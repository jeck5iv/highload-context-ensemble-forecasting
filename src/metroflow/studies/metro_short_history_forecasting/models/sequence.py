from __future__ import annotations

from ..common import *
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)

def _scaled_station_sequence_parts(
    train_all_df: pd.DataFrame,
    pred_all_df: pd.DataFrame,
    pred_anchor_df: pd.DataFrame,
    cfg: MultiStationForecastConfig,
) -> dict:
    """Create pooled sequence windows with per-station standardization."""
    window = int(cfg.seq_window)
    h = int(cfg.horizon_steps)
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    X_eval_parts: list[np.ndarray] = []
    y_eval_parts: list[np.ndarray] = []
    eval_indices: list[int] = []
    eval_means: list[float] = []
    eval_stds: list[float] = []

    train_groups = {k: g.sort_values(cfg.time_col) for k, g in train_all_df.groupby("station_key", sort=False)}
    pred_groups = {k: g.sort_values(cfg.time_col) for k, g in pred_all_df.groupby("station_key", sort=False)}

    for station_key, anchor_g in pred_anchor_df.groupby("station_key", sort=False):
        train_g = train_groups.get(station_key)
        pred_g = pred_groups.get(station_key)
        if train_g is None or pred_g is None:
            continue
        train_values = train_g["count"].astype(float).values
        pred_values = pred_g["count"].astype(float).values
        if len(train_values) < window + h + 1 or len(pred_values) < h + 1:
            continue
        mean_ = float(np.mean(train_values))
        std_ = float(np.std(train_values) + 1e-8)
        train_scaled = (train_values - mean_) / std_
        pred_scaled = (pred_values - mean_) / std_

        X_train_local = []
        y_train_local = []
        for anchor in range(window - 1, len(train_scaled) - h):
            X_train_local.append(train_scaled[anchor - window + 1: anchor + 1])
            y_train_local.append(float(train_scaled[anchor + 1: anchor + h + 1].sum()))
        if X_train_local:
            X_train_parts.append(np.asarray(X_train_local, dtype=np.float32))
            y_train_parts.append(np.asarray(y_train_local, dtype=np.float32))

        concat_scaled = np.concatenate([train_scaled, pred_scaled])
        pred_positions = {pd.Timestamp(ts): i for i, ts in enumerate(pd.to_datetime(pred_g[cfg.time_col]).tolist())}
        for idx, row in anchor_g.sort_values(cfg.time_col).iterrows():
            ts = pd.Timestamp(row[cfg.time_col])
            pred_pos = pred_positions.get(ts)
            if pred_pos is None:
                continue
            anchor = len(train_scaled) + int(pred_pos)
            if anchor - window + 1 < 0 or anchor + h >= len(concat_scaled):
                continue
            x = concat_scaled[anchor - window + 1: anchor + 1]
            y_scaled = float(concat_scaled[anchor + 1: anchor + h + 1].sum())
            X_eval_parts.append(np.asarray(x, dtype=np.float32))
            y_eval_parts.append(np.asarray(y_scaled, dtype=np.float32))
            eval_indices.append(int(idx))
            eval_means.append(mean_)
            eval_stds.append(std_)

    if not X_train_parts or not X_eval_parts:
        raise ValueError("No sequence windows were constructed")
    return {
        "X_train": np.concatenate(X_train_parts, axis=0),
        "y_train": np.concatenate(y_train_parts, axis=0),
        "X_eval": np.stack(X_eval_parts).astype(np.float32),
        "y_eval": np.asarray(y_eval_parts, dtype=np.float32),
        "eval_indices": np.asarray(eval_indices, dtype=int),
        "eval_means": np.asarray(eval_means, dtype=float),
        "eval_stds": np.asarray(eval_stds, dtype=float),
    }

def _sequence_eval_pred_to_frame_order(payload: dict, raw_pred: np.ndarray, eval_df: pd.DataFrame, cfg: MultiStationForecastConfig) -> np.ndarray:
    pred_unscaled = raw_pred * payload["eval_stds"] + cfg.horizon_steps * payload["eval_means"]
    pred_unscaled = np.maximum(pred_unscaled, 0.0)
    series = pd.Series(pred_unscaled, index=payload["eval_indices"], dtype=float)
    missing = eval_df.index.difference(series.index)
    if len(missing):
                                                                                               
        fallback = eval_df.loc[missing, "target_daylag"].fillna(eval_df.loc[missing, "lag_96"]).fillna(eval_df["target_h"].mean())
        series.loc[missing] = fallback.values.astype(float)
    return series.loc[eval_df.index].values.astype(float)

def _predict_scaled_sequence_model(model, X_scaled: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_scaled), max(1, int(batch_size))):
            xb = torch.tensor(X_scaled[start:start + int(batch_size)], dtype=torch.float32).unsqueeze(-1).to(device)
            preds.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0).astype(float) if preds else np.empty(0, dtype=float)

def _run_global_sequence_models(cfg: MultiStationForecastConfig, split: dict) -> dict[str, dict]:
    """Fit pooled global sequence experts on per-station standardized windows."""
    import torch
    from torch.utils.data import DataLoader
    from metroflow.models.sequence import SeqDataset, LSTMRegressor, TransformerLiteRegressor, train_seq_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    tune = _scaled_station_sequence_parts(split["train_tune_all"], split["val_all"], split["val_df"], cfg)
    full = _scaled_station_sequence_parts(split["train_full_all"], split["test_all"], split["test_df"], cfg)

    train_loader = DataLoader(
        SeqDataset(tune["X_train"], tune["y_train"]),
        batch_size=min(int(cfg.seq_batch_size), max(1, len(tune["X_train"]))),
        shuffle=True,
    )
    val_loader = DataLoader(
        SeqDataset(tune["X_eval"], tune["y_eval"]),
        batch_size=min(int(cfg.seq_batch_size), max(1, len(tune["X_eval"]))),
        shuffle=False,
    )
    full_loader = DataLoader(
        SeqDataset(full["X_train"], full["y_train"]),
        batch_size=min(int(cfg.seq_batch_size), max(1, len(full["X_train"]))),
        shuffle=True,
    )

    out: dict[str, dict] = {}

    if cfg.run_global_lstm:
        lstm = LSTMRegressor(hidden_size=32, num_layers=1, dropout=0.0)
        lstm, val_loss, best_epoch = train_seq_model(
            lstm, train_loader, device=device, val_loader=val_loader,
            epochs=int(cfg.lstm_max_epochs), patience=int(cfg.seq_patience),
        )
        val_raw = _predict_scaled_sequence_model(lstm, tune["X_eval"], cfg.seq_batch_size, device)
        val_pred = _sequence_eval_pred_to_frame_order(tune, val_raw, split["val_df"], cfg)

        lstm_full = LSTMRegressor(hidden_size=32, num_layers=1, dropout=0.0)
        lstm_full, _, _ = train_seq_model(
            lstm_full, full_loader, device=device, val_loader=None,
            epochs=max(1, int(best_epoch)), patience=int(cfg.seq_patience),
        )
        test_raw = _predict_scaled_sequence_model(lstm_full, full["X_eval"], cfg.seq_batch_size, device)
        test_pred = _sequence_eval_pred_to_frame_order(full, test_raw, split["test_df"], cfg)
        out["Global LSTM"] = {
            "val_pred": val_pred,
            "test_pred": test_pred,
            "best_epoch": int(best_epoch),
            "val_loss_scaled": float(val_loss) if np.isfinite(val_loss) else np.nan,
            "n_train_windows_tune": int(len(tune["X_train"])),
            "n_train_windows_full": int(len(full["X_train"])),
        }

    if cfg.run_global_transformer:
        tf = TransformerLiteRegressor(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1)
        tf, val_loss, best_epoch = train_seq_model(
            tf, train_loader, device=device, val_loader=val_loader,
            epochs=int(cfg.transformer_max_epochs), patience=int(cfg.seq_patience),
        )
        val_raw = _predict_scaled_sequence_model(tf, tune["X_eval"], cfg.seq_batch_size, device)
        val_pred = _sequence_eval_pred_to_frame_order(tune, val_raw, split["val_df"], cfg)

        tf_full = TransformerLiteRegressor(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1)
        tf_full, _, _ = train_seq_model(
            tf_full, full_loader, device=device, val_loader=None,
            epochs=max(1, int(best_epoch)), patience=int(cfg.seq_patience),
        )
        test_raw = _predict_scaled_sequence_model(tf_full, full["X_eval"], cfg.seq_batch_size, device)
        test_pred = _sequence_eval_pred_to_frame_order(full, test_raw, split["test_df"], cfg)
        out["Global Transformer-lite"] = {
            "val_pred": val_pred,
            "test_pred": test_pred,
            "best_epoch": int(best_epoch),
            "val_loss_scaled": float(val_loss) if np.isfinite(val_loss) else np.nan,
            "n_train_windows_tune": int(len(tune["X_train"])),
            "n_train_windows_full": int(len(full["X_train"])),
        }

    return out

def _phase_space_reconstruct_windows(windows: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
    """Causal Taken-style delay embedding over already-causal historical windows."""
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D windows for PSR, got shape={x.shape}")
    m = max(2, int(embedding_dim))
    tau = max(1, int(delay))
    reconstructed_len = x.shape[1] - (m - 1) * tau
    if reconstructed_len < 2:
        raise ValueError(
            f"PSR settings embedding_dim={m}, delay={tau} are incompatible with window={x.shape[1]}"
        )
    parts = [x[:, j * tau:j * tau + reconstructed_len] for j in range(m)]
    return np.stack(parts, axis=-1).astype(np.float32)

def _causal_moving_average_windows(windows: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32)
    width = max(1, int(width))
    out = np.empty_like(x, dtype=np.float32)
    # seq_window is small (24 in the current setup), so this loop is cheap and memory-stable.
    for t in range(x.shape[1]):
        lo = max(0, t - width + 1)
        out[:, t] = x[:, lo:t + 1].mean(axis=1)
    return out

def _multiscale_decompose_windows(windows: np.ndarray, short_width: int, long_width: int) -> np.ndarray:
    """Leakage-free multi-scale decomposition for the decomposition/attention comparator.

    Lu & Wang's SMA-VMD-Informer-BiLSTM uses VMD components.  Exact VMD on every rolling
    causal window would be very slow and brittle for the present one-week setting, so this
    lightweight comparator preserves the central methodology—multi-scale decomposition plus
    parallel attention/BiLSTM fusion—using causal high/mid/low components.
    """
    x = np.asarray(windows, dtype=np.float32)
    short_ma = _causal_moving_average_windows(x, short_width)
    long_ma = _causal_moving_average_windows(x, max(int(long_width), int(short_width) + 1))
    high = x - short_ma
    mid = short_ma - long_ma
    low = long_ma
    return np.stack([high, mid, low], axis=-1).astype(np.float32)

def _predict_tensor_sequence_model(model, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), max(1, int(batch_size))):
            xb = torch.tensor(X[start:start + int(batch_size)], dtype=torch.float32).to(device)
            preds.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0).astype(float) if preds else np.empty(0, dtype=float)

def _train_tensor_sequence_regressor(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: MultiStationForecastConfig,
    device: str,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int | None = None,
) -> tuple[object, float, int]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    max_epochs = max(1, int(cfg.article_seq_max_epochs if epochs is None else epochs))
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(cfg.seq_batch_size), max(1, len(train_ds))),
        shuffle=True,
    )
    val_loader = None
    if X_val is not None and y_val is not None and len(X_val):
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=min(int(cfg.seq_batch_size), max(1, len(val_ds))),
            shuffle=False,
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.article_seq_learning_rate),
        weight_decay=float(cfg.article_seq_weight_decay),
    )
    loss_fn = nn.L1Loss()
    best_val = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    patience = max(1, int(cfg.article_seq_patience))

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).reshape(-1)
            loss = loss_fn(pred, yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if val_loader is None:
            best_epoch = epoch
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).reshape(-1)
                val_losses.append(float(loss_fn(pred, yb.reshape(-1)).detach().cpu()))
        current = float(np.mean(val_losses)) if val_losses else float("inf")
        if current + 1e-8 < best_val:
            best_val = current
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if not np.isfinite(best_val):
        best_val = np.nan
    return model, float(best_val), int(best_epoch)

def _run_article_sequence_models(cfg: MultiStationForecastConfig, split: dict) -> dict[str, dict]:
    """Fit recent article-grounded lightweight neural comparators.

    The models intentionally stay compact.  They are used as literature-motivated comparators
    under the same short-history multi-station protocol, not as exact reproductions of the papers'
    original longer-data experimental environments.
    """
    import torch
    from metroflow.models.article_sequence import (
        BiLSTMCTPALiteRegressor,
        CNNBiLSTMAttentionRegressor,
        MSDInformerBiLSTMLiteRegressor,
        PSRCNNLSTMRegressor,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.random_state + 91)
    np.random.seed(cfg.random_state + 91)

    tune = _scaled_station_sequence_parts(split["train_tune_all"], split["val_all"], split["val_df"], cfg)
    full = _scaled_station_sequence_parts(split["train_full_all"], split["test_all"], split["test_df"], cfg)
    out: dict[str, dict] = {}

    def fit_and_register(name: str, model_factory, tune_x: np.ndarray, tune_val_x: np.ndarray, full_x: np.ndarray, full_eval_x: np.ndarray):
        model = model_factory()
        model, val_loss, best_epoch = _train_tensor_sequence_regressor(
            model,
            tune_x,
            tune["y_train"],
            cfg,
            device,
            X_val=tune_val_x,
            y_val=tune["y_eval"],
        )
        val_raw = _predict_tensor_sequence_model(model, tune_val_x, cfg.seq_batch_size, device)
        val_pred = _sequence_eval_pred_to_frame_order(tune, val_raw, split["val_df"], cfg)

        full_model = model_factory()
        full_model, _, _ = _train_tensor_sequence_regressor(
            full_model,
            full_x,
            full["y_train"],
            cfg,
            device,
            epochs=max(1, int(best_epoch)),
        )
        test_raw = _predict_tensor_sequence_model(full_model, full_eval_x, cfg.seq_batch_size, device)
        test_pred = _sequence_eval_pred_to_frame_order(full, test_raw, split["test_df"], cfg)
        out[name] = {
            "val_pred": val_pred,
            "test_pred": test_pred,
            "best_epoch": int(best_epoch),
            "val_loss_scaled": float(val_loss) if np.isfinite(val_loss) else np.nan,
            "n_train_windows_tune": int(len(tune_x)),
            "n_train_windows_full": int(len(full_x)),
        }

    if cfg.run_psr_cnn_lstm:
        tune_x = _phase_space_reconstruct_windows(tune["X_train"], cfg.psr_embedding_dim, cfg.psr_delay)
        tune_val_x = _phase_space_reconstruct_windows(tune["X_eval"], cfg.psr_embedding_dim, cfg.psr_delay)
        full_x = _phase_space_reconstruct_windows(full["X_train"], cfg.psr_embedding_dim, cfg.psr_delay)
        full_eval_x = _phase_space_reconstruct_windows(full["X_eval"], cfg.psr_embedding_dim, cfg.psr_delay)
        fit_and_register(
            "PSR-CNN-LSTM-lite",
            lambda: PSRCNNLSTMRegressor(embedding_dim=int(cfg.psr_embedding_dim)),
            tune_x,
            tune_val_x,
            full_x,
            full_eval_x,
        )

    if cfg.run_cnn_bilstm_attention:
        fit_and_register(
            "CNN-BiLSTM-Attention-lite",
            lambda: CNNBiLSTMAttentionRegressor(input_dim=1),
            tune["X_train"][:, :, None].astype(np.float32),
            tune["X_eval"][:, :, None].astype(np.float32),
            full["X_train"][:, :, None].astype(np.float32),
            full["X_eval"][:, :, None].astype(np.float32),
        )

    if cfg.run_msd_informer_bilstm_lite:
        tune_x = _multiscale_decompose_windows(tune["X_train"], cfg.msd_short_window, cfg.msd_long_window)
        tune_val_x = _multiscale_decompose_windows(tune["X_eval"], cfg.msd_short_window, cfg.msd_long_window)
        full_x = _multiscale_decompose_windows(full["X_train"], cfg.msd_short_window, cfg.msd_long_window)
        full_eval_x = _multiscale_decompose_windows(full["X_eval"], cfg.msd_short_window, cfg.msd_long_window)
        fit_and_register(
            "MSD-Informer-BiLSTM-lite",
            lambda: MSDInformerBiLSTMLiteRegressor(input_dim=3, max_len=int(cfg.seq_window)),
            tune_x,
            tune_val_x,
            full_x,
            full_eval_x,
        )

    if cfg.run_bilstm_ctpa_lite:
        fit_and_register(
            "BiLSTM-CTPA-lite",
            lambda: BiLSTMCTPALiteRegressor(input_dim=1),
            tune["X_train"][:, :, None].astype(np.float32),
            tune["X_eval"][:, :, None].astype(np.float32),
            full["X_train"][:, :, None].astype(np.float32),
            full["X_eval"][:, :, None].astype(np.float32),
        )

    return out

__all__ = [
    "_scaled_station_sequence_parts",
    "_sequence_eval_pred_to_frame_order",
    "_predict_scaled_sequence_model",
    "_run_global_sequence_models",
    "_phase_space_reconstruct_windows",
    "_causal_moving_average_windows",
    "_multiscale_decompose_windows",
    "_predict_tensor_sequence_model",
    "_train_tensor_sequence_regressor",
    "_run_article_sequence_models",
]
