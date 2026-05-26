
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from metroflow.config import ExperimentConfig
from metroflow.evaluation.metrics import calc_metrics


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerLiteRegressor(nn.Module):
    def __init__(self, input_size=1, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        h = self.encoder(x)
        last = h[:, -1, :]
        return self.head(last)


def standardize(train_values, other_values_list):
    mean_ = np.mean(train_values)
    std_ = np.std(train_values) + 1e-8
    train_scaled = (train_values - mean_) / std_
    others_scaled = [(arr - mean_) / std_ for arr in other_values_list]
    return train_scaled, others_scaled, mean_, std_


def make_supervised_windows(values, timestamps, window, horizon):
    X, y, t = [], [], []
    values = np.asarray(values, dtype=np.float32)
    timestamps = pd.Series(timestamps).reset_index(drop=True)
    for anchor in range(window - 1, len(values) - horizon):
        X.append(values[anchor - window + 1: anchor + 1])
        y.append(values[anchor + 1: anchor + horizon + 1].sum())
        t.append(timestamps.iloc[anchor])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(t)


def train_seq_model(model, train_loader, device, val_loader=None, epochs=80, lr=1e-3, patience=8):
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float('inf')
    best_epoch = epochs
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if val_loader is None:
            continue
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_epoch


def predict_seq_model(model, X_scaled, mean_, std_, horizon, device, clip_nonnegative=True):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
        pred_scaled = model(X_tensor).cpu().numpy().reshape(-1)
    pred = pred_scaled * std_ + horizon * mean_
    if clip_nonnegative:
        pred = np.maximum(pred, 0.0)
    return pred


def build_seq_eval_frames(train_all_df, pred_all_df, pred_anchor_df, time_col, seq_window, horizon):
    train_series = train_all_df['count'].astype(float).values
    pred_series = pred_all_df['count'].astype(float).values
    train_scaled, [pred_scaled], mean_train, std_train = standardize(train_series, [pred_series])
    X_train, y_train, _ = make_supervised_windows(train_scaled, train_all_df[time_col].reset_index(drop=True), seq_window, horizon)
    concat_scaled = np.concatenate([train_scaled, pred_scaled])
    concat_ts = pd.concat([train_all_df[time_col], pred_all_df[time_col]], axis=0).reset_index(drop=True)
    X_all, y_all, t_all = make_supervised_windows(concat_scaled, concat_ts, seq_window, horizon)
    mask_eval = pd.Series(pd.to_datetime(t_all)).isin(pd.to_datetime(pred_anchor_df[time_col]).values).values
    X_eval = X_all[mask_eval]
    y_eval = y_all[mask_eval]
    t_eval = pd.to_datetime(t_all[mask_eval])
    return X_train, y_train, X_eval, y_eval, t_eval, mean_train, std_train


def run_sequence_models(cfg: ExperimentConfig, split: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    time_col = cfg.time_col
    h = cfg.horizon_steps
    seq_window = cfg.seq_window
    y_val = split['val_df']['target_h'].values.astype(float)
    y_test = split['test_df']['target_h'].values.astype(float)

    X_train, y_train, X_val, y_val_seq, _, mean_tune, std_tune = build_seq_eval_frames(
        split['train_tune_all'], split['val_all'], split['val_df'], time_col, seq_window, h
    )
    X_train_full, y_train_full, X_test, y_test_seq, _, mean_full, std_full = build_seq_eval_frames(
        split['train_full_all'], split['test_all'], split['test_df'], time_col, seq_window, h
    )

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=min(64, max(1, len(X_train))), shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val_seq), batch_size=min(64, max(1, len(X_val))), shuffle=False)
    full_loader = DataLoader(SeqDataset(X_train_full, y_train_full), batch_size=min(64, max(1, len(X_train_full))), shuffle=True)

    out = {}

    if cfg.models.run_lstm:
        lstm = LSTMRegressor(hidden_size=32, num_layers=1, dropout=0.0)
        lstm, _, best_epoch = train_seq_model(lstm, train_loader, device=device, val_loader=val_loader,
                                              epochs=cfg.models.lstm_max_epochs, patience=cfg.models.seq_patience)
        pred_val = predict_seq_model(lstm, X_val, mean_tune, std_tune, h, device)
        lstm_full = LSTMRegressor(hidden_size=32, num_layers=1, dropout=0.0)
        lstm_full, _, _ = train_seq_model(lstm_full, full_loader, device=device, val_loader=None,
                                          epochs=best_epoch, patience=cfg.models.seq_patience)
        pred_test = predict_seq_model(lstm_full, X_test, mean_full, std_full, h, device)
        out['LSTM'] = {
            'val_pred': pred_val,
            'test_pred': pred_test,
            'val_metrics': calc_metrics(y_val, pred_val, gamma=cfg.features.load_weight_gamma),
            'test_metrics': calc_metrics(y_test, pred_test, gamma=cfg.features.load_weight_gamma),
        }

    if cfg.models.run_transformer:
        tf = TransformerLiteRegressor(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1)
        tf, _, best_epoch = train_seq_model(tf, train_loader, device=device, val_loader=val_loader,
                                            epochs=cfg.models.transformer_max_epochs, patience=cfg.models.seq_patience)
        pred_val = predict_seq_model(tf, X_val, mean_tune, std_tune, h, device)
        tf_full = TransformerLiteRegressor(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1)
        tf_full, _, _ = train_seq_model(tf_full, full_loader, device=device, val_loader=None,
                                        epochs=best_epoch, patience=cfg.models.seq_patience)
        pred_test = predict_seq_model(tf_full, X_test, mean_full, std_full, h, device)
        out['Transformer-lite'] = {
            'val_pred': pred_val,
            'test_pred': pred_test,
            'val_metrics': calc_metrics(y_val, pred_val, gamma=cfg.features.load_weight_gamma),
            'test_metrics': calc_metrics(y_test, pred_test, gamma=cfg.features.load_weight_gamma),
        }

    return out
