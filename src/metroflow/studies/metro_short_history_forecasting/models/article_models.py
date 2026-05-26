from __future__ import annotations

from ..common import *
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)
from ..features import (
    BASE_FEATURES,
    RICH_FEATURES,
    _available_features,
    _xgb_matrix,
    _boost_lw_sample_weight,
    _aligned_sample_weight,
)
from ..evaluation.metrics import (
    calc_metrics_df,
    _calc_metric_values,
    evaluate_predictions,
    BOOTSTRAP_METRICS,
    PREDICTION_SINGLE_MODEL_CANDIDATES,
    ARTICLE_COMPARATOR_MODELS,
    _choose_bootstrap_reference_models,
    _safe_divide,
    _station_metric_component_arrays,
    station_bootstrap_ci,
)
from .boosting_local import (
    NORMALIZED_MAGNITUDE_PREFIXES,
    NORMALIZED_MAGNITUDE_COLS,
    _station_scale_series,
    _normalize_boost_frame,
    _fit_predict_xgb_normalized,
    _run_global_megaboost_normalized,
    _fit_predict_xgb_residual,
    _run_global_residual_megaboost,
    _run_validation_residual_megaboost,
    _prediction_metric,
    _blend_anchor_prediction,
    _select_correction_alpha,
    _line_code_column,
    _run_line_local_normalized_megaboost,
    _run_line_local_seasonal_residual_megaboost,
    _station_seasonal_fallback,
    _run_station_local_normalized_megaboost,
    _run_station_local_seasonal_residual_megaboost,
    _station_bias_table,
    _apply_station_bias,
    _run_station_bias_seasonal,
)
from .meta import (
    _seasonal_anchor_series,
    seasonal_daylag_predictions,
    _positive_blend,
    _context_blend,
    _build_meta_tables,
    _run_meta_models,
    _meta_feature_matrix,
)

def _station_cluster_descriptors(train_df: pd.DataFrame, cfg: MultiStationForecastConfig) -> pd.DataFrame:
    """Build train-only station descriptors for the Km-XGBoost-lite comparator."""
    work = train_df.dropna(subset=["target_h"]).copy()
    if work.empty:
        raise ValueError("No target_h rows available for station clustering")

    target = work.groupby("station_key", sort=True)["target_h"]
    stats = target.agg(
        cluster_mean_target="mean",
        cluster_median_target="median",
        cluster_std_target="std",
        cluster_q80_target=lambda x: x.quantile(0.80),
        cluster_q95_target=lambda x: x.quantile(0.95),
    )
    if "count" in work.columns:
        stats["cluster_mean_count"] = work.groupby("station_key", sort=True)["count"].mean()

    n_bands = 5
    slot = pd.to_numeric(work.get("slot_in_day"), errors="coerce").fillna(0).astype(int) % max(int(cfg.day_lag), 1)
    band = np.floor(slot.values.astype(float) * n_bands / max(int(cfg.day_lag), 1)).astype(int)
    band = np.clip(band, 0, n_bands - 1)
    work["__cluster_band"] = band
    profile = work.groupby(["station_key", "__cluster_band"], sort=True)["target_h"].mean().unstack()
    profile = profile.reindex(columns=list(range(n_bands)))
    profile.columns = [f"cluster_band_{int(c)}_target" for c in profile.columns]
    scale = stats["cluster_mean_target"].replace(0, np.nan)
    profile = profile.div(scale, axis=0)
    profile.columns = [f"{c}_ratio" for c in profile.columns]

    desc = stats.join(profile, how="left").reset_index()
    numeric = [c for c in desc.columns if c != "station_key"]
    for c in numeric:
        values = pd.to_numeric(desc[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fallback = float(values.median()) if values.notna().any() else 0.0
        desc[c] = values.fillna(fallback).astype(float)
    return desc

def _fit_station_kmeans_mapping(train_df: pd.DataFrame, cfg: MultiStationForecastConfig, n_clusters: int) -> tuple[pd.Series, pd.DataFrame]:
    desc = _station_cluster_descriptors(train_df, cfg)
    feature_cols = [c for c in desc.columns if c != "station_key"]
    if len(desc) < 2 or not feature_cols:
        mapping = pd.Series(0, index=desc["station_key"].astype(str), dtype=int)
        desc["cluster_id"] = 0
        return mapping, desc

    k = max(1, min(int(n_clusters), len(desc)))
    if k == 1:
        labels = np.zeros(len(desc), dtype=int)
    else:
        X = StandardScaler().fit_transform(desc[feature_cols].values.astype(float))
        model = KMeans(n_clusters=k, random_state=cfg.random_state + 17 * k, n_init=20)
        labels = model.fit_predict(X).astype(int)
    desc["cluster_id"] = labels
    mapping = pd.Series(labels, index=desc["station_key"].astype(str), dtype=int)
    return mapping, desc

def _with_station_cluster(df: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["__station_cluster"] = out["station_key"].astype(str).map(mapping).fillna(-1).astype(int)
    return out

def _run_km_xgboost_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    fallback_val_pred: np.ndarray,
    fallback_test_pred: np.ndarray,
    selection_metric: str = "LW-MAE",
) -> dict:
    """Cluster stations on train-only descriptors, then fit gated cluster-local normalized XGBoost models."""
    candidate_payloads: list[dict] = []
    candidate_rows: list[dict] = []
    fallback_val = pd.Series(np.asarray(fallback_val_pred, dtype=float), index=val_df.index, dtype=float)
    fallback_test = pd.Series(np.asarray(fallback_test_pred, dtype=float), index=test_df.index, dtype=float)

    for grid_i, n_clusters in enumerate(cfg.km_xgb_cluster_grid, start=1):
        mapping, desc = _fit_station_kmeans_mapping(train_df, cfg, int(n_clusters))
        train_c = _with_station_cluster(train_df, mapping)
        val_c = _with_station_cluster(val_df, mapping)
        train_full_c = _with_station_cluster(train_full_df, mapping)
        test_c = _with_station_cluster(test_df, mapping)
        val_series = fallback_val.copy()
        test_series = fallback_test.copy()
        cluster_rows: list[dict] = []

        for cluster_i, (cluster_id, val_cluster) in enumerate(val_c.groupby("__station_cluster", sort=True)):
            train_cluster = train_c[train_c["__station_cluster"].eq(cluster_id)].copy()
            train_full_cluster = train_full_c[train_full_c["__station_cluster"].eq(cluster_id)].copy()
            test_cluster = test_c[test_c["__station_cluster"].eq(cluster_id)].copy()
            fallback_cluster_pred = fallback_val.loc[val_cluster.index].values.astype(float)
            fallback_metric = _prediction_metric(val_cluster, fallback_cluster_pred, cfg, selection_metric)
            row = {
                "n_clusters": int(n_clusters),
                "cluster_id": int(cluster_id),
                "n_train": int(len(train_cluster)),
                "n_val": int(len(val_cluster)),
                "n_train_full": int(len(train_full_cluster)),
                "n_test": int(len(test_cluster)),
                "fallback_metric": fallback_metric,
                "cluster_metric": np.nan,
                "selected": "fallback",
                "status": "fallback",
            }
            if (
                int(cluster_id) < 0
                or len(train_cluster) < int(cfg.km_xgb_min_train_rows)
                or len(val_cluster) < int(cfg.km_xgb_min_validation_rows)
                or test_cluster.empty
            ):
                row["status"] = "fallback_insufficient_rows"
                cluster_rows.append(row)
                continue
            try:
                local = _run_global_megaboost_normalized(
                    cfg,
                    train_cluster,
                    val_cluster,
                    train_full_cluster,
                    test_cluster,
                    feature_cols,
                    label=f"{label} K={n_clusters} C={cluster_id}",
                    sample_weight_mode="lw",
                    selection_metric=selection_metric,
                    param_grid=cfg.local_boosting_param_grid,
                    top_k=cfg.top_k_local_boost_models,
                    seed_offset=90_000 + 1_000 * grid_i + 100 * cluster_i,
                )
                cluster_metric = _prediction_metric(val_cluster, local["val_pred"], cfg, selection_metric)
                use_cluster = bool(cluster_metric < fallback_metric)
                row.update({
                    "cluster_metric": cluster_metric,
                    "selected": "cluster" if use_cluster else "fallback",
                    "status": "cluster_selected" if use_cluster else "fallback_validation_gate",
                    "selected_names": "|".join(local.get("selected_names", [])),
                })
                if use_cluster:
                    val_series.loc[val_cluster.index] = np.asarray(local["val_pred"], dtype=float)
                    test_series.loc[test_cluster.index] = np.asarray(local["test_pred"], dtype=float)
            except Exception as exc:
                row["status"] = f"fallback_error:{type(exc).__name__}"
            cluster_rows.append(row)

        val_pred = val_series.loc[val_df.index].values.astype(float)
        test_pred = test_series.loc[test_df.index].values.astype(float)
        metric = _prediction_metric(val_df, val_pred, cfg, selection_metric)
        cluster_table = pd.DataFrame(cluster_rows)
        candidate_payloads.append({
            "n_clusters": int(n_clusters),
            "station_cluster_descriptors": desc,
            "cluster_selection_table": cluster_table,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "selection_metric_value": metric,
        })
        candidate_rows.append({
            "n_clusters": int(n_clusters),
            selection_metric: metric,
            "n_selected_clusters": int(cluster_table["selected"].eq("cluster").sum()) if not cluster_table.empty else 0,
            "n_candidate_clusters": int(len(cluster_table)),
        })

    candidate_table = pd.DataFrame(candidate_rows).sort_values([selection_metric, "n_clusters"]).reset_index(drop=True)
    if candidate_table.empty:
        raise ValueError("Km-XGBoost-lite did not produce any cluster candidates")
    best_k = int(candidate_table.loc[0, "n_clusters"])
    selected_payload = next(x for x in candidate_payloads if int(x["n_clusters"]) == best_k)
    return {
        "label": label,
        "val_pred": selected_payload["val_pred"],
        "test_pred": selected_payload["test_pred"],
        "test_metrics_micro": calc_metrics_df(test_df, selected_payload["test_pred"], cfg, mode="micro"),
        "candidate_table": candidate_table,
        "selected_n_clusters": best_k,
        "selected_cluster_table": selected_payload["cluster_selection_table"],
        "station_cluster_descriptors": selected_payload["station_cluster_descriptors"],
        "fallback_family": "MegaBoost rich norm LW",
        "selection_metric": selection_metric,
    }

def _fit_predict_extra_trees_normalized(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, object]:
    train_n, norm_features = _normalize_boost_frame(train_df, feature_cols)
    pred_n, _ = _normalize_boost_frame(pred_df, feature_cols)
    train_part = train_n.dropna(subset=["target_norm"]).copy()
    if train_part.empty:
        raise ValueError("No normalized target rows for ExtraTrees")
    fit_weight = _aligned_sample_weight(train_n, train_part, sample_weight)
    model = ExtraTreesRegressor(
        n_estimators=int(cfg.et_lgbm_extra_trees_estimators),
        min_samples_leaf=int(cfg.et_lgbm_extra_trees_min_samples_leaf),
        max_features=0.85,
        n_jobs=-1,
        random_state=cfg.random_state + seed_offset,
    )
    model.fit(_xgb_matrix(train_part, norm_features), train_part["target_norm"].astype(float), sample_weight=fit_weight)
    pred_norm = np.asarray(model.predict(_xgb_matrix(pred_n, norm_features)), dtype=float)
    pred = np.maximum(pred_norm * pred_n["__station_scale"].values.astype(float), 0.0)
    return pred, model

def _fit_predict_lgbm_normalized(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, object]:
    if LGBMRegressor is None:
        raise ModuleNotFoundError("lightgbm is not installed")
    train_n, norm_features = _normalize_boost_frame(train_df, feature_cols)
    pred_n, _ = _normalize_boost_frame(pred_df, feature_cols)
    train_part = train_n.dropna(subset=["target_norm"]).copy()
    if train_part.empty:
        raise ValueError("No normalized target rows for LightGBM")
    fit_weight = _aligned_sample_weight(train_n, train_part, sample_weight)
    model = LGBMRegressor(
        objective="mae",
        n_estimators=int(cfg.et_lgbm_n_estimators),
        learning_rate=float(cfg.et_lgbm_learning_rate),
        num_leaves=int(cfg.et_lgbm_num_leaves),
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=12.0,
        reg_alpha=2.0,
        random_state=cfg.random_state + seed_offset,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(_xgb_matrix(train_part, norm_features), train_part["target_norm"].astype(float), sample_weight=fit_weight)
    pred_norm = np.asarray(model.predict(_xgb_matrix(pred_n, norm_features)), dtype=float)
    pred = np.maximum(pred_norm * pred_n["__station_scale"].values.astype(float), 0.0)
    return pred, model

def _blend_metric(df: pd.DataFrame, left: np.ndarray, right: np.ndarray, weight_left: float, cfg: MultiStationForecastConfig, metric: str) -> float:
    pred = np.maximum(float(weight_left) * np.asarray(left, dtype=float) + (1.0 - float(weight_left)) * np.asarray(right, dtype=float), 0.0)
    return _prediction_metric(df, pred, cfg, metric)

def _optimize_pair_weight_pso(
    df: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    cfg: MultiStationForecastConfig,
    metric: str = "LW-MAE",
    seed_offset: int = 0,
) -> tuple[float, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state + seed_offset)
    n_particles = max(3, int(cfg.et_lgbm_pso_particles))
    n_iters = max(1, int(cfg.et_lgbm_pso_iters))
    pos = rng.uniform(0.0, 1.0, size=n_particles)
    vel = rng.normal(0.0, 0.08, size=n_particles)
    pbest = pos.copy()
    pbest_score = np.asarray([_blend_metric(df, left, right, w, cfg, metric) for w in pos], dtype=float)
    gbest_idx = int(np.argmin(pbest_score))
    gbest = float(pbest[gbest_idx])
    gbest_score = float(pbest_score[gbest_idx])
    rows = [{"iteration": 0, "best_weight_left": gbest, metric: gbest_score}]

    for it in range(1, n_iters + 1):
        r1 = rng.uniform(0.0, 1.0, size=n_particles)
        r2 = rng.uniform(0.0, 1.0, size=n_particles)
        vel = 0.60 * vel + 1.35 * r1 * (pbest - pos) + 1.35 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0.0, 1.0)
        scores = np.asarray([_blend_metric(df, left, right, w, cfg, metric) for w in pos], dtype=float)
        improve = scores < pbest_score
        pbest[improve] = pos[improve]
        pbest_score[improve] = scores[improve]
        current_idx = int(np.argmin(pbest_score))
        if float(pbest_score[current_idx]) < gbest_score:
            gbest_score = float(pbest_score[current_idx])
            gbest = float(pbest[current_idx])
        rows.append({"iteration": int(it), "best_weight_left": gbest, metric: gbest_score})
    return gbest, pd.DataFrame(rows)

def _run_et_lgbm_pso_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    selection_metric: str = "LW-MAE",
) -> dict:
    train_weight = _boost_lw_sample_weight(train_df, cfg)
    train_full_weight = _boost_lw_sample_weight(train_full_df, cfg)
    et_val, _ = _fit_predict_extra_trees_normalized(cfg, train_df, val_df, feature_cols, sample_weight=train_weight, seed_offset=111)
    lgbm_val, _ = _fit_predict_lgbm_normalized(cfg, train_df, val_df, feature_cols, sample_weight=train_weight, seed_offset=222)
    weight_et, pso_trace = _optimize_pair_weight_pso(val_df, et_val, lgbm_val, cfg, selection_metric, seed_offset=333)
    val_pred = np.maximum(weight_et * et_val + (1.0 - weight_et) * lgbm_val, 0.0)

    et_test, et_model = _fit_predict_extra_trees_normalized(cfg, train_full_df, test_df, feature_cols, sample_weight=train_full_weight, seed_offset=444)
    lgbm_test, lgbm_model = _fit_predict_lgbm_normalized(cfg, train_full_df, test_df, feature_cols, sample_weight=train_full_weight, seed_offset=555)
    test_pred = np.maximum(weight_et * et_test + (1.0 - weight_et) * lgbm_test, 0.0)
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "selected_weight_extra_trees": float(weight_et),
        "selected_weight_lightgbm": float(1.0 - weight_et),
        "pso_trace": pso_trace,
        "component_validation_metrics": pd.DataFrame([
            {"component": "ExtraTrees", **calc_metrics_df(val_df, et_val, cfg, mode="micro")},
            {"component": "LightGBM", **calc_metrics_df(val_df, lgbm_val, cfg, mode="micro")},
            {"component": "ET-LGBM-PSO-lite", **calc_metrics_df(val_df, val_pred, cfg, mode="micro")},
        ]),
        "models": [et_model, lgbm_model],
        "selection_metric": selection_metric,
    }

def _fit_predict_lsboost_normalized(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, object]:
    train_n, norm_features = _normalize_boost_frame(train_df, feature_cols)
    pred_n, _ = _normalize_boost_frame(pred_df, feature_cols)
    train_part = train_n.dropna(subset=["target_norm"]).copy()
    if train_part.empty:
        raise ValueError("No normalized target rows for LSBoost-like model")
    fit_weight = _aligned_sample_weight(train_n, train_part, sample_weight)
    model = GradientBoostingRegressor(
        loss="absolute_error",
        n_estimators=int(cfg.tp_boost_ls_estimators),
        learning_rate=float(cfg.tp_boost_ls_learning_rate),
        max_depth=2,
        min_samples_leaf=8,
        random_state=cfg.random_state + seed_offset,
    )
    model.fit(_xgb_matrix(train_part, norm_features), train_part["target_norm"].astype(float), sample_weight=fit_weight)
    pred_norm = np.asarray(model.predict(_xgb_matrix(pred_n, norm_features)), dtype=float)
    pred = np.maximum(pred_norm * pred_n["__station_scale"].values.astype(float), 0.0)
    return pred, model

def _time_partition_id(df: pd.DataFrame, cfg: MultiStationForecastConfig) -> pd.Series:
    slot = pd.to_numeric(df.get("slot_in_day"), errors="coerce").fillna(0).astype(int) % max(int(cfg.day_lag), 1)
    fractions = np.asarray([0.0, 1.0 / 6.0, 1.0 / 3.0, 7.0 / 12.0, 19.0 / 24.0, 1.0], dtype=float)
    edges = np.rint(fractions * max(int(cfg.day_lag), 1)).astype(int)
    edges[0] = 0
    edges[-1] = max(int(cfg.day_lag), 1)
    edges = np.maximum.accumulate(edges)
    # searchsorted on interior right edges produces contiguous operational-day partitions.
    partition = np.searchsorted(edges[1:-1], slot.values, side="right").astype(int)
    return pd.Series(partition, index=df.index, dtype=int)

def _select_pair_weight_grid(
    df: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    cfg: MultiStationForecastConfig,
    metric: str = "LW-MAE",
) -> tuple[float, pd.DataFrame]:
    rows = []
    for w in np.linspace(0.0, 1.0, 21):
        rows.append({"weight_left": float(w), metric: _blend_metric(df, left, right, float(w), cfg, metric)})
    table = pd.DataFrame(rows).sort_values([metric, "weight_left"]).reset_index(drop=True)
    return float(table.loc[0, "weight_left"]), table

def _run_tp_boost_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    fallback_val_pred: np.ndarray,
    fallback_test_pred: np.ndarray,
    selection_metric: str = "LW-MAE",
) -> dict:
    """Time-partitioned LSBoost/XGBoost fusion, gated against pooled rich normalized boosting."""
    train_p = train_df.copy()
    val_p = val_df.copy()
    train_full_p = train_full_df.copy()
    test_p = test_df.copy()
    train_p["__time_partition"] = _time_partition_id(train_p, cfg)
    val_p["__time_partition"] = _time_partition_id(val_p, cfg)
    train_full_p["__time_partition"] = _time_partition_id(train_full_p, cfg)
    test_p["__time_partition"] = _time_partition_id(test_p, cfg)

    fallback_val = pd.Series(np.asarray(fallback_val_pred, dtype=float), index=val_df.index, dtype=float)
    fallback_test = pd.Series(np.asarray(fallback_test_pred, dtype=float), index=test_df.index, dtype=float)
    val_series = fallback_val.copy()
    test_series = fallback_test.copy()
    rows: list[dict] = []
    weight_tables: dict[int, pd.DataFrame] = {}

    xgb_params = dict(cfg.local_boosting_param_grid[min(1, len(cfg.local_boosting_param_grid) - 1)])
    for part_i, (part_id, val_part) in enumerate(val_p.groupby("__time_partition", sort=True)):
        train_part = train_p[train_p["__time_partition"].eq(part_id)].copy()
        train_full_part = train_full_p[train_full_p["__time_partition"].eq(part_id)].copy()
        test_part = test_p[test_p["__time_partition"].eq(part_id)].copy()
        fallback_pred = fallback_val.loc[val_part.index].values.astype(float)
        fallback_metric = _prediction_metric(val_part, fallback_pred, cfg, selection_metric)
        row = {
            "time_partition": int(part_id),
            "n_train": int(len(train_part)),
            "n_val": int(len(val_part)),
            "n_train_full": int(len(train_full_part)),
            "n_test": int(len(test_part)),
            "fallback_metric": fallback_metric,
            "partition_metric": np.nan,
            "selected_weight_lsboost": np.nan,
            "selected": "fallback",
            "status": "fallback",
        }
        if (
            len(train_part) < int(cfg.tp_boost_min_train_rows)
            or len(val_part) < int(cfg.tp_boost_min_validation_rows)
            or test_part.empty
        ):
            row["status"] = "fallback_insufficient_rows"
            rows.append(row)
            continue
        try:
            train_weight = _boost_lw_sample_weight(train_part, cfg)
            train_full_weight = _boost_lw_sample_weight(train_full_part, cfg)
            xgb_params["random_state"] = cfg.random_state + 120_000 + 100 * part_i
            xgb_val, _ = _fit_predict_xgb_normalized(train_part, val_part, feature_cols, xgb_params, sample_weight=train_weight)
            ls_val, _ = _fit_predict_lsboost_normalized(cfg, train_part, val_part, feature_cols, sample_weight=train_weight, seed_offset=130_000 + 100 * part_i)
            weight_ls, weight_table = _select_pair_weight_grid(val_part, ls_val, xgb_val, cfg, selection_metric)
            blended_val = np.maximum(weight_ls * ls_val + (1.0 - weight_ls) * xgb_val, 0.0)
            partition_metric = _prediction_metric(val_part, blended_val, cfg, selection_metric)
            use_partition = bool(partition_metric < fallback_metric)
            row.update({
                "partition_metric": partition_metric,
                "selected_weight_lsboost": float(weight_ls),
                "selected": "partition" if use_partition else "fallback",
                "status": "partition_selected" if use_partition else "fallback_validation_gate",
            })
            weight_tables[int(part_id)] = weight_table
            if use_partition:
                xgb_params["random_state"] = cfg.random_state + 140_000 + 100 * part_i
                xgb_test, _ = _fit_predict_xgb_normalized(train_full_part, test_part, feature_cols, xgb_params, sample_weight=train_full_weight)
                ls_test, _ = _fit_predict_lsboost_normalized(cfg, train_full_part, test_part, feature_cols, sample_weight=train_full_weight, seed_offset=150_000 + 100 * part_i)
                blended_test = np.maximum(weight_ls * ls_test + (1.0 - weight_ls) * xgb_test, 0.0)
                val_series.loc[val_part.index] = blended_val
                test_series.loc[test_part.index] = blended_test
        except Exception as exc:
            row["status"] = f"fallback_error:{type(exc).__name__}"
        rows.append(row)

    val_pred = val_series.loc[val_df.index].values.astype(float)
    test_pred = test_series.loc[test_df.index].values.astype(float)
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "partition_selection_table": pd.DataFrame(rows),
        "partition_weight_tables": weight_tables,
        "fallback_family": "MegaBoost rich norm LW",
        "selection_metric": selection_metric,
    }

def _fit_predict_adaboost_normalized(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, object]:
    train_n, norm_features = _normalize_boost_frame(train_df, feature_cols)
    pred_n, _ = _normalize_boost_frame(pred_df, feature_cols)
    train_part = train_n.dropna(subset=["target_norm"]).copy()
    if train_part.empty:
        raise ValueError("No normalized target rows for AdaBoost")
    fit_weight = _aligned_sample_weight(train_n, train_part, sample_weight)
    tree = DecisionTreeRegressor(
        max_depth=int(cfg.ml_fusion_ada_tree_depth),
        min_samples_leaf=12,
        random_state=cfg.random_state + seed_offset,
    )
    kwargs = dict(
        n_estimators=int(cfg.ml_fusion_ada_estimators),
        learning_rate=float(cfg.ml_fusion_ada_learning_rate),
        loss="linear",
        random_state=cfg.random_state + seed_offset + 7,
    )
    try:
        model = AdaBoostRegressor(estimator=tree, **kwargs)
    except TypeError:  # sklearn < 1.2 compatibility
        model = AdaBoostRegressor(base_estimator=tree, **kwargs)
    model.fit(
        _xgb_matrix(train_part, norm_features).fillna(0.0),
        train_part["target_norm"].astype(float),
        sample_weight=fit_weight,
    )
    pred_norm = np.asarray(model.predict(_xgb_matrix(pred_n, norm_features).fillna(0.0)), dtype=float)
    pred = np.maximum(pred_norm * pred_n["__station_scale"].values.astype(float), 0.0)
    return pred, model

def _run_ml_fusion_svr_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    xgb_val_pred: np.ndarray,
    xgb_test_pred: np.ndarray,
) -> dict:
    """AdaBoost/LightGBM/XGB prediction fusion with a validation-fitted SVR meta-combiner.

    The XGB branch reuses the already fitted rich normalized MegaBoost prediction.  This keeps
    the comparator inexpensive while preserving the paper-level idea of fusing heterogeneous
    tree learners through a nonlinear SVR-style combiner.  The fusion model is not exposed to
    the main meta-pool.
    """
    if LGBMRegressor is None:
        raise ModuleNotFoundError("lightgbm is not installed")
    train_weight = _boost_lw_sample_weight(train_df, cfg)
    train_full_weight = _boost_lw_sample_weight(train_full_df, cfg)
    ada_val, ada_val_model = _fit_predict_adaboost_normalized(
        cfg, train_df, val_df, feature_cols, sample_weight=train_weight, seed_offset=160_000
    )
    ada_test, ada_test_model = _fit_predict_adaboost_normalized(
        cfg, train_full_df, test_df, feature_cols, sample_weight=train_full_weight, seed_offset=160_100
    )
    lgbm_val, lgbm_val_model = _fit_predict_lgbm_normalized(
        cfg, train_df, val_df, feature_cols, sample_weight=train_weight, seed_offset=161_000
    )
    lgbm_test, lgbm_test_model = _fit_predict_lgbm_normalized(
        cfg, train_full_df, test_df, feature_cols, sample_weight=train_full_weight, seed_offset=161_100
    )

    X_val_all = np.column_stack([
        np.asarray(xgb_val_pred, dtype=float),
        ada_val.astype(float),
        lgbm_val.astype(float),
    ])
    X_test = np.column_stack([
        np.asarray(xgb_test_pred, dtype=float),
        ada_test.astype(float),
        lgbm_test.astype(float),
    ])
    y_val = val_df["target_h"].astype(float).values
    rng = np.random.default_rng(cfg.random_state + 162_000)
    max_rows = max(500, int(cfg.ml_fusion_svr_max_rows))
    if len(y_val) > max_rows:
        fit_idx = np.sort(rng.choice(np.arange(len(y_val)), size=max_rows, replace=False))
    else:
        fit_idx = np.arange(len(y_val))
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_val_all[fit_idx])
    svr = SVR(
        kernel="rbf",
        C=float(cfg.ml_fusion_svr_c),
        epsilon=float(cfg.ml_fusion_svr_epsilon),
        gamma="scale",
        cache_size=512,
    )
    svr.fit(X_fit_scaled, y_val[fit_idx])
    val_pred = np.maximum(np.asarray(svr.predict(scaler.transform(X_val_all)), dtype=float), 0.0)
    test_pred = np.maximum(np.asarray(svr.predict(scaler.transform(X_test)), dtype=float), 0.0)
    component_rows = []
    for component, pred in [
        ("MegaBoost rich norm LW", xgb_val_pred),
        ("AdaBoost normalized", ada_val),
        ("LightGBM normalized", lgbm_val),
        (label, val_pred),
    ]:
        component_rows.append({"component": component, **calc_metrics_df(val_df, np.asarray(pred, dtype=float), cfg, mode="micro")})
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "component_table": pd.DataFrame(component_rows),
        "models": [ada_val_model, ada_test_model, lgbm_val_model, lgbm_test_model, svr],
        "validation_calibrated": True,
        "fusion_fit_rows": int(len(fit_idx)),
    }

def _augment_multiscale_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def col(name: str) -> pd.Series:
        return pd.to_numeric(out.get(name, 0.0), errors="coerce").astype(float)
    lag1 = col("lag_1")
    lag4 = col("lag_4")
    roll4 = col("roll_mean_4")
    roll8 = col("roll_mean_8")
    roll16 = col("roll_mean_16")
    daylag = col("target_daylag")
    out["msd_low_short"] = roll4
    out["msd_low_long"] = roll16
    out["msd_mid_component"] = roll4 - roll16
    out["msd_high_component"] = lag1 - roll4
    out["msd_day_component"] = daylag - roll16
    out["msd_local_acceleration"] = lag1 - 2.0 * col("lag_2") + col("lag_3")
    out["msd_recent_spread"] = lag1 - lag4
    out["msd_roll_gap"] = roll8 - roll16
    return out

MULTISCALE_PROXY_FEATURES = [
    "msd_low_short", "msd_low_long", "msd_mid_component", "msd_high_component",
    "msd_day_component", "msd_local_acceleration", "msd_recent_spread", "msd_roll_gap",
]

def _fit_predict_lgbm_normalized_custom(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    sample_weight: np.ndarray | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, object]:
    if LGBMRegressor is None:
        raise ModuleNotFoundError("lightgbm is not installed")
    train_n, norm_features = _normalize_boost_frame(train_df, feature_cols)
    pred_n, _ = _normalize_boost_frame(pred_df, feature_cols)
    train_part = train_n.dropna(subset=["target_norm"]).copy()
    if train_part.empty:
        raise ValueError("No normalized target rows for custom LightGBM")
    fit_weight = _aligned_sample_weight(train_n, train_part, sample_weight)
    base = dict(
        objective="mae",
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=12.0,
        reg_alpha=2.0,
        random_state=cfg.random_state + seed_offset,
        n_jobs=-1,
        verbosity=-1,
    )
    base.update(params)
    model = LGBMRegressor(**base)
    model.fit(_xgb_matrix(train_part, norm_features), train_part["target_norm"].astype(float), sample_weight=fit_weight)
    pred_norm = np.asarray(model.predict(_xgb_matrix(pred_n, norm_features)), dtype=float)
    pred = np.maximum(pred_norm * pred_n["__station_scale"].values.astype(float), 0.0)
    return pred, model

def _run_vmd_lightgbm_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    selection_metric: str = "MAE",
) -> dict:
    """Causal multi-scale LightGBM proxy for decomposition-driven metro forecasting papers."""
    if LGBMRegressor is None:
        raise ModuleNotFoundError("lightgbm is not installed")
    train_m = _augment_multiscale_proxy_features(train_df)
    val_m = _augment_multiscale_proxy_features(val_df)
    train_full_m = _augment_multiscale_proxy_features(train_full_df)
    test_m = _augment_multiscale_proxy_features(test_df)
    features = list(dict.fromkeys([*feature_cols, *MULTISCALE_PROXY_FEATURES]))
    train_weight = _boost_lw_sample_weight(train_m, cfg)
    train_full_weight = _boost_lw_sample_weight(train_full_m, cfg)
    candidates = []
    for i, (leaves, estimators) in enumerate(zip(cfg.vmd_lgbm_candidate_leaves, cfg.vmd_lgbm_candidate_estimators), start=1):
        params = {
            "num_leaves": int(leaves),
            "n_estimators": int(estimators),
            "learning_rate": 0.035 if int(estimators) <= 480 else 0.028,
            "min_child_samples": 18 + 2 * i,
        }
        pred_val, _ = _fit_predict_lgbm_normalized_custom(
            cfg, train_m, val_m, features, params, sample_weight=train_weight, seed_offset=170_000 + i
        )
        metrics = calc_metrics_df(val_m, pred_val, cfg, mode="micro")
        candidates.append({"candidate": f"cfg{i}", "params": params, "val_pred": pred_val, **metrics})
    table = pd.DataFrame([{k: v for k, v in c.items() if k not in {"params", "val_pred"}} for c in candidates]).sort_values(selection_metric).reset_index(drop=True)
    best_name = str(table.loc[0, "candidate"])
    selected = next(c for c in candidates if c["candidate"] == best_name)
    pred_test, model = _fit_predict_lgbm_normalized_custom(
        cfg,
        train_full_m,
        test_m,
        features,
        selected["params"],
        sample_weight=train_full_weight,
        seed_offset=171_000,
    )
    return {
        "label": label,
        "val_pred": np.asarray(selected["val_pred"], dtype=float),
        "test_pred": pred_test,
        "test_metrics_micro": calc_metrics_df(test_m, pred_test, cfg, mode="micro"),
        "candidate_table": table,
        "selected_candidate": best_name,
        "features": features,
        "models": [model],
        "selection_metric": selection_metric,
    }

def _feature_selection_sample(df: pd.DataFrame, cfg: MultiStationForecastConfig) -> pd.DataFrame:
    valid = df.dropna(subset=["target_h"]).copy()
    max_rows = max(1000, int(cfg.feature_selection_max_rows))
    if len(valid) <= max_rows:
        return valid
    return valid.sample(max_rows, random_state=cfg.random_state + 172_000)

def _select_mi_features(train_df: pd.DataFrame, feature_cols: list[str], cfg: MultiStationForecastConfig, top_k: int) -> pd.DataFrame:
    sample = _feature_selection_sample(train_df, cfg)
    X = _xgb_matrix(sample, feature_cols).fillna(0.0)
    y = sample["target_h"].astype(float).values
    if X.empty:
        return pd.DataFrame(columns=["feature", "score", "rank"])
    score = mutual_info_regression(X.values.astype(float), y, random_state=cfg.random_state + 172_100)
    table = pd.DataFrame({"feature": feature_cols, "score": score}).sort_values(["score", "feature"], ascending=[False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table.head(max(1, min(int(top_k), len(table)))).reset_index(drop=True)

def _run_st_lightgbm_mic_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict:
    """Mutual-information-filtered LightGBM proxy for ST-LightGBM-MIC style models."""
    if LGBMRegressor is None:
        raise ModuleNotFoundError("lightgbm is not installed")
    selected_table = _select_mi_features(train_df, feature_cols, cfg, cfg.st_mic_top_k_features)
    selected_features = selected_table["feature"].tolist()
    if not selected_features:
        raise ValueError("ST-LightGBM-MIC-lite selected no features")
    train_weight = _boost_lw_sample_weight(train_df, cfg)
    train_full_weight = _boost_lw_sample_weight(train_full_df, cfg)
    params = {
        "n_estimators": int(cfg.et_lgbm_n_estimators),
        "learning_rate": float(cfg.et_lgbm_learning_rate),
        "num_leaves": int(cfg.et_lgbm_num_leaves),
        "min_child_samples": 24,
    }
    pred_val, _ = _fit_predict_lgbm_normalized_custom(
        cfg, train_df, val_df, selected_features, params, sample_weight=train_weight, seed_offset=173_000
    )
    pred_test, model = _fit_predict_lgbm_normalized_custom(
        cfg, train_full_df, test_df, selected_features, params, sample_weight=train_full_weight, seed_offset=173_100
    )
    return {
        "label": label,
        "val_pred": pred_val,
        "test_pred": pred_test,
        "test_metrics_micro": calc_metrics_df(test_df, pred_test, cfg, mode="micro"),
        "selected_feature_table": selected_table,
        "features": selected_features,
        "models": [model],
    }

def _fit_predict_rff_residual_correction(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    anchor_train_pred: np.ndarray,
    anchor_pred: np.ndarray,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict]:
    train_part = train_df.dropna(subset=["target_h"]).copy()
    if train_part.empty:
        raise ValueError("No residual rows for SN-LSSVM-lite")
    X_train = _xgb_matrix(train_part, feature_cols).fillna(0.0).values.astype(float)
    X_pred = _xgb_matrix(pred_df, feature_cols).fillna(0.0).values.astype(float)
    residual = train_part["target_h"].astype(float).values - pd.Series(np.asarray(anchor_train_pred, dtype=float), index=train_df.index).loc[train_part.index].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pred_s = scaler.transform(X_pred)
    rff = RBFSampler(
        gamma=float(cfg.sn_lssvm_rbf_gamma),
        n_components=int(cfg.sn_lssvm_rbf_components),
        random_state=cfg.random_state + seed_offset,
    )
    Z_train = rff.fit_transform(X_train_s)
    Z_pred = rff.transform(X_pred_s)
    ridge = Ridge(alpha=float(cfg.sn_lssvm_ridge_alpha))
    ridge.fit(Z_train, residual)
    correction = np.asarray(ridge.predict(Z_pred), dtype=float)
    lo, hi = np.quantile(residual[np.isfinite(residual)], [0.05, 0.95]) if np.isfinite(residual).any() else (-np.inf, np.inf)
    correction = np.clip(correction, lo, hi)
    pred = np.maximum(np.asarray(anchor_pred, dtype=float) + correction, 0.0)
    return pred, {"scaler": scaler, "rff": rff, "ridge": ridge, "residual_clip": (float(lo), float(hi))}

def _run_sn_lssvm_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    anchor_train_pred: np.ndarray,
    anchor_val_pred: np.ndarray,
    anchor_train_full_pred: np.ndarray,
    anchor_test_pred: np.ndarray,
    selection_metric: str = "MAE",
) -> dict:
    corrected_val, tune_model = _fit_predict_rff_residual_correction(
        cfg, train_df, val_df, feature_cols, anchor_train_pred, anchor_val_pred, seed_offset=174_000
    )
    alpha, alpha_table = _select_correction_alpha(val_df, anchor_val_pred, corrected_val, cfg, selection_metric)
    val_pred = _blend_anchor_prediction(anchor_val_pred, corrected_val, alpha)
    corrected_test, full_model = _fit_predict_rff_residual_correction(
        cfg, train_full_df, test_df, feature_cols, anchor_train_full_pred, anchor_test_pred, seed_offset=174_100
    )
    test_pred = _blend_anchor_prediction(anchor_test_pred, corrected_test, alpha)
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "selected_alpha": float(alpha),
        "alpha_table": alpha_table,
        "models": [tune_model, full_model],
        "selection_metric": selection_metric,
        "anchor_family": "seasonal",
    }

def _select_correlation_features(train_df: pd.DataFrame, feature_cols: list[str], cfg: MultiStationForecastConfig, top_k: int) -> pd.DataFrame:
    sample = _feature_selection_sample(train_df, cfg)
    y = sample["target_h"].astype(float)
    rows = []
    for c in feature_cols:
        x = pd.to_numeric(sample[c], errors="coerce").astype(float)
        corr = float(x.corr(y)) if x.notna().any() else 0.0
        rows.append({"feature": c, "corr": corr, "abs_corr": abs(corr) if np.isfinite(corr) else 0.0})
    table = pd.DataFrame(rows).sort_values(["abs_corr", "feature"], ascending=[False, True]).reset_index(drop=True)
    table["corr_rank"] = np.arange(1, len(table) + 1)
    return table.head(max(1, min(int(top_k), len(table)))).reset_index(drop=True)

def _two_stage_feature_selection(train_df: pd.DataFrame, feature_cols: list[str], cfg: MultiStationForecastConfig) -> pd.DataFrame:
    corr_table = _select_correlation_features(train_df, feature_cols, cfg, cfg.twostage_corr_top_k_features)
    forecast_features = corr_table["feature"].tolist()
    sample = _feature_selection_sample(train_df, cfg)
    X = _xgb_matrix(sample, forecast_features).fillna(0.0)
    y = sample["target_h"].astype(float)
    if X.empty:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr", "importance", "importance_rank"])
    selector = ExtraTreesRegressor(
        n_estimators=220,
        min_samples_leaf=3,
        max_features=0.90,
        n_jobs=-1,
        random_state=cfg.random_state + 175_000,
    )
    selector.fit(X, y)
    imp = pd.DataFrame({"feature": forecast_features, "importance": selector.feature_importances_})
    table = corr_table.merge(imp, on="feature", how="left")
    table["importance"] = table["importance"].fillna(0.0)
    table = table.sort_values(["importance", "abs_corr", "feature"], ascending=[False, False, True]).reset_index(drop=True)
    table["importance_rank"] = np.arange(1, len(table) + 1)
    return table.head(max(1, min(int(cfg.twostage_final_top_k_features), len(table)))).reset_index(drop=True)

def _run_twostage_fs_msd_hybrid_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    selection_metric: str = "MAE",
) -> dict:
    """Two-stage feature selection + multiscale XGB/LightGBM hybrid kept outside the meta-pool."""
    if LGBMRegressor is None:
        raise ModuleNotFoundError("lightgbm is not installed")
    train_m = _augment_multiscale_proxy_features(train_df)
    val_m = _augment_multiscale_proxy_features(val_df)
    train_full_m = _augment_multiscale_proxy_features(train_full_df)
    test_m = _augment_multiscale_proxy_features(test_df)
    candidate_features = list(dict.fromkeys([*feature_cols, *MULTISCALE_PROXY_FEATURES]))
    feature_table = _two_stage_feature_selection(train_m, candidate_features, cfg)
    selected_features = feature_table["feature"].tolist()
    if not selected_features:
        raise ValueError("TwoStage-FS-MSD-Hybrid-lite selected no features")
    train_weight = _boost_lw_sample_weight(train_m, cfg)
    train_full_weight = _boost_lw_sample_weight(train_full_m, cfg)
    lgbm_params = {
        "n_estimators": max(360, int(cfg.et_lgbm_n_estimators) - 40),
        "learning_rate": float(cfg.et_lgbm_learning_rate),
        "num_leaves": int(cfg.et_lgbm_num_leaves),
        "min_child_samples": 22,
    }
    xgb_params = dict(cfg.boosting_param_grid[min(3, len(cfg.boosting_param_grid) - 1)])
    xgb_params["random_state"] = cfg.random_state + 176_000
    lgbm_val, _ = _fit_predict_lgbm_normalized_custom(
        cfg, train_m, val_m, selected_features, lgbm_params, sample_weight=train_weight, seed_offset=176_100
    )
    xgb_val, _ = _fit_predict_xgb_normalized(train_m, val_m, selected_features, xgb_params, sample_weight=train_weight)
    w_lgbm, blend_table = _select_pair_weight_grid(val_m, lgbm_val, xgb_val, cfg, selection_metric)
    val_pred = np.maximum(w_lgbm * lgbm_val + (1.0 - w_lgbm) * xgb_val, 0.0)
    xgb_params["random_state"] = cfg.random_state + 176_200
    lgbm_test, lgbm_model = _fit_predict_lgbm_normalized_custom(
        cfg, train_full_m, test_m, selected_features, lgbm_params, sample_weight=train_full_weight, seed_offset=176_300
    )
    xgb_test, xgb_model = _fit_predict_xgb_normalized(train_full_m, test_m, selected_features, xgb_params, sample_weight=train_full_weight)
    test_pred = np.maximum(w_lgbm * lgbm_test + (1.0 - w_lgbm) * xgb_test, 0.0)
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_m, test_pred, cfg, mode="micro"),
        "feature_selection_table": feature_table,
        "blend_table": blend_table,
        "selected_weight_lgbm": float(w_lgbm),
        "features": selected_features,
        "models": [lgbm_model, xgb_model],
        "selection_metric": selection_metric,
    }

def _stacking_catboost_meta_matrix(
    df: pd.DataFrame,
    et_pred: np.ndarray,
    lgbm_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> pd.DataFrame:
    X = pd.DataFrame({
        "pred_extra_trees": np.asarray(et_pred, dtype=float),
        "pred_lightgbm": np.asarray(lgbm_pred, dtype=float),
        "pred_xgboost": np.asarray(xgb_pred, dtype=float),
        "pred_seasonal_daylag": _seasonal_anchor_series(df).values.astype(float),
    }, index=df.index)
    for c in ["hour_sin", "hour_cos", "continuous_load_score", "hl_daylag"]:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float).values
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

def _run_stacking_catboost_lite(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict:
    """Article-inspired tree stack: ExtraTrees/LightGBM/XGBoost -> CatBoost meta-head.

    This is kept outside the prediction meta-ensemble.  It follows the high-level
    Stacking-CatBoost idea from metro passenger-flow literature, but uses only
    features available in the present closed dataset rather than unavailable
    passenger-return-probability inputs.
    """
    if CatBoostRegressor is None:
        raise ModuleNotFoundError("catboost is not installed")
    if LGBMRegressor is None:
        raise ModuleNotFoundError("lightgbm is not installed")

    # Use standard-error-oriented component fits: this comparator is intended as a
    # strong literature-grounded average-error baseline, not as a high-load model.
    et_val, _ = _fit_predict_extra_trees_normalized(cfg, train_df, val_df, feature_cols, sample_weight=None, seed_offset=9101)
    lgbm_val, _ = _fit_predict_lgbm_normalized(cfg, train_df, val_df, feature_cols, sample_weight=None, seed_offset=9102)
    xgb_params = dict(cfg.boosting_param_grid[min(3, len(cfg.boosting_param_grid) - 1)])
    xgb_params["random_state"] = cfg.random_state + 9103
    xgb_val, _ = _fit_predict_xgb_normalized(train_df, val_df, feature_cols, xgb_params, sample_weight=None)
    X_meta_val = _stacking_catboost_meta_matrix(val_df, et_val, lgbm_val, xgb_val)

    meta = CatBoostRegressor(
        loss_function="MAE",
        iterations=int(cfg.stacking_catboost_iterations),
        depth=int(cfg.stacking_catboost_depth),
        learning_rate=float(cfg.stacking_catboost_learning_rate),
        l2_leaf_reg=float(cfg.stacking_catboost_l2_leaf_reg),
        random_seed=cfg.random_state + 9104,
        verbose=False,
        allow_writing_files=False,
    )
    meta.fit(X_meta_val, val_df["target_h"].astype(float))
    val_pred = np.maximum(np.asarray(meta.predict(X_meta_val), dtype=float), 0.0)

    et_test, et_model = _fit_predict_extra_trees_normalized(cfg, train_full_df, test_df, feature_cols, sample_weight=None, seed_offset=9201)
    lgbm_test, lgbm_model = _fit_predict_lgbm_normalized(cfg, train_full_df, test_df, feature_cols, sample_weight=None, seed_offset=9202)
    xgb_params_full = dict(xgb_params)
    xgb_params_full["random_state"] = cfg.random_state + 9203
    xgb_test, xgb_model = _fit_predict_xgb_normalized(train_full_df, test_df, feature_cols, xgb_params_full, sample_weight=None)
    X_meta_test = _stacking_catboost_meta_matrix(test_df, et_test, lgbm_test, xgb_test)
    test_pred = np.maximum(np.asarray(meta.predict(X_meta_test), dtype=float), 0.0)

    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "component_validation_metrics": pd.DataFrame([
            {"component": "ExtraTrees", **calc_metrics_df(val_df, et_val, cfg, mode="micro")},
            {"component": "LightGBM", **calc_metrics_df(val_df, lgbm_val, cfg, mode="micro")},
            {"component": "XGBoost", **calc_metrics_df(val_df, xgb_val, cfg, mode="micro")},
            {"component": "Stacking-CatBoost-lite", **calc_metrics_df(val_df, val_pred, cfg, mode="micro")},
        ]),
        "models": [et_model, lgbm_model, xgb_model, meta],
    }

__all__ = [
    "_station_cluster_descriptors",
    "_fit_station_kmeans_mapping",
    "_with_station_cluster",
    "_run_km_xgboost_lite",
    "_fit_predict_extra_trees_normalized",
    "_fit_predict_lgbm_normalized",
    "_blend_metric",
    "_optimize_pair_weight_pso",
    "_run_et_lgbm_pso_lite",
    "_fit_predict_lsboost_normalized",
    "_time_partition_id",
    "_select_pair_weight_grid",
    "_run_tp_boost_lite",
    "_fit_predict_adaboost_normalized",
    "_run_ml_fusion_svr_lite",
    "_augment_multiscale_proxy_features",
    "MULTISCALE_PROXY_FEATURES",
    "_fit_predict_lgbm_normalized_custom",
    "_run_vmd_lightgbm_lite",
    "_feature_selection_sample",
    "_select_mi_features",
    "_run_st_lightgbm_mic_lite",
    "_fit_predict_rff_residual_correction",
    "_run_sn_lssvm_lite",
    "_select_correlation_features",
    "_two_stage_feature_selection",
    "_run_twostage_fs_msd_hybrid_lite",
    "_stacking_catboost_meta_matrix",
    "_run_stacking_catboost_lite",
]
