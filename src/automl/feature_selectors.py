"""Feature selection helpers used by the four kitAb AutoML techniques."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from .pipeline_defaults import DEFAULT_RANDOM_STATE, DEFAULT_SFS_MIN_IMPROVEMENT
from .utils import (
    CorrelationBundle,
    DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS,
    DEFAULT_CORRELATION_SCREENING_ID_COLS,
    DEFAULT_PRUNE_EXCLUDE_COLS,
    _bh_adjust,
    calculate_correlations_and_plot,
    compute_correlation_bundle,
    fit_regressor,
    make_regressor,
    reduce_correlated_features,
    remove_low_variance_features,
)

_ALLOWED_SFS_MODELS = frozenset({"elasticnet", "randomforest", "svm", "knn"})


def _default_candidate_feature_columns(
    df: pd.DataFrame,
    target_col: str,
) -> list[str]:
    tc = str(target_col)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        c
        for c in numeric
        if str(c) != tc and not str(c).startswith("target")
    ]


def _validate_model_type(value: str, *, param_name: str, allowed: frozenset) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        opts = ", ".join(sorted(allowed))
        raise ValueError(f"{param_name} must be one of: {opts}")
    return normalized


def sequential_forward_selector(
    split_train_df,
    target_col: str,
    candidate_features: list | None = None,
    n_features_to_select=10,
    cv: int | None = None,
    scoring: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_improvement: float | None = None,
    n_jobs: int | None = None,
    model_type="elasticnet",
    enet_alpha: float | None = None,
    enet_l1_ratio: float | None = None,
    svm_C: float | None = None,
    svm_epsilon: float | None = None,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int | None = None,
    rf_max_features=None,
    knn_n_neighbors: int | None = None,
    knn_weights: str | None = None,
    knn_algorithm: str | None = None,
    knn_leaf_size: int | None = None,
    knn_p: int | None = None,
    knn_metric: str | None = None,
    knn_metric_params: dict | None = None,
):
    cv_eff = 5 if cv is None else int(cv)
    scoring_eff = "spearman" if scoring is None else str(scoring)
    min_imp_eff = DEFAULT_SFS_MIN_IMPROVEMENT if min_improvement is None else float(min_improvement)
    n_jobs_eff = -1 if n_jobs is None else int(n_jobs)

    def _build_custom_folds(n_samples, n_splits, seed):
        if n_splits < 2 or n_splits > n_samples:
            return []

        base_size = n_samples // n_splits
        remainder = n_samples % n_splits
        segment_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_splits)]
        boundaries = np.cumsum([0] + segment_sizes)

        rng = np.random.default_rng(seed)
        idx = np.arange(n_samples)
        rng.shuffle(idx)

        folds = []
        for i in range(n_splits):
            val_start, val_end = boundaries[i], boundaries[i + 1]
            val_idx = idx[val_start:val_end]
            train_idx = np.concatenate([idx[:val_start], idx[val_end:]])
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            folds.append((train_idx, val_idx))
        return folds

    def _score(y_true, y_pred, metric):
        if metric == "neg_mean_squared_error":
            return -mean_squared_error(y_true, y_pred)
        if metric == "r2":
            return r2_score(y_true, y_pred)
        if metric == "spearman":
            corr, _ = spearmanr(y_true, y_pred)
            return float(corr) if not np.isnan(corr) else -np.inf
        raise ValueError(
            "Unsupported scoring for manual forward selection. "
            "Use 'neg_mean_squared_error', 'r2', or 'spearman'."
        )

    model_type_norm = _validate_model_type(
        model_type, param_name="model_type", allowed=_ALLOWED_SFS_MODELS
    )

    if candidate_features is None:
        candidate_features = _default_candidate_feature_columns(
            split_train_df, target_col
        )
    candidate_features_t = list(dict.fromkeys(list(candidate_features)))
    cols = candidate_features_t + [target_col]
    work = split_train_df.loc[:, cols].astype(np.float64, copy=True)

    if len(work) < 3 or len(candidate_features_t) == 0:
        return []

    X = work[candidate_features_t]
    y = work[target_col]
    n_avail = X.shape[1]

    n_select = int(n_features_to_select)
    n_select = max(1, min(n_select, n_avail))

    effective_cv = min(cv_eff, len(work))
    if effective_cv < 2:
        return list(X.columns[:n_select])

    folds = _build_custom_folds(len(work), effective_cv, random_state)
    if len(folds) == 0:
        return list(X.columns[:n_select])

    remaining = list(X.columns)
    selected: list[str] = []
    prev_best_score = -np.inf

    for _ in range(n_select):
        best_feat = None
        best_score = -np.inf

        for feat in remaining:
            candidate_set = selected + [feat]
            fold_scores = []

            for train_idx, val_idx in folds:
                X_tr = X.iloc[train_idx][candidate_set]
                y_tr = y.iloc[train_idx]
                X_val = X.iloc[val_idx][candidate_set]
                y_val = y.iloc[val_idx]

                if len(X_tr) < 2 or len(X_val) < 1:
                    continue

                mr: dict = {
                    "random_state": random_state,
                    "n_jobs": n_jobs_eff,
                    "n_samples_fit": len(X_tr),
                }
                if enet_alpha is not None:
                    mr["enet_alpha"] = enet_alpha
                if enet_l1_ratio is not None:
                    mr["enet_l1_ratio"] = enet_l1_ratio
                if rf_n_estimators is not None:
                    mr["rf_n_estimators"] = rf_n_estimators
                if rf_max_depth is not None:
                    mr["rf_max_depth"] = rf_max_depth
                if rf_min_samples_leaf is not None:
                    mr["rf_min_samples_leaf"] = rf_min_samples_leaf
                if rf_max_features is not None:
                    mr["rf_max_features"] = rf_max_features
                if svm_C is not None:
                    mr["svm_C"] = svm_C
                if svm_epsilon is not None:
                    mr["svm_epsilon"] = svm_epsilon
                if knn_n_neighbors is not None:
                    mr["knn_n_neighbors"] = knn_n_neighbors
                if knn_weights is not None:
                    mr["knn_weights"] = knn_weights
                if knn_algorithm is not None:
                    mr["knn_algorithm"] = knn_algorithm
                if knn_leaf_size is not None:
                    mr["knn_leaf_size"] = knn_leaf_size
                if knn_p is not None:
                    mr["knn_p"] = knn_p
                if knn_metric is not None:
                    mr["knn_metric"] = knn_metric
                if knn_metric_params is not None:
                    mr["knn_metric_params"] = knn_metric_params
                model = make_regressor(model_type_norm, **mr)
                fit_regressor(model, X_tr, y_tr)
                try:
                    y_pred = model.predict(X_val)
                    fold_scores.append(_score(y_val, y_pred, scoring_eff))
                except Exception:
                    continue

            if len(fold_scores) == 0:
                continue

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_feat = feat

        if best_feat is None:
            break

        if prev_best_score != -np.inf:
            improvement = best_score - prev_best_score
            if improvement < min_imp_eff:
                break
        elif not np.isfinite(best_score):
            break

        prev_best_score = best_score
        selected.append(best_feat)
        remaining.remove(best_feat)

        if len(remaining) == 0:
            break

    if len(selected) == 0:
        selected = list(X.columns[:1])

    return selected


def correlation_selector(
    bundle: CorrelationBundle,
    target_col: str,
    *,
    candidate_features: Iterable[str] | None = None,
    p_threshold: float = 0.05,
    fdr_alpha: float = 0.05,
    use_fdr: bool = True,
    min_abs_rho: float | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    target_col = str(target_col)
    if target_col not in bundle.target_cols:
        raise ValueError(f"target_col {target_col!r} not in bundle.target_cols")

    in_bundle = set(bundle.feature_cols)
    if candidate_features is not None:
        want = [str(f) for f in candidate_features]
        feat = [f for f in want if f in in_bundle]
    else:
        feat = list(bundle.feature_cols)
    if not feat:
        corr_df = pd.DataFrame(
            columns=["feature", "spearman_r", "spearman_p", "spearman_p_adj", "n_samples"]
        )
        return corr_df, []

    corr_df = pd.DataFrame(
        {
            "feature": feat,
            "spearman_r": [bundle.target_feature_rho.loc[f, target_col] for f in feat],
            "spearman_p": [bundle.target_feature_pvalue.loc[f, target_col] for f in feat],
            "n_samples": [bundle.target_feature_n.loc[f, target_col] for f in feat],
        }
    )

    if use_fdr:
        corr_df["spearman_p_adj"] = _bh_adjust(corr_df["spearman_p"].values)
        significant = corr_df[corr_df["spearman_p_adj"] < fdr_alpha].copy()
    else:
        corr_df["spearman_p_adj"] = corr_df["spearman_p"].values
        significant = corr_df[corr_df["spearman_p"] < p_threshold].copy()

    if min_abs_rho is not None:
        significant = significant[
            significant["spearman_r"].abs() >= min_abs_rho
        ].copy()

    significant_tuples: list[tuple[str, float]] = []
    if len(significant) > 0:
        sig = significant.copy()
        sig["spearman_r"] = pd.to_numeric(sig["spearman_r"], errors="coerce")
        sig = sig.dropna(subset=["spearman_r"])
        if len(sig) > 0:
            sig = sig.assign(_abs_r=sig["spearman_r"].abs())
            sig = sig.sort_values("_abs_r", ascending=False).drop(columns=["_abs_r"])
            significant_tuples = list(
                zip(
                    sig["feature"].astype(str).tolist(),
                    sig["spearman_r"].astype(float).tolist(),
                )
            )

    return corr_df, significant_tuples


select_features_by_target_correlation = correlation_selector


def cv_shuffled_fold_ilocs(
    n_samples: int,
    n_splits: int,
    random_state: int,
) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray]]]:
    n = int(n_samples)
    if n_splits < 2 or n_splits > n:
        raise ValueError(f"n_splits must be between 2 and N={n}.")

    base_size = n // n_splits
    remainder = n % n_splits
    segment_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_splits)]
    boundaries = np.cumsum([0] + segment_sizes)

    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)

    fold_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_start, test_end = boundaries[i], boundaries[i + 1]
        test_idx = idx[test_start:test_end]
        train_idx = np.concatenate([idx[:test_start], idx[test_end:]])
        fold_pairs.append((train_idx, test_idx))

    return segment_sizes, fold_pairs


def _sorted_split_labels(series: pd.Series) -> list:
    uniq = list(pd.unique(series))
    try:
        return sorted(uniq, key=lambda x: int(float(x)))
    except (TypeError, ValueError):
        pass
    try:
        return sorted(uniq, key=float)
    except (TypeError, ValueError):
        return sorted(uniq, key=str)


def cv_split_col_ilocs(
    split_values: pd.Series,
) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray]]]:
    s = split_values.reset_index(drop=True)
    if s.isna().any():
        n_bad = int(s.isna().sum())
        raise ValueError(
            f"split_col has {n_bad} NaN row(s); every row used for CV must have a fold label."
        )
    uniques = _sorted_split_labels(s)
    k = len(uniques)
    if k < 2:
        raise ValueError(
            f"split_col must have at least 2 distinct values for CV; found {k}."
        )
    segment_sizes: list[int] = []
    fold_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for u in uniques:
        test_mask = (s == u).to_numpy()
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        segment_sizes.append(int(len(test_idx)))
        if len(train_idx) == 0:
            raise ValueError(f"split value {u!r}: no training rows (all rows have this fold).")
        fold_pairs.append((train_idx, test_idx))
    return segment_sizes, fold_pairs


__all__ = [
    "DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS",
    "DEFAULT_CORRELATION_SCREENING_ID_COLS",
    "DEFAULT_PRUNE_EXCLUDE_COLS",
    "CorrelationBundle",
    "compute_correlation_bundle",
    "correlation_selector",
    "select_features_by_target_correlation",
    "reduce_correlated_features",
    "remove_low_variance_features",
    "sequential_forward_selector",
    "cv_shuffled_fold_ilocs",
    "calculate_correlations_and_plot",
]
