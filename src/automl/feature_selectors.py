from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from .pipeline_defaults import (
    DEFAULT_INTERCORR_IMPORTANCE_METRIC,
    DEFAULT_INTERCORR_REDUCTION_MODE,
    DEFAULT_INTERCORR_THRESHOLD,
    DEFAULT_LOW_VARIANCE_EPSILON,
    DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SFS_MIN_IMPROVEMENT,
)
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
    apply_minmax_to_train_test_features,
    remove_low_variance_features,
)

_ALLOWED_STABILITY_MODELS = frozenset({"elasticnet", "randomforest", "svm"})
_ALLOWED_RFE_MODELS = frozenset({"elasticnet", "randomforest", "svm"})
_ALLOWED_SFS_MODELS = frozenset({"elasticnet", "randomforest", "svm", "knn"})

# Adaptive subsample fraction: max(0.4, min(0.8, 1/log10(5 + n_train))).
FOLD_TRAIN_LOG10_CONST_5 = "fold_train_log10_const_5"
_LEGACY_STABILITY_REDUCTION_SF_SENTINEL = "__stability_reduction_sf_fold_train_log10__"

# Adaptive subsample count: max(100, min(300, round(500/log10(n_train)))).
FOLD_TRAIN_LOG10_NS_500 = "fold_train_log10_ns_500"

STABILITY_PREREDUCTION_N_SUBSAMPLES = 50


def stability_reduction_subsample_fraction_fold_train_log10(n_train: int) -> float:
    nt = max(1, int(n_train))
    x = 1.0 / math.log10(5.0 + float(nt))
    return max(0.4, min(0.8, x))


def stability_reduction_n_subsamples_fold_train_log500(n_train: int) -> int:
    nt = max(2, int(n_train))
    x = 500.0 / math.log10(float(nt))
    k = int(round(x))
    return max(100, min(300, k))


def _resolve_stability_n_subsamples(spec: int | str, n_train: int) -> int:
    if spec == FOLD_TRAIN_LOG10_NS_500:
        return stability_reduction_n_subsamples_fold_train_log500(n_train)
    return int(spec)


def _resolve_stability_sample_fraction(spec: float | str, n_train: int) -> float:
    if spec in (FOLD_TRAIN_LOG10_CONST_5, _LEGACY_STABILITY_REDUCTION_SF_SENTINEL):
        return stability_reduction_subsample_fraction_fold_train_log10(n_train)
    return float(spec)


# Keys under YAML ``hyperparameters`` / CLI JSON (aliases → run_feature_selection_on_one_fold kwargs).
_STABILITY_HP_ALIASES: dict[str, str] = {
    "n_subsamples": "stability_n_subsamples",
    "stability_n_subsamples": "stability_n_subsamples",
    "sample_fraction": "stability_sample_fraction",
    "stability_sample_fraction": "stability_sample_fraction",
    "l1_ratio": "stability_elasticnet_l1_ratio",
    "elasticnet_l1_ratio": "stability_elasticnet_l1_ratio",
    "stability_elasticnet_l1_ratio": "stability_elasticnet_l1_ratio",
    "stability_l1_ratio": "stability_elasticnet_l1_ratio",
    "alpha": "stability_elasticnet_alpha",
    "elasticnet_alpha": "stability_elasticnet_alpha",
    "stability_elasticnet_alpha": "stability_elasticnet_alpha",
    "stability_alpha": "stability_elasticnet_alpha",
    "coef_threshold": "stability_coef_threshold",
    "stability_coef_threshold": "stability_coef_threshold",
    "svm_c": "stability_svm_c",
    "svm_C": "stability_svm_c",
    "stability_svm_c": "stability_svm_c",
    "svm_epsilon": "stability_svm_epsilon",
    "stability_svm_epsilon": "stability_svm_epsilon",
    "rf_n_estimators": "stability_rf_n_estimators",
    "stability_rf_n_estimators": "stability_rf_n_estimators",
    "rf_max_depth": "stability_rf_max_depth",
    "stability_rf_max_depth": "stability_rf_max_depth",
    "rf_min_samples_leaf": "stability_rf_min_samples_leaf",
    "stability_rf_min_samples_leaf": "stability_rf_min_samples_leaf",
    "stability_rf_max_features": "stability_rf_max_features",
    "rf_max_features": "stability_rf_max_features",
    "max_features": "stability_rf_max_features",
    "n_estimators": "stability_rf_n_estimators",
    "max_depth": "stability_rf_max_depth",
    "min_samples_leaf": "stability_rf_min_samples_leaf",
    "C": "stability_svm_c",
    "epsilon": "stability_svm_epsilon",
}

_STABILITY_HP_CANONICAL = frozenset(_STABILITY_HP_ALIASES.values())

# Shared prefilter keys (allowed for every selector type).
_SHARED_HP_ALIASES: dict[str, str] = {
    "low_variance_relative_std_threshold": "low_variance_relative_std_threshold",
    "low_var_threshold": "low_variance_relative_std_threshold",
    "low_variance_epsilon": "low_variance_epsilon",
    "intercorr_threshold": "intercorr_threshold",
    "intercorr_importance_metric": "intercorr_importance_metric",
    "intercorr_metric": "intercorr_importance_metric",
    "intercorr_reduction_mode": "intercorr_reduction_mode",
    "intercorr_mode": "intercorr_reduction_mode",
    "correlation_reduction_min_n_features": "correlation_reduction_min_n_features",
    "stability_reduction_n_features": "stability_reduction_n_features",
    "stability_reduction_features": "stability_reduction_n_features",
    "stability_reduction_model": "stability_reduction_model",
    "stability_reduction_n_subsamples": "stability_reduction_n_subsamples",
    "stability_reduction_sample_fraction": "stability_reduction_sample_fraction",
    "stability_reduction_coef_threshold": "stability_reduction_coef_threshold",
    "stability_reduction_l1_ratio": "stability_reduction_elasticnet_l1_ratio",
    "stability_reduction_elasticnet_l1_ratio": "stability_reduction_elasticnet_l1_ratio",
    "stability_reduction_alpha": "stability_reduction_elasticnet_alpha",
    "stability_reduction_elasticnet_alpha": "stability_reduction_elasticnet_alpha",
    "stability_reduction_svm_c": "stability_reduction_svm_c",
    "stability_reduction_svm_C": "stability_reduction_svm_c",
    "stability_reduction_svm_epsilon": "stability_reduction_svm_epsilon",
    "stability_reduction_rf_n_estimators": "stability_reduction_rf_n_estimators",
    "stability_reduction_rf_max_depth": "stability_reduction_rf_max_depth",
    "stability_reduction_rf_min_samples_leaf": "stability_reduction_rf_min_samples_leaf",
    "stability_reduction_rf_max_features": "stability_reduction_rf_max_features",
    "stability_reduction_min_n_features": "stability_reduction_min_n_features",
    "min_n_features": "stability_reduction_min_n_features",
    "stability_prereduction_n_features": "stability_prereduction_n_features",
    "prereduction_n_features": "stability_prereduction_n_features",
}

_CORRELATION_HP_ALIASES: dict[str, str] = {
    "correlation_p_threshold": "correlation_p_threshold",
    "p_threshold": "correlation_p_threshold",
    "correlation_fdr_alpha": "correlation_fdr_alpha",
    "fdr_alpha": "correlation_fdr_alpha",
    "correlation_use_fdr": "correlation_use_fdr",
    "use_fdr": "correlation_use_fdr",
    "correlation_normalize": "correlation_normalize",
    "normalize": "correlation_normalize",
    "correlation_make_plots": "correlation_make_plots",
    "make_plots": "correlation_make_plots",
    "correlation_screening_min_abs_rho": "correlation_screening_min_abs_rho",
    "min_abs_rho": "correlation_screening_min_abs_rho",
    "screening_min_abs_rho": "correlation_screening_min_abs_rho",
    "correlation_post_threshold": "correlation_post_threshold",
    "post_threshold": "correlation_post_threshold",
    "correlation_post_importance_metric": "correlation_post_importance_metric",
    "post_importance_metric": "correlation_post_importance_metric",
}

_SFS_HP_ALIASES: dict[str, str] = {
    "sfs_cv": "sfs_cv",
    "cv": "sfs_cv",
    "sfs_scoring": "sfs_scoring",
    "scoring": "sfs_scoring",
    "sfs_min_improvement": "sfs_min_improvement",
    "min_improvement": "sfs_min_improvement",
    "sfs_n_jobs": "sfs_n_jobs",
    "n_jobs": "sfs_n_jobs",
    "sfs_enet_alpha": "sfs_enet_alpha",
    "elasticnet_alpha": "sfs_enet_alpha",
    "sfs_enet_l1_ratio": "sfs_enet_l1_ratio",
    "elasticnet_l1_ratio": "sfs_enet_l1_ratio",
    "alpha": "sfs_enet_alpha",
    "l1_ratio": "sfs_enet_l1_ratio",
    "sfs_svm_c": "sfs_svm_c",
    "sfs_svm_epsilon": "sfs_svm_epsilon",
    "svm_c": "sfs_svm_c",
    "svm_C": "sfs_svm_c",
    "C": "sfs_svm_c",
    "svm_epsilon": "sfs_svm_epsilon",
    "epsilon": "sfs_svm_epsilon",
    "sfs_rf_n_estimators": "sfs_rf_n_estimators",
    "rf_n_estimators": "sfs_rf_n_estimators",
    "n_estimators": "sfs_rf_n_estimators",
    "sfs_rf_max_depth": "sfs_rf_max_depth",
    "rf_max_depth": "sfs_rf_max_depth",
    "max_depth": "sfs_rf_max_depth",
    "sfs_rf_min_samples_leaf": "sfs_rf_min_samples_leaf",
    "rf_min_samples_leaf": "sfs_rf_min_samples_leaf",
    "min_samples_leaf": "sfs_rf_min_samples_leaf",
    "sfs_rf_max_features": "sfs_rf_max_features",
    "rf_max_features": "sfs_rf_max_features",
    "max_features": "sfs_rf_max_features",
    "sfs_knn_n_neighbors": "sfs_knn_n_neighbors",
    "knn_n_neighbors": "sfs_knn_n_neighbors",
    "n_neighbors": "sfs_knn_n_neighbors",
}

_RFE_HP_ALIASES: dict[str, str] = {
    "rfe_enet_alpha": "rfe_enet_alpha",
    "enet_alpha": "rfe_enet_alpha",
    "rfe_enet_l1_ratio": "rfe_enet_l1_ratio",
    "enet_l1_ratio": "rfe_enet_l1_ratio",
    "rfe_rf_n_estimators": "rfe_rf_n_estimators",
    "rf_n_estimators": "rfe_rf_n_estimators",
    "n_estimators": "rfe_rf_n_estimators",
    "rfe_rf_max_depth": "rfe_rf_max_depth",
    "rf_max_depth": "rfe_rf_max_depth",
    "max_depth": "rfe_rf_max_depth",
    "rfe_rf_min_samples_leaf": "rfe_rf_min_samples_leaf",
    "rf_min_samples_leaf": "rfe_rf_min_samples_leaf",
    "min_samples_leaf": "rfe_rf_min_samples_leaf",
    "rfe_svm_c": "rfe_svm_c",
    "svm_c": "rfe_svm_c",
    "svm_C": "rfe_svm_c",
    "C": "rfe_svm_c",
    "rfe_svm_epsilon": "rfe_svm_epsilon",
    "svm_epsilon": "rfe_svm_epsilon",
    "epsilon": "rfe_svm_epsilon",
    "rfe_step": "rfe_step",
    "step": "rfe_step",
    "rfe_rf_max_features": "rfe_rf_max_features",
    "rf_max_features": "rfe_rf_max_features",
    "max_features": "rfe_rf_max_features",
}

_SELECTOR_HP_ALIAS_PARTS: dict[str, dict[str, str]] = {
    "stability": {**_SHARED_HP_ALIASES, **_STABILITY_HP_ALIASES},
    "correlation": {**_SHARED_HP_ALIASES, **_CORRELATION_HP_ALIASES},
    "sfs": {**_SHARED_HP_ALIASES, **_SFS_HP_ALIASES},
    "rfe": {**_SHARED_HP_ALIASES, **_RFE_HP_ALIASES},
}


def _canonical_selector_hp_key(selector: str, k: str) -> str:
    sel = str(selector).strip().lower()
    if sel not in _SELECTOR_HP_ALIAS_PARTS:
        raise ValueError(f"Unknown selector {selector!r} for hyperparameters")
    amap = _SELECTOR_HP_ALIAS_PARTS[sel]
    ks = str(k).strip()
    if ks in amap:
        return amap[ks]
    kl = ks.lower()
    for ak, ck in amap.items():
        if ak.lower() == kl:
            return ck
    allowed = sorted(set(amap.keys()) | set(amap.values()))
    raise ValueError(f"Unknown hyperparameter {k!r} for selector {sel!r}; allowed: {allowed}")


def _coerce_selector_hyperparameter(canon: str, v) -> object:
    if canon in (
        "correlation_use_fdr",
        "correlation_normalize",
        "correlation_make_plots",
    ):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(int(v))
        s = str(v).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"{canon} must be boolean, got {v!r}")
    if canon == "correlation_screening_min_abs_rho":
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        return float(v)
    if canon == "stability_reduction_sample_fraction":
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        if isinstance(v, str):
            s = str(v).strip().lower().replace("-", "_")
            if s in (
                FOLD_TRAIN_LOG10_CONST_5,
                "fold_train_log10",
                "fold_train_log",
                "log10_5_plus_n",
                "log10_decay",
            ) or str(v).strip() == _LEGACY_STABILITY_REDUCTION_SF_SENTINEL:
                return FOLD_TRAIN_LOG10_CONST_5
        return float(v)
    if canon in ("stability_rf_max_depth", "stability_reduction_rf_max_depth"):
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        return int(v)
    if canon == "rfe_rf_max_depth":
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        return int(v)
    if canon == "sfs_rf_max_depth":
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        return int(v)
    if canon in (
        "stability_rf_max_features",
        "stability_reduction_rf_max_features",
        "sfs_rf_max_features",
        "rfe_rf_max_features",
    ):
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        if isinstance(v, str):
            s = v.strip()
            if s.lower() in ("sqrt", "log2"):
                return s
            try:
                return float(s) if "." in s else int(s)
            except ValueError:
                return s
        return v
    if canon in ("stability_reduction_min_n_features", "correlation_reduction_min_n_features"):
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        i = int(v)
        if i < 1:
            raise ValueError(f"{canon} must be >= 1")
        return i
    if canon == "stability_reduction_n_subsamples":
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        if isinstance(v, str):
            s = str(v).strip().lower().replace("-", "_")
            if s in (
                FOLD_TRAIN_LOG10_NS_500,
                "fold_train_log10_ns500",
                "log500_over_log10_n",
            ):
                return FOLD_TRAIN_LOG10_NS_500
        return int(v)
    if canon == "stability_n_subsamples":
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        if isinstance(v, str):
            s = str(v).strip().lower().replace("-", "_")
            if s in (
                FOLD_TRAIN_LOG10_NS_500,
                "fold_train_log10_ns500",
                "log500_over_log10_n",
            ):
                return FOLD_TRAIN_LOG10_NS_500
        return int(v)
    if canon in (
        "stability_rf_n_estimators",
        "stability_rf_min_samples_leaf",
        "stability_reduction_n_features",
        "stability_reduction_rf_n_estimators",
        "stability_reduction_rf_min_samples_leaf",
        "stability_prereduction_n_features",
        "sfs_cv",
        "sfs_n_jobs",
        "sfs_rf_n_estimators",
        "sfs_rf_min_samples_leaf",
        "sfs_knn_n_neighbors",
        "rfe_step",
        "rfe_rf_n_estimators",
        "rfe_rf_min_samples_leaf",
    ):
        return int(v)
    if canon == "stability_sample_fraction":
        if v is None or (isinstance(v, str) and str(v).strip().lower() in ("", "none", "null")):
            return None
        if isinstance(v, str):
            s = str(v).strip().lower().replace("-", "_")
            if s in (
                FOLD_TRAIN_LOG10_CONST_5,
                "fold_train_log10",
                "fold_train_log",
                "log10_5_plus_n",
                "log10_decay",
            ) or str(v).strip() == _LEGACY_STABILITY_REDUCTION_SF_SENTINEL:
                return FOLD_TRAIN_LOG10_CONST_5
        return float(v)
    if canon in (
        "low_variance_relative_std_threshold",
        "low_variance_epsilon",
        "intercorr_threshold",
        "correlation_p_threshold",
        "correlation_fdr_alpha",
        "correlation_post_threshold",
        "stability_elasticnet_l1_ratio",
        "stability_elasticnet_alpha",
        "stability_coef_threshold",
        "stability_svm_c",
        "stability_svm_epsilon",
        "stability_reduction_elasticnet_l1_ratio",
        "stability_reduction_elasticnet_alpha",
        "stability_reduction_coef_threshold",
        "stability_reduction_svm_c",
        "stability_reduction_svm_epsilon",
        "sfs_min_improvement",
        "sfs_enet_alpha",
        "sfs_enet_l1_ratio",
        "sfs_svm_c",
        "sfs_svm_epsilon",
        "rfe_enet_alpha",
        "rfe_enet_l1_ratio",
        "rfe_svm_c",
        "rfe_svm_epsilon",
    ):
        return float(v)
    if canon in (
        "intercorr_importance_metric",
        "intercorr_reduction_mode",
        "correlation_post_importance_metric",
        "sfs_scoring",
        "stability_reduction_model",
    ):
        return str(v).strip()
    raise ValueError(f"Unhandled hyperparameter {canon!r}")


def parse_selector_hyperparameters_mapping(selector: str, raw: dict | None) -> dict:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("hyperparameters must be a mapping")
    sel = str(selector).strip().lower()
    if sel not in _SELECTOR_HP_ALIAS_PARTS:
        raise ValueError(f"Unsupported selector {selector!r} for hyperparameters")
    out: dict = {}
    for k, v in raw.items():
        canon = _canonical_selector_hp_key(sel, str(k))
        out[canon] = _coerce_selector_hyperparameter(canon, v)
    return out


# Minimum rows (after dropping null targets) required to run stability selection subsampling.
STABILITY_SELECTION_MIN_SAMPLES = 4


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


def _default_stability_coef_threshold(model_type: str, n_features: int) -> float:
    mt = str(model_type).strip().lower()
    if mt == "randomforest":
        return 1.0 / float(max(1, int(n_features)))
    if mt in ("elasticnet", "svm"):
        return 1e-6
    raise ValueError(f"unsupported model_type for stability: {model_type!r}")


def _stability_feature_frequencies(
    merged_df: pd.DataFrame,
    target_col: str,
    candidate_features: list[str],
    *,
    n_subsamples: int | str = FOLD_TRAIN_LOG10_NS_500,
    sample_fraction: float | str = FOLD_TRAIN_LOG10_CONST_5,
    model_type: str,
    l1_ratio: float | None = None,
    alpha: float | None = None,
    coef_threshold: float | None = None,
    svm_C: float | None = None,
    svm_epsilon: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = False,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int | None = None,
    rf_max_features=None,
    minmax_scale: bool = True,
) -> pd.Series:
    n_train = len(merged_df)
    n_subsamples = _resolve_stability_n_subsamples(n_subsamples, n_train)
    sample_fraction = _resolve_stability_sample_fraction(sample_fraction, n_train)
    X = merged_df[candidate_features].to_numpy(dtype=np.float64, copy=True)
    y = merged_df[target_col].to_numpy(dtype=np.float64, copy=True)

    if minmax_scale:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    rng = np.random.default_rng(random_state)
    n_feats = len(candidate_features)
    feature_counts = np.zeros(n_feats)
    subsize = max(2, int(len(y) * sample_fraction))

    ct = (
        float(coef_threshold)
        if coef_threshold is not None
        else _default_stability_coef_threshold(model_type, n_feats)
    )

    for i in range(n_subsamples):
        idx = rng.choice(len(y), size=subsize, replace=False)
        X_sub = X_scaled[idx]
        y_sub = y[idx]

        rs = int(rng.integers(0, 2**31))
        mr: dict = {"random_state": rs, "n_jobs": 1}
        if alpha is not None:
            mr["enet_alpha"] = alpha
        if l1_ratio is not None:
            mr["enet_l1_ratio"] = l1_ratio
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
        model = make_regressor(model_type, **mr)
        fit_regressor(model, X_sub, y_sub)
        if model_type == "elasticnet":
            w = np.ravel(np.asarray(model.coef_, dtype=np.float64))
        elif model_type == "randomforest":
            w = np.ravel(np.asarray(model.feature_importances_, dtype=np.float64))
        elif model_type == "svm":
            w = np.ravel(np.asarray(model.coef_, dtype=np.float64))
        else:
            raise ValueError(f"unsupported model_type for stability: {model_type!r}")
        if w.size != n_feats:
            raise ValueError(
                f"stability selection: coef/importance length {w.size} != n_features {n_feats}"
            )
        if model_type == "randomforest":
            non_zero = w > ct
        else:
            non_zero = np.abs(w) > ct

        feature_counts[non_zero] += 1
        if verbose and (i + 1) % 50 == 0:
            print(
                f"Stability selection target={target_col}: {i + 1}/{n_subsamples} subsamples"
            )

    return pd.Series(feature_counts / n_subsamples, index=candidate_features)


def _selected_names_from_stability_frequencies_elbow(freq: pd.Series) -> list[str]:
    freq_sorted = freq.sort_values(ascending=False)
    n = int(len(freq_sorted))
    if n == 0:
        return []
    if n == 1:
        return [freq_sorted.index[0]]
    x = np.arange(n, dtype=float)
    y_vals = freq_sorted.to_numpy(dtype=float, copy=False)
    start, end = np.array([x[0], y_vals[0]]), np.array([x[-1], y_vals[-1]])
    line_vec = end - start
    norm = np.linalg.norm(line_vec)
    if norm == 0.0:
        elbow_idx = 0
    else:
        line_dir = line_vec / norm
        pts = np.stack([x, y_vals], axis=1)
        diffs = pts - start
        proj_lengths = diffs @ line_dir
        proj_points = np.outer(proj_lengths, line_dir)
        orthogonal = diffs - proj_points
        dists = np.linalg.norm(orthogonal, axis=1)
        elbow_idx = int(np.argmax(dists))
    k_feats = max(1, elbow_idx + 1)
    return freq_sorted.iloc[:k_feats].index.tolist()


def stability_reduction_select_top_n(
    merged_df: pd.DataFrame,
    target_col: str,
    candidate_features: list[str],
    *,
    n_features: int,
    model_type: str = "randomforest",
    n_subsamples: int | str = FOLD_TRAIN_LOG10_NS_500,
    sample_fraction: float | str = FOLD_TRAIN_LOG10_CONST_5,
    l1_ratio: float | None = None,
    alpha: float | None = None,
    coef_threshold: float | None = None,
    svm_C: float | None = None,
    svm_epsilon: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = False,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int | None = None,
    rf_max_features=None,
    minmax_scale: bool = True,
) -> tuple[pd.Series, list[str]]:
    model_type = _validate_model_type(
        model_type, param_name="model_type", allowed=_ALLOWED_STABILITY_MODELS
    )
    candidate_features = list(dict.fromkeys(list(candidate_features)))
    if len(candidate_features) == 0:
        raise ValueError("candidate_features must be non-empty")
    missing = [f for f in candidate_features if f not in merged_df.columns]
    if missing:
        raise ValueError(f"candidate_features not in merged_df: {missing[:5]!r}")
    n_rows = len(merged_df)
    if n_rows < STABILITY_SELECTION_MIN_SAMPLES:
        raise ValueError(
            f"merged_df must have at least {STABILITY_SELECTION_MIN_SAMPLES} rows "
            f"for stability reduction (got {n_rows})"
        )
    k = max(1, min(int(n_features), len(candidate_features)))
    freq = _stability_feature_frequencies(
        merged_df,
        target_col,
        candidate_features,
        n_subsamples=n_subsamples,
        sample_fraction=sample_fraction,
        model_type=model_type,
        l1_ratio=l1_ratio,
        alpha=alpha,
        coef_threshold=coef_threshold,
        svm_C=svm_C,
        svm_epsilon=svm_epsilon,
        random_state=random_state,
        verbose=verbose,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_max_features=rf_max_features,
        minmax_scale=minmax_scale,
    )
    freq_sorted = freq.sort_values(ascending=False)
    selected = freq_sorted.iloc[:k].index.tolist()
    return freq, selected


def stability_selector(
    merged_df: pd.DataFrame,
    target_col: str,
    candidate_features: list[str] | None = None,
    *,
    n_subsamples: int | str = FOLD_TRAIN_LOG10_NS_500,
    sample_fraction: float | str = FOLD_TRAIN_LOG10_CONST_5,
    model_type: str = "elasticnet",
    l1_ratio: float | None = None,
    alpha: float | None = None,
    coef_threshold: float | None = None,
    svm_C: float | None = None,
    svm_epsilon: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int | None = None,
    rf_max_features=None,
    minmax_scale: bool = True,
) -> tuple[pd.Series, list[str]]:
    model_type = _validate_model_type(
        model_type, param_name="model_type", allowed=_ALLOWED_STABILITY_MODELS
    )

    if candidate_features is None:
        candidate_features = _default_candidate_feature_columns(merged_df, target_col)
    candidate_features = list(dict.fromkeys(list(candidate_features)))

    if len(candidate_features) == 0:
        raise ValueError("candidate_features must be non-empty (or omit None to infer)")

    missing = [f for f in candidate_features if f not in merged_df.columns]
    if missing:
        raise ValueError(f"candidate_features not in merged_df: {missing[:5]!r}")

    n_rows = len(merged_df)
    if n_rows < STABILITY_SELECTION_MIN_SAMPLES:
        raise ValueError(
            f"merged_df must have at least {STABILITY_SELECTION_MIN_SAMPLES} rows "
            f"for stability selection (got {n_rows})"
        )

    freq = _stability_feature_frequencies(
        merged_df,
        target_col,
        candidate_features,
        n_subsamples=n_subsamples,
        sample_fraction=sample_fraction,
        model_type=model_type,
        l1_ratio=l1_ratio,
        alpha=alpha,
        coef_threshold=coef_threshold,
        svm_C=svm_C,
        svm_epsilon=svm_epsilon,
        random_state=random_state,
        verbose=verbose,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_max_features=rf_max_features,
        minmax_scale=minmax_scale,
    )
    selected = _selected_names_from_stability_frequencies_elbow(freq)
    return freq, selected


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

        prev_best_score = best_score
        selected.append(best_feat)
        remaining.remove(best_feat)

        if len(remaining) == 0:
            break

    if len(selected) == 0:
        selected = list(X.columns[:1])

    return selected


def select_features_floating_sfs(
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
            "Unsupported scoring for floating selection. "
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

    def _cv_score_for_feature_set(feature_set: list[str]) -> float:
        if len(feature_set) == 0:
            return -np.inf

        fold_scores = []
        X_use = X[feature_set]
        for train_idx, val_idx in folds:
            X_tr = X_use.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X_use.iloc[val_idx]
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
            return -np.inf
        return float(np.mean(fold_scores))

    remaining = list(X.columns)
    selected: list[str] = []
    current_score = -np.inf
    max_outer = max(n_avail * n_select, n_avail) * 4 + 10
    outer_steps = 0

    while len(selected) < n_select and outer_steps < max_outer:
        outer_steps += 1

        best_feat = None
        best_score = -np.inf
        for feat in remaining:
            candidate_set = selected + [feat]
            score = _cv_score_for_feature_set(candidate_set)
            if score > best_score:
                best_score = score
                best_feat = feat

        if best_feat is None or not np.isfinite(best_score):
            break

        if current_score != -np.inf:
            improvement = best_score - current_score
            if improvement < min_imp_eff:
                break

        selected.append(best_feat)
        remaining.remove(best_feat)
        current_score = best_score

        while len(selected) > 1:
            best_rm_feat = None
            best_rm_score = current_score

            for feat in selected:
                subset = [f for f in selected if f != feat]
                score = _cv_score_for_feature_set(subset)
                if score > best_rm_score:
                    best_rm_score = score
                    best_rm_feat = feat

            if best_rm_feat is None:
                break
            if (best_rm_score - current_score) < min_imp_eff:
                break

            selected.remove(best_rm_feat)
            remaining.append(best_rm_feat)
            current_score = best_rm_score

    if len(selected) == 0:
        selected = list(X.columns[:1])

    return selected


def recursive_elimination_selector(
    split_train_df,
    target_col: str,
    n_features_to_select: int,
    candidate_features: list | None = None,
    rfe_estimator: str = "elasticnet",
    random_state: int = DEFAULT_RANDOM_STATE,
    enet_alpha: float | None = None,
    enet_l1_ratio: float | None = None,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int | None = None,
    rf_max_features=None,
    svm_C: float | None = None,
    svm_epsilon: float | None = None,
    step: int | None = None,
):
    est = _validate_model_type(
        rfe_estimator, param_name="rfe_estimator", allowed=_ALLOWED_RFE_MODELS
    )

    if candidate_features is None:
        candidate_features = _default_candidate_feature_columns(
            split_train_df, target_col
        )
    feats_t = list(dict.fromkeys(list(candidate_features)))

    if len(feats_t) == 0:
        return []

    cols_t = feats_t + [target_col]
    work_t = split_train_df.loc[:, cols_t].astype(np.float64, copy=True)
    if len(work_t) < 3:
        return feats_t[:1]

    X_t = work_t[feats_t]
    y_t = work_t[target_col]

    if len(X_t) < 3:
        return feats_t[:1]

    n_select_t = max(1, min(int(n_features_to_select), X_t.shape[1]))

    mr: dict = {"random_state": random_state, "n_jobs": -1}
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
    estimator = make_regressor(est, **mr)
    step_t = max(1, int(step if step is not None else 1))
    selector = RFE(estimator=estimator, n_features_to_select=n_select_t, step=step_t)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        selector.fit(X_t, y_t)
    selected_cols_t = X_t.columns[selector.support_].tolist()
    return selected_cols_t if len(selected_cols_t) > 0 else feats_t[:1]


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


# Backward-compatible name (notebooks, older call sites).
select_features_by_target_correlation = correlation_selector

_VALID_PIPELINE_STEPS = frozenset({"stability", "correlation", "sfs", "rfe"})


def cv_shuffled_fold_ilocs(
    n_samples: int,
    n_splits: int,
    random_state: int,
) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray]]]:
    N = int(n_samples)
    if n_splits < 2 or n_splits > N:
        raise ValueError(f"n_splits must be between 2 and N={N}.")

    base_size = N // n_splits
    remainder = N % n_splits
    segment_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_splits)]
    boundaries = np.cumsum([0] + segment_sizes)

    rng = np.random.default_rng(random_state)
    idx = np.arange(N)
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


def _parse_feature_selection_pipeline(feature_selection_pipeline: str) -> list[str]:
    pipeline_steps = [s.strip() for s in feature_selection_pipeline.split("->") if s.strip()]
    if len(pipeline_steps) == 0:
        raise ValueError("feature_selection_pipeline must name one selector step.")
    if len(pipeline_steps) > 1:
        raise ValueError(
            "feature_selection_pipeline must be a single step (no '->'). "
            f"Use one of: {sorted(_VALID_PIPELINE_STEPS)}"
        )
    invalid = [s for s in pipeline_steps if s not in _VALID_PIPELINE_STEPS]
    if invalid:
        raise ValueError(
            f"Invalid pipeline step(s): {invalid}. Allowed: {sorted(_VALID_PIPELINE_STEPS)}"
        )
    return pipeline_steps


_PIPELINE_STEPS_NEEDING_MODEL = frozenset({"stability", "sfs", "rfe"})


def _normalize_selection_estimator(name: str) -> str:
    n = str(name).strip().lower()
    allowed = {"elasticnet", "randomforest", "svm", "knn"}
    if n not in allowed:
        raise ValueError(
            f"Unknown selection_estimator {name!r}. Use one of: {sorted(allowed)}"
        )
    return n


def run_feature_selection_on_one_fold(
    train_df_k: pd.DataFrame,
    test_df_k: pd.DataFrame,
    *,
    target_col: str,
    feature_selection_pipeline: str,
    model_type: str = "elasticnet",
    candidate_features: list[str] | None = None,
    low_variance_relative_std_threshold: float = DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD,
    low_variance_epsilon: float = DEFAULT_LOW_VARIANCE_EPSILON,
    intercorr_threshold: float = DEFAULT_INTERCORR_THRESHOLD,
    intercorr_importance_metric: str = DEFAULT_INTERCORR_IMPORTANCE_METRIC,
    intercorr_reduction_mode: str = DEFAULT_INTERCORR_REDUCTION_MODE,
    correlation_reduction_min_n_features: int | None = None,
    stability_reduction_n_features: int | None = None,
    stability_reduction_min_n_features: int | None = None,
    stability_reduction_model: str | None = None,
    stability_reduction_n_subsamples: int | str | None = None,
    stability_reduction_sample_fraction: float | str | None = None,
    stability_reduction_coef_threshold: float | None = None,
    stability_reduction_elasticnet_l1_ratio: float | None = None,
    stability_reduction_elasticnet_alpha: float | None = None,
    stability_reduction_svm_c: float | None = None,
    stability_reduction_svm_epsilon: float | None = None,
    stability_reduction_rf_n_estimators: int | None = None,
    stability_reduction_rf_max_depth: int | None = None,
    stability_reduction_rf_min_samples_leaf: int | None = None,
    stability_reduction_rf_max_features: str | float | int | None = None,
    stability_prereduction_n_features: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    stability_n_subsamples: int | str = FOLD_TRAIN_LOG10_NS_500,
    stability_sample_fraction: float | str = FOLD_TRAIN_LOG10_CONST_5,
    stability_elasticnet_l1_ratio: float | None = None,
    stability_elasticnet_alpha: float | None = None,
    stability_coef_threshold: float | None = None,
    stability_svm_c: float | None = None,
    stability_svm_epsilon: float | None = None,
    stability_rf_n_estimators: int | None = None,
    stability_rf_max_depth: int | None = None,
    stability_rf_min_samples_leaf: int | None = None,
    stability_rf_max_features: str | float | int | None = None,
    correlation_p_threshold: float | None = None,
    correlation_fdr_alpha: float | None = None,
    correlation_use_fdr: bool | None = None,
    correlation_normalize: bool | None = None,
    correlation_make_plots: bool | None = None,
    correlation_screening_min_abs_rho: float | None = None,
    correlation_post_threshold: float | None = None,
    correlation_post_importance_metric: str | None = None,
    sfs_cv: int | None = None,
    sfs_scoring: str | None = None,
    sfs_min_improvement: float | None = None,
    sfs_n_jobs: int | None = None,
    sfs_enet_alpha: float | None = None,
    sfs_enet_l1_ratio: float | None = None,
    sfs_svm_c: float | None = None,
    sfs_svm_epsilon: float | None = None,
    sfs_rf_n_estimators: int | None = None,
    sfs_rf_max_depth: int | None = None,
    sfs_rf_min_samples_leaf: int | None = None,
    sfs_rf_max_features: str | float | int | None = None,
    sfs_knn_n_neighbors: int | None = None,
    rfe_enet_alpha: float | None = None,
    rfe_enet_l1_ratio: float | None = None,
    rfe_rf_n_estimators: int | None = None,
    rfe_rf_max_depth: int | None = None,
    rfe_rf_min_samples_leaf: int | None = None,
    rfe_rf_max_features: str | float | int | None = None,
    rfe_svm_c: float | None = None,
    rfe_svm_epsilon: float | None = None,
    rfe_step: int | None = None,
    verbose: bool = False,
    selection_max_features: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pipeline_steps = _parse_feature_selection_pipeline(feature_selection_pipeline)
    needs_model = bool(set(pipeline_steps) & _PIPELINE_STEPS_NEEDING_MODEL)
    if needs_model:
        model_type = _normalize_selection_estimator(model_type)
    else:
        model_type = str(model_type).strip().lower()

    irm = str(intercorr_reduction_mode).strip().lower()
    if irm not in ("cluster", "pairwise"):
        raise ValueError("intercorr_reduction_mode must be 'cluster' or 'pairwise'")

    intercorr_importance_metric = str(intercorr_importance_metric).strip().lower()
    if intercorr_importance_metric not in ("spearman", "pearson"):
        raise ValueError(
            "intercorr_importance_metric must be 'spearman' or 'pearson'"
        )
    if correlation_post_importance_metric is not None:
        correlation_post_importance_metric = str(
            correlation_post_importance_metric
        ).strip().lower()
        if correlation_post_importance_metric not in ("spearman", "pearson"):
            raise ValueError(
                "correlation_post_importance_metric must be 'spearman' or 'pearson'"
            )

    tcol = str(target_col)
    if tcol not in train_df_k.columns:
        raise ValueError(f"target_col {tcol!r} not in train dataframe columns")

    train_df_k = train_df_k.copy()
    test_df_k = test_df_k.copy()

    if candidate_features is None:
        cand = [
            c
            for c in train_df_k.columns
            if not (str(c).startswith("target") or c == "name")
        ]
    else:
        cols_set = set(train_df_k.columns)
        missing = [c for c in candidate_features if c not in cols_set]
        if missing:
            raise ValueError(
                f"candidate_features not in train_df_k.columns: {missing!r}"
            )
        cand = list(dict.fromkeys(candidate_features))
        if len(cand) == 0:
            raise ValueError("candidate_features must be non-empty when provided")

    train_df_k, test_df_k = apply_minmax_to_train_test_features(
        train_df_k, test_df_k, cand
    )

    kept_k, removed_k, rel_std_k = remove_low_variance_features(
        X=train_df_k,
        candidate_features=cand,
        relative_std_threshold=low_variance_relative_std_threshold,
        epsilon=low_variance_epsilon,
    )

    if len(kept_k) == 0:
        finite_feats = rel_std_k.index[np.isfinite(rel_std_k.values)].tolist()
        kept_k = finite_feats if len(finite_feats) > 0 else cand[:1]

    kept_k_set = set(kept_k)
    removed_k = [c for c in cand if c not in kept_k_set]

    train_df_k = train_df_k.drop(columns=removed_k, errors="ignore").copy()
    test_df_k = test_df_k.drop(columns=removed_k, errors="ignore").copy()

    prefilter_k = reduce_correlated_features(
        split_train_df=train_df_k,
        target_col=tcol,
        candidate_features=kept_k,
        correlation_threshold=intercorr_threshold,
        reduction_mode=irm,
        importance_metric=intercorr_importance_metric,
    )

    current_features = list(prefilter_k)
    if len(current_features) == 0:
        current_features = list(kept_k[:1])

    features_after_correlation_reduction = list(current_features)
    correlation_reduction_summary: dict | None = None
    if correlation_reduction_min_n_features is not None:
        mn_cr = int(correlation_reduction_min_n_features)
        if mn_cr < 1:
            raise ValueError("correlation_reduction_min_n_features must be >= 1")
        n_pref_cr = len(current_features)
        if n_pref_cr <= 1:
            correlation_reduction_summary = {
                "skipped": True,
                "reason": "n_prefilter_features_at_or_below_minimum_for_elbow",
                "n_prefilter_features": n_pref_cr,
                "mode": "spearman_abs_elbow",
                "min_n_features": mn_cr,
                "features_after_reduction": list(current_features),
            }
        else:
            bundle_cr = compute_correlation_bundle(
                train_df_k,
                tcol,
                list(current_features),
                exclude_cols=DEFAULT_PRUNE_EXCLUDE_COLS,
                id_like_cols=frozenset(),
                corr_method="spearman",
            )
            rho = bundle_cr.target_feature_rho.reindex(list(current_features))[tcol]
            spearman_abs = rho.abs().fillna(0.0)
            elbow_list = _selected_names_from_stability_frequencies_elbow(spearman_abs)
            elbow_n = len(elbow_list)
            k_keep = max(mn_cr, elbow_n)
            k_keep = min(k_keep, n_pref_cr)
            ranked = spearman_abs.sort_values(ascending=False)
            current_features = ranked.iloc[:k_keep].index.tolist()
            if len(current_features) == 0:
                current_features = [ranked.index[0]]
            features_after_correlation_reduction = list(current_features)
            correlation_reduction_summary = {
                "skipped": False,
                "mode": "spearman_abs_elbow",
                "n_prefilter_features": n_pref_cr,
                "n_elbow": elbow_n,
                "min_n_features": mn_cr,
                "n_after_reduction": len(current_features),
                "features_after_reduction": list(current_features),
            }

    features_after_prereduction = list(current_features)
    stability_prereduction_summary: dict | None = None
    if stability_prereduction_n_features is not None:
        n_pre_cap = int(stability_prereduction_n_features)
        if n_pre_cap < 1:
            raise ValueError("stability_prereduction_n_features must be >= 1")
        n_pref_pre = len(current_features)
        if n_pref_pre <= n_pre_cap:
            stability_prereduction_summary = {
                "skipped": True,
                "reason": "n_prefilter_features_at_or_below_cap",
                "n_prefilter_features": n_pref_pre,
                "n_features_cap": n_pre_cap,
                "features_after_prereduction": list(current_features),
            }
        elif n_pref_pre <= 1:
            stability_prereduction_summary = {
                "skipped": True,
                "reason": "n_prefilter_features_at_or_below_minimum_for_stability",
                "n_prefilter_features": n_pref_pre,
                "features_after_prereduction": list(current_features),
            }
        elif len(train_df_k) < STABILITY_SELECTION_MIN_SAMPLES:
            stability_prereduction_summary = {
                "skipped": True,
                "reason": "insufficient_train_rows_for_stability",
                "n_train": int(len(train_df_k)),
                "n_prefilter_features": n_pref_pre,
                "features_after_prereduction": list(current_features),
            }
        else:
            md_pre = stability_reduction_model
            if md_pre is None or str(md_pre).strip() == "":
                md_pre = "elasticnet"
            sr_model_pre = _validate_model_type(
                str(md_pre),
                param_name="stability_prereduction(backing_model)",
                allowed=_ALLOWED_STABILITY_MODELS,
            )
            pre_ns = STABILITY_PREREDUCTION_N_SUBSAMPLES
            pre_sf = stability_reduction_subsample_fraction_fold_train_log10(
                len(train_df_k)
            )
            try:
                _, sel_pre = stability_reduction_select_top_n(
                    train_df_k,
                    tcol,
                    list(current_features),
                    n_features=n_pre_cap,
                    model_type=sr_model_pre,
                    n_subsamples=pre_ns,
                    sample_fraction=pre_sf,
                    l1_ratio=stability_reduction_elasticnet_l1_ratio,
                    alpha=stability_reduction_elasticnet_alpha,
                    coef_threshold=stability_reduction_coef_threshold,
                    svm_C=stability_reduction_svm_c,
                    svm_epsilon=stability_reduction_svm_epsilon,
                    random_state=random_state,
                    verbose=verbose,
                    rf_n_estimators=stability_reduction_rf_n_estimators,
                    rf_max_depth=stability_reduction_rf_max_depth,
                    rf_min_samples_leaf=stability_reduction_rf_min_samples_leaf,
                    rf_max_features=stability_reduction_rf_max_features,
                    minmax_scale=False,
                )
                if len(sel_pre) > 0:
                    current_features = list(sel_pre)
                features_after_prereduction = list(current_features)
                stability_prereduction_summary = {
                    "skipped": False,
                    "n_prefilter_features": n_pref_pre,
                    "n_after_reduction": len(current_features),
                    "n_features_cap": n_pre_cap,
                    "model_type": sr_model_pre,
                    "n_subsamples": pre_ns,
                    "sample_fraction": pre_sf,
                    "n_train": int(len(train_df_k)),
                    "features_after_prereduction": list(current_features),
                }
            except Exception as e:
                if verbose:
                    print(f"Stability prereduction skipped for target={tcol!r}: {e}")
                stability_prereduction_summary = {
                    "skipped": True,
                    "reason": "error",
                    "error": str(e),
                    "n_prefilter_features": n_pref_pre,
                    "n_features_cap": n_pre_cap,
                    "features_after_prereduction": list(current_features),
                }

    stability_reduction_summary: dict | None = None
    _sr_cap_set = stability_reduction_n_features is not None
    _sr_elbow = (
        not _sr_cap_set
        and stability_reduction_model is not None
        and str(stability_reduction_model).strip() != ""
    )
    if _sr_cap_set or _sr_elbow:
        if stability_reduction_model is None or str(stability_reduction_model).strip() == "":
            stability_reduction_model = "elasticnet"
        sr_model = _validate_model_type(
            str(stability_reduction_model),
            param_name="stability_reduction_model",
            allowed=_ALLOWED_STABILITY_MODELS,
        )
        if _sr_cap_set:
            n_cap = int(stability_reduction_n_features)
            if n_cap < 1:
                raise ValueError("stability_reduction_n_features must be >= 1")
        else:
            n_cap = None

        n_pref = len(current_features)
        if _sr_cap_set and n_pref <= n_cap:
            stability_reduction_summary = {
                "skipped": True,
                "reason": "n_prefilter_features_at_or_below_cap",
                "n_prefilter_features": n_pref,
                "n_features_cap": n_cap,
                "stability_reduction_mode": "top_n",
                "model_type": sr_model,
                "features_after_reduction": list(current_features),
            }
        elif _sr_elbow and n_pref <= 1:
            stability_reduction_summary = {
                "skipped": True,
                "reason": "n_prefilter_features_at_or_below_minimum_for_elbow",
                "n_prefilter_features": n_pref,
                "stability_reduction_mode": "elbow",
                "model_type": sr_model,
                "features_after_reduction": list(current_features),
            }
        else:
            _sr_ns_spec = stability_reduction_n_subsamples
            if _sr_ns_spec == FOLD_TRAIN_LOG10_NS_500:
                sr_ns = stability_reduction_n_subsamples_fold_train_log500(len(train_df_k))
                _sr_ns_rule = FOLD_TRAIN_LOG10_NS_500
            elif stability_reduction_n_subsamples is not None:
                sr_ns = int(stability_reduction_n_subsamples)
                _sr_ns_rule = "fixed"
            else:
                sr_ns = stability_reduction_n_subsamples_fold_train_log500(len(train_df_k))
                _sr_ns_rule = FOLD_TRAIN_LOG10_NS_500
            _sr_sf_spec = stability_reduction_sample_fraction
            if _sr_sf_spec in (
                FOLD_TRAIN_LOG10_CONST_5,
                _LEGACY_STABILITY_REDUCTION_SF_SENTINEL,
            ):
                sr_sf = stability_reduction_subsample_fraction_fold_train_log10(
                    len(train_df_k)
                )
                _sr_sf_rule = FOLD_TRAIN_LOG10_CONST_5
            elif stability_reduction_sample_fraction is not None:
                sr_sf = float(stability_reduction_sample_fraction)
                _sr_sf_rule = "fixed"
            else:
                sr_sf = stability_reduction_subsample_fraction_fold_train_log10(
                    len(train_df_k)
                )
                _sr_sf_rule = FOLD_TRAIN_LOG10_CONST_5
            try:
                if _sr_cap_set:
                    _, sel_sr = stability_reduction_select_top_n(
                        train_df_k,
                        tcol,
                        list(current_features),
                        n_features=n_cap,
                        model_type=sr_model,
                        n_subsamples=sr_ns,
                        sample_fraction=sr_sf,
                        l1_ratio=stability_reduction_elasticnet_l1_ratio,
                        alpha=stability_reduction_elasticnet_alpha,
                        coef_threshold=stability_reduction_coef_threshold,
                        svm_C=stability_reduction_svm_c,
                        svm_epsilon=stability_reduction_svm_epsilon,
                        random_state=random_state,
                        verbose=verbose,
                        rf_n_estimators=stability_reduction_rf_n_estimators,
                        rf_max_depth=stability_reduction_rf_max_depth,
                        rf_min_samples_leaf=stability_reduction_rf_min_samples_leaf,
                        rf_max_features=stability_reduction_rf_max_features,
                        minmax_scale=False,
                    )
                else:
                    freq_sr, sel_elbow = stability_selector(
                        train_df_k,
                        tcol,
                        list(current_features),
                        n_subsamples=sr_ns,
                        sample_fraction=sr_sf,
                        model_type=sr_model,
                        l1_ratio=stability_reduction_elasticnet_l1_ratio,
                        alpha=stability_reduction_elasticnet_alpha,
                        coef_threshold=stability_reduction_coef_threshold,
                        svm_C=stability_reduction_svm_c,
                        svm_epsilon=stability_reduction_svm_epsilon,
                        random_state=random_state,
                        verbose=verbose,
                        rf_n_estimators=stability_reduction_rf_n_estimators,
                        rf_max_depth=stability_reduction_rf_max_depth,
                        rf_min_samples_leaf=stability_reduction_rf_min_samples_leaf,
                        rf_max_features=stability_reduction_rf_max_features,
                        minmax_scale=False,
                    )
                    elbow_n = len(sel_elbow)
                    if stability_reduction_min_n_features is not None:
                        mn_i = int(stability_reduction_min_n_features)
                        if mn_i < 1:
                            raise ValueError(
                                "stability_reduction_min_n_features must be >= 1"
                            )
                        k_keep = max(mn_i, elbow_n)
                    else:
                        mn_i = None
                        k_keep = elbow_n
                    k_keep = min(k_keep, n_pref)
                    freq_sorted_sr = freq_sr.sort_values(ascending=False)
                    sel_sr = freq_sorted_sr.iloc[:k_keep].index.tolist()
                if len(sel_sr) > 0:
                    current_features = list(sel_sr)
                stability_reduction_summary = {
                    "skipped": False,
                    "n_prefilter_features": n_pref,
                    "n_after_reduction": len(current_features),
                    "stability_reduction_mode": "top_n" if _sr_cap_set else "elbow",
                    "model_type": sr_model,
                    "n_subsamples": sr_ns,
                    "n_subsamples_rule": _sr_ns_rule,
                    "n_train_for_n_subsamples": int(len(train_df_k)),
                    "sample_fraction": sr_sf,
                    "sample_fraction_rule": _sr_sf_rule,
                    "n_train_for_sample_fraction": int(len(train_df_k)),
                    "features_after_reduction": list(current_features),
                }
                if _sr_cap_set:
                    stability_reduction_summary["n_features_cap"] = n_cap
                else:
                    stability_reduction_summary["n_elbow"] = elbow_n
                    if stability_reduction_min_n_features is not None:
                        stability_reduction_summary["min_n_features"] = mn_i
            except Exception as e:
                if verbose:
                    print(f"Stability reduction skipped for target={tcol!r}: {e}")
                stability_reduction_summary = {
                    "skipped": True,
                    "reason": "error",
                    "error": str(e),
                    "n_prefilter_features": n_pref,
                    "stability_reduction_mode": "top_n" if _sr_cap_set else "elbow",
                    "model_type": sr_model,
                    "n_subsamples": sr_ns,
                    "n_subsamples_rule": _sr_ns_rule,
                    "n_train_for_n_subsamples": int(len(train_df_k)),
                    "sample_fraction": sr_sf,
                    "sample_fraction_rule": _sr_sf_rule,
                    "n_train_for_sample_fraction": int(len(train_df_k)),
                    "features_after_reduction": list(current_features),
                }
                if _sr_cap_set:
                    stability_reduction_summary["n_features_cap"] = n_cap

    features_after_stability_reduction = list(current_features)

    after_step: dict[str, list[str]] = {}
    correlation_selection_summary: dict | None = None
    stability_selector_subsampling: dict | None = None

    for step in pipeline_steps:
        if step == "stability":
            union_feats = sorted(current_features)
            if len(union_feats) == 0:
                union_feats = list(kept_k[:1])
            _stab_ns_spec = stability_n_subsamples
            if _stab_ns_spec == FOLD_TRAIN_LOG10_NS_500:
                stab_ns = stability_reduction_n_subsamples_fold_train_log500(len(train_df_k))
                stab_ns_rule = FOLD_TRAIN_LOG10_NS_500
            else:
                stab_ns = int(stability_n_subsamples)
                stab_ns_rule = "fixed"
            _stab_sf_spec = stability_sample_fraction
            if _stab_sf_spec in (
                FOLD_TRAIN_LOG10_CONST_5,
                _LEGACY_STABILITY_REDUCTION_SF_SENTINEL,
            ):
                stab_sf = stability_reduction_subsample_fraction_fold_train_log10(
                    len(train_df_k)
                )
                stab_sf_rule = FOLD_TRAIN_LOG10_CONST_5
            else:
                stab_sf = float(stability_sample_fraction)
                stab_sf_rule = "fixed"
            stability_selector_subsampling = {
                "n_subsamples": stab_ns,
                "n_subsamples_rule": stab_ns_rule,
                "n_train": int(len(train_df_k)),
                "sample_fraction": stab_sf,
                "sample_fraction_rule": stab_sf_rule,
            }
            try:
                freq, sel = stability_selector(
                    train_df_k,
                    tcol,
                    union_feats,
                    n_subsamples=stab_ns,
                    sample_fraction=stab_sf,
                    model_type=model_type,
                    l1_ratio=stability_elasticnet_l1_ratio,
                    alpha=stability_elasticnet_alpha,
                    coef_threshold=stability_coef_threshold,
                    svm_C=stability_svm_c,
                    svm_epsilon=stability_svm_epsilon,
                    rf_n_estimators=stability_rf_n_estimators,
                    rf_max_depth=stability_rf_max_depth,
                    rf_min_samples_leaf=stability_rf_min_samples_leaf,
                    rf_max_features=stability_rf_max_features,
                    random_state=random_state,
                    verbose=verbose,
                    minmax_scale=False,
                )
                if len(sel) > 0:
                    # sel is already sorted by descending selection frequency
                    if selection_max_features is not None and len(sel) > selection_max_features:
                        sel = sel[:selection_max_features]
                    current_features = list(sel)
            except ValueError as e:
                if verbose:
                    print(f"Stability selection cannot be run for target={tcol!r}: {e}")

            after_step["stability"] = list(current_features)

        elif step == "correlation":
            ccp: dict = {}
            if correlation_p_threshold is not None:
                ccp["p_threshold"] = correlation_p_threshold
            if correlation_fdr_alpha is not None:
                ccp["fdr_alpha"] = correlation_fdr_alpha
            if correlation_use_fdr is not None:
                ccp["use_fdr"] = correlation_use_fdr
            if correlation_normalize is not None:
                ccp["normalize"] = correlation_normalize
            if correlation_make_plots is not None:
                ccp["make_plots"] = correlation_make_plots
            _, sig_tuples = calculate_correlations_and_plot(
                merged_df=train_df_k,
                target_col=tcol,
                correlation_threshold=correlation_screening_min_abs_rho,
                candidate_features=current_features,
                verbose=verbose,
                **ccp,
            )

            n_sig_raw = len(sig_tuples)
            # sig_features is ordered by descending |rho| (from calculate_correlations_and_plot)
            sig_features = [el[0] for el in sig_tuples]
            allowed = set(current_features)
            sig_filtered = [f for f in sig_features if f in allowed]

            if len(sig_filtered) == 0:
                # No significant features found – fall back to capped prefilter output.
                if selection_max_features is not None and len(current_features) > selection_max_features:
                    current_features = current_features[:selection_max_features]
                correlation_selection_summary = {
                    "skipped": True,
                    "reason": (
                        "no_significant_target_correlations"
                        if n_sig_raw == 0
                        else "significant_features_not_in_prefilter_allowlist"
                    ),
                    "n_significant_before_allowlist": n_sig_raw,
                    "n_after_allowlist": 0,
                    "n_after_frac_cap": len(current_features),
                    "min_abs_rho_config": correlation_screening_min_abs_rho,
                }
                after_step["correlation"] = list(current_features)
            else:
                rcw: dict = {"reduction_mode": irm}
                if correlation_post_threshold is not None:
                    rcw["correlation_threshold"] = correlation_post_threshold
                if correlation_post_importance_metric is not None:
                    rcw["importance_metric"] = correlation_post_importance_metric
                reduced_corr = reduce_correlated_features(
                    split_train_df=train_df_k,
                    target_col=tcol,
                    candidate_features=sig_filtered,
                    **rcw,
                )
                n_after_reduction = len(reduced_corr)
                if len(reduced_corr) > 0:
                    # Re-order survivors by original |rho| rank and apply cap.
                    reduced_set = set(reduced_corr)
                    reduced_by_rho = [f for f in sig_filtered if f in reduced_set]
                    if selection_max_features is not None and len(reduced_by_rho) > selection_max_features:
                        reduced_by_rho = reduced_by_rho[:selection_max_features]
                    current_features = reduced_by_rho
                else:
                    # Inter-feature reduction removed everything – cap the prefilter fallback.
                    if selection_max_features is not None and len(current_features) > selection_max_features:
                        current_features = current_features[:selection_max_features]

                correlation_selection_summary = {
                    "skipped": False,
                    "n_significant_before_allowlist": n_sig_raw,
                    "n_after_allowlist": len(sig_filtered),
                    "n_after_interfeature_reduction": n_after_reduction,
                    "n_after_frac_cap": len(current_features),
                    "min_abs_rho_config": correlation_screening_min_abs_rho,
                }

                after_step["correlation"] = list(current_features)

        elif step == "sfs":
            # Upper bound on features to add; early stopping via min_improvement is fine.
            n_avail = len(current_features)
            if selection_max_features is not None:
                n_to_select = max(1, min(int(selection_max_features), n_avail))
            else:
                n_to_select = n_avail
            try:
                sfs_call: dict = {
                    "n_features_to_select": n_to_select,
                    "random_state": random_state,
                    "model_type": model_type,
                }
                if sfs_cv is not None:
                    sfs_call["cv"] = sfs_cv
                if sfs_scoring is not None:
                    sfs_call["scoring"] = sfs_scoring
                if sfs_min_improvement is not None:
                    sfs_call["min_improvement"] = sfs_min_improvement
                if sfs_n_jobs is not None:
                    sfs_call["n_jobs"] = sfs_n_jobs
                if sfs_enet_alpha is not None:
                    sfs_call["enet_alpha"] = sfs_enet_alpha
                if sfs_enet_l1_ratio is not None:
                    sfs_call["enet_l1_ratio"] = sfs_enet_l1_ratio
                if sfs_svm_c is not None:
                    sfs_call["svm_C"] = sfs_svm_c
                if sfs_svm_epsilon is not None:
                    sfs_call["svm_epsilon"] = sfs_svm_epsilon
                if sfs_rf_n_estimators is not None:
                    sfs_call["rf_n_estimators"] = sfs_rf_n_estimators
                if sfs_rf_max_depth is not None:
                    sfs_call["rf_max_depth"] = sfs_rf_max_depth
                if sfs_rf_min_samples_leaf is not None:
                    sfs_call["rf_min_samples_leaf"] = sfs_rf_min_samples_leaf
                if sfs_rf_max_features is not None:
                    sfs_call["rf_max_features"] = sfs_rf_max_features
                if sfs_knn_n_neighbors is not None:
                    sfs_call["knn_n_neighbors"] = sfs_knn_n_neighbors
                sel = sequential_forward_selector(
                    train_df_k,
                    tcol,
                    current_features,
                    **sfs_call,
                )
                if len(sel) > 0:
                    current_features = list(sel)
            except Exception as e:
                if verbose:
                    print(f"SFS cannot be run for target={tcol!r}: {e}")
            after_step["sfs"] = list(current_features)

        elif step == "rfe":
            # Must always eliminate down to the allowed count.
            n_avail = len(current_features)
            if selection_max_features is not None:
                n_to_select = max(1, min(int(selection_max_features), n_avail))
            else:
                n_to_select = max(1, n_avail)
            # RFE requires coef_ or feature_importances_; knn is not supported.
            rfe_model_type = model_type
            if rfe_model_type == "knn":
                rfe_model_type = "elasticnet"
                if verbose:
                    print(
                        f"RFE does not support knn; falling back to elasticnet "
                        f"for target={tcol!r}."
                    )
            try:
                rfe_call: dict = {
                    "n_features_to_select": n_to_select,
                    "candidate_features": current_features,
                    "rfe_estimator": rfe_model_type,
                    "random_state": random_state,
                }
                if rfe_enet_alpha is not None:
                    rfe_call["enet_alpha"] = rfe_enet_alpha
                if rfe_enet_l1_ratio is not None:
                    rfe_call["enet_l1_ratio"] = rfe_enet_l1_ratio
                if rfe_rf_n_estimators is not None:
                    rfe_call["rf_n_estimators"] = rfe_rf_n_estimators
                if rfe_rf_max_depth is not None:
                    rfe_call["rf_max_depth"] = rfe_rf_max_depth
                if rfe_rf_min_samples_leaf is not None:
                    rfe_call["rf_min_samples_leaf"] = rfe_rf_min_samples_leaf
                if rfe_rf_max_features is not None:
                    rfe_call["rf_max_features"] = rfe_rf_max_features
                if rfe_svm_c is not None:
                    rfe_call["svm_C"] = rfe_svm_c
                if rfe_svm_epsilon is not None:
                    rfe_call["svm_epsilon"] = rfe_svm_epsilon
                if rfe_step is not None:
                    rfe_call["step"] = rfe_step
                sel = recursive_elimination_selector(
                    train_df_k,
                    tcol,
                    **rfe_call,
                )
                if len(sel) > 0:
                    current_features = list(sel)
            except Exception as e:
                if verbose:
                    print(f"RFE cannot be run for target={tcol!r}: {e}")
            after_step["rfe"] = list(current_features)

    prefilter_list = list(prefilter_k) if not isinstance(prefilter_k, list) else list(prefilter_k)

    result = {
        "target_col": tcol,
        "feature_scaling": "minmax_train_fit_transform_test",
        "selected_features": list(current_features),
        "n_selected_features": len(current_features),
        "features_lowvar": list(kept_k),
        "features_intercorr_prefilter": prefilter_list,
        "features_intercorr_prefilter_count": len(prefilter_list),
        "features_after_correlation_reduction": features_after_correlation_reduction,
        "features_after_correlation_reduction_count": len(
            features_after_correlation_reduction
        ),
        "features_after_prereduction": features_after_prereduction,
        "features_after_prereduction_count": len(features_after_prereduction),
        "features_after_stability_reduction": features_after_stability_reduction,
        "features_after_stability_reduction_count": len(
            features_after_stability_reduction
        ),
        "after_step": after_step,
        "feature_selection_pipeline": feature_selection_pipeline,
        "model_type": model_type,
        "selection_max_features": selection_max_features,
    }
    if correlation_reduction_summary is not None:
        result["correlation_reduction_summary"] = correlation_reduction_summary
    if stability_reduction_summary is not None:
        result["stability_reduction_summary"] = stability_reduction_summary
    if stability_prereduction_summary is not None:
        result["stability_prereduction_summary"] = stability_prereduction_summary
    if stability_selector_subsampling is not None:
        result["stability_selector_subsampling"] = stability_selector_subsampling
    if correlation_selection_summary is not None:
        result["correlation_selection_summary"] = correlation_selection_summary
    return train_df_k, test_df_k, result


__all__ = [
    "parse_selector_hyperparameters_mapping",
    "DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS",
    "DEFAULT_CORRELATION_SCREENING_ID_COLS",
    "DEFAULT_PRUNE_EXCLUDE_COLS",
    "CorrelationBundle",
    "compute_correlation_bundle",
    "correlation_selector",
    "select_features_by_target_correlation",
    "reduce_correlated_features",
    "make_regressor",
    "remove_low_variance_features",
    "stability_selector",
    "stability_reduction_select_top_n",
    "FOLD_TRAIN_LOG10_CONST_5",
    "FOLD_TRAIN_LOG10_NS_500",
    "STABILITY_PREREDUCTION_N_SUBSAMPLES",
    "stability_reduction_subsample_fraction_fold_train_log10",
    "stability_reduction_n_subsamples_fold_train_log500",
    "sequential_forward_selector",
    "select_features_floating_sfs",
    "recursive_elimination_selector",
    "cv_shuffled_fold_ilocs",
    "run_feature_selection_on_one_fold",
    "calculate_correlations_and_plot",
]
