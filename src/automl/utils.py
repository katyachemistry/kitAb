"""Correlation matrices, developability AutoML prep (inputs + JSON merge), variance screening, plots."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from utils.load_results_to_dataframe import load_json_results

DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS: frozenset[str] = frozenset(
    {
        "antibody_id",
        "residue_number",
        "n_total_rows",
        "n_filtered_rows",
        "n_beta_sheet_rows",
        "n_exposed_rows",
    }
)

DEFAULT_CORRELATION_SCREENING_ID_COLS: frozenset[str] = frozenset(
    {
        "structure_id",
        "base",
        "heavy",
        "light",
        "dataset",
        "name",
        "antibody_name",
    }
)

DEFAULT_PRUNE_EXCLUDE_COLS: frozenset[str] = frozenset(
    {
        "antibody_id",
        "structure_id",
        "residue_number",
        "n_total_rows",
        "n_filtered_rows",
        "n_beta_sheet_rows",
        "n_exposed_rows",
        "base",
        "heavy",
        "light",
        "name",
        "dataset",
        "antibody_name",
        "index",
        "target",
        "viscosity",
        "target_viscosity",
    }
)


def make_regressor(
    model_type: str,
    *,
    random_state: int | np.integer | None = None,
    n_jobs: int = -1,
    enet_alpha: float | None = None,
    enet_l1_ratio: float | None = None,
    enet_max_iter: int | None = None,
    rf_n_estimators: int | None = None,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int | None = None,
    rf_max_features=None,
    svm_C: float | None = None,
    svm_epsilon: float | None = None,
    svm_kernel: str = "linear",
    knn_n_neighbors: int | None = None,
    knn_weights: str | None = None,
    knn_algorithm: str | None = None,
    knn_leaf_size: int | None = None,
    knn_p: int | None = None,
    knn_metric: str | None = None,
    knn_metric_params: dict | None = None,
    gpr_kernel_length_scale: float | None = None,
    gpr_n_restarts_optimizer: int | None = None,
    gpr_normalize_y: bool | None = None,
    gpr_alpha: float | None = None,
    n_samples_fit: int | None = None,
    linear_fit_intercept: bool = True,
):
    """Build a sklearn regressor.

    ``model_type``: ``linear`` (ordinary least squares), ``elasticnet``, ``randomforest``,
    ``svm``, ``knn``, or ``gpr`` (Gaussian Process Regression). ``linear`` and ``gpr`` are
    not accepted by feature-selection helpers in ``feature_selectors`` (stability / SFS / RFE
    / fold pipeline); use other model types there.

    For ``knn``, pass ``n_samples_fit`` (training row count) to cap ``n_neighbors``.

    For ``gpr``, the kernel is ``ConstantKernel * RBF`` by default with ``normalize_y=True``
    (suitable for MinMax-scaled inputs). Supports overrides ``gpr_kernel_length_scale``,
    ``gpr_n_restarts_optimizer``, ``gpr_normalize_y``, ``gpr_alpha``.

    Hyperparameters default to ``None``: omitted from the sklearn constructor so the
    library defaults apply (e.g. ElasticNet ``alpha=1.0``). Pass a value only when you
    want to override sklearn.
    """
    mt = str(model_type).strip().lower()
    if mt == "linear":
        return LinearRegression(fit_intercept=linear_fit_intercept, n_jobs=n_jobs)
    if mt == "elasticnet":
        kw: dict = {}
        if enet_alpha is not None:
            kw["alpha"] = enet_alpha
        if enet_l1_ratio is not None:
            kw["l1_ratio"] = enet_l1_ratio
        if enet_max_iter is not None:
            kw["max_iter"] = enet_max_iter
        if random_state is not None:
            kw["random_state"] = int(random_state)
        return ElasticNet(**kw)
    if mt == "randomforest":
        kw: dict = {"n_jobs": n_jobs}
        if rf_n_estimators is not None:
            kw["n_estimators"] = rf_n_estimators
        if rf_max_depth is not None:
            kw["max_depth"] = rf_max_depth
        if rf_min_samples_leaf is not None:
            kw["min_samples_leaf"] = rf_min_samples_leaf
        if rf_max_features is not None:
            kw["max_features"] = rf_max_features
        if random_state is not None:
            kw["random_state"] = int(random_state)
        return RandomForestRegressor(**kw)
    if mt == "svm":
        kw: dict = {"kernel": svm_kernel}
        if svm_C is not None:
            kw["C"] = svm_C
        if svm_epsilon is not None:
            kw["epsilon"] = svm_epsilon
        return SVR(**kw)
    if mt == "knn":
        kw: dict = {}
        nn = knn_n_neighbors
        if nn is not None and n_samples_fit is not None:
            nn = max(1, min(int(nn), int(n_samples_fit)))
        if nn is not None:
            kw["n_neighbors"] = nn
        if knn_weights is not None:
            kw["weights"] = knn_weights
        if knn_algorithm is not None:
            kw["algorithm"] = knn_algorithm
        if knn_leaf_size is not None:
            kw["leaf_size"] = knn_leaf_size
        if knn_p is not None:
            kw["p"] = knn_p
        if knn_metric is not None:
            kw["metric"] = knn_metric
        if knn_metric_params is not None:
            kw["metric_params"] = knn_metric_params
        return KNeighborsRegressor(n_jobs=n_jobs, **kw)
    if mt == "gpr":
        kw: dict = {}
        ls = gpr_kernel_length_scale if gpr_kernel_length_scale is not None else 1.0
        kw["kernel"] = ConstantKernel(1.0) * RBF(length_scale=float(ls))
        kw["normalize_y"] = bool(gpr_normalize_y) if gpr_normalize_y is not None else True
        if gpr_n_restarts_optimizer is not None:
            kw["n_restarts_optimizer"] = int(gpr_n_restarts_optimizer)
        if gpr_alpha is not None:
            kw["alpha"] = float(gpr_alpha)
        if random_state is not None:
            kw["random_state"] = int(random_state)
        return GaussianProcessRegressor(**kw)
    raise ValueError(
        f"Unknown model_type {model_type!r}; expected linear, elasticnet, randomforest, svm, knn, or gpr."
    )


_EVAL_MODEL_BLOCK_NAMES = frozenset(
    {"linear", "elasticnet", "randomforest", "svm", "knn", "gpr"}
)

# YAML keys under each model block → :func:`make_regressor` keyword (omit if None → sklearn default).
_EVAL_HP_KEY_TO_MAKE_REGRESSOR: dict[str, dict[str, str]] = {
    "linear": {
        "fit_intercept": "linear_fit_intercept",
        "linear_fit_intercept": "linear_fit_intercept",
    },
    "elasticnet": {
        "alpha": "enet_alpha",
        "enet_alpha": "enet_alpha",
        "l1_ratio": "enet_l1_ratio",
        "enet_l1_ratio": "enet_l1_ratio",
        "max_iter": "enet_max_iter",
        "enet_max_iter": "enet_max_iter",
    },
    "randomforest": {
        "n_estimators": "rf_n_estimators",
        "rf_n_estimators": "rf_n_estimators",
        "max_depth": "rf_max_depth",
        "rf_max_depth": "rf_max_depth",
        "min_samples_leaf": "rf_min_samples_leaf",
        "rf_min_samples_leaf": "rf_min_samples_leaf",
        "max_features": "rf_max_features",
        "rf_max_features": "rf_max_features",
    },
    "svm": {
        "C": "svm_C",
        "c": "svm_C",
        "svm_c": "svm_C",
        "svm_C": "svm_C",
        "epsilon": "svm_epsilon",
        "svm_epsilon": "svm_epsilon",
        "kernel": "svm_kernel",
        "svm_kernel": "svm_kernel",
    },
    "knn": {
        "n_neighbors": "knn_n_neighbors",
        "knn_n_neighbors": "knn_n_neighbors",
        "weights": "knn_weights",
        "knn_weights": "knn_weights",
        "algorithm": "knn_algorithm",
        "knn_algorithm": "knn_algorithm",
        "leaf_size": "knn_leaf_size",
        "knn_leaf_size": "knn_leaf_size",
        "p": "knn_p",
        "knn_p": "knn_p",
        "metric": "knn_metric",
        "knn_metric": "knn_metric",
        "metric_params": "knn_metric_params",
        "knn_metric_params": "knn_metric_params",
    },
    "gpr": {
        "kernel_length_scale": "gpr_kernel_length_scale",
        "gpr_kernel_length_scale": "gpr_kernel_length_scale",
        "length_scale": "gpr_kernel_length_scale",
        "n_restarts_optimizer": "gpr_n_restarts_optimizer",
        "gpr_n_restarts_optimizer": "gpr_n_restarts_optimizer",
        "normalize_y": "gpr_normalize_y",
        "gpr_normalize_y": "gpr_normalize_y",
        "alpha": "gpr_alpha",
        "gpr_alpha": "gpr_alpha",
    },
}


def _canonical_eval_hp_param(model: str, key: str) -> str:
    m = str(model).strip().lower()
    amap = _EVAL_HP_KEY_TO_MAKE_REGRESSOR.get(m)
    if amap is None:
        raise ValueError(f"Unknown eval model block {model!r}")
    ks = str(key).strip()
    if ks in amap:
        return amap[ks]
    kl = ks.lower()
    for ak, mk in amap.items():
        if ak.lower() == kl:
            return mk
    allowed = sorted(set(amap.keys()) | set(amap.values()))
    raise ValueError(
        f"Unknown eval hyperparameter {key!r} for model {m!r}; allowed: {allowed}"
    )


def _coerce_eval_hp_value(mr_key: str, value):
    if value is None:
        return None
    if mr_key == "linear_fit_intercept":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        s = str(value).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"{mr_key} must be boolean, got {value!r}")
    if mr_key in (
        "enet_alpha",
        "enet_l1_ratio",
        "svm_C",
        "svm_epsilon",
    ):
        return float(value)
    if mr_key in (
        "enet_max_iter",
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "knn_n_neighbors",
        "knn_leaf_size",
        "knn_p",
    ):
        return int(value)
    if mr_key == "rf_max_features":
        if isinstance(value, str):
            s = value.strip()
            if s.lower() in ("sqrt", "log2"):
                return s
            if "." in s:
                return float(s)
            try:
                return int(s)
            except ValueError:
                return s
        return value
    if mr_key in ("svm_kernel", "knn_weights", "knn_algorithm", "knn_metric"):
        return str(value).strip()
    if mr_key == "knn_metric_params":
        if not isinstance(value, dict):
            raise ValueError("knn_metric_params must be a mapping")
        return value
    if mr_key in ("gpr_kernel_length_scale", "gpr_alpha"):
        return float(value)
    if mr_key == "gpr_n_restarts_optimizer":
        return int(value)
    if mr_key == "gpr_normalize_y":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        s = str(value).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"{mr_key} must be boolean, got {value!r}")
    raise ValueError(f"Unhandled eval hyperparameter {mr_key!r}")


def parse_eval_hyperparameters_mapping(raw) -> dict[str, dict]:
    """Map YAML / JSON to per-model kwargs for :func:`make_regressor` (final fold evaluation).

    Expected shape: one top-level key per eval model (``linear``, ``elasticnet``, ``randomforest``,
    ``svm``, ``knn``, ``gpr``), each mapping to a nested mapping of hyperparameters. Skipped / null
    values are omitted so sklearn defaults apply. Unknown model or parameter names raise
    ``ValueError``. ``raw`` may be a JSON object string.
    """
    if raw is None:
        return {}
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            raw = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"eval hyperparameters must be JSON object string: {e}") from e
    if not isinstance(raw, dict):
        raise TypeError("eval hyperparameters must be a mapping (or JSON string of an object)")
    if not raw:
        return {}
    out: dict[str, dict] = {}
    for mk, sub in raw.items():
        model = str(mk).strip().lower()
        if model not in _EVAL_MODEL_BLOCK_NAMES:
            opts = ", ".join(sorted(_EVAL_MODEL_BLOCK_NAMES))
            raise ValueError(
                f"Unknown eval model block {mk!r}; expected top-level keys: {opts}"
            )
        if sub is None:
            continue
        if not isinstance(sub, dict):
            raise ValueError(f"eval hyperparameters[{mk!r}] must be a mapping")
        merged: dict = {}
        for pk, pv in sub.items():
            if pv is None:
                continue
            mr = _canonical_eval_hp_param(model, str(pk))
            merged[mr] = _coerce_eval_hp_value(mr, pv)
        if merged:
            out[model] = merged
    return out


def fit_regressor(model, X, y) -> None:
    """Fit ``model`` in place. ElasticNet / SVR non-convergence becomes ``ConvergenceWarning`` → error."""
    if isinstance(model, (ElasticNet, SVR)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X, y)
    else:
        model.fit(X, y)


@dataclass
class CorrelationBundle:
    """Precomputed Spearman correlations (pairwise complete observations).

    ``prepared_df`` holds rows after target dropna and per-column coercion; matrix
    entries use the same frame as ``feature_feature_corr`` / ``target_feature_*``.
    """

    prepared_df: pd.DataFrame
    feature_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    feature_feature_corr: pd.DataFrame
    target_feature_rho: pd.DataFrame
    target_feature_pvalue: pd.DataFrame
    target_feature_n: pd.DataFrame


def apply_minmax_to_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit :class:`~sklearn.preprocessing.MinMaxScaler` on train columns and transform both splits.

    Uses the same defaults as ``stability_selector`` (feature range ``[0, 1]``).
    Non-listed columns (e.g. ``name``, targets) are left unchanged. The scaler is
    fit **only** on ``train_df[feature_cols]``; test columns are transformed with
    ``transform`` for inference-safe scaling.
    """
    cols = list(dict.fromkeys(feature_cols))
    if not cols:
        return train_df.copy(), test_df.copy()
    missing_te = [c for c in cols if c not in test_df.columns]
    if missing_te:
        raise ValueError(
            "apply_minmax_to_train_test_features: test_df missing scaled columns "
            f"{missing_te!r}"
        )
    scaler = MinMaxScaler()
    tr = train_df[cols].to_numpy(dtype=np.float64, copy=True)
    scaler.fit(tr)
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out[cols] = scaler.transform(tr)
    te = test_df[cols].to_numpy(dtype=np.float64, copy=True)
    test_out[cols] = scaler.transform(te)
    return train_out, test_out


def remove_low_variance_features(
    X: pd.DataFrame,
    candidate_features: list[str] | None = None,
    relative_std_threshold: float = 0.15,
    epsilon: float = 1e-8,
):
    """Drop features with low relative std on ``X``.

    In the fold pipeline, ``X`` is train-split features after
    :func:`apply_minmax_to_train_test_features` (MinMax fitted on train only).
    """

    def _column_as_series(df: pd.DataFrame, col: str) -> pd.Series:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            return s.iloc[:, 0]
        return s

    if candidate_features is None:
        candidate_features = X.select_dtypes(include=[np.number]).columns.tolist()
    candidate_features = list(dict.fromkeys(list(candidate_features)))

    if len(candidate_features) == 0:
        empty = pd.Series(dtype=float)
        return [], [], empty

    X_feat = pd.DataFrame(
        {
            c: _column_as_series(X, c).astype(np.float64, copy=True)
            for c in candidate_features
        }
    )

    mean = X_feat.mean(axis=0)
    std = X_feat.std(axis=0, ddof=0)
    relative_std = std / (mean.abs() + epsilon)

    keep_mask = relative_std >= relative_std_threshold
    kept_features = relative_std.index[keep_mask].tolist()
    removed_features = relative_std.index[~keep_mask].tolist()

    return kept_features, removed_features, relative_std


def _prepare_features_and_targets_for_correlation_matrix(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric matrix for Spearman: coerce numerics; one-hot other categoricals."""
    idx = data.index
    feat_frames: list[pd.DataFrame] = []
    expanded_feature_names: list[str] = []

    for c in feature_cols:
        c = str(c)
        if c not in data.columns:
            continue
        s = data[c]
        s_num = pd.to_numeric(s, errors="coerce")
        non_missing = int(s.notna().sum())
        converted = int(s_num.notna().sum())
        if non_missing == 0:
            feat_frames.append(pd.DataFrame({c: s_num}, index=idx))
            expanded_feature_names.append(c)
        elif converted >= max(1, int(0.8 * non_missing)):
            feat_frames.append(pd.DataFrame({c: s_num}, index=idx))
            expanded_feature_names.append(c)
        else:
            dummies = pd.get_dummies(
                s, prefix=c, prefix_sep="__", dtype=float, dummy_na=True
            )
            feat_frames.append(dummies)
            expanded_feature_names.extend(dummies.columns.tolist())

    tdf = pd.DataFrame(
        {str(t): data[str(t)] for t in target_cols if str(t) in data.columns},
        index=idx,
    )
    if not feat_frames:
        work = tdf
    else:
        work = pd.concat(feat_frames + [tdf], axis=1)
    return work, expanded_feature_names


def _infer_feature_columns_for_bundle(
    merged_df: pd.DataFrame,
    target_cols: list[str],
    *,
    exclude_cols: frozenset[str],
    id_like_cols: frozenset[str],
) -> list[str]:
    """Feature names for correlation: numeric candidates minus targets / excludes."""
    target_set = set(target_cols)
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        return [
            c
            for c in numeric_cols
            if c not in exclude_cols
            and c not in id_like_cols
            and c not in target_set
            and not str(c).startswith("target")
        ]
    return [
        c
        for c in merged_df.columns
        if c not in exclude_cols
        and c not in id_like_cols
        and c not in target_set
        and not str(c).startswith("target")
    ]


def _bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment. pvals: array-like. NaNs are preserved (not used in adjustment)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return p
    p_valid = p[valid]
    n = len(p_valid)
    order = np.argsort(p_valid)
    p_sorted = p_valid[order]
    ratios = n * p_sorted / np.arange(1, n + 1)
    adj_sorted = np.minimum(1, np.minimum.accumulate(ratios[::-1])[::-1])
    rank_of_original = np.argsort(order)
    out[valid] = adj_sorted[rank_of_original]
    return out


def compute_correlation_bundle(
    merged_df: pd.DataFrame,
    target_cols: str | Iterable[str],
    candidate_features: list[str] | None = None,
    *,
    dropna_how: str = "any",
    normalize: bool = False,
    min_periods: int = 3,
    exclude_cols: Iterable[str] | None = None,
    id_like_cols: Iterable[str] | None = None,
    corr_method: Literal["spearman", "pearson"] = "spearman",
) -> CorrelationBundle:
    """Target–feature and feature–feature correlations (Spearman or Pearson).

    Rows with missing values in **any** target (when ``dropna_how='any'``) are dropped
    before coercion. Pairwise correlations use :meth:`pandas.DataFrame.corr` with
    ``method=corr_method`` (pairwise complete observations). Target–feature *p*-values
    use :func:`scipy.stats.spearmanr` or :func:`scipy.stats.pearsonr` on the same
    prepared columns (aligned with rho from the full correlation matrix).

    Parameters
    ----------
    corr_method
        ``"spearman"`` (default) or ``"pearson"`` for both feature–feature and
        target–feature matrices and *p*-values.
    exclude_cols, id_like_cols
        Used only when ``candidate_features`` is ``None``. Defaults match
        :data:`DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS` and
        :data:`DEFAULT_CORRELATION_SCREENING_ID_COLS`. For inter-feature pruning, pass
        ``exclude_cols=DEFAULT_PRUNE_EXCLUDE_COLS`` and ``id_like_cols=()`` instead.
    """
    if isinstance(target_cols, str):
        tcols = [target_cols]
    else:
        tcols = [str(t) for t in target_cols]
    if not tcols:
        raise ValueError("target_cols must be non-empty")

    for t in tcols:
        if t not in merged_df.columns:
            raise ValueError(f"target column {t!r} not in merged_df")

    cm = str(corr_method).strip().lower()
    if cm not in ("spearman", "pearson"):
        raise ValueError("corr_method must be 'spearman' or 'pearson'")

    excl = (
        frozenset(exclude_cols)
        if exclude_cols is not None
        else DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS
    )
    id_like = (
        frozenset(id_like_cols)
        if id_like_cols is not None
        else DEFAULT_CORRELATION_SCREENING_ID_COLS
    )

    base = merged_df.dropna(subset=tcols, how=dropna_how).copy()

    if candidate_features is None:
        feats = _infer_feature_columns_for_bundle(
            base, tcols, exclude_cols=excl, id_like_cols=id_like
        )
    else:
        feats = [
            str(f)
            for f in candidate_features
            if str(f) in base.columns and str(f) not in tcols
        ]

    feats = list(dict.fromkeys(feats))
    if not feats:
        empty_ff = pd.DataFrame()
        empty_tf = pd.DataFrame()
        return CorrelationBundle(
            prepared_df=base.iloc[:0].copy(),
            feature_cols=tuple(),
            target_cols=tuple(tcols),
            feature_feature_corr=empty_ff,
            target_feature_rho=empty_tf,
            target_feature_pvalue=empty_tf.copy(),
            target_feature_n=empty_tf.copy(),
        )

    work, feats = _prepare_features_and_targets_for_correlation_matrix(base, feats, tcols)
    work = work.dropna(how="all")
    all_cols = list(feats) + list(tcols)

    if normalize and len(work) > 0:
        for c in all_cols:
            if c not in work.columns:
                continue
            x = pd.to_numeric(work[c], errors="coerce")
            lo, hi = x.min(), x.max()
            work[c] = (x - lo) / (hi - lo) if hi > lo else x

    full = work[all_cols].corr(method=cm, min_periods=min_periods)
    ff = full.loc[feats, feats]
    tf_r = full.loc[feats, tcols]

    tf_p = pd.DataFrame(np.nan, index=feats, columns=tcols, dtype=float)
    tf_n = pd.DataFrame(np.nan, index=feats, columns=tcols, dtype=float)
    _r_p = spearmanr if cm == "spearman" else pearsonr
    for t in tcols:
        for f in feats:
            sub = work[[t, f]].dropna()
            n = len(sub)
            tf_n.loc[f, t] = float(n)
            if n >= min_periods:
                _, p = _r_p(sub[t], sub[f])
                tf_p.loc[f, t] = float(p)
            else:
                tf_p.loc[f, t] = np.nan

    return CorrelationBundle(
        prepared_df=work,
        feature_cols=tuple(feats),
        target_cols=tuple(tcols),
        feature_feature_corr=ff,
        target_feature_rho=tf_r,
        target_feature_pvalue=tf_p,
        target_feature_n=tf_n,
    )


def reduce_correlated_features(
    split_train_df: pd.DataFrame,
    target_col: str,
    candidate_features: list[str] | None = None,
    *,
    correlation_threshold: float = 0.85,
    reduction_mode: Literal["cluster", "pairwise"] = "cluster",
    importance_metric: str = "spearman",
    correlation_bundle: CorrelationBundle | None = None,
) -> list[str]:
    """Drop redundant features using feature–feature and target–feature correlations.

    ``importance_metric`` selects Spearman or Pearson for both matrices; the survivor
    in each cluster or pairwise tie-break is the feature with larger ``|rho(target)|``.

    - ``reduction_mode="cluster"``: hierarchical clustering on feature–feature |rho|.
    - ``reduction_mode="pairwise"``: greedy resolution of strongest violating pairs.
    """
    importance_metric = importance_metric.lower()
    if importance_metric not in ("spearman", "pearson"):
        raise ValueError("importance_metric must be 'spearman' or 'pearson'")

    mode = str(reduction_mode).strip().lower()
    if mode not in ("cluster", "pairwise"):
        raise ValueError("reduction_mode must be 'cluster' or 'pairwise'")

    target_col = str(target_col)
    if target_col not in split_train_df.columns:
        raise ValueError(f"target_col {target_col!r} not in split_train_df columns")

    exclude = DEFAULT_PRUNE_EXCLUDE_COLS

    if candidate_features is None:
        if correlation_bundle is None:
            bundle = compute_correlation_bundle(
                split_train_df,
                [target_col],
                candidate_features=None,
                exclude_cols=exclude,
                id_like_cols=frozenset(),
                corr_method=importance_metric,  # type: ignore[arg-type]
            )
        else:
            bundle = correlation_bundle
        if target_col not in bundle.target_cols:
            raise ValueError(f"target_col {target_col!r} not in correlation_bundle.target_cols")
        sig_features = [f for f in bundle.feature_cols if f in split_train_df.columns]
    else:
        sig_features = [
            f
            for f in candidate_features
            if f in split_train_df.columns
            and f not in exclude
            and f != target_col
            and not str(f).startswith("target")
        ]
        sig_features = list(dict.fromkeys(sig_features))
        if correlation_bundle is None or not set(sig_features).issubset(
            correlation_bundle.feature_cols
        ):
            bundle = compute_correlation_bundle(
                split_train_df,
                [target_col],
                candidate_features=sig_features if sig_features else None,
                exclude_cols=exclude,
                id_like_cols=frozenset(),
                corr_method=importance_metric,  # type: ignore[arg-type]
            )
        else:
            bundle = correlation_bundle
        if target_col not in bundle.target_cols:
            raise ValueError(f"target_col {target_col!r} not in correlation_bundle.target_cols")

    if len(sig_features) < 2:
        return sig_features

    if mode == "cluster":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        rho_tgt = bundle.target_feature_rho.reindex(sig_features)[target_col]
        target_score = {
            f: (0.0 if pd.isna(rho_tgt.loc[f]) else float(rho_tgt.loc[f]))
            for f in sig_features
        }

        sub_ff = bundle.feature_feature_corr.reindex(
            index=sig_features, columns=sig_features
        ).abs()
        sub_ff = sub_ff.fillna(0.0)
        # Copy: parquet / pandas COW can yield read-only .values; fill_diagonal writes in place.
        ff_arr = np.array(sub_ff, dtype=np.float64, copy=True, order="C")
        np.fill_diagonal(ff_arr, 1.0)
        dist_arr = 1.0 - ff_arr
        np.fill_diagonal(dist_arr, 0.0)

        if len(sig_features) == 2:
            cluster_labels = (
                np.array([1, 1])
                if dist_arr[0, 1] <= (1.0 - correlation_threshold)
                else np.array([1, 2])
            )
        else:
            condensed = squareform(dist_arr, checks=False)
            Z = linkage(condensed, method="average")
            cluster_labels = fcluster(
                Z, t=(1.0 - correlation_threshold), criterion="distance"
            )

        cluster_to_features: dict[int, list[str]] = {}
        for feat, label in zip(sig_features, cluster_labels):
            cluster_to_features.setdefault(int(label), []).append(feat)

        kept_features: list[str] = []
        for label in sorted(cluster_to_features):
            members = cluster_to_features[label]
            best_feat = max(members, key=lambda f: abs(target_score.get(f, 0.0)))
            kept_features.append(best_feat)

        kept_set = set(kept_features)
        return [f for f in sig_features if f in kept_set]

    sub_ff = bundle.feature_feature_corr.reindex(
        index=sig_features, columns=sig_features
    ).abs().fillna(0.0)

    rho_abs: dict[str, float] = {}
    for f in sig_features:
        v = bundle.target_feature_rho.loc[f, target_col]
        rho_abs[f] = abs(float(v)) if pd.notna(v) else 0.0

    kept = set(sig_features)
    max_iter = len(sig_features) * max(len(sig_features), 1) + 10

    for _ in range(max_iter):
        best_pair: tuple[float, str, str] | None = None
        for fi in sig_features:
            if fi not in kept:
                continue
            for fj in sig_features:
                if fj <= fi or fj not in kept:
                    continue
                r = float(sub_ff.loc[fi, fj])
                if r >= correlation_threshold:
                    cand = (-r, fi, fj)
                    if best_pair is None or cand < best_pair:
                        best_pair = cand
        if best_pair is None:
            break
        _, fi, fj = best_pair
        if rho_abs[fi] > rho_abs[fj] or (
            rho_abs[fi] == rho_abs[fj] and fi < fj
        ):
            kept.discard(fj)
        else:
            kept.discard(fi)

    return [f for f in sig_features if f in kept]


def calculate_correlations_and_plot(
    merged_df,
    target_col: str,
    p_threshold=0.05,
    fdr_alpha=0.05,
    use_fdr=True,
    normalize=False,
    make_plots=False,
    correlation_threshold=None,
    correlation_exclude_cols: Iterable[str] | None = None,
    correlation_id_like_cols: Iterable[str] | None = None,
    candidate_features: Iterable[str] | None = None,
    verbose: bool = True,
):
    """Spearman screening vs one target with optional plots. Returns ``(corr_df, significant_tuples)``.

    Pass ``candidate_features`` (last argument) to restrict to a subset of columns; if
    ``None``, numeric columns are used minus the usual ID / metadata exclusions.
    """
    from .feature_selectors import correlation_selector

    exclude_set = (
        frozenset(correlation_exclude_cols)
        if correlation_exclude_cols is not None
        else DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS
    )
    id_like_set = (
        frozenset(correlation_id_like_cols)
        if correlation_id_like_cols is not None
        else DEFAULT_CORRELATION_SCREENING_ID_COLS
    )

    merged_df_t = merged_df.copy()
    merged_df_t[target_col] = pd.to_numeric(merged_df_t[target_col], errors="coerce")

    n_before = len(merged_df_t)
    merged_df_t = merged_df_t.dropna(subset=[target_col])
    n_after = len(merged_df_t)
    if n_before > n_after:
        print(
            f"Dropped {n_before - n_after} rows with NaN/invalid target '{target_col}' "
            f"(using {n_after} for correlations)."
        )

    if candidate_features is not None:
        bundle_candidates = [
            str(c)
            for c in dict.fromkeys(candidate_features)
            if str(c) in merged_df_t.columns
            and str(c) != str(target_col)
            and not str(c).startswith("target")
        ]
    else:
        numeric_cols = merged_df_t.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            bundle_candidates = [
                col
                for col in numeric_cols
                if col not in exclude_set
                and col not in id_like_set
                and col != target_col
                and not str(col).startswith("target")
            ]
        else:
            bundle_candidates = [
                col
                for col in merged_df_t.columns
                if col not in exclude_set
                and col not in id_like_set
                and col != target_col
                and not str(col).startswith("target")
            ]

    bundle = compute_correlation_bundle(
        merged_df_t,
        [target_col],
        candidate_features=bundle_candidates,
        exclude_cols=exclude_set,
        id_like_cols=id_like_set,
        normalize=normalize,
    )
    corr_df, significant_tuples = correlation_selector(
        bundle,
        target_col,
        p_threshold=p_threshold,
        fdr_alpha=fdr_alpha,
        use_fdr=use_fdr,
        min_abs_rho=correlation_threshold,
    )

    if significant_tuples:
        order = {t[0]: i for i, t in enumerate(significant_tuples)}
        significant = corr_df[corr_df["feature"].isin(order)].copy()
        significant["_ord"] = significant["feature"].map(order)
        significant = significant.sort_values("_ord").drop(columns=["_ord"])
    else:
        significant = corr_df.iloc[:0].copy()

    plot_df = (
        bundle.prepared_df
        if len(bundle.prepared_df) > 0 and target_col in bundle.prepared_df.columns
        else merged_df_t
    )

    if verbose:
        print(f"[target={target_col}] Total features tested: {len(corr_df)}")
        print(
            f"[target={target_col}] Significant (raw p < {p_threshold}): "
            f"{(corr_df['spearman_p'] < p_threshold).sum()}"
        )
        if use_fdr:
            print(
                f"[target={target_col}] Significant after FDR correction (adj p < {fdr_alpha}): "
                f"{len(significant)}"
            )
        else:
            print(f"[target={target_col}] Significant (no FDR): {len(significant)}")
        if correlation_threshold is not None:
            print(
                f"[target={target_col}] After applying |rho| >= {correlation_threshold}: {len(significant)}"
            )
        if normalize:
            print(
                f"[target={target_col}] Note: Features were min-max normalized before correlation calculation"
            )
        print(
            f"[target={target_col}] Top correlations by absolute Spearman r "
            f"({'FDR-significant only' if use_fdr else 'raw p < ' + str(p_threshold)}):"
        )

        if len(significant) > 0:
            sig = significant.copy()
            sig["spearman_r"] = pd.to_numeric(sig["spearman_r"], errors="coerce")
            sig = sig.dropna(subset=["spearman_r"])
            if len(sig) > 0:
                cols = ["feature", "spearman_r", "spearman_p", "spearman_p_adj"]
                print(
                    sig.nlargest(10, "spearman_r", keep="all")[
                        [c for c in cols if c in sig.columns]
                    ]
                )
            else:
                print("(none)")
        else:
            print("(no significant correlations)")

    if make_plots and len(significant) > 0:
        import matplotlib.pyplot as plt

        n_plots = len(significant)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        plot_i = 0
        for idx, row in significant.iterrows():
            fc = row["feature"]
            if fc not in plot_df.columns:
                continue
            data = plot_df[[target_col, fc]].dropna()
            if len(data) < 1:
                continue
            ax = axes[plot_i]
            ax.scatter(data[fc], data[target_col], alpha=0.6)
            ax.set_xlabel(fc, fontsize=10)
            ax.set_ylabel(target_col, fontsize=10)
            p_label = "p_adj" if use_fdr else "p"
            ax.set_title(
                f"ρ={row['spearman_r']:.3f}, {p_label}={row['spearman_p_adj']:.3e}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            plot_i += 1

        for j in range(plot_i, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    return corr_df, significant_tuples


# --- AutoML pipeline: inputs, column prep, developability JSON merge -----------------


DEVELOPABILITY_FEATURE_GROUPS: Final[dict[str, list[str]]] = {
    "aggregation": [],
    "thermostability": [],
    "polyreactivity": [],
    "viscosity": [],
}

_ALLOWED_DEVELOPABILITY_CATEGORIES: frozenset[str] = frozenset(
    DEVELOPABILITY_FEATURE_GROUPS.keys()
)

TARGET_COLUMN_PREFIX: Final[str] = "target_"
EXTERNAL_FEATURE_PREFIX: Final[str] = "external_feature_"


def _pipeline_dtype_is_int_float_or_bool(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_integer_dtype(series)
        or pd.api.types.is_floating_dtype(series)
    )


def natural_sort_key(value: object) -> list:
    """Sort e.g. IgG1, IgG2, IgG10 by numeric chunks (``analyze.ipynb`` GINKGO cell)."""
    s = str(value).strip()
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def one_hot_encode_non_numeric_columns(
    df: pd.DataFrame,
    *,
    skip_columns: Iterable[str] | None = None,
    skip_target_prefixed: bool = True,
    natural_sort_keys: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot-encode columns that are not integer / float / bool.

    ``natural_sort_keys`` is accepted for API compatibility with older call sites and
    is ignored (dummy column order follows ``pandas.get_dummies``).
    """
    _ = natural_sort_keys
    out = df.copy()
    skip: set[str] = {"name"}
    skip.update(str(c) for c in (skip_columns or ()))
    if skip_target_prefixed:
        skip.update(
            c for c in out.columns if str(c).startswith(TARGET_COLUMN_PREFIX)
        )

    source_cols_one_hot: list[str] = []

    for col in list(out.columns):
        if col in skip:
            continue
        s = out[col]
        if _pipeline_dtype_is_int_float_or_bool(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(
            s
        ):
            continue

        non_missing = s[s.notna()]
        if non_missing.empty:
            continue

        dummies = pd.get_dummies(
            s, prefix=str(col), prefix_sep="__", dtype=float, dummy_na=True
        )
        out = out.drop(columns=[col])
        out = pd.concat([out, dummies], axis=1)
        source_cols_one_hot.append(str(col))

    return out, source_cols_one_hot


def prefixed_target_column_name(col: str) -> str:
    c = str(col)
    if c.startswith(TARGET_COLUMN_PREFIX):
        return c
    return f"{TARGET_COLUMN_PREFIX}{c}"


def canonical_target_column_names(raw_target_names: Iterable[str]) -> list[str]:
    return [prefixed_target_column_name(c) for c in raw_target_names]


def coerce_target_columns_to_string_and_check_unique(
    df: pd.DataFrame,
    target_cols: Iterable[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in target_cols:
        c = str(col)
        if c not in out.columns:
            raise ValueError(f"Target column {c!r} not in dataframe columns.")
        s = out[c]
        mask = s.notna()

        if pd.api.types.is_string_dtype(s):
            str_s = s.copy()
        else:
            str_s = pd.Series(pd.NA, index=s.index, dtype=pd.StringDtype())
            if mask.any():
                str_s.loc[mask] = s.loc[mask].map(str)

        non_null = str_s.dropna()
        if not non_null.empty and non_null.duplicated().any():
            dup_vals = non_null[non_null.duplicated(keep=False)].unique()
            sample = dup_vals[: min(10, len(dup_vals))].tolist()
            raise ValueError(
                f"Target column {c!r} has duplicate non-null values "
                f"(sample: {sample!r})."
            )
        out[c] = str_s

    return out


def prefixed_external_feature_column_name(col: str) -> str:
    c = str(col)
    if c.startswith(EXTERNAL_FEATURE_PREFIX):
        return c
    return f"{EXTERNAL_FEATURE_PREFIX}{c}"


def _external_feature_scalar_ok(x: object) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except (ValueError, TypeError):
        pass
    if isinstance(x, (str, bytes, bytearray)):
        return True
    if isinstance(x, (np.str_, np.bytes_)):
        return True
    if isinstance(x, (bool, np.bool_)):
        return True
    if isinstance(x, (int, float, complex, np.integer, np.floating, np.complexfloating)):
        return True
    if isinstance(x, (pd.Timestamp, datetime, date, np.datetime64, np.timedelta64)):
        return False
    return False


def validate_external_feature_columns_are_numeric_or_string(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> None:
    for col in feature_cols:
        c = str(col)
        if c not in df.columns:
            raise ValueError(f"External feature column {c!r} not in dataframe columns.")
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_string_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        if isinstance(s.dtype, pd.CategoricalDtype):
            for x in s:
                if not _external_feature_scalar_ok(x):
                    raise ValueError(
                        f"convert feature column {c} to numeric or string"
                    )
            continue
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(
            s
        ):
            raise ValueError(f"convert feature column {c} to numeric or string")
        if pd.api.types.is_object_dtype(s):
            for x in s:
                if not _external_feature_scalar_ok(x):
                    raise ValueError(
                        f"convert feature column {c} to numeric or string"
                    )
            continue

        raise ValueError(f"convert feature column {c} to numeric or string")


def apply_pipeline_column_renames(
    df: pd.DataFrame,
    *,
    target_cols: list[str],
    external_feature_cols: list[str],
    antibody_name_col: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    rename_map: dict[str, str] = {}

    if antibody_name_col not in out.columns:
        raise ValueError(
            f"antibody_name_col {antibody_name_col!r} not in dataframe columns."
        )

    tset = set(target_cols)
    eset = set(external_feature_cols)
    overlap = tset & eset
    if overlap:
        raise ValueError(
            f"Columns cannot be both targets and external features: {sorted(overlap)!r}"
        )
    if antibody_name_col in tset or antibody_name_col in eset:
        raise ValueError(
            "antibody_name_col must not appear in target_cols or external_feature_cols."
        )

    if external_feature_cols:
        validate_external_feature_columns_are_numeric_or_string(
            out, external_feature_cols
        )

    for c in external_feature_cols:
        if c not in out.columns:
            raise ValueError(f"external feature column {c!r} not in dataframe columns.")
        new_c = prefixed_external_feature_column_name(c)
        if new_c != c:
            rename_map[c] = new_c

    for c in target_cols:
        if c not in out.columns:
            raise ValueError(f"target column {c!r} not in dataframe columns.")
        new_c = prefixed_target_column_name(c)
        if new_c != c:
            rename_map[c] = new_c

    if antibody_name_col != "name":
        if "name" in out.columns:
            raise ValueError(
                f"Cannot rename antibody column {antibody_name_col!r} to 'name': "
                "column 'name' already exists."
            )
        rename_map[antibody_name_col] = "name"

    grouped: dict[str, list[str]] = {}
    for old, new in rename_map.items():
        grouped.setdefault(new, []).append(old)
    for new, olds in grouped.items():
        if len(olds) > 1:
            raise ValueError(
                f"Column rename collision: multiple columns map to {new!r}: {olds!r}"
            )

    for old, new in rename_map.items():
        if old == new:
            continue
        if new in out.columns:
            raise ValueError(
                f"Cannot rename {old!r} -> {new!r}: column {new!r} already exists."
            )

    if rename_map:
        out = out.rename(columns=rename_map, errors="raise")

    applied = {o: n for o, n in rename_map.items() if o != n}
    return out, applied


def parse_whitespace_column_list(spec: str | None) -> list[str]:
    if spec is None:
        return []
    parts = str(spec).split()
    return list(dict.fromkeys(p for p in parts if p))


def normalize_developability_category(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower()
    if not key:
        return None
    if key not in _ALLOWED_DEVELOPABILITY_CATEGORIES:
        opts = ", ".join(sorted(_ALLOWED_DEVELOPABILITY_CATEGORIES))
        raise ValueError(
            f"developability_category must be one of: {opts}; got {name!r}"
        )
    return key


def _normalize_merge_key_like_notebook(df: pd.DataFrame, key: str) -> pd.DataFrame:
    out = df.copy()
    if key not in out.columns:
        raise ValueError(f"Merge key column {key!r} not in dataframe.")
    s = out[key]
    try:
        out[key] = s.astype(int)
    except Exception:
        pass
    s = out[key]
    out[key] = s.astype(str)
    return out


def verify_developability_merge_matches_prepared(
    left_merged: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    merge_on: str,
    how: Literal["inner", "left", "right", "outer"],
) -> None:
    """For ``inner`` / ``left`` joins, ensure row count and prepared columns are unchanged.

    Checks that ``len(merged) == len(left_merged)`` (no dropped or duplicated prepared
    rows from many-to-one matches on the JSON side) and that every column present on
    both frames matches exactly, including NaN positions (``Series.equals``).

    No-op when ``how`` is ``right`` or ``outer`` (prepared row set is not preserved).
    """
    if how not in ("inner", "left"):
        return

    if len(merged) != len(left_merged):
        raise ValueError(
            f"Merge sanity check failed: prepared rows {len(left_merged)} but merged "
            f"rows {len(merged)} (how={how!r}). Expect one merged row per prepared row "
            "when merge keys are unique; duplicate JSON keys or missing matches change "
            "the row count."
        )

    ls = left_merged.sort_values(merge_on, kind="mergesort").reset_index(drop=True)
    ms = merged.sort_values(merge_on, kind="mergesort").reset_index(drop=True)
    shared = [c for c in left_merged.columns if c in merged.columns]
    for c in shared:
        if not ls[c].equals(ms[c]):
            n_pre = int(ls[c].isna().sum())
            n_mer = int(ms[c].isna().sum())
            raise ValueError(
                f"Merge sanity check failed: column {c!r} differs between prepared and "
                f"merged data (NaN count prepared vs merged: {n_pre} vs {n_mer})."
            )


def merge_prepared_with_developability_json(
    prepared_df: pd.DataFrame,
    results_dir: Path | str,
    *,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    merge_on: str = "name",
    developability_base: str | None = None,
    suffixes: tuple[str, str] = ("", "_developability"),
    merge_sanity_check: bool = True,
) -> pd.DataFrame:
    """Merge JSON descriptor rows into ``prepared_df`` on ``merge_on``.

    When ``merge_sanity_check`` is True and ``how`` is ``inner`` or ``left``, runs
    :func:`verify_developability_merge_matches_prepared` so row count matches the
    prepared table and no prepared column gains or loses NaNs vs its pre-merge state.
    """
    path = Path(results_dir)
    if not str(path).strip():
        raise ValueError(
            "results_dir is empty; set developability_results_dir on AutoMLPipelineInput "
            "or pass a path to the JSON results folder."
        )

    left = _normalize_merge_key_like_notebook(prepared_df, merge_on)
    right = load_json_results(path, name_from_stem=True)
    if developability_base is not None:
        right = right.copy()
        right["base"] = developability_base
    right = _normalize_merge_key_like_notebook(right, merge_on)

    merged = left.merge(right, on=merge_on, how=how, suffixes=suffixes)
    if merge_sanity_check:
        verify_developability_merge_matches_prepared(
            left, merged, merge_on=merge_on, how=how
        )
    return merged


@dataclass
class AutoMLPipelineInput:
    """Pipeline inputs: experimental table + developability JSON folder (see module docstring)."""

    original_dataset: pd.DataFrame
    target_cols: str
    feature_cols: str | None = None
    antibody_name_col: str = "name"
    developability_results_dir: Path | str = ""
    developability_category: str | None = None

    @property
    def parsed_target_cols(self) -> list[str]:
        cols = parse_whitespace_column_list(self.target_cols)
        if not cols:
            raise ValueError("target_cols must list at least one column name.")
        return cols

    @property
    def parsed_external_feature_cols(self) -> list[str]:
        return parse_whitespace_column_list(self.feature_cols)

    @property
    def developability_category_key(self) -> str | None:
        return normalize_developability_category(self.developability_category)

    @property
    def developability_results_path(self) -> Path:
        return Path(self.developability_results_dir)

    def developability_candidate_columns(self) -> list[str]:
        key = self.developability_category_key
        if key is None:
            return []
        return list(DEVELOPABILITY_FEATURE_GROUPS[key])

    def validate_against_dataset(self) -> None:
        df = self.original_dataset
        targets = self.parsed_target_cols
        externals = self.parsed_external_feature_cols
        tset, eset = set(targets), set(externals)
        if tset & eset:
            raise ValueError(
                f"Columns cannot be both targets and external features: {sorted(tset & eset)!r}"
            )
        if self.antibody_name_col in tset or self.antibody_name_col in eset:
            raise ValueError(
                "antibody_name_col must not appear in target_cols or feature_cols."
            )
        if self.antibody_name_col not in df.columns:
            raise ValueError(
                f"antibody_name_col {self.antibody_name_col!r} not in original_dataset columns."
            )
        for c in targets:
            if c not in df.columns:
                raise ValueError(f"target column {c!r} not in original_dataset columns.")
        for c in externals:
            if c not in df.columns:
                raise ValueError(
                    f"external feature column {c!r} not in original_dataset columns."
                )
        if externals:
            validate_external_feature_columns_are_numeric_or_string(df, externals)

    def data_frame_with_canonical_column_names(
        self,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        return apply_pipeline_column_renames(
            self.original_dataset,
            target_cols=self.parsed_target_cols,
            external_feature_cols=self.parsed_external_feature_cols,
            antibody_name_col=self.antibody_name_col,
        )

    def data_frame_canonical_and_encoded(
        self,
        *,
        natural_sort_keys: bool = False,
        skip_columns: Iterable[str] | None = None,
        skip_target_prefixed: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
        df, renames = self.data_frame_with_canonical_column_names()
        df = coerce_target_columns_to_string_and_check_unique(
            df,
            canonical_target_column_names(self.parsed_target_cols),
        )
        df_enc, encoded = one_hot_encode_non_numeric_columns(
            df,
            skip_columns=skip_columns,
            skip_target_prefixed=skip_target_prefixed,
            natural_sort_keys=natural_sort_keys,
        )
        return df_enc, renames, encoded

    def merge_developability_results(
        self,
        prepared_df: pd.DataFrame,
        *,
        how: Literal["inner", "left", "right", "outer"] = "inner",
        merge_on: str = "name",
        developability_base: str | None = None,
        merge_sanity_check: bool = True,
    ) -> pd.DataFrame:
        return merge_prepared_with_developability_json(
            prepared_df,
            self.developability_results_path,
            how=how,
            merge_on=merge_on,
            developability_base=developability_base,
            merge_sanity_check=merge_sanity_check,
        )

    def data_frame_canonical_encoded_and_merged(
        self,
        *,
        natural_sort_keys: bool = False,
        skip_columns: Iterable[str] | None = None,
        skip_target_prefixed: bool = True,
        merge_how: Literal["inner", "left", "right", "outer"] = "inner",
        merge_on: str = "name",
        developability_base: str | None = None,
        merge_sanity_check: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
        prepared, renames, encoded = self.data_frame_canonical_and_encoded(
            natural_sort_keys=natural_sort_keys,
            skip_columns=skip_columns,
            skip_target_prefixed=skip_target_prefixed,
        )
        merged = self.merge_developability_results(
            prepared,
            how=merge_how,
            merge_on=merge_on,
            developability_base=developability_base,
            merge_sanity_check=merge_sanity_check,
        )
        return merged, renames, encoded


__all__ = [
    "DEFAULT_CORRELATION_SCREENING_EXCLUDE_COLS",
    "DEFAULT_CORRELATION_SCREENING_ID_COLS",
    "DEFAULT_PRUNE_EXCLUDE_COLS",
    "make_regressor",
    "fit_regressor",
    "CorrelationBundle",
    "apply_minmax_to_train_test_features",
    "remove_low_variance_features",
    "compute_correlation_bundle",
    "reduce_correlated_features",
    "calculate_correlations_and_plot",
    "DEVELOPABILITY_FEATURE_GROUPS",
    "AutoMLPipelineInput",
    "TARGET_COLUMN_PREFIX",
    "EXTERNAL_FEATURE_PREFIX",
    "apply_pipeline_column_renames",
    "canonical_target_column_names",
    "coerce_target_columns_to_string_and_check_unique",
    "natural_sort_key",
    "one_hot_encode_non_numeric_columns",
    "normalize_developability_category",
    "validate_external_feature_columns_are_numeric_or_string",
    "parse_whitespace_column_list",
    "prefixed_external_feature_column_name",
    "prefixed_target_column_name",
    "merge_prepared_with_developability_json",
    "verify_developability_merge_matches_prepared",
]
