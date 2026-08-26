#!/usr/bin/env python3
"""Nested CV for fixed pipeline variants with comprehensive analysis artifacts.

Pipelines:
  * intercorr_svm: low-var + intercorr prune, then SVM on all survivors
  * sfs_svm: low-var + intercorr + SFS (SVM, frac=0.15); pick eval among
    svm / knn / linear / randomforest
  * sfs_knn: same as sfs_svm but SFS selector model is KNN
  * elasticnet: nested alpha/l1 grid on all input features
    (median impute + StandardScaler; no selector)

Artifacts written under --out-dir:
  predictions/oof.parquet
  metrics/outer_fold_metrics.{parquet,csv}
  metrics/{variant_split,structure_model,averaged}_summary.csv
  features/outer_feature_usage.parquet
  features/outer_feature_sets.json
  features/inner_selected_features.parquet
  inner_selection/{outer_choice,inner_scores_long,inner_grid_scores}.parquet
  checkpoints/.../outer_*.json  (resume-safe)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.nested_cv import write_inner_fold_dir
from analysis.nested_elasticnet_all_methods import (
    EXCLUDED_TARGETS,
    METHOD_SLUGS,
    _discover_kitab,
    _discover_propermab,
    _discover_tap,
    _fisher_mean,
    _features,
    _target_dirs,
    _write_json_atomic,
)
from analysis.nested_elasticnet_all_targets import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    _spearman,
)
from automl.feature_selectors import (
    reduce_correlated_features,
    remove_low_variance_features,
    sequential_forward_selector,
)
from automl.pipeline_defaults import (
    DEFAULT_EVAL_HYPERPARAMETERS_RAW,
    DEFAULT_INTERCORR_IMPORTANCE_METRIC,
    DEFAULT_INTERCORR_REDUCTION_MODE,
    DEFAULT_INTERCORR_THRESHOLD,
    DEFAULT_LOW_VARIANCE_EPSILON,
    DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SFS_MIN_IMPROVEMENT,
)
from automl.utils import fit_regressor, make_regressor, parse_eval_hyperparameters_mapping

REPO = Path("/storage/antibody_data/PairedStructures/kitAb")

PIPELINES: dict[str, dict[str, Any]] = {
    "intercorr_svm": {
        "label": "Intercorr prune + SVM",
        "kind": "selector",
        "selector_name": None,
        "selector_model": None,
        "features_frac": None,
        "eval_models": ["svm"],
        "apply_intercorr": True,
    },
    "sfs_svm": {
        "label": "SFS (SVM selector)",
        "kind": "selector",
        "selector_name": "sfs",
        "selector_model": "svm",
        "features_frac": 0.15,
        "eval_models": ["svm", "knn", "linear", "randomforest"],
        "apply_intercorr": True,
    },
    "sfs_knn": {
        "label": "SFS (KNN selector)",
        "kind": "selector",
        "selector_name": "sfs",
        "selector_model": "knn",
        "features_frac": 0.15,
        "eval_models": ["svm", "knn", "linear", "randomforest"],
        "apply_intercorr": True,
    },
    "elasticnet": {
        "label": "Nested ElasticNet",
        "kind": "elasticnet",
        "selector_name": None,
        "selector_model": None,
        "features_frac": None,
        "eval_models": ["elasticnet"],
        "apply_intercorr": False,
        "alphas": list(DEFAULT_ALPHAS),
        "l1_ratios": list(DEFAULT_L1_RATIOS),
    },
}

SFS_INNER_CV = 5
PERM_REPEATS = 10
CONFIG_COMPARE_KEYS = (
    "pipelines",
    "eval_hyperparameters",
    "intercorr_threshold",
    "intercorr_importance_metric",
    "intercorr_reduction_mode",
    "low_variance_relative_std_threshold",
    "low_variance_epsilon",
    "sfs_inner_cv",
    "sfs_min_improvement",
    "random_state",
    "excluded_targets",
    "perm_repeats",
)


def _selection_max_features(n_train: int, features_frac: float) -> int:
    return max(1, int(float(features_frac) * n_train))


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _metric_bundle(y: pd.Series | np.ndarray, yhat: pd.Series | np.ndarray) -> dict[str, Any]:
    y_arr = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=float)
    yhat_arr = np.asarray(pd.to_numeric(pd.Series(yhat), errors="coerce"), dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(yhat_arr)
    y_arr, yhat_arr = y_arr[mask], yhat_arr[mask]
    if len(y_arr) < 2 or len(np.unique(y_arr)) < 2 or len(np.unique(yhat_arr)) < 2:
        return {
            "spearman": None,
            "spearman_p": None,
            "pearson_r": None,
            "pearson_p": None,
            "r2": None,
            "mse": None,
            "n": int(len(y_arr)),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho, rho_p = spearmanr(y_arr, yhat_arr)
        pearson_r, pearson_p = pearsonr(y_arr, yhat_arr)
    return {
        "spearman": _json_float(float(rho)),
        "spearman_p": _json_float(float(rho_p)),
        "pearson_r": _json_float(float(pearson_r)),
        "pearson_p": _json_float(float(pearson_p)),
        "r2": _json_float(float(r2_score(y_arr, yhat_arr))),
        "mse": _json_float(float(mean_squared_error(y_arr, yhat_arr))),
        "n": int(len(y_arr)),
    }


def _ranks_desc(values: np.ndarray) -> list[int]:
    order = (-np.asarray(values, dtype=float)).argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks.tolist()


def _fit_minmax(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_mat = train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    test_mat = test_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    scaler.fit(train_mat)
    train_out[cols] = scaler.transform(train_mat)
    test_out[cols] = scaler.transform(test_mat)
    return train_out, test_out, scaler


def _run_selector_feature_selection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    candidate_features: list[str],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    cand = list(dict.fromkeys(candidate_features))
    train_k, test_k, scaler = _fit_minmax(train_df, test_df, cand)
    kept, _removed, _rel = remove_low_variance_features(
        X=train_k,
        candidate_features=cand,
        relative_std_threshold=DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD,
        epsilon=DEFAULT_LOW_VARIANCE_EPSILON,
    )
    if len(kept) == 0:
        kept = cand[:1]
    drop = [feature for feature in cand if feature not in set(kept)]
    train_k = train_k.drop(columns=drop, errors="ignore").copy()
    test_k = test_k.drop(columns=drop, errors="ignore").copy()

    after_intercorr = list(kept)
    if pipeline["apply_intercorr"]:
        prefilter = reduce_correlated_features(
            split_train_df=train_k,
            target_col=target_col,
            candidate_features=kept,
            correlation_threshold=DEFAULT_INTERCORR_THRESHOLD,
            reduction_mode=DEFAULT_INTERCORR_REDUCTION_MODE,
            importance_metric=DEFAULT_INTERCORR_IMPORTANCE_METRIC,
        )
        after_intercorr = list(prefilter) if len(prefilter) > 0 else list(kept[:1])

    selected = list(after_intercorr)
    if pipeline["selector_name"] == "sfs":
        n_select = _selection_max_features(
            len(train_k), float(pipeline["features_frac"])
        )
        try:
            selected = sequential_forward_selector(
                train_k,
                target_col,
                after_intercorr,
                n_features_to_select=n_select,
                cv=SFS_INNER_CV,
                scoring="spearman",
                random_state=DEFAULT_RANDOM_STATE,
                min_improvement=DEFAULT_SFS_MIN_IMPROVEMENT,
                n_jobs=1,
                model_type=str(pipeline["selector_model"]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"SFS failed for {target_col} with selector="
                f"{pipeline['selector_model']}: {exc}"
            ) from exc
        allowed = set(after_intercorr)
        selected = [str(feature) for feature in selected if str(feature) in allowed]
        if not selected:
            raise RuntimeError(f"SFS returned no features for {target_col}")

    scaler_stats = {
        feature: {
            "data_min": float(scaler.data_min_[idx]),
            "data_max": float(scaler.data_max_[idx]),
            "scale": float(scaler.scale_[idx]),
            "min": float(scaler.min_[idx]),
        }
        for idx, feature in enumerate(cand)
    }
    return {
        "train_df": train_k,
        "test_df": test_k,
        "input_features": cand,
        "after_lowvar": list(kept),
        "after_intercorr": list(after_intercorr),
        "selected_features": list(selected),
        "scaling_method": "minmax_train_fit",
        "scaler_stats": scaler_stats,
    }


def _make_elasticnet(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=100_000,
                    tol=1e-5,
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ]
    )


def _xy(
    df: pd.DataFrame, target_col: str, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    x = df.loc[mask, features].apply(pd.to_numeric, errors="coerce")
    names = (
        df.loc[mask, "name"].astype(str)
        if "name" in df.columns
        else pd.Series([str(i) for i in range(int(mask.sum()))], index=y.loc[mask].index)
    )
    return x, y.loc[mask], names


def _extract_attributions(
    model: Any,
    *,
    model_type: str,
    feature_names: list[str],
    scaling_method: str,
    scaler_stats: dict[str, dict[str, float]] | None,
    pipeline_model: Pipeline | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mt = str(model_type).lower()
    if mt in {"linear", "svm", "elasticnet"}:
        if pipeline_model is not None:
            enet = pipeline_model.named_steps["elasticnet"]
            scaler = pipeline_model.named_steps["scaler"]
            coef_model = np.ravel(np.asarray(enet.coef_, dtype=float))
            intercept_model = float(np.ravel(enet.intercept_)[0])
            coef_original = coef_model / scaler.scale_
            intercept_original = float(
                intercept_model
                - np.sum(coef_model * scaler.mean_ / scaler.scale_)
            )
            abs_coef = np.abs(coef_model)
            abs_sum = float(abs_coef.sum())
            ranks = _ranks_desc(abs_coef)
            for idx, feature in enumerate(feature_names):
                rows.append(
                    {
                        "feature": feature,
                        "coef_model_space": float(coef_model[idx]),
                        "coef_abs_model_space": float(abs_coef[idx]),
                        "coef_l1_norm": (
                            float(abs_coef[idx] / abs_sum) if abs_sum > 0 else 0.0
                        ),
                        "coef_original_units": float(coef_original[idx]),
                        "is_nonzero": bool(abs_coef[idx] > 0),
                        "importance_mdi": None,
                        "rank_by_abs": int(ranks[idx]),
                        "intercept_model_space": intercept_model,
                        "intercept_original_units": intercept_original,
                        "scaling_method": scaling_method,
                    }
                )
            return rows

        coef_model = np.ravel(np.asarray(model.coef_, dtype=float))
        intercept_model = float(np.ravel(model.intercept_)[0])
        abs_coef = np.abs(coef_model)
        abs_sum = float(abs_coef.sum())
        ranks = _ranks_desc(abs_coef)
        for idx, feature in enumerate(feature_names):
            stats = (scaler_stats or {}).get(feature, {})
            scale = float(stats.get("scale", np.nan))
            min_off = float(stats.get("min", np.nan))
            coef_original = (
                float(coef_model[idx] * scale) if np.isfinite(scale) else None
            )
            rows.append(
                {
                    "feature": feature,
                    "coef_model_space": float(coef_model[idx]),
                    "coef_abs_model_space": float(abs_coef[idx]),
                    "coef_l1_norm": (
                        float(abs_coef[idx] / abs_sum) if abs_sum > 0 else 0.0
                    ),
                    "coef_original_units": coef_original,
                    "is_nonzero": bool(abs_coef[idx] > 0),
                    "importance_mdi": None,
                    "rank_by_abs": int(ranks[idx]),
                    "intercept_model_space": intercept_model,
                    "intercept_original_units": (
                        float(intercept_model + np.dot(coef_model, [
                            float((scaler_stats or {}).get(f, {}).get("min", 0.0))
                            for f in feature_names
                        ]))
                        if scaler_stats is not None
                        else None
                    ),
                    "scaling_method": scaling_method,
                    "scaler_data_min": stats.get("data_min"),
                    "scaler_data_max": stats.get("data_max"),
                    "scaler_scale": stats.get("scale"),
                    "scaler_min": min_off if np.isfinite(min_off) else None,
                }
            )
        return rows

    if mt == "randomforest":
        importance = np.asarray(model.feature_importances_, dtype=float)
        ranks = _ranks_desc(importance)
        for idx, feature in enumerate(feature_names):
            rows.append(
                {
                    "feature": feature,
                    "coef_model_space": None,
                    "coef_abs_model_space": None,
                    "coef_l1_norm": None,
                    "coef_original_units": None,
                    "is_nonzero": True,
                    "importance_mdi": float(importance[idx]),
                    "rank_by_abs": int(ranks[idx]),
                    "intercept_model_space": None,
                    "intercept_original_units": None,
                    "scaling_method": scaling_method,
                }
            )
        return rows

    # knn or unknown: usage only
    for feature in feature_names:
        rows.append(
            {
                "feature": feature,
                "coef_model_space": None,
                "coef_abs_model_space": None,
                "coef_l1_norm": None,
                "coef_original_units": None,
                "is_nonzero": True,
                "importance_mdi": None,
                "rank_by_abs": None,
                "intercept_model_space": None,
                "intercept_original_units": None,
                "scaling_method": scaling_method,
            }
        )
    return rows


def _permutation_rows(
    model: Any,
    x_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=PERM_REPEATS,
            random_state=DEFAULT_RANDOM_STATE,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
    means = np.asarray(result.importances_mean, dtype=float)
    stds = np.asarray(result.importances_std, dtype=float)
    ranks = _ranks_desc(means)
    return {
        feature: {
            "perm_importance_mean": float(means[idx]),
            "perm_importance_std": float(stds[idx]),
            "perm_importance_rank": int(ranks[idx]),
        }
        for idx, feature in enumerate(feature_names)
    }


def _fit_predict_selector(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    selected_features: list[str],
    eval_model: str,
    eval_hp_by_model: dict[str, dict],
    scaling_method: str,
    scaler_stats: dict[str, dict[str, float]] | None,
    compute_attribution: bool,
) -> dict[str, Any]:
    cols = [
        feature
        for feature in selected_features
        if feature in train_df.columns and feature in test_df.columns
    ]
    if not cols:
        raise RuntimeError(
            f"No selected features present for eval_model={eval_model!r} on {target_col}"
        )
    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    y_test = pd.to_numeric(test_df[target_col], errors="coerce")
    train_mask = y_train.notna()
    test_mask = y_test.notna()
    y_train = y_train.loc[train_mask]
    y_test = y_test.loc[test_mask]
    if len(y_train) < 2 or len(y_test) < 1:
        raise RuntimeError(f"Insufficient labeled rows for {target_col}")
    x_train = train_df.loc[train_mask, cols].apply(pd.to_numeric, errors="coerce")
    x_test = test_df.loc[test_mask, cols].apply(pd.to_numeric, errors="coerce")
    names = test_df.loc[test_mask, "name"].astype(str).tolist()

    model_kwargs: dict[str, Any] = {
        "random_state": DEFAULT_RANDOM_STATE,
        "n_jobs": 1,
        "n_samples_fit": len(y_train),
    }
    model_kwargs.update(eval_hp_by_model.get(eval_model, {}))
    model = make_regressor(eval_model, **model_kwargs)
    fit_regressor(model, x_train, y_train)
    yhat = np.asarray(model.predict(x_test), dtype=np.float64).ravel()
    metrics = _metric_bundle(y_test, yhat)
    evaluation = {
        **{f"{key}": metrics[key] for key in ("spearman", "pearson_r", "r2", "mse")},
        "spearman_rho": metrics["spearman"],
        "spearman_p": metrics["spearman_p"],
        "pearson_p": metrics["pearson_p"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_used_at_eval": int(len(cols)),
        "eval_features_used": list(cols),
    }
    attributions: list[dict[str, Any]] = []
    if compute_attribution:
        attributions = _extract_attributions(
            model,
            model_type=eval_model,
            feature_names=cols,
            scaling_method=scaling_method,
            scaler_stats=scaler_stats,
        )
        perm = _permutation_rows(model, x_test, y_test, cols)
        for row in attributions:
            row.update(perm.get(row["feature"], {}))
    return {
        "names": names,
        "y": pd.Series(y_test.to_numpy(dtype=float)),
        "yhat": pd.Series(yhat),
        "evaluation": evaluation,
        "attributions": attributions,
        "final_features": cols,
    }


def _build_feature_usage_rows(
    *,
    input_features: list[str],
    after_lowvar: list[str],
    after_intercorr: list[str],
    selected_features: list[str],
    final_features: list[str],
    attributions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attr_map = {row["feature"]: row for row in attributions}
    lowvar = set(after_lowvar)
    intercorr = set(after_intercorr)
    selected = set(selected_features)
    final = set(final_features)
    rows: list[dict[str, Any]] = []
    for feature in input_features:
        attr = attr_map.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "survived_lowvar": feature in lowvar,
                "survived_intercorr": feature in intercorr,
                "selected_by_sfs_or_prune": feature in selected,
                "final_model_input": feature in final,
                "is_nonzero": attr.get("is_nonzero"),
                "coef_model_space": attr.get("coef_model_space"),
                "coef_abs_model_space": attr.get("coef_abs_model_space"),
                "coef_l1_norm": attr.get("coef_l1_norm"),
                "coef_original_units": attr.get("coef_original_units"),
                "importance_mdi": attr.get("importance_mdi"),
                "rank_by_abs": attr.get("rank_by_abs"),
                "perm_importance_mean": attr.get("perm_importance_mean"),
                "perm_importance_std": attr.get("perm_importance_std"),
                "perm_importance_rank": attr.get("perm_importance_rank"),
                "intercept_model_space": attr.get("intercept_model_space"),
                "intercept_original_units": attr.get("intercept_original_units"),
                "scaling_method": attr.get("scaling_method"),
                "scaler_data_min": attr.get("scaler_data_min"),
                "scaler_data_max": attr.get("scaler_data_max"),
                "scaler_scale": attr.get("scaler_scale"),
                "scaler_min": attr.get("scaler_min"),
            }
        )
    return rows


def _select_best_eval_model(
    inner_dir: Path,
    *,
    target_col: str,
    candidate_features: list[str],
    pipeline: dict[str, Any],
    eval_hp_by_model: dict[str, dict],
) -> tuple[str, float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    meta = json.loads((inner_dir / "meta.json").read_text())
    n_inner = int(meta["n_splits"])
    eval_models = list(pipeline["eval_models"])
    inner_scores: list[dict[str, Any]] = []
    inner_feature_rows: list[dict[str, Any]] = []
    pooled_by_model: dict[str, tuple[list[pd.Series], list[pd.Series]]] = {
        model: ([], []) for model in eval_models
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for inner_k in range(n_inner):
            train = pd.read_parquet(inner_dir / f"fold_{inner_k}_train.parquet")
            val = pd.read_parquet(inner_dir / f"fold_{inner_k}_test.parquet")
            selection = _run_selector_feature_selection(
                train,
                val,
                target_col=target_col,
                candidate_features=candidate_features,
                pipeline=pipeline,
            )
            for feature in selection["selected_features"]:
                inner_feature_rows.append(
                    {
                        "inner_fold": inner_k,
                        "feature": feature,
                        "stage": "selected",
                    }
                )
            for eval_model in eval_models:
                fit = _fit_predict_selector(
                    selection["train_df"],
                    selection["test_df"],
                    target_col=target_col,
                    selected_features=selection["selected_features"],
                    eval_model=eval_model,
                    eval_hp_by_model=eval_hp_by_model,
                    scaling_method=selection["scaling_method"],
                    scaler_stats=selection["scaler_stats"],
                    compute_attribution=False,
                )
                pooled_by_model[eval_model][0].append(fit["y"])
                pooled_by_model[eval_model][1].append(fit["yhat"])
                inner_scores.append(
                    {
                        "eval_model": eval_model,
                        "inner_fold": inner_k,
                        "spearman": fit["evaluation"]["spearman"],
                        "r2": fit["evaluation"]["r2"],
                        "mse": fit["evaluation"]["mse"],
                        "pearson_r": fit["evaluation"]["pearson_r"],
                        "n_selected_features": len(selection["selected_features"]),
                        "selected_features": list(selection["selected_features"]),
                    }
                )

    ranked: list[tuple[str, float | None]] = []
    for eval_model in eval_models:
        ys, yhats = pooled_by_model[eval_model]
        pooled = _spearman(
            pd.concat(ys, ignore_index=True),
            pd.concat(yhats, ignore_index=True),
        )
        ranked.append((eval_model, pooled))
        for row in inner_scores:
            if row["eval_model"] == eval_model:
                row["pooled_inner_spearman"] = pooled

    valid = [(model, score) for model, score in ranked if score is not None]
    if not valid:
        raise RuntimeError(f"No valid inner eval-model score for {target_col}")
    best_model, best_score = max(valid, key=lambda item: (item[1], item[0]))
    return best_model, best_score, inner_scores, inner_feature_rows


def _run_outer_selector(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    pipeline_key: str,
    pipeline: dict[str, Any],
    work_root: Path,
    eval_hp_by_model: dict[str, dict],
) -> dict[str, Any]:
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")
    inner_dir = work_root / target_col / f"outer_{outer_k}" / "inner_folds"
    write_inner_fold_dir(fold_dir, outer_k=outer_k, dest=inner_dir)
    meta = json.loads((inner_dir / "meta.json").read_text())
    n_inner = int(meta["n_splits"])

    if len(pipeline["eval_models"]) == 1:
        best_eval = pipeline["eval_models"][0]
        inner_scores: list[dict[str, Any]] = []
        inner_feature_rows: list[dict[str, Any]] = []
        ys: list[pd.Series] = []
        yhats: list[pd.Series] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            for inner_k in range(n_inner):
                train = pd.read_parquet(inner_dir / f"fold_{inner_k}_train.parquet")
                val = pd.read_parquet(inner_dir / f"fold_{inner_k}_test.parquet")
                selection = _run_selector_feature_selection(
                    train,
                    val,
                    target_col=target_col,
                    candidate_features=features,
                    pipeline=pipeline,
                )
                for feature in selection["selected_features"]:
                    inner_feature_rows.append(
                        {
                            "inner_fold": inner_k,
                            "feature": feature,
                            "stage": "selected",
                        }
                    )
                fit = _fit_predict_selector(
                    selection["train_df"],
                    selection["test_df"],
                    target_col=target_col,
                    selected_features=selection["selected_features"],
                    eval_model=best_eval,
                    eval_hp_by_model=eval_hp_by_model,
                    scaling_method=selection["scaling_method"],
                    scaler_stats=selection["scaler_stats"],
                    compute_attribution=False,
                )
                ys.append(fit["y"])
                yhats.append(fit["yhat"])
                inner_scores.append(
                    {
                        "eval_model": best_eval,
                        "inner_fold": inner_k,
                        "spearman": fit["evaluation"]["spearman"],
                        "r2": fit["evaluation"]["r2"],
                        "mse": fit["evaluation"]["mse"],
                        "pearson_r": fit["evaluation"]["pearson_r"],
                        "n_selected_features": len(selection["selected_features"]),
                        "selected_features": list(selection["selected_features"]),
                    }
                )
        best_inner = _spearman(
            pd.concat(ys, ignore_index=True),
            pd.concat(yhats, ignore_index=True),
        )
        for row in inner_scores:
            row["pooled_inner_spearman"] = best_inner
    else:
        best_eval, best_inner, inner_scores, inner_feature_rows = _select_best_eval_model(
            inner_dir,
            target_col=target_col,
            candidate_features=features,
            pipeline=pipeline,
            eval_hp_by_model=eval_hp_by_model,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        selection = _run_selector_feature_selection(
            outer_train,
            outer_test,
            target_col=target_col,
            candidate_features=features,
            pipeline=pipeline,
        )
        fit = _fit_predict_selector(
            selection["train_df"],
            selection["test_df"],
            target_col=target_col,
            selected_features=selection["selected_features"],
            eval_model=best_eval,
            eval_hp_by_model=eval_hp_by_model,
            scaling_method=selection["scaling_method"],
            scaler_stats=selection["scaler_stats"],
            compute_attribution=True,
        )

    feature_usage = _build_feature_usage_rows(
        input_features=selection["input_features"],
        after_lowvar=selection["after_lowvar"],
        after_intercorr=selection["after_intercorr"],
        selected_features=selection["selected_features"],
        final_features=fit["final_features"],
        attributions=fit["attributions"],
    )
    print(
        f"{pipeline_key} {target_col} outer {outer_k}: "
        f"rho={fit['evaluation']['spearman']} r2={fit['evaluation']['r2']} "
        f"eval={best_eval} n_selected={len(selection['selected_features'])}",
        flush=True,
    )
    return {
        "pipeline": pipeline_key,
        "pipeline_label": pipeline["label"],
        "target_col": target_col,
        "outer_fold": outer_k,
        "n_test": int(fit["evaluation"]["n_test"]),
        "n_train": int(fit["evaluation"]["n_train"]),
        "spearman": fit["evaluation"]["spearman"],
        "spearman_p": fit["evaluation"].get("spearman_p"),
        "pearson_r": fit["evaluation"]["pearson_r"],
        "pearson_p": fit["evaluation"].get("pearson_p"),
        "r2": fit["evaluation"]["r2"],
        "mse": fit["evaluation"]["mse"],
        "eval_model": best_eval,
        "alpha": None,
        "l1_ratio": None,
        "inner_pooled_spearman": best_inner,
        "n_selected_features": len(selection["selected_features"]),
        "n_final_features": len(fit["final_features"]),
        "n_nonzero": sum(
            1 for row in feature_usage if row.get("is_nonzero") and row["final_model_input"]
        ),
        "selected_features": selection["selected_features"],
        "final_features": fit["final_features"],
        "feature_usage": feature_usage,
        "inner_scores": inner_scores,
        "inner_feature_rows": inner_feature_rows,
        "grid_scores": [],
        "outer_evaluation": fit["evaluation"],
        "n_inner_folds": n_inner,
        "selection_rule": "max pooled inner spearman",
        "oof_rows": [
            {
                "name": str(name),
                "target_col": target_col,
                "outer_fold": outer_k,
                "pipeline": pipeline_key,
                "eval_model": best_eval,
                "alpha": None,
                "l1_ratio": None,
                "y": float(y),
                "yhat": float(yhat),
            }
            for name, y, yhat in zip(fit["names"], fit["y"], fit["yhat"])
        ],
    }


def _run_outer_elasticnet(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    pipeline_key: str,
    pipeline: dict[str, Any],
    work_root: Path,
) -> dict[str, Any]:
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")
    inner_dir = work_root / target_col / f"outer_{outer_k}" / "inner_folds"
    write_inner_fold_dir(fold_dir, outer_k=outer_k, dest=inner_dir)
    n_inner = int(json.loads((inner_dir / "meta.json").read_text())["n_splits"])
    alphas = list(pipeline["alphas"])
    l1_ratios = list(pipeline["l1_ratios"])

    grid_scores: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ConstantInputWarning)
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                ys: list[pd.Series] = []
                yhats: list[pd.Series] = []
                fold_scores: list[float | None] = []
                fold_r2: list[float | None] = []
                for inner_k in range(n_inner):
                    train = pd.read_parquet(
                        inner_dir / f"fold_{inner_k}_train.parquet"
                    )
                    val = pd.read_parquet(inner_dir / f"fold_{inner_k}_test.parquet")
                    x_train, y_train, _ = _xy(train, target_col, features)
                    x_val, y_val, _ = _xy(val, target_col, features)
                    model = _make_elasticnet(alpha, l1_ratio)
                    model.fit(x_train, y_train)
                    pred = pd.Series(model.predict(x_val), index=y_val.index)
                    ys.append(y_val)
                    yhats.append(pred)
                    metrics = _metric_bundle(y_val, pred)
                    fold_scores.append(metrics["spearman"])
                    fold_r2.append(metrics["r2"])
                pooled_y = pd.concat(ys, ignore_index=True)
                pooled_yhat = pd.concat(yhats, ignore_index=True)
                pooled = _metric_bundle(pooled_y, pooled_yhat)
                grid_scores.append(
                    {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "pooled_inner_spearman": pooled["spearman"],
                        "pooled_inner_r2": pooled["r2"],
                        "inner_fold_spearman": fold_scores,
                        "inner_fold_r2": fold_r2,
                    }
                )

        valid = [row for row in grid_scores if row["pooled_inner_spearman"] is not None]
        if not valid:
            raise RuntimeError(
                f"No valid inner score for {target_col}, outer fold {outer_k}"
            )
        best = max(
            valid,
            key=lambda row: (
                row["pooled_inner_spearman"],
                row["alpha"],
                row["l1_ratio"],
            ),
        )
        x_train, y_train, _ = _xy(outer_train, target_col, features)
        x_test, y_test, names = _xy(outer_test, target_col, features)
        model = _make_elasticnet(best["alpha"], best["l1_ratio"])
        model.fit(x_train, y_train)
        pred = pd.Series(model.predict(x_test), index=y_test.index)

    metrics = _metric_bundle(y_test, pred)
    attributions = _extract_attributions(
        model.named_steps["elasticnet"],
        model_type="elasticnet",
        feature_names=features,
        scaling_method="standardize_after_median_impute",
        scaler_stats=None,
        pipeline_model=model,
    )
    perm = _permutation_rows(model, x_test, y_test, features)
    for row in attributions:
        row.update(perm.get(row["feature"], {}))
    feature_usage = _build_feature_usage_rows(
        input_features=features,
        after_lowvar=features,
        after_intercorr=features,
        selected_features=features,
        final_features=features,
        attributions=attributions,
    )
    nonzero_features = [
        row["feature"] for row in attributions if row.get("is_nonzero")
    ]
    print(
        f"{pipeline_key} {target_col} outer {outer_k}: "
        f"rho={metrics['spearman']} r2={metrics['r2']} "
        f"alpha={best['alpha']} l1={best['l1_ratio']} n_nonzero={len(nonzero_features)}",
        flush=True,
    )
    inner_scores = [
        {
            "eval_model": "elasticnet",
            "inner_fold": inner_k,
            "alpha": row["alpha"],
            "l1_ratio": row["l1_ratio"],
            "spearman": row["inner_fold_spearman"][inner_k]
            if inner_k < len(row["inner_fold_spearman"])
            else None,
            "r2": row["inner_fold_r2"][inner_k]
            if inner_k < len(row["inner_fold_r2"])
            else None,
            "pooled_inner_spearman": row["pooled_inner_spearman"],
            "is_best": (
                row["alpha"] == best["alpha"] and row["l1_ratio"] == best["l1_ratio"]
            ),
        }
        for row in grid_scores
        for inner_k in range(n_inner)
    ]
    return {
        "pipeline": pipeline_key,
        "pipeline_label": pipeline["label"],
        "target_col": target_col,
        "outer_fold": outer_k,
        "n_test": int(metrics["n"]),
        "n_train": int(len(y_train)),
        "spearman": metrics["spearman"],
        "spearman_p": metrics["spearman_p"],
        "pearson_r": metrics["pearson_r"],
        "pearson_p": metrics["pearson_p"],
        "r2": metrics["r2"],
        "mse": metrics["mse"],
        "eval_model": "elasticnet",
        "alpha": best["alpha"],
        "l1_ratio": best["l1_ratio"],
        "inner_pooled_spearman": best["pooled_inner_spearman"],
        "n_selected_features": len(features),
        "n_final_features": len(features),
        "n_nonzero": len(nonzero_features),
        "selected_features": list(features),
        "final_features": list(features),
        "nonzero_features": nonzero_features,
        "feature_usage": feature_usage,
        "inner_scores": inner_scores,
        "inner_feature_rows": [],
        "grid_scores": grid_scores,
        "outer_evaluation": {
            **metrics,
            "n_train": int(len(y_train)),
            "n_test": int(metrics["n"]),
            "n_features_used_at_eval": len(features),
            "eval_features_used": list(features),
        },
        "n_inner_folds": n_inner,
        "selection_rule": "max pooled inner spearman",
        "oof_rows": [
            {
                "name": str(name),
                "target_col": target_col,
                "outer_fold": outer_k,
                "pipeline": pipeline_key,
                "eval_model": "elasticnet",
                "alpha": best["alpha"],
                "l1_ratio": best["l1_ratio"],
                "y": float(y),
                "yhat": float(yhat),
            }
            for name, y, yhat in zip(names, y_test, pred)
        ],
    }


def _run_outer(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    pipeline_key: str,
    pipeline: dict[str, Any],
    work_root: Path,
    eval_hp_by_model: dict[str, dict],
) -> dict[str, Any]:
    if pipeline["kind"] == "elasticnet":
        return _run_outer_elasticnet(
            fold_dir,
            target_col=target_col,
            outer_k=outer_k,
            features=features,
            pipeline_key=pipeline_key,
            pipeline=pipeline,
            work_root=work_root,
        )
    return _run_outer_selector(
        fold_dir,
        target_col=target_col,
        outer_k=outer_k,
        features=features,
        pipeline_key=pipeline_key,
        pipeline=pipeline,
        work_root=work_root,
        eval_hp_by_model=eval_hp_by_model,
    )


def _task_checkpoint(out_dir: Path, task: dict[str, Any]) -> Path:
    return (
        out_dir
        / "checkpoints"
        / task["pipeline"]
        / METHOD_SLUGS[task["method"]]
        / task["variant"]
        / task["Dataset_stem"]
        / task["split"]
        / task["target_col"]
        / f"outer_{task['outer_k']}.json"
    )


def _checkpoint_matches_task(result: dict[str, Any], task: dict[str, Any]) -> bool:
    expected = {
        "pipeline": task["pipeline"],
        "method": task["method"],
        "variant": task["variant"],
        "structure_model": task["structure_model"],
        "Dataset_stem": task["Dataset_stem"],
        "split": task["split"],
        "target_col": task["target_col"],
        "outer_fold": task["outer_k"],
        "fold_root": str(task["fold_dir"]),
        "n_input_features": len(task["features"]),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            return False
    required = {
        "oof_rows",
        "feature_usage",
        "r2",
        "mse",
        "pearson_r",
        "selected_features",
        "final_features",
    }
    if not required.issubset(result):
        return False
    oof_rows = result.get("oof_rows")
    if not isinstance(oof_rows, list) or len(oof_rows) == 0:
        return False
    if int(result.get("n_test", -1)) != len(oof_rows):
        return False
    names = [str(row.get("name", "")) for row in oof_rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        return False
    for row in oof_rows:
        try:
            y = float(row["y"])
            yhat = float(row["yhat"])
        except (KeyError, TypeError, ValueError):
            return False
        if not np.isfinite(y) or not np.isfinite(yhat):
            return False
    return True


def _load_checkpoint(path: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return result if _checkpoint_matches_task(result, task) else None


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    result = _run_outer(
        task["fold_dir"],
        target_col=task["target_col"],
        outer_k=task["outer_k"],
        features=task["features"],
        pipeline_key=task["pipeline"],
        pipeline=PIPELINES[task["pipeline"]],
        work_root=task["work_root"],
        eval_hp_by_model=task["eval_hp_by_model"],
    )
    result.update(
        {
            "method": task["method"],
            "variant": task["variant"],
            "structure_model": task["structure_model"],
            "Dataset_stem": task["Dataset_stem"],
            "split": task["split"],
            "n_input_features": len(task["features"]),
            "fold_root": str(task["fold_dir"]),
        }
    )
    return result


def _manifest_fingerprint(manifest: pd.DataFrame) -> str:
    if manifest.empty:
        raise RuntimeError("Empty task manifest; refusing to fingerprint")
    payload = manifest.sort_values(
        [
            "pipeline",
            "method",
            "variant",
            "Dataset_stem",
            "split",
            "Target_col",
            "fold_root",
        ]
    ).to_json(orient="records")
    return hashlib.sha1(payload.encode()).hexdigest()


def _build_pipeline_tasks(
    *,
    out_dir: Path,
    eval_hp_by_model: dict[str, dict],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    kitab = _discover_kitab()
    propermab = _discover_propermab()
    tap = _discover_tap()
    structural_keys = set(kitab) | set(propermab)
    if set(kitab) != set(propermab):
        raise RuntimeError(
            "kitAb/ProperMAb structural root mismatch: "
            f"kitAb-only={sorted(set(kitab) - set(propermab))}; "
            f"ProperMAb-only={sorted(set(propermab) - set(kitab))}"
        )
    tap_keys = {(stem, split) for stem, split, _, _ in structural_keys}
    if set(tap) != tap_keys:
        raise RuntimeError(
            f"TAP root mismatch: missing={sorted(tap_keys - set(tap))}; "
            f"extra={sorted(set(tap) - tap_keys)}"
        )

    configs: list[dict[str, Any]] = []
    for method, root_map, mode, suffix in (
        ("kitAb", kitab, "kitab", ""),
        ("PROPERMAB", propermab, "propermab", "_propermab"),
    ):
        for (stem, split, backend, variant), root in sorted(root_map.items()):
            configs.append(
                {
                    "method": method,
                    "variant": f"{backend}_{variant}{suffix}",
                    "structure_model": backend,
                    "Dataset_stem": stem,
                    "split": split,
                    "root": root,
                    "feature_mode": mode,
                }
            )

    for (stem, split, backend, variant), root in sorted(propermab.items()):
        if backend != "abb2" or variant != 1:
            continue
        configs.append(
            {
                "method": "Sequence features baseline",
                "variant": f"abb2_{variant}_propermab",
                "structure_model": "abb2",
                "Dataset_stem": stem,
                "split": split,
                "root": root,
                "feature_mode": "sequence",
            }
        )

    for (stem, split), root in sorted(tap.items()):
        configs.append(
            {
                "method": "TAP",
                "variant": "tap",
                "structure_model": "tap",
                "Dataset_stem": stem,
                "split": split,
                "root": root,
                "feature_mode": "tap",
            }
        )

    structural_targets: dict[tuple[str, str], set[str]] = {}
    for config in configs:
        if config["method"] not in {"kitAb", "PROPERMAB"}:
            continue
        targets = {
            name
            for name in _target_dirs(config["root"])
            if name not in EXCLUDED_TARGETS
        }
        key = (config["Dataset_stem"], config["split"])
        previous = structural_targets.get(key)
        if previous is None:
            structural_targets[key] = targets
        elif previous != targets:
            raise RuntimeError(
                f"Target mismatch for {key}: {sorted(previous ^ targets)}"
            )

    tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for pipeline_key, pipeline in PIPELINES.items():
        for config in configs:
            targets = _target_dirs(config["root"])
            if not targets:
                raise FileNotFoundError(f"No target folds under {config['root']}")
            for target_col, fold_dir in sorted(targets.items()):
                if target_col in EXCLUDED_TARGETS:
                    continue
                meta = json.loads((fold_dir / "meta.json").read_text())
                n_splits = int(meta["n_splits"])
                features = _features(
                    fold_dir,
                    target_col,
                    feature_mode=config["feature_mode"],
                )
                manifest_rows.append(
                    {
                        "pipeline": pipeline_key,
                        "pipeline_label": pipeline["label"],
                        **{
                            key: config[key]
                            for key in (
                                "method",
                                "variant",
                                "structure_model",
                                "Dataset_stem",
                                "split",
                            )
                        },
                        "Target_col": target_col,
                        "fold_root": str(fold_dir),
                        "n_outer_folds": n_splits,
                        "n_input_features": len(features),
                        "features_frac": pipeline["features_frac"],
                        "eval_models": ",".join(pipeline["eval_models"]),
                    }
                )
                for outer_k in range(n_splits):
                    task = {
                        "pipeline": pipeline_key,
                        **{
                            key: config[key]
                            for key in (
                                "method",
                                "variant",
                                "structure_model",
                                "Dataset_stem",
                                "split",
                            )
                        },
                        "target_col": target_col,
                        "fold_dir": fold_dir,
                        "features": features,
                        "outer_k": outer_k,
                        "eval_hp_by_model": eval_hp_by_model,
                        "work_root": (
                            out_dir
                            / "work"
                            / pipeline_key
                            / METHOD_SLUGS[config["method"]]
                            / config["variant"]
                            / config["Dataset_stem"]
                            / config["split"]
                        ),
                    }
                    task["checkpoint"] = _task_checkpoint(out_dir, task)
                    tasks.append(task)
    return tasks, pd.DataFrame(manifest_rows)


def _identity_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline": result["pipeline"],
        "method": result["method"],
        "variant": result["variant"],
        "structure_model": result["structure_model"],
        "Dataset_stem": result["Dataset_stem"],
        "split": result["split"],
        "target_col": result["target_col"],
        "outer_fold": result["outer_fold"],
    }


def _pooled_metrics(group: pd.DataFrame) -> dict[str, Any]:
    metrics = _metric_bundle(group["y"], group["yhat"])
    return {
        "Spearman_pooled_oof": metrics["spearman"],
        "Pearson_pooled_oof": metrics["pearson_r"],
        "R2_pooled_oof": metrics["r2"],
        "MSE_pooled_oof": metrics["mse"],
        "n_oof": int(metrics["n"]),
    }


def _summarize(results: list[dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    predictions_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    features_dir = out_dir / "features"
    inner_dir = out_dir / "inner_selection"
    for path in (predictions_dir, metrics_dir, features_dir, inner_dir):
        path.mkdir(parents=True, exist_ok=True)

    oof_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    inner_score_rows: list[dict[str, Any]] = []
    inner_feature_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    feature_sets: dict[str, dict[str, list[str]]] = {}

    for result in results:
        identity = _identity_from_result(result)
        for row in result["oof_rows"]:
            oof_rows.append({**row, **{k: result[k] for k in (
                "method", "variant", "structure_model", "Dataset_stem", "split"
            )}})
        outer_rows.append(
            {
                **identity,
                "pipeline_label": result.get("pipeline_label"),
                "n_test": result.get("n_test"),
                "n_train": result.get("n_train"),
                "spearman": result.get("spearman"),
                "spearman_p": result.get("spearman_p"),
                "pearson_r": result.get("pearson_r"),
                "pearson_p": result.get("pearson_p"),
                "r2": result.get("r2"),
                "mse": result.get("mse"),
                "eval_model": result.get("eval_model"),
                "alpha": result.get("alpha"),
                "l1_ratio": result.get("l1_ratio"),
                "inner_pooled_spearman": result.get("inner_pooled_spearman"),
                "n_selected_features": result.get("n_selected_features"),
                "n_final_features": result.get("n_final_features"),
                "n_nonzero": result.get("n_nonzero"),
                "n_input_features": result.get("n_input_features"),
                "n_inner_folds": result.get("n_inner_folds"),
                "selection_rule": result.get("selection_rule"),
                "fold_root": result.get("fold_root"),
            }
        )
        choice_rows.append(
            {
                **identity,
                "chosen_eval_model": result.get("eval_model"),
                "chosen_alpha": result.get("alpha"),
                "chosen_l1_ratio": result.get("l1_ratio"),
                "inner_pooled_spearman_at_choice": result.get("inner_pooled_spearman"),
                "n_inner_folds": result.get("n_inner_folds"),
                "selection_rule": result.get("selection_rule"),
            }
        )
        for row in result.get("feature_usage", []):
            feature_rows.append({**identity, **row})
        for row in result.get("inner_scores", []):
            inner_score_rows.append({**identity, **row})
        for row in result.get("inner_feature_rows", []):
            inner_feature_rows.append({**identity, **row})
        for row in result.get("grid_scores", []):
            for inner_k, spearman in enumerate(row.get("inner_fold_spearman", [])):
                grid_rows.append(
                    {
                        **identity,
                        "alpha": row["alpha"],
                        "l1_ratio": row["l1_ratio"],
                        "inner_fold": inner_k,
                        "inner_spearman": spearman,
                        "inner_r2": (
                            row.get("inner_fold_r2", [None] * (inner_k + 1))[inner_k]
                            if row.get("inner_fold_r2")
                            else None
                        ),
                        "pooled_inner_spearman": row.get("pooled_inner_spearman"),
                        "pooled_inner_r2": row.get("pooled_inner_r2"),
                        "is_best": (
                            row.get("alpha") == result.get("alpha")
                            and row.get("l1_ratio") == result.get("l1_ratio")
                        ),
                    }
                )

        set_key = "|".join(
            [
                str(result["pipeline"]),
                str(result["method"]),
                str(result["variant"]),
                str(result["Dataset_stem"]),
                str(result["split"]),
                str(result["target_col"]),
            ]
        )
        fold_key = f"fold_{int(result['outer_fold']) + 1}"
        bucket = feature_sets.setdefault(
            set_key,
            {"selected": {}, "final_model_input": {}, "active_nonzero": {}},
        )
        bucket["selected"][fold_key] = list(result.get("selected_features") or [])
        bucket["final_model_input"][fold_key] = list(result.get("final_features") or [])
        nonzero = result.get("nonzero_features")
        if nonzero is None:
            nonzero = [
                row["feature"]
                for row in result.get("feature_usage", [])
                if row.get("final_model_input") and row.get("is_nonzero")
            ]
        bucket["active_nonzero"][fold_key] = list(nonzero)

    oof = pd.DataFrame(oof_rows)
    oof.to_parquet(predictions_dir / "oof.parquet", index=False)
    oof.to_parquet(out_dir / "oof.parquet", index=False)

    outer = pd.DataFrame(outer_rows)
    outer.to_parquet(metrics_dir / "outer_fold_metrics.parquet", index=False)
    outer.to_csv(metrics_dir / "outer_fold_metrics.csv", index=False)
    outer.to_csv(out_dir / "outer_summary.csv", index=False)

    pd.DataFrame(choice_rows).to_parquet(
        inner_dir / "outer_choice.parquet", index=False
    )
    if inner_score_rows:
        # Drop bulky selected_features lists from long score table.
        slim_scores = []
        for row in inner_score_rows:
            slim = dict(row)
            slim.pop("selected_features", None)
            slim_scores.append(slim)
        pd.DataFrame(slim_scores).to_parquet(
            inner_dir / "inner_scores_long.parquet", index=False
        )
    if grid_rows:
        pd.DataFrame(grid_rows).to_parquet(
            inner_dir / "inner_grid_scores.parquet", index=False
        )
    if feature_rows:
        pd.DataFrame(feature_rows).to_parquet(
            features_dir / "outer_feature_usage.parquet", index=False
        )
    if inner_feature_rows:
        pd.DataFrame(inner_feature_rows).to_parquet(
            features_dir / "inner_selected_features.parquet", index=False
        )
    (features_dir / "outer_feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2)
    )

    variant_rows: list[dict[str, Any]] = []
    keys = [
        "pipeline",
        "method",
        "variant",
        "structure_model",
        "Dataset_stem",
        "split",
        "target_col",
    ]
    for key, group in oof.groupby(keys, sort=True):
        row = {**dict(zip(keys, key)), **_pooled_metrics(group)}
        mode = group["eval_model"].mode()
        row["eval_model_mode"] = mode.iloc[0] if len(mode) else None
        variant_rows.append(row)
    variant_summary = pd.DataFrame(variant_rows)
    variant_summary.to_csv(metrics_dir / "variant_split_summary.csv", index=False)
    variant_summary.to_csv(out_dir / "variant_split_summary.csv", index=False)

    structure_rows: list[dict[str, Any]] = []
    structure_keys = [
        "pipeline",
        "method",
        "Dataset_stem",
        "target_col",
        "structure_model",
    ]
    for key, group in variant_summary.groupby(structure_keys, sort=True):
        structure_rows.append(
            {
                **dict(zip(structure_keys, key)),
                "structure_model_Spearman": _fisher_mean(
                    group["Spearman_pooled_oof"]
                ),
                "structure_model_Pearson": _fisher_mean(
                    group["Pearson_pooled_oof"]
                ),
                "structure_model_R2": float(
                    pd.to_numeric(group["R2_pooled_oof"], errors="coerce").mean()
                ),
                "structure_model_MSE": float(
                    pd.to_numeric(group["MSE_pooled_oof"], errors="coerce").mean()
                ),
                "n_variant_split_runs": len(group),
            }
        )
    structure_summary = pd.DataFrame(structure_rows)
    structure_summary.to_csv(
        metrics_dir / "structure_model_summary.csv", index=False
    )
    structure_summary.to_csv(out_dir / "structure_model_summary.csv", index=False)

    averaged_rows: list[dict[str, Any]] = []
    average_keys = ["pipeline", "method", "Dataset_stem", "target_col"]
    for key, group in structure_summary.groupby(average_keys, sort=True):
        spearman = group["structure_model_Spearman"]
        pearson = group["structure_model_Pearson"]
        r2 = group["structure_model_R2"]
        mse = group["structure_model_MSE"]
        row = {
            **dict(zip(average_keys, key)),
            "Spearman": _fisher_mean(spearman),
            "Pearson": _fisher_mean(pearson),
            "R2": float(pd.to_numeric(r2, errors="coerce").mean()),
            "MSE": float(pd.to_numeric(mse, errors="coerce").mean()),
            "lower": float(spearman.min()),
            "upper": float(spearman.max()),
            "n_structure_models": len(group),
            "n_variant_split_runs": int(group["n_variant_split_runs"].sum()),
        }
        for record in group.itertuples():
            row[f"{record.structure_model}_Spearman"] = (
                record.structure_model_Spearman
            )
            row[f"{record.structure_model}_R2"] = record.structure_model_R2
        averaged_rows.append(row)
    averaged = pd.DataFrame(averaged_rows)
    averaged.to_csv(metrics_dir / "averaged_results.csv", index=False)
    averaged.to_csv(out_dir / "averaged_results.csv", index=False)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "runs/nested_pipeline_all_methods",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    eval_hp_by_model = parse_eval_hyperparameters_mapping(
        DEFAULT_EVAL_HYPERPARAMETERS_RAW
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "run_config.json"
    run_config: dict[str, Any] = {
        "pipelines": PIPELINES,
        "jobs": args.jobs,
        "eval_hyperparameters": DEFAULT_EVAL_HYPERPARAMETERS_RAW,
        "intercorr_threshold": DEFAULT_INTERCORR_THRESHOLD,
        "intercorr_importance_metric": DEFAULT_INTERCORR_IMPORTANCE_METRIC,
        "intercorr_reduction_mode": DEFAULT_INTERCORR_REDUCTION_MODE,
        "low_variance_relative_std_threshold": (
            DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD
        ),
        "low_variance_epsilon": DEFAULT_LOW_VARIANCE_EPSILON,
        "sfs_inner_cv": SFS_INNER_CV,
        "sfs_min_improvement": DEFAULT_SFS_MIN_IMPROVEMENT,
        "random_state": DEFAULT_RANDOM_STATE,
        "excluded_targets": sorted(EXCLUDED_TARGETS),
        "perm_repeats": PERM_REPEATS,
    }

    tasks, manifest = _build_pipeline_tasks(
        out_dir=args.out_dir,
        eval_hp_by_model=eval_hp_by_model,
    )
    run_config["manifest_sha1"] = _manifest_fingerprint(manifest)

    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but missing {config_path}; "
                "use a fresh --out-dir or omit --resume"
            )
        previous = json.loads(config_path.read_text())
        for key in CONFIG_COMPARE_KEYS:
            if previous.get(key) != run_config[key]:
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"{previous.get(key)!r} != {run_config[key]!r}"
                )
        if previous.get("manifest_sha1") != run_config["manifest_sha1"]:
            raise ValueError(
                "Resume configuration mismatch for manifest_sha1 "
                "(discovered roots/targets/features changed)"
            )
    elif any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"{args.out_dir} is not empty; pass --resume or choose another --out-dir"
        )

    config_path.write_text(json.dumps(run_config, indent=2, default=str))
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    counts = manifest.groupby(["pipeline", "method"]).agg(
        configurations=("variant", "size"),
        dataset_targets=("Target_col", "size"),
        outer_folds=("n_outer_folds", "sum"),
    )
    print(counts.to_string(), flush=True)
    print(
        f"\nTotal: {len(manifest)} pipeline/variant/split/target configurations; "
        f"{len(tasks)} outer-fold tasks",
        flush=True,
    )
    if args.dry_run:
        print("Dry run complete; no models fitted.", flush=True)
        return

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        checkpoint = (
            _load_checkpoint(task["checkpoint"], task) if args.resume else None
        )
        if checkpoint is None:
            pending.append(task)
        else:
            results.append(checkpoint)
    print(
        f"Resume: {len(results)} complete, {len(pending)} pending; "
        f"launching {min(args.jobs, len(pending))} workers",
        flush=True,
    )

    if pending:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(pending)),
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = {pool.submit(_run_task, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                if not _checkpoint_matches_task(result, task):
                    raise RuntimeError(
                        f"Invalid result for {task['checkpoint']}; refusing to write"
                    )
                _write_json_atomic(task["checkpoint"], result)
                results.append(result)
                if index % 25 == 0 or index == len(futures):
                    print(
                        f"Completed {index}/{len(futures)} pending outer folds "
                        f"({len(results)}/{len(tasks)} total)",
                        flush=True,
                    )

    if len(results) != len(tasks):
        raise RuntimeError(f"Expected {len(tasks)} results, got {len(results)}")
    results.sort(
        key=lambda result: (
            result["pipeline"],
            result["method"],
            result["variant"],
            result["Dataset_stem"],
            result["split"],
            result["target_col"],
            result["outer_fold"],
        )
    )
    averaged = _summarize(results, args.out_dir)
    print(
        f"\nCompleted {len(results)} outer folds. "
        f"Wrote {len(averaged)} averaged pipeline/method/dataset/target rows to "
        f"{args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
