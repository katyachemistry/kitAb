"""Cross-validation engine for the four kitAb AutoML techniques.

One call to :func:`run_outer_fold` evaluates a single technique on a single
outer fold. In ``nested`` mode the inner folds (built from the remaining outer
folds, with outer-test samples removed) choose the eval model for the SFS
techniques and the ``(alpha, l1_ratio)`` pair for ElasticNet; the choice is then
refit on the whole outer-train split. In ``flat`` mode selection happens on
outer-train only, which is what small variant series need.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from automl.feature_selectors import sequential_forward_selector
from automl.folds import write_inner_fold_dir
from automl.techniques import PipelineSettings, Technique
from automl.utils import (
    fit_regressor,
    make_regressor,
    parse_eval_hyperparameters_mapping,
    reduce_correlated_features,
    remove_low_variance_features,
)

NAME_COL = "name"


class TechniqueRunError(RuntimeError):
    """A technique could not be evaluated on a fold."""


def json_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def metric_bundle(
    y: pd.Series | np.ndarray, yhat: pd.Series | np.ndarray
) -> dict[str, Any]:
    """Spearman / Pearson / R2 / MSE, with ``None`` where undefined."""
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
        "spearman": json_float(float(rho)),
        "spearman_p": json_float(float(rho_p)),
        "pearson_r": json_float(float(pearson_r)),
        "pearson_p": json_float(float(pearson_p)),
        "r2": json_float(float(r2_score(y_arr, yhat_arr))),
        "mse": json_float(float(mean_squared_error(y_arr, yhat_arr))),
        "n": int(len(y_arr)),
    }


def pooled_spearman(y: pd.Series, yhat: pd.Series) -> float | None:
    return metric_bundle(y, yhat)["spearman"]


def _ranks_desc(values: np.ndarray) -> list[int]:
    order = (-np.asarray(values, dtype=float)).argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks.tolist()


def selection_max_features(n_train: int, features_frac: float) -> int:
    return max(1, int(float(features_frac) * n_train))


def eval_hyperparameters_by_model(settings: PipelineSettings) -> dict[str, dict]:
    return parse_eval_hyperparameters_mapping(settings.eval_hyperparameters)


def _fit_minmax(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_mat = (
        train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    )
    test_mat = (
        test_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    )
    scaler.fit(train_mat)
    train_out[cols] = scaler.transform(train_mat)
    test_out[cols] = scaler.transform(test_mat)
    return train_out, test_out, scaler


def select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    candidate_features: list[str],
    technique: Technique,
    settings: PipelineSettings,
) -> dict[str, Any]:
    """Low-variance filter, optional intercorrelation prune, optional SFS.

    Returns min-max scaled copies of both frames plus the surviving feature
    lists at each stage.
    """
    cand = list(dict.fromkeys(candidate_features))
    train_k, test_k, scaler = _fit_minmax(train_df, test_df, cand)
    kept, _removed, _rel = remove_low_variance_features(
        X=train_k,
        candidate_features=cand,
        relative_std_threshold=settings.low_variance_relative_std_threshold,
        epsilon=settings.low_variance_epsilon,
    )
    if len(kept) == 0:
        kept = cand[:1]
    drop = [feature for feature in cand if feature not in set(kept)]
    train_k = train_k.drop(columns=drop, errors="ignore").copy()
    test_k = test_k.drop(columns=drop, errors="ignore").copy()

    after_intercorr = list(kept)
    if technique.apply_intercorr:
        prefilter = reduce_correlated_features(
            split_train_df=train_k,
            target_col=target_col,
            candidate_features=kept,
            correlation_threshold=settings.intercorr_threshold,
            reduction_mode=settings.intercorr_reduction_mode,
            importance_metric=settings.intercorr_importance_metric,
        )
        after_intercorr = list(prefilter) if len(prefilter) > 0 else list(kept[:1])

    selected = list(after_intercorr)
    if technique.selector == "sfs":
        n_select = selection_max_features(
            len(train_k), float(technique.features_frac)
        )
        try:
            selected = sequential_forward_selector(
                train_k,
                target_col,
                after_intercorr,
                n_features_to_select=n_select,
                cv=settings.sfs.inner_cv,
                scoring="spearman",
                random_state=settings.random_state,
                min_improvement=settings.sfs.min_improvement,
                n_jobs=1,
                model_type=str(technique.selector_model),
            )
        except Exception as exc:
            raise TechniqueRunError(
                f"SFS failed for {target_col} with selector="
                f"{technique.selector_model}: {exc}"
            ) from exc
        allowed = set(after_intercorr)
        selected = [str(feature) for feature in selected if str(feature) in allowed]
        if not selected:
            raise TechniqueRunError(f"SFS returned no features for {target_col}")

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


class RejectMissing(BaseEstimator, TransformerMixin):
    """Fail fast if any feature value is missing; do not impute."""

    def fit(self, X, y=None):
        self._check(X, stage="fit")
        return self

    def transform(self, X):
        self._check(X, stage="transform")
        return X

    @staticmethod
    def _check(X, *, stage: str) -> None:
        arr = np.asarray(X, dtype=float)
        n_missing = int(np.isnan(arr).sum())
        if n_missing:
            raise ValueError(
                f"Missing feature values are not allowed "
                f"({n_missing} NaN(s) during elastic-net {stage})"
            )


def make_elasticnet_pipeline(
    alpha: float, l1_ratio: float, *, random_state: int
) -> Pipeline:
    return Pipeline(
        [
            ("reject_missing", RejectMissing()),
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=100_000,
                    tol=1e-5,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_elasticnet_cv_pipeline(
    alphas: list[float] | tuple[float, ...],
    l1_ratios: list[float] | tuple[float, ...],
    *,
    cv: int,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        [
            ("reject_missing", RejectMissing()),
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNetCV(
                    alphas=[float(a) for a in alphas],
                    l1_ratio=[float(x) for x in l1_ratios],
                    cv=max(2, int(cv)),
                    max_iter=100_000,
                    tol=1e-5,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_selector_pipeline(
    eval_model: str,
    *,
    eval_hp_by_model: dict[str, dict],
    random_state: int,
    n_samples_fit: int,
) -> Pipeline:
    """Min-max scaler + eval regressor, so the estimator consumes raw values."""
    model_kwargs: dict[str, Any] = {
        "random_state": random_state,
        "n_jobs": 1,
        "n_samples_fit": int(n_samples_fit),
    }
    model_kwargs.update(eval_hp_by_model.get(eval_model, {}))
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("model", make_regressor(eval_model, **model_kwargs)),
        ]
    )


def xy(
    df: pd.DataFrame, target_col: str, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    x = df.loc[mask, features].apply(pd.to_numeric, errors="coerce")
    names = (
        df.loc[mask, NAME_COL].astype(str)
        if NAME_COL in df.columns
        else pd.Series([str(i) for i in range(int(mask.sum()))], index=y.loc[mask].index)
    )
    return x, y.loc[mask], names


def extract_attributions(
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
                intercept_model - np.sum(coef_model * scaler.mean_ / scaler.scale_)
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
            coef_original = float(coef_model[idx] * scale) if np.isfinite(scale) else None
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
                        float(
                            intercept_model
                            + np.dot(
                                coef_model,
                                [
                                    float(
                                        (scaler_stats or {}).get(f, {}).get("min", 0.0)
                                    )
                                    for f in feature_names
                                ],
                            )
                        )
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
    *,
    repeats: int,
    random_state: int,
) -> dict[str, dict[str, float]]:
    if repeats <= 0:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=repeats,
            random_state=random_state,
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
    settings: PipelineSettings,
    compute_attribution: bool,
) -> dict[str, Any]:
    cols = [
        feature
        for feature in selected_features
        if feature in train_df.columns and feature in test_df.columns
    ]
    if not cols:
        raise TechniqueRunError(
            f"No selected features present for eval_model={eval_model!r} on {target_col}"
        )
    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    y_test = pd.to_numeric(test_df[target_col], errors="coerce")
    train_mask = y_train.notna()
    test_mask = y_test.notna()
    y_train = y_train.loc[train_mask]
    y_test = y_test.loc[test_mask]
    if len(y_train) < 2 or len(y_test) < 1:
        raise TechniqueRunError(f"Insufficient labeled rows for {target_col}")
    x_train = train_df.loc[train_mask, cols].apply(pd.to_numeric, errors="coerce")
    x_test = test_df.loc[test_mask, cols].apply(pd.to_numeric, errors="coerce")
    names = test_df.loc[test_mask, NAME_COL].astype(str).tolist()

    model_kwargs: dict[str, Any] = {
        "random_state": settings.random_state,
        "n_jobs": 1,
        "n_samples_fit": len(y_train),
    }
    model_kwargs.update(eval_hp_by_model.get(eval_model, {}))
    model = make_regressor(eval_model, **model_kwargs)
    fit_regressor(model, x_train, y_train)
    yhat = np.asarray(model.predict(x_test), dtype=np.float64).ravel()
    metrics = metric_bundle(y_test, yhat)
    evaluation = {
        **{key: metrics[key] for key in ("spearman", "pearson_r", "r2", "mse")},
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
        attributions = extract_attributions(
            model,
            model_type=eval_model,
            feature_names=cols,
            scaling_method=scaling_method,
            scaler_stats=scaler_stats,
        )
        perm = _permutation_rows(
            model,
            x_test,
            y_test,
            cols,
            repeats=settings.permutation_repeats,
            random_state=settings.random_state,
        )
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


def build_feature_usage_rows(
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
                "selected": feature in selected,
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
            }
        )
    return rows


def choose_eval_model_over_folds(
    inner_dir: Path,
    *,
    target_col: str,
    candidate_features: list[str],
    technique: Technique,
    settings: PipelineSettings,
    eval_hp_by_model: dict[str, dict],
) -> tuple[str, float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every candidate eval model over a set of folds; return the best.

    Used both for the inner folds of nested CV and for the full-dataset refit,
    so the final model is chosen by exactly the rule cross-validation used.
    """
    meta = json.loads((inner_dir / "meta.json").read_text())
    n_inner = int(meta["n_splits"])
    eval_models = list(technique.eval_models)
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
            selection = select_features(
                train,
                val,
                target_col=target_col,
                candidate_features=candidate_features,
                technique=technique,
                settings=settings,
            )
            for feature in selection["selected_features"]:
                inner_feature_rows.append(
                    {"inner_fold": inner_k, "feature": feature, "stage": "selected"}
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
                    settings=settings,
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
                    }
                )

    ranked: list[tuple[str, float | None]] = []
    for eval_model in eval_models:
        ys, yhats = pooled_by_model[eval_model]
        pooled = pooled_spearman(
            pd.concat(ys, ignore_index=True), pd.concat(yhats, ignore_index=True)
        )
        ranked.append((eval_model, pooled))
        for row in inner_scores:
            if row["eval_model"] == eval_model:
                row["pooled_inner_spearman"] = pooled

    valid = [(model, score) for model, score in ranked if score is not None]
    if not valid:
        raise TechniqueRunError(f"No valid inner eval-model score for {target_col}")
    best_model, best_score = max(valid, key=lambda item: (item[1], item[0]))
    return best_model, best_score, inner_scores, inner_feature_rows


def _run_outer_selector(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    technique: Technique,
    settings: PipelineSettings,
    work_root: Path,
    eval_hp_by_model: dict[str, dict],
) -> dict[str, Any]:
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")
    inner_dir = work_root / target_col / f"outer_{outer_k}" / "inner_folds"
    write_inner_fold_dir(fold_dir, outer_k=outer_k, dest=inner_dir)
    best_eval, best_inner, inner_scores, inner_feature_rows = choose_eval_model_over_folds(
        inner_dir,
        target_col=target_col,
        candidate_features=features,
        technique=technique,
        settings=settings,
        eval_hp_by_model=eval_hp_by_model,
    )
    n_inner = int(json.loads((inner_dir / "meta.json").read_text())["n_splits"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        selection = select_features(
            outer_train,
            outer_test,
            target_col=target_col,
            candidate_features=features,
            technique=technique,
            settings=settings,
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
            settings=settings,
            compute_attribution=True,
        )

    return _selector_result(
        technique=technique,
        target_col=target_col,
        outer_k=outer_k,
        selection=selection,
        fit=fit,
        eval_model=best_eval,
        inner_pooled_spearman=best_inner,
        inner_scores=inner_scores,
        inner_feature_rows=inner_feature_rows,
        n_inner_folds=n_inner,
        cv_mode="nested",
        selection_rule="max pooled inner spearman",
    )


def _run_outer_selector_flat(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    technique: Technique,
    settings: PipelineSettings,
    eval_hp_by_model: dict[str, dict],
) -> dict[str, Any]:
    """Single-level CV: select on outer-train only; no inner eval-model search."""
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")
    eval_model = technique.selector_model or technique.eval_models[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        selection = select_features(
            outer_train,
            outer_test,
            target_col=target_col,
            candidate_features=features,
            technique=technique,
            settings=settings,
        )
        fit = _fit_predict_selector(
            selection["train_df"],
            selection["test_df"],
            target_col=target_col,
            selected_features=selection["selected_features"],
            eval_model=eval_model,
            eval_hp_by_model=eval_hp_by_model,
            scaling_method=selection["scaling_method"],
            scaler_stats=selection["scaler_stats"],
            settings=settings,
            compute_attribution=True,
        )
    return _selector_result(
        technique=technique,
        target_col=target_col,
        outer_k=outer_k,
        selection=selection,
        fit=fit,
        eval_model=eval_model,
        inner_pooled_spearman=None,
        inner_scores=[],
        inner_feature_rows=[],
        n_inner_folds=0,
        cv_mode="flat",
        selection_rule="flat CV train-only selection",
    )


def _selector_result(
    *,
    technique: Technique,
    target_col: str,
    outer_k: int,
    selection: dict[str, Any],
    fit: dict[str, Any],
    eval_model: str,
    inner_pooled_spearman: float | None,
    inner_scores: list[dict[str, Any]],
    inner_feature_rows: list[dict[str, Any]],
    n_inner_folds: int,
    cv_mode: str,
    selection_rule: str,
) -> dict[str, Any]:
    feature_usage = build_feature_usage_rows(
        input_features=selection["input_features"],
        after_lowvar=selection["after_lowvar"],
        after_intercorr=selection["after_intercorr"],
        selected_features=selection["selected_features"],
        final_features=fit["final_features"],
        attributions=fit["attributions"],
    )
    return {
        "technique": technique.key,
        "technique_label": technique.label,
        "cv_mode": cv_mode,
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
        "eval_model": eval_model,
        "alpha": None,
        "l1_ratio": None,
        "inner_pooled_spearman": inner_pooled_spearman,
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
        "n_inner_folds": n_inner_folds,
        "selection_rule": selection_rule,
        "oof_rows": [
            {
                "name": str(name),
                "target_col": target_col,
                "outer_fold": outer_k,
                "technique": technique.key,
                "eval_model": eval_model,
                "alpha": None,
                "l1_ratio": None,
                "y": float(y),
                "yhat": float(yhat),
            }
            for name, y, yhat in zip(fit["names"], fit["y"], fit["yhat"])
        ],
    }


def choose_elasticnet_grid_point_over_folds(
    inner_dir: Path,
    *,
    target_col: str,
    features: list[str],
    technique: Technique,
    settings: PipelineSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Pooled-Spearman ``(alpha, l1_ratio)`` search over a set of folds."""
    n_inner = int(json.loads((inner_dir / "meta.json").read_text())["n_splits"])
    grid_scores: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ConstantInputWarning)
        inner_frames = [
            (
                pd.read_parquet(inner_dir / f"fold_{inner_k}_train.parquet"),
                pd.read_parquet(inner_dir / f"fold_{inner_k}_test.parquet"),
            )
            for inner_k in range(n_inner)
        ]
        for alpha in technique.alphas:
            for l1_ratio in technique.l1_ratios:
                ys: list[pd.Series] = []
                yhats: list[pd.Series] = []
                fold_scores: list[float | None] = []
                fold_r2: list[float | None] = []
                for train, val in inner_frames:
                    x_train, y_train, _ = xy(train, target_col, features)
                    x_val, y_val, _ = xy(val, target_col, features)
                    model = make_elasticnet_pipeline(
                        alpha, l1_ratio, random_state=settings.random_state
                    )
                    model.fit(x_train, y_train)
                    pred = pd.Series(model.predict(x_val), index=y_val.index)
                    ys.append(y_val)
                    yhats.append(pred)
                    metrics = metric_bundle(y_val, pred)
                    fold_scores.append(metrics["spearman"])
                    fold_r2.append(metrics["r2"])
                pooled = metric_bundle(
                    pd.concat(ys, ignore_index=True),
                    pd.concat(yhats, ignore_index=True),
                )
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
        raise TechniqueRunError(
            f"No valid ElasticNet grid score for {target_col} under {inner_dir}"
        )
    best = max(
        valid,
        key=lambda row: (row["pooled_inner_spearman"], row["alpha"], row["l1_ratio"]),
    )
    return best, grid_scores, n_inner


def _run_outer_elasticnet(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    technique: Technique,
    settings: PipelineSettings,
    work_root: Path,
) -> dict[str, Any]:
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")
    inner_dir = work_root / target_col / f"outer_{outer_k}" / "inner_folds"
    write_inner_fold_dir(fold_dir, outer_k=outer_k, dest=inner_dir)
    best, grid_scores, n_inner = choose_elasticnet_grid_point_over_folds(
        inner_dir,
        target_col=target_col,
        features=features,
        technique=technique,
        settings=settings,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ConstantInputWarning)
        x_train, y_train, _ = xy(outer_train, target_col, features)
        x_test, y_test, names = xy(outer_test, target_col, features)
        model = make_elasticnet_pipeline(
            best["alpha"], best["l1_ratio"], random_state=settings.random_state
        )
        model.fit(x_train, y_train)
        pred = pd.Series(model.predict(x_test), index=y_test.index)

    inner_scores = [
        {
            "eval_model": "elasticnet",
            "inner_fold": inner_k,
            "alpha": row["alpha"],
            "l1_ratio": row["l1_ratio"],
            "spearman": row["inner_fold_spearman"][inner_k],
            "r2": row["inner_fold_r2"][inner_k],
            "pooled_inner_spearman": row["pooled_inner_spearman"],
            "is_best": (
                row["alpha"] == best["alpha"] and row["l1_ratio"] == best["l1_ratio"]
            ),
        }
        for row in grid_scores
        for inner_k in range(n_inner)
    ]
    return _elasticnet_result(
        technique=technique,
        target_col=target_col,
        outer_k=outer_k,
        features=features,
        model=model,
        x_test=x_test,
        y_test=y_test,
        pred=pred,
        names=names,
        n_train=int(len(y_train)),
        alpha=best["alpha"],
        l1_ratio=best["l1_ratio"],
        inner_pooled_spearman=best["pooled_inner_spearman"],
        inner_scores=inner_scores,
        grid_scores=grid_scores,
        n_inner_folds=n_inner,
        cv_mode="nested",
        selection_rule="max pooled inner spearman",
        settings=settings,
    )


def _run_outer_elasticnet_flat(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    technique: Technique,
    settings: PipelineSettings,
) -> dict[str, Any]:
    """Single-level CV: ElasticNetCV on outer-train only (no leftover inner folds)."""
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")
    x_train, y_train, _ = xy(outer_train, target_col, features)
    x_test, y_test, names = xy(outer_test, target_col, features)
    n_train = int(len(y_train))
    cv = min(5, max(2, n_train // 2)) if n_train >= 4 else 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ConstantInputWarning)
        model = make_elasticnet_cv_pipeline(
            technique.alphas,
            technique.l1_ratios,
            cv=cv,
            random_state=settings.random_state,
        )
        model.fit(x_train, y_train)
        pred = pd.Series(model.predict(x_test), index=y_test.index)
    fitted = model.named_steps["elasticnet"]
    return _elasticnet_result(
        technique=technique,
        target_col=target_col,
        outer_k=outer_k,
        features=features,
        model=model,
        x_test=x_test,
        y_test=y_test,
        pred=pred,
        names=names,
        n_train=n_train,
        alpha=float(fitted.alpha_),
        l1_ratio=float(fitted.l1_ratio_),
        inner_pooled_spearman=None,
        inner_scores=[],
        grid_scores=[],
        n_inner_folds=cv,
        cv_mode="flat",
        selection_rule="flat ElasticNetCV on outer-train",
        settings=settings,
    )


def _elasticnet_result(
    *,
    technique: Technique,
    target_col: str,
    outer_k: int,
    features: list[str],
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    pred: pd.Series,
    names: pd.Series,
    n_train: int,
    alpha: float,
    l1_ratio: float,
    inner_pooled_spearman: float | None,
    inner_scores: list[dict[str, Any]],
    grid_scores: list[dict[str, Any]],
    n_inner_folds: int,
    cv_mode: str,
    selection_rule: str,
    settings: PipelineSettings,
) -> dict[str, Any]:
    metrics = metric_bundle(y_test, pred)
    attributions = extract_attributions(
        model.named_steps["elasticnet"],
        model_type="elasticnet",
        feature_names=features,
        scaling_method="standardize",
        scaler_stats=None,
        pipeline_model=model,
    )
    perm = _permutation_rows(
        model,
        x_test,
        y_test,
        features,
        repeats=settings.permutation_repeats,
        random_state=settings.random_state,
    )
    for row in attributions:
        row.update(perm.get(row["feature"], {}))
    feature_usage = build_feature_usage_rows(
        input_features=features,
        after_lowvar=features,
        after_intercorr=features,
        selected_features=features,
        final_features=features,
        attributions=attributions,
    )
    nonzero_features = [row["feature"] for row in attributions if row.get("is_nonzero")]
    return {
        "technique": technique.key,
        "technique_label": technique.label,
        "cv_mode": cv_mode,
        "target_col": target_col,
        "outer_fold": outer_k,
        "n_test": int(metrics["n"]),
        "n_train": n_train,
        "spearman": metrics["spearman"],
        "spearman_p": metrics["spearman_p"],
        "pearson_r": metrics["pearson_r"],
        "pearson_p": metrics["pearson_p"],
        "r2": metrics["r2"],
        "mse": metrics["mse"],
        "eval_model": "elasticnet",
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "inner_pooled_spearman": inner_pooled_spearman,
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
        "n_inner_folds": n_inner_folds,
        "selection_rule": selection_rule,
        "oof_rows": [
            {
                "name": str(name),
                "target_col": target_col,
                "outer_fold": outer_k,
                "technique": technique.key,
                "eval_model": "elasticnet",
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "y": float(y),
                "yhat": float(yhat),
            }
            for name, y, yhat in zip(names, y_test, pred)
        ],
    }


def run_outer_fold(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    technique: Technique,
    settings: PipelineSettings,
    work_root: Path,
    eval_hp_by_model: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Evaluate one technique on one outer fold."""
    hp = eval_hp_by_model or eval_hyperparameters_by_model(settings)
    flat = settings.cv.mode == "flat"
    if technique.kind == "elasticnet":
        if flat:
            return _run_outer_elasticnet_flat(
                fold_dir,
                target_col=target_col,
                outer_k=outer_k,
                features=features,
                technique=technique,
                settings=settings,
            )
        return _run_outer_elasticnet(
            fold_dir,
            target_col=target_col,
            outer_k=outer_k,
            features=features,
            technique=technique,
            settings=settings,
            work_root=work_root,
        )
    if flat:
        return _run_outer_selector_flat(
            fold_dir,
            target_col=target_col,
            outer_k=outer_k,
            features=features,
            technique=technique,
            settings=settings,
            eval_hp_by_model=hp,
        )
    return _run_outer_selector(
        fold_dir,
        target_col=target_col,
        outer_k=outer_k,
        features=features,
        technique=technique,
        settings=settings,
        work_root=work_root,
        eval_hp_by_model=hp,
    )
