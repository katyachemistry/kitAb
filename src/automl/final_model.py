"""Refit the winning technique on the whole dataset and persist it.

The estimator is an sklearn ``Pipeline`` that owns its scaler, so callers
predict directly on raw descriptor values in ``feature_cols`` order.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ConstantInputWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from automl.cv_engine import (
    TechniqueRunError,
    choose_elasticnet_grid_point_over_folds,
    choose_eval_model_over_folds,
    eval_hyperparameters_by_model,
    extract_attributions,
    make_elasticnet_pipeline,
    make_selector_pipeline,
    select_features,
    xy,
)
from automl.folds import full_dataset_frame
from automl.model_io import save_model
from automl.select_best import TechniqueScore
from automl.techniques import PipelineSettings, Technique

SELECTION_RULE = (
    "Refit on all labelled rows. The technique is the highest mean inner "
    "Spearman from nested CV. ElasticNet (alpha, l1_ratio) and the SFS eval "
    "model are the mode of that technique's per-fold inner-CV choices."
)
NESTED_PROCEDURE_METRIC_NOTE = (
    "cv_* metrics are the nested AutoML procedure: each outer fold uses the "
    "technique with the best inner-CV Spearman. They are not the CV score of "
    "the finally deployed technique alone, and they are not computed on the "
    "rows this model was finally fitted on."
)
HP_SELECTION_RULE = (
    "Hyperparameters come from nested inner CV (mode across outer-train "
    "splits of the deployed technique). They are not re-tuned on outer-test."
)


@dataclass
class FinalModel:
    estimator: Pipeline
    technique: Technique
    target_col: str
    feature_cols: list[str]
    eval_model: str
    alpha: float | None
    l1_ratio: float | None
    n_train: int
    selection_spearman: float | None
    attributions: list[dict[str, Any]]


def fit_final_model(
    fold_dir: Path,
    *,
    target_col: str,
    features: list[str],
    technique: Technique,
    settings: PipelineSettings,
    nested_hp: dict[str, Any] | None = None,
) -> FinalModel:
    """Fit the technique on every labelled row under ``fold_dir``.

    ``nested_hp`` supplies inner-CV ``eval_model`` / ``alpha`` / ``l1_ratio``
    from the nested run so the final fit does not re-tune on outer-test.
    """
    full = full_dataset_frame(fold_dir)
    eval_hp_by_model = eval_hyperparameters_by_model(settings)
    hp = dict(nested_hp or {})

    if technique.kind == "elasticnet":
        alpha = hp.get("alpha")
        l1_ratio = hp.get("l1_ratio")
        if alpha is None or l1_ratio is None:
            best, _grid, _n_folds = choose_elasticnet_grid_point_over_folds(
                fold_dir,
                target_col=target_col,
                features=features,
                technique=technique,
                settings=settings,
            )
            alpha = best["alpha"]
            l1_ratio = best["l1_ratio"]
            selection_spearman = best["pooled_inner_spearman"]
        else:
            alpha = float(alpha)
            l1_ratio = float(l1_ratio)
            selection_spearman = hp.get("selection_spearman")
        x_all, y_all, _ = xy(full, target_col, features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator = make_elasticnet_pipeline(
                alpha, l1_ratio, random_state=settings.random_state
            )
            estimator.fit(x_all, y_all)
        attributions = extract_attributions(
            estimator.named_steps["elasticnet"],
            model_type="elasticnet",
            feature_names=list(features),
            scaling_method="standardize",
            scaler_stats=None,
            pipeline_model=estimator,
        )
        return FinalModel(
            estimator=estimator,
            technique=technique,
            target_col=target_col,
            feature_cols=list(features),
            eval_model="elasticnet",
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            n_train=int(len(y_all)),
            selection_spearman=selection_spearman,
            attributions=attributions,
        )

    if technique.searches_eval_model:
        eval_model = hp.get("eval_model")
        selection_spearman = hp.get("selection_spearman")
        if eval_model is None:
            eval_model, selection_spearman, _scores, _rows = choose_eval_model_over_folds(
                fold_dir,
                target_col=target_col,
                candidate_features=features,
                technique=technique,
                settings=settings,
                eval_hp_by_model=eval_hp_by_model,
            )
        else:
            eval_model = str(eval_model)
    else:
        eval_model, selection_spearman = technique.eval_models[0], hp.get(
            "selection_spearman"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        selection = select_features(
            full,
            full,
            target_col=target_col,
            candidate_features=features,
            technique=technique,
            settings=settings,
        )
    selected = [c for c in selection["selected_features"] if c in full.columns]
    if not selected:
        raise TechniqueRunError(
            f"Final refit for {target_col} selected no features "
            f"(technique={technique.key})"
        )

    y_all = pd.to_numeric(full[target_col], errors="coerce")
    labelled = y_all.notna()
    x_all = full.loc[labelled, selected].apply(pd.to_numeric, errors="coerce")
    y_all = y_all.loc[labelled]
    estimator = make_selector_pipeline(
        eval_model,
        eval_hp_by_model=eval_hp_by_model,
        random_state=settings.random_state,
        n_samples_fit=int(len(y_all)),
    )
    estimator.fit(x_all, y_all)
    attributions = extract_attributions(
        estimator.named_steps["model"],
        model_type=eval_model,
        feature_names=selected,
        scaling_method="minmax_train_fit",
        scaler_stats={
            feature: selection["scaler_stats"][feature]
            for feature in selected
            if feature in selection["scaler_stats"]
        },
    )
    return FinalModel(
        estimator=estimator,
        technique=technique,
        target_col=target_col,
        feature_cols=selected,
        eval_model=eval_model,
        alpha=None,
        l1_ratio=None,
        n_train=int(len(y_all)),
        selection_spearman=selection_spearman,
        attributions=attributions,
    )


def build_meta(
    final: FinalModel,
    *,
    score: TechniqueScore,
    settings: PipelineSettings,
    dataset_stem: str,
    dataset_yaml_key: str,
    descriptor_source: str,
    competing_scores: list[TechniqueScore],
    fold_winners: list[dict[str, Any]] | None = None,
    technique_selection: str = "inner",
    selection_rule: str | None = None,
    deployed_outer_spearman: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_stem": dataset_stem,
        "dataset_yaml_key": dataset_yaml_key,
        "descriptor_source": descriptor_source,
        "target_col": final.target_col,
        "technique": final.technique.key,
        "technique_label": final.technique.label,
        "eval_model": final.eval_model,
        "hyperparameters": (
            {"alpha": final.alpha, "l1_ratio": final.l1_ratio}
            if final.alpha is not None
            else dict(settings.eval_hyperparameters.get(final.eval_model, {}))
        ),
        "feature_cols": list(final.feature_cols),
        "n_features": len(final.feature_cols),
        "training_row_count": final.n_train,
        "random_state": settings.random_state,
        "cv_mode": score.cv_mode,
        "technique_selection": technique_selection,
        "n_outer_folds": score.n_outer_folds,
        "n_folds_won": score.n_folds_won,
        "cv_spearman_pooled_oof": score.spearman_pooled_oof,
        "cv_pearson_pooled_oof": score.pearson_pooled_oof,
        "cv_r2_pooled_oof": score.r2_pooled_oof,
        "cv_feature_jaccard": score.feature_jaccard,
        "deployed_outer_spearman": deployed_outer_spearman,
        "fold_winners": list(fold_winners or []),
        "cv_metric_note": NESTED_PROCEDURE_METRIC_NOTE,
        "final_selection_spearman": final.selection_spearman,
        "selection_rule": selection_rule or SELECTION_RULE,
        "hp_selection_rule": HP_SELECTION_RULE,
        "technique_comparison": [s.as_row() for s in competing_scores],
        "scaling": "included_in_estimator_pipeline",
        "inference_note": (
            "Predict on raw (unscaled) descriptor values in feature_cols order; "
            "the estimator pipeline applies its own scaler. Load with "
            "kitab.models.predict_with_model(model_dir, features)."
        ),
    }


def save_final_model(
    model_dir: Path, final: FinalModel, meta: dict[str, Any]
) -> Path:
    return save_model(model_dir, estimator=final.estimator, meta=meta)
