"""Nested technique-selection aggregation (no model fitting)."""

from __future__ import annotations

import pytest

from automl.select_best import (
    MISSING_INNER_ERROR,
    inner_selection_available,
    nested_hyperparameters_for_technique,
    pick_fold_winner,
    resolve_target_selection,
    score_techniques,
    select_best,
)
from automl.techniques import TechniqueConfigError, pipeline_settings_from_block


def _fold(
    technique: str,
    outer_fold: int,
    *,
    inner: float | None,
    y: list[float],
    yhat: list[float],
    eval_model: str = "svm",
    alpha: float | None = None,
    l1_ratio: float | None = None,
) -> dict:
    return {
        "technique": technique,
        "technique_label": technique,
        "target_col": "target_x",
        "cv_mode": "nested",
        "outer_fold": outer_fold,
        "inner_pooled_spearman": inner,
        "spearman": None,
        "oof_rows": [
            {
                "name": f"{technique}_{outer_fold}_{i}",
                "y": float(yi),
                "yhat": float(yh),
            }
            for i, (yi, yh) in enumerate(zip(y, yhat))
        ],
        "eval_model": eval_model,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "n_test": len(y),
        "n_selected_features": 2,
        "selected_features": ["f1", "f2"],
    }


def _two_fold_results() -> list[dict]:
    y = [1.0, 2.0, 3.0, 4.0]
    return [
        _fold("elasticnet", 0, inner=0.80, y=y, yhat=y),
        _fold("sfs_svm", 0, inner=0.95, y=y, yhat=[4.0, 3.0, 1.0, 2.0]),
        _fold("elasticnet", 1, inner=0.90, y=y, yhat=y),
        _fold("sfs_svm", 1, inner=0.10, y=y, yhat=list(reversed(y))),
    ]


def test_inner_procedure_is_not_max_of_outer_spearman():
    results = _two_fold_results()
    competing = score_techniques(results, technique_order=["elasticnet", "sfs_svm"])
    outer_best = select_best(competing)
    inner = resolve_target_selection(
        results,
        technique_order=["elasticnet", "sfs_svm"],
    )
    assert outer_best.technique == "elasticnet"
    assert inner.technique_selection == "inner"
    assert [row["technique"] for row in inner.fold_winners] == ["sfs_svm", "elasticnet"]
    # mean inner: elasticnet 0.85, sfs_svm 0.525
    assert inner.deployed.technique == "elasticnet"
    assert inner.deployed.n_folds_won == 1
    assert inner.procedure.spearman_pooled_oof is not None
    assert outer_best.spearman_pooled_oof is not None
    assert inner.procedure.spearman_pooled_oof < outer_best.spearman_pooled_oof


def test_missing_inner_scores_do_not_fall_back_to_outer():
    results = _two_fold_results()
    for row in results:
        row["inner_pooled_spearman"] = None
        row["cv_mode"] = "flat"
    assert inner_selection_available(results) is False
    with pytest.raises(ValueError, match="inner_pooled_spearman"):
        resolve_target_selection(
            results,
            technique_order=["elasticnet", "sfs_svm"],
        )


def test_single_technique_flat_does_not_select():
    y = [1.0, 2.0, 3.0, 4.0]
    results = [
        _fold("elasticnet", 0, inner=None, y=y, yhat=y),
        _fold("elasticnet", 1, inner=None, y=y, yhat=y),
    ]
    selection = resolve_target_selection(results, technique_order=["elasticnet"])
    assert selection.deployed.technique == "elasticnet"
    assert selection.fold_winners == []
    assert "single technique" in selection.selection_rule


def test_pick_fold_winner_tie_breaks_on_technique_name():
    y = [1.0, 2.0, 3.0]
    fold = [
        _fold("elasticnet", 0, inner=0.5, y=y, yhat=y),
        _fold("sfs_svm", 0, inner=0.5, y=y, yhat=y),
    ]
    assert pick_fold_winner(fold)["technique"] == "sfs_svm"


def test_deployed_technique_uses_mean_inner_not_majority():
    y = [1.0, 2.0, 3.0, 4.0]
    results = [
        _fold("elasticnet", 0, inner=0.51, y=y, yhat=y),
        _fold("sfs_svm", 0, inner=0.50, y=y, yhat=y),
        _fold("elasticnet", 1, inner=0.51, y=y, yhat=y),
        _fold("sfs_svm", 1, inner=0.50, y=y, yhat=y),
        _fold("elasticnet", 2, inner=0.10, y=y, yhat=y),
        _fold("sfs_svm", 2, inner=0.90, y=y, yhat=y),
    ]
    selection = resolve_target_selection(
        results,
        technique_order=["elasticnet", "sfs_svm"],
    )
    # majority would be elasticnet (2 folds); mean inner prefers sfs_svm
    assert selection.deployed.technique == "sfs_svm"
    assert selection.deployed.n_folds_won == 1
    assert [s.n_folds_won for s in selection.competing if s.technique == "elasticnet"] == [2]


def test_nested_hyperparameters_use_mode_of_inner_choices():
    y = [1.0, 2.0, 3.0]
    results = [
        _fold("elasticnet", 0, inner=0.8, y=y, yhat=y, alpha=0.01, l1_ratio=1.0),
        _fold("elasticnet", 1, inner=0.7, y=y, yhat=y, alpha=0.01, l1_ratio=1.0),
        _fold("elasticnet", 2, inner=0.2, y=y, yhat=y, alpha=0.3, l1_ratio=0.5),
    ]
    hp = nested_hyperparameters_for_technique(results, "elasticnet")
    assert hp["alpha"] == 0.01
    assert hp["l1_ratio"] == 1.0


def test_pipeline_settings_reject_outer_technique_selection():
    default = pipeline_settings_from_block({"cv": {"mode": "nested"}})
    assert default.cv.technique_selection == "inner"
    with pytest.raises(TechniqueConfigError, match="technique_selection"):
        pipeline_settings_from_block(
            {"cv": {"mode": "nested", "technique_selection": "outer"}}
        )


def test_missing_inner_error_mentions_flat_cv():
    assert "Flat CV" in MISSING_INNER_ERROR


def test_technique_selection_ignores_outer_test_scores():
    results = _two_fold_results()
    baseline = resolve_target_selection(
        results, technique_order=["elasticnet", "sfs_svm"]
    )
    for row in results:
        row["spearman"] = 0.99 if row["technique"] == "sfs_svm" else -0.99
        for oof in row["oof_rows"]:
            oof["yhat"] = -oof["y"] if row["technique"] == "elasticnet" else oof["y"]
    mutated = resolve_target_selection(
        results, technique_order=["elasticnet", "sfs_svm"]
    )
    assert [row["technique"] for row in mutated.fold_winners] == [
        row["technique"] for row in baseline.fold_winners
    ]
    assert mutated.deployed.technique == baseline.deployed.technique
    assert pick_fold_winner(
        [r for r in results if r["outer_fold"] == 0]
    )["technique"] == "sfs_svm"
