"""Score the four techniques and pick the winner.

Selection is nested CV only. Each outer fold keeps the technique with the
highest *inner* pooled Spearman. Those fold-wise out-of-fold predictions are
pooled into the nested-procedure score.

The deployed technique is the one with the highest *mean* inner Spearman
across outer-train splits (ties: technique name) — the same inner-CV
criterion, never outer-test. Hyperparameters for the final refit are the
mode of that technique's per-fold inner choices.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any, Iterable

import pandas as pd

from automl.cv_engine import json_float, metric_bundle

INNER_SELECTION_RULE = (
    "nested procedure: each outer fold uses the technique with the highest "
    "inner pooled Spearman; deployed technique is the highest mean inner "
    "Spearman across those outer-train splits"
)
SINGLE_TECHNIQUE_RULE = "single technique; no selection among techniques"
MISSING_INNER_ERROR = (
    "Nested technique selection needs inner_pooled_spearman on every outer fold. "
    "Flat CV does not compute inner scores, so it cannot choose among techniques "
    "without using outer-test predictions. Use nested CV (the default), or pass "
    "a single --techniques value."
)


@dataclass
class TechniqueScore:
    technique: str
    technique_label: str
    target_col: str
    cv_mode: str
    n_outer_folds: int
    n_oof: int
    spearman_pooled_oof: float | None
    pearson_pooled_oof: float | None
    r2_pooled_oof: float | None
    mse_pooled_oof: float | None
    mean_fold_spearman: float | None
    mean_inner_spearman: float | None
    eval_model: str | None
    alpha: float | None
    l1_ratio: float | None
    mean_n_selected_features: float | None
    feature_jaccard: float | None
    n_folds_won: int = 0
    is_best: bool = False

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetSelection:
    """Per-target nested technique decision."""

    competing: list[TechniqueScore]
    deployed: TechniqueScore
    procedure: TechniqueScore
    fold_winners: list[dict[str, Any]]
    technique_selection: str
    selection_rule: str

    def as_winner_row(self) -> dict[str, Any]:
        row = self.procedure.as_row()
        row.update(
            {
                "technique": self.deployed.technique,
                "technique_label": self.deployed.technique_label,
                "eval_model": self.deployed.eval_model,
                "alpha": self.deployed.alpha,
                "l1_ratio": self.deployed.l1_ratio,
                "mean_n_selected_features": self.deployed.mean_n_selected_features,
                "feature_jaccard": self.deployed.feature_jaccard,
                "n_folds_won": self.deployed.n_folds_won,
                "is_best": True,
                "technique_selection": self.technique_selection,
                "selection_rule": self.selection_rule,
                "deployed_outer_spearman": self.deployed.spearman_pooled_oof,
                "fold_winner_techniques": ",".join(
                    str(row["technique"]) for row in self.fold_winners
                ),
            }
        )
        return row


def _mode_or_none(values: Iterable[Any]) -> Any:
    series = pd.Series([v for v in values if v is not None])
    if series.empty:
        return None
    mode = series.mode()
    return mode.iloc[0] if len(mode) else None


def _try_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return json_float(float(value))
    except (TypeError, ValueError):
        return None


def mean_pairwise_jaccard(feature_sets: list[list[str]]) -> float | None:
    """Average Jaccard similarity between the selected feature sets of each fold."""
    sets = [set(fs) for fs in feature_sets if fs]
    if len(sets) < 2:
        return None
    scores: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        if not union:
            continue
        scores.append(len(left & right) / len(union))
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def score_technique(results: list[dict[str, Any]]) -> TechniqueScore:
    """Pool the outer folds of one technique on one target into a single score."""
    if not results:
        raise ValueError("score_technique requires at least one outer-fold result")
    ordered = sorted(results, key=lambda r: int(r["outer_fold"]))
    oof = pd.DataFrame(
        [row for result in ordered for row in result.get("oof_rows", [])]
    )
    if oof.empty:
        raise ValueError(
            f"No out-of-fold predictions for technique {ordered[0].get('technique')!r}"
        )
    pooled = metric_bundle(oof["y"], oof["yhat"])
    fold_spearman = pd.to_numeric(
        pd.Series([r.get("spearman") for r in ordered]), errors="coerce"
    )
    inner_spearman = pd.to_numeric(
        pd.Series([r.get("inner_pooled_spearman") for r in ordered]), errors="coerce"
    )
    n_selected = pd.to_numeric(
        pd.Series([r.get("n_selected_features") for r in ordered]), errors="coerce"
    )
    return TechniqueScore(
        technique=str(ordered[0]["technique"]),
        technique_label=str(ordered[0].get("technique_label") or ordered[0]["technique"]),
        target_col=str(ordered[0]["target_col"]),
        cv_mode=str(ordered[0].get("cv_mode") or "nested"),
        n_outer_folds=len(ordered),
        n_oof=int(pooled["n"]),
        spearman_pooled_oof=pooled["spearman"],
        pearson_pooled_oof=pooled["pearson_r"],
        r2_pooled_oof=pooled["r2"],
        mse_pooled_oof=pooled["mse"],
        mean_fold_spearman=(
            float(fold_spearman.mean()) if fold_spearman.notna().any() else None
        ),
        mean_inner_spearman=(
            float(inner_spearman.mean()) if inner_spearman.notna().any() else None
        ),
        eval_model=_mode_or_none(r.get("eval_model") for r in ordered),
        alpha=_mode_or_none(r.get("alpha") for r in ordered),
        l1_ratio=_mode_or_none(r.get("l1_ratio") for r in ordered),
        mean_n_selected_features=(
            float(n_selected.mean()) if n_selected.notna().any() else None
        ),
        feature_jaccard=mean_pairwise_jaccard(
            [list(r.get("selected_features") or []) for r in ordered]
        ),
    )


def score_techniques(
    results: list[dict[str, Any]], *, technique_order: list[str] | None = None
) -> list[TechniqueScore]:
    """One score per technique, in ``technique_order`` when given."""
    by_technique: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_technique.setdefault(str(result["technique"]), []).append(result)
    keys = list(technique_order or sorted(by_technique))
    keys.extend(key for key in sorted(by_technique) if key not in keys)
    return [score_technique(by_technique[key]) for key in keys if key in by_technique]


def select_best(scores: list[TechniqueScore]) -> TechniqueScore:
    """Highest pooled out-of-fold Spearman; ties broken by technique name.

    Diagnostic only — not used to choose the deployed technique.
    """
    ranked = [s for s in scores if s.spearman_pooled_oof is not None]
    if not ranked:
        raise ValueError(
            "No technique produced a usable pooled Spearman correlation; "
            "check that the target has enough labelled rows"
        )
    return max(ranked, key=lambda s: (s.spearman_pooled_oof, s.technique))


def inner_selection_available(results: list[dict[str, Any]]) -> bool:
    """True when every outer fold has at least one finite inner pooled Spearman."""
    by_fold: dict[int, list[float]] = defaultdict(list)
    for result in results:
        fold = int(result["outer_fold"])
        score = _try_finite(result.get("inner_pooled_spearman"))
        if score is not None:
            by_fold[fold].append(score)
    folds = {int(result["outer_fold"]) for result in results}
    return bool(folds) and all(by_fold.get(fold) for fold in folds)


def pick_fold_winner(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest inner pooled Spearman on one outer fold; ties broken by name."""
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for result in fold_results:
        score = _try_finite(result.get("inner_pooled_spearman"))
        if score is None:
            continue
        ranked.append((score, str(result["technique"]), result))
    if not ranked:
        raise ValueError(
            "pick_fold_winner requires at least one finite inner_pooled_spearman"
        )
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def select_deployed_technique(
    fold_winners: list[dict[str, Any]],
    competing: list[TechniqueScore],
) -> TechniqueScore:
    """Highest mean inner pooled Spearman; ties broken by technique name.

    Outer-test predictions are not used. This is the full-data analogue of
    :func:`pick_fold_winner`.
    """
    if not fold_winners:
        raise ValueError("select_deployed_technique requires fold winners")
    if not competing:
        raise ValueError("select_deployed_technique requires competing scores")
    ranked = [
        score
        for score in competing
        if score.mean_inner_spearman is not None
    ]
    if not ranked:
        raise ValueError(
            "select_deployed_technique requires finite mean inner Spearman"
        )
    return max(ranked, key=lambda s: (s.mean_inner_spearman, s.technique))


def _pool_winner_oof(winners: list[dict[str, Any]]) -> dict[str, Any]:
    oof = pd.DataFrame(
        [row for result in winners for row in result.get("oof_rows", [])]
    )
    if oof.empty:
        raise ValueError("No out-of-fold predictions for the nested procedure")
    return metric_bundle(oof["y"], oof["yhat"])


def _fold_winner_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "outer_fold": int(result["outer_fold"]),
        "technique": str(result["technique"]),
        "technique_label": str(
            result.get("technique_label") or result["technique"]
        ),
        "inner_pooled_spearman": _try_finite(result.get("inner_pooled_spearman")),
        "outer_spearman": _try_finite(result.get("spearman")),
        "eval_model": result.get("eval_model"),
        "alpha": result.get("alpha"),
        "l1_ratio": result.get("l1_ratio"),
        "n_test": result.get("n_test"),
        "n_selected_features": result.get("n_selected_features"),
    }


def _annotate_wins(
    competing: list[TechniqueScore], fold_winners: list[dict[str, Any]], deployed: str
) -> list[TechniqueScore]:
    wins = Counter(str(row["technique"]) for row in fold_winners)
    return [
        replace(
            score,
            n_folds_won=int(wins.get(score.technique, 0)),
            is_best=score.technique == deployed,
        )
        for score in competing
    ]


def _procedure_score_from_winners(
    winners: list[dict[str, Any]],
    *,
    deployed: TechniqueScore,
    pooled: dict[str, Any],
) -> TechniqueScore:
    unique = list(dict.fromkeys(str(row["technique"]) for row in winners))
    fold_spearman = pd.to_numeric(
        pd.Series([row.get("spearman") for row in winners]), errors="coerce"
    )
    inner_spearman = pd.to_numeric(
        pd.Series([row.get("inner_pooled_spearman") for row in winners]),
        errors="coerce",
    )
    n_selected = pd.to_numeric(
        pd.Series([row.get("n_selected_features") for row in winners]),
        errors="coerce",
    )
    mixed = len(unique) != 1
    return TechniqueScore(
        technique=deployed.technique,
        technique_label=deployed.technique_label,
        target_col=deployed.target_col,
        cv_mode=deployed.cv_mode,
        n_outer_folds=len(winners),
        n_oof=int(pooled["n"]),
        spearman_pooled_oof=pooled["spearman"],
        pearson_pooled_oof=pooled["pearson_r"],
        r2_pooled_oof=pooled["r2"],
        mse_pooled_oof=pooled["mse"],
        mean_fold_spearman=(
            float(fold_spearman.mean()) if fold_spearman.notna().any() else None
        ),
        mean_inner_spearman=(
            float(inner_spearman.mean()) if inner_spearman.notna().any() else None
        ),
        eval_model=deployed.eval_model if not mixed else None,
        alpha=deployed.alpha if not mixed else None,
        l1_ratio=deployed.l1_ratio if not mixed else None,
        mean_n_selected_features=(
            float(n_selected.mean()) if n_selected.notna().any() else None
        ),
        feature_jaccard=mean_pairwise_jaccard(
            [list(row.get("selected_features") or []) for row in winners]
        ),
        n_folds_won=deployed.n_folds_won,
        is_best=True,
    )


def nested_hyperparameters_for_technique(
    results: list[dict[str, Any]], technique: str
) -> dict[str, Any]:
    """Inner-CV hyperparameters for ``technique`` (mode across outer folds).

    Each fold's ``eval_model`` / ``(alpha, l1_ratio)`` was chosen on that
    fold's outer-train only. Ties in the mode fall back to the fold with the
    highest inner Spearman.
    """
    subset = [row for row in results if str(row["technique"]) == technique]
    if not subset:
        raise ValueError(f"No nested-fold results for technique {technique!r}")

    def _best_inner(row: dict[str, Any]) -> tuple[float, str]:
        score = _try_finite(row.get("inner_pooled_spearman"))
        return (score if score is not None else float("-inf"), str(row.get("eval_model") or ""))

    fallback = max(subset, key=_best_inner)
    inners = [
        v
        for v in (_try_finite(row.get("inner_pooled_spearman")) for row in subset)
        if v is not None
    ]
    return {
        "eval_model": _mode_or_none(row.get("eval_model") for row in subset)
        or fallback.get("eval_model"),
        "alpha": _mode_or_none(row.get("alpha") for row in subset)
        if any(row.get("alpha") is not None for row in subset)
        else fallback.get("alpha"),
        "l1_ratio": _mode_or_none(row.get("l1_ratio") for row in subset)
        if any(row.get("l1_ratio") is not None for row in subset)
        else fallback.get("l1_ratio"),
        "selection_spearman": (float(sum(inners) / len(inners)) if inners else None),
    }


def resolve_target_selection(
    results: list[dict[str, Any]],
    *,
    technique_order: list[str] | None = None,
) -> TargetSelection:
    """Pick the nested-procedure score and the technique to refit.

    Technique × outer-fold results are assumed to already exist (computed in
    parallel). This function does not run models. Outer-test Spearman is never
    used to choose a technique.
    """
    if not results:
        raise ValueError("resolve_target_selection requires at least one fold result")
    competing = score_techniques(results, technique_order=technique_order)
    n_techniques = len({str(row["technique"]) for row in results})

    if n_techniques == 1 and not inner_selection_available(results):
        deployed = replace(competing[0], is_best=True, n_folds_won=0)
        return TargetSelection(
            competing=[deployed],
            deployed=deployed,
            procedure=replace(deployed, is_best=True),
            fold_winners=[],
            technique_selection="inner",
            selection_rule=SINGLE_TECHNIQUE_RULE,
        )

    if not inner_selection_available(results):
        raise ValueError(MISSING_INNER_ERROR)

    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_fold[int(result["outer_fold"])].append(result)
    winners = [pick_fold_winner(by_fold[fold]) for fold in sorted(by_fold)]
    fold_winner_rows = [_fold_winner_row(row) for row in winners]
    if n_techniques == 1:
        deployed = competing[0]
        selection_rule = SINGLE_TECHNIQUE_RULE
    else:
        deployed = select_deployed_technique(winners, competing)
        selection_rule = INNER_SELECTION_RULE
    competing = _annotate_wins(competing, fold_winner_rows, deployed.technique)
    deployed = next(score for score in competing if score.technique == deployed.technique)
    pooled = _pool_winner_oof(winners)
    procedure = _procedure_score_from_winners(
        winners, deployed=deployed, pooled=pooled
    )
    return TargetSelection(
        competing=competing,
        deployed=deployed,
        procedure=procedure,
        fold_winners=fold_winner_rows,
        technique_selection="inner",
        selection_rule=selection_rule,
    )
