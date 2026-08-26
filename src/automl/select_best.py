"""Score the four techniques against each other and pick the winner.

Techniques are ranked by Spearman correlation over the pooled out-of-fold
predictions, which is the same criterion used in the kitAb paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable

import pandas as pd

from automl.cv_engine import metric_bundle


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
    eval_model: str | None
    alpha: float | None
    l1_ratio: float | None
    mean_n_selected_features: float | None
    feature_jaccard: float | None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _mode_or_none(values: Iterable[Any]) -> Any:
    series = pd.Series([v for v in values if v is not None])
    if series.empty:
        return None
    mode = series.mode()
    return mode.iloc[0] if len(mode) else None


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
    """Highest pooled out-of-fold Spearman; ties broken by technique name."""
    ranked = [s for s in scores if s.spearman_pooled_oof is not None]
    if not ranked:
        raise ValueError(
            "No technique produced a usable pooled Spearman correlation; "
            "check that the target has enough labelled rows"
        )
    return max(ranked, key=lambda s: (s.spearman_pooled_oof, s.technique))
