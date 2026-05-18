"""Utilities for summarizing feature-selection outputs across a batch/grid.

Each worker JSON from ``run_fold_pipeline_config`` corresponds to one outer fold and
one grid cell (selector × selection model × eval feature cap × hyperparameters). These
helpers aggregate how often each feature name appears across those cells, keyed by
``fold_index``. Nothing here is invoked by the batch driver yet — import when needed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal


def _feature_names_from_fold_result(
    data: dict,
    *,
    feature_source: Literal["selected_features", "eval_reported"],
) -> list[str] | None:
    """Feature list for one result payload; ``None`` means this file contributes no votes."""
    if feature_source == "eval_reported":
        evaluation = data.get("evaluation") or {}
        for model_result in evaluation.values():
            if not isinstance(model_result, dict) or model_result.get("error"):
                continue
            used = model_result.get("eval_features_used")
            if isinstance(used, list) and len(used) > 0:
                return [str(f) for f in used]
        return None

    feats = data.get("selected_features")
    if not isinstance(feats, list):
        return None
    return [str(f) for f in feats]


def selection_combo_key(data: dict) -> tuple[str, str, str, str, float | None]:
    """Stable tuple identifying one grid cell (for logging / dedup); not used internally.

    Order: ``target_col``, ``selector_name``, ``model_type``, ``dataset_stem``,
    ``eval_features_frac``. Omitted JSON fields become ``\"\"`` / ``None``.
    """
    tc = str(data.get("target_col", "") or "")
    sel = str(data.get("selector_name", "") or "")
    mt = str(data.get("model_type", "") or "")
    ds = str(data.get("dataset_stem", "") or "")
    frac_raw = data.get("eval_features_frac")
    try:
        frac = float(frac_raw) if frac_raw is not None else None
    except (TypeError, ValueError):
        frac = None
    return (tc, sel, mt, ds, frac)


def aggregate_selected_feature_votes_across_grid(
    result_json_paths: Iterable[Path | str],
    *,
    dataset_stem: str | None = None,
    target_col: str | None = None,
    dataset_yaml_key: str | None = None,
    feature_source: Literal["selected_features", "eval_reported"] = "selected_features",
    dedupe_features_within_combo: bool = True,
) -> dict[int, dict[str, int]]:
    """Count, per outer fold, how many grid JSONs include each feature.

    Each input path should be one fold-level result file (one ``fold_index`` per file).
    For every file that passes optional metadata filters, increment by 1 each feature
    that appears in that file's feature list (so a count of *N* means the feature showed
    up in *N* selector–model–(hyperparameter) combinations in the provided set).

    Parameters
    ----------
    result_json_paths
        Paths to JSON written by ``run_fold_pipeline_config`` (or compatible dicts).
    dataset_stem, target_col, dataset_yaml_key
        If not ``None``, skip files whose top-level string field does not match exactly
        (after ``str()``). Use together to restrict to one dataset block / one target.
    feature_source
        ``selected_features``: post-selector list from the JSON (default).
        ``eval_reported``: first non-error ``eval_features_used`` in ``evaluation``, if any;
        files without that list are skipped (no votes).
    dedupe_features_within_combo
        If True, each feature counts at most once per JSON even if the list repeats a name.

    Returns
    -------
    dict[int, dict[str, int]]
        ``fold_index`` (0-based, as stored in JSON) → feature name → vote count.
        Fold keys and per-fold feature keys are sorted for stable, diff-friendly output.
    """
    agg: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    for raw_path in result_json_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        if dataset_stem is not None and str(data.get("dataset_stem", "")) != str(
            dataset_stem
        ):
            continue
        if target_col is not None and str(data.get("target_col", "")) != str(
            target_col
        ):
            continue
        if dataset_yaml_key is not None and str(data.get("dataset_yaml_key", "")) != str(
            dataset_yaml_key
        ):
            continue

        fi = data.get("fold_index")
        try:
            fold_k = int(fi)
        except (TypeError, ValueError):
            continue

        feats = _feature_names_from_fold_result(data, feature_source=feature_source)
        if feats is None:
            continue
        if dedupe_features_within_combo:
            feats = list(dict.fromkeys(feats))
        for name in feats:
            agg[fold_k][name] += 1

    out: dict[int, dict[str, int]] = {}
    for fk in sorted(agg.keys()):
        inner = agg[fk]
        out[fk] = {name: inner[name] for name in sorted(inner.keys())}
    return out


def iter_batch_result_json_paths(
    batch_dir: Path | str,
    *,
    recursive: bool = False,
) -> Iterator[Path]:
    """Yield ``*.json`` paths under a batch output directory.

    Config-driven batches often write one subdirectory per dataset stem; use
    ``recursive=True`` to include those nested fold result files.
    """
    root = Path(batch_dir)
    if not root.is_dir():
        return iter(())
    if recursive:
        return (p for p in root.rglob("*.json") if p.is_file())
    return (p for p in root.glob("*.json") if p.is_file())


__all__ = [
    "aggregate_selected_feature_votes_across_grid",
    "iter_batch_result_json_paths",
    "selection_combo_key",
]
