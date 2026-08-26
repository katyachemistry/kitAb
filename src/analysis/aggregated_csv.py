from __future__ import annotations

import re
from pathlib import Path
from typing import Any

COL_RUN_ID = "Target-Selector-Model"
COL_TARGET = "Target_col"
COL_TRACK = "Track"
COL_DATASET = "Dataset_stem"
COL_SOURCE = "Developability_source"
COL_SPEAR = "Spearman"
COL_SPEAR_POOLED = "Spearman_pooled_oof"
COL_PEAR = "Pearson"
COL_PEAR_POOLED = "Pearson_pooled_oof"
COL_R2 = "R2"
COL_R2_POOLED = "R2_pooled_oof"
COL_N_OOF = "n_oof"
COL_N_FOLDS_PRESENT = "n_folds_present"
COL_FEATURES = "selected_features_by_fold"

ANALYSIS_RESULTS_DIRNAME = "analysis_results"

_FOLD_SPEAR_RE = re.compile(r"^fold_(\d+)_spearman$")


def is_our_source(source: str) -> bool:
    s = str(source)
    if s.startswith("descriptors_") or "__descriptors_" in s:
        return True
    return "_results" in s or "_our_" in s or "_our__" in s or s.endswith("_our")


def selector_model_frac(run_id: str, target_col: str) -> str:
    prefix = target_col + "-"
    if run_id.startswith(prefix):
        return run_id[len(prefix) :]
    parts = run_id.split("-", 1)
    if len(parts) == 2:
        return parts[1]
    return run_id


def eval_model_slug(run_id: str, target_col: str) -> str | None:
    suf = selector_model_frac(run_id, target_col)
    parts = suf.rsplit("-", 3)
    if len(parts) == 4 and str(parts[-1]).startswith("frac"):
        return str(parts[-2]).strip().lower()
    return None


def row_matches_eval_model_filter(
    run_id: str, target_col: str, eval_model_lc: str | None
) -> bool:
    if eval_model_lc is None:
        return True
    slug = eval_model_slug(run_id, target_col)
    return slug is not None and slug == eval_model_lc


def run_suffix_after_target(run_id: str, target_col: str) -> str:
    prefix = target_col + "-"
    if run_id.startswith(prefix):
        return run_id[len(prefix) :]
    parts = run_id.split("-", 1)
    return parts[1] if len(parts) == 2 else run_id


def fold_spearman_columns(columns: Any) -> list[str]:
    pairs: list[tuple[int, str]] = []
    for c in columns:
        m = _FOLD_SPEAR_RE.match(str(c))
        if m:
            pairs.append((int(m.group(1)), str(c)))
    return [c for _, c in sorted(pairs)]


def expand_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in inputs:
        p = Path(pat)
        if any(ch in pat for ch in "*?["):
            parent = p.parent if str(p.parent) not in ("", ".") else Path.cwd()
            paths.extend(sorted(parent.glob(p.name)))
        elif p.is_file():
            paths.append(p)
    return paths


def expand_glob_pattern(pattern: str) -> list[Path]:
    p = Path(pattern)
    if any(ch in pattern for ch in "*?["):
        parent = p.parent if str(p.parent) not in ("", ".") else Path.cwd()
        return sorted(parent.glob(p.name))
    return [p] if p.is_file() else []


def resolve_output_dir(path: Path | None = None) -> Path:
    if path is None:
        return (Path.cwd() / ANALYSIS_RESULTS_DIRNAME).resolve()
    p = Path(path)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
