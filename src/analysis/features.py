"""Feature naming, usage from batch CSVs, and model shortlist tie-breakers."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, TextIO

from utils.load_results_to_dataframe import _flatten_dict

from .aggregated_csv import (
    COL_DATASET,
    COL_FEATURES,
    COL_RUN_ID,
    COL_SOURCE,
    COL_TARGET,
    COL_TRACK,
    eval_model_slug,
    run_suffix_after_target,
    selector_model_frac,
)

DEVELOPABILITY_JSON_GROUPS: tuple[str, ...] = (
    "surface",
    "general",
    "sequence_motives",
)
FEATURE_PREFIX = "feature_"

SUMMARY_COL_SPEAR = "best_spearman"
SUMMARY_COL_PEAR = "best_pearson"
COL_SPEAR_FEAT = "best_spearman_features"
COL_PEAR_FEAT = "best_pearson_features"
COL_SPEAR_RUN = "best_spearman_Target-Selector-Model"
COL_PEAR_RUN = "best_pearson_Target-Selector-Model"

_FRAC_SUFFIX_RE = re.compile(r"frac(\d{3})$")

EVAL_MODEL_PREFERENCE: tuple[str, ...] = (
    "elasticnet",
    "linear",
    "svm",
    "knn",
    "randomforest",
    "gpr",
)
_EVAL_MODEL_RANK: dict[str, int] = {
    name: i for i, name in enumerate(EVAL_MODEL_PREFERENCE)
}
_DEFAULT_EVAL_RANK = len(EVAL_MODEL_PREFERENCE)


def normalize_feature_name(name: str) -> str:
    n = str(name).strip()
    if n.startswith(FEATURE_PREFIX):
        n = n[len(FEATURE_PREFIX) :]
    for group in DEVELOPABILITY_JSON_GROUPS:
        prefix = f"{group}_"
        if n.startswith(prefix):
            return n[len(prefix) :]
    return n


def normalize_feature_count_dict(counts: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, count in counts.items():
        key = normalize_feature_name(name)
        out[key] = out.get(key, 0) + int(count)
    return dict(sorted(out.items()))


_REFERENCE_CSV_SKIP_COLS = frozenset({"name", "pdb_file", "error"})


def validate_reference_features(reference: Path) -> None:
    ref = reference.resolve()
    if not ref.is_file():
        raise FileNotFoundError(f"Reference features file not found: {ref}")
    bad_suf = ref.suffix.lower()
    if bad_suf in (".pdb", ".cif", ".mmcif", ".ent"):
        raise ValueError(
            f"--reference-json must be developability JSON or features.csv, not a structure file ({ref}).\n"
            "Example: pdgf38_results/AB-001.json (output of run_developability.py)."
        )
    if bad_suf == ".csv":
        with ref.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        feature_cols = [
            c.strip()
            for c in fieldnames
            if c and c.strip() and c.strip() not in _REFERENCE_CSV_SKIP_COLS
        ]
        if not feature_cols:
            raise ValueError(f"No feature columns in reference CSV: {ref}")
        return
    if bad_suf != ".json":
        raise ValueError(
            f"--reference-json must be developability JSON or features.csv, got: {ref}"
        )


def validate_reference_json(reference_json: Path) -> None:
    validate_reference_features(reference_json)


def canonical_feature_names_from_csv(reference_csv: Path) -> list[str]:
    with reference_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    return sorted(
        c.strip()
        for c in fieldnames
        if c and c.strip() and c.strip() not in _REFERENCE_CSV_SKIP_COLS
    )


def canonical_feature_names_from_json(reference_json: Path) -> list[str]:
    raw = reference_json.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        suf = reference_json.suffix.lower()
        hint = ""
        if suf in (".pdb", ".cif", ".mmcif", ".ent"):
            hint = (
                " This path looks like a structure file. "
                "--reference-json must be developability JSON output "
                "(e.g. pdgf38_results/AB-001.json from run_developability.py), not a PDB."
            )
        raise ValueError(
            f"Could not parse --reference-json as JSON: {reference_json}{hint}"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object: {reference_json}")
    flat = dict(_flatten_dict(data))
    return sorted({normalize_feature_name(k) for k in flat.keys()})


def canonical_feature_names(reference: Path) -> list[str]:
    ref = reference.resolve()
    if ref.suffix.lower() == ".csv":
        return canonical_feature_names_from_csv(ref)
    return canonical_feature_names_from_json(ref)


def feature_names_from_json(path: Path) -> set[str]:
    return set(canonical_feature_names(path))


def feature_names_from_usage_csv(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return names
        col = None
        for candidate in ("feature", "Feature", "FEATURE"):
            if candidate in reader.fieldnames:
                col = candidate
                break
        if col is None:
            raise ValueError(
                f"No 'feature' column in {path}; columns={reader.fieldnames!r}"
            )
        for row in reader:
            name = (row.get(col) or "").strip()
            if name:
                names.add(name)
    return names


def features_missing_from_usage(reference_json: Path, usage_csv: Path) -> list[str]:
    return sorted(feature_names_from_json(reference_json) - feature_names_from_usage_csv(usage_csv))


def _parse_selected_by_fold(cell: str) -> dict[str, Any] | None:
    if not cell or not str(cell).strip():
        return None
    try:
        data = json.loads(cell)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def count_fold_selections(cell: str) -> dict[str, int]:
    data = _parse_selected_by_fold(cell)
    if not data:
        return {}
    local: dict[str, int] = defaultdict(int)
    for feats in data.values():
        if not isinstance(feats, list):
            continue
        in_fold = {f for f in feats if isinstance(f, str)}
        for f in in_fold:
            local[normalize_feature_name(f)] += 1
    return dict(local)


def features_in_all_folds(cell: str) -> set[str]:
    data = _parse_selected_by_fold(cell)
    if not data:
        return set()
    fold_sets: list[set[str]] = []
    for feats in data.values():
        if not isinstance(feats, list):
            return set()
        fold_sets.append({normalize_feature_name(f) for f in feats if isinstance(f, str)})
    if not fold_sets:
        return set()
    out = fold_sets[0].copy()
    for s in fold_sets[1:]:
        out &= s
    return out


def _row_signature(row: dict[str, str]) -> str:
    ds = row.get(COL_DATASET, "")
    tgt = row.get(COL_TARGET, "")
    run_id = row.get(COL_RUN_ID, "")
    suffix = run_suffix_after_target(run_id, tgt)
    return f"{ds}|{tgt}|{suffix}"


def aggregate_usage_from_aggregated_csvs(
    aggregated_paths: list[Path],
) -> tuple[dict[str, int], dict[str, set[str]]]:
    total: dict[str, int] = defaultdict(int)
    all_folds_sigs: dict[str, set[str]] = defaultdict(set)

    for path in aggregated_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if COL_FEATURES not in (reader.fieldnames or []):
                print(
                    f"Warning: skip {path.name}: no column {COL_FEATURES!r}",
                    file=sys.stderr,
                )
                continue
            need_sig = (
                COL_DATASET in (reader.fieldnames or [])
                and COL_TARGET in (reader.fieldnames or [])
                and COL_RUN_ID in (reader.fieldnames or [])
            )
            if not need_sig:
                print(
                    f"Warning: {path.name}: missing columns for dataset_target_model "
                    f"({COL_DATASET}, {COL_TARGET}, {COL_RUN_ID}); counts still computed.",
                    file=sys.stderr,
                )
            for row in reader:
                cell = row.get(COL_FEATURES, "")
                for feat, n in count_fold_selections(cell).items():
                    total[feat] += n
                if need_sig:
                    sig = _row_signature(row)
                    for feat in features_in_all_folds(cell):
                        all_folds_sigs[feat].add(sig)

    return dict(total), {k: v for k, v in all_folds_sigs.items()}


def build_feature_usage_rows(
    *,
    canonical: set[str],
    usage: dict[str, int],
    all_folds: dict[str, set[str]],
) -> list[tuple[str, int, str]]:
    extras = sorted(k for k in usage if k not in canonical)
    rows: list[tuple[str, int, str]] = []
    for name in sorted(canonical):
        sigs = all_folds.get(name, set())
        rows.append((name, usage.get(name, 0), ";".join(sorted(sigs))))
    for name in extras:
        sigs = all_folds.get(name, set())
        rows.append((name, usage[name], ";".join(sorted(sigs))))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows


def write_feature_usage_csv(
    dest: TextIO,
    rows: list[tuple[str, int, str]],
) -> None:
    w = csv.writer(dest)
    w.writerow(["feature", "count", "dataset_target_model_all_folds"])
    w.writerows(rows)


def write_global_feature_usage_from_aggregated(
    *,
    reference_json: Path,
    aggregated_paths: list[Path],
    out_path: Path | None,
) -> list[tuple[str, int, str]]:
    validate_reference_features(reference_json)
    canonical = set(canonical_feature_names(reference_json))
    if not aggregated_paths:
        raise FileNotFoundError("No aggregated CSV files matched the given pattern.")

    usage, all_folds = aggregate_usage_from_aggregated_csvs(aggregated_paths)
    extras = sorted(k for k in usage if k not in canonical)
    if extras:
        print(
            f"Note: {len(extras)} feature name(s) in aggregated files are not in "
            f"{reference_json.name} (first few: {extras[:5]})",
            file=sys.stderr,
        )

    rows = build_feature_usage_rows(canonical=canonical, usage=usage, all_folds=all_folds)
    if out_path is not None:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            write_feature_usage_csv(f, rows)
        print(
            f"Wrote {len(rows)} rows ({len(canonical)} canonical + {len(extras)} extra) "
            f"from {len(aggregated_paths)} files to {out_path}",
            file=sys.stderr,
        )
    else:
        write_feature_usage_csv(sys.stdout, rows)
    return rows


def print_json_features_missing_from_usage(
    *,
    reference_json: Path,
    usage_csv: Path,
    out_path: Path | None = None,
) -> int:
    validate_reference_features(reference_json)
    if not usage_csv.is_file():
        print(f"Error: CSV not found: {usage_csv}", file=sys.stderr)
        return 1

    json_features = set(canonical_feature_names(reference_json))
    usage_features = feature_names_from_usage_csv(usage_csv)
    missing = sorted(json_features - usage_features)

    print(f"Flattened scalar features in JSON: {len(json_features)}")
    print(f"Features listed in usage CSV:      {len(usage_features)}")
    print(f"In JSON but not in usage CSV:      {len(missing)}")
    for name in missing:
        print(name)

    if out_path is not None:
        out_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
        print(f"Wrote {len(missing)} lines to {out_path}", file=sys.stderr)
    return 0


def parse_metric_cell(cell: object) -> float | None:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    if isinstance(cell, (int, float)):
        v = float(cell)
        return v if math.isfinite(v) else None
    s = str(cell).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_features_frac_from_run_id(run_id: str, target_col: str) -> float | None:
    suf = selector_model_frac(run_id, target_col)
    m = _FRAC_SUFFIX_RE.search(suf)
    if not m:
        return None
    return int(m.group(1)) / 100.0


def _feature_sets_by_fold(features_cell: str) -> list[set[str]]:
    if not features_cell or not str(features_cell).strip():
        return []
    try:
        data = json.loads(features_cell)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    out: list[set[str]] = []
    for feats in data.values():
        if isinstance(feats, list):
            out.append({str(x) for x in feats if isinstance(x, str)})
    return out


def _jaccard_sets(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def mean_jaccard_across_folds(features_cell: str) -> float | None:
    sets = _feature_sets_by_fold(features_cell)
    if not sets:
        return None
    if len(sets) == 1:
        return 1.0
    vals: list[float] = []
    for a, b in combinations(sets, 2):
        vals.append(_jaccard_sets(a, b))
    return float(sum(vals) / len(vals)) if vals else None


def eval_model_preference_rank(run_id: str, target_col: str) -> int:
    em = eval_model_slug(run_id, target_col)
    if em is None:
        return _DEFAULT_EVAL_RANK
    return _EVAL_MODEL_RANK.get(em, _DEFAULT_EVAL_RANK)


def row_tiebreak_key(
    row: dict[str, Any],
    *,
    features_col: str = COL_FEATURES,
    run_id_col: str = COL_RUN_ID,
    target_col: str = COL_TARGET,
) -> tuple[float, float, int, str]:
    tgt = str(row.get(target_col, ""))
    run_id = str(row.get(run_id_col, ""))
    jac = mean_jaccard_across_folds(str(row.get(features_col, "")))
    jac_sort = -jac if jac is not None else float("inf")
    frac = parse_features_frac_from_run_id(run_id, tgt)
    frac_sort = frac if frac is not None else float("inf")
    em_rank = eval_model_preference_rank(run_id, tgt)
    return (jac_sort, frac_sort, em_rank, run_id)


def shortlist_sort_key(
    row: dict[str, Any],
    *,
    metric_col: str,
    features_col: str = COL_FEATURES,
    run_id_col: str = COL_RUN_ID,
    target_col: str = COL_TARGET,
    parse_metric: Callable[[object], float | None] = parse_metric_cell,
) -> tuple[float, float, float, int, str]:
    spe = parse_metric(row.get(metric_col, ""))
    spe_sort = -spe if spe is not None else float("inf")
    tie = row_tiebreak_key(
        row,
        features_col=features_col,
        run_id_col=run_id_col,
        target_col=target_col,
    )
    return (spe_sort, *tie)


def pick_preferred_row(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    metric_col: str,
    primary_metric_a: float | None = None,
    primary_metric_b: float | None = None,
    metric_eps: float = 1e-12,
) -> dict[str, Any]:
    if primary_metric_a is None:
        primary_metric_a = parse_metric_cell(a.get(metric_col, ""))
    if primary_metric_b is None:
        primary_metric_b = parse_metric_cell(b.get(metric_col, ""))
    if primary_metric_a is not None and primary_metric_b is not None:
        if primary_metric_a > primary_metric_b + metric_eps:
            return a
        if primary_metric_b > primary_metric_a + metric_eps:
            return b
    return a if row_tiebreak_key(a) <= row_tiebreak_key(b) else b


def select_close_top_models(
    rows: list[dict[str, Any]],
    *,
    metric_col: str,
    max_rank: int = 3,
    margin: float = 0.1,
    parse_metric: Callable[[object], float | None] = parse_metric_cell,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        v = parse_metric(r.get(metric_col, ""))
        if v is not None:
            scored.append((v, r))
    if not scored:
        return []
    scored.sort(key=lambda t: t[0], reverse=True)
    best_val = scored[0][0]
    threshold = best_val - float(margin)
    candidates = [r for v, r in scored if v >= threshold]
    candidates.sort(
        key=lambda r: shortlist_sort_key(r, metric_col=metric_col, parse_metric=parse_metric)
    )
    return candidates[: max(1, int(max_rank))]


def collapse_best_track_per_group(
    summary_rows: list[dict[str, Any]],
    *,
    dataset_key: str = "Dataset_stem",
    source_key: str = "Developability_source",
    target_key: str = "Target_col",
    track_key: str = "Track",
    spearman_key: str = "best_spearman",
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        ds = str(r.get(dataset_key, ""))
        src = str(r.get(source_key, ""))
        tgt = str(r.get(target_key, ""))
        if not ds or not src or not tgt:
            continue
        by_key[(ds, src, tgt)].append(r)

    out: list[dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        rows = by_key[key]
        best_row: dict[str, Any] | None = None
        for r in rows:
            if best_row is None:
                best_row = r
                continue
            best_row = pick_preferred_row(
                best_row,
                r,
                metric_col=spearman_key,
                primary_metric_a=parse_metric_cell(best_row.get(spearman_key, "")),
                primary_metric_b=parse_metric_cell(r.get(spearman_key, "")),
            )
        if best_row is not None:
            out.append(dict(best_row))
        elif rows:
            out.append(dict(rows[0]))
    return out
