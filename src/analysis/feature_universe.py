"""Feature names and usage counts from developability JSON and batch CSVs."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

from utils.load_results_to_dataframe import _flatten_dict

from .aggregated_csv import (
    COL_DATASET,
    COL_FEATURES,
    COL_RUN_ID,
    COL_SOURCE,
    COL_TARGET,
    run_suffix_after_target,
)

FEATURE_PREFIX = "feature_"

SUMMARY_COL_SPEAR = "best_spearman"
SUMMARY_COL_PEAR = "best_pearson"
COL_SPEAR_FEAT = "best_spearman_features"
COL_PEAR_FEAT = "best_pearson_features"
COL_SPEAR_RUN = "best_spearman_Target-Selector-Model"
COL_PEAR_RUN = "best_pearson_Target-Selector-Model"
COL_TRACK = "Track"


def validate_reference_json(reference_json: Path) -> None:
    ref = reference_json.resolve()
    if not ref.is_file():
        raise FileNotFoundError(f"Reference JSON not found: {ref}")
    bad_suf = ref.suffix.lower()
    if bad_suf in (".pdb", ".cif", ".mmcif", ".ent"):
        raise ValueError(
            f"--reference-json must be developability JSON, not a structure file ({ref}).\n"
            "Example: pdgf38_results/AB-001.json (output of run_developability.py)."
        )


def canonical_feature_names(reference_json: Path) -> list[str]:
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
    return sorted({f"{FEATURE_PREFIX}{k}" for k in flat.keys()})


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
            local[f] += 1
    return dict(local)


def features_in_all_folds(cell: str) -> set[str]:
    data = _parse_selected_by_fold(cell)
    if not data:
        return set()
    fold_sets: list[set[str]] = []
    for feats in data.values():
        if not isinstance(feats, list):
            return set()
        fold_sets.append({f for f in feats if isinstance(f, str)})
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
    validate_reference_json(reference_json)
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
    validate_reference_json(reference_json)
    if not usage_csv.is_file():
        print(f"Error: CSV not found: {usage_csv}", file=sys.stderr)
        return 1

    json_features = feature_names_from_json(reference_json)
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
