#!/usr/bin/env python3
"""Random-shuffle Spearman baseline on preserved AutoML fold parquets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from automl.feature_selectors import cv_shuffled_fold_ilocs, cv_split_col_ilocs  # noqa: E402
from automl.prepare_run import drop_invalid_target_rows, load_merge_and_expand  # noqa: E402


def _slug_to_variant(dataset_stem: str, yaml_block_key: str) -> str:
    prefix = f"{dataset_stem}_"
    key = str(yaml_block_key)
    if key.startswith(prefix):
        return key[len(prefix):]
    return key


def _fmt_spearman(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _finite_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2 or np.ptp(y_true) == 0.0 or np.ptp(y_pred) == 0.0:
        return None
    rho = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    if rho is None or not math.isfinite(float(rho)):
        return None
    return float(rho)


def _resolve_existing_path(path_raw: str | Path, repo_root: Path) -> Path:
    path = Path(path_raw)
    candidates = [path]
    text = str(path)
    if "/FASTAb/" in text:
        candidates.append(Path(text.replace("/FASTAb/", "/kitAb/")))
        candidates.append(
            Path(text.replace("/FASTAb/", "/kitAb/").replace("/scenario2_prefinal/", "/our_abb2/"))
        )
    if text.startswith("/storage/antibody_data/PairedStructures/FASTAb/"):
        rel = text.split("/storage/antibody_data/PairedStructures/FASTAb/", 1)[1]
        candidates.append(repo_root / rel)
        candidates.append(repo_root / rel.replace("scenario2_prefinal/", "our_abb2/"))
    if not path.is_absolute():
        candidates.append(repo_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _load_manifest(batch_root: Path) -> dict[str, Any]:
    manifest_path = batch_root / "batch_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing batch manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _manifest_dataset_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("datasets") or []:
        if not isinstance(entry, dict):
            continue
        yaml_key = str(entry.get("dataset_yaml_key") or "")
        if not yaml_key:
            continue
        out[yaml_key] = {
            "yaml_block_key": str(entry.get("yaml_block_key") or yaml_key),
            "dataset_stem": str(entry.get("dataset_stem") or ""),
            "dataset_path": entry.get("dataset_path"),
            "developability_results_paths": entry.get("developability_results_paths") or [],
            "split_col": entry.get("split_col"),
            "random_state": entry.get("random_state"),
            "developability_feature_groups_by_target": entry.get(
                "developability_feature_groups_by_target"
            )
            or {},
        }
    return out


def _master_tsv_path(batch_root: Path, manifest: dict[str, Any], repo_root: Path) -> Path:
    raw = manifest.get("master_jobs_tsv")
    if raw:
        resolved = _resolve_existing_path(str(raw), repo_root)
        if resolved.is_file():
            return resolved
    fallback = batch_root / "parallel_jobs_master.tsv"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Missing parallel_jobs_master.tsv under {batch_root}")


def _iter_unique_fold_jobs(
    master_tsv: Path,
    *,
    manifest_lookup: dict[str, dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, int]] = set()
    jobs: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(master_tsv.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            raise ValueError(f"Bad master TSV line {line_no}: expected >=10 columns")
        fold_dir_s, fold_s, dataset_stem, target_col, dataset_yaml_key = parts[:5]
        random_state_s = parts[9]
        fold_k = int(fold_s)
        fold_dir = _resolve_existing_path(fold_dir_s, repo_root)
        lookup = manifest_lookup.get(dataset_yaml_key, {})
        yaml_block_key = lookup.get("yaml_block_key") or re.sub(r"__rs\d+$", "", dataset_yaml_key)
        ds = lookup.get("dataset_stem") or dataset_stem
        key = (dataset_yaml_key, target_col, str(fold_dir), fold_k)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(
            {
                "fold_dir": fold_dir,
                "fold_index": fold_k,
                "dataset_stem": ds,
                "target_col": target_col,
                "dataset_yaml_key": dataset_yaml_key,
                "yaml_block_key": yaml_block_key,
                "variant": _slug_to_variant(ds, yaml_block_key),
                "random_state": int(random_state_s),
            }
        )
    return jobs


def _random_fold_spearman(job: dict[str, Any], *, seed: int) -> float | None:
    fold_dir = Path(job["fold_dir"])
    fold_k = int(job["fold_index"])
    target_col = str(job["target_col"])
    test_path = fold_dir / f"fold_{fold_k}_test.parquet"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    test_df = pd.read_parquet(test_path)
    if target_col not in test_df.columns:
        raise KeyError(f"{target_col!r} not in {test_path}")
    y_true = test_df[target_col].to_numpy(dtype=np.float64, copy=True)
    rng_seed = (
        int(seed)
        + 1_000_003 * int(job["random_state"])
        + 10_007 * int(fold_k)
        + (abs(hash((job["dataset_yaml_key"], target_col))) % 1_000_003)
    )
    rng = np.random.default_rng(rng_seed)
    y_pred = np.array(y_true, copy=True)
    rng.shuffle(y_pred)
    return _finite_spearman(y_true, y_pred)


def _reconstructed_fold_targets(
    job: dict[str, Any],
    *,
    manifest_lookup: dict[str, dict[str, Any]],
    repo_root: Path,
) -> list[np.ndarray]:
    entry = manifest_lookup.get(str(job["dataset_yaml_key"]))
    if not entry:
        raise KeyError(f"No manifest dataset entry for {job['dataset_yaml_key']!r}")
    dataset_path = _resolve_existing_path(str(entry["dataset_path"]), repo_root)
    dev_sources = [
        _resolve_existing_path(str(path), repo_root)
        for path in entry.get("developability_results_paths", [])
    ]
    target_col = str(job["target_col"])
    groups_by_target = entry.get("developability_feature_groups_by_target") or {}
    target_cols = list(groups_by_target.keys()) or [target_col]
    if target_col not in target_cols:
        target_cols.append(target_col)
    dev_groups = list(groups_by_target.get(target_col) or [])
    split_col = entry.get("split_col")
    random_state = int(entry.get("random_state") or job["random_state"])

    merged, _expanded_features = load_merge_and_expand(
        dataset_path,
        name_col="name",
        feature_cols=[],
        target_cols=target_cols,
        developability_sources=dev_sources,
        developability_feature_groups=dev_groups,
        split_col=str(split_col) if split_col else None,
    )
    merged = drop_invalid_target_rows(merged, target_col, max_nan_frac=0.7)
    merged = merged.reset_index(drop=True)
    if split_col:
        _segment_sizes, fold_ilocs = cv_split_col_ilocs(merged[str(split_col)])
    else:
        # Existing prepare_run.py default for these run configs.
        _segment_sizes, fold_ilocs = cv_shuffled_fold_ilocs(
            len(merged), n_splits=5, random_state=random_state
        )
    out: list[np.ndarray] = []
    for _train_idx, test_idx in fold_ilocs:
        out.append(merged.iloc[test_idx][target_col].to_numpy(dtype=np.float64, copy=True))
    return out


def _random_fold_spearman_from_targets(
    job: dict[str, Any],
    y_true: np.ndarray,
    *,
    seed: int,
) -> float | None:
    fold_k = int(job["fold_index"])
    rng_seed = (
        int(seed)
        + 1_000_003 * int(job["random_state"])
        + 10_007 * fold_k
        + (abs(hash((job["dataset_yaml_key"], job["target_col"]))) % 1_000_003)
    )
    rng = np.random.default_rng(rng_seed)
    y_pred = np.array(y_true, copy=True)
    rng.shuffle(y_pred)
    return _finite_spearman(y_true, y_pred)


def build_random_baseline_rows(
    batch_root: Path,
    *,
    repo_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    manifest = _load_manifest(batch_root)
    lookup = _manifest_dataset_lookup(manifest)
    master_tsv = _master_tsv_path(batch_root, manifest, repo_root)
    jobs = _iter_unique_fold_jobs(master_tsv, manifest_lookup=lookup, repo_root=repo_root)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    reconstructed_cache: dict[tuple[str, str], list[np.ndarray]] = {}
    for job in jobs:
        try:
            rho = _random_fold_spearman(job, seed=seed)
        except FileNotFoundError:
            cache_key = (str(job["dataset_yaml_key"]), str(job["target_col"]))
            if cache_key not in reconstructed_cache:
                reconstructed_cache[cache_key] = _reconstructed_fold_targets(
                    job,
                    manifest_lookup=lookup,
                    repo_root=repo_root,
                )
            targets_by_fold = reconstructed_cache[cache_key]
            fold_k = int(job["fold_index"])
            if fold_k < 0 or fold_k >= len(targets_by_fold):
                raise IndexError(
                    f"Fold index {fold_k} out of range for reconstructed {cache_key}"
                )
            rho = _random_fold_spearman_from_targets(
                job, targets_by_fold[fold_k], seed=seed
            )
        if rho is None:
            continue
        grouped[(job["dataset_stem"], job["variant"], job["target_col"])].append(rho)

    rows: list[dict[str, Any]] = []
    for dataset_stem, variant, target_col in sorted(grouped):
        values = grouped[(dataset_stem, variant, target_col)]
        spearman = float(np.mean(values)) if values else None
        rows.append(
            {
                "Dataset_stem": dataset_stem,
                "Variant": variant,
                "Target_col": target_col,
                "Spearman": spearman,
                "Jaccard_norm": None,
                "best_Target-Selector-Model": f"{target_col}-random_baseline-shuffle",
                "best_selector_model_frac": "random_baseline-shuffle-frac000",
                "feature_usage": "{}",
            }
        )
    return rows


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "Dataset_stem",
        "Variant",
        "Target_col",
        "Spearman",
        "Jaccard_norm",
        "best_Target-Selector-Model",
        "best_selector_model_frac",
        "feature_usage",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "Spearman": _fmt_spearman(row.get("Spearman")),
                    "Jaccard_norm": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write random-shuffle Spearman baseline results from existing fold parquets."
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        required=True,
        help="AutoML batch root containing batch_manifest.json and parallel_jobs_master.tsv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path, e.g. our_abb2/analysis_results/results_random_baseline.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args()

    batch_root = _resolve_existing_path(args.batch_root, args.repo_root)
    rows = build_random_baseline_rows(
        batch_root,
        repo_root=args.repo_root.resolve(),
        seed=int(args.seed),
    )
    write_results_csv(args.out, rows)
    print(f"Wrote {len(rows)} random baseline rows to {args.out}")


if __name__ == "__main__":
    main()
