#!/usr/bin/env python3
"""Nested Elastic Net comparison for all matched kitAb/ProperMAb ABB2-1 folds."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.nested_elasticnet_all_targets import (
    DEFAULT_ALPHAS,
    DEFAULT_L1_RATIOS,
    _feature_columns,
    _run_outer,
    _spearman,
)
from analysis.nested_elasticnet_all_methods import _root_key

REPO = Path("/storage/antibody_data/PairedStructures/kitAb")


def _discover_roots() -> dict[str, dict[tuple[str, str], Path]]:
    runs = REPO / "runs"
    kitab_candidates = [
        *runs.glob(
            "*_cv_prepare__our_abb2_final_set_of_features_descriptors_"
            "*_abb2_1_results*"
        ),
        *runs.glob(
            "hutchinson2023enhancement_top200tm1_igg_cv_prepare__"
            "our_abb2_no_sequence_motives_descriptors_*_abb2_1_results*"
        ),
    ]
    propermab_candidates = list(
        runs.glob(
            "*_cv_prepare__nested_propermab_abb2__*_abb2_1_propermab*"
        )
    )
    roots = {
        "kitAb": {_root_key(path): path for path in kitab_candidates if path.is_dir()},
        "ProperMAb": {
            _root_key(path): path for path in propermab_candidates if path.is_dir()
        },
    }
    unmatched_kitab = sorted(set(roots["kitAb"]) - set(roots["ProperMAb"]))
    unmatched_propermab = sorted(set(roots["ProperMAb"]) - set(roots["kitAb"]))
    if unmatched_kitab or unmatched_propermab:
        raise RuntimeError(
            "Unmatched fold roots: "
            f"kitAb-only={unmatched_kitab}, ProperMAb-only={unmatched_propermab}"
        )
    return roots


def _target_dirs(root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in root.glob("target_*")
        if path.is_dir() and (path / "meta.json").is_file()
    }


def _features(fold_dir: Path, target: str) -> list[str]:
    meta = json.loads((fold_dir / "meta.json").read_text())
    sample = pd.read_parquet(fold_dir / "fold_0_train.parquet")
    features = [
        str(feature)
        for feature in meta.get("feature_cols", [])
        if str(feature) in sample.columns
    ]
    return features or _feature_columns(sample, target)


def _validate_paired_folds(kitab_dir: Path, propermab_dir: Path) -> int:
    kitab_meta = json.loads((kitab_dir / "meta.json").read_text())
    propermab_meta = json.loads((propermab_dir / "meta.json").read_text())
    n_splits = int(kitab_meta["n_splits"])
    if int(propermab_meta["n_splits"]) != n_splits:
        raise ValueError(f"Fold-count mismatch: {kitab_dir} vs {propermab_dir}")
    for fold in range(n_splits):
        kitab_names = set(
            pd.read_parquet(
                kitab_dir / f"fold_{fold}_test.parquet", columns=["name"]
            )["name"].astype(str)
        )
        propermab_names = set(
            pd.read_parquet(
                propermab_dir / f"fold_{fold}_test.parquet", columns=["name"]
            )["name"].astype(str)
        )
        if kitab_names != propermab_names:
            raise ValueError(
                f"Outer-test names differ for {kitab_dir.name}, fold {fold}"
            )
    return n_splits


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    result = _run_outer(
        task["fold_dir"],
        target_col=task["target_col"],
        outer_k=task["outer_k"],
        features=task["features"],
        alphas=task["alphas"],
        l1_ratios=task["l1_ratios"],
        work_root=task["work_root"],
    )
    result.update(
        {
            "method": task["method"],
            "Dataset_stem": task["Dataset_stem"],
            "split": task["split"],
            "n_input_features": len(task["features"]),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS)
    )
    parser.add_argument(
        "--l1-ratios", nargs="+", type=float, default=list(DEFAULT_L1_RATIOS)
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or REPO / f"runs/nested_elasticnet_all_abb2_1_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = _discover_roots()

    tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for key in sorted(roots["kitAb"]):
        stem, split = key
        method_targets = {
            method: _target_dirs(method_roots[key])
            for method, method_roots in roots.items()
        }
        if set(method_targets["kitAb"]) != set(method_targets["ProperMAb"]):
            raise ValueError(f"Target mismatch for {stem}, {split}")
        for target in sorted(method_targets["kitAb"]):
            n_splits = _validate_paired_folds(
                method_targets["kitAb"][target],
                method_targets["ProperMAb"][target],
            )
            for method in ("kitAb", "ProperMAb"):
                fold_dir = method_targets[method][target]
                features = _features(fold_dir, target)
                manifest_rows.append(
                    {
                        "Dataset_stem": stem,
                        "split": split,
                        "Target_col": target,
                        "method": method,
                        "fold_root": str(fold_dir),
                        "n_outer_folds": n_splits,
                        "n_input_features": len(features),
                    }
                )
                for outer_k in range(n_splits):
                    tasks.append(
                        {
                            "fold_dir": fold_dir,
                            "target_col": target,
                            "outer_k": outer_k,
                            "features": features,
                            "alphas": list(args.alphas),
                            "l1_ratios": list(args.l1_ratios),
                            "work_root": (
                                out_dir / "work" / method / stem / split
                            ),
                            "method": method,
                            "Dataset_stem": stem,
                            "split": split,
                        }
                    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(
        f"Running {len(tasks)} outer folds across "
        f"{manifest[['Dataset_stem', 'Target_col']].drop_duplicates().shape[0]} "
        f"dataset-target pairs ({args.jobs} workers)",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.jobs, len(tasks)),
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"Completed {index}/{len(futures)} outer folds", flush=True)
    results.sort(
        key=lambda result: (
            result["Dataset_stem"],
            result["split"],
            result["target_col"],
            result["method"],
            result["outer_fold"],
        )
    )

    oof_rows: list[dict[str, Any]] = []
    result_metadata: list[dict[str, Any]] = []
    for result in results:
        for row in result.pop("oof_rows"):
            oof_rows.append(
                {
                    **row,
                    "method": result["method"],
                    "Dataset_stem": result["Dataset_stem"],
                    "split": result["split"],
                }
            )
        result_metadata.append(result)
    oof = pd.DataFrame(oof_rows)
    oof.to_parquet(out_dir / "oof.parquet", index=False)
    (out_dir / "inner_selection.json").write_text(
        json.dumps(result_metadata, indent=2, default=str)
    )

    method_rows: list[dict[str, Any]] = []
    group_cols = ["Dataset_stem", "split", "target_col", "method"]
    for key, group in oof.groupby(group_cols, sort=True):
        stem, split, target, method = key
        outer = [
            result
            for result in result_metadata
            if result["Dataset_stem"] == stem
            and result["split"] == split
            and result["target_col"] == target
            and result["method"] == method
        ]
        method_rows.append(
            {
                "Dataset_stem": stem,
                "split": split,
                "Target_col": target,
                "method": method,
                "Spearman_pooled_oof": _spearman(group["y"], group["yhat"]),
                "n_oof": len(group),
                "n_input_features": outer[0]["n_input_features"],
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(out_dir / "method_summary_by_split.csv", index=False)

    value_cols = ["Spearman_pooled_oof", "n_oof", "n_input_features"]
    comparison = method_summary.pivot(
        index=["Dataset_stem", "split", "Target_col"],
        columns="method",
        values=value_cols,
    )
    comparison.columns = [
        f"{method}_{metric}" for metric, method in comparison.columns
    ]
    comparison = comparison.reset_index()
    comparison["delta_kitAb_minus_ProperMAb"] = (
        comparison["kitAb_Spearman_pooled_oof"]
        - comparison["ProperMAb_Spearman_pooled_oof"]
    )
    comparison["better"] = comparison["delta_kitAb_minus_ProperMAb"].map(
        lambda delta: (
            "kitAb" if delta > 0 else ("ProperMAb" if delta < 0 else "tie")
        )
    )
    comparison.to_csv(out_dir / "comparison_by_split.csv", index=False)

    aggregate = (
        comparison.groupby(["Dataset_stem", "Target_col"], as_index=False)
        .agg(
            n_split_repeats=("split", "size"),
            kitAb_Spearman_pooled_oof_mean=(
                "kitAb_Spearman_pooled_oof",
                "mean",
            ),
            kitAb_Spearman_pooled_oof_std=(
                "kitAb_Spearman_pooled_oof",
                "std",
            ),
            ProperMAb_Spearman_pooled_oof_mean=(
                "ProperMAb_Spearman_pooled_oof",
                "mean",
            ),
            ProperMAb_Spearman_pooled_oof_std=(
                "ProperMAb_Spearman_pooled_oof",
                "std",
            ),
        )
    )
    aggregate["delta_kitAb_minus_ProperMAb"] = (
        aggregate["kitAb_Spearman_pooled_oof_mean"]
        - aggregate["ProperMAb_Spearman_pooled_oof_mean"]
    )
    aggregate["better"] = aggregate["delta_kitAb_minus_ProperMAb"].map(
        lambda delta: (
            "kitAb" if delta > 0 else ("ProperMAb" if delta < 0 else "tie")
        )
    )
    aggregate.to_csv(out_dir / "comparison_seed_aggregated.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "alphas": args.alphas,
                "l1_ratios": args.l1_ratios,
                "jobs": args.jobs,
                "n_outer_tasks": len(tasks),
            },
            indent=2,
        )
    )
    print("\n" + aggregate.to_string(index=False), flush=True)
    print(f"\nWrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
