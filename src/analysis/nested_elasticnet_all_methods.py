#!/usr/bin/env python3
"""Run nested ElasticNet over every prepared kitAb/external feature universe.

Methods:
  * kitAb and ProperMAb: ABB2, ABB3, FlashABB; structure variants 1, 2, 3
  * ProperMAb sequence-feature baseline: ABB2 variants 1, 2, 3
  * TAP: one structure-independent feature set

Every outer-fold result is checkpointed immediately. Re-running with --resume
skips valid checkpoints, making this suitable for a long parallel batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Prevent 100 worker processes from each spawning a BLAS thread pool.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
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

REPO = Path("/storage/antibody_data/PairedStructures/kitAb")
RUNS = REPO / "runs"
BACKENDS = ("abb2", "abb3", "flashabb")
STRUCTURE_VARIANTS = (1, 2, 3)
SEQUENCE_FEATURES = (
    "cdr_h3_length",
    "aromatic_cdr",
    "theoretical_pi",
    "n_charged_res_fv",
    "fv_charge",
    "fv_csp",
)
METHOD_SLUGS = {
    "kitAb": "kitab",
    "PROPERMAB": "propermab",
    "Sequence features baseline": "sequence_baseline",
    "TAP": "tap",
}
# Match the publication Spearman figure exclusion policy.
EXCLUDED_TARGETS = frozenset({"target_Fab_pI"})
CONFIG_COMPARE_KEYS = ("alphas", "l1_ratios", "excluded_targets")
_SEED_RE = re.compile(r"__rs(\d+)$")


def _root_key(path: Path) -> tuple[str, str]:
    stem = path.name.split("_cv_prepare__", 1)[0]
    match = _SEED_RE.search(path.name)
    split = f"rs{match.group(1)}" if match else "sequence_aware"
    return stem, split


def _choose(
    selected: dict[tuple[str, str, str, int], tuple[int, Path]],
    *,
    path: Path,
    backend: str,
    variant: int,
    priority: int,
) -> None:
    stem, split = _root_key(path)
    key = (stem, split, backend, variant)
    previous = selected.get(key)
    if previous is None or priority > previous[0]:
        selected[key] = (priority, path)


def _discover_kitab() -> dict[tuple[str, str, str, int], Path]:
    selected: dict[tuple[str, str, str, int], tuple[int, Path]] = {}
    for backend in BACKENDS:
        markers = (
            (f"our_{backend}_final_set_of_features_descriptors_", 30),
            (f"our_{backend}_no_sequence_motives_descriptors_", 20),
            (f"our_{backend}_descriptors_", 10),
        )
        for marker, priority in markers:
            for path in RUNS.glob(f"*_cv_prepare__{marker}*"):
                if not path.is_dir():
                    continue
                for variant in STRUCTURE_VARIANTS:
                    token = f"_{backend}_{variant}_results"
                    if token in path.name:
                        _choose(
                            selected,
                            path=path,
                            backend=backend,
                            variant=variant,
                            priority=priority,
                        )
                        break
    return {key: value[1] for key, value in selected.items()}


def _discover_propermab() -> dict[tuple[str, str, str, int], Path]:
    selected: dict[tuple[str, str, str, int], tuple[int, Path]] = {}
    for backend in BACKENDS:
        markers = (
            (f"nested_propermab_{backend}__", 30),
            (f"descriptors_propermab_{backend}_", 10),
        )
        for marker, priority in markers:
            for path in RUNS.glob(f"*_cv_prepare__{marker}*"):
                if not path.is_dir() or "NO_PATCH_FEATURES" in path.name:
                    continue
                for variant in STRUCTURE_VARIANTS:
                    token = f"_{backend}_{variant}_propermab"
                    if token in path.name:
                        _choose(
                            selected,
                            path=path,
                            backend=backend,
                            variant=variant,
                            priority=priority,
                        )
                        break
    return {key: value[1] for key, value in selected.items()}


def _discover_tap() -> dict[tuple[str, str], Path]:
    selected: dict[tuple[str, str], Path] = {}
    for path in RUNS.glob("*_cv_prepare__descriptors_tap_*"):
        if path.is_dir():
            selected[_root_key(path)] = path
    return selected


def _target_dirs(root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in root.glob("target_*")
        if path.is_dir() and (path / "meta.json").is_file()
    }


def _features(
    fold_dir: Path,
    target_col: str,
    *,
    feature_mode: str,
) -> list[str]:
    meta = json.loads((fold_dir / "meta.json").read_text())
    sample = pd.read_parquet(fold_dir / "fold_0_train.parquet")
    features = [
        str(feature)
        for feature in meta.get("feature_cols", [])
        if str(feature) in sample.columns
    ]
    if not features:
        features = _feature_columns(sample, target_col)

    if feature_mode == "sequence":
        subtype = [
            feature
            for feature in features
            if feature.startswith("hc_subtype__")
            or feature.startswith("lc_subtype__")
        ]
        selected = subtype + [
            feature for feature in SEQUENCE_FEATURES if feature in sample.columns
        ]
        missing = [feature for feature in SEQUENCE_FEATURES if feature not in selected]
        if missing:
            raise ValueError(
                f"{fold_dir}: missing sequence-baseline features {missing}"
            )
        return selected

    non_subtype = [
        feature
        for feature in features
        if not feature.startswith("hc_subtype__")
        and not feature.startswith("lc_subtype__")
    ]
    if feature_mode in {"kitab", "propermab"} and len(non_subtype) < 20:
        raise ValueError(
            f"{fold_dir}: {feature_mode} root has only {len(non_subtype)} "
            "non-subtype features; likely a sequence-baseline overwrite"
        )
    if feature_mode == "tap":
        expected = {"PSH", "PPC", "PNC", "SFvCSP"}
        if not expected.issubset(features):
            raise ValueError(
                f"{fold_dir}: TAP features missing {sorted(expected - set(features))}"
            )
    return features


def _task_checkpoint(out_dir: Path, task: dict[str, Any]) -> Path:
    return (
        out_dir
        / "checkpoints"
        / METHOD_SLUGS[task["method"]]
        / task["variant"]
        / task["Dataset_stem"]
        / task["split"]
        / task["target_col"]
        / f"outer_{task['outer_k']}.json"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, default=str))
    temporary.replace(path)


def _checkpoint_matches_task(
    result: dict[str, Any],
    task: dict[str, Any],
) -> bool:
    expected = {
        "method": task["method"],
        "variant": task["variant"],
        "structure_model": task["structure_model"],
        "Dataset_stem": task["Dataset_stem"],
        "split": task["split"],
        "target_col": task["target_col"],
        "outer_fold": task["outer_k"],
        "fold_root": str(task["fold_dir"]),
        "n_input_features": len(task["features"]),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            return False
    oof_rows = result.get("oof_rows")
    if not isinstance(oof_rows, list) or len(oof_rows) == 0:
        return False
    if int(result.get("n_test", -1)) != len(oof_rows):
        return False
    names = [str(row.get("name", "")) for row in oof_rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        return False
    for row in oof_rows:
        try:
            y = float(row["y"])
            yhat = float(row["yhat"])
        except (KeyError, TypeError, ValueError):
            return False
        if not np.isfinite(y) or not np.isfinite(yhat):
            return False
    return True


def _load_checkpoint(path: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return result if _checkpoint_matches_task(result, task) else None


def _manifest_fingerprint(manifest: pd.DataFrame) -> str:
    if manifest.empty:
        raise RuntimeError("Empty task manifest; refusing to fingerprint")
    payload = manifest.sort_values(
        [
            "method",
            "variant",
            "Dataset_stem",
            "split",
            "Target_col",
            "fold_root",
        ]
    ).to_json(orient="records")
    return hashlib.sha1(payload.encode()).hexdigest()


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
            "variant": task["variant"],
            "structure_model": task["structure_model"],
            "Dataset_stem": task["Dataset_stem"],
            "split": task["split"],
            "n_input_features": len(task["features"]),
            "fold_root": str(task["fold_dir"]),
        }
    )
    return result


def _fisher_mean(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(array) == 0:
        return float("nan")
    return float(np.tanh(np.arctanh(np.clip(array, -0.999999, 0.999999)).mean()))


def _summarize(
    results: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    for result in results:
        for row in result["oof_rows"]:
            oof_rows.append(
                {
                    **row,
                    "method": result["method"],
                    "variant": result["variant"],
                    "structure_model": result["structure_model"],
                    "Dataset_stem": result["Dataset_stem"],
                    "split": result["split"],
                }
            )
        outer_rows.append(
            {
                key: result.get(key)
                for key in (
                    "method",
                    "variant",
                    "structure_model",
                    "Dataset_stem",
                    "split",
                    "target_col",
                    "outer_fold",
                    "n_test",
                    "spearman",
                    "alpha",
                    "l1_ratio",
                    "inner_pooled_spearman",
                    "n_nonzero",
                    "n_input_features",
                    "fold_root",
                )
            }
        )
    oof = pd.DataFrame(oof_rows)
    oof.to_parquet(out_dir / "oof.parquet", index=False)
    pd.DataFrame(outer_rows).to_csv(out_dir / "outer_summary.csv", index=False)

    variant_rows: list[dict[str, Any]] = []
    keys = [
        "method",
        "variant",
        "structure_model",
        "Dataset_stem",
        "split",
        "target_col",
    ]
    for key, group in oof.groupby(keys, sort=True):
        variant_rows.append(
            {
                **dict(zip(keys, key)),
                "Spearman_pooled_oof": _spearman(group["y"], group["yhat"]),
                "n_oof": len(group),
            }
        )
    variant_summary = pd.DataFrame(variant_rows)
    variant_summary.to_csv(out_dir / "variant_split_summary.csv", index=False)

    structure_rows: list[dict[str, Any]] = []
    structure_keys = ["method", "Dataset_stem", "target_col", "structure_model"]
    for key, group in variant_summary.groupby(structure_keys, sort=True):
        structure_rows.append(
            {
                **dict(zip(structure_keys, key)),
                "structure_model_Spearman": _fisher_mean(
                    group["Spearman_pooled_oof"]
                ),
                "n_variant_split_runs": len(group),
            }
        )
    structure_summary = pd.DataFrame(structure_rows)
    structure_summary.to_csv(
        out_dir / "structure_model_summary.csv", index=False
    )

    averaged_rows: list[dict[str, Any]] = []
    average_keys = ["method", "Dataset_stem", "target_col"]
    for key, group in structure_summary.groupby(average_keys, sort=True):
        values = group["structure_model_Spearman"]
        row = {
            **dict(zip(average_keys, key)),
            "Spearman": _fisher_mean(values),
            "lower": float(values.min()),
            "upper": float(values.max()),
            "n_structure_models": len(group),
            "n_variant_split_runs": int(group["n_variant_split_runs"].sum()),
        }
        for record in group.itertuples():
            row[f"{record.structure_model}_Spearman"] = (
                record.structure_model_Spearman
            )
        averaged_rows.append(row)
    averaged = pd.DataFrame(averaged_rows)
    averaged.to_csv(out_dir / "averaged_results.csv", index=False)
    return variant_summary, averaged


def _build_tasks(
    *,
    out_dir: Path,
    alphas: list[float],
    l1_ratios: list[float],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    kitab = _discover_kitab()
    propermab = _discover_propermab()
    tap = _discover_tap()
    structural_keys = set(kitab) | set(propermab)
    if set(kitab) != set(propermab):
        raise RuntimeError(
            "kitAb/ProperMAb structural root mismatch: "
            f"kitAb-only={sorted(set(kitab) - set(propermab))}; "
            f"ProperMAb-only={sorted(set(propermab) - set(kitab))}"
        )
    tap_keys = {(stem, split) for stem, split, _, _ in structural_keys}
    if set(tap) != tap_keys:
        raise RuntimeError(
            f"TAP root mismatch: missing={sorted(tap_keys - set(tap))}; "
            f"extra={sorted(set(tap) - tap_keys)}"
        )

    configs: list[dict[str, Any]] = []
    for method, root_map, mode, suffix in (
        ("kitAb", kitab, "kitab", ""),
        ("PROPERMAB", propermab, "propermab", "_propermab"),
    ):
        for (stem, split, backend, variant), root in sorted(root_map.items()):
            configs.append(
                {
                    "method": method,
                    "variant": f"{backend}_{variant}{suffix}",
                    "structure_model": backend,
                    "Dataset_stem": stem,
                    "split": split,
                    "root": root,
                    "feature_mode": mode,
                }
            )

    # Sequence baseline uses sequence features only (no structure descriptors);
    # one ABB2 fold root is enough — structure variant does not matter.
    for (stem, split, backend, variant), root in sorted(propermab.items()):
        if backend != "abb2" or variant != 1:
            continue
        configs.append(
            {
                "method": "Sequence features baseline",
                "variant": f"abb2_{variant}_propermab",
                "structure_model": "abb2",
                "Dataset_stem": stem,
                "split": split,
                "root": root,
                "feature_mode": "sequence",
            }
        )

    for (stem, split), root in sorted(tap.items()):
        configs.append(
            {
                "method": "TAP",
                "variant": "tap",
                "structure_model": "tap",
                "Dataset_stem": stem,
                "split": split,
                "root": root,
                "feature_mode": "tap",
            }
        )

    tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for config in configs:
        targets = _target_dirs(config["root"])
        if not targets:
            raise FileNotFoundError(f"No target folds under {config['root']}")
        for target_col, fold_dir in sorted(targets.items()):
            if target_col in EXCLUDED_TARGETS:
                continue
            meta = json.loads((fold_dir / "meta.json").read_text())
            n_splits = int(meta["n_splits"])
            features = _features(
                fold_dir,
                target_col,
                feature_mode=config["feature_mode"],
            )
            manifest_rows.append(
                {
                    **{
                        key: config[key]
                        for key in (
                            "method",
                            "variant",
                            "structure_model",
                            "Dataset_stem",
                            "split",
                        )
                    },
                    "Target_col": target_col,
                    "fold_root": str(fold_dir),
                    "n_outer_folds": n_splits,
                    "n_input_features": len(features),
                }
            )
            for outer_k in range(n_splits):
                task = {
                    **{
                        key: config[key]
                        for key in (
                            "method",
                            "variant",
                            "structure_model",
                            "Dataset_stem",
                            "split",
                        )
                    },
                    "target_col": target_col,
                    "fold_dir": fold_dir,
                    "features": features,
                    "outer_k": outer_k,
                    "alphas": alphas,
                    "l1_ratios": l1_ratios,
                    "work_root": (
                        out_dir
                        / "work"
                        / METHOD_SLUGS[config["method"]]
                        / config["variant"]
                        / config["Dataset_stem"]
                        / config["split"]
                    ),
                }
                task["checkpoint"] = _task_checkpoint(out_dir, task)
                tasks.append(task)
    return tasks, pd.DataFrame(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS)
    )
    parser.add_argument(
        "--l1-ratios", nargs="+", type=float, default=list(DEFAULT_L1_RATIOS)
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "runs/nested_elasticnet_all_methods",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if any(alpha <= 0 for alpha in args.alphas):
        parser.error("--alphas must all be > 0")
    if any(not 0 <= ratio <= 1 for ratio in args.l1_ratios):
        parser.error("--l1-ratios must be between 0 and 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "run_config.json"
    run_config = {
        "alphas": list(args.alphas),
        "l1_ratios": list(args.l1_ratios),
        "jobs": args.jobs,
        "excluded_targets": sorted(EXCLUDED_TARGETS),
    }

    tasks, manifest = _build_tasks(
        out_dir=args.out_dir,
        alphas=list(args.alphas),
        l1_ratios=list(args.l1_ratios),
    )
    run_config["manifest_sha1"] = _manifest_fingerprint(manifest)

    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but missing {config_path}; "
                "use a fresh --out-dir or omit --resume"
            )
        previous = json.loads(config_path.read_text())
        for key in CONFIG_COMPARE_KEYS:
            if previous.get(key) != run_config[key]:
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"{previous.get(key)!r} != {run_config[key]!r}"
                )
        if previous.get("manifest_sha1") != run_config["manifest_sha1"]:
            raise ValueError(
                "Resume configuration mismatch for manifest_sha1 "
                "(discovered roots/targets/features changed)"
            )
    elif any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"{args.out_dir} is not empty; pass --resume or choose another --out-dir"
        )
    config_path.write_text(json.dumps(run_config, indent=2))
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    counts = manifest.groupby("method").agg(
        configurations=("variant", "size"),
        dataset_targets=("Target_col", "size"),
        outer_folds=("n_outer_folds", "sum"),
    )
    print(counts.to_string(), flush=True)
    print(
        f"\nTotal: {len(manifest)} variant/split/target configurations; "
        f"{len(tasks)} outer-fold tasks",
        flush=True,
    )
    if args.dry_run:
        print("Dry run complete; no models fitted.", flush=True)
        return

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        checkpoint = (
            _load_checkpoint(task["checkpoint"], task) if args.resume else None
        )
        if checkpoint is None:
            pending.append(task)
        else:
            results.append(checkpoint)
    print(
        f"Resume: {len(results)} complete, {len(pending)} pending; "
        f"launching {min(args.jobs, len(pending))} workers",
        flush=True,
    )

    if pending:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(pending)),
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = {pool.submit(_run_task, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                if not _checkpoint_matches_task(result, task):
                    raise RuntimeError(
                        f"Invalid result for {task['checkpoint']}; refusing to write"
                    )
                _write_json_atomic(task["checkpoint"], result)
                results.append(result)
                if index % 50 == 0 or index == len(futures):
                    print(
                        f"Completed {index}/{len(futures)} pending outer folds "
                        f"({len(results)}/{len(tasks)} total)",
                        flush=True,
                    )

    if len(results) != len(tasks):
        raise RuntimeError(
            f"Expected {len(tasks)} results, got {len(results)}"
        )
    results.sort(
        key=lambda result: (
            result["method"],
            result["variant"],
            result["Dataset_stem"],
            result["split"],
            result["target_col"],
            result["outer_fold"],
        )
    )
    _, averaged = _summarize(results, args.out_dir)
    print(
        f"\nCompleted {len(results)} outer folds. "
        f"Wrote {len(averaged)} averaged method/dataset/target rows to "
        f"{args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
