#!/usr/bin/env python3
"""Run the four kitAb AutoML techniques, pick the best one, fit and save it.

For every dataset/target in the run config this script

1. builds cross-validation folds (``prepare_run.py``),
2. evaluates ``elasticnet``, ``intercorr_svm``, ``sfs_svm`` and ``sfs_knn`` on
   every outer fold, in parallel across folds and targets,
3. ranks the techniques by Spearman correlation over the pooled out-of-fold
   predictions, and
4. refits the winner on the whole dataset and writes ``estimator.joblib``.

Outer folds are checkpointed, so ``--resume`` picks up where a run stopped.
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from automl.cv_engine import (  # noqa: E402
    eval_hyperparameters_by_model,
    run_outer_fold,
)
from automl.final_model import build_meta, fit_final_model, save_final_model  # noqa: E402
from automl.folds import (  # noqa: E402
    discover_target_fold_dirs,
    feature_columns,
    fold_parquets_ready,
    read_fold_meta,
)
from automl.run_config import (  # noqa: E402
    REPO_ROOT,
    DatasetRecord,
    RunConfigError,
    as_bool,
    automl_config_path_from,
    parse_dataset_records,
    resolve_path,
    slug,
)
from automl.select_best import TechniqueScore, score_techniques, select_best  # noqa: E402
from automl.techniques import (  # noqa: E402
    CV_MODES,
    PipelineSettings,
    Technique,
    apply_pipeline_cli_overrides,
    load_pipeline_settings,
)

DEFAULT_PY = "conda run --no-capture-output -n developability python"


# --------------------------------------------------------------------------
# Phase 1: fold preparation
# --------------------------------------------------------------------------


def _prepare_group_key(record: DatasetRecord) -> tuple:
    groups = tuple(
        sorted(
            (str(target), tuple(value or []))
            for target, value in record.developability_feature_groups_by_target.items()
        )
    )
    return (
        record.dataset_path,
        record.developability_paths,
        record.name_col,
        record.targets_csv,
        record.features_csv,
        groups,
        tuple(record.include_features),
        record.run_dir,
        -1 if record.split_col else record.n_splits,
        record.random_state,
        record.max_target_nan_frac,
        record.split_col or "",
    )


def prepare_folds(
    records: list[DatasetRecord],
    *,
    py_parts: list[str],
    settings: PipelineSettings,
    force: bool,
) -> None:
    """Run ``prepare_run.py`` once per distinct dataset/seed/target-group."""
    groups: dict[tuple, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        groups[_prepare_group_key(record)].append(record)

    for group in groups.values():
        head = group[0]
        head.run_dir.mkdir(parents=True, exist_ok=True)
        head.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        if not force and all(
            fold_parquets_ready(fold_dir)
            for fold_dir in discover_target_fold_dirs(head.run_dir).values()
        ) and discover_target_fold_dirs(head.run_dir):
            print(
                f"[automl] Folds already prepared: {head.run_dir}",
                file=sys.stderr,
                flush=True,
            )
            continue

        head.jobs_file.write_text("")
        target_buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for target in head.target_cols:
            bucket = tuple(
                head.developability_feature_groups_by_target.get(str(target), [])
            )
            target_buckets[bucket].append(str(target))

        for bucket_groups, bucket_targets in target_buckets.items():
            prepare_env = os.environ.copy()
            prepare_env["PYTHONPATH"] = str(REPO_ROOT / "src")
            cmd = [
                *py_parts,
                "-m",
                "automl.prepare_run",
                str(head.dataset_path),
                "--name-col",
                head.name_col,
                "--target-cols",
                *bucket_targets,
                "--feature-cols",
                *[p.strip() for p in head.features_csv.split(",") if p.strip()],
                "--developability-results",
                *[str(p) for p in head.developability_paths],
                "--output-dir",
                str(head.run_dir),
                "--n-splits",
                str(head.n_splits),
                "--random-state",
                str(head.random_state),
                "--features-frac",
                str(settings.sfs.features_frac),
                "--max-target-nan-frac",
                str(head.max_target_nan_frac),
                "--jobs-file",
                str(head.jobs_file),
            ]
            if bucket_groups:
                cmd.extend(["--developability-feature-groups", *bucket_groups])
            if head.include_features:
                cmd.extend(["--include-features", *head.include_features])
            if head.split_col:
                cmd.extend(["--split-col", head.split_col])

            group_slug = "all" if not bucket_groups else "_".join(bucket_groups)
            log_path = head.run_dir / f"prepare_run__{slug(group_slug)}.log"
            print(
                f"[automl] Preparing folds for {head.dataset_path.name} "
                f"({len(bucket_targets)} target(s)); log → {log_path}",
                file=sys.stderr,
                flush=True,
            )
            with open(log_path, "a") as log_file:
                completed = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT / "src",
                    stdout=log_file,
                    stderr=sys.stderr,
                    env=prepare_env,
                )
            if completed.returncode != 0:
                raise SystemExit(
                    f"prepare_run.py failed for {head.dataset_path.name} "
                    f"(see {log_path})"
                )


# --------------------------------------------------------------------------
# Phase 2: cross-validate every technique on every outer fold
# --------------------------------------------------------------------------


def _checkpoint_path(batch_root: Path, task: dict[str, Any]) -> Path:
    return (
        batch_root
        / "checkpoints"
        / slug(task["yaml_key"])
        / slug(task["target_col"])
        / task["technique"].key
        / f"outer_{task['outer_k']}.json"
    )


def _checkpoint_matches(result: dict[str, Any], task: dict[str, Any]) -> bool:
    expected = {
        "technique": task["technique"].key,
        "yaml_key": task["yaml_key"],
        "target_col": task["target_col"],
        "outer_fold": task["outer_k"],
        "fold_root": str(task["fold_dir"]),
        "n_input_features": len(task["features"]),
        "cv_mode": task["cv_mode"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return False
    oof_rows = result.get("oof_rows")
    if not isinstance(oof_rows, list) or not oof_rows:
        return False
    if int(result.get("n_test", -1)) != len(oof_rows):
        return False
    names = [str(row.get("name", "")) for row in oof_rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        return False
    for row in oof_rows:
        try:
            y, yhat = float(row["y"]), float(row["yhat"])
        except (KeyError, TypeError, ValueError):
            return False
        if not np.isfinite(y) or not np.isfinite(yhat):
            return False
    return True


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ckpt_", suffix=".json", dir=path.parent)
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(payload, default=str))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    settings: PipelineSettings = task["settings"]
    result = run_outer_fold(
        task["fold_dir"],
        target_col=task["target_col"],
        outer_k=task["outer_k"],
        features=task["features"],
        technique=task["technique"],
        settings=settings,
        work_root=task["work_root"],
        eval_hp_by_model=eval_hyperparameters_by_model(settings),
    )
    result.update(
        {
            "yaml_key": task["yaml_key"],
            "dataset_stem": task["dataset_stem"],
            "descriptor_source": task["descriptor_source"],
            "fold_root": str(task["fold_dir"]),
            "n_input_features": len(task["features"]),
        }
    )
    print(
        f"[automl] {task['yaml_key']} {task['target_col']} "
        f"{task['technique'].key} fold {task['outer_k']}: "
        f"rho={result['spearman']} r2={result['r2']} eval={result['eval_model']}",
        flush=True,
    )
    return result


def build_tasks(
    records: list[DatasetRecord],
    *,
    settings: PipelineSettings,
    batch_root: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    techniques = settings.build_techniques()
    tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for record in records:
        fold_dirs = discover_target_fold_dirs(record.run_dir)
        if not fold_dirs:
            raise RunConfigError(f"No prepared target folds under {record.run_dir}")
        for target_col, fold_dir in sorted(fold_dirs.items()):
            meta = read_fold_meta(fold_dir)
            n_outer = int(meta["n_splits"])
            features = feature_columns(fold_dir)
            for technique in techniques:
                manifest_rows.append(
                    {
                        "yaml_key": record.yaml_key,
                        "dataset_stem": record.dataset_stem,
                        "descriptor_source": record.descriptor_source(),
                        "target_col": target_col,
                        "technique": technique.key,
                        "technique_label": technique.label,
                        "cv_mode": settings.cv.mode,
                        "n_outer_folds": n_outer,
                        "n_input_features": len(features),
                        "fold_root": str(fold_dir),
                    }
                )
                for outer_k in range(n_outer):
                    tasks.append(
                        {
                            "yaml_key": record.yaml_key,
                            "dataset_stem": record.dataset_stem,
                            "descriptor_source": record.descriptor_source(),
                            "target_col": target_col,
                            "fold_dir": fold_dir,
                            "features": features,
                            "technique": technique,
                            "settings": settings,
                            "outer_k": outer_k,
                            "cv_mode": settings.cv.mode,
                            "work_root": (
                                batch_root
                                / "work"
                                / slug(record.yaml_key)
                                / technique.key
                            ),
                        }
                    )
    return tasks, pd.DataFrame(manifest_rows)


def run_tasks(
    tasks: list[dict[str, Any]], *, batch_root: Path, jobs: int, resume: bool
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        checkpoint = _checkpoint_path(batch_root, task)
        task["checkpoint"] = checkpoint
        cached = None
        if resume and checkpoint.is_file():
            try:
                candidate = json.loads(checkpoint.read_text())
            except (json.JSONDecodeError, OSError):
                candidate = None
            if candidate is not None and _checkpoint_matches(candidate, task):
                cached = candidate
        if cached is None:
            pending.append(task)
        else:
            results.append(cached)

    print(
        f"[automl] {len(results)} fold(s) reused from checkpoints, "
        f"{len(pending)} to run on up to {jobs} worker(s)",
        file=sys.stderr,
        flush=True,
    )
    if not pending:
        return results

    if jobs == 1:
        for task in pending:
            result = _run_task(task)
            _write_json_atomic(task["checkpoint"], result)
            results.append(result)
        return results

    with ProcessPoolExecutor(
        max_workers=min(jobs, len(pending)), mp_context=mp.get_context("spawn")
    ) as pool:
        futures = {pool.submit(_run_task, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            result = future.result()
            if not _checkpoint_matches(result, task):
                raise RuntimeError(
                    f"Invalid result for {task['checkpoint']}; refusing to write"
                )
            _write_json_atomic(task["checkpoint"], result)
            results.append(result)
            if index % 25 == 0 or index == len(futures):
                print(
                    f"[automl] completed {index}/{len(futures)} pending fold(s)",
                    file=sys.stderr,
                    flush=True,
                )
    return results


# --------------------------------------------------------------------------
# Phase 3: pick the winner and refit it on all rows
# --------------------------------------------------------------------------


def _fit_and_save(job: dict[str, Any]) -> dict[str, Any]:
    final = fit_final_model(
        job["fold_dir"],
        target_col=job["target_col"],
        features=job["features"],
        technique=job["technique"],
        settings=job["settings"],
    )
    meta = build_meta(
        final,
        score=job["score"],
        settings=job["settings"],
        dataset_stem=job["dataset_stem"],
        dataset_yaml_key=job["yaml_key"],
        descriptor_source=job["descriptor_source"],
        competing_scores=job["competing_scores"],
    )
    save_final_model(job["model_dir"], final, meta)
    print(
        f"[automl] saved {job['technique'].key} model for "
        f"{job['yaml_key']} {job['target_col']} "
        f"({len(final.feature_cols)} features, n={final.n_train}) → {job['model_dir']}",
        flush=True,
    )
    return {
        "yaml_key": job["yaml_key"],
        "dataset_stem": job["dataset_stem"],
        "target_col": job["target_col"],
        "technique": final.technique.key,
        "eval_model": final.eval_model,
        "alpha": final.alpha,
        "l1_ratio": final.l1_ratio,
        "n_features": len(final.feature_cols),
        "training_row_count": final.n_train,
        "cv_spearman_pooled_oof": job["score"].spearman_pooled_oof,
        "cv_r2_pooled_oof": job["score"].r2_pooled_oof,
        "model_dir": str(job["model_dir"]),
    }


def build_final_model_jobs(
    results: list[dict[str, Any]],
    *,
    settings: PipelineSettings,
    models_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score techniques per target and describe the refit job for each winner."""
    techniques: dict[str, Technique] = {t.key: t for t in settings.build_techniques()}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["yaml_key"], result["target_col"])].append(result)

    jobs: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    for (yaml_key, target_col), group in sorted(grouped.items()):
        scores = score_techniques(group, technique_order=list(settings.techniques))
        best = select_best(scores)
        head = group[0]
        for score in scores:
            comparison_rows.append(
                {
                    "yaml_key": yaml_key,
                    "dataset_stem": head["dataset_stem"],
                    "descriptor_source": head["descriptor_source"],
                    **score.as_row(),
                    "is_best": score.technique == best.technique,
                }
            )
        winner_rows.append(
            {
                "yaml_key": yaml_key,
                "dataset_stem": head["dataset_stem"],
                "descriptor_source": head["descriptor_source"],
                **best.as_row(),
            }
        )
        fold_dir = Path(head["fold_root"])
        jobs.append(
            {
                "yaml_key": yaml_key,
                "dataset_stem": head["dataset_stem"],
                "descriptor_source": head["descriptor_source"],
                "target_col": target_col,
                "fold_dir": fold_dir,
                "features": feature_columns(fold_dir),
                "technique": techniques[best.technique],
                "settings": settings,
                "score": best,
                "competing_scores": scores,
                "model_dir": models_root
                / slug(f"{head['dataset_stem']}__{head['descriptor_source']}")
                / slug(target_col),
            }
        )
    return jobs, comparison_rows, winner_rows


def run_final_models(jobs: list[dict[str, Any]], *, jobs_parallel: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not jobs:
        return pd.DataFrame(rows)
    if jobs_parallel == 1 or len(jobs) == 1:
        rows = [_fit_and_save(job) for job in jobs]
        return pd.DataFrame(rows)
    with ProcessPoolExecutor(
        max_workers=min(jobs_parallel, len(jobs)), mp_context=mp.get_context("spawn")
    ) as pool:
        futures = [pool.submit(_fit_and_save, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def write_artifacts(
    results: list[dict[str, Any]],
    *,
    batch_root: Path,
    comparison_rows: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
) -> None:
    predictions_dir = batch_root / "predictions"
    metrics_dir = batch_root / "metrics"
    features_dir = batch_root / "features"
    inner_dir = batch_root / "inner_selection"
    for path in (predictions_dir, metrics_dir, features_dir, inner_dir):
        path.mkdir(parents=True, exist_ok=True)

    identity_keys = ("yaml_key", "dataset_stem", "descriptor_source", "target_col")
    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    feature_sets: dict[str, dict[str, list[str]]] = {}

    for result in results:
        identity = {key: result[key] for key in identity_keys}
        for row in result.get("oof_rows", []):
            oof_rows.append({**identity, **row})
        fold_rows.append(
            {
                **identity,
                **{
                    key: result.get(key)
                    for key in (
                        "technique",
                        "technique_label",
                        "cv_mode",
                        "outer_fold",
                        "n_train",
                        "n_test",
                        "spearman",
                        "spearman_p",
                        "pearson_r",
                        "pearson_p",
                        "r2",
                        "mse",
                        "eval_model",
                        "alpha",
                        "l1_ratio",
                        "inner_pooled_spearman",
                        "n_selected_features",
                        "n_final_features",
                        "n_nonzero",
                        "n_input_features",
                        "n_inner_folds",
                        "selection_rule",
                        "fold_root",
                    )
                },
            }
        )
        fold_identity = {
            **identity,
            "technique": result["technique"],
            "outer_fold": result["outer_fold"],
        }
        for row in result.get("feature_usage", []):
            feature_rows.append({**fold_identity, **row})
        for row in result.get("inner_scores", []):
            inner_rows.append({**fold_identity, **row})
        for row in result.get("grid_scores", []):
            grid_rows.append(
                {
                    **fold_identity,
                    "alpha": row["alpha"],
                    "l1_ratio": row["l1_ratio"],
                    "pooled_inner_spearman": row.get("pooled_inner_spearman"),
                    "pooled_inner_r2": row.get("pooled_inner_r2"),
                }
            )
        set_key = "|".join(
            [
                str(result["yaml_key"]),
                str(result["target_col"]),
                str(result["technique"]),
            ]
        )
        feature_sets.setdefault(set_key, {})[f"fold_{int(result['outer_fold']) + 1}"] = (
            list(result.get("selected_features") or [])
        )

    pd.DataFrame(oof_rows).to_parquet(predictions_dir / "oof.parquet", index=False)
    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["yaml_key", "target_col", "technique", "outer_fold"]
    )
    fold_metrics.to_csv(metrics_dir / "outer_fold_metrics.csv", index=False)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(metrics_dir / "technique_comparison.csv", index=False)
    comparison.to_csv(batch_root / "technique_comparison.csv", index=False)
    pd.DataFrame(winner_rows).to_csv(metrics_dir / "best_technique.csv", index=False)

    if feature_rows:
        pd.DataFrame(feature_rows).to_parquet(
            features_dir / "feature_usage.parquet", index=False
        )
    (features_dir / "selected_features.json").write_text(
        json.dumps(feature_sets, indent=2)
    )
    if inner_rows:
        pd.DataFrame(inner_rows).to_parquet(
            inner_dir / "inner_scores.parquet", index=False
        )
    if grid_rows:
        pd.DataFrame(grid_rows).to_parquet(
            inner_dir / "elasticnet_grid.parquet", index=False
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Generated run config YAML.")
    parser.add_argument(
        "--automl-config",
        type=Path,
        default=None,
        help="Pipeline YAML (default: automl_config in the run config, else src/automl.yaml).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker processes (default: n_cpu from the run config, else all cores).",
    )
    parser.add_argument(
        "--py",
        default=None,
        help='Python invocation for fold preparation, e.g. "conda run -n developability python".',
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Where to write fitted models (default: <batch_result_root>/../models).",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse fold checkpoints.")
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Rebuild fold parquets even when they already exist.",
    )
    parser.add_argument(
        "--no-final-model",
        action="store_true",
        help="Cross-validate and compare techniques without fitting the final model.",
    )
    parser.add_argument(
        "--techniques",
        default=None,
        help="Comma-separated techniques to run (default: pipeline.techniques in automl.yaml).",
    )
    parser.add_argument(
        "--cv-mode",
        choices=CV_MODES,
        default=None,
        help="Cross-validation mode: nested or flat (default: from automl.yaml).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the work plan and exit."
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit("Run config root must be a mapping")

    settings = load_pipeline_settings(
        args.automl_config if args.automl_config else automl_config_path_from(raw)
    )
    settings = apply_pipeline_cli_overrides(
        settings,
        techniques=args.techniques,
        cv_mode=args.cv_mode,
        no_final_model=args.no_final_model,
    )

    batch_root_raw = raw.get("batch_result_root") or raw.get("batch-result-root")
    batch_root = (
        resolve_path(batch_root_raw)
        if batch_root_raw
        else (REPO_ROOT / "runs" / f"batch_{config_path.stem}").resolve()
    )
    batch_root.mkdir(parents=True, exist_ok=True)
    models_root = (
        args.models_root.resolve()
        if args.models_root
        else (batch_root.parent / "models").resolve()
    )

    jobs = args.jobs
    if jobs is None:
        raw_cpu = raw.get("n_cpu") or raw.get("n-cpu")
        try:
            jobs = int(raw_cpu) if raw_cpu else len(os.sched_getaffinity(0))
        except (TypeError, ValueError, AttributeError):
            jobs = os.cpu_count() or 4
    jobs = max(1, int(jobs))

    py_parts = shlex.split(args.py or DEFAULT_PY)
    records = parse_dataset_records(raw, default_n_splits=settings.cv.n_splits)
    force = args.force_preprocess or any(r.force_preprocess for r in records)

    print(
        f"[automl] {len(records)} dataset run(s); techniques="
        f"{','.join(settings.techniques)}; cv={settings.cv.mode}; jobs={jobs}",
        file=sys.stderr,
        flush=True,
    )

    prepare_folds(records, py_parts=py_parts, settings=settings, force=force)
    tasks, manifest = build_tasks(records, settings=settings, batch_root=batch_root)
    manifest.to_csv(batch_root / "manifest.csv", index=False)
    (batch_root / "run_config.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "batch_result_root": str(batch_root),
                "models_root": str(models_root),
                "pipeline": settings.to_dict(),
                "n_tasks": len(tasks),
            },
            indent=2,
        )
    )
    print(
        f"[automl] {len(manifest)} dataset/target/technique combination(s); "
        f"{len(tasks)} outer-fold task(s)",
        file=sys.stderr,
        flush=True,
    )
    if args.dry_run:
        return

    results = run_tasks(tasks, batch_root=batch_root, jobs=jobs, resume=args.resume)
    if len(results) != len(tasks):
        raise SystemExit(f"Expected {len(tasks)} fold results, got {len(results)}")

    final_jobs, comparison_rows, winner_rows = build_final_model_jobs(
        results, settings=settings, models_root=models_root
    )
    write_artifacts(
        results,
        batch_root=batch_root,
        comparison_rows=comparison_rows,
        winner_rows=winner_rows,
    )

    if args.no_final_model or not settings.save_final_model:
        print(
            "[automl] skipping the final full-dataset refit "
            "(--no-final-model or pipeline.save_final_model: false)",
            file=sys.stderr,
        )
    else:
        summary = run_final_models(final_jobs, jobs_parallel=jobs)
        models_root.mkdir(parents=True, exist_ok=True)
        summary.sort_values(["dataset_stem", "target_col"]).to_csv(
            models_root / "model_summary.csv", index=False
        )
        print(
            f"[automl] wrote {len(summary)} model(s) under {models_root}",
            file=sys.stderr,
        )

    print(
        f"[automl] done in {time.monotonic() - started:.1f}s; results in {batch_root}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
