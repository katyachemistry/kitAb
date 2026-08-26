"""Smoke tests for the four-technique AutoML runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from automl.model_io import ESTIMATOR_FILENAME
from automl.run_automl import build_tasks
from automl.run_config import parse_dataset_records
from automl.techniques import apply_pipeline_cli_overrides, load_pipeline_settings


def _write_fast_automl_config(path: Path) -> None:
    path.write_text(
        """
pipeline:
  random_state: 42
  techniques: [elasticnet, intercorr_svm, sfs_svm, sfs_knn]
  cv:
    mode: flat
    n_splits: 3
  sfs:
    features_frac: 0.2
    min_improvement: 0.0
    inner_cv: 2
    eval_models: [linear, svm]
  elasticnet:
    alphas: [0.01, 0.1]
    l1_ratios: [0.5, 1.0]
  eval_hyperparameters:
    elasticnet:
      alpha: 0.01
  permutation_repeats: 0
  save_final_model: true
""",
        encoding="utf-8",
    )


def _write_run_config(path: Path, repo_root: Path, fixtures_dir: Path, batch_root: Path) -> None:
    csv_path = fixtures_dir / "csv" / "ab21_mini.csv"
    dev_path = fixtures_dir / "descriptors" / "ab21_mini_abb2_1" / "results"
    payload = {
        "batch_result_root": str(batch_root),
        "ab21_mini_abb2_1": {
            "path": str(csv_path),
            "developability_results_path": str(dev_path),
            "target_cols": "target_viscosity",
            "run_dir": str(batch_root / "folds"),
            "n_splits": 3,
            "random_state": 42,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_build_tasks_includes_all_four_techniques(
    repo_root: Path, tmp_path: Path, fixtures_dir: Path
):
    os.environ["PYTHONPATH"] = str(repo_root / "src")
    automl_cfg = tmp_path / "automl.yaml"
    _write_fast_automl_config(automl_cfg)
    settings = apply_pipeline_cli_overrides(
        load_pipeline_settings(automl_cfg),
        cv_mode="flat",
    )
    run_cfg = tmp_path / "run_config.yaml"
    batch_root = tmp_path / "batch"
    _write_run_config(run_cfg, repo_root, fixtures_dir, batch_root)

    from automl.run_automl import prepare_folds

    records = parse_dataset_records(
        yaml.safe_load(run_cfg.read_text()), default_n_splits=settings.cv.n_splits
    )
    prepare_folds(records, py_parts=[sys.executable], settings=settings, force=True)
    tasks, manifest = build_tasks(records, settings=settings, batch_root=batch_root)

    techniques = sorted({t["technique"].key for t in tasks})
    assert techniques == ["elasticnet", "intercorr_svm", "sfs_knn", "sfs_svm"]
    n_outer = int(manifest.iloc[0]["n_outer_folds"])
    assert len(tasks) == len(manifest) * n_outer


def test_four_technique_automl_writes_estimator(
    repo_root: Path, tmp_path: Path, fixtures_dir: Path
):
    os.environ["PYTHONPATH"] = str(repo_root / "src")
    automl_cfg = tmp_path / "automl.yaml"
    _write_fast_automl_config(automl_cfg)
    run_cfg = tmp_path / "run_config.yaml"
    batch_root = tmp_path / "batch"
    models_root = tmp_path / "models"
    _write_run_config(run_cfg, repo_root, fixtures_dir, batch_root)

    from automl import run_automl

    argv = [
        "run_automl.py",
        str(run_cfg),
        "--automl-config",
        str(automl_cfg),
        "--jobs",
        "2",
        "--cv-mode",
        "flat",
        "--models-root",
        str(models_root),
        "--py",
        sys.executable,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        run_automl.main()
    finally:
        sys.argv = old_argv

    estimators = list(models_root.glob(f"**/{ESTIMATOR_FILENAME}"))
    assert estimators, f"expected estimator.joblib under {models_root}"
    assert (batch_root / "technique_comparison.csv").is_file()
    assert (batch_root / "metrics" / "best_technique.csv").is_file()


def test_no_final_model_skips_estimator(
    repo_root: Path, tmp_path: Path, fixtures_dir: Path
):
    os.environ["PYTHONPATH"] = str(repo_root / "src")
    automl_cfg = tmp_path / "automl.yaml"
    _write_fast_automl_config(automl_cfg)
    run_cfg = tmp_path / "run_config.yaml"
    batch_root = tmp_path / "batch"
    models_root = tmp_path / "models_skip"
    _write_run_config(run_cfg, repo_root, fixtures_dir, batch_root)

    from automl import run_automl

    argv = [
        "run_automl.py",
        str(run_cfg),
        "--automl-config",
        str(automl_cfg),
        "--jobs",
        "2",
        "--cv-mode",
        "flat",
        "--no-final-model",
        "--models-root",
        str(models_root),
        "--py",
        sys.executable,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        run_automl.main()
    finally:
        sys.argv = old_argv

    assert list(models_root.glob(f"**/{ESTIMATOR_FILENAME}")) == []
    assert (batch_root / "technique_comparison.csv").is_file()
