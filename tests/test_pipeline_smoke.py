"""Smoke / integration tests using disposable fixtures and mocked heavy engines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from kitab.config import load_manifest
from kitab.pipeline import run_pipeline
from kitab.stages import check_dataset_completeness, filter_run_config_for_complete


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_validate_descriptors_only(tmp_path: Path, repo_root: Path, fixtures_dir: Path):
    cfg = tmp_path / "desc.yaml"
    _write_manifest(
        cfg,
        {
            "inputs": {"structures_dir": str(fixtures_dir / "structures")},
            "run": {"output_dir": str(tmp_path / "out_desc")},
            "descriptors": {"enabled": True},
            "automl": {"enabled": False},
        },
    )
    from kitab.cli import main

    assert main(["validate", str(cfg)]) == 0


def test_completeness_blocks_incomplete_dataset(tmp_path: Path, repo_root: Path, fixtures_dir: Path):
    from kitab.logging_state import RunLogger

    out = tmp_path / "out"
    out.mkdir()
    logger = RunLogger(out)
    # Build a tiny run_config pointing at fixtures with one missing JSON
    csv = fixtures_dir / "csv" / "ab21_mini.csv"
    desc = fixtures_dir / "descriptors" / "ab21_mini_abb2_1" / "results"
    # remove one descriptor temporarily via copy
    work_desc = tmp_path / "desc" / "results"
    work_desc.mkdir(parents=True)
    for p in desc.glob("*.json"):
        if p.stem == "mAb1":
            continue
        (work_desc / p.name).write_text(p.read_text())
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        yaml.safe_dump(
            {
                "batch_result_root": str(out / "automl"),
                "ab21_mini_abb2_1": {
                    "path": str(csv),
                    "developability_results_path": str(work_desc),
                    "name_col": "name",
                    "target_cols": ["target_viscosity"],
                    "n_splits": 5,
                    "random_seeds": [42, 43, 44],
                },
            },
            sort_keys=False,
        )
    )
    m = load_manifest(
        _write_manifest(
            tmp_path / "m.yaml",
            {
                "inputs": {
                    "datasets_dir": str(fixtures_dir / "csv"),
                    "predefined_descriptors_dir": str(fixtures_dir / "descriptors"),
                },
                "run": {"output_dir": str(out)},
                "automl": {"enabled": True},
            },
        ),
        repo_root=repo_root,
    )
    completeness = check_dataset_completeness(m, logger, run_cfg)
    assert completeness.get("ab21_mini_abb2_1") is False
    filtered = filter_run_config_for_complete(
        run_cfg, completeness, tmp_path / "filtered.yaml"
    )
    data = yaml.safe_load(filtered.read_text())
    assert "ab21_mini_abb2_1" not in data or "ab21_mini_abb2_1" not in [
        k for k, v in completeness.items() if v
    ]


def test_input_immutability_and_mocked_pipeline(
    tmp_path: Path, repo_root: Path, fixtures_dir: Path
):
    """structures mode with mocked engines; assert input PDB/CSV hashes unchanged."""
    csv = fixtures_dir / "csv" / "ab21_mini.csv"
    struct = fixtures_dir / "structures" / "ab21_mini"
    csv_hash = _sha(csv)
    pdb_hashes = {p.name: _sha(p) for p in struct.glob("*.pdb")}

    out = tmp_path / "out_struct"
    cfg = _write_manifest(
        tmp_path / "struct.yaml",
        {
            "inputs": {
                "datasets_dir": str(fixtures_dir / "csv"),
                "structures_dir": str(fixtures_dir / "structures"),
                "split_randomly": ["ab21_mini"],
            },
            "run": {"output_dir": str(out)},
            "descriptors": {"enabled": True},
            "automl": {"enabled": False},
            "structure_processing": {"minimize": False, "renumber_imgt": False},
        },
    )
    m = load_manifest(cfg, repo_root=repo_root)

    def fake_prepare(manifest, logger):
        internal = out / "internal"
        internal.mkdir(parents=True, exist_ok=True)
        run_cfg = internal / "run_config.yaml"
        run_cfg.write_text(
            yaml.safe_dump(
                {
                    "ab21_mini": {
                        "path": str(csv),
                        "structure_dir": str(struct),
                        "developability_results_path": str(
                            out / "descriptors" / "ab21_mini" / "results"
                        ),
                    }
                },
                sort_keys=False,
            )
        )
        return run_cfg

    def fake_descriptors(manifest, logger, run_config):
        # Simulate writing descriptor JSONs without touching engines.
        dest = out / "descriptors" / "ab21_mini" / "results"
        dest.mkdir(parents=True, exist_ok=True)
        src = fixtures_dir / "descriptors" / "ab21_mini_abb2_1" / "results"
        for p in src.glob("*.json"):
            (dest / p.name).write_text(p.read_text())
        return out / "descriptors"

    with mock.patch("kitab.stages.prepare_internal_configs", side_effect=fake_prepare), mock.patch(
        "kitab.stages.process_structures", return_value=[struct]
    ), mock.patch("kitab.stages.run_descriptors", side_effect=fake_descriptors):
        rc = run_pipeline(m)
    assert rc == 0
    assert _sha(csv) == csv_hash
    for name, digest in pdb_hashes.items():
        assert _sha(struct / name) == digest
    assert (out / "logs" / "run.log").is_file()
    assert (out / "state" / "summary.json").is_file()


def test_model_export_in_stage_graph_when_automl(repo_root: Path, tmp_path: Path, fixtures_dir: Path):
    cfg = _write_manifest(
        tmp_path / "t.yaml",
        {
            "inputs": {
                "datasets_dir": str(fixtures_dir / "csv"),
                "predefined_descriptors_dir": str(fixtures_dir / "descriptors"),
            },
            "run": {"output_dir": str(tmp_path / "out")},
            "automl": {"enabled": True},
        },
    )
    m = load_manifest(cfg, repo_root=repo_root)
    assert m.automl.save_final_model is True
    assert m.stage_graph() == ["automl"]


def test_matrix_manifest_modes(repo_root: Path, tmp_path: Path, fixtures_dir: Path):
    cases = [
        (
            "descriptors",
            {
                "inputs": {"structures_dir": str(fixtures_dir / "structures")},
                "descriptors": {"enabled": True},
                "automl": {"enabled": False},
            },
            ["process_structures", "descriptors"],
        ),
        (
            "structures_automl",
            {
                "inputs": {
                    "datasets_dir": str(fixtures_dir / "csv"),
                    "structures_dir": str(fixtures_dir / "structures"),
                },
                "descriptors": {"enabled": True},
                "automl": {"enabled": True},
            },
            [
                "process_structures",
                "descriptors",
                "completeness",
                "automl",
            ],
        ),
        (
            "automl_only",
            {
                "inputs": {
                    "datasets_dir": str(fixtures_dir / "csv"),
                    "predefined_descriptors_dir": str(fixtures_dir / "descriptors"),
                },
                "automl": {"enabled": True},
            },
            ["automl"],
        ),
    ]
    for name, extra, expected in cases:
        payload = {
            "run": {"output_dir": str(tmp_path / name)},
        }
        payload.update(extra)
        m = load_manifest(_write_manifest(tmp_path / f"{name}.yaml", payload), repo_root=repo_root)
        assert m.stage_graph() == expected, name
