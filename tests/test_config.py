from __future__ import annotations

from pathlib import Path

import pytest

from kitab.config import ConfigError, load_manifest, parse_manifest_dict


def test_validate_starter_manifest(tmp_path: Path, repo_root: Path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        f"""
inputs:
  datasets_dir: {repo_root / 'tests/fixtures/csv'}
run:
  output_dir: {tmp_path / 'out'}
structure_prediction:
  enabled: true
descriptors:
  enabled: true
automl:
  enabled: true
tuning:
  enabled: false
"""
    )
    m = load_manifest(cfg, repo_root=repo_root)
    assert m.mode == "predict"
    assert m.structure_prediction.enabled is True
    assert m.structure_prediction.model == ["abb2"]
    assert m.tuning.enabled is False
    assert "gpr" not in m.automl.eval_models or m.automl.eval_models == "all"
    assert m.stage_graph()[0] == "predict"


def test_validate_reports_multiple_errors(tmp_path: Path, repo_root: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        """
inputs:
  dataset_dir: missing
run: {}
structure_prediction:
  modell: abb9
"""
    )
    with pytest.raises(ConfigError) as excinfo:
        load_manifest(cfg, repo_root=repo_root)
    text = str(excinfo.value)
    assert "run.output_dir" in text
    assert "dataset_dir" in text or "unknown inputs" in text
    assert "modell" in text or "unknown structure_prediction" in text


def test_legacy_scenario1(repo_root: Path):
    m = load_manifest(repo_root / "configs/scenario1.yaml", repo_root=repo_root)
    assert m.legacy is True
    assert m.mode == "predict"
    assert m.tuning.enabled is False
    assert m.warnings


def test_legacy_scenario4(repo_root: Path):
    m = load_manifest(repo_root / "configs/scenario4.yaml", repo_root=repo_root)
    assert m.mode == "automl"
    assert m.stage_graph() == ["automl", "analysis", "tuning"]


def test_structure_models_accept_list_and_csv(repo_root: Path):
    base = {
        "inputs": {"datasets_dir": "tests/fixtures/csv"},
        "run": {"output_dir": "runs/x"},
        "structure_prediction": {"enabled": True, "model": ["abb2", "abb3"]},
        "automl": {"enabled": False},
    }
    listed = parse_manifest_dict(
        base, source_path=Path("t.yaml"), repo_root=repo_root
    )
    assert listed.structure_prediction.model == ["abb2", "abb3"]

    csv_raw = dict(base)
    csv_raw["structure_prediction"] = {"enabled": True, "model": "abb2, abb3"}
    csv_parsed = parse_manifest_dict(
        csv_raw, source_path=Path("t.yaml"), repo_root=repo_root
    )
    assert csv_parsed.structure_prediction.model == ["abb2", "abb3"]


def test_structure_models_reject_unknown(repo_root: Path):
    with pytest.raises(ConfigError, match="structure_prediction.model"):
        parse_manifest_dict(
            {
                "inputs": {"datasets_dir": "tests/fixtures/csv"},
                "run": {"output_dir": "runs/x"},
                "structure_prediction": {"enabled": True, "model": ["abb2", "abb9"]},
                "automl": {"enabled": False},
            },
            source_path=Path("t.yaml"),
            repo_root=repo_root,
        )


def test_structure_prediction_runs_ignored(repo_root: Path):
    m = parse_manifest_dict(
        {
            "inputs": {"datasets_dir": "tests/fixtures/csv"},
            "run": {"output_dir": "runs/x"},
            "structure_prediction": {"enabled": True, "runs": 3},
            "automl": {"enabled": False},
        },
        source_path=Path("t.yaml"),
        repo_root=repo_root,
    )
    assert not hasattr(m.structure_prediction, "runs")
    assert any("structure_prediction.runs" in w for w in m.warnings)


def test_allowed_suffixes_ignored(repo_root: Path):
    m = parse_manifest_dict(
        {
            "inputs": {
                "datasets_dir": "tests/fixtures/csv",
                "predefined_descriptors_dir": "tests/fixtures/descriptors",
                "allowed_suffixes": ["_abb2_1"],
            },
            "run": {"output_dir": "runs/x"},
            "automl": {"enabled": True},
        },
        source_path=Path("t.yaml"),
        repo_root=repo_root,
    )
    assert not hasattr(m.inputs, "allowed_suffixes")
    assert any("allowed_suffixes" in w for w in m.warnings)


def test_structures_layout_ignored(repo_root: Path):
    m = parse_manifest_dict(
        {
            "inputs": {"structures_dir": "tests/fixtures/structures"},
            "run": {"output_dir": "runs/x"},
            "descriptors": {"enabled": True, "structures_layout": "flat"},
            "automl": {"enabled": False},
        },
        source_path=Path("t.yaml"),
        repo_root=repo_root,
    )
    assert not hasattr(m.descriptors, "structures_layout")
    assert any("structures_layout" in w for w in m.warnings)


def test_dataset_stem_for_name():
    from utils.prepare_run_config import _dataset_stem_for_name

    stems = {"ab21", "pdgf38", "ab21_mini"}
    assert _dataset_stem_for_name("ab21", stems) == "ab21"
    assert _dataset_stem_for_name("ab21_abb2_1", stems) == "ab21"
    assert _dataset_stem_for_name("ab21_mini_abb2_1", stems) == "ab21_mini"
    assert _dataset_stem_for_name("ab210", stems) is None
    assert _dataset_stem_for_name("other", stems) is None


def test_eval_models_accept_list(repo_root: Path):
    m = parse_manifest_dict(
        {
            "inputs": {"datasets_dir": "tests/fixtures/csv"},
            "run": {"output_dir": "runs/x"},
            "structure_prediction": {"enabled": True},
            "descriptors": {"enabled": True},
            "automl": {"eval_models": ["linear", "svm"]},
        },
        source_path=Path("t.yaml"),
        repo_root=repo_root,
    )
    assert m.automl.eval_models == "linear,svm"


def test_gpr_rejected(repo_root: Path):
    with pytest.raises(ConfigError, match="gpr"):
        parse_manifest_dict(
            {
                "inputs": {"datasets_dir": "tests/fixtures/csv"},
                "run": {"output_dir": "runs/x"},
                "structure_prediction": {"enabled": True},
                "automl": {"eval_models": "gpr"},
            },
            source_path=Path("t.yaml"),
            repo_root=repo_root,
        )


def test_cli_override_enable_tuning(repo_root: Path, tmp_path: Path, fixtures_dir: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"""
inputs:
  datasets_dir: {fixtures_dir / 'csv'}
  structures_dir: {fixtures_dir / 'structures'}
run:
  output_dir: {tmp_path / 'out'}
descriptors:
  enabled: true
automl:
  enabled: true
tuning:
  enabled: false
"""
    )
    m = load_manifest(cfg, repo_root=repo_root, cli_overrides={"enable_tuning": True})
    assert m.tuning.enabled is True
    assert "tuning" in m.stage_graph()


def test_example_manifests_validate(repo_root: Path):
    for name in (
        "predict-and-automl.yaml",
        "existing-structures.yaml",
        "descriptors-only.yaml",
        "automl-only.yaml",
        "full-with-tuning.yaml",
    ):
        path = repo_root / "examples" / "configs" / name
        from kitab.config import load_yaml, is_legacy_config, parse_manifest_dict

        raw = load_yaml(path)
        assert not is_legacy_config(raw)
        parse_manifest_dict(raw, source_path=path, repo_root=repo_root)
