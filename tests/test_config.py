from __future__ import annotations

from pathlib import Path

import pytest

from kitab.config import ConfigError, SUPPORTED_TECHNIQUES, load_manifest, parse_manifest_dict


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
    assert m.automl.enabled is True
    assert m.automl.techniques == list(SUPPORTED_TECHNIQUES)
    assert m.automl.cv_mode == "nested"
    assert m.automl.save_final_model is True
    assert m.stage_graph()[-1] == "automl"
    assert any("tuning" in w for w in m.warnings)


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
    assert m.automl.enabled is True
    assert m.warnings


def test_legacy_scenario4(repo_root: Path):
    m = load_manifest(repo_root / "configs/scenario4.yaml", repo_root=repo_root)
    assert m.mode == "automl"
    assert m.stage_graph() == ["automl"]


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


def test_ignored_automl_yaml_keys_warn(repo_root: Path):
    m = parse_manifest_dict(
        {
            "inputs": {"datasets_dir": "tests/fixtures/csv"},
            "run": {"output_dir": "runs/x"},
            "structure_prediction": {"enabled": True},
            "descriptors": {"enabled": True},
            "automl": {
                "enabled": True,
                "eval_models": ["linear", "svm"],
                "techniques": ["elasticnet"],
                "cv_mode": "flat",
                "config_path": "src/automl.yaml",
            },
        },
        source_path=Path("t.yaml"),
        repo_root=repo_root,
    )
    assert m.automl.techniques == list(SUPPORTED_TECHNIQUES)
    assert m.automl.cv_mode == "nested"
    assert any("eval_models" in w for w in m.warnings)
    assert any("techniques" in w for w in m.warnings)


def test_cli_automl_overrides(repo_root: Path, tmp_path: Path, fixtures_dir: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"""
inputs:
  datasets_dir: {fixtures_dir / 'csv'}
  predefined_descriptors_dir: {fixtures_dir / 'descriptors'}
run:
  output_dir: {tmp_path / 'out'}
automl:
  enabled: true
"""
    )
    m = load_manifest(
        cfg,
        repo_root=repo_root,
        cli_overrides={
            "techniques": "elasticnet,sfs_svm",
            "cv_mode": "flat",
            "no_final_model": True,
        },
    )
    assert m.automl.techniques == ["elasticnet", "sfs_svm"]
    assert m.automl.cv_mode == "flat"
    assert m.automl.save_final_model is False


def test_cli_has_no_enable_tuning():
    import argparse

    from kitab.cli import _add_common_overrides

    parser = argparse.ArgumentParser()
    _add_common_overrides(parser)
    option_strings = {opt for a in parser._actions for opt in a.option_strings}
    assert "--enable-tuning" not in option_strings
    assert "--techniques" in option_strings
    assert "--cv-mode" in option_strings
    assert "--no-final-model" in option_strings


def test_example_manifests_validate(repo_root: Path):
    for name in (
        "predict-and-automl.yaml",
        "existing-structures.yaml",
        "descriptors-only.yaml",
        "automl-only.yaml",
        "reproduce-paper-abb2.yaml",
        "reproduce-paper-abb3.yaml",
        "reproduce-paper-flashabb.yaml",
    ):
        path = repo_root / "examples" / "configs" / name
        from kitab.config import load_yaml, is_legacy_config, parse_manifest_dict

        raw = load_yaml(path)
        assert not is_legacy_config(raw)
        parse_manifest_dict(raw, source_path=path, repo_root=repo_root)
