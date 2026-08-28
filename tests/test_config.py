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
"""
    )
    m = load_manifest(cfg, repo_root=repo_root)
    assert m.mode == "predict"
    assert m.structure_prediction.enabled is True
    assert m.structure_prediction.model == ["abb2"]
    assert m.structure_prediction.runs == 1
    assert m.automl.enabled is True
    assert m.automl.techniques == list(SUPPORTED_TECHNIQUES)
    assert m.automl.cv_mode == "nested"
    assert m.automl.technique_selection == "inner"
    assert m.automl.save_final_model is True
    assert m.stage_graph()[-1] == "automl"


def test_rejects_old_top_level_keys(tmp_path: Path, repo_root: Path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        f"""
inputs:
  datasets_dir: {repo_root / 'tests/fixtures/csv'}
run:
  output_dir: {tmp_path / 'out'}
automl:
  enabled: true
tuning:
  enabled: false
"""
    )
    with pytest.raises(ConfigError, match="tuning"):
        load_manifest(cfg, repo_root=repo_root)


def test_rejects_legacy_scenario_yaml(tmp_path: Path, repo_root: Path):
    cfg = tmp_path / "old.yaml"
    cfg.write_text(
        """
input_csvs_folder: tests/fixtures/csv
result_folder: runs/legacy_check
structure_prediction:
  model: abb2
"""
    )
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_manifest(cfg, repo_root=repo_root)


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


def test_canonical_scenario1(repo_root: Path):
    m = load_manifest(repo_root / "configs/scenario1.yaml", repo_root=repo_root)
    assert m.mode == "predict"
    assert m.automl.enabled is True
    assert m.structure_prediction.model == ["abb2"]
    assert m.structure_prediction.device == "cuda:1"


def test_canonical_scenario4(repo_root: Path):
    m = load_manifest(repo_root / "configs/scenario4.yaml", repo_root=repo_root)
    assert m.mode == "automl"
    assert m.stage_graph() == ["automl"]
    assert m.automl.enabled is True
    assert m.descriptors.enabled is False


def test_repo_yamls_validate(repo_root: Path):
    from kitab.config import load_yaml, parse_manifest_dict

    paths = [
        *(repo_root / "configs").glob("*.yaml"),
        *(repo_root / "examples" / "configs").glob("*.yaml"),
    ]
    assert paths
    for path in paths:
        raw = load_yaml(path)
        parse_manifest_dict(raw, source_path=path, repo_root=repo_root)


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


def test_structure_prediction_runs(repo_root: Path):
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
    assert m.structure_prediction.runs == 3

    with pytest.raises(ConfigError, match="structure_prediction.runs"):
        parse_manifest_dict(
            {
                "inputs": {"datasets_dir": "tests/fixtures/csv"},
                "run": {"output_dir": "runs/x"},
                "structure_prediction": {"enabled": True, "runs": 0},
                "automl": {"enabled": False},
            },
            source_path=Path("t.yaml"),
            repo_root=repo_root,
        )


def test_structure_processing_rejects_n_cpu(repo_root: Path):
    with pytest.raises(ConfigError, match="unknown structure_processing"):
        parse_manifest_dict(
            {
                "inputs": {"datasets_dir": "tests/fixtures/csv"},
                "run": {"output_dir": "runs/x", "n_cpu": 8},
                "structure_prediction": {"enabled": True},
                "structure_processing": {
                    "enabled": True,
                    "minimize": True,
                    "n_cpu": 16,
                },
                "automl": {"enabled": False},
            },
            source_path=Path("t.yaml"),
            repo_root=repo_root,
        )


def test_run_n_cpu_is_processing_pool(repo_root: Path):
    from kitab.stages import processing_n_cpu

    m = parse_manifest_dict(
        {
            "inputs": {"datasets_dir": "tests/fixtures/csv"},
            "run": {"output_dir": "runs/x", "n_cpu": 8},
            "structure_prediction": {"enabled": True},
            "structure_processing": {"enabled": True, "minimize": True},
            "automl": {"enabled": False},
        },
        source_path=Path("t.yaml"),
        repo_root=repo_root,
    )
    assert m.run.n_cpu == 8
    assert processing_n_cpu(m) == 8


def test_allowed_suffixes_rejected(repo_root: Path):
    with pytest.raises(ConfigError, match="allowed_suffixes"):
        parse_manifest_dict(
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


def test_structures_layout_rejected(repo_root: Path):
    with pytest.raises(ConfigError, match="structures_layout"):
        parse_manifest_dict(
            {
                "inputs": {"structures_dir": "tests/fixtures/structures"},
                "run": {"output_dir": "runs/x"},
                "descriptors": {"enabled": True, "structures_layout": "flat"},
                "automl": {"enabled": False},
            },
            source_path=Path("t.yaml"),
            repo_root=repo_root,
        )


def test_dataset_stem_for_name():
    from utils.prepare_run_config import _dataset_stem_for_name

    stems = {"ab21", "pdgf38", "ab21_mini"}
    assert _dataset_stem_for_name("ab21", stems) == "ab21"
    assert _dataset_stem_for_name("ab21_abb2_1", stems) == "ab21"
    assert _dataset_stem_for_name("ab21_mini_abb2_1", stems) == "ab21_mini"
    assert _dataset_stem_for_name("ab210", stems) is None
    assert _dataset_stem_for_name("other", stems) is None


def test_cli_owned_automl_yaml_keys_warn(repo_root: Path):
    m = parse_manifest_dict(
        {
            "inputs": {"datasets_dir": "tests/fixtures/csv"},
            "run": {"output_dir": "runs/x"},
            "structure_prediction": {"enabled": True},
            "descriptors": {"enabled": True},
            "automl": {
                "enabled": True,
                "techniques": ["elasticnet"],
                "cv_mode": "flat",
                "technique_selection": "outer",
            },
        },
        source_path=Path("t.yaml"),
        repo_root=repo_root,
    )
    assert m.automl.techniques == list(SUPPORTED_TECHNIQUES)
    assert m.automl.cv_mode == "nested"
    assert m.automl.technique_selection == "inner"
    assert any("techniques" in w for w in m.warnings)
    assert any("cv_mode" in w for w in m.warnings)
    assert any("technique_selection" in w for w in m.warnings)


def test_removed_automl_yaml_keys_rejected(repo_root: Path):
    with pytest.raises(ConfigError, match="eval_models|config_path"):
        parse_manifest_dict(
            {
                "inputs": {"datasets_dir": "tests/fixtures/csv"},
                "run": {"output_dir": "runs/x"},
                "structure_prediction": {"enabled": True},
                "descriptors": {"enabled": True},
                "automl": {
                    "enabled": True,
                    "eval_models": ["linear", "svm"],
                    "config_path": "src/automl.yaml",
                },
            },
            source_path=Path("t.yaml"),
            repo_root=repo_root,
        )


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
    assert m.automl.technique_selection == "inner"
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
        from kitab.config import load_yaml, parse_manifest_dict

        raw = load_yaml(path)
        parse_manifest_dict(raw, source_path=path, repo_root=repo_root)


def test_prepare_from_manifest_automl_only(
    tmp_path: Path, repo_root: Path, fixtures_dir: Path
):
    import yaml
    from kitab.logging_state import RunLogger
    from kitab.stages import prepare_internal_configs
    from utils.prepare_run_config import build_run_config, prepare_from_manifest

    out = tmp_path / "automl_out"
    cfg = tmp_path / "automl.yaml"
    cfg.write_text(
        f"""
inputs:
  datasets_dir: {fixtures_dir / 'csv'}
  predefined_descriptors_dir: {fixtures_dir / 'descriptors'}
run:
  output_dir: {out}
automl:
  enabled: true
"""
    )
    m = load_manifest(cfg, repo_root=repo_root)
    plan = prepare_from_manifest(m, resume=True)
    assert plan["uses_predefined_descriptors"] is True
    assert plan["calculate_descriptors"] is False
    assert plan["is_scenario3"] is False
    assert Path(plan["output_dir"]) == out.resolve()

    run_cfg = build_run_config(plan, repo_root)
    assert "batch_result_root" in run_cfg
    assert Path(run_cfg["batch_result_root"]) == (out / "automl").resolve()
    assert "ab21_mini_abb2_1" in run_cfg
    block = run_cfg["ab21_mini_abb2_1"]
    assert "path" in block
    assert "developability_results_path" in block
    assert "target_cols" in block
    assert "name_col" in block

    out.mkdir(parents=True, exist_ok=True)
    written = prepare_internal_configs(m, RunLogger(out))
    assert written == out / "internal" / "run_config.yaml"
    assert written.is_file()
    assert not (out / "internal" / "generic_config.yaml").exists()
    dumped = yaml.safe_load(written.read_text())
    assert dumped["batch_result_root"] == run_cfg["batch_result_root"]
    assert "ab21_mini_abb2_1" in dumped


def test_prepare_from_manifest_existing_structures(
    tmp_path: Path, repo_root: Path, fixtures_dir: Path
):
    from utils.prepare_run_config import build_run_config, prepare_from_manifest

    out = tmp_path / "struct_out"
    cfg = tmp_path / "struct.yaml"
    cfg.write_text(
        f"""
inputs:
  datasets_dir: {fixtures_dir / 'csv'}
  structures_dir: {fixtures_dir / 'structures'}
  split_randomly: [ab21_mini]
run:
  output_dir: {out}
descriptors:
  enabled: true
automl:
  enabled: true
"""
    )
    m = load_manifest(cfg, repo_root=repo_root)
    plan = prepare_from_manifest(m, resume=True)
    assert plan["uses_existing_structures"] is True
    assert plan["calculate_descriptors"] is True
    assert plan["is_scenario3"] is False

    run_cfg = build_run_config(plan, repo_root)
    assert Path(run_cfg["batch_result_root"]) == (out / "automl").resolve()
    assert "ab21_mini" in run_cfg
    block = run_cfg["ab21_mini"]
    assert block["path"].endswith("ab21_mini.csv")
    assert "structure_dir" in block
    assert Path(block["developability_results_path"]) == (
        out / "descriptors" / "ab21_mini" / "results"
    ).resolve()
    assert "n_splits" in block
    assert "random_seeds" in block


def test_prepare_from_manifest_prediction_runs(
    tmp_path: Path, repo_root: Path, fixtures_dir: Path
):
    from utils.prepare_run_config import build_run_config, prepare_from_manifest

    out = tmp_path / "pred_out"
    cfg = tmp_path / "pred.yaml"
    cfg.write_text(
        f"""
inputs:
  datasets_dir: {fixtures_dir / 'csv'}
  split_randomly: [ab21_mini]
run:
  output_dir: {out}
structure_prediction:
  enabled: true
  model: abb2
  runs: 3
descriptors:
  enabled: true
automl:
  enabled: true
"""
    )
    m = load_manifest(cfg, repo_root=repo_root)
    assert m.structure_prediction.runs == 3
    plan = prepare_from_manifest(m, resume=True)
    assert plan["structure_prediction"]["runs"] == 3
    run_cfg = build_run_config(plan, repo_root)
    assert "ab21_mini_abb2_1" in run_cfg
    assert "ab21_mini_abb2_2" in run_cfg
    assert "ab21_mini_abb2_3" in run_cfg
    assert "ab21_mini_abb2_4" not in run_cfg
    for i in (1, 2, 3):
        block = run_cfg[f"ab21_mini_abb2_{i}"]
        assert Path(block["structure_dir"]) == (
            out / "structures" / f"ab21_mini_abb2_{i}"
        ).resolve()

