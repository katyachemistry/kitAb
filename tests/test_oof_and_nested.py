"""Tests for pooled OOF reproduction and nested leftover-fold construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.nested_cv import NESTED_ELIGIBLE, write_inner_fold_dir
from analysis.oof_predictions import (
    pooled_metrics_from_oof,
    reproduce_oof_from_result_json,
)
from automl.run_fold_pipeline_config import (
    _evaluate_fold_models,
    _spearman_stat_p,
)
from automl.utils import apply_minmax_to_train_test_features


def test_is_our_source_accepts_isolated_sequence_baseline_label():
    from analysis.aggregated_csv import is_our_source

    assert is_our_source("descriptors_propermab_abb2_ginkgo_ig_folded_abb2_1_propermab")
    assert is_our_source(
        "propermab_sequence_baseline__descriptors_propermab_abb2_ginkgo_ig_folded_abb2_1_propermab"
    )
    assert is_our_source("descriptors_tap_ginkgo_ig_folded")
    assert is_our_source("our_abb2_final_set_of_features_descriptors_jain_abb2_1_results")
    assert not is_our_source("tap")
    assert not is_our_source("propermab_sequence_baseline")


def _tiny_fold_dir(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    n = 20
    names = [f"m{i}" for i in range(n)]
    y = np.arange(n, dtype=np.float64)
    x1 = y + rng.normal(0, 0.1, size=n)
    x2 = rng.normal(0, 1, size=n)
    df = pd.DataFrame({"name": names, "target_visc": y, "feat_a": x1, "feat_b": x2})
    train = df.iloc[:16].copy()
    test = df.iloc[16:].copy()
    fold_dir = tmp_path / "folds" / "target_visc"
    fold_dir.mkdir(parents=True)
    train.to_parquet(fold_dir / "fold_0_train.parquet", index=False)
    test.to_parquet(fold_dir / "fold_0_test.parquet", index=False)
    (fold_dir / "meta.json").write_text(
        json.dumps(
            {
                "target_col": "target_visc",
                "feature_cols": ["feat_a", "feat_b"],
                "n_splits": 5,
                "random_state": 42,
                "N": n,
            }
        )
    )
    return fold_dir, test


def test_evaluate_fold_models_writes_oof_rows(tmp_path: Path):
    fold_dir, test = _tiny_fold_dir(tmp_path)
    train = pd.read_parquet(fold_dir / "fold_0_train.parquet")
    test = pd.read_parquet(fold_dir / "fold_0_test.parquet")
    tr, te = apply_minmax_to_train_test_features(train, test, ["feat_a", "feat_b"])
    ev, oof = _evaluate_fold_models(
        tr,
        te,
        target_col="target_visc",
        feature_cols=["feat_a", "feat_b"],
        eval_models=["linear", "gpr"],
        random_state=42,
        features_frac=0.5,
    )
    assert "gpr" not in ev
    assert "linear" in ev
    assert len(oof) == len(test)
    assert set(oof["name"]) == set(test["name"])


def test_reproduce_oof_matches_stored_spearman(tmp_path: Path):
    fold_dir, _ = _tiny_fold_dir(tmp_path)
    train = pd.read_parquet(fold_dir / "fold_0_train.parquet")
    test = pd.read_parquet(fold_dir / "fold_0_test.parquet")
    tr, te = apply_minmax_to_train_test_features(train, test, ["feat_a", "feat_b"])
    ev, oof = _evaluate_fold_models(
        tr,
        te,
        target_col="target_visc",
        feature_cols=["feat_a", "feat_b"],
        eval_models=["linear"],
        random_state=42,
        features_frac=0.5,
    )
    jp = tmp_path / "result.json"
    payload = {
        "fold_dir": str(fold_dir),
        "fold_index": 0,
        "random_state": 42,
        "selector_name": "correlation",
        "model_type": "elasticnet",
        "eval_models": ["linear"],
        "eval_features_frac": 0.5,
        "target_col": "target_visc",
        "selected_features": ["feat_a", "feat_b"],
        "evaluation": ev,
        "dataset_stem": "toy",
        "dataset_yaml_key": "toy_abb2_1",
        "pipeline_track_name": "track_linear",
    }
    jp.write_text(json.dumps(payload))
    oof2, checks = reproduce_oof_from_result_json(jp)
    assert checks[0]["match"] is True
    assert len(oof2) == len(oof)
    pooled = pooled_metrics_from_oof(oof2)
    assert pooled["n_oof"] == len(test)
    sp, _ = _spearman_stat_p(
        oof2["y"].to_numpy(), oof2["yhat"].to_numpy()
    )
    assert sp is not None


def test_nested_post_grid_floating_sfs_writes_oof(tmp_path: Path):
    from analysis.nested_cv import run_nested_floating_sfs_job
    from automl.run_fold_pipeline_config import oof_sidecar_path

    fold_dir, test = _tiny_fold_dir(tmp_path)
    result_dir = tmp_path / "outer_0" / "inner_results"
    result_dir.mkdir(parents=True)
    for i, (model_type, feats) in enumerate(
        (("linear", ["feat_a"]), ("elasticnet", ["feat_a", "feat_b"]))
    ):
        (result_dir / f"inner0__grid__c{i}.json").write_text(
            json.dumps(
                {
                    "fold_index": 0,
                    "selector_name": "correlation",
                    "model_type": model_type,
                    "pipeline_track_name": "track_linear",
                    "selected_features": feats,
                }
            )
        )

    output = result_dir / "inner0__final_floating_sfs__elasticnet.json"
    run_nested_floating_sfs_job(
        inner_fold_dir=fold_dir,
        inner_k=0,
        dataset_stem="toy",
        target_col="target_visc",
        dataset_yaml_key="toy_abb2_1",
        selection_model="elasticnet",
        max_feature_fraction=0.5,
        track_name="track_linear",
        eval_models=["linear"],
        random_state=42,
        eval_hyperparameters={},
        output_json=output,
    )

    payload = json.loads(output.read_text())
    assert payload["selector_name"] == "final_floating_sfs"
    assert payload["final_floating_sfs_summary"]["vote_counts_by_feature"] == {
        "feat_a": 2,
        "feat_b": 1,
    }
    assert payload["selected_features"]
    oof = pd.read_parquet(oof_sidecar_path(output))
    assert len(oof) == len(test)
    assert set(oof["eval_model"]) == {"linear"}


def test_nested_outer_refit_supports_floating_sfs(tmp_path: Path):
    from analysis.nested_cv import refit_winner_on_outer
    from automl.run_fold_pipeline_config import oof_sidecar_path

    fold_dir, test = _tiny_fold_dir(tmp_path)
    dest = tmp_path / "outer_result.json"
    info = refit_winner_on_outer(
        fold_dir,
        0,
        {
            "pipeline_track_name": "track_linear",
            "selector_name": "final_floating_sfs",
            "model_type": "elasticnet",
            "eval_model": "linear",
            "eval_features_frac": 0.5,
            "eval_hyperparameters": {},
            "random_state": 42,
            "floating_vote_counts": {"feat_a": 2, "feat_b": 1},
            "Spearman_pooled_oof": 0.5,
        },
        dest_json=dest,
    )
    assert info["winner_run"].startswith(
        "target_visc-final_floating_sfs-elasticnet-linear"
    )
    assert len(pd.read_parquet(oof_sidecar_path(dest))) == len(test)


def test_inner_fold_excludes_outer_test(tmp_path: Path):
    fold_dir = tmp_path / "orig"
    fold_dir.mkdir()
    frames = []
    for k in range(4):
        names = [f"f{k}_{i}" for i in range(5)]
        tr_names = [f"f{j}_{i}" for j in range(4) if j != k for i in range(5)]
        test = pd.DataFrame(
            {
                "name": names,
                "target_visc": np.arange(5) + 10 * k,
                "feat_a": np.arange(5) + 0.1 * k,
            }
        )
        train = pd.DataFrame(
            {
                "name": tr_names,
                "target_visc": np.arange(len(tr_names)),
                "feat_a": np.arange(len(tr_names), dtype=float),
            }
        )
        train.to_parquet(fold_dir / f"fold_{k}_train.parquet", index=False)
        test.to_parquet(fold_dir / f"fold_{k}_test.parquet", index=False)
        frames.append(test)
    (fold_dir / "meta.json").write_text(
        json.dumps(
            {
                "target_col": "target_visc",
                "feature_cols": ["feat_a"],
                "n_splits": 4,
                "random_state": 42,
                "N": 20,
            }
        )
    )
    inner = write_inner_fold_dir(fold_dir, outer_k=0, dest=tmp_path / "inner")
    meta = json.loads((inner / "meta.json").read_text())
    assert meta["n_splits"] == 3
    outer_names = set(pd.read_parquet(fold_dir / "fold_0_test.parquet")["name"])
    for k in range(3):
        tr = pd.read_parquet(inner / f"fold_{k}_train.parquet")
        te = pd.read_parquet(inner / f"fold_{k}_test.parquet")
        assert outer_names.isdisjoint(set(tr["name"]))
        assert outer_names.isdisjoint(set(te["name"]))
        assert len(te) == 5


def test_write_folds_rejects_vanished_fold_after_nan_drop(tmp_path: Path):
    from automl.prepare_run import write_folds_for_target

    df = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "target_visc": [1.0, 2.0, 3.0, np.nan],
            "feat_a": [0.1, 0.2, 0.3, 0.4],
            "fold": [0, 0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="vanished|too small|imbalance"):
        write_folds_for_target(
            df,
            ["feat_a"],
            name_col="name",
            user_target_col="target_visc",
            all_target_cols=["target_visc"],
            output_dir=tmp_path / "folds",
            n_splits=5,
            random_state=42,
            features_frac=0.1,
            dataset_stem="toy",
            split_col="fold",
        )


def test_write_folds_allows_published_unbalanced_splits(tmp_path: Path):
    from automl.prepare_run import write_folds_for_target

    names = [f"m{i}" for i in range(10)]
    df = pd.DataFrame(
        {
            "name": names,
            "target_visc": np.arange(10, dtype=float),
            "feat_a": np.linspace(0.1, 1.0, 10),
            "fold": [0] * 7 + [1] * 3,
        }
    )
    kwargs = dict(
        expanded_feature_cols=["feat_a"],
        name_col="name",
        user_target_col="target_visc",
        all_target_cols=["target_visc"],
        n_splits=5,
        random_state=42,
        features_frac=0.1,
        dataset_stem="toy",
        split_col="fold",
    )
    with pytest.raises(ValueError, match="imbalance"):
        write_folds_for_target(df, output_dir=tmp_path / "reject", **kwargs)
    fold_root, job_lines = write_folds_for_target(
        df,
        output_dir=tmp_path / "allow",
        allow_unbalanced_splits=True,
        **kwargs,
    )
    assert (fold_root / "meta.json").is_file()
    assert len(job_lines) == 2


def test_sfs_stops_when_all_scores_are_nonfinite():
    from automl.feature_selectors import sequential_forward_selector

    n = 8
    df = pd.DataFrame(
        {
            "target_visc": np.ones(n),
            "feat_a": np.arange(n, dtype=float),
            "feat_b": np.arange(n, dtype=float)[::-1],
            "feat_c": np.linspace(0, 1, n),
        }
    )
    sel = sequential_forward_selector(
        df,
        "target_visc",
        ["feat_a", "feat_b", "feat_c"],
        n_features_to_select=3,
        cv=2,
        min_improvement=0.02,
        random_state=0,
        model_type="elasticnet",
    )
    assert 1 <= len(sel) <= 3
    assert len(sel) == 1


def test_nested_eligible_pairs():
    pairs = {(s, t) for s, t in NESTED_ELIGIBLE}
    assert ("ginkgo_ig_folded", "target_HAC") in pairs
    assert ("ab21", "target_viscosity") not in pairs
    assert ("ginkgo_ig_folded", "target_Tm1") in pairs
    assert len(NESTED_ELIGIBLE) == 24


def test_nested_yaml_key_suffix():
    from analysis.nested_cv import nested_yaml_key, resolve_nested_pairs

    assert nested_yaml_key("ginkgo_ig_folded", "abb2") == "ginkgo_ig_folded_abb2_1"
    assert (
        nested_yaml_key("ginkgo_ig_folded", "abb2", suffix="_propermab")
        == "ginkgo_ig_folded_abb2_1_propermab"
    )
    pairs = resolve_nested_pairs(
        Path("/unused"),
        include_stems=["ginkgo_ig_folded"],
        yaml_key_suffix="_propermab",
    )
    assert all(stem == "ginkgo_ig_folded" for stem, _ in pairs)
    assert ("ginkgo_ig_folded", "target_HAC") in pairs
    assert ("ginkgo_ig_folded", "target_Tm1") in pairs


def test_nested_yaml_key_suffix_goes_before_random_split_tag():
    from analysis.nested_cv import nested_yaml_key

    assert (
        nested_yaml_key(
            "hutchinson2023enhancement_top200tm1_igg",
            "abb2",
            suffix="_propermab",
        )
        == "hutchinson2023enhancement_top200tm1_igg_abb2_1_propermab__rs42"
    )
    assert nested_yaml_key("ginkgo_ig_folded", "abb2", mode="stem") == "ginkgo_ig_folded"
    assert (
        nested_yaml_key(
            "hutchinson2023enhancement_top200tm1_igg",
            "abb2",
            mode="stem",
        )
        == "hutchinson2023enhancement_top200tm1_igg__rs42"
    )


def test_nested_yaml_key_supports_structure_variants_2_and_3():
    from analysis.nested_cv import (
        all_backend_pairs_mode,
        backend_yaml_key_mode,
        discover_all_backend_pairs,
        nested_yaml_key,
    )

    assert backend_yaml_key_mode(2) == "backend_2"
    assert all_backend_pairs_mode(3) == "all_backend_3"
    assert nested_yaml_key("dataset_a", "abb3", mode="backend_2") == "dataset_a_abb3_2"
    assert (
        nested_yaml_key(
            "garbinski2023_tm1_folded_08_4",
            "abb2",
            suffix="_propermab",
            mode="backend_3",
        )
        == "garbinski2023_tm1_folded_08_4_abb2_3_propermab"
    )


def test_discover_all_backend_pairs_uses_structure_variant_mode(tmp_path: Path):
    from analysis.nested_cv import discover_all_backend_pairs, nested_yaml_key

    automl = tmp_path / "automl"
    for yaml_key, stem, target in (
        ("dataset_a_abb2_2", "dataset_a", "target_x"),
        ("dataset_a_abb2_1", "dataset_a", "target_y"),
        ("dataset_a_abb2_3", "dataset_a", "target_z"),
    ):
        path = automl / yaml_key / f"{target}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset_yaml_key": yaml_key,
                    "dataset_stem": stem,
                    "target_col": target,
                }
            )
        )

    assert nested_yaml_key("dataset_a", "abb2", mode="backend_2") == "dataset_a_abb2_2"
    assert discover_all_backend_pairs(
        automl,
        backend="abb2",
        yaml_key_mode="backend_2",
    ) == [("dataset_a", "target_x")]
    assert discover_all_backend_pairs(
        automl,
        backend="abb2",
        yaml_key_mode="backend_3",
    ) == [("dataset_a", "target_z")]


def test_resolve_nested_pairs_keeps_stem_mode_under_all_backend_1(tmp_path: Path):
    from analysis.nested_cv import resolve_nested_pairs

    automl = tmp_path / "automl"
    for yaml_key, stem, target in (
        ("ginkgo_ig_folded", "ginkgo_ig_folded", "target_Tm1"),
        ("jetha2019homology_RT__rs42", "jetha2019homology_RT", "target_HICRT"),
        ("ab21__rs42", "ab21", "target_viscosity"),
    ):
        path = automl / yaml_key / f"{target}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset_yaml_key": yaml_key,
                    "dataset_stem": stem,
                    "target_col": target,
                }
            )
        )

    pairs = resolve_nested_pairs(
        automl,
        backend="abb2",
        pairs_mode="all_backend_1",
        yaml_key_mode="stem",
        yaml_key_suffix="",
        exclude_stems={"ab21"},
    )
    assert pairs == [
        ("ginkgo_ig_folded", "target_Tm1"),
        ("jetha2019homology_RT", "target_HICRT"),
    ]


def test_remap_fold_dir_uses_longest_prefix():
    from analysis.nested_cv import remap_fold_dir

    mapping = {
        "/old/runs/ds": "/new/runs/ds",
        "/old/runs": "/should_not_win",
    }
    out = remap_fold_dir(Path("/old/runs/ds/target_Tm1"), mapping)
    assert out == Path("/new/runs/ds/target_Tm1")


def test_discover_nested_jobs_requires_meta_and_applies_fold_map(tmp_path: Path):
    from analysis.nested_cv import discover_nested_jobs

    yaml_key = "dataset_a_abb2_1"
    automl = tmp_path / "automl" / yaml_key
    automl.mkdir(parents=True)
    old_fold = tmp_path / "old_folds" / "target_x"
    new_fold = tmp_path / "new_folds" / "target_x"
    new_fold.mkdir(parents=True)
    (new_fold / "meta.json").write_text(json.dumps({"n_splits": 2}))
    (automl / "0.json").write_text(
        json.dumps(
            {
                "dataset_stem": "dataset_a",
                "target_col": "target_x",
                "dataset_yaml_key": yaml_key,
                "fold_dir": str(old_fold),
                "selector_name": "rfe",
                "model_type": "elasticnet",
                "eval_features_frac": 0.1,
                "pipeline_track_name": "track_linear",
                "eval_models": ["elasticnet"],
                "eval_hyperparameters": {},
            }
        )
    )
    jobs = discover_nested_jobs(
        tmp_path,
        tmp_path / "nested",
        automl_root=tmp_path / "automl",
        pairs=[("dataset_a", "target_x")],
        fold_dir_map={str(tmp_path / "old_folds"): str(tmp_path / "new_folds")},
    )
    assert jobs
    assert Path(jobs[0]["orig_fold_dir"]) == new_fold.resolve()


def test_nested_report_matches_empty_tap_variant(tmp_path: Path):
    from analysis.nested_cv import write_nested_report

    pair_root = tmp_path / "nested" / "ginkgo_ig_folded" / "target_Tm1"
    pair_root.mkdir(parents=True)
    (pair_root / "nested_summary.json").write_text(
        json.dumps({"Spearman_pooled_oof": 0.22, "n_oof": 10})
    )
    flat = tmp_path / "flat.csv"
    pd.DataFrame(
        [
            {
                "Dataset_stem": "ginkgo_ig_folded",
                "Target_col": "target_Tm1",
                "Variant": "",
                "Spearman": 0.41,
            }
        ]
    ).to_csv(flat, index=False)
    report = write_nested_report(
        tmp_path / "nested",
        automl_root=tmp_path / "automl",
        dest=tmp_path / "report.csv",
        pairs=[("ginkgo_ig_folded", "target_Tm1")],
        yaml_key_mode="stem",
        flat_results_csv=flat,
        flat_variant="",
    )
    assert len(report) == 1
    assert report.loc[0, "yaml_key"] == "ginkgo_ig_folded"
    assert report.loc[0, "flat_Spearman"] == 0.41


def test_prepare_nested_folds_descriptor_and_run_dirs():
    from analysis.prepare_nested_folds import (
        descriptor_dir,
        isolated_run_dir,
        iter_existing_tap_run_dirs,
        original_tap_run_dir,
    )

    repo = Path("/repo")
    assert descriptor_dir(
        repo, "tap", "ginkgo_ig_folded", "ginkgo_ig_folded"
    ) == repo / "descriptors_tap" / "ginkgo_ig_folded"
    assert original_tap_run_dir(
        repo, "ginkgo_ig_folded", "ginkgo_ig_folded"
    ) == repo / "runs" / "ginkgo_ig_folded_cv_prepare__descriptors_tap_ginkgo_ig_folded"
    assert original_tap_run_dir(repo, "ab21", "ab21__rs42") == (
        repo / "runs" / "ab21_cv_prepare__descriptors_tap_ab21__rs42"
    )
    assert iter_existing_tap_run_dirs(repo) == []
    yaml_key = "hutchinson2023enhancement_top200tm1_igg_abb2_1_propermab__rs42"
    assert descriptor_dir(
        repo,
        "propermab_sequence_baseline",
        "hutchinson2023enhancement_top200tm1_igg",
        yaml_key,
    ) == (
        repo
        / "descriptors_propermab_abb2"
        / "hutchinson2023enhancement_top200tm1_igg_abb2_1_propermab"
    )
    stem = "jain2017biophysical_folded_08_5"
    key = f"{stem}_abb3_1_propermab"
    assert isolated_run_dir(repo, "propermab_abb3", stem, key) == (
        repo / "runs" / f"{stem}_cv_prepare__nested_propermab_abb3__{key}"
    )


def test_iter_existing_tap_run_dirs(tmp_path: Path):
    from analysis.prepare_nested_folds import iter_existing_tap_run_dirs

    runs = tmp_path / "runs"
    tap = runs / "ab21_cv_prepare__descriptors_tap_ab21__rs42"
    other = runs / "ab21_cv_prepare__nested_tap__ab21__rs42"
    tap.mkdir(parents=True)
    other.mkdir(parents=True)
    found = iter_existing_tap_run_dirs(tmp_path)
    assert found == [("ab21", "ab21__rs42", tap)]


def test_iter_automl_yaml_jobs_and_fold_map_aliases(tmp_path: Path):
    from analysis.prepare_nested_folds import fold_map_aliases, iter_automl_yaml_jobs

    automl = tmp_path / "automl"
    d1 = automl / "jetha2019homology_RT_abb2_2_propermab__rs44"
    d1.mkdir(parents=True)
    (d1 / "a.json").write_text(
        json.dumps(
            {
                "dataset_stem": "jetha2019homology_RT",
                "target_col": "target_HICRT",
                "fold_dir": "/FASTAb/runs/jetha_cv_prepare/target_HICRT",
            }
        )
    )
    (d1 / "b.json").write_text(
        json.dumps(
            {
                "dataset_stem": "jetha2019homology_RT",
                "target_col": "target_ACRT",
            }
        )
    )
    jobs = iter_automl_yaml_jobs(automl)
    assert jobs == [
        (
            "jetha2019homology_RT",
            "jetha2019homology_RT_abb2_2_propermab__rs44",
            ["target_ACRT", "target_HICRT"],
        )
    ]
    aliases = fold_map_aliases(
        Path("/FASTAb/runs/old"),
        tmp_path / "runs" / "new",
    )
    assert aliases["/FASTAb/runs/old"] == str((tmp_path / "runs" / "new").resolve())
    assert aliases["/kitAb/runs/old"] == str((tmp_path / "runs" / "new").resolve())


def test_reproduce_oof_uses_fold_dir_map(tmp_path: Path):
    from analysis.oof_predictions import reproduce_oof_from_result_json

    old_root = tmp_path / "old_folds"
    fold_dir, _ = _tiny_fold_dir(tmp_path)
    new_root = fold_dir.parent
    payload = {
        "fold_dir": str(old_root / "target_visc"),
        "fold_index": 0,
        "random_state": 42,
        "selector_name": "correlation",
        "model_type": "elasticnet",
        "eval_models": ["linear"],
        "eval_features_frac": 0.5,
        "target_col": "target_visc",
        "selected_features": ["feat_a", "feat_b"],
        "evaluation": {"linear": {"spearman_rho": 1.0}},
        "dataset_stem": "toy",
        "dataset_yaml_key": "toy_abb2_1",
        "pipeline_track_name": "track_linear",
    }
    jp = tmp_path / "result.json"
    jp.write_text(json.dumps(payload))
    _, checks = reproduce_oof_from_result_json(
        jp, fold_dir_map={str(old_root): str(new_root)}
    )
    assert checks
    assert checks[0]["recomputed_spearman"] is not None


def test_discover_all_abb2_pairs_excludes_stems_and_replicates(tmp_path: Path):
    from analysis.nested_cv import discover_all_abb2_pairs

    automl = tmp_path / "automl"
    fixtures = [
        ("dataset_a_abb2_1", "dataset_a", "target_x"),
        ("dataset_a_abb2_1", "dataset_a", "target_y"),
        ("dataset_a_abb2_2", "dataset_a", "target_x"),
        ("ab21_abb2_1__rs42", "ab21", "target_viscosity"),
    ]
    for i, (yaml_key, stem, target) in enumerate(fixtures):
        path = automl / yaml_key / f"{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset_yaml_key": yaml_key,
                    "dataset_stem": stem,
                    "target_col": target,
                }
            )
        )

    pairs = discover_all_abb2_pairs(automl, exclude_stems={"ab21"})
    assert pairs == [("dataset_a", "target_x"), ("dataset_a", "target_y")]


def test_discover_all_backend_pairs_uses_requested_backend(tmp_path: Path):
    from analysis.nested_cv import discover_all_backend_pairs, nested_yaml_key

    automl = tmp_path / "automl"
    for yaml_key, stem, target in (
        ("dataset_a_abb3_1", "dataset_a", "target_x"),
        ("dataset_a_abb2_1", "dataset_a", "target_wrong_backend"),
        ("jetha2019homology_RT_abb3_1__rs42", "jetha2019homology_RT", "target_RT"),
    ):
        path = automl / yaml_key / f"{target}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset_yaml_key": yaml_key,
                    "dataset_stem": stem,
                    "target_col": target,
                }
            )
        )

    assert nested_yaml_key("dataset_a", "abb3") == "dataset_a_abb3_1"
    assert (
        nested_yaml_key("jetha2019homology_RT", "abb3")
        == "jetha2019homology_RT_abb3_1__rs42"
    )
    assert discover_all_backend_pairs(automl, backend="abb3") == [
        ("dataset_a", "target_x"),
        ("jetha2019homology_RT", "target_RT"),
    ]


def test_nested_report_uses_matching_abb2_1_flat_variant(tmp_path: Path):
    from analysis.nested_cv import write_nested_report

    pair_root = tmp_path / "nested" / "dataset_a_abb2_1" / "target_x"
    pair_root.mkdir(parents=True)
    (pair_root / "nested_summary.json").write_text(
        json.dumps(
            {
                "Spearman_pooled_oof": 0.3,
                "n_oof": 20,
                "n_unique_winners": 2,
                "winner_stability": 0.5,
            }
        )
    )
    flat = tmp_path / "flat.csv"
    pd.DataFrame(
        [
            {
                "Dataset_stem": "dataset_a",
                "Target_col": "target_x",
                "Variant": "abb2_1",
                "Spearman": 0.4,
            },
            {
                "Dataset_stem": "dataset_a",
                "Target_col": "target_x",
                "Variant": "abb2_3",
                "Spearman": 0.9,
            },
        ]
    ).to_csv(flat, index=False)

    report = write_nested_report(
        tmp_path / "nested",
        automl_root=tmp_path / "automl",
        dest=tmp_path / "report.csv",
        flat_results_csv=flat,
        pairs=[("dataset_a", "target_x")],
    )
    assert report.loc[0, "flat_Spearman"] == 0.4
    assert report.loc[0, "optimism_gap_flat_minus_nested"] == pytest.approx(0.1)


def test_nested_report_uses_matching_flashabb_1_flat_variant(tmp_path: Path):
    from analysis.nested_cv import write_nested_report

    pair_root = tmp_path / "nested" / "dataset_a_flashabb_1" / "target_x"
    pair_root.mkdir(parents=True)
    (pair_root / "nested_summary.json").write_text(
        json.dumps(
            {
                "Spearman_pooled_oof": 0.2,
                "n_oof": 20,
                "n_unique_winners": 1,
                "winner_stability": 1.0,
            }
        )
    )
    flat = tmp_path / "flat.csv"
    pd.DataFrame(
        [
            {
                "Dataset_stem": "dataset_a",
                "Target_col": "target_x",
                "Variant": "abb2_1",
                "Spearman": 0.9,
            },
            {
                "Dataset_stem": "dataset_a",
                "Target_col": "target_x",
                "Variant": "flashabb_1",
                "Spearman": 0.35,
            },
        ]
    ).to_csv(flat, index=False)

    report = write_nested_report(
        tmp_path / "nested",
        automl_root=tmp_path / "automl",
        dest=tmp_path / "report.csv",
        backend="flashabb",
        flat_results_csv=flat,
        pairs=[("dataset_a", "target_x")],
    )
    assert report.loc[0, "flat_Spearman"] == 0.35
    assert report.loc[0, "optimism_gap_flat_minus_nested"] == pytest.approx(0.15)


def test_nested_report_uses_custom_flat_variant_and_yaml_suffix(tmp_path: Path):
    from analysis.nested_cv import write_nested_report

    pair_root = (
        tmp_path / "nested" / "ginkgo_ig_folded_abb2_1_propermab" / "target_Tm1"
    )
    pair_root.mkdir(parents=True)
    (pair_root / "nested_summary.json").write_text(
        json.dumps(
            {
                "Spearman_pooled_oof": 0.25,
                "n_oof": 20,
                "n_unique_winners": 1,
                "winner_stability": 1.0,
            }
        )
    )
    flat = tmp_path / "flat.csv"
    pd.DataFrame(
        [
            {
                "Dataset_stem": "ginkgo_ig_folded",
                "Target_col": "target_Tm1",
                "Variant": "abb2_1",
                "Spearman": 0.9,
            },
            {
                "Dataset_stem": "ginkgo_ig_folded",
                "Target_col": "target_Tm1",
                "Variant": "abb2_1_propermab",
                "Spearman": 0.41,
            },
        ]
    ).to_csv(flat, index=False)

    report = write_nested_report(
        tmp_path / "nested",
        automl_root=tmp_path / "automl",
        dest=tmp_path / "report.csv",
        flat_results_csv=flat,
        pairs=[("ginkgo_ig_folded", "target_Tm1")],
        yaml_key_suffix="_propermab",
        flat_variant="abb2_1_propermab",
    )
    assert len(report) == 1
    assert report.loc[0, "yaml_key"] == "ginkgo_ig_folded_abb2_1_propermab"
    assert report.loc[0, "flat_Spearman"] == 0.41
    assert report.loc[0, "nested_Spearman_pooled_oof"] == 0.25


def test_winner_rank_prefers_pooled_spearman():
    from analysis.aggregated_csv import COL_SPEAR, COL_SPEAR_POOLED
    from analysis.analyze_results import _best_in_group

    rows = [
        {COL_SPEAR: "0.90", COL_SPEAR_POOLED: "0.10", "id": "mean_high"},
        {COL_SPEAR: "0.20", COL_SPEAR_POOLED: "0.80", "id": "pooled_high"},
    ]
    best = _best_in_group(rows, COL_SPEAR)
    assert best is not None
    assert best["id"] == "pooled_high"


def test_pearson_and_r2_reporting_prefer_pooled():
    from analysis.aggregated_csv import (
        COL_PEAR,
        COL_PEAR_POOLED,
        COL_R2,
        COL_R2_POOLED,
    )
    from analysis.analyze_results import _pearson_report_value, _r2_report_value

    row = {
        COL_PEAR: "0.90",
        COL_PEAR_POOLED: "0.40",
        COL_R2: "0.80",
        COL_R2_POOLED: "0.30",
    }
    assert _pearson_report_value(row) == 0.40
    assert _r2_report_value(row) == 0.30

    # Falls back to mean-of-folds when the pooled column is absent or blank.
    assert _pearson_report_value({COL_PEAR: "0.90", COL_PEAR_POOLED: ""}) == 0.90
    assert _r2_report_value({COL_R2: "0.80"}) == 0.80


def test_results_csv_reports_pooled_sample_counts(tmp_path: Path):
    from analysis.aggregated_csv import COL_N_FOLDS_PRESENT, COL_N_OOF
    from analysis.analyze_results import (
        COL_RESULT_N_FOLDS_PRESENT,
        COL_RESULT_N_OOF,
        build_results_per_target,
        write_results_csv,
    )

    summary = [
        {
            "Dataset_stem": "ds",
            "Developability_source": "descriptors_abb2",
            "Target_col": "target_x",
            "Track": "track_linear",
            "best_spearman": 0.5,
            "best_spearman_Target-Selector-Model": "target_x-correlation-elasticnet-frac015",
            "best_spearman_selector_model_frac": "correlation-elasticnet-frac015",
            "best_spearman_features": "",
            "spearman_winner_pearson": 0.4,
            "spearman_winner_r2": 0.3,
            COL_N_OOF: 21,
            COL_N_FOLDS_PRESENT: 5,
        }
    ]
    rows = build_results_per_target(summary)
    assert len(rows) == 1
    assert rows[0][COL_RESULT_N_OOF] == 21
    assert rows[0][COL_RESULT_N_FOLDS_PRESENT] == 5

    dest = tmp_path / "results.csv"
    write_results_csv(dest, rows)
    header, first = dest.read_text().splitlines()[:2]
    assert COL_N_OOF in header.split(",")
    assert COL_N_FOLDS_PRESENT in header.split(",")
    assert first.endswith("21,5")


def test_incomplete_fold_runs_are_not_rankable():
    from analysis.aggregated_csv import COL_N_FOLDS_PRESENT, COL_SPEAR_POOLED
    from analysis.analyze_results import (
        _expected_folds_by_dataset_target,
        _rows_with_complete_folds,
    )

    complete = {
        "Dataset_stem": "ds",
        "Target_col": "t",
        COL_SPEAR_POOLED: "0.26",
        COL_N_FOLDS_PRESENT: "5",
        "id": "complete",
    }
    partial = {
        "Dataset_stem": "ds",
        "Target_col": "t",
        COL_SPEAR_POOLED: "0.29",
        COL_N_FOLDS_PRESENT: "2",
        "id": "partial",
    }
    expected = _expected_folds_by_dataset_target([complete, partial])
    assert expected == {("ds", "t"): 5}

    kept, dropped = _rows_with_complete_folds([complete, partial], expected)
    assert [r["id"] for r in kept] == ["complete"]
    assert dropped == 1

    # A group of only partial runs still yields a winner rather than vanishing.
    kept, dropped = _rows_with_complete_folds([partial], expected)
    assert [r["id"] for r in kept] == ["partial"]
    assert dropped == 0

    # Aggregated CSVs without the pooled columns are unaffected.
    legacy = [{"Dataset_stem": "ds", "Target_col": "t", "id": "legacy"}]
    kept, dropped = _rows_with_complete_folds(
        legacy, _expected_folds_by_dataset_target(legacy)
    )
    assert [r["id"] for r in kept] == ["legacy"]
    assert dropped == 0


def test_pending_oof_jobs_skips_existing_parquets(tmp_path: Path):
    from analysis.oof_predictions import pending_oof_jobs, write_jobs_tsv

    done = tmp_path / "done.oof.parquet"
    done.write_bytes(b"parquet")
    missing = tmp_path / "missing.oof.parquet"
    jobs = [
        {
            "backend": "propermab_abb2",
            "json_path": "/a.json",
            "oof_path": str(done),
            "automl_root": "/automl",
        },
        {
            "backend": "propermab_flashabb",
            "json_path": "/b.json",
            "oof_path": str(missing),
            "automl_root": "/automl",
        },
    ]
    pending = pending_oof_jobs(jobs)
    assert [j["json_path"] for j in pending] == ["/b.json"]
    dest = tmp_path / "pending.tsv"
    write_jobs_tsv(dest, pending)
    lines = dest.read_text().splitlines()
    assert len(lines) == 2
    assert "b.json" in lines[1]


def test_validate_empty_oof_dir_fails(tmp_path: Path):
    from analysis.oof_predictions import validate_check_files

    summary = validate_check_files(tmp_path, max_mismatch_rate=0.0)
    assert summary["ok"] is False
    assert summary["n_eval_checks"] == 0


def _write_check(path: Path, checks: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"json_path": str(path), "checks": checks}))


def test_validate_tolerates_rf_noise_but_fails_real_disagreement(tmp_path: Path):
    from analysis.oof_predictions import validate_check_files

    _write_check(
        tmp_path / "abb2" / "ds" / "a.oof_check.json",
        [
            {"eval_model": "linear", "stored_spearman": 0.5, "recomputed_spearman": 0.5, "match": True},
            {"eval_model": "randomforest", "stored_spearman": 0.11330, "recomputed_spearman": 0.11027, "match": False},
        ],
    )
    ok = validate_check_files(tmp_path, max_mismatch_rate=0.0, tolerance=0.1)
    assert ok["ok"] is True
    assert ok["n_exact"] == 1
    assert ok["n_within_tolerance"] == 1
    assert ok["per_eval_model"]["randomforest"]["n_within_tolerance"] == 1
    assert ok["per_backend"]["abb2"]["n_eval_checks"] == 2

    _write_check(
        tmp_path / "abb2" / "ds" / "b.oof_check.json",
        [{"eval_model": "linear", "stored_spearman": 0.9, "recomputed_spearman": 0.1, "match": False}],
    )
    bad = validate_check_files(tmp_path, max_mismatch_rate=0.0, tolerance=0.1)
    assert bad["ok"] is False
    assert bad["n_beyond_tolerance"] == 1
    assert bad["examples_beyond_tolerance"][0]["eval_model"] == "linear"


def test_validate_tolerates_linear_near_tie_noise_at_default(tmp_path: Path):
    from analysis.oof_predictions import VALIDATE_NOISE_TOLERANCE, validate_check_files

    # Observed on jetha2019homology_RT HICRT fold 0: one discrete charge
    # feature with float jitter; sklearn 1.8 OLS reorders near-ties.
    _write_check(
        tmp_path / "abb2" / "ds" / "jetha_linear.oof_check.json",
        [
            {
                "eval_model": "linear",
                "stored_spearman": -0.17488231054342254,
                "recomputed_spearman": -0.04040240801649134,
                "match": False,
            },
            {
                "eval_model": "svm",
                "stored_spearman": -0.14425926362933106,
                "recomputed_spearman": -0.14425926362933106,
                "match": True,
            },
        ],
    )
    summary = validate_check_files(tmp_path, max_mismatch_rate=0.0)
    assert VALIDATE_NOISE_TOLERANCE >= 0.1544
    assert summary["ok"] is True
    assert summary["n_exact"] == 1
    assert summary["n_within_tolerance"] == 1
    assert summary["n_beyond_tolerance"] == 0
    assert summary["per_eval_model"]["linear"]["ok"] is True


def test_validate_counts_undefined_recomputed_separately(tmp_path: Path):
    from analysis.oof_predictions import validate_check_files

    _write_check(
        tmp_path / "abb2" / "ds" / "c.oof_check.json",
        [{"eval_model": "linear", "stored_spearman": -0.238, "recomputed_spearman": None, "match": False}],
    )
    s = validate_check_files(tmp_path, max_mismatch_rate=0.0, tolerance=0.1)
    assert s["n_recomputed_undefined"] == 1
    assert s["n_beyond_tolerance"] == 0
    assert s["ok"] is True


def test_randomforest_eval_is_deterministic(tmp_path: Path):
    fold_dir, _ = _tiny_fold_dir(tmp_path)
    train = pd.read_parquet(fold_dir / "fold_0_train.parquet")
    test = pd.read_parquet(fold_dir / "fold_0_test.parquet")
    tr, te = apply_minmax_to_train_test_features(train, test, ["feat_a", "feat_b"])
    seen = set()
    for _ in range(4):
        ev, _oof = _evaluate_fold_models(
            tr,
            te,
            target_col="target_visc",
            feature_cols=["feat_a", "feat_b"],
            eval_models=["randomforest"],
            random_state=42,
            features_frac=1.0,
            n_jobs=-1,
        )
        seen.add(ev["randomforest"]["spearman_rho"])
    assert len(seen) == 1


def test_selection_matched_takes_max(tmp_path: Path):
    from analysis.random_spearman_baseline import selection_matched_from_oof

    y = np.arange(20, dtype=float)
    a = tmp_path / "a.oof.parquet"
    b = tmp_path / "b.oof.parquet"
    pd.DataFrame(
        {
            "name": [f"n{i}" for i in range(20)],
            "y": y,
            "yhat": y,
            "eval_model": "linear",
            "dataset_stem": "toy",
            "dataset_yaml_key": "toy_abb2_1",
            "target_col": "target_visc",
            "selector_name": "correlation",
            "model_type": "elasticnet",
            "eval_features_frac": 0.1,
            "pipeline_track_name": "t1",
            "fold_index": 0,
        }
    ).to_parquet(a, index=False)
    pd.DataFrame(
        {
            "name": [f"n{i}" for i in range(20)],
            "y": y,
            "yhat": -y,
            "eval_model": "linear",
            "dataset_stem": "toy",
            "dataset_yaml_key": "toy_abb2_1",
            "target_col": "target_visc",
            "selector_name": "sfs",
            "model_type": "linear",
            "eval_features_frac": 0.1,
            "pipeline_track_name": "t1",
            "fold_index": 0,
        }
    ).to_parquet(b, index=False)
    rows = selection_matched_from_oof(tmp_path, seed=0)
    assert len(rows) == 1
    assert rows[0]["Spearman"] is not None


def test_nested_jobs_tsv_json_survives_tab_split(tmp_path: Path):
    import json
    from analysis.nested_cv import (
        pending_nested_jobs,
        write_nested_jobs_tsv,
        load_nested_jobs_tsv,
    )

    done = tmp_path / "done.json"
    done.write_text("{}")
    hp = {"rfe_step": 1, "intercorr_importance_metric": "pearson"}
    ev = {"elasticnet": {"enet_alpha": 0.01}, "knn": {"knn_weights": "distance"}}
    jobs = [
        {
            "stem": "ds",
            "target": "t",
            "yaml_key": "ds_abb2_1",
            "orig_fold_dir": "/orig",
            "inner_fold_dir": "/inner",
            "outer_k": "0",
            "inner_k": "0",
            "n_splits": "5",
            "selector_name": "rfe",
            "model_type": "elasticnet",
            "eval_features_frac": "0.15",
            "pipeline_track_name": "track_linear",
            "eval_models": "linear,elasticnet",
            "random_state": "42",
            "correlation_min_abs_rho": "none",
            "selector_hyperparameters": json.dumps(hp, separators=(",", ":")),
            "eval_hyperparameters": json.dumps(ev, separators=(",", ":")),
            "output_json": str(tmp_path / "pending.json"),
        },
        {
            "stem": "ds",
            "target": "t",
            "yaml_key": "ds_abb2_1",
            "orig_fold_dir": "/orig",
            "inner_fold_dir": "/inner",
            "outer_k": "0",
            "inner_k": "1",
            "n_splits": "5",
            "selector_name": "rfe",
            "model_type": "elasticnet",
            "eval_features_frac": "0.15",
            "pipeline_track_name": "track_linear",
            "eval_models": "linear,elasticnet",
            "random_state": "42",
            "correlation_min_abs_rho": "none",
            "selector_hyperparameters": json.dumps(hp, separators=(",", ":")),
            "eval_hyperparameters": json.dumps(ev, separators=(",", ":")),
            "output_json": str(done),
        },
    ]
    src = tmp_path / "jobs.tsv"
    write_nested_jobs_tsv(src, jobs)
    fields = src.read_text().splitlines()[1].split("\t")
    assert json.loads(fields[15]) == hp
    assert json.loads(fields[16]) == ev

    dest = tmp_path / "pending.tsv"
    write_nested_jobs_tsv(dest, pending_nested_jobs(load_nested_jobs_tsv(src)))
    pending_lines = dest.read_text().splitlines()
    assert len(pending_lines) == 2
    pfields = pending_lines[1].split("\t")
    assert json.loads(pfields[15]) == hp
    assert pfields[17].endswith("pending.json")


def test_nested_resume_matches_renamed_semantic_result(tmp_path: Path):
    from analysis.nested_cv import pending_nested_jobs
    from automl.run_fold_pipeline_config import oof_sidecar_path

    result_dir = tmp_path / "inner_results"
    result_dir.mkdir()
    old_path = result_dir / "inner0__rfe__elasticnet__frac015__c16.json"
    payload = {
        "dataset_yaml_key": "ds_abb2_1",
        "target_col": "target_x",
        "fold_index": 0,
        "selector_name": "rfe",
        "model_type": "elasticnet",
        "eval_features_frac": 0.15,
        "pipeline_track_name": "track_linear",
        "eval_models": ["linear", "elasticnet"],
        "random_state": 42,
        "correlation_min_abs_rho": None,
        "selector_hyperparameters": {"rfe_step": 1},
        "eval_hyperparameters": {"elasticnet": {"enet_alpha": 0.01}},
    }
    old_path.write_text(json.dumps(payload))
    pd.DataFrame({"y": [0.0], "yhat": [0.0]}).to_parquet(
        oof_sidecar_path(old_path), index=False
    )
    job = {
        "yaml_key": "ds_abb2_1",
        "target": "target_x",
        "inner_k": "0",
        "selector_name": "rfe",
        "model_type": "elasticnet",
        "eval_features_frac": "0.15",
        "pipeline_track_name": "track_linear",
        "eval_models": "linear,elasticnet",
        "random_state": "42",
        "correlation_min_abs_rho": "none",
        "selector_hyperparameters": '{"rfe_step":1}',
        "eval_hyperparameters": '{"elasticnet":{"enet_alpha":0.01}}',
        "output_json": str(result_dir / "renamed_semantic_hash.json"),
    }
    assert pending_nested_jobs([job], require_oof=True) == []


def test_floating_votes_ignore_duplicate_semantic_results(tmp_path: Path):
    from analysis.nested_cv import _floating_vote_map

    result_dir = tmp_path / "inner_results"
    result_dir.mkdir()
    payload = {
        "dataset_yaml_key": "ds_abb2_1",
        "target_col": "target_x",
        "fold_index": 0,
        "selector_name": "rfe",
        "model_type": "elasticnet",
        "eval_features_frac": 0.15,
        "pipeline_track_name": "track_linear",
        "eval_models": ["linear"],
        "random_state": 42,
        "correlation_min_abs_rho": None,
        "selector_hyperparameters": {"rfe_step": 1},
        "eval_hyperparameters": {},
        "selected_features": ["feat_a", "feat_b"],
    }
    (result_dir / "old_c16.json").write_text(json.dumps(payload))
    (result_dir / "new_h123.json").write_text(json.dumps(payload))

    assert _floating_vote_map(
        result_dir, track_name="track_linear", inner_k=0
    ) == {"feat_a": 1, "feat_b": 1}


def test_inner_winner_ignores_duplicate_semantic_oof(tmp_path: Path):
    from analysis.nested_cv import _inner_pooled_winners
    from automl.run_fold_pipeline_config import oof_sidecar_path

    outer_dir = tmp_path / "outer_0"
    result_dir = outer_dir / "inner_results"
    fold_dir = outer_dir / "inner_folds"
    result_dir.mkdir(parents=True)
    fold_dir.mkdir()
    (fold_dir / "meta.json").write_text(json.dumps({"n_splits": 1}))
    payload = {
        "dataset_yaml_key": "ds_abb2_1",
        "target_col": "target_x",
        "fold_index": 0,
        "selector_name": "rfe",
        "model_type": "elasticnet",
        "eval_features_frac": 0.15,
        "pipeline_track_name": "track_linear",
        "eval_models": ["linear"],
        "random_state": 42,
        "correlation_min_abs_rho": None,
        "selector_hyperparameters": {"rfe_step": 1},
        "eval_hyperparameters": {},
    }
    oof = pd.DataFrame(
        {
            "y": [0.0, 1.0],
            "yhat": [0.0, 1.0],
            "eval_model": ["linear", "linear"],
        }
    )
    for name in ("old_c16.json", "new_h123.json"):
        path = result_dir / name
        path.write_text(json.dumps(payload))
        oof.to_parquet(oof_sidecar_path(path), index=False)

    winner = _inner_pooled_winners(result_dir)
    assert winner["inner_oof_rows"] == 2
    assert winner["n_oof"] == 2
