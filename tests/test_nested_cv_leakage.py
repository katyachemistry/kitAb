"""Nested-CV leakage: SFS scaling, group folds, categoricals, outer-test holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from automl.cv_engine import TechniqueRunError, select_features
from automl.feature_selectors import (
    minmax_scale_train_val,
    resolve_sfs_cv_folds,
    sequential_forward_selector,
)
from automl.folds import (
    encode_categoricals_train_only,
    fold_index_pairs_for_frame,
    write_inner_fold_dir,
)
from automl.techniques import build_technique, pipeline_settings_from_block


def _toy_sfs_frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "name": [f"m{i}" for i in range(n)],
            "target_y": y,
            "good": y + 0.01 * rng.normal(size=n),
            "noise": rng.normal(size=n),
            "huge_val_only": np.r_[np.ones(n - 3), np.array([1000.0, 2000.0, 3000.0])],
        }
    )


def test_minmax_scale_train_val_is_fold_local():
    x_train = np.array([[0.0], [1.0]])
    x_val = np.array([[10.0]])
    tr, va = minmax_scale_train_val(x_train, x_val)
    np.testing.assert_allclose(tr.ravel(), [0.0, 1.0])
    np.testing.assert_allclose(va.ravel(), [10.0])
    leaked_tr, leaked_va = minmax_scale_train_val(
        np.vstack([x_train, x_val]), x_val
    )
    assert leaked_tr[1, 0] == pytest.approx(0.1)
    assert va[0, 0] != pytest.approx(leaked_va[0, 0])


def test_sfs_scaler_fit_excludes_validation_rows(monkeypatch):
    seen_max: list[float] = []
    original_fit = MinMaxScaler.fit

    def _fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        seen_max.append(float(np.nanmax(arr)))
        return original_fit(self, X, y)

    monkeypatch.setattr(MinMaxScaler, "fit", _fit)
    df = _toy_sfs_frame()
    train_idx = np.arange(9)
    val_idx = np.arange(9, 12)
    sequential_forward_selector(
        df,
        "target_y",
        ["good", "noise", "huge_val_only"],
        n_features_to_select=1,
        cv=2,
        min_improvement=0.0,
        n_jobs=1,
        model_type="elasticnet",
        cv_folds=[(train_idx, val_idx)],
    )
    assert seen_max, "expected MinMaxScaler.fit during SFS"
    assert max(seen_max) < 100.0


def test_sfs_uses_predefined_folds_not_shuffle(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("row-shuffle folds must not be used when cv_folds is set")

    monkeypatch.setattr(
        "automl.feature_selectors.shuffled_row_folds", _boom
    )
    df = _toy_sfs_frame()
    folds = [
        (np.arange(0, 8), np.arange(8, 12)),
        (np.concatenate([np.arange(0, 4), np.arange(8, 12)]), np.arange(4, 8)),
    ]
    selected = sequential_forward_selector(
        df,
        "target_y",
        ["good", "noise"],
        n_features_to_select=1,
        cv=5,
        min_improvement=0.0,
        n_jobs=1,
        model_type="elasticnet",
        cv_folds=folds,
    )
    assert selected
    resolved = resolve_sfs_cv_folds(12, cv=5, random_state=0, cv_folds=folds)
    assert len(resolved) == 2
    np.testing.assert_array_equal(resolved[0][1], folds[0][1])


def _write_group_fold_dir(path: Path, frame: pd.DataFrame, groups: list[list[str]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    n = len(groups)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "target_col": "target_y",
                "feature_cols": ["good", "noise"],
                "n_splits": n,
            }
        )
    )
    name_to_i = {n: i for i, n in enumerate(frame["name"].astype(str))}
    for k, val_names in enumerate(groups):
        val_set = set(val_names)
        val_idx = [name_to_i[n] for n in val_names]
        train_idx = [i for i, n in enumerate(frame["name"].astype(str)) if n not in val_set]
        frame.iloc[train_idx].to_parquet(path / f"fold_{k}_train.parquet", index=False)
        frame.iloc[val_idx].to_parquet(path / f"fold_{k}_test.parquet", index=False)
    return path


def test_sfs_respects_predefined_group_folds_and_drops_outer_test(tmp_path: Path):
    df = _toy_sfs_frame(12)
    groups = [
        ["m0", "m1", "m2"],
        ["m3", "m4", "m5"],
        ["m6", "m7", "m8"],
        ["m9", "m10", "m11"],
    ]
    fold_dir = _write_group_fold_dir(tmp_path / "folds", df, groups)
    inner = write_inner_fold_dir(fold_dir, outer_k=0, dest=tmp_path / "inner")
    outer_test_names = {"m0", "m1", "m2"}
    inner_train = pd.read_parquet(inner / "fold_0_train.parquet")
    assert outer_test_names.isdisjoint(set(inner_train["name"].astype(str)))
    pairs = fold_index_pairs_for_frame(
        inner, inner_train.reset_index(drop=True), exclude_names=outer_test_names
    )
    assert len(pairs) >= 2
    names = inner_train["name"].astype(str).reset_index(drop=True)
    for train_idx, val_idx in pairs:
        used = set(names.iloc[np.concatenate([train_idx, val_idx])])
        assert used.isdisjoint(outer_test_names)
        assert len(set(train_idx) & set(val_idx)) == 0


def test_select_features_rejects_outer_test_in_train(tmp_path: Path):
    df = _toy_sfs_frame(12)
    groups = [
        ["m0", "m1", "m2"],
        ["m3", "m4", "m5"],
        ["m6", "m7", "m8"],
        ["m9", "m10", "m11"],
    ]
    fold_dir = _write_group_fold_dir(tmp_path / "folds", df, groups)
    settings = pipeline_settings_from_block(
        {"sfs": {"eval_models": ["svm"], "inner_cv": 2, "min_improvement": 0.0}}
    )
    technique = build_technique("sfs_svm", settings)
    train = pd.read_parquet(fold_dir / "fold_0_train.parquet")
    test = pd.read_parquet(fold_dir / "fold_0_test.parquet")
    leaked = pd.concat([train, test.iloc[:1]], ignore_index=True)
    with pytest.raises(TechniqueRunError, match="leaked"):
        select_features(
            leaked,
            test,
            target_col="target_y",
            candidate_features=["good", "noise"],
            technique=technique,
            settings=settings,
            fold_dir=fold_dir,
            exclude_names=set(test["name"].astype(str)),
        )


def test_encode_categoricals_train_only_drops_test_only_level():
    train = pd.DataFrame(
        {"name": ["a", "b"], "color": ["red", "blue"], "x": [0.1, 0.2]}
    )
    test = pd.DataFrame(
        {"name": ["c"], "color": ["green"], "x": [0.3]}
    )
    train_e, test_e, cols = encode_categoricals_train_only(
        train, test, ["color", "x"]
    )
    dummy_cols = [c for c in cols if c.startswith("color__")]
    assert "color__red" in dummy_cols
    assert "color__blue" in dummy_cols
    assert not any("green" in c for c in dummy_cols)
    assert "color" not in train_e.columns
    assert list(test_e[dummy_cols].sum(axis=1)) == [0.0]


def test_load_merge_keeps_categorical_until_train_split():
    from automl.prepare_run import one_hot_listed_non_numeric_features

    raw = pd.DataFrame({"color": ["red", "green"], "x": [1.0, 2.0]})
    expanded, cols = one_hot_listed_non_numeric_features(raw, ["color", "x"])
    assert any(c.startswith("color__") for c in cols)
    # Nested CV no longer calls this on the full table; train-only encoding
    # is the path used by select_features / ElasticNet.
    train_e, test_e, train_cols = encode_categoricals_train_only(
        raw.iloc[:1], raw.iloc[1:], ["color", "x"]
    )
    assert "color__red" in train_cols
    assert not any("green" in c for c in train_cols)
