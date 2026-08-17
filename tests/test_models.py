from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from kitab.models import (
    ModelSchemaError,
    enrich_model_meta,
    predict_with_tuned_model,
    validate_model_roundtrip,
    write_meta_with_checksum,
)
import pytest


def test_model_roundtrip(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    est = LinearRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    est.fit(X, y)
    joblib.dump(est, model_dir / "estimator.joblib", compress=3)
    meta = {
        "eval_model": "linear",
        "target_col": "target_viscosity",
        "feature_cols": ["f1"],
        "dataset_stem": "ab21_mini",
    }
    meta = enrich_model_meta(meta, model_dir=model_dir, manifest_checksum="abc")
    write_meta_with_checksum(model_dir, meta)
    validate_model_roundtrip(model_dir)
    pred = predict_with_tuned_model(model_dir, pd.DataFrame({"f1": [4.0]}))
    assert pred.shape == (1,)
    with pytest.raises(ModelSchemaError):
        predict_with_tuned_model(model_dir, pd.DataFrame({"other": [1.0]}))


def test_untuned_eval_hyperparameters_from_manifest():
    from automl.tune_eval_hyperparameters import _hp_from_manifest_block

    _raw, kwargs = _hp_from_manifest_block({}, "elasticnet")
    assert kwargs["enet_alpha"] == 0.01

    _raw, kwargs = _hp_from_manifest_block(
        {"eval_hyperparameters": {"elasticnet": {"alpha": 0.5}}},
        "elasticnet",
    )
    assert kwargs["enet_alpha"] == 0.5
