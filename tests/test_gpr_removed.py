from __future__ import annotations

import pytest

from automl.pipeline_defaults import DEFAULT_EVAL_MODEL_ORDER, DEFAULT_EVAL_HYPERPARAMETERS_RAW
from automl.run_fold_pipeline_config import _parse_eval_models
from automl.utils import make_regressor


def test_all_excludes_gpr():
    models = _parse_eval_models("all")
    assert models is not None
    assert "gpr" not in models
    assert list(DEFAULT_EVAL_MODEL_ORDER) == models
    assert "gpr" not in DEFAULT_EVAL_HYPERPARAMETERS_RAW


def test_explicit_gpr_fails():
    with pytest.raises(ValueError, match="removed"):
        _parse_eval_models("gpr")
    with pytest.raises(ValueError, match="removed"):
        make_regressor("gpr")
