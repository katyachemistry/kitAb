"""Persist and load the fitted kitAb model for one dataset/target."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

ESTIMATOR_FILENAME = "estimator.joblib"
META_FILENAME = "meta.json"


def save_model(model_dir: Path, *, estimator: Any, meta: dict[str, Any]) -> Path:
    """Write estimator + metadata, renaming temp files so readers never see a partial dir."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload.setdefault("schema_version", 2)

    fd, tmp_est = tempfile.mkstemp(prefix=".estimator_", suffix=".joblib", dir=model_dir)
    os.close(fd)
    try:
        joblib.dump(estimator, tmp_est, compress=3)
        os.replace(tmp_est, model_dir / ESTIMATOR_FILENAME)
    finally:
        if os.path.exists(tmp_est):
            try:
                os.unlink(tmp_est)
            except OSError:
                pass

    fd, tmp_meta = tempfile.mkstemp(prefix=".meta_", suffix=".json", dir=model_dir)
    os.close(fd)
    try:
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_meta, model_dir / META_FILENAME)
    finally:
        if os.path.exists(tmp_meta):
            try:
                os.unlink(tmp_meta)
            except OSError:
                pass
    return model_dir


def load_model(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load the fitted estimator and its metadata."""
    model_dir = Path(model_dir)
    est_path = model_dir / ESTIMATOR_FILENAME
    meta_path = model_dir / META_FILENAME
    if not (est_path.is_file() and meta_path.is_file()):
        raise FileNotFoundError(
            f"No kitAb model under {model_dir} "
            f"(expected {ESTIMATOR_FILENAME} + {META_FILENAME})"
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(est_path), meta
