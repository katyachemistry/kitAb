"""Tuned-model load/predict helpers with schema validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ModelSchemaError(ValueError):
    """Feature schema mismatch for a tuned kitAb model."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def enrich_model_meta(
    meta: dict[str, Any],
    *,
    model_dir: Path,
    manifest_checksum: str | None = None,
    package_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    out = dict(meta)
    est = model_dir / "estimator.joblib"
    meta_path = model_dir / "meta.json"
    out["schema_version"] = int(out.get("schema_version") or 1)
    if package_versions:
        out["package_versions"] = dict(package_versions)
    if manifest_checksum:
        out["resolved_manifest_checksum"] = manifest_checksum
    if est.is_file():
        out["estimator_sha256"] = sha256_file(est)
    # meta checksum is computed after writing without this key.
    out.pop("meta_sha256", None)
    return out


def write_meta_with_checksum(model_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(model_dir)
    meta_path = model_dir / "meta.json"
    payload = dict(meta)
    payload.pop("meta_sha256", None)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    meta_path.write_text(text, encoding="utf-8")
    payload["meta_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def load_tuned_model(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    from automl.tune_eval_hyperparameters import load_tuned_model as _load

    return _load(Path(model_dir))


def predict_with_tuned_model(
    model_dir: Path,
    features: pd.DataFrame | dict[str, Any],
) -> np.ndarray:
    estimator, meta = load_tuned_model(model_dir)
    cols = list(meta.get("feature_cols") or [])
    if not cols:
        raise ModelSchemaError(f"No feature_cols in meta.json under {model_dir}")
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ModelSchemaError(
            f"Missing required feature columns ({len(missing)}): {missing[:10]}"
        )
    X = df.loc[:, cols]
    if X.isna().any().any():
        raise ModelSchemaError("Feature matrix contains NaN values")
    try:
        X_np = X.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ModelSchemaError("Feature columns must be numeric") from exc
    return np.asarray(estimator.predict(X_np))


def validate_model_roundtrip(model_dir: Path) -> None:
    estimator, meta = load_tuned_model(model_dir)
    cols = list(meta.get("feature_cols") or [])
    if not cols:
        raise ModelSchemaError("empty feature_cols")
    # smoke predict on zeros
    df = pd.DataFrame({c: [0.0] for c in cols})
    _ = predict_with_tuned_model(model_dir, df)
    est_path = Path(model_dir) / "estimator.joblib"
    if not est_path.is_file():
        raise FileNotFoundError(est_path)
    _ = joblib.load(est_path)
