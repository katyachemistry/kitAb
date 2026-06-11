#!/usr/bin/env python3
"""Prepare folds per dataset, expand to a master TSV, run GNU parallel fold workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise SystemExit(1) from e

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from automl.feature_selectors import parse_selector_hyperparameters_mapping
from automl.pipeline_defaults import (
    DEFAULT_EVAL_MODELS,
    DEFAULT_FEATURES_FRAC_CSV,
    DEFAULT_RANDOM_STATE,
)
from automl.utils import parse_eval_hyperparameters_mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_SELECTOR_TYPES = frozenset({"stability", "correlation", "sfs", "rfe"})
_MODEL_HP_BLOCKS = frozenset({"elasticnet", "randomforest", "svm", "knn"})
_ALLOWED_FLOATING_SFS_MODELS = frozenset({"elasticnet", "randomforest", "svm", "knn"})
_PIPELINE_ROOT_KEYS = ("pipeline", "defaults", "shared", "global", "fit_settings")
_AUTOML_CONFIG_KEYS = ("automl_config", "automl-config", "automl")
_DEFAULT_AUTOML_CONFIG = _REPO_ROOT / "src" / "automl.yaml"

_PIPELINE_NON_TRACK_KEYS: frozenset[str] = frozenset({
    "random_state", "random-state", "random_seeds", "random-seeds",
    "features_frac", "features-frac", "features_fracs", "features-fracs",
    "eval_models", "eval-models",
    "eval_hyperparameters", "eval-hyperparameters",
    "eval_models_hyperparameters", "eval-models-hyperparameters",
    "hyperparameters", "hyperparameter",
    "low_variance_relative_std_threshold", "low-variance-relative-std-threshold",
    "low_var_threshold", "low-var-threshold",
    "low_variance_epsilon", "low-variance-epsilon",
    "intercorr_threshold", "intercorr-threshold",
    "intercorr_importance_metric", "intercorr-importance-metric",
    "intercorr_metric", "intercorr-metric",
    "intercorr_reduction_mode", "intercorr-reduction-mode",
    "intercorr_mode", "intercorr-mode",
    "correlation_reduction", "correlation-reduction",
    "stability_reduction", "stability-reduction",
})


def _parquet_train_row_count(train_pq: Path) -> int:
    p = Path(train_pq)
    if not p.is_file():
        raise FileNotFoundError(f"Missing fold train parquet: {p}")
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:
        import pandas as pd

        return int(len(pd.read_parquet(p)))


def _dedupe_features_fracs_across_folds(
    features_fracs: list[float],
    fold_to_n_train: dict[str, int],
) -> tuple[list[float], list[float]]:
    if not features_fracs:
        return ([], [])
    if not fold_to_n_train:
        uniq = sorted({float(x) for x in features_fracs}, reverse=True)
        return (uniq, [])
    fold_keys = sorted(fold_to_n_train.keys(), key=lambda kk: int(str(kk)))
    if any(int(fold_to_n_train[k]) < 1 for k in fold_keys):
        uniq = sorted({float(x) for x in features_fracs}, reverse=True)
        return (uniq, [])

    def _cap_signature(fv: float) -> tuple[int, ...]:
        return tuple(
            max(1, int(fv * int(fold_to_n_train[k])))
            for k in fold_keys
        )

    best_by_sig: dict[tuple[int, ...], float] = {}
    for f in features_fracs:
        fv = float(f)
        sig = _cap_signature(fv)
        old = best_by_sig.get(sig)
        if old is None or fv > old:
            best_by_sig[sig] = fv
    kept = sorted(best_by_sig.values(), reverse=True)
    kept_set = set(kept)
    dropped = sorted({float(x) for x in features_fracs if float(x) not in kept_set})
    return (kept, dropped)


def _parse_features_fracs(block: dict, yaml_key: str) -> list[float]:
    raw = _get(
        block,
        "features_frac",
        "features-frac",
        "features_fracs",
        "features-fracs",
        default=DEFAULT_FEATURES_FRAC_CSV,
    )
    if raw is None:
        raw = DEFAULT_FEATURES_FRAC_CSV
    out: list[float] = []
    if isinstance(raw, (list, tuple)):
        for x in raw:
            out.append(float(x))
    else:
        s = str(raw).strip()
        if not s:
            s = DEFAULT_FEATURES_FRAC_CSV
        for part in s.split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    if not out:
        raise ValueError(
            f"Dataset block {yaml_key!r}: features_frac must expand to at least one number"
        )
    seen: set[float] = set()
    uniq: list[float] = []
    for v in out:
        key = round(float(v), 12)
        if key not in seen:
            seen.add(key)
            uniq.append(float(v))
    return uniq


def _natural_selector_keys(block: dict) -> list[str]:
    keys = [k for k in block if isinstance(k, str) and k.startswith("selector")]
    return sorted(
        keys,
        key=lambda k: (
            int(m.group(1))
            if (m := re.search(r"(\d+)$", k))
            else 10_000,
            k,
        ),
    )


def _get(d: dict, *names: str, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def _phase1_cache_ok(jobs_file: Path) -> bool:
    if not jobs_file.is_file() or jobs_file.stat().st_size == 0:
        return False
    text = jobs_file.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            return False
        fold_dir_s, k_s, ds, tg = parts
        if not fold_dir_s or not k_s or not ds or not tg:
            return False
        fold_dir = Path(fold_dir_s)
        k = k_s
        if not fold_dir.is_dir():
            return False
        if not (fold_dir / "meta.json").is_file():
            return False
        if not (fold_dir / f"fold_{k}_train.parquet").is_file():
            return False
        if not (fold_dir / f"fold_{k}_test.parquet").is_file():
            return False
    return True


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _tab_tok(s: str) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").replace("\r", "")


def _slug(s: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_")
    return t or "x"


def _developability_run_dir_suffix(dev_paths: tuple[Path, ...]) -> str:
    rr = _REPO_ROOT.resolve()
    tokens: list[str] = []
    for p in dev_paths:
        rp = Path(p).resolve()
        try:
            tokens.append(rp.relative_to(rr).as_posix())
        except ValueError:
            tokens.append(str(rp))
    token = "__".join(tokens)
    s = _slug(token.replace("/", "_"))
    if not s:
        s = "dev"
    if len(s) > 96:
        h = hashlib.sha256(token.encode()).hexdigest()[:14]
        lead = Path(dev_paths[0]).name if dev_paths else "dev"
        s = _slug(f"{lead}_{h}")
    return s


def _strip_shared_correlation_reduction_for_tracked_pipeline(shared_base: dict) -> None:
    drop: list[str] = []
    for k in shared_base:
        if not isinstance(k, str):
            continue
        kl = k.strip().lower().replace("-", "_")
        if kl == "correlation_reduction" or kl.startswith("correlation_reduction_"):
            drop.append(k)
    for k in drop:
        shared_base.pop(k, None)


def _is_track_key(key: str, value) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    kl = str(key).strip().lower().replace("-", "_")
    if kl in _PIPELINE_NON_TRACK_KEYS:
        return False
    if kl.startswith("stability_reduction_") or kl.startswith("correlation_reduction_"):
        return False
    return any(str(k).startswith("selector") for k in value)


def _extract_tracks_from_pipeline(
    block: dict,
) -> list[tuple[str | None, dict]]:
    track_items = [(k, v) for k, v in block.items() if _is_track_key(k, v)]
    if not track_items:
        return [(None, block)]

    shared_base = {k: v for k, v in block.items() if not _is_track_key(k, v)}
    _strip_shared_correlation_reduction_for_tracked_pipeline(shared_base)

    result: list[tuple[str | None, dict]] = []
    for track_name, track_block in track_items:
        merged: dict = {**shared_base}
        for tk, tv in track_block.items():
            merged[tk] = tv
        result.append((str(track_name), merged))
    return result


def _merge_raw_selector_hyperparameters(spec: dict, model: str, stype: str) -> dict:
    raw = _get(spec, "hyperparameters", "hyperparameter")
    if raw is None:
        return {}
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            raw = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"hyperparameters must be a mapping or JSON object string: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError("hyperparameters must be a mapping (or JSON string of an object)")
    st = str(stype).strip().lower()
    if st == "correlation":
        merged: dict = {}
        for k, v in raw.items():
            kl = str(k).strip().lower()
            if isinstance(v, dict) and kl in _MODEL_HP_BLOCKS:
                continue
            merged[str(k)] = v
        return merged
    model_l = str(model).strip().lower()
    shared: dict = {}
    extra: dict = {}
    for k, v in raw.items():
        kl = str(k).strip().lower()
        if isinstance(v, dict) and kl in _MODEL_HP_BLOCKS:
            if kl == model_l:
                extra.update(v)
        else:
            shared[str(k)] = v
    return {**shared, **extra}


_STABILITY_REDUCTION_NESTED_HP: dict[str, str] = {
    "n_subsamples": "stability_reduction_n_subsamples",
    "stability_n_subsamples": "stability_reduction_n_subsamples",
    "sample_fraction": "stability_reduction_sample_fraction",
    "subsample_fraction": "stability_reduction_sample_fraction",
    "coef_threshold": "stability_reduction_coef_threshold",
    "l1_ratio": "stability_reduction_elasticnet_l1_ratio",
    "elasticnet_l1_ratio": "stability_reduction_elasticnet_l1_ratio",
    "alpha": "stability_reduction_elasticnet_alpha",
    "elasticnet_alpha": "stability_reduction_elasticnet_alpha",
    "svm_c": "stability_reduction_svm_c",
    "svm_C": "stability_reduction_svm_c",
    "svm_epsilon": "stability_reduction_svm_epsilon",
    "rf_n_estimators": "stability_reduction_rf_n_estimators",
    "n_estimators": "stability_reduction_rf_n_estimators",
    "rf_max_depth": "stability_reduction_rf_max_depth",
    "max_depth": "stability_reduction_rf_max_depth",
    "rf_min_samples_leaf": "stability_reduction_rf_min_samples_leaf",
    "min_samples_leaf": "stability_reduction_rf_min_samples_leaf",
    "rf_max_features": "stability_reduction_rf_max_features",
    "max_features": "stability_reduction_rf_max_features",
    "min_n_features": "stability_reduction_min_n_features",
}


def _correlation_reduction_raw_from_block(block: dict) -> dict:
    out: dict = {}
    cr = _get(block, "correlation_reduction", "correlation-reduction")
    if not isinstance(cr, dict) or not cr:
        return out
    mn = _get(cr, "min_n_features", "min-n-features")
    if mn is not None:
        out["correlation_reduction_min_n_features"] = mn
    return out


def _stability_reduction_raw_from_block(block: dict) -> dict:
    out: dict = {}
    sr = _get(block, "stability_reduction", "stability-reduction")
    if not isinstance(sr, dict) or not sr:
        return out
    nf = sr.get("n_features")
    if nf is None:
        nf = sr.get("n")
    if nf is None:
        nf = sr.get("features")
    if nf is not None:
        out["stability_reduction_n_features"] = nf
    md = sr.get("model")
    if md is not None and str(md).strip():
        out["stability_reduction_model"] = str(md).strip()
    pre = _get(sr, "prereduction", "pre-reduction", "pre_reduction")
    if isinstance(pre, dict) and pre:
        pnf = pre.get("n_features")
        if pnf is None:
            pnf = pre.get("n")
        if pnf is None:
            pnf = pre.get("features")
        if pnf is not None:
            out["stability_prereduction_n_features"] = pnf
    hp = _get(sr, "hyperparameters", "hyperparameter")
    if isinstance(hp, dict):
        for k, v in hp.items():
            kl = str(k).strip().lower().replace("-", "_")
            canon = _STABILITY_REDUCTION_NESTED_HP.get(kl)
            if canon is None:
                for ak, ck in _STABILITY_REDUCTION_NESTED_HP.items():
                    if ak.lower() == kl:
                        canon = ck
                        break
            if canon is not None and v is not None:
                out[canon] = v
    # Top-level subsampling / estimator knobs (override nested hyperparameters when both set).
    _sr_structural = frozenset(
        {
            "n_features",
            "n",
            "features",
            "model",
            "hyperparameters",
            "hyperparameter",
            "prereduction",
            "pre_reduction",
            "pre-reduction",
        }
    )

    def _sr_key_to_canon(k: str) -> str | None:
        kl = str(k).strip().lower().replace("-", "_")
        canon = _STABILITY_REDUCTION_NESTED_HP.get(kl)
        if canon is not None:
            return canon
        for ak, ck in _STABILITY_REDUCTION_NESTED_HP.items():
            if ak.lower() == kl:
                return ck
        return None

    for k, v in sr.items():
        if not isinstance(k, str) or k in _sr_structural or v is None:
            continue
        canon = _sr_key_to_canon(k)
        if canon is not None:
            out[canon] = v
    return out


def _shared_prefilter_raw_from_block(block: dict) -> dict:
    out: dict = {}
    v = _get(
        block,
        "low_variance_relative_std_threshold",
        "low-variance-relative-std-threshold",
        "low_var_threshold",
        "low-var-threshold",
    )
    if v is not None:
        out["low_variance_relative_std_threshold"] = v
    v = _get(block, "low_variance_epsilon", "low-variance-epsilon")
    if v is not None:
        out["low_variance_epsilon"] = v
    v = _get(block, "intercorr_threshold", "intercorr-threshold")
    if v is not None:
        out["intercorr_threshold"] = v
    v = _get(
        block,
        "intercorr_importance_metric",
        "intercorr-importance-metric",
        "intercorr_metric",
        "intercorr-metric",
    )
    if v is not None:
        out["intercorr_importance_metric"] = v
    v = _get(
        block,
        "intercorr_reduction_mode",
        "intercorr-reduction-mode",
        "intercorr_mode",
        "intercorr-mode",
    )
    if v is not None:
        out["intercorr_reduction_mode"] = v
    out.update(_correlation_reduction_raw_from_block(block))
    out.update(_stability_reduction_raw_from_block(block))
    # Top-level ``correlation_reduction_*`` / ``stability_reduction_*`` override nested values.
    for k, v in block.items():
        if not isinstance(k, str) or v is None:
            continue
        if (
            k.startswith("correlation_reduction_")
            or k.startswith("stability_reduction_")
            or k.startswith("stability_prereduction_")
        ):
            out[k] = v
    return out


def _correlation_min_abs_rho_from_spec(spec: dict, stype: str) -> float | None:
    if stype != "correlation":
        return None
    for key in (
        "threshold",
        "correlation_threshold",
        "min_abs_rho",
        "correlation_min_abs_rho",
    ):
        if key not in spec or spec[key] is None:
            continue
        s = str(spec[key]).strip()
        if s == "":
            continue
        return float(spec[key])
    return None


def _selector_job_specs(block: dict, dataset_yaml_key: str, track_name: str | None = None) -> list[dict]:
    sks = _natural_selector_keys(block)
    if not sks:
        raise ValueError(
            f"Dataset block {dataset_yaml_key!r}: no selector1, selector2, … entries found"
        )
    pipe_pref = _shared_prefilter_raw_from_block(block)
    jobs: list[dict] = []
    for sk in sks:
        spec = block[sk]
        if not isinstance(spec, dict):
            raise ValueError(f"{dataset_yaml_key}.{sk} must be a mapping")
        stype = str(spec.get("type", "")).strip().lower()
        if stype not in _ALLOWED_SELECTOR_TYPES:
            raise ValueError(
                f"{dataset_yaml_key}.{sk}: type must be one of {_ALLOWED_SELECTOR_TYPES}, got {stype!r}"
            )
        corr_min = _correlation_min_abs_rho_from_spec(spec, stype)
        raw_model = spec.get("model", "elasticnet")
        if raw_model is None or str(raw_model).strip() == "":
            models = ["elasticnet"]
        else:
            models = [
                m.strip().lower()
                for m in str(raw_model).split(",")
                if m.strip()
            ]
            if not models:
                models = ["elasticnet"]
        for m in models:
            merged_raw = {**pipe_pref, **_merge_raw_selector_hyperparameters(spec, m, stype)}
            hp = parse_selector_hyperparameters_mapping(stype, merged_raw)
            jobs.append(
                {
                    "selector": stype,
                    "model": m,
                    "correlation_min_abs_rho": corr_min,
                    "hyperparameters": hp,
                    "track_name": track_name,
                }
            )
    return jobs


def _parse_final_floating_sfs_from_block(block: dict, label: str) -> dict | None:
    raw = _get(block, "final_floating_sfs", "final-floating-sfs")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{label}.final_floating_sfs must be a mapping")

    frac_raw = _get(
        raw,
        "max_feature_fraction",
        "max-feature-fraction",
    )
    if frac_raw is None:
        raise ValueError(
            f"{label}.final_floating_sfs.max_feature_fraction is required"
        )
    try:
        max_feature_fraction = float(frac_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label}.final_floating_sfs.max_feature_fraction must be a float"
        ) from None
    if not (0.0 < max_feature_fraction <= 1.0):
        raise ValueError(
            f"{label}.final_floating_sfs.max_feature_fraction must be in (0, 1], "
            f"got {max_feature_fraction!r}"
        )

    raw_model = _get(raw, "model", default="elasticnet")
    if isinstance(raw_model, (list, tuple)):
        models = [str(m).strip().lower() for m in raw_model if str(m).strip()]
    else:
        models = [
            m.strip().lower() for m in str(raw_model).split(",") if str(m).strip()
        ]
    if not models:
        models = ["elasticnet"]
    bad = [m for m in models if m not in _ALLOWED_FLOATING_SFS_MODELS]
    if bad:
        opts = ", ".join(sorted(_ALLOWED_FLOATING_SFS_MODELS))
        raise ValueError(
            f"{label}.final_floating_sfs.model contains unsupported model(s) {bad!r}; "
            f"use: {opts}"
        )
    models = list(dict.fromkeys(models))
    return {
        "max_feature_fraction": max_feature_fraction,
        "models": models,
    }


def _pop_pipeline_block(raw: dict) -> dict | None:
    for key in _PIPELINE_ROOT_KEYS:
        if key not in raw or raw[key] is None:
            continue
        pl = raw.pop(key)
        if not isinstance(pl, dict):
            raise ValueError(f"YAML key {key!r} must be a mapping")
        return pl
    return None


def _pop_automl_config_path(raw: dict) -> str | None:
    for key in _AUTOML_CONFIG_KEYS:
        if key not in raw or raw[key] is None:
            continue
        text = str(raw.pop(key)).strip()
        if text:
            return text
    return None


def _default_automl_config_path() -> Path | None:
    if _DEFAULT_AUTOML_CONFIG.is_file():
        return _DEFAULT_AUTOML_CONFIG.resolve()
    return None


def _pipeline_block_from_automl_yaml(raw: object, *, source: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"AutoML config root must be a mapping: {source}")
    block = dict(raw)
    pipeline = _pop_pipeline_block(block)
    if pipeline is not None:
        return pipeline
    if block:
        return block
    raise ValueError(f"AutoML config is empty: {source}")


def _resolve_pipeline_config(
    run_raw: dict,
    *,
    automl_config_cli: Path | None = None,
) -> dict | None:
    inline_pipeline = _pop_pipeline_block(run_raw)

    automl_path: Path | None
    if automl_config_cli is not None:
        automl_path = (
            automl_config_cli.resolve()
            if automl_config_cli.is_absolute()
            else (_REPO_ROOT / automl_config_cli).resolve()
        )
    else:
        yaml_path_str = _pop_automl_config_path(run_raw)
        if yaml_path_str:
            automl_path = _resolve_path(yaml_path_str)
        else:
            automl_path = _default_automl_config_path()

    file_pipeline: dict | None = None
    if automl_path is not None:
        if not automl_path.is_file():
            print(f"AutoML config not found: {automl_path}", file=sys.stderr)
            sys.exit(1)
        automl_raw = yaml.safe_load(automl_path.read_text())
        try:
            file_pipeline = _pipeline_block_from_automl_yaml(
                automl_raw, source=automl_path
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(
            f"[batch] AutoML pipeline from {automl_path}",
            file=sys.stderr,
            flush=True,
        )

    if inline_pipeline is not None and file_pipeline is not None:
        print(
            "Warning: inline 'pipeline' block in the run config is ignored; "
            f"using AutoML settings from {automl_path}",
            file=sys.stderr,
        )
    elif inline_pipeline is not None:
        print(
            "[batch] AutoML pipeline from inline 'pipeline' block in run config (legacy)",
            file=sys.stderr,
            flush=True,
        )

    return file_pipeline if file_pipeline is not None else inline_pipeline


def _pipeline_settings_from_block(
    block: dict, label: str
) -> tuple[list[float], str, list[dict], dict[str, dict], dict | list | None]:
    features_fracs = _parse_features_fracs(block, label)
    eval_models = str(_get(block, "eval_models", "eval-models", default=DEFAULT_EVAL_MODELS) or DEFAULT_EVAL_MODELS)
    eval_hp_raw = _get(
        block,
        "eval_hyperparameters",
        "eval-hyperparameters",
        "eval_models_hyperparameters",
        "eval-models-hyperparameters",
    )
    if eval_hp_raw is None:
        eval_hp_raw = _get(block, "hyperparameters", "hyperparameter")
    try:
        eval_hyperparameters = parse_eval_hyperparameters_mapping(eval_hp_raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{label}: invalid eval hyperparameters: {e}") from e

    track_entries = _extract_tracks_from_pipeline(block)
    if len(track_entries) == 1 and track_entries[0][0] is None:
        selector_jobs = _selector_job_specs(block, label, track_name=None)
        final_floating_sfs: dict | list | None = _parse_final_floating_sfs_from_block(block, label)
    else:
        selector_jobs = []
        ffs_list: list[dict] = []
        for track_name, track_block in track_entries:
            track_label = f"{label}.{track_name}"
            selector_jobs.extend(
                _selector_job_specs(track_block, track_label, track_name=track_name)
            )
            ffs = _parse_final_floating_sfs_from_block(track_block, track_label)
            if ffs is not None:
                ffs_list.append({"track_name": track_name, **ffs})
        final_floating_sfs = ffs_list if ffs_list else None

    return (
        features_fracs,
        eval_models,
        selector_jobs,
        eval_hyperparameters,
        final_floating_sfs,
    )


def _coerce_random_seeds(raw: object, *, yaml_key: str) -> list[int]:
    if raw is None:
        return [DEFAULT_RANDOM_STATE]
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError(
                f"Dataset block {yaml_key!r}: random_state / random_seeds list is empty"
            )
        return [int(float(x)) for x in raw]
    s = str(raw).strip()
    if not s:
        return [DEFAULT_RANDOM_STATE]
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            raise ValueError(
                f"Dataset block {yaml_key!r}: empty random_state / random_seeds string"
            )
        return [int(float(p)) for p in parts]
    return [int(float(s))]


def _resolve_random_state_raw(block: dict, pipeline: dict | None) -> object:
    for src in (block, pipeline or {}):
        for names in (
            ("random_seeds", "random-seeds"),
            ("random_state", "random-state"),
        ):
            r = _get(src, *names)
            if r is not None:
                return r
    return DEFAULT_RANDOM_STATE


def _parse_dataset_records(root: dict, pipeline: dict | None) -> list[dict]:
    out: list[dict] = []
    shared: tuple[list[float], str, list[dict], dict[str, dict], dict | None] | None = None
    if pipeline is not None:
        if not pipeline:
            raise ValueError(
                "Top-level 'pipeline' (or 'defaults' / 'shared') block is empty; "
                "add features_frac, eval_models, and selector1, …"
            )
        shared = _pipeline_settings_from_block(pipeline, "pipeline")

    for yaml_key, block in root.items():
        if not isinstance(block, dict):
            continue
        has_dev_key = any(
            k in block
            for k in (
                "developability_results_path",
                "developability-results-path",
                "developability_results_paths",
                "developability-results-paths",
            )
        )
        if "path" not in block or not has_dev_key:
            continue
        dev_paths = _parse_developability_results_paths(block, yaml_key=yaml_key)

        dataset_path = _resolve_path(block["path"])
        dev_paths_resolved = tuple(_resolve_path(p) for p in dev_paths)
        name_col = str(_get(block, "name_col", default="name") or "name")
        targets_csv = str(_get(block, "target_cols") or "").strip()
        features_csv = str(_get(block, "feature_cols") or "").strip()
        if not targets_csv:
            raise ValueError(f"Dataset block {yaml_key!r}: target_cols is required")
        target_cols, _feature_cols_unused = _targets_features_as_lists(targets_csv, features_csv)
        if not target_cols:
            raise ValueError(f"Dataset block {yaml_key!r}: target_cols is required")
        dev_feat_groups_by_target = _parse_developability_feature_groups_by_target(
            block, target_cols, yaml_key=yaml_key
        )

        n_splits = int(_get(block, "n_splits", "n-splits", default=5) or 5)
        if shared is not None:
            (
                features_fracs,
                eval_models,
                selector_jobs,
                eval_hyperparameters,
                final_floating_sfs,
            ) = shared
        else:
            (
                features_fracs,
                eval_models,
                selector_jobs,
                eval_hyperparameters,
                final_floating_sfs,
            ) = (
                _pipeline_settings_from_block(block, yaml_key)
            )

        stem = dataset_path.stem
        run_dir_user = block.get("run_dir") or block.get("run-dir")

        force_preprocess = bool(
            _get(block, "force_preprocess", "force-preprocess", default=False)
        )
        skip_pre = _get(block, "skip_preprocessing", "skip-preprocessing")
        if skip_pre is False:
            force_preprocess = True

        mtn_raw = _get(block, "max_target_nan_frac", "max-target-nan-frac", default=0.7)
        try:
            max_target_nan_frac = float(mtn_raw if mtn_raw is not None else 0.7)
        except (TypeError, ValueError):
            raise ValueError(
                f"Dataset block {yaml_key!r}: max_target_nan_frac must be a float in (0, 1]"
            ) from None
        if not (0.0 < max_target_nan_frac <= 1.0):
            raise ValueError(
                f"Dataset block {yaml_key!r}: max_target_nan_frac must be in (0, 1], got {max_target_nan_frac!r}"
            )

        split_raw = _get(block, "split_col", "split-col")
        split_col: str | None
        if split_raw is None or str(split_raw).strip() == "":
            split_col = None
        else:
            split_col = str(split_raw).strip()

        raw_rs = _resolve_random_state_raw(block, pipeline)
        seeds = _coerce_random_seeds(raw_rs, yaml_key=yaml_key)
        if split_col and len(seeds) > 1:
            dropped = seeds[1:]
            seeds = [seeds[0]]
            print(
                f"Dataset block {yaml_key!r}: split_col is set — using random_state={seeds[0]} "
                f"only (ignoring extra pipeline/dataset seeds {dropped!r}).",
                file=sys.stderr,
            )

        for rs in seeds:
            eff_yaml_key = f"{yaml_key}__rs{rs}" if len(seeds) > 1 else yaml_key
            if run_dir_user:
                base_rd = _resolve_path(run_dir_user)
                if len(seeds) > 1:
                    run_dir_p = (base_rd.parent / f"{base_rd.name}__rs{rs}").resolve()
                else:
                    run_dir_p = base_rd.resolve()
            else:
                dev_tag = _developability_run_dir_suffix(dev_paths_resolved)
                rs_suffix = f"__rs{rs}" if len(seeds) > 1 else ""
                run_dir_p = (
                    _REPO_ROOT / "runs" / f"{stem}_cv_prepare__{dev_tag}{rs_suffix}"
                ).resolve()

            out.append(
                {
                    "yaml_key": eff_yaml_key,
                    "dataset_path": dataset_path,
                    "dataset_stem": stem,
                    "developability_paths": dev_paths_resolved,
                    "name_col": name_col,
                    "targets_csv": targets_csv,
                    "features_csv": features_csv,
                    "developability_feature_groups_by_target": dev_feat_groups_by_target,
                    "run_dir": run_dir_p,
                    "jobs_file": run_dir_p / "parallel_jobs.txt",
                    "n_splits": n_splits,
                    "random_state": rs,
                    "features_fracs": features_fracs,
                    "eval_models": eval_models,
                    "eval_hyperparameters": eval_hyperparameters,
                    "selector_jobs": selector_jobs,
                    "final_floating_sfs": final_floating_sfs,
                    "force_preprocess": force_preprocess,
                    "max_target_nan_frac": max_target_nan_frac,
                    "split_col": split_col,
                    "yaml_block_key": yaml_key,
                }
            )
    if not out:
        raise ValueError("No dataset blocks found (need path + developability_results_path(s))")
    return out


def _targets_features_as_lists(targets_csv: str, features_csv: str) -> tuple[list[str], list[str]]:
    return (
        [p.strip() for p in targets_csv.split(",") if p.strip()],
        [p.strip() for p in features_csv.split(",") if p.strip()],
    )


def _parse_developability_results_paths(block: dict, *, yaml_key: str) -> list[str]:
    single = _get(block, "developability_results_path", "developability-results-path")
    multi = _get(block, "developability_results_paths", "developability-results-paths")

    if single is None and multi is None:
        raise ValueError(
            f"Dataset block {yaml_key!r}: provide developability_results_path or developability_results_paths."
        )
    if single is not None and multi is not None:
        raise ValueError(
            f"Dataset block {yaml_key!r}: use only one of developability_results_path or "
            "developability_results_paths."
        )

    raw = multi if multi is not None else single
    if isinstance(raw, (list, tuple)):
        out = [str(x).strip() for x in raw if str(x).strip()]
    else:
        s = str(raw).strip()
        out = [p.strip() for p in s.split(",") if p.strip()] if s else []
    if not out:
        raise ValueError(
            f"Dataset block {yaml_key!r}: developability results path list is empty."
        )
    return out


def _parse_developability_feature_groups(block: dict) -> list[str]:
    raw = _get(
        block,
        "developability_features",
        "developability-features",
        "developability_feature_groups",
        "developability-feature-groups",
    )
    return _parse_developability_groups_value(raw)


def _parse_developability_feature_groups_by_target(
    block: dict,
    target_cols: list[str],
    *,
    yaml_key: str,
) -> dict[str, list[str]]:
    dev_keys = (
        "developability_features",
        "developability-features",
        "developability_feature_groups",
        "developability-feature-groups",
    )
    present = [(k, block.get(k)) for k in dev_keys if k in block and block.get(k) is not None]
    if len(present) > 1:
        has_mapping = any(isinstance(v, dict) for _, v in present)
        has_shared = any(not isinstance(v, dict) for _, v in present)
        if has_mapping and has_shared:
            keys = [k for k, _ in present]
            raise ValueError(
                f"Dataset block {yaml_key!r}: developability features are defined both as "
                f"a shared value and as per-target mapping ({keys!r}). Use only one form."
            )

    raw = _get(block, *dev_keys)
    if raw is None:
        return {str(t): [] for t in target_cols}

    if not isinstance(raw, dict):
        groups = _parse_developability_feature_groups(block)
        return {str(t): list(groups) for t in target_cols}

    default_groups = _parse_developability_groups_value(raw.get("default", raw.get("*")))
    out: dict[str, list[str]] = {}
    unknown_keys = sorted(
        k for k in raw.keys() if str(k) not in set(target_cols) | {"default", "*"}
    )
    if unknown_keys:
        raise ValueError(
            f"Dataset block {yaml_key!r}: unknown developability_features mapping key(s) "
            f"{unknown_keys!r}; expected target names from target_cols or 'default'."
        )
    for t in target_cols:
        if t in raw:
            out[t] = _parse_developability_groups_value(raw[t])
        else:
            out[t] = list(default_groups)
    return out


def _parse_developability_groups_value(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = [str(x).strip() for x in raw if str(x).strip()]
    else:
        s = str(raw).strip()
        if not s:
            return []
        out = [p.strip() for p in s.split(",") if p.strip()]
    if not out:
        return []
    return out


def _prepare_key(d: dict) -> tuple:
    grp_by_target = d.get("developability_feature_groups_by_target") or {}
    grp_items = tuple(
        sorted((str(t), tuple(v or [])) for t, v in dict(grp_by_target).items())
    )
    split_col_k = d.get("split_col") or ""
    n_splits_key = -1 if split_col_k else int(d["n_splits"])
    return (
        d["dataset_path"],
        tuple(d.get("developability_paths") or ()),
        d["name_col"],
        d["targets_csv"],
        d["features_csv"],
        grp_items,
        d["run_dir"],
        n_splits_key,
        d["random_state"],
        tuple(sorted(d["features_fracs"])),
        float(d["max_target_nan_frac"]),
        split_col_k,
    )


def _output_json_for_job(
    batch_root: Path,
    *,
    dataset_yaml_key: str,
    dataset_stem: str,
    fold_dir: Path,
    fold_k: str,
    selector: str,
    model: str,
    eval_frac_slug: str,
    track_name: str | None = None,
    hp_slug: str = "",
    eval_hp_slug: str = "",
) -> Path:
    sub = batch_root / _slug(dataset_yaml_key)
    hp_part = f"__{hp_slug}" if hp_slug else ""
    ev_part = f"__{eval_hp_slug}" if eval_hp_slug else ""
    track_part = f"__{_slug(track_name)}" if track_name else ""
    fname = (
        f"{_slug(dataset_stem)}__{_slug(fold_dir.name)}__fold{fold_k}{track_part}__"
        f"{_slug(selector)}__{_slug(model)}__{eval_frac_slug}{hp_part}{ev_part}.json"
    )
    return (sub / fname).resolve()


def _expand_master_lines_for_dataset(drec: dict, batch_root: Path) -> list[str]:
    jf = drec["jobs_file"]
    parsed_lines: list[tuple[str, str, str, str, Path]] = []
    fold_to_n_train: dict[str, int] = {}
    for base_line in jf.read_text().splitlines():
        base_line = base_line.strip()
        if not base_line:
            continue
        parts = base_line.split("\t")
        if len(parts) != 4:
            print(f"Bad phase-1 jobs line (need 4 tab cols): {base_line!r}", file=sys.stderr)
            raise SystemExit(1)
        fold_dir_s, k_s, stem_line, p_target = parts
        fold_dir_p = Path(fold_dir_s)
        stem_meta = _tab_tok(stem_line)
        if stem_meta != drec["dataset_stem"]:
            print(
                f"Jobs line dataset stem {stem_meta!r} != block stem {drec['dataset_stem']!r} "
                f"for {drec['yaml_key']}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        train_pq = fold_dir_p / f"fold_{k_s}_train.parquet"
        try:
            n_train_fold = _parquet_train_row_count(train_pq)
        except FileNotFoundError as e:
            print(f"{e} (jobs line {base_line!r})", file=sys.stderr)
            raise SystemExit(1) from e
        fold_to_n_train[k_s] = int(n_train_fold)
        parsed_lines.append((fold_dir_s, k_s, stem_line, p_target, fold_dir_p))

    fracs_kept, dropped_fracs = _dedupe_features_fracs_across_folds(
        drec["features_fracs"],
        fold_to_n_train,
    )
    if dropped_fracs:
        fold_nt = ", ".join(
            f"{k}→{fold_to_n_train[k]}" for k in sorted(fold_to_n_train, key=lambda kk: int(str(kk)))
        )
        print(
            f"[batch] {drec['yaml_key']}: features_frac dedupe across folds "
            f"(n_train: {fold_nt}): keep {fracs_kept} drop {dropped_fracs}",
            file=sys.stderr,
        )

    master_rows: list[str] = []
    for fold_dir_s, k_s, stem_line, p_target, fold_dir_p in parsed_lines:
        for job in drec["selector_jobs"]:
            sel = job["selector"]
            mod = job["model"]
            rho = job["correlation_min_abs_rho"]
            hp = job.get("hyperparameters") or {}
            track_name: str | None = job.get("track_name") or None
            rho_tok = "none" if rho is None else str(rho)
            hp_json = json.dumps(hp, sort_keys=True, separators=(",", ":"))
            hp_slug = ""
            if hp:
                hp_slug = "h" + hashlib.sha256(hp_json.encode()).hexdigest()[:12]
            ev_hp = drec["eval_hyperparameters"]
            eval_hp_json = json.dumps(ev_hp, sort_keys=True, separators=(",", ":"))
            eval_hp_slug = ""
            if ev_hp:
                eval_hp_slug = "e" + hashlib.sha256(eval_hp_json.encode()).hexdigest()[:12]
            for eval_frac in fracs_kept:
                ef_slug = f"frac{int(round(float(eval_frac) * 100)):03d}"
                out_json = _output_json_for_job(
                    batch_root,
                    dataset_yaml_key=drec["yaml_key"],
                    dataset_stem=drec["dataset_stem"],
                    fold_dir=fold_dir_p,
                    fold_k=k_s,
                    selector=sel,
                    model=mod,
                    eval_frac_slug=ef_slug,
                    track_name=track_name,
                    hp_slug=hp_slug,
                    eval_hp_slug=eval_hp_slug,
                )
                row = "\t".join(
                    [
                        _tab_tok(fold_dir_s),
                        _tab_tok(k_s),
                        _tab_tok(stem_line),
                        _tab_tok(p_target),
                        _tab_tok(drec["yaml_key"]),
                        _tab_tok(sel),
                        _tab_tok(mod),
                        _tab_tok(drec["eval_models"]),
                        str(out_json),
                        str(drec["random_state"]),
                        _tab_tok(rho_tok),
                        _tab_tok(str(eval_frac)),
                        _tab_tok(hp_json),
                        _tab_tok(eval_hp_json),
                        _tab_tok(track_name or ""),
                    ]
                )
                master_rows.append(row)
    return master_rows


def main() -> None:
    t_script0 = time.monotonic()
    p = argparse.ArgumentParser(
        description=(
            "Prepare folds (once per dataset) then GNU parallel over fold×selector×model jobs "
            "(one parallel batch per dataset for progress reporting)."
        ),
    )
    p.add_argument(
        "config",
        type=Path,
        help="YAML config path (repo-relative or absolute).",
    )
    p.add_argument(
        "--parallel-jobs",
        default=None,
        help="GNU parallel job slots (overrides YAML; default nproc or 4).",
    )
    p.add_argument(
        "--py",
        default=None,
        help='Python invocation, e.g. "conda run -n developability python" (overrides YAML).',
    )
    p.add_argument(
        "--no-preprocessing-skip",
        action="store_true",
        help="Always run phase 1 (ignore cache); overrides YAML.",
    )
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not run aggregate_batch_results.py after parallel finishes.",
    )
    p.add_argument(
        "--clean-folds",
        action="store_true",
        help=(
            "Remove fold parquet files after parallel completes. Default: keep them until "
            "hyperparameter tuning succeeds."
        ),
    )
    p.add_argument(
        "--no-clean-folds",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--automl-config",
        type=Path,
        default=None,
        help=(
            "AutoML YAML with the shared pipeline block (overrides automl_config in run config; "
            "default: src/automl.yaml)."
        ),
    )
    args = p.parse_args()

    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (_REPO_ROOT / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict):
        print("YAML root must be a mapping", file=sys.stderr)
        sys.exit(1)

    yaml_parallel = raw.pop("parallel_jobs", None)
    yaml_parallel = raw.pop("parallel-jobs", yaml_parallel)
    yaml_py = raw.pop("py", None)

    batch_result_root = raw.pop("batch_result_root", None)
    batch_result_root = raw.pop("batch-result-root", batch_result_root)
    if batch_result_root:
        batch_root = _resolve_path(batch_result_root)
    else:
        batch_root = (_REPO_ROOT / "runs" / f"batch_{cfg_path.stem}").resolve()

    batch_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[batch] Batch directory created (phase 1 runs before manifest / master TSV appear): "
        f"{batch_root}",
        file=sys.stderr,
        flush=True,
    )

    pipeline_cfg = _resolve_pipeline_config(raw, automl_config_cli=args.automl_config)

    parallel_jobs = args.parallel_jobs
    if parallel_jobs is None and yaml_parallel is not None:
        parallel_jobs = str(yaml_parallel).strip()
    if not parallel_jobs:
        try:
            parallel_jobs = str(len(os.sched_getaffinity(0)))
        except Exception:
            parallel_jobs = "4"

    py_invocation = args.py if args.py is not None else (yaml_py or "conda run -n developability python")
    py_parts = shlex.split(py_invocation)

    datasets = _parse_dataset_records(raw, pipeline_cfg)

    by_run_dir: dict[Path, list[dict]] = defaultdict(list)
    for drec in datasets:
        by_run_dir[drec["run_dir"].resolve()].append(drec)
    for rd, grp in by_run_dir.items():
        devs = {tuple(Path(p).resolve() for p in d["developability_paths"]) for d in grp}
        if len(devs) > 1:
            keys = ", ".join(sorted(repr(d["yaml_key"]) for d in grp))
            print(
                f"Warning: dataset blocks {keys} share run_dir {rd} but different "
                "developability_results_path(s); phase-1 cache may reuse the wrong merge. "
                "Use separate run_dir values or omit run_dir (default includes a developability suffix).",
                file=sys.stderr,
            )

    by_prepare: dict[tuple, list[dict]] = defaultdict(list)
    for drec in datasets:
        by_prepare[_prepare_key(drec)].append(drec)

    for key, group in by_prepare.items():
        (
            dataset_path,
            developability_paths,
            name_col,
            targets_csv,
            features_csv,
            dev_groups_by_target_items,
            run_dir,
            _n_splits_cache_key,
            random_state,
            _features_fracs_sorted,
            max_target_nan_frac,
            _split_col_key,
        ) = key
        features_frac_prepare = float(max(_features_fracs_sorted))
        force_preprocess = (
            any(d["force_preprocess"] for d in group) or args.no_preprocessing_skip
        )
        target_cols, feature_cols = _targets_features_as_lists(targets_csv, features_csv)
        dev_groups_by_target = {str(t): list(v) for t, v in dev_groups_by_target_items}
        jobs_file: Path = group[0]["jobs_file"]
        yaml_n_splits = int(group[0]["n_splits"])

        run_dir.mkdir(parents=True, exist_ok=True)
        jobs_file.parent.mkdir(parents=True, exist_ok=True)

        _gkeys = ", ".join(sorted(d["yaml_key"] for d in group))
        print(
            f"[batch] Phase 1 — dataset file={dataset_path.name} yaml_keys=[{_gkeys}] "
            f"run_dir={run_dir} jobs_file={jobs_file}",
            file=sys.stderr,
            flush=True,
        )

        if force_preprocess:
            print(
                f"Forcing phase 1: prepare_run.py -> {run_dir}",
                file=sys.stderr,
                flush=True,
            )
            run_phase1 = True
        elif _phase1_cache_ok(jobs_file):
            run_phase1 = False
            print(f"Skipping phase 1 (cache): {jobs_file}", file=sys.stderr, flush=True)
        else:
            print(f"Running phase 1 -> {run_dir}", file=sys.stderr, flush=True)
            run_phase1 = True

        if run_phase1:
            jobs_file.write_text("")
            target_buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for t in target_cols:
                groups_for_target = tuple(dev_groups_by_target.get(str(t), []))
                target_buckets[groups_for_target].append(str(t))

            for bucket_groups, bucket_targets in target_buckets.items():
                prep_cmd = [
                    *py_parts,
                    str(_REPO_ROOT / "src/automl/prepare_run.py"),
                    str(dataset_path),
                    "--name-col",
                    name_col,
                    "--target-cols",
                    *bucket_targets,
                    "--feature-cols",
                    *feature_cols,
                    "--developability-results",
                    *[str(p) for p in developability_paths],
                    "--output-dir",
                    str(run_dir),
                    "--n-splits",
                    str(yaml_n_splits),
                    "--random-state",
                    str(random_state),
                    "--features-frac",
                    str(features_frac_prepare),
                    "--max-target-nan-frac",
                    str(max_target_nan_frac),
                    "--jobs-file",
                    str(jobs_file),
                ]
                if bucket_groups:
                    prep_cmd.extend(
                        ["--developability-feature-groups", *list(bucket_groups)]
                    )
                sc = group[0].get("split_col")
                if sc:
                    prep_cmd.extend(["--split-col", str(sc)])
                group_slug = "all" if not bucket_groups else "_".join(bucket_groups)
                log_path = run_dir / f"prepare_run__{_slug(group_slug)}.log"
                _tpreview = (
                    ", ".join(bucket_targets[:8])
                    + (" …" if len(bucket_targets) > 8 else "")
                )
                print(
                    f"[batch] Spawning prepare_run.py (stdout → {log_path}; "
                    f"progress lines → stderr here): targets={_tpreview}",
                    file=sys.stderr,
                    flush=True,
                )
                with open(log_path, "a") as logf:
                    r = subprocess.run(
                        prep_cmd,
                        cwd=_REPO_ROOT,
                        stdout=logf,
                        stderr=sys.stderr,
                    )
                if r.returncode != 0:
                    print(f"prepare_run.py failed (see {log_path})", file=sys.stderr)
                    sys.exit(r.returncode)

        if not _phase1_cache_ok(jobs_file):
            print(f"Invalid or empty jobs file after phase 1: {jobs_file}", file=sys.stderr)
            sys.exit(1)
        _n_job_lines = sum(
            1 for line in jobs_file.read_text().splitlines() if line.strip()
        )
        print(
            f"[batch] Phase 1 done for group dataset={dataset_path.name} — "
            f"jobs_file={jobs_file} lines={_n_job_lines}",
            file=sys.stderr,
            flush=True,
        )

    master_path = batch_root / "parallel_jobs_master.tsv"
    manifest_path = batch_root / "batch_manifest.json"

    master_rows: list[str] = []
    utc_now = datetime.now(timezone.utc).isoformat()
    manifest_datasets: list[dict] = []
    lines_per_dataset: list[tuple[dict, list[str]]] = []

    for drec in datasets:
        jf = drec["jobs_file"]
        manifest_datasets.append(
            {
                "dataset_yaml_key": drec["yaml_key"],
                "yaml_block_key": drec.get("yaml_block_key", drec["yaml_key"]),
                "dataset_stem": drec["dataset_stem"],
                "dataset_path": str(drec["dataset_path"]),
                "run_dir": str(drec["run_dir"]),
                "developability_results_paths": [str(p) for p in drec["developability_paths"]],
                "phase1_jobs_file": str(jf),
                "random_state": drec["random_state"],
                "features_fracs": drec["features_fracs"],
                "eval_models": drec["eval_models"],
                "eval_hyperparameters": drec["eval_hyperparameters"],
                "selector_jobs": drec["selector_jobs"],
                "final_floating_sfs": drec.get("final_floating_sfs"),
                "developability_feature_groups_by_target": drec.get(
                    "developability_feature_groups_by_target", {}
                ),
                "split_col": drec.get("split_col"),
            }
        )
        chunk_lines = _expand_master_lines_for_dataset(drec, batch_root)
        lines_per_dataset.append((drec, chunk_lines))
        master_rows.extend(chunk_lines)

    master_path.write_text("\n".join(master_rows) + ("\n" if master_rows else ""))

    manifest = {
        "created_utc": utc_now,
        "config_path": str(cfg_path),
        "batch_result_root": str(batch_root),
        "master_jobs_tsv": str(master_path),
        "master_tsv_format": "15col-v1",
        "master_tsv_columns": [
            "fold_dir",
            "k",
            "dataset_stem",
            "pipeline_target_col",
            "dataset_yaml_key",
            "selector_name",
            "model_to_use",
            "eval_models",
            "output_json",
            "random_state",
            "correlation_min_abs_rho",
            "eval_features_frac",
            "selector_hyperparameters_json",
            "eval_hyperparameters_json",
            "pipeline_track_name",
        ],
        "job_line_count": len(master_rows),
        "parallel_jobs": parallel_jobs,
        "py_invocation": py_invocation,
        "datasets": manifest_datasets,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote master jobs ({len(master_rows)} lines): {master_path}", file=sys.stderr)
    print(f"Wrote manifest: {manifest_path}", file=sys.stderr)

    if not master_rows:
        print("No jobs to run.", file=sys.stderr)
        sys.exit(0)

    par_cmd_base = [
        "parallel",
        "--will-cite",
        f"--jobs={parallel_jobs}",
        "--line-buffer",
        "--colsep",
        "\t",
        *py_parts,
        str(_REPO_ROOT / "src/automl/run_fold_pipeline_config.py"),
        "--fold-dir",
        "{1}",
        "--fold",
        "{2}",
        "--dataset-stem",
        "{3}",
        "--pipeline-target-col",
        "{4}",
        "--dataset-yaml-key",
        "{5}",
        "--selector-name",
        "{6}",
        "--model-to-use",
        "{7}",
        "--eval-models",
        "{8}",
        "--output-json",
        "{9}",
        "--random-state",
        "{10}",
        "--correlation-min-abs-rho",
        "{11}",
        "--eval-features-frac",
        "{12}",
        "--selector-hyperparameters",
        "{13}",
        "--eval-hyperparameters",
        "{14}",
        "--pipeline-track-name",
        "{15}",
        "--quiet",
        "::::",
    ]

    chunk_paths: list[Path] = []
    for idx, (drec, chunk_lines) in enumerate(lines_per_dataset):
        if not chunk_lines:
            print(
                f"[batch] {drec['dataset_path'].name}: no jobs for {drec['yaml_key']} (skip)",
                file=sys.stderr,
            )
            continue
        chunk_path = batch_root / f"_parallel_chunk_{idx:04d}_{_slug(drec['yaml_key'])}.tsv"
        chunk_path.write_text("\n".join(chunk_lines) + ("\n" if chunk_lines else ""))
        chunk_paths.append(chunk_path)
        par_run = [*par_cmd_base, str(chunk_path)]
        print(
            f"[batch] Parallel {len(chunk_lines)} job(s): {drec['yaml_key']} …",
            file=sys.stderr,
            flush=True,
        )
        r = subprocess.run(par_run, cwd=_REPO_ROOT)
        if r.returncode != 0:
            sys.exit(r.returncode)
        elapsed = time.monotonic() - t_script0
        print(
            f"[batch] {drec['dataset_path'].name} ready — {drec['yaml_key']} "
            f"(elapsed {elapsed:.1f}s since script start)",
            file=sys.stderr,
            flush=True,
        )
        if not args.no_aggregate:
            agg_one = [
                *py_parts,
                str(_REPO_ROOT / "src/automl/aggregate_batch_results.py"),
                "--manifest",
                str(manifest_path),
                "--only-dataset-yaml-key",
                str(drec["yaml_key"]),
                "--no-plots",
            ]
            print(
                f"[batch] Aggregating results for {drec['yaml_key']} …",
                file=sys.stderr,
                flush=True,
            )
            r_agg = subprocess.run(agg_one, cwd=_REPO_ROOT)
            if r_agg.returncode != 0:
                sys.exit(r_agg.returncode)

    for cp in chunk_paths:
        try:
            cp.unlink(missing_ok=True)
        except OSError:
            pass

    if any(d.get("final_floating_sfs") for d in manifest_datasets):
        ffs_cmd = [
            *py_parts,
            str(_REPO_ROOT / "src/automl/run_final_floating_sfs_batch.py"),
            "--manifest",
            str(manifest_path),
        ]
        print(
            "Running per-fold post-grid final floating SFS (batch_manifest.json)...",
            file=sys.stderr,
        )
        r_ffs = subprocess.run(ffs_cmd, cwd=_REPO_ROOT)
        if r_ffs.returncode != 0:
            sys.exit(r_ffs.returncode)

    if not args.no_aggregate:
        agg_cmd = [
            *py_parts,
            str(_REPO_ROOT / "src/automl/aggregate_batch_results.py"),
            "--manifest",
            str(manifest_path),
            "--no-plots",
        ]
        print(
            "Aggregating full batch (batch_manifest.json; refreshes all datasets, "
            "includes floating-SFS JSONs if any)…",
            file=sys.stderr,
        )
        r2 = subprocess.run(agg_cmd, cwd=_REPO_ROOT)
        if r2.returncode != 0:
            sys.exit(r2.returncode)

    if args.clean_folds and not args.no_clean_folds:
        n_removed = 0
        seen_dirs: set[Path] = set()
        for drec in datasets:
            rd: Path = drec["run_dir"]
            if rd in seen_dirs:
                continue
            seen_dirs.add(rd)
            for pq in sorted(rd.rglob("*.parquet")):
                try:
                    pq.unlink()
                    n_removed += 1
                except OSError as e:
                    print(f"Warning: could not remove {pq}: {e}", file=sys.stderr)
        if n_removed:
            print(
                f"Removed {n_removed} fold parquet file(s) from run dir(s). "
                "(Omit --clean-folds to keep them for hyperparameter tuning.)",
                file=sys.stderr,
            )

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
