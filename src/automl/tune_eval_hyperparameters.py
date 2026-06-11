#!/usr/bin/env python3
"""Grid-search eval-regressor hyperparameters for analysis-shortlisted models."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.aggregated_csv import (
    COL_DATASET,
    COL_FEATURES,
    COL_PEAR,
    COL_RUN_ID,
    COL_SOURCE,
    COL_SPEAR,
    COL_TARGET,
    COL_TRACK,
    eval_model_slug,
    expand_glob_pattern,
    is_our_source,
    row_is_gpr_eval,
)
from analysis.features import (
    collapse_best_track_per_group,
    parse_features_frac_from_run_id,
    pick_preferred_row,
    select_close_top_models,
)
from automl.run_fold_pipeline_config import (
    _json_float,
    _pearson_stat_p,
    _spearman_stat_p,
    _top_k_column_indices_for_eval,
)
from automl.utils import (
    _canonical_eval_hp_param,
    _coerce_eval_hp_value,
    fit_regressor,
    make_regressor,
)

_NAME_COL = "name"

_EVAL_HP_GRIDS: dict[str, list[dict]] = {
    "linear": [
        {"fit_intercept": True},
        {"fit_intercept": False},
    ],
    "elasticnet": [
        {"alpha": 0.0001, "l1_ratio": 0.5},
        {"alpha": 0.001, "l1_ratio": 0.1},
        {"alpha": 0.001, "l1_ratio": 0.5},
        {"alpha": 0.001, "l1_ratio": 0.9},
        {"alpha": 0.01, "l1_ratio": 0.1},
        {"alpha": 0.01, "l1_ratio": 0.5},
        {"alpha": 0.01, "l1_ratio": 0.9},
        {"alpha": 0.1, "l1_ratio": 0.5},
        {"alpha": 1.0, "l1_ratio": 0.5},
    ],
    "randomforest": [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 300, "max_depth": None},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 300, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 20},
    ],
    "svm": [
        {"C": 0.01, "epsilon": 0.01},
        {"C": 0.1, "epsilon": 0.01},
        {"C": 1.0, "epsilon": 0.01},
        {"C": 10.0, "epsilon": 0.01},
        {"C": 1.0, "epsilon": 0.1},
    ],
    "knn": [
        {"n_neighbors": 3, "weights": "uniform"},
        {"n_neighbors": 5, "weights": "uniform"},
        {"n_neighbors": 10, "weights": "uniform"},
        {"n_neighbors": 5, "weights": "distance"},
        {"n_neighbors": 10, "weights": "distance"},
        {"n_neighbors": 15, "weights": "distance"},
    ],
    "gpr": [
        {"alpha": 1e-10},
        {"alpha": 0.01},
        {"alpha": 0.1},
        {"alpha": 1.0},
    ],
}


def _eval_hp_grid_for_model(model: str) -> list[dict]:
    m = str(model).strip().lower()
    raw = _EVAL_HP_GRIDS.get(m)
    if not raw:
        return [{}]
    return [dict(pt) for pt in raw]


def _eval_hp_grid_to_make_regressor_kwargs(model: str, hp_point: dict) -> dict:
    m = str(model).strip().lower()
    out: dict = {}
    for pk, pv in hp_point.items():
        if pv is None:
            continue
        mr = _canonical_eval_hp_param(m, str(pk))
        out[mr] = _coerce_eval_hp_value(mr, pv)
    return out


_FOLD_KEY_RE = re.compile(r"^fold_(\d+)$")


def _parquet_fold_index(fold_key: str) -> int | None:
    m = _FOLD_KEY_RE.match(str(fold_key).strip())
    if not m:
        return None
    return int(m.group(1)) - 1


_SUMMARY_SKIP_DATASETS = frozenset({"pipeline_time_summary", "jaccard_summary"})


def _features_frac_for_run(
    run_id: str,
    target_col: str,
    fold_dir: Path,
) -> float | None:
    frac = parse_features_frac_from_run_id(run_id, target_col)
    if frac is not None:
        return frac
    meta_path = fold_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(meta, dict):
        return None
    raw = meta.get("features_frac", meta.get("eval_features_frac"))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def _developability_label_from_run_dir(run_dir: str, dev_paths: list[str] | None) -> str:
    from automl.aggregate_batch_results import _developability_label_from_run_dir as _lbl

    return _lbl(run_dir, dev_paths or None)


def _manifest_by_developability_source(manifest: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for block in manifest.get("datasets") or []:
        if not isinstance(block, dict):
            continue
        yk = block.get("dataset_yaml_key")
        rd = block.get("run_dir")
        if not yk:
            continue
        dev_paths = block.get("developability_results_paths") or []
        label = _developability_label_from_run_dir(str(rd or ""), dev_paths or None)
        out[str(label)] = block
    return out


def _read_aggregated_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_file"] = p.name
                rows.append(row)
    return rows


def _summary_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = str(row.get(COL_DATASET, "")).strip()
            if ds in _SUMMARY_SKIP_DATASETS:
                continue
            src = str(row.get(COL_SOURCE, "")).strip()
            tgt = str(row.get(COL_TARGET, "")).strip()
            if not ds or not src or not tgt:
                continue
            if not is_our_source(src):
                continue
            rows.append(row)
    return rows


def _filter_aggregated_for_summary(
    agg_rows: list[dict[str, str]],
    summary_row: dict[str, str],
    *,
    no_gpr: bool,
    ignore_track: bool = False,
) -> list[dict[str, str]]:
    ds = str(summary_row.get(COL_DATASET, ""))
    src = str(summary_row.get(COL_SOURCE, ""))
    tgt = str(summary_row.get(COL_TARGET, ""))
    trk = str(summary_row.get(COL_TRACK, "")).strip()
    out: list[dict[str, str]] = []
    for r in agg_rows:
        if str(r.get(COL_DATASET, "")) != ds:
            continue
        if str(r.get(COL_SOURCE, "")) != src:
            continue
        if str(r.get(COL_TARGET, "")) != tgt:
            continue
        if not ignore_track and trk and str(r.get(COL_TRACK, "")).strip() != trk:
            continue
        if no_gpr and row_is_gpr_eval(r):
            continue
        out.append(r)
    return out


def _parse_features_by_fold(cell: str) -> dict[str, list[str]]:
    if not cell or not str(cell).strip():
        return {}
    try:
        data = json.loads(cell)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for fk, feats in data.items():
        if not isinstance(feats, list):
            continue
        out[str(fk)] = [str(x) for x in feats if isinstance(x, str)]
    return out


def _fold_keys_sorted(features_by_fold: dict[str, list[str]]) -> list[str]:
    def _key(k: str) -> int:
        m = _FOLD_KEY_RE.match(k)
        return int(m.group(1)) if m else 10_000

    return sorted(features_by_fold.keys(), key=_key)


def _fold_parquets_ready(fold_dir: Path, fold_keys: list[str]) -> bool:
    if not (fold_dir / "meta.json").is_file():
        return False
    for fk in fold_keys:
        k = _parquet_fold_index(fk)
        if k is None or k < 0:
            continue
        if not (fold_dir / f"fold_{k}_train.parquet").is_file():
            return False
        if not (fold_dir / f"fold_{k}_test.parquet").is_file():
            return False
    return bool(fold_keys)


def _eval_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    *,
    target_col: str,
    eval_model: str,
    features_frac: float | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    present = [c for c in cols if c in train_df.columns and c in test_df.columns]
    if not present:
        return None
    y_tr = train_df[target_col].to_numpy(dtype=np.float64, copy=True)
    y_te = test_df[target_col].to_numpy(dtype=np.float64, copy=True)
    X_tr = train_df.loc[:, present].to_numpy(dtype=np.float64, copy=True)
    X_te = test_df.loc[:, present].to_numpy(dtype=np.float64, copy=True)
    n_train = int(len(y_tr))
    used = present
    if features_frac is not None:
        k_cap = min(len(present), max(1, int(float(features_frac) * n_train)))
        idx = _top_k_column_indices_for_eval(
            eval_model,
            X_tr,
            y_tr,
            k_max=k_cap,
            random_state=random_state,
            n_train=n_train,
        )
        used = [present[i] for i in idx.tolist()]
        X_tr = train_df.loc[:, used].to_numpy(dtype=np.float64, copy=True)
        X_te = test_df.loc[:, used].to_numpy(dtype=np.float64, copy=True)
    return X_tr, X_te, y_tr, y_te, used


def _mean_cv_metrics(
    *,
    fold_dir: Path,
    target_col: str,
    features_by_fold: dict[str, list[str]],
    eval_model: str,
    hp_kwargs: dict,
    random_state: int,
    features_frac: float | None,
) -> tuple[float | None, float | None]:
    fold_keys = _fold_keys_sorted(features_by_fold)
    if not _fold_parquets_ready(fold_dir, fold_keys):
        return None, None

    spears: list[float] = []
    pears: list[float] = []
    n_expected = 0
    for fk in fold_keys:
        k = _parquet_fold_index(fk)
        if k is None or k < 0:
            continue
        feats = features_by_fold.get(fk) or []
        if not feats:
            continue
        n_expected += 1
        train_df = pd.read_parquet(fold_dir / f"fold_{k}_train.parquet")
        test_df = pd.read_parquet(fold_dir / f"fold_{k}_test.parquet")
        mats = _eval_feature_matrix(
            train_df,
            test_df,
            feats,
            target_col=target_col,
            eval_model=eval_model,
            features_frac=features_frac,
            random_state=random_state,
        )
        if mats is None:
            continue
        X_tr, X_te, y_tr, y_te, _used = mats
        n_train = int(len(y_tr))
        mkw: dict = {
            "random_state": random_state,
            "n_jobs": -1,
            "n_samples_fit": n_train,
            **hp_kwargs,
        }
        try:
            model = make_regressor(eval_model, **mkw)
            fit_regressor(model, X_tr, y_tr)
            if eval_model == "gpr":
                y_pred, _ = model.predict(X_te, return_std=True)
            else:
                y_pred = model.predict(X_te)
        except Exception:
            continue
        sp, _ = _spearman_stat_p(y_te, y_pred)
        pe, _ = _pearson_stat_p(y_te, y_pred)
        if sp is None:
            continue
        spears.append(float(sp))
        if pe is not None:
            pears.append(float(pe))

    if not spears or len(spears) != n_expected:
        return None, None
    return float(np.mean(spears)), float(np.mean(pears)) if pears else None


def _grid_search_candidate(
    *,
    fold_dir: Path,
    target_col: str,
    features_by_fold: dict[str, list[str]],
    eval_model: str,
    random_state: int,
    features_frac: float | None,
) -> tuple[dict, float | None, float | None]:
    grid = _eval_hp_grid_for_model(eval_model)
    best_hp_raw: dict = {}
    best_spear: float | None = None
    best_pear: float | None = None
    for hp_raw in grid:
        try:
            hp_kwargs = _eval_hp_grid_to_make_regressor_kwargs(eval_model, hp_raw)
        except (ValueError, TypeError):
            continue
        sp, pe = _mean_cv_metrics(
            fold_dir=fold_dir,
            target_col=target_col,
            features_by_fold=features_by_fold,
            eval_model=eval_model,
            hp_kwargs=hp_kwargs,
            random_state=random_state,
            features_frac=features_frac,
        )
        if sp is None:
            continue
        if best_spear is None or sp > best_spear + 1e-12:
            best_spear = sp
            best_pear = pe
            best_hp_raw = dict(hp_raw)
    return best_hp_raw, best_spear, best_pear


def _concat_deduped_train(fold_dir: Path) -> pd.DataFrame | None:
    train_parts: list[pd.DataFrame] = []
    for p in sorted(fold_dir.glob("fold_*_train.parquet"), key=lambda x: int(x.stem.split("_")[1])):
        train_parts.append(pd.read_parquet(p))
    if not train_parts:
        return None
    train_df = pd.concat(train_parts, ignore_index=True)
    if _NAME_COL in train_df.columns:
        return train_df.drop_duplicates(subset=[_NAME_COL], keep="first").copy()
    return train_df.drop_duplicates(keep="first").copy()


def _fit_final_model(
    *,
    fold_dir: Path,
    target_col: str,
    feature_cols: list[str],
    eval_model: str,
    hp_kwargs: dict,
    random_state: int,
    features_frac: float | None,
) -> tuple[Any | None, list[str]]:
    train_df = _concat_deduped_train(fold_dir)
    if train_df is None:
        return None, []
    cols = [c for c in feature_cols if c in train_df.columns]
    if not cols:
        return None, []
    y = train_df[target_col].to_numpy(dtype=np.float64, copy=True)
    X = train_df.loc[:, cols].to_numpy(dtype=np.float64, copy=True)
    n_train = int(len(y))
    used_cols = cols
    if features_frac is not None:
        k_cap = min(len(cols), max(1, int(float(features_frac) * n_train)))
        idx = _top_k_column_indices_for_eval(
            eval_model,
            X,
            y,
            k_max=k_cap,
            random_state=random_state,
            n_train=n_train,
        )
        used_cols = [cols[i] for i in idx.tolist()]
        X = train_df.loc[:, used_cols].to_numpy(dtype=np.float64, copy=True)
    mkw: dict = {
        "random_state": random_state,
        "n_jobs": -1,
        "n_samples_fit": n_train,
        **hp_kwargs,
    }
    model = make_regressor(eval_model, **mkw)
    fit_regressor(model, X, y)
    return model, used_cols


def load_tuned_model(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load sklearn estimator and metadata from a tuned model directory."""
    model_dir = Path(model_dir)
    est_path = model_dir / "estimator.joblib"
    meta_path = model_dir / "meta.json"
    legacy_path = model_dir / "model.joblib"
    if est_path.is_file() and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return joblib.load(est_path), meta
    if legacy_path.is_file():
        bundle = joblib.load(legacy_path)
        if isinstance(bundle, dict) and "model" in bundle:
            meta = {k: v for k, v in bundle.items() if k != "model"}
            return bundle["model"], meta
    raise FileNotFoundError(
        f"No tuned model under {model_dir} (expected estimator.joblib + meta.json)"
    )


def save_tuned_model(
    model_dir: Path,
    *,
    estimator: Any,
    meta: dict[str, Any],
) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_dir / "estimator.joblib", compress=3)
    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return model_dir


def _union_features(features_by_fold: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for fk in _fold_keys_sorted(features_by_fold):
        for f in features_by_fold.get(fk) or []:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def _default_models_root(batch_root: Path) -> Path:
    """``<automl>/tuned_models`` when batch results live at ``<automl>``."""
    return batch_root.resolve() / "tuned_models"


def _model_out_dir(
    models_root: Path,
    summary_row: dict[str, str],
) -> Path:
    ds = str(summary_row.get(COL_DATASET, ""))
    src = str(summary_row.get(COL_SOURCE, ""))
    tgt = str(summary_row.get(COL_TARGET, ""))
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{ds}__{src}__{tgt}").strip("_")
    return models_root / stem


@dataclass
class TuningResult:
    metrics_path: Path
    expected_jobs: int
    tuned_jobs: int
    tuned_run_dirs: list[Path] = field(default_factory=list)


def run_tuning(
    *,
    batch_root: Path,
    best_metrics_summary: Path,
    aggregated_paths: list[Path],
    out_dir: Path,
    manifest_path: Path,
    margin: float = 0.1,
    max_rank: int = 3,
    no_gpr: bool = False,
    models_root: Path | None = None,
    metrics_name: str = "tuned_eval_hyperparameters_metrics.csv",
    limit: int | None = None,
) -> TuningResult:
    manifest = _load_manifest(manifest_path)
    manifest_by_src = _manifest_by_developability_source(manifest)
    agg_rows = _read_aggregated_rows(aggregated_paths)
    # One tuning job per (dataset, source, target): best summary row across tracks.
    summary = collapse_best_track_per_group(_summary_rows(best_metrics_summary))

    if models_root is None:
        models_root = _default_models_root(batch_root)
    models_root = models_root.resolve()
    models_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / metrics_name

    fieldnames = [
        COL_DATASET,
        COL_SOURCE,
        COL_TARGET,
        "random_state",
        "shortlisted_run_ids",
        "n_shortlisted",
        "winning_run_id",
        "winning_track",
        "eval_model",
        "cv_spearman_mean",
        "cv_pearson_mean",
        "best_hyperparameters_json",
        "model_path",
    ]
    result_rows: list[dict[str, Any]] = []
    tuned_run_dirs: list[Path] = []
    expected_jobs = len(summary) if limit is None else min(int(limit), len(summary))

    for i, summary_row in enumerate(summary):
        if limit is not None and i >= limit:
            break
        src = str(summary_row.get(COL_SOURCE, ""))
        tgt = str(summary_row.get(COL_TARGET, ""))
        block = manifest_by_src.get(src)
        if block is None:
            print(f"Warning: no manifest entry for source {src!r}; skip.", file=sys.stderr)
            continue

        pool = _filter_aggregated_for_summary(
            agg_rows,
            summary_row,
            no_gpr=no_gpr,
            ignore_track=True,
        )
        if not pool:
            print(
                f"Warning: no aggregated rows for {summary_row.get(COL_DATASET)!r} "
                f"{src!r} {tgt!r}; skip.",
                file=sys.stderr,
            )
            continue

        shortlisted = select_close_top_models(
            pool,
            metric_col=COL_SPEAR,
            max_rank=max_rank,
            margin=margin,
        )
        if not shortlisted:
            continue

        random_state = int(block.get("random_state", 42))
        run_dir = Path(str(block.get("run_dir", "")))
        fold_dir = run_dir / tgt
        if not fold_dir.is_dir():
            print(f"Warning: missing fold dir {fold_dir}; skip.", file=sys.stderr)
            continue

        best_run: dict[str, str] | None = None
        best_hp_raw: dict = {}
        best_hp_kwargs: dict = {}
        best_spear: float | None = None
        best_pear: float | None = None

        for cand in shortlisted:
            run_id = str(cand.get(COL_RUN_ID, ""))
            em = eval_model_slug(run_id, tgt)
            if em is None:
                continue
            feats_by_fold = _parse_features_by_fold(str(cand.get(COL_FEATURES, "")))
            if not feats_by_fold:
                continue
            if not _fold_parquets_ready(fold_dir, _fold_keys_sorted(feats_by_fold)):
                print(
                    f"Warning: fold parquets missing under {fold_dir} "
                    f"(re-run AutoML to regenerate fold parquets); skip {run_id!r}.",
                    file=sys.stderr,
                )
                continue
            feats_frac = _features_frac_for_run(run_id, tgt, fold_dir)
            hp_raw, sp, pe = _grid_search_candidate(
                fold_dir=fold_dir,
                target_col=tgt,
                features_by_fold=feats_by_fold,
                eval_model=em,
                random_state=random_state,
                features_frac=feats_frac,
            )
            if sp is None:
                continue
            if best_run is None:
                best_spear = sp
                best_pear = pe
                best_run = cand
                best_hp_raw = hp_raw
                try:
                    best_hp_kwargs = _eval_hp_grid_to_make_regressor_kwargs(em, hp_raw)
                except (ValueError, TypeError):
                    best_hp_kwargs = {}
            else:
                cand_pick = pick_preferred_row(
                    best_run,
                    cand,
                    metric_col=COL_SPEAR,
                    primary_metric_a=best_spear,
                    primary_metric_b=sp,
                )
                if cand_pick is cand:
                    best_spear = sp
                    best_pear = pe
                    best_run = cand
                    best_hp_raw = hp_raw
                    try:
                        best_hp_kwargs = _eval_hp_grid_to_make_regressor_kwargs(em, hp_raw)
                    except (ValueError, TypeError):
                        best_hp_kwargs = {}

        if best_run is None or best_spear is None:
            print(
                f"Warning: grid search found no valid model for "
                f"{summary_row.get(COL_DATASET)!r} {src!r} {tgt!r}.",
                file=sys.stderr,
            )
            continue

        run_id = str(best_run.get(COL_RUN_ID, ""))
        em = eval_model_slug(run_id, tgt) or ""
        feats_by_fold = _parse_features_by_fold(str(best_run.get(COL_FEATURES, "")))
        feature_cols = _union_features(feats_by_fold)
        win_frac = _features_frac_for_run(run_id, tgt, fold_dir)
        model, used_cols = _fit_final_model(
            fold_dir=fold_dir,
            target_col=tgt,
            feature_cols=feature_cols,
            eval_model=em,
            hp_kwargs=best_hp_kwargs,
            random_state=random_state,
            features_frac=win_frac,
        )
        if model is None:
            print(f"Warning: could not fit final model for {run_id!r}.", file=sys.stderr)
            continue

        model_dir = _model_out_dir(models_root, summary_row)
        meta = {
            "eval_model": em,
            "target_col": tgt,
            "feature_cols": used_cols,
            "features_frac": win_frac,
            "hyperparameters": best_hp_raw,
            "winning_run_id": run_id,
            "winning_track": str(best_run.get(COL_TRACK, "")),
            "random_state": random_state,
            "dataset_stem": str(summary_row.get(COL_DATASET, "")),
            "developability_source": src,
            "cv_spearman_mean": _json_float(best_spear),
            "cv_pearson_mean": _json_float(best_pear),
            "scaling": "none",
            "inference_note": (
                "Predict on raw (unscaled) descriptor values in feature_cols order. "
                "Load with automl.tune_eval_hyperparameters.load_tuned_model(model_dir)."
            ),
        }
        save_tuned_model(model_dir, estimator=model, meta=meta)
        if run_dir not in tuned_run_dirs:
            tuned_run_dirs.append(run_dir)

        result_rows.append({
            COL_DATASET: str(summary_row.get(COL_DATASET, "")),
            COL_SOURCE: src,
            COL_TARGET: tgt,
            "random_state": random_state,
            "shortlisted_run_ids": json.dumps(
                [str(c.get(COL_RUN_ID, "")) for c in shortlisted],
                ensure_ascii=False,
            ),
            "n_shortlisted": len(shortlisted),
            "winning_run_id": run_id,
            "winning_track": str(best_run.get(COL_TRACK, "")),
            "eval_model": em,
            "cv_spearman_mean": best_spear,
            "cv_pearson_mean": best_pear if best_pear is not None else "",
            "best_hyperparameters_json": json.dumps(best_hp_raw, ensure_ascii=False),
            "model_path": str(model_dir),
        })
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in result_rows:
            w.writerow(r)

    print(
        f"Wrote {len(result_rows)} tuned model(s) under {models_root} "
        f"and metrics to {metrics_path}",
        file=sys.stderr,
    )
    return TuningResult(
        metrics_path=metrics_path,
        expected_jobs=expected_jobs,
        tuned_jobs=len(result_rows),
        tuned_run_dirs=tuned_run_dirs,
    )


def _clean_fold_parquets(run_dirs: list[Path]) -> int:
    n_removed = 0
    seen: set[str] = set()
    for run_path in run_dirs:
        rd = str(run_path)
        if not rd or rd in seen:
            continue
        seen.add(rd)
        if not run_path.is_dir():
            continue
        for pq in sorted(run_path.rglob("*.parquet")):
            try:
                pq.unlink()
                n_removed += 1
            except OSError as e:
                print(f"Warning: could not remove {pq}: {e}", file=sys.stderr)
    return n_removed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument(
        "--best-metrics-summary",
        type=Path,
        required=True,
        help="Analysis results CSV (e.g. analysis_results/results.csv).",
    )
    p.add_argument(
        "--aggregated-glob",
        type=str,
        required=True,
        help="Glob for aggregated_*.csv (e.g. <automl>/aggregated_*.csv).",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="batch_manifest.json (default: <batch-root>/batch_manifest.json).",
    )
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--max-rank", type=int, default=3)
    p.add_argument("--no-gpr", action="store_true")
    p.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help=(
            "Directory for tuned model artifacts (default: <batch-root>/tuned_models, "
            "i.e. <automl>/tuned_models when batch-root is <automl>)."
        ),
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--clean-folds",
        action="store_true",
        help=(
            "After all tuning jobs succeed, remove fold parquets only from run_dir(s) "
            "that produced tuned models."
        ),
    )
    args = p.parse_args()

    batch_root = args.batch_root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else batch_root / "batch_manifest.json"
    )
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    agg_paths = expand_glob_pattern(args.aggregated_glob)
    if not agg_paths:
        raise SystemExit(f"No aggregated CSVs matched: {args.aggregated_glob!r}")

    if not args.best_metrics_summary.is_file():
        raise SystemExit(f"Missing best-metrics summary: {args.best_metrics_summary}")

    models_root = (
        args.models_root.resolve()
        if args.models_root is not None
        else _default_models_root(batch_root)
    )

    result = run_tuning(
        batch_root=batch_root,
        best_metrics_summary=args.best_metrics_summary.resolve(),
        aggregated_paths=agg_paths,
        out_dir=args.out_dir.resolve(),
        manifest_path=manifest_path,
        margin=float(args.margin),
        max_rank=int(args.max_rank),
        no_gpr=bool(args.no_gpr),
        models_root=models_root,
        limit=args.limit,
    )

    if result.expected_jobs > 0 and result.tuned_jobs < result.expected_jobs:
        raise SystemExit(
            f"Tuning incomplete: {result.tuned_jobs}/{result.expected_jobs} model(s) "
            f"written to {result.metrics_path}. Fold parquets and other artifacts were kept."
        )

    if args.clean_folds:
        if result.tuned_jobs == 0:
            raise SystemExit("No tuned models; refusing --clean-folds.")
        n = _clean_fold_parquets(result.tuned_run_dirs)
        if n:
            print(
                f"Removed {n} fold parquet file(s) from {len(result.tuned_run_dirs)} "
                f"successfully tuned run_dir(s).",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
