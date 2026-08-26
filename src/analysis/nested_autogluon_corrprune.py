#!/usr/bin/env python3
"""Nested CV: correlation prune (train-only) + AutoGluon Tabular regression.

Uses existing sequence-aware fold parquets. For each outer fold:
  inner loop on leftover folds with corr-prune on inner-train only,
  AutoGluon fit with tuning_data = inner-val,
  select model by pooled inner-val Spearman; then corr-prune on full outer-train,
  refit AutoGluon, predict outer-test.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.nested_cv import write_inner_fold_dir
from automl.pipeline_defaults import (
    DEFAULT_INTERCORR_THRESHOLD,
    DEFAULT_INTERCORR_IMPORTANCE_METRIC,
    DEFAULT_INTERCORR_REDUCTION_MODE,
)
from automl.utils import reduce_correlated_features

REPO = Path("/storage/antibody_data/PairedStructures/kitAb")

DEFAULT_FOLD_ROOT = (
    REPO
    / "runs/jain2017biophysical_folded_08_5_cv_prepare__"
    "our_abb2_final_set_of_features_descriptors_jain2017biophysical_folded_08_5_abb2_1_results"
)

JAIN2017_TARGETS = ("target_ELISA", "target_HICRT", "target_BVPELISA")

# Models that work in bio-ds (LightGBM may fail on older libstdc++).
_DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "CAT": {},
    "XGB": {},
    "RF": {},
    "XT": {},
    "KNN": {},
    "LR": {},
}
# Faster inner selection: tree + linear only (drop RF/XT/KNN).
_FAST_HYPERPARAMETERS: dict[str, Any] = {
    "CAT": {},
    "XGB": {},
    "LR": {},
}
_HP_ALIASES: dict[str, str] = {
    "cat": "CAT",
    "catboost": "CAT",
    "xgb": "XGB",
    "xgboost": "XGB",
    "rf": "RF",
    "randomforest": "RF",
    "xt": "XT",
    "extratrees": "XT",
    "knn": "KNN",
    "kneighbors": "KNN",
    "lr": "LR",
    "linear": "LR",
    "linearmodel": "LR",
}
_AG_MODEL_TO_HP_KEY: dict[str, str] = {
    "LightGBM": "GBM",
    "LightGBMXT": "GBM",
    "CatBoost": "CAT",
    "XGBoost": "XGB",
    "RandomForest": "RF",
    "ExtraTrees": "XT",
    "KNeighbors": "KNN",
    "LinearModel": "LR",
    "NeuralNetTorch": "NN_TORCH",
    "NeuralNetFastAI": "FASTAI",
}


def _spearman(y: pd.Series, yhat: pd.Series) -> float | None:
    y = pd.to_numeric(y, errors="coerce")
    yhat = pd.to_numeric(yhat, errors="coerce")
    mask = y.notna() & yhat.notna()
    y, yhat = y.loc[mask], yhat.loc[mask]
    if len(y) < 2 or y.nunique() < 2 or yhat.nunique() < 2:
        return None
    rho = float(y.corr(yhat, method="spearman"))
    return rho if np.isfinite(rho) else None


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    skip = {"name", "pdb_file", "error", "fold", target_col}
    return [
        c
        for c in df.columns
        if c not in skip
        and not str(c).startswith("target_")
        and not str(c).startswith("hc_subtype__")
        and not str(c).startswith("lc_subtype__")
    ]


def _corr_prune(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    candidate_features: list[str],
) -> list[str]:
    kept = reduce_correlated_features(
        train_df,
        target_col,
        candidate_features,
        correlation_threshold=DEFAULT_INTERCORR_THRESHOLD,
        reduction_mode=DEFAULT_INTERCORR_REDUCTION_MODE,  # type: ignore[arg-type]
        importance_metric=DEFAULT_INTERCORR_IMPORTANCE_METRIC,
    )
    return kept if kept else candidate_features[:1]


def _ag_table(
    df: pd.DataFrame,
    *,
    target_col: str,
    features: list[str],
) -> pd.DataFrame:
    cols = [target_col] + [f for f in features if f in df.columns]
    out = df[cols].copy()
    for c in features:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    return out.dropna(subset=[target_col])


def _parse_model_keys(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(_DEFAULT_HYPERPARAMETERS)
    out: list[str] = []
    for token in raw:
        key = _HP_ALIASES.get(token.strip().lower(), token.strip().upper())
        if key not in _DEFAULT_HYPERPARAMETERS:
            allowed = ", ".join(sorted(_DEFAULT_HYPERPARAMETERS))
            raise ValueError(f"Unknown model {token!r}; allowed: {allowed}")
        if key not in out:
            out.append(key)
    return out


def _hyperparameters_from_keys(keys: list[str]) -> dict[str, Any]:
    return {k: {} for k in keys}


def _fit_predictor(
    train_df: pd.DataFrame,
    *,
    target_col: str,
    features: list[str],
    path: Path,
    tuning_data: pd.DataFrame | None,
    time_limit: int,
    hyperparameters: dict[str, Any] | str | None = None,
    ag_feature_prune: bool = False,
    num_cpus: int | None = None,
) -> Any:
    from autogluon.tabular import TabularPredictor

    train_ag = _ag_table(train_df, target_col=target_col, features=features)
    tuning_ag = (
        _ag_table(tuning_data, target_col=target_col, features=features)
        if tuning_data is not None
        else None
    )
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    fit_kwargs: dict[str, Any] = {
        "time_limit": int(time_limit),
        "presets": None,
        "num_bag_folds": 0,
        "num_stack_levels": 0,
        "fit_weighted_ensemble": False,
    }
    if ag_feature_prune:
        fit_kwargs["feature_prune_kwargs"] = {}
    if num_cpus is not None:
        fit_kwargs["num_cpus"] = int(num_cpus)
    if hyperparameters is not None:
        fit_kwargs["hyperparameters"] = hyperparameters
    else:
        fit_kwargs["hyperparameters"] = _DEFAULT_HYPERPARAMETERS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor = TabularPredictor(
            label=target_col,
            problem_type="regression",
            eval_metric="pearsonr",
            path=str(path),
            verbosity=0,
        )
        predictor.fit(
            train_ag,
            tuning_data=tuning_ag,
            **fit_kwargs,
        )
    return predictor


def _predict_all_models(
    predictor: Any,
    val_df: pd.DataFrame,
    *,
    target_col: str,
    features: list[str],
) -> dict[str, pd.Series]:
    val_ag = _ag_table(val_df, target_col=target_col, features=features)
    y_true = val_ag[target_col]
    out: dict[str, pd.Series] = {}
    for model in predictor.model_names():
        try:
            pred = predictor.predict(val_ag, model=model)
            out[str(model)] = pd.Series(pred, index=y_true.index)
        except Exception:
            continue
    return out


def _ag_features(predictor: Any) -> list[str]:
    try:
        meta = predictor.feature_metadata
        if meta is not None:
            return list(meta.get_features())
    except Exception:
        pass
    try:
        return list(predictor.features())
    except Exception:
        return []


def _hyperparameters_for_model(model_name: str) -> dict[str, Any] | str | None:
    if "WeightedEnsemble" in model_name or "Ensemble" in model_name:
        return None
    for prefix, hp_key in _AG_MODEL_TO_HP_KEY.items():
        if model_name.startswith(prefix):
            return {hp_key: {}}
    return None


def _run_outer_fold(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    all_features: list[str],
    inner_time_limit: int,
    outer_time_limit: int,
    work_root: Path,
    model_keys: list[str],
    ag_feature_prune: bool,
    cpus_per_fit: int,
) -> dict[str, Any]:
    fold_dir = Path(fold_dir)
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")

    inner_dir = work_root / f"outer_{outer_k}" / "inner_folds"
    write_inner_fold_dir(fold_dir, outer_k=outer_k, dest=inner_dir)
    inner_meta = json.loads((inner_dir / "meta.json").read_text())
    n_inner = int(inner_meta["n_splits"])

    inner_preds_by_model: dict[str, list[tuple[pd.Series, pd.Series]]] = defaultdict(list)
    inner_detail: list[dict[str, Any]] = []

    for inner_k in range(n_inner):
        t_inner = time.perf_counter()
        inner_train = pd.read_parquet(inner_dir / f"fold_{inner_k}_train.parquet")
        inner_val = pd.read_parquet(inner_dir / f"fold_{inner_k}_test.parquet")
        corr_feats = _corr_prune(
            inner_train,
            target_col=target_col,
            candidate_features=all_features,
        )

        ag_path = work_root / f"outer_{outer_k}" / f"inner_{inner_k}" / "predictor"
        predictor = _fit_predictor(
            inner_train,
            target_col=target_col,
            features=corr_feats,
            path=ag_path,
            tuning_data=inner_val,
            time_limit=inner_time_limit,
            hyperparameters=_hyperparameters_from_keys(model_keys),
            ag_feature_prune=ag_feature_prune,
            num_cpus=cpus_per_fit,
        )
        print(
            f"    outer {outer_k} inner {inner_k}: "
            f"fit {time.perf_counter() - t_inner:.0f}s  "
            f"n_feats={len(corr_feats)}  models={predictor.model_names()}",
            flush=True,
        )
        val_ag = _ag_table(inner_val, target_col=target_col, features=corr_feats)
        y_true = val_ag[target_col]
        preds_by_model = _predict_all_models(
            predictor,
            inner_val,
            target_col=target_col,
            features=corr_feats,
        )
        fold_scores: dict[str, float | None] = {}
        for model, yhat in preds_by_model.items():
            inner_preds_by_model[model].append((y_true, yhat))
            fold_scores[model] = _spearman(y_true, yhat)
        inner_detail.append(
            {
                "inner_fold": inner_k,
                "n_corr_features": len(corr_feats),
                "corr_features": corr_feats,
                "ag_features": _ag_features(predictor),
                "fold_spearman_by_model": fold_scores,
            }
        )

    pooled_scores: dict[str, float | None] = {}
    for model, parts in inner_preds_by_model.items():
        ys = pd.concat([p[0] for p in parts], ignore_index=True)
        yhats = pd.concat([p[1] for p in parts], ignore_index=True)
        pooled_scores[model] = _spearman(ys, yhats)

    valid = {
        m: s for m, s in pooled_scores.items() if s is not None and np.isfinite(s)
    }
    if not valid:
        raise RuntimeError(f"No valid inner model scores for outer_k={outer_k}")
    chosen_model = max(valid, key=lambda m: valid[m])
    chosen_hp = _hyperparameters_for_model(chosen_model)

    outer_corr_feats = _corr_prune(
        outer_train,
        target_col=target_col,
        candidate_features=all_features,
    )
    outer_ag_path = work_root / f"outer_{outer_k}" / "outer_predictor"
    outer_hp = chosen_hp or _hyperparameters_from_keys(model_keys)
    outer_predictor = _fit_predictor(
        outer_train,
        target_col=target_col,
        features=outer_corr_feats,
        path=outer_ag_path,
        tuning_data=None,
        time_limit=outer_time_limit,
        hyperparameters=outer_hp,
        ag_feature_prune=ag_feature_prune,
        num_cpus=cpus_per_fit,
    )
    test_ag = _ag_table(outer_test, target_col=target_col, features=outer_corr_feats)
    y_test = test_ag[target_col]
    try:
        yhat = outer_predictor.predict(test_ag, model=chosen_model)
    except Exception:
        yhat = outer_predictor.predict(test_ag)
    yhat = pd.Series(yhat, index=y_test.index)
    outer_sp = _spearman(y_test, yhat)
    print(
        f"  outer {outer_k} complete: Spearman={outer_sp}  model={chosen_model}",
        flush=True,
    )

    oof_rows = [
        {
            "name": name,
            "y": float(y),
            "yhat": float(pred),
            "outer_fold": outer_k,
            "chosen_model": chosen_model,
        }
        for name, y, pred in zip(
            outer_test.loc[y_test.index, "name"].astype(str), y_test, yhat
        )
    ]
    return {
        "outer_k": outer_k,
        "oof_rows": oof_rows,
        "outer_fold": {
            "outer_fold": outer_k,
            "n_test": int(len(y_test)),
            "spearman": outer_sp,
            "chosen_model": chosen_model,
            "chosen_inner_pooled_spearman": valid[chosen_model],
        },
        "features": {
            "outer_fold": outer_k,
            "corr_features_outer": outer_corr_feats,
            "ag_features_outer": _ag_features(outer_predictor),
            "inner_detail": inner_detail,
        },
        "inner_selection": {
            "outer_fold": outer_k,
            "pooled_inner_spearman_by_model": pooled_scores,
            "chosen_model": chosen_model,
            "chosen_hyperparameters": chosen_hp,
        },
    }


def run_nested_autogluon(
    fold_dir: Path,
    *,
    target_col: str,
    inner_time_limit: int,
    outer_time_limit: int,
    work_root: Path,
    model_keys: list[str],
    ag_feature_prune: bool,
    outer_jobs: int,
    cpus_per_fit: int,
) -> dict[str, Any]:
    fold_dir = Path(fold_dir)
    meta_path = fold_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json under {fold_dir}")
    n_outer = int(json.loads(meta_path.read_text())["n_splits"])
    sample_df = pd.read_parquet(fold_dir / "fold_0_train.parquet")
    all_features = _feature_columns(sample_df, target_col)
    common = {
        "target_col": target_col,
        "all_features": all_features,
        "inner_time_limit": inner_time_limit,
        "outer_time_limit": outer_time_limit,
        "work_root": work_root,
        "model_keys": model_keys,
        "ag_feature_prune": ag_feature_prune,
        "cpus_per_fit": cpus_per_fit,
    }

    if outer_jobs == 1:
        results = [
            _run_outer_fold(fold_dir, outer_k=k, **common)
            for k in range(n_outer)
        ]
    else:
        results = []
        with ProcessPoolExecutor(
            max_workers=min(outer_jobs, n_outer),
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(_run_outer_fold, fold_dir, outer_k=k, **common): k
                for k in range(n_outer)
            }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda r: r["outer_k"])

    outer_oof_rows = [row for result in results for row in result["oof_rows"]]
    outer_fold_records = [result["outer_fold"] for result in results]
    feature_records = [result["features"] for result in results]
    inner_selection_records = [result["inner_selection"] for result in results]

    oof = pd.DataFrame(outer_oof_rows)
    pooled = _spearman(oof["y"], oof["yhat"])
    return {
        "target_col": target_col,
        "fold_dir": str(fold_dir),
        "n_outer_folds": n_outer,
        "n_oof": len(oof),
        "pooled_spearman": pooled,
        "outer_folds": outer_fold_records,
        "oof": oof,
        "features": feature_records,
        "inner_selection": inner_selection_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=DEFAULT_FOLD_ROOT,
        help="Parent dir with target_* subdirs containing fold parquets.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(JAIN2017_TARGETS),
        help="Target_col subdir names (default: ELISA, HICRT, BVPELISA).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="AutoGluon time_limit (seconds) for every fit; overrides inner/outer limits.",
    )
    parser.add_argument(
        "--inner-time-limit",
        type=int,
        default=60,
        help="Seconds per inner-fold AutoGluon fit (default: 60).",
    )
    parser.add_argument(
        "--outer-time-limit",
        type=int,
        default=90,
        help="Seconds per outer refit AutoGluon fit (default: 90).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model families for inner selection (default: CAT XGB RF XT KNN LR). "
            "Aliases: cat, xgb, rf, xt, knn, lr."
        ),
    )
    parser.add_argument(
        "--ag-feature-prune",
        action="store_true",
        help=(
            "Enable AutoGluon internal feature pruning (feature_prune_kwargs={}). "
            "Much slower (~2x per fit) but matches the original plan."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Fast mode: CAT+XGB+LR only, no AutoGluon feature prune, "
            "inner=45s / outer=60s and 5 outer workers unless overridden."
        ),
    )
    parser.add_argument(
        "--outer-jobs",
        type=int,
        default=None,
        help=(
            "Concurrent outer-fold processes (default: 1; --fast defaults to 5). "
            "Each outer fold remains a fully independent nested-CV unit."
        ),
    )
    parser.add_argument(
        "--cpus-per-fit",
        type=int,
        default=None,
        help="CPU limit for each AutoGluon fit (default: auto, capped at 4).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/nested_autogluon_jain2017_abb2_1_<stamp>).",
    )
    args = parser.parse_args()

    if args.fast:
        if args.models is None:
            args.models = list(_FAST_HYPERPARAMETERS)
        if args.time_limit is None:
            args.inner_time_limit = 45
            args.outer_time_limit = 60

    inner_time_limit = int(args.time_limit or args.inner_time_limit)
    outer_time_limit = int(args.time_limit or args.outer_time_limit)
    model_keys = _parse_model_keys(args.models)
    outer_jobs = int(args.outer_jobs or (5 if args.fast else 1))
    if outer_jobs < 1:
        parser.error("--outer-jobs must be >= 1")
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    cpus_per_fit = int(
        args.cpus_per_fit
        or max(1, min(4, available_cpus // outer_jobs))
    )
    if cpus_per_fit < 1:
        parser.error("--cpus-per-fit must be >= 1")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (
        REPO / f"runs/nested_autogluon_jain2017_abb2_1_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    nested_report = pd.read_csv(
        REPO / "runs/reanalysis_20260818T113409Z/nested_report.csv"
    )
    kitab_nested = {
        (str(r.Dataset_stem), str(r.Target_col)): float(r.nested_Spearman_pooled_oof)
        for r in nested_report.itertuples()
    }

    summary_rows: list[dict[str, Any]] = []
    all_oof: list[pd.DataFrame] = []

    for target in args.targets:
        fold_dir = args.fold_root / target
        if not (fold_dir / "meta.json").is_file():
            raise FileNotFoundError(f"Missing fold dir: {fold_dir}")
        work_root = out_dir / target / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {target} ===", flush=True)
        print(
            f"  models={model_keys}  inner={inner_time_limit}s  outer={outer_time_limit}s  "
            f"ag_feature_prune={args.ag_feature_prune}  outer_jobs={outer_jobs}  "
            f"cpus_per_fit={cpus_per_fit}",
            flush=True,
        )
        result = run_nested_autogluon(
            fold_dir,
            target_col=target,
            inner_time_limit=inner_time_limit,
            outer_time_limit=outer_time_limit,
            work_root=work_root,
            model_keys=model_keys,
            ag_feature_prune=bool(args.ag_feature_prune),
            outer_jobs=outer_jobs,
            cpus_per_fit=cpus_per_fit,
        )
        oof = result["oof"].copy()
        oof["target_col"] = target
        all_oof.append(oof)
        oof.to_parquet(out_dir / f"oof_{target}.parquet", index=False)

        kitab_sp = kitab_nested.get(
            ("jain2017biophysical_folded_08_5", target)
        )
        row = {
            "Dataset_stem": "jain2017biophysical_folded_08_5",
            "Target_col": target,
            "pooled_spearman_autogluon": result["pooled_spearman"],
            "pooled_spearman_kitab_nested": kitab_sp,
            "delta_autogluon_minus_kitab": (
                (result["pooled_spearman"] - kitab_sp)
                if result["pooled_spearman"] is not None and kitab_sp is not None
                else None
            ),
            "n_oof": result["n_oof"],
        }
        for rec in result["outer_folds"]:
            row[f"outer_{rec['outer_fold']}_spearman"] = rec["spearman"]
            row[f"outer_{rec['outer_fold']}_model"] = rec["chosen_model"]
        summary_rows.append(row)

        (out_dir / f"features_{target}.json").write_text(
            json.dumps(result["features"], indent=2)
        )
        (out_dir / f"inner_selection_{target}.json").write_text(
            json.dumps(result["inner_selection"], indent=2)
        )
        kitab_text = f"{kitab_sp:.3f}" if kitab_sp is not None else "NA"
        print(
            f"  pooled Spearman={result['pooled_spearman']:.3f}  "
            f"kitAb nested={kitab_text}",
            flush=True,
        )
        for rec in result["outer_folds"]:
            print(
                f"  outer {rec['outer_fold']}: "
                f"Spearman={rec['spearman']:.3f}  model={rec['chosen_model']}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if all_oof:
        pd.concat(all_oof, ignore_index=True).to_parquet(
            out_dir / "oof.parquet", index=False
        )

    print("\n=== comparison ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nWrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
