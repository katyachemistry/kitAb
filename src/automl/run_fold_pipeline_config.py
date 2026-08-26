#!/usr/bin/env python3
"""One CV fold: feature selection, then optional eval-model fit/predict."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from automl.feature_selectors import (
    parse_selector_hyperparameters_mapping,
    run_feature_selection_on_one_fold,
)
from automl.pipeline_defaults import (
    DEFAULT_EVAL_MODEL_ORDER,
    DEFAULT_EVAL_MODELS,
    DEFAULT_RANDOM_STATE,
)
from automl.utils import fit_regressor, make_regressor, parse_eval_hyperparameters_mapping

_EVAL_MODEL_ORDER = DEFAULT_EVAL_MODEL_ORDER
_EVAL_MODEL_SET = frozenset(_EVAL_MODEL_ORDER)
_REMOVED_EVAL_MODELS = frozenset({"gpr"})


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_") or "cfg"


def _parse_eval_models(spec: str) -> list[str] | None:
    s = str(spec).strip().lower()
    if s in ("", "none", "skip", "off"):
        return None
    if s == "all":
        return list(_EVAL_MODEL_ORDER)
    parts = [p for p in re.split(r"[\s,]+", s) if p]
    if not parts:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p in _REMOVED_EVAL_MODELS:
            raise ValueError(
                f"Eval model {p!r} has been removed from kitAb. "
                f"Supported: {', '.join(_EVAL_MODEL_ORDER)}"
            )
        if p not in _EVAL_MODEL_SET:
            opts = ", ".join(_EVAL_MODEL_ORDER)
            raise ValueError(f"Unknown eval model {p!r}; expected one of: {opts}, or all")
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _json_float(x: float | None) -> float | None:
    if x is None:
        return None
    xf = float(x)
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf


def _finite_bivariate_variation(y: np.ndarray, yhat: np.ndarray) -> bool:
    a = np.asarray(y, dtype=np.float64).ravel()
    b = np.asarray(yhat, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        return False
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return False
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return False
    return True


def _spearman_stat_p(y: np.ndarray, yhat: np.ndarray) -> tuple[float | None, float | None]:
    if not _finite_bivariate_variation(y, yhat):
        return None, None
    r = spearmanr(y, yhat)
    try:
        stat, pv = float(r.statistic), float(r.pvalue)
    except AttributeError:
        stat, pv = float(r[0]), float(r[1])
    return _json_float(stat), _json_float(pv)


def _pearson_stat_p(y: np.ndarray, yhat: np.ndarray) -> tuple[float | None, float | None]:
    if not _finite_bivariate_variation(y, yhat):
        return None, None
    r = pearsonr(y, yhat)
    try:
        stat, pv = float(r.statistic), float(r.pvalue)
    except AttributeError:
        stat, pv = float(r[0]), float(r[1])
    return _json_float(stat), _json_float(pv)


def _ranking_scores_from_fitted_model(model) -> np.ndarray | None:
    if hasattr(model, "coef_") and model.coef_ is not None:
        c = np.asarray(model.coef_, dtype=np.float64)
        if c.ndim == 2:
            c = np.mean(np.abs(c), axis=0)
        else:
            c = np.abs(c.ravel())
        return c
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=np.float64)
    return None


def _top_k_column_indices_for_eval(
    m_type: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    k_max: int,
    random_state: int,
    n_train: int,
) -> np.ndarray:
    n_cols = X_tr.shape[1]
    if n_cols <= k_max:
        return np.arange(n_cols, dtype=int)

    if m_type == "knn":
        probe_type = "elasticnet"
    else:
        probe_type = m_type

    probe = make_regressor(
        probe_type,
        random_state=random_state,
        n_jobs=-1,
        n_samples_fit=n_train,
    )
    fit_regressor(probe, X_tr, y_tr)
    scores = _ranking_scores_from_fitted_model(probe)
    if scores is None or scores.shape[0] != n_cols:
        probe = make_regressor(
            "elasticnet",
            random_state=random_state,
            n_jobs=-1,
            n_samples_fit=n_train,
        )
        fit_regressor(probe, X_tr, y_tr)
        scores = _ranking_scores_from_fitted_model(probe)
    if scores is None or scores.shape[0] != n_cols:
        raise ValueError(
            f"eval ranking scores unavailable for probe {probe_type!r} "
            f"(n_cols={n_cols}); refusing column-order fallback"
        )

    order = np.argsort(-scores)
    picked = order[:k_max]
    return np.sort(picked)


_OOF_NAME_COL = "name"
_SKIP_EVAL_MODELS = frozenset({"gpr"})

# RandomForestRegressor.predict sums per-tree predictions across threads in a
# nondeterministic order, so repeated runs differ by ~1e-15. That is enough to
# swap near-tied predictions and move the fold Spearman by up to ~0.1 on small
# test folds, so the eval fit is pinned to one thread.
_EVAL_FIT_N_JOBS: dict[str, int] = {"randomforest": 1}


def _test_row_names(test_df: pd.DataFrame, n_test: int) -> list[str]:
    if _OOF_NAME_COL in test_df.columns:
        return [str(x) for x in test_df[_OOF_NAME_COL].tolist()]
    return [str(i) for i in range(n_test)]


def _evaluate_fold_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    eval_models: list[str],
    random_state: int,
    features_frac: float,
    eval_hp_by_model: dict[str, dict] | None = None,
    n_jobs: int = -1,
) -> tuple[dict[str, dict], pd.DataFrame]:
    tcol = str(target_col)
    models = [m for m in eval_models if str(m).strip().lower() not in _SKIP_EVAL_MODELS]
    cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    if len(cols) == 0:
        return (
            {
                m: {
                    "error": "no_selected_features_present_in_train_and_test",
                    "n_features": 0,
                }
                for m in models
            },
            pd.DataFrame(columns=["name", "y", "yhat", "eval_model"]),
        )

    y_tr = train_df[tcol].to_numpy(dtype=np.float64, copy=True)
    y_te = test_df[tcol].to_numpy(dtype=np.float64, copy=True)
    X_tr = train_df.loc[:, cols].to_numpy(dtype=np.float64, copy=True)
    X_te = test_df.loc[:, cols].to_numpy(dtype=np.float64, copy=True)

    n_train = int(len(y_tr))
    n_test = int(len(y_te))
    n_feat = len(cols)
    k_cap = min(n_feat, max(1, int(features_frac * n_train)))
    names = _test_row_names(test_df, n_test)
    out: dict[str, dict] = {}
    ev_hp = eval_hp_by_model or {}
    oof_rows: list[dict] = []

    for m in models:
        try:
            idx = _top_k_column_indices_for_eval(
                m,
                X_tr,
                y_tr,
                k_max=k_cap,
                random_state=random_state,
                n_train=n_train,
            )
            X_tr_u = X_tr[:, idx]
            X_te_u = X_te[:, idx]
            used_names = [cols[i] for i in idx.tolist()]
            n_used = int(X_tr_u.shape[1])

            mkw: dict = {
                "random_state": random_state,
                "n_jobs": _EVAL_FIT_N_JOBS.get(m, n_jobs),
                "n_samples_fit": n_train,
            }
            mkw.update(ev_hp.get(m, {}))
            model = make_regressor(m, **mkw)
            fit_regressor(model, X_tr_u, y_tr)
            y_pred = np.asarray(model.predict(X_te_u), dtype=np.float64).ravel()
            r2 = r2_score(y_te, y_pred)
            mse = mean_squared_error(y_te, y_pred)
            rho_f, rho_p = _spearman_stat_p(y_te, y_pred)
            pr_f, pr_p = _pearson_stat_p(y_te, y_pred)
            entry: dict = {
                "r2": _json_float(r2),
                "mse": _json_float(mse),
                "spearman_rho": rho_f,
                "spearman_p": rho_p,
                "pearson_r": pr_f,
                "pearson_p": pr_p,
                "n_train": n_train,
                "n_test": n_test,
                "n_features_after_selection": n_feat,
                "n_features_allowed_by_frac": k_cap,
                "n_features_used_at_eval": n_used,
            }
            if n_used < n_feat:
                entry["eval_features_used"] = used_names
                if m == "knn":
                    entry["knn_ranking_probe"] = "elasticnet"
            out[m] = entry
            for nm, yt, yh in zip(names, y_te.tolist(), y_pred.tolist()):
                oof_rows.append(
                    {
                        "name": str(nm),
                        "y": float(yt),
                        "yhat": float(yh),
                        "eval_model": str(m),
                    }
                )
        except Exception as e:
            out[m] = {
                "error": str(e),
                "n_train": n_train,
                "n_test": n_test,
                "n_features_after_selection": n_feat,
                "n_features_allowed_by_frac": k_cap,
            }
    oof_df = pd.DataFrame(oof_rows, columns=["name", "y", "yhat", "eval_model"])
    return out, oof_df


def oof_sidecar_path(result_json: Path) -> Path:
    p = Path(result_json)
    return p.with_name(p.stem + ".oof.parquet")


def write_oof_parquet(path: Path, oof_df: pd.DataFrame, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = oof_df.copy()
    if extra:
        for k, v in extra.items():
            df[k] = v
    df.to_parquet(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "One fold: single selector step (stability, correlation, sfs, or rfe) + selection model, "
            "then optional fit/predict on multiple eval models."
        ),
    )
    p.add_argument(
        "--fold-dir",
        type=Path,
        required=True,
        help="Directory with meta.json and fold_{k}_train/test.parquet.",
    )
    p.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold index k (non-negative).",
    )
    p.add_argument(
        "--dataset-stem",
        default=None,
        help="Experimental table stem (column 3 in tab-separated jobs file).",
    )
    p.add_argument(
        "--pipeline-target-col",
        default=None,
        help=(
            "Optional: expected target column from the job line (e.g. master TSV col 4). "
            "If set, must equal meta.json target_col or the run aborts. "
            "The fold always uses meta.json; this flag only catches mismatched job rows."
        ),
    )
    p.add_argument(
        "--dataset-yaml-key",
        default=None,
        help="YAML dataset block key (e.g. dataset1); stored in JSON for aggregation.",
    )
    p.add_argument(
        "--selector-name",
        required=True,
        help=(
            'Single selector: stability, correlation, sfs, or rfe '
            '(no chained "->" pipelines).'
        ),
    )
    p.add_argument(
        "--model-to-use",
        required=True,
        help=(
            "elasticnet | randomforest | svm | knn (selection model; ignored for "
            "correlation-only; knn falls back to elasticnet for rfe)."
        ),
    )
    p.add_argument(
        "--eval-models",
        default=DEFAULT_EVAL_MODELS,
        help=(
            "After selection: fit on train / predict test with each listed model. "
            "Comma- or space-separated: linear, elasticnet, randomforest, svm, knn; "
            "or 'all' (default), or 'none' to skip."
        ),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Exact output path (overrides --result-dir).",
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help=(
            "Directory for result JSON when --output-json is omitted. "
            "Filename: [<dataset_stem>_]<fold_dir_basename>_fold{k}_<selector>_<model>.json"
        ),
    )
    p.add_argument(
        "--sfs-cv",
        type=int,
        default=None,
        help=(
            "Inner CV folds for forward SFS (omit or pass with no value for internal default: 5)."
        ),
    )
    p.add_argument(
        "--sfs-scoring",
        type=str,
        default=None,
        help="SFS scoring: spearman, r2, or neg_mean_squared_error (omit for internal default: spearman).",
    )
    p.add_argument(
        "--sfs-min-improvement",
        type=float,
        default=None,
        help=(
            "Minimum CV gain to add a feature in SFS (omit for internal default: 0.02)."
        ),
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=(
            "RNG seed for selectors and eval models. When meta.json contains random_state "
            "(from prepare_run / YAML), that value is used and this flag is ignored if it differs."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output from selectors (e.g. stability subsample progress).",
    )
    p.add_argument(
        "--correlation-min-abs-rho",
        type=str,
        default="none",
        help=(
            "For correlation selector: minimum |Spearman ρ| vs target after p/FDR screening "
            "(use 'none' for YAML default / no extra |ρ| cutoff). Master TSV column 11."
        ),
    )
    p.add_argument(
        "--eval-features-frac",
        type=str,
        default=None,
        help=(
            "Evaluation feature cap as fraction of training rows (overrides meta.json). "
            "Master TSV column 12 from prepare_parallel_from_config. Omit for legacy single-column runs."
        ),
    )
    p.add_argument(
        "--selector-hyperparameters",
        type=str,
        default="{}",
        help=(
            "JSON object of selector-specific overrides for run_feature_selection_on_one_fold "
            "(shared prefilter keys + per-selector params; see parse_selector_hyperparameters_mapping). "
            "Master TSV column 13."
        ),
    )
    p.add_argument(
        "--eval-hyperparameters",
        type=str,
        default="{}",
        help=(
            "JSON object: top-level keys linear, elasticnet, randomforest, svm, knn; each maps to "
            "hyperparameters for final eval make_regressor (omit keys for sklearn defaults). "
            "GPR keys: kernel_length_scale, n_restarts_optimizer, normalize_y, alpha. "
            "Master TSV column 14."
        ),
    )
    p.add_argument(
        "--pipeline-track-name",
        default=None,
        help=(
            "Optional pipeline track name (e.g. 'track_linear'). "
            "Stored in result JSON as pipeline_track_name for downstream filtering. "
            "Master TSV column 15."
        ),
    )
    args = p.parse_args()

    try:
        eval_models = _parse_eval_models(args.eval_models)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    fold_dir = Path(args.fold_dir).resolve()
    k = int(args.fold)
    if k < 0:
        print("--fold must be >= 0", file=sys.stderr)
        sys.exit(1)

    meta_path = fold_dir / "meta.json"
    train_pq = fold_dir / f"fold_{k}_train.parquet"
    test_pq = fold_dir / f"fold_{k}_test.parquet"

    for path, label in (
        (meta_path, "meta.json"),
        (train_pq, f"fold_{k}_train.parquet"),
        (test_pq, f"fold_{k}_test.parquet"),
    ):
        if not path.is_file():
            print(f"Missing {label}: {path}", file=sys.stderr)
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    cli_rs = int(args.random_state)
    meta_rs = meta.get("random_state")
    if meta_rs is not None:
        random_state = int(meta_rs)
        if cli_rs != random_state and not args.quiet:
            print(
                f"Using meta.json random_state={random_state} "
                f"(ignoring --random-state {cli_rs}).",
                file=sys.stderr,
            )
    else:
        random_state = cli_rs
        if not args.quiet:
            print(
                "Warning: meta.json has no random_state; using --random-state from CLI only. "
                "Re-run prepare_run so folds and selectors share one seed.",
                file=sys.stderr,
            )

    target_col = str(meta["target_col"])
    if args.pipeline_target_col is not None and str(args.pipeline_target_col) != target_col:
        print(
            f"Job line target {args.pipeline_target_col!r} != meta.json target_col {target_col!r} "
            f"(fold_dir={fold_dir})",
            file=sys.stderr,
        )
        sys.exit(1)
    features_frac = float(
        meta.get("features_frac", meta.get("sfs_n_features_fraction", 0.1))
    )
    if args.eval_features_frac is not None:
        ef = str(args.eval_features_frac).strip()
        if ef and ef.lower() not in ("none", "meta"):
            features_frac = float(ef)
    if "feature_cols" not in meta:
        print(
            "meta.json must contain feature_cols (from prepare_run.py); "
            "re-run prepare for this fold directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    candidate_features = [str(c) for c in meta["feature_cols"]]

    try:
        rho_t = str(args.correlation_min_abs_rho).strip().lower()
        if rho_t in ("", "none", "nan", "-", "null"):
            corr_min_rho = None
        else:
            corr_min_rho = float(rho_t)
    except ValueError:
        print(
            f"Invalid --correlation-min-abs-rho: {args.correlation_min_abs_rho!r} "
            "(use 'none' or a float).",
            file=sys.stderr,
        )
        sys.exit(1)

    train_df = pd.read_parquet(train_pq)
    test_df = pd.read_parquet(test_pq)

    n_train = len(train_df)
    selection_max_features = max(1, int(features_frac * n_train))
    if not args.quiet:
        print(
            f"[fold] {fold_dir.name} fold={k} n_train={n_train} "
            f"features_frac={features_frac} effective_cap={selection_max_features} "
            f"n_candidates={len(candidate_features)}",
            file=sys.stderr,
        )

    hp_json = str(args.selector_hyperparameters).strip()
    if not hp_json or hp_json.lower() in ("none",):
        hp_json = "{}"
    try:
        hp_obj = json.loads(hp_json)
    except json.JSONDecodeError as e:
        print(f"Invalid --selector-hyperparameters JSON: {e}", file=sys.stderr)
        sys.exit(1)
    sel_l = str(args.selector_name).strip().lower()
    try:
        hp_kwargs = parse_selector_hyperparameters_mapping(sel_l, hp_obj)
    except (ValueError, TypeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    ev_json = str(args.eval_hyperparameters).strip()
    if not ev_json or ev_json.lower() in ("none",):
        ev_json = "{}"
    try:
        ev_obj = json.loads(ev_json)
    except json.JSONDecodeError as e:
        print(f"Invalid --eval-hyperparameters JSON: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        eval_hp_by_model = parse_eval_hyperparameters_mapping(ev_obj)
    except (ValueError, TypeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    fs_kw = dict(
        train_df_k=train_df,
        test_df_k=test_df,
        target_col=target_col,
        feature_selection_pipeline=args.selector_name,
        model_type=args.model_to_use,
        candidate_features=candidate_features,
        correlation_screening_min_abs_rho=corr_min_rho,
        random_state=random_state,
        sfs_cv=args.sfs_cv,
        sfs_scoring=args.sfs_scoring,
        sfs_min_improvement=args.sfs_min_improvement,
        verbose=not args.quiet,
        selection_max_features=selection_max_features,
    )
    fs_kw.update(hp_kwargs)

    t_pipeline_start = time.monotonic()
    try:
        train_df_k, test_df_k, result = run_feature_selection_on_one_fold(**fs_kw)
    except (ValueError, OSError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    evaluation: dict[str, dict] | None = None
    oof_df = pd.DataFrame(columns=["name", "y", "yhat", "eval_model"])
    if eval_models is not None:
        evaluation, oof_df = _evaluate_fold_models(
            train_df_k,
            test_df_k,
            target_col=target_col,
            feature_cols=list(result.get("selected_features", [])),
            eval_models=eval_models,
            random_state=random_state,
            features_frac=features_frac,
            eval_hp_by_model=eval_hp_by_model,
        )
    pipeline_time_seconds = round(time.monotonic() - t_pipeline_start, 3)

    out_payload = {
        "fold_dir": str(fold_dir),
        "fold_index": k,
        "random_state": random_state,
        "selector_name": args.selector_name,
        "eval_models": eval_models,
        "eval_features_frac": features_frac,
        **result,
        "evaluation": evaluation,
        "pipeline_time_seconds": pipeline_time_seconds,
    }
    if hp_kwargs:
        out_payload["selector_hyperparameters"] = hp_kwargs
    if eval_hp_by_model:
        out_payload["eval_hyperparameters"] = eval_hp_by_model
    if args.dataset_stem:
        out_payload["dataset_stem"] = str(args.dataset_stem)
    if args.dataset_yaml_key:
        out_payload["dataset_yaml_key"] = str(args.dataset_yaml_key)
    pipeline_track = (args.pipeline_track_name or "").strip()
    if pipeline_track:
        out_payload["pipeline_track_name"] = pipeline_track
    if str(args.selector_name).strip().lower() == "correlation":
        out_payload["correlation_min_abs_rho"] = corr_min_rho

    out_path = args.output_json
    if out_path is None:
        sel_s = _slug(args.selector_name)
        mod_s = _slug(args.model_to_use)
        if args.dataset_stem:
            ds_s = _slug(str(args.dataset_stem))
            fname = f"{ds_s}_{fold_dir.name}_fold{k}_{sel_s}_{mod_s}.json"
        else:
            fname = f"{fold_dir.name}_fold{k}_{sel_s}_{mod_s}.json"
        if args.result_dir is not None:
            out_path = Path(args.result_dir) / fname
        else:
            out_path = fold_dir / f"result_fold{k}_{sel_s}_{mod_s}.json"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    if eval_models is not None and len(oof_df) > 0:
        write_oof_parquet(
            oof_sidecar_path(out_path),
            oof_df,
            extra={
                "fold_index": k,
                "dataset_yaml_key": str(args.dataset_yaml_key or ""),
                "dataset_stem": str(args.dataset_stem or ""),
                "target_col": target_col,
                "selector_name": str(args.selector_name),
                "model_type": str(result.get("model_type", args.model_to_use)),
                "eval_features_frac": float(features_frac),
                "pipeline_track_name": pipeline_track,
                "source_json": str(out_path),
            },
        )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
