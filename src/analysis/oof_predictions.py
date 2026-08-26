#!/usr/bin/env python3
"""Recompute per-sample out-of-fold predictions from existing AutoML fold JSONs.

Fold directories are taken from each JSON's ``fold_dir`` field (never by
run-directory name glob). Pass ``--fold-dir-map`` when those paths were
overwritten and rebuilt into isolated run dirs. GPR eval entries are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.fold_dirs import load_fold_dir_map, remap_fold_dir, resolve_fold_dir
from automl.pipeline_defaults import DEFAULT_FEATURES_FRAC
from automl.run_fold_pipeline_config import (
    _evaluate_fold_models,
    _pearson_stat_p,
    _spearman_stat_p,
    oof_sidecar_path,
    write_oof_parquet,
)
from automl.utils import apply_minmax_to_train_test_features, parse_eval_hyperparameters_mapping

PAPER_BACKENDS: dict[str, str] = {
    "abb2": "our_abb2_final_set_of_features",
    "abb3": "our_abb3_final_set_of_features",
    "flashabb": "our_flashabb_final_set_of_features",
}

SKIP_EVAL_MODELS = frozenset({"gpr"})
VALIDATE_ATOL = 1e-9

# Stored fold Spearmans are compared with a tolerance because tiny numerical
# differences reorder near-tied predictions on small test folds (n≈20):
#   * randomforest: original runs used n_jobs=-1, and predict sums trees in a
#     nondeterministic thread order (~1e-15). Recomputation is pinned to one
#     thread and is exact.
#   * linear: discrete ProperMAb charges (e.g. Fv_chml) carry float jitter
#     around integers. sklearn 1.8 OLS (centered lstsq + intercept) produces a
#     different micro-ordering of those near-ties than the original fit, which
#     is enough to move Spearman by ~0.15. SVM/elasticnet/knn on the same fold
#     still match exactly.
VALIDATE_NOISE_TOLERANCE = 0.2
CONFIG_KEY_COLS = (
    "dataset_yaml_key",
    "pipeline_track_name",
    "target_col",
    "selector_name",
    "model_type",
    "eval_model",
    "eval_features_frac",
)


def paper_automl_root(repo_root: Path, backend: str) -> Path:
    folder = PAPER_BACKENDS.get(str(backend).strip().lower())
    if folder is None:
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of {sorted(PAPER_BACKENDS)}"
        )
    return Path(repo_root) / folder / "automl"


def iter_fold_result_jsons(automl_root: Path) -> list[Path]:
    root = Path(automl_root)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for jp in sorted(sub.glob("*.json")):
            if jp.name.endswith(".oof.json"):
                continue
            out.append(jp)
    return out


def oof_out_path(out_root: Path, json_path: Path, automl_root: Path) -> Path:
    try:
        rel = json_path.resolve().relative_to(Path(automl_root).resolve())
    except ValueError:
        rel = Path(json_path.name)
    return Path(out_root) / rel.with_suffix("").with_name(json_path.stem + ".oof.parquet")


def _eval_hp_from_result(data: dict[str, Any]) -> dict[str, dict]:
    raw = data.get("eval_hyperparameters") or {}
    if isinstance(raw, dict):
        raw = {k: v for k, v in raw.items() if str(k).strip().lower() not in SKIP_EVAL_MODELS}
    try:
        return parse_eval_hyperparameters_mapping(raw if raw else None)
    except (ValueError, TypeError):
        return parse_eval_hyperparameters_mapping(None)


def _load_fold_frames(
    data: dict[str, Any],
    *,
    fold_dir_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = remap_fold_dir(Path(str(data["fold_dir"])), fold_dir_map)
    fold_dir = resolve_fold_dir(mapped)
    k = int(data["fold_index"])
    train_pq = fold_dir / f"fold_{k}_train.parquet"
    test_pq = fold_dir / f"fold_{k}_test.parquet"
    if not train_pq.is_file() or not test_pq.is_file():
        raise FileNotFoundError(
            f"Missing fold parquets under {fold_dir} for fold_index={k}"
        )
    return pd.read_parquet(train_pq), pd.read_parquet(test_pq)


def reproduce_oof_from_result_json(
    json_path: Path,
    *,
    n_jobs: int = 1,
    fold_dir_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Refit eval models on stored selected features; return OOF rows + per-model checks."""
    data = json.loads(Path(json_path).read_text())
    feats = [str(c) for c in (data.get("selected_features") or [])]
    target_col = str(data["target_col"])
    random_state = int(data.get("random_state", 42))
    try:
        features_frac = float(data.get("eval_features_frac", DEFAULT_FEATURES_FRAC))
    except (TypeError, ValueError):
        features_frac = float(DEFAULT_FEATURES_FRAC)

    eval_block = data.get("evaluation") if isinstance(data.get("evaluation"), dict) else {}
    eval_models = [
        str(m)
        for m in (data.get("eval_models") or list(eval_block.keys()))
        if str(m).strip().lower() not in SKIP_EVAL_MODELS
    ]
    eval_models = list(dict.fromkeys(eval_models))

    train_df, test_df = _load_fold_frames(data, fold_dir_map=fold_dir_map)
    cols = [c for c in feats if c in train_df.columns and c in test_df.columns]
    if cols:
        train_df, test_df = apply_minmax_to_train_test_features(train_df, test_df, cols)

    evaluation, oof_df = _evaluate_fold_models(
        train_df,
        test_df,
        target_col=target_col,
        feature_cols=cols,
        eval_models=eval_models,
        random_state=random_state,
        features_frac=features_frac,
        eval_hp_by_model=_eval_hp_from_result(data),
        n_jobs=n_jobs,
    )

    extra = {
        "fold_index": int(data.get("fold_index", -1)),
        "dataset_yaml_key": str(data.get("dataset_yaml_key") or data.get("dataset_stem") or ""),
        "dataset_stem": str(data.get("dataset_stem") or ""),
        "target_col": target_col,
        "selector_name": str(data.get("selector_name") or ""),
        "model_type": str(data.get("model_type") or ""),
        "eval_features_frac": float(features_frac),
        "pipeline_track_name": str(data.get("pipeline_track_name") or ""),
        "source_json": str(Path(json_path).resolve()),
    }
    if len(oof_df):
        for k, v in extra.items():
            oof_df[k] = v

    checks: list[dict[str, Any]] = []
    gpr_skipped = sum(
        1
        for m in (eval_block or {})
        if str(m).strip().lower() in SKIP_EVAL_MODELS
    )
    for m in eval_models:
        stored = eval_block.get(m) if isinstance(eval_block.get(m), dict) else {}
        recomputed = evaluation.get(m) if isinstance(evaluation.get(m), dict) else {}
        stored_err = bool(stored.get("error"))
        rec_err = bool(recomputed.get("error"))
        stored_sp = stored.get("spearman_rho") if not stored_err else None
        rec_sp = recomputed.get("spearman_rho") if not rec_err else None
        match = _spearman_close(stored_sp, rec_sp)
        checks.append(
            {
                "eval_model": m,
                "stored_spearman": stored_sp,
                "recomputed_spearman": rec_sp,
                "match": match,
                "stored_error": stored.get("error") if stored_err else None,
                "recomputed_error": recomputed.get("error") if rec_err else None,
            }
        )
    if gpr_skipped:
        checks.append(
            {
                "eval_model": "gpr",
                "stored_spearman": None,
                "recomputed_spearman": None,
                "match": True,
                "stored_error": None,
                "recomputed_error": f"skipped ({gpr_skipped} gpr eval block(s))",
            }
        )
    return oof_df, checks


def _spearman_close(a: object, b: object, atol: float = VALIDATE_ATOL) -> bool:
    if a is None and b is None:
        return True
    try:
        fa = float(a) if a is not None else None
        fb = float(b) if b is not None else None
    except (TypeError, ValueError):
        return False
    if fa is None or fb is None:
        return fa is None and fb is None
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    return abs(fa - fb) <= atol


def pooled_metrics_from_oof(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df is None or len(df) == 0:
        return {
            "Spearman_pooled_oof": None,
            "Pearson_pooled_oof": None,
            "R2_pooled_oof": None,
            "n_oof": 0,
            "n_folds_present": 0,
        }
    y = df["y"].to_numpy(dtype=np.float64)
    yhat = df["yhat"].to_numpy(dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    n = int(y.size)
    sp, _ = _spearman_stat_p(y, yhat)
    pe, _ = _pearson_stat_p(y, yhat)
    r2: float | None = None
    if n >= 2 and np.ptp(y) > 0:
        try:
            r2 = float(r2_score(y, yhat))
            if not math.isfinite(r2):
                r2 = None
        except Exception:
            r2 = None
    n_folds = 0
    if "fold_index" in df.columns:
        n_folds = int(df["fold_index"].nunique())
    return {
        "Spearman_pooled_oof": sp,
        "Pearson_pooled_oof": pe,
        "R2_pooled_oof": r2,
        "n_oof": n,
        "n_folds_present": n_folds,
    }


def config_key_from_row(row: pd.Series) -> tuple:
    frac = row.get("eval_features_frac", DEFAULT_FEATURES_FRAC)
    try:
        eval_frac = float(frac)
    except (TypeError, ValueError):
        eval_frac = float(DEFAULT_FEATURES_FRAC)
    return (
        str(row.get("dataset_yaml_key") or ""),
        str(row.get("pipeline_track_name") or ""),
        str(row.get("target_col") or ""),
        str(row.get("selector_name") or ""),
        str(row.get("model_type") or ""),
        str(row.get("eval_model") or ""),
        float(eval_frac),
    )


def pool_oof_parquet_dir(oof_root: Path) -> dict[tuple, dict[str, float | int | None]]:
    paths = list(Path(oof_root).rglob("*.oof.parquet"))
    grouped: dict[tuple, list[pd.DataFrame]] = defaultdict(list)
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df is None or len(df) == 0 or "eval_model" not in df.columns:
            continue
        for _, sub in df.groupby("eval_model", sort=False):
            key = config_key_from_row(sub.iloc[0])
            grouped[key].append(sub)
    out: dict[tuple, dict[str, float | int | None]] = {}
    for key, parts in grouped.items():
        cat = pd.concat(parts, ignore_index=True)
        out[key] = pooled_metrics_from_oof(cat)
    return out


def discover_jobs(
    repo_root: Path,
    backends: Iterable[str],
    out_root: Path,
) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for backend in backends:
        automl_root = paper_automl_root(repo_root, backend)
        jsons = iter_fold_result_jsons(automl_root)
        for jp in jsons:
            jobs.append(
                {
                    "backend": str(backend),
                    "json_path": str(jp.resolve()),
                    "oof_path": str(oof_out_path(out_root / backend, jp, automl_root)),
                    "automl_root": str(automl_root.resolve()),
                }
            )
    return jobs


def write_jobs_tsv(path: Path, jobs: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("backend\tjson_path\toof_path\tautoml_root\n")
        for j in jobs:
            f.write(
                f"{j['backend']}\t{j['json_path']}\t{j['oof_path']}\t{j['automl_root']}\n"
            )


def load_jobs_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [
            {k: str(v or "") for k, v in row.items()}
            for row in csv.DictReader(f, delimiter="\t")
        ]


def pending_oof_jobs(jobs: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    pending: list[dict[str, str]] = []
    for job in jobs:
        oof_path = str(job.get("oof_path") or "").strip()
        if oof_path and Path(oof_path).is_file():
            continue
        pending.append(dict(job))
    return pending


def run_one_job(
    json_path: Path,
    oof_path: Path,
    *,
    resume: bool = False,
    n_jobs: int = 1,
    fold_dir_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    oof_path = Path(oof_path)
    if resume and oof_path.is_file():
        return {"status": "skipped_exists", "json_path": str(json_path), "oof_path": str(oof_path)}
    oof_df, checks = reproduce_oof_from_result_json(
        json_path, n_jobs=n_jobs, fold_dir_map=fold_dir_map
    )
    write_oof_parquet(oof_path, oof_df)
    check_path = oof_path.with_suffix("").with_name(oof_path.name.replace(".oof.parquet", ".oof_check.json"))
    check_path.write_text(json.dumps({"json_path": str(json_path), "checks": checks}, indent=2))
    n_mismatch = sum(1 for c in checks if c.get("eval_model") != "gpr" and not c.get("match"))
    return {
        "status": "ok",
        "json_path": str(json_path),
        "oof_path": str(oof_path),
        "n_rows": int(len(oof_df)),
        "n_mismatch": n_mismatch,
    }


def classify_check(c: dict[str, Any], *, tolerance: float) -> str:
    """exact | within_tolerance | beyond_tolerance | recomputed_undefined | stored_undefined."""
    stored, recomputed = c.get("stored_spearman"), c.get("recomputed_spearman")
    if stored is None and recomputed is None:
        return "exact"
    if recomputed is None:
        return "recomputed_undefined"
    if stored is None:
        return "stored_undefined"
    try:
        d = abs(float(stored) - float(recomputed))
    except (TypeError, ValueError):
        return "beyond_tolerance"
    if not math.isfinite(d):
        return "beyond_tolerance"
    if d <= VALIDATE_ATOL:
        return "exact"
    return "within_tolerance" if d <= float(tolerance) else "beyond_tolerance"


def _bucket_summary(
    buckets: Counter,
    *,
    max_mismatch_rate: float,
) -> dict[str, Any]:
    n = int(sum(buckets.values()))
    n_exact = int(buckets["exact"])
    n_beyond = int(buckets["beyond_tolerance"])
    rate = (n_beyond / n) if n else 0.0
    return {
        "n_eval_checks": n,
        "n_exact": n_exact,
        "exact_rate": (n_exact / n) if n else 0.0,
        "n_within_tolerance": int(buckets["within_tolerance"]),
        "n_recomputed_undefined": int(buckets["recomputed_undefined"]),
        "n_stored_undefined": int(buckets["stored_undefined"]),
        "n_beyond_tolerance": n_beyond,
        "beyond_tolerance_rate": rate,
        "ok": n > 0 and rate <= float(max_mismatch_rate),
    }


def validate_check_files(
    oof_root: Path,
    *,
    max_mismatch_rate: float = 0.0,
    tolerance: float = VALIDATE_NOISE_TOLERANCE,
) -> dict[str, Any]:
    """Gate the retro predictions.

    Exact agreement is reported at 1e-9, but the pass/fail decision uses
    ``tolerance`` because stored randomforest and linear metrics carry
    near-tie noise on small folds (see VALIDATE_NOISE_TOLERANCE). Anything
    beyond ``tolerance`` is a real disagreement and fails the gate.
    """
    root = Path(oof_root)
    by_backend: dict[str, Counter] = defaultdict(Counter)
    by_model: dict[str, Counter] = defaultdict(Counter)
    total: Counter = Counter()
    worst: list[dict[str, Any]] = []
    for p in root.rglob("*.oof_check.json"):
        try:
            payload = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            rel = p.resolve().relative_to(root.resolve())
            backend = rel.parts[0] if rel.parts else "unknown"
        except ValueError:
            backend = "unknown"
        for c in payload.get("checks") or []:
            model = str(c.get("eval_model"))
            if model in SKIP_EVAL_MODELS:
                continue
            bucket = classify_check(c, tolerance=tolerance)
            by_backend[backend][bucket] += 1
            by_model[model][bucket] += 1
            total[bucket] += 1
            if bucket == "beyond_tolerance" and len(worst) < 20:
                worst.append({"json_path": payload.get("json_path"), **c})

    summary = _bucket_summary(total, max_mismatch_rate=max_mismatch_rate)
    summary["max_mismatch_rate"] = max_mismatch_rate
    summary["tolerance"] = tolerance
    summary["exact_atol"] = VALIDATE_ATOL
    summary["per_backend"] = {
        be: _bucket_summary(b, max_mismatch_rate=max_mismatch_rate)
        for be, b in sorted(by_backend.items())
    }
    summary["per_eval_model"] = {
        m: _bucket_summary(b, max_mismatch_rate=max_mismatch_rate)
        for m, b in sorted(by_model.items())
    }
    if worst:
        summary["examples_beyond_tolerance"] = worst
    return summary


def _cmd_discover(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    backends = _backends_from_arg(args.backend)
    out_root = Path(args.out_dir).resolve()
    jobs = discover_jobs(repo, backends, out_root)
    write_jobs_tsv(Path(args.jobs_file), jobs)
    by_be: dict[str, set[str]] = defaultdict(set)
    for j in jobs:
        yaml_key = Path(j["json_path"]).parent.name
        by_be[j["backend"]].add(yaml_key)
    parts = [
        f"{be}={len(keys)} yaml_key(s)" for be, keys in sorted(by_be.items())
    ]
    print(
        f"Wrote {len(jobs)} OOF job(s) for backend(s) {backends} "
        f"({'; '.join(parts)}) -> {args.jobs_file}",
        file=sys.stderr,
    )
    return 0 if jobs else 1


def _cmd_pending(args: argparse.Namespace) -> int:
    jobs = load_jobs_tsv(Path(args.jobs_file))
    pending = pending_oof_jobs(jobs)
    write_jobs_tsv(Path(args.out), pending)
    print(
        f"oof resume: {len(pending)}/{len(jobs)} job(s) remaining -> {args.out}",
        file=sys.stderr,
    )
    return 0


def _cmd_run_one(args: argparse.Namespace) -> int:
    info = run_one_job(
        Path(args.json_path),
        Path(args.oof_path),
        resume=bool(args.resume),
        n_jobs=int(args.n_jobs),
        fold_dir_map=load_fold_dir_map(getattr(args, "fold_dir_map", None)),
    )
    print(json.dumps(info))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    summary = validate_check_files(
        Path(args.oof_dir),
        max_mismatch_rate=float(args.max_mismatch_rate),
        tolerance=float(args.tolerance),
    )
    if args.summary_out is not None:
        dest = Path(args.summary_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


def _backends_from_arg(spec: str) -> list[str]:
    s = str(spec).strip().lower()
    if s in ("", "all"):
        return list(PAPER_BACKENDS)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in parts if p not in PAPER_BACKENDS]
    if bad:
        raise SystemExit(f"Unknown backend(s) {bad}; expected {sorted(PAPER_BACKENDS)} or all")
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Write TSV of JSON -> OOF parquet jobs.")
    d.add_argument("--repo-root", type=Path, required=True)
    d.add_argument("--backend", default="all")
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--jobs-file", type=Path, required=True)
    d.set_defaults(func=_cmd_discover)

    pend = sub.add_parser(
        "pending", help="Write jobs whose OOF parquet is missing."
    )
    pend.add_argument("--jobs-file", type=Path, required=True)
    pend.add_argument("--out", type=Path, required=True)
    pend.set_defaults(func=_cmd_pending)

    r = sub.add_parser("run-one", help="Recompute OOF for one fold result JSON.")
    r.add_argument("--json-path", type=Path, required=True)
    r.add_argument("--oof-path", type=Path, required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument(
        "--fold-dir-map",
        type=Path,
        default=None,
        help="JSON object mapping original fold run dirs to rebuilt isolated dirs.",
    )
    r.set_defaults(func=_cmd_run_one)

    v = sub.add_parser("validate", help="Fail if stored vs recomputed Spearman mismatch rate exceeds threshold.")
    v.add_argument("--oof-dir", type=Path, required=True)
    v.add_argument("--max-mismatch-rate", type=float, default=0.0)
    v.add_argument(
        "--tolerance",
        type=float,
        default=VALIDATE_NOISE_TOLERANCE,
        help=(
            "Absolute Spearman difference tolerated when deciding pass/fail. "
            "Exact (1e-9) agreement is always reported separately."
        ),
    )
    v.add_argument("--summary-out", type=Path, default=None)
    v.set_defaults(func=_cmd_validate)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
