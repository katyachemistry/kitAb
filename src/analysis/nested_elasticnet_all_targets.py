#!/usr/bin/env python3
"""Nested Elastic Net on all targets in a sequence-aware fold root.

For each outer fold, alpha and l1_ratio are selected by pooled Spearman across
the predefined inner validation folds. Scaling is fitted only on the
corresponding training split; missing feature values raise an error. The
selected model is then refitted on the complete outer-training set and
evaluated on the untouched outer test set.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.nested_cv import write_inner_fold_dir

REPO = Path("/storage/antibody_data/PairedStructures/kitAb")
DEFAULT_FOLD_ROOT = (
    REPO
    / "runs/jain2017biophysical_folded_08_5_cv_prepare__"
    "our_abb2_final_set_of_features_descriptors_jain2017biophysical_folded_08_5_abb2_1_results"
)
DEFAULT_ALPHAS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_L1_RATIOS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


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


def _xy(
    df: pd.DataFrame, target_col: str, features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    x = df.loc[mask, features].apply(pd.to_numeric, errors="coerce")
    return x, y.loc[mask]


class _RejectMissing(BaseEstimator, TransformerMixin):
    """Fail fast if any feature value is missing; do not impute."""

    def fit(self, X, y=None):
        self._check(X, stage="fit")
        return self

    def transform(self, X):
        self._check(X, stage="transform")
        return X

    @staticmethod
    def _check(X, *, stage: str) -> None:
        arr = np.asarray(X, dtype=float)
        n_missing = int(np.isnan(arr).sum())
        if n_missing:
            raise ValueError(
                f"Missing feature values are not allowed "
                f"({n_missing} NaN(s) during elastic-net {stage})"
            )


def _make_model(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline(
        [
            ("reject_missing", _RejectMissing()),
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=100_000,
                    tol=1e-5,
                    random_state=42,
                ),
            ),
        ]
    )


def _run_outer(
    fold_dir: Path,
    *,
    target_col: str,
    outer_k: int,
    features: list[str],
    alphas: list[float],
    l1_ratios: list[float],
    work_root: Path,
) -> dict[str, Any]:
    outer_train = pd.read_parquet(fold_dir / f"fold_{outer_k}_train.parquet")
    outer_test = pd.read_parquet(fold_dir / f"fold_{outer_k}_test.parquet")
    inner_dir = work_root / target_col / f"outer_{outer_k}" / "inner_folds"
    write_inner_fold_dir(fold_dir, outer_k=outer_k, dest=inner_dir)
    n_inner = int(json.loads((inner_dir / "meta.json").read_text())["n_splits"])

    grid_scores: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                ys: list[pd.Series] = []
                yhats: list[pd.Series] = []
                fold_scores: list[float | None] = []
                for inner_k in range(n_inner):
                    train = pd.read_parquet(
                        inner_dir / f"fold_{inner_k}_train.parquet"
                    )
                    val = pd.read_parquet(inner_dir / f"fold_{inner_k}_test.parquet")
                    x_train, y_train = _xy(train, target_col, features)
                    x_val, y_val = _xy(val, target_col, features)
                    model = _make_model(alpha, l1_ratio)
                    model.fit(x_train, y_train)
                    pred = pd.Series(model.predict(x_val), index=y_val.index)
                    ys.append(y_val)
                    yhats.append(pred)
                    fold_scores.append(_spearman(y_val, pred))
                pooled = _spearman(
                    pd.concat(ys, ignore_index=True),
                    pd.concat(yhats, ignore_index=True),
                )
                grid_scores.append(
                    {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "pooled_inner_spearman": pooled,
                        "inner_fold_spearman": fold_scores,
                    }
                )

        valid = [r for r in grid_scores if r["pooled_inner_spearman"] is not None]
        if not valid:
            raise RuntimeError(
                f"No valid inner score for {target_col}, outer fold {outer_k}"
            )
        best = max(
            valid,
            key=lambda r: (
                r["pooled_inner_spearman"],
                r["alpha"],
                r["l1_ratio"],
            ),
        )
        x_train, y_train = _xy(outer_train, target_col, features)
        x_test, y_test = _xy(outer_test, target_col, features)
        model = _make_model(best["alpha"], best["l1_ratio"])
        model.fit(x_train, y_train)
        pred = pd.Series(model.predict(x_test), index=y_test.index)

    enet = model.named_steps["elasticnet"]
    outer_spearman = _spearman(y_test, pred)
    print(
        f"{target_col} outer {outer_k}: rho={outer_spearman} "
        f"alpha={best['alpha']} l1_ratio={best['l1_ratio']}",
        flush=True,
    )
    return {
        "target_col": target_col,
        "outer_fold": outer_k,
        "n_test": len(y_test),
        "spearman": outer_spearman,
        "alpha": best["alpha"],
        "l1_ratio": best["l1_ratio"],
        "inner_pooled_spearman": best["pooled_inner_spearman"],
        "n_nonzero": int(np.count_nonzero(enet.coef_)),
        "grid_scores": grid_scores,
        "oof_rows": [
            {
                "name": name,
                "target_col": target_col,
                "outer_fold": outer_k,
                "y": float(y),
                "yhat": float(yhat),
            }
            for name, y, yhat in zip(
                outer_test.loc[y_test.index, "name"].astype(str), y_test, pred
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Target directory names (default: every target_* directory).",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument(
        "--l1-ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_L1_RATIOS),
    )
    parser.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    targets = args.targets or sorted(
        p.name
        for p in args.fold_root.glob("target_*")
        if p.is_dir() and (p / "meta.json").is_file()
    )
    if not targets:
        raise FileNotFoundError(f"No target fold directories under {args.fold_root}")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if any(a <= 0 for a in args.alphas):
        parser.error("--alphas must all be > 0")
    if any(not 0 <= r <= 1 for r in args.l1_ratios):
        parser.error("--l1-ratios must be between 0 and 1")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or REPO / f"runs/nested_elasticnet_jain2017_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "work"

    tasks: list[tuple[Path, str, int, list[str]]] = []
    for target in targets:
        fold_dir = args.fold_root / target
        n_outer = int(json.loads((fold_dir / "meta.json").read_text())["n_splits"])
        meta = json.loads((fold_dir / "meta.json").read_text())
        sample = pd.read_parquet(fold_dir / "fold_0_train.parquet")
        features = [
            str(feature)
            for feature in meta.get("feature_cols", [])
            if str(feature) in sample.columns
        ]
        if not features:
            features = _feature_columns(sample, target)
        for outer_k in range(n_outer):
            tasks.append((fold_dir, target, outer_k, features))

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.jobs, len(tasks)),
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(
                _run_outer,
                fold_dir,
                target_col=target,
                outer_k=outer_k,
                features=features,
                alphas=list(args.alphas),
                l1_ratios=list(args.l1_ratios),
                work_root=work_root,
            ): (target, outer_k)
            for fold_dir, target, outer_k, features in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: (r["target_col"], r["outer_fold"]))

    oof = pd.DataFrame(
        row for result in results for row in result.pop("oof_rows")
    )
    summary_rows: list[dict[str, Any]] = []
    for target in targets:
        target_results = [r for r in results if r["target_col"] == target]
        target_oof = oof[oof["target_col"] == target]
        row: dict[str, Any] = {
            "Dataset_stem": "jain2017biophysical_folded_08_5",
            "Target_col": target,
            "nested_Spearman_pooled_oof": _spearman(
                target_oof["y"], target_oof["yhat"]
            ),
            "n_oof": len(target_oof),
        }
        for result in target_results:
            k = result["outer_fold"]
            row[f"outer_{k}_spearman"] = result["spearman"]
            row[f"outer_{k}_alpha"] = result["alpha"]
            row[f"outer_{k}_l1_ratio"] = result["l1_ratio"]
            row[f"outer_{k}_n_nonzero"] = result["n_nonzero"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    oof.to_parquet(out_dir / "oof.parquet", index=False)
    (out_dir / "inner_selection.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "fold_root": str(args.fold_root),
                "targets": targets,
                "alphas": args.alphas,
                "l1_ratios": args.l1_ratios,
                "jobs": args.jobs,
            },
            indent=2,
        )
    )
    print("\n" + summary.to_string(index=False), flush=True)
    print(f"\nWrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
