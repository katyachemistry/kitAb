"""Fold-directory helpers shared by the AutoML runner and the final refit.

``prepare_run.py`` writes one directory per target under a dataset run dir::

    <run_dir>/<target>/meta.json
    <run_dir>/<target>/fold_{k}_train.parquet
    <run_dir>/<target>/fold_{k}_test.parquet
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NAME_COL = "name"


def read_fold_meta(fold_dir: Path) -> dict[str, Any]:
    meta_path = Path(fold_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing fold metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid fold metadata (not a mapping): {meta_path}")
    return meta


def discover_target_fold_dirs(run_dir: Path) -> dict[str, Path]:
    """Map pipeline target column -> fold directory for one prepared dataset."""
    out: dict[str, Path] = {}
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return out
    for child in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if not (child / "meta.json").is_file():
            continue
        meta = read_fold_meta(child)
        target_col = str(meta.get("target_col") or child.name)
        out[target_col] = child
    return out


def feature_columns(fold_dir: Path) -> list[str]:
    meta = read_fold_meta(fold_dir)
    features = [str(c) for c in (meta.get("feature_cols") or [])]
    if not features:
        raise ValueError(f"No feature_cols recorded in {fold_dir}/meta.json")
    return features


def fold_parquets_ready(fold_dir: Path) -> bool:
    fold_dir = Path(fold_dir)
    if not (fold_dir / "meta.json").is_file():
        return False
    try:
        n_splits = int(read_fold_meta(fold_dir)["n_splits"])
    except (KeyError, TypeError, ValueError):
        return False
    return all(
        (fold_dir / f"fold_{k}_train.parquet").is_file()
        and (fold_dir / f"fold_{k}_test.parquet").is_file()
        for k in range(n_splits)
    )


def full_dataset_frame(fold_dir: Path) -> pd.DataFrame:
    """All rows the folds were built from, deduplicated by sample name.

    In k-fold CV every sample appears in k-1 training folds, so concatenating
    the train parquets and dropping duplicates recovers the whole dataset.
    """
    fold_dir = Path(fold_dir)
    parts = [
        pd.read_parquet(path)
        for path in sorted(
            fold_dir.glob("fold_*_train.parquet"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
    ]
    if not parts:
        raise FileNotFoundError(f"No fold_*_train.parquet under {fold_dir}")
    frame = pd.concat(parts, ignore_index=True)
    if NAME_COL in frame.columns:
        return frame.drop_duplicates(subset=[NAME_COL], keep="first").reset_index(drop=True)
    return frame.drop_duplicates(keep="first").reset_index(drop=True)


def write_inner_fold_dir(orig_fold_dir: Path, *, outer_k: int, dest: Path) -> Path:
    """Rebuild inner folds from the outer-train folds, dropping outer-test names."""
    orig = Path(orig_fold_dir)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    meta = read_fold_meta(orig)
    n_splits = int(meta["n_splits"])
    if outer_k < 0 or outer_k >= n_splits:
        raise ValueError(f"outer_k={outer_k} out of range for n_splits={n_splits}")

    outer_test = pd.read_parquet(orig / f"fold_{outer_k}_test.parquet")
    outer_names = set(outer_test[NAME_COL].astype(str))
    inner_indices = [i for i in range(n_splits) if i != outer_k]
    segment_sizes: list[int] = []
    for new_k, old_k in enumerate(inner_indices):
        test_df = pd.read_parquet(orig / f"fold_{old_k}_test.parquet")
        train_df = pd.read_parquet(orig / f"fold_{old_k}_train.parquet")
        train_df = train_df.loc[~train_df[NAME_COL].astype(str).isin(outer_names)].copy()
        test_df = test_df.loc[~test_df[NAME_COL].astype(str).isin(outer_names)].copy()
        if len(test_df) < 2:
            raise ValueError(
                f"inner test fold too small after dropping outer-test names: "
                f"outer_k={outer_k} old_k={old_k} n_test={len(test_df)}"
            )
        if len(train_df) < 2:
            raise ValueError(
                f"inner train fold too small after dropping outer-test names: "
                f"outer_k={outer_k} old_k={old_k} n_train={len(train_df)}"
            )
        train_df.to_parquet(dest / f"fold_{new_k}_train.parquet", index=False)
        test_df.to_parquet(dest / f"fold_{new_k}_test.parquet", index=False)
        segment_sizes.append(int(len(test_df)))

    inner_meta = dict(meta)
    inner_meta["n_splits"] = len(inner_indices)
    inner_meta["segment_sizes"] = segment_sizes
    inner_meta["nested_outer_fold"] = int(outer_k)
    inner_meta["nested_source_fold_dir"] = str(orig.resolve())
    (dest / "meta.json").write_text(json.dumps(inner_meta, indent=2))
    return dest


def non_numeric_feature_columns(df: pd.DataFrame, feature_cols: Iterable[str]) -> list[str]:
    """Feature columns that are not numeric (need train-only dummy encoding)."""
    out: list[str] = []
    for col in feature_cols:
        name = str(col)
        if name not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[name]):
            out.append(name)
    return out


def encode_categoricals_train_only(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode non-numeric features from *train* levels only.

    Test-only categories are dropped (all-zero columns are not created). Dummy
    column names follow ``{col}__{level}`` with ``dummy_na=True``.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()
    expanded: list[str] = []
    for col in [str(c) for c in feature_cols]:
        if col not in train_out.columns:
            continue
        series = train_out[col]
        if pd.api.types.is_numeric_dtype(series):
            expanded.append(col)
            continue
        train_dummies = pd.get_dummies(
            series, prefix=col, prefix_sep="__", dtype=float, dummy_na=True
        )
        if col in test_out.columns:
            test_dummies = pd.get_dummies(
                test_out[col], prefix=col, prefix_sep="__", dtype=float, dummy_na=True
            )
        else:
            test_dummies = pd.DataFrame(index=test_out.index)
        test_dummies = test_dummies.reindex(columns=list(train_dummies.columns), fill_value=0.0)
        train_out = train_out.drop(columns=[col])
        test_out = test_out.drop(columns=[col], errors="ignore")
        train_out = pd.concat([train_out, train_dummies], axis=1)
        test_out = pd.concat([test_out, test_dummies], axis=1)
        expanded.extend(str(c) for c in train_dummies.columns)
    return train_out, test_out, expanded


def shuffled_row_folds(
    n_samples: int, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random contiguous segments after a shuffle (SFS fallback only)."""
    if n_splits < 2 or n_splits > n_samples:
        return []
    base_size = n_samples // n_splits
    remainder = n_samples % n_splits
    segment_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_splits)]
    boundaries = np.cumsum([0] + segment_sizes)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        val_start, val_end = int(boundaries[i]), int(boundaries[i + 1])
        val_idx = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


def fold_index_pairs_for_frame(
    fold_dir: Path,
    frame: pd.DataFrame,
    *,
    exclude_names: Iterable[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map a fold directory's test sets onto *frame* row positions.

    Outer-test (or other held-out) names in ``exclude_names`` never appear in
    train or val indices. Fold parquets that share no names with *frame*
    (already held out) are skipped. This reuses MMseqs2 / predefined group
    splits instead of shuffling rows.
    """
    if NAME_COL not in frame.columns:
        return []
    excluded = {str(n) for n in (exclude_names or [])}
    names = frame[NAME_COL].astype(str)
    leaked = {n for n in names if n in excluded}
    if leaked:
        raise ValueError(
            f"held-out names present in SFS/train frame: {sorted(leaked)[:12]}"
        )
    pos = {str(name): i for i, name in enumerate(names)}
    meta = read_fold_meta(fold_dir)
    try:
        n_splits = int(meta["n_splits"])
    except (KeyError, TypeError, ValueError):
        return []
    all_pos = np.arange(len(frame), dtype=int)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        test_path = Path(fold_dir) / f"fold_{k}_test.parquet"
        if not test_path.is_file():
            continue
        test_names = pd.read_parquet(test_path, columns=[NAME_COL])[NAME_COL].astype(str)
        val_idx = np.array(
            [pos[n] for n in test_names if n in pos and n not in excluded],
            dtype=int,
        )
        if len(val_idx) < 1:
            continue
        val_set = set(val_idx.tolist())
        train_idx = np.array([i for i in all_pos if i not in val_set], dtype=int)
        if len(train_idx) < 2:
            continue
        pairs.append((train_idx, val_idx))
    return pairs
