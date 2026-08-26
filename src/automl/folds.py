"""Fold-directory helpers shared by the AutoML runner and the final refit.

``prepare_run.py`` writes one directory per target under a dataset run dir::

    <run_dir>/<target>/meta.json
    <run_dir>/<target>/fold_{k}_train.parquet
    <run_dir>/<target>/fold_{k}_test.parquet
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

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
