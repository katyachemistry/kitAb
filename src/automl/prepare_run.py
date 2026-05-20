#!/usr/bin/env python3
"""
Prepare parallel automl runs: merge developability once, then for each target write
CV fold train/test parquet files and ``meta.json`` for fold workers.

Loads experimental table, requires no NaNs in ID (targets may be NaN).

**Experimental vs developability features:** The united developability table is loaded
first; candidate developability feature columns are chosen from its schema (optionally
restricted by ``--developability-feature-groups``). Experimental columns are chosen from
the experimental table (``--feature-cols``). Only those columns are inner-merged on ID,
so the join is never a full Cartesian column explosion. Optional ``--feature-cols`` are
**in addition to** developability features. Group flags follow JSON top-level namespaces
from ``calculate_descriptors`` (``surface``, ``core``, ``general``, ``sequence_motives``).
Omit group flags to use all developability columns. Ignored for Propermab ``.csv``
(with a warning).

If ``--feature-cols`` is omitted or empty, ML features are **only** the selected
developability columns.
Inner-merges developability (JSON folder or propermab CSV), one-hot-encodes
listed non-numeric features, then per target drops invalid target rows, renames columns
builds the pipeline table (same column naming as fold workers), and splits with either
``cv_shuffled_fold_ilocs`` (default) or leave-one-fold-out from ``--split-col`` on the
experimental table (fold count = number of distinct labels, at least 2).

From repo root (``developability`` conda env)::

    conda run -n developability python src/automl/prepare_run.py data/exp.csv \\
        --name-col name --target-cols tm1 aggregation \\
        --feature-cols hc_subtype lc_subtype some_descriptor \\
        --developability-results path/to/json_or_csv \\
        --output-dir runs/myexp

Optional ``--jobs-file`` appends one tab-separated line per fold:
``fold_dir<TAB>k<TAB>dataset_stem<TAB>pipeline_target_col`` for GNU parallel.

For disposable fold parquets, use a temp directory under the repository root (e.g.
``<repo>/tmp/...``); ``src/prepare_then_parallel.sh`` defaults to ``mktemp`` there.

``src`` is prepended to ``sys.path`` like ``calculate_descriptors.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from automl.feature_selectors import cv_shuffled_fold_ilocs, cv_split_col_ilocs
from utils.load_results_to_dataframe import load_json_results


def drop_invalid_target_rows(
    df: pd.DataFrame,
    target_col: str,
    *,
    max_nan_frac: float = 0.5,
) -> pd.DataFrame:
    """Fail if fraction of NaN in ``target_col`` is ≥ ``max_nan_frac``; else warn and drop NaN rows."""
    if not (0.0 < float(max_nan_frac) <= 1.0):
        raise ValueError("max_nan_frac must be in (0, 1].")
    target_col = str(target_col)
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not in dataframe columns.")

    y = df[target_col]
    n = len(df)
    if n == 0:
        raise ValueError("Dataset is empty after prior steps.")

    n_nan = int(y.isna().sum())
    if n_nan == 0:
        return df

    frac = n_nan / n
    mnf = float(max_nan_frac)
    if frac >= mnf:
        raise ValueError(
            f"target_col {target_col!r}: {n_nan}/{n} rows ({frac:.0%}) are NaN; "
            f"require strictly fewer than {mnf:.0%} NaN (see --max-target-nan-frac)."
        )

    warnings.warn(
        f"target_col {target_col!r}: dropping {n_nan}/{n} rows with NaN ({frac:.0%}).",
        UserWarning,
        stacklevel=2,
    )
    return df.loc[y.notna()].copy()


def require_no_nans_in_dataframe(df: pd.DataFrame, *, context: str) -> None:
    """Raise if any column in ``df`` contains NaN."""
    if df.empty:
        return
    issues: list[str] = []
    for c in df.columns:
        if df[c].isna().any():
            n_nan = int(df[c].isna().sum())
            issues.append(f"{str(c)!r} ({n_nan} NaN)")
    if not issues:
        return
    detail = ", ".join(issues[:15])
    more = f" … and {len(issues) - 15} more column(s)" if len(issues) > 15 else ""
    raise ValueError(
        f"{context}: NaN values in {len(issues)} column(s): {detail}{more}"
    )


def require_no_nans_except_columns(
    df: pd.DataFrame,
    *,
    skip_columns: set[str],
    context: str,
) -> None:
    """Raise if any column outside ``skip_columns`` has NaN."""
    skip = {str(c) for c in skip_columns}
    sub = df[[c for c in df.columns if str(c) not in skip]]
    if sub.empty:
        return
    require_no_nans_in_dataframe(sub, context=context)


def standardize_pipeline_column_names(
    merged_df: pd.DataFrame,
    *,
    name_col: str,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, str, list[str]]:
    """Rename ID to ``name``, response to ``target_*``, features to ``feature_*``."""
    name_col = str(name_col)
    target_col = str(target_col)
    feats_in = [str(c) for c in feature_cols]

    rename_map: dict[str, str] = {}
    if name_col != "name":
        rename_map[name_col] = "name"

    new_target = target_col if target_col.startswith("target_") else f"target_{target_col}"
    if new_target != target_col:
        rename_map[target_col] = new_target

    new_feats: list[str] = []
    for f in feats_in:
        nf = f if f.startswith("feature_") else f"feature_{f}"
        new_feats.append(nf)
        if nf != f:
            rename_map[f] = nf

    out = merged_df.rename(columns=rename_map, copy=True)
    return out, new_target, new_feats


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".csv", ".txt") or suf == "":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset extension {path.suffix!r}; use .csv or .parquet.")


def _name_from_pdb_file_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("\\", "/", regex=False)
    return s.str.split("/").str[-1].str.split(".").str[0]


def load_propermab_features_csv(features_path: Path) -> pd.DataFrame:
    features_path = Path(features_path)
    feat_df = pd.read_csv(features_path)
    feat_df = feat_df.copy()
    if "pdb_file" in feat_df.columns:
        feat_df["name"] = _name_from_pdb_file_column(feat_df["pdb_file"])
    elif "name" in feat_df.columns:
        # Also support generic feature CSVs that already carry merge key ``name``.
        feat_df["name"] = feat_df["name"].astype(str)
    else:
        raise ValueError(
            f"CSV {features_path} has neither 'pdb_file' nor 'name' column "
            "(need one merge key to match experimental IDs)."
        )
    feat_df = feat_df.drop(columns=["pdb_file", "error"], errors="ignore")
    return feat_df


def one_hot_listed_non_numeric_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    new_order: list[str] = []
    for col in [str(c) for c in feature_cols]:
        if col not in out.columns:
            continue
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            new_order.append(col)
            continue
        dummies = pd.get_dummies(
            s, prefix=col, prefix_sep="__", dtype=float, dummy_na=True
        )
        out = out.drop(columns=[col])
        out = pd.concat([out, dummies], axis=1)
        new_order.extend(dummies.columns.tolist())
    return out, new_order


def _normalize_merge_keys(left: pd.DataFrame, name_col: str, right: pd.DataFrame) -> None:
    for df, col in ((left, name_col), (right, "name")):
        if col not in df.columns:
            continue
        s = df[col]
        try:
            df[col] = s.astype(int)
        except (ValueError, TypeError):
            pass
        df[col] = df[col].astype(str)


def load_developability_dataframe(
    developability_source: Path,
) -> tuple[pd.DataFrame, str]:
    """
    Load the united developability table (JSON directory or Propermab CSV).

    Returns ``(df_dev, id_hint)`` where ``df_dev`` has a ``name`` column for merging.
    """
    src = Path(developability_source)
    if src.is_dir():
        df_dev = load_json_results(src)
        if "name" not in df_dev.columns:
            raise ValueError(
                f"Developability JSON folder {src} has no 'name' column "
                "(expected from JSON filename stems)."
            )
        id_hint = f"JSON filename stems under {src}"
    elif src.is_file() and src.suffix.lower() == ".csv":
        df_dev = load_propermab_features_csv(src)
        id_hint = f"pdb_file basename or name in {src}"
    else:
        raise ValueError(
            f"developability_source must be a directory of *.json files or a .csv path; got {src}"
        )
    return df_dev, id_hint


def merge_experimental_with_developability_subset(
    experimental_df: pd.DataFrame,
    developability_df: pd.DataFrame,
    *,
    name_col: str,
    id_hint: str,
    developability_columns: list[str],
) -> pd.DataFrame:
    """
    Inner-merge experimental rows to developability features using only
    ``developability_columns`` from the developability frame (plus ``name``).
    """
    name_col = str(name_col)
    if name_col not in experimental_df.columns:
        raise ValueError(f"name_col {name_col!r} not in experimental dataset columns.")

    dev_cols_ordered = list(dict.fromkeys(developability_columns))
    miss_dev = [c for c in dev_cols_ordered if c not in developability_df.columns]
    if miss_dev:
        raise ValueError(
            "Developability table is missing column(s) selected for features: "
            f"{miss_dev!r}"
        )

    right = developability_df.loc[:, ["name", *dev_cols_ordered]].copy()
    left = experimental_df.copy()
    _normalize_merge_keys(left, name_col, right)

    if name_col == "name":
        merged = left.merge(right, on="name", how="inner")
    else:
        merged = left.merge(
            right,
            left_on=name_col,
            right_on="name",
            how="inner",
        )
        merged = merged.drop(columns=["name"], errors="ignore")

    if merged.empty:
        raise ValueError(
            "Inner merge yielded 0 rows; check that IDs in "
            f"{name_col!r} match {id_hint}."
        )
    return merged


def build_pipeline_dataframe(
    merged_df: pd.DataFrame,
    *,
    name_col: str,
    target_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    name_col = str(name_col)
    target_col = str(target_col)
    feats = [str(c) for c in feature_cols]

    if name_col not in merged_df.columns:
        raise ValueError(f"name_col {name_col!r} not in merged dataframe.")
    if target_col not in merged_df.columns:
        raise ValueError(f"target_col {target_col!r} not in merged dataframe.")

    missing = [c for c in feats if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Unknown feature column(s) after merge: {missing}")

    cols = [name_col, target_col, *feats]
    return merged_df.loc[:, cols].copy()


def _safe_dir_segment(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")
    return s or "target"


def _jobs_file_token(s: str) -> str:
    """Strip characters that would break tab-separated GNU parallel job lines."""
    return str(s).replace("\t", " ").replace("\n", " ").replace("\r", "")


def _subset_columns(df: pd.DataFrame, cols: list[str], *, context: str) -> pd.DataFrame:
    """Keep only ``cols`` (deduped, order preserved); raise if any column is missing."""
    order = list(dict.fromkeys(cols))
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing column(s): {missing}")
    return df.loc[:, order].copy()


def _validate_experimental_column_roles(
    name_col: str,
    targets: list[str],
    feats: list[str],
    *,
    split_col: str | None = None,
) -> None:
    """``dict.fromkeys`` subsets would drop duplicates; require disjoint roles."""
    name_col = str(name_col)
    tset = {str(t) for t in targets}
    fset = {str(f) for f in feats}
    if len(targets) != len(tset):
        raise ValueError("target_cols contains duplicate column name(s).")
    if len(feats) != len(fset):
        raise ValueError("feature_cols contains duplicate column name(s).")
    if name_col in tset:
        raise ValueError(f"name_col {name_col!r} must not be one of the target columns.")
    if name_col in fset:
        raise ValueError(
            f"name_col {name_col!r} must not appear in feature_cols "
            "(it is the merge key, not a model feature)."
        )
    overlap_tf = tset & fset
    if overlap_tf:
        raise ValueError(
            "The same column cannot be both a target and an experimental feature: "
            f"{sorted(overlap_tf)!r}"
        )
    if split_col:
        sc = str(split_col)
        if sc == name_col:
            raise ValueError(f"split_col {sc!r} must not be the same as name_col.")
        if sc in tset:
            raise ValueError(f"split_col {sc!r} must not be one of the target columns.")
        if sc in fset:
            raise ValueError(
                f"split_col {sc!r} must not appear in feature_cols (it is not a model feature)."
            )


# Top-level object keys in calculate_descriptors aggregated JSON (after flatten, columns are
# ``{group}_{...}`` with this separator).
_KNOWN_DEVELOPABILITY_JSON_GROUPS = frozenset(
    {"surface", "core", "general", "sequence_motives"}
)


_DEVELOPABILITY_EXCLUDE_META = frozenset(
    {
        "error",
        "index",
        "pdb_file",
        "antibody_id",
        "residue_number",
    }
)


def _developability_feature_column_candidates(df_dev: pd.DataFrame) -> list[str]:
    """Column names on the developability table that may be used as ML features (not ``name`` / meta)."""
    out: list[str] = []
    for c in df_dev.columns:
        cs = str(c)
        if cs == "name" or cs in _DEVELOPABILITY_EXCLUDE_META:
            continue
        if cs.startswith("target_"):
            continue
        out.append(cs)
    return out


def _filter_developability_columns_by_groups(
    columns: list[str],
    groups: list[str],
    *,
    developability_is_json_dir: bool,
    context: str,
) -> list[str]:
    """Keep columns whose name equals a group or starts with ``group_`` (JSON flatten layout)."""
    gn = [str(g).strip() for g in groups if str(g).strip()]
    if not gn:
        return list(columns)
    if not developability_is_json_dir:
        warnings.warn(
            f"{context}: --developability-feature-groups is ignored when developability "
            "source is not a directory of JSON files (e.g. Propermab .csv).",
            UserWarning,
            stacklevel=2,
        )
        return list(columns)
    unknown = sorted(set(gn) - _KNOWN_DEVELOPABILITY_JSON_GROUPS)
    if unknown:
        warnings.warn(
            f"{context}: unknown developability_feature_groups {unknown!r} "
            f"(expected subset of {sorted(_KNOWN_DEVELOPABILITY_JSON_GROUPS)}).",
            UserWarning,
            stacklevel=2,
        )
    out: list[str] = []
    for c in columns:
        cs = str(c)
        if any(cs == g or cs.startswith(f"{g}_") for g in gn):
            out.append(cs)
    if not out:
        raise ValueError(
            f"{context}: developability feature groups {gn!r} matched no columns "
            "in the developability table."
        )
    return out


def load_merge_and_expand(
    dataset_path: Path,
    *,
    name_col: str,
    feature_cols: list[str],
    target_cols: list[str],
    developability_sources: list[Path],
    developability_feature_groups: list[str] | None = None,
    split_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read experimental + developability table(s), pick feature columns, merge, one-hot."""
    exp = _read_table(Path(dataset_path))
    name_col = str(name_col)
    feats = [str(c) for c in feature_cols]
    targets = [str(c) for c in target_cols]
    dev_groups = (
        list(developability_feature_groups)
        if developability_feature_groups is not None
        else []
    )
    dev_sources = [Path(p) for p in developability_sources]
    if not dev_sources:
        raise ValueError("At least one developability source path is required.")
    multi_source = len(dev_sources) > 1

    if name_col not in exp.columns:
        raise ValueError(f"name_col {name_col!r} not in experimental dataset.")
    for t in targets:
        if t not in exp.columns:
            raise ValueError(f"target column {t!r} not in experimental dataset.")

    if split_col:
        sc = str(split_col)
        if sc not in exp.columns:
            raise ValueError(
                f"split_col {sc!r} not in experimental dataset columns."
            )

    _validate_experimental_column_roles(
        name_col, targets, feats, split_col=str(split_col) if split_col else None
    )

    merged_dev: pd.DataFrame | None = None
    merged_id_hint_parts: list[str] = []
    dev_cols_all: list[str] = []
    seen_dev_cols: set[str] = set()
    used_suffixes: set[str] = set()

    def _safe_source_suffix(src: Path) -> str:
        """Stable suffix used to disambiguate developability columns by source."""
        s = Path(src)
        if s.is_dir():
            raw = s.name
        else:
            raw = s.parent.name or s.stem
        token = re.sub(r"[^a-zA-Z0-9]+", "_", str(raw)).strip("_")
        if not token:
            token = "dev"
        base = token
        i = 2
        while token in used_suffixes:
            token = f"{base}_{i}"
            i += 1
        used_suffixes.add(token)
        return token

    for src in dev_sources:
        df_dev_one, id_hint_one = load_developability_dataframe(src)
        merged_id_hint_parts.append(id_hint_one)
        dev_candidates = _developability_feature_column_candidates(df_dev_one)
        dev_cols_one = _filter_developability_columns_by_groups(
            dev_candidates,
            dev_groups,
            developability_is_json_dir=src.is_dir(),
            context=f"load_merge_and_expand ({src})",
        )
        dev_cols_one = list(dict.fromkeys(dev_cols_one))

        if multi_source:
            suffix = _safe_source_suffix(src)
            rename_map = {c: f"{c}_{suffix}" for c in dev_cols_one}
            df_dev_one = df_dev_one.rename(columns=rename_map, copy=True)
            dev_cols_one = [rename_map[c] for c in dev_cols_one]

        dup_with_prior = sorted(set(dev_cols_one) & seen_dev_cols)
        if dup_with_prior:
            raise ValueError(
                "Duplicate developability feature column name(s) after source-suffix "
                f"renaming: {dup_with_prior!r}. Ensure sources have distinct directory names."
            )
        seen_dev_cols.update(dev_cols_one)
        dev_cols_all.extend(dev_cols_one)

        if merged_dev is None:
            merged_dev = df_dev_one.loc[:, ["name", *dev_cols_one]].copy()
        else:
            left = merged_dev.copy()
            right = df_dev_one.loc[:, ["name", *dev_cols_one]].copy()
            _normalize_merge_keys(left, "name", right)
            merged_dev = left.merge(right, on="name", how="inner")

    if merged_dev is None:
        raise ValueError("Failed to load developability sources.")
    id_hint = " AND ".join(merged_id_hint_parts)
    df_dev = merged_dev
    dev_cols = list(dict.fromkeys(dev_cols_all))

    if name_col in dev_cols:
        raise ValueError(
            f"name_col {name_col!r} matches a developability feature column name; "
            "pandas would suffix duplicate names on merge. Rename the developability field "
            "or use a different name_col."
        )

    target_dev_overlap = set(targets) & set(dev_cols)
    if target_dev_overlap:
        raise ValueError(
            "Target column name(s) match developability feature column(s); "
            f"rename targets or adjust developability output: {sorted(target_dev_overlap)!r}"
        )

    if not feats:
        exp_keep = list(dict.fromkeys([name_col, *targets]))
        if split_col:
            exp_keep.append(str(split_col))
        exp = _subset_columns(
            exp,
            exp_keep,
            context="Experimental table",
        )
        exp_id_cols = [name_col]
        if split_col:
            exp_id_cols.append(str(split_col))
        require_no_nans_in_dataframe(
            exp[exp_id_cols].copy(),
            context="Experimental table: name_col and split_col must have no NaNs",
        )
        if not dev_cols:
            raise ValueError(
                "No developability feature columns in the united developability table "
                "(after optional group filter). Add experimental feature_cols, or check "
                "developability_results_paths / developability_features."
            )
        combined_feats = dev_cols
    else:
        missing_feats = [c for c in feats if c not in exp.columns]
        if missing_feats:
            raise ValueError(
                "feature_cols must be columns of the experimental dataset only (not "
                "developability-only). Missing from experimental table "
                f"{Path(dataset_path)!s}: {missing_feats!r}"
            )
        exp_keep = list(dict.fromkeys([name_col, *targets, *feats]))
        if split_col and str(split_col) not in exp_keep:
            exp_keep.append(str(split_col))
        exp = _subset_columns(
            exp,
            exp_keep,
            context="Experimental table",
        )
        pre_merge_check_cols = [name_col, *feats]
        if split_col:
            pre_merge_check_cols.append(str(split_col))
        require_no_nans_in_dataframe(
            exp[pre_merge_check_cols].copy(),
            context="Experimental table: name_col, feature_cols, and split_col must have no NaNs",
        )
        overlap = set(feats) & set(dev_cols)
        if overlap:
            raise ValueError(
                "The same column name(s) appear in experimental feature_cols and in the "
                "developability table; merge would be ambiguous. "
                f"Overlap: {sorted(overlap)!r}"
            )
        combined_feats = list(dict.fromkeys([*feats, *dev_cols]))

    merged = merge_experimental_with_developability_subset(
        exp,
        df_dev,
        name_col=name_col,
        id_hint=id_hint,
        developability_columns=dev_cols,
    )
    missing_feat = [c for c in combined_feats if c not in merged.columns]
    if missing_feat:
        raise ValueError(
            "After merge, expected feature column(s) are missing (possible key/column "
            f"name clash): {missing_feat!r}"
        )
    merged, expanded_feature_cols = one_hot_listed_non_numeric_features(
        merged, combined_feats
    )
    merged_tail = list(dict.fromkeys([name_col, *targets, *expanded_feature_cols]))
    if split_col:
        sc = str(split_col)
        if sc not in merged.columns:
            raise ValueError(
                f"split_col {sc!r} missing after merge with developability "
                "(possible name clash with a developability column)."
            )
        merged_tail.append(sc)
        require_no_nans_in_dataframe(
            merged[[sc]].copy(),
            context=f"After merge: split_col {sc!r} must have no NaNs",
        )
    merged = _subset_columns(
        merged,
        list(dict.fromkeys(merged_tail)),
        context="After merge (name, targets, experimental + developability features)",
    )
    require_no_nans_except_columns(
        merged,
        skip_columns=set(targets),
        context="After merge (experimental + developability); target columns may be NaN",
    )
    return merged, expanded_feature_cols


def write_folds_for_target(
    merged: pd.DataFrame,
    expanded_feature_cols: list[str],
    *,
    name_col: str,
    user_target_col: str,
    all_target_cols: list[str],
    output_dir: Path,
    n_splits: int,
    random_state: int,
    features_frac: float,
    dataset_stem: str,
    max_target_nan_frac: float = 0.5,
    split_col: str | None = None,
) -> tuple[Path, list[str]]:
    """Drop target NaNs, build pipeline frame, split, write parquet + meta. Returns (fold_dir, job_lines)."""
    name_col = str(name_col)
    user_target_col = str(user_target_col)

    m = drop_invalid_target_rows(
        merged, user_target_col, max_nan_frac=max_target_nan_frac
    )
    # Other targets may still be NaN per row (multi-target experimental table).
    other_targets = {str(t) for t in all_target_cols if str(t) != user_target_col}
    require_no_nans_except_columns(
        m,
        skip_columns=other_targets,
        context=f"After dropping NaNs for target {user_target_col!r}",
    )
    if split_col:
        sc = str(split_col)
        require_no_nans_in_dataframe(
            m[[sc]].copy(),
            context=f"After dropping NaNs for target {user_target_col!r}: split_col {sc!r}",
        )

    m, pipe_target, pipe_feats = standardize_pipeline_column_names(
        m,
        name_col=name_col,
        target_col=user_target_col,
        feature_cols=expanded_feature_cols,
    )
    pipe_df = build_pipeline_dataframe(
        m,
        name_col="name",
        target_col=pipe_target,
        feature_cols=pipe_feats,
    )

    N = len(pipe_df)
    if split_col:
        sc = str(split_col)
        if sc not in m.columns:
            raise ValueError(
                f"split_col {sc!r} missing on merged frame before writing folds."
            )
        split_series = m[sc].reset_index(drop=True)
        segment_sizes, fold_ilocs = cv_split_col_ilocs(split_series)
        split_scheme = "column"
        meta_n_splits = len(fold_ilocs)
    else:
        segment_sizes, fold_ilocs = cv_shuffled_fold_ilocs(N, n_splits, random_state)
        split_scheme = "shuffled"
        meta_n_splits = int(n_splits)

    fold_root = Path(output_dir) / _safe_dir_segment(pipe_target)
    fold_root.mkdir(parents=True, exist_ok=True)

    sfs_row_basis = int(pipe_df.shape[0])

    meta = {
        "target_col": pipe_target,
        "feature_cols": pipe_feats,
        "n_splits": int(meta_n_splits),
        "random_state": int(random_state),
        "N": int(N),
        "segment_sizes": [int(x) for x in segment_sizes],
        "features_frac": float(features_frac),
        "sfs_row_count_basis": sfs_row_basis,
        "split_scheme": split_scheme,
        "split_col": str(split_col) if split_col else None,
    }
    meta_path = fold_root / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    n_folds = len(fold_ilocs)
    print(
        f"[prepare_run] {dataset_stem} target={user_target_col!r} "
        f"(pipeline column {pipe_target!r}): writing {n_folds} fold(s) under {fold_root.name}/ …",
        file=sys.stderr,
        flush=True,
    )
    job_lines: list[str] = []
    for k, (train_idx, test_idx) in enumerate(fold_ilocs):
        train_df = pipe_df.iloc[train_idx].copy()
        test_df = pipe_df.iloc[test_idx].copy()
        train_df.to_parquet(fold_root / f"fold_{k}_train.parquet", index=False)
        test_df.to_parquet(fold_root / f"fold_{k}_test.parquet", index=False)
        ds_t = _jobs_file_token(dataset_stem)
        tg_t = _jobs_file_token(pipe_target)
        job_lines.append(f"{fold_root.resolve()}\t{k}\t{ds_t}\t{tg_t}")
        print(
            f"[prepare_run] {dataset_stem}: created fold {k + 1}/{n_folds} "
            f"(index {k}) for {pipe_target!r} — "
            f"train={len(train_df)} test={len(test_df)} rows",
            file=sys.stderr,
            flush=True,
        )

    return fold_root, job_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge developability once, write CV fold parquet files per target for parallel workers."
        ),
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to experimental table (.csv or .parquet).",
    )
    parser.add_argument(
        "--name-col",
        required=True,
        help="ID column merged to developability (JSON stem or pdb_file basename).",
    )
    parser.add_argument(
        "--target-cols",
        nargs="+",
        required=True,
        metavar="COL",
        help="One or more response columns (original names before target_ prefix).",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        metavar="COL",
        help=(
            "Optional: columns on the experimental table only (space-separated). "
            "Developability feature columns (after optional --developability-feature-groups) "
            "are inner-merged in addition to these. Omit to use only developability columns. "
            "Non-numeric listed columns are one-hot-encoded after merge."
        ),
    )
    parser.add_argument(
        "--developability-results",
        required=True,
        nargs="+",
        type=Path,
        dest="developability_sources",
        help=(
            "One or more developability sources (space-separated): directory of *.json "
            "(stem = ID) and/or propermab features.csv."
        ),
    )
    parser.add_argument(
        "--developability-feature-groups",
        nargs="*",
        default=None,
        metavar="GROUP",
        dest="developability_feature_groups",
        help=(
            "Restrict developability-derived merged columns to these JSON top-level groups "
            "(surface, core, general, sequence_motives). Does not apply to --feature-cols "
            "(experimental table). Ignored for developability .csv. Omit for all groups."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Parent directory; each target is written under a subfolder (e.g. target_tm1).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--features-frac",
        type=float,
        default=0.1,
        dest="features_frac",
        help=(
            "Maximum fraction of row-count basis for how many features selection may keep; "
            "stored in meta.json as ``features_frac`` for fold workers."
        ),
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        default=None,
        help=(
            "Append tab-separated lines "
            "'fold_dir<TAB>k<TAB>dataset_stem<TAB>pipeline_target_col' for GNU parallel."
        ),
    )
    parser.add_argument(
        "--max-target-nan-frac",
        type=float,
        default=0.5,
        dest="max_target_nan_frac",
        metavar="F",
        help=(
            "Skip a target (raise) when this fraction or more of rows are NaN in that column "
            "(default 0.5). Use e.g. 0.7 for sparse multi-endpoint tables."
        ),
    )
    parser.add_argument(
        "--split-col",
        type=str,
        default=None,
        help=(
            "Optional column on the experimental CSV with discrete fold ids. "
            "Uses leave-one-fold-out CV; fold count is the number of distinct values (≥2). "
            "YAML/CLI --n-splits is ignored for fold count when this is set. "
            "Omit for shuffled contiguous blocks (default)."
        ),
    )
    args = parser.parse_args()
    if not (0.0 < float(args.max_target_nan_frac) <= 1.0):
        print("--max-target-nan-frac must be in (0, 1].", file=sys.stderr)
        sys.exit(1)

    split_col = (args.split_col or "").strip() or None

    if not args.dataset_path.exists():
        print(f"Dataset not found: {args.dataset_path}", file=sys.stderr)
        sys.exit(1)
    dev_sources = [Path(p) for p in args.developability_sources]
    for dev in dev_sources:
        if not dev.exists():
            print(f"Developability path not found: {dev}", file=sys.stderr)
            sys.exit(1)
        if not (dev.is_dir() or (dev.is_file() and dev.suffix.lower() == ".csv")):
            print(
                "Developability path must be a directory of JSON files or a .csv file "
                f"(got {dev})",
                file=sys.stderr,
            )
            sys.exit(1)

    out_parent = Path(args.output_dir)
    out_parent.mkdir(parents=True, exist_ok=True)

    dataset_stem = args.dataset_path.stem
    print(
        f"[prepare_run] {dataset_stem}: loading developability + merging "
        f"(this step can take a long time on large tables)…",
        file=sys.stderr,
        flush=True,
    )

    fc = [] if args.feature_cols is None else list(args.feature_cols)
    dfg = (
        None
        if args.developability_feature_groups is None
        else list(args.developability_feature_groups)
    )
    try:
        merged, expanded_feature_cols = load_merge_and_expand(
            args.dataset_path,
            name_col=args.name_col,
            feature_cols=fc,
            target_cols=args.target_cols,
            developability_sources=dev_sources,
            developability_feature_groups=dfg,
            split_col=split_col,
        )
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(
        f"[prepare_run] {dataset_stem}: merged {len(merged)} row(s); "
        f"writing CV folds for {len(args.target_cols)} target(s)…",
        file=sys.stderr,
        flush=True,
    )
    all_job_lines: list[str] = []
    for user_target in args.target_cols:
        try:
            fold_root, lines = write_folds_for_target(
                merged,
                expanded_feature_cols,
                name_col=args.name_col,
                user_target_col=user_target,
                all_target_cols=list(args.target_cols),
                output_dir=out_parent,
                n_splits=args.n_splits,
                random_state=args.random_state,
                features_frac=args.features_frac,
                dataset_stem=dataset_stem,
                max_target_nan_frac=float(args.max_target_nan_frac),
                split_col=split_col,
            )
        except ValueError as e:
            print(f"Target {user_target!r}: {e}", file=sys.stderr)
            sys.exit(1)
        print(
            f"[prepare_run] {dataset_stem} target={user_target!r}: "
            f"finished {len(lines)} fold(s) -> {fold_root}",
            file=sys.stderr,
            flush=True,
        )
        all_job_lines.extend(lines)

    for line in all_job_lines:
        print(line)
    if args.jobs_file is not None:
        args.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.jobs_file, "a") as jf:
            for line in all_job_lines:
                jf.write(line + "\n")
        print(
            f"[prepare_run] Appended {len(all_job_lines)} job line(s) to {args.jobs_file}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
