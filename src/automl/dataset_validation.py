"""Validate experimental CSV tables before AutoML or structure workflows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

STRUCTURE_SUFFIXES: tuple[str, ...] = (".pdb", ".cif", ".mmcif")


def find_structure_file(dataset_dir: Path, name: str) -> Path | None:
    dataset_dir = Path(dataset_dir)
    stem = str(name)
    for suffix in STRUCTURE_SUFFIXES:
        candidate = dataset_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def require_no_nans_in_dataframe(df: pd.DataFrame, *, context: str) -> None:
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
    skip = {str(c) for c in skip_columns}
    sub = df[[c for c in df.columns if str(c) not in skip]]
    if sub.empty:
        return
    require_no_nans_in_dataframe(sub, context=context)


def _stripped_name_series(df: pd.DataFrame, name_col: str) -> pd.Series:
    raw = df[name_col]
    if raw.isna().any():
        n_nan = int(raw.isna().sum())
        raise ValueError(
            f"Column {name_col!r} has {n_nan} NaN value(s); names must be present."
        )
    return raw.astype(str).str.strip()


def _collect_pipe_in_name_errors(
    names: pd.Series,
    *,
    name_col: str,
    context: str,
) -> list[str]:
    pipe_mask = names.str.contains("|", regex=False)
    if not pipe_mask.any():
        return []
    bad = sorted(names[pipe_mask].unique().tolist())
    sample = bad[:10]
    tail = f" … and {len(bad) - 10} more" if len(bad) > 10 else ""
    return [
        f"{context}: {name_col!r} must not contain '|': {sample!r}{tail}."
    ]


def collect_experimental_dataset_errors(
    df: pd.DataFrame,
    *,
    name_col: str = "name",
    heavy_col: str = "heavy",
    light_col: str = "light",
    allow_nan_in_columns: Iterable[str] | None = None,
    context: str = "Experimental dataset",
) -> list[str]:
    errors: list[str] = []
    name_col = str(name_col)
    heavy_col = str(heavy_col)
    light_col = str(light_col)
    ctx = str(context)

    if df.empty:
        errors.append(f"{ctx}: dataset has no rows.")
        return errors

    required = (name_col, heavy_col, light_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"{ctx}: missing required column(s) {missing!r}.")
        return errors

    skip_nan = {str(c) for c in (allow_nan_in_columns or ())}
    cols_to_check = [c for c in df.columns if str(c) not in skip_nan]
    if cols_to_check:
        try:
            require_no_nans_in_dataframe(
                df[cols_to_check].copy(),
                context=f"{ctx}: all columns must have no NaNs"
                + (
                    f" (except {sorted(skip_nan)!r})"
                    if skip_nan
                    else ""
                ),
            )
        except ValueError as exc:
            errors.append(str(exc))

    try:
        names = _stripped_name_series(df, name_col)
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    errors.extend(
        _collect_pipe_in_name_errors(names, name_col=name_col, context=ctx)
    )

    empty_mask = names == ""
    if empty_mask.any():
        n_empty = int(empty_mask.sum())
        errors.append(
            f"{ctx}: {n_empty} row(s) have empty {name_col!r} after stripping whitespace."
        )

    dup_name_mask = names.duplicated(keep=False)
    if dup_name_mask.any():
        dup_names = sorted(names[dup_name_mask].unique().tolist())
        sample = dup_names[:10]
        tail = f" … and {len(dup_names) - 10} more" if len(dup_names) > 10 else ""
        errors.append(
            f"{ctx}: duplicate value(s) in {name_col!r}: {sample!r}{tail}."
        )

    heavy = df[heavy_col]
    light = df[light_col]
    if heavy.isna().any() or light.isna().any():
        n_h = int(heavy.isna().sum())
        n_l = int(light.isna().sum())
        errors.append(
            f"{ctx}: {heavy_col!r} has {n_h} NaN and {light_col!r} has {n_l} NaN; "
            "both sequence columns must be present."
        )
        return errors

    seq_key = heavy.astype(str) + light.astype(str)
    dup_seq_mask = seq_key.duplicated(keep=False)
    if dup_seq_mask.any():
        dup_df = pd.DataFrame(
            {name_col: names, "_seq_key": seq_key}
        ).loc[dup_seq_mask]
        groups: dict[str, list[str]] = {}
        for nm, key in zip(dup_df[name_col], dup_df["_seq_key"], strict=True):
            groups.setdefault(str(key), []).append(str(nm))
        group_lines = [
            str(sorted(dict.fromkeys(group_names)))
            for group_names in groups.values()
        ]
        group_lines.sort()
        sample = group_lines[:5]
        tail = (
            f" … and {len(group_lines) - 5} more duplicate sequence group(s)"
            if len(group_lines) > 5
            else ""
        )
        errors.append(
            f"{ctx}: duplicate concatenated {heavy_col!r}+{light_col!r} sequence(s); "
            f"name groups: {', '.join(sample)}{tail}."
        )

    return errors


def validate_experimental_dataset(
    df: pd.DataFrame,
    *,
    name_col: str = "name",
    heavy_col: str = "heavy",
    light_col: str = "light",
    allow_nan_in_columns: Iterable[str] | None = None,
    context: str = "Experimental dataset",
) -> None:
    errors = collect_experimental_dataset_errors(
        df,
        name_col=name_col,
        heavy_col=heavy_col,
        light_col=light_col,
        allow_nan_in_columns=allow_nan_in_columns,
        context=context,
    )
    if errors:
        raise ValueError("\n".join(errors))


def collect_structure_validation_errors(
    df: pd.DataFrame,
    structures_dir: Path,
    *,
    name_col: str = "name",
    check_orphan_structures: bool = False,
    context: str | None = None,
) -> list[str]:
    errors: list[str] = []
    structures_dir = Path(structures_dir)
    name_col = str(name_col)
    ctx = context or f"Structures under {structures_dir}"

    if name_col not in df.columns:
        errors.append(f"{ctx}: missing required column {name_col!r}.")
        return errors

    if not structures_dir.is_dir():
        errors.append(f"{ctx}: structures directory does not exist: {structures_dir}")
        return errors

    try:
        names = _stripped_name_series(df, name_col)
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    errors.extend(
        _collect_pipe_in_name_errors(names, name_col=name_col, context=ctx)
    )

    csv_name_set: set[str] = set()
    for name in names:
        if not name:
            continue
        csv_name_set.add(name)
        if find_structure_file(structures_dir, name) is None:
            errors.append(
                f"{ctx}: no structure file for name {name!r} "
                f"(expected {structures_dir}/{name}.pdb or .cif)"
            )

    if check_orphan_structures:
        structure_files = [
            p
            for p in structures_dir.iterdir()
            if p.is_file() and p.suffix.lower() in set(STRUCTURE_SUFFIXES)
        ]
        for path in structure_files:
            if path.stem not in csv_name_set:
                errors.append(
                    f"{ctx}: structure {path.name} has no matching name in the dataset"
                )

    return errors


def validate_experimental_structures_present(
    df: pd.DataFrame,
    structures_dir: Path,
    *,
    name_col: str = "name",
    check_orphan_structures: bool = False,
    context: str | None = None,
) -> None:
    errors = collect_structure_validation_errors(
        df,
        structures_dir,
        name_col=name_col,
        check_orphan_structures=check_orphan_structures,
        context=context,
    )
    if errors:
        raise ValueError("\n".join(errors))
