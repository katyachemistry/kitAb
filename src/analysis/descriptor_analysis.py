#!/usr/bin/env python3
"""Descriptor run variability and target correlations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils.load_results_to_dataframe import load_json_results

DEFAULT_DATASET_TO_CSV: dict[str, str] = {
    "ab21": "ab21.csv",
    "garbinski2023": "garbinski2023_tm1_folded_08_4.csv",
    "GINKGO": "ginkgo_ig_folded.csv",
    "hutchinson2023enhancement": "hutchinson2023enhancement_top200tm1_igg.csv",
    "jain2017biophysical": "jain2017biophysical_folded_08_5.csv",
    "jain2023identifying": "jain2023identifying_folded_08_5.csv",
    "jain2024assessment": "jain2024assessment_folded_08_5.csv",
    "jetha2019homology": "jetha2019homology_RT.csv",
    "kraft2019herapin": "kraft2019herapin_relrt_folded_08_5.csv",
    "pdgf38": "pdgf38.csv",
}


def _descriptor_columns(
    df: pd.DataFrame,
    *,
    id_cols: frozenset[str],
    exclude_prefixes: tuple[str, ...] = (),
) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if c in id_cols:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _per_antibody_rcv_across_runs(
    wide: pd.DataFrame,
    *,
    eps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = wide.mean(axis=1)
    s = wide.std(axis=1, ddof=1 if wide.shape[1] > 1 else 0)
    denom = m.abs().clip(lower=eps)
    rcv = s / denom
    rcv = rcv.where(s.notna() & m.notna())
    return m, s, rcv


def descriptor_variability_for_runs(
    run_result_dirs: Sequence[Path | str],
    *,
    dataset: str = "dataset",
    id_cols: Sequence[str] = ("name",),
    eps: float = 1e-12,
    exclude_prefixes: tuple[str, ...] = (),
) -> pd.DataFrame:
    dirs = [Path(p).resolve() for p in run_result_dirs]
    if len(dirs) < 2:
        raise ValueError("Need at least two run result directories to compare runs.")

    id_set = frozenset(id_cols)
    frames: list[pd.DataFrame] = []
    for r, d in enumerate(dirs):
        sub = load_json_results(d)
        sub["_run"] = r
        frames.append(sub)

    long = pd.concat(frames, ignore_index=True)
    if not id_set.issubset(long.columns):
        missing = id_set - frozenset(long.columns)
        raise KeyError(f"Missing id column(s) {missing} after loading runs.")

    counts = long.groupby(list(id_cols))["_run"].nunique()
    n_runs = len(dirs)
    complete_ids = counts[counts == n_runs].index
    if isinstance(complete_ids, pd.MultiIndex):
        mask = long.set_index(list(id_cols)).index.isin(complete_ids)
    else:
        mask = long[list(id_cols)[0]].isin(complete_ids)
    long_c = long.loc[mask].copy()

    cols = _descriptor_columns(long_c, id_cols=id_set | {"_run"}, exclude_prefixes=exclude_prefixes)
    if not cols:
        raise ValueError("No numeric descriptor columns found.")

    rows: list[dict] = []
    idx_names = list(id_cols)

    for col in cols:
        try:
            pivot = long_c.pivot_table(index=idx_names, columns="_run", values=col, aggfunc="first")
        except Exception:
            continue
        if pivot.shape[1] < 2:
            continue
        pivot = pivot.apply(pd.to_numeric, errors="coerce")
        if not pivot.notna().all(axis=1).any():
            continue
        ok = pivot.notna().all(axis=1)
        pivot = pivot.loc[ok]
        if pivot.shape[0] == 0:
            continue

        _, _, rcv_ab = _per_antibody_rcv_across_runs(pivot, eps=eps)
        mu_r = pivot.mean(axis=0)
        cohort_mean = float(mu_r.mean())
        cohort_std = float(mu_r.std(ddof=1)) if len(mu_r) > 1 else float("nan")
        rel_cv_means = (
            cohort_std / max(abs(cohort_mean), eps) if pd.notna(cohort_std) else float("nan")
        )

        rows.append(
            {
                "dataset": dataset,
                "descriptor": col,
                "median_relative_cv": float(rcv_ab.median(skipna=True)),
                "mean_relative_cv": float(rcv_ab.mean(skipna=True)),
                "relative_cv_run_means": rel_cv_means,
                "n_antibodies": int(pivot.shape[0]),
                "n_runs": n_runs,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        "median_relative_cv",
        ascending=False,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def descriptor_variability_table(
    dataset_to_run_dirs: Mapping[str, Sequence[Path | str]],
    *,
    id_cols: Sequence[str] = ("name",),
    eps: float = 1e-12,
    exclude_prefixes: tuple[str, ...] = (),
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for ds, runs in dataset_to_run_dirs.items():
        parts.append(
            descriptor_variability_for_runs(
                runs,
                dataset=ds,
                id_cols=id_cols,
                eps=eps,
                exclude_prefixes=exclude_prefixes,
            )
        )
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return out
    return out.sort_values(
        ["dataset", "median_relative_cv"],
        ascending=[True, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _parse_kv_pairs(pairs: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Expected dataset=path1,path2,... got: {raw!r}")
        k, rest = raw.split("=", 1)
        k = k.strip()
        paths = [x.strip() for x in rest.split(",") if x.strip()]
        if len(paths) < 2:
            raise ValueError(f"Dataset {k!r}: need at least two comma-separated result dirs.")
        out[k] = paths
    return out


def run_variability(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build per-descriptor run-variability tables (median relative CV across runs)."
    )
    p.add_argument(
        "--dataset-runs",
        action="append",
        metavar="DS=DIR1,DIR2,...",
        required=True,
        help="Repeat for each dataset: name=results_dir_run1,results_dir_run2,...",
    )
    p.add_argument("--out", type=Path, default=None, help="Write CSV (default: stdout).")
    p.add_argument("--eps", type=float, default=1e-12)
    args = p.parse_args(argv)

    spec = _parse_kv_pairs(args.dataset_runs)
    tab = descriptor_variability_table(spec, eps=args.eps)
    if tab.empty:
        print("No variability rows produced.", file=sys.stderr)
        return 1
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(args.out, index=False)
        print(f"Wrote {len(tab)} rows to {args.out}")
    else:
        print(tab.to_string(index=False))
    return 0


def _flatten_json_numeric(obj: Any, prefix: str = "") -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_json_numeric(v, key))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out


def _load_descriptors_for_experiment(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            payload = json.load(f)
        flat = _flatten_json_numeric(payload)
        flat["name"] = str(path.stem)
        rows.append(flat)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _target_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("target_")]


def _numeric_descriptor_columns(df: pd.DataFrame) -> list[str]:
    skip = {"name"}
    return [
        c
        for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]


def _corr_block(
    merged: pd.DataFrame, descriptors: list[str], targets: list[str], dataset: str, data_csv: str
) -> Iterator[dict[str, Any]]:
    for target in targets:
        y = pd.to_numeric(merged[target], errors="coerce")
        for desc in descriptors:
            x = pd.to_numeric(merged[desc], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            row_base = {
                "dataset": dataset,
                "data_csv": data_csv,
                "target": target,
                "descriptor": desc,
                "n": n,
            }
            if n < 3:
                yield {**row_base, "pearson_r": float("nan"), "pearson_p": float("nan"),
                       "spearman_r": float("nan"), "spearman_p": float("nan")}
                continue
            xs = x[mask].astype(float)
            ys = y[mask].astype(float)
            if xs.nunique() < 2 or ys.nunique() < 2:
                yield {**row_base, "pearson_r": float("nan"), "pearson_p": float("nan"),
                       "spearman_r": float("nan"), "spearman_p": float("nan")}
                continue
            pr = pearsonr(xs, ys)
            sr = spearmanr(xs, ys)
            yield {
                **row_base,
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_r": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
            }


def run_correlations(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Descriptor vs assay target correlations.")
    p.add_argument(
        "--descriptors-root",
        type=Path,
        default=Path("descriptors_experiments"),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/descriptor_target_correlations.csv"),
    )
    p.add_argument(
        "--dataset-csv",
        action="append",
        metavar="FOLDER=FILENAME",
        help="Override tabular file for one experiment folder. Repeatable.",
    )
    args = p.parse_args(argv)

    overrides: dict[str, str] = {}
    if args.dataset_csv:
        for spec in args.dataset_csv:
            if "=" not in spec:
                print(f"Invalid --dataset-csv (expected FOLDER=file.csv): {spec!r}", file=sys.stderr)
                return 2
            k, v = spec.split("=", 1)
            overrides[k.strip()] = v.strip()

    dataset_to_csv = {**DEFAULT_DATASET_TO_CSV, **overrides}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    for exp_dir in sorted(args.descriptors_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        dataset = exp_dir.name
        csv_name = dataset_to_csv.get(dataset)
        if not csv_name:
            print(f"Skip {dataset!r}: no default data CSV mapping (use --dataset-csv).", file=sys.stderr)
            continue
        data_path = args.data_dir / csv_name
        if not data_path.is_file():
            print(f"Skip {dataset!r}: missing {data_path}", file=sys.stderr)
            continue

        results_dir = exp_dir / "results"
        if not results_dir.is_dir():
            print(f"Skip {dataset!r}: no results dir {results_dir}", file=sys.stderr)
            continue

        desc_df = _load_descriptors_for_experiment(results_dir)
        if desc_df.empty:
            print(f"Skip {dataset!r}: no JSON under {results_dir}", file=sys.stderr)
            continue

        tab = pd.read_csv(data_path)
        if "name" not in tab.columns:
            print(f"Skip {dataset!r}: no 'name' column in {data_path}", file=sys.stderr)
            continue
        tab["name"] = tab["name"].astype(str)

        targets = _target_columns(tab)
        if not targets:
            print(f"Skip {dataset!r}: no target_* columns in {data_path}", file=sys.stderr)
            continue

        desc_cols = _numeric_descriptor_columns(desc_df)
        merged = tab[["name"] + targets].merge(desc_df, on="name", how="inner")
        if merged.empty:
            print(
                f"Warning {dataset!r}: zero rows after merge on name "
                f"(tabular n={len(tab)}, descriptors n={len(desc_df)}).",
                file=sys.stderr,
            )
            continue

        for row in _corr_block(merged, desc_cols, targets, dataset, csv_name):
            all_rows.append(row)

    if not all_rows:
        print("No correlation rows produced.", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out.resolve()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("variability", help="Median relative CV of descriptors across runs.")
    sub.add_parser("correlations", help="Pearson/Spearman vs assay targets.")

    args, rest = parser.parse_known_args()
    if args.command == "variability":
        return run_variability(rest)
    if args.command == "correlations":
        return run_correlations(rest)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
