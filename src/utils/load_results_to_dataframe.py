#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _flatten_dict(obj, parent_key: str = "", sep: str = "_"):
    items = []
    if parent_key:
        prefix = parent_key + sep
    else:
        prefix = ""
    for k, v in obj.items():
        new_key = f"{prefix}{k}"
        if v is None or isinstance(v, (int, float, bool, str)):
            items.append((new_key, v))
        elif isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep))
        else:
            continue
    return items


def load_json_results(results_dir: Path, name_from_stem: bool = True) -> pd.DataFrame:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_dir}")

    jsons = sorted(results_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No *.json files in {results_dir}")

    rows = []
    for path in jsons:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {path.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(
                f"Warning: skipping {path.name}: top-level JSON value is not an object",
                file=sys.stderr,
            )
            continue
        flat = dict(_flatten_dict(data))
        if name_from_stem:
            flat["name"] = path.stem
        rows.append(flat)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Load JSON results from a folder and concat into a DataFrame."
    )
    parser.add_argument(
        "results_folder",
        type=Path,
        help="Folder containing *.json descriptor results (e.g. pdgf38_results)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base name for this dataset (e.g. pdgf38). Added as column 'base'.",
    )
    parser.add_argument(
        "--order-csv",
        type=Path,
        default=None,
        help="Optional CSV with name, heavy, light. Merge to add heavy/light.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write DataFrame as CSV or Parquet.",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Write Parquet instead of CSV when --out is set.",
    )
    args = parser.parse_args()

    try:
        df = load_json_results(args.results_folder)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if args.base is not None:
        df["base"] = args.base

    if args.order_csv is not None:
        order_path = args.order_csv
        if not order_path.exists():
            print(f"Order CSV not found: {order_path}", file=sys.stderr)
            sys.exit(1)
        order_df = pd.read_csv(order_path)
        for col in ["name", "heavy", "light"]:
            if col not in order_df.columns:
                print(
                    f"Order CSV must contain columns: name, heavy, light. Missing: {col}",
                    file=sys.stderr,
                )
                sys.exit(1)
        order_sub = order_df[["name", "heavy", "light"]].drop_duplicates("name")
        before = len(df)
        df = df.merge(order_sub, on="name", how="left")
        n_matched = df["heavy"].notna().sum()
        if n_matched < before:
            print(
                f"Warning: only {n_matched}/{before} rows matched order CSV by 'name'.",
                file=sys.stderr,
            )

    if args.out is not None:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.parquet:
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")
    else:
        print(
            f"Loaded {len(df)} rows, {len(df.columns)} columns. Pass --out to save CSV/Parquet."
        )


if __name__ == "__main__":
    main()
