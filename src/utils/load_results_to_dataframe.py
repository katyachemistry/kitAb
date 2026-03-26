#!/usr/bin/env python3
"""
Load descriptor results from a folder of JSON files (e.g. pdgf38_results) and
concat into a single DataFrame compatible with clean_validation.ipynb merge logic.

The notebook merges on (heavy, light) or (name) and expects columns: base, heavy, light, name,
plus all descriptor columns. This script:
  - Loads all *.json from the given results folder
  - Flattens nested dicts into columns (e.g. cluster_metrics.negative_cluster_largest_size)
  - Uses filename stem as "name" (e.g. R1-004)
  - Optionally merges with an order CSV (name, heavy, light) to add heavy, light, base

Usage (from repo root):
  python src/utils/load_results_to_dataframe.py pdgf38_results [--base pdgf38] [--order-csv path/to/pdgf38.csv] [--out merged.csv]

Example (for use with notebook):
  python src/utils/load_results_to_dataframe.py pdgf38_results --base pdgf38 --order-csv data/pdgf38.csv --out data/pdgf38_descriptors.csv
  # Then in notebook: results_df = pd.read_csv("data/pdgf38_descriptors.csv")
  # Or: from src.utils.load_results_to_dataframe import load_json_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _is_scalar(v):
    if v is None:
        return True
    if isinstance(v, (int, float, bool)):
        return True
    if isinstance(v, str):
        return True
    return False


def _flatten_dict(obj, parent_key: str = "", sep: str = "_"):
    """Flatten nested dict; only scalar values become columns. Lists/dicts are skipped."""
    items = []
    if parent_key:
        prefix = parent_key + sep
    else:
        prefix = ""
    for k, v in obj.items():
        new_key = f"{prefix}{k}"
        if _is_scalar(v):
            items.append((new_key, v))
        elif isinstance(v, dict):
            # Recurse; filter out non-scalar leaves in nested dicts
            items.extend(_flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # Skip or could stringify; skip to keep columns numeric where possible
            continue
        else:
            continue
    return items


def load_json_results(results_dir: Path, name_from_stem: bool = True) -> pd.DataFrame:
    """Load all JSON files from results_dir; flatten each to one row; concat."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_dir}")

    jsons = sorted(results_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No *.json files in {results_dir}")

    rows = []
    for path in jsons:
        with open(path, "r") as f:
            data = json.load(f)
        flat = dict(_flatten_dict(data))
        if name_from_stem:
            flat["name"] = path.stem
        rows.append(flat)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Load JSON results from a folder and concat into a DataFrame (for clean_validation merge)."
    )
    parser.add_argument(
        "results_folder",
        type=Path,
        help="Folder containing *.json descriptor results (e.g. pdgf38_results)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base name for this dataset (e.g. pdgf38). Added as column 'base' for notebook merge.",
    )
    parser.add_argument(
        "--order-csv",
        type=Path,
        default=None,
        help="Optional CSV with name, heavy, light (and optionally other cols). Merge to add heavy/light.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write DataFrame as CSV. If omitted, prints path to stdout and writes no file.",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="If --out is set, write Parquet instead of CSV.",
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
        order_path = Path(args.order_csv)
        if not order_path.exists():
            print(f"Order CSV not found: {order_path}", file=sys.stderr)
            sys.exit(1)
        order_df = pd.read_csv(order_path)
        for col in ["name", "heavy", "light"]:
            if col not in order_df.columns:
                print(f"Order CSV must contain columns: name, heavy, light. Missing: {col}", file=sys.stderr)
                sys.exit(1)
        # Merge on name so we get heavy, light (and any extra columns from order)
        order_sub = order_df[["name", "heavy", "light"]].drop_duplicates("name")
        before = len(df)
        df = df.merge(order_sub, on="name", how="left")
        n_matched = df["heavy"].notna().sum()
        if n_matched < before:
            print(f"Warning: only {n_matched}/{before} rows matched order CSV by 'name'.", file=sys.stderr)

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.parquet:
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")
    else:
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns. Pass --out to save CSV/Parquet.")

    return df


if __name__ == "__main__":
    main()
