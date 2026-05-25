#!/usr/bin/env python3
"""Validate CSV names against structure files in paired dataset folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from automl.dataset_validation import (
    collect_experimental_dataset_errors,
    collect_structure_validation_errors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    args = parser.parse_args()

    csv_dir = args.csv_dir.resolve()
    structures_root = args.structures_root.resolve()
    if not csv_dir.is_dir():
        raise SystemExit(f"Not a directory: {csv_dir}")
    if not structures_root.is_dir():
        raise SystemExit(f"Not a directory: {structures_root}")

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No *.csv files in {csv_dir}")

    all_errors: list[str] = []
    for csv_path in csv_files:
        stem = csv_path.stem
        dataset_dir = structures_root / stem
        errors: list[str] = []
        if not dataset_dir.is_dir():
            errors.append(f"{stem}: missing structure subfolder {dataset_dir}")
        else:
            df = pd.read_csv(csv_path)
            errors.extend(
                collect_experimental_dataset_errors(
                    df,
                    context=f"{csv_path.name}",
                )
            )
            if not errors:
                errors.extend(
                    collect_structure_validation_errors(
                        df,
                        dataset_dir,
                        check_orphan_structures=True,
                        context=stem,
                    )
                )
        all_errors.extend(errors)

    if all_errors:
        print("Structure / CSV name validation failed:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"Validated {len(csv_files)} dataset(s): names in CSVs match structures under {structures_root}"
    )


if __name__ == "__main__":
    main()
