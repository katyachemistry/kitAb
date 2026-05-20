#!/usr/bin/env python3
"""AbodyBuilder2 (ImmuneBuilder) inference from CSV(s) (name, heavy, light).

One model load per process; GPU is selected like ABB3 via --device (cuda:N maps to
CUDA_VISIBLE_DEVICES before torch import). Predictions are one sequence at a time
(ABodyBuilder2 has no multi-sequence batch API). --batch-size is accepted for CLI
parity with the ABB3 driver but is not used for inference grouping.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _early_set_cuda_visible_device() -> None:
    """Parse --device / ABB2_DEVICE before torch so only the chosen GPU is visible."""
    dev = ""
    i = 0
    while i < len(sys.argv) - 1:
        if sys.argv[i] == "--device":
            dev = sys.argv[i + 1].strip()
            break
        i += 1
    if not dev:
        dev = os.environ.get("ABB2_DEVICE", "").strip()
    if not dev:
        dev = "cuda:1"
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if dev.startswith("cuda:"):
        tail = dev.split(":", 1)[1].strip()
        if tail.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = tail
            return
    if dev == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    raise SystemExit(f"Unknown --device {dev!r} (use cuda:0, cuda:1, cpu)")


_early_set_cuda_visible_device()

import pandas as pd
import torch


def _build_dataset_jobs(
    csv_paths: list[Path] | None,
    data_dir: Path | None,
    dataset_names: list[str] | None,
) -> list[tuple[Path, str]]:
    if data_dir is not None:
        if csv_paths:
            raise SystemExit("Use either --data-dir or --csv, not both")
        d = data_dir.resolve()
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        paths = sorted(d.glob("*.csv"))
        if not paths:
            raise SystemExit(f"No *.csv files in {d}")
        if dataset_names:
            if len(dataset_names) != len(paths):
                raise SystemExit(
                    f"--dataset count ({len(dataset_names)}) must match "
                    f"CSV count ({len(paths)}) in --data-dir, or omit --dataset for stems"
                )
            return list(zip(paths, dataset_names))
        return [(p, p.stem) for p in paths]

    if not csv_paths:
        raise SystemExit("Provide --csv (one or more) or --data-dir")

    names = dataset_names or []
    if not names:
        return [(p.resolve(), p.stem) for p in csv_paths]
    if len(names) == len(csv_paths):
        return [(p.resolve(), n) for p, n in zip(csv_paths, names)]
    raise SystemExit(
        "With --csv: give one --dataset per --csv (same order), or omit --dataset "
        "to use each file's basename (without .csv)"
    )


def _validate_csv_columns(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, nrows=0)
    missing = [c for c in ("name", "heavy", "light") if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing columns {missing}")


def process_one_dataset(
    csv_path: Path,
    dataset: str,
    predictor,
    out_root: Path,
    runs: int,
    skip_existing: bool,
    n_threads_save: int,
) -> None:
    df = pd.read_csv(csv_path)
    missing = [c for c in ("name", "heavy", "light") if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing columns {missing}")

    print(f"\n>>> Dataset {dataset!r} ({csv_path.name})", flush=True)

    for run in range(1, runs + 1):
        run_dir = out_root / f"{dataset}_abb2_{run}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== run {run}/{runs} -> {run_dir} ===", flush=True)

        n_ok, n_skip, n_fail = 0, 0, 0

        for _, row in df.iterrows():
            name = str(row["name"]).strip() if pd.notna(row["name"]) else ""
            heavy = str(row["heavy"]).strip() if pd.notna(row["heavy"]) else ""
            light = str(row["light"]).strip() if pd.notna(row["light"]) else ""
            if not name or not heavy or not light:
                n_fail += 1
                continue

            safe = name.replace("|", "_")
            pdb_path = run_dir / f"{safe}.pdb"
            if skip_existing and pdb_path.is_file():
                n_skip += 1
                continue

            sequences = {"H": heavy, "L": light}
            antibody: object | None = None
            try:
                antibody = predictor.predict(sequences)
            except Exception as e:
                print(f"  FAIL {name!r}: {e}", flush=True)
                n_fail += 1
                continue

            if antibody is None:
                n_fail += 1
                continue

            try:
                antibody.save(
                    str(pdb_path),
                    check_for_strained_bonds=True,
                    n_threads=n_threads_save,
                )
                n_ok += 1
            except Exception as e:
                print(f"  FAIL {name!r} (save): {e}", flush=True)
                n_fail += 1

        print(
            f"  run {run}: wrote {n_ok}, skipped {n_skip}, failed {n_fail}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --csv data/pdgf38.csv --output-root structures --dataset pdgf38
  %(prog)s --data-dir data --output-root structures
        """.strip(),
    )
    p.add_argument(
        "--csv",
        action="append",
        default=None,
        metavar="PATH",
        help="Input CSV (repeat for multiple datasets).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Process every top-level *.csv in DIR (non-recursive).",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="STEM",
        help="Output folder prefix per CSV (same order as --csv or sorted *.csv).",
    )
    p.add_argument("--output-root", required=True, type=Path, help="structures/ parent")
    p.add_argument("--runs", type=int, default=1, help="Run folders: _abb2_1 …")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose .pdb already exists in that run directory",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Accepted for parity with the ABB3 driver; not used for ABB2 inference.",
    )
    p.add_argument(
        "--device",
        default="cuda:1",
        help='Logical device hint (default cuda:1). Maps to CUDA_VISIBLE_DEVICES '
        'before import; use cpu to force CPU.',
    )
    p.add_argument(
        "--refine-threads",
        type=int,
        default=1,
        help="n_threads passed to OpenMM refinement in antibody.save (default: 1).",
    )
    args = p.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    csv_arg_list = [Path(x) for x in args.csv] if args.csv else None
    dataset_names = list(args.dataset) if args.dataset else None
    jobs = _build_dataset_jobs(csv_arg_list, args.data_dir, dataset_names)

    for csv_path, _stem in jobs:
        if not csv_path.is_file():
            raise SystemExit(f"Not a file: {csv_path}")
        _validate_csv_columns(csv_path)

    from ImmuneBuilder import ABodyBuilder2

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
    predictor = ABodyBuilder2()
    print(f"ABodyBuilder2 device: {predictor.device}", flush=True)

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Datasets to run ({len(jobs)}):", flush=True)
    for csv_path, stem in jobs:
        print(f"  - {stem!r} <- {csv_path}", flush=True)

    n_threads_save = args.refine_threads
    if n_threads_save == 0:
        n_threads_save = -1

    for csv_path, dataset in jobs:
        process_one_dataset(
            csv_path,
            dataset,
            predictor,
            out_root,
            args.runs,
            args.skip_existing,
            n_threads_save,
        )

    print(f"\nDone. Outputs under {out_root}", flush=True)


if __name__ == "__main__":
    main()
