#!/usr/bin/env python3
"""Rerun FlashABB prediction + OpenMM minimization for failed structures only.

Reads a TSV produced from our_flashabb descriptor failures (dataset, structure, reason).
For each row, re-predicts the named antibody in the matching structures_flashabb run folder
and minimizes the PDB in place.

Example:
  python src/structure/rerun_failed_flashabb.py \\
    --failures-tsv our_flashabb/failed_structures.tsv \\
    --structures-root structures_flashabb \\
    --datasets-dir datasets \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from math import dist
from pathlib import Path

_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLASHABB_RUN_RE = re.compile(r"^(?P<stem>.+)_flashabb_(?P<run>\d+)$")


def _parse_failures(
    tsv_path: Path,
    *,
    structures_root: Path,
    datasets_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    with tsv_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            name = (row.get("structure") or "").strip()
            if not dataset or not name:
                continue
            match = _FLASHABB_RUN_RE.match(dataset)
            if not match:
                raise SystemExit(f"Cannot parse dataset folder name: {dataset!r}")
            stem = match.group("stem")
            run = int(match.group("run"))
            csv_path = datasets_dir / f"{stem}.csv"
            if not csv_path.is_file():
                raise SystemExit(f"CSV not found for {dataset}: {csv_path}")
            struct_dir = structures_root / dataset
            rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "run": run,
                    "name": name,
                    "csv_path": csv_path,
                    "struct_dir": struct_dir,
                }
            )
    if not rows:
        raise SystemExit(f"No failure rows found in {tsv_path}")
    return rows


def _group_rows(rows: list[dict]) -> dict[tuple[Path, str, int, Path], set[str]]:
    grouped: dict[tuple[Path, str, int, Path], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["csv_path"], row["stem"], row["run"], row["struct_dir"])
        grouped[key].add(row["name"])
    return grouped


def _bad_peptide_bonds(pdb_path: Path, chain: str = "H", lo: int = 100, hi: int = 130) -> list[tuple]:
    by_key: dict[tuple[int, str], dict[str, tuple[float, float, float]]] = {}
    order: list[tuple[int, str]] = []
    for line in pdb_path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM") or len(line) < 54 or line[21] != chain:
            continue
        key = (int(line[22:26]), line[26].strip())
        if key not in by_key:
            by_key[key] = {}
            order.append(key)
        by_key[key][line[12:16].strip()] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    bad = []
    for idx in range(len(order) - 1):
        left, right = order[idx], order[idx + 1]
        if not (lo <= left[0] <= hi or lo <= right[0] <= hi):
            continue
        if "C" not in by_key[left] or "N" not in by_key[right]:
            continue
        bond = dist(by_key[left]["C"], by_key[right]["N"])
        if bond < 1.0 or bond > 2.0:
            bad.append((left, right, bond))
    return bad


def _load_sequences(csv_path: Path, names: set[str]) -> list[tuple[str, str]]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in ("name", "heavy", "light") if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"{csv_path}: missing columns {missing_cols}")

    out: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        name = str(row["name"]).strip() if pd.notna(row["name"]) else ""
        heavy = str(row["heavy"]).strip() if pd.notna(row["heavy"]) else ""
        light = str(row["light"]).strip() if pd.notna(row["light"]) else ""
        if name in names:
            if not heavy or not light:
                raise SystemExit(f"{csv_path}: missing heavy/light for {name!r}")
            out.append((name, f"{heavy}|{light}"))
    missing = sorted(names - {n for n, _ in out})
    if missing:
        raise SystemExit(f"{csv_path}: names not found in CSV: {missing}")
    return out


def _predict_flashabb(
    model,
    struct_dir: Path,
    items: list[tuple[str, str]],
    *,
    batch_size: int,
) -> None:
    struct_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    seqs: list[str] = []

    def flush() -> None:
        nonlocal names, seqs
        if not names:
            return
        import torch

        with torch.no_grad():
            result = model(seqs)
        result.to_pdbs(names, pdb_dir=str(struct_dir))
        print(f"  predicted batch of {len(names)} -> {struct_dir}", flush=True)
        names = []
        seqs = []

    for name, seq in items:
        pdb_path = struct_dir / f"{name}.pdb"
        if pdb_path.is_file():
            pdb_path.unlink()
        names.append(name)
        seqs.append(seq)
        if len(names) >= batch_size:
            flush()
    flush()


def _geometry_ok(pdb_path: Path) -> tuple[bool, list, list]:
    bad_h = _bad_peptide_bonds(pdb_path, chain="H")
    bad_l = _bad_peptide_bonds(pdb_path, chain="L", lo=1, hi=130)
    return (not bad_h and not bad_l), bad_h, bad_l


def _report_geometry(name: str, pdb_path: Path) -> bool:
    ok, bad_h, bad_l = _geometry_ok(pdb_path)
    if ok:
        print(f"  geometry check {name}: OK", flush=True)
        return True
    print(f"  geometry check {name}: BAD bonds H={len(bad_h)} L={len(bad_l)}", flush=True)
    for left, right, bond in bad_h[:3]:
        print(
            f"    H {left[0]}{left[1]} -> {right[0]}{right[1]}: {bond:.2f} A",
            flush=True,
        )
    for left, right, bond in bad_l[:3]:
        print(
            f"    L {left[0]}{left[1]} -> {right[0]}{right[1]}: {bond:.2f} A",
            flush=True,
        )
    return False


def _minimize_pdbs(struct_dir: Path, names: set[str], *, jobs: int) -> None:
    from structure.postprocess_structures import refine

    tasks = []
    for name in sorted(names):
        pdb_path = struct_dir / f"{name}.pdb"
        if not pdb_path.is_file():
            raise SystemExit(f"Missing PDB after prediction: {pdb_path}")
        tasks.append(pdb_path)

    print(f"  minimizing {len(tasks)} structure(s) in place under {struct_dir}", flush=True)

    n_ok = n_warn = 0
    for pdb_path in tasks:
        ok = refine(str(pdb_path), str(pdb_path), n_threads=-1)
        if ok:
            n_ok += 1
            print(f"    minimize ok   {pdb_path.name}", flush=True)
        else:
            n_warn += 1
            print(f"    minimize warn {pdb_path.name}", flush=True)
    print(
        f"  minimize done: {n_ok} ok, {n_warn} warn, 0 failed",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failures-tsv",
        type=Path,
        default=_REPO_ROOT / "our_flashabb" / "failed_structures.tsv",
    )
    parser.add_argument(
        "--structures-root",
        type=Path,
        default=_REPO_ROOT / "structures_flashabb",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=_REPO_ROOT / "datasets",
    )
    parser.add_argument("--device", default=os.environ.get("FLASHABB_DEVICE", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--minimize-jobs", type=int, default=4)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Retry FlashABB+minimize per structure until geometry passes (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without predicting or minimizing",
    )
    args = parser.parse_args(argv)

    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")

    rows = _parse_failures(
        args.failures_tsv.resolve(),
        structures_root=args.structures_root.resolve(),
        datasets_dir=args.datasets_dir.resolve(),
    )
    grouped = _group_rows(rows)

    print(f"Failures TSV: {args.failures_tsv}", flush=True)
    print(f"Unique rerun jobs: {len(grouped)} dataset/run folder(s)", flush=True)
    for (csv_path, stem, run, struct_dir), names in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][2])
    ):
        print(
            f"  {struct_dir.name}: {len(names)} structure(s) "
            f"<- {csv_path.name} ({', '.join(sorted(names, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)))})",
            flush=True,
        )

    if args.dry_run:
        return 0

    flashabb_dir = os.environ.get("FLASHABB_DIR", str(_REPO_ROOT / "FlashABB"))
    if flashabb_dir and Path(flashabb_dir).is_dir() and flashabb_dir not in sys.path:
        sys.path.insert(0, flashabb_dir)

    from structure.run_abb_batch_from_csv import _early_set_cuda_visible_device, _resolve_torch_device
    from flash_abb import pretrained

    _early_set_cuda_visible_device(args.device)
    import torch

    device = _resolve_torch_device(args.device)
    print(f"Loading FlashABB on {device} (cuda available={torch.cuda.is_available()})", flush=True)
    t0 = time.monotonic()
    model = pretrained(device=str(device))
    print(f"FlashABB loaded in {time.monotonic() - t0:.1f}s", flush=True)

    still_bad: list[str] = []

    for (csv_path, stem, run, struct_dir), names in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][2])
    ):
        print(f"\n=== {struct_dir.name} ({len(names)} structure(s)) ===", flush=True)
        seq_by_name = dict(_load_sequences(csv_path, names))

        for name in sorted(names, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            pdb_path = struct_dir / f"{name}.pdb"
            ok = False
            for attempt in range(1, args.max_attempts + 1):
                if attempt > 1:
                    print(f"  retry {name}: attempt {attempt}/{args.max_attempts}", flush=True)
                _predict_flashabb(
                    model,
                    struct_dir,
                    [(name, seq_by_name[name])],
                    batch_size=1,
                )
                _minimize_pdbs(struct_dir, {name}, jobs=args.minimize_jobs)
                ok = _report_geometry(name, pdb_path)
                if ok:
                    break
            if not ok:
                still_bad.append(f"{struct_dir.name}/{name}")

    print("\nDone.", flush=True)
    if still_bad:
        print(f"Still broken after {args.max_attempts} attempt(s): {len(still_bad)}", flush=True)
        for entry in still_bad:
            print(f"  - {entry}", flush=True)
        print("\nRe-run descriptors only for structures that passed geometry check:", flush=True)
        print("  ./kitab.sh configs/scenario2.yaml --resume", flush=True)
        return 1

    print("All structures passed geometry check.", flush=True)
    print("Re-run descriptors, e.g.:", flush=True)
    print("  ./kitab.sh configs/scenario2.yaml --resume", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
