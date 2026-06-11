#!/usr/bin/env python3
"""CLI: build PropKa-only input PDB (with optional IMGT insertion renumbering)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils.propka_pdb import prepare_propka_input


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare PropKa input PDB (renumber insertion-code collisions when needed)."
    )
    parser.add_argument("source_pdb", type=Path, help="Original IMGT-numbered structure PDB")
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        required=True,
        help="Directory for temporary PropKa PDB/map files (e.g. propka/tmp_structures)",
    )
    parser.add_argument(
        "--stem",
        required=True,
        help="Base name for temp outputs (e.g. mAb27_full, mAb27_H, mAb27_L)",
    )
    parser.add_argument(
        "--chain",
        choices=("H", "L"),
        default=None,
        help="Optional chain filter when preparing split H/L PropKa inputs",
    )
    args = parser.parse_args()

    result = prepare_propka_input(
        args.source_pdb,
        tmp_dir=args.tmp_dir,
        stem=args.stem,
        chain_filter=args.chain,
    )
    # stdout: absolute path to PDB file PropKa should read
    print(result.propka_pdb)
    if result.map_path:
        print(f"PropKa map: {result.map_path}", file=sys.stderr)
        print(f"Renumbered {result.remapped_count} residue(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
