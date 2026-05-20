#!/usr/bin/env python3
"""
Renumber ABodyBuilder3 PDB files to IMGT numbering using ANARCI.

Input:
  one or more directories containing ABB3-style PDB files.

Output:
  by default, for each input directory creates a sibling directory with _imgt suffix,
  e.g. abb3_dir1 -> abb3_dir1_imgt, and saves PDB files under original names.
  optionally, --out-dir mirrors all processed files under a single output directory.

Notes:
  * PDB insertion codes are written in the standard PDB location: columns 23-26
    contain residue number, column 27 contains insertion code. Therefore IMGT
    insertions will display as e.g. "H 112A" in the ATOM records.
  * Existing ABB3 residue numbers are ignored. Numbering is reconstructed from
    the residue sequence of each chain.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from anarci import anarci
except ImportError as exc:
    raise SystemExit(
        "ERROR: Could not import ANARCI.\n"
        "Install it first, for example in a conda environment where ANARCI works:\n"
        "  pip install anarci\n"
        "or follow the ANARCI repository installation instructions.\n"
    ) from exc


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common PDB aliases. ABB3 normally uses standard residues, but these help.
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
}

ATOMISH_RECORDS = ("ATOM  ", "HETATM")
REN_NUMBERABLE_RECORDS = ("ATOM  ", "HETATM", "TER   ")


@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    resseq: int
    icode: str


@dataclass
class ChainInfo:
    chain_id: str
    residues: List[ResidueKey]
    resnames: List[str]
    sequence: str


@dataclass
class NumberingResult:
    chain_type: str
    species: str
    start_index: int
    end_index: int
    mapping_by_index: Dict[int, Tuple[int, str]]


class RenumberingError(RuntimeError):
    pass


def parse_pdb_chains(pdb_path: Path, selected_chains: Optional[set[str]] = None) -> Dict[str, ChainInfo]:
    """
    Extract chain sequences from ATOM/HETATM records.

    Residues are deduplicated by (chain_id, resseq, insertion_code), preserving
    the first order in which they appear in the file.
    """
    residues_by_chain: Dict[str, OrderedDict[ResidueKey, str]] = {}

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(ATOMISH_RECORDS):
                continue
            if len(line) < 27:
                continue

            chain_id = line[21]
            if selected_chains is not None and chain_id not in selected_chains:
                continue

            resname = line[17:20].strip().upper()
            if resname not in THREE_TO_ONE:
                # Ignore non-protein HETATM residues, water, ligands, ions, etc.
                if line.startswith("HETATM"):
                    continue
                raise RenumberingError(
                    f"{pdb_path}: unsupported residue {resname!r} in chain {chain_id!r}. "
                    "Add it to THREE_TO_ONE if this is intentional."
                )

            raw_resseq = line[22:26].strip()
            if not raw_resseq:
                raise RenumberingError(f"{pdb_path}: empty residue number in line: {line.rstrip()}")
            try:
                resseq = int(raw_resseq)
            except ValueError as exc:
                raise RenumberingError(
                    f"{pdb_path}: cannot parse residue number {raw_resseq!r} in line: {line.rstrip()}"
                ) from exc

            icode = line[26]
            key = ResidueKey(chain_id=chain_id, resseq=resseq, icode=icode)

            residues_by_chain.setdefault(chain_id, OrderedDict())
            if key not in residues_by_chain[chain_id]:
                residues_by_chain[chain_id][key] = resname

    chains: Dict[str, ChainInfo] = {}
    for chain_id, od in residues_by_chain.items():
        residues = list(od.keys())
        resnames = list(od.values())
        sequence = "".join(THREE_TO_ONE[x] for x in resnames)
        if sequence:
            chains[chain_id] = ChainInfo(
                chain_id=chain_id,
                residues=residues,
                resnames=resnames,
                sequence=sequence,
            )

    return chains


def normalize_icode(icode: object) -> str:
    """
    Normalize ANARCI insertion code to one PDB-character insertion code.

    ANARCI usually returns ' ' for no insertion and 'A', 'B', ... for insertions.
    """
    if icode is None:
        return " "
    icode_str = str(icode)
    if icode_str == "" or icode_str == " ":
        return " "
    if len(icode_str) != 1:
        raise RenumberingError(f"ANARCI returned a multi-character insertion code: {icode_str!r}")
    return icode_str


def run_anarci_imgt(name: str, sequence: str, allow_partial: bool = False) -> NumberingResult:
    """
    Run ANARCI and return IMGT numbering for one antibody/TCR domain sequence.

    For ABB3 heavy/light chain outputs, we expect exactly one domain per chain
    and expect ANARCI to cover the full chain. If it does not, fail unless
    --allow-partial-domain is set.
    """
    numbering, alignment_details, _hit_tables = anarci(
        [(name, sequence)],
        scheme="imgt",
        output=False,
    )

    if not numbering or numbering[0] is None:
        raise RenumberingError(f"ANARCI did not number {name}.")

    domains = numbering[0]
    if len(domains) != 1:
        raise RenumberingError(
            f"ANARCI found {len(domains)} domains for {name}. "
            "This script expects one variable domain per PDB chain."
        )

    domain_numbering, start_index, end_index = domains[0]
    details = alignment_details[0][0] if alignment_details and alignment_details[0] else {}

    if not allow_partial and (start_index != 0 or end_index != len(sequence) - 1):
        raise RenumberingError(
            f"ANARCI numbered only residues {start_index}..{end_index} of {name} "
            f"(sequence length {len(sequence)}). Refusing to create mixed numbering. "
            "Use --allow-partial-domain if you intentionally want to keep unmapped residues unchanged."
        )

    observed: List[Tuple[Tuple[int, str], str]] = []
    for (position, aa) in domain_numbering:
        # position is usually a tuple like (112, 'A'); aa is '-' for a numbering gap.
        if aa == "-":
            continue
        if not isinstance(position, tuple) or len(position) != 2:
            raise RenumberingError(f"Unexpected ANARCI position object for {name}: {position!r}")
        resnum, icode = position
        observed.append(((int(resnum), normalize_icode(icode)), aa))

    domain_seq = sequence[start_index : end_index + 1]
    if len(observed) != len(domain_seq):
        raise RenumberingError(
            f"ANARCI returned {len(observed)} non-gap positions for {name}, "
            f"but the numbered sequence segment has {len(domain_seq)} residues."
        )

    mapping_by_index: Dict[int, Tuple[int, str]] = {}
    for local_i, ((resnum, icode), aa) in enumerate(observed):
        seq_i = start_index + local_i
        if sequence[seq_i] != aa:
            raise RenumberingError(
                f"Sequence mismatch for {name} at index {seq_i}: "
                f"PDB-derived sequence has {sequence[seq_i]!r}, ANARCI returned {aa!r}."
            )
        mapping_by_index[seq_i] = (resnum, icode)

    return NumberingResult(
        chain_type=str(details.get("chain_type", "")),
        species=str(details.get("species", "")),
        start_index=int(start_index),
        end_index=int(end_index),
        mapping_by_index=mapping_by_index,
    )


def build_residue_mapping(
    pdb_path: Path,
    selected_chains: Optional[set[str]] = None,
    allow_partial: bool = False,
) -> Tuple[Dict[ResidueKey, Tuple[int, str]], List[dict]]:
    """
    Return mapping from old residue identifiers to new IMGT residue identifiers.
    """
    chains = parse_pdb_chains(pdb_path, selected_chains=selected_chains)
    if not chains:
        raise RenumberingError(f"{pdb_path}: no protein chains found.")

    old_to_new: Dict[ResidueKey, Tuple[int, str]] = {}
    report_rows: List[dict] = []

    for chain_id, chain in chains.items():
        result = run_anarci_imgt(f"{pdb_path.stem}:{chain_id}", chain.sequence, allow_partial=allow_partial)

        for idx, old_key in enumerate(chain.residues):
            if idx in result.mapping_by_index:
                old_to_new[old_key] = result.mapping_by_index[idx]
            elif allow_partial:
                # Keep the original number if ANARCI did not cover it.
                old_to_new[old_key] = (old_key.resseq, old_key.icode)
            else:
                raise RenumberingError(
                    f"{pdb_path}: residue index {idx} in chain {chain_id!r} has no ANARCI numbering."
                )

        report_rows.append(
            {
                "pdb": str(pdb_path),
                "chain_id": chain_id,
                "chain_type": result.chain_type,
                "species": result.species,
                "sequence_length": len(chain.sequence),
                "anarci_start_index": result.start_index,
                "anarci_end_index": result.end_index,
                "first_old_residue": f"{chain.residues[0].resseq}{chain.residues[0].icode.strip()}",
                "last_old_residue": f"{chain.residues[-1].resseq}{chain.residues[-1].icode.strip()}",
                "first_imgt_residue": format_residue_id(*old_to_new[chain.residues[0]]),
                "last_imgt_residue": format_residue_id(*old_to_new[chain.residues[-1]]),
            }
        )

    return old_to_new, report_rows


def format_residue_id(resseq: int, icode: str) -> str:
    return f"{resseq}{icode.strip()}"


def rewrite_pdb(
    input_pdb: Path,
    output_pdb: Path,
    old_to_new: Dict[ResidueKey, Tuple[int, str]],
    overwrite: bool = False,
) -> None:
    if output_pdb.exists() and not overwrite:
        raise RenumberingError(f"Output already exists: {output_pdb}. Use --overwrite to replace it.")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    with input_pdb.open("r", encoding="utf-8", errors="replace") as inp, output_pdb.open(
        "w", encoding="utf-8"
    ) as out:
        for line in inp:
            if line.startswith(REN_NUMBERABLE_RECORDS) and len(line) >= 27:
                chain_id = line[21]
                raw_resseq = line[22:26].strip()
                if raw_resseq:
                    try:
                        old_key = ResidueKey(chain_id=chain_id, resseq=int(raw_resseq), icode=line[26])
                    except ValueError:
                        old_key = None

                    if old_key is not None and old_key in old_to_new:
                        new_resseq, new_icode = old_to_new[old_key]
                        if not (-999 <= new_resseq <= 9999):
                            raise RenumberingError(
                                f"New residue number {new_resseq} does not fit PDB columns for {input_pdb}."
                            )
                        line = line[:22] + f"{new_resseq:4d}{new_icode:1s}" + line[27:]

            out.write(line)


def find_pdbs(input_dirs: Iterable[Path], pattern: str, recursive: bool) -> List[Path]:
    pdbs: List[Path] = []
    for d in input_dirs:
        if not d.exists():
            raise RenumberingError(f"Input path does not exist: {d}")
        if not d.is_dir():
            raise RenumberingError(f"Input path is not a directory: {d}")
        iterator = d.rglob(pattern) if recursive else d.glob(pattern)
        pdbs.extend(sorted(p for p in iterator if p.is_file()))
    return sorted(set(pdbs))


def choose_output_path(
    input_pdb: Path,
    input_dirs: List[Path],
    out_dir: Optional[Path],
    suffix: str,
) -> Path:
    """
    If --out-dir is not provided:
      abb3_dir1/sample.pdb -> abb3_dir1_imgt/sample.pdb

    If --recursive is used:
      abb3_dir1/sub/sample.pdb -> abb3_dir1_imgt/sub/sample.pdb

    If --out-dir is provided:
      mirror files under --out-dir, preserving original filenames.
    """
    for root in input_dirs:
        try:
            rel = input_pdb.relative_to(root)
        except ValueError:
            continue

        if out_dir is None:
            output_root = root.with_name(f"{root.name}{suffix}")
            return output_root / rel

        return out_dir / rel

    # Fallback; should rarely happen because PDBs come from input_dirs.
    if out_dir is None:
        output_root = input_pdb.parent.with_name(f"{input_pdb.parent.name}{suffix}")
        return output_root / input_pdb.name

    return out_dir / input_pdb.name


def parse_chain_selection(value: Optional[str]) -> Optional[set[str]]:
    if value is None or value.strip() == "":
        return None
    chains = [x.strip() for x in value.split(",") if x.strip()]
    if not chains:
        return None
    bad = [x for x in chains if len(x) != 1]
    if bad:
        raise argparse.ArgumentTypeError(f"Chain IDs must be single characters, got: {bad}")
    return set(chains)


def write_report(report_path: Path, rows: List[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "pdb",
        "output_pdb",
        "message",
        "chain_id",
        "chain_type",
        "species",
        "sequence_length",
        "anarci_start_index",
        "anarci_end_index",
        "first_old_residue",
        "last_old_residue",
        "first_imgt_residue",
        "last_imgt_residue",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Renumber ABB3 PDB files to IMGT numbering with ANARCI."
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing PDB files.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, files are written next to inputs.",
    )
    parser.add_argument(
        "--glob",
        default="*.pdb",
        help="File glob to select PDBs inside each input directory. Default: *.pdb",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
    )
    parser.add_argument(
        "--suffix",
        default="_imgt",
        help="Suffix for output directories. Default: _imgt",
    )
    parser.add_argument(
        "--chains",
        type=parse_chain_selection,
        default=None,
        help="Comma-separated PDB chain IDs to renumber, e.g. H,L. Default: all protein chains.",
    )
    parser.add_argument(
        "--allow-partial-domain",
        action="store_true",
        help=(
            "Allow ANARCI to number only part of a chain; unmapped residues keep original numbers. "
            "Not recommended for normal ABB3 variable-domain outputs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional TSV report path.",
    )

    args = parser.parse_args(argv)

    input_dirs = [p.resolve() for p in args.input_dirs]
    pdbs = find_pdbs(input_dirs, pattern=args.glob, recursive=args.recursive)

    if not pdbs:
        print("No PDB files found.", file=sys.stderr)
        return 2

    all_report_rows: List[dict] = []
    n_ok = 0
    n_fail = 0

    for pdb_path in pdbs:
        out_path = choose_output_path(
            input_pdb=pdb_path,
            input_dirs=input_dirs,
            out_dir=args.out_dir.resolve() if args.out_dir is not None else None,
            suffix=args.suffix,
        )

        try:
            old_to_new, chain_rows = build_residue_mapping(
                pdb_path,
                selected_chains=args.chains,
                allow_partial=args.allow_partial_domain,
            )
            rewrite_pdb(pdb_path, out_path, old_to_new, overwrite=args.overwrite)
            n_ok += 1
            print(f"OK\t{pdb_path}\t->\t{out_path}")

            for row in chain_rows:
                row.update(
                    {
                        "status": "ok",
                        "output_pdb": str(out_path),
                        "message": "",
                    }
                )
                all_report_rows.append(row)

        except Exception as exc:
            n_fail += 1
            msg = str(exc)
            print(f"FAIL\t{pdb_path}\t{msg}", file=sys.stderr)
            all_report_rows.append(
                {
                    "status": "fail",
                    "pdb": str(pdb_path),
                    "output_pdb": str(out_path),
                    "message": msg,
                }
            )

    if args.report is not None:
        write_report(args.report, all_report_rows)
        print(f"Report written: {args.report}")

    print(f"Done. Successful: {n_ok}; failed: {n_fail}; total: {len(pdbs)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
