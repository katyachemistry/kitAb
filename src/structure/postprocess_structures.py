#!/usr/bin/env python3
"""IMGT renumbering (ANARCI) and OpenMM minimization. Subcommands: renumber, minimize."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils.parsers import Atom, parse_cif, parse_structure, write_structure_pdb  # noqa: E402

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
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
}

ATOMISH_RECORDS = ("ATOM  ", "HETATM")

# IMGT CDR3: insertion codes at 112 sort C-terminal anchor backwards.
IMGT_REVERSE_INSERTION_RESNUMS: frozenset = frozenset({112})


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


STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


def normalize_pdb_icode(icode: str) -> str:
    icode = (icode or "").strip().replace(".", "")
    if not icode:
        return " "
    if len(icode) != 1:
        raise RenumberingError(f"Unsupported insertion code {icode!r}")
    return icode


def _chains_from_residue_maps(
    residues_by_chain: Dict[str, OrderedDict[ResidueKey, str]],
    source: Path,
) -> Dict[str, ChainInfo]:
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
    if not chains:
        raise RenumberingError(f"{source}: no protein chains found.")
    return chains


def parse_pdb_chains(pdb_path: Path, selected_chains: Optional[set[str]] = None) -> Dict[str, ChainInfo]:
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

            key = ResidueKey(
                chain_id=chain_id,
                resseq=resseq,
                icode=normalize_pdb_icode(line[26]),
            )

            residues_by_chain.setdefault(chain_id, OrderedDict())
            if key not in residues_by_chain[chain_id]:
                residues_by_chain[chain_id][key] = resname

    return _chains_from_residue_maps(residues_by_chain, pdb_path)


def parse_cif_chains(cif_path: Path, selected_chains: Optional[set[str]] = None) -> Dict[str, ChainInfo]:
    atoms = parse_cif(
        str(cif_path),
        allowed_chains=selected_chains,
        all_chains=(selected_chains is None),
    )
    if not atoms:
        raise RenumberingError(f"{cif_path}: no protein ATOM records in mmCIF.")

    residues_by_chain: Dict[str, OrderedDict[ResidueKey, str]] = {}
    for atom in atoms:
        resname = atom.residue_name.upper()
        if resname not in THREE_TO_ONE:
            continue
        key = ResidueKey(
            chain_id=atom.chain,
            resseq=atom.residue_number,
            icode=normalize_pdb_icode(atom.insertion_code),
        )
        if selected_chains is not None and key.chain_id not in selected_chains:
            continue
        residues_by_chain.setdefault(key.chain_id, OrderedDict())
        if key not in residues_by_chain[key.chain_id]:
            residues_by_chain[key.chain_id][key] = resname

    return _chains_from_residue_maps(residues_by_chain, cif_path)


def run_anarci_imgt(name: str, sequence: str, allow_partial: bool = False) -> NumberingResult:
    try:
        from anarci import anarci
    except ImportError as exc:
        raise SystemExit(
            "ERROR: Could not import ANARCI.\n"
            "Install it first, for example in a conda environment where ANARCI works:\n"
            "  pip install anarci\n"
            "or follow the ANARCI repository installation instructions.\n"
        ) from exc

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
        if aa == "-":
            continue
        if not isinstance(position, tuple) or len(position) != 2:
            raise RenumberingError(f"Unexpected ANARCI position object for {name}: {position!r}")
        resnum, icode = position
        if icode is None:
            icode_norm = " "
        else:
            icode_str = str(icode)
            if icode_str == "" or icode_str == " ":
                icode_norm = " "
            elif len(icode_str) != 1:
                raise RenumberingError(
                    f"ANARCI returned a multi-character insertion code: {icode_str!r}"
                )
            else:
                icode_norm = icode_str
        observed.append(((int(resnum), icode_norm), aa))

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


def imgt_residue_sort_key(resseq: int, icode: str) -> Tuple[int, str]:
    icode = icode.strip()
    if resseq in IMGT_REVERSE_INSERTION_RESNUMS:
        if not icode:
            return (resseq, "~")
        return (resseq, chr(ord("A") + ord("Z") - ord(icode.upper())))
    return (resseq, icode)


def build_residue_mapping(
    structure_path: Path,
    selected_chains: Optional[set[str]] = None,
    allow_partial: bool = False,
) -> Tuple[Dict[ResidueKey, Tuple[int, str]], List[dict], Set[ResidueKey]]:
    if structure_path.suffix.lower() in (".cif", ".mmcif"):
        chains = parse_cif_chains(structure_path, selected_chains=selected_chains)
    else:
        chains = parse_pdb_chains(structure_path, selected_chains=selected_chains)

    old_to_new: Dict[ResidueKey, Tuple[int, str]] = {}
    output_keys: Set[ResidueKey] = set()
    report_rows: List[dict] = []

    for chain_id, chain in chains.items():
        result = run_anarci_imgt(
            f"{structure_path.stem}:{chain_id}", chain.sequence, allow_partial=allow_partial
        )

        for idx, old_key in enumerate(chain.residues):
            if idx in result.mapping_by_index:
                old_to_new[old_key] = result.mapping_by_index[idx]
                output_keys.add(old_key)
            elif not allow_partial:
                raise RenumberingError(
                    f"{structure_path}: residue index {idx} in chain {chain_id!r} has no ANARCI numbering."
                )

        if not any(k.chain_id == chain_id for k in output_keys):
            raise RenumberingError(
                f"{structure_path}: chain {chain_id!r} has no ANARCI-numbered residues."
            )

        mapped_residues = [k for k in chain.residues if k in output_keys]
        report_rows.append(
            {
                "pdb": str(structure_path),
                "chain_id": chain_id,
                "chain_type": result.chain_type,
                "species": result.species,
                "sequence_length": len(chain.sequence),
                "anarci_start_index": result.start_index,
                "anarci_end_index": result.end_index,
                "first_old_residue": f"{mapped_residues[0].resseq}{mapped_residues[0].icode.strip()}",
                "last_old_residue": f"{mapped_residues[-1].resseq}{mapped_residues[-1].icode.strip()}",
                "first_imgt_residue": f"{old_to_new[mapped_residues[0]][0]}{old_to_new[mapped_residues[0]][1].strip()}",
                "last_imgt_residue": f"{old_to_new[mapped_residues[-1]][0]}{old_to_new[mapped_residues[-1]][1].strip()}",
            }
        )

    return old_to_new, report_rows, output_keys


def format_pdb_atom_line(serial: int, atom: Atom, resseq: int, icode: str) -> str:
    icode_c = icode if (icode and icode.strip()) else " "
    name = atom.name.strip()
    name_field = name[:4] if len(name) >= 4 else f" {name:<3}"[:4]
    resname = atom.residue_name[:3].upper()
    element = (atom.element or "").strip()[:2] or name_field.strip()[:1]
    element = f"{element:>2s}"[:2]
    line = (
        f"ATOM  {serial:5d} {name_field:4s} {resname:>3} {atom.chain:1s}"
        f"{resseq:4d}{icode_c:1s}   {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"  1.00  0.00          {element:>2s}\n"
    )
    return line if len(line) >= 80 else line.rstrip("\n").ljust(80) + "\n"


def _write_slim_pdb(
    output_pdb: Path,
    atom_lines: List[Tuple[str, int, str, str, str]],
    *,
    overwrite: bool,
) -> None:
    if output_pdb.exists() and not overwrite:
        raise RenumberingError(f"Output already exists: {output_pdb}. Use --overwrite to replace it.")
    if not atom_lines:
        raise RenumberingError(f"No ATOM records to write for {output_pdb}.")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    atom_lines.sort(
        key=lambda item: (
            item[0],
            *imgt_residue_sort_key(item[1], item[2]),
            item[4][12:16] if len(item[4]) > 16 else item[4],
        )
    )

    with output_pdb.open("w", encoding="utf-8") as out:
        out.write("REMARK   1 IMGT numbered with ANARCI (antibody chains only)\n")
        serial = 1
        prev_chain: Optional[str] = None
        prev_residue: Optional[Tuple[int, str]] = None
        prev_resname = ""

        for i, (chain_id, new_resseq, new_icode, resname, line) in enumerate(atom_lines):
            if prev_chain is not None and chain_id != prev_chain:
                out.write(
                    f"TER   {serial:5d}      {prev_resname:>3}{prev_chain:>2}"
                    f"{prev_residue[0]:4d}{prev_residue[1]:1s}\n"
                )
                serial += 1
            out.write(line if line.endswith("\n") else f"{line}\n")
            serial += 1
            prev_chain = chain_id
            prev_residue = (new_resseq, new_icode)
            prev_resname = resname

        if prev_chain is not None and prev_residue is not None:
            out.write(
                f"TER   {serial:5d}      {prev_resname:>3}{prev_chain:>2}"
                f"{prev_residue[0]:4d}{prev_residue[1]:1s}\n"
            )
        out.write("END\n")


def write_antibody_pdb(
    input_pdb: Path,
    output_pdb: Path,
    old_to_new: Dict[ResidueKey, Tuple[int, str]],
    output_keys: Set[ResidueKey],
    *,
    selected_chains: Optional[set[str]] = None,
    overwrite: bool = False,
) -> None:
    records: List[Tuple[str, int, str, str, str]] = []
    with input_pdb.open("r", encoding="utf-8", errors="replace") as inp:
        for line in inp:
            if not line.startswith("ATOM  "):
                continue
            resname = line[17:20].strip().upper()
            if resname not in THREE_TO_ONE:
                continue
            if len(line) < 27:
                continue
            raw_resseq = line[22:26].strip()
            if not raw_resseq:
                continue
            try:
                old_key = ResidueKey(
                    chain_id=line[21],
                    resseq=int(raw_resseq),
                    icode=normalize_pdb_icode(line[26]),
                )
            except ValueError:
                continue
            if old_key not in output_keys:
                continue
            if selected_chains is not None and old_key.chain_id not in selected_chains:
                continue
            new_resseq, new_icode = old_to_new[old_key]
            if not (-999 <= new_resseq <= 9999):
                raise RenumberingError(
                    f"New residue number {new_resseq} does not fit PDB columns for {input_pdb}."
                )
            full_line = f"ATOM  {0:5d}" + line[11:22] + f"{new_resseq:4d}{new_icode:1s}" + line[27:]
            if len(full_line) < 80:
                full_line = full_line.rstrip("\n").ljust(80) + "\n"
            else:
                full_line = full_line.rstrip("\n")[:80] + "\n"
            records.append((old_key.chain_id, new_resseq, new_icode, resname, full_line))

    if not records:
        raise RenumberingError(f"{input_pdb}: no ATOM records to write after filtering.")

    records_sorted = sorted(
        records,
        key=lambda item: (
            item[0],
            *imgt_residue_sort_key(item[1], item[2]),
            item[4][12:16],
        ),
    )
    final_lines: List[Tuple[str, int, str, str, str]] = []
    serial = 1
    for chain_id, new_resseq, new_icode, resname, line in records_sorted:
        final_lines.append(
            (
                chain_id,
                new_resseq,
                new_icode,
                resname,
                f"ATOM  {serial:5d}" + line[11:],
            )
        )
        serial += 1
    _write_slim_pdb(output_pdb, final_lines, overwrite=overwrite)


def write_antibody_pdb_from_mmcif(
    input_cif: Path,
    output_pdb: Path,
    old_to_new: Dict[ResidueKey, Tuple[int, str]],
    output_keys: Set[ResidueKey],
    *,
    selected_chains: Optional[set[str]] = None,
    overwrite: bool = False,
) -> None:
    atoms = parse_cif(
        str(input_cif),
        allowed_chains=selected_chains,
        all_chains=(selected_chains is None),
    )
    records: List[Tuple[str, int, str, str, str]] = []
    serial = 1
    pending: List[Tuple[str, int, str, str, str]] = []

    for atom in atoms:
        resname = atom.residue_name.upper()
        if resname not in THREE_TO_ONE:
            continue
        old_key = ResidueKey(
            chain_id=atom.chain,
            resseq=atom.residue_number,
            icode=normalize_pdb_icode(atom.insertion_code),
        )
        if old_key not in output_keys:
            continue
        if selected_chains is not None and old_key.chain_id not in selected_chains:
            continue
        new_resseq, new_icode = old_to_new[old_key]
        if not (-999 <= new_resseq <= 9999):
            raise RenumberingError(
                f"New residue number {new_resseq} does not fit PDB columns for {input_cif}."
            )
        line = format_pdb_atom_line(serial, atom, new_resseq, new_icode)
        pending.append((old_key.chain_id, new_resseq, new_icode, resname, line))
        serial += 1

    if not pending:
        raise RenumberingError(f"{input_cif}: no ATOM records to write after filtering.")

    pending.sort(
        key=lambda item: (
            item[0],
            *imgt_residue_sort_key(item[1], item[2]),
            item[4][12:16],
        ),
    )
    final_lines: List[Tuple[str, int, str, str, str]] = []
    for idx, (chain_id, new_resseq, new_icode, resname, line) in enumerate(pending, start=1):
        final_lines.append((chain_id, new_resseq, new_icode, resname, f"ATOM  {idx:5d}" + line[11:]))

    _write_slim_pdb(output_pdb, final_lines, overwrite=overwrite)


def find_structure_files(
    input_dirs: Iterable[Path],
    *,
    recursive: bool,
    patterns: Optional[List[str]] = None,
) -> List[Path]:
    globs = patterns or ["*.pdb", "*.cif", "*.mmcif"]
    found: List[Path] = []
    for d in input_dirs:
        if not d.exists():
            raise RenumberingError(f"Input path does not exist: {d}")
        if not d.is_dir():
            raise RenumberingError(f"Input path is not a directory: {d}")
        for pattern in globs:
            iterator = d.rglob(pattern) if recursive else d.glob(pattern)
            found.extend(p for p in iterator if p.is_file() and p.suffix.lower() in STRUCTURE_SUFFIXES)
    return sorted(set(found))


def choose_output_path(
    input_structure: Path,
    input_dirs: List[Path],
    out_dir: Optional[Path],
    suffix: str,
) -> Path:
    if input_structure.suffix.lower() in (".cif", ".mmcif"):
        out_name = f"{input_structure.stem}.pdb"
    else:
        out_name = input_structure.name
    for root in input_dirs:
        try:
            rel = input_structure.relative_to(root)
        except ValueError:
            continue
        rel = rel.with_name(out_name)

        if out_dir is None:
            output_root = root.with_name(f"{root.name}{suffix}")
            return output_root / rel

        return out_dir / rel

    if out_dir is None:
        output_root = input_structure.parent.with_name(f"{input_structure.parent.name}{suffix}")
        return output_root / out_name

    return out_dir / out_name


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


def renumber_structure_file(
    input_path: Path,
    output_pdb: Path,
    *,
    selected_chains: Optional[set[str]] = None,
    allow_partial: bool = False,
    overwrite: bool = False,
) -> Tuple[Dict[ResidueKey, Tuple[int, str]], List[dict]]:
    old_to_new, report_rows, output_keys = build_residue_mapping(
        input_path,
        selected_chains=selected_chains,
        allow_partial=allow_partial,
    )
    if input_path.suffix.lower() in (".cif", ".mmcif"):
        write_antibody_pdb_from_mmcif(
            input_path,
            output_pdb,
            old_to_new,
            output_keys,
            selected_chains=selected_chains,
            overwrite=overwrite,
        )
    else:
        write_antibody_pdb(
            input_path,
            output_pdb,
            old_to_new,
            output_keys,
            selected_chains=selected_chains,
            overwrite=overwrite,
        )
    return old_to_new, report_rows


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


def main_renumber(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Renumber antibody PDB/mmCIF structures to IMGT numbering with ANARCI."
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing PDB or mmCIF files.",
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
        default=None,
        help=(
            "File glob inside each input directory (repeatable). "
            "Default: *.pdb, *.cif, *.mmcif"
        ),
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
            "Allow ANARCI to number only the variable domain; terminal residues outside "
            "the domain are omitted from output (for experimental Fv/PDB structures)."
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
    glob_patterns = [args.glob] if args.glob else None
    structures = find_structure_files(
        input_dirs,
        recursive=args.recursive,
        patterns=glob_patterns,
    )

    if not structures:
        print("No PDB/mmCIF files found.", file=sys.stderr)
        return 2

    all_report_rows: List[dict] = []
    n_ok = 0
    n_fail = 0

    for structure_path in structures:
        out_path = choose_output_path(
            input_structure=structure_path,
            input_dirs=input_dirs,
            out_dir=args.out_dir.resolve() if args.out_dir is not None else None,
            suffix=args.suffix,
        )

        try:
            _old_to_new, chain_rows = renumber_structure_file(
                structure_path,
                out_path,
                selected_chains=args.chains,
                allow_partial=args.allow_partial_domain,
                overwrite=args.overwrite,
            )
            n_ok += 1
            print(f"OK\t{structure_path}\t->\t{out_path}")

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
            print(f"FAIL\t{structure_path}\t{msg}", file=sys.stderr)
            all_report_rows.append(
                {
                    "status": "fail",
                    "pdb": str(structure_path),
                    "output_pdb": str(out_path),
                    "message": msg,
                }
            )

    if args.report is not None:
        write_report(args.report, all_report_rows)
        print(f"Report written: {args.report}")

    print(f"Done. Successful: {n_ok}; failed: {n_fail}; total: {len(structures)}")
    return 0 if n_fail == 0 else 1

_MINIMIZE_DEPS_LOADED = False


def _ensure_minimize_deps() -> None:
    global _MINIMIZE_DEPS_LOADED
    global np, pdbfixer, spatial
    global ENERGY, LENGTH, spring_unit, CLASH_CUTOFF, atom_radii, radii_sums, cutoffs, forcefield
    global LangevinIntegrator, CustomExternalForce, CustomTorsionForce, OpenMMException, Platform, app, unit
    if _MINIMIZE_DEPS_LOADED:
        return
    import numpy as np  # noqa: F401
    import pdbfixer  # noqa: F401
    from openmm import (  # noqa: F401
        CustomExternalForce,
        CustomTorsionForce,
        LangevinIntegrator,
        OpenMMException,
        Platform,
        app,
        unit,
    )
    from scipy import spatial  # noqa: F401

    ENERGY = unit.kilocalories_per_mole
    LENGTH = unit.angstroms
    spring_unit = ENERGY / (LENGTH**2)
    CLASH_CUTOFF = 0.63
    atom_radii = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
    radii_sums = {
        i + j: atom_radii[i] + atom_radii[j]
        for i in atom_radii
        for j in atom_radii
    }
    cutoffs = {k: CLASH_CUTOFF * v for k, v in radii_sums.items()}
    forcefield = app.ForceField("amber14/protein.ff14SB.xml")
    _MINIMIZE_DEPS_LOADED = True


def minimize_energy(topology, positions, k1=2.5, k2=2.5, n_threads=-1):
    _ensure_minimize_deps()
    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology)

    force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", k1 * spring_unit)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)
    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            if atom.name in ["CA", "CB", "N", "C"]:
                force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    if k2 > 0.0:
        cis_force = CustomTorsionForce("10*k2*(1+cos(theta))^2")
        cis_force.addGlobalParameter("k2", k2 * ENERGY)
        for chain in modeller.topology.chains():
            residues = list(chain.residues())
            rel = [
                {a.name: a.index for a in res.atoms() if a.name in ["N", "CA", "C"]}
                for res in residues
            ]
            for i in range(1, len(residues)):
                if residues[i].name == "PRO":
                    continue
                cis_force.addTorsion(
                    rel[i - 1]["CA"], rel[i - 1]["C"], rel[i]["N"], rel[i]["CA"]
                )
        system.addForce(cis_force)

    integrator = LangevinIntegrator(0, 0.01, 0.0)
    if n_threads > 0:
        platform = Platform.getPlatformByName("CPU")
        simulation = app.Simulation(
            modeller.topology, system, integrator,
            platform, {"Threads": str(n_threads)},
        )
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    return simulation


def chirality_fixer(simulation):
    topology = simulation.topology
    positions = simulation.context.getState(getPositions=True).getPositions()

    d_stereoisomers = []
    for residue in topology.residues():
        if residue.name == "GLY":
            continue
        atom_indices = {
            a.name: a.index
            for a in residue.atoms()
            if a.name in ["N", "CA", "C", "CB"]
        }
        vectors = [
            positions[atom_indices[i]] - positions[atom_indices["CA"]]
            for i in ["N", "C", "CB"]
        ]
        if np.dot(np.cross(vectors[0], vectors[1]), vectors[2]) < 0.0 * LENGTH**3:
            indices = {x.name: x.index for x in residue.atoms() if x.name in ["HA", "CA"]}
            positions[indices["HA"]] = (
                2 * positions[indices["CA"]] - positions[indices["HA"]]
            )
            particle_mass = simulation.system.getParticleMass(indices["HA"])
            simulation.system.setParticleMass(indices["HA"], 0.0)
            d_stereoisomers.append((indices["HA"], particle_mass))

    if d_stereoisomers:
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        for atom in d_stereoisomers:
            simulation.system.setParticleMass(*atom)
        simulation.minimizeEnergy()
    return simulation


def bond_check(topology, positions):
    for chain in topology.chains():
        residues = [
            {a.name: a.index for a in res.atoms() if a.name in ["N", "C"]}
            for res in chain.residues()
        ]
        for i in range(len(residues) - 1):
            v = np.linalg.norm(
                positions[residues[i]["C"]] - positions[residues[i + 1]["N"]]
            )
            if abs(v - 1.329 * LENGTH) > 0.1 * LENGTH:
                return False
    return True


def cis_check(topology, positions):
    pos = np.array(positions.value_in_unit(LENGTH))
    for chain in topology.chains():
        residues = list(chain.residues())
        rel = [
            {a.name: a.index for a in res.atoms() if a.name in ["N", "CA", "C"]}
            for res in residues
        ]
        for i in range(1, len(residues)):
            if residues[i].name == "PRO":
                continue
            r, nr = rel[i - 1], rel[i]
            p0, p1, p2, p3 = (
                pos[r["CA"]],
                pos[r["C"]],
                pos[nr["N"]],
                pos[nr["CA"]],
            )
            ab = p1 - p0
            cd = p2 - p1
            db = p3 - p2
            u = np.cross(-ab, cd)
            v = np.cross(db, cd)
            if np.dot(u, v) > 0:
                return False
    return True


def stereo_check(topology, positions):
    pos = np.array(positions.value_in_unit(LENGTH))
    for residue in topology.residues():
        if residue.name == "GLY":
            continue
        idx = {
            a.name: a.index
            for a in residue.atoms()
            if a.name in ["N", "CA", "C", "CB"]
        }
        vecs = pos[[idx[i] for i in ["N", "C", "CB"]]] - pos[idx["CA"]]
        if np.linalg.det(vecs) < 0:
            return False
    return True


def clash_check(topology, positions):
    heavies = [x for x in topology.atoms() if x.element.symbol != "H"]
    pos = np.array(positions.value_in_unit(LENGTH))[[x.index for x in heavies]]
    tree = spatial.KDTree(pos)
    pairs = tree.query_pairs(r=max(cutoffs.values()))
    for pair in pairs:
        ai, aj = heavies[pair[0]], heavies[pair[1]]
        if ai.residue.index == aj.residue.index:
            continue
        if (ai.name == "C" and aj.name == "N") or (ai.name == "N" and aj.name == "C"):
            continue
        d = np.linalg.norm(pos[pair[0]] - pos[pair[1]])
        if ai.name == "SG" and aj.name == "SG" and d > 1.88:
            continue
        if d < cutoffs[ai.element.symbol + aj.element.symbol]:
            return False
    return True


def strained_sidechain_bonds_check(topology, positions):
    atoms = list(topology.atoms())
    pos = np.array(positions.value_in_unit(LENGTH))
    system = forcefield.createSystem(topology)
    bonds = [x for x in system.getForces() if type(x).__name__ == "HarmonicBondForce"][0]
    n_bonds = bonds.getNumBonds()
    ii = np.empty(n_bonds, dtype=int)
    jj = np.empty(n_bonds, dtype=int)
    k = np.empty(n_bonds)
    x0 = np.empty(n_bonds)
    for n in range(n_bonds):
        ii[n], jj[n], _x0, _k = bonds.getBondParameters(n)
        k[n] = _k.value_in_unit(spring_unit)
        x0[n] = _x0.value_in_unit(LENGTH)
    distance = np.linalg.norm(pos[ii] - pos[jj], axis=-1)
    check = k * (distance - x0) ** 2 > 100
    return [atoms[x].residue for x in ii[check]]


def strained_sidechain_bonds_fixer(strained_residues, topology, positions, n_threads=-1):
    bb_atoms = ["N", "CA", "C"]
    bad_side_chains = [
        atom
        for residue in strained_residues
        for atom in residue.atoms()
        if atom.name not in bb_atoms
    ]
    modeller = app.Modeller(topology, positions)
    modeller.delete(bad_side_chains)

    fd, tmp_file = tempfile.mkstemp(suffix=".pdb", prefix="side_chain_fix_")
    os.close(fd)
    try:
        with open(tmp_file, "w") as handle:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

        fixer = pdbfixer.PDBFixer(tmp_file)
    finally:
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(modeller.topology)
    integrator = LangevinIntegrator(0, 0.01, 0.0)
    if n_threads > 0:
        platform = Platform.getPlatformByName("CPU")
        simulation = app.Simulation(
            modeller.topology, system, integrator,
            platform, {"Threads": str(n_threads)},
        )
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    return simulation.topology, simulation.context.getState(getPositions=True).getPositions()


def refine_once(input_file, output_file, check_for_strained_bonds=True, n=6, n_threads=-1):
    _ensure_minimize_deps()
    k1s = [2.5, 1, 0.5, 0.25, 0.1, 0.001]
    k2s = [2.5, 5, 7.5, 15, 25, 50]
    success = False

    fixer = pdbfixer.PDBFixer(input_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    k1 = k1s[0]
    k2 = -1 if cis_check(fixer.topology, fixer.positions) else k2s[0]
    topology, positions = fixer.topology, fixer.positions

    for i in range(n):
        try:
            simulation = minimize_energy(topology, positions, k1=k1, k2=k2, n_threads=n_threads)
            topology, positions = (
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
            )
            acceptable_bonds = bond_check(topology, positions)
            trans_peptide_bonds = cis_check(topology, positions)
        except OpenMMException as e:
            if i == n - 1 and "positions" not in locals():
                print(f"OpenMM failed to refine {input_file}", flush=True)
                raise e
            topology, positions = fixer.topology, fixer.positions
            continue

        if not acceptable_bonds:
            k1 = k1s[min(i, len(k1s) - 1)]
        if not trans_peptide_bonds:
            k2 = k2s[min(i, len(k2s) - 1)]
        else:
            k2 = -1

        if acceptable_bonds and trans_peptide_bonds:
            try:
                simulation = chirality_fixer(simulation)
                topology, positions = (
                    simulation.topology,
                    simulation.context.getState(getPositions=True).getPositions(),
                )
            except OpenMMException:
                topology, positions = fixer.topology, fixer.positions
                continue

            if check_for_strained_bonds:
                try:
                    strained_bonds = strained_sidechain_bonds_check(topology, positions)
                    if len(strained_bonds) > 0:
                        needs_recheck = True
                        topology, positions = strained_sidechain_bonds_fixer(
                            strained_bonds, topology, positions, n_threads=n_threads
                        )
                    else:
                        needs_recheck = False
                except OpenMMException:
                    topology, positions = fixer.topology, fixer.positions
                    continue
            else:
                needs_recheck = False

            tests = bond_check(topology, positions) and cis_check(topology, positions)
            if needs_recheck:
                tests = tests and not strained_sidechain_bonds_check(topology, positions)
            if tests and stereo_check(topology, positions) and clash_check(topology, positions):
                success = True
                break

    # If all minimization attempts failed the structure was reset to the PDBFixer heavy-atom
    # topology (no hydrogens).  Try to add hydrogens as a last resort so that downstream tools
    # (e.g. ProperMAB) don't fail with "No template found … missing N hydrogen atoms".
    if not success:
        h_count = sum(1 for a in topology.atoms() if a.element is not None and a.element.symbol == "H")
        if h_count == 0:
            try:
                modeller = app.Modeller(topology, positions)
                modeller.addHydrogens(forcefield)
                topology, positions = modeller.topology, modeller.positions
                print(
                    f"[refine] minimization failed for {input_file}; "
                    "wrote heavy-atom structure with fallback H addition.",
                    flush=True,
                )
            except Exception as h_err:
                print(
                    f"[refine] minimization failed for {input_file} and fallback H addition "
                    f"also failed ({h_err}); writing heavy-atom structure.",
                    flush=True,
                )

    with open(output_file, "w") as out_handle:
        app.PDBFile.writeFile(topology, positions, out_handle, keepIds=True)
    return success


def refine(input_file, output_file, check_for_strained_bonds=True, tries=3, n=6, n_threads=-1):
    for _ in range(tries):
        if refine_once(
            input_file, output_file,
            check_for_strained_bonds=check_for_strained_bonds,
            n=n, n_threads=n_threads,
        ):
            return True
    return False


def _minimize_one(task: tuple) -> tuple[str, str, bool]:
    input_pdb, output_pdb, n_threads, skip_existing = task
    if skip_existing and os.path.exists(output_pdb):
        return input_pdb, "skipped", True
    try:
        success = refine(input_pdb, output_pdb, n_threads=n_threads)
        status = "ok" if success else "warn"
        return input_pdb, status, success
    except Exception as e:
        return input_pdb, f"FAIL: {e}", False


_MINIMIZE_STAGING_DIRNAME = ".minimize_input_pdb"


def prepare_minimize_input_dir(input_dir: Path) -> Tuple[Path, List[Path]]:
    cif_files = sorted(input_dir.glob("*.cif")) + sorted(input_dir.glob("*.mmcif"))
    pdb_files = sorted(
        p
        for p in input_dir.glob("*.pdb")
        if not (p.stem.endswith("_H") or p.stem.endswith("_L"))
    )

    if not cif_files and not pdb_files:
        return input_dir, []

    staging = input_dir / _MINIMIZE_STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    structure_files = [(p, f"{p.stem}.pdb") for p in cif_files]
    structure_files.extend((p, p.name) for p in pdb_files)

    seen_outputs = set()
    for structure_path, out_name in structure_files:
        if out_name in seen_outputs:
            raise ValueError(
                f"Cannot stage multiple structures with the same output name: {out_name}"
            )
        seen_outputs.add(out_name)

        out_pdb = staging / out_name
        atoms = parse_structure(str(structure_path))
        if not atoms:
            raise ValueError(f"No antibody atoms parsed from {structure_path}")
        write_structure_pdb(
            atoms,
            str(out_pdb),
            remark=f"Antibody-only input from {structure_path.name} for minimization",
        )
        print(f"  stage  {structure_path.name} -> {out_pdb.name}", flush=True)

    staged_pdbs = sorted(
        p
        for p in staging.glob("*.pdb")
        if not (p.stem.endswith("_H") or p.stem.endswith("_L"))
    )
    return staging, staged_pdbs


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: postprocess_structures.py {renumber|minimize} ...")
    command = argv[0]
    rest = argv[1:]
    if command == "renumber":
        return main_renumber(rest)
    if command == "minimize":
        _run_minimize_cli(rest)
        return 0
    raise SystemExit(f"unknown subcommand {command!r}; use renumber or minimize")


def _run_minimize_cli(rest: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Minimize PDB/mmCIF structures with OpenMM.")
    parser.add_argument("--input-dir", required=True, metavar="DIR")
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    parser.add_argument("--n-threads", type=int, default=1, metavar="N")
    parser.add_argument("--jobs", type=int, default=8, metavar="J")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(rest)
    _ensure_minimize_deps()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir, pdb_files = prepare_minimize_input_dir(input_dir)
    if not pdb_files:
        print(f"No PDB/mmCIF structure files found in {input_dir}", flush=True)
        return

    print(f"Minimizing {len(pdb_files)} structure(s): {work_dir} -> {output_dir}", flush=True)
    tasks = [
        (str(p), str(output_dir / p.name), args.n_threads, args.skip_existing)
        for p in pdb_files
    ]
    n_ok = n_warn = n_skip = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_minimize_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            name, status, _ = future.result()
            short = Path(name).name
            if status == "skipped":
                n_skip += 1
            elif status == "ok":
                n_ok += 1
                print(f"  ok   {short}", flush=True)
            elif status == "warn":
                n_warn += 1
                print(f"  warn {short}  (did not fully converge)", flush=True)
            else:
                n_fail += 1
                print(f"  {status}  {short}", flush=True)

    print(f"\nDone: {n_ok} ok, {n_warn} warn, {n_skip} skipped, {n_fail} failed.", flush=True)
    print(f"Output: {output_dir}", flush=True)

    staging = input_dir / _MINIMIZE_STAGING_DIRNAME
    if staging.is_dir():
        shutil.rmtree(staging)
        print(f"Removed minimization staging: {staging}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
