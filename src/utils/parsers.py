from typing import Dict, List, Tuple, Optional, Iterable, Iterator
from dataclasses import dataclass
from collections import defaultdict
import re
import shlex
from pathlib import Path
import logging

from utils.chemistry import AA_1_TO_3

_RESNUM_PATTERN = re.compile(r"^(-?\d+)([A-Za-z])?$")
hbond_pattern = re.compile(r"(-?\d+),\s*(-?\d+\.?\d*)")

CHARGED_RESIDUE_TYPES = frozenset({"ASP", "GLU", "LYS", "ARG", "HIS", "TYR", "CYS", "N+", "C-"})

ResKey4 = Tuple[str, int, str, str]

_PKA_INSERTION_REMAP_WARNED: set[Tuple[ResKey4, ResKey4]] = set()


def _build_residue_key_index(
    pdb_residue_set: set[ResKey4],
) -> Dict[Tuple[str, int, str], List[ResKey4]]:
    index: Dict[Tuple[str, int, str], List[ResKey4]] = defaultdict(list)
    for key in pdb_residue_set:
        index[(key[0], key[1], key[2])].append(key)
    for keys in index.values():
        keys.sort(key=lambda k: k[3])
    return index


def resolve_external_residue_key(
    external_key: ResKey4,
    pdb_residue_set: set[ResKey4],
    *,
    residue_index: Optional[Dict[Tuple[str, int, str], List[ResKey4]]] = None,
) -> Optional[ResKey4]:
    """Map PropKa/DSSP residue keys onto parsed structure keys.

    PropKa often omits IMGT insertion codes (e.g. ``TYR 112 H``) while the
    structure uses ``112B``. When exactly one structure residue matches
    ``(residue_name, residue_number, chain)``, use that key.
    """
    if external_key in pdb_residue_set:
        return external_key

    res_name, res_num, chain, inscode = external_key
    if inscode:
        return None

    if residue_index is None:
        residue_index = _build_residue_key_index(pdb_residue_set)

    candidates = residue_index.get((res_name, res_num, chain), [])
    if len(candidates) == 1:
        mapped = candidates[0]
        if mapped != external_key and (external_key, mapped) not in _PKA_INSERTION_REMAP_WARNED:
            _PKA_INSERTION_REMAP_WARNED.add((external_key, mapped))
            logger.info(
                "Mapped external residue key %r -> structure key %r (insertion-code mismatch)",
                external_key,
                mapped,
            )
        return mapped

    if not candidates:
        return None

    unlettered = (res_name, res_num, chain, "")
    if unlettered in pdb_residue_set:
        return unlettered
    return None

_PKA_CACHE: Dict[Tuple[str, int], Dict[Tuple[str, int, str, str], float]] = {}

STANDARD_AA = frozenset(AA_1_TO_3.values())
ALLOWED_CHAINS = frozenset({"H", "L"})

_SASA_TOTAL_CACHE: Dict[str, Optional[float]] = {}

_DSSP_FILE_CACHE: Dict[str, Tuple[List[str], Optional[int]]] = {}
_DSSP_FULL_CACHE: Dict[
    Tuple[str, int],
    Tuple[
        Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]],
        Dict[Tuple[str, int, str, str], List[Tuple[int, float]]],
        Dict[int, Tuple[str, int, str, str]],
    ],
] = {}

logger = logging.getLogger(__name__)

def _write_per_file_log(source_path: str, message: str) -> None:
    try:
        p = Path(source_path)
        log_dir = p.parent / "_logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{p.name}.log"
        with open(log_path, "a") as f:
            f.write(message + "\n")
    except Exception:
        return

def parse_residue_number_field(s: str) -> Optional[Tuple[int, str]]:
    s = (s or "").strip()
    if not s:
        return None
    match = _RESNUM_PATTERN.match(s)
    if not match:
        return None
    num = int(match.group(1))
    ins = (match.group(2) or "")
    return (num, ins)


def parse_dssp_residue_fields(line: str) -> Optional[Tuple[int, str, str]]:
    """Parse residue number, insertion code, and chain from a DSSP data line.

    Columns 6-10 (``line[5:10]``) hold the PDB residue number; the insertion
    code may be the last character of that field. mkdssp sometimes right-pads
    three-digit numbers with two spaces, pushing the insertion letter into
    column 11 (``line[10]``). Chain ID stays at column 12 (``line[11]``).
    """
    if len(line) < 12:
        return None
    parsed_res = parse_residue_number_field(line[5:10])
    if parsed_res is None:
        return None
    residue_number, insertion_code = parsed_res
    if not insertion_code:
        spill = line[10]
        if spill.isalpha():
            insertion_code = spill
    chain = line[11:12].strip()
    if not chain:
        return None
    return residue_number, insertion_code, chain


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    residue_name: str
    chain: str
    residue_number: int
    insertion_code: str
    x: float
    y: float
    z: float
    element: str

@dataclass
class SASAEntry:
    residue_name: str
    chain: str
    residue_number: int
    total_side_abs: float
    total_side_rel: float
    main_chain_abs: float
    main_chain_rel: float
    non_polar_abs: float = 0.0
    non_polar_rel: float = 0.0
    all_polar_abs: float = 0.0
    all_polar_rel: float = 0.0


@dataclass(frozen=True)
class SASAParseResult:
    entries: Dict[Tuple[str, int, str, str], SASAEntry]
    total_sasa: float


@dataclass(frozen=True)
class SASARawRecord:
    residue_name: str
    chain: str
    residue_number: int
    insertion_code: str
    total_side_abs: str
    total_side_rel: str
    main_chain_abs: str
    main_chain_rel: str
    non_polar_abs: str = ""
    non_polar_rel: str = ""
    all_polar_abs: str = ""
    all_polar_rel: str = ""

_SASA_RAW_CACHE: Dict[str, List[SASARawRecord]] = {}
_SASA_CACHE: Dict[str, SASAParseResult] = {}

def parse_pdb(
    pdb_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> List[Atom]:

    chains = frozenset(allowed_chains) if allowed_chains is not None else ALLOWED_CHAINS
    atoms: List[Atom] = []
    n_total = 0
    n_skipped = 0
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                n_total += 1
                try:
                    serial = int(line[6:11].strip())
                    name = line[12:16].strip()
                    alt_loc = line[16:17].strip()
                    residue_name = line[17:20].strip()
                    chain = line[21:22].strip()
                    if chain not in chains or residue_name not in STANDARD_AA:
                        continue
                    residue_number = int(line[22:26].strip())
                    insertion_code = (line[26:27].strip() or "")
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = line[76:78].strip() if len(line) > 76 else name[0]
                    if alt_loc and alt_loc != "A":
                        continue
                    atom = Atom(
                        serial=serial,
                        name=name,
                        residue_name=residue_name,
                        chain=chain,
                        residue_number=residue_number,
                        insertion_code=insertion_code,
                        x=x,
                        y=y,
                        z=z,
                        element=element,
                    )
                    atoms.append(atom)
                except (ValueError, IndexError):
                    n_skipped += 1
                    continue
    except FileNotFoundError:
        logger.info("PDB file %r not found; returning empty atom list", pdb_path)
    except Exception as e:
        logger.warning("Failed to read PDB file %r: %s; returning partial/empty atoms", pdb_path, e)

    if n_total > 0 and not atoms:
        logger.warning("Parsed 0 ATOM records from %r (chains=%r)", pdb_path, chains)
    elif n_total > 0 and n_skipped / n_total > 0.1:
        logger.warning(
            "Skipped %d of %d ATOM records while parsing %r",
            n_skipped,
            n_total,
            pdb_path,
        )

    return atoms

def _clean_cif_optional(value: str) -> str:
    value = value.strip()
    return "" if value in {"", ".", "?"} else value


def _iter_cif_loop_rows(
    lines: List[str],
    start_idx: int,
    n_cols: int,
) -> Iterator[Tuple[List[str], int]]:
    i = start_idx
    pending: List[str] = []

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("#") or stripped.startswith("_") or stripped == "loop_":
            if pending:
                yield pending, i
            yield [], i
            return

        try:
            lexer = shlex.shlex(raw.rstrip("\n"), posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError:
            tokens = []

        i += 1
        if not tokens:
            continue

        pending.extend(tokens)
        while len(pending) >= n_cols:
            row = pending[:n_cols]
            pending = pending[n_cols:]
            yield row, i

    if pending:
        yield pending, i


def parse_cif(
    cif_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
    *,
    all_chains: bool = False,
) -> List[Atom]:
    if all_chains:
        chain_filter: Optional[frozenset] = None
    elif allowed_chains is not None:
        chain_filter = frozenset(allowed_chains)
    else:
        chain_filter = ALLOWED_CHAINS
    atoms: List[Atom] = []

    try:
        with open(cif_path, "r") as f:
            lines = list(f)
    except FileNotFoundError:
        logger.info("CIF file %r not found; returning empty atom list", cif_path)
        return atoms
    except Exception as e:
        logger.warning("Failed to read CIF file %r: %s; returning empty atom list", cif_path, e)
        return atoms

    # prefer auth_* columns (DSSP/PropKa/SASA use author residue numbering)
    i = 0
    col_indices: Optional[Dict[str, int]] = None
    required = {
        "group_PDB", "id", "type_symbol", "label_atom_id",
        "label_comp_id", "label_asym_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z",
    }
    n_total = 0
    n_skipped = 0
    first_model: Optional[str] = None

    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip() != "loop_":
            continue
        col_names: List[str] = []
        while i < len(lines):
            L = lines[i]
            i += 1
            L = L.strip()
            if not L or L.startswith("#"):
                continue
            if not L.startswith("_atom_site."):
                i -= 1
                break
            name = L[len("_atom_site.") :].strip()
            col_names.append(name)
        if not col_names:
            continue
        col_indices = {name: idx for idx, name in enumerate(col_names)}
        if not required.issubset(col_indices):
            continue
        chain_col = "auth_asym_id" if "auth_asym_id" in col_indices else "label_asym_id"
        resname_col = "auth_comp_id" if "auth_comp_id" in col_indices else "label_comp_id"
        resnum_col = "auth_seq_id" if "auth_seq_id" in col_indices else "label_seq_id"
        ins_code_col = "pdbx_PDB_ins_code" if "pdbx_PDB_ins_code" in col_indices else None
        atom_name_col = "auth_atom_id" if "auth_atom_id" in col_indices else "label_atom_id"
        alt_loc_col = "label_alt_id" if "label_alt_id" in col_indices else None
        model_col = "pdbx_PDB_model_num" if "pdbx_PDB_model_num" in col_indices else None
        max_col_idx = max(col_indices.values())
        for parts, next_i in _iter_cif_loop_rows(lines, i, len(col_names)):
            if not parts:
                i = next_i
                break
            if len(parts) <= max_col_idx:
                n_skipped += 1
                continue
            try:
                n_total += 1
                group = parts[col_indices["group_PDB"]].strip()
                if group != "ATOM":
                    continue
                if model_col is not None:
                    model = _clean_cif_optional(parts[col_indices[model_col]])
                    if model:
                        if first_model is None:
                            first_model = model
                        elif model != first_model:
                            continue
                serial_text = _clean_cif_optional(parts[col_indices["id"]])
                serial = int(serial_text or "0")
                name = _clean_cif_optional(parts[col_indices[atom_name_col]])
                element = _clean_cif_optional(parts[col_indices["type_symbol"]]) or (name[:1] or "?")
                alt_loc = ""
                if alt_loc_col is not None:
                    alt_loc = _clean_cif_optional(parts[col_indices[alt_loc_col]])
                residue_name = _clean_cif_optional(parts[col_indices[resname_col]]).upper()
                chain = _clean_cif_optional(parts[col_indices[chain_col]])
                if chain_filter is not None and chain not in chain_filter:
                    continue
                if residue_name not in STANDARD_AA:
                    continue
                resnum_str = _clean_cif_optional(parts[col_indices[resnum_col]])
                parsed_resnum = parse_residue_number_field(resnum_str)
                if parsed_resnum is None:
                    continue
                residue_number, insertion_code = parsed_resnum
                if ins_code_col and ins_code_col in col_indices:
                    ins = _clean_cif_optional(parts[col_indices[ins_code_col]])
                    if ins:
                        insertion_code = ins
                x = float(_clean_cif_optional(parts[col_indices["Cartn_x"]]) or "0")
                y = float(_clean_cif_optional(parts[col_indices["Cartn_y"]]) or "0")
                z = float(_clean_cif_optional(parts[col_indices["Cartn_z"]]) or "0")
                if alt_loc and alt_loc != "A":
                    continue
                atoms.append(
                    Atom(
                        serial=serial,
                        name=name,
                        residue_name=residue_name,
                        chain=chain,
                        residue_number=residue_number,
                        insertion_code=insertion_code,
                        x=x,
                        y=y,
                        z=z,
                        element=element,
                    )
                )
            except (ValueError, IndexError, KeyError):
                n_skipped += 1
                continue
        else:
            i = len(lines)
        break

    if n_total > 0 and not atoms:
        logger.warning("Parsed 0 ATOM records from CIF %r (chain_filter=%r)", cif_path, chain_filter)
    elif n_total > 0 and n_skipped / n_total > 0.1:
        logger.warning(
            "Skipped %d of %d atom_site records while parsing CIF %r",
            n_skipped,
            n_total,
            cif_path,
        )

    return atoms


def collect_pdb_bfactor_values(pdb_path: str) -> set[float]:
    """Collect unique B-factors (PDB columns 61-66) from ATOM/HETATM records."""
    bfactors: set[float] = set()
    try:
        with open(pdb_path, "r") as handle:
            for line in handle:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                if len(line) < 66:
                    continue
                try:
                    bfactors.add(float(line[60:66].strip()))
                except ValueError:
                    continue
    except OSError:
        return set()
    return bfactors


def abb2_bfactor_all_zero(pdb_path: str) -> bool:
    """True when every ATOM/HETATM B-factor is 0.00 (unrefined abb2 structure)."""
    bfactors = collect_pdb_bfactor_values(pdb_path)
    if not bfactors:
        return False
    return all(abs(value) < 1e-6 for value in bfactors)


def parse_structure(
    path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> List[Atom]:

    if path.lower().endswith((".cif", ".mmcif")):
        return parse_cif(path, allowed_chains=allowed_chains)
    return parse_pdb(path, allowed_chains=allowed_chains)


def _format_atom_pdb_line(serial: int, atom: Atom) -> str:
    icode = (atom.insertion_code or "").strip()
    icode_c = icode[0] if icode else " "
    _stripped = atom.name.strip()
    name_field = _stripped[:4] if len(_stripped) >= 4 else f" {_stripped:<3}"[:4]
    resname = atom.residue_name[:3].upper()
    element = (atom.element or "").strip()[:2] or name_field.strip()[:1]
    element = f"{element:>2s}"[:2]
    line = (
        f"ATOM  {serial:5d} {name_field:4s} {resname:>3} {atom.chain:1s}"
        f"{atom.residue_number:4d}{icode_c:1s}   {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"  1.00  0.00          {element:>2s}\n"
    )
    return line if len(line) >= 80 else line.rstrip("\n").ljust(80) + "\n"


def write_structure_pdb(
    atoms: List[Atom],
    output_path: str,
    *,
    remark: Optional[str] = None,
) -> None:
    if not atoms:
        raise ValueError(f"No atoms to write to {output_path!r}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as handle:
        if remark:
            handle.write(f"REMARK   1 {remark}\n")
        serial = 1
        prev_chain: Optional[str] = None
        prev_residue: Optional[Tuple[int, str, str]] = None

        for atom in atoms:
            residue = (atom.residue_number, atom.insertion_code, atom.residue_name)
            if prev_chain is not None and atom.chain != prev_chain:
                assert prev_residue is not None
                resname, resnum, icode = prev_residue[2], prev_residue[0], prev_residue[1]
                icode_c = (icode.strip() or " ")[:1]
                handle.write(
                    f"TER   {serial:5d}      {resname:>3} {prev_chain:1s}"
                    f"{resnum:4d}{icode_c:1s}\n"
                )
                serial += 1
            handle.write(_format_atom_pdb_line(serial, atom))
            serial += 1
            prev_chain = atom.chain
            prev_residue = residue

        if prev_chain is not None and prev_residue is not None:
            resname, resnum, icode = prev_residue[2], prev_residue[0], prev_residue[1]
            icode_c = (icode.strip() or " ")[:1]
            handle.write(
                f"TER   {serial:5d}      {resname:>3} {prev_chain:1s}"
                f"{resnum:4d}{icode_c:1s}\n"
            )
        handle.write("END\n")


def residue_key_from_atom(atom: Atom) -> Tuple[str, int, str, str]:
    return (atom.residue_name, atom.residue_number, atom.chain, getattr(atom, "insertion_code", ""))


def ca_xyz_by_residue(
    atoms: List[Atom],
) -> Dict[Tuple[str, int, str, str], Tuple[float, float, float]]:
    ca: Dict[Tuple[str, int, str, str], Tuple[float, float, float]] = {}
    for atom in atoms:
        name = (atom.name or "").strip().upper()
        if name != "CA":
            continue
        key = residue_key_from_atom(atom)
        if key not in ca:
            ca[key] = (float(atom.x), float(atom.y), float(atom.z))
    return ca


def load_sasa_raw(sasa_path: str) -> List[SASARawRecord]:
    abs_path = str(Path(sasa_path).resolve())
    cached = _SASA_RAW_CACHE.get(abs_path)
    if cached is not None:
        return cached

    records: List[SASARawRecord] = []
    n_total = 0
    n_skipped = 0
    total_sasa: Optional[float] = None

    try:
        with open(sasa_path, "r") as f:
            for line in f:
                if line.startswith("TOTAL"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            total_sasa = float(parts[1])
                        except ValueError:
                            pass
                    continue
                if not line.startswith("RES "):
                    continue
                n_total += 1
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    residue_name = parts[1]
                    chain = parts[2]
                    parsed = parse_residue_number_field(parts[3])
                    if parsed is None:
                        continue
                    residue_number, insertion_code = parsed
                    total_side_abs = parts[6] if len(parts) > 6 else ""
                    total_side_rel = parts[7] if len(parts) > 7 else ""
                    main_chain_abs = parts[8] if len(parts) > 8 else ""
                    main_chain_rel = parts[9] if len(parts) > 9 else ""
                    non_polar_abs = parts[10] if len(parts) > 10 else ""
                    non_polar_rel = parts[11] if len(parts) > 11 else ""
                    all_polar_abs = parts[12] if len(parts) > 12 else ""
                    all_polar_rel = parts[13] if len(parts) > 13 else ""
                    records.append(
                        SASARawRecord(
                            residue_name=residue_name,
                            chain=chain,
                            residue_number=residue_number,
                            insertion_code=insertion_code,
                            total_side_abs=total_side_abs,
                            total_side_rel=total_side_rel,
                            main_chain_abs=main_chain_abs,
                            main_chain_rel=main_chain_rel,
                            non_polar_abs=non_polar_abs,
                            non_polar_rel=non_polar_rel,
                            all_polar_abs=all_polar_abs,
                            all_polar_rel=all_polar_rel,
                        )
                    )
                except (ValueError, IndexError):
                    n_skipped += 1
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"SASA file not found: {sasa_path!r}")
    except Exception as e:
        msg = f"Failed to read SASA file {sasa_path!r}: {e}; returning partial/empty records"
        logger.warning(msg)
        _write_per_file_log(sasa_path, msg)
        if total_sasa is None:
            raise ValueError(
                f"SASA file {sasa_path!r} must contain a TOTAL line with a valid numeric total SASA (Å²)."
            ) from e
        _SASA_RAW_CACHE[abs_path] = records
        _SASA_TOTAL_CACHE[abs_path] = total_sasa
        return records

    if total_sasa is None:
        raise ValueError(
            f"SASA file {sasa_path!r} must contain a TOTAL line with a valid numeric total SASA (Å²)."
        )

    if n_total > 0 and not records:
        msg = f"Parsed 0 RES records from SASA file {sasa_path!r}"
        logger.warning(msg)
        _write_per_file_log(sasa_path, msg)
    elif n_total > 0 and n_skipped / n_total > 0.1:
        msg = (
            f"Skipped {n_skipped} of {n_total} RES records while parsing SASA file {sasa_path!r}"
        )
        logger.warning(msg)
        _write_per_file_log(sasa_path, msg)

    _SASA_RAW_CACHE[abs_path] = records
    _SASA_TOTAL_CACHE[abs_path] = total_sasa
    return records


def parse_sasa(sasa_path: str) -> SASAParseResult:
    abs_path = str(Path(sasa_path).resolve())
    if not Path(abs_path).is_file():
        raise FileNotFoundError(f"SASA file not found: {sasa_path!r}")
    cached = _SASA_CACHE.get(abs_path)
    if cached is not None:
        return cached

    sasa_data: Dict[Tuple[str, int, str, str], SASAEntry] = {}

    for rec in load_sasa_raw(sasa_path):
        try:
            if rec.total_side_abs and rec.total_side_abs != "N/A":
                total_side_abs = float(rec.total_side_abs)
            else:
                total_side_abs = 0.0
            if rec.total_side_rel and rec.total_side_rel != "N/A":
                total_side_rel = float(rec.total_side_rel) / 100.0
            else:
                total_side_rel = 0.0

            if rec.main_chain_abs and rec.main_chain_abs != "N/A":
                main_chain_abs = float(rec.main_chain_abs)
            else:
                main_chain_abs = 0.0
            if rec.main_chain_rel and rec.main_chain_rel != "N/A":
                main_chain_rel = float(rec.main_chain_rel) / 100.0
            else:
                main_chain_rel = 0.0
            if rec.non_polar_abs and rec.non_polar_abs != "N/A":
                non_polar_abs = float(rec.non_polar_abs)
            else:
                non_polar_abs = 0.0
            if rec.non_polar_rel and rec.non_polar_rel != "N/A":
                non_polar_rel = float(rec.non_polar_rel) / 100.0
            else:
                non_polar_rel = 0.0
            if rec.all_polar_abs and rec.all_polar_abs != "N/A":
                all_polar_abs = float(rec.all_polar_abs)
            else:
                all_polar_abs = 0.0
            if rec.all_polar_rel and rec.all_polar_rel != "N/A":
                all_polar_rel = float(rec.all_polar_rel) / 100.0
            else:
                all_polar_rel = 0.0

            key = (rec.residue_name, rec.residue_number, rec.chain, rec.insertion_code)
            sasa_data[key] = SASAEntry(
                residue_name=rec.residue_name,
                chain=rec.chain,
                residue_number=rec.residue_number,
                total_side_abs=total_side_abs,
                total_side_rel=total_side_rel,
                main_chain_abs=main_chain_abs,
                main_chain_rel=main_chain_rel,
                non_polar_abs=non_polar_abs,
                non_polar_rel=non_polar_rel,
                all_polar_abs=all_polar_abs,
                all_polar_rel=all_polar_rel,
            )
        except ValueError:
            continue

    total = _SASA_TOTAL_CACHE.get(abs_path)
    if total is None:
        raise ValueError(
            f"SASA file {sasa_path!r} must contain a TOTAL line with a valid numeric total SASA (Å²)."
        )
    result = SASAParseResult(entries=sasa_data, total_sasa=float(total))
    _SASA_CACHE[abs_path] = result
    return result


def parse_pka(
    pka_path: str,
    pdb_atoms: Optional[List[Atom]] = None
) -> Dict[Tuple[str, int, str, str], float]:

    abs_path = str(Path(pka_path).resolve())
    atoms_id = id(pdb_atoms) if pdb_atoms is not None else 0
    cache_key = (abs_path, atoms_id)
    cached = _PKA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    pdb_residue_set: set = set()
    if pdb_atoms:
        for atom in pdb_atoms:
            inscode = getattr(atom, "insertion_code", "") or ""
            pdb_residue_set.add((atom.residue_name, atom.residue_number, atom.chain, inscode))
    residue_index = _build_residue_key_index(pdb_residue_set) if pdb_atoms else None

    pka_data: Dict[Tuple[str, int, str, str], float] = {}

    try:
        with open(pka_path, "r") as f:
            in_data_section = False
            skip_next_separator = False
            n_total = 0
            n_skipped = 0

            for raw_line in f:
                line = raw_line.rstrip("\n")

                if not in_data_section:
                    if "RESIDUE" in line and "pKa" in line:
                        in_data_section = True
                        skip_next_separator = True 
                    continue

                if skip_next_separator:
                    skip_next_separator = False
                    continue

                if not line.strip():
                    continue
                if line.strip().startswith("---"):
                    continue

                try:
                    n_total += 1
                    if len(line) < 15:
                        continue

                    # split() not fixed columns — slicing can truncate pKa values
                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    residue_name = parts[0].strip()
                    if not residue_name or residue_name not in CHARGED_RESIDUE_TYPES:
                        continue

                    residue_num_str = parts[1].strip()
                    if not residue_num_str:
                        continue
                    match = _RESNUM_PATTERN.match(residue_num_str)
                    if not match:
                        continue
                    residue_number = int(match.group(1))
                    insertion_code = (match.group(2) or "")

                    chain = parts[2].strip()
                    if not chain:
                        continue

                    pka_str = parts[3].strip().rstrip("*")
                    if not pka_str:
                        continue
                    try:
                        pka_value = float(pka_str)
                    except ValueError:
                        continue

                    residue_key = (residue_name, residue_number, chain, insertion_code)

                    if pdb_atoms and residue_name not in ("N+", "C-"):
                        mapped_key = resolve_external_residue_key(
                            residue_key,
                            pdb_residue_set,
                            residue_index=residue_index,
                        )
                        if mapped_key is None:
                            continue
                        residue_key = mapped_key

                    if residue_key not in pka_data:
                        pka_data[residue_key] = pka_value

                except (ValueError, IndexError):
                    n_skipped += 1
                    continue

    except FileNotFoundError:
        msg = f"pKa file {pka_path!r} not found; returning empty pKa map"
        logger.info(msg)
        _write_per_file_log(pka_path, msg)
        _PKA_CACHE[cache_key] = pka_data
        return pka_data
    except Exception as e:
        msg = f"Failed to read pKa file {pka_path!r}: {e}; returning partial/empty pKa map"
        logger.warning(msg)
        _write_per_file_log(pka_path, msg)
        _PKA_CACHE[cache_key] = pka_data
        return pka_data

    if n_total > 0 and not pka_data:
        msg = f"Parsed 0 charged residues with pKa values from {pka_path!r}"
        logger.info(msg)
        _write_per_file_log(pka_path, msg)
    elif n_total > 0 and n_skipped / n_total > 0.1:
        msg = (
            f"Skipped {n_skipped} of {n_total} potential pKa lines while parsing {pka_path!r}"
        )
        logger.warning(msg)
        _write_per_file_log(pka_path, msg)

    _PKA_CACHE[cache_key] = pka_data
    return pka_data


def load_dssp_lines(dssp_path: str) -> Tuple[List[str], Optional[int]]:
    abs_path = str(Path(dssp_path).resolve())
    cached = _DSSP_FILE_CACHE.get(abs_path)
    if cached is not None:
        return cached

    lines: List[str] = []
    header_line_idx: Optional[int] = None

    try:
        with open(dssp_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "#  RESIDUE AA STRUCTURE" in line:
                header_line_idx = i
                break
    except FileNotFoundError:
        msg = f"DSSP file {dssp_path!r} not found; returning empty DSSP lines"
        logger.info(msg)
        _write_per_file_log(dssp_path, msg)
        lines = []
        header_line_idx = None
    except Exception as e:
        msg = f"Failed to read DSSP file {dssp_path!r}: {e}; returning empty DSSP lines"
        logger.warning(msg)
        _write_per_file_log(dssp_path, msg)
        lines = []
        header_line_idx = None

    _DSSP_FILE_CACHE[abs_path] = (lines, header_line_idx)
    return lines, header_line_idx


def _parse_dssp_hbond_pair(hbond_str: str) -> Optional[Tuple[int, float]]:
    if not hbond_str or hbond_str.strip() == "":
        return None
    parts = hbond_str.split(",")
    if len(parts) != 2:
        return None
    offset_str = parts[0].strip()
    energy_str = parts[1].strip()
    if not offset_str or not energy_str:
        return None
    try:
        residue_offset = int(offset_str)
        energy = float(energy_str)
        if residue_offset == 0 and energy == 0.0:
            return None
        return (residue_offset, energy)
    except ValueError:
        return None


def parse_dssp(
    dssp_path: str,
    pdb_atoms: Optional[List[Atom]] = None,
) -> Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]]:
    abs_path = str(Path(dssp_path).resolve())
    atoms_id = id(pdb_atoms) if pdb_atoms is not None else 0
    cache_key = (abs_path, atoms_id)

    if cache_key in _DSSP_FULL_CACHE:
        dssp_data, _, _ = _DSSP_FULL_CACHE[cache_key]
        return dssp_data

    pdb_residue_set: set[Tuple[str, int, str, str]] = set()
    if pdb_atoms:
        for atom in pdb_atoms:
            pdb_residue_set.add(residue_key_from_atom(atom))
    residue_index = _build_residue_key_index(pdb_residue_set) if pdb_atoms else None

    dssp_data: Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]] = {}
    hbond_data: Dict[Tuple[str, int, str, str], List[Tuple[int, float]]] = {}
    dssp_seq_to_pdb: Dict[int, Tuple[str, int, str, str]] = {}

    lines, header_line_idx = load_dssp_lines(dssp_path)
    if header_line_idx is None:
        if lines:
            msg = f"DSSP file {dssp_path!r} has no expected header; returning empty data"
            logger.warning(msg)
            _write_per_file_log(dssp_path, msg)
        _DSSP_FULL_CACHE[cache_key] = (dssp_data, hbond_data, dssp_seq_to_pdb)
        return dssp_data

    for line_idx, line in enumerate(lines[header_line_idx + 1 :], start=header_line_idx + 1):
        line = line.rstrip("\n")
        if not line.strip():
            continue

        try:
            parsed_fields = parse_dssp_residue_fields(line)
            if parsed_fields is None:
                continue
            residue_number, insertion_code, chain = parsed_fields

            if len(line) < 14:
                continue
            aa_1letter = line[13:14].strip()
            if not aa_1letter or aa_1letter not in AA_1_TO_3:
                continue

            residue_name = AA_1_TO_3[aa_1letter]
            residue_key = (residue_name, residue_number, chain, insertion_code)

            if pdb_atoms:
                mapped_key = resolve_external_residue_key(
                    residue_key,
                    pdb_residue_set,
                    residue_index=residue_index,
                )
                if mapped_key is None:
                    continue
                residue_key = mapped_key

            dssp_seq_str = line[0:5].strip()
            if dssp_seq_str:
                try:
                    dssp_seq_num = int(dssp_seq_str)
                    dssp_seq_to_pdb[dssp_seq_num] = residue_key
                except ValueError:
                    pass

            secondary_structure = line[16] if len(line) > 16 else " "

            nh_o_1_str = ""
            oh_n_1_str = ""
            nh_o_2_str = ""
            oh_n_2_str = ""
            if len(line) > 40:
                matches = list(hbond_pattern.finditer(line[40:]))
                if len(matches) >= 4:
                    nh_o_1_str = f"{matches[0].group(1)},{matches[0].group(2)}"
                    oh_n_1_str = f"{matches[1].group(1)},{matches[1].group(2)}"
                    nh_o_2_str = f"{matches[2].group(1)},{matches[2].group(2)}"
                    oh_n_2_str = f"{matches[3].group(1)},{matches[3].group(2)}"
                elif len(matches) >= 2:
                    nh_o_1_str = f"{matches[0].group(1)},{matches[0].group(2)}"
                    oh_n_1_str = f"{matches[1].group(1)},{matches[1].group(2)}"

                hbond_pairs: List[Tuple[int, float]] = []
                for m in matches:
                    hbond_str = f"{m.group(1)},{m.group(2)}"
                    pair = _parse_dssp_hbond_pair(hbond_str)
                    if pair is not None:
                        hbond_pairs.append(pair)
                if hbond_pairs:
                    hbond_data[residue_key] = hbond_pairs

            _nh_o_1 = _parse_dssp_hbond_pair(nh_o_1_str)
            nh_o_1_energy = _nh_o_1[1] if _nh_o_1 is not None else None
            _oh_n_1 = _parse_dssp_hbond_pair(oh_n_1_str)
            oh_n_1_energy = _oh_n_1[1] if _oh_n_1 is not None else None
            _nh_o_2 = _parse_dssp_hbond_pair(nh_o_2_str)
            nh_o_2_energy = _nh_o_2[1] if _nh_o_2 is not None else None
            _oh_n_2 = _parse_dssp_hbond_pair(oh_n_2_str)
            oh_n_2_energy = _oh_n_2[1] if _oh_n_2 is not None else None

            dssp_data[residue_key] = {
                "secondary_structure": secondary_structure,
                "N-H-->O_1": nh_o_1_energy,
                "N-H-->O_2": nh_o_2_energy,
                "O-->H-N_1": oh_n_1_energy,
                "O-->H-N_2": oh_n_2_energy,
            }
        except (ValueError, IndexError):
            continue

    _DSSP_FULL_CACHE[cache_key] = (dssp_data, hbond_data, dssp_seq_to_pdb)
    return dssp_data


def parse_dssp_hbonds(
    dssp_path: str,
    pdb_atoms: Optional[List[Atom]] = None,
) -> Tuple[
    Dict[Tuple[str, int, str, str], List[Tuple[int, float]]],
    Dict[int, Tuple[str, int, str, str]],
]:
    abs_path = str(Path(dssp_path).resolve())
    atoms_id = id(pdb_atoms) if pdb_atoms is not None else 0
    cache_key = (abs_path, atoms_id)

    if cache_key in _DSSP_FULL_CACHE:
        _, hbond_data, dssp_seq_to_pdb = _DSSP_FULL_CACHE[cache_key]
        return hbond_data, dssp_seq_to_pdb

    _ = parse_dssp(dssp_path, pdb_atoms)
    _, hbond_data, dssp_seq_to_pdb = _DSSP_FULL_CACHE.get(
        cache_key, ({}, {}, {})
    )
    return hbond_data, dssp_seq_to_pdb
