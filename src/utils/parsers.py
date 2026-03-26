from typing import Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass
import math
import re
from pathlib import Path
import logging

# residue number with insertion code (e.g. 111A)
_RESNUM_PATTERN = re.compile(r"^(\d+)([A-Za-z])?$")
hbond_pattern = re.compile(r"(-?\d+),\s*(-?\d+\.?\d*)")

# Residues we extract pKa values for from PropKa output.
#
# Note: this is intentionally broader than the residue/atom sets used for
# geometric salt-bridge detection. Charge/pI counting and salt-bridge
# geometry are handled separately elsewhere in the pipeline.
CHARGED_RESIDUE_TYPES = frozenset({"ASP", "GLU", "LYS", "ARG", "HIS", "TYR", "CYS"})

_PKA_CACHE: Dict[Tuple[str, int], Dict[Tuple[str, int, str, str], float]] = {}

STANDARD_AA = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})
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
    """
    "111" -> (111, ""), "111A" -> (111, "A"). Returns None if not parseable.
    """
    s = (s or "").strip()
    if not s:
        return None
    match = _RESNUM_PATTERN.match(s)
    if not match:
        return None
    num = int(match.group(1))
    ins = (match.group(2) or "")
    return (num, ins)

@dataclass(frozen=True)
class Atom:
    """Atom from a PDB file. Frozen so Atom instances can be used as dict keys (e.g. in H-bond counting)."""
    serial: int 
    name: str  # CA
    residue_name: str  # 3-letter
    chain: str
    residue_number: int
    insertion_code: str  # PDB column 27; '' when none (e.g. residues 111B, 112A)
    x: float
    y: float
    z: float
    element: str  # C

@dataclass
class SASAEntry:
    """SASA data for a residue"""
    residue_name: str
    chain: str
    residue_number: int
    total_side_abs: float
    total_side_rel: float
    main_chain_abs: float
    main_chain_rel: float


@dataclass(frozen=True)
class SASARawRecord:
    """
    Raw SASA record as parsed from the file.
    """
    residue_name: str
    chain: str
    residue_number: int
    insertion_code: str  # '' when not present
    total_side_abs: str
    total_side_rel: str
    main_chain_abs: str
    main_chain_rel: str

_SASA_RAW_CACHE: Dict[str, List[SASARawRecord]] = {}
_SASA_CACHE: Dict[str, Dict[Tuple[str, int, str, str], SASAEntry]] = {}

# format: ATOM  serial  name  alt  res  chain  resnum  x  y  z  occ  temp  element
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

def _split_cif_line(line: str) -> List[str]:

    values: List[str] = []
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i] in " \t":
            i += 1
        if i >= n:
            break
        if line[i] in "\"'":
            quote = line[i]
            i += 1
            start = i
            while i < n and line[i] != quote:
                if line[i] == "\\":
                    i += 1
                i += 1
            values.append(line[start:i].strip())
            if i < n:
                i += 1 
            continue
        start = i
        while i < n and line[i] not in " \t":
            i += 1
        values.append(line[start:i].strip())
    return values


def parse_cif(cif_path: str, allowed_chains: Optional[Iterable[str]] = None) -> List[Atom]:

    chains = frozenset(allowed_chains) if allowed_chains is not None else ALLOWED_CHAINS
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

    # find atom_site loop: loop_ followed by _atom_site.* column names, then data
    i = 0
    col_indices: Optional[Dict[str, int]] = None
    required = {
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z",
    }

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
        # Parse data lines
        n_total = 0
        n_skipped = 0
        while i < len(lines):
            data_line = lines[i].rstrip("\n")
            i += 1
            if not data_line.strip():
                continue
            if data_line.startswith("#") or data_line.startswith("_") or data_line.startswith("loop_"):
                i -= 1
                break
            parts = _split_cif_line(data_line)
            if len(parts) <= max(col_indices.values()):
                continue
            try:
                n_total += 1
                group = parts[col_indices["group_PDB"]].strip()
                if group != "ATOM":
                    continue
                serial = int(parts[col_indices["id"]].strip() or "0")
                element = (parts[col_indices["type_symbol"]].strip() or "") or (parts[col_indices[atom_name_col]].strip()[:1] or "?")
                name = parts[col_indices[atom_name_col]].strip()
                alt_loc = (parts[col_indices["label_alt_id"]].strip() or "").replace(".", "")
                residue_name = parts[col_indices[resname_col]].strip()
                chain = parts[col_indices[chain_col]].strip()
                if chain not in chains or residue_name not in STANDARD_AA:
                    continue
                resnum_str = parts[col_indices[resnum_col]].strip().replace("?", "")
                if not resnum_str:
                    continue
                residue_number = int(resnum_str)
                insertion_code = ""
                if ins_code_col and ins_code_col in col_indices:
                    ins = parts[col_indices[ins_code_col]].strip().replace("?", "").replace(".", "")
                    insertion_code = ins if ins else ""
                x = float(parts[col_indices["Cartn_x"]].strip().replace("?", "0"))
                y = float(parts[col_indices["Cartn_y"]].strip().replace("?", "0"))
                z = float(parts[col_indices["Cartn_z"]].strip().replace("?", "0"))
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
        break  # one atom_site loop per file

    if n_total > 0 and not atoms:
        logger.warning("Parsed 0 ATOM records from CIF %r (chains=%r)", cif_path, chains)
    elif n_total > 0 and n_skipped / n_total > 0.1:
        logger.warning(
            "Skipped %d of %d atom_site records while parsing CIF %r",
            n_skipped,
            n_total,
            cif_path,
        )

    return atoms


def parse_structure(
    path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> List[Atom]:

    if path.lower().endswith(".cif"):
        return parse_cif(path, allowed_chains=allowed_chains)
    return parse_pdb(path, allowed_chains=allowed_chains)


def residue_key_from_atom(atom: Atom) -> Tuple[str, int, str, str]:
    return (atom.residue_name, atom.residue_number, atom.chain, getattr(atom, "insertion_code", ""))

# format: RES resName chain resNum (original residue number, may have letter)
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
                    # FreeSASA residue format (split parts):
                    # RES <resName> <chain> <num> <all_abs> <all_rel> <side_abs> <side_rel> <main_abs> <main_rel> ...
                    total_side_abs = parts[6] if len(parts) > 6 else ""
                    total_side_rel = parts[7] if len(parts) > 7 else ""
                    main_chain_abs = parts[8] if len(parts) > 8 else ""
                    main_chain_rel = parts[9] if len(parts) > 9 else ""
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
                        )
                    )
                except (ValueError, IndexError):
                    n_skipped += 1
                    continue
    except FileNotFoundError:
        msg = f"SASA file {sasa_path!r} not found; returning empty records list"
        logger.info(msg)
        _write_per_file_log(sasa_path, msg)
        _SASA_RAW_CACHE[abs_path] = []
        _SASA_TOTAL_CACHE[abs_path] = None
        return []
    except Exception as e:
        msg = f"Failed to read SASA file {sasa_path!r}: {e}; returning partial/empty records"
        logger.warning(msg)
        _write_per_file_log(sasa_path, msg)
        _SASA_RAW_CACHE[abs_path] = records
        _SASA_TOTAL_CACHE[abs_path] = total_sasa
        return records

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


def get_sasa_total(sasa_path: str) -> Optional[float]:
    """Return cached total SASA (from TOTAL line) for path; load file if not yet cached."""
    abs_path = str(Path(sasa_path).resolve())
    if abs_path not in _SASA_TOTAL_CACHE:
        load_sasa_raw(sasa_path)
    return _SASA_TOTAL_CACHE.get(abs_path)


def parse_sasa(sasa_path: str) -> Dict[Tuple[str, int, str, str], SASAEntry]:
    abs_path = str(Path(sasa_path).resolve())
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

            key = (rec.residue_name, rec.residue_number, rec.chain, rec.insertion_code)
            sasa_data[key] = SASAEntry(
                residue_name=rec.residue_name,
                chain=rec.chain,
                residue_number=rec.residue_number,
                total_side_abs=total_side_abs,
                total_side_rel=total_side_rel,
                main_chain_abs=main_chain_abs,
                main_chain_rel=main_chain_rel,
            )
        except ValueError:
            continue

    _SASA_CACHE[abs_path] = sasa_data
    return sasa_data


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

                    # PropKa output has fixed-width columns, but whitespace
                    # padding differs between residue types and pKa magnitudes.
                    # Using split() avoids column-slicing errors like:
                    #   99.99 -> 9.99 and 11.87 -> 1.87
                    # which severely distorts titration/charge calculations.
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

                    pka_str = parts[3].strip()
                    if not pka_str:
                        continue
                    try:
                        pka_value = float(pka_str)
                    except ValueError:
                        continue

                    residue_key = (residue_name, residue_number, chain, insertion_code)

                    if pdb_atoms and residue_key not in pdb_residue_set:
                        continue

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


def get_pka_file_path(pdb_path: str) -> Optional[str]:
    """
    pdgf38/AB-001.pdb -> pdgf38_propka/AB-001_full.pka
    """
    pdb_path_obj = Path(pdb_path)
    pdb_basename = pdb_path_obj.stem 
    pdb_dir = pdb_path_obj.parent

    pka_dir = pdb_dir.parent / f"{pdb_dir.name}_propka"
    pka_path = pka_dir / f"{pdb_basename}_full.pka"
    
    if pka_path.exists():
        return str(pka_path)
    
    return None

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


def parse_dssp(
    dssp_path: str,
    pdb_atoms: Optional[List[Atom]] = None,
) -> Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]]:
    """
    - secondary structure 
    - donor H-bond energies
    - acceptor H-bond energies
    """
    from developability.descriptors import AA_1_TO_3  # lazy import to avoid cycles

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
            parsed_res = parse_residue_number_field(line[5:10])
            if parsed_res is None:
                continue
            residue_number, insertion_code = parsed_res

            if len(line) < 12:
                continue
            chain = line[11:12].strip()
            if not chain:
                continue

            if len(line) < 14:
                continue
            aa_1letter = line[13:14].strip()
            if not aa_1letter or aa_1letter not in AA_1_TO_3:
                continue

            residue_name = AA_1_TO_3[aa_1letter]
            residue_key = (residue_name, residue_number, chain, insertion_code)

            if pdb_atoms and residue_key not in pdb_residue_set:
                continue

            # per-residue map from seq index to key (needed for hbonds)
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

                # H-bond offset/energy pairs for this residue (sequential-index-based)
                hbond_pairs: List[Tuple[int, float]] = []
                for m in matches:
                    hbond_str = f"{m.group(1)},{m.group(2)}"
                    pair = _parse_dssp_hbond_pair(hbond_str)
                    if pair is not None:
                        hbond_pairs.append(pair)
                if hbond_pairs:
                    hbond_data[residue_key] = hbond_pairs

            def _parse_hbond_energy(hbond_str: str) -> Optional[float]:
                if not hbond_str or hbond_str.strip() == "":
                    return None
                parts = hbond_str.split(",")
                if len(parts) != 2:
                    return None
                energy_str = parts[1].strip()
                if not energy_str:
                    return None
                try:
                    return float(energy_str)
                except ValueError:
                    return None

            nh_o_1_energy = _parse_hbond_energy(nh_o_1_str)
            oh_n_1_energy = _parse_hbond_energy(oh_n_1_str)
            nh_o_2_energy = _parse_hbond_energy(nh_o_2_str)
            oh_n_2_energy = _parse_hbond_energy(oh_n_2_str)

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
    _, hbond_data, dssp_seq_to_pdb = _DSSP_FULL_CACHE.get(cache_key, ({}, {} ,{}))
    return hbond_data, dssp_seq_to_pdb


def parse_motif_to_3letter(motif: str, aa_map: Optional[Dict[str, str]] = None) -> List[str]:

    from developability.descriptors import AA_1_TO_3 as _AA_1_TO_3

    if aa_map is None:
        aa_map = _AA_1_TO_3

    parts = [p.strip() for p in motif.split("-") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid motif: {motif!r}")
    result: List[str] = []
    for p in parts:
        u = p.upper()
        if len(u) == 1:
            three = aa_map.get(u)
            if three is None:
                raise ValueError(f"Unknown 1-letter code in motif: {p!r}")
            result.append(three)
        elif len(u) == 3:
            result.append(u)
        else:
            raise ValueError(f"Residue in motif must be 1- or 3-letter: {p!r}")
    return result

