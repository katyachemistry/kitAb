"""
Parsers for PDB structure files, CIF structure files, SASA files and pKa files.
"""
# TODO: add arbitrary heavy and light chain names: after parse_pdb is outside every function that uses it, make it a global variable

from typing import Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass
import math
import re
from pathlib import Path

# residue number with insertion code (original PDB-style, may have trailing letter e.g. 111A)
_RESNUM_PATTERN = re.compile(r"^(\d+)([A-Za-z])?$")


def parse_residue_number_field(s: str) -> Optional[Tuple[int, str]]:
    """
    Parse original residue number string (numeric + optional insertion code).
    Used by SASA, DSSP and PropKA parsers for unified 4-tuple keys.
    E.g. "111" -> (111, ""), "111A" -> (111, "A"). Returns None if not parseable.
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

STANDARD_AA = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})
ALLOWED_CHAINS = frozenset({"H", "L"})


@dataclass
class Atom:
    """Atom from a PDB file."""
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
    """SASA data for a residue."""
    residue_name: str
    chain: str
    residue_number: int
    total_side_rel: float
    main_chain_rel: float


@dataclass(frozen=True)
class SASARawRecord:
    """
    Raw SASA record as parsed from the file.
    Residue number is original PDB-style (4th column); may have insertion_code.
    """
    residue_name: str
    chain: str
    residue_number: int
    insertion_code: str  # '' when not present
    total_side_rel: str
    main_chain_rel: str


_SASA_RAW_CACHE: Dict[str, List[SASARawRecord]] = {}
_SASA_CACHE: Dict[str, Dict[Tuple[str, int, str, str], SASAEntry]] = {}

# format: ATOM  serial  name  alt  res  chain  resnum  x  y  z  occ  temp  element
def parse_pdb(pdb_path: str) -> List[Atom]:

    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    serial = int(line[6:11].strip())
                    name = line[12:16].strip()
                    alt_loc = line[16:17].strip()
                    residue_name = line[17:20].strip()
                    chain = line[21:22].strip()
                    if chain not in ALLOWED_CHAINS or residue_name not in STANDARD_AA:
                        continue
                    residue_number = int(line[22:26].strip())
                    insertion_code = (line[26:27].strip() or '')
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = line[76:78].strip() if len(line) > 76 else name[0]
                    
                    # if alternate location is present, only take first
                    if alt_loc == '' or alt_loc == 'A':
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
                            element=element
                        )
                        atoms.append(atom)
                except (ValueError, IndexError) as e:
                    continue
    
    return atoms # add atom list check downstream!!


def _split_cif_line(line: str) -> List[str]:
    """
    Split a CIF data line into values. Handles quoted strings (single or double)
    that may contain spaces. Unquoted ? and . are kept as-is.
    """
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
                    i += 1  # skip escaped char
                i += 1
            values.append(line[start:i].strip())
            if i < n:
                i += 1  # skip closing quote
            continue
        start = i
        while i < n and line[i] not in " \t":
            i += 1
        values.append(line[start:i].strip())
    return values


def parse_cif(cif_path: str, allowed_chains: Optional[Iterable[str]] = None) -> List[Atom]:
    """
    Parse an mmCIF structure file and return a list of Atom (same as parse_pdb).

    Only ATOM rows are kept; HETATM are skipped. Uses auth_* fields for
    chain/residue/atom to match PDB convention. Filters by standard amino acids
    and allowed chain IDs (default ALLOWED_CHAINS). Alternate location: only
    '' or 'A' are kept.
    """
    chains = frozenset(allowed_chains) if allowed_chains is not None else ALLOWED_CHAINS
    atoms: List[Atom] = []

    with open(cif_path, "r") as f:
        lines = list(f)

    # Find atom_site loop: loop_ followed by _atom_site.* column names, then data
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
        # Collect _atom_site.* column names
        col_names: List[str] = []
        while i < len(lines):
            L = lines[i]
            i += 1
            L = L.strip()
            if not L or L.startswith("#"):
                continue
            if not L.startswith("_atom_site."):
                # Back up one line so the next parser can use it
                i -= 1
                break
            name = L[len("_atom_site.") :].strip()
            col_names.append(name)
        if not col_names:
            continue
        # Build column index map (use short names without _atom_site. prefix)
        col_indices = {name: idx for idx, name in enumerate(col_names)}
        if not required.issubset(col_indices):
            continue
        # Prefer auth_* for chain/residue/insertion; fall back to label_*
        chain_col = "auth_asym_id" if "auth_asym_id" in col_indices else "label_asym_id"
        resname_col = "auth_comp_id" if "auth_comp_id" in col_indices else "label_comp_id"
        resnum_col = "auth_seq_id" if "auth_seq_id" in col_indices else "label_seq_id"
        ins_code_col = "pdbx_PDB_ins_code" if "pdbx_PDB_ins_code" in col_indices else None
        atom_name_col = "auth_atom_id" if "auth_atom_id" in col_indices else "label_atom_id"
        # Parse data lines
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
                continue
        break  # one atom_site loop per file

    return atoms


def parse_structure(
    path: str, allowed_chains: Optional[Iterable[str]] = None
) -> List[Atom]:
    """
    Parse a structure file (PDB or mmCIF) and return a list of Atom.
    Chooses parser by suffix: .cif -> parse_cif, otherwise -> parse_pdb.
    """
    if path.lower().endswith(".cif"):
        return parse_cif(path, allowed_chains=allowed_chains)
    return parse_pdb(path)  # parse_pdb does not yet accept allowed_chains


def residue_key_from_atom(atom: Atom) -> Tuple[str, int, str, str]:
    return (atom.residue_name, atom.residue_number, atom.chain, getattr(atom, "insertion_code", ""))

def load_sasa_raw(sasa_path: str) -> List[SASARawRecord]:
    """
    Load raw SASA records from file, with simple caching by absolute path.

    This function is shared by both high-level SASA parsers so the SASA
    file is only read once per path in a given process.
    """
    abs_path = str(Path(sasa_path).resolve())
    cached = _SASA_RAW_CACHE.get(abs_path)
    if cached is not None:
        return cached

    records: List[SASARawRecord] = []

    # Format: RES resName chain resNum ... (4th column = original residue number, may have letter e.g. 111A)
    try:
        with open(sasa_path, "r") as f:
            for line in f:
                if not line.startswith("RES "):
                    continue
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
                    total_side_rel = parts[7] if len(parts) > 7 else ""
                    main_chain_rel = parts[9] if len(parts) > 9 else ""
                    records.append(
                        SASARawRecord(
                            residue_name=residue_name,
                            chain=chain,
                            residue_number=residue_number,
                            insertion_code=insertion_code,
                            total_side_rel=total_side_rel,
                            main_chain_rel=main_chain_rel,
                        )
                    )
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        records = []
    except Exception:
        records = []

    _SASA_RAW_CACHE[abs_path] = records
    return records


def parse_sasa(sasa_path: str) -> Dict[Tuple[str, int, str, str], SASAEntry]:
    """
    Parse SASA file into normalized SASAEntry records.
    Keys are 4-tuples (residue_name, residue_number, chain, insertion_code);
    residue number is from the 4th column (original PDB-style, may include letter).
    """
    abs_path = str(Path(sasa_path).resolve())
    cached = _SASA_CACHE.get(abs_path)
    if cached is not None:
        return cached

    sasa_data: Dict[Tuple[str, int, str, str], SASAEntry] = {}

    for rec in load_sasa_raw(sasa_path):
        try:
            if rec.total_side_rel and rec.total_side_rel != "N/A":
                total_side_rel = float(rec.total_side_rel) / 100.0
            else:
                total_side_rel = 0.0

            if rec.main_chain_rel and rec.main_chain_rel != "N/A":
                main_chain_rel = float(rec.main_chain_rel) / 100.0
            else:
                main_chain_rel = 0.0

            key = (rec.residue_name, rec.residue_number, rec.chain, rec.insertion_code)
            sasa_data[key] = SASAEntry(
                residue_name=rec.residue_name,
                chain=rec.chain,
                residue_number=rec.residue_number,
                total_side_rel=total_side_rel,
                main_chain_rel=main_chain_rel,
            )
        except ValueError:
            continue

    _SASA_CACHE[abs_path] = sasa_data
    return sasa_data


def distance(atom1: Atom, atom2: Atom) -> float:
    """Euclidean distance between two atoms."""
    dx = atom1.x - atom2.x
    dy = atom1.y - atom2.y
    dz = atom1.z - atom2.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def angle_between_vectors(v1: Tuple[float, float, float], 
                          v2: Tuple[float, float, float]) -> float:
    """
    Angle between two vectors, in degrees.
    """
    dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    mag1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
    mag2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

backbone_atoms = {'N', 'CA', 'C', 'O', 'H', 'HA', 'HA2', 'HA3'}
def is_backbone_atom(atom_name: str) -> bool:
    return atom_name in backbone_atoms or atom_name.startswith('H') and len(atom_name) <= 3


# residue types that can have pKa values
CHARGED_RESIDUE_TYPES = frozenset({"ASP", "GLU", "LYS", "ARG", "HIS"})

_PKA_CACHE: Dict[Tuple[str, int], Dict[Tuple[str, int, str, str], float]] = {}

def parse_pka(
    pka_path: str,
    pdb_atoms: Optional[List[Atom]] = None
) -> Dict[Tuple[str, int, str, str], float]:
    """
    Extract pKa values for ASP, GLU, LYS, ARG, and HIS residues.
    Returns keys as (residue_name, residue_number, chain, insertion_code).
    Insertion code is parsed from PropKa when present (e.g. 111A); otherwise ''.
    Results are cached per (file path, atoms list) to avoid redundant parsing.
    """
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

            for raw_line in f:
                line = raw_line.rstrip("\n")

                # Find header line once
                if not in_data_section:
                    if "RESIDUE" in line and "pKa" in line:
                        in_data_section = True
                        skip_next_separator = True  # Next line is the separator
                    continue

                # Skip the separator line immediately after the header
                if skip_next_separator:
                    skip_next_separator = False
                    continue

                if not line.strip():
                    continue
                if line.strip().startswith("---"):
                    continue

                try:
                    if len(line) < 15:
                        continue

                    residue_name = line[0:3].strip()
                    if not residue_name or residue_name not in CHARGED_RESIDUE_TYPES:
                        continue

                    residue_num_str = line[4:8].strip()
                    if not residue_num_str:
                        continue
                    match = _RESNUM_PATTERN.match(residue_num_str)
                    if not match:
                        continue
                    residue_number = int(match.group(1))
                    insertion_code = (match.group(2) or "")

                    if len(line) < 11:
                        continue
                    chain = line[10:11].strip()
                    if not chain and len(line) >= 9:
                        chain = line[8:9].strip()
                    if not chain:
                        continue

                    if len(line) < 18:
                        continue
                    pka_str = line[12:18].strip()
                    if not pka_str and len(line) >= 17:
                        pka_str = line[10:17].strip()
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
                    continue

    except FileNotFoundError:
        _PKA_CACHE[cache_key] = pka_data
        return pka_data
    except Exception:
        _PKA_CACHE[cache_key] = pka_data
        return pka_data

    _PKA_CACHE[cache_key] = pka_data
    return pka_data


def get_pka_file_path(pdb_path: str) -> Optional[str]:
    """
    Looks for {pdb_folder}_propka/{stem}_full.pka as a sibling of the structure.
    E.g. pdgf38/AB-001.pdb -> pdgf38_propka/AB-001_full.pka
    """
    pdb_path_obj = Path(pdb_path)
    pdb_basename = pdb_path_obj.stem 
    pdb_dir = pdb_path_obj.parent

    pka_dir = pdb_dir.parent / f"{pdb_dir.name}_propka"
    pka_path = pka_dir / f"{pdb_basename}_full.pka"
    
    if pka_path.exists():
        return str(pka_path)
    
    return None

