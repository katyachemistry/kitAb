"""
This module calculates developability determinants.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple
import os
import math
from collections import defaultdict
import re
import numpy as np
from scipy.spatial import cKDTree
from scipy import optimize
from sklearn.cluster import DBSCAN

from utils.parsers import (
    parse_pdb,
    parse_structure,
    parse_sasa,
    load_sasa_raw,
    parse_residue_number_field,
    Atom,
    SASAEntry,
    distance,
    angle_between_vectors,
    is_backbone_atom,
    parse_pka,
    get_pka_file_path,
    residue_key_from_atom,
)

AA_1_TO_3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

POSITIVE_ATOMS = frozenset({
    ("NH1", "ARG"),
    ("NH2", "ARG"),
    ("NZ", "LYS"),
})

NEGATIVE_ATOMS = frozenset({
    ("OD1", "ASP"),
    ("OD2", "ASP"),
    ("OE1", "GLU"),
    ("OE2", "GLU"),
})

# Residue type sets for residue category density (and related helpers)
AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP"})
HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"})
POLAR_RESIDUES = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "CYS", "GLU", "ASP", "LYS", "ARG", "HIS"})
NEGATIVE_CHARGED_RESIDUES = frozenset({"ASP", "GLU"})
POSITIVE_CHARGED_RESIDUES = frozenset({"LYS", "ARG", "HIS"})
CHARGED_RESIDUES = NEGATIVE_CHARGED_RESIDUES | POSITIVE_CHARGED_RESIDUES  # all charged (same as CHARGED_RESIDUE_TYPES from parsers)

DONORS_ANY = frozenset({"N"})
DONOR_EXCLUDED = frozenset({("N", "PRO")})

DONOR_INFO: Dict[Tuple[str, str], str] = {
    ("NE2", "GLN"): "CD",
    ("ND2", "ASN"): "CG",
    ("NE",  "ARG"): "CZ",
    ("NH1", "ARG"): "CZ",
    ("NH2", "ARG"): "CZ",
    ("NZ",  "LYS"): "CE",
    ("ND1", "HIS"): "CG",
    ("NE2", "HIS"): "CD2",
    ("OG",  "SER"): "CB",
    ("OG1", "THR"): "CB",
    ("OH",  "TYR"): "CZ",
}

ACCEPTORS_ANY = frozenset({"O"})

ACCEPTORS_SPECIFIC = frozenset({
    ("OE1", "GLN"),
    ("OE2", "GLU"),
    ("OD1", "ASN"),
    ("OD2", "ASP"),
    ("ND1", "HIS"),
    ("NE2", "HIS"),
    ("OG", "SER"),
    ("OG1", "THR"),
    ("OH", "TYR"),
})

MAX_HBOND_DISTANCE = 3.2
MAX_SALT_BRIDGE_DISTANCE = 4.0
INTER_CHAIN_INTERFACE_CUTOFF = 5.0  # Å; residue is in inter-chain interface if any heavy atom is within this of another chain's heavy atom

# angle base -> donor -> acceptor
MIN_HBOND_ANGLE = 120.0

MIN_BACKBONE_SEPARATION = 3

# Generic terminal pKa constants used when estimating net charge / pI.
# For antibody VH+VL this corresponds to 2 N-termini and 2 C-termini.
NTERM_PKA = 8.0
CTERM_PKA = 3.1

_ATOM_LOOKUP_CACHE: Dict[int, Dict[Tuple[str, int, str], Atom]] = {}

# maps id(atoms) -> (residue_name, resnum, chain, insertion_code) -> sequence index
_RES_SEQ_INDEX_CACHE: Dict[int, Dict[Tuple[str, int, str, str], int]] = {}

# maps structure path -> list of atoms
_ATOMS_CACHE: Dict[str, List[Atom]] = {}

# cache DSSP file lines and header index per absolute path
_DSSP_FILE_CACHE: Dict[str, Tuple[List[str], Optional[int]]] = {}

# cache geometry-based H-bond pairs per structure path to avoid re-enumeration
_HBOND_PAIRS_CACHE: Dict[str, List[Tuple[Atom, Atom]]] = {}

# cache unique residue counts per structure (used by many *_average helpers)
_RES_COUNT_CACHE: Dict[str, int] = {}

# cache: pdb abs path -> set of residue_key (4-tuple) that are in inter-chain interface
_INTER_CHAIN_INTERFACE_CACHE: Dict[str, Set[Tuple[str, int, str, str]]] = {}

# cache residue_key (4-tuple) -> "CDR1"|"CDR2"|"CDR3"|"framework" per atoms list
_RESIDUE_REGION_CACHE: Dict[int, Dict[Tuple[str, int, str, str], str]] = {}

# cache: pdb abs path -> sequence per chain (list of 3-letter codes)
_PDB_SEQUENCE_CACHE: Dict[str, Dict[str, List[str]]] = {}

# cache: pdb abs path -> set of unique residue keys (4-tuples)
_PDB_RESIDUE_KEYS_CACHE: Dict[str, Set[Tuple[str, int, str, str]]] = {}

# CDR regions by original residue number (inclusive). Framework = everything else.
# Residues with insertion codes (e.g. 111A) use the numeric part only (111 in 105-117 -> CDR3).
CDR1_RANGE = (27, 38)
CDR2_RANGE = (56, 65)
CDR3_RANGE = (105, 117)


def get_residue_region(residue_number: int) -> str:
    """
    Classify a residue by original PDB residue number (integer part) as CDR1, CDR2, CDR3, or framework.

    Uses only the numeric residue number; insertion codes (e.g. 111A) are irrelevant — the integer
    part is what matters (e.g. 111 → CDR3). PDB/parsers store residue_number as int and
    insertion_code separately.

    Args:
        residue_number: PDB residue number (integer part only).

    Returns:
        "CDR1", "CDR2", "CDR3", or "framework".
    """
    if CDR1_RANGE[0] <= residue_number <= CDR1_RANGE[1]:
        return "CDR1"
    if CDR2_RANGE[0] <= residue_number <= CDR2_RANGE[1]:
        return "CDR2"
    if CDR3_RANGE[0] <= residue_number <= CDR3_RANGE[1]:
        return "CDR3"
    return "framework"


def get_residue_region_map(atoms: List[Atom]) -> Dict[Tuple[str, int, str, str], str]:
    """
    Map each residue (by 4-tuple key) to "CDR1", "CDR2", "CDR3", or "framework".

    Uses original residue numbers only (insertion codes do not affect classification).
    Result is cached by id(atoms) so repeated calls with the same atom list are cheap.

    Args:
        atoms: List of atoms (e.g. from _get_atoms_for_path).

    Returns:
        Dict mapping (residue_name, residue_number, chain, insertion_code) -> region string.
    """
    atoms_id = id(atoms)
    cached = _RESIDUE_REGION_CACHE.get(atoms_id)
    if cached is not None:
        return cached

    out: Dict[Tuple[str, int, str, str], str] = {}
    seen: Set[Tuple[str, int, str, str]] = set()
    for atom in atoms:
        key = residue_key_from_atom(atom)
        if key in seen:
            continue
        seen.add(key)
        out[key] = get_residue_region(atom.residue_number)

    _RESIDUE_REGION_CACHE[atoms_id] = out
    return out


def _get_atoms_for_path(pdb_path: str) -> List[Atom]:
    abs_path = os.path.abspath(pdb_path)
    cached = _ATOMS_CACHE.get(abs_path)
    if cached is not None:
        return cached
    atoms = parse_structure(pdb_path)
    _ATOMS_CACHE[abs_path] = atoms
    return atoms


def _compute_inter_chain_interface_from_by_chain(
    by_chain: Dict[str, List[Atom]],
) -> Set[Tuple[str, int, str, str]]:
    """
    Compute interface residue set from heavy atoms already grouped by chain.
    Used so callers can reuse a single pass (e.g. when building all_residues + by_chain together).
    """
    interface: Set[Tuple[str, int, str, str]] = set()
    chains = list(by_chain.keys())
    cutoff = INTER_CHAIN_INTERFACE_CUTOFF
    for i, c1 in enumerate(chains):
        for c2 in chains[i + 1 :]:
            coords1 = np.array([[a.x, a.y, a.z] for a in by_chain[c1]], dtype=np.float64)
            coords2 = np.array([[a.x, a.y, a.z] for a in by_chain[c2]], dtype=np.float64)
            tree1 = cKDTree(coords1)
            tree2 = cKDTree(coords2)
            for a in by_chain[c1]:
                if tree2.query_ball_point([a.x, a.y, a.z], cutoff):
                    interface.add(residue_key_from_atom(a))
            for a in by_chain[c2]:
                if tree1.query_ball_point([a.x, a.y, a.z], cutoff):
                    interface.add(residue_key_from_atom(a))
    return interface


def get_inter_chain_interface_residues(
    pdb_path: str,
    by_chain_heavy: Optional[Dict[str, List[Atom]]] = None,
) -> Set[Tuple[str, int, str, str]]:
    """
    Residues that have at least one heavy atom within INTER_CHAIN_INTERFACE_CUTOFF (5 Å)
    of any heavy atom on another chain. Result is cached by PDB path so callers can reuse it.

    If by_chain_heavy is provided (e.g. from a fused loop that already grouped heavy atoms
    by chain), it is used and the result is cached; avoids a second pass over the structure.

    Returns:
        Set of 4-tuple residue keys (residue_name, residue_number, chain, insertion_code).
    """
    abs_path = os.path.abspath(pdb_path)
    cached = _INTER_CHAIN_INTERFACE_CACHE.get(abs_path)
    if cached is not None:
        return cached

    if by_chain_heavy is not None:
        interface = _compute_inter_chain_interface_from_by_chain(by_chain_heavy)
    else:
        atoms = _get_atoms_for_path(pdb_path)
        by_chain: Dict[str, List[Atom]] = defaultdict(list)
        for a in atoms:
            if getattr(a, "element", "X") != "H":
                by_chain[a.chain].append(a)
        interface = _compute_inter_chain_interface_from_by_chain(by_chain)

    _INTER_CHAIN_INTERFACE_CACHE[abs_path] = interface
    return interface


def is_residue_in_inter_chain_interface(pdb_path: str, residue_key: Tuple[str, int, str, str]) -> bool:
    """True if the residue (4-tuple key) has any heavy atom within 5 Å of another chain's heavy atom."""
    return residue_key in get_inter_chain_interface_residues(pdb_path)


def _load_dssp_lines(dssp_path: str) -> Tuple[List[str], Optional[int]]:
    """
    Load DSSP file lines and locate the header marking the start of data.

    Both DSSP parsers in this module use this helper so that, when they are
    called in the same run, the underlying DSSP file is only read once.
    """
    abs_path = os.path.abspath(dssp_path)
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
        lines = []
        header_line_idx = None
    except Exception:
        lines = []
        header_line_idx = None

    _DSSP_FILE_CACHE[abs_path] = (lines, header_line_idx)
    return lines, header_line_idx


def _count_unique_residues_in_pdb(pdb_path: str) -> int:
    abs_path = os.path.abspath(pdb_path)
    cached = _RES_COUNT_CACHE.get(abs_path)
    if cached is not None:
        return cached
    atoms = _get_atoms_for_path(pdb_path)
    count = len({residue_key_from_atom(atom) for atom in atoms})
    _RES_COUNT_CACHE[abs_path] = count
    return count


def _get_residue_seq_index(atoms: List[Atom]) -> Dict[Tuple[str, int, str, str], int]:
    atoms_id = id(atoms)
    if atoms_id in _RES_SEQ_INDEX_CACHE:
        return _RES_SEQ_INDEX_CACHE[atoms_id]
    
    chain_to_residues: Dict[str, List[Tuple[str, int, str, str]]] = {}
    seen_per_chain: Dict[str, set] = {}
    for atom in atoms:
        res_key = residue_key_from_atom(atom)  # (resname, resnum, chain, insertion letter)
        chain = res_key[2]
        if chain not in seen_per_chain:
            seen_per_chain[chain] = set()
            chain_to_residues[chain] = []
        if res_key not in seen_per_chain[chain]:
            seen_per_chain[chain].add(res_key)
            chain_to_residues[chain].append(res_key)
    
    seq_index: Dict[Tuple[str, int, str, str], int] = {}
    for chain, residues in chain_to_residues.items():
        for idx, res_key in enumerate(residues):
            seq_index[res_key] = idx
    
    _RES_SEQ_INDEX_CACHE[atoms_id] = seq_index
    return seq_index

def is_donor(atom: Atom) -> bool:
    key = (atom.name, atom.residue_name)
    return key not in DONOR_EXCLUDED and (
        atom.name in DONORS_ANY or key in DONOR_INFO
    )

def is_acceptor(atom: Atom) -> bool:
    return (atom.name, atom.residue_name) in ACCEPTORS_SPECIFIC or atom.name in ACCEPTORS_ANY

def get_donor_base_atom(
    donor: Atom, 
    atoms: List[Atom],
    backbone_base_cache: Optional[Dict[Tuple[str, int], Atom]] = None
) -> Optional[Atom]:
    """
    Get the base atom for a donor atom (DSSP-style).
    
    The base atom is the atom covalently bonded to the donor:
    - Backbone N → previous residue C (residue_number - 1)
    - Side-chain donors → bonded heavy atom in same residue
    
    Note: This function assumes implicit hydrogen atoms in the PDB structure.
    For PDBs with explicit hydrogens, the base atom selection logic may need
    adjustment, but typically works correctly for heavy-atom-only PDBs.
    
    Args:
        donor: The donor atom
        atoms: List of all atoms
        backbone_base_cache: Optional cache dict mapping (chain, resnum) -> base C atom
                           for backbone N donors. If None, will search each time.
        
    Returns:
        Base atom, or None if not found
    """
    # Build or retrieve atom lookup for this atoms list
    atoms_id = id(atoms)
    if atoms_id not in _ATOM_LOOKUP_CACHE:
        _ATOM_LOOKUP_CACHE[atoms_id] = {
            (atom.chain, atom.residue_number, atom.name): atom for atom in atoms
        }
    atom_lookup = _ATOM_LOOKUP_CACHE[atoms_id]
    
    # Backbone N: base is C from previous residue in sequence order
    if donor.name == "N":
        seq_index = _get_residue_seq_index(atoms)
        donor_res_key = residue_key_from_atom(donor)
        donor_idx = seq_index.get(donor_res_key)
        if donor_idx is None or donor_idx == 0:
            # No previous residue in sequence (true N-terminus or unmapped), so
            # no backbone base atom for angle calculation.
            return None
        
        # Find previous residue in the same chain by walking seq_index keys
        donor_chain = donor.chain
        prev_res_key = None
        for (resname, resnum, chain, inscode), idx in seq_index.items():
            if chain == donor_chain and idx == donor_idx - 1:
                prev_res_key = (resname, resnum, chain, inscode)
                break
        
        if prev_res_key is None:
            return None
        
        prev_resnum = prev_res_key[1]
        key_prev_c = (donor_chain, prev_resnum, "C")
        base = atom_lookup.get(key_prev_c)
        if base is not None:
            return base
        
        # If there are atoms for the previous residue but no C atom, treat as
        # a corrupted backbone and raise; otherwise (true chain break), return None.
        has_prev_residue_atoms = any(
            ch == donor_chain and res == prev_resnum for (ch, res, _name) in atom_lookup.keys()
        )
        if has_prev_residue_atoms:
            raise ValueError(
                f"Backbone donor base atom not found for N in residue "
                f"{donor.residue_name} {donor.residue_number} chain {donor.chain}"
            )
        return None
    
    # Side-chain donors: use known topology from DONOR_INFO
    key = (donor.name, donor.residue_name)
    if key not in DONOR_INFO:
        raise ValueError(
            f"Side-chain donor base atom definition missing for {donor.name} in "
            f"{donor.residue_name} {donor.residue_number} chain {donor.chain}"
        )
    
    base_name = DONOR_INFO[key]
    base_key = (donor.chain, donor.residue_number, base_name)
    base = atom_lookup.get(base_key)
    if base is None:
        raise ValueError(
            f"Side-chain donor base atom {base_name} not found for donor {donor.name} in "
            f"{donor.residue_name} {donor.residue_number} chain {donor.chain}"
        )
    return base


def check_hydrogen_bond(
    donor: Atom, 
    acceptor: Atom, 
    atoms: List[Atom],
    backbone_base_cache: Optional[Dict[Tuple[str, int], Atom]] = None,
    precomputed_base: Optional[Atom] = None,
    precomputed_base_vec: Optional[np.ndarray] = None,
) -> bool:
    """
    Check if a hydrogen bond exists between donor and acceptor (DSSP-style).
    
    Criteria:
    1. D-A distance ≤ 3.2 Å
    2. Angle Base→Donor→Acceptor ≥ 120° (where Base is covalently bonded to donor)
       Angle is measured in degrees (DSSP-style).
    
    For backbone donors, base atom is mandatory.
    For side-chain donors, base atom is preferred but can fallback to distance-only.
    
    Args:
        donor: Donor atom
        acceptor: Acceptor atom
        atoms: List of all atoms (for finding base atom)
        backbone_base_cache: Optional cache for backbone base atoms (for performance)
        
    Returns:
        True if hydrogen bond criteria are met
    """
    # Check distance
    da_dist = distance(donor, acceptor)
    if da_dist > MAX_HBOND_DISTANCE:
        return False
    
    # Get base atom for donor (use precomputed value if provided)
    base = precomputed_base
    if base is None:
        base = get_donor_base_atom(donor, atoms, backbone_base_cache)
    
    # Base atom is mandatory for all donors: geometry must be defined.
    if base is None:
        raise ValueError(
            f"Base atom not found for donor {donor.name} in "
            f"{donor.residue_name} {donor.residue_number} chain {donor.chain}"
        )
    
    # Calculate angle: Base→Donor→Acceptor
    # Vector from donor to base
    if precomputed_base_vec is not None:
        db_vec = precomputed_base_vec
    else:
        db_vec = (base.x - donor.x, base.y - donor.y, base.z - donor.z)
    # Vector from donor to acceptor
    da_vec = (acceptor.x - donor.x, acceptor.y - donor.y, acceptor.z - donor.z)
    
    # Calculate angle Base-Donor-Acceptor (returns degrees, DSSP-style)
    angle = angle_between_vectors(db_vec, da_vec)
    
    return angle >= MIN_HBOND_ANGLE

# CHECK FROM HERE

def get_sasa_weight(atom: Atom, sasa_data: Dict[Tuple[str, int, str, str], SASAEntry]) -> float:
    key = residue_key_from_atom(atom)
    if key not in sasa_data:
        return 0.0

    sasa_entry = sasa_data[key]
    
    if is_backbone_atom(atom.name):
        rel_sasa = sasa_entry.main_chain_rel
    else:
        rel_sasa = sasa_entry.total_side_rel
    
    # rel_sasa is a fraction (0-1). Scale to [0, 100].
    return rel_sasa * 100.0


def get_residue_sasa_weight(
    residue_key: Tuple[str, int, str, str],
    sasa_data: Dict[Tuple[str, int, str, str], SASAEntry],
    use_side_chain: bool = True,
) -> float:
    if residue_key not in sasa_data:
        return 0.0

    sasa_entry = sasa_data[residue_key]
    
    if use_side_chain:
        rel_sasa = sasa_entry.total_side_rel
    else:
        rel_sasa = sasa_entry.main_chain_rel
    
    # rel_sasa is a fraction (0-1). Scale to [0, 100].
    return rel_sasa * 100.0


def net_charge_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float
) -> Optional[float]:
    """
    Compute protein net charge at a given pH using PropKA per-residue pKa values.

    Uses Henderson-Hasselbalch: ASP/GLU contribute -1/(1+10^(pKa-pH));
    LYS/ARG contribute +1/(1+10^(pH-pKa)). Only residues present in pka_data
    (PropKA output) are included.

    Args:
        pka_data: Dict mapping (residue_name, residue_number, chain, insertion_code) -> pKa (float).
        pH: pH at which to compute net charge.

    Returns:
        Net charge (float), or None if pka_data is empty.
    """
    if not pka_data:
        return None
    net = 0.0
    for key, pka in pka_data.items():
        res_name = key[0]
        if res_name in ("ASP", "GLU"):
            # Fraction deprotonated (negative charge)
            net -= 1.0 / (1.0 + np.power(10.0, pka - pH))
        elif res_name in ("LYS", "ARG", "HIS"):
            # Fraction protonated (positive charge)
            net += 1.0 / (1.0 + np.power(10.0, pH - pka))

    # Add generic N- and C-terminus contributions per polypeptide chain.
    # Each distinct chain in pka_data is assumed to have one N-terminus and one C-terminus.
    chains = {key[2] for key in pka_data.keys()}
    n_chains = len(chains)
    if n_chains > 0:
        # N-terminus: positive charge contribution
        net += n_chains * (1.0 / (1.0 + np.power(10.0, pH - NTERM_PKA)))
        # C-terminus: negative charge contribution
        net -= n_chains * (1.0 / (1.0 + np.power(10.0, CTERM_PKA - pH)))

    return float(net)


def pi_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float]
) -> Optional[float]:
    """
    Compute protein isoelectric point (pI) using PropKA per-residue pKa values.

    Finds the pH at which net charge is zero by minimizing (net_charge(pH))^2
    over pH in (0, 14), using scipy.optimize.minimize_scalar.

    Args:
        pka_data: Dict mapping (residue_name, residue_number, chain, insertion_code) -> pKa (float).

    Returns:
        pI (float), or None if pka_data is empty or optimization failed.
    """
    if not pka_data:
        return None

    def objective(pH: float) -> float:
        q = net_charge_from_pka(pka_data, pH)
        return (q * q) if q is not None else float("nan")

    try:
        result = optimize.minimize_scalar(objective, bounds=(0.0, 14.0), method="bounded")
        if result.success:
            return float(result.x)
        return None
    except Exception:
        return None


# ============================================================================
# SCM (Surface Charge Metric) Score
# ============================================================================
# Main-chain atom names (side-chain = not in this set); match reference PROPERMAB.
SCM_MAIN_CHAIN_ATOMS = frozenset({"CA", "HA", "N", "C", "O", "HN", "H"})


def _residue_fractional_charge_at_pH(
    residue_name: str,
    pka_value: Optional[float],
    pH: float
) -> float:
    """Fractional charge of a residue at pH (for SCM: ASP/GLU negative, LYS/ARG positive)."""
    if pka_value is None:
        return 0.0
    if residue_name in ("ASP", "GLU"):
        # Acidic: negative fractional charge
        return -1.0 / (1.0 + np.power(10.0, pka_value - pH))
    if residue_name in ("LYS", "ARG", "HIS"):
        # Basic: positive fractional charge
        return 1.0 / (1.0 + np.power(10.0, pH - pka_value))
    return 0.0


def scm_score_from_pka(
    pdb_path: str,
    sasa_path: str,
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    d_cutoff: float = 10.0,
    sasa_cutoff: float = 0.25,
) -> Optional[float]:
    """
    SCM (Surface Charge Metric) score using PropKA charges and per-residue SASA.

    SCM_atom,i = sum of partial charges of side-chain atoms that belong to an
    exposed residue and are within distance R = d_cutoff (default 10 Å) of atom i.
    SCM score = | Σ over all atoms i of ( SCM_atom,i × H(-SCM_atom,i) ) |
    i.e. absolute value of the sum of negative SCM_atom,i (Heaviside H(-x) = 1 when x ≤ 0).

    Uses residue-level SASA (exposed = total_side_rel > sasa_cutoff, default 10%).
    Residue fractional charges at pH from PropKA are distributed equally over
    side-chain atoms of that residue.

    Args:
        pdb_path: Path to PDB file.
        sasa_path: Path to SASA file (per-residue total_side_rel, etc.).
        pka_data: PropKA per-residue pKa dict (residue_key_4tuple -> pKa).
        pH: pH for charge calculation.
        d_cutoff: Distance cutoff in Å (default 10).
        sasa_cutoff: Residue exposure threshold (fraction, default 0.25 = 25%).

    Returns:
        SCM score (float), or None if missing data or computation fails.
    """
    from utils.parsers import parse_sasa

    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        return None

    try:
        sasa_data = parse_sasa(sasa_path)
    except Exception:
        return None

    def _get_pka(key_4: Tuple) -> Optional[float]:
        return pka_data.get(key_4)

    try:
        n = len(atoms)
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        
        # Side-chain mask: atom name not in main-chain list
        is_sidechain = np.array([a.name.strip() not in SCM_MAIN_CHAIN_ATOMS for a in atoms], dtype=bool)
        
        key4_list: List[Tuple[str, int, str, str]] = [
            residue_key_from_atom(a) for a in atoms
        ]

        # Residue exposed: total_side_rel > sasa_cutoff
        residue_exposed = np.zeros(n, dtype=bool)
        for i, key4 in enumerate(key4_list):
            entry = sasa_data.get(key4)
            if entry is not None and getattr(entry, "total_side_rel", 0) is not None:
                residue_exposed[i] = entry.total_side_rel > sasa_cutoff
        
        # Per-residue fractional charge at pH and side-chain atom counts
        residue_charge: Dict[Tuple[str, int, str, str], float] = {}
        residue_sidechain_count: Dict[Tuple[str, int, str, str], int] = {}
        for i, (a, key4) in enumerate(zip(atoms, key4_list)):
            if key4 not in residue_charge:
                pka_val = _get_pka(key4)
                residue_charge[key4] = _residue_fractional_charge_at_pH(a.residue_name, pka_val, pH)
            if is_sidechain[i]:
                residue_sidechain_count[key4] = residue_sidechain_count.get(key4, 0) + 1
        
        # Per-atom charge: distribute residue charge equally over side-chain atoms
        atom_charge = np.zeros(n, dtype=np.float64)
        for i, key4 in enumerate(key4_list):
            if is_sidechain[i]:
                count = residue_sidechain_count.get(key4, 1)
                atom_charge[i] = residue_charge.get(key4, 0.0) / max(1, count)

        # SCM_atom,i = sum of charge(j) for j side-chain, exposed, within d_cutoff of i.
        # Only exposed residues are considered as centers i for the SCM score.
        # Use a precomputed validity mask and pairwise neighbor enumeration for efficiency.
        tree = cKDTree(coords)
        scm_atom = np.zeros(n, dtype=np.float64)
        valid = is_sidechain & residue_exposed
        # query_pairs returns each unordered pair (i, j) once; accumulate contributions
        # symmetrically while respecting exposed-center and side-chain/exposed-source rules.
        for i, j in tree.query_pairs(d_cutoff):
            if i == j:
                continue
            if residue_exposed[i] and valid[j]:
                scm_atom[i] += atom_charge[j]
            if residue_exposed[j] and valid[i]:
                scm_atom[j] += atom_charge[i]

        # SCM score = | sum of SCM_atom,i for i where SCM_atom,i < 0 |
        neg_sum = np.sum(scm_atom[scm_atom < 0])
        return float(np.abs(neg_sum))
    except Exception:
        return None


def sum_total_side_rel_within_cutoff(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]],
    cutoff: float = 5.0,
) -> Dict[Tuple[str, int, str, str], float]:
    """
    For each residue, sum total_side_rel (relative side-chain ASA, %) of all residues
    within cutoff Å (Cα–Cα distance). Includes the residue itself in the sum.
    """
    residue_keys: List[Tuple[str, int, str, str]] = []
    residue_coords: List[Tuple[float, float, float]] = []
    residue_total_side_rel: List[float] = []

    seen = set()
    atoms_by_res: Dict[Tuple, List[Atom]] = {}
    ca_by_res: Dict[Tuple, Tuple[float, float, float]] = {}
    for atom in pdb_atoms:
        key4 = residue_key_from_atom(atom)
        if key4 not in seen:
            seen.add(key4)
            residue_keys.append(key4)
            atoms_by_res[key4] = []
        atoms_by_res[key4].append(atom)
        if atom.name.strip() == "CA":
            ca_by_res[key4] = (atom.x, atom.y, atom.z)

    for key4 in residue_keys:
        atoms_res = atoms_by_res[key4]
        sasa_entry = sasa_output_data.get(key4) or {}
        val = sasa_entry.get("total_side_rel")
        total_side_rel = float(val) if val is not None else 0.0
        residue_total_side_rel.append(total_side_rel)

        # Cα position if present, otherwise centroid
        ca_coord = ca_by_res.get(key4)
        if ca_coord is not None:
            residue_coords.append(ca_coord)
        else:
            cx = sum(a.x for a in atoms_res) / len(atoms_res)
            cy = sum(a.y for a in atoms_res) / len(atoms_res)
            cz = sum(a.z for a in atoms_res) / len(atoms_res)
            residue_coords.append((cx, cy, cz))

    coords = np.array(residue_coords, dtype=np.float64)
    total_side_rel_arr = np.array(residue_total_side_rel, dtype=np.float64)
    tree = cKDTree(coords)
    out: Dict[Tuple[str, int, str, str], float] = {}
    for i, key4 in enumerate(residue_keys):
        indices = tree.query_ball_point(coords[i], cutoff)
        out[key4] = float(np.sum(total_side_rel_arr[indices]))
    return out


# ============================================================================
# Salt-Bridge Detection Functions
# ============================================================================

def is_positively_charged(atom: Atom) -> bool:
    """
    Check if an atom is positively charged (can form salt bridges).
    
    Eligible atoms:
    - ARG: NH1, NH2
    - LYS: NZ
    - HIS: excluded by default (ambiguous protonation state)
    
    Args:
        atom: The atom to check
        
    Returns:
        True if the atom is positively charged
    """
    return (atom.name, atom.residue_name) in POSITIVE_ATOMS


def is_negatively_charged(atom: Atom) -> bool:
    """
    Check if an atom is negatively charged (can form salt bridges).
    
    Eligible atoms:
    - ASP: OD1, OD2
    - GLU: OE1, OE2
    
    Args:
        atom: The atom to check
        
    Returns:
        True if the atom is negatively charged
    """
    return (atom.name, atom.residue_name) in NEGATIVE_ATOMS


def _load_pka_for_atoms(
    pdb_path: str,
    atoms: List[Atom],
    pka_path: Optional[str],
) -> Dict[Tuple[str, int, str, str], float]:
    """
    Helper to load PropKA pKa data for a given structure and atom list,
    with the same auto-detection semantics used by the public salt-bridge
    functions.
    """
    pka_data: Dict[Tuple[str, int, str, str], float] = {}

    if pka_path is None:
        # Try to auto-detect pKa file path
        pka_path = get_pka_file_path(pdb_path)

    if pka_path and os.path.exists(pka_path):
        pka_data = parse_pka(pka_path, atoms)

    return pka_data


def _find_salt_bridge_contacts(
    atoms: List[Atom],
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
) -> Dict[
    Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
    List[Tuple[Atom, Atom]],
]:
    """
    Core geometry- and charge-based salt-bridge detector.

    Finds all atom-level contacts between positively and negatively charged
    residues that satisfy the distance cutoff and charge-state rules. The
    result groups contacts by residue pair; callers can either attach SASA
    weights or simply count unique pairs.
    """
    # Identify positively and negatively charged atoms
    positive_atoms = [atom for atom in atoms if is_positively_charged(atom)]
    negative_atoms = [atom for atom in atoms if is_negatively_charged(atom)]

    if len(negative_atoms) == 0 or len(positive_atoms) == 0:
        return {}

    # Precompute which residues are charged at the given pH, using the same
    # Henderson–Hasselbalch rules as net_charge_from_pka and the same
    # missing-pKa conventions (ASP/GLU/LYS/ARG charged, HIS uncharged).
    charged_residues: Dict[Tuple[str, int, str, str], bool] = {}
    for atom in atoms:
        key4 = residue_key_from_atom(atom)
        if key4 in charged_residues:
            continue
        pka_val = pka_data.get(key4)
        res_name = atom.residue_name
        if pka_val is None:
            charged_residues[key4] = res_name in ("ASP", "GLU", "LYS", "ARG")
        elif res_name in ("ASP", "GLU"):
            charged_residues[key4] = pka_val < pH
        elif res_name in ("LYS", "ARG", "HIS"):
            charged_residues[key4] = pka_val > pH
        else:
            charged_residues[key4] = False

    # Build KD-tree for negative atoms for efficient neighbor search
    negative_coords = np.array([[a.x, a.y, a.z] for a in negative_atoms])
    negative_tree = cKDTree(negative_coords)

    # Group contacts per residue pair
    contacts_by_pair: Dict[
        Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
        List[Tuple[Atom, Atom]],
    ] = {}

    for pos_atom in positive_atoms:
        pos_coord = (pos_atom.x, pos_atom.y, pos_atom.z)
        pos_key = residue_key_from_atom(pos_atom)

        # Check if positive residue is charged at given pH
        if not charged_residues.get(pos_key, False):
            continue  # Skip if not charged

        # Query all negative atoms within MAX_SALT_BRIDGE_DISTANCE
        indices = negative_tree.query_ball_point(pos_coord, MAX_SALT_BRIDGE_DISTANCE)

        for idx in indices:
            neg_atom = negative_atoms[idx]
            neg_key = residue_key_from_atom(neg_atom)

            # Check if negative residue is charged at given pH
            if not charged_residues.get(neg_key, False):
                continue  # Skip if not charged

            # Exclude contacts within the same residue (same 4-tuple key)
            if pos_key == neg_key:
                continue

            pair = (pos_key, neg_key)
            contacts_by_pair.setdefault(pair, []).append((pos_atom, neg_atom))

    return contacts_by_pair


def detect_salt_bridges(
    pdb_path: str,
    sasa_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> Dict[Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]], Tuple[float, float]]:
    """
    Detect salt bridges between residue pairs with SASA weighting.

    A salt bridge is formed when any charged atom from a positively charged residue
    (ARG, LYS) is within MAX_SALT_BRIDGE_DISTANCE (4.0 Å) of any charged atom from
    a negatively charged residue (ASP, GLU).

    Residues are only considered charged if their pKa values (from pKa file) indicate
    they are charged at the specified pH:
    - ASP/GLU: charged when pKa < pH
    - LYS/ARG: charged when pKa > pH

    Each residue pair counts as only one salt bridge, regardless of how many
    atom-atom contacts exist between them. If multiple contacts exist, the maximum
    weight from any contact is used.

    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)

    Returns:
        Dictionary mapping ((pos_res_name, pos_res_num, pos_chain, pos_ins),
                            (neg_res_name, neg_res_num, neg_chain, neg_ins))
        -> (pos_weight, neg_weight)
        where weights are the maximum SASA-based weights from any atom-atom contact in the pair.
        Keys are ordered such that the positive residue comes first.
    """
    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        return {}

    sasa_data = parse_sasa(sasa_path)
    pka_data = _load_pka_for_atoms(pdb_path, atoms, pka_path)

    # Find all geometrically and electrostatically valid salt-bridge contacts once.
    contacts_by_pair = _find_salt_bridge_contacts(atoms, pka_data, pH)
    if not contacts_by_pair:
        return {}

    # Cache SASA-derived weights per atom (by id) to avoid recomputing inside contact loops.
    atom_sasa_weight: Dict[int, float] = {}

    def get_cached_sasa_weight(atom: Atom) -> float:
        key = id(atom)
        cached = atom_sasa_weight.get(key)
        if cached is not None:
            return cached
        w = get_sasa_weight(atom, sasa_data)
        atom_sasa_weight[key] = w
        return w

    # For each residue pair, keep the maximum SASA-based weight observed over all
    # atom-atom contacts, matching the original behavior.
    salt_bridge_pairs: Dict[
        Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
        Tuple[float, float],
    ] = {}

    for pair, contacts in contacts_by_pair.items():
        pos_key, neg_key = pair
        max_pos_weight = 0.0
        max_neg_weight = 0.0

        for pos_atom, neg_atom in contacts:
            pos_weight = get_cached_sasa_weight(pos_atom)
            neg_weight = get_cached_sasa_weight(neg_atom)
            if pos_weight > max_pos_weight:
                max_pos_weight = pos_weight
            if neg_weight > max_neg_weight:
                max_neg_weight = neg_weight

        salt_bridge_pairs[pair] = (max_pos_weight, max_neg_weight)

    return dict(salt_bridge_pairs)


def calculate_salt_bridge_density(
    pdb_path: str,
    sasa_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> Tuple[Dict[Tuple[str, int, str, str], float], Dict[Tuple[str, int, str, str], bool]]:
    """
    Calculate salt bridge density per residue, using SASA-based weights.
    
    Number of salt bridges per residue, weighted by scaled relative SASA:
    - Per-contact weight = 100 * REL, where REL is total-side REL for side-chain
      atoms or main-chain REL for backbone atoms.
    
    SASA-derived weights are summed per residue and a square-root transform is then
    applied to the final per-residue weights, introducing diminishing returns for
    residues with many salt bridges.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)
        
    Returns:
        Tuple of:
        1. Dictionary mapping (residue_name, residue_number, chain) -> weighted salt bridge density
        2. Dictionary mapping (residue_name, residue_number, chain) -> bool indicating
           if residue has any inter-chain contacts
    """
    salt_bridges = detect_salt_bridges(pdb_path, sasa_path, pka_path, pH)
    
    # Aggregate by residue (raw weights, no sqrt here; callers can apply their
    # own transforms). Only residues that participate in at least one salt
    # bridge appear in the output dicts.
    residue_weights = defaultdict(float)
    residue_inter_chain: Dict[Tuple[str, int, str, str], bool] = {}
    
    for (pos_key, neg_key), (pos_weight, neg_weight) in salt_bridges.items():
        # Check if this is an inter-chain contact
        is_inter_chain = (pos_key[2] != neg_key[2])
        
        # Each residue gets the weight based on its own atom's location
        # Add weights (no explicit cap; square-root transform at the end)
        residue_weights[pos_key] += pos_weight
        residue_weights[neg_key] += neg_weight
        
        # Mark residues as having inter-chain contacts if applicable
        if is_inter_chain:
            residue_inter_chain[pos_key] = True
            residue_inter_chain[neg_key] = True
    
    return dict(residue_weights), residue_inter_chain


# Type alias for salt bridge result: (pos_key, neg_key) -> (pos_weight, neg_weight)
_SaltBridgesDict = Dict[
    Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
    Tuple[float, float],
]


def calculate_salt_bridge_density_average(
    pdb_path: str,
    sasa_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.4,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    weighted: bool = True,
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    salt_bridges: Optional[_SaltBridgesDict] = None,
) -> float:
    """
    Calculate average salt bridge density with configurable numerator and denominator.

    A single pass over salt bridge data computes both SASA-weighted sums and raw
    counts per residue, so weighted and unweighted averages are efficient.

    To avoid recomputing salt bridges when calling this function multiple times
    (e.g. for different residue sets or weighted vs unweighted), compute once
    and pass the result: salt_bridges = detect_salt_bridges(...), then pass
    salt_bridges=salt_bridges into each call.

    Examples:
        - Average (SASA-weighted) salt bridge density of CDR residues over all residues:
          calculate_salt_bridge_density_average(pdb, sasa, residues_for_density=cdr_residues)
        - Average unweighted salt bridge count of CDR residues over CDR residues only:
          calculate_salt_bridge_density_average(pdb, sasa, residues_for_density=cdr_residues,
                                                weighted=False, residues_for_average=cdr_residues)
        - Multiple averages without recomputing salt bridges:
          sb = detect_salt_bridges(pdb, sasa, pka_path, pH)
          avg1 = calculate_salt_bridge_density_average(pdb, sasa, ..., salt_bridges=sb, ...)
          avg2 = calculate_salt_bridge_density_average(pdb, sasa, ..., salt_bridges=sb, ...)

    Args:
        pdb_path: Path to PDB structure file (used when salt_bridges is None, and for
            denominator when residues_for_average is None).
        sasa_path: Path to SASA file (used only when salt_bridges is None).
        pka_path: Optional path to pKa file (used only when salt_bridges is None).
        pH: pH value for charge state (used only when salt_bridges is None).
        residues_for_density: If provided, only these residues contribute to the numerator.
            Residue keys are 4-tuples (residue_name, residue_number, chain, insertion_code).
            If None, all residues that have salt bridge density are included.
        weighted: If True, use SASA-weighted density (with sqrt transform). If False,
            use raw count of salt bridges per residue.
        residues_for_average: Residues over which to average (denominator). If provided,
            denominator is len(set(residues_for_average)). If None, denominator is total
            number of unique residues in the PDB.
        salt_bridges: Optional pre-computed result from detect_salt_bridges(). When
            provided, salt bridge detection is skipped (pdb_path is still used for
            denominator when residues_for_average is None).

    Returns:
        Sum of (density or count) over residues_for_density, divided by denominator.
        Returns 0.0 if denominator is 0 or there are no salt bridges.
    """
    if salt_bridges is None:
        salt_bridges = detect_salt_bridges(pdb_path, sasa_path, pka_path, pH)
    if not salt_bridges:
        return 0.0

    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    residue_counts: Dict[Tuple[str, int, str, str], int] = defaultdict(int)
    for (pos_key, neg_key), (pos_w, neg_w) in salt_bridges.items():
        residue_weights_raw[pos_key] += pos_w
        residue_weights_raw[neg_key] += neg_w
        residue_counts[pos_key] += 1
        residue_counts[neg_key] += 1

    density_set: Optional[Set[Tuple[str, int, str, str]]] = (
        set(residues_for_density) if residues_for_density is not None else None
    )
    if residues_for_average is not None:
        average_over_set = set(residues_for_average)
        denom = len(average_over_set)
    else:
        denom = _count_unique_residues_in_pdb(pdb_path)

    if denom == 0:
        return 0.0

    keys_to_sum = (
        residue_weights_raw.keys()
        if density_set is None
        else (set(residue_weights_raw.keys()) & density_set)
    )
    if weighted:
        total = sum(math.sqrt(residue_weights_raw[k]) for k in keys_to_sum)
    else:
        total = sum(residue_counts[k] for k in keys_to_sum)

    return total / float(denom)


# ============================================================================
# Sequence motif count
# ============================================================================

def _parse_motif(motif: str) -> List[str]:
    """
    Parse a motif string into a list of 3-letter residue codes.

    Examples: "Asp-Gly" -> ["ASP", "GLY"], "Trp-Pro-Trp" -> ["TRP", "PRO", "TRP"],
    "Lys" -> ["LYS"]. Accepts 1-letter (e.g. K) or 3-letter (e.g. Lys, LYS) codes.
    """
    parts = [p.strip() for p in motif.split("-") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid motif: {motif!r}")
    result: List[str] = []
    for p in parts:
        u = p.upper()
        if len(u) == 1:
            three = AA_1_TO_3.get(u)
            if three is None:
                raise ValueError(f"Unknown 1-letter code in motif: {p!r}")
            result.append(three)
        elif len(u) == 3:
            result.append(u)
        else:
            raise ValueError(f"Residue in motif must be 1- or 3-letter: {p!r}")
    return result


def _get_sequence_per_chain(pdb_path: str) -> Dict[str, List[str]]:
    """
    Return ordered residue sequence (3-letter codes) per chain.
    Residues are sorted by residue number then insertion code.
    """
    abs_path = os.path.abspath(pdb_path)
    cached = _PDB_SEQUENCE_CACHE.get(abs_path)
    if cached is not None:
        return cached

    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        _PDB_SEQUENCE_CACHE[abs_path] = {}
        return {}
    chain_to_keys: Dict[str, List[Tuple[str, int, str, str]]] = {}
    for atom in atoms:
        key = residue_key_from_atom(atom)
        chain = key[2]
        keys = chain_to_keys.setdefault(chain, [])
        # Avoid duplicates while preserving insertion-order per chain
        if not keys or keys[-1] != key:
            keys.append(key)
    out: Dict[str, List[str]] = {}
    for chain, keys in chain_to_keys.items():
        sorted_keys = sorted(keys, key=lambda k: (k[1], k[3]))
        out[chain] = [k[0] for k in sorted_keys]

    _PDB_SEQUENCE_CACHE[abs_path] = out
    return out


def _count_motif_in_sequence(sequence: List[str], motif_list: List[str]) -> int:
    """Count non-overlapping occurrences of motif_list in sequence (sliding window)."""
    n = len(motif_list)
    if n == 0 or len(sequence) < n:
        return 0
    count = 0
    i = 0
    while i <= len(sequence) - n:
        if sequence[i : i + n] == motif_list:
            count += 1
            i += n
        else:
            i += 1
    return count


def sequence_motif_count(pdb_path: str, motif: str) -> int:
    """
    Count how many times a sequence motif appears in the structure's chains.

    The motif is given as a string with residues separated by "-", e.g. "Asp-Gly",
    "Trp-Pro-Trp", or "Lys". Each residue can be 1-letter (K) or 3-letter (Lys, LYS).
    Counts are summed over all chains; each chain is treated as a separate sequence.
    Overlapping occurrences are not double-counted (non-overlapping match).

    Args:
        pdb_path: Path to PDB structure file
        motif: Motif string, e.g. "Asp-Gly", "Trp-Pro-Trp", "Lys"

    Returns:
        Total number of times the motif appears across all chains

    Raises:
        ValueError: If motif is empty or contains an unknown residue code
    """
    motif_list = _parse_motif(motif)
    chains_to_sequence = _get_sequence_per_chain(pdb_path)
    total = 0
    for sequence in chains_to_sequence.values():
        total += _count_motif_in_sequence(sequence, motif_list)
    return total


# ============================================================================
# Aromatic Residue Detection Functions (Phe/Tyr/Trp)
# ============================================================================

def is_aromatic_residue(residue_name: str) -> bool:
    """
    Check if a residue is aromatic (Phe, Tyr, or Trp).
    
    Args:
        residue_name: 3-letter residue code
        
    Returns:
        True if the residue is aromatic
    """
    return residue_name in AROMATIC_RESIDUES


def get_residue_keys_by_type(
    pdb_path: str,
    residue_types: Iterable[str],
) -> Set[Tuple[str, int, str, str]]:
    """
    Return the set of residue keys in the structure whose residue name is in residue_types.

    residue_types: e.g. AROMATIC_RESIDUES, HYDROPHOBIC_RESIDUES, POLAR_RESIDUES, CHARGED_RESIDUES.
    """
    types_set = frozenset(residue_types)
    abs_path = os.path.abspath(pdb_path)
    # Cache unique residue keys per PDB for reuse across many descriptors.
    residue_keys = _PDB_RESIDUE_KEYS_CACHE.get(abs_path)
    if residue_keys is None:
        atoms = _get_atoms_for_path(pdb_path)
        if not atoms:
            _PDB_RESIDUE_KEYS_CACHE[abs_path] = set()
            return set()
        seen: Set[Tuple[str, int, str, str]] = set()
        residue_keys = set()
        for atom in atoms:
            key4 = residue_key_from_atom(atom)
            if key4 not in seen:
                seen.add(key4)
                residue_keys.add(key4)
        _PDB_RESIDUE_KEYS_CACHE[abs_path] = residue_keys

    return {k for k in residue_keys if k[0] in types_set}


def get_aromatic_residue_keys(pdb_path: str) -> Set[Tuple[str, int, str, str]]:
    """Residue keys for all Phe, Tyr, and Trp in the structure."""
    return get_residue_keys_by_type(pdb_path, AROMATIC_RESIDUES)


def get_hydrophobic_residue_keys(pdb_path: str) -> Set[Tuple[str, int, str, str]]:
    """Residue keys for all hydrophobic residues (ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, GLY)."""
    return get_residue_keys_by_type(pdb_path, HYDROPHOBIC_RESIDUES)


def get_polar_residue_keys(pdb_path: str) -> Set[Tuple[str, int, str, str]]:
    """Residue keys for all polar residues (SER, THR, ASN, GLN, TYR, CYS, GLU, ASP, LYS, ARG, HIS)."""
    return get_residue_keys_by_type(pdb_path, POLAR_RESIDUES)


def get_charged_residue_keys(pdb_path: str) -> Set[Tuple[str, int, str, str]]:
    """Residue keys for all charged residue types (ASP, GLU, LYS, ARG, HIS)."""
    return get_residue_keys_by_type(pdb_path, CHARGED_RESIDUES)


def get_negative_charged_residue_keys(pdb_path: str) -> Set[Tuple[str, int, str, str]]:
    """Residue keys for negative charged types (ASP, GLU)."""
    return get_residue_keys_by_type(pdb_path, NEGATIVE_CHARGED_RESIDUES)


def get_positive_charged_residue_keys(pdb_path: str) -> Set[Tuple[str, int, str, str]]:
    """Residue keys for positive charged types (LYS, ARG, HIS)."""
    return get_residue_keys_by_type(pdb_path, POSITIVE_CHARGED_RESIDUES)


def calculate_aromatic_density(
    pdb_path: str,
    sasa_path: str,
) -> Dict[Tuple[str, int, str, str], float]:
    """
    Calculate aromatic residue density per residue, weighted by SASA.
    
    For each Phe, Tyr, or Trp residue, counts it as 1 * scaled_SASA.
    Weight = 100 * total-side REL (aromatic residues are side-chain).
    
    SASA-derived weights are combined per residue and a square-root transform is
    applied to the final per-residue weights to keep the metric bounded while
    preserving relative differences.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> weighted aromatic density.
        Only includes Phe, Tyr, and Trp residues.
    """
    # Parse files
    sasa_data = parse_sasa(sasa_path)
    
    # Use precomputed aromatic residue keys and compute weights directly.
    aromatic_keys = get_aromatic_residue_keys(pdb_path)
    aromatic_weights: Dict[Tuple[str, int, str, str], float] = {}
    for key4 in aromatic_keys:
        weight = get_residue_sasa_weight(key4, sasa_data, use_side_chain=True)
        aromatic_weights[key4] = weight
    
    # Apply square-root transform to the final per-residue weights in place.
    for key in list(aromatic_weights.keys()):
        aromatic_weights[key] = math.sqrt(aromatic_weights[key])
    
    return aromatic_weights


# Type alias for residue category density cache: residue_key -> raw SASA weight (0-100)
_ResidueDensityRawDict = Dict[Tuple[str, int, str, str], float]


def compute_residue_density_raw(
    pdb_path: str,
    sasa_path: str,
) -> _ResidueDensityRawDict:
    """
    Compute per-residue raw SASA weight for all residues in the structure. No sqrt.

    Uses side-chain relative SASA. Use this to cache density data when calling
    calculate_residue_category_density_average multiple times with different
    residue categories or weighted vs unweighted.
    """
    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        return {}
    sasa_data = parse_sasa(sasa_path)
    residue_keys = {residue_key_from_atom(a) for a in atoms}

    weights: _ResidueDensityRawDict = {}
    for key4 in residue_keys:
        weight = get_residue_sasa_weight(key4, sasa_data, use_side_chain=True)
        weights[key4] = weight
    return weights


def calculate_residue_category_density_average(
    pdb_path: str,
    sasa_path: str,
    *,
    residue_category: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    weighted: bool = True,
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    density_raw: Optional[_ResidueDensityRawDict] = None,
) -> float:
    """
    Calculate average density for a category of residues with configurable denominator.

    The category is any list/set of residue keys (e.g. polar, aromatic, inter-chain,
    CDR, or "exposed only") determined elsewhere and passed in. Density can be
    SASA-weighted or raw count; the average can be over total residues or over
    the category itself.

    Examples:
        - Density of polar residues (only exposed), averaged over total residue count:
          calculate_residue_category_density_average(pdb, sasa, residue_category=polar_exposed_keys)
        - Density of aromatic inter-chain residues, averaged over their count:
          calculate_residue_category_density_average(pdb, sasa,
              residue_category=aromatic_inter_chain_keys,
              weighted=False, residues_for_average=aromatic_inter_chain_keys)
        - Multiple averages without recomputing:
          raw = compute_residue_density_raw(pdb, sasa)
          avg1 = calculate_residue_category_density_average(pdb, sasa, density_raw=raw, ...)
          avg2 = calculate_residue_category_density_average(pdb, sasa, density_raw=raw, ...)

    Args:
        pdb_path: Path to PDB structure file (used when density_raw is None, and for
            denominator when residues_for_average is None).
        sasa_path: Path to SASA file (used only when density_raw is None).
        residue_category: Residue keys to include in the numerator (the "category").
            Residue keys are 4-tuples (residue_name, residue_number, chain, insertion_code).
            If None, all residues present in density_raw are included.
        weighted: If True, use SASA-weighted density (with sqrt transform). If False,
            use raw count (1 per residue in the category).
        residues_for_average: Residues over which to average (denominator). If provided,
            denominator is len(set(residues_for_average)). If None, denominator is total
            number of unique residues in the PDB.
        density_raw: Optional pre-computed result from compute_residue_density_raw().
            When provided, PDB/SASA are not loaded again (pdb_path still used for
            denominator when residues_for_average is None).

    Returns:
        Sum of (density or count) over residue_category, divided by denominator.
        Returns 0.0 if denominator is 0 or no residues in category.
    """
    if density_raw is None:
        density_raw = compute_residue_density_raw(pdb_path, sasa_path)
    if not density_raw:
        return 0.0

    category_set: Optional[Set[Tuple[str, int, str, str]]] = (
        set(residue_category) if residue_category is not None else None
    )
    if residues_for_average is not None:
        denom = len(set(residues_for_average))
    else:
        denom = _count_unique_residues_in_pdb(pdb_path)

    if denom == 0:
        return 0.0

    keys_to_sum = (
        density_raw.keys()
        if category_set is None
        else [k for k in density_raw.keys() if k in category_set]
    )
    if weighted:
        total = sum(math.sqrt(density_raw[k]) for k in keys_to_sum)
    else:
        total = float(len(list(keys_to_sum)))

    return total / float(denom)


# ============================================================================
# Weighted Contact Number (WCN) Functions
# ============================================================================

def calculate_weighted_contact_number(
    pdb_path: str,
    *,
    residue_category: Optional[Iterable[Tuple[str, int, str, str]]] = None,
) -> Dict[Tuple[str, int, str, str], float]:
    """
    Calculate Weighted Contact Number (WCN) per residue (distance-weighted: 1/r²).

    WCN_i = Σ(j≠i) 1/(r_ij²), where r_ij is Cα–Cα distance. Always computed over
    all residues in the structure; if residue_category is provided, only those
    keys are returned (e.g. polar, aromatic).

    Args:
        pdb_path: Path to PDB structure file
        residue_category: If provided, return WCN only for these residue keys.

    Returns:
        Dictionary mapping residue_key -> WCN value (float)
    """
    atoms = _get_atoms_for_path(pdb_path)
    ca_atoms = [atom for atom in atoms if atom.name == "CA"]
    if len(ca_atoms) == 0:
        return {}

    ca_dict: Dict[Tuple[str, int, str, str], Atom] = {}
    for atom in ca_atoms:
        residue_key = residue_key_from_atom(atom)
        # Keep the first Cα encountered for each residue key to avoid
        # overwriting with alternative locations if present.
        if residue_key not in ca_dict:
            ca_dict[residue_key] = atom

    residue_keys = list(ca_dict.keys())
    ca_coords = np.array([
        [ca_dict[key].x, ca_dict[key].y, ca_dict[key].z] for key in residue_keys
    ])
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    with np.errstate(divide="ignore"):
        inv_dist_sq = 1.0 / dist_sq
    wcn_array = np.sum(inv_dist_sq, axis=1)

    wcn_values: Dict[Tuple[str, int, str, str], float] = {}
    for key, wcn in zip(residue_keys, wcn_array):
        wcn_values[key] = float(wcn)

    if residue_category is not None:
        category_set = set(residue_category)
        wcn_values = {k: v for k, v in wcn_values.items() if k in category_set}
    return wcn_values


def calculate_weighted_contact_number_average(
    pdb_path: str,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    wcn_values: Optional[Dict[Tuple[str, int, str, str], float]] = None,
) -> float:
    """
    Average Weighted Contact Number (WCN) with configurable numerator and denominator.

    Same interface as other density averages: residues_for_density (which residues
    to sum), residues_for_average (denominator), and optional wcn_values (cache from
    calculate_weighted_contact_number to avoid recomputing).

    Examples:
        - Average WCN of polar residues over all residues:
          calculate_weighted_contact_number_average(pdb, residues_for_density=get_polar_residue_keys(pdb))
        - Average WCN of CDR residues over CDR only:
          calculate_weighted_contact_number_average(pdb, residues_for_density=cdr, residues_for_average=cdr)
        - Multiple averages with one WCN computation:
          wcn = calculate_weighted_contact_number(pdb)
          a = calculate_weighted_contact_number_average(pdb, wcn_values=wcn, residues_for_density=polar_keys)
          b = calculate_weighted_contact_number_average(pdb, wcn_values=wcn, residues_for_density=cdr_keys)
    """
    if wcn_values is None:
        wcn_values = calculate_weighted_contact_number(pdb_path)
    if not wcn_values:
        return 0.0

    density_set: Optional[Set[Tuple[str, int, str, str]]] = (
        set(residues_for_density) if residues_for_density is not None else None
    )
    if residues_for_average is not None:
        denom = len(set(residues_for_average))
    else:
        denom = _count_unique_residues_in_pdb(pdb_path)
    if denom == 0:
        return 0.0

    keys_to_sum = (
        wcn_values.keys()
        if density_set is None
        else [k for k in wcn_values.keys() if k in density_set]
    )
    total = sum(wcn_values[k] for k in keys_to_sum)
    return total / float(denom)


# ============================================================================
# DSSP File Parsing Functions (RESIDUE = original PDB residue number, 2nd column; may have letter)
# ============================================================================

def parse_dssp(
    dssp_path: str,
    pdb_atoms: Optional[List[Atom]] = None
) -> Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]]:
    """
    Parse DSSP file and extract secondary structure and H-bond energy data.

    RESIDUE column uses original PDB residue numbers and may include an insertion
    code (e.g. 111A). Keys are 4-tuples (residue_name, residue_number, chain, insertion_code).

    Extracts:
    - Secondary structure (first character from STRUCTURE column)
    - N-H-->O_1, N-H-->O_2 (donor H-bond energies)
    - O-->H-N_1, O-->H-N_2 (acceptor H-bond energies)

    Args:
        dssp_path: Path to DSSP file
        pdb_atoms: Optional list of atoms from PDB file for validation

    Returns:
        Dictionary mapping (residue_name, residue_number, chain, insertion_code) -> {
            'secondary_structure': str,
            'N-H-->O_1': float or None, 'N-H-->O_2': float or None,
            'O-->H-N_1': float or None, 'O-->H-N_2': float or None
        }
    """
    pdb_residue_set: Set[Tuple[str, int, str, str]] = set()
    if pdb_atoms:
        for atom in pdb_atoms:
            pdb_residue_set.add(residue_key_from_atom(atom))

    dssp_data: Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]] = {}

    lines, header_line_idx = _load_dssp_lines(dssp_path)
    if header_line_idx is None:
        # No header found or file missing, return empty dict
        return dssp_data

    # Precompiled H-bond regex (module-level import of re)
    hbond_pattern = re.compile(r"(-?\d+),\s*(-?\d+\.?\d*)")

    # Parse data lines (starting after header)
    for line in lines[header_line_idx + 1 :]:
        line = line.rstrip("\n")
        if not line.strip():
            continue

        # Parse DSSP format
        # Format: "    1    1 H Q              0   0  210      0, 0.0     2,-0.3     0, 0.0    26,-0.1"
        # First number (0-4): sequential DSSP number
        # Second number (5-9): PDB residue number (this is what we need!)
        # Use fixed-width for initial fields, then split for H-bond fields
        try:
            # RESIDUE (PDB residue number, may have insertion code): columns 5-10
            parsed_res = parse_residue_number_field(line[5:10])
            if parsed_res is None:
                continue
            residue_number, insertion_code = parsed_res

            # Chain: column 11
            if len(line) < 12:
                continue
            chain = line[11:12].strip()
            if not chain:
                continue

            # AA (1-letter): column 13
            if len(line) < 14:
                continue
            aa_1letter = line[13:14].strip()
            if not aa_1letter or aa_1letter not in AA_1_TO_3:
                continue

            # Convert to 3-letter code
            residue_name = AA_1_TO_3[aa_1letter]

            # STRUCTURE: first character at column 16 (0-based index 16)
            secondary_structure = line[16] if len(line) > 16 else " "

            # H-bond columns: use regex to find all "residue,energy" patterns
            # H-bonds appear after ACC column (around position 40+)
            # Find all H-bond patterns in the line after column 40
            if len(line) > 40:
                matches = list(hbond_pattern.finditer(line[40:]))
                if len(matches) >= 4:
                    # Extract the 4 H-bond pairs
                    nh_o_1_str = f"{matches[0].group(1)},{matches[0].group(2)}"
                    oh_n_1_str = f"{matches[1].group(1)},{matches[1].group(2)}"
                    nh_o_2_str = f"{matches[2].group(1)},{matches[2].group(2)}"
                    oh_n_2_str = f"{matches[3].group(1)},{matches[3].group(2)}"
                elif len(matches) >= 2:
                    # Some residues may have fewer H-bonds
                    nh_o_1_str = f"{matches[0].group(1)},{matches[0].group(2)}"
                    oh_n_1_str = f"{matches[1].group(1)},{matches[1].group(2)}"
                    nh_o_2_str = ""
                    oh_n_2_str = ""
                else:
                    nh_o_1_str = ""
                    oh_n_1_str = ""
                    nh_o_2_str = ""
                    oh_n_2_str = ""
            else:
                nh_o_1_str = ""
                oh_n_1_str = ""
                nh_o_2_str = ""
                oh_n_2_str = ""

            nh_o_1_energy = _parse_hbond_energy(nh_o_1_str)
            oh_n_1_energy = _parse_hbond_energy(oh_n_1_str)
            nh_o_2_energy = _parse_hbond_energy(nh_o_2_str)
            oh_n_2_energy = _parse_hbond_energy(oh_n_2_str)

            residue_key = (residue_name, residue_number, chain, insertion_code)

            # Validate residue is present in PDB if provided
            if pdb_atoms and residue_key not in pdb_residue_set:
                # Residue not in PDB, skip
                continue

            dssp_data[residue_key] = {
                "secondary_structure": secondary_structure,
                "N-H-->O_1": nh_o_1_energy,
                "N-H-->O_2": nh_o_2_energy,
                "O-->H-N_1": oh_n_1_energy,
                "O-->H-N_2": oh_n_2_energy,
            }

        except (ValueError, IndexError):
            # Skip malformed lines
            continue

    return dssp_data


def _parse_hbond_energy(hbond_str: str) -> Optional[float]:
    """
    Parse H-bond energy from DSSP format "residue,energy".
    
    Args:
        hbond_str: String like "0, 0.0" or "105,-0.5" or "-2,-0.3"
        
    Returns:
        Energy value (float) or None if not parseable
    """
    if not hbond_str or hbond_str.strip() == '':
        return None
    
    # Split by comma
    parts = hbond_str.split(',')
    if len(parts) != 2:
        return None
    
    # Extract energy (second part)
    energy_str = parts[1].strip()
    if not energy_str:
        return None
    
    try:
        energy = float(energy_str)
        return energy
    except ValueError:
        return None


# ============================================================================
# C-alpha DBSCAN clustering by residue group (charge / hydrophobic / polar)
# ============================================================================

def compute_residue_cluster_labels(
    atoms: List[Atom],
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    eps: float = 6.0,
    min_samples: int = 2,
) -> Tuple[
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
]:
    """
    Collect C-alpha coordinates for four residue groups (negative, positive, hydrophobic, polar),
    run DBSCAN on each group's 3D coordinates, and return cluster labels per residue.
    
    Charge groups use PropKA: negative = ASP/GLU when pKa < pH; positive = ARG/LYS/HIS when pKa > pH.
    Hydrophobic = ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, GLY.
    Polar = SER, THR, ASN, GLN, TYR, CYS, plus any uncharged ASP/GLU/LYS/ARG/HIS.
    
    Each residue (by CA) is assigned to at most one group. Labels: -1 = noise, 0,1,2,... = cluster id.
    Residues not in a group do not appear in that group's dict.
    
    Returns:
        (negative_labels, positive_labels, hydrophobic_labels, polar_labels)
        each dict: residue_key (4-tuple) -> int label
    """
    # Collect CA per residue (one CA per residue_key)
    ca_by_res: Dict[Tuple[str, int, str, str], Tuple[float, float, float]] = {}
    for atom in atoms:
        if atom.name != "CA":
            continue
        key = residue_key_from_atom(atom)
        ca_by_res[key] = (atom.x, atom.y, atom.z)
    
    def get_pka(key):
        return pka_data.get(key)
    
    # Classify each CA into one group
    neg_keys: List[Tuple[str, int, str, str]] = []
    neg_coords: List[Tuple[float, float, float]] = []
    pos_keys: List[Tuple[str, int, str, str]] = []
    pos_coords: List[Tuple[float, float, float]] = []
    hydro_keys: List[Tuple[str, int, str, str]] = []
    hydro_coords: List[Tuple[float, float, float]] = []
    polar_keys: List[Tuple[str, int, str, str]] = []
    polar_coords: List[Tuple[float, float, float]] = []
    
    for key, xyz in ca_by_res.items():
        res_name = key[0]
        pka_val = get_pka(key)
        # First, assign to charged groups if PropKA says the residue is charged,
        # using the same charge rules as net_charge_from_pka and the same
        # missing-pKa conventions.
        if res_name in ("ASP", "GLU"):
            if pka_val is None or pka_val < pH:
                neg_keys.append(key)
                neg_coords.append(xyz)
        elif res_name in ("ARG", "LYS", "HIS"):
            if pka_val is None:
                # Treat ARG/LYS as charged when pKa missing; HIS as uncharged.
                if res_name in ("ARG", "LYS"):
                    pos_keys.append(key)
                    pos_coords.append(xyz)
            elif pka_val > pH:
                pos_keys.append(key)
                pos_coords.append(xyz)
        # Uncharged residues fall through to hydrophobic / polar grouping.
        elif res_name in HYDROPHOBIC_RESIDUES:
            hydro_keys.append(key)
            hydro_coords.append(xyz)
        elif res_name in POLAR_RESIDUES:
            polar_keys.append(key)
            polar_coords.append(xyz)
    
    def run_dbscan(keys: List, coords: List) -> Dict[Tuple[str, int, str, str], int]:
        if not coords:
            return {}
        X = np.array(coords)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return dict(zip(keys, clustering.labels_.tolist()))
    
    neg_labels = run_dbscan(neg_keys, neg_coords)
    pos_labels = run_dbscan(pos_keys, pos_coords)
    hydro_labels = run_dbscan(hydro_keys, hydro_coords)
    polar_labels = run_dbscan(polar_keys, polar_coords)
    
    return neg_labels, pos_labels, hydro_labels, polar_labels


# ============================================================================
# SASA File Parsing Functions (for output columns)
# ============================================================================

def parse_sasa_for_output(
    sasa_path: str,
    pdb_atoms: Optional[List[Atom]] = None
) -> Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]]:
    """
    Parse SASA file and extract Total-Side REL and Main-Chain REL for output.
    Keys are 4-tuples (residue_name, residue_number, chain, insertion_code); 4th column = original residue number.
    """
    pdb_residue_set: Set[Tuple[str, int, str, str]] = set()
    if pdb_atoms:
        for atom in pdb_atoms:
            pdb_residue_set.add(residue_key_from_atom(atom))

    sasa_output_data: Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]] = {}

    for rec in load_sasa_raw(sasa_path):
        residue_key = (rec.residue_name, rec.residue_number, rec.chain, rec.insertion_code)

        if pdb_atoms and residue_key not in pdb_residue_set:
            continue

        total_side_rel: Optional[float] = None
        if rec.total_side_rel and rec.total_side_rel != "N/A":
            try:
                total_side_rel = float(rec.total_side_rel)
            except ValueError:
                total_side_rel = None

        main_chain_rel: Optional[float] = None
        if rec.main_chain_rel and rec.main_chain_rel != "N/A":
            try:
                main_chain_rel = float(rec.main_chain_rel)
            except ValueError:
                main_chain_rel = None

        sasa_output_data[residue_key] = {
            "total_side_rel": total_side_rel,
            "main_chain_rel": main_chain_rel,
        }

    return sasa_output_data


# ============================================================================
# Shared H-bond enumeration (single tree, single pass over donor-acceptor pairs)
# ============================================================================

def _enumerate_hbonds(pdb_path: str) -> List[Tuple[Atom, Atom]]:
    """
    Enumerate all hydrogen bonds in the structure (distance + angle + backbone-separation rules).

    Single implementation used by calculate_global_hbond_density (weighted aggregation)
    and largest_hbond_component_size (graph construction). Returns list of (donor_atom, acceptor_atom)
    so callers can apply their own aggregation (SASA weights + sqrt, or edge set for union-find).

    Rules: D-A distance <= MAX_HBOND_DISTANCE, angle Base->Donor->Acceptor >= MIN_HBOND_ANGLE,
    same-residue excluded, backbone-backbone with |seq_i - seq_j| < MIN_BACKBONE_SEPARATION excluded.
    """
    abs_path = os.path.abspath(pdb_path)
    cached = _HBOND_PAIRS_CACHE.get(abs_path)
    if cached is not None:
        return cached

    atoms = _get_atoms_for_path(pdb_path)
    seq_index = _get_residue_seq_index(atoms)
    donors = [atom for atom in atoms if is_donor(atom)]
    acceptors = [atom for atom in atoms if is_acceptor(atom)]

    if len(acceptors) == 0:
        _HBOND_PAIRS_CACHE[abs_path] = []
        return []

    acceptor_coords = np.array([[a.x, a.y, a.z] for a in acceptors])
    acceptor_keys = [residue_key_from_atom(a) for a in acceptors]
    acceptor_tree = cKDTree(acceptor_coords)
    backbone_base_cache: Dict[Tuple[str, int], Atom] = {}
    backbone_base_vec_cache: Dict[Tuple[str, int, str, str], np.ndarray] = {}
    sidechain_base_cache: Dict[int, Optional[Atom]] = {}
    sidechain_base_vec_cache: Dict[int, np.ndarray] = {}

    pairs: List[Tuple[Atom, Atom]] = []

    for donor in donors:
        donor_coord = np.array([donor.x, donor.y, donor.z])
        donor_res_key = residue_key_from_atom(donor)
        indices = acceptor_tree.query_ball_point(donor_coord, MAX_HBOND_DISTANCE)
        if not indices:
            continue

        is_backbone_donor = is_backbone_atom(donor.name)
        precomputed_base: Optional[Atom] = None
        precomputed_base_vec: Optional[np.ndarray] = None
        if is_backbone_donor:
            precomputed_base = get_donor_base_atom(donor, atoms, backbone_base_cache)
            if precomputed_base is None:
                continue
            precomputed_base_vec = backbone_base_vec_cache.get(donor_res_key)
            if precomputed_base_vec is None:
                precomputed_base_vec = np.array([
                    precomputed_base.x - donor.x,
                    precomputed_base.y - donor.y,
                    precomputed_base.z - donor.z,
                ])
                backbone_base_vec_cache[donor_res_key] = precomputed_base_vec

        if is_backbone_donor:
            valid_indices = []
            valid_acceptors = []
            for idx in indices:
                acceptor = acceptors[idx]
                acceptor_res_key = acceptor_keys[idx]
                if donor_res_key == acceptor_res_key:
                    continue
                if is_backbone_atom(acceptor.name) and donor.chain == acceptor.chain:
                    di = seq_index.get(donor_res_key)
                    ai = seq_index.get(acceptor_res_key)
                    if di is not None and ai is not None:
                        if abs(di - ai) < MIN_BACKBONE_SEPARATION:
                            continue
                valid_indices.append(idx)
                valid_acceptors.append(acceptor)

            if not valid_acceptors:
                continue

            da_coords = np.array([[a.x, a.y, a.z] for a in valid_acceptors])
            da_vecs = da_coords - donor_coord
            db_vec = precomputed_base_vec
            db_norm = np.linalg.norm(db_vec)
            if db_norm == 0.0:
                continue
            da_norms = np.linalg.norm(da_vecs, axis=1)
            nonzero_mask = da_norms > 0.0
            if not np.any(nonzero_mask):
                continue
            dot_products = np.einsum("ij,j->i", da_vecs[nonzero_mask], db_vec)
            cos_theta = dot_products / (da_norms[nonzero_mask] * db_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_theta))
            angle_ok = angles >= MIN_HBOND_ANGLE
            if not np.any(angle_ok):
                continue
            passed_indices = np.nonzero(nonzero_mask)[0][angle_ok]
            for local_idx in passed_indices:
                acceptor = valid_acceptors[local_idx]
                pairs.append((donor, acceptor))
        else:
            # Side-chain donors: cache base atom and base vector per donor.
            donor_id = id(donor)
            base = sidechain_base_cache.get(donor_id)
            if base is None and donor_id not in sidechain_base_cache:
                base = get_donor_base_atom(donor, atoms, backbone_base_cache)
                sidechain_base_cache[donor_id] = base
            if base is None:
                continue
            base_vec = sidechain_base_vec_cache.get(donor_id)
            if base_vec is None:
                base_vec = np.array([base.x - donor.x, base.y - donor.y, base.z - donor.z])
                sidechain_base_vec_cache[donor_id] = base_vec

            for idx in indices:
                acceptor = acceptors[idx]
                acceptor_res_key = acceptor_keys[idx]
                if donor_res_key == acceptor_res_key:
                    continue
                if (is_backbone_atom(donor.name) and is_backbone_atom(acceptor.name) and donor.chain == acceptor.chain):
                    di = seq_index.get(donor_res_key)
                    ai = seq_index.get(acceptor_res_key)
                    if di is not None and ai is not None:
                        if abs(di - ai) < MIN_BACKBONE_SEPARATION:
                            continue
                if check_hydrogen_bond(
                    donor,
                    acceptor,
                    atoms,
                    backbone_base_cache,
                    precomputed_base=base,
                    precomputed_base_vec=base_vec,
                ):
                    pairs.append((donor, acceptor))

    _HBOND_PAIRS_CACHE[abs_path] = pairs
    return pairs


# Each hydrogen bond is counted once.
# Each donor-acceptor pair is checked once (we iterate for donor in donors, then for acceptor in acceptors, so we don't check acceptor-donor pairs separately).
# Each bond contributes to both residues:
# Donor residue gets donor_weight (based on donor atom's SASA)
# Acceptor residue gets acceptor_weight (based on acceptor atom's SASA)
# We don't skip residues — we ensure each donor-acceptor pair is only evaluated once in the nested loop.
# So a single bond adds to both residues, but with different weights. No double-counting of the same bond for the same residue.

# Type alias for H-bond density cache: (residue_weights_raw, residue_counts)
_HbondDensityRaw = Tuple[
    Dict[Tuple[str, int, str, str], float],
    Dict[Tuple[str, int, str, str], int],
]


def _aggregate_hbond_pairs_to_raw(
    pairs: List[Tuple[Atom, Atom]],
    sasa_data: Dict[Tuple[str, int, str, str], SASAEntry],
) -> Tuple[
    Dict[Tuple[str, int, str, str], float],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], bool],
]:
    """Aggregate H-bond pairs into per-residue raw weights, counts, and inter-chain flags."""
    atom_sasa_weight: Dict[int, float] = {}

    def get_cached_sasa_weight(atom: Atom) -> float:
        key = id(atom)
        cached = atom_sasa_weight.get(key)
        if cached is not None:
            return cached
        w = get_sasa_weight(atom, sasa_data)
        atom_sasa_weight[key] = w
        return w

    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    residue_counts: Dict[Tuple[str, int, str, str], int] = defaultdict(int)
    residue_inter_chain: Dict[Tuple[str, int, str, str], bool] = defaultdict(bool)

    for donor, acceptor in pairs:
        donor_key = residue_key_from_atom(donor)
        acceptor_key = residue_key_from_atom(acceptor)
        donor_weight = get_cached_sasa_weight(donor)
        acceptor_weight = get_cached_sasa_weight(acceptor)
        residue_weights_raw[donor_key] += donor_weight
        residue_weights_raw[acceptor_key] += acceptor_weight
        residue_counts[donor_key] += 1
        residue_counts[acceptor_key] += 1
        if donor.chain != acceptor.chain:
            residue_inter_chain[donor_key] = True
            residue_inter_chain[acceptor_key] = True

    return dict(residue_weights_raw), dict(residue_counts), dict(residue_inter_chain)


def calculate_global_hbond_density(
    pdb_path: str,
    sasa_path: str,
) -> Tuple[Dict[Tuple[str, int, str, str], float], Dict[Tuple[str, int, str, str], bool], Dict[Tuple[str, int, str, str], int]]:
    """
    Calculate global hydrogen bond density per residue.
    
    Number of hydrogen bonds per residue, weighted by scaled relative SASA:
    - Per-contact weight = 100 * REL, where REL is total-side REL for side-chain
      atoms or main-chain REL for backbone atoms.
    
    Each hydrogen bond is counted once and contributes to both residues involved.
    All H-bonds between a residue pair are accumulated (no limit per pair).
    
    A square-root transform is applied to the final per-residue weights to introduce
    diminishing returns with increasing numbers of bonds, so residues with many
    contacts do not dominate the metric.
    
    Uses scipy's cKDTree for efficient O(N log N) neighbor search instead of O(N²).
    KD-tree correctly uses [x, y, z] coordinates for spatial queries.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        
    Returns:
        Tuple of:
        1. Dictionary mapping (residue_name, residue_number, chain) -> weighted HB density
        2. Dictionary mapping (residue_name, residue_number, chain) -> bool indicating
           if residue has any inter-chain contacts
        3. Dictionary mapping (residue_name, residue_number, chain) -> number of H-bonds
    """
    pairs = _enumerate_hbonds(pdb_path)
    if not pairs:
        return {}, {}, {}
    sasa_data = parse_sasa(sasa_path)
    weights_raw, counts, inter_chain = _aggregate_hbond_pairs_to_raw(pairs, sasa_data)
    sqrt_residue_hbonds = {key: math.sqrt(w) for key, w in weights_raw.items()}
    return sqrt_residue_hbonds, inter_chain, counts


def compute_hbond_density_raw(
    pdb_path: str,
    sasa_path: str,
) -> _HbondDensityRaw:
    """
    Compute per-residue raw SASA-weighted sums and H-bond counts. No sqrt.

    Use this to cache H-bond density data when calling
    calculate_global_hbond_density_average multiple times with different
    residue sets or weighted vs unweighted.
    """
    pairs = _enumerate_hbonds(pdb_path)
    if not pairs:
        return {}, {}
    sasa_data = parse_sasa(sasa_path)
    weights_raw, counts, _ = _aggregate_hbond_pairs_to_raw(pairs, sasa_data)
    return weights_raw, counts


def calculate_global_hbond_density_average(
    pdb_path: str,
    sasa_path: str,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    weighted: bool = True,
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    hbond_density_raw: Optional[_HbondDensityRaw] = None,
) -> float:
    """
    Calculate average H-bond density with configurable numerator and denominator.

    Same interface as salt bridge density average: optional residues_for_density
    (which residues contribute to the sum), weighted (SASA-weighted vs raw count),
    residues_for_average (denominator), and hbond_density_raw (pre-computed cache
    from compute_hbond_density_raw() to avoid recomputing when calling multiple times).

    Examples:
        - Average (SASA-weighted) H-bond density of CDR residues over all residues:
          calculate_global_hbond_density_average(pdb, sasa, residues_for_density=cdr_residues)
        - Average unweighted H-bond count over CDR residues only:
          calculate_global_hbond_density_average(pdb, sasa, residues_for_density=cdr_residues,
                                                 weighted=False, residues_for_average=cdr_residues)
        - Multiple averages without recomputing:
          raw = compute_hbond_density_raw(pdb, sasa)
          avg1 = calculate_global_hbond_density_average(pdb, sasa, hbond_density_raw=raw, ...)
          avg2 = calculate_global_hbond_density_average(pdb, sasa, hbond_density_raw=raw, ...)

    Args:
        pdb_path: Path to PDB structure file (used when hbond_density_raw is None, and for
            denominator when residues_for_average is None).
        sasa_path: Path to SASA file (used only when hbond_density_raw is None).
        residues_for_density: If provided, only these residues contribute to the numerator.
        weighted: If True, use SASA-weighted density (with sqrt transform). If False, use raw count.
        residues_for_average: Residues over which to average (denominator). If None, total PDB residues.
        hbond_density_raw: Optional (weights_raw, counts) from compute_hbond_density_raw().

    Returns:
        Sum of (density or count) over residues_for_density, divided by denominator.
    """
    if hbond_density_raw is None:
        hbond_density_raw = compute_hbond_density_raw(pdb_path, sasa_path)
    weights_raw, counts = hbond_density_raw
    if not weights_raw:
        return 0.0

    density_set: Optional[Set[Tuple[str, int, str, str]]] = (
        set(residues_for_density) if residues_for_density is not None else None
    )
    if residues_for_average is not None:
        denom = len(set(residues_for_average))
    else:
        denom = _count_unique_residues_in_pdb(pdb_path)

    if denom == 0:
        return 0.0

    keys_to_sum = (
        weights_raw.keys()
        if density_set is None
        else [k for k in weights_raw.keys() if k in density_set]
    )
    if weighted:
        # Precompute sqrt-transformed weights once per call to avoid repeated
        # sqrt when multiple averages are taken over the same raw data.
        sqrt_cache = {k: math.sqrt(w) for k, w in weights_raw.items()}
        total = sum(sqrt_cache[k] for k in keys_to_sum)
    else:
        total = sum(counts[k] for k in keys_to_sum)

    return total / float(denom)


def largest_hbond_component_size(pdb_path: str) -> int:
    """
    Size of the largest connected component of the H-bond network (geometry-based).

    Uses the same H-bond detection as calculate_global_hbond_density (distance + angle
    criteria) via _enumerate_hbonds. Builds an undirected graph of residue pairs
    connected by H-bonds, then returns the number of residues in the largest component.

    Args:
        pdb_path: Path to PDB structure file.

    Returns:
        Number of residues in the largest connected component (0 if no H-bonds).
    """
    pairs = _enumerate_hbonds(pdb_path)
    if not pairs:
        return 0

    edges: Set[frozenset] = {
        frozenset({residue_key_from_atom(donor), residue_key_from_atom(acceptor)})
        for donor, acceptor in pairs
    }
    if not edges:
        return 0

    parent: Dict[Tuple, Tuple] = {}

    def find(x: Tuple) -> Tuple:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: Tuple, y: Tuple) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for edge in edges:
        u, v = tuple(edge)
        union(u, v)

    root_count: Dict[Tuple, int] = defaultdict(int)
    for node in parent:
        root_count[find(node)] += 1

    return max(root_count.values()) if root_count else 0


# ============================================================================
# DSSP-based Hydrogen Bond Density Calculation (Backbone Only)
# ============================================================================

def _parse_dssp_hbond_pair(hbond_str: str) -> Optional[Tuple[int, float]]:
    """
    Parse DSSP H-bond pair string to extract residue offset and energy.
    
    DSSP format: "residue_offset,energy" where residue_offset is relative to current residue.
    For example: "2,-0.3" means H-bond with residue at current + 2, energy -0.3.
    
    Args:
        hbond_str: String like "0, 0.0" or "105,-0.5" or "-2,-0.3" or ""
        
    Returns:
        Tuple of (residue_offset, energy) or None if not parseable or zero energy
    """
    if not hbond_str or hbond_str.strip() == '':
        return None
    
    # Split by comma
    parts = hbond_str.split(',')
    if len(parts) != 2:
        return None
    
    # Extract residue offset (first part) and energy (second part)
    offset_str = parts[0].strip()
    energy_str = parts[1].strip()
    
    if not offset_str or not energy_str:
        return None
    
    try:
        residue_offset = int(offset_str)
        energy = float(energy_str)
        
        # Only return non-zero bonds (0, 0.0 means no bond)
        if residue_offset == 0 and energy == 0.0:
            return None
        
        return (residue_offset, energy)
    except ValueError:
        return None


def parse_dssp_hbonds(
    dssp_path: str,
    pdb_atoms: Optional[List[Atom]] = None
) -> Tuple[Dict[Tuple[str, int, str, str], List[Tuple[int, float]]], Dict[int, Tuple[str, int, str, str]]]:
    """
    Parse DSSP file and extract H-bond pairs (residue offset + energy).
    
    Extracts all H-bonds from N-H-->O and O-->H-N columns (both pairs for bifurcated bonds).
    Only includes bonds with non-zero energy.
    
    Note: H-bond offsets in DSSP are relative to sequential DSSP number (first column),
    not PDB residue number (second column). This function returns both the H-bond data
    and a mapping from sequential DSSP number to PDB residue key.
    
    Args:
        dssp_path: Path to DSSP file
        pdb_atoms: Optional list of atoms from PDB file for validation
        
    Returns:
        Tuple of:
        1. Dictionary mapping (residue_name, residue_number, chain, insertion_code) -> list of (residue_offset, energy) tuples
        2. Dictionary mapping sequential_DSSP_number -> (residue_name, residue_number, chain, insertion_code)
    """
    pdb_residue_set: Set[Tuple[str, int, str, str]] = set()
    if pdb_atoms:
        for atom in pdb_atoms:
            pdb_residue_set.add(residue_key_from_atom(atom))

    hbond_data: Dict[Tuple[str, int, str, str], List[Tuple[int, float]]] = {}
    dssp_seq_to_pdb: Dict[int, Tuple[str, int, str, str]] = {}

    lines, header_line_idx = _load_dssp_lines(dssp_path)
    if header_line_idx is None:
        # No header found or file missing, return empty dicts
        return hbond_data, dssp_seq_to_pdb

    # Parse data lines (starting after header)
    for line in lines[header_line_idx + 1 :]:
        line = line.rstrip("\n")
        if not line.strip():
            continue

        # Parse DSSP format
        # Format: "    1    1 H Q              0   0  210      0, 0.0     2,-0.3     0, 0.0    26,-0.1"
        # First number (columns 0-4): Sequential DSSP number
        # Second number (columns 5-9): PDB residue number
        try:
            # Sequential DSSP number: columns 0-4 (first number)
            dssp_seq_str = line[0:5].strip()
            if not dssp_seq_str:
                continue
            dssp_seq_num = int(dssp_seq_str)

            # RESIDUE (original PDB residue number, 2nd column): columns 5-10
            parsed_res = parse_residue_number_field(line[5:10])
            if parsed_res is None:
                continue
            residue_number, insertion_code = parsed_res

            # Chain: column 11
            if len(line) < 12:
                continue
            chain = line[11:12].strip()
            if not chain:
                continue

            # AA (1-letter): column 13
            if len(line) < 14:
                continue
            aa_1letter = line[13:14].strip()
            if not aa_1letter or aa_1letter not in AA_1_TO_3:
                continue

            residue_name = AA_1_TO_3[aa_1letter]
            residue_key = (residue_name, residue_number, chain, insertion_code)

            if pdb_atoms and residue_key not in pdb_residue_set:
                continue

            dssp_seq_to_pdb[dssp_seq_num] = residue_key

            # H-bond columns: use regex to find all "residue,energy" patterns
            # H-bonds appear after ACC column (around position 40+)
            # Offsets are relative to sequential DSSP number
            import re

            hbond_pattern = r"(-?\d+),\s*(-?\d+\.?\d*)"

            # Find all H-bond patterns in the line after column 40
            hbond_pairs = []
            if len(line) > 40:
                matches = list(re.finditer(hbond_pattern, line[40:]))
                for match in matches:
                    hbond_str = f"{match.group(1)},{match.group(2)}"
                    hbond_pair = _parse_dssp_hbond_pair(hbond_str)
                    if hbond_pair is not None:
                        hbond_pairs.append(hbond_pair)

            if hbond_pairs:
                hbond_data[residue_key] = hbond_pairs

        except (ValueError, IndexError):
            # Skip malformed lines
            continue

    return hbond_data, dssp_seq_to_pdb


# Type alias for DSSP H-bond energy density cache: (residue_weights_raw, residue_counts)
# Weights are SASA * |energy| per bond; counts are number of backbone H-bonds per residue.
_DsspHbondEnergyDensityRaw = Tuple[
    Dict[Tuple[str, int, str, str], float],
    Dict[Tuple[str, int, str, str], int],
]


def _aggregate_dssp_hbond_energy_to_raw(
    dssp_hbonds: Dict[Tuple[str, int, str, str], List[Tuple[int, float]]],
    dssp_seq_to_pdb: Dict[int, Tuple[str, int, str, str]],
    pdb_to_dssp_seq: Dict[Tuple[str, int, str, str], int],
    sasa_data: Dict[Tuple[str, int, str, str], SASAEntry],
) -> Tuple[
    Dict[Tuple[str, int, str, str], float],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], bool],
]:
    """Aggregate DSSP H-bond (offset, energy) data into per-residue raw weights (SASA*|energy|), counts, and inter-chain flags."""
    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    residue_counts: Dict[Tuple[str, int, str, str], int] = defaultdict(int)
    residue_inter_chain: Dict[Tuple[str, int, str, str], bool] = defaultdict(bool)

    for residue_key, hbond_pairs in dssp_hbonds.items():
        if residue_key not in pdb_to_dssp_seq:
            continue
        dssp_seq_num = pdb_to_dssp_seq[residue_key]
        residue_weight = get_residue_sasa_weight(
            residue_key, sasa_data, use_side_chain=False
        )
        if residue_weight == 0.0:
            continue

        for residue_offset, energy in hbond_pairs:
            target_dssp_seq = dssp_seq_num + residue_offset
            if target_dssp_seq not in dssp_seq_to_pdb:
                continue
            target_residue_key = dssp_seq_to_pdb[target_dssp_seq]
            is_inter_chain = (residue_key[2] != target_residue_key[2])
            target_weight = get_residue_sasa_weight(
                target_residue_key, sasa_data, use_side_chain=False
            )
            if target_weight == 0.0:
                continue
            energy_scale = abs(energy)
            if energy_scale == 0.0:
                continue

            residue_weights_raw[residue_key] += residue_weight * energy_scale
            residue_weights_raw[target_residue_key] += target_weight * energy_scale
            residue_counts[residue_key] += 1
            residue_counts[target_residue_key] += 1
            if is_inter_chain:
                residue_inter_chain[residue_key] = True
                residue_inter_chain[target_residue_key] = True

    return dict(residue_weights_raw), dict(residue_counts), dict(residue_inter_chain)


def calculate_hbond_energy_density_dssp_backbone_only(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
) -> Tuple[Dict[Tuple[str, int, str, str], float], Dict[Tuple[str, int, str, str], bool]]:
    """
    Calculate hydrogen bond energy density per residue using DSSP data (backbone only).

    Residue keys are 4-tuples (residue_name, residue_number, chain, insertion_code).
    H-bonds are from DSSP (N-H-->O and O-->H-N; backbone only, no N in Pro).
    Each bond is weighted by main-chain SASA and by the magnitude of the DSSP H-bond energy.

    Args:
        pdb_path: Path to PDB structure file (for residue mapping)
        sasa_path: Path to SASA file (for main-chain REL values)
        dssp_path: Path to DSSP file (for H-bond detection)

    Returns:
        Tuple of:
        1. Dictionary mapping residue_key -> weighted HB energy density (sqrt applied)
        2. Dictionary mapping residue_key -> bool indicating if residue has any inter-chain contacts
    """
    atoms = _get_atoms_for_path(pdb_path)
    sasa_data = parse_sasa(sasa_path)
    dssp_hbonds, dssp_seq_to_pdb = parse_dssp_hbonds(dssp_path, atoms)
    pdb_to_dssp_seq = {pdb_key: dssp_seq for dssp_seq, pdb_key in dssp_seq_to_pdb.items()}

    weights_raw, _, inter_chain = _aggregate_dssp_hbond_energy_to_raw(
        dssp_hbonds, dssp_seq_to_pdb, pdb_to_dssp_seq, sasa_data
    )
    sqrt_residue_hbonds = {key: math.sqrt(w) for key, w in weights_raw.items()}
    return sqrt_residue_hbonds, inter_chain


def compute_dssp_hbond_energy_density_raw(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
) -> _DsspHbondEnergyDensityRaw:
    """
    Compute per-residue raw (SASA * |energy|) sums and backbone H-bond counts from DSSP. No sqrt.

    Use this to cache DSSP H-bond energy density when calling
    calculate_hbond_energy_density_dssp_backbone_only_average multiple times.
    """
    atoms = _get_atoms_for_path(pdb_path)
    sasa_data = parse_sasa(sasa_path)
    dssp_hbonds, dssp_seq_to_pdb = parse_dssp_hbonds(dssp_path, atoms)
    pdb_to_dssp_seq = {pdb_key: dssp_seq for dssp_seq, pdb_key in dssp_seq_to_pdb.items()}
    weights_raw, counts, _ = _aggregate_dssp_hbond_energy_to_raw(
        dssp_hbonds, dssp_seq_to_pdb, pdb_to_dssp_seq, sasa_data
    )
    return weights_raw, counts


def calculate_hbond_energy_density_dssp_backbone_only_average(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    weighted: bool = True,
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    dssp_hbond_energy_density_raw: Optional[_DsspHbondEnergyDensityRaw] = None,
) -> float:
    """
    Average DSSP backbone H-bond energy density with configurable numerator and denominator.

    Same interface as salt bridge / global H-bond: residues_for_density (which residues
    to sum), weighted (SASA * |energy| with sqrt vs raw count), residues_for_average
    (denominator), and dssp_hbond_energy_density_raw (cache from
    compute_dssp_hbond_energy_density_raw()).
    DSSP provides backbone H-bonds only (no N in Pro).
    """
    if dssp_hbond_energy_density_raw is None:
        dssp_hbond_energy_density_raw = compute_dssp_hbond_energy_density_raw(
            pdb_path, sasa_path, dssp_path
        )
    weights_raw, counts = dssp_hbond_energy_density_raw
    if not weights_raw:
        return 0.0

    density_set = set(residues_for_density) if residues_for_density is not None else None
    denom = len(set(residues_for_average)) if residues_for_average is not None else _count_unique_residues_in_pdb(pdb_path)
    if denom == 0:
        return 0.0

    keys_to_sum = set(weights_raw.keys()) if density_set is None else (set(weights_raw.keys()) & density_set)
    if weighted:
        total = sum(math.sqrt(weights_raw[k]) for k in keys_to_sum)
    else:
        total = sum(counts.get(k, 0) for k in keys_to_sum)
    return total / float(denom)

