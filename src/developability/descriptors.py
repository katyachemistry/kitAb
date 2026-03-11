"""
Thermostability descriptors for antibody structures.

This module calculates developability determinants from antibody structures
and SASA values. The first property is thermostability.
"""

from typing import Dict, List, Optional, Set, Tuple
import math
import os
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy import optimize
from sklearn.cluster import DBSCAN

from utils.parsers import (
    parse_pdb, parse_sasa, Atom, SASAEntry,
    distance, angle_between_vectors, is_backbone_atom,
    parse_pka, get_pka_file_path, CHARGED_RESIDUE_TYPES,
    residue_key_from_atom,
)

# Mapping from 1-letter to 3-letter amino acid codes
AA_1_TO_3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}


# Hydrogen bond donors (immutable set for performance)
# Format: (atom_name, residue_name) where "ANY" means any residue
DONORS = frozenset({
    # Backbone
    ("N", "ANY"),   # backbone N–H (except Pro)
    
    # Side chains
    ("NE2", "GLN"),
    ("ND2", "ASN"),
    ("NE", "ARG"),
    ("NH1", "ARG"),
    ("NH2", "ARG"),
    ("NZ", "LYS"),
    ("ND1", "HIS"),
    ("NE2", "HIS"),
    ("OG", "SER"),
    ("OG1", "THR"),
    ("OH", "TYR"),
})

# Hydrogen bond acceptors (immutable set for performance)
ACCEPTORS = frozenset({
    # Backbone
    ("O", "ANY"),  # backbone C=O
    
    # Side chains
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

# Maximum D-A distance for hydrogen bonds (Angstroms)
MAX_HBOND_DISTANCE = 3.2

# Minimum angle Base→Donor→Acceptor for hydrogen bonds (degrees)
MIN_HBOND_ANGLE = 120.0

# Minimum residue separation for backbone-backbone H-bonds (DSSP-style)
MIN_BACKBONE_SEPARATION = 3

# Minimum SASA threshold: values below 1% are clipped to 1% to give maximum weight
# This ensures the most buried residues get maximum weight (1 / 0.01 = 100)
MIN_SASA_THRESHOLD = 0.01  # 1% as a fraction (since we parse percentages and divide by 100)

# Maximum weight per residue: cap total accumulated weight at 100
# This prevents residues with multiple buried atoms (backbone + side-chain) from exceeding 100
MAX_WEIGHT_PER_RESIDUE = 100.0

# ============================================================================
# Salt-Bridge Detection Constants
# ============================================================================

# Maximum distance for salt bridges (Angstroms)
MAX_SALT_BRIDGE_DISTANCE = 4.0

# Positively charged atoms (bases)
# Format: (atom_name, residue_name)
POSITIVE_CHARGED_ATOMS = frozenset({
    ("NH1", "ARG"),
    ("NH2", "ARG"),
    ("NZ", "LYS"),
    # HIS excluded by default (ambiguous protonation state)
})

# Negatively charged atoms (acids)
# Format: (atom_name, residue_name)
NEGATIVE_CHARGED_ATOMS = frozenset({
    ("OD1", "ASP"),
    ("OD2", "ASP"),
    ("OE1", "GLU"),
    ("OE2", "GLU"),
})

# Aromatic residues
AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP"})


def is_donor(atom: Atom) -> bool:
    """
    Check if an atom can act as a hydrogen bond donor.
    
    Special cases:
    - Proline backbone N is NOT a donor
    - Amide N (Asn/Gln) is donor only, NOT acceptor
    """
    # Proline backbone N is NOT a donor
    if atom.name == "N" and atom.residue_name == "PRO":
        return False
    
    # Check if atom matches donor criteria
    if (atom.name, atom.residue_name) in DONORS:
        return True
    if (atom.name, "ANY") in DONORS:
        return True
    
    return False


def is_acceptor(atom: Atom) -> bool:
    """
    Check if an atom can act as a hydrogen bond acceptor.
    
    Special cases:
    - Amide N (Asn/Gln) is donor only, NOT acceptor
    """
    # Amide N (Asn/Gln) is NOT an acceptor
    if atom.name in ("ND2", "NE2") and atom.residue_name in ("ASN", "GLN"):
        return False
    
    # Check if atom matches acceptor criteria
    if (atom.name, atom.residue_name) in ACCEPTORS:
        return True
    if (atom.name, "ANY") in ACCEPTORS:
        return True
    
    return False


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
    # Backbone N: base is C from previous residue
    if donor.name == "N" and is_backbone_atom(donor.name):
        # Use cache if provided
        if backbone_base_cache is not None:
            cache_key = (donor.chain, donor.residue_number - 1)
            if cache_key in backbone_base_cache:
                return backbone_base_cache[cache_key]
        
        # Find C atom from previous residue (same chain)
        for atom in atoms:
            if (atom.name == "C" and
                atom.chain == donor.chain and
                atom.residue_number == donor.residue_number - 1):
                # Update cache if provided
                if backbone_base_cache is not None:
                    cache_key = (donor.chain, donor.residue_number - 1)
                    backbone_base_cache[cache_key] = atom
                return atom
        return None
    
    # Side-chain donors: find bonded heavy atom in same residue
    # Maximum covalent bond distance (typically 1.5-2.0 Å for heavy atoms)
    max_covalent_distance = 2.0
    
    candidates = []
    for atom in atoms:
        if atom == donor:
            continue
        if (atom.residue_name == donor.residue_name and
            atom.residue_number == donor.residue_number and
            atom.chain == donor.chain):
            # Skip hydrogen atoms
            if atom.element == "H":
                continue
            dist = distance(donor, atom)
            if dist < max_covalent_distance:
                candidates.append((dist, atom))
    
    if candidates:
        # Return closest heavy atom
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    return None


def check_hydrogen_bond(
    donor: Atom, 
    acceptor: Atom, 
    atoms: List[Atom],
    backbone_base_cache: Optional[Dict[Tuple[str, int], Atom]] = None
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
    
    # Get base atom for donor
    base = get_donor_base_atom(donor, atoms, backbone_base_cache)
    
    # For backbone donors, base is mandatory
    if is_backbone_atom(donor.name):
        if base is None:
            return False
    else:
        # For side-chain donors, allow fallback to distance-only if base not found
        if base is None:
            return True  # Distance-only fallback for flexible side chains
    
    # Calculate angle: Base→Donor→Acceptor
    # Vector from donor to base
    db_vec = (base.x - donor.x, base.y - donor.y, base.z - donor.z)
    # Vector from donor to acceptor
    da_vec = (acceptor.x - donor.x, acceptor.y - donor.y, acceptor.z - donor.z)
    
    # Calculate angle Base-Donor-Acceptor (returns degrees, DSSP-style)
    angle = angle_between_vectors(db_vec, da_vec)
    
    return angle >= MIN_HBOND_ANGLE


def get_sasa_weight(atom: Atom, sasa_data: Dict[Tuple[str, int, str], SASAEntry], weighting: str = "inverse") -> float:
    key = (atom.residue_name, atom.residue_number, atom.chain)
    if key not in sasa_data:
        return 0.0
    
    sasa_entry = sasa_data[key]
    
    if is_backbone_atom(atom.name):
        rel_sasa = sasa_entry.main_chain_rel
    else:
        rel_sasa = sasa_entry.total_side_rel
    
    # Note: rel_sasa is already a fraction (0-1), not a percentage
    # parse_sasa() already divides percentages by 100.0
    rel_sasa_fraction = rel_sasa
    
    if weighting == "negative_linear":
        clamped = min(max(rel_sasa_fraction, 0.0), 1.0)
        return 1.0 - clamped
    else:
        if rel_sasa_fraction < MIN_SASA_THRESHOLD:
            rel_sasa_fraction = MIN_SASA_THRESHOLD
        return 1.0 / rel_sasa_fraction


def get_residue_sasa_weight(
    residue_key: Tuple[str, int, str],
    sasa_data: Dict[Tuple[str, int, str], SASAEntry],
    use_side_chain: bool = True,
    weighting: str = "inverse"
) -> float:
    if residue_key not in sasa_data:
        return 0.0
    
    sasa_entry = sasa_data[residue_key]
    
    if use_side_chain:
        rel_sasa = sasa_entry.total_side_rel
    else:
        rel_sasa = sasa_entry.main_chain_rel
    
    # Note: rel_sasa is already a fraction (0-1), not a percentage
    # parse_sasa() already divides percentages by 100.0
    rel_sasa_fraction = rel_sasa
    
    if weighting == "negative_linear":
        clamped = min(max(rel_sasa_fraction, 0.0), 1.0)
        return 1.0 - clamped
    else:
        if rel_sasa_fraction < MIN_SASA_THRESHOLD:
            rel_sasa_fraction = MIN_SASA_THRESHOLD
        return 1.0 / rel_sasa_fraction


# ============================================================================
# Charge State Functions
# ============================================================================

def is_residue_charged(
    residue_name: str,
    pka_value: Optional[float],
    pH: float
) -> bool:
    """
    Determine if a residue is charged at a given pH based on its pKa value.
    
    Rules:
    - ASP/GLU: charged (deprotonated, negative) when pKa < pH
    - LYS/ARG: charged (protonated, positive) when pKa > pH
    
    If pKa is None (residue not in pKa file), returns True (conservative default:
    assume charged).
    
    Args:
        residue_name: 3-letter residue code (ASP, GLU, LYS, ARG)
        pka_value: pKa value from pKa file, or None if not found
        pH: pH value to check charge state at
        
    Returns:
        True if residue is charged at given pH, False otherwise
    """
    # If pKa not available, assume charged (conservative default)
    if pka_value is None:
        return True
    
    # Check charge state based on residue type
    if residue_name in ("ASP", "GLU"):
        # Acidic: charged (deprotonated, negative) when pKa < pH
        return pka_value < pH
    elif residue_name in ("LYS", "ARG"):
        # Basic: charged (protonated, positive) when pKa > pH
        return pka_value > pH
    
    # Not a charged residue type, return False
    return False


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
        elif res_name in ("LYS", "ARG"):
            # Fraction protonated (positive charge)
            net += 1.0 / (1.0 + np.power(10.0, pH - pka))
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
        return -1.0 / (1.0 + np.power(10.0, pka_value - pH))
    if residue_name in ("LYS", "ARG"):
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

    atoms = parse_pdb(pdb_path)
    if not atoms:
        return None

    try:
        sasa_data = parse_sasa(sasa_path)
    except Exception:
        return None

    def _get_pka(key_4: Tuple) -> Optional[float]:
        return pka_data.get(key_4) or pka_data.get((key_4[0], key_4[1], key_4[2], ""))

    try:
        n = len(atoms)
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)

        # Side-chain mask: atom name not in main-chain list
        is_sidechain = np.array([a.name.strip() not in SCM_MAIN_CHAIN_ATOMS for a in atoms], dtype=bool)

        # Residue key (3-tuple) for SASA lookup
        def res_key_3(a: Atom) -> Tuple[str, int, str]:
            return (a.residue_name, a.residue_number, a.chain)

        # Residue exposed: total_side_rel > sasa_cutoff
        residue_exposed = np.zeros(n, dtype=bool)
        for i, a in enumerate(atoms):
            key3 = res_key_3(a)
            entry = sasa_data.get(key3)
            if entry is not None and getattr(entry, "total_side_rel", 0) is not None:
                residue_exposed[i] = entry.total_side_rel > sasa_cutoff

        # Per-residue fractional charge at pH
        residue_charge: Dict[Tuple[str, int, str, str], float] = {}
        for a in atoms:
            key4 = residue_key_from_atom(a)
            if key4 not in residue_charge:
                pka_val = _get_pka(key4)
                residue_charge[key4] = _residue_fractional_charge_at_pH(a.residue_name, pka_val, pH)

        # Per-atom charge: distribute residue charge equally over side-chain atoms
        atom_charge = np.zeros(n, dtype=np.float64)
        residue_sidechain_count: Dict[Tuple, int] = {}
        for i, a in enumerate(atoms):
            key4 = residue_key_from_atom(a)
            if is_sidechain[i]:
                residue_sidechain_count[key4] = residue_sidechain_count.get(key4, 0) + 1
        for i, a in enumerate(atoms):
            if is_sidechain[i]:
                key4 = residue_key_from_atom(a)
                count = residue_sidechain_count.get(key4, 1)
                atom_charge[i] = residue_charge.get(key4, 0.0) / max(1, count)

        # SCM_atom,i = sum of charge(j) for j side-chain, exposed, within d_cutoff of i
        tree = cKDTree(coords)
        scm_atom = np.zeros(n, dtype=np.float64)
        for i in range(n):
            indices = tree.query_ball_point(coords[i], d_cutoff)
            for j in indices:
                if i == j:
                    continue
                if not is_sidechain[j]:
                    continue
                if not residue_exposed[j]:
                    continue
                scm_atom[i] += atom_charge[j]

        # SCM score = | sum of SCM_atom,i for i where SCM_atom,i < 0 |
        neg_sum = np.sum(scm_atom[scm_atom < 0])
        return float(np.abs(neg_sum))
    except Exception:
        return None


def sum_total_side_rel_within_cutoff(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[Tuple[str, int, str], Dict[str, Optional[float]]],
    cutoff: float = 5.0,
) -> Dict[Tuple[str, int, str, str], float]:
    """
    For each residue, sum total_side_rel (relative side-chain ASA, %) of all residues
    within cutoff Å (Cα–Cα distance). Includes the residue itself in the sum.

    Args:
        pdb_atoms: List of atoms from PDB.
        sasa_output_data: Dict (residue_name, residue_number, chain) -> {'total_side_rel': float or None, ...}.
        cutoff: Distance cutoff in Å (default 5.0).

    Returns:
        Dict mapping (residue_name, residue_number, chain, insertion_code) -> float (sum of total_side_rel in %).
    """
    # Unique residues with 4-tuple key and Cα position (or centroid)
    residue_keys: List[Tuple[str, int, str, str]] = []
    residue_coords: List[Tuple[float, float, float]] = []
    residue_total_side_rel: List[float] = []

    seen = set()
    atoms_by_res: Dict[Tuple, List[Atom]] = {}
    for atom in pdb_atoms:
        key4 = residue_key_from_atom(atom)
        if key4 not in seen:
            seen.add(key4)
            residue_keys.append(key4)
        atoms_by_res.setdefault(key4, []).append(atom)

    for key4 in residue_keys:
        atoms_res = atoms_by_res[key4]
        key3 = (key4[0], key4[1], key4[2])
        sasa_entry = sasa_output_data.get(key3) or {}
        val = sasa_entry.get("total_side_rel")
        total_side_rel = float(val) if val is not None else 0.0
        residue_total_side_rel.append(total_side_rel)

        # Cα position or centroid
        ca_atoms = [a for a in atoms_res if a.name.strip() == "CA"]
        if ca_atoms:
            a0 = ca_atoms[0]
            residue_coords.append((a0.x, a0.y, a0.z))
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
    return (atom.name, atom.residue_name) in POSITIVE_CHARGED_ATOMS


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
    return (atom.name, atom.residue_name) in NEGATIVE_CHARGED_ATOMS


def detect_salt_bridges(
    pdb_path: str,
    sasa_path: str,
    weighting: str = "inverse",
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> Dict[Tuple[Tuple[str, int, str], Tuple[str, int, str]], Tuple[float, float]]:
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
        weighting: Weighting strategy ("inverse" or "negative_linear")
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)
        
    Returns:
        Dictionary mapping ((pos_res_name, pos_res_num, pos_chain), (neg_res_name, neg_res_num, neg_chain)) 
        -> (pos_weight, neg_weight)
        where weights are the maximum inverse SASA weights from any atom-atom contact in the pair.
        Keys are ordered such that the positive residue comes first.
    """
    # Parse files
    atoms = parse_pdb(pdb_path)
    sasa_data = parse_sasa(sasa_path)
    
    # Parse pKa file if available
    pka_data = {}
    if pka_path is None:
        # Try to auto-detect pKa file path
        pka_path = get_pka_file_path(pdb_path)
    
    if pka_path and os.path.exists(pka_path):
        pka_data = parse_pka(pka_path, atoms)
    
    # Identify positively and negatively charged atoms
    positive_atoms = [atom for atom in atoms if is_positively_charged(atom)]
    negative_atoms = [atom for atom in atoms if is_negatively_charged(atom)]
    
    # Build KD-tree for negative atoms for efficient neighbor search
    if len(negative_atoms) == 0 or len(positive_atoms) == 0:
        return {}
    
    negative_coords = np.array([[a.x, a.y, a.z] for a in negative_atoms])
    negative_tree = cKDTree(negative_coords)
    
    # Track salt bridges per residue pair
    # Key: ((pos_res_name, pos_res_num, pos_chain), (neg_res_name, neg_res_num, neg_chain))
    # Value: (max pos_atom weight, max neg_atom weight)
    # Each residue pair counts as only one salt bridge, using the maximum weight from any contact
    salt_bridge_pairs = {}
    
    # Find all salt bridges using KD-tree for efficient neighbor search
    for pos_atom in positive_atoms:
        pos_coord = np.array([pos_atom.x, pos_atom.y, pos_atom.z])
        
        # Get residue key for positive atom (4-tuple with insertion code)
        pos_key = residue_key_from_atom(pos_atom)
        
        # Check if positive residue is charged at given pH (pKa keys use '' for insertion)
        pos_pka = pka_data.get(pos_key) or pka_data.get((pos_key[0], pos_key[1], pos_key[2], ''))
        if not is_residue_charged(pos_atom.residue_name, pos_pka, pH):
            continue  # Skip if not charged
        
        # Query all negative atoms within MAX_SALT_BRIDGE_DISTANCE
        indices = negative_tree.query_ball_point(pos_coord, MAX_SALT_BRIDGE_DISTANCE)
        
        for idx in indices:
            neg_atom = negative_atoms[idx]
            
            # Get residue key for negative atom (4-tuple with insertion code)
            neg_key = residue_key_from_atom(neg_atom)
            
            # Check if negative residue is charged at given pH
            neg_pka = pka_data.get(neg_key) or pka_data.get((neg_key[0], neg_key[1], neg_key[2], ''))
            if not is_residue_charged(neg_atom.residue_name, neg_pka, pH):
                continue  # Skip if not charged
            
            # Exclude contacts within the same residue (same 4-tuple key)
            if pos_key == neg_key:
                continue
            
            # Check distance (already filtered by KD-tree, but verify)
            dist = distance(pos_atom, neg_atom)
            if dist > MAX_SALT_BRIDGE_DISTANCE:
                continue
            
            pos_weight = get_sasa_weight(pos_atom, sasa_data, weighting)
            neg_weight = get_sasa_weight(neg_atom, sasa_data, weighting)
            
            # Store as ordered pair (positive first, negative second)
            # Use tuple of tuples for hashability
            pair = (pos_key, neg_key)
            
            # For each residue pair, only count one salt bridge using maximum weight
            if pair in salt_bridge_pairs:
                # Update with maximum weights if this contact has higher weights
                current_pos_weight, current_neg_weight = salt_bridge_pairs[pair]
                salt_bridge_pairs[pair] = (
                    max(current_pos_weight, pos_weight),
                    max(current_neg_weight, neg_weight)
                )
            else:
                # First contact for this residue pair
                salt_bridge_pairs[pair] = (pos_weight, neg_weight)
    
    return dict(salt_bridge_pairs)


def calculate_salt_bridge_density(
    pdb_path: str,
    sasa_path: str,
    weighting: str = "inverse",
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> Tuple[Dict[Tuple[str, int, str], float], Dict[Tuple[str, int, str], bool]]:
    """
    Calculate salt bridge density per residue, weighted by inverse SASA.
    
    Number of salt bridges per residue, weighted by INVERSE relative SASA:
    - Weight = 1 / (total-side REL) if atom is in side chain
    - Weight = 1 / (main-chain REL) if atom is in backbone
    
    Note: Uses INVERSE SASA (1 / relative_ASA), not raw SASA. This means:
    - Buried residues (low SASA) contribute HIGHER weights
    - Exposed residues (high SASA) contribute LOWER weights
    
    The total weight per residue is capped at MAX_WEIGHT_PER_RESIDUE (100.0) to prevent
    residues with multiple buried atoms from accumulating excessive weights.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        weighting: Weighting strategy ("inverse" or "negative_linear")
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)
        
    Returns:
        Tuple of:
        1. Dictionary mapping (residue_name, residue_number, chain) -> weighted salt bridge density
           (weighted by inverse SASA, capped at 100.0 per residue)
        2. Dictionary mapping (residue_name, residue_number, chain) -> bool indicating
           if residue has any inter-chain contacts
    """
    salt_bridges = detect_salt_bridges(pdb_path, sasa_path, weighting, pka_path, pH)
    
    # Aggregate by residue with weight capping
    residue_weights = defaultdict(float)
    residue_inter_chain = defaultdict(bool)
    
    for (pos_key, neg_key), (pos_weight, neg_weight) in salt_bridges.items():
        # Check if this is an inter-chain contact
        is_inter_chain = (pos_key[2] != neg_key[2])
        
        # Each residue gets the weight based on its own atom's location
        # Add weights and clamp to maximum per residue
        residue_weights[pos_key] = min(
            residue_weights[pos_key] + pos_weight,
            MAX_WEIGHT_PER_RESIDUE
        )
        residue_weights[neg_key] = min(
            residue_weights[neg_key] + neg_weight,
            MAX_WEIGHT_PER_RESIDUE
        )
        
        # Mark residues as having inter-chain contacts if applicable
        if is_inter_chain:
            residue_inter_chain[pos_key] = True
            residue_inter_chain[neg_key] = True
    
    return dict(residue_weights), dict(residue_inter_chain)


def calculate_salt_bridge_density_average(
    pdb_path: str,
    sasa_path: str,
    weighting: str = "inverse",
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> float:
    """
    Calculate average salt bridge density across all residues.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        weighting: Weighting strategy ("inverse" or "negative_linear")
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)
        
    Returns:
        Average weighted salt bridge density per residue
    """
    residue_densities, _ = calculate_salt_bridge_density(pdb_path, sasa_path, weighting, pka_path, pH)
    
    if len(residue_densities) == 0:
        return 0.0
    
    return sum(residue_densities.values()) / len(residue_densities)


# ============================================================================
# Unweighted Salt-Bridge Detection Functions (no SASA weighting)
# ============================================================================

def detect_salt_bridges_unweighted(
    pdb_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> Dict[Tuple[Tuple[str, int, str], Tuple[str, int, str]], int]:
    """
    Detect salt bridges between residue pairs without SASA weighting.
    
    A salt bridge is formed when any charged atom from a positively charged residue
    (ARG, LYS) is within MAX_SALT_BRIDGE_DISTANCE (4.0 Å) of any charged atom from
    a negatively charged residue (ASP, GLU).
    
    Residues are only considered charged if their pKa values (from pKa file) indicate
    they are charged at the specified pH:
    - ASP/GLU: charged when pKa < pH
    - LYS/ARG: charged when pKa > pH
    
    Each residue pair counts as only one salt bridge, regardless of how many
    atom-atom contacts exist between them.
    
    Args:
        pdb_path: Path to PDB structure file
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)
        
    Returns:
        Dictionary mapping ((pos_res_name, pos_res_num, pos_chain), (neg_res_name, neg_res_num, neg_chain)) -> count
        where count is always 1 per pair.
        Keys are ordered such that the positive residue comes first.
    """
    # Parse PDB file
    atoms = parse_pdb(pdb_path)
    
    # Parse pKa file if available
    pka_data = {}
    if pka_path is None:
        # Try to auto-detect pKa file path
        pka_path = get_pka_file_path(pdb_path)
    
    if pka_path and os.path.exists(pka_path):
        pka_data = parse_pka(pka_path, atoms)
    
    # Identify positively and negatively charged atoms
    positive_atoms = [atom for atom in atoms if is_positively_charged(atom)]
    negative_atoms = [atom for atom in atoms if is_negatively_charged(atom)]
    
    # Build KD-tree for negative atoms for efficient neighbor search
    if len(negative_atoms) == 0 or len(positive_atoms) == 0:
        return {}
    
    negative_coords = np.array([[a.x, a.y, a.z] for a in negative_atoms])
    negative_tree = cKDTree(negative_coords)
    
    # Track salt bridges per residue pair (using set to ensure uniqueness)
    salt_bridge_pairs = set()
    
    # Find all salt bridges using KD-tree for efficient neighbor search
    for pos_atom in positive_atoms:
        pos_coord = np.array([pos_atom.x, pos_atom.y, pos_atom.z])
        
        # Get residue key for positive atom (4-tuple with insertion code)
        pos_key = residue_key_from_atom(pos_atom)
        
        # Check if positive residue is charged at given pH (pKa keys use '' for insertion)
        pos_pka = pka_data.get(pos_key) or pka_data.get((pos_key[0], pos_key[1], pos_key[2], ''))
        if not is_residue_charged(pos_atom.residue_name, pos_pka, pH):
            continue  # Skip if not charged
        
        # Query all negative atoms within MAX_SALT_BRIDGE_DISTANCE
        indices = negative_tree.query_ball_point(pos_coord, MAX_SALT_BRIDGE_DISTANCE)
        
        for idx in indices:
            neg_atom = negative_atoms[idx]
            
            # Get residue key for negative atom (4-tuple with insertion code)
            neg_key = residue_key_from_atom(neg_atom)
            
            # Check if negative residue is charged at given pH
            neg_pka = pka_data.get(neg_key) or pka_data.get((neg_key[0], neg_key[1], neg_key[2], ''))
            if not is_residue_charged(neg_atom.residue_name, neg_pka, pH):
                continue  # Skip if not charged
            
            # Exclude contacts within the same residue (same 4-tuple key)
            if pos_key == neg_key:
                continue
            
            # Check distance (already filtered by KD-tree, but verify)
            dist = distance(pos_atom, neg_atom)
            if dist > MAX_SALT_BRIDGE_DISTANCE:
                continue
            
            # Store as ordered pair (positive first, negative second)
            pair = (pos_key, neg_key)
            salt_bridge_pairs.add(pair)
    
    # Convert set to dictionary with count = 1 for each pair
    return {pair: 1 for pair in salt_bridge_pairs}


def calculate_salt_bridge_density_unweighted(
    pdb_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> Dict[Tuple[str, int, str], int]:
    """
    Calculate salt bridge density per residue without SASA weighting.
    
    Counts the number of salt bridges each residue participates in.
    Each residue pair contributes 1 to both residues.
    
    Args:
        pdb_path: Path to PDB structure file
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> salt bridge count
    """
    salt_bridges = detect_salt_bridges_unweighted(pdb_path, pka_path, pH)
    
    # Aggregate by residue
    residue_counts = defaultdict(int)
    
    for (pos_key, neg_key), count in salt_bridges.items():
        # Each residue in the pair gets the count
        residue_counts[pos_key] += count
        residue_counts[neg_key] += count
    
    return dict(residue_counts)


def calculate_salt_bridge_density_average_unweighted(
    pdb_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.4
) -> float:
    """
    Calculate average salt bridge density across all residues without SASA weighting.
    
    Args:
        pdb_path: Path to PDB structure file
        pka_path: Optional path to pKa file. If None, will try to auto-detect from pdb_path.
        pH: pH value for charge state determination (default: 7.4)
        
    Returns:
        Average salt bridge count per residue
    """
    residue_counts = calculate_salt_bridge_density_unweighted(pdb_path, pka_path, pH)
    
    if len(residue_counts) == 0:
        return 0.0
    
    return sum(residue_counts.values()) / len(residue_counts)


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


def calculate_aromatic_density(
    pdb_path: str,
    sasa_path: str,
    weighting: str = "inverse"
) -> Dict[Tuple[str, int, str], float]:
    """
    Calculate aromatic residue density per residue, weighted by inverse SASA.
    
    For each Phe, Tyr, or Trp residue, counts it as 1 * inverse_SASA.
    Weight = 1 / (total-side REL) since aromatic residues are side-chain residues.
    
    Note: Uses INVERSE SASA (1 / relative_ASA), not raw SASA. This means:
    - Buried residues (low SASA) contribute HIGHER weights
    - Exposed residues (high SASA) contribute LOWER weights
    
    The total weight per residue is capped at MAX_WEIGHT_PER_RESIDUE (100.0) to prevent
    residues from accumulating excessive weights.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> weighted aromatic density
        (weighted by inverse SASA, capped at 100.0 per residue)
        Only includes Phe, Tyr, and Trp residues
    """
    # Parse files
    atoms = parse_pdb(pdb_path)
    sasa_data = parse_sasa(sasa_path)
    
    # Track aromatic residues and their weights
    # Key: (residue_name, residue_number, chain)
    # Value: weight (1 * inverse_SASA)
    aromatic_weights = {}
    
    # Find all aromatic residues
    seen_residues = set()
    for atom in atoms:
        residue_key = residue_key_from_atom(atom)
        
        # Skip if we've already processed this residue
        if residue_key in seen_residues:
            continue
        
        # Check if residue is aromatic
        if is_aromatic_residue(atom.residue_name):
            seen_residues.add(residue_key)
            
            # Get SASA weight for this residue (use side-chain SASA)
            weight = get_residue_sasa_weight(residue_key, sasa_data, use_side_chain=True, weighting=weighting)
            
            # Each aromatic residue contributes 1 * inverse_SASA
            # Cap at maximum per residue
            aromatic_weights[residue_key] = min(weight, MAX_WEIGHT_PER_RESIDUE)
    
    return aromatic_weights


def calculate_aromatic_density_average(
    pdb_path: str,
    sasa_path: str,
    weighting: str = "inverse"
) -> float:
    """
    Calculate average aromatic residue density across all residues.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        
    Returns:
        Average weighted aromatic density per residue (only counting Phe, Tyr, Trp)
    """
    residue_densities = calculate_aromatic_density(pdb_path, sasa_path, weighting)
    
    if len(residue_densities) == 0:
        return 0.0
    
    return sum(residue_densities.values()) / len(residue_densities)


def calculate_aromatic_density_unweighted(
    pdb_path: str
) -> Dict[Tuple[str, int, str], int]:
    """
    Calculate aromatic residue density per residue without SASA weighting.
    
    Counts each Phe, Tyr, or Trp residue as 1.
    
    Args:
        pdb_path: Path to PDB structure file
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> count (always 1 for aromatic residues)
        Only includes Phe, Tyr, and Trp residues
    """
    # Parse PDB file
    atoms = parse_pdb(pdb_path)
    
    # Track aromatic residues
    aromatic_residues = {}
    
    # Find all aromatic residues
    seen_residues = set()
    for atom in atoms:
        residue_key = residue_key_from_atom(atom)
        
        # Skip if we've already processed this residue
        if residue_key in seen_residues:
            continue
        
        # Check if residue is aromatic
        if is_aromatic_residue(atom.residue_name):
            seen_residues.add(residue_key)
            # Each aromatic residue counts as 1
            aromatic_residues[residue_key] = 1
    
    return aromatic_residues


def calculate_aromatic_density_average_unweighted(
    pdb_path: str
) -> float:
    """
    Calculate average aromatic residue density across all residues without SASA weighting.
    
    Args:
        pdb_path: Path to PDB structure file
        
    Returns:
        Average aromatic residue count per residue (only counting Phe, Tyr, Trp)
    """
    residue_counts = calculate_aromatic_density_unweighted(pdb_path)
    
    if len(residue_counts) == 0:
        return 0.0
    
    return sum(residue_counts.values()) / len(residue_counts)


# ============================================================================
# Weighted Contact Number (WCN) Functions
# ============================================================================

def calculate_weighted_contact_number(
    pdb_path: str
) -> Dict[Tuple[str, int, str], float]:
    """
    Calculate Weighted Contact Number (WCN) per residue.
    
    For a residue i, the weighted contact number is defined as the inverse-distance-weighted
    sum over all other residues j:
    
    WCN_i = Σ(j≠i) 1/(r_ij^2)
    
    where:
    - r_ij is the Euclidean distance between Cα atoms of residues i and j
    - The exponent 2 reflects the empirical decay of interaction influence with distance
    
    Args:
        pdb_path: Path to PDB structure file
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> WCN value (float)
    """
    # Parse PDB file
    atoms = parse_pdb(pdb_path)
    
    # Extract Cα atoms only
    ca_atoms = [atom for atom in atoms if atom.name == "CA"]
    
    if len(ca_atoms) == 0:
        return {}
    
    # Build mapping from residue key to Cα atom
    ca_dict = {}
    for atom in ca_atoms:
        residue_key = residue_key_from_atom(atom)
        ca_dict[residue_key] = atom
    
    # Calculate WCN for each residue
    wcn_values = {}
    
    # Convert to numpy arrays for efficient distance calculation
    residue_keys = list(ca_dict.keys())
    ca_coords = np.array([[ca_dict[key].x, ca_dict[key].y, ca_dict[key].z] 
                          for key in residue_keys])
    
    # Calculate pairwise distances using numpy broadcasting
    # This is O(N^2) but vectorized, so it's reasonably fast for typical protein sizes
    for i, residue_key_i in enumerate(residue_keys):
        # Get coordinates of residue i
        coord_i = ca_coords[i]
        
        # Calculate distances to all other residues
        # coord_i is shape (3,), ca_coords is shape (N, 3)
        # Broadcasting: (3,) - (N, 3) -> (N, 3)
        diff = ca_coords - coord_i
        distances_sq = np.sum(diff ** 2, axis=1)  # Shape: (N,)
        
        # Calculate WCN: sum of 1/(r_ij^2) for all j ≠ i
        # Exclude residue i itself (where distance = 0)
        wcn = 0.0
        for j in range(len(residue_keys)):
            if i != j:  # j ≠ i
                dist_sq = distances_sq[j]
                if dist_sq > 0:  # Avoid division by zero (shouldn't happen for j≠i, but safety check)
                    wcn += 1.0 / dist_sq
        
        wcn_values[residue_key_i] = wcn
    
    return wcn_values


def calculate_weighted_contact_number_average(
    pdb_path: str
) -> float:
    """
    Calculate average Weighted Contact Number (WCN) across all residues.
    
    Args:
        pdb_path: Path to PDB structure file
        
    Returns:
        Average WCN value per residue
    """
    wcn_values = calculate_weighted_contact_number(pdb_path)
    
    if len(wcn_values) == 0:
        return 0.0
    
    return sum(wcn_values.values()) / len(wcn_values)


# ============================================================================
# DSSP File Parsing Functions
# ============================================================================

def parse_dssp(
    dssp_path: str,
    pdb_atoms: Optional[List[Atom]] = None
) -> Dict[Tuple[str, int, str], Dict[str, Optional[float]]]:
    """
    Parse DSSP file and extract secondary structure and H-bond energy data.
    
    Extracts:
    - Secondary structure (first character from STRUCTURE column)
    - N-H-->O_1, N-H-->O_2 (donor H-bond energies)
    - O-->H-N_1, O-->H-N_2 (acceptor H-bond energies)
    
    Only extracts energy values (the part after the comma in "residue,energy" format).
    
    Args:
        dssp_path: Path to DSSP file
        pdb_atoms: Optional list of atoms from PDB file for amino acid validation
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> {
            'secondary_structure': str (first char of STRUCTURE),
            'N-H-->O_1': float or None,
            'N-H-->O_2': float or None,
            'O-->H-N_1': float or None,
            'O-->H-N_2': float or None
        }
        Only includes residues that match PDB atoms (if provided) with same amino acid.
    """
    # Build mapping from PDB atoms for validation
    pdb_residue_map = {}
    # Use 3-tuple keys (res, num, chain) so DSSP file lookups match; DSSP has no insertion codes
    if pdb_atoms:
        for atom in pdb_atoms:
            key_3 = (atom.residue_name, atom.residue_number, atom.chain)
            if key_3 not in pdb_residue_map:
                pdb_residue_map[key_3] = atom.residue_name
    
    dssp_data = {}
    
    try:
        with open(dssp_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (usually line 28, contains "#  RESIDUE AA STRUCTURE")
        header_line_idx = None
        for i, line in enumerate(lines):
            if '#  RESIDUE AA STRUCTURE' in line:
                header_line_idx = i
                break
        
        if header_line_idx is None:
            # No header found, return empty dict
            return dssp_data
        
        # Parse data lines (starting after header)
        for line in lines[header_line_idx + 1:]:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            
            # Parse DSSP format
            # Format: "    1    1 H Q              0   0  210      0, 0.0     2,-0.3     0, 0.0    26,-0.1"
            # First number (0-4): sequential DSSP number
            # Second number (5-9): PDB residue number (this is what we need!)
            # Use fixed-width for initial fields, then split for H-bond fields
            try:
                # RESIDUE (PDB residue number): columns 5-9 (second number)
                residue_num_str = line[5:10].strip()
                if not residue_num_str:
                    continue
                residue_number = int(residue_num_str)
                
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
                
                # STRUCTURE: starts at column 16, first character is secondary structure
                if len(line) < 17:
                    secondary_structure = ' '
                else:
                    structure_str = line[16:24].strip()
                    secondary_structure = structure_str[0] if structure_str else ' '
                
                # H-bond columns: use regex to find all "residue,energy" patterns
                # H-bonds appear after ACC column (around position 40+)
                import re
                hbond_pattern = r'(-?\d+),\s*(-?\d+\.?\d*)'
                
                # Find all H-bond patterns in the line after column 40
                if len(line) > 40:
                    matches = list(re.finditer(hbond_pattern, line[40:]))
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
                        nh_o_2_str = ''
                        oh_n_2_str = ''
                    else:
                        nh_o_1_str = ''
                        oh_n_1_str = ''
                        nh_o_2_str = ''
                        oh_n_2_str = ''
                else:
                    nh_o_1_str = ''
                    oh_n_1_str = ''
                    nh_o_2_str = ''
                    oh_n_2_str = ''
                
                nh_o_1_energy = _parse_hbond_energy(nh_o_1_str)
                oh_n_1_energy = _parse_hbond_energy(oh_n_1_str)
                nh_o_2_energy = _parse_hbond_energy(nh_o_2_str)
                oh_n_2_energy = _parse_hbond_energy(oh_n_2_str)
                
                residue_key = (residue_name, residue_number, chain)
                
                # Validate amino acid matches PDB if provided
                if pdb_atoms:
                    if residue_key not in pdb_residue_map:
                        # Residue not in PDB, skip
                        continue
                    if pdb_residue_map[residue_key] != residue_name:
                        # Amino acid mismatch, skip
                        continue
                
                dssp_data[residue_key] = {
                    'secondary_structure': secondary_structure,
                    'N-H-->O_1': nh_o_1_energy,
                    'N-H-->O_2': nh_o_2_energy,
                    'O-->H-N_1': oh_n_1_energy,
                    'O-->H-N_2': oh_n_2_energy
                }
            
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
    
    except FileNotFoundError:
        # File not found, return empty dict
        return dssp_data
    except Exception as e:
        # Other errors, return empty dict
        return dssp_data
    
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

# Residue groups for clustering (C-alpha 3D coordinates)
HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "GLY"})  # PRO removed
POLAR_RESIDUES = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "CYS", "GLU", "ASP", "LYS", "ARG", "HIS"})


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
    
    Charge groups use PropKA: negative = ASP/GLU when pKa < pH; positive = ARG/LYS when pKa > pH.
    Hydrophobic = ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, GLY.
    Polar = SER, THR, ASN, GLN, TYR, CYS.
    
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
        return pka_data.get(key) or pka_data.get((key[0], key[1], key[2], ""))
    
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
        if res_name in ("ASP", "GLU") and is_residue_charged(res_name, pka_val, pH):
            neg_keys.append(key)
            neg_coords.append(xyz)
        elif res_name in ("ARG", "LYS") and is_residue_charged(res_name, pka_val, pH):
            pos_keys.append(key)
            pos_coords.append(xyz)
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
) -> Dict[Tuple[str, int, str], Dict[str, Optional[float]]]:
    """
    Parse SASA file and extract Total-Side REL and Main-Chain REL values for output.
    
    Extracts:
    - Total-Side REL (as percentage, not fraction)
    - Main-Chain REL (as percentage, not fraction)
    
    Args:
        sasa_path: Path to SASA file
        pdb_atoms: Optional list of atoms from PDB file for validation
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> {
            'total_side_rel': float (percentage) or None,
            'main_chain_rel': float (percentage) or None
        }
        Only includes residues that match PDB atoms (if provided) with same amino acid.
    """
    # Use 3-tuple keys (res, num, chain) so SASA file lookups match; SASA has no insertion codes
    pdb_residue_map = {}
    if pdb_atoms:
        for atom in pdb_atoms:
            key_3 = (atom.residue_name, atom.residue_number, atom.chain)
            if key_3 not in pdb_residue_map:
                pdb_residue_map[key_3] = atom.residue_name
    
    sasa_output_data = {}
    
    try:
        with open(sasa_path, 'r') as f:
            for line in f:
                if line.startswith('RES '):
                    # Parse RES line
                    # Format: RES RESNAME CHAIN NUM  All-atoms  Total-Side  Main-Chain ...
                    #         ABS REL  ABS REL  ABS REL ...
                    # Example: RES GLN H   1   195.74 109.5 147.09 107.5  48.65 115.8 ...
                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            residue_name = parts[1]
                            chain = parts[2]
                            residue_number = int(parts[3])
                            
                            # Extract Total-Side REL (parts[7]) and Main-Chain REL (parts[9])
                            # These are already in percentage format
                            total_side_rel_str = parts[7] if len(parts) > 7 else ''
                            main_chain_rel_str = parts[9] if len(parts) > 9 else ''
                            
                            # Parse values (handle 'N/A' case)
                            total_side_rel = None
                            if total_side_rel_str and total_side_rel_str != 'N/A':
                                try:
                                    total_side_rel = float(total_side_rel_str)
                                except ValueError:
                                    total_side_rel = None
                            
                            main_chain_rel = None
                            if main_chain_rel_str and main_chain_rel_str != 'N/A':
                                try:
                                    main_chain_rel = float(main_chain_rel_str)
                                except ValueError:
                                    main_chain_rel = None
                            
                            residue_key = (residue_name, residue_number, chain)
                            
                            # Validate amino acid matches PDB if provided
                            if pdb_atoms:
                                if residue_key not in pdb_residue_map:
                                    # Residue not in PDB, skip
                                    continue
                                if pdb_residue_map[residue_key] != residue_name:
                                    # Amino acid mismatch, skip
                                    continue
                            
                            sasa_output_data[residue_key] = {
                                'total_side_rel': total_side_rel,
                                'main_chain_rel': main_chain_rel
                            }
                        
                        except (ValueError, IndexError) as e:
                            # Skip malformed lines
                            continue
    
    except FileNotFoundError:
        # File not found, return empty dict
        return sasa_output_data
    except Exception as e:
        # Other errors, return empty dict
        return sasa_output_data
    
    return sasa_output_data


# Each hydrogen bond is counted once.
# Each donor-acceptor pair is checked once (we iterate for donor in donors, then for acceptor in acceptors, so we don't check acceptor-donor pairs separately).
# Each bond contributes to both residues:
# Donor residue gets donor_weight (based on donor atom's SASA)
# Acceptor residue gets acceptor_weight (based on acceptor atom's SASA)
# We don't skip residues — we ensure each donor-acceptor pair is only evaluated once in the nested loop.
# So a single bond adds to both residues, but with different weights. No double-counting of the same bond for the same residue.
def calculate_global_hbond_density(
    pdb_path: str,
    sasa_path: str,
    weighting: str = "inverse"
) -> Tuple[Dict[Tuple[str, int, str], float], Dict[Tuple[str, int, str], bool], Dict[Tuple[str, int, str], int]]:
    """
    Calculate global hydrogen bond density per residue.
    
    Number of hydrogen bonds per residue, weighted by INVERSE relative SASA:
    - Weight = 1 / (total-side REL) if atom is in side chain
    - Weight = 1 / (main-chain REL) if atom is in backbone
    
    Note: Uses INVERSE SASA (1 / relative_ASA), not raw SASA. This means:
    - Buried residues (low SASA) contribute HIGHER weights
    - Exposed residues (high SASA) contribute LOWER weights
    
    Each hydrogen bond is counted once and contributes to both residues involved.
    All H-bonds between a residue pair are accumulated (no limit per pair).
    
    The total weight per residue is capped at MAX_WEIGHT_PER_RESIDUE (100.0) to prevent
    residues with multiple buried atoms (e.g., both backbone and side-chain) from
    accumulating excessive weights.
    
    Uses scipy's cKDTree for efficient O(N log N) neighbor search instead of O(N²).
    KD-tree correctly uses [x, y, z] coordinates for spatial queries.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        weighting: Weighting strategy ("inverse" or "negative_linear")
        
    Returns:
        Tuple of:
        1. Dictionary mapping (residue_name, residue_number, chain) -> weighted HB density
           (weighted by inverse SASA, capped at 100.0 per residue)
        2. Dictionary mapping (residue_name, residue_number, chain) -> bool indicating
           if residue has any inter-chain contacts
        3. Dictionary mapping (residue_name, residue_number, chain) -> number of H-bonds
    """
    # Parse files
    atoms = parse_pdb(pdb_path)
    sasa_data = parse_sasa(sasa_path)
    
    # Identify donors and acceptors
    donors = [atom for atom in atoms if is_donor(atom)]
    acceptors = [atom for atom in atoms if is_acceptor(atom)]
    
    # Build KD-tree for acceptors for efficient neighbor search
    if len(acceptors) == 0:
        return {}, {}
    
    acceptor_coords = np.array([[a.x, a.y, a.z] for a in acceptors])
    acceptor_tree = cKDTree(acceptor_coords)
    
    # Build cache for backbone base atoms (performance optimization for large systems)
    # Maps (chain, residue_number) -> base C atom for backbone N donors
    backbone_base_cache: Dict[Tuple[str, int], Atom] = {}
    
    # Track weighted hydrogen bonds per residue
    # Key: (residue_name, residue_number, chain)
    # Value: sum of weighted hydrogen bonds
    residue_hbonds = defaultdict(float)
    residue_inter_chain = defaultdict(bool)
    residue_hbond_count = defaultdict(int)
    
    # Find all hydrogen bonds using KD-tree for efficient neighbor search
    for donor in donors:
        donor_coord = np.array([donor.x, donor.y, donor.z])
        
        # Query all acceptors within MAX_HBOND_DISTANCE
        # KD-tree uses [x, y, z] coordinates correctly
        indices = acceptor_tree.query_ball_point(donor_coord, MAX_HBOND_DISTANCE)
        
        for idx in indices:
            acceptor = acceptors[idx]
            
            # Don't form bonds within the same residue
            if (donor.residue_name == acceptor.residue_name and
                donor.residue_number == acceptor.residue_number and
                donor.chain == acceptor.chain):
                continue
            
            # DSSP-style: Exclude trivial local backbone H-bonds
            # Skip backbone-backbone bonds where |res_i - res_j| < 3
            if (is_backbone_atom(donor.name) and is_backbone_atom(acceptor.name) and
                donor.chain == acceptor.chain):
                res_sep = abs(donor.residue_number - acceptor.residue_number)
                if res_sep < MIN_BACKBONE_SEPARATION:
                    continue
            
            # Check if hydrogen bond exists (angle check)
            # Pass cache for performance
            if check_hydrogen_bond(donor, acceptor, atoms, backbone_base_cache):
                # Check if this is an inter-chain contact
                is_inter_chain = (donor.chain != acceptor.chain)
                
                # Get weights for each atom based on its location (inverse SASA)
                donor_weight = get_sasa_weight(donor, sasa_data, weighting)
                acceptor_weight = get_sasa_weight(acceptor, sasa_data, weighting)
                
                # Each residue gets the weight based on its own atom's location
                donor_key = residue_key_from_atom(donor)
                acceptor_key = residue_key_from_atom(acceptor)
                
                # Add weights and clamp to maximum per residue
                residue_hbonds[donor_key] = min(
                    residue_hbonds[donor_key] + donor_weight,
                    MAX_WEIGHT_PER_RESIDUE
                )
                residue_hbonds[acceptor_key] = min(
                    residue_hbonds[acceptor_key] + acceptor_weight,
                    MAX_WEIGHT_PER_RESIDUE
                )
                
                # Mark residues as having inter-chain contacts if applicable
                if is_inter_chain:
                    residue_inter_chain[donor_key] = True
                    residue_inter_chain[acceptor_key] = True
                
                # Count number of H-bonds per residue (each bond contributes 1 to both)
                residue_hbond_count[donor_key] += 1
                residue_hbond_count[acceptor_key] += 1
    
    return dict(residue_hbonds), dict(residue_inter_chain), dict(residue_hbond_count)


def calculate_global_hbond_density_average(
    pdb_path: str,
    sasa_path: str,
    weighting: str = "inverse"
) -> float:
    """
    Calculate average global hydrogen bond density across all residues.
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        weighting: Weighting strategy ("inverse" or "negative_linear")
        
    Returns:
        Average weighted hydrogen bond density per residue
    """
    residue_densities, _, _ = calculate_global_hbond_density(pdb_path, sasa_path, weighting)
    
    if len(residue_densities) == 0:
        return 0.0
    
    return sum(residue_densities.values()) / len(residue_densities)


# ============================================================================
# Unweighted Hydrogen Bond Detection Functions (no SASA weighting)
# ============================================================================

def calculate_global_hbond_density_unweighted(
    pdb_path: str
) -> Dict[Tuple[str, int, str], int]:
    """
    Calculate global hydrogen bond density per residue without SASA weighting.
    
    Counts the number of hydrogen bonds per residue. Each bond contributes 1 to
    both the donor and acceptor residues.
    
    Uses scipy's cKDTree for efficient O(N log N) neighbor search instead of O(N²).
    KD-tree correctly uses [x, y, z] coordinates for spatial queries.
    
    Args:
        pdb_path: Path to PDB structure file
        
    Returns:
        Dictionary mapping (residue_name, residue_number, chain) -> H-bond count
    """
    # Parse PDB file
    atoms = parse_pdb(pdb_path)
    
    # Identify donors and acceptors
    donors = [atom for atom in atoms if is_donor(atom)]
    acceptors = [atom for atom in atoms if is_acceptor(atom)]
    
    # Build KD-tree for acceptors for efficient neighbor search
    if len(acceptors) == 0:
        return {}
    
    acceptor_coords = np.array([[a.x, a.y, a.z] for a in acceptors])
    acceptor_tree = cKDTree(acceptor_coords)
    
    # Build cache for backbone base atoms (performance optimization for large systems)
    # Maps (chain, residue_number) -> base C atom for backbone N donors
    backbone_base_cache: Dict[Tuple[str, int], Atom] = {}
    
    # Track hydrogen bonds per residue
    # Key: (residue_name, residue_number, chain)
    # Value: count of hydrogen bonds
    residue_hbonds = defaultdict(int)
    
    # Find all hydrogen bonds using KD-tree for efficient neighbor search
    for donor in donors:
        donor_coord = np.array([donor.x, donor.y, donor.z])
        
        # Query all acceptors within MAX_HBOND_DISTANCE
        # KD-tree uses [x, y, z] coordinates correctly
        indices = acceptor_tree.query_ball_point(donor_coord, MAX_HBOND_DISTANCE)
        
        for idx in indices:
            acceptor = acceptors[idx]
            
            # Don't form bonds within the same residue
            if (donor.residue_name == acceptor.residue_name and
                donor.residue_number == acceptor.residue_number and
                donor.chain == acceptor.chain):
                continue
            
            # DSSP-style: Exclude trivial local backbone H-bonds
            # Skip backbone-backbone bonds where |res_i - res_j| < 3
            if (is_backbone_atom(donor.name) and is_backbone_atom(acceptor.name) and
                donor.chain == acceptor.chain):
                res_sep = abs(donor.residue_number - acceptor.residue_number)
                if res_sep < MIN_BACKBONE_SEPARATION:
                    continue
            
            # Check if hydrogen bond exists (angle check)
            # Pass cache for performance
            if check_hydrogen_bond(donor, acceptor, atoms, backbone_base_cache):
                # Each residue gets a count of 1 for this bond
                donor_key = residue_key_from_atom(donor)
                acceptor_key = residue_key_from_atom(acceptor)
                
                residue_hbonds[donor_key] += 1
                residue_hbonds[acceptor_key] += 1
    
    return dict(residue_hbonds)


def calculate_global_hbond_density_average_unweighted(
    pdb_path: str
) -> float:
    """
    Calculate average global hydrogen bond density across all residues without SASA weighting.
    
    Args:
        pdb_path: Path to PDB structure file
        
    Returns:
        Average hydrogen bond count per residue
    """
    residue_densities = calculate_global_hbond_density_unweighted(pdb_path)
    
    if len(residue_densities) == 0:
        return 0.0
    
    return sum(residue_densities.values()) / len(residue_densities)


def largest_hbond_component_size(pdb_path: str) -> int:
    """
    Size of the largest connected component of the H-bond network (geometry-based).

    Uses the same H-bond detection as calculate_global_hbond_density_unweighted
    (distance + angle criteria). Builds an undirected graph of residue pairs connected
    by H-bonds, then returns the number of residues in the largest connected component.

    Args:
        pdb_path: Path to PDB structure file.

    Returns:
        Number of residues in the largest connected component (0 if no H-bonds).
    """
    atoms = parse_pdb(pdb_path)
    donors = [atom for atom in atoms if is_donor(atom)]
    acceptors = [atom for atom in atoms if is_acceptor(atom)]

    if len(acceptors) == 0:
        return 0

    acceptor_coords = np.array([[a.x, a.y, a.z] for a in acceptors])
    acceptor_tree = cKDTree(acceptor_coords)
    backbone_base_cache: Dict[Tuple[str, int], Atom] = {}

    edges: Set[frozenset] = set()
    for donor in donors:
        donor_coord = np.array([donor.x, donor.y, donor.z])
        indices = acceptor_tree.query_ball_point(donor_coord, MAX_HBOND_DISTANCE)
        for idx in indices:
            acceptor = acceptors[idx]
            if (donor.residue_name == acceptor.residue_name and
                donor.residue_number == acceptor.residue_number and
                donor.chain == acceptor.chain):
                continue
            if (is_backbone_atom(donor.name) and is_backbone_atom(acceptor.name) and
                donor.chain == acceptor.chain):
                res_sep = abs(donor.residue_number - acceptor.residue_number)
                if res_sep < MIN_BACKBONE_SEPARATION:
                    continue
            if check_hydrogen_bond(donor, acceptor, atoms, backbone_base_cache):
                donor_key = residue_key_from_atom(donor)
                acceptor_key = residue_key_from_atom(acceptor)
                edge = frozenset({donor_key, acceptor_key})
                edges.add(edge)

    if not edges:
        return 0

    # Union-Find to get connected components
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
) -> Tuple[Dict[Tuple[str, int, str], List[Tuple[int, float]]], Dict[int, Tuple[str, int, str]]]:
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
        1. Dictionary mapping (residue_name, residue_number, chain) -> list of (residue_offset, energy) tuples
        2. Dictionary mapping sequential_DSSP_number -> (residue_name, residue_number, chain)
        Only includes residues that match PDB atoms (if provided) with same amino acid.
    """
    # Use 3-tuple keys (res, num, chain) so DSSP file lookups match
    pdb_residue_map = {}
    if pdb_atoms:
        for atom in pdb_atoms:
            key_3 = (atom.residue_name, atom.residue_number, atom.chain)
            if key_3 not in pdb_residue_map:
                pdb_residue_map[key_3] = atom.residue_name
    
    hbond_data = {}
    dssp_seq_to_pdb: Dict[int, Tuple[str, int, str]] = {}
    
    try:
        with open(dssp_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (usually line 28, contains "#  RESIDUE AA STRUCTURE")
        header_line_idx = None
        for i, line in enumerate(lines):
            if '#  RESIDUE AA STRUCTURE' in line:
                header_line_idx = i
                break
        
        if header_line_idx is None:
            # No header found, return empty dicts
            return hbond_data, dssp_seq_to_pdb
        
        # Parse data lines (starting after header)
        for line in lines[header_line_idx + 1:]:
            line = line.rstrip('\n')
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
                
                # RESIDUE (PDB residue number): columns 5-9 (second number)
                residue_num_str = line[5:10].strip()
                if not residue_num_str:
                    continue
                residue_number = int(residue_num_str)
                
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
                
                residue_key = (residue_name, residue_number, chain)
                
                # Validate amino acid matches PDB if provided
                if pdb_atoms:
                    if residue_key not in pdb_residue_map:
                        # Residue not in PDB, skip
                        continue
                    if pdb_residue_map[residue_key] != residue_name:
                        # Amino acid mismatch, skip
                        continue
                
                # Map sequential DSSP number to PDB residue key
                dssp_seq_to_pdb[dssp_seq_num] = residue_key
                
                # H-bond columns: use regex to find all "residue,energy" patterns
                # H-bonds appear after ACC column (around position 40+)
                # Offsets are relative to sequential DSSP number
                import re
                hbond_pattern = r'(-?\d+),\s*(-?\d+\.?\d*)'
                
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
            
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
    
    except FileNotFoundError:
        # File not found, return empty dicts
        return hbond_data, dssp_seq_to_pdb
    except Exception as e:
        # Other errors, return empty dicts
        return hbond_data, dssp_seq_to_pdb
    
    return hbond_data, dssp_seq_to_pdb


def calculate_hbond_energy_density_dssp_backbone_only(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
    weighting: str = "inverse"
) -> Tuple[Dict[Tuple[str, int, str], float], Dict[Tuple[str, int, str], bool]]:
    """
    Calculate hydrogen bond energy density per residue using DSSP data (backbone only).
    
    Detects H-bonds from DSSP file (N-H-->O and O-->H-N columns) and weights them
    by INVERSE main-chain relative SASA. This is for backbone/main-chain H-bonds only
    since DSSP primarily reports backbone H-bonds.
    
    Weighting:
    - Weight = 1 / (main-chain REL) for each residue
    - SASA values below 1% are clipped to 1% (minimum weight = 1 / 0.01 = 100)
    - No maximum cap per residue (unlike regular H-bond density)
    
    Each H-bond is counted for both residues involved (donor and acceptor),
    weighted by each residue's own main-chain SASA.
    
    Args:
        pdb_path: Path to PDB structure file (for residue mapping)
        sasa_path: Path to SASA file (for main-chain REL values)
        dssp_path: Path to DSSP file (for H-bond detection)
        weighting: Weighting strategy ("inverse" or "negative_linear")
        
    Returns:
        Tuple of:
        1. Dictionary mapping (residue_name, residue_number, chain) -> weighted HB density
           (weighted by inverse main-chain SASA, no maximum cap)
        2. Dictionary mapping (residue_name, residue_number, chain) -> bool indicating
           if residue has any inter-chain contacts
    """
    # Parse files
    atoms = parse_pdb(pdb_path)
    sasa_data = parse_sasa(sasa_path)
    dssp_hbonds, dssp_seq_to_pdb = parse_dssp_hbonds(dssp_path, atoms)
    
    # Build mapping from PDB residue key to sequential DSSP number
    # This helps us find the sequential DSSP number for each residue
    pdb_to_dssp_seq: Dict[Tuple[str, int, str], int] = {}
    for dssp_seq, pdb_key in dssp_seq_to_pdb.items():
        pdb_to_dssp_seq[pdb_key] = dssp_seq
    
    # Track weighted hydrogen bonds per residue
    # Key: (residue_name, residue_number, chain)
    # Value: sum of weighted hydrogen bonds
    residue_hbonds = defaultdict(float)
    residue_inter_chain = defaultdict(bool)
    
    # Process each residue's H-bonds from DSSP
    for residue_key, hbond_pairs in dssp_hbonds.items():
        # Get sequential DSSP number for this residue
        if residue_key not in pdb_to_dssp_seq:
            # Residue not in DSSP mapping, skip
            continue
        
        dssp_seq_num = pdb_to_dssp_seq[residue_key]
        
        # Get weight for this residue (main-chain only)
        residue_weight = get_residue_sasa_weight(residue_key, sasa_data, use_side_chain=False, weighting=weighting)
        
        # If residue has no SASA data, skip it
        if residue_weight == 0.0:
            continue
        
        # Process each H-bond pair (residue_offset, energy)
        # Offset is relative to sequential DSSP number, not PDB residue number
        for residue_offset, energy in hbond_pairs:
            # Calculate target sequential DSSP number
            target_dssp_seq = dssp_seq_num + residue_offset
            
            # Find target residue key using sequential DSSP number
            if target_dssp_seq not in dssp_seq_to_pdb:
                # Target residue not found, skip this bond
                continue
            
            target_residue_key = dssp_seq_to_pdb[target_dssp_seq]
            
            # Check if this is an inter-chain contact
            is_inter_chain = (residue_key[2] != target_residue_key[2])
            
            # Get weight for target residue (main-chain only)
            target_weight = get_residue_sasa_weight(target_residue_key, sasa_data, use_side_chain=False, weighting=weighting)
            
            # If target residue has no SASA data, skip it
            if target_weight == 0.0:
                continue
            
            # Add weights to both residues (each bond contributes to both)
            # No maximum cap for DSSP-based H-bond energy density
            residue_hbonds[residue_key] += residue_weight
            residue_hbonds[target_residue_key] += target_weight
            
            # Mark residues as having inter-chain contacts if applicable
            if is_inter_chain:
                residue_inter_chain[residue_key] = True
                residue_inter_chain[target_residue_key] = True
    
    return dict(residue_hbonds), dict(residue_inter_chain)


def calculate_hbond_energy_density_dssp_backbone_only_average(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
    weighting: str = "inverse"
) -> float:
    """
    Calculate average hydrogen bond energy density using DSSP data (backbone only).
    
    Args:
        pdb_path: Path to PDB structure file
        sasa_path: Path to SASA file
        dssp_path: Path to DSSP file
        weighting: Weighting strategy ("inverse" or "negative_linear")
        
    Returns:
        Average weighted hydrogen bond density per residue
    """
    residue_densities, _ = calculate_hbond_energy_density_dssp_backbone_only(
        pdb_path, sasa_path, dssp_path, weighting
    )
    
    if len(residue_densities) == 0:
        return 0.0
    
    return sum(residue_densities.values()) / len(residue_densities)

