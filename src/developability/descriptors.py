from typing import Dict, Iterable, List, Optional, Set, Tuple
import os
import math
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy import optimize
from sklearn.cluster import DBSCAN
import logging

from developability.structure_context import StructureContext, ResKey4
from developability.descriptor_utils import (
    CDR_RANGES_CA,
    get_residue_region,
    get_residue_region_map,
    iter_unique_residues,
    _get_atoms_for_path,
    get_residue_keys_by_type,
    normalize_hydropathy,
    _residue_fractional_charge_at_pH,
    is_residue_charged,
    atom_sasa_weight,
    get_residue_sasa_weight,
    residue_main_sasa,
    residue_side_sasa,
    _get_structure_context,
    _count_residues_in_pdb,
    _get_residue_seq_index,
    _get_charged_residues_at_pH,
    _RES_INDEX_TO_KEY_CACHE,
    _residue_category_group,
    get_exposed_residues,
    _sasa_lookup,
    is_donor,
    is_acceptor,
    get_donor_base_atom,
    donor_max_hbonds,
    acceptor_max_hbonds,
    _aggregate_dssp_hbond_energy_to_raw,
    _INTER_CHAIN_INTERFACE_CACHE,
    _compute_inter_chain_interface_from_by_chain,
    get_inter_chain_interface_residues)

from utils.parsers import (
    get_sasa_total,
    parse_dssp,
    parse_sasa,
    Atom,
    SASAEntry,
    residue_key_from_atom,
    parse_motif_to_3letter
)
from utils.geometry import (
    is_backbone_atom
)

logger = logging.getLogger(__name__)

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
    ("ND1","HIS"),
    ("NE2","HIS")
})

NEGATIVE_ATOMS = frozenset({
    ("OD1", "ASP"),
    ("OD2", "ASP"),
    ("OE1", "GLU"),
    ("OE2", "GLU"),
})

NEGATIVE_CHARGED_RESIDUES = frozenset({res for _atom, res in NEGATIVE_ATOMS})
POSITIVE_CHARGED_RESIDUES = frozenset({res for _atom, res in POSITIVE_ATOMS})
POLAR_RESIDUES = frozenset({"SER", "THR", "ASN", "GLN", "TYR", "GLU", "ASP", "LYS", "ARG", "HIS"})
AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP"})
HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS"})


MAX_HBOND_DISTANCE = 3.2
MAX_SALT_BRIDGE_DISTANCE = 4.0
INTER_CHAIN_INTERFACE_CUTOFF = 5.0
MIN_HBOND_ANGLE = 120.0 # angle base -> donor -> acceptor
MIN_BACKBONE_SEPARATION = 3
NTERM_PKA = 8.0
CTERM_PKA = 3.1

_HBOND_PAIRS_CACHE: Dict[Tuple[str, float, float, int], List[Tuple[Atom, Atom]]] = {}
_PDB_SEQUENCE_CACHE: Dict[str, Dict[str, List[str]]] = {}
SCM_MAIN_CHAIN_ATOMS = frozenset({"CA", "HA", "N", "C", "O", "HN", "H"})

# Charge

def net_charge_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float
) -> Optional[float]:

    if not pka_data:
        return None
    net = 0.0
    for key, pka in pka_data.items():
        res_name = key[0]
        net += _residue_fractional_charge_at_pH(res_name, pka, pH)

    chains = {key[2] for key in pka_data.keys()}
    n_chains = len(chains)
    if n_chains > 0:
        # N-terminus
        net += n_chains * (1.0 / (1.0 + np.power(10.0, pH - NTERM_PKA)))
        # C-terminus
        net -= n_chains * (1.0 / (1.0 + np.power(10.0, CTERM_PKA - pH)))

    return float(net)

def pi_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float]
) -> Optional[float]:
    if not pka_data:
        return None

    def charge_at_pH(pH: float) -> float:
        q = net_charge_from_pka(pka_data, pH)
        return q if q is not None else 0.0

    try:
        q0 = charge_at_pH(0.0)
        q14 = charge_at_pH(14.0)
        if q0 * q14 > 0:
            return None
        return float(optimize.brentq(charge_at_pH, 0.0, 14.0))
    except (ValueError, Exception):
        return None

def compute_dipole_moment_magnitude(
    pdb_atoms: List[Atom],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
) -> Optional[float]:
    """
    Dipole moment magnitude from C-alpha positions and pH-dependent charges (q = ±1).
    μ = Σ qi*ri over charged residues, |μ| = sqrt(μx² + μy² + μz²).
    """
    ca_dipole: Dict[ResKey4, Tuple[float, float, float]] = {}
    for atom in pdb_atoms:
        if atom.name != "CA":
            continue
        key = residue_key_from_atom(atom)
        ca_dipole[key] = (atom.x, atom.y, atom.z)

    def get_pka(k: ResKey4) -> Optional[float]:
        return pka_output_data.get(k) or pka_output_data.get((k[0], k[1], k[2], ""))

    mux, muy, muz = 0.0, 0.0, 0.0
    for key, (x, y, z) in ca_dipole.items():
        res_name = key[0]
        pka_val = get_pka(key)
        if res_name in NEGATIVE_CHARGED_RESIDUES and is_residue_charged(res_name, pka_val, pH):
            mux -= x
            muy -= y
            muz -= z
        elif res_name in POSITIVE_CHARGED_RESIDUES and is_residue_charged(res_name, pka_val, pH):
            mux += x
            muy += y
            muz += z
    return math.sqrt(mux * mux + muy * muy + muz * muz)

# Spatial Charge Map with optional weighting by SASA (default: True)

def scm_score_from_pka(
    pdb_path: str,
    sasa_path: str,
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    d_cutoff: float = 10.0,
    sasa_cutoff: float = 0.25,
    sasa_weighting: bool = True,
) -> Optional[float]:

    from utils.parsers import parse_sasa

    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        return None

    try:
        sasa_data = parse_sasa(sasa_path)
    except Exception as e:
        logger.warning("SCM: failed to parse SASA %s: %s", sasa_path, e)
        return None

    try:
        n = len(atoms)
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        
        is_sidechain = np.array(
            [a.name.strip() not in SCM_MAIN_CHAIN_ATOMS for a in atoms],
            dtype=bool,
        )
        
        key4_list: List[Tuple[str, int, str, str]] = [
            residue_key_from_atom(a) for a in atoms
        ]

        residue_exposure = get_exposed_residues(sasa_data, sasa_cutoff)
        residue_exposed = np.array(
            [residue_exposure.get(key4, False) for key4 in key4_list],
            dtype=bool,
        )

        residue_sasa: Dict[Tuple[str, int, str, str], float] = {}
        for key4 in key4_list:
            entry = sasa_data.get(key4)
            residue_sasa[key4] = float(getattr(entry, "total_side_rel", 0.0) or 0.0)
        
        residue_charge: Dict[Tuple[str, int, str, str], float] = {}
        residue_charged_atom_count: Dict[Tuple[str, int, str, str], int] = {}
        for i, (a, key4) in enumerate(zip(atoms, key4_list)):
            if key4 not in residue_charge:
                residue_charge[key4] = _residue_fractional_charge_at_pH(
                    a.residue_name, pka_data.get(key4), pH
                )
            if is_sidechain[i] and (
                (a.name, a.residue_name) in POSITIVE_ATOMS
                or (a.name, a.residue_name) in NEGATIVE_ATOMS
            ):
                residue_charged_atom_count[key4] = (
                    residue_charged_atom_count.get(key4, 0) + 1
                )
        
        atom_charge = np.zeros(n, dtype=np.float64)
        atom_sasa_share = np.zeros(n, dtype=np.float64)
        for i, (a, key4) in enumerate(zip(atoms, key4_list)):
            if not is_sidechain[i]:
                continue
            if (a.name, a.residue_name) not in POSITIVE_ATOMS and (
                (a.name, a.residue_name) not in NEGATIVE_ATOMS
            ):
                continue
            total_charged = residue_charged_atom_count.get(key4, 0)
            if total_charged <= 0:
                continue
            per_atom_charge = residue_charge.get(key4, 0.0) / float(total_charged)
            atom_charge[i] = per_atom_charge
            atom_sasa_share[i] = residue_sasa.get(key4, 0.0) / float(total_charged)

        valid_center = np.ones(n, dtype=bool)
        is_charged_atom = np.array(
            [
                (a.name, a.residue_name) in POSITIVE_ATOMS
                or (a.name, a.residue_name) in NEGATIVE_ATOMS
                for a in atoms
            ],
            dtype=bool,
        )
        if sasa_weighting:
            valid_source = is_charged_atom
        else:
            valid_source = is_charged_atom & residue_exposed
        tree = cKDTree(coords)
        scm_atom = np.zeros(n, dtype=np.float64)
        for i, j in tree.query_pairs(d_cutoff):
            if valid_center[i] and valid_source[j]:
                contrib_j = atom_charge[j]
                if sasa_weighting:
                    contrib_j *= atom_sasa_share[j]
                scm_atom[i] += contrib_j
            if valid_center[j] and valid_source[i]:
                contrib_i = atom_charge[i]
                if sasa_weighting:
                    contrib_i *= atom_sasa_share[i]
                scm_atom[j] += contrib_i

        neg_sum = np.sum(scm_atom[scm_atom < 0])
        return float(np.abs(neg_sum))
    except Exception as e:
        logger.warning("SCM score computation failed: %s", e, exc_info=True)
        return None


def sum_total_side_rel_within_cutoff(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[Tuple[str, int, str, str], Dict[str, Optional[float]]],
    cutoff: float = 5.0,
    sap_mode: bool = False,  # SAP residue-level approximation (hydrophobicity weighting)
    positive_charge_mode: bool = False,  # weight by positive fractional charge at pH
    negative_charge_mode: bool = False,  # weight by negative fractional charge at pH
    pka_output_data: Optional[Dict[Tuple[str, int, str, str], float]] = None,
    pH: float = 7.0,
) -> float:
    """
    for each residue, sum total_side_rel of all residues
    within cutoff (Cα–Cα distance), weighted by either hydrophobicity or fractional charge at pH
    """
    residue_keys: List[Tuple[str, int, str, str]] = []
    residue_coords: List[Tuple[float, float, float]] = []
    residue_total_side_rel: List[float] = []

    atoms_by_res: Dict[Tuple, List[Atom]] = {}
    ca_by_res: Dict[Tuple, Tuple[float, float, float]] = {}
    for key4 in iter_unique_residues(pdb_atoms):
        residue_keys.append(key4)
        atoms_by_res[key4] = []
    for atom in pdb_atoms:
        key4 = residue_key_from_atom(atom)
        if key4 in atoms_by_res:
            atoms_by_res[key4].append(atom)
            if atom.name.strip() == "CA":
                ca_by_res[key4] = (atom.x, atom.y, atom.z)

    for key4 in residue_keys:
        sasa_entry = sasa_output_data.get(key4) or {}
        val = sasa_entry.get("total_side_rel")
        if val is not None:
            total_side_rel = float(val)
        else:
            raise ValueError(f"No total_side_rel found for residue {key4}")
        residue_total_side_rel.append(total_side_rel)

        ca_coord = ca_by_res.get(key4)
        if ca_coord is not None:
            residue_coords.append(ca_coord)
        else:
            raise ValueError(f"No CA atom found for residue {key4}")

    coords = np.array(residue_coords, dtype=np.float64)
    total_side_rel_arr = np.array(residue_total_side_rel, dtype=np.float64)
    tree = cKDTree(coords)
    out: Dict[Tuple[str, int, str, str], float] = {}
    for i, key4 in enumerate(residue_keys):
        indices = tree.query_ball_point(coords[i], cutoff)
        value = float(np.sum(total_side_rel_arr[indices]))
        if sap_mode or positive_charge_mode or negative_charge_mode:
            res_name = key4[0]
            if sap_mode:
                weight = float(KYTE_DOOLITTLE.get(res_name, 0.0))
            else:
                pka = None
                if pka_output_data is not None:
                    pka = pka_output_data.get(key4) or pka_output_data.get(
                        (key4[0], key4[1], key4[2], "")
                    )
                q = float(_residue_fractional_charge_at_pH(res_name, pka, pH))
                if positive_charge_mode and negative_charge_mode:
                    weight = q
                elif positive_charge_mode:
                    weight = q if q > 0.0 else 0.0
                elif negative_charge_mode:
                    weight = q if q < 0.0 else 0.0
                else:
                    weight = 1.0
            value *= weight
        out[key4] = value

    return float(np.sum(list(out.values())))

# Generic helpers
ResidueDensityRawDict = Dict[ResKey4, float]

def average_over_residues(
    *,
    weights_raw: Dict[ResKey4, float],
    counts: Optional[Dict[ResKey4, int]] = None,
    residues_for_density: Optional[Iterable[ResKey4]] = None,
    residues_for_average: Optional[Iterable[ResKey4]] = None,
    denom_total_residues: int,
    weighted: bool = True,
    sqrt_weights: bool = True,
) -> float:
    """
    generic helper to average per-residue quantities over a residue set

    - numerator: sum over residues_for_density (or all keys in weights_raw if None)
      if weighted=True and sqrt_weights=True, uses sqrt(weights_raw[k])
      if weighted=True and sqrt_weights=False, uses weights_raw[k] directly
      if weighted=False, uses counts[k] if provided, else 0.0
    - denominator: len(set(residues_for_average)) if provided, otherwise
      denom_total_residues (total number of residues in the PDB)
    """
    if not weights_raw:
        return 0.0

    if residues_for_average is not None:
        denom = len(set(residues_for_average))
    else:
        denom = denom_total_residues
    if denom == 0:
        return 0.0

    density_set: Optional[Set[ResKey4]] = (
        set(residues_for_density) if residues_for_density is not None else None
    )
    keys_to_sum = (
        weights_raw.keys()
        if density_set is None
        else [k for k in density_set if k in weights_raw]
    )

    if weighted:
        if sqrt_weights:
            total = sum(math.sqrt(weights_raw[k]) for k in keys_to_sum)
        else:
            total = sum(weights_raw[k] for k in keys_to_sum)
    else:
        if counts is None:
            total = 0.0
        else:
            total = sum(counts.get(k, 0) for k in keys_to_sum)

    return total / float(denom)

def compute_residue_density_raw(
    pdb_path: str,
    sasa_path: str,
) -> ResidueDensityRawDict:
    """
    compute per-residue raw SASA weight for all residues in the structure. No sqrt
    """
    ctx = _get_structure_context(pdb_path, sasa_path=sasa_path)
    atoms = ctx.atoms
    if not atoms:
        return {}
    sasa_data = ctx.sasa_residue
    residue_keys = list(iter_unique_residues(atoms))

    weights: ResidueDensityRawDict = {}
    for key4 in residue_keys:
        weight = get_residue_sasa_weight(key4, sasa_data, use_side_chain=True)
        weights[key4] = weight
    return weights

def calculate_residue_category_density_average(
    pdb_path: str,
    sasa_path: str,
    *,
    residue_category: Optional[Iterable[ResKey4]] = None,
    weighted: bool = True,
    residues_for_average: Optional[Iterable[ResKey4]] = None,
    density_raw: Optional[ResidueDensityRawDict] = None,
) -> float:
    """
    the category is any list/set of residue keys (e.g. polar, aromatic, inter-chain,
    CDR, or exposed) determined elsewhere and passed in
    density can be SASA-weighted or raw count; the average can be over total residues or over
    the category itself
    """
    if density_raw is None:
        density_raw = compute_residue_density_raw(pdb_path, sasa_path)
    if not density_raw:
        return 0.0

    total_res = _count_residues_in_pdb(pdb_path)
    if weighted:
        return average_over_residues(
            weights_raw=density_raw,
            counts=None,
            residues_for_density=residue_category,
            residues_for_average=residues_for_average,
            denom_total_residues=total_res,
            weighted=True,
            sqrt_weights=True,
        )
    return average_over_residues(
        weights_raw=density_raw,
        counts={k: 1 for k in density_raw},
        residues_for_density=residue_category,
        residues_for_average=residues_for_average,
        denom_total_residues=total_res,
        weighted=False,
    )

# Salt bridge contacts
def _find_salt_bridge_contacts(
    atoms: List[Atom],
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
) -> Dict[
    Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
    List[Tuple[Atom, Atom]],
]:

    charged_pos_residues, charged_neg_residues = _get_charged_residues_at_pH(
        atoms, pka_data, pH
    )
    if not charged_pos_residues or not charged_neg_residues:
        return {}

    positive_atoms = [
        atom
        for atom in atoms
        if (atom.name, atom.residue_name) in POSITIVE_ATOMS
        and residue_key_from_atom(atom) in charged_pos_residues
    ]
    negative_atoms = [
        atom
        for atom in atoms
        if (atom.name, atom.residue_name) in NEGATIVE_ATOMS
        and residue_key_from_atom(atom) in charged_neg_residues
    ]

    if len(negative_atoms) == 0 or len(positive_atoms) == 0:
        return {}

    negative_coords = np.array([[a.x, a.y, a.z] for a in negative_atoms])
    negative_tree = cKDTree(negative_coords)

    contacts_by_pair: Dict[
        Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
        List[Tuple[Atom, Atom]],
    ] = {}

    for pos_atom in positive_atoms:
        pos_coord = (pos_atom.x, pos_atom.y, pos_atom.z)
        pos_key = residue_key_from_atom(pos_atom)

        if pos_key not in charged_pos_residues:
            continue

        indices = negative_tree.query_ball_point(pos_coord, MAX_SALT_BRIDGE_DISTANCE)

        for idx in indices:
            neg_atom = negative_atoms[idx]
            neg_key = residue_key_from_atom(neg_atom)

            if neg_key not in charged_neg_residues:
                continue

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
    a salt bridge is formed when any charged atom from a positively charged residue
    is within MAX_SALT_BRIDGE_DISTANCE of any charged atom from
    a negatively charged residue, according to pKa

    each residue pair counts as only one salt bridge, regardless of how many
    atom-atom contacts exist between them
    """
    ctx = StructureContext(pdb_path, sasa_path=sasa_path, pka_path=pka_path)
    atoms = ctx.atoms
    if not atoms:
        return {}

    sasa_data = ctx.sasa_residue
    pka_data = ctx.pka_residue

    contacts_by_pair = _find_salt_bridge_contacts(atoms, pka_data, pH)
    if not contacts_by_pair:
        return {}

    salt_bridge_pairs: Dict[
        Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
        Tuple[float, float],
    ] = {}

    for pair, contacts in contacts_by_pair.items():
        pos_key, neg_key = pair
        pos_weight = get_residue_sasa_weight(pos_key, sasa_data, use_side_chain=True)
        neg_weight = get_residue_sasa_weight(neg_key, sasa_data, use_side_chain=True)
        salt_bridge_pairs[pair] = (pos_weight, neg_weight)

    return dict(salt_bridge_pairs)

_SaltBridgesDict = Dict[
    Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]],
    Tuple[float, float],
]

# def _compute_salt_bridge_residue_raw(
#     pdb_path: str,
#     sasa_path: str,
#     pka_path: Optional[str],
#     pH: float,
#     salt_bridges: Optional[_SaltBridgesDict] = None,
# ) -> Tuple[
#     _SaltBridgesDict,
#     Dict[Tuple[str, int, str, str], float],
#     Dict[Tuple[str, int, str, str], int],
#     Dict[Tuple[str, int, str, str], bool],
# ]:
#     if salt_bridges is None:
#         salt_bridges = detect_salt_bridges(pdb_path, sasa_path, pka_path, pH)
#     if not salt_bridges:
#         return {}, {}, {}, {}

#     residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
#     residue_counts: Dict[Tuple[str, int, str, str], int] = defaultdict(int)
#     residue_inter_chain: Dict[Tuple[str, int, str, str], bool] = {}

#     for (pos_key, neg_key), (pos_w, neg_w) in salt_bridges.items():
#         residue_weights_raw[pos_key] += pos_w
#         residue_weights_raw[neg_key] += neg_w
#         residue_counts[pos_key] += 1
#         residue_counts[neg_key] += 1
#         if pos_key[2] != neg_key[2]:
#             residue_inter_chain[pos_key] = True
#             residue_inter_chain[neg_key] = True

#     return (
#         salt_bridges,
#         dict(residue_weights_raw),
#         dict(residue_counts),
#         residue_inter_chain,
#     )

# Count of salt bridges for given residues, weighted by sqrt relative SASA and averaged over given residues
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

    if salt_bridges is None:
        salt_bridges = detect_salt_bridges(pdb_path, sasa_path, pka_path, pH)
    if not salt_bridges:
        return {}, {}, {}, {}

    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    residue_counts: Dict[Tuple[str, int, str, str], int] = defaultdict(int)
    residue_inter_chain: Dict[Tuple[str, int, str, str], bool] = {}

    for (pos_key, neg_key), (pos_w, neg_w) in salt_bridges.items():
        residue_weights_raw[pos_key] += pos_w
        residue_weights_raw[neg_key] += neg_w
        residue_counts[pos_key] += 1
        residue_counts[neg_key] += 1
        if pos_key[2] != neg_key[2]:
            residue_inter_chain[pos_key] = True
            residue_inter_chain[neg_key] = True
    
    return average_over_residues(
        weights_raw=dict(residue_weights_raw),
        counts=dict(residue_counts),
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path),
        weighted=weighted,
        sqrt_weights=True,
    )

# near the `_PDB_SEQUENCE_CACHE` definition in descriptors.py

def _get_sequence_per_chain(pdb_path: str) -> Dict[str, List[str]]:
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
        if not keys or keys[-1] != key:
            keys.append(key)

    out: Dict[str, List[str]] = {}
    for chain, keys in chain_to_keys.items():
        sorted_keys = sorted(keys, key=lambda k: (k[1], k[3]))
        out[chain] = [k[0] for k in sorted_keys]

    _PDB_SEQUENCE_CACHE[abs_path] = out
    return out

def get_full_sequence_from_pdb(pdb_path: str, chain_order: Optional[List[str]] = None) -> str:
    per_chain = _get_sequence_per_chain(pdb_path)
    if not per_chain:
        return ""
    chains = chain_order or sorted(per_chain)
    return "".join("".join(per_chain[c]) for c in chains if c in per_chain)


def get_full_sequence_with_index_map_from_pdb(
    pdb_path: str, chain_order: Optional[List[str]] = None
) -> Tuple[str, Dict[Tuple[str, int, str, str], int]]:
    """
    Return the concatenated full sequence string together with a mapping from
    4-tuple residue keys (res_name, res_num, chain, insertion_code) to
    0-based indices in that sequence.
    """
    abs_path = os.path.abspath(pdb_path)
    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        return "", {}

    # Build per-chain unique residue list using the same residue keys as elsewhere.
    chain_to_keys: Dict[str, List[Tuple[str, int, str, str]]] = {}
    for atom in atoms:
        key = residue_key_from_atom(atom)
        chain = key[2]
        keys = chain_to_keys.setdefault(chain, [])
        if not keys or keys[-1] != key:
            keys.append(key)

    # Sort residues within each chain by (res_num, insertion_code) to get a stable order.
    for chain, keys in list(chain_to_keys.items()):
        chain_to_keys[chain] = sorted(keys, key=lambda k: (k[1], k[3]))

    chains = chain_order or sorted(chain_to_keys)

    full_seq_parts: List[str] = []
    index_map: Dict[Tuple[str, int, str, str], int] = {}
    offset = 0
    for chain in chains:
        keys = chain_to_keys.get(chain)
        if not keys:
            continue
        # Residue name is stored in the first element of the key tuple.
        full_seq_parts.append("".join(k[0] for k in keys))
        for local_idx, key in enumerate(keys):
            index_map[key] = offset + local_idx
        offset += len(keys)

    full_seq = "".join(full_seq_parts)
    return full_seq, index_map

# Sequence-related

def count_motif_overlapping(seq, motif: str) -> int:

    if not motif:
        return 0
    # Single string
    if isinstance(seq, str):
        if not seq:
            return 0
        n = len(motif)
        return sum(1 for i in range(len(seq) - n + 1) if seq[i : i + n] == motif)
    # Iterable of strings
    total = 0
    n = len(motif)
    for s in seq:
        if not s:
            continue
        total += sum(1 for i in range(len(s) - n + 1) if s[i : i + n] == motif)
    return total

# Packing-related

def calculate_weighted_contact_number_average(
    pdb_path: str,
    *,
    residue_category: Optional[Iterable[ResKey4]] = None,
    residues_for_density: Optional[Iterable[ResKey4]] = None,
    residues_for_average: Optional[Iterable[ResKey4]] = None,
    wcn_values: Optional[Dict[ResKey4, float]] = None,
) -> float:
    """
    WCN_i = Σ(j≠i) 1/(r_ij²), where r_ij is Cα–Cα distance.

    WCN is first computed for all residues in the structure, then:
    - ``residue_category``: limits which residues have WCN values kept at all
      (e.g. only CDR, only buried, only aromatic). Think "which keys exist in
      the weights dictionary".
    - ``residues_for_density``: optional subset of those keys that actually
      contribute to the numerator of the average. If ``None``, all keys in the
      (possibly category-filtered) weights dictionary are used.
    - ``residues_for_average``: optional set that defines the denominator
      (number of residues you are averaging over). If ``None``, the
      denominator is the total number of residues in the PDB.

    Passing ``residue_category`` as a set avoids repeated ``set()`` construction
    when the same category is reused.
    """
    ctx = _get_structure_context(pdb_path)
    ca_coords_dict = ctx.ca_coords
    if not ca_coords_dict:
        return {}

    residue_keys = list(ca_coords_dict.keys())
    ca_coords = np.array(
        [[xyz[0], xyz[1], xyz[2]] for xyz in ca_coords_dict.values()],
        dtype=np.float64,
    )
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    with np.errstate(divide="ignore"):
        inv_dist_sq = 1.0 / dist_sq
    wcn_array = np.sum(inv_dist_sq, axis=1)

    wcn_values: Dict[ResKey4, float] = {}
    for key, wcn in zip(residue_keys, wcn_array):
        wcn_values[key] = float(wcn)

    if residue_category is not None:
        category_set = (
            residue_category
            if isinstance(residue_category, set)
            else set(residue_category)
        )
        wcn_values = {k: v for k, v in wcn_values.items() if k in category_set}
    if not wcn_values:
        return 0.0

    return average_over_residues(
        weights_raw=wcn_values,
        counts=None,
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path),
        weighted=True,
        sqrt_weights=False,
    )

# Clustering
def compute_residue_DBSCAN_cluster_labels(
    atoms: List[Atom],
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    eps: float = 4.0,
    min_samples: int = 2,
) -> Tuple[
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
]:
    """
    collect C-alpha coordinates for five residue groups (negative, positive, aromatic, hydrophobic, polar),
    run DBSCAN on each group's 3D coordinates, and return cluster labels per category, for each residue 
    """
    ca_by_res: Dict[Tuple[str, int, str, str], Tuple[float, float, float]] = {}
    for atom in atoms:
        if atom.name != "CA":
            continue
        key = residue_key_from_atom(atom)
        ca_by_res[key] = (atom.x, atom.y, atom.z)
    
    def get_pka(key):
        return pka_data.get(key)
    
    neg_keys: List[Tuple[str, int, str, str]] = []
    neg_coords: List[Tuple[float, float, float]] = []
    pos_keys: List[Tuple[str, int, str, str]] = []
    pos_coords: List[Tuple[float, float, float]] = []
    aromatic_keys: List[Tuple[str, int, str, str]] = []
    aromatic_coords: List[Tuple[float, float, float]] = []
    hydro_keys: List[Tuple[str, int, str, str]] = []
    hydro_coords: List[Tuple[float, float, float]] = []
    polar_keys: List[Tuple[str, int, str, str]] = []
    polar_coords: List[Tuple[float, float, float]] = []
    
    for key, xyz in ca_by_res.items():
        res_name = key[0]
        pka_val = get_pka(key)
        group = _residue_category_group(res_name, pka_val, pH)
        if group == "negative":
            neg_keys.append(key)
            neg_coords.append(xyz)
        elif group == "positive":
            pos_keys.append(key)
            pos_coords.append(xyz)
        elif group == "aromatic":
            aromatic_keys.append(key)
            aromatic_coords.append(xyz)
        elif group == "hydrophobic":
            hydro_keys.append(key)
            hydro_coords.append(xyz)
        elif group == "polar":
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
    aromatic_labels = run_dbscan(aromatic_keys, aromatic_coords)
    hydro_labels = run_dbscan(hydro_keys, hydro_coords)
    polar_labels = run_dbscan(polar_keys, polar_coords)
    
    return neg_labels, pos_labels, aromatic_labels, hydro_labels, polar_labels


def summarize_dbscan_clusters(
    labels: Dict[ResKey4, int],
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
) -> Tuple[int, int, float]:

    if not labels:
        return 0, 0, 0.0

    from collections import defaultdict

    cluster_sizes: Dict[int, int] = defaultdict(int)
    cluster_asa: Dict[int, float] = defaultdict(float)

    for key, label in labels.items():
        if label == -1:
            continue
        cluster_sizes[label] += 1
        entry = _sasa_lookup(sasa_output_data, key) or {}
        total_side_rel = entry.get("total_side_rel")
        if total_side_rel is not None:
            cluster_asa[label] += float(total_side_rel)

    if not cluster_sizes:
        return 0, 0, 0.0

    largest_cluster_size = max(cluster_sizes.values())
    n_clusters = len(cluster_sizes)
    total_rel_asa = float(sum(cluster_asa.values()))
    return largest_cluster_size, n_clusters, total_rel_asa


SURFACE_EXPOSED_THRESHOLD_DEFAULT = 25.0
RIPLEY_K_DISTANCE = 8.0
RIPLEY_K_N_SAMPLES = 1000
PSH_PAIR_RADIUS = 7.5
CDR_VICINITY_RADIUS = 4.0

def ripley_k_statistic(
    obs_coords,
    allowed_coords,
    distance: float = 6.0,
    n: int = 1000,
) -> float:
    """
    Ripley's K-like statistic for a set of observed points embedded in a space of
    allowed points

    k_o is the fraction of observed point pairs whose distance is less than the
    cutoff distance. k_e is the expected fraction under a null model where the
    same number of points are randomly scattered over the allowed coordinates.
    The reported statistic is the ratio k_o / k_e
    """
    obs_coords = np.asarray(obs_coords, dtype=float)
    allowed_coords = np.asarray(allowed_coords, dtype=float)
    feature_size = obs_coords.shape[0]
    if feature_size < 2:
        return float("nan")

    denominator = feature_size * (feature_size - 1)

    def _get_number_of_pairs(coords, dist: float) -> int:
        if len(coords) == 0:
            return 0
        kd_tree = cKDTree(coords)
        neighbor_pairs = kd_tree.query_pairs(r=dist)
        return len(neighbor_pairs)

    k_o = _get_number_of_pairs(obs_coords, distance) / denominator

    rng = np.random.default_rng()
    k_e_null = []
    allowed_size = allowed_coords.shape[0]
    replace = allowed_size < feature_size
    for _ in range(int(n)):
        indices = rng.choice(allowed_size, size=feature_size, replace=replace)
        new_coords = allowed_coords[indices]
        k_e_null.append(_get_number_of_pairs(new_coords, distance) / denominator)

    k_e = float(np.mean(k_e_null)) if k_e_null else float("nan")
    if k_e == 0.0 or math.isnan(k_e):
        return float("nan")
    return k_o / k_e

def get_pka_for_key(key: ResKey4, pka_output_data: Dict[ResKey4, float]) -> Optional[float]:
    return pka_output_data.get(key) or pka_output_data.get(
        (key[0], key[1], key[2], "")
    )

def compute_ripley(coords: List[Tuple[float, float, float]], allowed_coords: List[Tuple[float, float, float]], ripley_distance: float = RIPLEY_K_DISTANCE, ripley_n: int = RIPLEY_K_N_SAMPLES) -> Optional[float]:
        if len(coords) < 2:
            return 0.0
        value = ripley_k_statistic(coords, allowed_coords, distance=ripley_distance, n=ripley_n)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)

def compute_surface_ripley_descriptors(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
    surface_exposed_threshold: float = SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    ripley_distance: float = RIPLEY_K_DISTANCE,
    ripley_n: int = RIPLEY_K_N_SAMPLES,
) -> Dict[str, Optional[float]]:

    result: Dict[str, Optional[float]] = {
        "ripley_k_negative": None,
        "ripley_k_positive": None,
        "ripley_k_aromatic": None,
        "ripley_k_hydrophobic": None,
        "ripley_k_polar": None,
    }
    if not sasa_output_data:
        return result

    ca_by_res: Dict[ResKey4, Tuple[float, float, float]] = {}
    atoms_by_res: Dict[ResKey4, List[Atom]] = defaultdict(list)
    for atom in pdb_atoms:
        key = residue_key_from_atom(atom)
        atoms_by_res[key].append(atom)
        if atom.name == "CA":
            ca_by_res[key] = (atom.x, atom.y, atom.z)

    residue_exposure = get_exposed_residues(sasa_output_data, surface_exposed_threshold)
    exposed_keys: Set[ResKey4] = {
        key for key, is_exposed in residue_exposure.items() if is_exposed
    }

    exposed_keys_with_ca = [k for k in exposed_keys if k in ca_by_res]
    if len(exposed_keys_with_ca) < 2:
        result["ripley_k_negative"] = 0.0
        result["ripley_k_positive"] = 0.0
        result["ripley_k_aromatic"] = 0.0
        result["ripley_k_hydrophobic"] = 0.0
        result["ripley_k_polar"] = 0.0
        return result

    allowed_coords = np.array(
        [ca_by_res[k] for k in exposed_keys_with_ca],
        dtype=float,
    )

    neg_coords: List[Tuple[float, float, float]] = []
    pos_coords: List[Tuple[float, float, float]] = []
    aromatic_coords: List[Tuple[float, float, float]] = []
    hydro_coords: List[Tuple[float, float, float]] = []
    polar_coords: List[Tuple[float, float, float]] = []
    for key in exposed_keys_with_ca:
        xyz = ca_by_res[key]
        res_name = key[0]
        pka_val = get_pka_for_key(key, pka_output_data)
        group = _residue_category_group(res_name, pka_val, pH)
        if group == "negative":
            neg_coords.append(xyz)
        elif group == "positive":
            pos_coords.append(xyz)
        elif group == "aromatic":
            aromatic_coords.append(xyz)
        elif group == "hydrophobic":
            hydro_coords.append(xyz)
        elif group == "polar":
            polar_coords.append(xyz)

    result["ripley_k_negative"] = compute_ripley(neg_coords)
    result["ripley_k_positive"] = compute_ripley(pos_coords)
    result["ripley_k_aromatic"] = compute_ripley(aromatic_coords)
    result["ripley_k_hydrophobic"] = compute_ripley(hydro_coords)
    result["ripley_k_polar"] = compute_ripley(polar_coords)

    return result


def compute_surface_pair_descriptors(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
    surface_exposed_threshold: float = SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    pair_radius: float = PSH_PAIR_RADIUS,
    cdr_vicinity_radius: float = CDR_VICINITY_RADIUS,
) -> Dict[str, Optional[float]]:
    """
    Compute PSH/PPC/PNC surface spatial descriptors (all surface and CDR vicinity)
    """
    result: Dict[str, Optional[float]] = {
        "psh_all_surface": None,
        "psh_cdr_vicinity": None,
        "ppc_all_surface": None,
        "ppc_cdr_vicinity": None,
        "pnc_all_surface": None,
        "pnc_cdr_vicinity": None,
    }
    if not sasa_output_data:
        return result

    ca_by_res: Dict[ResKey4, Tuple[float, float, float]] = {}
    atoms_by_res: Dict[ResKey4, List[Atom]] = defaultdict(list)
    for atom in pdb_atoms:
        key = residue_key_from_atom(atom)
        atoms_by_res[key].append(atom)
        if atom.name == "CA":
            ca_by_res[key] = (atom.x, atom.y, atom.z)

    # Use shared exposure helper instead of duplicating SASA threshold logic
    residue_exposure = get_exposed_residues(sasa_output_data, surface_exposed_threshold)
    exposed_keys: Set[ResKey4] = {
        key for key, is_exposed in residue_exposure.items() if is_exposed
    }

    exposed_keys_with_ca = [k for k in exposed_keys if k in ca_by_res]
    if len(exposed_keys_with_ca) < 2:
        result["psh_all_surface"] = 0.0
        result["psh_cdr_vicinity"] = 0.0
        result["ppc_all_surface"] = 0.0
        result["ppc_cdr_vicinity"] = 0.0
        result["pnc_all_surface"] = 0.0
        result["pnc_cdr_vicinity"] = 0.0
        return result

    exposed_residue_keys: List[ResKey4] = []
    H_values: List[float] = []
    for key in exposed_keys:
        h_val = normalize_hydropathy(key[0])
        if h_val is None or key not in atoms_by_res:
            continue
        exposed_residue_keys.append(key)
        H_values.append(h_val)

    if len(exposed_residue_keys) < 2:
        result["psh_all_surface"] = 0.0
        result["psh_cdr_vicinity"] = 0.0
        result["ppc_all_surface"] = 0.0
        result["ppc_cdr_vicinity"] = 0.0
        result["pnc_all_surface"] = 0.0
        result["pnc_cdr_vicinity"] = 0.0
        return result

    points_list: List[List[float]] = []
    labels_list: List[int] = []
    for idx, key in enumerate(exposed_residue_keys):
        for atom in atoms_by_res[key]:
            if getattr(atom, "element", None) == "H":
                continue
            points_list.append([atom.x, atom.y, atom.z])
            labels_list.append(idx)

    if not points_list:
        result["psh_all_surface"] = 0.0
        result["psh_cdr_vicinity"] = 0.0
        result["ppc_all_surface"] = 0.0
        result["ppc_cdr_vicinity"] = 0.0
        result["pnc_all_surface"] = 0.0
        result["pnc_cdr_vicinity"] = 0.0
        return result

    points_arr = np.array(points_list, dtype=float)
    labels_arr = np.array(labels_list, dtype=int)
    atom_tree = cKDTree(points_arr)
    pair_min_dist: Dict[Tuple[int, int], float] = {}
    for i in range(points_arr.shape[0]):
        res_i = labels_arr[i]
        neighbors = atom_tree.query_ball_point(points_arr[i], pair_radius)
        for j in neighbors:
            if j <= i:
                continue
            res_j = labels_arr[j]
            if res_i == res_j:
                continue
            pair = (min(res_i, res_j), max(res_i, res_j))
            dist_ij = float(np.linalg.norm(points_arr[i] - points_arr[j]))
            if dist_ij <= 0.0:
                continue
            current = pair_min_dist.get(pair)
            if current is None or dist_ij < current:
                pair_min_dist[pair] = dist_ij

    if not pair_min_dist:
        result["psh_all_surface"] = 0.0
        result["psh_cdr_vicinity"] = 0.0
        result["ppc_all_surface"] = 0.0
        result["ppc_cdr_vicinity"] = 0.0
        result["pnc_all_surface"] = 0.0
        result["pnc_cdr_vicinity"] = 0.0
        return result

    H_array = np.array(H_values, dtype=float)
    pos_charge = np.zeros(len(exposed_residue_keys), dtype=float)
    neg_charge = np.zeros(len(exposed_residue_keys), dtype=float)
    for idx, key in enumerate(exposed_residue_keys):
        res_name = key[0]
        pka_val = get_pka_for_key(key, pka_output_data)
        if res_name in POSITIVE_CHARGED_RESIDUES and is_residue_charged(res_name, pka_val, pH):
            pos_charge[idx] = 1.0
        elif res_name in NEGATIVE_CHARGED_RESIDUES and is_residue_charged(res_name, pka_val, pH):
            neg_charge[idx] = 1.0

    psh_all_surface_acc = 0.0
    ppc_all_surface_acc = 0.0
    pnc_all_surface_acc = 0.0
    for (ri, rj), dist in pair_min_dist.items():
        inv_r2 = 1.0 / (dist * dist)
        h1 = H_array[ri]
        h2 = H_array[rj]
        psh_all_surface_acc += (h1 * h2) * inv_r2
        if pos_charge[ri] > 0.0 and pos_charge[rj] > 0.0:
            ppc_all_surface_acc += (pos_charge[ri] * pos_charge[rj]) * inv_r2
        if neg_charge[ri] > 0.0 and neg_charge[rj] > 0.0:
            pnc_all_surface_acc += (neg_charge[ri] * neg_charge[rj]) * inv_r2

    result["psh_all_surface"] = psh_all_surface_acc
    result["ppc_all_surface"] = ppc_all_surface_acc
    result["pnc_all_surface"] = pnc_all_surface_acc

    def is_cdr_residue(k: ResKey4) -> bool:
        num = k[1]
        for start, end in CDR_RANGES_CA:
            if start <= num <= end:
                return True
        return False

    seed_indices = {
        idx for idx, key in enumerate(exposed_residue_keys) if is_cdr_residue(key)
    }
    seed_points_list: List[List[float]] = []
    for idx, key in enumerate(exposed_residue_keys):
        if idx not in seed_indices:
            continue
        for atom in atoms_by_res[key]:
            if getattr(atom, "element", None) == "H":
                continue
            seed_points_list.append([atom.x, atom.y, atom.z])

    cdr_vicinity_indices = set(seed_indices)
    if seed_points_list:
        seed_points_arr = np.array(seed_points_list, dtype=float)
        seed_tree = cKDTree(seed_points_arr)
        for idx, key in enumerate(exposed_residue_keys):
            if idx in seed_indices:
                continue
            for atom in atoms_by_res[key]:
                if getattr(atom, "element", None) == "H":
                    continue
                coord = np.array([atom.x, atom.y, atom.z], dtype=float)
                if seed_tree.query_ball_point(coord, cdr_vicinity_radius):
                    cdr_vicinity_indices.add(idx)
                    break

    if cdr_vicinity_indices and H_array is not None:
        psh_cdr_acc = 0.0
        ppc_cdr_acc = 0.0
        pnc_cdr_acc = 0.0
        cdr_set = set(cdr_vicinity_indices)
        for (ri, rj), dist in pair_min_dist.items():
            if ri in cdr_set and rj in cdr_set:
                inv_r2 = 1.0 / (dist * dist)
                psh_cdr_acc += (H_array[ri] * H_array[rj]) * inv_r2
                if pos_charge[ri] > 0.0 and pos_charge[rj] > 0.0:
                    ppc_cdr_acc += (pos_charge[ri] * pos_charge[rj]) * inv_r2
                if neg_charge[ri] > 0.0 and neg_charge[rj] > 0.0:
                    pnc_cdr_acc += (neg_charge[ri] * neg_charge[rj]) * inv_r2
        result["psh_cdr_vicinity"] = psh_cdr_acc
        result["ppc_cdr_vicinity"] = ppc_cdr_acc
        result["pnc_cdr_vicinity"] = pnc_cdr_acc
    else:
        result["psh_cdr_vicinity"] = 0.0
        result["ppc_cdr_vicinity"] = 0.0
        result["pnc_cdr_vicinity"] = 0.0

    return result


# def compute_surface_spatial_descriptors(
#     pdb_atoms: List[Atom],
#     sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
#     pka_output_data: Dict[ResKey4, float],
#     pH: float,
#     surface_exposed_threshold: float = SURFACE_EXPOSED_THRESHOLD_DEFAULT,
#     ripley_distance: float = RIPLEY_K_DISTANCE,
#     ripley_n: int = RIPLEY_K_N_SAMPLES,
#     pair_radius: float = PSH_PAIR_RADIUS,
#     cdr_vicinity_radius: float = CDR_VICINITY_RADIUS,
# ) -> Dict[str, Optional[float]]:
#     """
#     Backwards-compatible wrapper that computes both Ripley K and PSH/PPC/PNC descriptors.
#     """
#     ripley = compute_surface_ripley_descriptors(
#         pdb_atoms,
#         sasa_output_data,
#         pka_output_data,
#         pH,
#         surface_exposed_threshold=surface_exposed_threshold,
#         ripley_distance=ripley_distance,
#         ripley_n=ripley_n,
#     )
#     pair = compute_surface_pair_descriptors(
#         pdb_atoms,
#         sasa_output_data,
#         pka_output_data,
#         pH,
#         surface_exposed_threshold=surface_exposed_threshold,
#         pair_radius=pair_radius,
#         cdr_vicinity_radius=cdr_vicinity_radius,
#     )
#     out: Dict[str, Optional[float]] = {}
#     out.update(ripley)
#     out.update(pair)
#     return out

# Inter-chain buried SASA
def compute_inter_chain_buried_sasa(complex_sasa_path: str) -> Optional[float]:
    """Inter-chain buried SASA = (H_total + L_total) - complex_total. Uses cached SASA totals from parsers."""
    from pathlib import Path
    path = Path(complex_sasa_path)
    if not path.exists():
        return None
    stem = path.stem
    suffix = path.suffix
    base = stem[:-5] if stem.endswith("_full") else stem
    sasa_H_path = str(path.with_name(f"{base}_H_full{suffix}"))
    sasa_L_path = str(path.with_name(f"{base}_L_full{suffix}"))

    complex_total = get_sasa_total(str(path.resolve()))
    h_total = get_sasa_total(sasa_H_path)
    l_total = get_sasa_total(sasa_L_path)
    if complex_total is not None and h_total is not None and l_total is not None:
        return h_total + l_total - complex_total
    return None

# H-bonds
def _enumerate_hbonds(pdb_path: str) -> List[Tuple[Atom, Atom]]:
    """
    Enumerate all hydrogen bonds in the structure (distance + angle + occupancy + backbone-separation rules).

    Single implementation used by calculate_global_hbond_density (weighted aggregation)
    and largest_hbond_component_size (graph construction). Returns list of (donor_atom, acceptor_atom)
    so callers can apply their own aggregation (SASA weights + sqrt, or edge set for union-find).

    Distance: enforced via KD-tree for every donor (backbone and sidechain); candidates within
    MAX_HBOND_DISTANCE. Then angle Base->Donor->Acceptor >= MIN_HBOND_ANGLE, donor/acceptor
    not over capacity, same-residue excluded, backbone-backbone with |seq_i - seq_j| <
    MIN_BACKBONE_SEPARATION excluded.
    """
    abs_path = os.path.abspath(pdb_path)
    cache_key = (abs_path, MAX_HBOND_DISTANCE, MIN_HBOND_ANGLE, MIN_BACKBONE_SEPARATION)
    cached = _HBOND_PAIRS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    atoms = _get_atoms_for_path(pdb_path)
    seq_index = _get_residue_seq_index(atoms)
    donors = [atom for atom in atoms if is_donor(atom)]
    acceptors = [atom for atom in atoms if is_acceptor(atom)]

    if not donors or not acceptors:
        _HBOND_PAIRS_CACHE[cache_key] = []
        return []

    acceptor_coords = np.array([[a.x, a.y, a.z] for a in acceptors])
    acceptor_keys = [residue_key_from_atom(a) for a in acceptors]
    acceptor_tree = cKDTree(acceptor_coords)
    donor_coords = np.array([[d.x, d.y, d.z] for d in donors])
    backbone_base_cache: Dict[ResKey4, Atom] = {}
    backbone_base_vec_cache: Dict[ResKey4, np.ndarray] = {}
    backbone_base_vec_norm_sq_cache: Dict[ResKey4, float] = {}
    sidechain_base_cache: Dict[Atom, Optional[Atom]] = {}
    sidechain_base_vec_cache: Dict[Atom, np.ndarray] = {}

    pairs: List[Tuple[Atom, Atom]] = []

    donor_hbond_counts: Dict[Atom, int] = {}
    acceptor_hbond_counts: Dict[Atom, int] = {}

    for donor_idx, donor in enumerate(donors):
        donor_coord = donor_coords[donor_idx]
        donor_res_key = residue_key_from_atom(donor)
        # Distance filter: all candidates come from KD-tree (same for backbone and sidechain)
        indices = acceptor_tree.query_ball_point(donor_coord, MAX_HBOND_DISTANCE)
        if not indices:
            continue

        is_backbone_donor = is_backbone_atom(donor.name)
        donor_max = donor_max_hbonds(donor)

        # Resolve base atom and base->donor vector for angle check (same for backbone and sidechain)
        if is_backbone_donor:
            precomputed_base = get_donor_base_atom(donor, atoms, backbone_base_cache)
            if precomputed_base is None:
                continue
            base_vec = backbone_base_vec_cache.get(donor_res_key)
            if base_vec is None:
                base_vec = np.array([
                    precomputed_base.x - donor.x,
                    precomputed_base.y - donor.y,
                    precomputed_base.z - donor.z,
                ])
                backbone_base_vec_cache[donor_res_key] = base_vec
                backbone_base_vec_norm_sq_cache[donor_res_key] = float(np.sum(base_vec**2))
            db_norm_sq = backbone_base_vec_norm_sq_cache[donor_res_key]
        else:
            base = sidechain_base_cache.get(donor)
            if base is None and donor not in sidechain_base_cache:
                base = get_donor_base_atom(donor, atoms, backbone_base_cache)
                sidechain_base_cache[donor] = base
            if base is None:
                continue
            base_vec = sidechain_base_vec_cache.get(donor)
            if base_vec is None:
                base_vec = np.array([base.x - donor.x, base.y - donor.y, base.z - donor.z])
                sidechain_base_vec_cache[donor] = base_vec
            db_norm_sq = float(np.sum(base_vec**2))

        # Valid candidates: within distance (KD-tree), acceptor not full, not same residue, backbone separation if applicable
        valid_indices = []
        valid_acceptors = []
        for idx in indices:
            acceptor = acceptors[idx]
            if acceptor_hbond_counts.get(acceptor, 0) >= acceptor_max_hbonds(acceptor):
                continue
            acceptor_res_key = acceptor_keys[idx]
            if donor_res_key == acceptor_res_key:
                continue
            if is_backbone_donor and is_backbone_atom(acceptor.name) and donor.chain == acceptor.chain:
                di = seq_index.get(donor_res_key)
                ai = seq_index.get(acceptor_res_key)
                if di is not None and ai is not None:
                    if abs(di - ai) < MIN_BACKBONE_SEPARATION:
                        continue
            valid_indices.append(idx)
            valid_acceptors.append(acceptor)

        if not valid_acceptors:
            continue

        if donor_hbond_counts.get(donor, 0) >= donor_max:
            continue

        # Angle check: Base->Donor->Acceptor >= MIN_HBOND_ANGLE (120°) <=> cos_theta <= -0.5 (vectorized)
        da_coords = acceptor_coords[valid_indices]
        da_vecs = da_coords - donor_coord
        if db_norm_sq == 0.0:
            continue
        da_norm_sq = np.sum(da_vecs**2, axis=1)
        nonzero_mask = da_norm_sq > 0.0
        if not np.any(nonzero_mask):
            continue
        dot_products = np.einsum("ij,j->i", da_vecs, base_vec)
        cos_threshold_sq = 0.25  # (-0.5)**2
        angle_ok = (
            (dot_products <= 0)
            & (dot_products**2 >= cos_threshold_sq * da_norm_sq * db_norm_sq)
            & nonzero_mask
        )
        if not np.any(angle_ok):
            continue
        passed_indices = np.nonzero(angle_ok)[0]
        for local_idx in passed_indices:
            if donor_hbond_counts.get(donor, 0) >= donor_max:
                break
            acceptor = valid_acceptors[local_idx]
            acc_max = acceptor_max_hbonds(acceptor)
            if acceptor_hbond_counts.get(acceptor, 0) >= acc_max:
                continue
            pairs.append((donor, acceptor))
            donor_hbond_counts[donor] = donor_hbond_counts.get(donor, 0) + 1
            acceptor_hbond_counts[acceptor] = acceptor_hbond_counts.get(acceptor, 0) + 1

    pairs.sort(key=lambda p: (residue_key_from_atom(p[0]), residue_key_from_atom(p[1])))
    _HBOND_PAIRS_CACHE[cache_key] = pairs
    return pairs

# Each hydrogen bond is counted once.
# Each donor-acceptor pair is checked once (we iterate for donor in donors, then for acceptor in acceptors, so we don't check acceptor-donor pairs separately).
# Each bond contributes to both residues:
# Donor residue gets donor_weight (based on donor atom's SASA)
# Acceptor residue gets acceptor_weight (based on acceptor atom's SASA)
# We don't skip residues — we ensure each donor-acceptor pair is only evaluated once in the nested loop.
# So a single bond adds to both residues, but with different weights. No double-counting of the same bond for the same residue.

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
    atom_sasa_cache: Dict[Atom, float] = {}
    residue_key_cache: Dict[Atom, ResKey4] = {}

    def get_cached_sasa_weight(atom: Atom) -> float:
        cached = atom_sasa_cache.get(atom)
        if cached is not None:
            return cached
        w = atom_sasa_weight(atom, sasa_data)
        atom_sasa_cache[atom] = w
        return w

    def get_cached_residue_key(atom: Atom) -> ResKey4:
        key = residue_key_cache.get(atom)
        if key is not None:
            return key
        key = residue_key_from_atom(atom)
        residue_key_cache[atom] = key
        return key

    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    residue_counts: Dict[Tuple[str, int, str, str], int] = defaultdict(int)
    residue_inter_chain: Dict[Tuple[str, int, str, str], bool] = defaultdict(bool)

    for donor, acceptor in pairs:
        donor_key = get_cached_residue_key(donor)
        acceptor_key = get_cached_residue_key(acceptor)
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

    pairs = _enumerate_hbonds(pdb_path)
    if not pairs:
        return {}, {}, {}
    ctx = _get_structure_context(pdb_path, sasa_path=sasa_path)
    sasa_data = ctx.sasa_residue
    weights_raw, counts, inter_chain = _aggregate_hbond_pairs_to_raw(pairs, sasa_data)
    sqrt_residue_hbonds = {key: math.sqrt(w) for key, w in weights_raw.items()}
    return sqrt_residue_hbonds, inter_chain, counts


def compute_hbond_density_raw(
    pdb_path: str,
    sasa_path: str,
) -> _HbondDensityRaw:

    pairs = _enumerate_hbonds(pdb_path)
    if not pairs:
        return {}, {}
    ctx = _get_structure_context(pdb_path, sasa_path=sasa_path)
    sasa_data = ctx.sasa_residue
    weights_raw, counts, _ = _aggregate_hbond_pairs_to_raw(pairs, sasa_data)
    return weights_raw, counts


def calculate_global_hbond_density_average(
    pdb_path: str,
    sasa_path: str,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    weighted: bool = True,
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    weights_raw: Optional[Dict[Tuple[str, int, str, str], float]] = None,
    counts: Optional[Dict[Tuple[str, int, str, str], int]] = None,
) -> float:
 
    if weights_raw is None or counts is None:
        weights_raw, counts = compute_hbond_density_raw(pdb_path, sasa_path)
    if not weights_raw or not counts:
        return 0.0

    return average_over_residues(
        weights_raw=weights_raw,
        counts=counts,
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path),
        weighted=weighted,
        sqrt_weights=True,
    )


def largest_hbond_component_size(pdb_path: str) -> int:
    """
    Size of the largest connected component of the H-bond network (geometry-based).
    Returns:
        Number of residues in the largest connected component (0 if no H-bonds).
    """
    pairs = _enumerate_hbonds(pdb_path)
    if not pairs:
        return 0

    atom_key_cache: Dict[Atom, ResKey4] = {}

    def get_res_key(atom: Atom) -> ResKey4:
        key = atom_key_cache.get(atom)
        if key is not None:
            return key
        key = residue_key_from_atom(atom)
        atom_key_cache[atom] = key
        return key

    edges: Set[Tuple[ResKey4, ResKey4]] = set()
    for donor, acceptor in pairs:
        edges.add((get_res_key(donor), get_res_key(acceptor)))
    if not edges:
        return 0

    parent: Dict[ResKey4, ResKey4] = {}

    def find(x: ResKey4) -> ResKey4:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: ResKey4, y: ResKey4) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for u, v in edges:
        union(u, v)

    root_count: Dict[ResKey4, int] = defaultdict(int)
    for node in parent:
        root_count[find(node)] += 1

    return max(root_count.values()) if root_count else 0

_DsspHbondEnergyDensityRaw = Tuple[
    Dict[Tuple[str, int, str, str], float],
    Dict[Tuple[str, int, str, str], int],
]


# def calculate_hbond_energy_density_dssp_backbone_only(
#     pdb_path: str,
#     sasa_path: str,
#     dssp_path: str,
# ) -> Tuple[Dict[Tuple[str, int, str, str], float], Dict[Tuple[str, int, str, str], bool]]:

#     ctx = StructureContext(pdb_path, sasa_path=sasa_path, dssp_path=dssp_path)
#     atoms = ctx.atoms
#     sasa_data = ctx.sasa_residue
#     dssp_hbonds, dssp_seq_to_pdb = ctx.dssp_hbonds
#     pdb_to_dssp_seq = {pdb_key: dssp_seq for dssp_seq, pdb_key in dssp_seq_to_pdb.items()}

#     weights_raw, _, inter_chain = _aggregate_dssp_hbond_energy_to_raw(
#         dssp_hbonds, dssp_seq_to_pdb, pdb_to_dssp_seq, sasa_data
#     )
#     sqrt_residue_hbonds = {key: math.sqrt(w) for key, w in weights_raw.items()}
#     return sqrt_residue_hbonds, inter_chain


def compute_dssp_hbond_energy_density_raw(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
) -> _DsspHbondEnergyDensityRaw:

    ctx = StructureContext(pdb_path, sasa_path=sasa_path, dssp_path=dssp_path)
    atoms = ctx.atoms
    sasa_data = ctx.sasa_residue
    dssp_hbonds, dssp_seq_to_pdb = ctx.dssp_hbonds
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

    if dssp_hbond_energy_density_raw is None:
        dssp_hbond_energy_density_raw = compute_dssp_hbond_energy_density_raw(
            pdb_path, sasa_path, dssp_path
        )
    weights_raw, counts = dssp_hbond_energy_density_raw
    if not weights_raw:
        return 0.0

    return average_over_residues(
        weights_raw=weights_raw,
        counts=counts,
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path),
        weighted=weighted,
        sqrt_weights=True,
    )


from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# 3-letter to 1-letter for sequence building (inverse of AA_1_TO_3)
THREE_TO_ONE = {v: k for k, v in AA_1_TO_3.items()}

# Residue sets used by the notebook
GLN_ASN_RESIDUES = frozenset({"GLN", "ASN"})

# Metrics for which we compute median / beta_sheet_median / buried_median / exposed_median
METRICS = ["hbond_density", "salt_bridge_density", "wcn", "hbond_energy_dssp_density"]

# Kyte-Doolittle hydropathy scores (same as run_developability / notebook)
KYTE_DOOLITTLE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}

# Cluster label column names for max total_side_rel per cluster
CLUSTER_LABEL_COLS = [
    "negative_cluster_labels",
    "positive_cluster_labels",
    "aromatic_cluster_labels",
    "hydrophobic_cluster_labels",
    "polar_cluster_labels",
]


def compute_downstream_descriptors(
    df: pd.DataFrame,
    *,
    structure_id: Optional[str] = None,
    base: Optional[str] = None,
    heavy: Optional[str] = None,
    light: Optional[str] = None,
    inter_chain_buried_sasa: Optional[float] = None,
    pdb_path: Optional[Union[str, Path]] = None,
    heavy_chain_id: str = "H",
    light_chain_id: str = "L",
    pH: float = 7.0,
    use_descriptors_motif: bool = False,
) -> Dict[str, Any]:

    df = df.copy()
    n_total = len(df)
    result: Dict[str, Any] = {}

    if structure_id is not None:
        result["structure_id"] = structure_id
    if base is not None:
        result["base"] = base
    if heavy is not None:
        result["heavy"] = heavy
    if light is not None:
        result["light"] = light

    # Numeric columns
    if "total_side_rel" in df.columns:
        df["total_side_rel"] = pd.to_numeric(df["total_side_rel"], errors="coerce")

    # ----- Inter-chain buried SASA -----
    if inter_chain_buried_sasa is not None:
        result["inter_chain_buried_sasa"] = float(inter_chain_buried_sasa)
    elif "inter_chain_buried_sasa" in df.columns and len(df) > 0:
        result["inter_chain_buried_sasa"] = pd.to_numeric(
            df["inter_chain_buried_sasa"], errors="coerce"
        ).iloc[0]
    else:
        result["inter_chain_buried_sasa"] = np.nan

    # ----- Inter-chain contact number -----
    if "inter_chain_contact" in df.columns:
        inter = df["inter_chain_contact"].astype(str).str.lower()
        result["inter_chain_contact_number"] = (inter == "true").sum()
    else:
        result["inter_chain_contact_number"] = np.nan

    # ----- Sequence and motifs -----
    # Motif counts are now computed in run_developability.py after the per-residue
    # table is built; we leave their slots to be filled there.

    # ----- Per-chain charge at pH (using descriptors' fractional charge) -----
    if "pka" in df.columns and "residue_name" in df.columns:
        df["pka"] = pd.to_numeric(df["pka"], errors="coerce")

        def res_charge(row: pd.Series) -> float:
            res = row["residue_name"]
            pka_val = row["pka"]
            if pd.isna(res) or pd.isna(pka_val):
                return 0.0
            return _residue_fractional_charge_at_pH(res, float(pka_val), pH)

        df["res_charge_pH"] = df.apply(res_charge, axis=1)
        if "chain" in df.columns:
            chain = df["chain"].astype(str).str.upper()
            heavy_mask = chain.str.startswith(heavy_chain_id)
            light_mask = chain.str.startswith(light_chain_id)
            heavy_charge = df.loc[heavy_mask, "res_charge_pH"].sum() if heavy_mask.any() else np.nan
            light_charge = df.loc[light_mask, "res_charge_pH"].sum() if light_mask.any() else np.nan
            result["heavy_charge_pH7"] = heavy_charge
            result["light_charge_pH7"] = light_charge
            if pd.notna(heavy_charge) and pd.notna(light_charge):
                result["heavy_light_charge_product_pH7"] = float(heavy_charge * light_charge)
            else:
                result["heavy_light_charge_product_pH7"] = np.nan
        else:
            result["heavy_charge_pH7"] = np.nan
            result["light_charge_pH7"] = np.nan
            result["heavy_light_charge_product_pH7"] = np.nan
            result["net_charge_propka_pH7"] = df["res_charge_pH"].sum()
    else:
        result["heavy_charge_pH7"] = np.nan
        result["light_charge_pH7"] = np.nan
        result["heavy_light_charge_product_pH7"] = np.nan

    # ----- Medians for metrics -----
    for metric in METRICS:
        if metric in df.columns:
            result[f"{metric}_median"] = df[metric].median()
        else:
            result[f"{metric}_median"] = np.nan

    # ----- H-bonds and salt bridges (structure-level) -----
    hbond_cols = ["N-H-->O_1", "N-H-->O_2", "O-->H-N_1", "O-->H-N_2"]
    existing_hbond = [c for c in hbond_cols if c in df.columns]
    if existing_hbond:
        df[existing_hbond] = df[existing_hbond].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        per_res = (df[existing_hbond] != 0).sum(axis=1)
        result["n_hydrogen_bonds"] = float(per_res.sum() / 2.0)
    else:
        result["n_hydrogen_bonds"] = 0.0
        per_res = pd.Series(0.0, index=df.index)

    if "salt_bridge_density" in df.columns:
        df["salt_bridge_density"] = pd.to_numeric(df["salt_bridge_density"], errors="coerce")
        result["n_salt_bridges"] = float(df["salt_bridge_density"].fillna(0).sum() / 2.0)
    else:
        result["n_salt_bridges"] = 0.0

    if "number_of_hbonds" in df.columns:
        deg = pd.to_numeric(df["number_of_hbonds"], errors="coerce").fillna(0)
        E = deg.sum() / 2.0
        N = len(df)
        result["mean_hbond_degree"] = (2.0 * E / N) if N > 0 else np.nan
    else:
        result["mean_hbond_degree"] = np.nan

    # ----- Beta sheet medians -----
    if "secondary_structure" in df.columns:
        beta_df = df[df["secondary_structure"] == "E"]
        for metric in METRICS:
            if metric in df.columns:
                result[f"{metric}_beta_sheet_median"] = beta_df[metric].median()
            else:
                result[f"{metric}_beta_sheet_median"] = np.nan
    else:
        for metric in METRICS:
            result[f"{metric}_beta_sheet_median"] = np.nan

    # ----- Buried / exposed (total_side_rel < 25 = buried) -----
    if "total_side_rel" in df.columns:
        buried_df = df[df["total_side_rel"] < 25]
        exposed_df = df[df["total_side_rel"] >= 25]
    else:
        buried_df = pd.DataFrame()
        exposed_df = df

    for metric in METRICS:
        if metric in df.columns:
            result[f"{metric}_buried_median"] = buried_df[metric].median() if len(buried_df) > 0 else np.nan
            result[f"{metric}_exposed_median"] = exposed_df[metric].median() if len(exposed_df) > 0 else np.nan
        else:
            result[f"{metric}_buried_median"] = np.nan
            result[f"{metric}_exposed_median"] = np.nan

    n_buried = len(buried_df)

    # ----- Met/Tyr counts -----
    if "residue_name" in df.columns:
        result["n_Met"] = (df["residue_name"] == "MET").sum()
        result["n_Tyr"] = (df["residue_name"] == "TYR").sum()
        result["n_Met_exposed"] = (exposed_df["residue_name"] == "MET").sum() if len(exposed_df) > 0 else 0
        result["n_Tyr_exposed"] = (exposed_df["residue_name"] == "TYR").sum() if len(exposed_df) > 0 else 0
    else:
        result["n_Met"] = result["n_Tyr"] = result["n_Met_exposed"] = result["n_Tyr_exposed"] = np.nan

    # ----- CDR region (using descriptors CDR ranges) -----
    cdr_mask = _cdr_mask(df)
    range_df = df[cdr_mask]
    n_cdr = len(range_df)

    if "hbond_density" in df.columns and n_cdr > 0:
        result["hbond_density_CDRs_mean"] = range_df["hbond_density"].mean()
    else:
        result["hbond_density_CDRs_mean"] = np.nan
    if "salt_bridge_density" in df.columns and n_cdr > 0:
        result["salt_bridge_density_CDRs_mean"] = range_df["salt_bridge_density"].mean()
    else:
        result["salt_bridge_density_CDRs_mean"] = np.nan

    total_hbond_part = per_res.sum()
    if total_hbond_part > 0:
        result["ratio_hbonds_CDR_to_total"] = per_res.loc[cdr_mask].sum() / total_hbond_part
    else:
        result["ratio_hbonds_CDR_to_total"] = np.nan

    if "salt_bridge_density" in df.columns:
        total_salt = df["salt_bridge_density"].sum()
        if total_salt > 0 and n_cdr > 0:
            result["ratio_salt_bridges_CDR_to_total"] = range_df["salt_bridge_density"].sum() / total_salt
        else:
            result["ratio_salt_bridges_CDR_to_total"] = 0.0 if total_salt == 0 else np.nan
    else:
        result["ratio_salt_bridges_CDR_to_total"] = 0.0

    if n_cdr > 0 and "residue_name" in df.columns:
        result["cdr_total_length"] = n_cdr
        result["fraction_gly_CDRs"] = (range_df["residue_name"] == "GLY").sum() / n_cdr
        result["fraction_pro_CDRs"] = (range_df["residue_name"] == "PRO").sum() / n_cdr
        result["fraction_aromatic_CDRs"] = range_df["residue_name"].isin(AROMATIC_RESIDUES).sum() / n_cdr
        result["fraction_gln_asn_CDRs"] = range_df["residue_name"].isin(GLN_ASN_RESIDUES).sum() / n_cdr
        hydro_cdr = range_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum()
        polar_cdr = range_df["residue_name"].isin(POLAR_RESIDUES).sum()
        result["ratio_hydrophobic_to_polar_CDRs"] = (hydro_cdr / polar_cdr) if polar_cdr > 0 else np.nan
    else:
        result["cdr_total_length"] = np.nan
        result["fraction_gly_CDRs"] = result["fraction_pro_CDRs"] = result["fraction_aromatic_CDRs"] = result["fraction_gln_asn_CDRs"] = np.nan
        result["ratio_hydrophobic_to_polar_CDRs"] = np.nan

    # ----- Fraction buried and total_side_rel median -----
    result["fraction_buried"] = (n_buried / n_total) if n_total > 0 else np.nan
    if "total_side_rel" in df.columns:
        result["total_side_rel_median"] = df["total_side_rel"].median()
    else:
        result["total_side_rel_median"] = np.nan

    # ----- Fraction hydrophobic / negative / positive among buried -----
    if n_buried > 0 and "residue_name" in df.columns:
        result["fraction_hydrophobic_buried"] = buried_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / n_buried
        result["fraction_negative_buried"] = buried_df["residue_name"].isin(NEGATIVE_CHARGED_RESIDUES).sum() / n_buried
        result["fraction_positive_buried"] = buried_df["residue_name"].isin(POSITIVE_CHARGED_RESIDUES).sum() / n_buried
    else:
        result["fraction_hydrophobic_buried"] = result["fraction_negative_buried"] = result["fraction_positive_buried"] = np.nan

    # ----- total_side_rel sums by category (aromatic, negative, positive, polar, hydrophobic) -----
    # All, buried, exposed, inter_chain, CDRs; fraction_*_exposed; hydrophobic_to_charged/polar ratios; sap_sum
    if "residue_name" in df.columns and "total_side_rel" in df.columns:
        _zero = 0.0
        # All
        result["aromatic_total_side_rel_sum"] = df[df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
        result["negative_total_side_rel_sum"] = df[df["residue_name"].isin(NEGATIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
        result["positive_total_side_rel_sum"] = df[df["residue_name"].isin(POSITIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
        result["polar_total_side_rel_sum"] = df[df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
        result["hydrophobic_total_side_rel_sum"] = df[df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
        # Buried
        result["aromatic_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
        result["negative_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(NEGATIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
        result["positive_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(POSITIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
        result["polar_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
        result["hydrophobic_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
        # Exposed
        result["aromatic_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
        result["negative_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(NEGATIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
        result["positive_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(POSITIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
        result["polar_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
        result["hydrophobic_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
        # Fraction exposed: type_exposed_sum / all_exposed
        all_exposed = float(exposed_df["total_side_rel"].sum()) if len(exposed_df) > 0 else 0.0
        if all_exposed > 0:
            result["fraction_hydrophobic_exposed"] = result["hydrophobic_exposed_total_side_rel_sum"] / all_exposed
            result["fraction_negative_exposed"] = result["negative_exposed_total_side_rel_sum"] / all_exposed
            result["fraction_positive_exposed"] = result["positive_exposed_total_side_rel_sum"] / all_exposed
        else:
            result["fraction_hydrophobic_exposed"] = result["fraction_negative_exposed"] = result["fraction_positive_exposed"] = np.nan
        # Whole-structure totals (buried + exposed) and ratios
        hydrophobic_total = result["hydrophobic_buried_total_side_rel_sum"] + result["hydrophobic_exposed_total_side_rel_sum"]
        negative_total = result["negative_buried_total_side_rel_sum"] + result["negative_exposed_total_side_rel_sum"]
        positive_total = result["positive_buried_total_side_rel_sum"] + result["positive_exposed_total_side_rel_sum"]
        polar_total = result["polar_buried_total_side_rel_sum"] + result["polar_exposed_total_side_rel_sum"]
        charged_total = negative_total + positive_total
        result["hydrophobic_to_charged_total_side_rel_ratio"] = (hydrophobic_total / charged_total) if charged_total > 0 else np.nan
        result["hydrophobic_to_polar_total_side_rel_ratio"] = (hydrophobic_total / polar_total) if polar_total > 0 else np.nan
        # Inter-chain
        if "inter_chain_contact" in df.columns:
            inter_mask = df["inter_chain_contact"].astype(str).str.lower() == "true"
            inter_chain_df = df[inter_mask]
            if len(inter_chain_df) > 0:
                result["aromatic_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() or _zero
                result["negative_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(NEGATIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() or _zero
                result["positive_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(POSITIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() or _zero
                result["polar_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() or _zero
                result["hydrophobic_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() or _zero
            else:
                result["aromatic_inter_chain_total_side_rel_sum"] = result["negative_inter_chain_total_side_rel_sum"] = result["positive_inter_chain_total_side_rel_sum"] = result["polar_inter_chain_total_side_rel_sum"] = result["hydrophobic_inter_chain_total_side_rel_sum"] = _zero
        else:
            result["aromatic_inter_chain_total_side_rel_sum"] = result["negative_inter_chain_total_side_rel_sum"] = result["positive_inter_chain_total_side_rel_sum"] = result["polar_inter_chain_total_side_rel_sum"] = result["hydrophobic_inter_chain_total_side_rel_sum"] = _zero
        # CDRs
        if n_cdr > 0:
            result["aromatic_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() or _zero
            result["negative_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(NEGATIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() or _zero
            result["positive_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(POSITIVE_CHARGED_RESIDUES)]["total_side_rel"].sum() or _zero
            result["polar_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() or _zero
            result["hydrophobic_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() or _zero
        else:
            result["aromatic_CDRs_total_side_rel_sum"] = result["negative_CDRs_total_side_rel_sum"] = result["positive_CDRs_total_side_rel_sum"] = result["polar_CDRs_total_side_rel_sum"] = result["hydrophobic_CDRs_total_side_rel_sum"] = _zero
        # SAP sum
        if "SAP" in df.columns:
            result["sap_sum"] = float(pd.to_numeric(df["SAP"], errors="coerce").sum())
        else:
            result["sap_sum"] = np.nan
    else:
        _nan_keys = (
            "aromatic_total_side_rel_sum", "negative_total_side_rel_sum", "positive_total_side_rel_sum",
            "polar_total_side_rel_sum", "hydrophobic_total_side_rel_sum",
            "aromatic_buried_total_side_rel_sum", "negative_buried_total_side_rel_sum", "positive_buried_total_side_rel_sum",
            "polar_buried_total_side_rel_sum", "hydrophobic_buried_total_side_rel_sum",
            "aromatic_exposed_total_side_rel_sum", "negative_exposed_total_side_rel_sum", "positive_exposed_total_side_rel_sum",
            "polar_exposed_total_side_rel_sum", "hydrophobic_exposed_total_side_rel_sum",
            "fraction_hydrophobic_exposed", "fraction_negative_exposed", "fraction_positive_exposed",
            "hydrophobic_to_charged_total_side_rel_ratio", "hydrophobic_to_polar_total_side_rel_ratio",
            "aromatic_inter_chain_total_side_rel_sum", "negative_inter_chain_total_side_rel_sum", "positive_inter_chain_total_side_rel_sum",
            "polar_inter_chain_total_side_rel_sum", "hydrophobic_inter_chain_total_side_rel_sum",
            "aromatic_CDRs_total_side_rel_sum", "negative_CDRs_total_side_rel_sum", "positive_CDRs_total_side_rel_sum",
            "polar_CDRs_total_side_rel_sum", "hydrophobic_CDRs_total_side_rel_sum",
            "sap_sum",
        )
        for k in _nan_keys:
            result[k] = np.nan if k in ("sap_sum", "fraction_hydrophobic_exposed", "fraction_negative_exposed", "fraction_positive_exposed", "hydrophobic_to_charged_total_side_rel_ratio", "hydrophobic_to_polar_total_side_rel_ratio") else 0.0

    # ----- Kyte-Doolittle sums (overall, beta sheet, buried, exposed) -----
    if "residue_name" in df.columns:
        result["hydrophobic_kyte_doolittle_sum"] = df["residue_name"].apply(
            lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
        ).sum()
        if "secondary_structure" in df.columns:
            beta_df = df[df["secondary_structure"] == "E"]
            if len(beta_df) > 0:
                result["fraction_hydrophobic_beta_sheet"] = beta_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / len(beta_df)
                result["fraction_gln_asn_beta_sheet"] = beta_df["residue_name"].isin(GLN_ASN_RESIDUES).sum() / len(beta_df)
                result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = beta_df["residue_name"].apply(
                    lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
                ).sum()
            else:
                result["fraction_hydrophobic_beta_sheet"] = result["fraction_gln_asn_beta_sheet"] = np.nan
                result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = 0.0
        else:
            result["fraction_hydrophobic_beta_sheet"] = result["fraction_gln_asn_beta_sheet"] = np.nan
            result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = 0.0
        if n_buried > 0:
            result["hydrophobic_buried_kyte_doolittle_sum"] = buried_df["residue_name"].apply(
                lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
            ).sum()
        else:
            result["hydrophobic_buried_kyte_doolittle_sum"] = 0.0
        if len(exposed_df) > 0:
            result["hydrophobic_exposed_kyte_doolittle_sum"] = exposed_df["residue_name"].apply(
                lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
            ).sum()
        else:
            result["hydrophobic_exposed_kyte_doolittle_sum"] = 0.0
    else:
        result["hydrophobic_kyte_doolittle_sum"] = 0.0
        result["fraction_hydrophobic_beta_sheet"] = result["fraction_gln_asn_beta_sheet"] = np.nan
        result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = result["hydrophobic_buried_kyte_doolittle_sum"] = result["hydrophobic_exposed_kyte_doolittle_sum"] = 0.0

    # ----- Kyte-Doolittle mean and SASA-weighted -----
    if "residue_name" in df.columns and "total_side_rel" in df.columns:
        df["kyte_doolittle"] = df["residue_name"].apply(
            lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
        )
        result["kyte_doolittle_mean"] = df["kyte_doolittle"].mean()
        valid = df["total_side_rel"].notna() & df["kyte_doolittle"].notna()
        if valid.sum() > 0:
            wsum = (df.loc[valid, "total_side_rel"] * df.loc[valid, "kyte_doolittle"]).sum()
            asa_sum = df.loc[valid, "total_side_rel"].sum()
            result["kyte_doolittle_weighted_by_side_asa"] = (wsum / asa_sum) if asa_sum > 0 else np.nan
        else:
            result["kyte_doolittle_weighted_by_side_asa"] = np.nan
    else:
        result["kyte_doolittle_mean"] = result["kyte_doolittle_weighted_by_side_asa"] = np.nan

    # ----- Interface: ratio hydrophobic/polar residues -----
    if "inter_chain_contact" in df.columns and "residue_name" in df.columns:
        inter = df["inter_chain_contact"].astype(str).str.lower() == "true"
        inter_df = df[inter]
        if len(inter_df) > 0:
            polar_inter = inter_df["residue_name"].isin(POLAR_RESIDUES).sum()
            hydro_inter = inter_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum()
            result["ratio_hydrophobic_to_polar_residues_interface"] = (hydro_inter / polar_inter) if polar_inter > 0 else np.nan
        else:
            result["ratio_hydrophobic_to_polar_residues_interface"] = np.nan
        result["hydrophobic_to_polar_sasa_interface_ratio"] = np.nan  # optional SASA-based; not computed here
    else:
        result["ratio_hydrophobic_to_polar_residues_interface"] = result["hydrophobic_to_polar_sasa_interface_ratio"] = np.nan

    # ----- Fraction hydrophobic at interface; Kyte-Doolittle sum at interface -----
    if "inter_chain_contact" in df.columns and "residue_name" in df.columns:
        inter = df["inter_chain_contact"].astype(str).str.lower() == "true"
        inter_df = df[inter]
        if len(inter_df) > 0:
            result["fraction_hydrophobic_inter_chain"] = inter_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / len(inter_df)
            result["hydrophobic_inter_chain_kyte_doolittle_sum"] = inter_df["residue_name"].apply(
                lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
            ).sum()
        else:
            result["fraction_hydrophobic_inter_chain"] = np.nan
            result["hydrophobic_inter_chain_kyte_doolittle_sum"] = 0.0
    else:
        result["fraction_hydrophobic_inter_chain"] = np.nan
        result["hydrophobic_inter_chain_kyte_doolittle_sum"] = 0.0

    # ----- Fraction hydrophobic in CDRs; Kyte-Doolittle sum in CDRs -----
    if n_cdr > 0 and "residue_name" in df.columns:
        result["fraction_hydrophobic_CDRs"] = range_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / n_cdr
        result["hydrophobic_CDRs_kyte_doolittle_sum"] = range_df["residue_name"].apply(
            lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
        ).sum()
    else:
        result["fraction_hydrophobic_CDRs"] = np.nan
        result["hydrophobic_CDRs_kyte_doolittle_sum"] = 0.0

    # ----- Cluster max total_side_rel -----
    for col in CLUSTER_LABEL_COLS:
        feat = col.replace("_cluster_labels", "_cluster_max_total_side_rel")
        if col in df.columns and "total_side_rel" in df.columns:
            d = df.copy()
            d[col] = d[col].astype(str).str.strip()
            valid = d[col].str.len() > 0
            if valid.any():
                sums = d.loc[valid].groupby(col)["total_side_rel"].sum()
                result[feat] = float(sums.max()) if len(sums) > 0 else np.nan
            else:
                result[feat] = np.nan
        else:
            result[feat] = np.nan

    # ----- Row counts -----
    result["n_total_rows"] = n_total
    result["n_filtered_rows"] = n_buried
    result["n_beta_sheet_rows"] = len(df[df["secondary_structure"] == "E"]) if "secondary_structure" in df.columns else 0
    result["n_exposed_rows"] = len(exposed_df)

    # ----- Inter-chain density means/medians -----
    if "inter_chain_contact" in df.columns:
        inter = df["inter_chain_contact"].astype(str).str.lower() == "true"
        inter_df = df[inter]
        if len(inter_df) > 0:
            result["hbond_density_inter_chain_mean"] = inter_df["hbond_density"].mean() if "hbond_density" in df.columns else np.nan
            result["salt_bridge_density_inter_chain_median"] = inter_df["salt_bridge_density"].median() if "salt_bridge_density" in df.columns else np.nan
            result["hbond_energy_dssp_density_inter_chain_median"] = inter_df["hbond_energy_dssp_density"].median() if "hbond_energy_dssp_density" in df.columns else np.nan
        else:
            result["hbond_density_inter_chain_mean"] = result["salt_bridge_density_inter_chain_median"] = result["hbond_energy_dssp_density_inter_chain_median"] = np.nan
    else:
        result["hbond_density_inter_chain_mean"] = result["salt_bridge_density_inter_chain_median"] = result["hbond_energy_dssp_density_inter_chain_median"] = np.nan

    # ----- Structure-level from first row (pass-through) -----
    ripley = ["ripley_k_negative", "ripley_k_positive", "ripley_k_aromatic", "ripley_k_hydrophobic", "ripley_k_polar"]
    psh_ppc_pnc = ["psh_all_surface_exposed", "psh_cdr_vicinity", "ppc_all_surface_exposed", "ppc_cdr_vicinity", "pnc_all_surface_exposed", "pnc_cdr_vicinity"]
    whole = (
        ["dipole_moment_magnitude", "largest_hbond_component_size", "net_charge", "protein_pi", "scm_score"]
        + [f"net_charge_pH{p}" for p in [4, 5, 6, 7, 8, 9, 10]]
        + [f"scm_score_pH{p}" for p in [4, 5, 6, 7, 8, 9, 10]]
    )
    for col in ripley + psh_ppc_pnc + whole:
        if col in df.columns and len(df) > 0:
            result[col] = pd.to_numeric(df[col], errors="coerce").iloc[0]
        else:
            result[col] = np.nan

    return result


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute structure-level downstream descriptors from developability per-residue CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to developability CSV (per-residue).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path (default: stdout).")
    parser.add_argument("--structure-id", type=str, default=None)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--heavy", type=str, default=None)
    parser.add_argument("--light", type=str, default=None)
    parser.add_argument("--inter-chain-buried-sasa", type=float, default=None)
    parser.add_argument("--pdb", type=Path, default=None, help="PDB path (optional; currently used only for metadata).")
    parser.add_argument("--heavy-chain", type=str, default="H")
    parser.add_argument("--light-chain", type=str, default="L")
    parser.add_argument("--pH", type=float, default=7.0)
    parser.add_argument("--use-descriptors-motif", action="store_true", help="(Deprecated) no effect; overlapping motif counts are always used.")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    out = compute_downstream_descriptors(
        df,
        structure_id=args.structure_id,
        base=args.base,
        heavy=args.heavy,
        light=args.light,
        inter_chain_buried_sasa=args.inter_chain_buried_sasa,
        pdb_path=args.pdb,
        heavy_chain_id=args.heavy_chain,
        light_chain_id=args.light_chain,
        pH=args.pH,
        use_descriptors_motif=args.use_descriptors_motif,
    )
    # Convert nan to None for JSON
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, (np.floating, float)) and np.isnan(obj):
            return None
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        return obj

    out = _sanitize(out)
    text = json.dumps(out, indent=2)
    if args.output is None:
        print(text)
    else:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

