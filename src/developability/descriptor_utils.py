import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

from developability.structure_context import StructureContext, ResKey4
from utils.geometry import is_backbone_atom
from utils.parsers import Atom, parse_structure, residue_key_from_atom, SASAEntry
from utils.chemistry import get_standard_residue_pka


def _residue_keys_cache_key(atoms: List["Atom"]) -> Tuple[ResKey4, ...]:
    """
    Stable cache key for residue-annotated structures.

    This intentionally does NOT depend on atom ordering: the same structure can
    be represented by different atom list orderings (different parsers, filtering),
    and order-dependent cache keys would cause cache misses and duplicated caches.
    """
    keys = {residue_key_from_atom(a) for a in atoms}
    return tuple(sorted(keys, key=lambda k: (k[2], k[1], k[3], k[0])))


def is_hydrogen_atom(atom: Atom) -> bool:
    """
    Best-effort hydrogen detection that does not rely solely on the `element`
    field being correct/clean.
    """
    elem = (getattr(atom, "element", "") or "").strip().upper()
    if elem in {"H", "D"}:
        return True
    name = (getattr(atom, "name", "") or "").strip().upper()
    if not name:
        return False
    # Common PDB hydrogen naming patterns: H, HA, HB2, 1H, 2HA, etc.
    if name.startswith("H"):
        return True
    if len(name) >= 2 and name[0].isdigit() and name[1] == "H":
        return True
    return False


def _sasa_exposure_cache_key(
    sasa_data: Dict[ResKey4, SASAEntry],
    sasa_cutoff: float,
) -> Tuple[Tuple[ResKey4, Optional[float]], ...]:
    items: List[Tuple[ResKey4, Optional[float]]] = []
    for res_key, entry in sasa_data.items():
        value = getattr(entry, "total_side_rel", None)
        items.append((res_key, None if value is None else float(value)))
    # include cutoff as a synthetic entry to distinguish different thresholds
    items.append((("CUTOFF", -1, "", ""), float(sasa_cutoff)))  # type: ignore[arg-type]
    return tuple(sorted(items))


_ATOMS_CACHE: Dict[str, List[Atom]] = {}
CDR_RANGES_CA = [(27, 38), (56, 65), (105, 117)]

_CDR_RESIDUE_CACHE: Dict[int, str] = {}  # residue_number -> "CDR" | "framework"
_RESIDUE_REGION_CACHE: Dict[
    Tuple[Tuple[str, int, str, str], ...],
    Dict[Tuple[str, int, str, str], str],
] = {}

_STRUCTURE_CONTEXT_CACHE: Dict[Tuple[str, Optional[str]], "StructureContext"] = {}
_RES_COUNT_CACHE: Dict[str, int] = {}
_RES_SEQ_INDEX_CACHE: Dict[
    Tuple[Tuple[str, int, str, str], ...],
    Dict[Tuple[str, int, str, str], int],
] = {}
_RES_INDEX_TO_KEY_CACHE: Dict[
    Tuple[Tuple[str, int, str, str], ...],
    Dict[str, Dict[int, Tuple[str, int, str, str]]],
] = {}

_ATOM_LOOKUP_CACHE: Dict[
    Tuple[Tuple[str, int, str, str], ...],
    Dict[Tuple[str, int, str, str, str], Atom],
] = {}

_INTER_CHAIN_INTERFACE_CACHE: Dict[str, Set[ResKey4]] = {}

# One-time warning cache for insertion-code fallback in SASA lookup.
_SASA_INSERTION_FALLBACK_WARNED: Set[Tuple[ResKey4, ResKey4]] = set()

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import logging
import os
import math
from collections import defaultdict

logger = logging.getLogger(__name__)
import numpy as np
from scipy.spatial import cKDTree
from scipy import optimize
from sklearn.cluster import DBSCAN
import pandas as pd
import re

KYTE_DOOLITTLE_SCORES = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}
KD_MIN = min(KYTE_DOOLITTLE_SCORES.values())
KD_MAX = max(KYTE_DOOLITTLE_SCORES.values())

def normalize_hydropathy(res_name: str) -> Optional[float]:
    """
    Normalize Kyte-Doolittle hydropathy score for a residue to [1, 2].
    Returns None if residue has no defined score.
    """
    score = KYTE_DOOLITTLE_SCORES.get(res_name)
    if score is None:
        return None
    if KD_MAX == KD_MIN:
        return 1.5
    return 1.0 + (score - KD_MIN) / (KD_MAX - KD_MIN)


# Charge-residue sets.
#
# Keep geometric salt-bridge detection separate from charge counting:
# salt bridges still depend on explicit salt-bridge partner atoms
# (see POSITIVE_ATOMS / NEGATIVE_ATOMS), while the charged/residue classification
# is used for charge/pI style metrics and charge-based clustering terms.
POSITIVE_CHARGED_RESIDUES = frozenset({"LYS", "ARG", "HIS"})

AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP"})
HYDROPHOBIC_RESIDUES = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS"}
)
POLAR_RESIDUES = frozenset(
    {"SER", "THR", "ASN", "GLN", "TYR", "GLU", "ASP", "LYS", "ARG", "HIS"}
)

POSITIVE_ATOMS = frozenset(
    {
        ("NH1", "ARG"),
        ("NH2", "ARG"),
        ("NZ", "LYS"),
        ("ND1", "HIS"),
        ("NE2", "HIS"),
    }
)

NEGATIVE_ATOMS = frozenset(
    {
        ("OD1", "ASP"),
        ("OD2", "ASP"),
        ("OE1", "GLU"),
        ("OE2", "GLU"),
        # Phenolic / thiol deprotonation.
        ("OH", "TYR"),
        ("SG", "CYS"),
    }
)

# Residues that can be negatively charged at a given pH (charge/pI logic).
NEGATIVE_CHARGED_RESIDUES = frozenset({res for _atom, res in NEGATIVE_ATOMS})

INTER_CHAIN_INTERFACE_CUTOFF = 5.0
MAX_SALT_BRIDGE_DISTANCE = 4.0

DONORS_ANY = frozenset({"N"})
DONOR_EXCLUDED = frozenset({("N", "PRO")})

DONOR_METADATA: Dict[Tuple[str, str], Tuple[str, int]] = {
    ("NE2", "GLN"): ("CD", 2),
    ("ND2", "ASN"): ("CG", 2),
    ("NE", "ARG"): ("CZ", 1),
    ("NH1", "ARG"): ("CZ", 2),
    ("NH2", "ARG"): ("CZ", 2),
    ("NZ", "LYS"): ("CE", 3),
    ("ND1", "HIS"): ("CG", 1),
    ("NE2", "HIS"): ("CD2", 1),
    ("OG", "SER"): ("CB", 1),
    ("OG1", "THR"): ("CB", 1),
    ("OH", "TYR"): ("CZ", 1),
}

DONOR_INFO: Dict[Tuple[str, str], str] = {
    key: base for key, (base, _max_hbonds) in DONOR_METADATA.items()
}

DONOR_MAX_HBONDS: Dict[Tuple[str, str], int] = {
    ("N", "ANY"): 1,  # any residue except PRO (filtered by DONOR_EXCLUDED)
    **{key: max_hbonds for key, (_base, max_hbonds) in DONOR_METADATA.items()},
}

ACCEPTORS_ANY = frozenset({"O"})

# Acceptor metadata: (atom_name, residue_name) -> max_hbonds
ACCEPTOR_METADATA: Dict[Tuple[str, str], int] = {
    ("O", "ANY"): 2,
    ("OE1", "GLN"): 1,
    ("OE2", "GLU"): 2,
    ("OD1", "ASN"): 1,
    ("OD2", "ASP"): 2,
    ("ND1", "HIS"): 1,
    ("NE2", "HIS"): 1,
    ("OG", "SER"): 1,
    ("OG1", "THR"): 1,
    ("OH", "TYR"): 1,
}

ACCEPTORS_SPECIFIC = frozenset(
    {
        key
        for key in ACCEPTOR_METADATA.keys()
        if not (key[0] == "O" and key[1] == "ANY")
    }
)

ACCEPTOR_MAX_HBONDS: Dict[Tuple[str, str], int] = dict(ACCEPTOR_METADATA)
CHARGED_THRESHOLD = 0.1  # |fractional charge| above this is considered charged
_CHARGE_CACHE: Dict[Tuple[str, Optional[float], float], Tuple[float, Optional[str]]] = {}


def _residue_fractional_charge_at_pH(
    residue_name: str,
    pka_value: Optional[float],
    pH: float,
) -> float:
    """fractional charge at pH (Henderson-Hasselbalch eq)"""
    res_name = (residue_name or "").strip().upper()
    effective_pka = pka_value if pka_value is not None else get_standard_residue_pka(res_name)
    cache_key = (res_name, effective_pka, float(pH))
    cached = _CHARGE_CACHE.get(cache_key)
    if cached is not None:
        return cached[0]

    # Biologically grounded fallback: if PropKa did not provide a pKa for a
    # titratable residue, use a standard residue pKa value instead of forcing
    # an always-charged (ASP/GLU/ARG/LYS) or never-charged (HIS) state.
    pka_value = effective_pka

    if pka_value is None:
        q = 0.0
    elif res_name in NEGATIVE_CHARGED_RESIDUES:
        # Acidic (deprotonated) state carries -1 charge.
        q = -1.0 / (1.0 + np.power(10.0, pka_value - pH))
    elif res_name in POSITIVE_CHARGED_RESIDUES:
        q = 1.0 / (1.0 + np.power(10.0, pH - pka_value))
    else:
        q = 0.0
    if q > CHARGED_THRESHOLD:
        sign = "positive"
    elif q < -CHARGED_THRESHOLD:
        sign = "negative"
    else:
        sign = None
    _CHARGE_CACHE[cache_key] = (q, sign)
    return q


def is_residue_charged(
    residue_name: str, pka_value: Optional[float], pH: float
) -> Optional[str]:
    res_name = (residue_name or "").strip().upper()
    effective_pka = pka_value if pka_value is not None else get_standard_residue_pka(res_name)
    cache_key = (res_name, effective_pka, float(pH))
    cached = _CHARGE_CACHE.get(cache_key)
    if cached is None:
        _residue_fractional_charge_at_pH(res_name, effective_pka, pH)
        cached = _CHARGE_CACHE.get(cache_key)
    return cached[1] if cached is not None else None


def get_aromatic_residue_keys(pdb_path: str) -> Set[ResKey4]:
    return get_residue_keys_by_type(pdb_path, AROMATIC_RESIDUES)

def _get_charged_residues_at_pH(
    atoms: List[Atom],
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
) -> Tuple[Set[Tuple[str, int, str, str]], Set[Tuple[str, int, str, str]]]:
    """
    Return sets of positively and negatively charged residues at a given pH,
    using PropKa-style per-residue pKa data when available and type-based
    defaults otherwise.
    """
    positive: Set[Tuple[str, int, str, str]] = set()
    negative: Set[Tuple[str, int, str, str]] = set()
    seen: Set[Tuple[str, int, str, str]] = set()
    for atom in atoms:
        res_key = residue_key_from_atom(atom)
        if res_key in seen:
            continue
        seen.add(res_key)
        res_name = atom.residue_name
        pka_val = pka_data.get(res_key)
        if pka_val is None:
            pka_val = get_standard_residue_pka(res_name)
        sign = is_residue_charged(res_name, pka_val, pH)
        if sign == "positive":
            positive.add(res_key)
        elif sign == "negative":
            negative.add(res_key)
    return positive, negative


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
    """
    Aggregate DSSP H-bond (offset, energy) data into per-residue raw weights
    (SASA * |energy|), counts, and inter-chain flags.
    """
    residue_sasa_cache: Dict[ResKey4, float] = {}
    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    residue_counts: Dict[Tuple[str, int, str, str], int] = defaultdict(int)
    residue_inter_chain: Dict[Tuple[str, int, str, str], bool] = defaultdict(bool)

    def get_cached_sasa(res_key: ResKey4) -> float:
        w = residue_sasa_cache.get(res_key)
        if w is not None:
            return w
        w = get_residue_sasa_weight(res_key, sasa_data, use_side_chain=False)
        residue_sasa_cache[res_key] = w
        return w

    for residue_key, hbond_pairs in dssp_hbonds.items():
        if residue_key not in pdb_to_dssp_seq:
            continue
        dssp_seq_num = pdb_to_dssp_seq[residue_key]
        residue_weight = get_cached_sasa(residue_key)
        if residue_weight == 0.0:
            continue

        for residue_offset, energy in hbond_pairs:
            target_dssp_seq = dssp_seq_num + residue_offset
            if target_dssp_seq not in dssp_seq_to_pdb:
                continue
            target_residue_key = dssp_seq_to_pdb[target_dssp_seq]
            is_inter_chain = residue_key[2] != target_residue_key[2]
            target_weight = get_cached_sasa(target_residue_key)
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


def _residue_category_group(
    res_name: str,
    pka_val: Optional[float],
    pH: float,
) -> Optional[str]:
    """
    Classify residue into charge/type group for clustering/Ripley (with HIS as positive).
    Returns "negative" | "positive" | "aromatic" | "hydrophobic" | "polar", or None.
    When pKa is None, charged residues are assigned by type (ASP/GLU -> negative, LYS/ARG/HIS -> positive).
    Aromatic (PHE, TYR, TRP) is checked before hydrophobic/polar so they get a dedicated group.
    """
    if res_name in NEGATIVE_CHARGED_RESIDUES:
        if pka_val is None:
            pka_val = get_standard_residue_pka(res_name)
        if is_residue_charged(res_name, pka_val, pH):
            return "negative"
        return None
    if res_name in POSITIVE_CHARGED_RESIDUES:
        if pka_val is None:
            pka_val = get_standard_residue_pka(res_name)
        if is_residue_charged(res_name, pka_val, pH):
            return "positive"
        return None
    if res_name in AROMATIC_RESIDUES:
        return "aromatic"
    if res_name in HYDROPHOBIC_RESIDUES:
        return "hydrophobic"
    if res_name in POLAR_RESIDUES:
        return "polar"
    return None


_EXPOSED_RESIDUE_CACHE: Dict[
    Tuple[Tuple[ResKey4, Optional[float]], ...],
    Dict[ResKey4, bool],
] = {}


def get_exposed_residues(
    sasa_data: Dict[ResKey4, SASAEntry],
    sasa_cutoff: float,
) -> Dict[ResKey4, bool]:

    key = _sasa_exposure_cache_key(sasa_data, float(sasa_cutoff))
    cached = _EXPOSED_RESIDUE_CACHE.get(key)
    if cached is not None:
        return cached

    exposure: Dict[ResKey4, bool] = {}
    for res_key, entry in sasa_data.items():
        if entry is None or getattr(entry, "total_side_rel", None) is None:
            exposure[res_key] = False
        else:
            exposure[res_key] = bool(entry.total_side_rel > sasa_cutoff)

    _EXPOSED_RESIDUE_CACHE[key] = exposure
    return exposure


def _compute_inter_chain_interface_from_by_chain(
    by_chain: Dict[str, List[Atom]],
) -> Set[ResKey4]:
    """
    Compute inter-chain interface residues from a mapping of chain -> atoms using
    a distance cutoff on heavy-atom coordinates.
    """
    interface: Set[ResKey4] = set()
    chains = list(by_chain.keys())
    cutoff = INTER_CHAIN_INTERFACE_CUTOFF
    for i, c1 in enumerate(chains):
        for c2 in chains[i + 1 :]:
            coords1 = np.array([[a.x, a.y, a.z] for a in by_chain[c1]], dtype=np.float64)
            coords2 = np.array([[a.x, a.y, a.z] for a in by_chain[c2]], dtype=np.float64)
            tree1 = cKDTree(coords1)
            tree2 = cKDTree(coords2)
            pairs = tree1.query_ball_tree(tree2, cutoff)
            # Skip if no atom in c1 has any neighbor in c2 within cutoff
            if not any(pairs):
                continue

            atoms1 = by_chain[c1]
            atoms2 = by_chain[c2]
            for i_atom, neighbor_indices in enumerate(pairs):
                if not neighbor_indices:
                    continue
                interface.add(residue_key_from_atom(atoms1[i_atom]))
                for j_atom in neighbor_indices:
                    interface.add(residue_key_from_atom(atoms2[j_atom]))
    return interface


def get_inter_chain_interface_residues(
    pdb_path: str,
    by_chain_heavy: Optional[Dict[str, List[Atom]]] = None,
) -> Set[ResKey4]:
    """
    Return the set of inter-chain interface residues for a PDB structure, using
    a cached result keyed by absolute path.
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
            if not is_hydrogen_atom(a):
                by_chain[a.chain].append(a)
        interface = _compute_inter_chain_interface_from_by_chain(by_chain)

    _INTER_CHAIN_INTERFACE_CACHE[abs_path] = interface
    return interface

def get_residue_region(residue_number: int) -> str:
    """Return 'CDR' if residue_number is in any CDR range, else 'framework'. Cached by residue number."""
    cached = _CDR_RESIDUE_CACHE.get(residue_number)
    if cached is not None:
        return cached
    for start, end in CDR_RANGES_CA:
        if start <= residue_number <= end:
            _CDR_RESIDUE_CACHE[residue_number] = "CDR"
            return "CDR"
    _CDR_RESIDUE_CACHE[residue_number] = "framework"
    return "framework"


def iter_unique_residues(atoms: List[Atom]) -> Iterable[ResKey4]:
    """Yield unique residue keys (4-tuples) in first-seen order for a list of atoms."""
    seen: Set[ResKey4] = set()
    for atom in atoms:
        key = residue_key_from_atom(atom)
        if key in seen:
            continue
        seen.add(key)
        yield key


def get_residue_region_map(atoms: List[Atom]) -> Dict[Tuple[str, int, str, str], str]:
    """Map each residue (by 4-tuple key) to "CDR" or "framework" (numeric part of residue number)."""
    cache_key = _residue_keys_cache_key(atoms)
    cached = _RESIDUE_REGION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    out: Dict[Tuple[str, int, str, str], str] = {}
    for atom in atoms:
        key = residue_key_from_atom(atom)
        if key in out:
            continue
        out[key] = get_residue_region(atom.residue_number)

    _RESIDUE_REGION_CACHE[cache_key] = out
    return out


def _get_atoms_for_path(pdb_path: str) -> List[Atom]:
    """Return cached or freshly parsed atoms for pdb_path (abs path as cache key)."""
    abs_path = os.path.abspath(pdb_path)
    cached = _ATOMS_CACHE.get(abs_path)
    if cached is not None:
        return cached
    atoms = parse_structure(pdb_path)
    _ATOMS_CACHE[abs_path] = atoms
    return atoms


def _get_residue_seq_index(
    atoms: List[Atom],
) -> Dict[Tuple[str, int, str, str], int]:
    """
    Build and cache per-chain ordered residue indices for a list of atoms.

    Returns a mapping from 4-tuple residue keys to 0-based sequence indices
    within each chain. Also populates _RES_INDEX_TO_KEY_CACHE with the reverse
    mapping (chain -> index -> residue key). Cached by a stable residue-key
    sequence rather than id(atoms) to avoid leaking distinct list objects.
    """
    cache_key = _residue_keys_cache_key(atoms)
    if cache_key in _RES_SEQ_INDEX_CACHE:
        return _RES_SEQ_INDEX_CACHE[cache_key]

    # Deterministic residue ordering: within each chain sort by (res_num, insertion_code, res_name).
    # This makes indices invariant to atom list ordering and avoids subtle downstream differences
    # (e.g. backbone separation checks) if atom ordering changes.
    chain_to_residue_set: Dict[str, Set[ResKey4]] = defaultdict(set)
    for atom in atoms:
        rk = residue_key_from_atom(atom)
        chain_to_residue_set[rk[2]].add(rk)

    seq_index: Dict[Tuple[str, int, str, str], int] = {}
    index_to_key: Dict[str, Dict[int, Tuple[str, int, str, str]]] = {}
    for chain, residue_set in chain_to_residue_set.items():
        residues = sorted(residue_set, key=lambda k: (k[1], k[3], k[0]))
        chain_index_map: Dict[int, Tuple[str, int, str, str]] = {}
        for idx, res_key in enumerate(residues):
            seq_index[res_key] = idx
            chain_index_map[idx] = res_key
        index_to_key[chain] = chain_index_map

    _RES_SEQ_INDEX_CACHE[cache_key] = seq_index
    _RES_INDEX_TO_KEY_CACHE[cache_key] = index_to_key
    return seq_index


def is_donor(atom: Atom) -> bool:
    """
    Return True if atom is an H-bond donor according to backbone/sidechain rules.
    """
    key = (atom.name, atom.residue_name)
    return key not in DONOR_EXCLUDED and (
        atom.name in DONORS_ANY or key in DONOR_METADATA
    )


def is_acceptor(atom: Atom) -> bool:
    """
    Return True if atom is an H-bond acceptor according to backbone/sidechain rules.
    """
    return (atom.name, atom.residue_name) in ACCEPTORS_SPECIFIC or atom.name in ACCEPTORS_ANY


def donor_max_hbonds(atom: Atom) -> int:
    """
    Maximum number of H-bonds allowed for a donor atom.
    Backbone N uses the generic ("N", "ANY") rule; other donors fall back to 1.
    """
    key = (atom.name, atom.residue_name)
    if key in DONOR_MAX_HBONDS:
        return DONOR_MAX_HBONDS[key]
    if atom.name == "N":
        return DONOR_MAX_HBONDS.get(("N", "ANY"), 1)
    return 1


def acceptor_max_hbonds(atom: Atom) -> int:
    """
    Maximum number of H-bonds allowed for an acceptor atom.
    Backbone O uses the generic ("O", "ANY") rule; other acceptors fall back to 2.
    """
    key = (atom.name, atom.residue_name)
    if key in ACCEPTOR_MAX_HBONDS:
        return ACCEPTOR_MAX_HBONDS[key]
    if atom.name == "O":
        return ACCEPTOR_MAX_HBONDS.get(("O", "ANY"), 2)
    return 2


def get_donor_base_atom(
    donor: Atom,
    atoms: List[Atom],
    backbone_base_cache: Optional[Dict[ResKey4, Atom]] = None,
) -> Optional[Atom]:
    """
    the base atom is the atom covalently bonded to the donor:
    - backbone N → previous residue C (residue_number - 1)
    - side-chain donors → bonded heavy atom in same residue

    returns None if not found
    """
    cache_key = _residue_keys_cache_key(atoms)
    if cache_key not in _ATOM_LOOKUP_CACHE:
        _ATOM_LOOKUP_CACHE[cache_key] = {
            (atom.chain, atom.residue_number, atom.insertion_code, atom.name): atom
            for atom in atoms
        }
    atom_lookup = _ATOM_LOOKUP_CACHE[cache_key]

    if donor.name == "N":
        seq_index = _get_residue_seq_index(atoms)
        donor_res_key = residue_key_from_atom(donor)
        donor_idx = seq_index.get(donor_res_key)
        if donor_idx is None or donor_idx == 0:
            # no previous residue in sequence (true N-terminus or unmapped), so
            # no backbone base atom for angle calculation
            return None

        donor_chain = donor.chain
        if backbone_base_cache is None:
            backbone_base_cache = {}
        chain_index_to_res = _RES_INDEX_TO_KEY_CACHE.get(cache_key, {})
        prev_chain_map = chain_index_to_res.get(donor_chain, {})
        prev_res_key = prev_chain_map.get(donor_idx - 1)

        if prev_res_key is None:
            return None

        prev_resname, prev_resnum, prev_chain, prev_inscode = prev_res_key
        key_prev_c = (prev_chain, prev_resnum, prev_inscode, "C")
        base = atom_lookup.get(key_prev_c)
        if base is not None:
            backbone_base_cache[donor_res_key] = base
            return base

        has_prev_residue_atoms = any(
            ch == prev_chain and res == prev_resnum and ins == prev_inscode
            for (ch, res, ins, _name) in atom_lookup.keys()
        )
        if has_prev_residue_atoms:
            raise ValueError(
                f"Backbone donor base atom not found for N in residue "
                f"{donor.residue_name} {donor.residue_number} chain {donor.chain}"
            )
        return None

    key = (donor.name, donor.residue_name)
    if key not in DONOR_METADATA:
        raise ValueError(
            f"Side-chain donor base atom definition missing for {donor.name} in "
            f"{donor.residue_name} {donor.residue_number} chain {donor.chain}"
        )

    base_name, _max_hbonds = DONOR_METADATA[key]
    base_key = (donor.chain, donor.residue_number, donor.insertion_code, base_name)
    base = atom_lookup.get(base_key)
    if base is None:
        raise ValueError(
            f"Side-chain donor base atom {base_name} not found for donor {donor.name} in "
            f"{donor.residue_name} {donor.residue_number} chain {donor.chain}"
        )
    return base


def _get_structure_context(
    pdb_path: str,
    sasa_path: Optional[str] = None,
) -> StructureContext:
    """Return a cached or new StructureContext for (pdb_path, sasa_path)."""
    key = (os.path.abspath(pdb_path), sasa_path)
    ctx = _STRUCTURE_CONTEXT_CACHE.get(key)
    if ctx is None:
        ctx = StructureContext(pdb_path, sasa_path=sasa_path)
        _STRUCTURE_CONTEXT_CACHE[key] = ctx
    return ctx


def _count_residues_in_pdb(pdb_path: str) -> int:
    """Return the number of unique residues in a PDB structure."""
    abs_path = os.path.abspath(pdb_path)
    cached = _RES_COUNT_CACHE.get(abs_path)
    if cached is not None:
        return cached
    atoms = _get_atoms_for_path(pdb_path)
    count = len(set(iter_unique_residues(atoms)))
    _RES_COUNT_CACHE[abs_path] = count
    return count


def _get_region_map_for_pdb(pdb_path: str) -> Dict[Tuple[str, int, str, str], str]:
    """Region map (ResKey4 -> 'CDR'|'framework') for this PDB, using the same residue-key-based cache as get_residue_region_map(atoms)."""
    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        return {}
    return get_residue_region_map(atoms)


# ---- SASA helpers (scaled rel SASA, per-residue / per-atom weights) ----

def _rel_sasa_scaled(entry: Optional[SASAEntry], attr: str) -> float:
    if entry is None:
        return 0.0
    value = getattr(entry, attr, None)
    if value is None:
        return 0.0
    return float(value) * 100.0


def residue_side_sasa(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
) -> float:
    entry = sasa_data.get(residue_key)
    return _rel_sasa_scaled(entry, "total_side_rel")

def residue_side_sasa_rel(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
) -> float:
    """Relative side-chain SASA as a fraction in [0, 1] (not percent-scaled)."""
    entry = sasa_data.get(residue_key)
    if entry is None:
        return 0.0
    v = getattr(entry, "total_side_rel", None)
    return float(v) if v is not None else 0.0

def residue_side_sasa_abs(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
) -> float:
    entry = sasa_data.get(residue_key)
    if entry is None:
        return 0.0
    return float(getattr(entry, "total_side_abs", 0.0) or 0.0)


def residue_main_sasa(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
) -> float:
    entry = sasa_data.get(residue_key)
    return _rel_sasa_scaled(entry, "main_chain_rel")

def residue_main_sasa_abs(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
) -> float:
    entry = sasa_data.get(residue_key)
    if entry is None:
        return 0.0
    return float(getattr(entry, "main_chain_abs", 0.0) or 0.0)


def atom_sasa_weight(atom: Atom, sasa_data: Dict[ResKey4, SASAEntry]) -> float:
    key = residue_key_from_atom(atom)
    entry = sasa_data.get(key)
    if is_backbone_atom(atom.name):
        return _rel_sasa_scaled(entry, "main_chain_rel")
    return _rel_sasa_scaled(entry, "total_side_rel")


def get_residue_sasa_weight(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
    use_side_chain: bool = True,
) -> float:
    if use_side_chain:
        return residue_side_sasa(residue_key, sasa_data)
    return residue_main_sasa(residue_key, sasa_data)


def _sasa_lookup(
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
    key: ResKey4,
) -> Dict[str, Optional[float]]:
    """Lookup SASA entry by 4-tuple residue key (with 3-tuple fallback)."""
    entry = sasa_output_data.get(key)
    if entry is not None:
        return entry
    if len(key) == 4:
        fallback_key = (key[0], key[1], key[2], "")
        fallback = sasa_output_data.get(fallback_key)
        if fallback is not None:
            # Falling back from a non-empty insertion code to "" is usually a key mismatch
            # that would otherwise be silent. Warn once per (missing_key -> fallback_key).
            if key[3] and (key, fallback_key) not in _SASA_INSERTION_FALLBACK_WARNED:
                _SASA_INSERTION_FALLBACK_WARNED.add((key, fallback_key))
                logger.warning(
                    "SASA lookup fell back from insertion-coded residue key %r to %r. "
                    "This can hide insertion-code mismatches.",
                    key,
                    fallback_key,
                )
            return fallback
        return {}
    return {}


# ---- PDB residue keys (by path, by type) ----

_PDB_RESIDUE_KEYS_CACHE: Dict[str, Set[ResKey4]] = {}
_RESIDUE_TYPE_CACHE: Dict[Tuple[str, frozenset], Set[ResKey4]] = {}


def _get_pdb_residue_keys(pdb_path: str) -> Set[ResKey4]:
    """Return the set of residue keys for the PDB, using cache or parsing the structure."""
    abs_path = os.path.abspath(pdb_path)
    all_residue_keys = _PDB_RESIDUE_KEYS_CACHE.get(abs_path)
    if all_residue_keys is None:
        atoms = _get_atoms_for_path(pdb_path)
        if not atoms:
            _PDB_RESIDUE_KEYS_CACHE[abs_path] = set()
            return set()
        all_residue_keys = set(iter_unique_residues(atoms))
        _PDB_RESIDUE_KEYS_CACHE[abs_path] = all_residue_keys
    return all_residue_keys


def get_residue_keys_by_type(
    pdb_path: str,
    residue_types: Iterable[str],
) -> Set[ResKey4]:
    abs_path = os.path.abspath(pdb_path)
    type_set = frozenset(residue_types)
    cache_key = (abs_path, type_set)
    cached = _RESIDUE_TYPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    all_residue_keys = _get_pdb_residue_keys(pdb_path)
    result = {k for k in all_residue_keys if k[0] in type_set}
    _RESIDUE_TYPE_CACHE[cache_key] = result
    return result

# def _extract_residue_num(residue_number: Any) -> Optional[int]:
#     """Extract integer residue number from PDB-style field (e.g. 111, '111A')."""
#     if pd.isna(residue_number):
#         return None
#     s = str(residue_number).strip()
#     m = re.search(r"(\\d+)", s)
#     return int(m.group(1)) if m else None
