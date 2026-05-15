import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import logging

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.distance import pdist
from scipy import optimize
from sklearn.cluster import DBSCAN

from developability.structure_context import StructureContext, ResKey4
from utils.parsers import (
    Atom,
    ca_xyz_by_residue,
    parse_structure,
    residue_key_from_atom,
    SASAEntry,
)
from utils.chemistry import (
    ACCEPTOR_MAX_HBONDS,
    ACCEPTOR_METADATA,
    ACCEPTORS_ANY,
    ACCEPTORS_SPECIFIC,
    AROMATIC_RESIDUES,
    DONOR_EXCLUDED,
    DONOR_INFO,
    DONOR_MAX_HBONDS,
    DONOR_METADATA,
    DONORS_ANY,
    EXPOSURE_REL_ASA_THRESHOLD,
    get_standard_residue_pka,
    HYDROPHOBIC_RESIDUES,
    INTER_CHAIN_INTERFACE_CUTOFF,
    is_backbone_atom,
    KD_MAX,
    KD_MIN,
    KYTE_DOOLITTLE,
    MAX_SALT_BRIDGE_DISTANCE,
    NEGATIVE_ATOMS,
    NEGATIVE_CHARGED_RESIDUES,
    normalize_hydropathy,
    POLAR_RESIDUES,
    POSITIVE_ATOMS,
    POSITIVE_CHARGED_RESIDUES,
)

logger = logging.getLogger(__name__)


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
) -> Tuple[Tuple[ResKey4, float], ...]:
    items: List[Tuple[ResKey4, float]] = [
        (res_key, entry.total_side_rel) for res_key, entry in sasa_data.items()
    ]
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

_CONVEX_HULL_CA_VOLUME_CACHE: Dict[str, float] = {}

# Cache keyed by id(atoms); safe because _get_atoms_for_path returns the same
# cached list object for a given path, so id is stable within a pipeline run.
_HEAVY_ATOM_TREE_CACHE: Dict[
    int,
    Tuple["cKDTree", np.ndarray, List[ResKey4], Dict[ResKey4, List[int]]],
] = {}

_RESIDUE_LOCAL_CURVATURE_CACHE: Dict[
    Tuple[int, ResKey4, float, int, bool],
    Optional[float],
] = {}

_CDR_VICINITY_RESIDUE_KEYS_CACHE: Dict[
    Tuple[Tuple[ResKey4, ...], float, Tuple[ResKey4, ...]],
    frozenset,
] = {}

# One-time warning cache for insertion-code fallback in SASA lookup.
_SASA_INSERTION_FALLBACK_WARNED: Set[Tuple[ResKey4, ResKey4]] = set()

_CHARGE_CACHE: Dict[Tuple[str, Optional[float], float], float] = {}


def _residue_fractional_charge_at_pH(
    residue_name: str,
    pka_value: Optional[float],
    pH: float,
) -> float:
    """Fractional charge at pH via Henderson-Hasselbalch. Returns 0.0 for non-titratable residues."""
    res_name = (residue_name or "").strip().upper()
    effective_pka = pka_value if pka_value is not None else get_standard_residue_pka(res_name)
    cache_key = (res_name, effective_pka, float(pH))
    cached = _CHARGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if effective_pka is None:
        q = 0.0
    elif res_name in NEGATIVE_CHARGED_RESIDUES:
        q = -1.0 / (1.0 + np.power(10.0, effective_pka - pH))
    elif res_name in POSITIVE_CHARGED_RESIDUES:
        q = 1.0 / (1.0 + np.power(10.0, pH - effective_pka))
    else:
        q = 0.0
    _CHARGE_CACHE[cache_key] = q
    return q


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
        q = _residue_fractional_charge_at_pH(atom.residue_name, pka_data.get(res_key), pH)
        if q > 0.0:
            positive.add(res_key)
        elif q < 0.0:
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
    Classify residue into charge/type group for clustering/Ripley.
    Returns "negative" | "positive" | "aromatic" | "hydrophobic" | "polar", or None.
    Titratable residues (ASP, GLU, LYS, ARG, HIS, TYR, CYS) are classified by the
    sign of their fractional charge at pH; if q==0 they fall through to aromatic/
    hydrophobic/polar below. Aromatic is checked before hydrophobic/polar.
    """
    q = _residue_fractional_charge_at_pH(res_name, pka_val, pH)
    if q < 0.0:
        return "negative"
    if q > 0.0:
        return "positive"
    if res_name in AROMATIC_RESIDUES:
        return "aromatic"
    if res_name in HYDROPHOBIC_RESIDUES:
        return "hydrophobic"
    if res_name in POLAR_RESIDUES:
        return "polar"
    return None


def _residue_category_group_surface_pcf(
    res_name: str,
    pka_val: Optional[float],
    pH: float,
) -> Optional[str]:
    """
    Like :func:`_residue_category_group` but without a separate **aromatic** bucket
    (for exposed pair-correlation clustering). Uncharged Phe/Tyr/Trp follow the same
    hydrophobic / polar ordering as other neutral side chains (hydrophobic first).
    """
    q = _residue_fractional_charge_at_pH(res_name, pka_val, pH)
    if q < 0.0:
        return "negative"
    if q > 0.0:
        return "positive"
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

    exposure: Dict[ResKey4, bool] = {
        res_key: entry.total_side_rel > sasa_cutoff
        for res_key, entry in sasa_data.items()
    }

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


def get_heavy_atom_tree(
    atoms: List[Atom],
) -> Tuple["cKDTree", np.ndarray, List[ResKey4], Dict[ResKey4, List[int]]]:
    """Build (and cache) a KDTree over all heavy atoms of the structure.

    Cache key is ``id(atoms)``.  This is safe as long as callers obtain their
    atom list from ``_get_atoms_for_path``, which caches the list object —
    meaning the same structure always produces the same Python list object and
    therefore the same ``id``.

    Returns
    -------
    tree : cKDTree
        Spatial index over heavy-atom coordinates.
    coords : ndarray, shape (n, 3)
        Heavy-atom coordinates in the same order as the tree.
    atom_res_keys : list of ResKey4
        ``atom_res_keys[i]`` is the ResKey4 of the i-th heavy atom in the tree.
    heavy_atoms_by_res : dict
        Maps each ResKey4 to the list of atom indices (into ``coords``/``tree``)
        that belong to that residue.
    """
    cache_key = id(atoms)
    cached = _HEAVY_ATOM_TREE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    coord_list: List[Tuple[float, float, float]] = []
    key_list: List[ResKey4] = []
    for a in atoms:
        if is_hydrogen_atom(a):
            continue
        coord_list.append((a.x, a.y, a.z))
        key_list.append(residue_key_from_atom(a))

    coords = np.array(coord_list, dtype=np.float64) if coord_list else np.zeros((0, 3), dtype=np.float64)
    tree = cKDTree(coords) if len(coords) > 0 else cKDTree(np.zeros((1, 3), dtype=np.float64))

    heavy_atoms_by_res: Dict[ResKey4, List[int]] = {}
    for i, rk in enumerate(key_list):
        heavy_atoms_by_res.setdefault(rk, []).append(i)

    result = (tree, coords, key_list, heavy_atoms_by_res)
    _HEAVY_ATOM_TREE_CACHE[cache_key] = result
    return result


def compute_cdr_vicinity_residue_keys(
    atoms: List[Atom],
    cdr_keys: Set[ResKey4],
    radius: float,
) -> Set[ResKey4]:
    """
    Residues in the CDR “vicinity”: any residue that has at least one heavy atom
    within ``radius`` Å (minimum heavy–heavy distance) of some heavy atom of a CDR
    residue. All residues in ``cdr_keys`` are included.

    Heavy atoms match :func:`get_heavy_atom_tree` / :func:`is_hydrogen_atom`.
    Cached per structure (order-invariant residue keying), radius, and CDR set.
    """
    if not cdr_keys:
        return set()

    cdr_key_tuple = tuple(sorted(cdr_keys, key=lambda k: (k[2], k[1], k[3], k[0])))
    cache_key = (_residue_keys_cache_key(atoms), float(radius), cdr_key_tuple)
    cached = _CDR_VICINITY_RESIDUE_KEYS_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)

    _tree, heavy_coords, _key_list, heavy_atoms_by_res = get_heavy_atom_tree(atoms)
    seed_indices: List[int] = []
    for k in cdr_keys:
        seed_indices.extend(heavy_atoms_by_res.get(k, []))

    out: Set[ResKey4] = set(cdr_keys)
    if not seed_indices:
        _CDR_VICINITY_RESIDUE_KEYS_CACHE[cache_key] = frozenset(out)
        return set(out)

    seed_coords = heavy_coords[np.asarray(seed_indices, dtype=np.int64)]
    seed_tree = cKDTree(seed_coords)
    r = float(radius)

    for res_key, atom_idxs in heavy_atoms_by_res.items():
        if res_key in out:
            continue
        for ai in atom_idxs:
            if seed_tree.query_ball_point(heavy_coords[ai], r=r):
                out.add(res_key)
                break

    _CDR_VICINITY_RESIDUE_KEYS_CACHE[cache_key] = frozenset(out)
    return set(out)


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
    *,
    atom_lookup: Optional[Dict[Tuple[str, int, str, str], Atom]] = None,
    seq_index: Optional[Dict[Tuple[str, int, str, str], int]] = None,
    chain_index_to_res: Optional[Dict[str, Dict[int, Tuple[str, int, str, str]]]] = None,
    residue_cache_key: Optional[Tuple[ResKey4, ...]] = None,
) -> Optional[Atom]:
    """
    the base atom is the atom covalently bonded to the donor:
    - backbone N → previous residue C (residue_number - 1)
    - side-chain donors → bonded heavy atom in same residue

    returns None if not found
    """
    cache_key = residue_cache_key if residue_cache_key is not None else _residue_keys_cache_key(atoms)
    if atom_lookup is None:
        if cache_key not in _ATOM_LOOKUP_CACHE:
            _ATOM_LOOKUP_CACHE[cache_key] = {
                (atom.chain, atom.residue_number, atom.insertion_code, atom.name): atom
                for atom in atoms
            }
        atom_lookup = _ATOM_LOOKUP_CACHE[cache_key]

    if donor.name == "N":
        if seq_index is None:
            seq_index = _get_residue_seq_index(atoms)
        donor_res_key = residue_key_from_atom(donor)
        if backbone_base_cache is not None:
            cached_base = backbone_base_cache.get(donor_res_key)
            if cached_base is not None:
                return cached_base
        donor_idx = seq_index.get(donor_res_key)
        if donor_idx is None or donor_idx == 0:
            # no previous residue in sequence (true N-terminus or unmapped), so
            # no backbone base atom for angle calculation
            return None

        donor_chain = donor.chain
        if backbone_base_cache is None:
            backbone_base_cache = {}
        if chain_index_to_res is None:
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


def _ca_coords_array(atoms: List[Atom]) -> np.ndarray:
    """
    One Cα per residue as (n, 3) float64 (see ``ca_xyz_by_residue``).
    Used for convex-hull volume helpers.
    """
    d = ca_xyz_by_residue(atoms)
    if not d:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(list(d.values()), dtype=np.float64)


def convex_hull_volume_from_points(pts: np.ndarray) -> float:
    """
    Convex-hull volume (Å³) of arbitrary 3D points (rows of ``pts``).

    Returns 0.0 when there are fewer than four points, the array is empty, or
    Qhull cannot build a 3D hull (degenerate / coplanar).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return 0.0


def convex_hull_ca_volume(atoms: List[Atom]) -> float:
    """
    Convex-hull volume (Å³) of per-residue Cα coordinates.

    Uses ``scipy.spatial.ConvexHull`` (Qhull): the smallest convex polyhedron
    containing those points. This overestimates true solvent-excluded volume
    but is fast and useful for normalization.

    Returns 0.0 when there are fewer than four points or when Qhull cannot build
    a proper 3D hull (e.g. coplanar or collinear point clouds).
    """
    return convex_hull_volume_from_points(_ca_coords_array(atoms))


def spherical_shell_volume(r: float, delta_r: float) -> float:
    """Volume (Å³) of the spherical shell ``[r, r + delta_r)`` (``r >= 0``)."""
    if delta_r <= 0.0 or r < 0.0:
        return 0.0
    r_hi = r + delta_r
    return (4.0 / 3.0) * math.pi * (r_hi**3 - r**3)


def _per_bin_delta_r(
    n_bins: int, delta_r: Union[float, Sequence[float]]
) -> List[float]:
    """Scalar broadcast or per-bin widths; length must match ``n_bins`` when a sequence."""
    if isinstance(delta_r, (int, float)):
        return [float(delta_r)] * n_bins
    seq = [float(x) for x in delta_r]
    if len(seq) != n_bins:
        raise ValueError(
            f"delta_r has length {len(seq)} but bin_starts has {n_bins} bins"
        )
    return seq


def _pair_correlation_g_values_for_bins(
    coords: np.ndarray,
    rho: float,
    bin_starts: Sequence[float],
    delta_r: Union[float, Sequence[float]],
) -> List[float]:
    """
    Pair-correlation ``g(r)`` values for multiple distance shells from one point set.

    This computes the condensed pairwise distance vector once and reuses it for
    all shells, while preserving the same counting semantics as
    :func:`pair_correlation_g_for_bin`: each unordered pair contributes to the
    neighbor count of both endpoints, so ``mean_n = 2 * n_pairs_in_shell / N``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    bin_list = [float(r) for r in bin_starts]
    n_bin = len(bin_list)
    zeros = [0.0] * n_bin
    if rho <= 0.0:
        return zeros
    if coords.ndim != 2 or coords.shape[1] != 3:
        return zeros
    n = int(coords.shape[0])
    if n < 2:
        return zeros

    drs = _per_bin_delta_r(n_bin, delta_r)
    pairwise_distances = pdist(coords, metric="euclidean")
    out: List[float] = []
    for i, r_lo in enumerate(bin_list):
        dr = drs[i]
        V = spherical_shell_volume(r_lo, dr)
        if V <= 0.0:
            out.append(0.0)
            continue
        n_pairs_in_shell = int(
            np.count_nonzero(
                (pairwise_distances >= r_lo)
                & (pairwise_distances < r_lo + dr)
            )
        )
        mean_n = (2.0 * float(n_pairs_in_shell)) / float(n)
        out.append((1.0 / rho) * (mean_n / V))
    return out


def pair_correlation_g_for_bin(
    coords: np.ndarray,
    rho: float,
    r_lo: float,
    delta_r: float,
) -> float:
    """
    Pair correlation value g(r) for a single distance shell ``[r_lo, r_lo + delta_r)``.

    ``coords`` are per-residue Cα positions (N, 3) for one category.
    Neighbors are **same-type only** (all rows of ``coords``). Uses
    ``g(r) = (1/ρ) · (1/N) · Σ_i n_i(r) / V(r)`` with ``n_i`` counting
    ``j ≠ i`` in the shell and ``ρ = N / V_reference`` supplied by the caller.
    """
    V = spherical_shell_volume(r_lo, delta_r)
    if V <= 0.0 or rho <= 0.0:
        return 0.0
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return 0.0
    n = coords.shape[0]
    if n < 2:
        return 0.0
    g_values = _pair_correlation_g_values_for_bins(
        coords, rho, (float(r_lo),), float(delta_r)
    )
    return float(g_values[0]) if g_values else 0.0


def pair_correlation_clustering_score(
    coords_same_type: np.ndarray,
    v_reference: float,
    bin_starts: Sequence[float] = (3.0, 4.0, 5.0),
    delta_r: Union[float, Sequence[float]] = 1.0,
) -> float:
    """
    Clustering score ``(1/K) Σ_k g(r_k)`` over distance bins.

    ``v_reference`` is ``V_total`` for ``ρ = N / V_reference`` (typically convex-hull
    volume of all Cα points in scope, e.g. all exposed residues).

    Returns ``0.0`` when ``N < 1``, ``v_reference <= 0``, or no bins.
    """
    coords_same_type = np.asarray(coords_same_type, dtype=np.float64)
    if coords_same_type.ndim != 2 or coords_same_type.shape[1] != 3:
        return 0.0
    n = int(coords_same_type.shape[0])
    if n < 1 or v_reference <= 0.0:
        return 0.0
    rho = n / v_reference
    gs = _pair_correlation_g_values_for_bins(
        coords_same_type, rho, bin_starts, delta_r
    )
    return float(np.mean(gs)) if gs else 0.0


def pair_correlation_clustering_score_random_surface_null(
    coords_observed: np.ndarray,
    allowed_coords: np.ndarray,
    v_reference: float,
    bin_starts: Sequence[float] = (3.0, 4.0, 5.0),
    delta_r: Union[float, Sequence[float]] = 1.0,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Mean pair-correlation score over bins, normalized by a Ripley-style null on
    the **discrete** surface support.

    Observed score is ``S_obs = mean_k g(r_k)`` (same as
    :func:`pair_correlation_clustering_score`). The null draws ``N`` Cα
    locations uniformly at random from ``allowed_coords`` (typically all exposed
    residues’ Cα coordinates in scope), **without replacement** when ``len(allowed) >= N``, otherwise **with
    replacement** — same rule as ``ripley_k_statistic`` in ``descriptors.py``.

    ``n_permutations`` is clamped to at least ``1`` (non‑positive values would
    yield an undefined null mean).

    Returns ``S_obs / mean(S_null)`` over permutations, or ``0.0`` when ``N < 2``,
    coordinates are not shape ``(N, 3)``, ``v_reference <= 0``, too few allowed
    sites, or the null mean is non‑positive.
    """
    coords_observed = np.asarray(coords_observed, dtype=np.float64)
    allowed_coords = np.asarray(allowed_coords, dtype=np.float64)
    n = int(coords_observed.shape[0])
    n_allow = int(allowed_coords.shape[0])
    if (
        coords_observed.ndim != 2
        or coords_observed.shape[1] != 3
        or n < 2
        or n_allow < 2
        or v_reference <= 0.0
        or allowed_coords.ndim != 2
        or allowed_coords.shape[1] != 3
    ):
        return 0.0
    n_perm = max(1, int(n_permutations))
    s_obs = pair_correlation_clustering_score(
        coords_observed, v_reference, bin_starts, delta_r
    )
    replace = n_allow < n
    gen = rng if rng is not None else np.random.default_rng(0)
    null_means: List[float] = []
    for _ in range(n_perm):
        idx = gen.choice(n_allow, size=n, replace=replace)
        sample = allowed_coords[idx]
        null_means.append(
            pair_correlation_clustering_score(sample, v_reference, bin_starts, delta_r)
        )
    s_e = float(np.mean(null_means)) if null_means else 0.0
    if s_e <= 0.0 or not math.isfinite(s_e) or not math.isfinite(s_obs):
        return 0.0
    return float(s_obs / s_e)


def pair_correlation_clustering_score_random_surface_null_by_bin(
    coords_observed: np.ndarray,
    allowed_coords: np.ndarray,
    v_reference: float,
    bin_starts: Sequence[float] = (3.0, 4.0, 5.0),
    delta_r: Union[float, Sequence[float]] = 1.0,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """
    Same normalization as :func:`pair_correlation_clustering_score_random_surface_null`,
    but **one ratio per distance shell** ``[r, r + Δr)``: ``g_obs(r) / mean_perm(g_null(r))``.

    Uses the **same** random Cα draws per permutation for every shell (one sample per
    permutation, then ``g(r)`` at each ``r``), matching the structure of the global
    null while avoiding averaging shells into a single score.
    """
    coords_observed = np.asarray(coords_observed, dtype=np.float64)
    allowed_coords = np.asarray(allowed_coords, dtype=np.float64)
    n = int(coords_observed.shape[0])
    n_allow = int(allowed_coords.shape[0])
    bin_list = [float(r) for r in bin_starts]
    n_bin = len(bin_list)
    zeros = [0.0] * n_bin
    if (
        coords_observed.ndim != 2
        or coords_observed.shape[1] != 3
        or n < 2
        or n_allow < 2
        or v_reference <= 0.0
        or allowed_coords.ndim != 2
        or allowed_coords.shape[1] != 3
        or n_bin == 0
    ):
        return zeros
    n_perm = max(1, int(n_permutations))
    rho = n / v_reference
    replace = n_allow < n
    gen = rng if rng is not None else np.random.default_rng(0)

    s_obs = _pair_correlation_g_values_for_bins(
        coords_observed, rho, bin_list, delta_r
    )
    null_sum = np.zeros(n_bin, dtype=np.float64)
    for _ in range(n_perm):
        idx = gen.choice(n_allow, size=n, replace=replace)
        sample = allowed_coords[idx]
        sample_g = _pair_correlation_g_values_for_bins(
            sample, rho, bin_list, delta_r
        )
        null_sum += np.asarray(sample_g, dtype=np.float64)
    null_mean = null_sum / float(n_perm)
    out: List[float] = []
    for bi in range(n_bin):
        se = float(null_mean[bi])
        so = float(s_obs[bi])
        if se <= 0.0 or not math.isfinite(se) or not math.isfinite(so):
            out.append(0.0)
        else:
            out.append(float(so / se))
    return out


def convex_hull_ca_volume_for_pdb(pdb_path: str) -> float:
    """Cached convex-hull volume over Cα coordinates (see ``convex_hull_ca_volume``)."""
    abs_path = os.path.abspath(pdb_path)
    cached = _CONVEX_HULL_CA_VOLUME_CACHE.get(abs_path)
    if cached is not None:
        return cached
    vol = convex_hull_ca_volume(_get_atoms_for_path(pdb_path))
    _CONVEX_HULL_CA_VOLUME_CACHE[abs_path] = vol
    return vol


def _local_pca_curvature_ratio(neighbor_coords: np.ndarray) -> float:
    """
    Fraction of total variance along the smallest PCA axis of a 3D point cloud.

    With eigenvalues ``λ₁ ≥ λ₂ ≥ λ₃`` of the (3×3) covariance of centered points,
    returns ``λ₃ / (λ₁ + λ₂ + λ₃)`` — near 0 for a flat patch, larger when the
    cloud has substantial thickness along the normal direction.

    ``neighbor_coords`` must be shape ``(N, 3)``. Uses population covariance
    ``(1/N) XᵀX`` after centering. Non-finite coordinates should be filtered
    by the caller.
    """
    pts = np.asarray(neighbor_coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return 0.0
    n = int(pts.shape[0])
    if n < 1:
        return 0.0
    mean = np.mean(pts, axis=0)
    xc = pts - mean
    cov = (xc.T @ xc) / float(n)
    vals = np.linalg.eigh(cov)[0]
    vals = np.clip(vals, 0.0, None)
    total = float(np.sum(vals))
    if total <= 1e-18:
        return 0.0
    # eigh returns ascending order → smallest eigenvalue is index 0 (λ₃).
    return float(vals[0] / total)


def residue_mean_local_curvature(
    atoms: List[Atom],
    residue_key: ResKey4,
    *,
    neighborhood_radius: float = 8.0,
    min_neighbors: int = 4,
    exclude_hydrogens: bool = True,
) -> Optional[float]:
    """
    Mean local surface “curvature” over heavy atoms of one residue.

    For each qualifying atom in ``residue_key``, take all structure atoms within
    ``neighborhood_radius`` Å (optionally excluding hydrogens), build the 3×3
    covariance of their positions, and compute ``λ₃ / (λ₁ + λ₂ + λ₃)`` from the
    eigenvalues (smallest over sum). Average these ratios over all atoms that
    have at least ``min_neighbors`` points in their neighborhood.

    Returns ``None`` if the residue is missing, has no heavy atoms (when
    excluding H), or no atom yields a neighborhood large enough to score.

    Parameters
    ----------
    atoms
        Full atom list for the structure (same frame as the residue key).
    residue_key
        ``(residue_name, residue_number, chain, insertion_code)``.
    neighborhood_radius
        Ball radius (Å) around each atom for the local point cloud.
    min_neighbors
        Minimum number of atoms required in the ball (including the center atom
        if it appears in ``atoms``).
    exclude_hydrogens
        If True, hydrogens are omitted from the KD-tree and from target atoms,
        and this function reuses the shared cached heavy-atom KD-tree from
        :func:`get_heavy_atom_tree`. If False, the full-atom environment is
        rebuilt locally to preserve the original semantics exactly; that path is
        intentionally less optimized.
    """
    if neighborhood_radius <= 0.0 or min_neighbors < 1:
        return None
    if not atoms:
        return None
    cache_key = (
        id(atoms),
        residue_key,
        float(neighborhood_radius),
        int(min_neighbors),
        bool(exclude_hydrogens),
    )
    if cache_key in _RESIDUE_LOCAL_CURVATURE_CACHE:
        return _RESIDUE_LOCAL_CURVATURE_CACHE[cache_key]

    r = float(neighborhood_radius)

    if exclude_hydrogens:
        tree, coords, _atom_res_keys, heavy_atoms_by_res = get_heavy_atom_tree(atoms)
        res_atom_indices = heavy_atoms_by_res.get(residue_key, [])
        if not res_atom_indices or coords.shape[0] < min_neighbors:
            _RESIDUE_LOCAL_CURVATURE_CACHE[cache_key] = None
            return None

        ratios: List[float] = []
        for atom_idx in res_atom_indices:
            idx = tree.query_ball_point(coords[atom_idx], r)
            if len(idx) < min_neighbors:
                continue
            ratios.append(_local_pca_curvature_ratio(coords[idx]))

        result = float(np.mean(ratios)) if ratios else None
        _RESIDUE_LOCAL_CURVATURE_CACHE[cache_key] = result
        return result

    res_atoms: List[Atom] = []
    for a in atoms:
        if residue_key_from_atom(a) == residue_key:
            res_atoms.append(a)
    if not res_atoms:
        _RESIDUE_LOCAL_CURVATURE_CACHE[cache_key] = None
        return None

    if len(atoms) < min_neighbors:
        _RESIDUE_LOCAL_CURVATURE_CACHE[cache_key] = None
        return None

    coords = np.array(
        [[float(a.x), float(a.y), float(a.z)] for a in atoms],
        dtype=np.float64,
    )
    tree = cKDTree(coords)

    ratios = []
    for a in res_atoms:
        idx = tree.query_ball_point((float(a.x), float(a.y), float(a.z)), r)
        if len(idx) < min_neighbors:
            continue
        ratios.append(_local_pca_curvature_ratio(coords[idx]))

    result = float(np.mean(ratios)) if ratios else None
    _RESIDUE_LOCAL_CURVATURE_CACHE[cache_key] = result
    return result


def sum_residue_mean_local_curvature(
    atoms: List[Atom],
    residue_keys: Iterable[ResKey4],
    *,
    neighborhood_radius: float = 8.0,
    min_neighbors: int = 4,
    exclude_hydrogens: bool = True,
) -> float:
    """
    Sum of :func:`residue_mean_local_curvature` over many residues.

    Terms where the per-residue mean is undefined (``None`` or non-finite) are
    omitted from the sum.
    """
    total = 0.0
    for key in residue_keys:
        v = residue_mean_local_curvature(
            atoms,
            key,
            neighborhood_radius=neighborhood_radius,
            min_neighbors=min_neighbors,
            exclude_hydrogens=exclude_hydrogens,
        )
        if v is not None and math.isfinite(v):
            total += float(v)
    return total


def mean_residue_curvature_over_residues(
    atoms: List[Atom],
    residue_keys: Iterable[ResKey4],
    *,
    neighborhood_radius: float = 8.0,
    min_neighbors: int = 4,
    exclude_hydrogens: bool = True,
) -> float:
    """
    Mean per-residue local curvature over ``residue_keys``.

    For each key, :func:`residue_mean_local_curvature` is evaluated; undefined
    (``None`` or non-finite) values count as **0** in the sum. The average is
    divided by **len(residue_keys)** (not by the count of defined values).
    Returns ``0.0`` when ``residue_keys`` is empty.
    """
    keys = list(residue_keys)
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        v = residue_mean_local_curvature(
            atoms,
            key,
            neighborhood_radius=neighborhood_radius,
            min_neighbors=min_neighbors,
            exclude_hydrogens=exclude_hydrogens,
        )
        if v is not None and math.isfinite(v):
            total += float(v)
    return total / float(len(keys))


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
    return entry.total_side_rel

def residue_side_sasa_abs(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
) -> float:
    entry = sasa_data.get(residue_key)
    if entry is None:
        return 0.0
    return entry.total_side_abs


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
    return entry.main_chain_abs


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

