import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import logging

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.distance import pdist

from developability.structure_context import StructureContext, ResKey4
from utils.parsers import (
    Atom,
    parse_structure,
    residue_key_from_atom,
    SASAEntry,
)
from utils.chemistry import (
    ACCEPTOR_MAX_HBONDS,
    ACCEPTORS_ANY,
    ACCEPTORS_SPECIFIC,
    AROMATIC_RESIDUES,
    DONOR_EXCLUDED,
    DONOR_MAX_HBONDS,
    DONOR_METADATA,
    DONORS_ANY,
    EXPOSURE_REL_ASA_THRESHOLD,
    get_standard_residue_pka,
    NONPOLAR_RESIDUES,
    BACKBONE_ATOMS,
    KYTE_DOOLITTLE,
    MAX_SALT_BRIDGE_DISTANCE,
    NEGATIVE_ATOMS,
    NEGATIVE_CHARGED_RESIDUES,
    POSITIVE_ATOMS,
    POSITIVE_CHARGED_RESIDUES,
)

logger = logging.getLogger(__name__)


def _residue_keys_cache_key(atoms: List["Atom"]) -> Tuple[ResKey4, ...]:

    keys = {residue_key_from_atom(a) for a in atoms}
    return tuple(sorted(keys, key=lambda k: (k[2], k[1], k[3], k[0])))


def is_hydrogen_atom(atom: Atom) -> bool:

    elem = (getattr(atom, "element", "") or "").strip().upper()
    if elem in {"H", "D"}:
        return True
    name = (getattr(atom, "name", "") or "").strip().upper()
    if not name:
        return False
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
    items.append((("CUTOFF", -1, "", ""), float(sasa_cutoff)))  # cache key includes cutoff
    return tuple(sorted(items))


_ATOMS_CACHE: Dict[Tuple[str, Optional[Tuple[str, ...]]], List[Atom]] = {}
CDR_RANGES_CA = [(27, 38), (56, 65), (105, 117)]

# IMGT CDR3: at res 112, insertion codes run C→B→A→(none) toward the C-terminus.
IMGT_REVERSE_INSERTION_RESNUMS: frozenset = frozenset({112})


def imgt_residue_sort_key(key: Tuple[str, int, str, str]) -> Tuple[int, str]:
    """Sort IMGT residues; res 112 uses reverse insertion-code order."""
    res_num, ins_code = key[1], key[3]
    if res_num in IMGT_REVERSE_INSERTION_RESNUMS:
        if not ins_code:
            return (res_num, "~")
        return (res_num, chr(ord("A") + ord("Z") - ord(ins_code.upper())))
    return (res_num, ins_code)

_CDR_RESIDUE_CACHE: Dict[int, str] = {}  
_RESIDUE_REGION_CACHE: Dict[
    Tuple[Tuple[str, int, str, str], ...],
    Dict[Tuple[str, int, str, str], str],
] = {}

_STRUCTURE_CONTEXT_CACHE: Dict[Tuple[str, Optional[str], Optional[Tuple[str, ...]]], "StructureContext"] = {}
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

_HEAVY_ATOM_TREE_CACHE: Dict[
    int,
    Tuple[int, "cKDTree", np.ndarray, List[ResKey4], Dict[ResKey4, List[int]]],
] = {}

_CDR_VICINITY_RESIDUE_KEYS_CACHE: Dict[
    Tuple[Tuple[ResKey4, ...], float, Tuple[ResKey4, ...]],
    frozenset,
] = {}

_SASA_INSERTION_FALLBACK_WARNED: Set[Tuple[ResKey4, ResKey4]] = set()

_CHARGE_CACHE: Dict[Tuple[str, Optional[float], float], float] = {}

def _residue_fractional_charge_at_pH(
    residue_name: str,
    pka_value: Optional[float],
    pH: float,
    use_fallback: bool = True,
) -> float:
    """Fractional charge at pH (Henderson–Hasselbalch). Non-titratable → 0.

    use_fallback=True: missing pKa → standard table value (no PropKa file).
    use_fallback=False: missing pKa → 0 (e.g. disulfide CYS must not pick up CYS pKa 8.37).
    """
    res_name = (residue_name or "").strip().upper()
    if pka_value is not None:
        effective_pka: Optional[float] = pka_value
    elif use_fallback:
        effective_pka = get_standard_residue_pka(res_name)
    else:
        effective_pka = None
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


def get_aromatic_residue_keys(
    pdb_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> Set[ResKey4]:
    return get_residue_keys_by_type(
        pdb_path, AROMATIC_RESIDUES, allowed_chains=allowed_chains
    )

MIN_ABS_CHARGE_THRESHOLD: float = 0.05
"""|q| must exceed this to count as charged (filters noise from TYR etc. at pH 7.5)."""


def _get_charged_residues_at_pH(
    atoms: List[Atom],
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    min_abs_charge: float = MIN_ABS_CHARGE_THRESHOLD,
) -> Tuple[Set[Tuple[str, int, str, str]], Set[Tuple[str, int, str, str]]]:

    # With PropKa data: missing entry → q=0 (disulfide CYS etc.). Without: use standard pKa.
    use_fallback = not bool(pka_data)
    positive: Set[Tuple[str, int, str, str]] = set()
    negative: Set[Tuple[str, int, str, str]] = set()
    seen: Set[Tuple[str, int, str, str]] = set()
    for atom in atoms:
        res_key = residue_key_from_atom(atom)
        if res_key in seen:
            continue
        seen.add(res_key)
        q = _residue_fractional_charge_at_pH(
            atom.residue_name, pka_data.get(res_key), pH, use_fallback=use_fallback
        )
        if q > min_abs_charge:
            positive.add(res_key)
        elif q < -min_abs_charge:
            negative.add(res_key)
    return positive, negative


def _aggregate_dssp_hbond_energy_to_raw(
    dssp_hbonds: Dict[Tuple[str, int, str, str], List[Tuple[int, float]]],
    dssp_seq_to_pdb: Dict[int, Tuple[str, int, str, str]],
    pdb_to_dssp_seq: Dict[Tuple[str, int, str, str], int],
    sasa_data: Dict[Tuple[str, int, str, str], SASAEntry],
) -> Dict[Tuple[str, int, str, str], float]:

    residue_sasa_cache: Dict[ResKey4, float] = {}
    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)

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
            target_weight = get_cached_sasa(target_residue_key)
            if target_weight == 0.0:
                continue
            energy_scale = abs(energy)
            if energy_scale == 0.0:
                continue

            residue_weights_raw[residue_key] += residue_weight * energy_scale
            residue_weights_raw[target_residue_key] += target_weight * energy_scale

    return dict(residue_weights_raw)


def _residue_category_group_surface_pcf(
    res_name: str,
    pka_val: Optional[float],
    pH: float,
    min_abs_charge: float = MIN_ABS_CHARGE_THRESHOLD,
    use_fallback_pka: bool = True,
) -> Optional[str]:
    """Surface patch class: negative | positive | hydrophobic | None.

    Uses NONPOLAR_RESIDUES (includes neutral Phe, Trp, Tyr) after charge assignment.
    """
    q = _residue_fractional_charge_at_pH(res_name, pka_val, pH, use_fallback=use_fallback_pka)
    if q < -min_abs_charge:
        return "negative"
    if q > min_abs_charge:
        return "positive"
    if res_name in NONPOLAR_RESIDUES:
        return "hydrophobic"
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

def get_residue_region(residue_number: int) -> str:
    """'CDR' or 'framework' from IMGT CA numbering (cached)."""
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
    """Unique residue keys in atom-list order."""
    seen: Set[ResKey4] = set()
    for atom in atoms:
        key = residue_key_from_atom(atom)
        if key in seen:
            continue
        seen.add(key)
        yield key


def get_residue_region_map(atoms: List[Atom]) -> Dict[Tuple[str, int, str, str], str]:
    """Residue key → 'CDR' or 'framework'."""
    cache_key = _residue_keys_cache_key(atoms)
    cached = _RESIDUE_REGION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    out = {key: get_residue_region(key[1]) for key in iter_unique_residues(atoms)}
    _RESIDUE_REGION_CACHE[cache_key] = out
    return out


def _get_atoms_for_path(
    pdb_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> List[Atom]:
    """Parsed atoms for pdb_path (cached by abs path and chain filter)."""
    abs_path = os.path.abspath(pdb_path)
    chains_key = tuple(sorted(allowed_chains)) if allowed_chains is not None else None
    cache_key = (abs_path, chains_key)
    cached = _ATOMS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    atoms = parse_structure(pdb_path, allowed_chains=allowed_chains)
    _ATOMS_CACHE[cache_key] = atoms
    return atoms


def get_heavy_atom_tree(
    atoms: List[Atom],
) -> Tuple["cKDTree", np.ndarray, List[ResKey4], Dict[ResKey4, List[int]]]:
    """KDTree over heavy atoms (cached by id(atoms)).

    Returns tree, coords, atom_res_keys[i], and heavy_atoms_by_res[ResKey4] → indices.
    """
    cache_key = id(atoms)
    cached = _HEAVY_ATOM_TREE_CACHE.get(cache_key)
    if cached is not None and cached[0] == id(atoms):
        return cached[1], cached[2], cached[3], cached[4]

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

    _HEAVY_ATOM_TREE_CACHE[cache_key] = (id(atoms), tree, coords, key_list, heavy_atoms_by_res)
    return tree, coords, key_list, heavy_atoms_by_res


def compute_cdr_vicinity_residue_keys(
    atoms: List[Atom],
    cdr_keys: Set[ResKey4],
    radius: float,
) -> Set[ResKey4]:
    """Residues within radius Å of any CDR heavy atom (CDR keys included). Cached."""
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
    """Per-chain 0-based residue index and reverse map (cached, order-stable)."""
    cache_key = _residue_keys_cache_key(atoms)
    if cache_key in _RES_SEQ_INDEX_CACHE:
        return _RES_SEQ_INDEX_CACHE[cache_key]

    # Sort residues per chain so indices don't depend on atom order in the PDB.
    chain_to_residue_set: Dict[str, Set[ResKey4]] = defaultdict(set)
    for atom in atoms:
        rk = residue_key_from_atom(atom)
        chain_to_residue_set[rk[2]].add(rk)

    seq_index: Dict[Tuple[str, int, str, str], int] = {}
    index_to_key: Dict[str, Dict[int, Tuple[str, int, str, str]]] = {}
    for chain, residue_set in chain_to_residue_set.items():
        residues = sorted(residue_set, key=lambda k: (*imgt_residue_sort_key(k), k[0]))
        chain_index_map: Dict[int, Tuple[str, int, str, str]] = {}
        for idx, res_key in enumerate(residues):
            seq_index[res_key] = idx
            chain_index_map[idx] = res_key
        index_to_key[chain] = chain_index_map

    _RES_SEQ_INDEX_CACHE[cache_key] = seq_index
    _RES_INDEX_TO_KEY_CACHE[cache_key] = index_to_key
    return seq_index


def is_donor(atom: Atom) -> bool:
    """True if atom can donate an H-bond (backbone/side-chain rules)."""
    key = (atom.name, atom.residue_name)
    return key not in DONOR_EXCLUDED and (
        atom.name in DONORS_ANY or key in DONOR_METADATA
    )


def is_acceptor(atom: Atom) -> bool:
    """True if atom can accept an H-bond."""
    return (atom.name, atom.residue_name) in ACCEPTORS_SPECIFIC or atom.name in ACCEPTORS_ANY


def donor_max_hbonds(atom: Atom) -> int:
    """Max H-bonds for this donor (backbone N uses generic rule)."""
    key = (atom.name, atom.residue_name)
    if key in DONOR_MAX_HBONDS:
        return DONOR_MAX_HBONDS[key]
    if atom.name == "N":
        return DONOR_MAX_HBONDS.get(("N", "ANY"), 1)
    return 1


def acceptor_max_hbonds(atom: Atom) -> int:
    """Max H-bonds for this acceptor (backbone O uses generic rule)."""
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
    """Atom bonded to the donor H (prev C for backbone N, else side-chain partner). None if missing."""
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
            return None  # N-terminus: no previous C for angle

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
    allowed_chains: Optional[Iterable[str]] = None,
) -> StructureContext:
    """StructureContext for (pdb_path, sasa_path, allowed_chains), cached."""
    chains_key = tuple(sorted(allowed_chains)) if allowed_chains is not None else None
    key = (os.path.abspath(pdb_path), sasa_path, chains_key)
    ctx = _STRUCTURE_CONTEXT_CACHE.get(key)
    if ctx is None:
        ctx = StructureContext(
            pdb_path, sasa_path=sasa_path, allowed_chains=allowed_chains
        )
        _STRUCTURE_CONTEXT_CACHE[key] = ctx
    return ctx


def _count_residues_in_pdb(
    pdb_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> int:
    """Unique residue count."""
    return len(_get_pdb_residue_keys(pdb_path, allowed_chains=allowed_chains))
def convex_hull_volume_from_points(pts: np.ndarray) -> float:
    """Convex hull volume (Å³); 0 if <4 points or degenerate."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return 0.0
def spherical_shell_volume(r: float, delta_r: float) -> float:
    """Volume of shell [r, r + delta_r) in Å³."""
    if delta_r <= 0.0 or r < 0.0:
        return 0.0
    r_hi = r + delta_r
    return (4.0 / 3.0) * math.pi * (r_hi**3 - r**3)


def _per_bin_delta_r(
    n_bins: int, delta_r: Union[float, Sequence[float]]
) -> List[float]:
    """Per-bin shell width; scalar delta_r broadcast or one value per bin."""
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
    """g(r) per distance shell; one pdist pass, both endpoints count toward shell occupancy."""
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
def pair_correlation_clustering_score_random_surface_null_by_bin(
    coords_observed: np.ndarray,
    allowed_coords: np.ndarray,
    v_reference: float,
    bin_starts: Sequence[float] = (3.0, 4.0, 5.0),
    delta_r: Union[float, Sequence[float]] = 1.0,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Observed g(r) / mean random-surface null, one ratio per shell (same perm draw each shell)."""
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
    gen = rng if rng is not None else np.random.default_rng(0)
    if n > n_allow:
        return zeros

    s_obs = _pair_correlation_g_values_for_bins(
        coords_observed, rho, bin_list, delta_r
    )
    null_sum = np.zeros(n_bin, dtype=np.float64)
    for _ in range(n_perm):
        idx = gen.choice(n_allow, size=n, replace=False)
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
    """total_side_rel × 100 (FreeSASA; can slightly exceed 100 for very exposed sites)."""
    entry = sasa_data.get(residue_key)
    return _rel_sasa_scaled(entry, "total_side_rel")


def residue_main_sasa(
    residue_key: ResKey4,
    sasa_data: Dict[ResKey4, SASAEntry],
) -> float:
    entry = sasa_data.get(residue_key)
    return _rel_sasa_scaled(entry, "main_chain_rel")
def atom_sasa_weight(atom: Atom, sasa_data: Dict[ResKey4, SASAEntry]) -> float:
    key = residue_key_from_atom(atom)
    entry = sasa_data.get(key)
    if atom.name in BACKBONE_ATOMS:
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
    entry = sasa_output_data.get(key)
    if entry is not None:
        return entry
    fallback_key = (key[0], key[1], key[2], "")
    fallback = sasa_output_data.get(fallback_key)
    if fallback is not None:
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

_PDB_RESIDUE_KEYS_CACHE: Dict[Tuple[str, Optional[Tuple[str, ...]]], Set[ResKey4]] = {}
_RESIDUE_TYPE_CACHE: Dict[Tuple[str, Optional[Tuple[str, ...]], frozenset], Set[ResKey4]] = {}


def _get_pdb_residue_keys(
    pdb_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> Set[ResKey4]:
    abs_path = os.path.abspath(pdb_path)
    chains_key = tuple(sorted(allowed_chains)) if allowed_chains is not None else None
    cache_key = (abs_path, chains_key)
    all_residue_keys = _PDB_RESIDUE_KEYS_CACHE.get(cache_key)
    if all_residue_keys is None:
        atoms = _get_atoms_for_path(pdb_path, allowed_chains=allowed_chains)
        if not atoms:
            _PDB_RESIDUE_KEYS_CACHE[cache_key] = set()
            return set()
        all_residue_keys = set(iter_unique_residues(atoms))
        _PDB_RESIDUE_KEYS_CACHE[cache_key] = all_residue_keys
    return all_residue_keys


def get_residue_keys_by_type(
    pdb_path: str,
    residue_types: Iterable[str],
    allowed_chains: Optional[Iterable[str]] = None,
) -> Set[ResKey4]:
    abs_path = os.path.abspath(pdb_path)
    chains_key = tuple(sorted(allowed_chains)) if allowed_chains is not None else None
    type_set = frozenset(residue_types)
    cache_key = (abs_path, chains_key, type_set)
    cached = _RESIDUE_TYPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    all_residue_keys = _get_pdb_residue_keys(pdb_path, allowed_chains=allowed_chains)
    result = {k for k in all_residue_keys if k[0] in type_set}
    _RESIDUE_TYPE_CACHE[cache_key] = result
    return result

