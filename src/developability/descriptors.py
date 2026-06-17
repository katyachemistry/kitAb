from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
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
    imgt_residue_sort_key,
    _ATOM_LOOKUP_CACHE,
    _RES_INDEX_TO_KEY_CACHE,
    iter_unique_residues,
    _get_atoms_for_path,
    _residue_keys_cache_key,
    _residue_fractional_charge_at_pH,
    is_hydrogen_atom,
    atom_sasa_weight,
    get_residue_sasa_weight,
    _get_structure_context,
    _count_residues_in_pdb,
    _get_residue_seq_index,
    _get_charged_residues_at_pH,
    _residue_category_group_surface_pcf,
    _lookup_pka_value,
    get_exposed_residues,
    _sasa_lookup,
    is_donor,
    is_acceptor,
    get_donor_base_atom,
    donor_max_hbonds,
    acceptor_max_hbonds,
    _aggregate_dssp_hbond_energy_to_raw,
    convex_hull_volume_from_points,
    pair_correlation_clustering_score_random_surface_null_by_bin,
    get_heavy_atom_tree,
)

from utils.parsers import (
    ca_xyz_by_residue,
    parse_sasa,
    Atom,
    SASAEntry,
    residue_key_from_atom,
)
from utils.chemistry import (
    AROMATIC_RESIDUES,
    CTERM_PKA,  # kept for commented standard termini pKa fallback
    DBSCAN_EPS_MIN_SAMPLES_BY_CATEGORY_DEFAULT,
    EXPOSURE_REL_ASA_THRESHOLD,
    get_ff19sb_atom_charge_with_source,
    BACKBONE_ATOMS,
    residue_hydrophobicity,
    MAX_HBOND_DISTANCE,
    MAX_SALT_BRIDGE_DISTANCE,
    MIN_BACKBONE_SEPARATION,
    MIN_HBOND_ANGLE,
    NEGATIVE_ATOMS,
    NEGATIVE_CHARGED_RESIDUES,
    SALT_BRIDGE_NEGATIVE_ATOMS,
    NTERM_PKA,  # kept for commented standard termini pKa fallback
    PCF_CLUSTER_BIN_STARTS_DEFAULT,
    PCF_CLUSTER_BIN_WIDTHS_DEFAULT,
    PCF_CLUSTER_N_PERMUTATIONS_DEFAULT,
    POSITIVE_ATOMS,
    POSITIVE_CHARGED_RESIDUES,
    SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    THREE_TO_ONE,
)

logger = logging.getLogger(__name__)

_HBOND_PAIRS_CACHE: Dict[
    Tuple[str, float, float, int, Optional[Tuple[str, ...]]],
    List[Tuple[Atom, Atom]],
] = {}

_RESIDUE_LOCAL_PLANARITY_CACHE: Dict[
    Tuple[int, ResKey4, float, int, bool],
    Optional[float],
] = {}
_PLANARITY_ATOMS_REFS: Dict[int, int] = {}

_IONIZABLE_ATOM_NAMES_BY_RES: Dict[str, Set[str]] = {}
for _atom_name, _res_name in POSITIVE_ATOMS | NEGATIVE_ATOMS:
    _IONIZABLE_ATOM_NAMES_BY_RES.setdefault(_res_name, set()).add(_atom_name)

# Charge

def net_charge_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float
) -> Optional[float]:

    if not pka_data:
        return None
    net = 0.0
    nterm_pka_by_chain: Dict[str, float] = {}
    cterm_pka_by_chain: Dict[str, float] = {}
    chains: set = set()

    for key, pka in pka_data.items():
        res_name, _, chain, _ = key
        chains.add(chain)
        if res_name == "N+":
            nterm_pka_by_chain[chain] = pka
        elif res_name == "C-":
            cterm_pka_by_chain[chain] = pka
        else:
            net += _residue_fractional_charge_at_pH(res_name, pka, pH)

    for chain in chains:
        if chain not in nterm_pka_by_chain:
            raise ValueError(
                f"Missing PropKa N+ entry for chain {chain!r}; "
                "termini pKa values must come from the PropKa file."
            )
        if chain not in cterm_pka_by_chain:
            raise ValueError(
                f"Missing PropKa C- entry for chain {chain!r}; "
                "termini pKa values must come from the PropKa file."
            )
        nterm_pka = nterm_pka_by_chain[chain]
        cterm_pka = cterm_pka_by_chain[chain]
        # Standard termini pKa fallback (disabled — PropKa required):
        # nterm_pka = nterm_pka_by_chain.get(chain, NTERM_PKA)
        # cterm_pka = cterm_pka_by_chain.get(chain, CTERM_PKA)
        # N-terminus
        net += 1.0 / (1.0 + np.power(10.0, pH - nterm_pka))
        # C-terminus
        net -= 1.0 / (1.0 + np.power(10.0, cterm_pka - pH))

    return float(net)


def asymmetry_score_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float],
    heavy_chain: str,
    light_chain: str,
    pH: float = 7.5,
) -> Optional[float]:
    """Heavy × light net charge at pH (from PropKa)."""
    if not pka_data:
        return None
    pka_heavy = {k: v for k, v in pka_data.items() if k[2] == heavy_chain}
    pka_light = {k: v for k, v in pka_data.items() if k[2] == light_chain}
    heavy_charge = net_charge_from_pka(pka_heavy, pH)
    light_charge = net_charge_from_pka(pka_light, pH)
    if heavy_charge is None or light_charge is None:
        return None
    return float(heavy_charge * light_charge)


def pi_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float]
) -> Optional[float]:
    if not pka_data:
        return None

    try:
        q0 = net_charge_from_pka(pka_data, 0.0) or 0.0
        q14 = net_charge_from_pka(pka_data, 14.0) or 0.0
        if q0 * q14 > 0:
            return None
        return float(optimize.brentq(
            lambda pH: net_charge_from_pka(pka_data, pH) or 0.0,
            0.0, 14.0,
        ))
    except Exception:
        return None

def compute_dipole_moment_magnitude(
    pdb_atoms: List[Atom],
    *,
    debug_charge_stats: Optional[Dict[str, int]] = None,
) -> Optional[float]:
    """‖μ‖ in e·Å from ff19SB charges; origin at atom centroid. No pH/pKa. Missing charges → 0."""

    if not pdb_atoms:
        return 0.0

    residue_atom_names: Dict[ResKey4, Set[str]] = {}
    for a in pdb_atoms:
        key4 = residue_key_from_atom(a)
        residue_atom_names.setdefault(key4, set()).add((a.name or "").strip().upper())

    ref_center = np.mean(
        np.array([(a.x, a.y, a.z) for a in pdb_atoms], dtype=np.float64),
        axis=0,
    )
    rx, ry, rz = float(ref_center[0]), float(ref_center[1]), float(ref_center[2])

    lookup_counts: Dict[str, int] = {
        "direct": 0,
        "fallback_oxt_to_o": 0,
        "fallback_h123_to_h": 0,
        "missing": 0,
    }

    mux, muy, muz = 0.0, 0.0, 0.0
    for atom in pdb_atoms:
        res_name = (atom.residue_name or "").strip().upper()
        q, source = get_ff19sb_atom_charge_with_source(
            res_name,
            atom.name or "",
            residue_atom_names=residue_atom_names.get(residue_key_from_atom(atom)),
        )
        lookup_counts[source] = lookup_counts.get(source, 0) + 1
        if q == 0.0:
            continue
        mux += q * (float(atom.x) - rx)
        muy += q * (float(atom.y) - ry)
        muz += q * (float(atom.z) - rz)

    if debug_charge_stats is not None:
        debug_charge_stats.clear()
        debug_charge_stats.update(lookup_counts)
        debug_charge_stats["n_atoms"] = len(pdb_atoms)
        debug_charge_stats["n_nonzero_charge"] = (
            lookup_counts["direct"] + lookup_counts["fallback_oxt_to_o"] + lookup_counts["fallback_h123_to_h"]
        )

    return float(math.sqrt(mux * mux + muy * muy + muz * muz))


def residue_neighbor_score(
    pdb_atoms: List[Atom],
    sasa_data: Dict[ResKey4, SASAEntry],
    d_cutoff: float,
    *,
    pka_data: Optional[Dict[ResKey4, float]] = None,
    pH: float = 7.5,
    source: str = "charge",
    sasa_weight_sources: bool = True,
    sasa_cutoff: float = EXPOSURE_REL_ASA_THRESHOLD,
    ionizable_only: bool = False,
    center_weight: str = "sasa",
    weight_center_by_sasa: bool = False,
    reduce: str = "neg_abs",
) -> Optional[float]:
    """SASA-weighted neighbor sum within d_cutoff (min heavy-atom distance).

    Each center i collects weighted contributions from residues j in range, applies
    center_weight (optionally × sasa(i)), then aggregates via reduce.

    source: charge | hydrophobicity | charge_pos | charge_neg | sasa_weighted_count
    sasa_weight_sources: weight j by sasa(j), or only count exposed j when False
    center_weight: sasa | hydrophobicity | charge | charge_pos | charge_neg | none
    reduce: neg_abs (|sum of negatives|, SCM anionic) | pos_abs | sum (SAP)
    """
    if not pdb_atoms or not sasa_data:
        return None

    needs_charge = source in ("charge", "charge_pos", "charge_neg") or center_weight in (
        "charge", "charge_pos", "charge_neg"
    )

    ionizable_keys: Optional[Set[ResKey4]] = None
    if ionizable_only:
        ionizable_keys = set()
        for a in pdb_atoms:
            allowed = _IONIZABLE_ATOM_NAMES_BY_RES.get((a.residue_name or "").strip().upper())
            if allowed and (a.name or "").strip().upper() in allowed:
                ionizable_keys.add(residue_key_from_atom(a))
        if not ionizable_keys:
            return None
        res_keys: List[ResKey4] = list(ionizable_keys)
    else:
        res_keys = list(iter_unique_residues(pdb_atoms))

    if not res_keys:
        return None

    m = len(res_keys)
    res_key_to_idx: Dict[ResKey4, int] = {k: i for i, k in enumerate(res_keys)}

    sasa_arr = np.zeros(m, dtype=np.float64)
    charge_arr = np.zeros(m, dtype=np.float64)

    for i, key4 in enumerate(res_keys):
        entry = sasa_data.get(key4)
        sasa_arr[i] = float(getattr(entry, "total_side_rel", 0.0) or 0.0)
        if needs_charge:
            pka = _lookup_pka_value(key4, pka_data, atoms=pdb_atoms) if pka_data else None
            charge_arr[i] = _residue_fractional_charge_at_pH(key4[0], pka, pH)

    exposed_arr: np.ndarray = sasa_arr > sasa_cutoff

    if source == "charge":
        raw_source = charge_arr
    elif source == "hydrophobicity":
        raw_source = np.array(
            [residue_hydrophobicity(k[0]) for k in res_keys], dtype=np.float64
        )
    elif source == "charge_pos":
        raw_source = np.maximum(0.0, charge_arr)
    elif source == "charge_neg":
        raw_source = np.minimum(0.0, charge_arr)
    elif source == "sasa_weighted_count":
        raw_source = np.ones(m, dtype=np.float64)
    else:
        raise ValueError(f"residue_neighbor_score: unknown source={source!r}")

    if sasa_weight_sources:
        source_contrib = raw_source * sasa_arr
        valid_source = np.ones(m, dtype=bool)
    else:
        source_contrib = raw_source
        valid_source = exposed_arr

    if center_weight == "sasa":
        center_arr = sasa_arr
    elif center_weight == "hydrophobicity":
        center_arr = np.array(
            [residue_hydrophobicity(k[0]) for k in res_keys], dtype=np.float64
        )
    elif center_weight == "charge":
        center_arr = charge_arr
    elif center_weight == "charge_pos":
        center_arr = np.maximum(0.0, charge_arr)
    elif center_weight == "charge_neg":
        center_arr = np.minimum(0.0, charge_arr)
    elif center_weight == "none":
        center_arr = np.ones(m, dtype=np.float64)
    else:
        raise ValueError(f"residue_neighbor_score: unknown center_weight={center_weight!r}")

    heavy_tree, heavy_coords, heavy_atom_res_keys, heavy_atoms_by_res = get_heavy_atom_tree(pdb_atoms)

    score_arr = np.zeros(m, dtype=np.float64)
    for ri, key_i in enumerate(res_keys):
        atom_indices_i = heavy_atoms_by_res.get(key_i, [])
        if not atom_indices_i:
            continue
        seen_rj: Set[int] = set()
        for nb_atom_idxs in heavy_tree.query_ball_point(heavy_coords[atom_indices_i], r=d_cutoff):
            for nb_atom_idx in nb_atom_idxs:
                rj = res_key_to_idx.get(heavy_atom_res_keys[nb_atom_idx])
                if rj is None or rj == ri or rj in seen_rj:
                    continue
                if not valid_source[rj]:
                    continue
                seen_rj.add(rj)
                score_arr[ri] += source_contrib[rj]

    if weight_center_by_sasa:
        center_arr = center_arr * sasa_arr
    result_arr = score_arr * center_arr

    if reduce == "neg_abs":
        neg_mask = result_arr < 0.0
        return float(np.abs(np.sum(result_arr[neg_mask])))
    elif reduce == "pos_abs":
        pos_mask = result_arr > 0.0
        return float(np.sum(result_arr[pos_mask]))
    elif reduce == "sum":
        return float(np.sum(result_arr))
    else:
        raise ValueError(f"residue_neighbor_score: unknown reduce={reduce!r}")


def scm_score_from_pka(
    pdb_path: str,
    sasa_path: str,
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    d_cutoff: float = 10.0,
    sasa_cutoff: float = EXPOSURE_REL_ASA_THRESHOLD,
    reduce: str = "neg_abs",
    allowed_chains: Optional[Iterable[str]] = None,
) -> Optional[float]:

    ctx = _get_structure_context(
        pdb_path, sasa_path=sasa_path, allowed_chains=allowed_chains
    )
    atoms = ctx.atoms
    if not atoms:
        return None
    sasa_data = ctx.sasa_residue
    if not sasa_data:
        return None
    try:
        return residue_neighbor_score(
            atoms,
            sasa_data,
            d_cutoff,
            pka_data=pka_data,
            pH=pH,
            source="charge",
            sasa_weight_sources=True,
            sasa_cutoff=sasa_cutoff,
            ionizable_only=True,
            center_weight="sasa",
            reduce=reduce,
        )
    except Exception as e:
        logger.warning("SCM score computation failed: %s", e, exc_info=True)
        return None


def scm_score_by_atoms(
    pdb_path: str,
    d_cutoff: float = 10.0,
    allowed_chains: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, float]]:
    """SCM from ff19SB charges in a d_cutoff shell around each atom."""
    atoms = _get_atoms_for_path(pdb_path, allowed_chains=allowed_chains)
    if not atoms:
        return None

    try:
        coords = np.array([(a.x, a.y, a.z) for a in atoms], dtype=np.float64)
        charges = np.array(
            [
                get_ff19sb_atom_charge_with_source(
                    (a.residue_name or "").strip().upper(),
                    a.name or "",
                )[0]
                for a in atoms
            ],
            dtype=np.float64,
        )

        if coords.shape[0] == 0:
            return None
        if coords.shape[0] == 1:
            return {"scm_neg_ff19sb": 0.0, "scm_pos_ff19sb": 0.0}

        tree = cKDTree(coords)
        neighbors_by_atom = tree.query_ball_point(coords, r=d_cutoff)
        charge_in_shell = np.fromiter(
            (charges[nb].sum() for nb in neighbors_by_atom),
            dtype=np.float64,
            count=len(neighbors_by_atom),
        )
        scmi = charge_in_shell - charges

        neg_mask = scmi < 0.0
        pos_mask = scmi > 0.0
        return {
            "scm_neg_ff19sb": float(np.sum(scmi[neg_mask])),
            "scm_pos_ff19sb": float(np.sum(scmi[pos_mask])),
        }
    except Exception as e:
        logger.warning("Atom-level SCM computation failed: %s", e, exc_info=True)
        return None

def compute_sap_shell_synergy_scores(
    pdb_atoms: List[Atom],
    sasa_data: Dict[ResKey4, SASAEntry],
    pka_data: Optional[Dict[ResKey4, float]],
    pH: float,
    d_cutoff: float = 10.0,
) -> Dict[str, float]:
    """SAP-style surface metrics: aromatic/histidine exposure and charge–aromatic synergy.

    Per residue i, sum SASA-weighted neighbor env (pos, neg, aro, his), then
    sap_aro_score, sap_his_score, sap_pos_aro_synergy, sap_aro_neg_contrast.
    """
    out: Dict[str, float] = {
        "sap_pos_aro_synergy": 0.0,
        "sap_aro_neg_contrast": 0.0,
    }
    if not pdb_atoms or not sasa_data:
        return out

    pka_lookup = pka_data if pka_data is not None else {}

    res_keys: List[ResKey4] = list(iter_unique_residues(pdb_atoms))
    if not res_keys:
        return out

    m = len(res_keys)
    res_key_to_idx: Dict[ResKey4, int] = {k: i for i, k in enumerate(res_keys)}

    sasa_arr = np.zeros(m, dtype=np.float64)
    charge_arr = np.zeros(m, dtype=np.float64)
    aro_arr = np.zeros(m, dtype=np.float64)
    his_arr = np.zeros(m, dtype=np.float64)

    for i, key4 in enumerate(res_keys):
        entry = sasa_data.get(key4)
        sasa_arr[i] = float(getattr(entry, "total_side_rel", 0.0) or 0.0)
        aa = (key4[0] or "").strip().upper()
        pka = _lookup_pka_value(key4, pka_lookup, atoms=pdb_atoms) if pka_lookup else None
        charge_arr[i] = _residue_fractional_charge_at_pH(aa, pka, pH)
        aro_arr[i] = 1.0 if aa in AROMATIC_RESIDUES else 0.0
        his_arr[i] = 1.0 if aa == "HIS" else 0.0

    pos_src = np.maximum(0.0, charge_arr) * sasa_arr
    neg_src = np.minimum(0.0, charge_arr) * sasa_arr
    aro_src = aro_arr * sasa_arr
    his_src = his_arr * sasa_arr

    pos_env = np.zeros(m, dtype=np.float64)
    neg_env = np.zeros(m, dtype=np.float64)
    aro_env = np.zeros(m, dtype=np.float64)
    his_env = np.zeros(m, dtype=np.float64)

    heavy_tree, heavy_coords, heavy_atom_res_keys, heavy_atoms_by_res = get_heavy_atom_tree(pdb_atoms)

    for ri, key_i in enumerate(res_keys):
        atom_indices_i = heavy_atoms_by_res.get(key_i, [])
        if not atom_indices_i:
            continue
        seen_rj: Set[int] = set()
        for nb_atom_idxs in heavy_tree.query_ball_point(heavy_coords[atom_indices_i], r=d_cutoff):
            for nb_atom_idx in nb_atom_idxs:
                rj = res_key_to_idx.get(heavy_atom_res_keys[nb_atom_idx])
                if rj is None or rj == ri or rj in seen_rj:
                    continue
                seen_rj.add(rj)
                pos_env[ri] += pos_src[rj]
                neg_env[ri] += neg_src[rj]
                aro_env[ri] += aro_src[rj]
                his_env[ri] += his_src[rj]

    w = sasa_arr
    out["sap_aro_score"] = float(np.sum(w * aro_env))
    out["sap_his_score"] = float(np.sum(w * his_env))
    out["sap_pos_aro_synergy"] = float(np.sum(w * pos_env * aro_env))
    out["sap_aro_neg_contrast"] = float(np.sum(w * aro_env * neg_env))
    return out


ResidueDensityRawDict = Dict[ResKey4, float]

def average_over_residues(
    *,
    weights_raw: Dict[ResKey4, float],
    residues_for_density: Optional[Iterable[ResKey4]] = None,
    residues_for_average: Optional[Union[str, Iterable[ResKey4]]] = None,
    denom_total_residues: int,
) -> float:
    """Weighted sum over residues_for_density, divided by a chosen denominator.

    residues_for_average: None → total PDB residue count; 'no' → raw sum only;
    else len(set(...)).
    """
    if not weights_raw:
        return 0.0

    if isinstance(residues_for_average, str):
        if residues_for_average == "no":
            denom = 1
        else:
            raise ValueError(
                f"residues_for_average string must be 'no', got {residues_for_average!r}"
            )
    elif residues_for_average is not None:
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
    return sum(weights_raw[k] for k in keys_to_sum) / float(denom)

def compute_residue_density_raw(
    pdb_path: str,
    sasa_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> ResidueDensityRawDict:
    """Relative side-chain SASA weight per residue."""
    ctx = _get_structure_context(
        pdb_path, sasa_path=sasa_path, allowed_chains=allowed_chains
    )
    atoms = ctx.atoms
    if not atoms:
        return {}
    sasa_data = ctx.sasa_residue
    residue_keys = list(iter_unique_residues(atoms))

    if sasa_path and ("sasa" in getattr(ctx, "parse_errors", {}) or not sasa_data):
        return {k: float("nan") for k in residue_keys}

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
    residues_for_average: Optional[Union[str, Iterable[ResKey4]]] = None,
    density_raw: Optional[ResidueDensityRawDict] = None,
    allowed_chains: Optional[Iterable[str]] = None,
) -> float:
    """Mean SASA-weighted density for residue_category.

    residue_category: keys in the numerator (all if None).
    residues_for_average: denominator — see average_over_residues.
    """
    if density_raw is None:
        density_raw = compute_residue_density_raw(
            pdb_path, sasa_path, allowed_chains=allowed_chains
        )
    if not density_raw:
        return 0.0

    return average_over_residues(
        weights_raw=density_raw,
        residues_for_density=residue_category,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path, allowed_chains=allowed_chains),
    )

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
        if (atom.name, atom.residue_name) in SALT_BRIDGE_NEGATIVE_ATOMS
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
    pH: float = 7.5,
    allowed_chains: Optional[Iterable[str]] = None,
) -> Dict[Tuple[Tuple[str, int, str, str], Tuple[str, int, str, str]], Tuple[float, float]]:
    """Salt bridges from charged atom pairs within MAX_SALT_BRIDGE_DISTANCE at pH.

    One bridge per residue pair; SASA weights returned for pos and neg keys.
    """
    ctx = StructureContext(
        pdb_path,
        sasa_path=sasa_path,
        pka_path=pka_path,
        allowed_chains=allowed_chains,
    )
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

def calculate_salt_bridge_density_average(
    pdb_path: str,
    sasa_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.5,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    residues_for_average: Optional[Union[str, Iterable[Tuple[str, int, str, str]]]] = None,
    salt_bridges: Optional[_SaltBridgesDict] = None,
    allowed_chains: Optional[Iterable[str]] = None,
) -> float:
    """Average SASA-weighted salt-bridge density over a residue set."""
    if salt_bridges is None:
        salt_bridges = detect_salt_bridges(
            pdb_path, sasa_path, pka_path, pH, allowed_chains=allowed_chains
        )
    if not salt_bridges:
        return 0.0

    residue_weights_raw: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    for (pos_key, neg_key), (pos_w, neg_w) in salt_bridges.items():
        residue_weights_raw[pos_key] += pos_w
        residue_weights_raw[neg_key] += neg_w

    return average_over_residues(
        weights_raw=dict(residue_weights_raw),
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path, allowed_chains=allowed_chains),
    )


def get_full_sequence_with_index_map_from_pdb(
    pdb_path: str,
    chain_order: Optional[List[str]] = None,
    allowed_chains: Optional[Iterable[str]] = None,
) -> Tuple[str, Dict[Tuple[str, int, str, str], int]]:
    """Concatenated sequence plus 0-based index for each (res, num, chain, icode) key."""
    atoms = _get_atoms_for_path(pdb_path, allowed_chains=allowed_chains)
    if not atoms:
        return "", {}

    chain_to_keys: Dict[str, List[Tuple[str, int, str, str]]] = {}
    for atom in atoms:
        key = residue_key_from_atom(atom)
        chain = key[2]
        keys = chain_to_keys.setdefault(chain, [])
        if not keys or keys[-1] != key:
            keys.append(key)

    for chain, keys in list(chain_to_keys.items()):
        chain_to_keys[chain] = sorted(keys, key=imgt_residue_sort_key)

    chains = chain_order or sorted(chain_to_keys)

    full_seq_parts: List[str] = []
    index_map: Dict[Tuple[str, int, str, str], int] = {}
    offset = 0
    for chain in chains:
        keys = chain_to_keys.get(chain)
        if not keys:
            continue

        full_seq_parts.append("".join(THREE_TO_ONE.get(k[0], "X") for k in keys))
        for local_idx, key in enumerate(keys):
            index_map[key] = offset + local_idx
        offset += len(keys)

    full_seq = "".join(full_seq_parts)
    return full_seq, index_map


def count_motif_overlapping(seq, motif: str) -> int:

    if not motif:
        return 0
    if isinstance(seq, str):
        if not seq:
            return 0
        n = len(motif)
        return sum(1 for i in range(len(seq) - n + 1) if seq[i : i + n] == motif)
    total = 0
    n = len(motif)
    for s in seq:
        if not s:
            continue
        total += sum(1 for i in range(len(s) - n + 1) if s[i : i + n] == motif)
    return total

def compute_residue_DBSCAN_cluster_labels(
    atoms: List[Atom],
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
) -> Tuple[
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
]:

    ca_by_res = ca_xyz_by_residue(atoms)

    neg_keys: List[Tuple[str, int, str, str]] = []
    neg_coords: List[Tuple[float, float, float]] = []
    pos_keys: List[Tuple[str, int, str, str]] = []
    pos_coords: List[Tuple[float, float, float]] = []
    hydro_keys: List[Tuple[str, int, str, str]] = []
    hydro_coords: List[Tuple[float, float, float]] = []

    for key, xyz in ca_by_res.items():
        res_name = key[0]
        group = _residue_category_group_surface_pcf(
            res_name, _lookup_pka_value(key, pka_data, atoms=atoms), pH
        )
        if group == "negative":
            neg_keys.append(key)
            neg_coords.append(xyz)
        elif group == "positive":
            pos_keys.append(key)
            pos_coords.append(xyz)
        elif group == "hydrophobic":
            hydro_keys.append(key)
            hydro_coords.append(xyz)

    def run_dbscan(
        keys: List,
        coords: List,
        category: str,
    ) -> Dict[Tuple[str, int, str, str], int]:
        if not coords:
            return {}
        eps, min_samples = DBSCAN_EPS_MIN_SAMPLES_BY_CATEGORY_DEFAULT[category]
        X = np.array(coords)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return dict(zip(keys, clustering.labels_.tolist()))

    neg_labels = run_dbscan(neg_keys, neg_coords, "negative")
    pos_labels = run_dbscan(pos_keys, pos_coords, "positive")
    hydro_labels = run_dbscan(hydro_keys, hydro_coords, "hydrophobic")

    return neg_labels, pos_labels, hydro_labels


def _dbscan_cluster_member_counts_and_side_abs_sums(
    labels: Dict[ResKey4, int],
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
) -> Tuple[Dict[int, int], Dict[int, float]]:

    cluster_sizes: Dict[int, int] = defaultdict(int)
    cluster_abs_sasa: Dict[int, float] = defaultdict(float)
    for key, label in labels.items():
        if label == -1:
            continue
        cluster_sizes[label] += 1
        entry = _sasa_lookup(sasa_output_data, key) or {}
        total_side_abs = entry.get("total_side_abs")
        if total_side_abs is not None:
            cluster_abs_sasa[label] += float(total_side_abs)
    return dict(cluster_sizes), dict(cluster_abs_sasa)


def summarize_dbscan_clusters(
    labels: Dict[ResKey4, int],
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
) -> float:

    if not labels:
        return 0.0
    _, cluster_abs_sasa = _dbscan_cluster_member_counts_and_side_abs_sums(
        labels, sasa_output_data
    )
    if not cluster_abs_sasa:
        return 0.0
    return float(sum(cluster_abs_sasa.values()))


def dbscan_cluster_side_abs_sasa_entropy(
    labels: Dict[ResKey4, int],
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
) -> float:

    if not labels:
        return 0.0
    _, cluster_abs_sasa = _dbscan_cluster_member_counts_and_side_abs_sums(
        labels, sasa_output_data
    )
    if not cluster_abs_sasa:
        return 0.0
    s_vals = np.asarray(list(cluster_abs_sasa.values()), dtype=np.float64)
    s_total = float(np.sum(s_vals))
    if s_total <= 0.0:
        return 0.0
    p = s_vals / s_total
    return float(-np.sum(np.where(p > 0.0, p * np.log(p), 0.0)))

def get_pka_for_key(key: ResKey4, pka_output_data: Dict[ResKey4, float]) -> Optional[float]:
    if key in pka_output_data:
        return pka_output_data[key]
    fallback_key = (key[0], key[1], key[2], "")
    if fallback_key in pka_output_data:
        return pka_output_data[fallback_key]
    return None


def _pcf_shell_numeric_token(x: float) -> str:
    v = float(x)
    ir = round(v)
    if abs(v - ir) < 1e-9:
        return str(int(ir))
    return f"{v:g}".replace(".", "p")


def _pcf_shell_key_suffix(r_lo: float, width: float) -> str:
    return f"{_pcf_shell_numeric_token(r_lo)}w{_pcf_shell_numeric_token(width)}A"


def _empty_pcf_cluster_scores(
    bin_starts: Tuple[float, ...],
    bin_widths: Tuple[float, ...],
) -> Dict[str, Optional[float]]:
    if len(bin_starts) != len(bin_widths):
        raise ValueError("bin_starts and bin_widths must have the same length")
    tags = [
        _pcf_shell_key_suffix(float(r), float(w))
        for r, w in zip(bin_starts, bin_widths)
    ]
    out: Dict[str, Optional[float]] = {}
    for cat in ("neg", "pos", "hyd"):
        for tag in tags:
            out[f"pcf_{cat}_{tag}"] = None
    return out


def compute_exposed_pair_correlation_cluster_scores(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[ResKey4, SASAEntry],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
    surface_exposed_threshold: float = SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    bin_starts: Tuple[float, ...] = PCF_CLUSTER_BIN_STARTS_DEFAULT,
    bin_widths: Tuple[float, ...] = PCF_CLUSTER_BIN_WIDTHS_DEFAULT,
    n_permutations: int = PCF_CLUSTER_N_PERMUTATIONS_DEFAULT,
    random_seed: int = 42,
) -> Dict[str, Optional[float]]:
    """PCF clustering scores for exposed Cα by charge/hydrophobic class and distance shell."""
    result = _empty_pcf_cluster_scores(bin_starts, bin_widths)
    if not sasa_output_data:
        return result

    ca_by_res = ca_xyz_by_residue(pdb_atoms)

    residue_exposure = get_exposed_residues(
        sasa_output_data, float(surface_exposed_threshold)
    )
    exposed_keys: Set[ResKey4] = {
        key for key, is_exposed in residue_exposure.items() if is_exposed
    }

    exposed_keys_with_ref: List[ResKey4] = sorted(
        [k for k in exposed_keys if k in ca_by_res],
        key=lambda k: (k[2], k[1], k[3], k[0]),
    )
    if len(exposed_keys_with_ref) < 2:
        for k in result:
            result[k] = 0.0
        return result

    allowed_coords = np.array(
        [ca_by_res[k] for k in exposed_keys_with_ref],
        dtype=np.float64,
    )
    v_ref = convex_hull_volume_from_points(allowed_coords)
    if v_ref <= 0.0:
        for k in result:
            result[k] = 0.0
        return result

    rng = np.random.default_rng(random_seed)

    neg: List[Tuple[float, float, float]] = []
    pos: List[Tuple[float, float, float]] = []
    hydro: List[Tuple[float, float, float]] = []

    for key in exposed_keys_with_ref:
        xyz = ca_by_res[key]
        res_name = key[0]
        pka_val = get_pka_for_key(key, pka_output_data)
        group = _residue_category_group_surface_pcf(res_name, pka_val, pH)
        if group == "negative":
            neg.append(xyz)
        elif group == "positive":
            pos.append(xyz)
        elif group == "hydrophobic":
            hydro.append(xyz)

    shell_tags = [
        _pcf_shell_key_suffix(float(r), float(w))
        for r, w in zip(bin_starts, bin_widths)
    ]

    for cat, coords_list in (
        ("neg", neg),
        ("pos", pos),
        ("hyd", hydro),
    ):
        per_shell = pair_correlation_clustering_score_random_surface_null_by_bin(
            np.array(coords_list, dtype=np.float64),
            allowed_coords,
            v_ref,
            bin_starts,
            bin_widths,
            n_permutations,
            rng,
        )
        for tag, val in zip(shell_tags, per_shell):
            result[f"pcf_{cat}_{tag}"] = val

    return result


def _local_pca_planarity_ratio(neighbor_coords: np.ndarray) -> float:
    """Smallest PCA eigenvalue / sum — low means locally flat."""
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
    # eigh: ascending eigenvalues; index 0 is the smallest.
    return float(vals[0] / total)


def residue_mean_local_planarity(
    atoms: List[Atom],
    residue_key: ResKey4,
    *,
    neighborhood_radius: float = 8.0,
    min_neighbors: int = 4,
    exclude_hydrogens: bool = True,
) -> Optional[float]:
    """Mean local planarity over heavy atoms of one residue (λ_min / trace in r Å shell).

    None if the residue has no scorable atoms (too few neighbors within radius).
    """
    if neighborhood_radius <= 0.0 or min_neighbors < 1:
        return None
    if not atoms:
        return None
    atoms_id = id(atoms)
    existing_ref = _PLANARITY_ATOMS_REFS.get(atoms_id)
    if existing_ref is None or existing_ref != atoms_id:
        stale = [k for k in _RESIDUE_LOCAL_PLANARITY_CACHE if k[0] == atoms_id]
        for k in stale:
            del _RESIDUE_LOCAL_PLANARITY_CACHE[k]
        _PLANARITY_ATOMS_REFS[atoms_id] = atoms_id
    cache_key = (
        atoms_id,
        residue_key,
        float(neighborhood_radius),
        int(min_neighbors),
        bool(exclude_hydrogens),
    )
    if cache_key in _RESIDUE_LOCAL_PLANARITY_CACHE:
        return _RESIDUE_LOCAL_PLANARITY_CACHE[cache_key]

    r = float(neighborhood_radius)

    if exclude_hydrogens:
        tree, coords, _atom_res_keys, heavy_atoms_by_res = get_heavy_atom_tree(atoms)
        res_atom_indices = heavy_atoms_by_res.get(residue_key, [])
        if not res_atom_indices or coords.shape[0] < min_neighbors:
            _RESIDUE_LOCAL_PLANARITY_CACHE[cache_key] = None
            return None

        ratios: List[float] = []
        for atom_idx in res_atom_indices:
            idx = tree.query_ball_point(coords[atom_idx], r)
            if len(idx) < min_neighbors:
                continue
            ratios.append(_local_pca_planarity_ratio(coords[idx]))

        result = float(np.mean(ratios)) if ratios else None
        _RESIDUE_LOCAL_PLANARITY_CACHE[cache_key] = result
        return result

    res_atoms: List[Atom] = []
    for a in atoms:
        if residue_key_from_atom(a) == residue_key:
            res_atoms.append(a)
    if not res_atoms:
        _RESIDUE_LOCAL_PLANARITY_CACHE[cache_key] = None
        return None

    if len(atoms) < min_neighbors:
        _RESIDUE_LOCAL_PLANARITY_CACHE[cache_key] = None
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
        ratios.append(_local_pca_planarity_ratio(coords[idx]))

    result = float(np.mean(ratios)) if ratios else None
    _RESIDUE_LOCAL_PLANARITY_CACHE[cache_key] = result
    return result


def _check_planarity_result(v: Optional[float], key: ResKey4) -> float:
    if v is None or not math.isfinite(v):
        logger.warning(
            "Local planarity undefined for residue %r (too few neighbours or "
            "isolated atom); contributing 0.0 to the aggregate.",
            key,
        )
        return 0.0
    return float(v)


def sum_residue_mean_local_planarity(
    atoms: List[Atom],
    residue_keys: Iterable[ResKey4],
    *,
    neighborhood_radius: float = 8.0,
    min_neighbors: int = 4,
    exclude_hydrogens: bool = True,
) -> float:
    total = 0.0
    for key in residue_keys:
        v = residue_mean_local_planarity(
            atoms,
            key,
            neighborhood_radius=neighborhood_radius,
            min_neighbors=min_neighbors,
            exclude_hydrogens=exclude_hydrogens,
        )
        total += _check_planarity_result(v, key)
    return total


def mean_residue_planarity_over_residues(
    atoms: List[Atom],
    residue_keys: Iterable[ResKey4],
    *,
    neighborhood_radius: float = 8.0,
    min_neighbors: int = 4,
    exclude_hydrogens: bool = True,
) -> float:
    """Mean local planarity over the given residues."""
    keys = list(residue_keys)
    if not keys:
        return 0.0
    return sum_residue_mean_local_planarity(
        atoms,
        keys,
        neighborhood_radius=neighborhood_radius,
        min_neighbors=min_neighbors,
        exclude_hydrogens=exclude_hydrogens,
    ) / float(len(keys))


def compute_inter_chain_buried_sasa(complex_sasa_path: str) -> Optional[float]:
    """Buried interface SASA: H_total + L_total − complex_total."""
    from pathlib import Path
    path = Path(complex_sasa_path)
    if not path.exists():
        return None
    stem = path.stem
    suffix = path.suffix
    base = stem[:-5] if stem.endswith("_full") else stem
    sasa_H_path = str(path.with_name(f"{base}_H_full{suffix}"))
    sasa_L_path = str(path.with_name(f"{base}_L_full{suffix}"))

    try:
        complex_total = parse_sasa(str(path.resolve())).total_sasa
        h_total = parse_sasa(sasa_H_path).total_sasa
        l_total = parse_sasa(sasa_L_path).total_sasa
    except (FileNotFoundError, ValueError, OSError):
        return None
    return h_total + l_total - complex_total

def _enumerate_hbonds(
    pdb_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> List[Tuple[Atom, Atom]]:

    abs_path = os.path.abspath(pdb_path)
    chains_key = tuple(sorted(allowed_chains)) if allowed_chains is not None else None
    cache_key = (abs_path, MAX_HBOND_DISTANCE, MIN_HBOND_ANGLE, MIN_BACKBONE_SEPARATION, chains_key)
    cached = _HBOND_PAIRS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    atoms = _get_atoms_for_path(pdb_path, allowed_chains=allowed_chains)
    residue_cache_key = _residue_keys_cache_key(atoms)
    seq_index = _get_residue_seq_index(atoms)
    atom_lookup = _ATOM_LOOKUP_CACHE.get(residue_cache_key)
    if atom_lookup is None:
        atom_lookup = {
            (atom.chain, atom.residue_number, atom.insertion_code, atom.name): atom
            for atom in atoms
        }
        _ATOM_LOOKUP_CACHE[residue_cache_key] = atom_lookup
    chain_index_to_res = _RES_INDEX_TO_KEY_CACHE.get(residue_cache_key, {})

    donors: List[Atom] = []
    donor_coords_list: List[Tuple[float, float, float]] = []
    donor_res_keys: List[ResKey4] = []
    donor_is_backbone: List[bool] = []
    donor_max_list: List[int] = []
    acceptors: List[Atom] = []
    acceptor_coords_list: List[Tuple[float, float, float]] = []
    acceptor_keys: List[ResKey4] = []
    acceptor_max_list: List[int] = []
    for atom in atoms:
        if is_donor(atom):
            donors.append(atom)
            donor_coords_list.append((atom.x, atom.y, atom.z))
            donor_res_keys.append(residue_key_from_atom(atom))
            donor_is_backbone.append(atom.name in BACKBONE_ATOMS)
            donor_max_list.append(donor_max_hbonds(atom))
        if is_acceptor(atom):
            acceptors.append(atom)
            acceptor_coords_list.append((atom.x, atom.y, atom.z))
            acceptor_keys.append(residue_key_from_atom(atom))
            acceptor_max_list.append(acceptor_max_hbonds(atom))

    if not donors or not acceptors:
        _HBOND_PAIRS_CACHE[cache_key] = []
        return []

    acceptor_coords = np.array(acceptor_coords_list, dtype=np.float64)
    acceptor_tree = cKDTree(acceptor_coords)
    donor_coords = np.array(donor_coords_list, dtype=np.float64)
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
        donor_res_key = donor_res_keys[donor_idx]
        indices = acceptor_tree.query_ball_point(donor_coord, MAX_HBOND_DISTANCE)
        if not indices:
            continue

        is_backbone_donor = donor_is_backbone[donor_idx]
        donor_max = donor_max_list[donor_idx]

        if is_backbone_donor:
            precomputed_base = get_donor_base_atom(
                donor,
                atoms,
                backbone_base_cache,
                atom_lookup=atom_lookup,
                seq_index=seq_index,
                chain_index_to_res=chain_index_to_res,
                residue_cache_key=residue_cache_key,
            )
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
                base = get_donor_base_atom(
                    donor,
                    atoms,
                    backbone_base_cache,
                    atom_lookup=atom_lookup,
                    seq_index=seq_index,
                    chain_index_to_res=chain_index_to_res,
                    residue_cache_key=residue_cache_key,
                )
                sidechain_base_cache[donor] = base
            if base is None:
                continue
            base_vec = sidechain_base_vec_cache.get(donor)
            if base_vec is None:
                base_vec = np.array([base.x - donor.x, base.y - donor.y, base.z - donor.z])
                sidechain_base_vec_cache[donor] = base_vec
            db_norm_sq = float(np.sum(base_vec**2))

        valid_indices = []
        valid_acceptors = []
        for idx in indices:
            acceptor = acceptors[idx]
            if acceptor_hbond_counts.get(acceptor, 0) >= acceptor_max_list[idx]:
                continue
            acceptor_res_key = acceptor_keys[idx]
            if donor_res_key == acceptor_res_key:
                continue
            if is_backbone_donor and acceptor.name in BACKBONE_ATOMS and donor.chain == acceptor.chain:
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

        da_coords = acceptor_coords[valid_indices]
        da_vecs = da_coords - donor_coord
        if db_norm_sq == 0.0:
            continue
        da_norm_sq = np.sum(da_vecs**2, axis=1)
        nonzero_mask = da_norm_sq > 0.0
        if not np.any(nonzero_mask):
            continue
        dot_products = np.einsum("ij,j->i", da_vecs, base_vec)
        cos_threshold_sq = 0.25  # cos(angle) <= -0.5
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
            acc_max = acceptor_max_list[valid_indices[local_idx]]
            if acceptor_hbond_counts.get(acceptor, 0) >= acc_max:
                continue
            pairs.append((donor, acceptor))
            donor_hbond_counts[donor] = donor_hbond_counts.get(donor, 0) + 1
            acceptor_hbond_counts[acceptor] = acceptor_hbond_counts.get(acceptor, 0) + 1

    pairs.sort(key=lambda p: (residue_key_from_atom(p[0]), residue_key_from_atom(p[1])))
    _HBOND_PAIRS_CACHE[cache_key] = pairs
    return pairs

_HbondDensityRaw = Dict[Tuple[str, int, str, str], float]

def _aggregate_hbond_pairs_to_raw(
    pairs: List[Tuple[Atom, Atom]],
    sasa_data: Dict[Tuple[str, int, str, str], SASAEntry],
) -> Dict[Tuple[str, int, str, str], float]:
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

    for donor, acceptor in pairs:
        donor_key = get_cached_residue_key(donor)
        acceptor_key = get_cached_residue_key(acceptor)
        residue_weights_raw[donor_key] += get_cached_sasa_weight(donor)
        residue_weights_raw[acceptor_key] += get_cached_sasa_weight(acceptor)

    return dict(residue_weights_raw)


def compute_hbond_density_raw(
    pdb_path: str,
    sasa_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> _HbondDensityRaw:
    pairs = _enumerate_hbonds(pdb_path, allowed_chains=allowed_chains)
    if not pairs:
        return {}
    ctx = _get_structure_context(
        pdb_path, sasa_path=sasa_path, allowed_chains=allowed_chains
    )
    sasa_data = ctx.sasa_residue
    weights_raw = _aggregate_hbond_pairs_to_raw(pairs, sasa_data)
    return weights_raw


def calculate_global_hbond_density_average(
    pdb_path: str,
    sasa_path: str,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    residues_for_average: Optional[Union[str, Iterable[Tuple[str, int, str, str]]]] = None,
    weights_raw: Optional[Dict[Tuple[str, int, str, str], float]] = None,
    allowed_chains: Optional[Iterable[str]] = None,
) -> float:
    if weights_raw is None:
        weights_raw = compute_hbond_density_raw(
            pdb_path, sasa_path, allowed_chains=allowed_chains
        )
    if not weights_raw:
        return 0.0

    return average_over_residues(
        weights_raw=weights_raw,
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path, allowed_chains=allowed_chains),
    )


_DsspHbondEnergyDensityRaw = Dict[Tuple[str, int, str, str], float]

# If almost no DSSP H-bonds line up with the structure, return NaN instead of 0.
DSSP_HBOND_MIN_ALIGNED_RESIDUE_FRACTION = 0.01


def compute_dssp_hbond_energy_density_raw(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
    allowed_chains: Optional[Iterable[str]] = None,
) -> _DsspHbondEnergyDensityRaw:

    ctx = StructureContext(
        pdb_path,
        sasa_path=sasa_path,
        dssp_path=dssp_path,
        allowed_chains=allowed_chains,
    )
    atoms = ctx.atoms
    sasa_data = ctx.sasa_residue
    dssp_hbonds, dssp_seq_to_pdb = ctx.dssp_hbonds

    def _canonical_residue_keys() -> List[Tuple[str, int, str, str]]:

        try:
            keys = ctx.residue_keys
        except Exception:
            keys = set(iter_unique_residues(atoms)) if atoms else set()
        return sorted(keys, key=lambda k: (k[2], *imgt_residue_sort_key(k), k[0]))

    if dssp_path and (
        "dssp_hbonds" in ctx.parse_errors
        or "dssp" in ctx.parse_errors
    ):
        res_keys = _canonical_residue_keys()
        return {k: float("nan") for k in res_keys}

    if sasa_path and ("sasa" in ctx.parse_errors or not sasa_data):
        res_keys = _canonical_residue_keys()
        return {k: float("nan") for k in res_keys}

    if dssp_path and ("dssp_hbonds" not in ctx.parse_errors and "dssp" not in ctx.parse_errors):
        if not dssp_hbonds or not dssp_seq_to_pdb:
            res_keys = _canonical_residue_keys()
            return {k: float("nan") for k in res_keys}

        try:
            n_struct = len(ctx.residue_keys)
        except Exception:
            n_struct = 0
        n_aligned = len(dssp_hbonds)
        if n_struct > 0 and (n_aligned / n_struct) < DSSP_HBOND_MIN_ALIGNED_RESIDUE_FRACTION:
            res_keys = _canonical_residue_keys()
            return {k: float("nan") for k in res_keys}

    pdb_to_dssp_seq = {pdb_key: dssp_seq for dssp_seq, pdb_key in dssp_seq_to_pdb.items()}
    return _aggregate_dssp_hbond_energy_to_raw(
        dssp_hbonds, dssp_seq_to_pdb, pdb_to_dssp_seq, sasa_data
    )

def calculate_hbond_energy_density_dssp_backbone_only_average(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    residues_for_average: Optional[Union[str, Iterable[Tuple[str, int, str, str]]]] = None,
    dssp_hbond_energy_density_raw: Optional[_DsspHbondEnergyDensityRaw] = None,
    allowed_chains: Optional[Iterable[str]] = None,
) -> float:
    if dssp_hbond_energy_density_raw is None:
        dssp_hbond_energy_density_raw = compute_dssp_hbond_energy_density_raw(
            pdb_path, sasa_path, dssp_path, allowed_chains=allowed_chains
        )
    weights_raw = dssp_hbond_energy_density_raw
    if not weights_raw:
        return float("nan")

    if residues_for_average is None:
        if residues_for_density is None:
            residues_for_average = list(weights_raw.keys())
        else:
            residues_for_average = [k for k in residues_for_density if k in weights_raw]
    elif isinstance(residues_for_average, str):
        if residues_for_average != "no":
            raise ValueError(
                f"residues_for_average must be None, 'no', or an iterable; got {residues_for_average!r}"
            )
    elif not list(residues_for_average):
        return float("nan")

    return average_over_residues(
        weights_raw=weights_raw,
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path, allowed_chains=allowed_chains),
    )

