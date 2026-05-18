from typing import Dict, Iterable, List, Optional, Set, Tuple, Any, Union
import os
import math
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy import optimize
from sklearn.cluster import DBSCAN
import logging
from pathlib import Path
import re
import sys
import pandas as pd

from developability.structure_context import StructureContext, ResKey4
from developability.descriptor_utils import (
    CDR_RANGES_CA,
    _ATOM_LOOKUP_CACHE,
    _RES_INDEX_TO_KEY_CACHE,
    get_residue_region,
    # get_residue_region_map,
    iter_unique_residues,
    _get_atoms_for_path,
    _residue_keys_cache_key,
    # get_residue_keys_by_type,
    _residue_fractional_charge_at_pH,
    is_hydrogen_atom,
    atom_sasa_weight,
    get_residue_sasa_weight,
    residue_side_sasa_abs,
    # residue_main_sasa,
    # residue_side_sasa,
    _get_structure_context,
    _count_residues_in_pdb,
    _get_residue_seq_index,
    _get_charged_residues_at_pH,
    # _RES_INDEX_TO_KEY_CACHE,
    _residue_category_group,
    _residue_category_group_surface_pcf,
    get_exposed_residues,
    _sasa_lookup,
    is_donor,
    is_acceptor,
    get_donor_base_atom,
    donor_max_hbonds,
    acceptor_max_hbonds,
    _aggregate_dssp_hbond_energy_to_raw,
    # _INTER_CHAIN_INTERFACE_CACHE,
    # _compute_inter_chain_interface_from_by_chain,
    get_inter_chain_interface_residues,
    convex_hull_volume_from_points,
    pair_correlation_clustering_score_random_surface_null_by_bin,
    get_heavy_atom_tree,
)

from utils.parsers import (
    ca_xyz_by_residue,
    parse_dssp,
    parse_sasa,
    Atom,
    SASAEntry,
    residue_key_from_atom,
    # parse_motif_to_3letter
)
from utils.chemistry import (
    _ANN_INDEX_AROMATIC_RESIDUES,
    _ANN_INDEX_NEGATIVE_RESIDUES,
    _ANN_INDEX_POSITIVE_RESIDUES,
    AA_1_TO_3,
    ANN_INDEX_N_PERMUTATIONS_DEFAULT,
    ANN_INDEX_SASA_CUTOFF_DEFAULT,
    AROMATIC_RESIDUES,
    CDR_VICINITY_RADIUS,
    CTERM_PKA,
    DBSCAN_EPS_MIN_SAMPLES_BY_CATEGORY_DEFAULT,
    EXPOSURE_REL_ASA_THRESHOLD,
    get_ff19sb_atom_charge,
    get_ff19sb_atom_charge_with_source,
    GLN_ASN_RESIDUES,
    HYDROPHOBIC_RESIDUES,
    INTER_CHAIN_INTERFACE_CUTOFF,
    is_backbone_atom,
    KYTE_DOOLITTLE,
    MAX_HBOND_DISTANCE,
    MAX_SALT_BRIDGE_DISTANCE,
    MIN_BACKBONE_SEPARATION,
    MIN_HBOND_ANGLE,
    NEGATIVE_ATOMS,
    NEGATIVE_CHARGED_RESIDUES,
    normalize_hydropathy,
    NTERM_PKA,
    PCF_CLUSTER_BIN_STARTS_DEFAULT,
    PCF_CLUSTER_BIN_WIDTHS_DEFAULT,
    PCF_CLUSTER_N_PERMUTATIONS_DEFAULT,
    POLAR_RESIDUES,
    POSITIVE_ATOMS,
    POSITIVE_CHARGED_RESIDUES,
    PSH_PAIR_RADIUS,
    RIPLEY_K_DISTANCE,
    RIPLEY_K_N_SAMPLES,
    SCM_MAIN_CHAIN_ATOMS,
    SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    THREE_TO_ONE,
    get_standard_residue_pka,
)

logger = logging.getLogger(__name__)

CLUSTER_LABEL_COLS = [
    "negative_cluster_labels",
    "positive_cluster_labels",
    "aromatic_cluster_labels",
    "hydrophobic_cluster_labels",
    "polar_cluster_labels",
]

_HBOND_PAIRS_CACHE: Dict[Tuple[str, float, float, int], List[Tuple[Atom, Atom]]] = {}
_PDB_SEQUENCE_CACHE: Dict[str, Dict[str, List[str]]] = {}

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
        nterm_pka = nterm_pka_by_chain.get(chain, NTERM_PKA)
        cterm_pka = cterm_pka_by_chain.get(chain, CTERM_PKA)
        # N-terminus: positive group, protonated at low pH
        net += 1.0 / (1.0 + np.power(10.0, pH - nterm_pka))
        # C-terminus: negative group, deprotonated at high pH
        net -= 1.0 / (1.0 + np.power(10.0, cterm_pka - pH))

    return float(net)


def simple_residue_charge_from_sequence(resname: str) -> float:
    """
    Fixed formal charge from 3-letter residue name (no pKa): Asp/Glu −1, Lys/Arg +1, His +0.1.
    """
    r = (resname or "").upper()
    if r in {"ASP", "GLU"}:
        return -1.0
    if r in {"LYS", "ARG"}:
        return 1.0
    if r == "HIS":
        return 0.1
    return 0.0


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
    except (ValueError, Exception):
        return None

def compute_dipole_moment_magnitude(
    pdb_atoms: List[Atom],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
    *,
    debug_charge_stats: Optional[Dict[str, int]] = None,
) -> Optional[float]:
    """
    Dipole moment magnitude ‖μ‖ in e·Å (not Debye).

    Uses Amber ff19SB atom partial charges from ``amino19.lib`` (including H;
    see ``utils.chemistry.FF19SB_ATOM_CHARGES``). Missing (residue, atom) pairs
    use charge ``0.0``.

    ``μ = Σ_i q_i (r_i − r_0)`` with *r_0* the centroid of all input atoms
    (translation-invariant). ``pka_output_data`` and ``pH`` are kept for API
    compatibility with callers; they are not used here.
    """
    _ = (pka_output_data, pH)  # unused; retained for call-site compatibility

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


def compute_hydrophobic_moment_magnitude(
    pdb_atoms: List[Atom],
) -> Optional[float]:
    """
    PROPERMAB-style hydrophobic moment magnitude.

    For each residue, compute the center of geometry over all residue atoms and
    sum ``hydropathy(residue) * r_residue`` using the raw Kyte-Doolittle scale.
    As in ProperMAb, this uses the absolute residue coordinates directly (no
    centroid subtraction), so the result is not translation-invariant.
    """
    if not pdb_atoms:
        return 0.0

    residue_atoms: Dict[ResKey4, List[Atom]] = defaultdict(list)
    for atom in pdb_atoms:
        residue_atoms[residue_key_from_atom(atom)].append(atom)

    residue_cogs: List[np.ndarray] = []
    residue_hydropathy: List[float] = []
    for key4 in sorted(residue_atoms.keys(), key=lambda k: (k[2], k[1], k[3], k[0])):
        kd = KYTE_DOOLITTLE.get((key4[0] or "").strip().upper())
        if kd is None:
            continue
        coords = np.array(
            [(float(a.x), float(a.y), float(a.z)) for a in residue_atoms[key4]],
            dtype=np.float64,
        )
        if coords.size == 0:
            continue
        residue_cogs.append(np.mean(coords, axis=0))
        residue_hydropathy.append(float(kd))

    if not residue_cogs:
        return 0.0

    cog_arr = np.asarray(residue_cogs, dtype=np.float64)
    hyd_arr = np.asarray(residue_hydropathy, dtype=np.float64).reshape((-1, 1))
    hyd_vector = np.sum(cog_arr * hyd_arr, axis=0)
    return float(np.linalg.norm(hyd_vector))

def residue_neighbor_score(
    pdb_atoms: List[Atom],
    sasa_data: Dict[ResKey4, SASAEntry],
    d_cutoff: float,
    *,
    pka_data: Optional[Dict[ResKey4, float]] = None,
    pH: float = 7.0,
    source: str = "charge",
    sasa_weight_sources: bool = True,
    sasa_cutoff: float = EXPOSURE_REL_ASA_THRESHOLD,
    ionizable_only: bool = False,
    center_weight: str = "sasa",
    weight_center_by_sasa: bool = False,
    reduce: str = "neg_abs",
) -> Optional[float]:
    """Generic SASA-weighted residue-pair score using min heavy-atom distance.

    For each residue *i* (center), accumulate contributions from all residues *j*
    (sources) whose heavy atoms lie within ``d_cutoff`` of any heavy atom of *i*.
    Then apply a per-center multiplier and aggregate.

    SASA values are ``total_side_rel`` (relative side-chain SASA, fraction in
    [0, 1]) throughout.  Distance is the minimum heavy-atom distance (via the
    shared cached KDTree from ``get_heavy_atom_tree``).

    Parameters
    ----------
    source : ``"charge"`` | ``"hydrophobicity"`` | ``"charge_pos"`` |
             ``"charge_neg"`` | ``"sasa_only"``
        Property that each source residue *j* contributes (before optional SASA
        weighting).  ``"charge"`` uses the pH-dependent fractional charge;
        ``"hydrophobicity"`` uses the Kyte–Doolittle scale; ``"charge_pos"`` /
        ``"charge_neg"`` use only the positive / negative part of the charge;
        ``"sasa_only"`` sets the property to 1 (pure SASA weighting).
    sasa_weight_sources : bool
        If *True*, source contribution = property(j) × sasa(j); all residues
        are valid sources.
        If *False*, source contribution = property(j) with no SASA multiplier;
        only exposed residues (sasa > ``sasa_cutoff``) are valid sources.
    ionizable_only : bool
        Restrict both centers and sources to residues that contain at least one
        ionizable atom (POSITIVE_ATOMS ∪ NEGATIVE_ATOMS).
    center_weight : ``"sasa"`` | ``"hydrophobicity"`` | ``"charge"`` |
                    ``"charge_pos"`` | ``"charge_neg"`` | ``"none"``
        Multiplier applied to each center's accumulated sum before aggregation.
    weight_center_by_sasa : bool
        If *True*, additionally multiply the center weight by ``sasa(i)``, so
        the effective center multiplier becomes ``center_weight(i) × sasa(i)``.
        Used by SAP to make the aggregation symmetric with the source weighting.
    reduce : ``"neg_abs"`` | ``"pos_abs"`` | ``"sum"``
        ``"neg_abs"``: ``|Σ_{i: result_i < 0} result_i|``  (SCM anionic clustering).
        ``"pos_abs"``: ``Σ_{i: result_i > 0} result_i``    (SCM cationic clustering).
        ``"sum"``:     ``Σ_i result_i``                     (SAP-style).
    """
    if not pdb_atoms or not sasa_data:
        return None

    needs_charge = source in ("charge", "charge_pos", "charge_neg") or center_weight in (
        "charge", "charge_pos", "charge_neg"
    )

    # --- identify which residues to include ---
    ionizable_keys: Optional[Set[ResKey4]] = None
    if ionizable_only:
        ionizable_atom_names: Dict[str, set] = {}
        for atom_name, res_name in POSITIVE_ATOMS:
            ionizable_atom_names.setdefault(res_name, set()).add(atom_name)
        for atom_name, res_name in NEGATIVE_ATOMS:
            ionizable_atom_names.setdefault(res_name, set()).add(atom_name)
        ionizable_keys = set()
        for a in pdb_atoms:
            allowed = ionizable_atom_names.get((a.residue_name or "").strip().upper())
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

    # --- per-residue arrays ---
    sasa_arr = np.zeros(m, dtype=np.float64)
    charge_arr = np.zeros(m, dtype=np.float64)

    for i, key4 in enumerate(res_keys):
        entry = sasa_data.get(key4)
        sasa_arr[i] = float(getattr(entry, "total_side_rel", 0.0) or 0.0)
        if needs_charge:
            pka = pka_data.get(key4) if pka_data else None
            charge_arr[i] = _residue_fractional_charge_at_pH(key4[0], pka, pH)

    exposed_arr: np.ndarray = sasa_arr > sasa_cutoff

    # --- source contribution per source j ---
    if source == "charge":
        raw_source = charge_arr
    elif source == "hydrophobicity":
        raw_source = np.array(
            [float(KYTE_DOOLITTLE.get(k[0], 0.0)) for k in res_keys], dtype=np.float64
        )
    elif source == "charge_pos":
        raw_source = np.maximum(0.0, charge_arr)
    elif source == "charge_neg":
        raw_source = np.minimum(0.0, charge_arr)
    elif source == "sasa_only":
        raw_source = np.ones(m, dtype=np.float64)
    else:
        raise ValueError(f"residue_neighbor_score: unknown source={source!r}")

    if sasa_weight_sources:
        source_contrib = raw_source * sasa_arr
        valid_source = np.ones(m, dtype=bool)
    else:
        source_contrib = raw_source
        valid_source = exposed_arr

    # --- center-level multiplier ---
    if center_weight == "sasa":
        center_arr = sasa_arr
    elif center_weight == "hydrophobicity":
        center_arr = np.array(
            [float(KYTE_DOOLITTLE.get(k[0], 0.0)) for k in res_keys], dtype=np.float64
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

    # --- pairwise accumulation via shared heavy-atom KDTree ---
    heavy_tree, heavy_coords, heavy_atom_res_keys, heavy_atoms_by_res = get_heavy_atom_tree(pdb_atoms)

    score_arr = np.zeros(m, dtype=np.float64)
    for ri, key_i in enumerate(res_keys):
        atom_indices_i = heavy_atoms_by_res.get(key_i, [])
        if not atom_indices_i:
            continue
        # Collect the set of distinct neighbor residue indices within d_cutoff.
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

    # --- reduce ---
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


# Spatial Charge Map — wrapper around residue_neighbor_score
def scm_score_from_pka(
    pdb_path: str,
    sasa_path: str,
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    d_cutoff: float = 10.0,
    sasa_cutoff: float = EXPOSURE_REL_ASA_THRESHOLD,
    sasa_weighting: bool = True,
    reduce: str = "neg_abs",
) -> Optional[float]:
    """Spatial Charge Map score.

    ``reduce="neg_abs"`` captures anionic charge clustering (default, original SCM).
    ``reduce="pos_abs"`` captures cationic charge clustering.
    """
    ctx = _get_structure_context(pdb_path, sasa_path=sasa_path)
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
            sasa_weight_sources=sasa_weighting,
            sasa_cutoff=sasa_cutoff,
            ionizable_only=True,
            center_weight="sasa" if sasa_weighting else "none",
            reduce=reduce,
        )
    except Exception as e:
        logger.warning("SCM score computation failed: %s", e, exc_info=True)
        return None


def scm_score_by_atoms(
    pdb_path: str,
    d_cutoff: float = 10.0,
) -> Optional[Dict[str, float]]:
    """Atom-level SCM using ff19SB atom partial charges.

    For each atom ``i``, compute:
        ``SCM_i = Σ_j q_j`` for atoms ``j`` within ``d_cutoff`` of ``i``
        (excluding ``i`` itself), where ``q_j`` is the atom partial charge.

    Returns two aggregated features:
      - ``scm_by_atoms_neg``: ``Σ_{i: SCM_i < 0} SCM_i``
      - ``scm_by_atoms_pos``: ``Σ_{i: SCM_i > 0} SCM_i``
    """
    atoms = _get_atoms_for_path(pdb_path)
    if not atoms:
        return None

    try:
        coords = np.array([(a.x, a.y, a.z) for a in atoms], dtype=np.float64)
        charges = np.array(
            [
                get_ff19sb_atom_charge(
                    (a.residue_name or "").strip().upper(),
                    a.name or "",
                )
                for a in atoms
            ],
            dtype=np.float64,
        )

        if coords.shape[0] == 0:
            return None
        if coords.shape[0] == 1:
            return {"scm_by_atoms_neg": 0.0, "scm_by_atoms_pos": 0.0}

        tree = cKDTree(coords)
        scmi = np.zeros(coords.shape[0], dtype=np.float64)
        neighbors_by_atom = tree.query_ball_point(coords, r=d_cutoff)
        for i, neighbor_indices in enumerate(neighbors_by_atom):
            if not neighbor_indices:
                continue
            # Exclude the center atom i; analogous to residue-level SCM (j != i).
            scmi[i] = float(sum(charges[j] for j in neighbor_indices if j != i))

        neg_mask = scmi < 0.0
        pos_mask = scmi > 0.0
        return {
            "scm_by_atoms_neg": float(np.sum(scmi[neg_mask])),
            "scm_by_atoms_pos": float(np.sum(scmi[pos_mask])),
        }
    except Exception as e:
        logger.warning("Atom-level SCM computation failed: %s", e, exc_info=True)
        return None


def scm_score_propermab_like_fast(
    pdb_path: str,
    sasa_path: str,
    d_cutoff: float = 10.0,
    sidechain_sasa_cutoff: float = 10.0,
) -> Optional[float]:
    """
    Fast approximation of ProperMAb's SCM score.

    ProperMAb computes atom-level SCM with force-field partial charges, excludes
    main-chain atoms from the *neighbor* set, keeps all atoms as centers, and
    only counts solvent-exposed neighbors. We do not have atom-level SASA here,
    so exposure is approximated from residue side-chain absolute SASA: all
    side-chain atoms of a residue are treated as exposed neighbors when that
    residue's ``total_side_abs`` exceeds ``sidechain_sasa_cutoff``.

    The final structure score matches ProperMAb's reduction:
    ``abs(sum(SCM_i for SCM_i < 0))``.
    """
    ctx = _get_structure_context(pdb_path, sasa_path=sasa_path)
    atoms = ctx.atoms
    if not atoms:
        return None
    sasa_data = ctx.sasa_residue
    if not sasa_data:
        return None

    try:
        residue_atom_names: Dict[ResKey4, Set[str]] = defaultdict(set)
        for atom in atoms:
            residue_atom_names[residue_key_from_atom(atom)].add(
                (atom.name or "").strip().upper()
            )

        coords = np.array([(a.x, a.y, a.z) for a in atoms], dtype=np.float64)
        charges = np.array(
            [
                get_ff19sb_atom_charge(
                    (a.residue_name or "").strip().upper(),
                    a.name or "",
                    residue_atom_names=residue_atom_names.get(residue_key_from_atom(a)),
                )
                for a in atoms
            ],
            dtype=np.float64,
        )

        if coords.shape[0] <= 1:
            return 0.0

        neighbor_allowed = np.zeros(coords.shape[0], dtype=bool)
        for i, atom in enumerate(atoms):
            if is_backbone_atom(atom):
                continue
            key4 = residue_key_from_atom(atom)
            entry = sasa_data.get(key4)
            side_abs = float(getattr(entry, "total_side_abs", 0.0) or 0.0) if entry is not None else 0.0
            if side_abs > float(sidechain_sasa_cutoff):
                neighbor_allowed[i] = True

        tree = cKDTree(coords)
        scmi = np.zeros(coords.shape[0], dtype=np.float64)
        neighbors_by_atom = tree.query_ball_point(coords, r=d_cutoff)
        for i, neighbor_indices in enumerate(neighbors_by_atom):
            if not neighbor_indices:
                continue
            scmi[i] = float(
                sum(
                    charges[j]
                    for j in neighbor_indices
                    if j != i and neighbor_allowed[j]
                )
            )

        neg_mask = scmi < 0.0
        return float(abs(np.sum(scmi[neg_mask])))
    except Exception as e:
        logger.warning("SCM pm-like computation failed: %s", e, exc_info=True)
        return None


def compute_atom_patch_area_cdr_pm_like_fast(
    pdb_atoms: List[Atom],
    atom_sasa_by_serial: Dict[int, float],
    cdr_keys: Set[ResKey4],
    *,
    charge_radius: float = 8.0,
    charge_sigma: float = 3.0,
    hydrophobic_radius: float = 6.0,
    hydrophobic_sigma: float = 2.5,
    cdr_distance_cutoff: float = 5.0,
    charge_eps: float = 4.0,
    charge_min_samples: int = 4,
    hydrophobic_eps: float = 3.0,
    hydrophobic_min_samples: int = 4,
    min_atom_sasa: float = 0.01,
    charge_seed_quantile: float = 0.75,
    hydrophobic_seed_quantile: float = 0.75,
    positive_charge_seed_threshold: Optional[float] = 0.53,
    negative_charge_seed_threshold: Optional[float] = 0.34,
    hydrophobic_seed_threshold: Optional[float] = 12.0,
    min_charge_seed_abs: float = 0.10,
    min_hydrophobic_seed: float = 0.50,
    charge_expand_ratio: float = 0.65,
    hydrophobic_expand_ratio: float = 0.65,
    cluster_expand_radius: Optional[float] = None,
) -> Dict[str, float]:
    """
    Fast exposed-atom patch proxy intended as a closer, surface-aware alternative
    to the residue C-alpha patch descriptors.

    Design:
      - centers / seed points are exposed atoms (atom SASA > 0)
      - electrostatic field is a short-range Gaussian-smoothed sum of ff19SB atom
        partial charges over *all* atoms, including hydrogens and terminal atoms
      - hydrophobic field is a short-range Gaussian-smoothed sum of positive
        Kyte-Doolittle residue weights over non-backbone atoms
      - seed thresholds default to fixed dataset-level cutoffs; quantile fallback is
        retained only when a fixed threshold is explicitly disabled with ``None``
      - clusters are built on seed-atom coordinates with DBSCAN
      - accepted clusters are expanded to nearby exposed atoms that meet a weaker
        same-sign field threshold before patch area is summed
      - CDR proximity is defined by atom-to-atom distance from the cluster to any
        atom belonging to a CDR residue

    Returns three new CDR-focused features:
      ``pos_patch_area_cdr_pm_like_fast``
      ``neg_patch_area_cdr_pm_like_fast``
      ``hyd_patch_area_cdr_pm_like_fast``
    """
    result = {
        "pos_patch_area_cdr_pm_like_fast": 0.0,
        "neg_patch_area_cdr_pm_like_fast": 0.0,
        "hyd_patch_area_cdr_pm_like_fast": 0.0,
    }
    if not pdb_atoms or not atom_sasa_by_serial or not cdr_keys:
        return result

    residue_atom_names: Dict[ResKey4, Set[str]] = defaultdict(set)
    for atom in pdb_atoms:
        residue_atom_names[residue_key_from_atom(atom)].add((atom.name or "").strip().upper())

    all_atoms = list(pdb_atoms)
    all_coords = np.asarray([(float(a.x), float(a.y), float(a.z)) for a in all_atoms], dtype=np.float64)
    if all_coords.shape[0] < 2:
        return result
    all_tree = cKDTree(all_coords)

    all_charge_weights = np.asarray(
        [
            float(
                get_ff19sb_atom_charge(
                    (a.residue_name or "").strip().upper(),
                    a.name or "",
                    residue_atom_names=residue_atom_names.get(residue_key_from_atom(a)),
                )
            )
            for a in all_atoms
        ],
        dtype=np.float64,
    )
    all_hydrophobic_weights = np.asarray(
        [
            float(max(KYTE_DOOLITTLE.get((a.residue_name or "").strip().upper(), 0.0), 0.0))
            if not is_backbone_atom(a)
            else 0.0
            for a in all_atoms
        ],
        dtype=np.float64,
    )

    exposed_atoms: List[Atom] = []
    exposed_coords: List[Tuple[float, float, float]] = []
    exposed_sasa: List[float] = []
    exposed_charge_field: List[float] = []
    exposed_hydrophobic_field: List[float] = []
    exposed_is_hydrophobic_seedable: List[bool] = []

    for atom in all_atoms:
        sasa_abs = float(atom_sasa_by_serial.get(int(atom.serial), 0.0) or 0.0)
        if sasa_abs <= float(min_atom_sasa):
            continue

        center = np.asarray((float(atom.x), float(atom.y), float(atom.z)), dtype=np.float64)

        charge_neighbors = all_tree.query_ball_point(center, r=float(charge_radius))
        charge_field = 0.0
        for j in charge_neighbors:
            other = all_atoms[j]
            if other.serial == atom.serial:
                continue
            d = float(np.linalg.norm(center - all_coords[j]))
            if d <= 1e-8:
                continue
            w = math.exp(-0.5 * (d / float(charge_sigma)) ** 2)
            charge_field += float(all_charge_weights[j]) * w

        hyd_neighbors = all_tree.query_ball_point(center, r=float(hydrophobic_radius))
        hyd_field = 0.0
        for j in hyd_neighbors:
            other = all_atoms[j]
            if other.serial == atom.serial:
                continue
            src_w = float(all_hydrophobic_weights[j])
            if src_w <= 0.0:
                continue
            d = float(np.linalg.norm(center - all_coords[j]))
            if d <= 1e-8:
                continue
            w = math.exp(-0.5 * (d / float(hydrophobic_sigma)) ** 2)
            hyd_field += src_w * w

        exposed_atoms.append(atom)
        exposed_coords.append((float(atom.x), float(atom.y), float(atom.z)))
        exposed_sasa.append(sasa_abs)
        exposed_charge_field.append(charge_field)
        exposed_hydrophobic_field.append(hyd_field)
        exposed_is_hydrophobic_seedable.append(
            (not is_backbone_atom(atom))
            and float(max(KYTE_DOOLITTLE.get((atom.residue_name or "").strip().upper(), 0.0), 0.0)) > 0.0
        )

    if len(exposed_atoms) < 2:
        return result

    exp_coords_arr = np.asarray(exposed_coords, dtype=np.float64)
    exp_sasa_arr = np.asarray(exposed_sasa, dtype=np.float64)
    exp_charge_field_arr = np.asarray(exposed_charge_field, dtype=np.float64)
    exp_hyd_field_arr = np.asarray(exposed_hydrophobic_field, dtype=np.float64)
    exp_hyd_seedable_arr = np.asarray(exposed_is_hydrophobic_seedable, dtype=bool)
    exp_tree = cKDTree(exp_coords_arr)

    cdr_atom_coords = np.asarray(
        [
            (float(atom.x), float(atom.y), float(atom.z))
            for atom in all_atoms
            if residue_key_from_atom(atom) in cdr_keys
        ],
        dtype=np.float64,
    )
    cdr_tree = cKDTree(cdr_atom_coords) if cdr_atom_coords.shape[0] > 0 else None

    def _seed_threshold(
        values: np.ndarray,
        *,
        positive: bool,
        quantile: float,
        floor: float,
    ) -> Optional[float]:
        if positive:
            pool = values[values > 0.0]
        else:
            pool = np.abs(values[values < 0.0])
        if pool.size == 0:
            return None
        return float(max(float(floor), float(np.quantile(pool, quantile))))

    pos_thr = (
        float(max(float(min_charge_seed_abs), float(positive_charge_seed_threshold)))
        if positive_charge_seed_threshold is not None
        else _seed_threshold(
            exp_charge_field_arr,
            positive=True,
            quantile=charge_seed_quantile,
            floor=min_charge_seed_abs,
        )
    )
    neg_thr = (
        float(max(float(min_charge_seed_abs), float(negative_charge_seed_threshold)))
        if negative_charge_seed_threshold is not None
        else _seed_threshold(
            exp_charge_field_arr,
            positive=False,
            quantile=charge_seed_quantile,
            floor=min_charge_seed_abs,
        )
    )
    hyd_thr = (
        float(max(float(min_hydrophobic_seed), float(hydrophobic_seed_threshold)))
        if hydrophobic_seed_threshold is not None
        else _seed_threshold(
            exp_hyd_field_arr[exp_hyd_seedable_arr] if np.any(exp_hyd_seedable_arr) else np.asarray([], dtype=np.float64),
            positive=True,
            quantile=hydrophobic_seed_quantile,
            floor=min_hydrophobic_seed,
        )
    )

    def _clustered_cdr_area(
        seed_mask: np.ndarray,
        *,
        field_values: np.ndarray,
        eligible_mask: np.ndarray,
        polarity: str,
        seed_threshold: float,
        expand_ratio: float,
        eps: float,
        min_samples: int,
    ) -> float:
        if seed_mask.sum() < max(2, int(min_samples)):
            return 0.0
        seed_indices = np.flatnonzero(seed_mask)
        seed_coords = exp_coords_arr[seed_indices]
        labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(seed_coords).labels_
        total = 0.0
        claimed = np.zeros(exp_coords_arr.shape[0], dtype=bool)
        expand_radius = float(cluster_expand_radius) if cluster_expand_radius is not None else float(eps)
        relaxed_threshold = float(seed_threshold) * float(expand_ratio)
        for label in sorted({int(x) for x in labels.tolist() if int(x) != -1}):
            members = labels == label
            if not np.any(members):
                continue
            member_seed_indices = seed_indices[members]
            member_seed_coords = exp_coords_arr[member_seed_indices]

            expanded_index_set: Set[int] = set(int(i) for i in member_seed_indices.tolist())
            for hit_list in exp_tree.query_ball_point(member_seed_coords, r=expand_radius):
                expanded_index_set.update(int(i) for i in hit_list)

            expanded_indices = np.asarray(sorted(expanded_index_set), dtype=np.int64)
            if polarity == "pos":
                expanded_indices = expanded_indices[
                    eligible_mask[expanded_indices] & (field_values[expanded_indices] >= relaxed_threshold)
                ]
            elif polarity == "neg":
                expanded_indices = expanded_indices[
                    eligible_mask[expanded_indices] & (field_values[expanded_indices] <= -relaxed_threshold)
                ]
            else:
                expanded_indices = expanded_indices[
                    eligible_mask[expanded_indices] & (field_values[expanded_indices] >= relaxed_threshold)
                ]

            if expanded_indices.size == 0:
                expanded_indices = member_seed_indices
            else:
                expanded_indices = np.unique(
                    np.concatenate([expanded_indices, member_seed_indices]).astype(np.int64, copy=False)
                )

            if cdr_tree is not None:
                near_cdr = cdr_tree.query_ball_point(exp_coords_arr[expanded_indices], r=float(cdr_distance_cutoff))
                if not any(bool(hits) for hits in near_cdr):
                    continue

            new_indices = expanded_indices[~claimed[expanded_indices]]
            if new_indices.size == 0:
                continue
            claimed[new_indices] = True
            total += float(np.sum(exp_sasa_arr[new_indices]))
        return float(total)

    pos_seed_mask = (
        exp_charge_field_arr >= float(pos_thr)
        if pos_thr is not None
        else np.zeros(exp_charge_field_arr.shape[0], dtype=bool)
    )
    neg_seed_mask = (
        exp_charge_field_arr <= -float(neg_thr)
        if neg_thr is not None
        else np.zeros(exp_charge_field_arr.shape[0], dtype=bool)
    )
    hyd_seed_mask = (
        exp_hyd_seedable_arr & (exp_hyd_field_arr >= float(hyd_thr))
        if hyd_thr is not None
        else np.zeros(exp_hyd_field_arr.shape[0], dtype=bool)
    )

    result["pos_patch_area_cdr_pm_like_fast"] = _clustered_cdr_area(
        pos_seed_mask,
        field_values=exp_charge_field_arr,
        eligible_mask=np.ones(exp_charge_field_arr.shape[0], dtype=bool),
        polarity="pos",
        seed_threshold=float(pos_thr),
        expand_ratio=float(charge_expand_ratio),
        eps=charge_eps,
        min_samples=charge_min_samples,
    )
    result["neg_patch_area_cdr_pm_like_fast"] = _clustered_cdr_area(
        neg_seed_mask,
        field_values=exp_charge_field_arr,
        eligible_mask=np.ones(exp_charge_field_arr.shape[0], dtype=bool),
        polarity="neg",
        seed_threshold=float(neg_thr),
        expand_ratio=float(charge_expand_ratio),
        eps=charge_eps,
        min_samples=charge_min_samples,
    )
    result["hyd_patch_area_cdr_pm_like_fast"] = _clustered_cdr_area(
        hyd_seed_mask,
        field_values=exp_hyd_field_arr,
        eligible_mask=exp_hyd_seedable_arr,
        polarity="hyd",
        seed_threshold=float(hyd_thr),
        expand_ratio=float(hydrophobic_expand_ratio),
        eps=hydrophobic_eps,
        min_samples=hydrophobic_min_samples,
    )
    return result


# SAP analog — wrapper around residue_neighbor_score.
# Now uses total_side_rel (relative side-chain SASA) and min heavy-atom distance.
def sum_side_abs_fraction_within_cutoff(
    pdb_atoms: List[Atom],
    sasa_data: Dict[ResKey4, SASAEntry],
    cutoff: float = 10.0,
    sap_mode: bool = False,
    positive_charge_mode: bool = False,
    negative_charge_mode: bool = False,
    pka_output_data: Optional[Dict[ResKey4, float]] = None,
    pH: float = 7.0,
) -> float:
    """SAP-style score: for each residue pair (i, j) within ``cutoff``
    (min heavy-atom distance), accumulate neighbor properties × SASA at center *i*,
    then weight each center's accumulated sum before summing over *i*.

    **Hydrophobicity** (``sap_mode``): all residues; Kyte–Doolittle at center and
    neighbor; ``ionizable_only=False``.

    **Full charge** (default flags): ionizable residues only; signed fractional
    charge at center and neighbor; symmetric ``charge × SASA`` weighting.

    **Positive / negative SAP** (``positive_charge_mode`` / ``negative_charge_mode``):
    every residue may be a center (charged or neutral). Neighbor contributions
    use only the positive part (``charge_pos``) or negative part (``charge_neg``)
    of fractional charge × SASA. Centers are weighted by relative side-chain
    SASA only (``center_weight="sasa"``, no extra charge factor).

    SASA: ``total_side_rel`` (relative, [0, 1]).
    Distance: min heavy-atom distance (shared cached KDTree).
    """
    if not pdb_atoms or not sasa_data:
        return 0.0
    if sap_mode:
        src = "hydrophobicity"
        ionizable = False
        ctr = "hydrophobicity"
        w_sasa_ctr = True
    elif positive_charge_mode:
        src = "charge_pos"
        ionizable = False
        ctr = "sasa"
        w_sasa_ctr = False
    elif negative_charge_mode:
        src = "charge_neg"
        ionizable = False
        ctr = "sasa"
        w_sasa_ctr = False
    else:
        src = "charge"
        ionizable = True
        ctr = "charge"
        w_sasa_ctr = True
    result = residue_neighbor_score(
        pdb_atoms,
        sasa_data,
        cutoff,
        pka_data=pka_output_data,
        pH=pH,
        source=src,
        sasa_weight_sources=True,
        ionizable_only=ionizable,
        center_weight=ctr,
        weight_center_by_sasa=w_sasa_ctr,
        reduce="sum",
    )
    return float(result) if result is not None else 0.0


def compute_sap_shell_synergy_scores(
    pdb_atoms: List[Atom],
    sasa_data: Dict[ResKey4, SASAEntry],
    pka_data: Optional[Dict[ResKey4, float]],
    pH: float,
    d_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    Single entry point for structure-level SAP metrics (10 Å min heavy-atom
    neighborhood, ``total_side_rel`` weights).

    **Charge SAP** (same as ``positive_charge_mode`` / ``negative_charge_mode``
    in :func:`sum_side_abs_fraction_within_cutoff`): ``sap_pos_charge_score``,
    ``sap_neg_charge_score`` = ``Σ_i total_side_rel(i) × env_{pos|neg}(i)``.

    **Symmetric hydrophobicity SAP** (``sap_mode`` in
    :func:`sum_side_abs_fraction_within_cutoff`): ``sap_hydro_score``.

    **Neighbor-shell channels** (pos/neg-style centering only — not
    ``sap_hydro_score``): ``sap_hyd_env_score``, ``sap_aromatic_score``,
    ``sap_histidine_score``.

    **Composites** use the same neighbor row sums ``pos_i, neg_i, hyd_i, aro_i,
    his_i`` as for synergies; denominators use ``1 + max(0, -neg_i)``.
    """
    out: Dict[str, float] = {
        # "sap_hydro_score": 0.0,
        # "sap_pos_charge_score": 0.0,
        # "sap_neg_charge_score": 0.0,
        # "sap_hyd_env_score": 0.0,
        "sap_aromatic_score": 0.0,
        "sap_histidine_score": 0.0,
        "sap_pos_aro_synergy": 0.0,
        # "sap_pos_hyd_synergy": 0.0,
        # "sap_pos_shield": 0.0,
        # "sap_heme_like": 0.0,
        "sap_aro_neg_contrast": 0.0,
        # "sap_hyd_neg_contrast": 0.0,
        # "sap_sticky_mix": 0.0,
    }
    if not pdb_atoms or not sasa_data:
        return out

    # out["sap_hydro_score"] = sum_side_abs_fraction_within_cutoff(
    #     pdb_atoms, sasa_data, cutoff=d_cutoff, sap_mode=True
    # )

    pka_lookup = pka_data if pka_data is not None else {}

    res_keys: List[ResKey4] = list(iter_unique_residues(pdb_atoms))
    if not res_keys:
        return out

    m = len(res_keys)
    res_key_to_idx: Dict[ResKey4, int] = {k: i for i, k in enumerate(res_keys)}

    sasa_arr = np.zeros(m, dtype=np.float64)
    charge_arr = np.zeros(m, dtype=np.float64)
    kd_arr = np.zeros(m, dtype=np.float64)
    aro_arr = np.zeros(m, dtype=np.float64)
    his_arr = np.zeros(m, dtype=np.float64)

    for i, key4 in enumerate(res_keys):
        entry = sasa_data.get(key4)
        sasa_arr[i] = float(getattr(entry, "total_side_rel", 0.0) or 0.0)
        aa = (key4[0] or "").strip().upper()
        pka = pka_lookup.get(key4)
        charge_arr[i] = _residue_fractional_charge_at_pH(aa, pka, pH)
        kd_arr[i] = float(KYTE_DOOLITTLE.get(aa, 0.0))
        aro_arr[i] = 1.0 if aa in AROMATIC_RESIDUES else 0.0
        his_arr[i] = 1.0 if aa == "HIS" else 0.0

    pos_src = np.maximum(0.0, charge_arr) * sasa_arr
    neg_src = np.minimum(0.0, charge_arr) * sasa_arr
    hyd_src = kd_arr * sasa_arr
    aro_src = aro_arr * sasa_arr
    his_src = his_arr * sasa_arr

    pos_env = np.zeros(m, dtype=np.float64)
    neg_env = np.zeros(m, dtype=np.float64)
    hyd_env = np.zeros(m, dtype=np.float64)
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
                hyd_env[ri] += hyd_src[rj]
                aro_env[ri] += aro_src[rj]
                his_env[ri] += his_src[rj]

    w = sasa_arr
    neg_mag = np.maximum(0.0, -neg_env)

    # out["sap_pos_charge_score"] = float(np.sum(w * pos_env))
    # out["sap_neg_charge_score"] = float(np.sum(w * neg_env))

    # out["sap_hyd_env_score"] = float(np.sum(w * hyd_env))
    out["sap_aromatic_score"] = float(np.sum(w * aro_env))
    out["sap_histidine_score"] = float(np.sum(w * his_env))

    out["sap_pos_aro_synergy"] = float(np.sum(w * pos_env * aro_env))
    # out["sap_pos_hyd_synergy"] = float(np.sum(w * pos_env * hyd_env))
    # out["sap_pos_shield"] = float(np.sum(w * pos_env / (1.0 + neg_mag)))
    # out["sap_heme_like"] = float(
    #     np.sum(w * (aro_env + 0.5 * his_env) * (pos_env + 0.5 * his_env) / (1.0 + neg_mag))
    # )
    out["sap_aro_neg_contrast"] = float(np.sum(w * aro_env * neg_env))
    # out["sap_hyd_neg_contrast"] = float(np.sum(w * hyd_env * neg_env))
    # out["sap_sticky_mix"] = float(np.sum(w * (pos_env + aro_env + hyd_env + neg_env)))
    return out


# Generic helpers
ResidueDensityRawDict = Dict[ResKey4, float]

def average_over_residues(
    *,
    weights_raw: Dict[ResKey4, float],
    counts: Optional[Dict[ResKey4, int]] = None,
    residues_for_density: Optional[Iterable[ResKey4]] = None,
    residues_for_average: Optional[Union[str, Iterable[ResKey4]]] = None,
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
    - denominator:
      * ``None``: ``denom_total_residues`` (total residues in the PDB)
      * ``"all"``: same as ``None``
      * ``"no"``: no division (returns the numerator sum as a scalar)
      * otherwise: ``len(set(residues_for_average))`` for an iterable of residue keys
    """
    if not weights_raw:
        return 0.0

    if isinstance(residues_for_average, str):
        if residues_for_average == "all":
            denom = denom_total_residues
        elif residues_for_average == "no":
            denom = 1
        else:
            raise ValueError(
                f"residues_for_average string must be 'all' or 'no', got {residues_for_average!r}"
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

    # If SASA was requested but failed to parse / yielded no usable records, propagate
    # failure as NaN rather than returning a "valid" zero density.
    if sasa_path and ("sasa" in getattr(ctx, "parse_errors", {}) or not sasa_data):
        return {k: float("nan") for k in residue_keys}

    weights: ResidueDensityRawDict = {}
    for key4 in residue_keys:
        weight = get_residue_sasa_weight(key4, sasa_data, use_side_chain=True)
        weights[key4] = weight
    return weights


def compute_residue_side_abs_density_raw(
    pdb_path: str,
    sasa_path: str,
) -> ResidueDensityRawDict:
    """
    Per-residue absolute side-chain SASA (Å²), same key space as :func:`compute_residue_density_raw`.

    On SASA parse failure, returns NaN-filled dict like the relative-density helper.
    """
    ctx = _get_structure_context(pdb_path, sasa_path=sasa_path)
    atoms = ctx.atoms
    if not atoms:
        return {}
    sasa_data = ctx.sasa_residue
    residue_keys = list(iter_unique_residues(atoms))

    if sasa_path and ("sasa" in getattr(ctx, "parse_errors", {}) or not sasa_data):
        return {k: float("nan") for k in residue_keys}

    weights: ResidueDensityRawDict = {}
    for key4 in residue_keys:
        weights[key4] = float(residue_side_sasa_abs(key4, sasa_data))
    return weights


def calculate_residue_category_density_average(
    pdb_path: str,
    sasa_path: str,
    *,
    residue_category: Optional[Iterable[ResKey4]] = None,
    weighted: bool = True,
    sqrt_weights: bool = True,
    residues_for_average: Optional[Union[str, Iterable[ResKey4]]] = None,
    density_raw: Optional[ResidueDensityRawDict] = None,
) -> float:
    """
    the category is any list/set of residue keys (e.g. polar, aromatic, inter-chain,
    CDR, or exposed) determined elsewhere and passed in
    density can be SASA-weighted or raw count; the average can be over total residues or over
    the category itself

    ``residues_for_average``: residue iterable for denominator size, or ``None`` /
    ``"all"`` for total PDB residue count, or ``"no"`` for an undivided sum (see
    :func:`average_over_residues`).
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
            sqrt_weights=sqrt_weights,
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

# Count of salt bridges for given residues, weighted by sqrt relative SASA and averaged over given residues
def calculate_salt_bridge_density_average(
    pdb_path: str,
    sasa_path: str,
    pka_path: Optional[str] = None,
    pH: float = 7.4,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    weighted: bool = True,
    sqrt_weights: bool = True,
    residues_for_average: Optional[Union[str, Iterable[Tuple[str, int, str, str]]]] = None,
    salt_bridges: Optional[_SaltBridgesDict] = None,
) -> float:

    if salt_bridges is None:
        salt_bridges = detect_salt_bridges(pdb_path, sasa_path, pka_path, pH)
    if not salt_bridges:
        # Keep return type consistent (scalar) so JSON output stays stable.
        return 0.0

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
        sqrt_weights=sqrt_weights,
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

    # Build per-chain unique residue sets. Avoid depending on atom ordering:
    # different parsers/filters can yield different atom list orders, and any
    # order-dependence here would make cached sequences unstable.
    chain_to_key_set: Dict[str, Set[ResKey4]] = defaultdict(set)
    for atom in atoms:
        key = residue_key_from_atom(atom)
        chain_to_key_set[key[2]].add(key)

    out: Dict[str, List[str]] = {}
    for chain, key_set in chain_to_key_set.items():
        sorted_keys = sorted(key_set, key=lambda k: (k[1], k[3], k[0]))
        # Convert 3-letter residue names to 1-letter codes for stable motif counting.
        out[chain] = [THREE_TO_ONE.get(k[0], "X") for k in sorted_keys]

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
        # Residue name is stored in the first element of the key tuple (3-letter).
        # Convert to 1-letter codes so sequence length == number of residues.
        full_seq_parts.append("".join(THREE_TO_ONE.get(k[0], "X") for k in keys))
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
CONTACT_ORDER_CA_CUTOFF = 8.0

def calculate_weighted_contact_number_average(
    pdb_path: str,
    *,
    residue_category: Optional[Iterable[ResKey4]] = None,
    residues_for_density: Optional[Iterable[ResKey4]] = None,
    residues_for_average: Optional[Union[str, Iterable[ResKey4]]] = None,
    wcn_values: Optional[Dict[ResKey4, float]] = None,
) -> float:
    """
    WCN_i = Σ(j≠i) 1/(r_ij²), where r_ij is distance between per-residue Cα atoms.

    WCN is first computed for all residues in the structure, then:
    - ``residue_category``: limits which residues have WCN values kept at all
      (e.g. only CDR, only buried, only aromatic). Think "which keys exist in
      the weights dictionary".
    - ``residues_for_density``: optional subset of those keys that actually
      contribute to the numerator of the average. If ``None``, all keys in the
      (possibly category-filtered) weights dictionary are used.
    - ``residues_for_average``: optional set that defines the denominator
      (number of residues you are averaging over). If ``None`` or ``"all"``,
      the denominator is the total number of residues in the PDB. ``"no"``
      skips division (sum only); see :func:`average_over_residues`.

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
    eps_min_samples_by_category: Optional[Dict[str, Tuple[float, int]]] = None,
) -> Tuple[
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
    Dict[Tuple[str, int, str, str], int],
]:
    """
    Collect Cα coordinates per residue for five residue groups
    (negative, positive, aromatic, hydrophobic, polar), run DBSCAN on each group's
    3D coordinates, and return cluster labels per category.

    Default ``eps`` / ``min_samples`` follow ``DBSCAN_EPS_MIN_SAMPLES_BY_CATEGORY_DEFAULT``
    (charged/polar: wider eps; hydrophobic/aromatic: tighter eps and larger
    ``min_samples``). Pass ``eps_min_samples_by_category`` to override any category.
    """
    dbscan_params = {
        **DBSCAN_EPS_MIN_SAMPLES_BY_CATEGORY_DEFAULT,
        **(eps_min_samples_by_category or {}),
    }
    ca_by_res = ca_xyz_by_residue(atoms)

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
        group = _residue_category_group(res_name, pka_data.get(key), pH)
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
    
    def run_dbscan(
        keys: List,
        coords: List,
        category: str,
    ) -> Dict[Tuple[str, int, str, str], int]:
        if not coords:
            return {}
        eps, min_samples = dbscan_params[category]
        X = np.array(coords)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return dict(zip(keys, clustering.labels_.tolist()))

    neg_labels = run_dbscan(neg_keys, neg_coords, "negative")
    pos_labels = run_dbscan(pos_keys, pos_coords, "positive")
    # aromatic_labels = run_dbscan(aromatic_keys, aromatic_coords, "aromatic")
    hydro_labels = run_dbscan(hydro_keys, hydro_coords, "hydrophobic")
    # polar_labels = run_dbscan(polar_keys, polar_coords, "polar")
    
    return neg_labels, pos_labels, hydro_labels


def _dbscan_cluster_member_counts_and_side_abs_sums(
    labels: Dict[ResKey4, int],
    sasa_output_data: Dict[ResKey4, Dict[str, Optional[float]]],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    For non-noise DBSCAN labels, return (residue count per label, sum of
    ``total_side_abs`` per label).     Noise ``-1`` is excluded.
    """
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
    """
    For DBSCAN clusters (noise label ``-1`` excluded), return the **sum** of
    absolute side-chain SASA over **all** non-noise clusters: each cluster
    contributes the sum of ``total_side_abs`` for its member residues.

    Units: Å² (same as ``total_side_abs``). Returns ``0.0`` when there are no
    clusters or ``labels`` is empty.
    """
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
    """
    Shannon entropy of DBSCAN cluster **total side-chain absolute SASA** masses.

    For each non-noise cluster label ``i``, let ``s_i`` be the sum of
    ``total_side_abs`` over residues in that cluster, ``S = Σ_i s_i``, and
    ``p_i = s_i / S``. Returns ``H = -Σ_i p_i log(p_i)`` (natural logarithm,
    units **nats**). Noise ``-1`` is excluded; clusters with ``s_i = 0`` contribute
    no entropy mass.

    Returns ``0.0`` when there are no clusters, ``S <= 0``, or ``labels`` empty.
    High ``H`` → SASA mass spread across similar-sized (in abs SASA) clusters;
    low ``H`` → one or few clusters dominate ``S``.
    """
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


def _ann_mean_nn_distance(coords: np.ndarray) -> float:
    """
    Mean distance from each point to its nearest *other* point (k=2 in KD-tree query).

    Matches PROPERMAB ``AverageNearestNeighbor.compute_nn_mean_distance``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    if n < 2:
        return float("nan")
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return float(np.mean(dists[:, 1]))


def _ann_index_ratio(
    feature_coords: np.ndarray,
    allowed_coords: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> Optional[float]:
    """
    ann_index = d_o_bar / d_e_bar with null from random locations among allowed_coords.

    Matches PROPERMAB ``AverageNearestNeighbor``: ``rng.choice(n_allow, size=n_feat)``
    uses replacement (default), so the null can place multiple features at the same
    exposed Cα location.
    """
    feature_coords = np.asarray(feature_coords, dtype=np.float64)
    allowed_coords = np.asarray(allowed_coords, dtype=np.float64)
    n_feat = int(feature_coords.shape[0])
    n_allow = int(allowed_coords.shape[0])
    if n_feat < 2 or n_allow < 2:
        return None

    d_o = _ann_mean_nn_distance(feature_coords)
    if math.isnan(d_o) or d_o <= 0.0:
        return None

    d_e_null_sum = 0.0
    for _ in range(int(n_permutations)):
        pick = rng.choice(n_allow, size=n_feat, replace=True)
        sample = allowed_coords[pick]
        d_e_null_sum += _ann_mean_nn_distance(sample)

    d_e = (d_e_null_sum / float(int(n_permutations))) if int(n_permutations) > 0 else float("nan")
    if math.isnan(d_e) or d_e <= 0.0:
        return None
    return float(d_o / d_e)


def compute_surface_ann_index_descriptors(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[ResKey4, SASAEntry],
    sasa_cutoff: float = ANN_INDEX_SASA_CUTOFF_DEFAULT,
    n_permutations: int = ANN_INDEX_N_PERMUTATIONS_DEFAULT,
    random_seed: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Average Nearest Neighbor (ANN) index for positive, negative, and aromatic residues
    on the solvent-exposed surface (PROPERMAB-style).

    For each property, coordinates are Cα of exposed residues
    matching the type.
    The null expectation ``d_e`` is estimated by repeatedly sampling the same number
    of points uniformly at random from the set of reference positions of *all* exposed
    residues, **with replacement** (matching PROPERMAB's ``rng.choice`` default).

    Returns
    -------
    dict
        ``pos_ann_index``, ``neg_ann_index``, ``aromatic_ann_index`` — each is
        ``d_o_mean / d_e_mean`` or ``None`` if undefined (too few points, etc.).
    """
    result: Dict[str, Optional[float]] = {
        "pos_ann_index": None,
        "neg_ann_index": None,
        "aromatic_ann_index": None,
    }
    if not sasa_output_data or not pdb_atoms:
        return result

    ca_by_res = ca_xyz_by_residue(pdb_atoms)

    residue_exposure = get_exposed_residues(sasa_output_data, float(sasa_cutoff))
    exposed_with_ref: List[ResKey4] = [
        k for k, exposed in residue_exposure.items() if exposed and k in ca_by_res
    ]
    if len(exposed_with_ref) < 2:
        return result

    allowed_coords = np.array([ca_by_res[k] for k in exposed_with_ref], dtype=np.float64)
    rng = np.random.default_rng(random_seed)

    def _coords_for(residue_set: Set[str]) -> np.ndarray:
        pts: List[Tuple[float, float, float]] = []
        for k in exposed_with_ref:
            res_name = (k[0] or "").strip().upper()
            if res_name in residue_set:
                pts.append(ca_by_res[k])
        return np.array(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)

    pos_c = _coords_for(_ANN_INDEX_POSITIVE_RESIDUES)
    neg_c = _coords_for(_ANN_INDEX_NEGATIVE_RESIDUES)
    aro_c = _coords_for(_ANN_INDEX_AROMATIC_RESIDUES)

    if pos_c.shape[0] >= 2:
        result["pos_ann_index"] = _ann_index_ratio(
            pos_c, allowed_coords, n_permutations, rng
        )
    if neg_c.shape[0] >= 2:
        result["neg_ann_index"] = _ann_index_ratio(
            neg_c, allowed_coords, n_permutations, rng
        )
    if aro_c.shape[0] >= 2:
        result["aromatic_ann_index"] = _ann_index_ratio(
            aro_c, allowed_coords, n_permutations, rng
        )

    return result


def ripley_k_statistic(
    obs_coords,
    allowed_coords,
    distance: float = 6.0,
    n: int = 1000,
    random_seed: int = 0,
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

    k_o = len(cKDTree(obs_coords).query_pairs(r=distance)) / denominator

    rng = np.random.default_rng(random_seed)
    k_e_null = []
    allowed_size = allowed_coords.shape[0]
    replace = allowed_size < feature_size
    for _ in range(int(n)):
        indices = rng.choice(allowed_size, size=feature_size, replace=replace)
        new_coords = allowed_coords[indices]
        k_e_null.append(len(cKDTree(new_coords).query_pairs(r=distance)) / denominator)

    k_e = float(np.mean(k_e_null)) if k_e_null else float("nan")
    if k_e == 0.0 or math.isnan(k_e):
        return float("nan")
    return k_o / k_e

def get_pka_for_key(key: ResKey4, pka_output_data: Dict[ResKey4, float]) -> Optional[float]:
    return pka_output_data.get(key) or pka_output_data.get(
        (key[0], key[1], key[2], "")
    )

def compute_ripley(coords: List[Tuple[float, float, float]], allowed_coords: List[Tuple[float, float, float]], ripley_distance: float = RIPLEY_K_DISTANCE, ripley_n: int = RIPLEY_K_N_SAMPLES, random_seed: int = 0) -> Optional[float]:
        if len(coords) < 2:
            return 0.0
        value = ripley_k_statistic(coords, allowed_coords, distance=ripley_distance, n=ripley_n, random_seed=random_seed)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)

def compute_surface_ripley_descriptors(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[ResKey4, SASAEntry],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
    surface_exposed_threshold: float = SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    ripley_distance: float = RIPLEY_K_DISTANCE,
    ripley_n: int = RIPLEY_K_N_SAMPLES,
    random_seed: int = 0,
) -> Dict[str, Optional[float]]:
    """
    Ripley-style ratios for charge/hydrophobicity classes on the **full** exposed
    surface (relative side SASA above threshold, with Cα coordinates). The
    null resamples ``N`` locations uniformly from that same exposed reference set (same
    rule as ``ripley_k_statistic``).
    """

    result: Dict[str, Optional[float]] = {
        "ripley_k_negative": None,
        "ripley_k_positive": None,
        "ripley_k_aromatic": None,
        "ripley_k_hydrophobic": None,
        "ripley_k_polar": None,
    }
    if not sasa_output_data:
        return result

    ca_by_res = ca_xyz_by_residue(pdb_atoms)

    residue_exposure = get_exposed_residues(sasa_output_data, surface_exposed_threshold)
    exposed_keys: Set[ResKey4] = {
        key for key, is_exposed in residue_exposure.items() if is_exposed
    }

    exposed_keys_with_ref = sorted(
        [k for k in exposed_keys if k in ca_by_res],
        key=lambda k: (k[2], k[1], k[3], k[0]),
    )
    if len(exposed_keys_with_ref) < 2:
        result["ripley_k_negative"] = 0.0
        result["ripley_k_positive"] = 0.0
        result["ripley_k_aromatic"] = 0.0
        result["ripley_k_hydrophobic"] = 0.0
        result["ripley_k_polar"] = 0.0
        return result

    allowed_coords = np.array(
        [ca_by_res[k] for k in exposed_keys_with_ref],
        dtype=float,
    )

    neg_coords: List[Tuple[float, float, float]] = []
    pos_coords: List[Tuple[float, float, float]] = []
    aromatic_coords: List[Tuple[float, float, float]] = []
    hydro_coords: List[Tuple[float, float, float]] = []
    polar_coords: List[Tuple[float, float, float]] = []
    for key in exposed_keys_with_ref:
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

    result["ripley_k_negative"] = compute_ripley(neg_coords, allowed_coords, random_seed=random_seed)
    result["ripley_k_positive"] = compute_ripley(pos_coords, allowed_coords, random_seed=random_seed)
    result["ripley_k_aromatic"] = compute_ripley(aromatic_coords, allowed_coords, random_seed=random_seed)
    result["ripley_k_hydrophobic"] = compute_ripley(hydro_coords, allowed_coords, random_seed=random_seed)
    result["ripley_k_polar"] = compute_ripley(polar_coords, allowed_coords, random_seed=random_seed)

    return result


def _pcf_shell_numeric_token(x: float) -> str:
    """Stable token for one distance in Å (no unit suffix), e.g. ``3.0`` -> ``3``, ``3.5`` -> ``3p5``."""
    v = float(x)
    ir = round(v)
    if abs(v - ir) < 1e-9:
        return str(int(ir))
    return f"{v:g}".replace(".", "p")


def _pcf_shell_key_suffix(r_lo: float, width: float) -> str:
    """Metric key fragment for shell ``[r_lo, r_lo + width)``, e.g. ``(3, 2)`` -> ``3w2A``."""
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


def _empty_pcf_buried_aromatic_hydrophobic_scores(
    bin_starts: Tuple[float, ...],
    bin_widths: Tuple[float, ...],
) -> Dict[str, Optional[float]]:
    """PCF keys for buried aromatic / hydrophobic only (``buried`` infix disambiguates from exposed)."""
    if len(bin_starts) != len(bin_widths):
        raise ValueError("bin_starts and bin_widths must have the same length")
    tags = [
        _pcf_shell_key_suffix(float(r), float(w))
        for r, w in zip(bin_starts, bin_widths)
    ]
    out: Dict[str, Optional[float]] = {}
    for cat in ("aromatic", "hydrophobic"):
        for tag in tags:
            out[f"pcf_cluster_score_buried_{cat}_{tag}"] = None
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
    random_seed: int = 0,
) -> Dict[str, Optional[float]]:
    """
    Pair-correlation clustering scores for surface-exposed Cα atoms,
    by residue category (negative / positive / hydrophobic; polar currently disabled).
    Uncharged
    Phe/Tyr/Trp are typed as hydrophobic or polar (no separate aromatic channel;
    Ripley / DBSCAN surface metrics still use a five-way split elsewhere).

    Only residues that are **exposed** (relative side SASA above threshold) and
    have a Cα coordinate are considered — same scope as ``compute_surface_ripley_descriptors``.

    For each category, ρ = N_type / V_ref with ``V_ref`` the convex-hull volume
    of **all** exposed Cα positions in scope. For **each** distance shell ``[r, r + Δr)``
    (defaults: 1 Å bins ``[3,4)``, ``[4,5)``, ``[5,6)`` plus 2 Å bins ``[3,5)``, ``[5,7)``;
    see ``PCF_CLUSTER_BIN_STARTS_DEFAULT`` / ``PCF_CLUSTER_BIN_WIDTHS_DEFAULT``), the reported
    value is ``g_obs(r) / mean(g_null(r))`` under the same Ripley-style null as
    ``pair_correlation_clustering_score_random_surface_null`` (uniform draws of
    ``N`` sites from the exposed set per permutation; default
    ``n_permutations`` is ``PCF_CLUSTER_N_PERMUTATIONS_DEFAULT``, 3000). Keys look like
    ``pcf_cluster_score_negative_3w1A``, ``…_4w1A``, ``…_5w1A``, ``…_3w2A``, ``…_5w2A``.
    """
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
    # polar: List[Tuple[float, float, float]] = []

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
        # elif group == "polar":
        #     polar.append(xyz)

    shell_tags = [
        _pcf_shell_key_suffix(float(r), float(w))
        for r, w in zip(bin_starts, bin_widths)
    ]

    for cat, coords_list in (
        ("neg", neg),
        ("pos", pos),
        ("hyd", hydro),
        # ("polar", polar),
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


def compute_buried_pair_correlation_cluster_scores_aromatic_hydrophobic(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[ResKey4, SASAEntry],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
    surface_exposed_threshold: float = SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    bin_starts: Tuple[float, ...] = PCF_CLUSTER_BIN_STARTS_DEFAULT,
    bin_widths: Tuple[float, ...] = PCF_CLUSTER_BIN_WIDTHS_DEFAULT,
    n_permutations: int = PCF_CLUSTER_N_PERMUTATIONS_DEFAULT,
    random_seed: int = 0,
) -> Dict[str, Optional[float]]:
    """
    Pair-correlation clustering scores for **buried** Cα atoms (complement of exposed:
    ``total_side_rel <= surface_exposed_threshold``), for **aromatic** and **hydrophobic**
    only, using the same residue typing as :func:`compute_exposed_pair_correlation_cluster_scores`.

    Reference volume ``V_ref`` is the convex hull of **all** buried Cα positions in scope.
    For each category and distance shell, the statistic matches the exposed implementation:
    ``g_obs(r) / mean(g_null(r))`` with null draws uniform on the buried Cα support
    (same ``bin_starts``, ``bin_widths``, ``n_permutations`` defaults as exposed PCF).

    Keys use the same shell suffixes as exposed PCF (``3w1A``, ``4w1A``, ``5w1A``, ``3w2A``, ``5w2A``).
    """
    result = _empty_pcf_buried_aromatic_hydrophobic_scores(bin_starts, bin_widths)
    if not sasa_output_data:
        return result

    ca_by_res = ca_xyz_by_residue(pdb_atoms)

    residue_exposure = get_exposed_residues(
        sasa_output_data, float(surface_exposed_threshold)
    )
    buried_keys: Set[ResKey4] = {
        key for key, is_exposed in residue_exposure.items() if not is_exposed
    }

    buried_keys_with_ref: List[ResKey4] = sorted(
        [k for k in buried_keys if k in ca_by_res],
        key=lambda k: (k[2], k[1], k[3], k[0]),
    )
    if len(buried_keys_with_ref) < 2:
        for k in result:
            result[k] = 0.0
        return result

    allowed_coords = np.array(
        [ca_by_res[k] for k in buried_keys_with_ref],
        dtype=np.float64,
    )
    v_ref = convex_hull_volume_from_points(allowed_coords)
    if v_ref <= 0.0:
        for k in result:
            result[k] = 0.0
        return result

    rng = np.random.default_rng(random_seed)

    aromatic: List[Tuple[float, float, float]] = []
    hydro: List[Tuple[float, float, float]] = []

    for key in buried_keys_with_ref:
        xyz = ca_by_res[key]
        res_name = key[0]
        pka_val = get_pka_for_key(key, pka_output_data)
        group = _residue_category_group(res_name, pka_val, pH)
        if group == "aromatic":
            aromatic.append(xyz)
        elif group == "hydrophobic":
            hydro.append(xyz)

    shell_tags = [
        _pcf_shell_key_suffix(float(r), float(w))
        for r, w in zip(bin_starts, bin_widths)
    ]

    for cat, coords_list in (
        ("aromatic", aromatic),
        ("hydrophobic", hydro),
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
            result[f"pcf_cluster_score_buried_{cat}_{tag}"] = val

    return result


def compute_surface_pair_descriptors(
    pdb_atoms: List[Atom],
    sasa_output_data: Dict[ResKey4, SASAEntry],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
    surface_exposed_threshold: float = SURFACE_EXPOSED_THRESHOLD_DEFAULT,
    pair_radius: float = PSH_PAIR_RADIUS,
    cdr_vicinity_radius: float = CDR_VICINITY_RADIUS,
    salt_bridge_residues: Optional[Set[ResKey4]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute PSH/PPC/PNC surface spatial descriptors (all surface and CDR vicinity)

    If `salt_bridge_residues` is provided, any residue in that set has its charge
    flag forced to zero for PPC/PNC (i.e., residues participating in salt bridges
    are excluded from same-charge clustering terms).
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

    ca_by_res = ca_xyz_by_residue(pdb_atoms)
    atoms_by_res: Dict[ResKey4, List[Atom]] = defaultdict(list)
    for atom in pdb_atoms:
        key = residue_key_from_atom(atom)
        atoms_by_res[key].append(atom)

    # Use shared exposure helper instead of duplicating SASA threshold logic
    residue_exposure = get_exposed_residues(sasa_output_data, surface_exposed_threshold)
    exposed_keys: Set[ResKey4] = {
        key for key, is_exposed in residue_exposure.items() if is_exposed
    }

    exposed_keys_with_ref = [k for k in exposed_keys if k in ca_by_res]
    if len(exposed_keys_with_ref) < 2:
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

    # Shared heavy-atom KDTree; pair (i, j) uses minimum heavy-atom distance.
    heavy_tree, heavy_coords, heavy_atom_res_keys, heavy_atoms_by_res = get_heavy_atom_tree(pdb_atoms)

    exposed_res_to_idx: Dict[ResKey4, int] = {key: idx for idx, key in enumerate(exposed_residue_keys)}
    exposed_res_set: Set[ResKey4] = set(exposed_residue_keys)

    # Verify that exposed residues have heavy atoms in the tree; bail if none do.
    if not any(heavy_atoms_by_res.get(k) for k in exposed_residue_keys):
        result["psh_all_surface"] = 0.0
        result["psh_cdr_vicinity"] = 0.0
        result["ppc_all_surface"] = 0.0
        result["ppc_cdr_vicinity"] = 0.0
        result["pnc_all_surface"] = 0.0
        result["pnc_cdr_vicinity"] = 0.0
        return result

    pair_min_dist: Dict[Tuple[int, int], float] = {}
    for idx, key in enumerate(exposed_residue_keys):
        atom_indices_i = heavy_atoms_by_res.get(key, [])
        if not atom_indices_i:
            continue
        neighbours_list = heavy_tree.query_ball_point(heavy_coords[atom_indices_i], pair_radius)
        for atom_i, neighbour_atom_idxs in zip(atom_indices_i, neighbours_list):
            for atom_j in neighbour_atom_idxs:
                nb_key = heavy_atom_res_keys[atom_j]
                if nb_key == key or nb_key not in exposed_res_set:
                    continue
                nb_local = exposed_res_to_idx[nb_key]
                if idx >= nb_local:
                    continue  # process each pair once (smaller idx first)
                pair = (idx, nb_local)
                dist_ij = float(np.linalg.norm(heavy_coords[atom_i] - heavy_coords[atom_j]))
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
        if salt_bridge_residues is not None and key in salt_bridge_residues:
            continue
        q = _residue_fractional_charge_at_pH(key[0], get_pka_for_key(key, pka_output_data), pH)
        if q > 0.0:
            pos_charge[idx] = q
        elif q < 0.0:
            neg_charge[idx] = q

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
        if neg_charge[ri] < 0.0 and neg_charge[rj] < 0.0:
            pnc_all_surface_acc += (neg_charge[ri] * neg_charge[rj]) * inv_r2

    result["psh_all_surface"] = psh_all_surface_acc
    result["ppc_all_surface"] = ppc_all_surface_acc
    result["pnc_all_surface"] = pnc_all_surface_acc

    seed_indices = {
        idx for idx, key in enumerate(exposed_residue_keys)
        if get_residue_region(key[1]) == "CDR"
    }
    seed_points_list: List[List[float]] = []
    for idx, key in enumerate(exposed_residue_keys):
        if idx not in seed_indices:
            continue
        for atom in atoms_by_res[key]:
            if is_hydrogen_atom(atom):
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
                if is_hydrogen_atom(atom):
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
                if neg_charge[ri] < 0.0 and neg_charge[rj] < 0.0:
                    pnc_cdr_acc += (neg_charge[ri] * neg_charge[rj]) * inv_r2
        result["psh_cdr_vicinity"] = psh_cdr_acc
        result["ppc_cdr_vicinity"] = ppc_cdr_acc
        result["pnc_cdr_vicinity"] = pnc_cdr_acc
    else:
        result["psh_cdr_vicinity"] = 0.0
        result["ppc_cdr_vicinity"] = 0.0
        result["pnc_cdr_vicinity"] = 0.0

    return result


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

    try:
        complex_total = parse_sasa(str(path.resolve())).total_sasa
        h_total = parse_sasa(sasa_H_path).total_sasa
        l_total = parse_sasa(sasa_L_path).total_sasa
    except (FileNotFoundError, ValueError, OSError):
        return None
    return h_total + l_total - complex_total

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
            donor_is_backbone.append(is_backbone_atom(atom.name))
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
        # Distance filter: all candidates come from KD-tree (same for backbone and sidechain)
        indices = acceptor_tree.query_ball_point(donor_coord, MAX_HBOND_DISTANCE)
        if not indices:
            continue

        is_backbone_donor = donor_is_backbone[donor_idx]
        donor_max = donor_max_list[donor_idx]

        # Resolve base atom and base->donor vector for angle check (same for backbone and sidechain)
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

        # Valid candidates: within distance (KD-tree), acceptor not full, not same residue, backbone separation if applicable
        valid_indices = []
        valid_acceptors = []
        for idx in indices:
            acceptor = acceptors[idx]
            if acceptor_hbond_counts.get(acceptor, 0) >= acceptor_max_list[idx]:
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
            acc_max = acceptor_max_list[valid_indices[local_idx]]
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
    residues_for_average: Optional[Union[str, Iterable[Tuple[str, int, str, str]]]] = None,
    sqrt_weights: bool = True,
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
        sqrt_weights=sqrt_weights,
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

# If DSSP parsing succeeds but essentially no H-bond records align to the structure,
# treat the descriptor as missing (NaN) rather than a true zero.
DSSP_HBOND_MIN_ALIGNED_RESIDUE_FRACTION = 0.01


def compute_dssp_hbond_energy_density_raw(
    pdb_path: str,
    sasa_path: str,
    dssp_path: str,
) -> _DsspHbondEnergyDensityRaw:

    ctx = StructureContext(pdb_path, sasa_path=sasa_path, dssp_path=dssp_path)
    atoms = ctx.atoms
    sasa_data = ctx.sasa_residue
    dssp_hbonds, dssp_seq_to_pdb = ctx.dssp_hbonds

    def _canonical_residue_keys() -> List[Tuple[str, int, str, str]]:
        """
        Deterministic ordering for per-residue payloads.

        Keys are `(resname, resnum, chain, insertion)` and are sorted by
        `(chain, resnum, insertion, resname)`.
        """
        try:
            keys = ctx.residue_keys
        except Exception:
            keys = set(iter_unique_residues(atoms)) if atoms else set()
        return sorted(keys, key=lambda k: (k[2], k[1], k[3], k[0]))

    # If DSSP H-bond parsing was requested but failed, treat as missing (NaN) rather than 0.0.
    if dssp_path and (
        "dssp_hbonds" in ctx.parse_errors
        or "dssp" in ctx.parse_errors
    ):
        # Return non-empty NaN payload so downstream averaging yields NaN.
        res_keys = _canonical_residue_keys()
        return ({k: float("nan") for k in res_keys}, {k: 0 for k in res_keys})

    # If SASA was requested but failed, the energy density cannot be computed reliably.
    if sasa_path and ("sasa" in ctx.parse_errors or not sasa_data):
        res_keys = _canonical_residue_keys()
        return ({k: float("nan") for k in res_keys}, {k: 0 for k in res_keys})

    # DSSP was requested and parsed (no parse_errors), but produced no aligned records
    # (e.g., chain-ID mismatch, numbering mismatch, or empty DSSP output).
    if dssp_path and ("dssp_hbonds" not in ctx.parse_errors and "dssp" not in ctx.parse_errors):
        if not dssp_hbonds or not dssp_seq_to_pdb:
            res_keys = _canonical_residue_keys()
            return ({k: float("nan") for k in res_keys}, {k: 0 for k in res_keys})

        # Coverage gate: if too few structure residues have any DSSP H-bond record,
        # treat as missing data (NaN) rather than a near-zero estimate.
        try:
            n_struct = len(ctx.residue_keys)
        except Exception:
            n_struct = 0
        n_aligned = len(dssp_hbonds)
        if n_struct > 0 and (n_aligned / n_struct) < DSSP_HBOND_MIN_ALIGNED_RESIDUE_FRACTION:
            res_keys = _canonical_residue_keys()
            return ({k: float("nan") for k in res_keys}, {k: 0 for k in res_keys})

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
    residues_for_average: Optional[Union[str, Iterable[Tuple[str, int, str, str]]]] = None,
    sqrt_weights: bool = True,
    dssp_hbond_energy_density_raw: Optional[_DsspHbondEnergyDensityRaw] = None,
) -> float:

    if dssp_hbond_energy_density_raw is None:
        dssp_hbond_energy_density_raw = compute_dssp_hbond_energy_density_raw(
            pdb_path, sasa_path, dssp_path
        )
    weights_raw, counts = dssp_hbond_energy_density_raw
    if not weights_raw:
        return float("nan")

    # Denominator policy for DSSP+SASA energy density:
    # average over *covered* residues only (those with a non-missing weight in weights_raw).
    # This avoids silently treating missing coverage as true zeros.
    if residues_for_average is None:
        if residues_for_density is None:
            residues_for_average = list(weights_raw.keys())
        else:
            residues_for_average = [k for k in residues_for_density if k in weights_raw]
    elif isinstance(residues_for_average, str):
        if residues_for_average not in ("all", "no"):
            raise ValueError(
                f"residues_for_average must be None, 'all', 'no', or an iterable; "
                f"got {residues_for_average!r}"
            )
    elif not list(residues_for_average):
        return float("nan")

    return average_over_residues(
        weights_raw=weights_raw,
        counts=counts,
        residues_for_density=residues_for_density,
        residues_for_average=residues_for_average,
        denom_total_residues=_count_residues_in_pdb(pdb_path),
        weighted=weighted,
        sqrt_weights=sqrt_weights,
    )


def calculate_hbond_energy_dssp_backbone_only_unweighted_average(
    pdb_path: str,
    dssp_path: str,
    *,
    residues_for_density: Optional[Iterable[Tuple[str, int, str, str]]] = None,
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
) -> float:
    """
    DSSP-only fallback metric (no SASA required): average absolute DSSP H-bond energy
    per residue, averaged over DSSP-indexed (covered) residues.

    This preserves a DSSP signal even when SASA is missing, while keeping SASA-weighted
    "density" metrics as NaN on SASA failure.
    """
    ctx = StructureContext(pdb_path, dssp_path=dssp_path)
    dssp_hbonds, dssp_seq_to_pdb = ctx.dssp_hbonds

    # If DSSP parsing was requested but failed, treat as missing (NaN).
    if dssp_path and ("dssp_hbonds" in ctx.parse_errors or "dssp" in ctx.parse_errors):
        return float("nan")

    # Coverage set: residues with a DSSP index aligned to the structure.
    dssp_indexed: Set[Tuple[str, int, str, str]] = set(dssp_seq_to_pdb.values()) if dssp_seq_to_pdb else set()
    if not dssp_hbonds or not dssp_indexed:
        return float("nan")

    # Gate extremely low alignment coverage (often numbering/chain mismatches).
    try:
        n_struct = len(ctx.residue_keys)
    except Exception:
        n_struct = 0
    if n_struct > 0 and (len(dssp_indexed) / n_struct) < DSSP_HBOND_MIN_ALIGNED_RESIDUE_FRACTION:
        return float("nan")

    # Aggregate absolute energy to both endpoints of each H-bond pair.
    pdb_to_dssp_seq = {pdb_key: dssp_seq for dssp_seq, pdb_key in dssp_seq_to_pdb.items()}
    energy_by_res: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    for res_key, pairs in dssp_hbonds.items():
        dssp_seq = pdb_to_dssp_seq.get(res_key)
        if dssp_seq is None:
            continue
        for offset, energy in pairs:
            if energy == 0.0:
                continue
            target_seq = dssp_seq + offset
            target_key = dssp_seq_to_pdb.get(target_seq)
            if target_key is None:
                continue
            e = abs(float(energy))
            energy_by_res[res_key] += e
            energy_by_res[target_key] += e

    # Denominator policy: average over *covered* residues only (DSSP-indexed),
    # optionally restricted to a subset.
    if residues_for_average is None:
        if residues_for_density is None:
            denom_set = dssp_indexed
        else:
            denom_set = set(residues_for_density) & dssp_indexed
    else:
        denom_set = set(residues_for_average) & dssp_indexed
    if not denom_set:
        return float("nan")

    if residues_for_density is None:
        numer_keys = denom_set
    else:
        numer_keys = set(residues_for_density) & denom_set
    total = sum(energy_by_res.get(k, 0.0) for k in numer_keys) # WRONG FORMULA REDO
    return total / float(len(denom_set))


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

