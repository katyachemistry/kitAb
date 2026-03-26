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
    get_residue_region,
    get_residue_region_map,
    iter_unique_residues,
    _get_atoms_for_path,
    get_residue_keys_by_type,
    normalize_hydropathy,
    _residue_fractional_charge_at_pH,
    is_residue_charged,
    is_hydrogen_atom,
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
from utils.chemistry import get_standard_residue_pka
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

THREE_TO_ONE = {v: k for k, v in AA_1_TO_3.items()}

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
    # Phenolic / thiol deprotonation.
    ("OH", "TYR"),
    ("SG", "CYS"),
})


# Residue sets used by the notebook
GLN_ASN_RESIDUES = frozenset({"GLN", "ASN"})

# Metrics for which we compute median / beta_sheet_median / buried_median / exposed_median
# METRICS = ["hbond_density", "salt_bridge_density", "wcn", "hbond_energy_dssp_density"]

KYTE_DOOLITTLE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}

CLUSTER_LABEL_COLS = [
    "negative_cluster_labels",
    "positive_cluster_labels",
    "aromatic_cluster_labels",
    "hydrophobic_cluster_labels",
    "polar_cluster_labels",
]

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


def _residue_fractional_charge_at_pH_for_charge_metrics(
    residue_name: str,
    pka_value: Optional[float],
    pH: float,
) -> float:
    """
    Fractional charge for net charge / pI and charge-weighting metrics.

    Negative residues (ASP/GLU plus TYR/CYS) are treated as titratable acidic
    groups: phenolic/thiol deprotonation -> -1 when deprotonated.
    """
    res_name = (residue_name or "").strip().upper()
    effective_pka = pka_value if pka_value is not None else get_standard_residue_pka(res_name)
    if effective_pka is None:
        return 0.0

    if res_name in NEGATIVE_CHARGED_RESIDUES:
        return -1.0 / (1.0 + np.power(10.0, effective_pka - pH))
    if res_name in POSITIVE_CHARGED_RESIDUES:
        return 1.0 / (1.0 + np.power(10.0, pH - effective_pka))
    return 0.0


def net_charge_from_pka(
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float
) -> Optional[float]:

    if not pka_data:
        return None
    net = 0.0
    for key, pka in pka_data.items():
        res_name = key[0]
        net += _residue_fractional_charge_at_pH_for_charge_metrics(res_name, pka, pH)

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


# ---------------------------------------------------------------------------
# Alternative sequence-based net charge (no structure / PropKa dependency)
# ---------------------------------------------------------------------------

# Copy of PROPERMAB-style simplified side-chain charge logic:
# - uses fixed pKa values only as a threshold for histidine protonation
# - returns integer net charge based on residue counts
_SEQUTILS_PKA_DICT = {
    "N_term": (8.6, 1),
    "C_term": (3.6, -1),
    "D": (3.9, -1),
    "E": (4.1, -1),
    "H": (6.5, 1),
    "Y": (10.1, -1),
    "K": (10.8, 1),
    "R": (12.5, 1),
}


# def net_charge_from_seq(seq: str, pH: float = 7.4) -> float:
#     """
#     Sequence-based net charge used for quick comparisons.

#     Note: this matches PROPERMAB's `calculate_seq_charge()` behavior:
#     it models His as titratable via a pH threshold and counts only
#     side-chain contributions (no explicit N/C termini terms).
#     """
#     if pH < _SEQUTILS_PKA_DICT["H"][0]:
#         return (
#             seq.count("H") + seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")
#         )
#     return seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")

def compute_dipole_moment_magnitude(
    pdb_atoms: List[Atom],
    pka_output_data: Dict[ResKey4, float],
    pH: float,
) -> Optional[float]:
    """
    Dipole moment magnitude from ionizable atom positions and pH-dependent fractional charges.

    We approximate the dipole as
      μ = Σ q_i * (r_i - r0),
    where r_i is the centroid of ionizable atoms for residue i, and r0 is a single
    reference centroid shared across all residues. Subtracting r0 makes the result
    invariant to global translation of the input coordinates.
    """
    # Ionizable atoms per residue for dipole estimation.
    # Constructed from the same POSITIVE_ATOMS / NEGATIVE_ATOMS definitions used
    # elsewhere in this module for charge handling (including Tyr/Cys).
    ionizable_atom_names_by_residue_lists: Dict[str, List[str]] = {}
    for atom_name, res_name in POSITIVE_ATOMS:
        ionizable_atom_names_by_residue_lists.setdefault(res_name, []).append(atom_name)
    for atom_name, res_name in NEGATIVE_ATOMS:
        ionizable_atom_names_by_residue_lists.setdefault(res_name, []).append(atom_name)
    ionizable_atom_names_by_residue: Dict[str, Tuple[str, ...]] = {
        res_name: tuple(sorted(atom_names))
        for res_name, atom_names in ionizable_atom_names_by_residue_lists.items()
    }

    # Collect per-residue representative coords as the centroid of ionizable atoms.
    coords_by_res: Dict[ResKey4, List[Tuple[float, float, float]]] = {}
    for atom in pdb_atoms:
        atom_name = (atom.name or "").strip().upper()
        res_name = (atom.residue_name or "").strip().upper()
        allowed = ionizable_atom_names_by_residue.get(res_name)
        if not allowed:
            continue
        if atom_name not in allowed:
            continue
        key = residue_key_from_atom(atom)
        coords_by_res.setdefault(key, []).append((atom.x, atom.y, atom.z))

    def get_pka(k: ResKey4) -> Optional[float]:
        return pka_output_data.get(k) or pka_output_data.get((k[0], k[1], k[2], ""))

    # Reference centroid for translation invariance.
    # Use the centroid of all atoms we were given (not just ionizable ones),
    # matching the approach in `struct_featurizer.py`.
    if not pdb_atoms:
        return 0.0
    ref_center = np.mean(
        np.array([(a.x, a.y, a.z) for a in pdb_atoms], dtype=np.float64),
        axis=0,
    )

    mux, muy, muz = 0.0, 0.0, 0.0
    for key, coords in coords_by_res.items():
        if not coords:
            continue

        res_name = key[0]
        pka_val = get_pka(key)
        q = float(_residue_fractional_charge_at_pH_for_charge_metrics(res_name, pka_val, pH))
        if q == 0.0:
            continue

        # Centroid of ionizable atom positions for stable, non-double-counted dipole vector.
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        x = float(np.mean(xs))
        y = float(np.mean(ys))
        z = float(np.mean(zs))

        dx = x - float(ref_center[0])
        dy = y - float(ref_center[1])
        dz = z - float(ref_center[2])
        mux += q * dx
        muy += q * dy
        muz += q * dz

    return math.sqrt(mux * mux + muy * muy + muz * muz)


# def compute_hydrophobic_moment_magnitude(
#     pdb_atoms: List[Atom],
#     *,
#     normalize_by_n: bool = False,
# ) -> float:
#     """
#     Magnitude of the first-order hydrophobic moment (amphiphilicity), analogous to
#     a dipole but with Kyte–Doolittle hydrophobicity instead of charge.

#     For each residue *i*, let *h_i* be the Kyte–Doolittle value and *r_i* the center
#     of geometry (mean position of all atoms in that residue). The moment vector is

#         **H** = Σ_i h_i * r_i

#     Some references include a factor 1/N with N the number of residues; set
#     ``normalize_by_n=True`` to use **H** = (1/N) Σ_i h_i * r_i.

#     This matches PROPERMAB ``StructFeaturizer.hyd_moment`` when
#     ``normalize_by_n=False`` (unnormalized sum, then Euclidean norm).

#     Parameters
#     ----------
#     pdb_atoms
#         Structure atoms (typically parsed PDB).
#     normalize_by_n
#         If True, scale the summed vector by 1/N before taking the norm (N = count of
#         residues that contribute: standard amino acids present in ``KYTE_DOOLITTLE``).

#     Returns
#     -------
#     float
#         ‖**H**‖ (or ‖**H**/N‖ when ``normalize_by_n`` is True). 0.0 if no residues
#         contribute.
#     """
#     if not pdb_atoms:
#         return 0.0

#     coords_by_res: Dict[ResKey4, List[Tuple[float, float, float]]] = defaultdict(list)
#     for atom in pdb_atoms:
#         key = residue_key_from_atom(atom)
#         coords_by_res[key].append((atom.x, atom.y, atom.z))

#     hx, hy, hz = 0.0, 0.0, 0.0
#     n_used = 0
#     for key, coords in coords_by_res.items():
#         if not coords:
#             continue
#         res_name = (key[0] or "").strip().upper()
#         h_i = KYTE_DOOLITTLE.get(res_name)
#         if h_i is None:
#             continue
#         arr = np.asarray(coords, dtype=np.float64)
#         r_i = np.mean(arr, axis=0)
#         hx += float(h_i) * float(r_i[0])
#         hy += float(h_i) * float(r_i[1])
#         hz += float(h_i) * float(r_i[2])
#         n_used += 1

#     if n_used == 0:
#         return 0.0
#     if normalize_by_n:
#         inv_n = 1.0 / float(n_used)
#         hx *= inv_n
#         hy *= inv_n
#         hz *= inv_n
#     return float(math.sqrt(hx * hx + hy * hy + hz * hz))


# Spatial Charge Map with optional weighting by SASA (default: True)

def scm_score_from_pka(
    pdb_path: str,
    sasa_path: str,
    pka_data: Dict[Tuple[str, int, str, str], float],
    pH: float,
    d_cutoff: float = 10.0,
    sasa_cutoff: float = 0.05,
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
        for a, key4 in zip(atoms, key4_list):
            if key4 not in residue_charge:
                residue_charge[key4] = _residue_fractional_charge_at_pH(
                    a.residue_name, pka_data.get(key4), pH
                )
            # Distribute residue charge only onto ionizable atoms.
            if (
                (a.name, a.residue_name) in POSITIVE_ATOMS
                or (a.name, a.residue_name) in NEGATIVE_ATOMS
            ):
                residue_charged_atom_count[key4] = (
                    residue_charged_atom_count.get(key4, 0) + 1
                )
        
        atom_charge = np.zeros(n, dtype=np.float64)
        atom_sasa_share = np.zeros(n, dtype=np.float64)
        for i, (a, key4) in enumerate(zip(atoms, key4_list)):
            # Only ionizable atoms get any per-atom charge/SASA share.
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


# PROPERMAB paper / Agrawal et al.: side-chain absolute SASA threshold (Å²).
# SCM_PROPERMAB_SIDE_CHAIN_ABS_SASA_CUTOFF_A2 = 10.0


# def scm_score_from_pka_propermab(
#     pdb_path: str,
#     sasa_path: str,
#     pka_data: Dict[Tuple[str, int, str, str], float],
#     pH: float,
#     d_cutoff: float = 10.0,
#     side_chain_abs_sasa_cutoff: float = SCM_PROPERMAB_SIDE_CHAIN_ABS_SASA_CUTOFF_A2,
#     chain_ids: Optional[Iterable[str]] = None,
# ) -> Optional[float]:
#     """
#     Spatial charge map (SCM) score as in the PROPERMAB paper (Agrawal et al.).

#     For each atom *i* in the Fv domain (see ``chain_ids``):

#         scm_i = sum_{j in E_i} q_j

#     where *E_i* is the set of **side-chain** atoms *j* that belong to residues whose
#     **side-chain solvent-accessible surface area is >** ``side_chain_abs_sasa_cutoff``
#     (default **10 Å²**), and whose distance to atom *i* is **≤** ``d_cutoff``
#     (default **10 Å**).

#     The domain score is::

#         scm = | sum_{i in A} scm_i * H(-scm_i) |

#     i.e. the absolute value of the sum of *scm_i* over atoms with *scm_i* < 0
#     (Heaviside *H*).

#     **Charges *q_j* (pKa-based):** residue fractional charge at *pH* is
#     ``_residue_fractional_charge_at_pH`` (same helper as elsewhere in this module).
#     That net charge is spread **evenly** over all **heavy** side-chain atoms of the
#     residue (non-backbone, non-hydrogen). This substitutes for force-field partial
#     charges used in the original formulation.

#     Parameters
#     ----------
#     chain_ids
#         If given, restrict *A* (and all bookkeeping) to these PDB chain IDs (e.g.
#         ``("H", "L")`` for Fv). If ``None``, all atoms from the parsed structure are
#         used.
#     """
#     from utils.parsers import parse_sasa

#     atoms = _get_atoms_for_path(pdb_path)
#     if not atoms:
#         return None

#     if chain_ids is not None:
#         allowed = frozenset(str(c) for c in chain_ids)
#         atoms = [a for a in atoms if (a.chain or "") in allowed]
#     if not atoms:
#         return None

#     try:
#         sasa_data = parse_sasa(sasa_path)
#     except Exception as e:
#         logger.warning("SCM (PROPERMAB): failed to parse SASA %s: %s", sasa_path, e)
#         return None

#     def _sasa_entry(key: ResKey4):
#         e = sasa_data.get(key)
#         if e is None:
#             e = sasa_data.get((key[0], key[1], key[2], ""))
#         return e

#     def _pka_val(key: ResKey4) -> Optional[float]:
#         v = pka_data.get(key)
#         if v is None:
#             v = pka_data.get((key[0], key[1], key[2], ""))
#         return v

#     try:
#         unique_keys: Set[ResKey4] = {residue_key_from_atom(a) for a in atoms}

#         residue_exposed_sc: Dict[ResKey4, bool] = {}
#         for key in unique_keys:
#             entry = _sasa_entry(key)
#             if entry is None:
#                 residue_exposed_sc[key] = False
#                 continue
#             abs_sa = getattr(entry, "total_side_abs", None)
#             if abs_sa is None:
#                 residue_exposed_sc[key] = False
#             else:
#                 try:
#                     residue_exposed_sc[key] = float(abs_sa) > float(
#                         side_chain_abs_sasa_cutoff
#                     )
#                 except (TypeError, ValueError):
#                     residue_exposed_sc[key] = False

#         sc_heavy_count: Dict[ResKey4, int] = defaultdict(int)
#         for a in atoms:
#             if is_hydrogen_atom(a) or is_backbone_atom(a):
#                 continue
#             sc_heavy_count[residue_key_from_atom(a)] += 1

#         q_residue: Dict[ResKey4, float] = {}
#         for key in unique_keys:
#             q_residue[key] = float(
#                 _residue_fractional_charge_at_pH(key[0], _pka_val(key), pH)
#             )

#         n = len(atoms)
#         coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)

#         source_coords_list: List[Tuple[float, float, float]] = []
#         source_q_list: List[float] = []
#         for a in atoms:
#             if is_hydrogen_atom(a) or is_backbone_atom(a):
#                 continue
#             rk = residue_key_from_atom(a)
#             if not residue_exposed_sc.get(rk, False):
#                 continue
#             n_sc = sc_heavy_count.get(rk, 0)
#             if n_sc <= 0:
#                 continue
#             qj = q_residue.get(rk, 0.0) / float(n_sc)
#             source_coords_list.append((a.x, a.y, a.z))
#             source_q_list.append(qj)

#         if not source_coords_list:
#             return 0.0

#         source_coords = np.asarray(source_coords_list, dtype=np.float64)
#         source_q = np.asarray(source_q_list, dtype=np.float64)
#         tree = cKDTree(source_coords)

#         scm_atom = np.zeros(n, dtype=np.float64)
#         for i in range(n):
#             idxs = tree.query_ball_point(coords[i], r=float(d_cutoff))
#             if not idxs:
#                 continue
#             scm_atom[i] = float(np.sum(source_q[list(idxs)]))

#         neg_sum = float(np.sum(scm_atom[scm_atom < 0.0]))
#         return float(abs(neg_sum))
#     except Exception as e:
#         logger.warning("SCM (PROPERMAB) computation failed: %s", e, exc_info=True)
#         return None

# SAP analog
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
                q = float(_residue_fractional_charge_at_pH_for_charge_metrics(res_name, pka, pH))
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

    # If SASA was requested but failed to parse / yielded no usable records, propagate
    # failure as NaN rather than returning a "valid" zero density.
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
    weighted: bool = True,
    sqrt_weights: bool = True,
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
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
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

# def calculate_relative_contact_order(
#     pdb_path: str,
#     *,
#     ca_cutoff: float = CONTACT_ORDER_CA_CUTOFF,
#     min_sequence_separation: int = 0,
#     include_inter_chain: bool = False,
#     chain_order: Optional[List[str]] = None,
# ) -> Optional[float]:
#     """
#     Relative contact order (CO):

#         CO = (1 / (N * L)) * Σ_{(i,j) in contacts} |i - j|

#     where:
#     - N is the number of (unique) residue-residue contacts
#     - |i-j| is the sequence separation in residues
#     - L is the total number of residues considered

#     Contacts are defined by Cα–Cα distance <= ``ca_cutoff``. By default, only
#     intra-chain contacts contribute (``include_inter_chain=False``)
#     """
#     ctx = _get_structure_context(pdb_path)
#     ca_coords_dict = ctx.ca_coords
#     if not ca_coords_dict:
#         return None

#     chain_to_keys: Dict[str, List[ResKey4]] = defaultdict(list)
#     for k in ca_coords_dict.keys():
#         chain_to_keys[k[2]].append(k)
#     for chain, keys in list(chain_to_keys.items()):
#         chain_to_keys[chain] = sorted(keys, key=lambda x: (x[1], x[3]))

#     chains = chain_order or sorted(chain_to_keys.keys())
#     chains = [c for c in chains if c in chain_to_keys]
#     if not chains:
#         return None

#     per_chain_index: Dict[ResKey4, int] = {}
#     global_index: Dict[ResKey4, int] = {}
#     global_keys: List[ResKey4] = []
#     offset = 0
#     for chain in chains:
#         keys = chain_to_keys.get(chain) or []
#         for local_i, key in enumerate(keys):
#             per_chain_index[key] = local_i
#             global_index[key] = offset + local_i
#             global_keys.append(key)
#         offset += len(keys)

#     L = len(global_keys)
#     if L < 2:
#         return None

#     coords = np.array([ca_coords_dict[k] for k in global_keys], dtype=np.float64)
#     tree = cKDTree(coords)

#     total_sep = 0.0
#     n_contacts = 0
#     for i, j in tree.query_pairs(r=ca_cutoff):
#         ki = global_keys[i]
#         kj = global_keys[j]
#         if not include_inter_chain and ki[2] != kj[2]:
#             continue
#         if include_inter_chain:
#             sep = abs(global_index[ki] - global_index[kj])
#         else:
#             sep = abs(per_chain_index[ki] - per_chain_index[kj])
#         if sep < int(min_sequence_separation):
#             continue
#         total_sep += float(sep)
#         n_contacts += 1

#     if n_contacts == 0:
#         return None
#     return float(total_sep) / float(n_contacts * L)

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
    n_cluster_members = int(sum(cluster_sizes.values()))
    avg_rel_asa = float(total_rel_asa) / float(n_cluster_members) if n_cluster_members > 0 else 0.0
    return largest_cluster_size, n_clusters, avg_rel_asa


# `parse_sasa()` stores `*_rel` fields as fractions in [0, 1] (it divides by 100).
# Defaults here should therefore use fraction-scale thresholds.
SURFACE_EXPOSED_THRESHOLD_DEFAULT = 0.08
RIPLEY_K_DISTANCE = 8.0
RIPLEY_K_N_SAMPLES = 1000
ANN_INDEX_N_PERMUTATIONS_DEFAULT = 1000
# PROPERMAB `StructFeaturizer.ann_index` uses relative side-chain RSA >= 0.05;
# our SASA parser stores relative values as fractions in [0, 1].
ANN_INDEX_SASA_CUTOFF_DEFAULT = 0.05
PSH_PAIR_RADIUS = 7.5
CDR_VICINITY_RADIUS = 4.0

# Residue sets for surface ANN index (match PROPERMAB `ann_index` prop filters).
_ANN_INDEX_POSITIVE_RESIDUES = frozenset({"ARG", "LYS", "HIS"})
_ANN_INDEX_NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})
_ANN_INDEX_AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP"})


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
    nn_distances: List[float] = []
    for point in coords:
        dists, _ = tree.query(point, k=2)
        nn_distances.append(float(dists[1]))
    return float(np.mean(nn_distances))


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

    d_e_null: List[float] = []
    for _ in range(int(n_permutations)):
        pick = rng.choice(n_allow, size=n_feat, replace=True)
        sample = allowed_coords[pick]
        d_e_null.append(_ann_mean_nn_distance(sample))

    d_e = float(np.mean(d_e_null)) if d_e_null else float("nan")
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

    For each property, coordinates are Cα of exposed residues matching the type.
    The null expectation ``d_e`` is estimated by repeatedly sampling the same number
    of points uniformly at random from the set of Cα positions of *all* exposed
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

    ca_by_res: Dict[ResKey4, Tuple[float, float, float]] = {}
    for atom in pdb_atoms:
        if (atom.name or "").strip() != "CA":
            continue
        key = residue_key_from_atom(atom)
        ca_by_res[key] = (atom.x, atom.y, atom.z)

    residue_exposure = get_exposed_residues(sasa_output_data, float(sasa_cutoff))
    exposed_with_ca: List[ResKey4] = [
        k for k, exposed in residue_exposure.items() if exposed and k in ca_by_res
    ]
    if len(exposed_with_ca) < 2:
        return result

    allowed_coords = np.array([ca_by_res[k] for k in exposed_with_ca], dtype=np.float64)
    rng = np.random.default_rng(random_seed)

    def _coords_for(residue_set: Set[str]) -> np.ndarray:
        pts: List[Tuple[float, float, float]] = []
        for k in exposed_with_ca:
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
    sasa_output_data: Dict[ResKey4, SASAEntry],
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

    result["ripley_k_negative"] = compute_ripley(neg_coords, allowed_coords)
    result["ripley_k_positive"] = compute_ripley(pos_coords, allowed_coords)
    result["ripley_k_aromatic"] = compute_ripley(aromatic_coords, allowed_coords)
    result["ripley_k_hydrophobic"] = compute_ripley(hydro_coords, allowed_coords)
    result["ripley_k_polar"] = compute_ripley(polar_coords, allowed_coords)

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
            if is_hydrogen_atom(atom):
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
        if salt_bridge_residues is not None and key in salt_bridge_residues:
            continue
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
    residues_for_average: Optional[Iterable[Tuple[str, int, str, str]]] = None,
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
    if not list(residues_for_average):
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
    total = sum(energy_by_res.get(k, 0.0) for k in numer_keys)
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

