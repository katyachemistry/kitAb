#!/usr/bin/env python3
"""
Command-line interface for calculating developability descriptors (H-bonds, salt bridges, aromatic, WCN, SASA, etc.).

Usage:
    python run_developability.py <pdb_file> <sasa_file> [options]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
# import numpy as np

# Add src directory to path so developability and utils packages are found
# This allows running from the repo root (e.g. python3 src/developability/run_developability.py ...)
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from developability.descriptors import (
    calculate_global_hbond_density_average,
    calculate_salt_bridge_density_average,
    detect_salt_bridges,
    calculate_residue_category_density_average,
    compute_residue_side_abs_density_raw,
    compute_hbond_density_raw,
    net_charge_from_pka,
    # simple_residue_charge_from_sequence,
    pi_from_pka,
    scm_score_from_pka,
    scm_score_by_atoms,
    compute_sap_shell_synergy_scores,
    # calculate_weighted_contact_number_average,
    parse_dssp,
    calculate_hbond_energy_density_dssp_backbone_only_average,
    # calculate_hbond_energy_dssp_backbone_only_unweighted_average,
    compute_residue_DBSCAN_cluster_labels,
    summarize_dbscan_clusters,
    dbscan_cluster_side_abs_sasa_entropy,
    _get_atoms_for_path,
    get_inter_chain_interface_residues,
    # compute_surface_ripley_descriptors,
    compute_exposed_pair_correlation_cluster_scores,
    # compute_buried_pair_correlation_cluster_scores_aromatic_hydrophobic,
    # compute_surface_ann_index_descriptors,
    # compute_surface_pair_descriptors,
    # compute_atom_patch_area_cdr_pm_like_fast,
    # scm_score_propermab_like_fast,
    # calculate_relative_contact_order,
    compute_dipole_moment_magnitude,
    # compute_hydrophobic_moment_magnitude,
    compute_inter_chain_buried_sasa,
)
from developability.descriptors import (
    count_motif_overlapping,
    get_full_sequence_with_index_map_from_pdb,
)
from utils.chemistry import (
    AROMATIC_RESIDUES,
    CHARGE_FRACTION_NEGATIVE_RESIDUES,
    CHARGE_FRACTION_POSITIVE_RESIDUES,
    EXPOSURE_REL_ASA_THRESHOLD,
    NET_CHARGE_EXPOSURE_REL_ASA_THRESHOLD,
    GLN_ASN_RESIDUES,
    HYDROPHOBIC_RESIDUES,
    KYTE_DOOLITTLE,
    POLAR_RESIDUES,
    get_ff19sb_atom_charge,
    get_ff19sb_residue_region_charges,
    is_backbone_atom,
)
from developability.structure_context import ResKey4, StructureContext
from developability.descriptor_utils import (
    CDR_RANGES_CA,
    get_residue_region_map,
    get_exposed_residues,
    get_aromatic_residue_keys,
    get_residue_keys_by_type,
    residue_side_sasa,
    sum_residue_mean_local_curvature,
    mean_residue_curvature_over_residues,
    _count_residues_in_pdb,
)
from utils.parsers import parse_pka, get_pka_file_path, residue_key_from_atom

NET_CHARGE_PHS = [3, 7.5]


def _to_4(key):
    return (key[0], key[1], key[2], key[3]) if len(key) == 4 else (key[0], key[1], key[2], "")


# def _lookup(d, key):
#     return d.get(key) or (d.get((key[0], key[1], key[2], "")) if len(key) == 4 else None)


def _scalar_or_dict_sum(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict) and value:
        return float(sum(value.values()))
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (f != f) else f  # exclude nan
    return None


_PCF_CLUSTER_SHELL_KEY = re.compile(
    r"^pcf_(neg|pos|hyd|polar)_(.+)$"
)


def _pcf_cluster_per_shell_and_mean_per_category(
    pcf: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    For each residue category, copy per-distance-shell scores (e.g. ``3w1A``, ``4w1A``, ``5w1A``, ``3w2A``, ``5w2A``
    from :func:`compute_exposed_pair_correlation_cluster_scores`) into the
    surface JSON payload.
    """
    out: Dict[str, Optional[float]] = {}
    for k, v in pcf.items():
        m = _PCF_CLUSTER_SHELL_KEY.match(str(k))
        if not m:
            continue
        if v is None:
            out[k] = None
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            out[k] = None
            continue
        if math.isfinite(fv):
            out[k] = fv
        else:
            out[k] = None
    return out


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter

    )
    
    parser.add_argument(
        'pdb_file',
        type=str,
        help='Path to PDB structure file'
    )
    
    parser.add_argument(
        'sasa_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to SASA file (FreeSASA format). Required unless --wcn-only.'
    )
    
    parser.add_argument(
        '--dssp-file',
        type=str,
        default=None,
        help='Path to DSSP file (optional). If provided, adds secondary structure and H-bond energy columns to output.'
    )
    
    parser.add_argument(
        '--pka-file',
        type=str,
        default=None,
        help='Path to pKa file (optional). If provided, salt bridge detection will use pH-dependent charge states. If not provided, will try to auto-detect from PDB path (GINKGO_propka/{basename}_full.pka).'
    )
    
    parser.add_argument(
        '--pH',
        type=float,
        default=7.5,
        help='pH value for charge state determination in salt bridge detection (default: 7.4). Only used if pKa file is provided.'
    )
    parser.add_argument(
        '--heavy-chain',
        type=str,
        default='H',
        help='Chain ID for heavy chain in PDB (default: H). Only this and --light-chain are parsed.'
    )
    parser.add_argument(
        '--light-chain',
        type=str,
        default='L',
        help='Chain ID for light chain in PDB (default: L). Only this and --heavy-chain are parsed.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    if args.sasa_file:
        sasa_path = Path(args.sasa_file)
        if "_full" not in sasa_path.stem:
            new_stem = sasa_path.stem + "_full"
            args.sasa_file = str(sasa_path.with_name(new_stem + sasa_path.suffix))
    
    if not Path(args.pdb_file).exists():
        print(f"Error: PDB file not found: {args.pdb_file}", file=sys.stderr)
        sys.exit(1)
    
    if args.sasa_file is None:
        print(f"Error: SASA file is required.", file=sys.stderr)
        sys.exit(1)
    if not Path(args.sasa_file).exists():
        print(f"Error: SASA file not found: {args.sasa_file}", file=sys.stderr)
        sys.exit(1)
    
    pdb_atoms = _get_atoms_for_path(args.pdb_file)
    residue_atom_names_by_key: Dict[ResKey4, Set[str]] = defaultdict(set)
    for atom in pdb_atoms:
        residue_atom_names_by_key[residue_key_from_atom(atom)].add(
            (atom.name or "").strip().upper()
        )
    
    dssp_data = {}
    if args.dssp_file:
        if not Path(args.dssp_file).exists():
            print(f"Warning: DSSP file not found: {args.dssp_file}. Continuing without DSSP data.", file=sys.stderr)
        else:
            dssp_data = parse_dssp(args.dssp_file, pdb_atoms)
            if dssp_data:
                print(f"Parsed DSSP data for {len(dssp_data)} residues", file=sys.stderr)
    
    
    try:
            
        ctx = StructureContext(
            args.pdb_file,
            sasa_path=args.sasa_file,
            pka_path=args.pka_file,
        )

        # Use the context's pKa mapping everywhere downstream. This centralizes
        # pKa handling and enables standard-pKa fallback when PropKa data are
        # missing or incomplete.
        pka_output_data = ctx.pka_residue
        if pka_output_data:
            print(
                f"Using pKa data for {len(pka_output_data)} residues",
                file=sys.stderr,
            )

        # Treat parse failures as missing data (None/NaN), not as valid empty results.
        sasa_failed = bool(args.sasa_file) and (
            "sasa" in ctx.parse_errors or not ctx.sasa_residue
        )

        cdr_keys = {
                        key
                        for key, region in get_residue_region_map(ctx.atoms).items()
                        if region == "CDR"
                    }
        # Heavy-atom CDR vicinity (5 Å); cached in ``compute_cdr_vicinity_residue_keys``.
        cdr_vicinity_keys: Set[ResKey4] = (
            ctx.get_cdr_vicinity_residue_keys(cdr_keys) if cdr_keys else set()
        )
        exposed_keys: Set[ResKey4] = set()
        # exposed_keys_for_net_charge: Set[ResKey4] = set()
        if not sasa_failed:
            exposed_flags = get_exposed_residues(ctx.sasa_residue, EXPOSURE_REL_ASA_THRESHOLD)
            exposed_keys = {key for key, is_exposed in exposed_flags.items() if is_exposed}
            # exposed_flags_net_charge = get_exposed_residues(
            #     ctx.sasa_residue, NET_CHARGE_EXPOSURE_REL_ASA_THRESHOLD
            # )
            # exposed_keys_for_net_charge = {
            #     key for key, is_exposed in exposed_flags_net_charge.items() if is_exposed
            # }
        # beta_keys = {
        #     _to_4(key)
        #     for key, entry in (dssp_data or {}).items()
        #     if entry.get("secondary_structure") == "E"
        # }
        # interface_keys = get_inter_chain_interface_residues(args.pdb_file)
        buried_keys: Set[ResKey4] = set()
        if not sasa_failed:
            buried_keys = set(ctx.sasa_residue.keys()) - exposed_keys
        # CDR composition: ``cdr3_length`` / hydro–polar ratio stay strict CDR. Counts
        # ``n_*_cdr_vicinity`` are in the heavy-atom CDR vicinity shell; ``fraction_*_in_cdr``
        # use strict CDR residues only (numerator and denominator ``n_cdr``).
        n_cdr = len(cdr_keys)
        cdr3_start, cdr3_end = CDR_RANGES_CA[2]
        cdr3_length = sum(
            1 for k in cdr_keys
            if cdr3_start <= int(k[1]) <= cdr3_end
        )
        # if n_cdr > 0:
        #     hydro_cdr = sum(1 for k in cdr_keys if k[0] in HYDROPHOBIC_RESIDUES)
        #     polar_cdr = sum(1 for k in cdr_keys if k[0] in POLAR_RESIDUES)
        #     ratio_hydrophobic_to_polar_CDRs = (hydro_cdr / polar_cdr) if polar_cdr > 0 else 0.0
        # else:
        #     ratio_hydrophobic_to_polar_CDRs = 0.0

        # if cdr_vicinity_keys:
        #     n_gly_cdr_vicinity = sum(1 for k in cdr_vicinity_keys if k[0] == "GLY")
        #     n_pro_cdr_vicinity = sum(1 for k in cdr_vicinity_keys if k[0] == "PRO")
        #     n_aromatic_cdr_vicinity = sum(1 for k in cdr_vicinity_keys if k[0] in AROMATIC_RESIDUES)
        #     n_positive_cdr_vicinity = sum(1 for k in cdr_vicinity_keys if k[0] in {"ARG", "LYS"})
        #     n_negative_cdr_vicinity = sum(1 for k in cdr_vicinity_keys if k[0] in {"ASP", "GLU"})
        #     n_gln_asn_cdr_vicinity = sum(1 for k in cdr_vicinity_keys if k[0] in GLN_ASN_RESIDUES)
        # else:
        #     n_gly_cdr_vicinity = n_pro_cdr_vicinity = n_aromatic_cdr_vicinity = 0
        #     n_positive_cdr_vicinity = n_negative_cdr_vicinity = n_gln_asn_cdr_vicinity = 0

        if n_cdr > 0:
            n_gly_cdr = sum(1 for k in cdr_keys if k[0] == "GLY")
            n_pro_cdr = sum(1 for k in cdr_keys if k[0] == "PRO")
            n_aromatic_cdr = sum(1 for k in cdr_keys if k[0] in AROMATIC_RESIDUES)
            n_positive_cdr = sum(1 for k in cdr_keys if k[0] in {"ARG", "LYS"})
            n_negative_cdr = sum(1 for k in cdr_keys if k[0] in {"ASP", "GLU"})
            n_gln_asn_cdr = sum(1 for k in cdr_keys if k[0] in GLN_ASN_RESIDUES)
            inv_cdr = 1.0 / float(n_cdr)
            fraction_gly_in_cdr = n_gly_cdr * inv_cdr
            fraction_pro_in_cdr = n_pro_cdr * inv_cdr
            fraction_aromatic_in_cdr = n_aromatic_cdr * inv_cdr
            fraction_positive_in_cdr = n_positive_cdr * inv_cdr
            fraction_negative_in_cdr = n_negative_cdr * inv_cdr
            fraction_gln_asn_in_cdr = n_gln_asn_cdr * inv_cdr
        else:
            (
                fraction_gly_in_cdr,
                fraction_pro_in_cdr,
                fraction_aromatic_in_cdr,
                fraction_positive_in_cdr,
                fraction_negative_in_cdr,
                fraction_gln_asn_in_cdr,
            ) = (0.0,) * 6

        # Fraction buried and composition of buried residues
        # n_total = None if sasa_failed else len(ctx.sasa_residue)
        # n_buried = None if sasa_failed else len(buried_keys)
        # if n_total is None or n_buried is None:
        #     fraction_buried = None
        # elif n_total > 0:
        #     fraction_buried = n_buried / n_total
        # else:
        #     fraction_buried = 0.0
        # if n_buried > 0:
        #     fraction_hydrophobic_buried = sum(1 for k in buried_keys if k[0] in HYDROPHOBIC_RESIDUES) / n_buried
        #     # fraction_negative_buried = sum(
        #     #     1 for k in buried_keys if k[0] in CHARGE_FRACTION_NEGATIVE_RESIDUES
        #     # ) / n_buried
        #     # fraction_positive_buried = sum(
        #     #     1 for k in buried_keys if k[0] in CHARGE_FRACTION_POSITIVE_RESIDUES
        #     # ) / n_buried
        # else:
        #     fraction_hydrophobic_buried = 0.0
        #     # fraction_negative_buried = fraction_positive_buried = 0.0

        # Beta-sheet composition and Kyte-Doolittle sum
        # n_beta = len(beta_keys)
        # if n_beta > 0:
        #     fraction_hydrophobic_beta_sheet = sum(1 for k in beta_keys if k[0] in HYDROPHOBIC_RESIDUES) / n_beta
        #     # fraction_gln_asn_beta_sheet = sum(1 for k in beta_keys if k[0] in GLN_ASN_RESIDUES) / n_beta
        #     # hydrophobic_beta_sheet_kyte_doolittle_sum = sum(KYTE_DOOLITTLE.get(k[0], 0.0) for k in beta_keys)
        # else:
        #     fraction_hydrophobic_beta_sheet = 0.0
        #     # fraction_gln_asn_beta_sheet = 0.0
        #     # hydrophobic_beta_sheet_kyte_doolittle_sum = 0.0

        def _total_side_rel_weight(k: Tuple[str, int, str, str]) -> float:
            """Relative side-chain SASA (fraction in [0, 1] from ``parse_sasa``)."""
            return float(getattr(ctx.sasa_residue[k], "total_side_rel", 0.0)) or 0.0

        # Kyte-Doolittle sum across the whole parsed sequence (all residues in structure)
        # (Keys use 3-letter residue names, matching KYTE_DOOLITTLE mapping.)
        kyte_doolittle_sum_all = (
            None
            if sasa_failed
            else sum(KYTE_DOOLITTLE.get(k[0], 0.0) for k in ctx.sasa_residue.keys())
        )

        # Mean over residues of KD × relative side-chain SASA; denominator is residue
        # count in scope (not sum of rel SASA weights).
        # avg_kd_times_total_side_rel_all = (
        #     None
        #     if sasa_failed
        #     else (
        #         sum(
        #             KYTE_DOOLITTLE.get(k[0], 0.0) * _total_side_rel_weight(k)
        #             for k in ctx.sasa_residue.keys()
        #         )
        #         / float(n_total)
        #         if n_total
        #         else 0.0
        #     )
        # )
        # Same, numerator restricted to exposed residues; divide by exposed count.
        # n_exposed_in_sasa = (
        #     None
        #     if sasa_failed
        #     else sum(1 for k in ctx.sasa_residue.keys() if k in exposed_keys)
        # )
        # avg_kd_times_total_side_rel_exposed_over_exposed = (
        #     None
        #     if sasa_failed
        #     else (
        #         sum(
        #             KYTE_DOOLITTLE.get(k[0], 0.0) * _total_side_rel_weight(k)
        #             for k in ctx.sasa_residue.keys()
        #             if k in exposed_keys
        #         )
        #         / float(n_exposed_in_sasa)
        #         if n_exposed_in_sasa
        #         else 0.0
        #     )
        # )
        # Same over CDR vicinity (heavy-atom 5 Å shell); divide by SASA residue count in that shell.
        n_cdr_vicinity_in_sasa = (
            None
            if sasa_failed
            else sum(1 for k in ctx.sasa_residue.keys() if k in cdr_vicinity_keys)
        )
        avg_kd_times_total_side_rel_cdr_vicinity_over_cdr_vicinity = (
            None
            if sasa_failed
            else (
                sum(
                    KYTE_DOOLITTLE.get(k[0], 0.0) * _total_side_rel_weight(k)
                    for k in ctx.sasa_residue.keys()
                    if k in cdr_vicinity_keys
                )
                / float(n_cdr_vicinity_in_sasa)
                if n_cdr_vicinity_in_sasa
                else 0.0
            )
        )
        # kyte_doolittle_mean_all = (
        #     None
        #     if sasa_failed or not n_total
        #     else (float(kyte_doolittle_sum_all) / n_total)  # type: ignore[arg-type]
        # )

        hydrophobic_keys = get_residue_keys_by_type(
            args.pdb_file, HYDROPHOBIC_RESIDUES
        )
        polar_keys = get_residue_keys_by_type(args.pdb_file, POLAR_RESIDUES)
        negative_keys = get_residue_keys_by_type(
            args.pdb_file, CHARGE_FRACTION_NEGATIVE_RESIDUES
        )
        positive_keys = get_residue_keys_by_type(
            args.pdb_file, CHARGE_FRACTION_POSITIVE_RESIDUES
        )
        aromatic_keys = get_aromatic_residue_keys(args.pdb_file)

        # Salt bridge densities (detect once, reuse for counts and all averages)
        salt_bridges = detect_salt_bridges(args.pdb_file, args.sasa_file, args.pka_file, args.pH)
        number_of_salt_bridges = len(salt_bridges)
        salt_bridge_residues: Set[Tuple[str, int, str, str]] = set()
        if salt_bridges:
            for (pos_key, neg_key) in salt_bridges.keys():
                salt_bridge_residues.add(pos_key)
                salt_bridge_residues.add(neg_key)

        # avg_salt = calculate_salt_bridge_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     args.pka_file,
        #     args.pH,
        #     salt_bridges=salt_bridges,
        #     sqrt_weights=False,
        # )

        avg_salt_cdr_vicinity_over_cdr_vicinity = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            args.pH,
            residues_for_density=cdr_vicinity_keys,
            residues_for_average=cdr_vicinity_keys,
            salt_bridges=salt_bridges,
            sqrt_weights=False,
        )

        # avg_salt_exposed_over_all = calculate_salt_bridge_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     args.pka_file,
        #     args.pH,
        #     residues_for_density=exposed_keys,
        #     salt_bridges=salt_bridges,
        #     # residues_for_average=exposed_keys,
        #     sqrt_weights=False,
        # )

        # avg_salt_beta_over_all = calculate_salt_bridge_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     args.pka_file,
        #     args.pH,
        #     residues_for_density=beta_keys,
        #     salt_bridges=salt_bridges,
        #     # residues_for_average=beta_keys,
        #     sqrt_weights=False,
        # )

        # avg_salt_interface = calculate_salt_bridge_density_average(
        #                     args.pdb_file,
        #                     args.sasa_file,
        #                     args.pka_file,
        #                     args.pH,
        #                     residues_for_density=interface_keys,
        #                     residues_for_average=interface_keys,
        #                     salt_bridges=salt_bridges,
        #                     sqrt_weights=False,
        #                 )

        avg_hbond_energy = None
        # avg_hbond_energy_buried_over_all = None
        # avg_hbond_energy_beta_over_all = None
        # avg_hbond_energy_dssp_unweighted = None
        # avg_hbond_energy_dssp_unweighted_buried_over_all = None
        # avg_hbond_energy_dssp_unweighted_beta_over_all = None
        # Explicit "DSSP-weighted" (SASA * |energy|) aliases for clarity in JSON output.
        # These are equivalent to the existing `avg_hbond_energy_*` values below and will
        # be NaN/None if SASA is missing or fails to parse.
        avg_hbond_energy_dssp_weighted = None
        # avg_hbond_energy_dssp_weighted_buried_over_all = None
        # avg_hbond_energy_dssp_weighted_beta_over_all = None
        avg_hbond_energy = calculate_hbond_energy_density_dssp_backbone_only_average(
            args.pdb_file,
            args.sasa_file,
            args.dssp_file,
            residues_for_density=None,
            residues_for_average=None,
            sqrt_weights=False,
        )

        # avg_hbond_energy_buried_over_all = calculate_hbond_energy_density_dssp_backbone_only_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     args.dssp_file,
        #     residues_for_density=buried_keys,
        #     residues_for_average=None,
        #     sqrt_weights=False,
        # )

        # avg_hbond_energy_beta_over_all = (
        #     calculate_hbond_energy_density_dssp_backbone_only_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         args.dssp_file,
        #         residues_for_density=beta_keys,
        #         residues_for_average=None,
        #         sqrt_weights=False,
        #     )
        #     if beta_keys
        #     else None
        # )

        # Alias weighted DSSP+SASA energy density metrics under explicit names.
        avg_hbond_energy_dssp_weighted = avg_hbond_energy
        # avg_hbond_energy_dssp_weighted_buried_over_all = avg_hbond_energy_buried_over_all
        # avg_hbond_energy_dssp_weighted_beta_over_all = avg_hbond_energy_beta_over_all

        # DSSP-only (no SASA) H-bond energy metric so missing SASA doesn't erase DSSP signal.
        # avg_hbond_energy_dssp_unweighted = calculate_hbond_energy_dssp_backbone_only_unweighted_average(
        #     args.pdb_file,
        #     args.dssp_file,
        #     residues_for_density=None,
        #     residues_for_average=None,
        # )
        # avg_hbond_energy_dssp_unweighted_buried_over_all = calculate_hbond_energy_dssp_backbone_only_unweighted_average(
        #     args.pdb_file,
        #     args.dssp_file,
        #     residues_for_density=buried_keys,
        #     residues_for_average=None,
        # )
        # avg_hbond_energy_dssp_unweighted_beta_over_all = (
        #     calculate_hbond_energy_dssp_backbone_only_unweighted_average(
        #         args.pdb_file,
        #         args.dssp_file,
        #         residues_for_density=beta_keys,
        #         residues_for_average=None,
        #     )
        #     if beta_keys
        #     else None
        # )

     
        # aromatic_exposed_keys = aromatic_keys & exposed_keys
        # hydrophobic_exposed_keys = hydrophobic_keys & exposed_keys
        # polar_exposed_keys = polar_keys & exposed_keys
        # negative_exposed_keys = negative_keys & exposed_keys
        # positive_exposed_keys = positive_keys & exposed_keys

        exposed_cdr_vicinity_keys = exposed_keys & cdr_vicinity_keys
        aromatic_cdr_vicinity_keys = aromatic_keys & exposed_cdr_vicinity_keys
        hydrophobic_cdr_vicinity_keys = hydrophobic_keys & exposed_cdr_vicinity_keys
        polar_cdr_vicinity_keys = polar_keys & exposed_cdr_vicinity_keys
        negative_cdr_vicinity_keys = negative_keys & exposed_cdr_vicinity_keys
        positive_cdr_vicinity_keys = positive_keys & exposed_cdr_vicinity_keys

        # Per-residue local PCA curvature (λ₃/trace): sums per chemistry class ∩ CDR
        # vicinity ∩ exposed, normalized by mean curvature over exposed CDR-vicinity residues.
        total_local_curvature_negative_cdr_vicinity = sum_residue_mean_local_curvature(
            pdb_atoms, negative_cdr_vicinity_keys
        )
        total_local_curvature_positive_cdr_vicinity = sum_residue_mean_local_curvature(
            pdb_atoms, positive_cdr_vicinity_keys
        )
        total_local_curvature_aromatic_cdr_vicinity = sum_residue_mean_local_curvature(
            pdb_atoms, aromatic_cdr_vicinity_keys
        )
        total_local_curvature_hydrophobic_cdr_vicinity = sum_residue_mean_local_curvature(
            pdb_atoms, hydrophobic_cdr_vicinity_keys
        )
        # total_local_curvature_polar_cdr_vicinity = sum_residue_mean_local_curvature(
        #     pdb_atoms, polar_cdr_vicinity_keys
        # )
        mean_curvature_cdr_vicinity = mean_residue_curvature_over_residues(
            pdb_atoms, exposed_cdr_vicinity_keys
        )
        _curv_denom = float(mean_curvature_cdr_vicinity)
        if _curv_denom > 1e-18 and math.isfinite(_curv_denom):
            normalized_local_curvature_negative_cdr_vicinity = (
                total_local_curvature_negative_cdr_vicinity / _curv_denom
            )
            normalized_local_curvature_positive_cdr_vicinity = (
                total_local_curvature_positive_cdr_vicinity / _curv_denom
            )
            normalized_local_curvature_aromatic_cdr_vicinity = (
                total_local_curvature_aromatic_cdr_vicinity / _curv_denom
            )
            normalized_local_curvature_hydrophobic_cdr_vicinity = (
                total_local_curvature_hydrophobic_cdr_vicinity / _curv_denom
            )
            # normalized_local_curvature_polar_cdr_vicinity = (
            #     total_local_curvature_polar_cdr_vicinity / _curv_denom
            # )
        else:
            normalized_local_curvature_negative_cdr_vicinity = 0.0
            normalized_local_curvature_positive_cdr_vicinity = 0.0
            normalized_local_curvature_aromatic_cdr_vicinity = 0.0
            normalized_local_curvature_hydrophobic_cdr_vicinity = 0.0
            # normalized_local_curvature_polar_cdr_vicinity = 0.0

        # SASA-weighted sums restricted to each type on the exposed surface; divide by
        # exposed residue count (same ``exposed_keys`` as ``get_exposed_residues`` with EXPOSURE_REL_ASA_THRESHOLD).
        # avg_aromatic_exposed_over_exposed = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=aromatic_exposed_keys,
        #         residues_for_average=exposed_keys,
        #         sqrt_weights=False,
        #     )
        #     if exposed_keys
        #     else 0.0
        # )
        # avg_hydrophobic_exposed_over_exposed = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=hydrophobic_exposed_keys,
        #         residues_for_average=exposed_keys,
        #         sqrt_weights=False,
        #     )
        #     if exposed_keys
        #     else 0.0
        # )

        # Beta-sheet (DSSP ``E``) aromatic / hydrophobic, averaged over all residues
        # aromatic_beta_keys = aromatic_keys & beta_keys
        # hydrophobic_beta_keys = hydrophobic_keys & beta_keys
        # avg_aromatic_beta_over_all = calculate_residue_category_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     residue_category=aromatic_beta_keys,
        #     residues_for_average=None,
        #     sqrt_weights=False,
        # ) if aromatic_beta_keys else 0.0
        # avg_hydrophobic_beta_over_all = calculate_residue_category_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     residue_category=hydrophobic_beta_keys,
        #     residues_for_average=None,
        #     sqrt_weights=False,
        # ) if hydrophobic_beta_keys else 0.0

        density_side_abs_raw: Optional[Dict[ResKey4, float]] = None
        if not sasa_failed:
            density_side_abs_raw = compute_residue_side_abs_density_raw(
                args.pdb_file, args.sasa_file
            )

        # Sum of side-chain rel SASA weights (no sqrt) over category ∩ CDR vicinity;
        # ``residues_for_average="no"`` skips division (see ``average_over_residues``).
        sum_aromatic_weighted_rel_side_asa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=aromatic_cdr_vicinity_keys,
                residues_for_average="no",
                sqrt_weights=False,
            )
            if aromatic_cdr_vicinity_keys
            else 0.0
        )

        sum_hydrophobic_weighted_rel_side_asa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=hydrophobic_cdr_vicinity_keys,
                residues_for_average="no",
                sqrt_weights=False,
            )
            if hydrophobic_cdr_vicinity_keys
            else 0.0
        )

        # avg_polar_exposed_over_exposed = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=polar_exposed_keys,
        #         residues_for_average=exposed_keys,
        #         sqrt_weights=False,
        #     )
        #     if exposed_keys
        #     else 0.0
        # )
        # avg_negative_exposed_over_exposed = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=negative_exposed_keys,
        #         residues_for_average=exposed_keys,
        #         sqrt_weights=False,
        #     )
        #     if exposed_keys
        #     else 0.0
        # )
        # avg_positive_exposed_over_exposed = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=positive_exposed_keys,
        #         residues_for_average=exposed_keys,
        #         sqrt_weights=False,
        #     )
        #     if exposed_keys
        #     else 0.0
        # )
        # _den_pr = (
        #     float(avg_polar_exposed_over_exposed)
        #     + float(avg_negative_exposed_over_exposed)
        #     + float(avg_positive_exposed_over_exposed)
        # )
        # if (
        #     math.isfinite(_den_pr)
        #     and abs(_den_pr) > 1e-18
        #     and math.isfinite(float(avg_hydrophobic_exposed_over_exposed))
        # ):
        #     ratio_avg_hydrophobic_to_negative_positive_polar_exposed = (
        #         float(avg_hydrophobic_exposed_over_exposed) / _den_pr
        #     )
        # else:
        #     ratio_avg_hydrophobic_to_negative_positive_polar_exposed = None

        # sum_polar_weighted_rel_side_asa_cdr_vicinity = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=polar_cdr_vicinity_keys,
        #         residues_for_average="no",
        #         sqrt_weights=False,
        #     )
        #     if polar_cdr_vicinity_keys
        #     else 0.0
        # )
        sum_negative_weighted_rel_side_asa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=negative_cdr_vicinity_keys,
                residues_for_average="no",
                sqrt_weights=False,
            )
            if negative_cdr_vicinity_keys
            else 0.0
        )
        sum_positive_weighted_rel_side_asa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=positive_cdr_vicinity_keys,
                residues_for_average="no",
                sqrt_weights=False,
            )
            if positive_cdr_vicinity_keys
            else 0.0
        )

        # Sum of absolute side-chain SASA (Å²) over category ∩ CDR vicinity (no division).
        sum_aromatic_side_abs_sasa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=aromatic_cdr_vicinity_keys,
                residues_for_average="no",
                sqrt_weights=False,
                density_raw=density_side_abs_raw,
            )
            if (density_side_abs_raw and aromatic_cdr_vicinity_keys)
            else 0.0
        )
        # sum_hydrophobic_side_abs_sasa_cdr_vicinity = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=hydrophobic_cdr_vicinity_keys,
        #         residues_for_average="no",
        #         sqrt_weights=False,
        #         density_raw=density_side_abs_raw,
        #     )
        #     if (density_side_abs_raw and hydrophobic_cdr_vicinity_keys)
        #     else 0.0
        # )
        # sum_polar_side_abs_sasa_cdr_vicinity = (
        #     calculate_residue_category_density_average(
        #         args.pdb_file,
        #         args.sasa_file,
        #         residue_category=polar_cdr_vicinity_keys,
        #         residues_for_average="no",
        #         sqrt_weights=False,
        #         density_raw=density_side_abs_raw,
        #     )
        #     if (density_side_abs_raw and polar_cdr_vicinity_keys)
        #     else 0.0
        # )
        sum_negative_side_abs_sasa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=negative_cdr_vicinity_keys,
                residues_for_average="no",
                sqrt_weights=False,
                density_raw=density_side_abs_raw,
            )
            if (density_side_abs_raw and negative_cdr_vicinity_keys)
            else 0.0
        )
        sum_positive_side_abs_sasa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=positive_cdr_vicinity_keys,
                residues_for_average="no",
                sqrt_weights=False,
                density_raw=density_side_abs_raw,
            )
            if (density_side_abs_raw and positive_cdr_vicinity_keys)
            else 0.0
        )

        # avg_wcn = calculate_weighted_contact_number_average(args.pdb_file)

        exposed_flags = get_exposed_residues(ctx.sasa_residue, EXPOSURE_REL_ASA_THRESHOLD)

        # Whole-structure WCN averages.

        # avg_wcn_buried_over_all = calculate_weighted_contact_number_average(
        #     args.pdb_file,
        #     residue_category=buried_keys,
        #     residues_for_density=None,
        #     residues_for_average=None,
        # )

        # CDR-only WCN, averaged over CDR residues
        # avg_wcn_cdr_over_cdr = calculate_weighted_contact_number_average(
        #     args.pdb_file,
        #     residue_category=cdr_keys,
        #     residues_for_density=cdr_keys,
        #     residues_for_average=cdr_keys,
        # ) if cdr_keys else 0.0

        # avg_wcn_interface_over_all = calculate_weighted_contact_number_average(
        #     args.pdb_file,
        #     residue_category=interface_keys,
        #     residues_for_density=None,
        #     residues_for_average=None,
        # )

        # Largest connected component size of the (geometry-based) H-bond network
        # largest_hbond_component_size_val = largest_hbond_component_size(args.pdb_file)

        # Cα DBSCAN cluster labels (negative/positive/aromatic/hydrophobic/polar; charge from PropKA)
        exposed_flags = get_exposed_residues(ctx.sasa_residue, EXPOSURE_REL_ASA_THRESHOLD)
        exposed_keys = {key for key, is_exposed in exposed_flags.items() if is_exposed}
        pdb_atoms_for_clustering = pdb_atoms

        neg_exposed_cluster_labels, pos_exposed_cluster_labels, hydro_exposed_cluster_labels = compute_residue_DBSCAN_cluster_labels(
                    pdb_atoms_for_clustering, pka_output_data, args.pH
                )

        neg_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            neg_exposed_cluster_labels, ctx.sasa_output
        )
        pos_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            pos_exposed_cluster_labels, ctx.sasa_output
        )
        # aromatic_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
        #     aromatic_exposed_cluster_labels, ctx.sasa_output
        # )
        hydro_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            hydro_exposed_cluster_labels, ctx.sasa_output
        )
        # polar_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
        #     polar_exposed_cluster_labels, ctx.sasa_output
        # )

        # Same CDR vicinity (heavy-atom 5 Å shell) as SASA-weighted CDR metrics, not CDR-only.
        pdb_atoms_for_cdr_clustering = [
            atom
            for atom in pdb_atoms
            if residue_key_from_atom(atom) in cdr_vicinity_keys
        ]
        (
            neg_cdr_cluster_labels,
            pos_cdr_cluster_labels,
            hydro_cdr_cluster_labels,
        ) = compute_residue_DBSCAN_cluster_labels(
            pdb_atoms_for_cdr_clustering, pka_output_data, args.pH
        )
        neg_cdr_vicinity_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            neg_cdr_cluster_labels, ctx.sasa_output
        )
        pos_cdr_vicinity_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            pos_cdr_cluster_labels, ctx.sasa_output
        )
        # aromatic_cdr_vicinity_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
        #     aromatic_cdr_cluster_labels, ctx.sasa_output
        # )
        hydro_cdr_vicinity_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            hydro_cdr_cluster_labels, ctx.sasa_output
        )
        # polar_cdr_vicinity_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
        #     polar_cdr_cluster_labels, ctx.sasa_output
        # )

        neg_cluster_side_abs_sasa_entropy = dbscan_cluster_side_abs_sasa_entropy(
            neg_exposed_cluster_labels, ctx.sasa_output
        )
        pos_cluster_side_abs_sasa_entropy = dbscan_cluster_side_abs_sasa_entropy(
            pos_exposed_cluster_labels, ctx.sasa_output
        )
        # aromatic_cluster_side_abs_sasa_entropy = dbscan_cluster_side_abs_sasa_entropy(
        #     aromatic_exposed_cluster_labels, ctx.sasa_output
        # )
        # hydro_cluster_side_abs_sasa_entropy = dbscan_cluster_side_abs_sasa_entropy(
        #     hydro_exposed_cluster_labels, ctx.sasa_output
        # )
        # polar_cluster_side_abs_sasa_entropy = dbscan_cluster_side_abs_sasa_entropy(
        #     polar_exposed_cluster_labels, ctx.sasa_output
        # )

        # Ripley K, PSH, PPC, PNC
        # ripley = compute_surface_ripley_descriptors(pdb_atoms, ctx.sasa_residue, pka_output_data, args.pH)
        # ripley_pm_like = compute_surface_ripley_descriptors(
        #     pdb_atoms,
        #     ctx.sasa_residue,
        #     pka_output_data,
        #     args.pH,
        #     surface_exposed_threshold=0.05,
        #     ripley_distance=6.0,
        # )
        pcf_cluster = compute_exposed_pair_correlation_cluster_scores(
            pdb_atoms,
            ctx.sasa_residue,
            pka_output_data,
            args.pH,
            surface_exposed_threshold=EXPOSURE_REL_ASA_THRESHOLD,
        )
        # pcf_cluster_buried_aromatic_hydro = (
        #     compute_buried_pair_correlation_cluster_scores_aromatic_hydrophobic(
        #         pdb_atoms,
        #         ctx.sasa_residue,
        #         pka_output_data,
        #         args.pH,
        #         surface_exposed_threshold=EXPOSURE_REL_ASA_THRESHOLD,
        #     )
        # )
        # ann_idx = compute_surface_ann_index_descriptors(pdb_atoms, ctx.sasa_residue)
        # ann_idx_pm_like = compute_surface_ann_index_descriptors(
        #     pdb_atoms,
        #     ctx.sasa_residue,
        #     sasa_cutoff=0.05,
        # )
        # pairs = compute_surface_pair_descriptors(
        #     pdb_atoms,
        #     ctx.sasa_residue,
        #     pka_output_data,
        #     args.pH,
        #     salt_bridge_residues=salt_bridge_residues,
        # )
        # ripley_k_negative = ripley["ripley_k_negative"]
        # ripley_k_positive = ripley["ripley_k_positive"]
        # ripley_k_aromatic = ripley["ripley_k_aromatic"]
        # ripley_k_hydrophobic = ripley["ripley_k_hydrophobic"]
        # ripley_k_polar = ripley["ripley_k_polar"]
        # pos_ann_index = ann_idx["pos_ann_index"]
        # neg_ann_index = ann_idx["neg_ann_index"]
        # aromatic_ann_index = ann_idx["aromatic_ann_index"]
        # pos_ann_index_pm_like = ann_idx_pm_like["pos_ann_index"]
        # neg_ann_index_pm_like = ann_idx_pm_like["neg_ann_index"]
        # aromatic_ann_index_pm_like = ann_idx_pm_like["aromatic_ann_index"]
        # ripley_k_negative_pm_like = ripley_pm_like["ripley_k_negative"]
        # ripley_k_positive_pm_like = ripley_pm_like["ripley_k_positive"]
        # ripley_k_aromatic_pm_like = ripley_pm_like["ripley_k_aromatic"]
        # ripley_k_hydrophobic_pm_like = ripley_pm_like["ripley_k_hydrophobic"]
        # ripley_k_polar_pm_like = ripley_pm_like["ripley_k_polar"]
        # psh_all_surface = pairs["psh_all_surface"]
        # psh_cdr_vicinity = pairs["psh_cdr_vicinity"]
        # ppc_all_surface = pairs["ppc_all_surface"]
        # ppc_cdr_vicinity = pairs["ppc_cdr_vicinity"]
        # pnc_all_surface = pairs["pnc_all_surface"]
        # pnc_cdr_vicinity = pairs["pnc_cdr_vicinity"]

        # H-bond density averages (structure-level)
        # avg_hbond = None
        avg_hbond_cdr_vicinity = None
        # avg_hbond_buried_over_all = None
        # avg_hbond_beta_over_all = None
        # avg_hbond_inter_chain = None
        weights_raw, counts = compute_hbond_density_raw(args.pdb_file, args.sasa_file)
        # number_of_hbonds = sum(counts.values()) // 2 if counts else 0
        _n_res = _count_residues_in_pdb(args.pdb_file)
        # mean_hbond_degree = (2.0 * number_of_hbonds / _n_res) if _n_res > 0 else 0.0

        # avg_hbond = calculate_global_hbond_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     weights_raw=weights_raw,
        #     counts=counts,
        #     residues_for_density=None,
        #     residues_for_average=None,
        #     sqrt_weights=False,
        # )
        avg_hbond_cdr_vicinity = calculate_global_hbond_density_average(
            args.pdb_file,
            args.sasa_file,
            weights_raw=weights_raw,
            counts=counts,
            residues_for_density=cdr_vicinity_keys,
            residues_for_average=cdr_vicinity_keys,
            sqrt_weights=False,
        )
        # avg_hbond_buried_over_all = calculate_global_hbond_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     weights_raw=weights_raw,
        #     counts=counts,
        #     residues_for_density=buried_keys,
        #     residues_for_average=None,
        #     sqrt_weights=False,
        # )

        # avg_hbond_beta_over_all = calculate_global_hbond_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     weights_raw=weights_raw,
        #     counts=counts,
        #     residues_for_density=beta_keys,
        #     residues_for_average=None,
        #     sqrt_weights=False,
        # )

        # avg_hbond_inter_chain = calculate_global_hbond_density_average(
        #     args.pdb_file,
        #     args.sasa_file,
        #     weights_raw=weights_raw,
        #     counts=counts,
        #     residues_for_density=interface_keys,
        #     residues_for_average=interface_keys,
        #     sqrt_weights=False,
        # )

        # Inter-chain buried SASA (structure-level)
        inter_chain_buried_sasa = compute_inter_chain_buried_sasa(args.sasa_file)
        # total_asa = get_sasa_total(args.sasa_file)

        # Dipole moment magnitude (structure-level)
        dipole_moment_magnitude = compute_dipole_moment_magnitude(pdb_atoms, pka_output_data, args.pH)
        # Hydrophobic moment (PROPERMAB-style: ‖Σ h_i r_i‖ over residue centers).
        # hydrophobic_moment_magnitude = compute_hydrophobic_moment_magnitude(pdb_atoms)

        # Relative contact order (CO) from Cα contacts
        # - "all" is intra-chain only across all chains (H+L for antibodies)
        # - heavy/light are computed within each chain
        # contact_order_all = calculate_relative_contact_order(
        #     args.pdb_file,
        #     ca_cutoff=8.0,
        #     min_sequence_separation=0,
        #     include_inter_chain=False,
        #     chain_order=[args.heavy_chain, args.light_chain],
        # )
        # contact_order_heavy = calculate_relative_contact_order(
        #     args.pdb_file,
        #     ca_cutoff=8.0,
        #     min_sequence_separation=0,
        #     include_inter_chain=False,
        #     chain_order=[args.heavy_chain],
        # )
        # contact_order_light = calculate_relative_contact_order(
        #     args.pdb_file,
        #     ca_cutoff=8.0,
        #     min_sequence_separation=0,
        #     include_inter_chain=False,
        #     chain_order=[args.light_chain],
        # )

        # net charge at different pHs; pI
        # seq_heavy, index_map_heavy = get_full_sequence_with_index_map_from_pdb(
        #     args.pdb_file, chain_order=[args.heavy_chain]
        # )
        # seq_light, index_map_light = get_full_sequence_with_index_map_from_pdb(
        #     args.pdb_file, chain_order=[args.light_chain]
        # )
        # seq_full = (seq_heavy or "") + (seq_light or "")
        net_charge_by_pH = {}
        # net_charge = net_charge_from_pka(pka_output_data, args.pH)  # PropKa-based
        # net_charge = net_charge_from_seq(seq_full, args.pH)

        cdr_keys_norm = {_to_4(k) for k in cdr_keys}

        # def _subsequence_for_keyset(
        #     seq: str,
        #     index_map: Dict[Tuple[str, int, str, str], int],
        #     keys_norm: Set[Tuple[str, int, str, str]],
        # ) -> str:
        #     """1-letter subsequence from seq for residues whose key is in keys_norm (chain order preserved)."""
        #     if not seq or not index_map:
        #         return ""
        #     indices = sorted(
        #         {
        #             int(idx)
        #             for key, idx in index_map.items()
        #             if _to_4(key) in keys_norm
        #             and idx is not None
        #             and 0 <= int(idx) < len(seq)
        #         }
        #     )
        #     return "".join(seq[i] for i in indices)

        # pka_output_data_cdr = {
        #     k: v for k, v in pka_output_data.items() if _to_4(k) in cdr_keys_norm
        # }
        # net_charge_cdr_from_pka = net_charge_from_pka(pka_output_data_cdr, args.pH)
        # Net-charge-on-surface: rel side-chain ASA > NET_CHARGE_EXPOSURE_REL_ASA_THRESHOLD
        # (0.05); all other descriptors keep EXPOSURE_REL_ASA_THRESHOLD (0.20).
        # exposed_keys_norm_net_charge = {_to_4(k) for k in exposed_keys_for_net_charge}
        # pka_output_data_exposed = {
        #     k: v for k, v in pka_output_data.items() if _to_4(k) in exposed_keys_norm_net_charge
        # }
        # exposed_net_charge = net_charge_from_pka(pka_output_data_exposed, args.pH)
        # pka_output_data_exposed_cdr = {
        #     k: v
        #     for k, v in pka_output_data.items()
        #     if _to_4(k) in exposed_keys_norm_net_charge and _to_4(k) in cdr_keys_norm
        # }
        # exposed_net_charge_cdr = net_charge_from_pka(pka_output_data_exposed_cdr, args.pH)

        # exposed_net_charge_simple = sum(
        #     simple_residue_charge_from_sequence(k[0]) for k in exposed_keys_for_net_charge
        # )
        # exposed_net_charge_cdr_simple = sum(
        #     simple_residue_charge_from_sequence(k[0])
        #     for k in cdr_keys
        #     if _to_4(k) in exposed_keys_norm_net_charge
        # )

        protein_pi = pi_from_pka(pka_output_data)
        for ph in NET_CHARGE_PHS:
            net_charge_by_pH[ph] = net_charge_from_pka(pka_output_data, ph)  # PropKa-based
            # net_charge_by_pH[ph] = net_charge_from_seq(seq_full, ph)
            
        # SCM score (requires SASA + pKa)
        weighted_scm_score_by_pH = {}
        for ph in NET_CHARGE_PHS:
            weighted_scm_score_by_pH[ph] = scm_score_from_pka(args.pdb_file, args.sasa_file, pka_output_data, ph, d_cutoff=10.0)
        # unweighted_scm_score_by_pH = {}
        # for ph in NET_CHARGE_PHS:
        #     unweighted_scm_score_by_pH[ph] = scm_score_from_pka(
        #         args.pdb_file,
        #         args.sasa_file,
        #         pka_output_data,
        #         ph,
        #         d_cutoff=10.0,
        #         sasa_weighting=False,
        #     )
        # Cationic clustering: same settings as weighted SCM but summing positive SCM_i
        weighted_scm_pos_score_by_pH = {}
        for ph in NET_CHARGE_PHS:
            weighted_scm_pos_score_by_pH[ph] = scm_score_from_pka(args.pdb_file, args.sasa_file, pka_output_data, ph, d_cutoff=10.0, reduce="pos_abs")
        scm_by_atoms = scm_score_by_atoms(args.pdb_file, d_cutoff=10.0) or {}
        scm_by_atoms_neg = scm_by_atoms.get("scm_by_atoms_neg")
        scm_by_atoms_pos = scm_by_atoms.get("scm_by_atoms_pos")
        # scm_pm_like = scm_score_propermab_like_fast(
        #     args.pdb_file,
        #     args.sasa_file,
        #     d_cutoff=10.0,
        #     sidechain_sasa_cutoff=10.0,
        # )
        # scm_propermab_score_by_pH = {}
        # for ph in NET_CHARGE_PHS:
        #     scm_propermab_score_by_pH[ph] = scm_score_from_pka_propermab(
        #         args.pdb_file,
        #         args.sasa_file,
        #         pka_output_data,
        #         ph,
        #         d_cutoff=10.0,
        #         chain_ids=(args.heavy_chain, args.light_chain),
        #     )

        # Per-chain net charge at pH 7 (heavy/light)
        _pka_heavy = {k: v for k, v in pka_output_data.items() if k[2] == args.heavy_chain}
        _pka_light = {k: v for k, v in pka_output_data.items() if k[2] == args.light_chain}
        heavy_charge = net_charge_from_pka(_pka_heavy, 7.5)
        light_charge = net_charge_from_pka(_pka_light, 7.5)
        # heavy_charge_pH74 = net_charge_from_seq(seq_heavy or "", 7.4)
        # light_charge_pH74 = net_charge_from_seq(seq_light or "", 7.4)
        asymmetry_score = heavy_charge * light_charge
        # asymmetry_substract = heavy_charge - light_charge
        residue_keys_all = set(ctx.residue_keys)
        # heavy_residue_keys = {k for k in residue_keys_all if k[2] == args.heavy_chain}
        # light_residue_keys = {k for k in residue_keys_all if k[2] == args.light_chain}
        # Fv_chml = (
        #     sum(simple_residue_charge_from_sequence(k[0]) for k in heavy_residue_keys)
        #     - sum(simple_residue_charge_from_sequence(k[0]) for k in light_residue_keys)
        # )
        # exposed_heavy_residue_keys = {k for k in exposed_keys_for_net_charge if k[2] == args.heavy_chain}
        # exposed_light_residue_keys = {k for k in exposed_keys_for_net_charge if k[2] == args.light_chain}
        # exposed_Fv_chml = (
        #     sum(simple_residue_charge_from_sequence(k[0]) for k in exposed_heavy_residue_keys)
        #     - sum(simple_residue_charge_from_sequence(k[0]) for k in exposed_light_residue_keys)
        # )

        # SAP-like structure-level metrics (multiple weighting modes)
        # sasa_output_data = ctx.sasa_output
        _sasa = ctx.sasa_residue

        def _clamp01(value: Optional[float]) -> float:
            if value is None:
                return 0.0
            return max(0.0, min(1.0, float(value)))

        def _ff19sb_total_charge_for_residue(key4: ResKey4) -> float:
            backbone_q, sidechain_q = get_ff19sb_residue_region_charges(
                key4[0],
                residue_atom_names=residue_atom_names_by_key.get(key4),
            )
            return float(backbone_q + sidechain_q)

        # def _ff19sb_exposed_charge_for_residue(key4: ResKey4) -> float:
        #     backbone_q, sidechain_q = get_ff19sb_residue_region_charges(
        #         key4[0],
        #         residue_atom_names=residue_atom_names_by_key.get(key4),
        #     )
        #     entry = _sasa.get(key4)
        #     if entry is None:
        #         return 0.0
        #     main_rel = _clamp01(getattr(entry, "main_chain_rel", 0.0))
        #     side_rel = _clamp01(getattr(entry, "total_side_rel", 0.0))
        #     return float(backbone_q * main_rel + sidechain_q * side_rel)

        net_charge_ff19sb = sum(_ff19sb_total_charge_for_residue(k) for k in residue_keys_all)
        # exposed_net_charge_ff19sb = sum(
        #     _ff19sb_exposed_charge_for_residue(k) for k in residue_keys_all
        # )
        # net_charge_cdr_ff19sb = sum(_ff19sb_total_charge_for_residue(k) for k in cdr_keys_norm)
        # exposed_net_charge_cdr_ff19sb = sum(
        #     _ff19sb_exposed_charge_for_residue(k) for k in cdr_keys_norm
        # )
        # Fv_chml_ff19sb = sum(_ff19sb_total_charge_for_residue(k) for k in heavy_residue_keys) - sum(
        #     _ff19sb_total_charge_for_residue(k) for k in light_residue_keys
        # )
        # exposed_Fv_chml_ff19sb = sum(
        #     _ff19sb_exposed_charge_for_residue(k) for k in heavy_residue_keys
        # ) - sum(_ff19sb_exposed_charge_for_residue(k) for k in light_residue_keys)

        def _atom_charge_ff19sb(atom) -> float:
            key4 = residue_key_from_atom(atom)
            return float(
                get_ff19sb_atom_charge(
                    (atom.residue_name or "").strip().upper(),
                    atom.name or "",
                    residue_atom_names=residue_atom_names_by_key.get(key4),
                )
            )

        def _atom_is_pm_like_exposed(atom) -> bool:
            key4 = residue_key_from_atom(atom)
            entry = _sasa.get(key4)
            if entry is None:
                return False
            if is_backbone_atom(atom):
                return float(getattr(entry, "main_chain_abs", 0.0) or 0.0) > 0.0
            return float(getattr(entry, "total_side_abs", 0.0) or 0.0) > 0.0

        net_charge_pm_like = 0.0
        # net_charge_cdr_pm_like = 0.0
        exposed_net_charge_pm_like = 0.0
        # exposed_net_charge_cdr_pm_like = 0.0
        heavy_charge_pm_like = 0.0
        light_charge_pm_like = 0.0
        exposed_heavy_charge_pm_like = 0.0
        exposed_light_charge_pm_like = 0.0
        for atom in pdb_atoms:
            key4 = residue_key_from_atom(atom)
            charge = _atom_charge_ff19sb(atom)
            net_charge_pm_like += charge
            # if key4 in cdr_keys_norm:
            #     net_charge_cdr_pm_like += charge
            if key4[2] == args.heavy_chain:
                heavy_charge_pm_like += charge
            elif key4[2] == args.light_chain:
                light_charge_pm_like += charge

            if not _atom_is_pm_like_exposed(atom):
                continue
            exposed_net_charge_pm_like += charge
            # if key4 in cdr_keys_norm:
            #     exposed_net_charge_cdr_pm_like += charge
            if key4[2] == args.heavy_chain:
                exposed_heavy_charge_pm_like += charge
            elif key4[2] == args.light_chain:
                exposed_light_charge_pm_like += charge

        # Fv_chml_pm_like = heavy_charge_pm_like - light_charge_pm_like
        # exposed_Fv_chml_pm_like = exposed_heavy_charge_pm_like - exposed_light_charge_pm_like

        _buried_tsr = [
            getattr(_sasa[k], "total_side_abs", None) for k in buried_keys if k in _sasa
        ]
        _exposed_tsr = [
            getattr(_sasa[k], "total_side_abs", None) for k in exposed_keys if k in _sasa
        ]
        _buried_tsr = [v for v in _buried_tsr if v is not None]
        _exposed_tsr = [v for v in _exposed_tsr if v is not None]
        # avg_total_side_abs_buried = (
        #     sum(_buried_tsr) / len(_buried_tsr) if _buried_tsr else 0.0
        # )
        # avg_total_side_abs_exposed = (
        #     sum(_exposed_tsr) / len(_exposed_tsr) if _exposed_tsr else 0.0
        # )

        # Sum of absolute SASA (side + main chain, Å²) over all hydrophobic residues with SASA.
        # total_hydrophobic_abs_sasa = sum(
        #     (float(getattr(_sasa[k], "total_side_abs", 0.0)) or 0.0)
        #     + (float(getattr(_sasa[k], "main_chain_abs", 0.0)) or 0.0)
        #     for k in hydrophobic_keys
        #     if k in _sasa
        # )
        # Match ProperMAb semantics more closely by using FreeSASA's residue-level
        # non-polar and all-polar areas directly instead of residue-class totals.
        hyd_asa_total = sum(
            float(getattr(entry, "non_polar_abs", 0.0)) or 0.0
            for entry in _sasa.values()
        )
        hph_asa_total = sum(
            float(getattr(entry, "all_polar_abs", 0.0)) or 0.0
            for entry in _sasa.values()
        )

        # Sum of side-chain absolute SASA (Å²) by residue type over a residue key set.
        def _category_side_abs_sum(residue_keys: Set[ResKey4], res_set) -> float:
            return sum(
                float(getattr(_sasa[k], "total_side_abs", 0.0)) or 0.0
                for k in residue_keys
                if k in _sasa and k[0] in res_set
            )

        _cat = [
            ("aromatic", AROMATIC_RESIDUES),
            ("negative", CHARGE_FRACTION_NEGATIVE_RESIDUES),
            ("positive", CHARGE_FRACTION_POSITIVE_RESIDUES),
            # ("polar", POLAR_RESIDUES),
            ("hydrophobic", HYDROPHOBIC_RESIDUES),
        ]
        all_sasa_keys: Set[ResKey4] = set(_sasa.keys())
        total_side_abs_sums: Dict[str, float] = {}
        for name, res_set in _cat:
            total_side_abs_sums[f"{name}_exposed_total_side_abs_sasa"] = _category_side_abs_sum(
                exposed_keys, res_set
            )
            total_side_abs_sums[f"{name}_all_total_side_abs_sasa"] = _category_side_abs_sum(
                all_sasa_keys, res_set
            )

        # def _safe_ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
        #     if numer is None or denom is None:
        #         return None
        #     denom_f = float(denom)
        #     if denom_f == 0.0 or (denom_f != denom_f):
        #         return None
        #     numer_f = float(numer)
        #     if numer_f != numer_f:
        #         return None
        #     return numer_f / denom_f

        # ratio_hydrophobic_exposed_total_side_abs_sasa_to_polar_negative_positive_exposed_total_side_abs_sasa = _safe_ratio(
        #     total_side_abs_sums.get("hydrophobic_exposed_total_side_abs_sasa"),
        #     (
        #         (total_side_abs_sums.get("polar_exposed_total_side_abs_sasa") or 0.0)
        #         + (total_side_abs_sums.get("negative_exposed_total_side_abs_sasa") or 0.0)
        #         + (total_side_abs_sums.get("positive_exposed_total_side_abs_sasa") or 0.0)
        #     ),
        # )

        # Side-chain absolute SASA (Å²) by residue type among inter-chain interface residues only.
        # inter_chain_side_abs_fraction_sums: Dict[str, float] = {}
        # for name, res_set in _cat:
        #     inter_chain_side_abs_fraction_sums[f"{name}_inter_chain_total_side_abs_sasa"] = (
        #         _category_side_abs_sum(interface_keys, res_set)
        #     )

        sap_scores = compute_sap_shell_synergy_scores(
            pdb_atoms,
            ctx.sasa_residue,
            pka_output_data,
            args.pH,
            d_cutoff=10.0,
        )
        # sap_hydro_score = sap_scores["sap_hydro_score"]
        # sap_pos_charge_score = sap_scores["sap_pos_charge_score"]
        # sap_neg_charge_score = sap_scores["sap_neg_charge_score"]
        pos_patch_area = pos_exposed_clusters_total_side_abs_sasa
        neg_patch_area = neg_exposed_clusters_total_side_abs_sasa
        hyd_patch_area = hydro_exposed_clusters_total_side_abs_sasa
        pos_patch_area_cdr = pos_cdr_vicinity_clusters_total_side_abs_sasa
        neg_patch_area_cdr = neg_cdr_vicinity_clusters_total_side_abs_sasa
        hyd_patch_area_cdr = hydro_cdr_vicinity_clusters_total_side_abs_sasa
        # atom_patch_pm_like_fast = compute_atom_patch_area_cdr_pm_like_fast(
        #     pdb_atoms,
        #     ctx.atom_sasa,
        #     cdr_keys_norm,
        # )
        # pos_patch_area_cdr_pm_like_fast = atom_patch_pm_like_fast["pos_patch_area_cdr_pm_like_fast"]
        # neg_patch_area_cdr_pm_like_fast = atom_patch_pm_like_fast["neg_patch_area_cdr_pm_like_fast"]
        # hyd_patch_area_cdr_pm_like_fast = atom_patch_pm_like_fast["hyd_patch_area_cdr_pm_like_fast"]

        # # Single pass: build all_residues and heavy-atoms-by-chain (for inter-chain interface)
        # all_residues = set()
        # by_chain_heavy = defaultdict(list)
        # for atom in pdb_atoms:
        #     all_residues.add(residue_key_from_atom(atom))
        #     if getattr(atom, "element", "X") != "H":
        #         by_chain_heavy[atom.chain].append(atom)
        # # Fill interface cache from same data so no second structure pass
        # get_inter_chain_interface_residues(args.pdb_file, by_chain_heavy=by_chain_heavy)

        # all_residues.update(aromatic_densities.keys())
        # all_residues.update(_to_4(k) for k in dssp_data.keys())
        # all_residues.update(_to_4(k) for k in sasa_output_data.keys())
        # all_residues.update(pka_output_data.keys())
            
        # # Build per-residue DataFrame (in memory) and compute aggregated structure-level descriptors
        # sorted_residues = sorted(all_residues, key=lambda x: (x[2], x[1], x[3] if len(x) == 4 else ''))
        # df = _build_output(
        #     sorted_residues=sorted_residues,
        #     dssp_data=dssp_data,
        #     sasa_output_data=sasa_output_data,
        #     pka_output_data=pka_output_data,
        #     psh_all_surface=psh_all_surface,
        #     psh_cdr_vicinity=psh_cdr_vicinity,
        #     ppc_all_surface=ppc_all_surface,
        #     ppc_cdr_vicinity=ppc_cdr_vicinity,
        #     pnc_all_surface=pnc_all_surface,
        #     pnc_cdr_vicinity=pnc_cdr_vicinity,
        #     dipole_moment_magnitude=dipole_moment_magnitude,
        #     largest_hbond_component_size_val=largest_hbond_component_size_val,
        #     net_charge=net_charge,
        #     protein_pi=protein_pi,
        #     # scm_score_val=scm_score_val,
        #     weighted_scm_score_by_pH=weighted_scm_score_by_pH,
        #     inter_chain_buried_sasa=inter_chain_buried_sasa,
        #     net_charge_by_pH=net_charge_by_pH,
        #     hbond_inter_chain=hbond_inter_chain,
        #     # salt_bridge_inter_chain=salt_bridge_inter_chain,
        #     hbond_energy_dssp_inter_chain=hbond_energy_dssp_inter_chain,
        #     sap_hydro_score=sap_hydro_score,
        #     sap_pos_charge_score=sap_pos_charge_score,
        #     sap_neg_charge_score=sap_neg_charge_score,
        #     )
        # aggregated = compute_downstream_descriptors(
        #     df,
        #     inter_chain_buried_sasa=inter_chain_buried_sasa,
        #     pdb_path=args.pdb_file,
        #     heavy_chain_id=args.heavy_chain,
        #     light_chain_id=args.light_chain,
        #     pH=args.pH,
        # )

        # Surface clustering, Ripley K, and pair statistics are structure-level;
        # ensure they are present explicitly in the aggregated JSON output.
        aggregated = {}
        
        aggregated.update(
            {
                        # All cluster and surface summary metrics under a single flat key
                        "surface": {
                            # DBSCAN cluster summaries
                            # "negative_exposed_clusters_total_side_abs_sasa": neg_exposed_clusters_total_side_abs_sasa,
                            # "positive_exposed_clusters_total_side_abs_sasa": pos_exposed_clusters_total_side_abs_sasa,
                            # "aromatic_exposed_clusters_total_side_abs_sasa": aromatic_exposed_clusters_total_side_abs_sasa,
                            # "hydrophobic_exposed_clusters_total_side_abs_sasa": hydro_exposed_clusters_total_side_abs_sasa,
                            # "polar_exposed_clusters_total_side_abs_sasa": polar_exposed_clusters_total_side_abs_sasa,
                            # "negative_cdr_vicinity_clusters_total_side_abs_sasa": neg_cdr_vicinity_clusters_total_side_abs_sasa,
                            # "positive_cdr_vicinity_clusters_total_side_abs_sasa": pos_cdr_vicinity_clusters_total_side_abs_sasa,
                            # "aromatic_cdr_vicinity_clusters_total_side_abs_sasa": aromatic_cdr_vicinity_clusters_total_side_abs_sasa,
                            # "hydrophobic_cdr_vicinity_clusters_total_side_abs_sasa": hydro_cdr_vicinity_clusters_total_side_abs_sasa,
                            # "polar_cdr_vicinity_clusters_total_side_abs_sasa": polar_cdr_vicinity_clusters_total_side_abs_sasa,
                            "neg_cluster_entropy": neg_cluster_side_abs_sasa_entropy,
                            "pos_cluster_entropy": pos_cluster_side_abs_sasa_entropy,
                            # "aromatic_exposed_cluster_side_abs_sasa_entropy_nats": aromatic_cluster_side_abs_sasa_entropy,
                            # "hydrophobic_exposed_cluster_side_abs_sasa_entropy_nats": hydro_cluster_side_abs_sasa_entropy,
                            # "polar_exposed_cluster_side_abs_sasa_entropy_nats": polar_cluster_side_abs_sasa_entropy,
                            # Ripley K and pairwise surface descriptors
                            # "ripley_k_negative": ripley_k_negative,
                            # "ripley_k_positive": ripley_k_positive,
                            # "ripley_k_aromatic": ripley_k_aromatic,
                            # "ripley_k_hydrophobic": ripley_k_hydrophobic,
                            # "ripley_k_polar": ripley_k_polar,
                            **_pcf_cluster_per_shell_and_mean_per_category(pcf_cluster),
                            # **pcf_cluster_buried_aromatic_hydro,
                            # "pos_ann_index": pos_ann_index,
                            # "neg_ann_index": neg_ann_index,
                            # "aromatic_ann_index": aromatic_ann_index,
                            # PPC/PNC/PSH
                            # "psh_all_surface_exposed": psh_all_surface,
                            # "psh_cdr_vicinity": psh_cdr_vicinity,
                            # "ppc_all_surface_exposed": ppc_all_surface,
                            # "ppc_cdr_vicinity": ppc_cdr_vicinity,
                            # "pnc_all_surface_exposed": pnc_all_surface,
                            # "pnc_cdr_vicinity": pnc_cdr_vicinity,
                            "hbond_density_cdr": avg_hbond_cdr_vicinity,
                            # "avg_aromatic_weighted_rel_side_asa_exposed_over_exposed": avg_aromatic_exposed_over_exposed,
                            # "avg_hydrophobic_weighted_rel_side_asa_exposed_over_exposed": avg_hydrophobic_exposed_over_exposed,
                            # "avg_polar_weighted_rel_side_asa_exposed_over_exposed": avg_polar_exposed_over_exposed,
                            # "avg_negative_weighted_rel_side_asa_exposed_over_exposed": avg_negative_exposed_over_exposed,
                            # "avg_positive_weighted_rel_side_asa_exposed_over_exposed": avg_positive_exposed_over_exposed,
                            "aro_exposure_cdr": sum_aromatic_weighted_rel_side_asa_cdr_vicinity,
                            "hyd_exposure_cdr": sum_hydrophobic_weighted_rel_side_asa_cdr_vicinity,
                            # "sum_polar_weighted_rel_side_asa_cdr_vicinity": sum_polar_weighted_rel_side_asa_cdr_vicinity,
                            "neg_exposure_cdr": sum_negative_weighted_rel_side_asa_cdr_vicinity,
                            "pos_exposure_cdr": sum_positive_weighted_rel_side_asa_cdr_vicinity,
                            "aro_sasa_cdr": sum_aromatic_side_abs_sasa_cdr_vicinity,
                            # "sum_hydrophobic_side_abs_sasa_cdr_vicinity": sum_hydrophobic_side_abs_sasa_cdr_vicinity,
                            # "sum_polar_side_abs_sasa_cdr_vicinity": sum_polar_side_abs_sasa_cdr_vicinity,
                            "neg_sasa_cdr": sum_negative_side_abs_sasa_cdr_vicinity,
                            "pos_sasa_cdr": sum_positive_side_abs_sasa_cdr_vicinity,
                            "neg_curvature_cdr": normalized_local_curvature_negative_cdr_vicinity,
                            "pos_curvature_cdr": normalized_local_curvature_positive_cdr_vicinity,
                            "aro_curvature_cdr": normalized_local_curvature_aromatic_cdr_vicinity,
                            "hyd_curvature_cdr": normalized_local_curvature_hydrophobic_cdr_vicinity,
                            # "normalized_local_curvature_polar_cdr_vicinity": normalized_local_curvature_polar_cdr_vicinity,
                            # "avg_kd_weighted_rel_side_asa_exposed_over_exposed": avg_kd_times_total_side_rel_exposed_over_exposed,
                            "exposure_weighted_hyd_score_cdr": avg_kd_times_total_side_rel_cdr_vicinity_over_cdr_vicinity,
                            "exposure_weighted_salt_bridge_score_cdr": avg_salt_cdr_vicinity_over_cdr_vicinity,
                            # "avg_salt_weighted_rel_side_asa_exposed_over_all": avg_salt_exposed_over_all,
                            "weighted_scm_score": weighted_scm_score_by_pH,
                            # "unweighted_scm_score_by_pH": unweighted_scm_score_by_pH,
                            # "weighted_scm_pos_score_by_pH": weighted_scm_pos_score_by_pH,
                            "scm_neg": scm_by_atoms_neg,
                            "scm_pos": scm_by_atoms_pos,
                            # "scm_pm_like": scm_pm_like,
                            # scm_score_from_pka_propermab not implemented in descriptors.
                            # "scm_propermab_score_by_pH": …
                            # SAP surface scores — TEMP: only hydro + pos/neg charge for testing.
                            # To restore full ``compute_sap_shell_synergy_scores`` output, comment out
                            # the following dict spread and uncomment the block below it.
                            # **{
                            #     k: _scalar_or_dict_sum(sap_scores[k])
                            #     for k in (
                            #         "sap_hydro_score",
                            #         "sap_pos_charge_score",
                            #         "sap_neg_charge_score",
                            #     )
                            #     if k in sap_scores
                            # },
                            **{
                                k: _scalar_or_dict_sum(v)
                                for k, v in sap_scores.items()
                            },
                            "pos_patch_area": pos_patch_area,
                            "neg_patch_area": neg_patch_area,
                            "hyd_patch_area": hyd_patch_area,
                            "pos_patch_area_cdr": pos_patch_area_cdr,
                            "neg_patch_area_cdr": neg_patch_area_cdr,
                            "hyd_patch_area_cdr": hyd_patch_area_cdr,
                            # "pos_patch_area_cdr_pm_like_fast": pos_patch_area_cdr_pm_like_fast,
                            # "neg_patch_area_cdr_pm_like_fast": neg_patch_area_cdr_pm_like_fast,
                            # "hyd_patch_area_cdr_pm_like_fast": hyd_patch_area_cdr_pm_like_fast,
                            # "pos_ann_index_pm_like": pos_ann_index_pm_like,
                            # "neg_ann_index_pm_like": neg_ann_index_pm_like,
                            # "aromatic_ann_index_pm_like": aromatic_ann_index_pm_like,
                            # "ripley_k_negative_pm_like": ripley_k_negative_pm_like,
                            # "ripley_k_positive_pm_like": ripley_k_positive_pm_like,
                            # "ripley_k_aromatic_pm_like": ripley_k_aromatic_pm_like,
                            # "ripley_k_hydrophobic_pm_like": ripley_k_hydrophobic_pm_like,
                            # "ripley_k_polar_pm_like": ripley_k_polar_pm_like,
                            # "avg_total_side_abs_exposed": avg_total_side_abs_exposed,
                            # "total_hydrophobic_abs_sasa": total_hydrophobic_abs_sasa,
                            **total_side_abs_sums,
                            # "ratio_hydrophobic_exposed_total_side_abs_sasa_to_polar_negative_positive_exposed_total_side_abs_sasa": ratio_hydrophobic_exposed_total_side_abs_sasa_to_polar_negative_positive_exposed_total_side_abs_sasa,

                        },
                        # H-bond density / energy averages, grouped separately
                        "core": {
                            # "number_of_hbonds": number_of_hbonds,
                            # "largest_hbond_component_size": largest_hbond_component_size_val,
                            # Per-end weights: side-chain rel SASA (×100) / backbone rel for backbone atoms.
                            # "avg_hbond_weighted_rel_side_asa_beta_over_all": avg_hbond_beta_over_all,
                            # "avg_hbond_weighted_rel_side_asa_inter_chain": avg_hbond_inter_chain,
                            # "avg_hbond_energy_all": avg_hbond_energy,
                            # "avg_hbond_energy_buried_over_all": avg_hbond_energy_buried_over_all,
                            # "avg_hbond_energy_beta_over_all": avg_hbond_energy_beta_over_all,
                            "hbond_energy_density": avg_hbond_energy_dssp_weighted,
                            # "avg_hbond_energy_dssp_weighted_buried_over_all": avg_hbond_energy_dssp_weighted_buried_over_all,
                            # "avg_hbond_energy_dssp_weighted_beta_over_all": avg_hbond_energy_dssp_weighted_beta_over_all,
                            # "avg_aromatic_weighted_rel_side_asa_beta_over_all": avg_aromatic_beta_over_all,
                            # "avg_hydrophobic_weighted_rel_side_asa_beta_over_all": avg_hydrophobic_beta_over_all,
                            # "fraction_gln_asn_beta_sheet": fraction_gln_asn_beta_sheet,
                            # "hydrophobic_beta_sheet_kyte_doolittle_sum": hydrophobic_beta_sheet_kyte_doolittle_sum,
                            # "avg_wcn_interface_over_all": avg_wcn_interface_over_all,
                            # "avg_salt_weighted_rel_side_asa_beta_over_all": avg_salt_beta_over_all,
                            # "avg_salt_weighted_rel_side_asa_inter_chain": avg_salt_interface,
                            # "fraction_negative_buried": fraction_negative_buried,
                            # "fraction_positive_buried": fraction_positive_buried,
                            "inter_chain_buried_sasa": inter_chain_buried_sasa,
                            # "inter_chain_side_abs_fraction_sums": inter_chain_side_abs_fraction_sums,

                        },
                        # Density numerators: side-chain rel SASA weights (linear, sqrt_weights=False).
                        "general": {
                            "dipole_moment": dipole_moment_magnitude,
                            # "hydrophobic_moment_magnitude": hydrophobic_moment_magnitude,
                            # "net_charge": net_charge,
                            # "net_charge_pm_like": net_charge_pm_like,
                            "net_charge_ff19sb": net_charge_ff19sb,
                            "protein_pi": protein_pi,
                            "net_charge_by_pH": net_charge_by_pH,
                            # "net_charge_cdr_from_pka": net_charge_cdr_from_pka,
                            # "net_charge_cdr_pm_like": net_charge_cdr_pm_like,
                            # "net_charge_cdr_ff19sb": net_charge_cdr_ff19sb,
                            # "exposed_net_charge": exposed_net_charge,
                            # "exposed_net_charge_cdr": exposed_net_charge_cdr,
                            # "exposed_net_charge_simple": exposed_net_charge_simple,
                            # "exposed_net_charge_cdr_simple": exposed_net_charge_cdr_simple,
                            # "exposed_net_charge_pm_like": exposed_net_charge_pm_like,
                            # "exposed_net_charge_cdr_pm_like": exposed_net_charge_cdr_pm_like,
                            # "exposed_net_charge_ff19sb": exposed_net_charge_ff19sb,
                            # "exposed_net_charge_cdr_ff19sb": exposed_net_charge_cdr_ff19sb,
                            # "heavy_charge_pH_7_5": heavy_charge,
                            # "light_charge_pH_7_5": light_charge,
                            # "Fv_chml": Fv_chml,
                            # "Fv_chml_pm_like": Fv_chml_pm_like,
                            # "exposed_Fv_chml": exposed_Fv_chml,
                            # "exposed_Fv_chml_pm_like": exposed_Fv_chml_pm_like,
                            # "Fv_chml_ff19sb": Fv_chml_ff19sb,
                            # "exposed_Fv_chml_ff19sb": exposed_Fv_chml_ff19sb,
                            "asymmetry_score": asymmetry_score,
                            # "asymmetry_substract": asymmetry_substract,
                            "hyd_asa_total": hyd_asa_total,
                            "hph_asa_total": hph_asa_total,
                            "cdr3_length": cdr3_length,
                            # "n_gly_cdr_vicinity": n_gly_cdr_vicinity,
                            # "n_pro_cdr_vicinity": n_pro_cdr_vicinity,
                            # "n_aromatic_cdr_vicinity": n_aromatic_cdr_vicinity,
                            # "n_positive_cdr_vicinity": n_positive_cdr_vicinity,
                            # "n_negative_cdr_vicinity": n_negative_cdr_vicinity,
                            # "n_gln_asn_cdr_vicinity": n_gln_asn_cdr_vicinity,
                            "gly_in_cdr": fraction_gly_in_cdr,
                            "pro_in_cdr": fraction_pro_in_cdr,
                            "aro_in_cdr": fraction_aromatic_in_cdr,
                            "pos_in_cdr": fraction_positive_in_cdr,
                            "neg_in_cdr": fraction_negative_in_cdr,
                            "gln_asn_in_cdr": fraction_gln_asn_in_cdr,
                            # "ratio_avg_hydrophobic_to_negative_positive_polar_exposed": ratio_avg_hydrophobic_to_negative_positive_polar_exposed,
                            "n_salt_bridges": number_of_salt_bridges,
                            "hyd_score": kyte_doolittle_sum_all,
                        },
                    },
                )

        # Motif counting should be per-chain (H and L separately) and then summed.
        # This avoids creating artificial motifs at the H/L concatenation boundary.
        #
        # For region-specific counts (CDR vicinity / beta / exposed / inter-chain), operate on contiguous
        # fragments in the original chain order, rather than a compressed subsequence,
        # to avoid artificial adjacency across gaps.
        chain_order = [args.heavy_chain, args.light_chain]
        chain_seqs_and_maps: List[Tuple[str, str, Dict[Tuple[str, int, str, str], int]]] = []
        for ch in chain_order:
            seq_ch, map_ch = get_full_sequence_with_index_map_from_pdb(args.pdb_file, chain_order=[ch])
            if seq_ch:
                chain_seqs_and_maps.append((ch, seq_ch, map_ch))

        if chain_seqs_and_maps:
                sequence_motives = aggregated.setdefault("sequence_motives", {})
                # full_seq_per_chain = [seq for _ch, seq, _m in chain_seqs_and_maps]

                # sequence_motives["n_motif_AsnGly"] = count_motif_overlapping(full_seq_per_chain, "NG")
                # sequence_motives["n_motif_AspSer"] = count_motif_overlapping(full_seq_per_chain, "DS")
                # sequence_motives["n_motif_AspAsp"] = count_motif_overlapping(full_seq_per_chain, "DD")
                # sequence_motives["n_motif_AspThr"] = count_motif_overlapping(full_seq_per_chain, "DT")
                # sequence_motives["n_motif_AspGlu"] = count_motif_overlapping(full_seq_per_chain, "DE")

                def _contiguous_fragments_for_keyset_per_chain(
                    key_set,
                ) -> Dict[str, List[str]]:
                    """
                    Return per-chain contiguous fragments (1-letter strings) for residues in key_set.
                    Fragment boundaries follow gaps in original chain indices.
                    """
                    if not key_set:
                        return {}
                    out: Dict[str, List[str]] = {}
                    for ch, seq_ch, map_ch in chain_seqs_and_maps:
                        if not seq_ch:
                            continue
                        present = [False] * len(seq_ch)
                        any_present = False
                        for k in key_set:
                            idx = map_ch.get(k)
                            if idx is not None and 0 <= idx < len(seq_ch):
                                present[idx] = True
                                any_present = True
                        if not any_present:
                            continue
                        frags: List[str] = []
                        i = 0
                        n = len(seq_ch)
                        while i < n:
                            if not present[i]:
                                i += 1
                                continue
                            j = i
                            while j < n and present[j]:
                                j += 1
                            frags.append(seq_ch[i:j])
                            i = j
                        if frags:
                            out[ch] = frags
                    return out

                def _flatten_fragments(frags_by_chain: Dict[str, List[str]]) -> List[str]:
                    if not frags_by_chain:
                        return []
                    flat: List[str] = []
                    for ch in chain_order:
                        xs = frags_by_chain.get(ch)
                        if xs:
                            flat.extend(xs)
                    return flat

                fragments_cdr_vicinity = _contiguous_fragments_for_keyset_per_chain(
                    cdr_vicinity_keys
                )
                # fragments_beta = _contiguous_fragments_for_keyset_per_chain(beta_keys)
                # fragments_exposed = _contiguous_fragments_for_keyset_per_chain(exposed_keys)
                # fragments_inter = _contiguous_fragments_for_keyset_per_chain(interface_keys)

                cdr_vicinity_frags_flat = _flatten_fragments(fragments_cdr_vicinity)
                # beta_frags_flat = _flatten_fragments(fragments_beta)
                # exposed_frags_flat = _flatten_fragments(fragments_exposed)
                # inter_frags_flat = _flatten_fragments(fragments_inter)

                # sequence_motives[f"n_motif_AsnGly_cdr_vicinity"] = count_motif_overlapping(
                #     cdr_vicinity_frags_flat, "NG"
                # )
                # sequence_motives[f"n_motif_AsnGly_inter_chain"] = count_motif_overlapping(inter_frags_flat, "NG")
                # sequence_motives[f"n_motif_AsnGly_beta_sheet"] = count_motif_overlapping(beta_frags_flat, "NG")

                # sequence_motives[f"n_motif_AspSer_cdr_vicinity"] = count_motif_overlapping(
                #     cdr_vicinity_frags_flat, "DS"
                # )
                # sequence_motives[f"n_motif_AspSer_inter_chain"] = count_motif_overlapping(inter_frags_flat, "DS")
                # sequence_motives[f"n_motif_AspSer_beta_sheet"] = count_motif_overlapping(beta_frags_flat, "DS")

                # sequence_motives[f"n_motif_AspThr_cdr_vicinity"] = count_motif_overlapping(
                #     cdr_vicinity_frags_flat, "DT"
                # )
                # sequence_motives[f"n_motif_AspThr_inter_chain"] = count_motif_overlapping(inter_frags_flat, "DT")
                # sequence_motives[f"n_motif_AspThr_beta_sheet"] = count_motif_overlapping(beta_frags_flat, "DT")

                sequence_motives[f"n_motif_AspAsp_cdr"] = count_motif_overlapping(
                    cdr_vicinity_frags_flat, "DD"
                )
                # Aromatic / His / basic dyads in CDR vicinity (1-letter, overlapping counts; per-chain fragments).
                # OK:
                sequence_motives["count_YY_cdr"] = count_motif_overlapping(
                    cdr_vicinity_frags_flat, "YY"
                )
                # sequence_motives["count_YH_plus_HY_cdr_vicinity"] = (
                #     count_motif_overlapping(cdr_vicinity_frags_flat, "YH")
                #     + count_motif_overlapping(cdr_vicinity_frags_flat, "HY")
                # )
                # sequence_motives["count_YK_plus_KY_cdr_vicinity"] = (
                #     count_motif_overlapping(cdr_vicinity_frags_flat, "YK")
                #     + count_motif_overlapping(cdr_vicinity_frags_flat, "KY")
                # )
                
                # OK:!!
                # sequence_motives["count_YR_plus_RY_cdr_vicinity"] = (
                #     count_motif_overlapping(cdr_vicinity_frags_flat, "YR")
                #     + count_motif_overlapping(cdr_vicinity_frags_flat, "RY")
                # )
                # sequence_motives["count_HH_cdr_vicinity"] = count_motif_overlapping(
                #     cdr_vicinity_frags_flat, "HH"
                # )
                # sequence_motives["count_HK_plus_KH_cdr_vicinity"] = (
                #     count_motif_overlapping(cdr_vicinity_frags_flat, "HK")
                #     + count_motif_overlapping(cdr_vicinity_frags_flat, "KH")
                # )
                # sequence_motives["count_HR_plus_RH_cdr_vicinity"] = (
                #     count_motif_overlapping(cdr_vicinity_frags_flat, "HR")
                #     + count_motif_overlapping(cdr_vicinity_frags_flat, "RH")
                # )
                # OK in general !!
                sequence_motives[f"n_motif_AspGlu"] = count_motif_overlapping(
                    cdr_vicinity_frags_flat, "DE"
                )

                # for motif, key in [
                #     ("DS", "AspSer"),
                #     ("DD", "AspAsp"),
                #     ("DT", "AspThr"),
                #     ("DE", "AspGlu"),
                # ]:
                #     sequence_motives[f"n_motif_{key}_cdr_vicinity"] = count_motif_overlapping(cdr_vicinity_frags_flat, motif)
                #     sequence_motives[f"n_motif_{key}_exposed"] = count_motif_overlapping(exposed_frags_flat, motif)
                # sequence_motives[f"n_motif_AsnGly_cdr_vicinity"] = count_motif_overlapping(cdr_vicinity_frags_flat, "NG")
                # sequence_motives[f"n_motif_AspGlu_cdr_vicinity"] = count_motif_overlapping(cdr_vicinity_frags_flat, "DE")

                # for motif, key in [
                #     ("NG", "AsnGly"),
                #     ("DS", "AspSer"),
                #     ("DT", "AspThr"),
                # ]:
                #     sequence_motives[f"n_motif_{key}_beta_sheet"] = count_motif_overlapping(beta_frags_flat, motif)
                #     sequence_motives[f"n_motif_{key}_inter_chain"] = count_motif_overlapping(inter_frags_flat, motif)
                # sequence_motives[f"n_motif_AspAsp_inter_chain"] = count_motif_overlapping(inter_frags_flat, "DD")
                # sequence_motives[f"n_motif_AspGlu_inter_chain"] = count_motif_overlapping(inter_frags_flat, "DE")

                # Structure-level motif "side-ASA sum" (relative side SASA, percent-scaled):
                # for each occurrence of a motif (e.g. "DG"), take residues i and i+1 in the
                # original chain, and add their side-chain SASA values (relative × 100).
                # Computed per-chain and summed across chains (no H/L boundary artifacts).
                sasa_residue = ctx.sasa_residue
                if sasa_residue:
                    # Precompute per-chain side-SASA arrays aligned to sequence indices
                    chain_side_sasa: Dict[str, List[float]] = {}
                    for ch, seq_ch, map_ch in chain_seqs_and_maps:
                        inv_map = {idx: k for k, idx in map_ch.items()}
                        arr = [0.0] * len(seq_ch)
                        for i in range(len(seq_ch)):
                            key4 = inv_map.get(i)
                            if key4 is not None:
                                arr[i] = float(residue_side_sasa(key4, sasa_residue))
                        chain_side_sasa[ch] = arr

                    def _motif_side_asa_sum(motif: str) -> float:
                        if not motif or len(motif) != 2:
                            return 0.0
                        a, b = motif[0], motif[1]
                        total = 0.0
                        for ch, seq_ch, _map_ch in chain_seqs_and_maps:
                            if len(seq_ch) < 2:
                                continue
                            asa = chain_side_sasa.get(ch)
                            if asa is None or len(asa) != len(seq_ch):
                                continue
                            # Scan matches (O(L)); L ~ chain length.
                            for i in range(len(seq_ch) - 1):
                                if seq_ch[i] == a and seq_ch[i + 1] == b:
                                    total += float(asa[i] + asa[i + 1])
                        return float(total)

                    for motif, key in [
                        # ("NG", "AsnGly"),
                        # ("DS", "AspSer"),
                        ("DD", "AspAsp"),
                        # ("DT", "AspThr"),
                        ("DE", "AspGlu"),
                    ]:
                        sequence_motives[f"sum_sasa_{key}"] = _motif_side_asa_sum(motif)

        def _sanitize_for_json(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sanitize_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize_for_json(x) for x in obj]
            if isinstance(obj, float):
                return round(obj, 3)
            try:
                if pd.isna(obj):
                    return None
            except (TypeError, ValueError):
                pass
            if hasattr(obj, "dtype") and hasattr(obj, "item"):
                try:
                    if "int" in str(obj.dtype):
                        return int(obj)
                    if "float" in str(obj.dtype):
                        return None if pd.isna(obj) else round(float(obj), 3)
                except (ValueError, TypeError):
                    pass
            return obj

        aggregated_clean = _sanitize_for_json(aggregated)
        # JSON output (only supported format).
        if args.output:
            Path(args.output).write_text(json.dumps(aggregated_clean, indent=2))
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(aggregated_clean, indent=2))

    
    except Exception as e:
        print(f"Error calculating developability descriptors: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

