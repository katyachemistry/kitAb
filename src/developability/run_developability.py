#!/usr/bin/env python3
"""
Command-line interface for calculating developability descriptors (H-bonds, salt bridges, aromatic, WCN, SASA, etc.).

Usage:
    python run_developability.py <pdb_file> <sasa_file> [options]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

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
    compute_residue_density_raw,
    compute_hbond_density_raw,
    largest_hbond_component_size,
    net_charge_from_pka,
    pi_from_pka,
    scm_score_from_pka,
    sum_total_side_rel_within_cutoff,
    calculate_weighted_contact_number_average,
    parse_dssp,
    calculate_hbond_energy_density_dssp_backbone_only_average,
    calculate_hbond_energy_dssp_backbone_only_unweighted_average,
    compute_residue_DBSCAN_cluster_labels,
    summarize_dbscan_clusters,
    _get_atoms_for_path,
    get_inter_chain_interface_residues,
    compute_surface_ripley_descriptors,
    compute_surface_pair_descriptors,
    calculate_relative_contact_order,
    compute_dipole_moment_magnitude,
    compute_inter_chain_buried_sasa,
    NEGATIVE_CHARGED_RESIDUES,
    POSITIVE_CHARGED_RESIDUES,
    POLAR_RESIDUES,
    HYDROPHOBIC_RESIDUES,
    AROMATIC_RESIDUES
)
from developability.descriptors import (
    count_motif_overlapping,
    get_full_sequence_with_index_map_from_pdb,
)
from developability.descriptors import GLN_ASN_RESIDUES, KYTE_DOOLITTLE
from developability.structure_context import StructureContext
from developability.descriptor_utils import (
    get_residue_region_map,
    get_exposed_residues,
    get_aromatic_residue_keys,
    get_residue_keys_by_type,
    residue_side_sasa,
    _count_residues_in_pdb,
)
from utils.parsers import parse_pka, get_pka_file_path, residue_key_from_atom, get_sasa_total

NET_CHARGE_PHS = [4, 5, 6, 7, 8, 9, 10]


def _to_4(key):
    return (key[0], key[1], key[2], key[3]) if len(key) == 4 else (key[0], key[1], key[2], "")


def _lookup(d, key):
    return d.get(key) or (d.get((key[0], key[1], key[2], "")) if len(key) == 4 else None)


def _scalar_or_dict_sum(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict) and value:
        return float(sum(value.values()))
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (f != f) else f  # exclude nan
    return None


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
        default=7.4,
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
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv'],
        default='json',
        help='Output format for aggregated descriptors (default: json). Ignored if --per-residue is used.'
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
    
    dssp_data = {}
    if args.dssp_file:
        if not Path(args.dssp_file).exists():
            print(f"Warning: DSSP file not found: {args.dssp_file}. Continuing without DSSP data.", file=sys.stderr)
        else:
            dssp_data = parse_dssp(args.dssp_file, pdb_atoms)
            if dssp_data:
                print(f"Parsed DSSP data for {len(dssp_data)} residues", file=sys.stderr)
    
    # sasa_output_data = {}
    # sasa_residue_data = {}
    # if args.sasa_file:
    #     if not Path(args.sasa_file).exists():
    #         print(
    #             f"Warning: SASA file not found: {args.sasa_file}. "
    #             f"Continuing without SASA output data.",
    #             file=sys.stderr,
    #         )
    #     else:
    #         try:
    #             ctx_sasa = StructureContext(args.pdb_file, sasa_path=args.sasa_file)
    #             sasa_output_data = ctx_sasa.sasa_output
    #             sasa_residue_data = ctx_sasa.sasa_residue
    #             if sasa_output_data:
    #                 print(
    #                     f"Parsed SASA output data for {len(sasa_output_data)} residues",
    #                     file=sys.stderr,
    #                 )
    #         except Exception as e:
    #             print(
    #                 f"Warning: failed to parse SASA output data from {args.sasa_file}: {e}. "
    #                 f"Continuing without SASA output data.",
    #                 file=sys.stderr,
    #             )
    
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
        exposed_keys: Set[ResKey4] = set()
        if not sasa_failed:
            exposed_flags = get_exposed_residues(ctx.sasa_residue, 0.25)
            exposed_keys = {key for key, is_exposed in exposed_flags.items() if is_exposed}
        beta_keys = {
            _to_4(key)
            for key, entry in (dssp_data or {}).items()
            if entry.get("secondary_structure") == "E"
        }
        interface_keys = get_inter_chain_interface_residues(args.pdb_file)
        buried_keys: Set[ResKey4] = set()
        if not sasa_failed:
            buried_keys = set(ctx.sasa_residue.keys()) - exposed_keys
        # CDR composition (residue name = key[0])
        n_cdr = len(cdr_keys)
        if n_cdr > 0:
            cdr_total_length = n_cdr
            fraction_gly_CDRs = sum(1 for k in cdr_keys if k[0] == "GLY") / n_cdr
            fraction_pro_CDRs = sum(1 for k in cdr_keys if k[0] == "PRO") / n_cdr
            fraction_aromatic_CDRs = sum(1 for k in cdr_keys if k[0] in AROMATIC_RESIDUES) / n_cdr
            fraction_gln_asn_CDRs = sum(1 for k in cdr_keys if k[0] in GLN_ASN_RESIDUES) / n_cdr
            hydro_cdr = sum(1 for k in cdr_keys if k[0] in HYDROPHOBIC_RESIDUES)
            polar_cdr = sum(1 for k in cdr_keys if k[0] in POLAR_RESIDUES)
            ratio_hydrophobic_to_polar_CDRs = (hydro_cdr / polar_cdr) if polar_cdr > 0 else None
        else:
            cdr_total_length = None
            fraction_gly_CDRs = fraction_pro_CDRs = fraction_aromatic_CDRs = fraction_gln_asn_CDRs = None
            ratio_hydrophobic_to_polar_CDRs = None

        # Fraction buried and composition of buried residues
        n_total = None if sasa_failed else len(ctx.sasa_residue)
        n_buried = None if sasa_failed else len(buried_keys)
        fraction_buried = (
            (n_buried / n_total) if (n_total and n_buried is not None and n_total > 0) else None
        )
        if n_buried > 0:
            fraction_hydrophobic_buried = sum(1 for k in buried_keys if k[0] in HYDROPHOBIC_RESIDUES) / n_buried
            fraction_negative_buried = sum(1 for k in buried_keys if k[0] in NEGATIVE_CHARGED_RESIDUES) / n_buried
            fraction_positive_buried = sum(1 for k in buried_keys if k[0] in POSITIVE_CHARGED_RESIDUES) / n_buried
        else:
            fraction_hydrophobic_buried = fraction_negative_buried = fraction_positive_buried = None

        # Beta-sheet composition and Kyte-Doolittle sum
        n_beta = len(beta_keys)
        if n_beta > 0:
            fraction_hydrophobic_beta_sheet = sum(1 for k in beta_keys if k[0] in HYDROPHOBIC_RESIDUES) / n_beta
            fraction_gln_asn_beta_sheet = sum(1 for k in beta_keys if k[0] in GLN_ASN_RESIDUES) / n_beta
            hydrophobic_beta_sheet_kyte_doolittle_sum = sum(KYTE_DOOLITTLE.get(k[0], 0.0) for k in beta_keys)
        else:
            fraction_hydrophobic_beta_sheet = fraction_gln_asn_beta_sheet = None
            hydrophobic_beta_sheet_kyte_doolittle_sum = 0.0

        # Kyte-Doolittle sum across the whole parsed sequence (all residues in structure)
        # (Keys use 3-letter residue names, matching KYTE_DOOLITTLE mapping.)
        kyte_doolittle_sum_all = (
            None
            if sasa_failed
            else sum(KYTE_DOOLITTLE.get(k[0], 0.0) for k in ctx.sasa_residue.keys())
        )
        kyte_doolittle_mean_all = (
            None
            if sasa_failed or not n_total
            else (float(kyte_doolittle_sum_all) / n_total)  # type: ignore[arg-type]
        )

        hydrophobic_keys = get_residue_keys_by_type(
            args.pdb_file, HYDROPHOBIC_RESIDUES
        )
        polar_keys = get_residue_keys_by_type(args.pdb_file, POLAR_RESIDUES)
        negative_keys = get_residue_keys_by_type(args.pdb_file, NEGATIVE_CHARGED_RESIDUES)
        positive_keys = get_residue_keys_by_type(args.pdb_file, POSITIVE_CHARGED_RESIDUES)
        aromatic_keys = get_aromatic_residue_keys(args.pdb_file)

        # Salt bridge densities (detect once, reuse for counts and all averages)
        salt_bridges = detect_salt_bridges(args.pdb_file, args.sasa_file, args.pka_file, args.pH)
        number_of_salt_bridges = len(salt_bridges)
        salt_bridge_residues: Set[Tuple[str, int, str, str]] = set()
        if salt_bridges:
            for (pos_key, neg_key) in salt_bridges.keys():
                salt_bridge_residues.add(pos_key)
                salt_bridge_residues.add(neg_key)

        avg_salt = calculate_salt_bridge_density_average(
            args.pdb_file, args.sasa_file, args.pka_file, args.pH, salt_bridges=salt_bridges
        )

        avg_salt_cdr = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            args.pH,
            residues_for_density=cdr_keys,
            residues_for_average=cdr_keys,
            salt_bridges=salt_bridges,
        )

        avg_salt_exposed_over_all = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            args.pH,
            residues_for_density=exposed_keys,
            salt_bridges=salt_bridges,
            # residues_for_average=exposed_keys,
        )

        avg_salt_beta_over_all = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            args.pH,
            residues_for_density=beta_keys,
            salt_bridges=salt_bridges,
            # residues_for_average=beta_keys,
        )

        avg_salt_interface = calculate_salt_bridge_density_average(
                            args.pdb_file,
                            args.sasa_file,
                            args.pka_file,
                            args.pH,
                            residues_for_density=interface_keys,
                            residues_for_average=interface_keys,
                            salt_bridges=salt_bridges,
                        )

        avg_hbond_energy = None
        avg_hbond_energy_buried_over_all = None
        avg_hbond_energy_beta_over_all = None
        avg_hbond_energy_dssp_unweighted = None
        avg_hbond_energy_dssp_unweighted_buried_over_all = None
        avg_hbond_energy_dssp_unweighted_beta_over_all = None
        avg_hbond_energy = calculate_hbond_energy_density_dssp_backbone_only_average(
            args.pdb_file,
            args.sasa_file,
            args.dssp_file,
            residues_for_density=None,
            residues_for_average=None,
        )

        avg_hbond_energy_buried_over_all = calculate_hbond_energy_density_dssp_backbone_only_average(
            args.pdb_file,
            args.sasa_file,
            args.dssp_file,
            residues_for_density=buried_keys,
            residues_for_average=None,
        )

        avg_hbond_energy_beta_over_all = (
            calculate_hbond_energy_density_dssp_backbone_only_average(
                args.pdb_file,
                args.sasa_file,
                args.dssp_file,
                residues_for_density=beta_keys,
                residues_for_average=None,
            )
            if beta_keys
            else None
        )

        # DSSP-only (no SASA) H-bond energy metric so missing SASA doesn't erase DSSP signal.
        avg_hbond_energy_dssp_unweighted = calculate_hbond_energy_dssp_backbone_only_unweighted_average(
            args.pdb_file,
            args.dssp_file,
            residues_for_density=None,
            residues_for_average=None,
        )
        avg_hbond_energy_dssp_unweighted_buried_over_all = calculate_hbond_energy_dssp_backbone_only_unweighted_average(
            args.pdb_file,
            args.dssp_file,
            residues_for_density=buried_keys,
            residues_for_average=None,
        )
        avg_hbond_energy_dssp_unweighted_beta_over_all = (
            calculate_hbond_energy_dssp_backbone_only_unweighted_average(
                args.pdb_file,
                args.dssp_file,
                residues_for_density=beta_keys,
                residues_for_average=None,
            )
            if beta_keys
            else None
        )

     
        avg_aromatic = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=aromatic_keys,
        )

        avg_hydrophobic = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=hydrophobic_keys,
        )

        # Buried-only aromatic / hydrophobic, averaged over all residues
        aromatic_buried_keys = aromatic_keys & buried_keys
        hydrophobic_buried_keys = hydrophobic_keys & buried_keys
        avg_aromatic_buried_over_all = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=aromatic_buried_keys,
            residues_for_average=None,
        ) if aromatic_buried_keys else None
        avg_hydrophobic_buried_over_all = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=hydrophobic_buried_keys,
            residues_for_average=None,
        ) if hydrophobic_buried_keys else None

        aromatic_cdr_keys = aromatic_keys & cdr_keys
        hydrophobic_cdr_keys = hydrophobic_keys & cdr_keys
        avg_aromatic_cdr_over_cdr = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=aromatic_cdr_keys,
            residues_for_average=aromatic_cdr_keys,
        ) if aromatic_cdr_keys else None

        avg_hydrophobic_cdr_over_cdr = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=hydrophobic_cdr_keys,
            residues_for_average=hydrophobic_cdr_keys,
        ) if hydrophobic_cdr_keys else None

        avg_polar = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=polar_keys,
        )
        avg_negative = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=negative_keys,
        )
        avg_positive = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=positive_keys,
        )

        polar_cdr_keys = polar_keys & cdr_keys
        negative_cdr_keys = negative_keys & cdr_keys
        positive_cdr_keys = positive_keys & cdr_keys
        avg_polar_cdr_over_cdr = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=polar_cdr_keys,
            residues_for_average=polar_cdr_keys,
        ) if polar_cdr_keys else None
        avg_negative_cdr_over_cdr = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=negative_cdr_keys,
            residues_for_average=negative_cdr_keys,
        ) if negative_cdr_keys else None
        avg_positive_cdr_over_cdr = calculate_residue_category_density_average(
            args.pdb_file,
            args.sasa_file,
            residue_category=positive_cdr_keys,
            residues_for_average=positive_cdr_keys,
        ) if positive_cdr_keys else None

        avg_wcn = calculate_weighted_contact_number_average(args.pdb_file)

        exposed_flags = get_exposed_residues(ctx.sasa_residue, 0.25)

        # Whole-structure WCN averages.

        avg_wcn_buried_over_all = calculate_weighted_contact_number_average(
            args.pdb_file,
            residue_category=buried_keys,
            residues_for_density=None,
            residues_for_average=None,
        )

        # CDR-only WCN, averaged over CDR residues
        avg_wcn_cdr_over_cdr = calculate_weighted_contact_number_average(
            args.pdb_file,
            residue_category=cdr_keys,
            residues_for_density=cdr_keys,
            residues_for_average=cdr_keys,
        ) if cdr_keys else None

        avg_wcn_interface_over_all = calculate_weighted_contact_number_average(
            args.pdb_file,
            residue_category=interface_keys,
            residues_for_density=None,
            residues_for_average=None,
        )
        
        # Largest connected component size of the (geometry-based) H-bond network
        largest_hbond_component_size_val = largest_hbond_component_size(args.pdb_file)

        # C-alpha DBSCAN cluster labels (negative/positive/aromatic/hydrophobic/polar; charge from PropKA)
        # Restrict clustering to surface-exposed residues when SASA data are available.
        pdb_atoms_for_clustering = pdb_atoms
        exposed_flags = get_exposed_residues(ctx.sasa_residue, 0.25)
        exposed_keys = {key for key, is_exposed in exposed_flags.items() if is_exposed}
        pdb_atoms_for_clustering = [
            atom
            for atom in pdb_atoms
            if residue_key_from_atom(atom) in exposed_keys
        ]

        neg_exposed_cluster_labels, pos_exposed_cluster_labels, aromatic_exposed_cluster_labels, hydro_exposed_cluster_labels, polar_exposed_cluster_labels = compute_residue_DBSCAN_cluster_labels(
                    pdb_atoms_for_clustering, pka_output_data, args.pH
                )

        neg_exposed_cluster_summary = summarize_dbscan_clusters(neg_exposed_cluster_labels, ctx.sasa_output)
        pos_exposed_cluster_summary = summarize_dbscan_clusters(pos_exposed_cluster_labels, ctx.sasa_output)
        aromatic_exposed_cluster_summary = summarize_dbscan_clusters(aromatic_exposed_cluster_labels, ctx.sasa_output)
        hydro_exposed_cluster_summary = summarize_dbscan_clusters(hydro_exposed_cluster_labels, ctx.sasa_output)
        polar_exposed_cluster_summary = summarize_dbscan_clusters(polar_exposed_cluster_labels, ctx.sasa_output)

        # Ripley K, PSH, PPC, PNC
        ripley = compute_surface_ripley_descriptors(pdb_atoms, ctx.sasa_residue, pka_output_data, args.pH)
        pairs = compute_surface_pair_descriptors(
            pdb_atoms,
            ctx.sasa_residue,
            pka_output_data,
            args.pH,
            salt_bridge_residues=salt_bridge_residues,
        )
        ripley_k_negative = ripley["ripley_k_negative"]
        ripley_k_positive = ripley["ripley_k_positive"]
        ripley_k_aromatic = ripley["ripley_k_aromatic"]
        ripley_k_hydrophobic = ripley["ripley_k_hydrophobic"]
        ripley_k_polar = ripley["ripley_k_polar"]
        psh_all_surface = pairs["psh_all_surface"]
        psh_cdr_vicinity = pairs["psh_cdr_vicinity"]
        ppc_all_surface = pairs["ppc_all_surface"]
        ppc_cdr_vicinity = pairs["ppc_cdr_vicinity"]
        pnc_all_surface = pairs["pnc_all_surface"]
        pnc_cdr_vicinity = pairs["pnc_cdr_vicinity"]

        # H-bond density averages (structure-level)
        avg_hbond = None
        avg_hbond_cdr = None
        avg_hbond_buried_over_all = None
        avg_hbond_beta_over_all = None
        avg_hbond_inter_chain = None
        weights_raw, counts = compute_hbond_density_raw(args.pdb_file, args.sasa_file)
        number_of_hbonds = sum(counts.values()) // 2 if counts else 0
        _n_res = _count_residues_in_pdb(args.pdb_file)
        mean_hbond_degree = (2.0 * number_of_hbonds / _n_res) if _n_res > 0 else None

        avg_hbond = calculate_global_hbond_density_average(args.pdb_file, args.sasa_file, weights_raw=weights_raw, counts=counts, residues_for_density=None, residues_for_average=None)
        avg_hbond_cdr = calculate_global_hbond_density_average(args.pdb_file, args.sasa_file, weights_raw=weights_raw, counts=counts, residues_for_density=cdr_keys, residues_for_average=cdr_keys)
        avg_hbond_buried_over_all = calculate_global_hbond_density_average(args.pdb_file, args.sasa_file, weights_raw=weights_raw, counts=counts, residues_for_density=buried_keys, residues_for_average=None)

        avg_hbond_beta_over_all = calculate_global_hbond_density_average(
            args.pdb_file,
            args.sasa_file,
            weights_raw=weights_raw,
            counts=counts,
            residues_for_density=beta_keys,
            residues_for_average=None,
        )

        avg_hbond_inter_chain = calculate_global_hbond_density_average(
            args.pdb_file,
            args.sasa_file,
            weights_raw=weights_raw,
            counts=counts,
            residues_for_density=interface_keys,
            residues_for_average=interface_keys,
        )

        # Inter-chain buried SASA (structure-level)
        inter_chain_buried_sasa = compute_inter_chain_buried_sasa(args.sasa_file)
        total_asa = get_sasa_total(args.sasa_file)

        # Dipole moment magnitude (structure-level)
        dipole_moment_magnitude = compute_dipole_moment_magnitude(pdb_atoms, pka_output_data, args.pH)

        # Relative contact order (CO) from Cα contacts
        # - "all" is intra-chain only across all chains (H+L for antibodies)
        # - heavy/light are computed within each chain
        contact_order_all = calculate_relative_contact_order(
            args.pdb_file,
            ca_cutoff=8.0,
            min_sequence_separation=0,
            include_inter_chain=False,
            chain_order=[args.heavy_chain, args.light_chain],
        )
        contact_order_heavy = calculate_relative_contact_order(
            args.pdb_file,
            ca_cutoff=8.0,
            min_sequence_separation=0,
            include_inter_chain=False,
            chain_order=[args.heavy_chain],
        )
        contact_order_light = calculate_relative_contact_order(
            args.pdb_file,
            ca_cutoff=8.0,
            min_sequence_separation=0,
            include_inter_chain=False,
            chain_order=[args.light_chain],
        )

        # net charge at different pHs; pI
        net_charge_by_pH = {}
        net_charge = net_charge_from_pka(pka_output_data, args.pH)
        protein_pi = pi_from_pka(pka_output_data)
        for ph in NET_CHARGE_PHS:
            net_charge_by_pH[ph] = net_charge_from_pka(pka_output_data, ph)
            
        # SCM score (requires SASA + pKa)
        weighted_scm_score_by_pH = {}
        for ph in NET_CHARGE_PHS:
            weighted_scm_score_by_pH[ph] = scm_score_from_pka(args.pdb_file, args.sasa_file, pka_output_data, ph, d_cutoff=10.0, sasa_cutoff=0.25)

        # Per-chain net charge at pH 7 (heavy/light)
        _pka_heavy = {k: v for k, v in pka_output_data.items() if k[2] == args.heavy_chain}
        _pka_light = {k: v for k, v in pka_output_data.items() if k[2] == args.light_chain}
        heavy_charge_pH7 = net_charge_from_pka(_pka_heavy, 7.0)
        light_charge_pH7 = net_charge_from_pka(_pka_light, 7.0)

        # SAP-like structure-level metrics (multiple weighting modes)
        sasa_output_data = ctx.sasa_output
        _sasa = ctx.sasa_residue
        _buried_tsr = [getattr(_sasa[k], "total_side_rel", None) for k in buried_keys if k in _sasa]
        _exposed_tsr = [getattr(_sasa[k], "total_side_rel", None) for k in exposed_keys if k in _sasa]
        _buried_tsr = [v for v in _buried_tsr if v is not None]
        _exposed_tsr = [v for v in _exposed_tsr if v is not None]
        avg_total_side_rel_buried = sum(_buried_tsr) / len(_buried_tsr) if _buried_tsr else None
        avg_total_side_rel_exposed = sum(_exposed_tsr) / len(_exposed_tsr) if _exposed_tsr else None

        # total_side_rel sums by residue type (all / buried / exposed)
        def _tsr_sum(keys, res_set):
            return sum(
                getattr(_sasa[k], "total_side_rel", 0.0) or 0.0
                for k in keys if k in _sasa and k[0] in res_set
            )
        _all_keys = set(_sasa.keys())
        _cat = [
            ("aromatic", AROMATIC_RESIDUES),
            ("negative", NEGATIVE_CHARGED_RESIDUES),
            ("positive", POSITIVE_CHARGED_RESIDUES),
            ("polar", POLAR_RESIDUES),
            ("hydrophobic", HYDROPHOBIC_RESIDUES),
        ]
        total_side_rel_sums = {}
        for name, res_set in _cat:
            total_side_rel_sums[f"{name}_total_side_rel_sum"] = _tsr_sum(_all_keys, res_set)
            total_side_rel_sums[f"{name}_buried_total_side_rel_sum"] = _tsr_sum(buried_keys, res_set)
            total_side_rel_sums[f"{name}_exposed_total_side_rel_sum"] = _tsr_sum(exposed_keys, res_set)
            total_side_rel_sums[f"{name}_inter_chain_total_side_rel_sum"] = _tsr_sum(interface_keys, res_set)

        # Convenience: inter-chain-only SASA-weighted sums by residue type
        inter_chain_total_side_rel_sums = {
            f"{name}_inter_chain_total_side_rel_sum": total_side_rel_sums.get(
                f"{name}_inter_chain_total_side_rel_sum", 0.0
            )
            for name, _ in _cat
        }

        sap_score = None
        sap_hydro_score = None
        sap_pos_charge_score = None
        sap_neg_charge_score = None
        sap_score = sum_total_side_rel_within_cutoff(pdb_atoms, sasa_output_data, cutoff=5.0)
        sap_hydro_score = sum_total_side_rel_within_cutoff(pdb_atoms, sasa_output_data, cutoff=5.0, sap_mode=True)
        sap_pos_charge_score = sum_total_side_rel_within_cutoff(pdb_atoms, sasa_output_data, cutoff=5.0, positive_charge_mode=True, pka_output_data=pka_output_data, pH=args.pH)
        sap_neg_charge_score = sum_total_side_rel_within_cutoff(pdb_atoms, sasa_output_data, cutoff=5.0, negative_charge_mode=True, pka_output_data=pka_output_data, pH=args.pH)

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
        #     sap_score=sap_score,
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
        (
            neg_largest,
            neg_n_clusters,
            neg_total_rel_asa,
        ) = neg_exposed_cluster_summary
        (
            pos_largest,
            pos_n_clusters,
            pos_total_rel_asa,
        ) = pos_exposed_cluster_summary
        (
            aromatic_largest,
            aromatic_n_clusters,
            aromatic_total_rel_asa,
        ) = aromatic_exposed_cluster_summary
        (
            hydro_largest,
            hydro_n_clusters,
            hydro_total_rel_asa,
        ) = hydro_exposed_cluster_summary
        (
            polar_largest,
            polar_n_clusters,
            polar_total_rel_asa,
        ) = polar_exposed_cluster_summary
        aggregated = {}

        aggregated.update(
            {
                # All cluster and surface summary metrics under a single flat key
                "cluster_metrics": {
                    # DBSCAN cluster summaries
                    "negative_exposed_cluster_largest_size": neg_largest,
                    "negative_exposed_cluster_n_clusters": neg_n_clusters,
                    "negative_exposed_cluster_total_side_rel": neg_total_rel_asa,
                    "positive_exposed_cluster_largest_size": pos_largest,
                    "positive_exposed_cluster_n_clusters": pos_n_clusters,
                    "positive_exposed_cluster_total_side_rel": pos_total_rel_asa,
                    "aromatic_exposed_cluster_largest_size": aromatic_largest,
                    "aromatic_exposed_cluster_n_clusters": aromatic_n_clusters,
                    "aromatic_exposed_cluster_total_side_rel": aromatic_total_rel_asa,
                    "hydrophobic_exposed_cluster_largest_size": hydro_largest,
                    "hydrophobic_exposed_cluster_n_clusters": hydro_n_clusters,
                    "hydrophobic_exposed_cluster_total_side_rel": hydro_total_rel_asa,
                    "polar_exposed_cluster_largest_size": polar_largest,
                    "polar_exposed_cluster_n_clusters": polar_n_clusters,
                    "polar_exposed_cluster_total_side_rel": polar_total_rel_asa,
                    # Ripley K and pairwise surface descriptors
                    "ripley_k_negative": ripley_k_negative,
                    "ripley_k_positive": ripley_k_positive,
                    "ripley_k_aromatic": ripley_k_aromatic,
                    "ripley_k_hydrophobic": ripley_k_hydrophobic,
                    "ripley_k_polar": ripley_k_polar,
                    # PPC/PNC/PSH
                    "psh_all_surface_exposed": psh_all_surface,
                    "psh_cdr_vicinity": psh_cdr_vicinity,
                    "ppc_all_surface_exposed": ppc_all_surface,
                    "ppc_cdr_vicinity": ppc_cdr_vicinity,
                    "pnc_all_surface_exposed": pnc_all_surface,
                    "pnc_cdr_vicinity": pnc_cdr_vicinity,
                },
                # H-bond density / energy averages, grouped separately
                "h_bonds_metrics": {
                    "number_of_hbonds": number_of_hbonds,
                    "mean_hbond_degree": mean_hbond_degree,
                    "largest_hbond_component_size": largest_hbond_component_size_val,
                    "avg_hbond_all": avg_hbond,
                    "avg_hbond_cdr": avg_hbond_cdr,
                    "avg_hbond_buried_over_all": avg_hbond_buried_over_all,
                    "avg_hbond_beta_over_all": avg_hbond_beta_over_all,
                    "avg_hbond_inter_chain": avg_hbond_inter_chain,
                    "avg_hbond_energy_all": avg_hbond_energy,
                    "avg_hbond_energy_buried_over_all": avg_hbond_energy_buried_over_all,
                    "avg_hbond_energy_beta_over_all": avg_hbond_energy_beta_over_all,
                    "avg_hbond_energy_dssp_unweighted_all": avg_hbond_energy_dssp_unweighted,
                    "avg_hbond_energy_dssp_unweighted_buried_over_all": avg_hbond_energy_dssp_unweighted_buried_over_all,
                    "avg_hbond_energy_dssp_unweighted_beta_over_all": avg_hbond_energy_dssp_unweighted_beta_over_all,
                },
                # Aromatic / hydrophobic density metrics
                "density_metrics": {
                    "avg_aromatic_all": avg_aromatic,
                    "avg_hydrophobic_all": avg_hydrophobic,
                    "avg_aromatic_buried_over_all": avg_aromatic_buried_over_all,
                    "avg_hydrophobic_buried_over_all": avg_hydrophobic_buried_over_all,
                    "avg_aromatic_cdr_over_cdr": avg_aromatic_cdr_over_cdr,
                    "avg_hydrophobic_cdr_over_cdr": avg_hydrophobic_cdr_over_cdr,
                    "avg_polar_all": avg_polar,
                    "avg_negative_all": avg_negative,
                    "avg_positive_all": avg_positive,
                    "avg_polar_cdr_over_cdr": avg_polar_cdr_over_cdr,
                    "avg_negative_cdr_over_cdr": avg_negative_cdr_over_cdr,
                    "avg_positive_cdr_over_cdr": avg_positive_cdr_over_cdr,
                },
                # CDR composition
                "cdr_metrics": {
                    "cdr_total_length": cdr_total_length,
                    "fraction_gly_CDRs": fraction_gly_CDRs,
                    "fraction_pro_CDRs": fraction_pro_CDRs,
                    "fraction_aromatic_CDRs": fraction_aromatic_CDRs,
                    "fraction_gln_asn_CDRs": fraction_gln_asn_CDRs,
                    "ratio_hydrophobic_to_polar_CDRs": ratio_hydrophobic_to_polar_CDRs,
                },
                "beta_sheet_metrics": {
                    "fraction_hydrophobic_beta_sheet": fraction_hydrophobic_beta_sheet,
                    "fraction_gln_asn_beta_sheet": fraction_gln_asn_beta_sheet,
                    "hydrophobic_beta_sheet_kyte_doolittle_sum": hydrophobic_beta_sheet_kyte_doolittle_sum,
                },
                "kyte_doolittle_metrics": {
                    "kyte_doolittle_sum_all": kyte_doolittle_sum_all,
                    "kyte_doolittle_mean_all": kyte_doolittle_mean_all,
                },
                # WCN metrics
                "wcn_metrics": {
                    "avg_wcn_all": avg_wcn,
                    "avg_wcn_cdr_over_cdr": avg_wcn_cdr_over_cdr,
                    "avg_wcn_buried_over_all": avg_wcn_buried_over_all,
                    "avg_wcn_interface_over_all": avg_wcn_interface_over_all,
                },
                "topology_metrics": {
                    "relative_contact_order_all": contact_order_all,
                    "relative_contact_order_heavy": contact_order_heavy,
                    "relative_contact_order_light": contact_order_light,
                },
                # Salt-bridge density averages (same partitions as printed above)
                "salt_bridges_metrics": {
                    "number_of_salt_bridges": number_of_salt_bridges,
                    "avg_salt_all": avg_salt,
                    "avg_salt_cdr": avg_salt_cdr,
                    "avg_salt_exposed_over_all": avg_salt_exposed_over_all,
                    "avg_salt_beta_over_all": avg_salt_beta_over_all,
                    "avg_salt_inter_chain": avg_salt_interface,
                },
                # Charge-related metrics
                "charge_metrics": {
                    "dipole_moment_magnitude": dipole_moment_magnitude,
                    "protein_pi": protein_pi,
                    "net_charge_by_pH": net_charge_by_pH,
                    "weighted_scm_score_by_pH": weighted_scm_score_by_pH,
                    "heavy_charge_pH7": heavy_charge_pH7,
                    "light_charge_pH7": light_charge_pH7,
                    "sap_pos_charge_score": _scalar_or_dict_sum(sap_pos_charge_score),
                    "sap_neg_charge_score": _scalar_or_dict_sum(sap_neg_charge_score),
                },
                "buried_metrics": {
                    "fraction_buried": fraction_buried,
                    "fraction_hydrophobic_buried": fraction_hydrophobic_buried,
                    "fraction_negative_buried": fraction_negative_buried,
                    "fraction_positive_buried": fraction_positive_buried,
                },
                "total_side_rel_sums": total_side_rel_sums,
                "inter_chain_total_side_rel_sums": inter_chain_total_side_rel_sums,
                "other_sasa_metrics": {
                    "total_asa": total_asa,
                    "inter_chain_buried_sasa": inter_chain_buried_sasa,
                    "avg_total_side_rel_buried": avg_total_side_rel_buried,
                    "avg_total_side_rel_exposed": avg_total_side_rel_exposed,
                }
            }
        )

        # Motif counting should be per-chain (H and L separately) and then summed.
        # This avoids creating artificial motifs at the H/L concatenation boundary.
        #
        # For region-specific counts (CDR/beta/exposed/inter-chain), operate on contiguous
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
                full_seq_per_chain = [seq for _ch, seq, _m in chain_seqs_and_maps]

                sequence_motives["n_motif_AsnGly"] = count_motif_overlapping(full_seq_per_chain, "NG")
                sequence_motives["n_motif_AsnAsp"] = count_motif_overlapping(full_seq_per_chain, "ND")
                sequence_motives["n_motif_AspGly"] = count_motif_overlapping(full_seq_per_chain, "DG")
                sequence_motives["n_motif_AspSer"] = count_motif_overlapping(full_seq_per_chain, "DS")
                sequence_motives["n_motif_AspAsp"] = count_motif_overlapping(full_seq_per_chain, "DD")
                sequence_motives["n_motif_AspThr"] = count_motif_overlapping(full_seq_per_chain, "DT")
                sequence_motives["n_motif_AspHis"] = count_motif_overlapping(full_seq_per_chain, "DH")
                sequence_motives["n_motif_AspGlu"] = count_motif_overlapping(full_seq_per_chain, "DE")
                sequence_motives["n_motif_ArgLys"] = count_motif_overlapping(full_seq_per_chain, "RK")

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

                fragments_cdr = _contiguous_fragments_for_keyset_per_chain(cdr_keys)
                fragments_beta = _contiguous_fragments_for_keyset_per_chain(beta_keys)
                fragments_exposed = _contiguous_fragments_for_keyset_per_chain(exposed_keys)
                fragments_inter = _contiguous_fragments_for_keyset_per_chain(interface_keys)

                cdr_frags_flat = _flatten_fragments(fragments_cdr)
                beta_frags_flat = _flatten_fragments(fragments_beta)
                exposed_frags_flat = _flatten_fragments(fragments_exposed)
                inter_frags_flat = _flatten_fragments(fragments_inter)

                for motif, key in [
                    ("NG", "AsnGly"),
                    ("ND", "AsnAsp"),
                    ("DG", "AspGly"),
                    ("DS", "AspSer"),
                    ("DD", "AspAsp"),
                    ("DT", "AspThr"),
                    ("DH", "AspHis"),
                    ("DE", "AspGlu"),
                    ("RK", "ArgLys"),
                ]:
                    sequence_motives[f"n_motif_{key}_CDRs"] = count_motif_overlapping(cdr_frags_flat, motif)
                    sequence_motives[f"n_motif_{key}_beta_sheet"] = count_motif_overlapping(beta_frags_flat, motif)
                    sequence_motives[f"n_motif_{key}_exposed"] = count_motif_overlapping(exposed_frags_flat, motif)
                    sequence_motives[f"n_motif_{key}_inter_chain"] = count_motif_overlapping(inter_frags_flat, motif)

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
                        ("NG", "AsnGly"),
                        ("ND", "AsnAsp"),
                        ("DG", "AspGly"),
                        ("DS", "AspSer"),
                        ("DD", "AspAsp"),
                        ("DT", "AspThr"),
                        ("DH", "AspHis"),
                        ("DE", "AspGlu"),
                        ("RK", "ArgLys"),
                    ]:
                        sequence_motives[f"side_asa_sum_motif_{key}"] = _motif_side_asa_sum(motif)

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
        if args.format == "csv":
            # Flatten nested dicts into a single-row CSV table.
            df = pd.json_normalize(aggregated_clean)
            # Round float columns for stable output.
            df = df.round(3)
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"Results written to {args.output}")
            else:
                # Print CSV to stdout.
                df.to_csv(sys.stdout, index=False)
        else:
            # JSON output (default).
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

