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

# Add src directory to path so developability and utils packages are found
# This allows running from the repo root (e.g. python3 src/developability/run_developability.py ...)
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from developability.descriptors import (
    # calculate_global_hbond_density,
    calculate_global_hbond_density_average,
    # calculate_salt_bridge_density,
    calculate_salt_bridge_density_average,
    calculate_residue_category_density_average,
    compute_residue_density_raw,
    compute_hbond_density_raw,
    largest_hbond_component_size,
    net_charge_from_pka,
    pi_from_pka,
    scm_score_from_pka,
    sum_total_side_rel_within_cutoff,
    # WCN functions
    calculate_weighted_contact_number_average,
    # DSSP parsing
    parse_dssp,
    # DSSP-based H-bond density
    # calculate_hbond_energy_density_dssp_backbone_only,
    calculate_hbond_energy_density_dssp_backbone_only_average,
    # C-alpha DBSCAN cluster labels by residue group
    compute_residue_DBSCAN_cluster_labels,
    summarize_dbscan_clusters,
    _get_atoms_for_path,
    get_inter_chain_interface_residues,
    # Structure-level surface/spatial and charge descriptors
    compute_surface_ripley_descriptors,
    compute_surface_pair_descriptors,
    compute_dipole_moment_magnitude,
    compute_inter_chain_buried_sasa,
    compute_downstream_descriptors,
    # Residue-type sets (for SASA-weighted densities)
    NEGATIVE_CHARGED_RESIDUES,
    POSITIVE_CHARGED_RESIDUES,
    POLAR_RESIDUES,
    HYDROPHOBIC_RESIDUES,
)
from developability.structure_context import StructureContext
from developability.descriptor_utils import (
    get_residue_region_map,
    get_exposed_residues,
    get_aromatic_residue_keys,
    get_residue_keys_by_type,
)
from developability.descriptors import (
    count_motif_overlapping,
    get_full_sequence_with_index_map_from_pdb,
)
from utils.parsers import parse_pka, get_pka_file_path, residue_key_from_atom

NET_CHARGE_PHS = [4, 5, 6, 7, 8, 9, 10]


def _to_4(key):
    """Normalize to 4-tuple (res_name, res_num, chain, insertion_code)."""
    return (key[0], key[1], key[2], key[3]) if len(key) == 4 else (key[0], key[1], key[2], "")


def _lookup(d, key):
    """Lookup by 4-tuple residue key."""
    return d.get(key) or (d.get((key[0], key[1], key[2], "")) if len(key) == 4 else None)


def _scalar_or_dict_sum(value: Any) -> Optional[float]:
    """Normalize SAP score: descriptor returns float; legacy code may use dict of per-residue values."""
    if value is None:
        return None
    if isinstance(value, dict) and value:
        return float(sum(value.values()))
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (f != f) else f  # exclude nan
    return None


# def _build_output(
#     *,
#     sorted_residues,
#     calc_hbonds,
#     calc_salt_bridges,
#     calc_aromatic,
#     calc_wcn,
#     hbond_densities,
#     hbond_counts,
#     # salt_bridge_densities,
#     aromatic_densities,
#     # wcn_values,
#     # hbond_energy_dssp_densities,
#     dssp_data,
#     sasa_output_data,
#     pka_output_data,
#     neg_cluster_labels,
#     pos_cluster_labels,
#     aromatic_cluster_labels,
#     hydro_cluster_labels,
#     polar_cluster_labels,
#     ripley_k_negative,
#     ripley_k_positive,
#     ripley_k_aromatic,
#     ripley_k_hydrophobic,
#     ripley_k_polar,
#     psh_all_surface,
#     psh_cdr_vicinity,
#     ppc_all_surface,
#     ppc_cdr_vicinity,
#     pnc_all_surface,
#     pnc_cdr_vicinity,
#     dipole_moment_magnitude,
#     largest_hbond_component_size_val,
#     net_charge,
#     protein_pi,
#     # scm_score_val,
#     weighted_scm_score_by_pH,
#     inter_chain_buried_sasa,
#     net_charge_by_pH,
#     hbond_inter_chain,
#     # salt_bridge_inter_chain,
#     hbond_energy_dssp_inter_chain,
#     sap_score,
#     sap_hydro_score,
#     sap_pos_charge_score,
#     sap_neg_charge_score,
# ):
    # rows = []
    # for idx, residue_key in enumerate(sorted_residues):
        # res_name, res_num, chain = residue_key[0], residue_key[1], residue_key[2]
        # ins = residue_key[3] if len(residue_key) == 4 else ""
        # res_num_str = str(res_num) + (ins if ins else "")
        # row = {
        #     "residue_name": res_name,
        #     "chain": chain,
        #     "residue_number": res_num_str,
        # }
        # if calc_hbonds:
        #     row["hbond_density"] = hbond_densities.get(residue_key, 0.0)
            # row["number_of_hbonds"] = hbond_counts.get(residue_key, 0)
        # if calc_salt_bridges:
        #     row["salt_bridge_dsalt_bridge_densitiesensity"] = .get(residue_key, 0.0)
        # if calc_aromatic:
        #     row["aromatic_density"] = aromatic_densities.get(residue_key, 0.0)
        # if calc_wcn:
        #     row["wcn"] = wcn_values.get(residue_key, 0.0)
        # if hbond_energy_dssp_densities:
        #     v = _lookup(hbond_energy_dssp_densities, residue_key)
            # row["hbond_energy_dssp_density"] = (v or 0.0)
        # if dssp_data:
        #     dssp_entry = _lookup(dssp_data, residue_key) or {}
        #     row["secondary_structure"] = dssp_entry.get("secondary_structure", "")
        #     for col in ["N-H-->O_1", "N-H-->O_2", "O-->H-N_1", "O-->H-N_2"]:
        #         val = dssp_entry.get(col)
        #         row[col] = val if val is not None else ""
        # if sasa_output_data:
        #     sasa_entry = _lookup(sasa_output_data, residue_key) or {}
        #     row["total_side_rel"] = sasa_entry.get("total_side_rel")
        #     # row["main_chain_rel"] = sasa_entry.get("main_chain_rel")
        # if pka_output_data:
        #     pka_val = pka_output_data.get(residue_key) or pka_output_data.get(
        #         (residue_key[0], residue_key[1], residue_key[2], "")
        #     )
        #     row["pka"] = pka_val
        # for col, d in [
        #     ("negative_cluster_labels", neg_cluster_labels),
        #     ("positive_cluster_labels", pos_cluster_labels),
        #     ("aromatic_cluster_labels", aromatic_cluster_labels),
        #     ("hydrophobic_cluster_labels", hydro_cluster_labels),
        #     ("polar_cluster_labels", polar_cluster_labels),
        # ]:
        #     row[col] = str(d[residue_key]) if residue_key in d else ""
        # Structure-level: first row only


            
    #         row["dipole_moment_magnitude"] = dipole_moment_magnitude
    #         row["largest_hbond_component_size"] = largest_hbond_component_size_val
    #         row["net_charge"] = net_charge
    #         row["protein_pi"] = protein_pi
    #         # row["scm_score"] = scm_score_val
    #         row["inter_chain_buried_sasa"] = inter_chain_buried_sasa
    #         row["SAP"] = sap_score
    #         row["SAP_hydro"] = sap_hydro_score
    #         row["SAP_pos_charge"] = sap_pos_charge_score
    #         row["SAP_neg_charge"] = sap_neg_charge_score
    #         for ph in NET_CHARGE_PHS:
    #             row[f"net_charge_pH{ph}"] = net_charge_by_pH.get(ph)
    #             row[f"scm_score_pH{ph}"] = weighted_scm_score_by_pH.get(ph)
    #     else:
    #         for col in (
    #             "ripley_k_negative", "ripley_k_positive", "ripley_k_hydrophobic", "ripley_k_polar",
    #             "psh_all_surface_exposed", "psh_cdr_vicinity", "ppc_all_surface_exposed", "ppc_cdr_vicinity",
    #             "pnc_all_surface_exposed", "pnc_cdr_vicinity",
    #             "dipole_moment_magnitude", "largest_hbond_component_size", "net_charge", "protein_pi",
    #             "scm_score", "inter_chain_buried_sasa",
    #         ):
    #             row[col] = ""
    #         for ph in NET_CHARGE_PHS:
    #             row[f"net_charge_pH{ph}"] = ""
    #         for col in ("SAP", "SAP_hydro", "SAP_pos_charge", "SAP_neg_charge"):
    #             row[col] = ""
    #             row[f"scm_score_pH{ph}"] = ""
    #     has_inter_chain = (
    #         hbond_inter_chain.get(residue_key, False)
    #         # or salt_bridge_inter_chain.get(residue_key, False)
    #         or bool(_lookup(hbond_energy_dssp_inter_chain, residue_key))
    #     )
    #     row["inter_chain_contact"] = "True" if has_inter_chain else "False"
    #     rows.append(row)
    # return pd.DataFrame(rows)


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
    
    pka_output_data = {}
    pka_path = args.pka_file or get_pka_file_path(args.pdb_file)
    if pka_path and Path(pka_path).exists():
        pka_output_data = parse_pka(pka_path, pdb_atoms)
        if pka_output_data:
            print(f"Parsed pKa data for {len(pka_output_data)} residues", file=sys.stderr)
   
    try:
            
        ctx = StructureContext(
            args.pdb_file,
            sasa_path=args.sasa_file,
            pka_path=args.pka_file,
        )

        cdr_keys = {
                        key
                        for key, region in get_residue_region_map(ctx.atoms).items()
                        if region == "CDR"
                    }
        exposed_flags = get_exposed_residues(ctx.sasa_residue, 0.25)
        exposed_keys = {
            key for key, is_exposed in exposed_flags.items() if is_exposed
        }
        beta_keys = {
            _to_4(key)
            for key, entry in (dssp_data or {}).items()
            if entry.get("secondary_structure") == "E"
        }
        interface_keys = get_inter_chain_interface_residues(args.pdb_file)
        buried_keys = set(ctx.sasa_residue.keys()) - exposed_keys
        hydrophobic_keys = get_residue_keys_by_type(
            args.pdb_file, HYDROPHOBIC_RESIDUES
        )
        polar_keys = get_residue_keys_by_type(args.pdb_file, POLAR_RESIDUES)
        negative_keys = get_residue_keys_by_type(args.pdb_file, NEGATIVE_CHARGED_RESIDUES)
        positive_keys = get_residue_keys_by_type(args.pdb_file, POSITIVE_CHARGED_RESIDUES)
        aromatic_keys = get_aromatic_residue_keys(args.pdb_file)

        # Salt bridge densities

        avg_salt = calculate_salt_bridge_density_average(
            args.pdb_file, args.sasa_file, args.pka_file, args.pH
        )

        avg_salt_cdr = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            args.pH,
            residues_for_density=cdr_keys,
            residues_for_average=cdr_keys,
        )

        avg_salt_exposed_over_all = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            args.pH,
            residues_for_density=exposed_keys,
            # residues_for_average=exposed_keys,
        )

        avg_salt_beta_over_all = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            args.pH,
            residues_for_density=beta_keys,
            # residues_for_average=beta_keys,
        )

        avg_salt_interface = calculate_salt_bridge_density_average(
                            args.pdb_file,
                            args.sasa_file,
                            args.pka_file,
                            args.pH,
                            residues_for_density=interface_keys,
                            residues_for_average=interface_keys,
                        )

        avg_hbond_energy = None
        avg_hbond_energy_buried_over_all = None
        avg_hbond_energy_beta_over_all = None
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
        
        largest_hbond_component_size_val = None

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

        neg_cluster_labels, pos_cluster_labels, aromatic_cluster_labels, hydro_cluster_labels, polar_cluster_labels = compute_residue_DBSCAN_cluster_labels(
                    pdb_atoms_for_clustering, pka_output_data, args.pH
                )

        neg_cluster_summary = summarize_dbscan_clusters(neg_cluster_labels, ctx.sasa_output)
        pos_cluster_summary = summarize_dbscan_clusters(pos_cluster_labels, ctx.sasa_output)
        aromatic_cluster_summary = summarize_dbscan_clusters(aromatic_cluster_labels, ctx.sasa_output)
        hydro_cluster_summary = summarize_dbscan_clusters(hydro_cluster_labels, ctx.sasa_output)
        polar_cluster_summary = summarize_dbscan_clusters(polar_cluster_labels, ctx.sasa_output)

        # Ripley K, PSH, PPC, PNC
        ripley = compute_surface_ripley_descriptors(pdb_atoms, ctx.sasa_residue, pka_output_data, args.pH)
        pairs = compute_surface_pair_descriptors(pdb_atoms, ctx.sasa_residue, pka_output_data, args.pH)
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

        # Dipole moment magnitude (structure-level)
        dipole_moment_magnitude = compute_dipole_moment_magnitude(pdb_atoms, pka_output_data, args.pH)

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

        # SAP-like structure-level metrics (multiple weighting modes)
        sasa_output_data = ctx.sasa_output
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
        ) = neg_cluster_summary
        (
            pos_largest,
            pos_n_clusters,
            pos_total_rel_asa,
        ) = pos_cluster_summary
        (
            aromatic_largest,
            aromatic_n_clusters,
            aromatic_total_rel_asa,
        ) = aromatic_cluster_summary
        (
            hydro_largest,
            hydro_n_clusters,
            hydro_total_rel_asa,
        ) = hydro_cluster_summary
        (
            polar_largest,
            polar_n_clusters,
            polar_total_rel_asa,
        ) = polar_cluster_summary
        aggregated = {}

        aggregated.update(
            {
                # All cluster and surface summary metrics under a single flat key
                "cluster_metrics": {
                    # DBSCAN cluster summaries
                    "negative_cluster_largest_size": neg_largest,
                    "negative_cluster_n_clusters": neg_n_clusters,
                    "negative_cluster_total_side_rel": neg_total_rel_asa,
                    "positive_cluster_largest_size": pos_largest,
                    "positive_cluster_n_clusters": pos_n_clusters,
                    "positive_cluster_total_side_rel": pos_total_rel_asa,
                    "aromatic_cluster_largest_size": aromatic_largest,
                    "aromatic_cluster_n_clusters": aromatic_n_clusters,
                    "aromatic_cluster_total_side_rel": aromatic_total_rel_asa,
                    "hydrophobic_cluster_largest_size": hydro_largest,
                    "hydrophobic_cluster_n_clusters": hydro_n_clusters,
                    "hydrophobic_cluster_total_side_rel": hydro_total_rel_asa,
                    "polar_cluster_largest_size": polar_largest,
                    "polar_cluster_n_clusters": polar_n_clusters,
                    "polar_cluster_total_side_rel": polar_total_rel_asa,
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
                    "avg_hbond_all": avg_hbond,
                    "avg_hbond_cdr": avg_hbond_cdr,
                    "avg_hbond_buried_over_all": avg_hbond_buried_over_all,
                    "avg_hbond_beta_over_all": avg_hbond_beta_over_all,
                    "avg_hbond_inter_chain": avg_hbond_inter_chain,
                    "avg_hbond_energy_all": avg_hbond_energy,
                    "avg_hbond_energy_buried_over_all": avg_hbond_energy_buried_over_all,
                    "avg_hbond_energy_beta_over_all": avg_hbond_energy_beta_over_all,
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
                # WCN metrics
                "wcn_metrics": {
                    "avg_wcn_all": avg_wcn,
                    "avg_wcn_cdr_over_cdr": avg_wcn_cdr_over_cdr,
                    "avg_wcn_buried_over_all": avg_wcn_buried_over_all,
                    "avg_wcn_interface_over_all": avg_wcn_interface_over_all,
                },
                # Salt-bridge density averages (same partitions as printed above)
                "salt_bridges_metrics": {
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
                    "sap_pos_charge_score": _scalar_or_dict_sum(sap_pos_charge_score),
                    "sap_neg_charge_score": _scalar_or_dict_sum(sap_neg_charge_score),
                },
                # Explicit top-level inter-chain buried SASA
                "inter_chain_buried_sasa": inter_chain_buried_sasa,
            }
        )

        full_seq, seq_index_map = get_full_sequence_with_index_map_from_pdb(args.pdb_file)
        if full_seq:
                sequence_motives = aggregated.setdefault("sequence_motives", {})

                sequence_motives["n_motif_AsnGly"] = count_motif_overlapping(full_seq, "NG")
                sequence_motives["n_motif_AsnAsp"] = count_motif_overlapping(full_seq, "ND")
                sequence_motives["n_motif_AspGly"] = count_motif_overlapping(full_seq, "DG")
                sequence_motives["n_motif_AspSer"] = count_motif_overlapping(full_seq, "DS")
                sequence_motives["n_motif_AspAsp"] = count_motif_overlapping(full_seq, "DD")
                sequence_motives["n_motif_AspThr"] = count_motif_overlapping(full_seq, "DT")
                sequence_motives["n_motif_AspHis"] = count_motif_overlapping(full_seq, "DH")

                def _seq_for_keys(key_set):
                    if not key_set or not full_seq:
                        return ""
                    # Map region residue keys into indices in the global full_seq,
                    # then build a subsequence in that order.
                    indices = [seq_index_map[k] for k in key_set if k in seq_index_map]
                    if not indices:
                        return ""
                    indices = sorted(set(indices))
                    return "".join(full_seq[i] for i in indices)

                cdr_seq = _seq_for_keys(cdr_keys)
                beta_seq = _seq_for_keys(beta_keys)
                exposed_seq = _seq_for_keys(exposed_keys)
                inter_seq = _seq_for_keys(interface_keys)

                for motif, key in [
                    ("NG", "AsnGly"),
                    ("ND", "AsnAsp"),
                    ("DG", "AspGly"),
                    ("DS", "AspSer"),
                    ("DD", "AspAsp"),
                    ("DT", "AspThr"),
                    ("DH", "AspHis"),
                ]:
                    sequence_motives[f"n_motif_{key}_CDRs"] = count_motif_overlapping(cdr_seq, motif)
                    sequence_motives[f"n_motif_{key}_beta_sheet"] = count_motif_overlapping(beta_seq, motif)
                    sequence_motives[f"n_motif_{key}_exposed"] = count_motif_overlapping(exposed_seq, motif)
                    sequence_motives[f"n_motif_{key}_inter_chain"] = count_motif_overlapping(inter_seq, motif)

                # try:
                #     def _motif_residue_keys(motif: str) -> Set[Tuple[str, int, str, str]]:
                #         keys: Set[Tuple[str, int, str, str]] = set()
                #         if not full_seq:
                #             return keys
                #         for i in range(len(full_seq) - 1):
                #             if full_seq[i : i + 2] == motif:
                #                 if i < len(sorted_residues):
                #                     keys.add(sorted_residues[i])
                #                 if i + 1 < len(sorted_residues):
                #                     keys.add(sorted_residues[i + 1])
                #         return keys

                #     # Compute SASA-based density once and reuse for all motifs.
                #     density_raw = compute_residue_density_raw(args.pdb_file, args.sasa_file)

                #     for motif, key in [
                #         ("NG", "AsnGly"),
                #         ("ND", "AsnAsp"),
                #         ("DG", "AspGly"),
                #         ("DS", "AspSer"),
                #         ("DD", "AspAsp"),
                #         ("DT", "AspThr"),
                #         ("DH", "AspHis"),
                #     ]:
                #         res_keys = _motif_residue_keys(motif)
                #         if res_keys and density_raw:
                #             aggregated[f"avg_sasa_weighted_{key}"] = calculate_residue_category_density_average(
                #                 args.pdb_file,
                #                 args.sasa_file,
                #                 residue_category=res_keys,
                #                 residues_for_average=None,  # average over all residues in the PDB
                #                 weighted=True,
                #                 density_raw=density_raw,
                #             )
                #         else:
                #             aggregated[f"avg_sasa_weighted_{key}"] = None

                #         sequence_motives[f"avg_sasa_weighted_{key}"] = aggregated[f"avg_sasa_weighted_{key}"]
                # except Exception:
                #     for _, key in [
                #         ("NG", "AsnGly"),
                #         ("ND", "AsnAsp"),
                #         ("DG", "AspGly"),
                #         ("DS", "AspSer"),
                #         ("DD", "AspAsp"),
                #         ("DT", "AspThr"),
                #         ("DH", "AspHis"),
                #     ]:
                #         aggregated[f"avg_sasa_weighted_{key}"] = None
                #         # Mirror into sequence_motives when SASA-weighted metrics fail
                #         sequence_motives[f"avg_sasa_weighted_{key}"] = None

        def _sanitize_for_json(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: _sanitize_for_json(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize_for_json(x) for x in obj]
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
                            return None if pd.isna(obj) else float(obj)
                    except (ValueError, TypeError):
                        pass
                return obj

        aggregated_clean = _sanitize_for_json(aggregated)
        if args.format == "csv":
            # Flatten nested dicts into a single-row CSV table.
            df = pd.json_normalize(aggregated_clean)
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

