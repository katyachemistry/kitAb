#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

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
    pi_from_pka,
    scm_score_from_pka,
    scm_score_by_atoms,
    compute_sap_shell_synergy_scores,
    parse_dssp,
    calculate_hbond_energy_density_dssp_backbone_only_average,
    compute_residue_DBSCAN_cluster_labels,
    summarize_dbscan_clusters,
    dbscan_cluster_side_abs_sasa_entropy,
    _get_atoms_for_path,
    compute_exposed_pair_correlation_cluster_scores,
    compute_dipole_moment_magnitude,
    compute_inter_chain_buried_sasa,
    asymmetry_score_from_pka,
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
    GLN_ASN_RESIDUES,
    HYDROPHOBIC_RESIDUES,
    KYTE_DOOLITTLE,
    get_ff19sb_residue_region_charges,
)
from developability.structure_context import ResKey4, StructureContext
from developability.descriptor_utils import (
    CDR_RANGES_CA,
    get_residue_region_map,
    get_exposed_residues,
    get_aromatic_residue_keys,
    get_residue_keys_by_type,
    residue_side_sasa,
)
from developability.descriptors import (
    sum_residue_mean_local_planarity,
    mean_residue_planarity_over_residues,
)
from utils.parsers import parse_pka, get_pka_file_path, residue_key_from_atom

NET_CHARGE_PHS = [3, 7.5]

def _to_4(key):
    return (key[0], key[1], key[2], key[3]) if len(key) == 4 else (key[0], key[1], key[2], "")

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
    r"^pcf_(neg|pos|hyd)_(.+)$"
)

def _pcf_cluster_per_shell_and_mean_per_category(
    pcf: Dict[str, Any],
) -> Dict[str, Optional[float]]:

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
        help='pH value for charge state determination in salt bridge detection (default: 7.5). Only used if pKa file is provided.'
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

        pka_output_data = ctx.pka_residue
        if pka_output_data:
            print(
                f"Using pKa data for {len(pka_output_data)} residues",
                file=sys.stderr,
            )

        sasa_failed = bool(args.sasa_file) and (
            "sasa" in ctx.parse_errors or not ctx.sasa_residue
        )

        cdr_keys = {
                        key
                        for key, region in get_residue_region_map(ctx.atoms).items()
                        if region == "CDR"
                    }
        cdr_vicinity_keys: Set[ResKey4] = (
            ctx.get_cdr_vicinity_residue_keys(cdr_keys) if cdr_keys else set()
        )
        exposed_keys: Set[ResKey4] = set()
        if not sasa_failed:
            exposed_flags = get_exposed_residues(ctx.sasa_residue, EXPOSURE_REL_ASA_THRESHOLD)
            exposed_keys = {key for key, is_exposed in exposed_flags.items() if is_exposed}

        n_cdr = len(cdr_keys)
        cdr3_start, cdr3_end = CDR_RANGES_CA[2]
        cdr3_length = sum(
            1 for k in cdr_keys
            if cdr3_start <= int(k[1]) <= cdr3_end
        )

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

        def _total_side_rel_weight(k: Tuple[str, int, str, str]) -> float:
            """Relative side-chain SASA (fraction in [0, 1] from ``parse_sasa``)."""
            return float(getattr(ctx.sasa_residue[k], "total_side_rel", 0.0)) or 0.0

        kyte_doolittle_sum_all = (
            None
            if sasa_failed
            else sum(KYTE_DOOLITTLE.get(k[0], 0.0) for k in ctx.sasa_residue.keys())
        )

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

        hydrophobic_keys = get_residue_keys_by_type(
            args.pdb_file, HYDROPHOBIC_RESIDUES
        )
        negative_keys = get_residue_keys_by_type(
            args.pdb_file, CHARGE_FRACTION_NEGATIVE_RESIDUES
        )
        positive_keys = get_residue_keys_by_type(
            args.pdb_file, CHARGE_FRACTION_POSITIVE_RESIDUES
        )
        aromatic_keys = get_aromatic_residue_keys(args.pdb_file)

        salt_bridges = detect_salt_bridges(
            args.pdb_file, args.sasa_file, args.pka_file, pH=args.pH
        )
        number_of_salt_bridges = len(salt_bridges)

        avg_salt_cdr_vicinity_over_cdr_vicinity = calculate_salt_bridge_density_average(
            args.pdb_file,
            args.sasa_file,
            args.pka_file,
            pH=args.pH,
            residues_for_density=cdr_vicinity_keys,
            residues_for_average=cdr_vicinity_keys,
            salt_bridges=salt_bridges,
        )

        avg_hbond_energy = calculate_hbond_energy_density_dssp_backbone_only_average(
            args.pdb_file,
            args.sasa_file,
            args.dssp_file,
            residues_for_density=None,
            residues_for_average=None,
        )

        avg_hbond_energy_dssp_weighted = avg_hbond_energy

        exposed_cdr_vicinity_keys = exposed_keys & cdr_vicinity_keys
        aromatic_cdr_vicinity_keys = aromatic_keys & exposed_cdr_vicinity_keys
        hydrophobic_cdr_vicinity_keys = hydrophobic_keys & exposed_cdr_vicinity_keys
        negative_cdr_vicinity_keys = negative_keys & exposed_cdr_vicinity_keys
        positive_cdr_vicinity_keys = positive_keys & exposed_cdr_vicinity_keys


        total_local_planarity_negative_cdr_vicinity = sum_residue_mean_local_planarity(
            pdb_atoms, negative_cdr_vicinity_keys
        )
        total_local_planarity_positive_cdr_vicinity = sum_residue_mean_local_planarity(
            pdb_atoms, positive_cdr_vicinity_keys
        )
        total_local_planarity_aromatic_cdr_vicinity = sum_residue_mean_local_planarity(
            pdb_atoms, aromatic_cdr_vicinity_keys
        )
        total_local_planarity_hydrophobic_cdr_vicinity = sum_residue_mean_local_planarity(
            pdb_atoms, hydrophobic_cdr_vicinity_keys
        )
        mean_planarity_cdr_vicinity = mean_residue_planarity_over_residues(
            pdb_atoms, exposed_cdr_vicinity_keys
        )
        _planarity_denom = float(mean_planarity_cdr_vicinity)
        if _planarity_denom > 1e-18 and math.isfinite(_planarity_denom):
            normalized_local_planarity_negative_cdr_vicinity = (
                total_local_planarity_negative_cdr_vicinity / _planarity_denom
            )
            normalized_local_planarity_positive_cdr_vicinity = (
                total_local_planarity_positive_cdr_vicinity / _planarity_denom
            )
            normalized_local_planarity_aromatic_cdr_vicinity = (
                total_local_planarity_aromatic_cdr_vicinity / _planarity_denom
            )
            normalized_local_planarity_hydrophobic_cdr_vicinity = (
                total_local_planarity_hydrophobic_cdr_vicinity / _planarity_denom
            )
        else:
            normalized_local_planarity_negative_cdr_vicinity = 0.0
            normalized_local_planarity_positive_cdr_vicinity = 0.0
            normalized_local_planarity_aromatic_cdr_vicinity = 0.0
            normalized_local_planarity_hydrophobic_cdr_vicinity = 0.0

        density_side_abs_raw: Optional[Dict[ResKey4, float]] = None
        if not sasa_failed:
            density_side_abs_raw = compute_residue_side_abs_density_raw(
                args.pdb_file, args.sasa_file
            )

        sum_aromatic_weighted_rel_side_asa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=aromatic_cdr_vicinity_keys,
                residues_for_average="no",
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
            )
            if hydrophobic_cdr_vicinity_keys
            else 0.0
        )

        sum_negative_weighted_rel_side_asa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=negative_cdr_vicinity_keys,
                residues_for_average="no",
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
                density_raw=density_side_abs_raw,
            )
            if (density_side_abs_raw and aromatic_cdr_vicinity_keys)
            else 0.0
        )
        sum_negative_side_abs_sasa_cdr_vicinity = (
            calculate_residue_category_density_average(
                args.pdb_file,
                args.sasa_file,
                residue_category=negative_cdr_vicinity_keys,
                residues_for_average="no",
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
                density_raw=density_side_abs_raw,
            )
            if (density_side_abs_raw and positive_cdr_vicinity_keys)
            else 0.0
        )

        exposed_flags = get_exposed_residues(ctx.sasa_residue, EXPOSURE_REL_ASA_THRESHOLD)
        exposed_keys = {key for key, is_exposed in exposed_flags.items() if is_exposed}

        neg_exposed_cluster_labels, pos_exposed_cluster_labels, hydro_exposed_cluster_labels = compute_residue_DBSCAN_cluster_labels(
            pdb_atoms, pka_output_data, args.pH
        )

        neg_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            neg_exposed_cluster_labels, ctx.sasa_output
        )
        pos_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            pos_exposed_cluster_labels, ctx.sasa_output
        )
        hydro_exposed_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            hydro_exposed_cluster_labels, ctx.sasa_output
        )

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
        hydro_cdr_vicinity_clusters_total_side_abs_sasa = summarize_dbscan_clusters(
            hydro_cdr_cluster_labels, ctx.sasa_output
        )

        neg_cluster_side_abs_sasa_entropy = dbscan_cluster_side_abs_sasa_entropy(
            neg_exposed_cluster_labels, ctx.sasa_output
        )
        pos_cluster_side_abs_sasa_entropy = dbscan_cluster_side_abs_sasa_entropy(
            pos_exposed_cluster_labels, ctx.sasa_output
        )

        pcf_cluster = compute_exposed_pair_correlation_cluster_scores(
            pdb_atoms,
            ctx.sasa_residue,
            pka_output_data,
            args.pH,
            surface_exposed_threshold=EXPOSURE_REL_ASA_THRESHOLD,
        )

        weights_raw = compute_hbond_density_raw(args.pdb_file, args.sasa_file)
        avg_hbond_cdr_vicinity = calculate_global_hbond_density_average(
            args.pdb_file,
            args.sasa_file,
            weights_raw=weights_raw,
            residues_for_density=cdr_vicinity_keys,
            residues_for_average=cdr_vicinity_keys,
        )

        inter_chain_buried_sasa = compute_inter_chain_buried_sasa(args.sasa_file)
        dipole_moment_magnitude = compute_dipole_moment_magnitude(pdb_atoms, pka_output_data, args.pH)

        net_charge_by_pH = {}
        protein_pi = pi_from_pka(pka_output_data)
        for ph in NET_CHARGE_PHS:
            net_charge_by_pH[ph] = net_charge_from_pka(pka_output_data, ph)  # PropKa-based
            
        weighted_scm_score_by_pH = {}
        for ph in NET_CHARGE_PHS:
            weighted_scm_score_by_pH[ph] = scm_score_from_pka(args.pdb_file, args.sasa_file, pka_output_data, ph, d_cutoff=10.0)
        scm_by_atoms = scm_score_by_atoms(args.pdb_file, d_cutoff=10.0) or {}
        scm_neg_ff19sb = scm_by_atoms.get("scm_neg_ff19sb")
        scm_pos_ff19sb = scm_by_atoms.get("scm_pos_ff19sb")

        asymmetry_score = asymmetry_score_from_pka(
            pka_output_data,
            args.heavy_chain,
            args.light_chain,
            args.pH,
        )
        residue_keys_all = set(ctx.residue_keys)
        _sasa = ctx.sasa_residue

        def _ff19sb_total_charge_for_residue(key4: ResKey4) -> float:
            backbone_q, sidechain_q = get_ff19sb_residue_region_charges(
                key4[0],
                residue_atom_names=residue_atom_names_by_key.get(key4),
            )
            return float(backbone_q + sidechain_q)

        net_charge_ff19sb = sum(_ff19sb_total_charge_for_residue(k) for k in residue_keys_all)

        hyd_asa_total = sum(
            float(getattr(entry, "non_polar_abs", 0.0)) or 0.0
            for entry in _sasa.values()
        )
        hph_asa_total = sum(
            float(getattr(entry, "all_polar_abs", 0.0)) or 0.0
            for entry in _sasa.values()
        )

        def _category_side_abs_sum(residue_keys: Set[ResKey4], res_set) -> float:
            return sum(
                float(getattr(_sasa[k], "total_side_abs", 0.0)) or 0.0
                for k in residue_keys
                if k in _sasa and k[0] in res_set
            )

        _cat = [
            ("aro", AROMATIC_RESIDUES),
            ("neg", CHARGE_FRACTION_NEGATIVE_RESIDUES),
            ("pos", CHARGE_FRACTION_POSITIVE_RESIDUES),
            ("hyd", HYDROPHOBIC_RESIDUES),
        ]
        all_sasa_keys: Set[ResKey4] = set(_sasa.keys())
        total_side_abs_sums: Dict[str, float] = {}
        for name, res_set in _cat:
            total_side_abs_sums[f"{name}_exposed_sasa"] = _category_side_abs_sum(
                exposed_keys, res_set
            )
            total_side_abs_sums[f"{name}_all_sasa"] = _category_side_abs_sum(
                all_sasa_keys, res_set
            )

        sap_scores = compute_sap_shell_synergy_scores(
            pdb_atoms,
            ctx.sasa_residue,
            pka_output_data,
            args.pH,
            d_cutoff=10.0,
        )
        pos_patch_area = pos_exposed_clusters_total_side_abs_sasa
        neg_patch_area = neg_exposed_clusters_total_side_abs_sasa
        hyd_patch_area = hydro_exposed_clusters_total_side_abs_sasa
        pos_patch_area_cdr = pos_cdr_vicinity_clusters_total_side_abs_sasa
        neg_patch_area_cdr = neg_cdr_vicinity_clusters_total_side_abs_sasa
        hyd_patch_area_cdr = hydro_cdr_vicinity_clusters_total_side_abs_sasa

        aggregated = {}

        aggregated.update(
            {
                "surface": {
                    **_pcf_cluster_per_shell_and_mean_per_category(pcf_cluster),
                    "neg_cluster_entropy": neg_cluster_side_abs_sasa_entropy,
                    "pos_cluster_entropy": pos_cluster_side_abs_sasa_entropy,
                    "hbond_density_cdr": avg_hbond_cdr_vicinity,
                    "aro_exposure_cdr": sum_aromatic_weighted_rel_side_asa_cdr_vicinity,
                    "hyd_exposure_cdr": sum_hydrophobic_weighted_rel_side_asa_cdr_vicinity,
                    "neg_exposure_cdr": sum_negative_weighted_rel_side_asa_cdr_vicinity,
                    "pos_exposure_cdr": sum_positive_weighted_rel_side_asa_cdr_vicinity,
                    "aro_sasa_cdr": sum_aromatic_side_abs_sasa_cdr_vicinity,
                    "neg_sasa_cdr": sum_negative_side_abs_sasa_cdr_vicinity,
                    "pos_sasa_cdr": sum_positive_side_abs_sasa_cdr_vicinity,
                    "neg_planarity_cdr": normalized_local_planarity_negative_cdr_vicinity,
                    "pos_planarity_cdr": normalized_local_planarity_positive_cdr_vicinity,
                    "aro_planarity_cdr": normalized_local_planarity_aromatic_cdr_vicinity,
                    "hyd_planarity_cdr": normalized_local_planarity_hydrophobic_cdr_vicinity,
                    "exposure_weighted_hyd_score_cdr": avg_kd_times_total_side_rel_cdr_vicinity_over_cdr_vicinity,
                    "exposure_weighted_salt_bridge_score_cdr": avg_salt_cdr_vicinity_over_cdr_vicinity,
                    "weighted_scm": weighted_scm_score_by_pH,
                    "scm_neg_ff19sb": scm_neg_ff19sb,
                    "scm_pos_ff19sb": scm_pos_ff19sb,
                    **{k: _scalar_or_dict_sum(v) for k, v in sap_scores.items()},
                    "pos_patch_area": pos_patch_area,
                    "neg_patch_area": neg_patch_area,
                    "hyd_patch_area": hyd_patch_area,
                    "pos_patch_area_cdr": pos_patch_area_cdr,
                    "neg_patch_area_cdr": neg_patch_area_cdr,
                    "hyd_patch_area_cdr": hyd_patch_area_cdr,
                    **total_side_abs_sums,
                },
                "core": {
                    "hbond_energy_density": avg_hbond_energy_dssp_weighted,
                    "inter_chain_buried_sasa": inter_chain_buried_sasa,
                },
                "general": {
                    "dipole_moment_ff19sb": dipole_moment_magnitude,
                    "net_charge_ff19sb": net_charge_ff19sb,
                    "protein_pi": protein_pi,
                    "net_charge_by_pH": net_charge_by_pH,
                    "asymmetry_score": asymmetry_score,
                    "hyd_asa_total": hyd_asa_total,
                    "hph_asa_total": hph_asa_total,
                    "cdr3_length": cdr3_length,
                    "fraction_gly_in_cdr": fraction_gly_in_cdr,
                    "fraction_pro_in_cdr": fraction_pro_in_cdr,
                    "fraction_aro_in_cdr": fraction_aromatic_in_cdr,
                    "fraction_pos_in_cdr": fraction_positive_in_cdr,
                    "fraction_neg_in_cdr": fraction_negative_in_cdr,
                    "fraction_gln_asn_in_cdr": fraction_gln_asn_in_cdr,
                    "number_of_salt_bridges": number_of_salt_bridges,
                    "hyd_score": kyte_doolittle_sum_all,
                },
            }
        )

        chain_order = [args.heavy_chain, args.light_chain]
        chain_seqs_and_maps: List[Tuple[str, str, Dict[Tuple[str, int, str, str], int]]] = []
        for ch in chain_order:
            seq_ch, map_ch = get_full_sequence_with_index_map_from_pdb(args.pdb_file, chain_order=[ch])
            if seq_ch:
                chain_seqs_and_maps.append((ch, seq_ch, map_ch))

        if chain_seqs_and_maps:
                sequence_motives = aggregated.setdefault("sequence_motives", {})

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
                cdr_vicinity_frags_flat = _flatten_fragments(fragments_cdr_vicinity)

                sequence_motives["count_AspAsp_cdr"] = count_motif_overlapping(
                    cdr_vicinity_frags_flat, "DD"
                )
                
                sequence_motives["count_AspGlu_cdr"] = count_motif_overlapping(
                    cdr_vicinity_frags_flat, "DE"
                )
                
                sequence_motives["count_TyrTyr_cdr"] = count_motif_overlapping(
                    cdr_vicinity_frags_flat, "YY"
                )

                sasa_residue = ctx.sasa_residue
                if sasa_residue:
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
                            for i in range(len(seq_ch) - 1):
                                if seq_ch[i] == a and seq_ch[i + 1] == b:
                                    total += float(asa[i] + asa[i + 1])
                        return float(total)

                    for motif, key in [("DD", "AspAsp"), ("DE", "AspGlu")]:
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

