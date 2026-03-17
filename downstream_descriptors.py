# """
# Compute structure-level downstream descriptors from per-residue developability output.

# Uses functions and constants from descriptors.py where applicable. Input is the
# per-residue table (DataFrame or path to CSV) produced by run_developability.py.
# Output is a single dict of structure-level metrics matching the aggregation
# logic in clean_validation.ipynb.

# Run with the developability conda environment, e.g.:
#   conda run -n developability python src/developability/downstream_descriptors.py <csv_path> [-o out.json]
# """

# from pathlib import Path
# import re
# import sys
# from typing import Any, Dict, List, Optional, Union

# _src_dir = Path(__file__).resolve().parent.parent
# if str(_src_dir) not in sys.path:
#     sys.path.insert(0, str(_src_dir))

# import numpy as np
# import pandas as pd

# # Import from descriptors for consistency
# from developability.descriptors import (
#     AA_1_TO_3,
#     AROMATIC_RESIDUES,
#     HYDROPHOBIC_RESIDUES,
#     POLAR_RESIDUES,
#     sequence_motif_count,
#     _cdr_mask,
# )
# from developability.descriptors import _residue_fractional_charge_at_pH  # used for per-chain charge

# # 3-letter to 1-letter for sequence building (inverse of AA_1_TO_3)
# THREE_TO_ONE = {v: k for k, v in AA_1_TO_3.items()}

# # Residue sets used by the notebook
# NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})
# POSITIVE_RESIDUES = frozenset({"ARG", "LYS"})
# GLN_ASN_RESIDUES = frozenset({"GLN", "ASN"})

# # Metrics for which we compute median / beta_sheet_median / buried_median / exposed_median
# METRICS = ["hbond_density", "salt_bridge_density", "wcn", "hbond_energy_dssp_density"]

# # Kyte-Doolittle hydropathy scores (same as run_developability / notebook)
# KYTE_DOOLITTLE = {
#     "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
#     "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
#     "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
#     "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
# }

# # Cluster label column names for max total_side_rel per cluster
# CLUSTER_LABEL_COLS = [
#     "negative_cluster_labels",
#     "positive_cluster_labels",
#     "hydrophobic_cluster_labels",
#     "polar_cluster_labels",
# ]


# def _ensure_dataframe(df_or_path: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
#     """Load CSV if path given, otherwise return DataFrame as-is."""
#     if isinstance(df_or_path, (str, Path)):
#         path = Path(df_or_path)
#         if path.suffix.lower() == ".csv":
#             return pd.read_csv(path)
#         raise ValueError(f"Unsupported file format: {path.suffix}")
#     return df_or_path.copy()


# def _build_full_sequence(df: pd.DataFrame) -> str:
#     """Build full Fv sequence (1-letter) from residue_name, ordered by chain then position."""
#     if "residue_name" not in df.columns:
#         return ""
#     if "chain" in df.columns:
#         parts: List[str] = []
#         for _ch, sub in df.groupby("chain"):
#             seq = "".join(THREE_TO_ONE.get(rn, "X") for rn in sub["residue_name"])
#             parts.append(seq)
#         return "".join(parts)
#     return "".join(THREE_TO_ONE.get(rn, "X") for rn in df["residue_name"])


# def _count_motif_overlapping(seq: str, motif: str) -> int:
#     """Count overlapping occurrences of motif in seq (notebook semantics)."""
#     if not seq or not motif:
#         return 0
#     n = len(motif)
#     return sum(1 for i in range(len(seq) - n + 1) if seq[i : i + n] == motif)


# def compute_downstream_descriptors(
#     df_or_path: Union[pd.DataFrame, str, Path],
#     *,
#     structure_id: Optional[str] = None,
#     base: Optional[str] = None,
#     heavy: Optional[str] = None,
#     light: Optional[str] = None,
#     inter_chain_buried_sasa: Optional[float] = None,
#     pdb_path: Optional[Union[str, Path]] = None,
#     heavy_chain_id: str = "H",
#     light_chain_id: str = "L",
#     pH: float = 7.0,
#     use_descriptors_motif: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Compute structure-level downstream descriptors from per-residue developability table.

#     Uses descriptors.py: get_residue_region (CDR ranges), residue sets, and optionally
#     sequence_motif_count(pdb_path, motif) when pdb_path is given and use_descriptors_motif=True.
#     Otherwise motif counts use overlapping count from sequence built from the table
#     (notebook semantics).

#     Args:
#         df_or_path: Per-residue DataFrame or path to developability CSV.
#         structure_id: Optional identifier (e.g. "base_0").
#         base: Optional base name.
#         heavy: Optional heavy chain sequence id.
#         light: Optional light chain sequence id.
#         inter_chain_buried_sasa: Override for structure-level buried SASA; if None,
#             taken from first row of table if column exists.
#         pdb_path: Optional PDB path for sequence_motif_count (descriptors.py).
#         heavy_chain_id: Chain id for heavy (default H).
#         light_chain_id: Chain id for light (default L).
#         pH: pH for per-residue charge (default 7.0).
#         use_descriptors_motif: If True and pdb_path set, use sequence_motif_count from
#             descriptors (non-overlapping). If False, use overlapping count from table sequence.

#     Returns:
#         Dict of structure-level descriptor names -> values (float, int, or nan).
#     """
#     df = _ensure_dataframe(df_or_path)
#     n_total = len(df)
#     result: Dict[str, Any] = {}

#     if structure_id is not None:
#         result["structure_id"] = structure_id
#     if base is not None:
#         result["base"] = base
#     if heavy is not None:
#         result["heavy"] = heavy
#     if light is not None:
#         result["light"] = light

#     # Numeric columns
#     if "total_side_rel" in df.columns:
#         df["total_side_rel"] = pd.to_numeric(df["total_side_rel"], errors="coerce")

#     # ----- Inter-chain buried SASA -----
#     if inter_chain_buried_sasa is not None:
#         result["inter_chain_buried_sasa"] = float(inter_chain_buried_sasa)
#     elif "inter_chain_buried_sasa" in df.columns and len(df) > 0:
#         result["inter_chain_buried_sasa"] = pd.to_numeric(
#             df["inter_chain_buried_sasa"], errors="coerce"
#         ).iloc[0]
#     else:
#         result["inter_chain_buried_sasa"] = np.nan

#     # ----- Inter-chain contact number -----
#     if "inter_chain_contact" in df.columns:
#         inter = df["inter_chain_contact"].astype(str).str.lower()
#         result["inter_chain_contact_number"] = (inter == "true").sum()
#     else:
#         result["inter_chain_contact_number"] = np.nan

#     # ----- Sequence and motifs -----
#     full_seq = _build_full_sequence(df)
#     if full_seq:
#         if use_descriptors_motif and pdb_path is not None:
#             result["n_motif_AsnGly"] = sequence_motif_count(str(pdb_path), "Asn-Gly")
#             result["n_motif_AsnAsp"] = sequence_motif_count(str(pdb_path), "Asn-Asp")
#             result["n_motif_AspGly"] = sequence_motif_count(str(pdb_path), "Asp-Gly")
#             result["n_motif_AspSer"] = sequence_motif_count(str(pdb_path), "Asp-Ser")
#             result["n_motif_AspAsp"] = sequence_motif_count(str(pdb_path), "Asp-Asp")
#             result["n_motif_AspThr"] = sequence_motif_count(str(pdb_path), "Asp-Thr")
#             result["n_motif_AspHis"] = sequence_motif_count(str(pdb_path), "Asp-His")
#         else:
#             result["n_motif_AsnGly"] = _count_motif_overlapping(full_seq, "NG")
#             result["n_motif_AsnAsp"] = _count_motif_overlapping(full_seq, "ND")
#             result["n_motif_AspGly"] = _count_motif_overlapping(full_seq, "DG")
#             result["n_motif_AspSer"] = _count_motif_overlapping(full_seq, "DS")
#             result["n_motif_AspAsp"] = _count_motif_overlapping(full_seq, "DD")
#             result["n_motif_AspThr"] = _count_motif_overlapping(full_seq, "DT")
#             result["n_motif_AspHis"] = _count_motif_overlapping(full_seq, "DH")
#         result["n_AspGlu"] = full_seq.count("D") + full_seq.count("E")
#         result["n_ArgLys"] = full_seq.count("R") + full_seq.count("K")
#     else:
#         for k in (
#             "n_motif_AsnGly", "n_motif_AsnAsp", "n_motif_AspGly", "n_motif_AspSer",
#             "n_motif_AspAsp", "n_motif_AspThr", "n_motif_AspHis", "n_AspGlu", "n_ArgLys",
#         ):
#             result[k] = np.nan

#     # ----- Per-chain charge at pH (using descriptors' fractional charge) -----
#     if "pka" in df.columns and "residue_name" in df.columns:
#         df["pka"] = pd.to_numeric(df["pka"], errors="coerce")

#         def res_charge(row: pd.Series) -> float:
#             res = row["residue_name"]
#             pka_val = row["pka"]
#             if pd.isna(res) or pd.isna(pka_val):
#                 return 0.0
#             return _residue_fractional_charge_at_pH(res, float(pka_val), pH)

#         df["res_charge_pH"] = df.apply(res_charge, axis=1)
#         if "chain" in df.columns:
#             chain = df["chain"].astype(str).str.upper()
#             heavy_mask = chain.str.startswith(heavy_chain_id)
#             light_mask = chain.str.startswith(light_chain_id)
#             heavy_charge = df.loc[heavy_mask, "res_charge_pH"].sum() if heavy_mask.any() else np.nan
#             light_charge = df.loc[light_mask, "res_charge_pH"].sum() if light_mask.any() else np.nan
#             result["heavy_charge_pH7"] = heavy_charge
#             result["light_charge_pH7"] = light_charge
#             if pd.notna(heavy_charge) and pd.notna(light_charge):
#                 result["heavy_light_charge_product_pH7"] = float(heavy_charge * light_charge)
#             else:
#                 result["heavy_light_charge_product_pH7"] = np.nan
#         else:
#             result["heavy_charge_pH7"] = np.nan
#             result["light_charge_pH7"] = np.nan
#             result["heavy_light_charge_product_pH7"] = np.nan
#             result["net_charge_propka_pH7"] = df["res_charge_pH"].sum()
#     else:
#         result["heavy_charge_pH7"] = np.nan
#         result["light_charge_pH7"] = np.nan
#         result["heavy_light_charge_product_pH7"] = np.nan

#     # ----- Medians for metrics -----
#     for metric in METRICS:
#         if metric in df.columns:
#             result[f"{metric}_median"] = df[metric].median()
#         else:
#             result[f"{metric}_median"] = np.nan

#     # ----- H-bonds and salt bridges (structure-level) -----
#     hbond_cols = ["N-H-->O_1", "N-H-->O_2", "O-->H-N_1", "O-->H-N_2"]
#     existing_hbond = [c for c in hbond_cols if c in df.columns]
#     if existing_hbond:
#         df[existing_hbond] = df[existing_hbond].apply(pd.to_numeric, errors="coerce").fillna(0.0)
#         per_res = (df[existing_hbond] != 0).sum(axis=1)
#         result["n_hydrogen_bonds"] = float(per_res.sum() / 2.0)
#     else:
#         result["n_hydrogen_bonds"] = 0.0
#         per_res = pd.Series(0.0, index=df.index)

#     if "salt_bridge_density" in df.columns:
#         df["salt_bridge_density"] = pd.to_numeric(df["salt_bridge_density"], errors="coerce")
#         result["n_salt_bridges"] = float(df["salt_bridge_density"].fillna(0).sum() / 2.0)
#     else:
#         result["n_salt_bridges"] = 0.0

#     if "number_of_hbonds" in df.columns:
#         deg = pd.to_numeric(df["number_of_hbonds"], errors="coerce").fillna(0)
#         E = deg.sum() / 2.0
#         N = len(df)
#         result["mean_hbond_degree"] = (2.0 * E / N) if N > 0 else np.nan
#     else:
#         result["mean_hbond_degree"] = np.nan

#     # ----- Beta sheet medians -----
#     if "secondary_structure" in df.columns:
#         beta_df = df[df["secondary_structure"] == "E"]
#         for metric in METRICS:
#             if metric in df.columns:
#                 result[f"{metric}_beta_sheet_median"] = beta_df[metric].median()
#             else:
#                 result[f"{metric}_beta_sheet_median"] = np.nan
#     else:
#         for metric in METRICS:
#             result[f"{metric}_beta_sheet_median"] = np.nan

#     # ----- Buried / exposed (total_side_rel < 25 = buried) -----
#     if "total_side_rel" in df.columns:
#         buried_df = df[df["total_side_rel"] < 25]
#         exposed_df = df[df["total_side_rel"] >= 25]
#     else:
#         buried_df = pd.DataFrame()
#         exposed_df = df

#     for metric in METRICS:
#         if metric in df.columns:
#             result[f"{metric}_buried_median"] = buried_df[metric].median() if len(buried_df) > 0 else np.nan
#             result[f"{metric}_exposed_median"] = exposed_df[metric].median() if len(exposed_df) > 0 else np.nan
#         else:
#             result[f"{metric}_buried_median"] = np.nan
#             result[f"{metric}_exposed_median"] = np.nan

#     n_buried = len(buried_df)

#     # ----- Met/Tyr counts -----
#     if "residue_name" in df.columns:
#         result["n_Met"] = (df["residue_name"] == "MET").sum()
#         result["n_Tyr"] = (df["residue_name"] == "TYR").sum()
#         result["n_Met_exposed"] = (exposed_df["residue_name"] == "MET").sum() if len(exposed_df) > 0 else 0
#         result["n_Tyr_exposed"] = (exposed_df["residue_name"] == "TYR").sum() if len(exposed_df) > 0 else 0
#     else:
#         result["n_Met"] = result["n_Tyr"] = result["n_Met_exposed"] = result["n_Tyr_exposed"] = np.nan

#     # ----- CDR region (using descriptors CDR ranges) -----
#     cdr_mask = _cdr_mask(df)
#     range_df = df[cdr_mask]
#     n_cdr = len(range_df)

#     if "hbond_density" in df.columns and n_cdr > 0:
#         result["hbond_density_CDRs_mean"] = range_df["hbond_density"].mean()
#     else:
#         result["hbond_density_CDRs_mean"] = np.nan
#     if "salt_bridge_density" in df.columns and n_cdr > 0:
#         result["salt_bridge_density_CDRs_mean"] = range_df["salt_bridge_density"].mean()
#     else:
#         result["salt_bridge_density_CDRs_mean"] = np.nan

#     total_hbond_part = per_res.sum()
#     if total_hbond_part > 0:
#         result["ratio_hbonds_CDR_to_total"] = per_res.loc[cdr_mask].sum() / total_hbond_part
#     else:
#         result["ratio_hbonds_CDR_to_total"] = np.nan

#     if "salt_bridge_density" in df.columns:
#         total_salt = df["salt_bridge_density"].sum()
#         if total_salt > 0 and n_cdr > 0:
#             result["ratio_salt_bridges_CDR_to_total"] = range_df["salt_bridge_density"].sum() / total_salt
#         else:
#             result["ratio_salt_bridges_CDR_to_total"] = 0.0 if total_salt == 0 else np.nan
#     else:
#         result["ratio_salt_bridges_CDR_to_total"] = 0.0

#     if n_cdr > 0 and "residue_name" in df.columns:
#         result["cdr_total_length"] = n_cdr
#         result["fraction_gly_CDRs"] = (range_df["residue_name"] == "GLY").sum() / n_cdr
#         result["fraction_pro_CDRs"] = (range_df["residue_name"] == "PRO").sum() / n_cdr
#         result["fraction_aromatic_CDRs"] = range_df["residue_name"].isin(AROMATIC_RESIDUES).sum() / n_cdr
#         result["fraction_gln_asn_CDRs"] = range_df["residue_name"].isin(GLN_ASN_RESIDUES).sum() / n_cdr
#         hydro_cdr = range_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum()
#         polar_cdr = range_df["residue_name"].isin(POLAR_RESIDUES).sum()
#         result["ratio_hydrophobic_to_polar_CDRs"] = (hydro_cdr / polar_cdr) if polar_cdr > 0 else np.nan
#     else:
#         result["cdr_total_length"] = np.nan
#         result["fraction_gly_CDRs"] = result["fraction_pro_CDRs"] = result["fraction_aromatic_CDRs"] = result["fraction_gln_asn_CDRs"] = np.nan
#         result["ratio_hydrophobic_to_polar_CDRs"] = np.nan

#     # ----- Fraction buried and total_side_rel median -----
#     result["fraction_buried"] = (n_buried / n_total) if n_total > 0 else np.nan
#     if "total_side_rel" in df.columns:
#         result["total_side_rel_median"] = df["total_side_rel"].median()
#     else:
#         result["total_side_rel_median"] = np.nan

#     # ----- Fraction hydrophobic / negative / positive among buried -----
#     if n_buried > 0 and "residue_name" in df.columns:
#         result["fraction_hydrophobic_buried"] = buried_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / n_buried
#         result["fraction_negative_buried"] = buried_df["residue_name"].isin(NEGATIVE_RESIDUES).sum() / n_buried
#         result["fraction_positive_buried"] = buried_df["residue_name"].isin(POSITIVE_RESIDUES).sum() / n_buried
#     else:
#         result["fraction_hydrophobic_buried"] = result["fraction_negative_buried"] = result["fraction_positive_buried"] = np.nan

#     # ----- total_side_rel sums by category (aromatic, negative, positive, polar, hydrophobic) -----
#     # All, buried, exposed, inter_chain, CDRs; fraction_*_exposed; hydrophobic_to_charged/polar ratios; sap_sum
#     if "residue_name" in df.columns and "total_side_rel" in df.columns:
#         _zero = 0.0
#         # All
#         result["aromatic_total_side_rel_sum"] = df[df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
#         result["negative_total_side_rel_sum"] = df[df["residue_name"].isin(NEGATIVE_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
#         result["positive_total_side_rel_sum"] = df[df["residue_name"].isin(POSITIVE_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
#         result["polar_total_side_rel_sum"] = df[df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
#         result["hydrophobic_total_side_rel_sum"] = df[df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() if len(df) > 0 else _zero
#         # Buried
#         result["aromatic_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
#         result["negative_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(NEGATIVE_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
#         result["positive_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(POSITIVE_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
#         result["polar_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
#         result["hydrophobic_buried_total_side_rel_sum"] = buried_df[buried_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() if len(buried_df) > 0 else _zero
#         # Exposed
#         result["aromatic_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
#         result["negative_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(NEGATIVE_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
#         result["positive_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(POSITIVE_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
#         result["polar_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
#         result["hydrophobic_exposed_total_side_rel_sum"] = exposed_df[exposed_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() if len(exposed_df) > 0 else _zero
#         # Fraction exposed: type_exposed_sum / all_exposed
#         all_exposed = float(exposed_df["total_side_rel"].sum()) if len(exposed_df) > 0 else 0.0
#         if all_exposed > 0:
#             result["fraction_hydrophobic_exposed"] = result["hydrophobic_exposed_total_side_rel_sum"] / all_exposed
#             result["fraction_negative_exposed"] = result["negative_exposed_total_side_rel_sum"] / all_exposed
#             result["fraction_positive_exposed"] = result["positive_exposed_total_side_rel_sum"] / all_exposed
#         else:
#             result["fraction_hydrophobic_exposed"] = result["fraction_negative_exposed"] = result["fraction_positive_exposed"] = np.nan
#         # Whole-structure totals (buried + exposed) and ratios
#         hydrophobic_total = result["hydrophobic_buried_total_side_rel_sum"] + result["hydrophobic_exposed_total_side_rel_sum"]
#         negative_total = result["negative_buried_total_side_rel_sum"] + result["negative_exposed_total_side_rel_sum"]
#         positive_total = result["positive_buried_total_side_rel_sum"] + result["positive_exposed_total_side_rel_sum"]
#         polar_total = result["polar_buried_total_side_rel_sum"] + result["polar_exposed_total_side_rel_sum"]
#         charged_total = negative_total + positive_total
#         result["hydrophobic_to_charged_total_side_rel_ratio"] = (hydrophobic_total / charged_total) if charged_total > 0 else np.nan
#         result["hydrophobic_to_polar_total_side_rel_ratio"] = (hydrophobic_total / polar_total) if polar_total > 0 else np.nan
#         # Inter-chain
#         if "inter_chain_contact" in df.columns:
#             inter_mask = df["inter_chain_contact"].astype(str).str.lower() == "true"
#             inter_chain_df = df[inter_mask]
#             if len(inter_chain_df) > 0:
#                 result["aromatic_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() or _zero
#                 result["negative_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(NEGATIVE_RESIDUES)]["total_side_rel"].sum() or _zero
#                 result["positive_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(POSITIVE_RESIDUES)]["total_side_rel"].sum() or _zero
#                 result["polar_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() or _zero
#                 result["hydrophobic_inter_chain_total_side_rel_sum"] = inter_chain_df[inter_chain_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() or _zero
#             else:
#                 result["aromatic_inter_chain_total_side_rel_sum"] = result["negative_inter_chain_total_side_rel_sum"] = result["positive_inter_chain_total_side_rel_sum"] = result["polar_inter_chain_total_side_rel_sum"] = result["hydrophobic_inter_chain_total_side_rel_sum"] = _zero
#         else:
#             result["aromatic_inter_chain_total_side_rel_sum"] = result["negative_inter_chain_total_side_rel_sum"] = result["positive_inter_chain_total_side_rel_sum"] = result["polar_inter_chain_total_side_rel_sum"] = result["hydrophobic_inter_chain_total_side_rel_sum"] = _zero
#         # CDRs
#         if n_cdr > 0:
#             result["aromatic_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(AROMATIC_RESIDUES)]["total_side_rel"].sum() or _zero
#             result["negative_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(NEGATIVE_RESIDUES)]["total_side_rel"].sum() or _zero
#             result["positive_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(POSITIVE_RESIDUES)]["total_side_rel"].sum() or _zero
#             result["polar_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(POLAR_RESIDUES)]["total_side_rel"].sum() or _zero
#             result["hydrophobic_CDRs_total_side_rel_sum"] = range_df[range_df["residue_name"].isin(HYDROPHOBIC_RESIDUES)]["total_side_rel"].sum() or _zero
#         else:
#             result["aromatic_CDRs_total_side_rel_sum"] = result["negative_CDRs_total_side_rel_sum"] = result["positive_CDRs_total_side_rel_sum"] = result["polar_CDRs_total_side_rel_sum"] = result["hydrophobic_CDRs_total_side_rel_sum"] = _zero
#         # SAP sum
#         if "SAP" in df.columns:
#             result["sap_sum"] = float(pd.to_numeric(df["SAP"], errors="coerce").sum())
#         else:
#             result["sap_sum"] = np.nan
#     else:
#         _nan_keys = (
#             "aromatic_total_side_rel_sum", "negative_total_side_rel_sum", "positive_total_side_rel_sum",
#             "polar_total_side_rel_sum", "hydrophobic_total_side_rel_sum",
#             "aromatic_buried_total_side_rel_sum", "negative_buried_total_side_rel_sum", "positive_buried_total_side_rel_sum",
#             "polar_buried_total_side_rel_sum", "hydrophobic_buried_total_side_rel_sum",
#             "aromatic_exposed_total_side_rel_sum", "negative_exposed_total_side_rel_sum", "positive_exposed_total_side_rel_sum",
#             "polar_exposed_total_side_rel_sum", "hydrophobic_exposed_total_side_rel_sum",
#             "fraction_hydrophobic_exposed", "fraction_negative_exposed", "fraction_positive_exposed",
#             "hydrophobic_to_charged_total_side_rel_ratio", "hydrophobic_to_polar_total_side_rel_ratio",
#             "aromatic_inter_chain_total_side_rel_sum", "negative_inter_chain_total_side_rel_sum", "positive_inter_chain_total_side_rel_sum",
#             "polar_inter_chain_total_side_rel_sum", "hydrophobic_inter_chain_total_side_rel_sum",
#             "aromatic_CDRs_total_side_rel_sum", "negative_CDRs_total_side_rel_sum", "positive_CDRs_total_side_rel_sum",
#             "polar_CDRs_total_side_rel_sum", "hydrophobic_CDRs_total_side_rel_sum",
#             "sap_sum",
#         )
#         for k in _nan_keys:
#             result[k] = np.nan if k in ("sap_sum", "fraction_hydrophobic_exposed", "fraction_negative_exposed", "fraction_positive_exposed", "hydrophobic_to_charged_total_side_rel_ratio", "hydrophobic_to_polar_total_side_rel_ratio") else 0.0

#     # ----- Kyte-Doolittle sums (overall, beta sheet, buried, exposed) -----
#     if "residue_name" in df.columns:
#         result["hydrophobic_kyte_doolittle_sum"] = df["residue_name"].apply(
#             lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
#         ).sum()
#         if "secondary_structure" in df.columns:
#             beta_df = df[df["secondary_structure"] == "E"]
#             if len(beta_df) > 0:
#                 result["fraction_hydrophobic_beta_sheet"] = beta_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / len(beta_df)
#                 result["fraction_gln_asn_beta_sheet"] = beta_df["residue_name"].isin(GLN_ASN_RESIDUES).sum() / len(beta_df)
#                 result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = beta_df["residue_name"].apply(
#                     lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
#                 ).sum()
#             else:
#                 result["fraction_hydrophobic_beta_sheet"] = result["fraction_gln_asn_beta_sheet"] = np.nan
#                 result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = 0.0
#         else:
#             result["fraction_hydrophobic_beta_sheet"] = result["fraction_gln_asn_beta_sheet"] = np.nan
#             result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = 0.0
#         if n_buried > 0:
#             result["hydrophobic_buried_kyte_doolittle_sum"] = buried_df["residue_name"].apply(
#                 lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
#             ).sum()
#         else:
#             result["hydrophobic_buried_kyte_doolittle_sum"] = 0.0
#         if len(exposed_df) > 0:
#             result["hydrophobic_exposed_kyte_doolittle_sum"] = exposed_df["residue_name"].apply(
#                 lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
#             ).sum()
#         else:
#             result["hydrophobic_exposed_kyte_doolittle_sum"] = 0.0
#     else:
#         result["hydrophobic_kyte_doolittle_sum"] = 0.0
#         result["fraction_hydrophobic_beta_sheet"] = result["fraction_gln_asn_beta_sheet"] = np.nan
#         result["hydrophobic_beta_sheet_kyte_doolittle_sum"] = result["hydrophobic_buried_kyte_doolittle_sum"] = result["hydrophobic_exposed_kyte_doolittle_sum"] = 0.0

#     # ----- Kyte-Doolittle mean and SASA-weighted -----
#     if "residue_name" in df.columns and "total_side_rel" in df.columns:
#         df["kyte_doolittle"] = df["residue_name"].apply(
#             lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
#         )
#         result["kyte_doolittle_mean"] = df["kyte_doolittle"].mean()
#         valid = df["total_side_rel"].notna() & df["kyte_doolittle"].notna()
#         if valid.sum() > 0:
#             wsum = (df.loc[valid, "total_side_rel"] * df.loc[valid, "kyte_doolittle"]).sum()
#             asa_sum = df.loc[valid, "total_side_rel"].sum()
#             result["kyte_doolittle_weighted_by_side_asa"] = (wsum / asa_sum) if asa_sum > 0 else np.nan
#         else:
#             result["kyte_doolittle_weighted_by_side_asa"] = np.nan
#     else:
#         result["kyte_doolittle_mean"] = result["kyte_doolittle_weighted_by_side_asa"] = np.nan

#     # ----- Interface: ratio hydrophobic/polar residues -----
#     if "inter_chain_contact" in df.columns and "residue_name" in df.columns:
#         inter = df["inter_chain_contact"].astype(str).str.lower() == "true"
#         inter_df = df[inter]
#         if len(inter_df) > 0:
#             polar_inter = inter_df["residue_name"].isin(POLAR_RESIDUES).sum()
#             hydro_inter = inter_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum()
#             result["ratio_hydrophobic_to_polar_residues_interface"] = (hydro_inter / polar_inter) if polar_inter > 0 else np.nan
#         else:
#             result["ratio_hydrophobic_to_polar_residues_interface"] = np.nan
#         result["hydrophobic_to_polar_sasa_interface_ratio"] = np.nan  # optional SASA-based; not computed here
#     else:
#         result["ratio_hydrophobic_to_polar_residues_interface"] = result["hydrophobic_to_polar_sasa_interface_ratio"] = np.nan

#     # ----- Fraction hydrophobic at interface; Kyte-Doolittle sum at interface -----
#     if "inter_chain_contact" in df.columns and "residue_name" in df.columns:
#         inter = df["inter_chain_contact"].astype(str).str.lower() == "true"
#         inter_df = df[inter]
#         if len(inter_df) > 0:
#             result["fraction_hydrophobic_inter_chain"] = inter_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / len(inter_df)
#             result["hydrophobic_inter_chain_kyte_doolittle_sum"] = inter_df["residue_name"].apply(
#                 lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
#             ).sum()
#         else:
#             result["fraction_hydrophobic_inter_chain"] = np.nan
#             result["hydrophobic_inter_chain_kyte_doolittle_sum"] = 0.0
#     else:
#         result["fraction_hydrophobic_inter_chain"] = np.nan
#         result["hydrophobic_inter_chain_kyte_doolittle_sum"] = 0.0

#     # ----- Fraction hydrophobic in CDRs; Kyte-Doolittle sum in CDRs -----
#     if n_cdr > 0 and "residue_name" in df.columns:
#         result["fraction_hydrophobic_CDRs"] = range_df["residue_name"].isin(HYDROPHOBIC_RESIDUES).sum() / n_cdr
#         result["hydrophobic_CDRs_kyte_doolittle_sum"] = range_df["residue_name"].apply(
#             lambda x: KYTE_DOOLITTLE.get(x, 0.0) if pd.notna(x) else 0.0
#         ).sum()
#     else:
#         result["fraction_hydrophobic_CDRs"] = np.nan
#         result["hydrophobic_CDRs_kyte_doolittle_sum"] = 0.0

#     # ----- Cluster max total_side_rel -----
#     for col in CLUSTER_LABEL_COLS:
#         feat = col.replace("_cluster_labels", "_cluster_max_total_side_rel")
#         if col in df.columns and "total_side_rel" in df.columns:
#             d = df.copy()
#             d[col] = d[col].astype(str).str.strip()
#             valid = d[col].str.len() > 0
#             if valid.any():
#                 sums = d.loc[valid].groupby(col)["total_side_rel"].sum()
#                 result[feat] = float(sums.max()) if len(sums) > 0 else np.nan
#             else:
#                 result[feat] = np.nan
#         else:
#             result[feat] = np.nan

#     # ----- Row counts -----
#     result["n_total_rows"] = n_total
#     result["n_filtered_rows"] = n_buried
#     result["n_beta_sheet_rows"] = len(df[df["secondary_structure"] == "E"]) if "secondary_structure" in df.columns else 0
#     result["n_exposed_rows"] = len(exposed_df)

#     # ----- Inter-chain density means/medians -----
#     if "inter_chain_contact" in df.columns:
#         inter = df["inter_chain_contact"].astype(str).str.lower() == "true"
#         inter_df = df[inter]
#         if len(inter_df) > 0:
#             result["hbond_density_inter_chain_mean"] = inter_df["hbond_density"].mean() if "hbond_density" in df.columns else np.nan
#             result["salt_bridge_density_inter_chain_median"] = inter_df["salt_bridge_density"].median() if "salt_bridge_density" in df.columns else np.nan
#             result["hbond_energy_dssp_density_inter_chain_median"] = inter_df["hbond_energy_dssp_density"].median() if "hbond_energy_dssp_density" in df.columns else np.nan
#         else:
#             result["hbond_density_inter_chain_mean"] = result["salt_bridge_density_inter_chain_median"] = result["hbond_energy_dssp_density_inter_chain_median"] = np.nan
#     else:
#         result["hbond_density_inter_chain_mean"] = result["salt_bridge_density_inter_chain_median"] = result["hbond_energy_dssp_density_inter_chain_median"] = np.nan

#     # ----- Structure-level from first row (pass-through) -----
#     ripley = ["ripley_k_negative", "ripley_k_positive", "ripley_k_hydrophobic", "ripley_k_polar"]
#     psh_ppc_pnc = ["psh_all_surface_exposed", "psh_cdr_vicinity", "ppc_all_surface_exposed", "ppc_cdr_vicinity", "pnc_all_surface_exposed", "pnc_cdr_vicinity"]
#     whole = ["dipole_moment_magnitude", "largest_hbond_component_size", "net_charge", "protein_pi", "scm_score"] + [f"net_charge_pH{p}" for p in [4, 5, 6, 7, 8, 9, 10]]
#     for col in ripley + psh_ppc_pnc + whole:
#         if col in df.columns and len(df) > 0:
#             result[col] = pd.to_numeric(df[col], errors="coerce").iloc[0]
#         else:
#             result[col] = np.nan

#     return result


# def main() -> None:
#     import argparse
#     import json

#     parser = argparse.ArgumentParser(
#         description="Compute structure-level downstream descriptors from developability per-residue CSV."
#     )
#     parser.add_argument("csv_path", type=Path, help="Path to developability CSV (per-residue).")
#     parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path (default: stdout).")
#     parser.add_argument("--structure-id", type=str, default=None)
#     parser.add_argument("--base", type=str, default=None)
#     parser.add_argument("--heavy", type=str, default=None)
#     parser.add_argument("--light", type=str, default=None)
#     parser.add_argument("--inter-chain-buried-sasa", type=float, default=None)
#     parser.add_argument("--pdb", type=Path, default=None, help="PDB path for sequence_motif_count (optional).")
#     parser.add_argument("--heavy-chain", type=str, default="H")
#     parser.add_argument("--light-chain", type=str, default="L")
#     parser.add_argument("--pH", type=float, default=7.0)
#     parser.add_argument("--use-descriptors-motif", action="store_true", help="Use descriptors.sequence_motif_count (non-overlapping).")
#     args = parser.parse_args()

#     out = compute_downstream_descriptors(
#         args.csv_path,
#         structure_id=args.structure_id,
#         base=args.base,
#         heavy=args.heavy,
#         light=args.light,
#         inter_chain_buried_sasa=args.inter_chain_buried_sasa,
#         pdb_path=args.pdb,
#         heavy_chain_id=args.heavy_chain,
#         light_chain_id=args.light_chain,
#         pH=args.pH,
#         use_descriptors_motif=args.use_descriptors_motif,
#     )
#     # Convert nan to None for JSON
#     def _sanitize(obj: Any) -> Any:
#         if isinstance(obj, dict):
#             return {k: _sanitize(v) for k, v in obj.items()}
#         if isinstance(obj, (list, tuple)):
#             return [_sanitize(x) for x in obj]
#         if isinstance(obj, (np.floating, float)) and np.isnan(obj):
#             return None
#         if isinstance(obj, (np.integer, np.int64)):
#             return int(obj)
#         return obj

#     out = _sanitize(out)
#     text = json.dumps(out, indent=2)
#     if args.output is None:
#         print(text)
#     else:
#         Path(args.output).write_text(text)
#         print(f"Wrote {args.output}", file=sys.stderr)


# if __name__ == "__main__":
#     main()
