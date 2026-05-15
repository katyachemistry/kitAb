
from typing import Dict, List, Optional, Set, Tuple, Iterable, TypeVar
import os
import logging

import numpy as np

from utils.parsers import (
    atom_sasa_path_from_residue_sasa_path,
    parse_atom_sasa,
    parse_structure,
    parse_sasa,
    parse_dssp,
    parse_dssp_hbonds,
    parse_pka,
    get_pka_file_path,
    Atom,
    SASAEntry,
    ca_xyz_by_residue,
    residue_key_from_atom,
)
from utils.chemistry import get_standard_residue_pka

logger = logging.getLogger(__name__)

# residue_name, residue_number, chain_id, insertion_code
ResKey4 = Tuple[str, int, str, str]

T = TypeVar("T")


class StructureContext:

    def _warn_missing_coverage(
        self,
        present_keys: Set[ResKey4],
        *,
        label: str,
        source_path: Optional[str] = None,
    ) -> None:
        """
        Warn when per-residue data does not cover residues in the parsed structure.

        This is intentionally conservative: downstream descriptors often treat missing
        residues as 0.0, which can silently bias results if coverage failures occur.
        """
        try:
            residue_keys = self.residue_keys
        except Exception:
            return
        if not residue_keys:
            return

        missing = residue_keys - present_keys
        if not missing:
            return

        # Chain-level severity: disjoint chain sets strongly suggests a chain-ID mismatch.
        struct_chains = {k[2] for k in residue_keys}
        present_chains = {k[2] for k in present_keys}
        disjoint_chains = struct_chains.isdisjoint(present_chains)

        # Missing counts by chain (helps differentiate "one chain missing" vs scattered gaps).
        missing_by_chain: Dict[str, int] = {}
        for k in missing:
            missing_by_chain[k[2]] = missing_by_chain.get(k[2], 0) + 1
        missing_chain_summary = ", ".join(
            f"{chain}:{n}" for chain, n in sorted(missing_by_chain.items(), key=lambda x: (-x[1], x[0]))
        )

        example_missing = list(missing)[:5]
        logger.warning(
            "%s missing for %d of %d residues in %r%s (present chains=%r, structure chains=%r%s); "
            "missing by chain: %s; examples of missing keys: %r",
            label,
            len(missing),
            len(residue_keys),
            self._pdb_path,
            f" (source={source_path!r})" if source_path else "",
            sorted(present_chains),
            sorted(struct_chains),
            ", DISJOINT CHAIN SETS -> likely chain-id mismatch" if disjoint_chains else "",
            missing_chain_summary,
            example_missing,
        )

    def _filter_to_structure_residues(
        self,
        data: Dict[ResKey4, T],
        *,
        label: str,
        source_path: Optional[str] = None,
    ) -> Dict[ResKey4, T]:
        """
        Enforce a single canonical "key space": residues present in the parsed structure
        (after any `allowed_chains` filtering).
        """
        if not data:
            return data
        try:
            residue_keys = self.residue_keys
        except Exception:
            return data

        filtered = {k: v for k, v in data.items() if k in residue_keys}
        dropped = len(data) - len(filtered)
        if dropped > 0:
            example_dropped = [k for k in data.keys() if k not in residue_keys][:5]
            logger.warning(
                "%s contained %d residue keys not present in parsed structure for %r%s; "
                "examples of dropped keys: %r",
                label,
                dropped,
                self._pdb_path,
                f" (source={source_path!r})" if source_path else "",
                example_dropped,
            )
        return filtered

    def __init__(
        self,
        pdb_path: str,
        allowed_chains: Optional[Iterable[str]] = None,
        sasa_path: Optional[str] = None,
        pka_path: Optional[str] = None,
        dssp_path: Optional[str] = None,
    ) -> None:
        self._pdb_path = pdb_path
        self._allowed_chains = allowed_chains
        self._sasa_path = sasa_path
        self._atom_sasa_path = (
            atom_sasa_path_from_residue_sasa_path(sasa_path) if sasa_path else None
        )
        self._pka_path = pka_path
        self._dssp_path = dssp_path
        self.parse_errors: Dict[str, str] = {}

        self._atoms: Optional[List[Atom]] = None
        self._residue_keys: Optional[Set[ResKey4]] = None
        self._ca_coords: Optional[Dict[ResKey4, Tuple[float, float, float]]] = None
        self._sasa_residue: Optional[Dict[ResKey4, SASAEntry]] = None
        self._sasa_output: Optional[Dict[ResKey4, Dict[str, Optional[float]]]] = None
        self._atom_sasa: Optional[Dict[int, float]] = None
        self._pka_residue: Optional[Dict[ResKey4, float]] = None
        self._dssp_per_residue: Optional[Dict[ResKey4, Dict[str, Optional[float]]]] = None
        self._dssp_hbonds: Optional[
            Tuple[Dict[ResKey4, List[Tuple[int, float]]], Dict[int, ResKey4]]
        ] = None

    @property
    def pdb_path(self) -> str:
        return self._pdb_path

    @property
    def atoms(self) -> List[Atom]:
        if self._atoms is None:
            self._atoms = parse_structure(self._pdb_path, allowed_chains=self._allowed_chains)
        return self._atoms

    @property
    def residue_keys(self) -> Set[ResKey4]:
        if self._residue_keys is None:
            keys: Set[ResKey4] = set()
            for atom in self.atoms:
                keys.add(residue_key_from_atom(atom))
            if not keys:
                raise ValueError(
                    f"No residues found in structure at {self._pdb_path!r} "
                    f"(parsed {len(self.atoms)} atoms but zero unique residue keys)."
                )
            self._residue_keys = keys
        return self._residue_keys

    @property
    def ca_coords(self) -> Dict[ResKey4, Tuple[float, float, float]]:
        """Per-residue Cα coordinates for WCN and distance-based surface metrics."""
        if self._ca_coords is None:
            ca_by_res = ca_xyz_by_residue(self.atoms)
            try:
                residue_keys = self.residue_keys
            except Exception:
                residue_keys = set()
            if residue_keys:
                missing = residue_keys - set(ca_by_res.keys())
                if missing:
                    example_missing = list(missing)[:5]
                    logger.warning(
                        "Cα coordinates missing for %d of %d residues in %r; "
                        "downstream geometric descriptors may be biased. Examples: %r",
                        len(missing),
                        len(residue_keys),
                        self._pdb_path,
                        example_missing,
                    )
            self._ca_coords = ca_by_res
        return self._ca_coords

    @property
    def sasa_residue(self) -> Dict[ResKey4, SASAEntry]:
        if self._sasa_residue is None:
            if not self._sasa_path:
                self._sasa_residue = {}
            else:
                try:
                    data = parse_sasa(self._sasa_path).entries
                    if not data:
                        raise ValueError(
                            f"SASA file {self._sasa_path!r} parsed but "
                            f"contained no per-residue records."
                        )
                    data = self._filter_to_structure_residues(
                        data, label="SASA", source_path=self._sasa_path
                    )

                    self._warn_missing_coverage(
                        set(data.keys()), label="SASA", source_path=self._sasa_path
                    )

                    self._sasa_residue = data
                except Exception as e:
                    self.parse_errors["sasa"] = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "Failed to parse SASA file %r for structure %r: %s. "
                        "Proceeding without SASA data.",
                        self._sasa_path,
                        self._pdb_path,
                        e,
                    )
                    self._sasa_residue = {}
        return self._sasa_residue

    @property
    def sasa_output(self) -> Dict[ResKey4, Dict[str, Optional[float]]]:
        """
        SASA data in dict-of-dicts format expected by cluster metrics and SAP functions.
        Fields mirror SASAEntry (total_side_rel, total_side_abs, main_chain_abs,
        main_chain_rel, non_polar_abs, non_polar_rel, all_polar_abs, all_polar_rel);
        all are floats (0.0 when absent/N/A per parse_sasa).
        """
        if self._sasa_output is None:
            self._sasa_output = {
                key: {
                    "total_side_rel": entry.total_side_rel,
                    "total_side_abs": entry.total_side_abs,
                    "main_chain_abs": entry.main_chain_abs,
                    "main_chain_rel": entry.main_chain_rel,
                    "non_polar_abs": entry.non_polar_abs,
                    "non_polar_rel": entry.non_polar_rel,
                    "all_polar_abs": entry.all_polar_abs,
                    "all_polar_rel": entry.all_polar_rel,
                }
                for key, entry in self.sasa_residue.items()
            }
        return self._sasa_output

    @property
    def atom_sasa(self) -> Dict[int, float]:
        """
        Atom-level absolute SASA keyed by PDB atom serial.

        This is populated from the compact companion file generated by
        FreeSASA ``--format=pdb --depth=atom`` and intentionally stores only the
        per-atom SASA needed for future exposed-atom descriptors.
        """
        if self._atom_sasa is None:
            atom_sasa_path = self._atom_sasa_path
            if not atom_sasa_path or not os.path.exists(atom_sasa_path):
                self._atom_sasa = {}
            else:
                try:
                    self._atom_sasa = parse_atom_sasa(atom_sasa_path, self.atoms)
                except Exception as e:
                    self.parse_errors["atom_sasa"] = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "Failed to parse atom SASA file %r for structure %r: %s. "
                        "Proceeding without atom SASA data.",
                        atom_sasa_path,
                        self._pdb_path,
                        e,
                    )
                    self._atom_sasa = {}
        return self._atom_sasa

    @property
    def pka_residue(self) -> Dict[ResKey4, float]:
        if self._pka_residue is None:
            pka_path = self._pka_path
            if pka_path is None:
                pka_path = get_pka_file_path(self._pdb_path)
            if pka_path and os.path.exists(pka_path):
                # Parse both raw and structure-filtered to detect systematic key mismatches.
                raw = parse_pka(pka_path, None)
                filtered = parse_pka(pka_path, self.atoms)

                if raw and not filtered:
                    raw_chains = sorted({k[2] for k in raw.keys()})
                    struct_chains = sorted({k[2] for k in self.residue_keys})
                    example_raw = list(raw.keys())[:5]
                    logger.warning(
                        "pKa file %r contained %d residue records but none matched the parsed structure for %r "
                        "(raw chains=%r, structure chains=%r). Example raw keys: %r",
                        pka_path,
                        len(raw),
                        self._pdb_path,
                        raw_chains,
                        struct_chains,
                        example_raw,
                    )
                elif raw:
                    dropped = len(raw) - len(filtered)
                    if dropped > 0:
                        example_dropped = [k for k in raw.keys() if k not in filtered][:5]
                        logger.warning(
                            "pKa file %r had %d of %d residue records that did not match the parsed structure for %r; "
                            "examples of dropped keys: %r",
                            pka_path,
                            dropped,
                            len(raw),
                            self._pdb_path,
                            example_dropped,
                        )

                # Warn about titratable residues in the structure not covered by PropKa.
                # Standard pKa values are filled below, but the warning makes sparse
                # PropKa coverage visible in logs.
                self._warn_missing_coverage(
                    set(filtered.keys()), label="pKa", source_path=pka_path
                )

                # Fill any missing titratable residues with standard pKa values so
                # charge-based descriptors (net charge, pI, clustering, etc.) remain
                # well-defined even when PropKa output has incomplete coverage.
                filled = dict(filtered)
                try:
                    residue_keys = self.residue_keys
                except Exception:
                    residue_keys = set()
                for key in residue_keys:
                    if key in filled:
                        continue
                    std = get_standard_residue_pka(key[0])
                    if std is not None:
                        filled[key] = float(std)

                self._pka_residue = filled
            else:
                # No PropKa file: use standard pKa values for titratable residues
                # present in the structure (as a biologically grounded fallback).
                filled: Dict[ResKey4, float] = {}
                try:
                    residue_keys = self.residue_keys
                except Exception:
                    residue_keys = set()
                for key in residue_keys:
                    std = get_standard_residue_pka(key[0])
                    if std is not None:
                        filled[key] = float(std)
                self._pka_residue = filled
        return self._pka_residue


    @property
    def dssp_per_residue(self) -> Dict[ResKey4, Dict[str, Optional[float]]]:
        if self._dssp_per_residue is None:
            if not self._dssp_path:
                self._dssp_per_residue = {}
            else:
                try:
                    data = parse_dssp(self._dssp_path, self.atoms)
                    data = self._filter_to_structure_residues(
                        data, label="DSSP", source_path=self._dssp_path
                    )
                    if data:
                        self._warn_missing_coverage(
                            set(data.keys()), label="DSSP", source_path=self._dssp_path
                        )
                    self._dssp_per_residue = data or {}
                except Exception as e:
                    self.parse_errors["dssp"] = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "Failed to parse DSSP file %r for structure %r: %s. "
                        "Proceeding without DSSP data.",
                        self._dssp_path,
                        self._pdb_path,
                        e,
                    )
                    self._dssp_per_residue = {}
        return self._dssp_per_residue

    @property
    def dssp_hbonds(self) -> Tuple[Dict[ResKey4, List[Tuple[int, float]]], Dict[int, ResKey4]]:
        if self._dssp_hbonds is None:
            if not self._dssp_path:
                self._dssp_hbonds = ({}, {})
            else:

                try:
                    hbond_data, dssp_seq_to_pdb = parse_dssp_hbonds(self._dssp_path, self.atoms)
                    hbond_data = self._filter_to_structure_residues(
                        hbond_data, label="DSSP H-bonds", source_path=self._dssp_path
                    )
                    # Keep DSSP index mapping consistent with the same canonical residue key space.
                    try:
                        residue_keys = self.residue_keys
                    except Exception:
                        residue_keys = set()

                    seq_to_pdb: Dict[int, ResKey4] = {}
                    if dssp_seq_to_pdb:
                        if residue_keys:
                            seq_to_pdb = {
                                i: k for i, k in dssp_seq_to_pdb.items() if k in residue_keys
                            }
                            dropped = len(dssp_seq_to_pdb) - len(seq_to_pdb)
                            if dropped > 0:
                                example_dropped = [
                                    (i, k)
                                    for i, k in dssp_seq_to_pdb.items()
                                    if k not in residue_keys
                                ][:5]
                                logger.warning(
                                    "DSSP seq->PDB mapping contained %d entries not present in parsed structure for %r "
                                    "(source=%r); examples of dropped mappings: %r",
                                    dropped,
                                    self._pdb_path,
                                    self._dssp_path,
                                    example_dropped,
                                )
                        else:
                            seq_to_pdb = dict(dssp_seq_to_pdb)

                    # Drop H-bond pairs whose target DSSP index no longer resolves.
                    pruned_hbonds: Dict[ResKey4, List[Tuple[int, float]]] = {}
                    if hbond_data and seq_to_pdb:
                        pdb_to_seq = {pdb_key: dssp_seq for dssp_seq, pdb_key in seq_to_pdb.items()}
                        dropped_pairs = 0
                        total_pairs = 0
                        dropped_sources = 0

                        for res_key, pairs in hbond_data.items():
                            dssp_seq = pdb_to_seq.get(res_key)
                            if dssp_seq is None:
                                dropped_sources += 1
                                continue
                            kept: List[Tuple[int, float]] = []
                            for offset, energy in pairs:
                                total_pairs += 1
                                target_seq = dssp_seq + offset
                                if target_seq not in seq_to_pdb:
                                    dropped_pairs += 1
                                    continue
                                kept.append((offset, energy))
                            if kept:
                                pruned_hbonds[res_key] = kept

                        if dropped_sources > 0 or dropped_pairs > 0:
                            logger.warning(
                                "Pruned DSSP H-bond data for %r (source=%r): dropped %d residues with no DSSP index, "
                                "and %d of %d H-bond pairs whose targets were outside the parsed structure key space.",
                                self._pdb_path,
                                self._dssp_path,
                                dropped_sources,
                                dropped_pairs,
                                total_pairs,
                            )
                    else:
                        pruned_hbonds = hbond_data or {}

                    # Note: DSSP H-bond data is not expected to cover every residue
                    # warn on dropped keys / pruned pairs and on parse failures.
                    self._dssp_hbonds = (pruned_hbonds or {}, seq_to_pdb or {})
                except Exception as e:
                    self.parse_errors["dssp_hbonds"] = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "Failed to parse DSSP H-bond data from %r for structure %r: %s. "
                        "Proceeding without DSSP H-bond data.",
                        self._dssp_path,
                        self._pdb_path,
                        e,
                    )
                    self._dssp_hbonds = ({}, {})
        return self._dssp_hbonds

    def get_cdr_vicinity_residue_keys(
        self,
        cdr_keys: Set[ResKey4],
        *,
        radius: Optional[float] = None,
    ) -> Set[ResKey4]:
        """
        Cached CDR vicinity (see ``descriptor_utils.compute_cdr_vicinity_residue_keys``).

        Uses ``CDR_VICINITY_HEAVY_ATOM_CUTOFF`` from ``utils.chemistry`` when
        ``radius`` is omitted.
        """
        from utils.chemistry import CDR_VICINITY_HEAVY_ATOM_CUTOFF
        from developability.descriptor_utils import compute_cdr_vicinity_residue_keys

        r = float(CDR_VICINITY_HEAVY_ATOM_CUTOFF if radius is None else radius)
        return compute_cdr_vicinity_residue_keys(self.atoms, cdr_keys, r)

