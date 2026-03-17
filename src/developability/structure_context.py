
from typing import Dict, List, Optional, Set, Tuple, Iterable
import os
import logging

import numpy as np

from utils.parsers import (
    parse_structure,
    parse_sasa,
    parse_dssp,
    parse_dssp_hbonds,
    parse_pka,
    get_pka_file_path,
    Atom,
    SASAEntry,
    residue_key_from_atom,
)

logger = logging.getLogger(__name__)

# residue_name, residue_number, chain_id, insertion_code
ResKey4 = Tuple[str, int, str, str]


class StructureContext:

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
        self._pka_path = pka_path
        self._dssp_path = dssp_path

        self._atoms: Optional[List[Atom]] = None
        self._residue_keys: Optional[Set[ResKey4]] = None
        self._seq_index: Optional[Dict[ResKey4, int]] = None
        self._ca_coords: Optional[Dict[ResKey4, Tuple[float, float, float]]] = None
        self._sasa_residue: Optional[Dict[ResKey4, SASAEntry]] = None
        self._sasa_output: Optional[Dict[ResKey4, Dict[str, Optional[float]]]] = None
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
            seen: Set[ResKey4] = set()
            keys: Set[ResKey4] = set()
            for atom in self.atoms:
                key4 = residue_key_from_atom(atom)
                if key4 not in seen:
                    seen.add(key4)
                    keys.add(key4)
            if not keys:
                raise ValueError(
                    f"No residues found in structure at {self._pdb_path!r} "
                    f"(parsed {len(self.atoms)} atoms but zero unique residue keys)."
                )
            self._residue_keys = keys
        return self._residue_keys

    @property
    def ca_coords(self) -> Dict[ResKey4, Tuple[float, float, float]]:
        if self._ca_coords is None:
            ca_by_res: Dict[ResKey4, Tuple[float, float, float]] = {}
            for atom in self.atoms:
                if atom.name != "CA":
                    continue
                key = residue_key_from_atom(atom)
                if key not in ca_by_res:
                    ca_by_res[key] = (atom.x, atom.y, atom.z)
            self._ca_coords = ca_by_res
        return self._ca_coords

    @property
    def sasa_residue(self) -> Dict[ResKey4, SASAEntry]:
        if self._sasa_residue is None:
            if not self._sasa_path:
                self._sasa_residue = {}
            else:
                try:
                    data = parse_sasa(self._sasa_path)
                    if not data:
                        raise ValueError(
                            f"SASA file {self._sasa_path!r} parsed but "
                            f"contained no per-residue records."
                        )
                    try:
                        residue_keys = self.residue_keys
                    except Exception:
                        residue_keys = set()

                    if residue_keys:
                        sasa_keys = set(data.keys())
                        if sasa_keys and len(sasa_keys) < len(residue_keys):
                            missing = residue_keys - sasa_keys
                            example_missing = list(missing)[:5]
                            logger.warning(
                                "SASA missing for %d of %d residues in %r; "
                                "examples of missing keys: %r",
                                len(missing),
                                len(residue_keys),
                                self._pdb_path,
                                example_missing,
                            )

                    self._sasa_residue = data
                except Exception:
                    self._sasa_residue = {}
        return self._sasa_residue

    @property
    def sasa_output(self) -> Dict[ResKey4, Dict[str, Optional[float]]]:
        """
        SASA output suitable for reporting
        """
        if self._sasa_output is None:
            sasa_res = self.sasa_residue
            output: Dict[ResKey4, Dict[str, Optional[float]]] = {}
            for key, entry in sasa_res.items():
                total_side_rel: Optional[float] = None
                main_chain_rel: Optional[float] = None
                if getattr(entry, "total_side_rel", None) is not None:
                    try:
                        total_side_rel = float(entry.total_side_rel)
                    except (TypeError, ValueError):
                        total_side_rel = None
                if getattr(entry, "main_chain_rel", None) is not None:
                    try:
                        main_chain_rel = float(entry.main_chain_rel)
                    except (TypeError, ValueError):
                        main_chain_rel = None
                output[key] = {
                    "total_side_rel": total_side_rel,
                    "main_chain_rel": main_chain_rel,
                }
            self._sasa_output = output
        return self._sasa_output

    @property
    def pka_residue(self) -> Dict[ResKey4, float]:
        if self._pka_residue is None:
            pka_path = self._pka_path
            if pka_path is None:
                pka_path = get_pka_file_path(self._pdb_path)
            if pka_path and os.path.exists(pka_path):
                self._pka_residue = parse_pka(pka_path, self.atoms)
            else:
                self._pka_residue = {}
        return self._pka_residue


    @property
    def dssp_per_residue(self) -> Dict[ResKey4, Dict[str, Optional[float]]]:
        if self._dssp_per_residue is None:
            if not self._dssp_path:
                self._dssp_per_residue = {}
            else:
                try:
                    data = parse_dssp(self._dssp_path, self.atoms)
                    if data:
                        try:
                            residue_keys = self.residue_keys
                        except Exception:
                            residue_keys = set()
                        if residue_keys:
                            dssp_keys = set(data.keys())
                            if dssp_keys and len(dssp_keys) < len(residue_keys):
                                missing = residue_keys - dssp_keys
                                example_missing = list(missing)[:5]
                                logger.warning(
                                    "DSSP missing for %d of %d residues in %r; "
                                    "examples of missing keys: %r",
                                    len(missing),
                                    len(residue_keys),
                                    self._pdb_path,
                                    example_missing,
                                )
                    self._dssp_per_residue = data or {}
                except Exception as e:
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
                    if hbond_data:
                        try:
                            residue_keys = self.residue_keys
                        except Exception:
                            residue_keys = set()
                        if residue_keys:
                            dssp_keys = set(hbond_data.keys())
                            if dssp_keys and len(dssp_keys) < len(residue_keys):
                                missing = residue_keys - dssp_keys
                                example_missing = list(missing)[:5]
                                logger.warning(
                                    "DSSP H-bond data missing for %d of %d residues in %r; "
                                    "examples of missing keys: %r",
                                    len(missing),
                                    len(residue_keys),
                                    self._pdb_path,
                                    example_missing,
                                )
                    self._dssp_hbonds = (hbond_data or {}, dssp_seq_to_pdb or {})
                except Exception as e:
                    logger.warning(
                        "Failed to parse DSSP H-bond data from %r for structure %r: %s. "
                        "Proceeding without DSSP H-bond data.",
                        self._dssp_path,
                        self._pdb_path,
                        e,
                    )
                    self._dssp_hbonds = ({}, {})
        return self._dssp_hbonds

