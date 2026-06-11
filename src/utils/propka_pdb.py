"""PropKa-only PDB renumbering for IMGT insertion-code collisions.

PropKa treats residues with the same (residue_number, chain) as one residue
regardless of insertion code. When IMGT CDR3 produces multiple rows at the
same number (e.g. 112 / 112A / 112B), build a temporary PDB with unique
residue numbers in a high spare range, run PropKa on that file only, and map
pKa values back to structure keys via a sidecar JSON file.

SASA, DSSP, and developability geometry always use the original IMGT PDB.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.parsers import (
    ResKey4,
    parse_pka,
    parse_structure,
    residue_key_from_atom,
    resolve_external_residue_key,
)

logger = logging.getLogger(__name__)

# IMGT CDR3: at res 112, insertion codes run C→B→A→(none) toward the C-terminus.
IMGT_REVERSE_INSERTION_RESNUMS: frozenset[int] = frozenset({112})

PROPKA_SPARE_POOLS: Dict[str, Tuple[int, int]] = {
    # Three-digit spare IDs fit the 4-character PDB resseq field and stay above
    # typical IMGT antibody numbering (H/L chains ~≤150).
    "H": (901, 999),
    "L": (801, 899),
}
DEFAULT_SPARE_POOL: Tuple[int, int] = (701, 799)

MAP_SUFFIX = ".propka_map.json"


def imgt_residue_sort_key(key: ResKey4) -> Tuple[int, str]:
    """Sort IMGT residues; res 112 uses reverse insertion-code order."""
    res_num, ins_code = key[1], key[3]
    if res_num in IMGT_REVERSE_INSERTION_RESNUMS:
        if not ins_code:
            return (res_num, "~")
        return (res_num, chr(ord("A") + ord("Z") - ord(ins_code.upper())))
    return (res_num, ins_code)


def _normalize_icode(icode: str) -> str:
    return (icode or "").strip()


def _spare_pool_for_chain(chain: str) -> Tuple[int, int]:
    return PROPKA_SPARE_POOLS.get(chain, DEFAULT_SPARE_POOL)


@dataclass
class PropkaRenumberResult:
    source_pdb: str
    propka_pdb: Optional[str]
    map_path: Optional[str]
    remapped_count: int
    propka_to_structure: Dict[ResKey4, ResKey4] = field(default_factory=dict)

    @property
    def uses_temp_pdb(self) -> bool:
        return bool(self.propka_pdb and self.remapped_count > 0)


def detect_collision_clusters(
    residue_keys: Iterable[ResKey4],
) -> List[List[ResKey4]]:
    """Return collision clusters: same (chain, resseq) with >1 distinct residue key."""
    by_chain_resseq: Dict[Tuple[str, int], List[ResKey4]] = defaultdict(list)
    for key in residue_keys:
        by_chain_resseq[(key[2], key[1])].append(key)

    clusters: List[List[ResKey4]] = []
    for keys in by_chain_resseq.values():
        unique = sorted(set(keys), key=imgt_residue_sort_key)
        if len(unique) > 1:
            clusters.append(unique)
    return clusters


def _build_renumber_map(
    residue_keys: Sequence[ResKey4],
) -> Dict[ResKey4, ResKey4]:
    """Map structure keys in collision clusters to unique PropKa-safe keys."""
    clusters = detect_collision_clusters(residue_keys)
    if not clusters:
        return {}

    next_num: Dict[str, int] = {}
    forward: Dict[ResKey4, ResKey4] = {}

    for cluster in clusters:
        for structure_key in cluster:
            chain = structure_key[2]
            pool_lo, pool_hi = _spare_pool_for_chain(chain)
            if chain not in next_num:
                next_num[chain] = pool_lo
            propka_num = next_num[chain]
            if propka_num > pool_hi:
                raise ValueError(
                    f"PropKa spare residue pool exhausted for chain {chain!r} "
                    f"(limit {pool_hi}); too many insertion-code collisions."
                )
            next_num[chain] = propka_num + 1
            propka_key: ResKey4 = (structure_key[0], propka_num, chain, "")
            forward[structure_key] = propka_key
            logger.info(
                "PropKa renumber %r -> %r (chain %s resseq %s collision cluster)",
                structure_key,
                propka_key,
                chain,
                structure_key[1],
            )
    return forward


def _rewrite_pdb_residue_fields(
    line: str,
    *,
    new_resseq: int,
    clear_icode: bool,
) -> str:
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
        return line
    if new_resseq < -999 or new_resseq > 9999:
        raise ValueError(f"PropKa spare resseq {new_resseq} out of PDB resseq field range")
    icode = " " if clear_icode else line[26:27]
    # Columns 23-26 (1-based) = resseq, column 27 = insertion code.
    return f"{line[:22]}{new_resseq:4d}{icode}{line[27:]}"


def _structure_key_from_atom_line(line: str) -> Optional[ResKey4]:
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 22:
        return None
    res_name = line[17:20].strip().upper()
    chain = line[21].strip() or "_"
    try:
        resseq = int(line[22:26].strip())
    except ValueError:
        return None
    icode = _normalize_icode(line[26:27])
    return (res_name, resseq, chain, icode)


def _write_propka_pdb(
    source_pdb: Path,
    output_pdb: Path,
    forward_map: Dict[ResKey4, ResKey4],
    *,
    chain_filter: Optional[str] = None,
) -> None:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    header_prefixes = ("REMARK", "HEADER", "TITLE", "COMPND", "SOURCE")
    lines = source_pdb.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    with output_pdb.open("w", encoding="utf-8") as dst:
        for line in lines:
            if line.startswith(header_prefixes):
                dst.write(line)
        for line in lines:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if chain_filter is not None and (len(line) <= 21 or line[21] != chain_filter):
                continue
            structure_key = _structure_key_from_atom_line(line)
            if structure_key is None:
                continue
            propka_key = forward_map.get(structure_key)
            if propka_key is not None:
                line = _rewrite_pdb_residue_fields(
                    line,
                    new_resseq=propka_key[1],
                    clear_icode=True,
                )
            dst.write(line)
        dst.write("END\n")


def _save_propka_map(
    map_path: Path,
    *,
    source_pdb: Path,
    propka_pdb: Path,
    forward_map: Dict[ResKey4, ResKey4],
) -> None:
    map_path.parent.mkdir(parents=True, exist_ok=True)
    propka_to_structure = {v: k for k, v in forward_map.items()}
    payload = {
        "source_pdb": str(source_pdb.resolve()),
        "propka_pdb": str(propka_pdb.resolve()),
        "remapped_count": len(forward_map),
        "propka_to_structure": [
            [list(propka_key), list(structure_key)]
            for propka_key, structure_key in sorted(
                propka_to_structure.items(),
                key=lambda item: (item[0][2], item[0][1], item[0][0], item[0][3]),
            )
        ],
    }
    map_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_propka_map(map_path: str | Path) -> Dict[ResKey4, ResKey4]:
    """Load PropKa-key -> structure-key mapping from sidecar JSON."""
    data = json.loads(Path(map_path).read_text(encoding="utf-8"))
    out: Dict[ResKey4, ResKey4] = {}
    for propka_entry, structure_entry in data.get("propka_to_structure", []):
        propka_key: ResKey4 = (
            str(propka_entry[0]).strip(),
            int(propka_entry[1]),
            str(propka_entry[2]).strip(),
            _normalize_icode(propka_entry[3] if len(propka_entry) > 3 else ""),
        )
        structure_key: ResKey4 = (
            str(structure_entry[0]).strip(),
            int(structure_entry[1]),
            str(structure_entry[2]).strip(),
            _normalize_icode(structure_entry[3] if len(structure_entry) > 3 else ""),
        )
        out[propka_key] = structure_key
    return out


def resolve_propka_map_path(pka_path: str | Path) -> Optional[Path]:
    """Return sidecar map path for a PropKa output file, if present."""
    pka = Path(pka_path)
    candidate = pka.parent / "tmp_structures" / f"{pka.stem}{MAP_SUFFIX}"
    return candidate if candidate.is_file() else None


def prepare_propka_input(
    source_pdb: str | Path,
    *,
    tmp_dir: str | Path,
    stem: str,
    chain_filter: Optional[str] = None,
) -> PropkaRenumberResult:
    """Build PropKa input PDB (original or renumbered) and optional sidecar map."""
    source = Path(source_pdb).resolve()
    tmp_root = Path(tmp_dir).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    atoms = parse_structure(str(source))
    if chain_filter is not None:
        atoms = [a for a in atoms if a.chain == chain_filter]
    residue_keys = sorted(
        {residue_key_from_atom(a) for a in atoms},
        key=imgt_residue_sort_key,
    )
    forward_map = _build_renumber_map(residue_keys)

    if not forward_map:
        return PropkaRenumberResult(
            source_pdb=str(source),
            propka_pdb=str(source),
            map_path=None,
            remapped_count=0,
        )

    propka_pdb = tmp_root / f"{stem}.pdb"
    map_path = tmp_root / f"{stem}{MAP_SUFFIX}"
    _write_propka_pdb(source, propka_pdb, forward_map, chain_filter=chain_filter)
    _save_propka_map(
        map_path,
        source_pdb=source,
        propka_pdb=propka_pdb,
        forward_map=forward_map,
    )
    propka_to_structure = {v: k for k, v in forward_map.items()}
    return PropkaRenumberResult(
        source_pdb=str(source),
        propka_pdb=str(propka_pdb),
        map_path=str(map_path),
        remapped_count=len(forward_map),
        propka_to_structure=propka_to_structure,
    )


def translate_pka_data_to_structure(
    pka_data: Dict[ResKey4, float],
    propka_to_structure: Dict[ResKey4, ResKey4],
    structure_atoms: Sequence,
) -> Dict[ResKey4, float]:
    """Map PropKa-output keys onto IMGT structure keys."""
    pdb_residue_set: Set[ResKey4] = {
        residue_key_from_atom(atom) for atom in structure_atoms
    }
    translated: Dict[ResKey4, float] = {}

    for propka_key, pka_value in pka_data.items():
        res_name = propka_key[0]
        if propka_key in propka_to_structure:
            structure_key = propka_to_structure[propka_key]
        elif res_name in ("N+", "C-"):
            structure_key = propka_key
        else:
            structure_key = resolve_external_residue_key(propka_key, pdb_residue_set)
            if structure_key is None:
                structure_key = propka_key

        if res_name not in ("N+", "C-") and structure_key not in pdb_residue_set:
            continue

        if structure_key in translated and translated[structure_key] != pka_value:
            logger.warning(
                "Duplicate structure pKa key %r from PropKa keys (keeping first value)",
                structure_key,
            )
            continue
        if structure_key not in translated:
            translated[structure_key] = pka_value

    return translated


def parse_pka_for_structure(
    pka_path: str | Path,
    structure_atoms: Sequence,
    *,
    propka_map_path: Optional[str | Path] = None,
) -> Dict[ResKey4, float]:
    """Parse a PropKa file onto structure residue keys (with optional renumber map)."""
    pka_path = Path(pka_path)
    if propka_map_path is None:
        propka_map_path = resolve_propka_map_path(pka_path)
    raw = parse_pka(str(pka_path), None)
    if propka_map_path and Path(propka_map_path).is_file():
        propka_to_structure = load_propka_map(propka_map_path)
        return translate_pka_data_to_structure(raw, propka_to_structure, structure_atoms)
    return parse_pka(str(pka_path), list(structure_atoms))
