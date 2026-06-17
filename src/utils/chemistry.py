from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

EXPOSURE_REL_ASA_THRESHOLD: float = 0.20

DBSCAN_EPS_MIN_SAMPLES_BY_CATEGORY_DEFAULT: Dict[str, Tuple[float, int]] = {
    "negative": (10.0, 2),
    "positive": (10.0, 2),
    "hydrophobic": (5.0, 3),
    "aromatic": (7.0, 3),
}

SURFACE_EXPOSED_THRESHOLD_DEFAULT: float = EXPOSURE_REL_ASA_THRESHOLD

PCF_CLUSTER_BIN_STARTS_DEFAULT: Tuple[float, ...] = (
    # 3.0, 4.0, 5.0,  # bin width 1 shells (3w1A, 4w1A, 5w1A)
    # 3.0, 5.0,  # bin width 2 shells (3w2A, 5w2A)
    4.0,
    7.0,
)
PCF_CLUSTER_BIN_WIDTHS_DEFAULT: Tuple[float, ...] = (
    # 1.0, 1.0, 1.0,  # bin width 1
    # 2.0, 2.0,  # bin width 2
    3.0,
    3.0,
)
PCF_CLUSTER_N_PERMUTATIONS_DEFAULT: int = 3000

CDR_VICINITY_HEAVY_ATOM_CUTOFF: float = 5.0

STANDARD_RESIDUE_PKA: Mapping[str, float] = {
    "ARG": 12.48,
    "ASP": 3.90,
    "GLU": 4.07,
    "HIS": 6.04,
    "LYS": 10.54,
    "TYR": 10.46,
    "CYS": 8.37
}

def get_standard_residue_pka(residue_name: str) -> Optional[float]:
    return STANDARD_RESIDUE_PKA.get((residue_name or "").strip().upper())

AA_1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

THREE_TO_ONE = {v: k for k, v in AA_1_TO_3.items()}

POSITIVE_ATOMS = frozenset(
    {
        ("NH1", "ARG"),
        ("NH2", "ARG"),
        ("NZ", "LYS"),
        ("ND1", "HIS"),
        ("NE2", "HIS"),
    }
)

NEGATIVE_ATOMS = frozenset(
    {
        ("OD1", "ASP"),
        ("OD2", "ASP"),
        ("OE1", "GLU"),
        ("OE2", "GLU"),
        ("OH", "TYR"),
        ("SG", "CYS"),
    }
)

# Only canonical strong-acid residues form proper salt bridges; TYR/CYS are
# ionizable but too weakly acidic at physiological pH to be salt-bridge partners.
SALT_BRIDGE_NEGATIVE_ATOMS = frozenset(
    {
        ("OD1", "ASP"),
        ("OD2", "ASP"),
        ("OE1", "GLU"),
        ("OE2", "GLU"),
    }
)

GLN_ASN_RESIDUES = frozenset({"GLN", "ASN"})

# Kyte-Doolittle (inactive in pipeline; legacy export for run_developability_fix_results.py).
KYTE_DOOLITTLE: Dict[str, float] = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}

WIMLEY_WHITE: Dict[str, float] = {
    "ALA": -0.20,
    "ARG": -0.41,
    "ASN": -0.51,
    "ASP": -1.53,
    "CYS": 0.31,
    "GLN": -0.71,
    "GLU": -2.51,
    "GLY": 0.00,
    "HIS": -0.20,
    "ILE": 0.35,
    "LEU": 0.71,
    "LYS": -0.59,
    "MET": 0.30,
    "PHE": 1.43,
    "PRO": -0.55,
    "SER": -0.15,
    "THR": -0.16,
    "TRP": 2.33,
    "TYR": 1.19,
    "VAL": -0.08,
}

# Active residue hydrophobicity lookup for developability descriptors.
RESIDUE_HYDROPHOBICITY: Mapping[str, float] = WIMLEY_WHITE


def residue_hydrophobicity(residue_name: str) -> float:
    """Per-residue hydrophobicity from the active scale (Wimley-White)."""
    return float(RESIDUE_HYDROPHOBICITY.get((residue_name or "").strip().upper(), 0.0))

NEGATIVE_CHARGED_RESIDUES = frozenset({res for _atom, res in NEGATIVE_ATOMS})
POSITIVE_CHARGED_RESIDUES = frozenset({res for _atom, res in POSITIVE_ATOMS})

AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP", "HIS"})


# NONPOLAR_RESIDUES = frozenset(
#     {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO", "CYS"}
# )
# https://pmc.ncbi.nlm.nih.gov/articles/PMC9475378/#s3
NONPOLAR_RESIDUES = frozenset(
    {"VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "CYS"}
)
HYDROPHOBIC_RESIDUES = NONPOLAR_RESIDUES

MAX_HBOND_DISTANCE = 3.2
MAX_SALT_BRIDGE_DISTANCE = 4.0
MIN_HBOND_ANGLE = 120.0
MIN_BACKBONE_SEPARATION = 3
NTERM_PKA = 8.0
CTERM_PKA = 3.1

SCM_MAIN_CHAIN_ATOMS = frozenset({"CA", "HA", "N", "C", "O", "HN", "H"})

_AMINO19_LIB_PATH = Path(__file__).resolve().parent.parent / "amino19.lib"
_AMINO19_ENTRY_ATOMS_RE = re.compile(r"^!entry\.([A-Za-z0-9]+)\.unit\.atoms table")
_AMINO19_ATOM_LINE_RE = re.compile(
    r'^\s*"([^"]+)"\s*"([^"]+)"\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s*$'
)


def _parse_amino19_lib_atom_charges(lib_path: Path) -> Dict[Tuple[str, str], float]:
    charges: Dict[Tuple[str, str], float] = {}
    if not lib_path.is_file():
        return charges

    current_res = ""
    in_atoms = False
    with open(lib_path, "r", encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("!!"):
                continue
            if not line.strip():
                continue

            m_entry = _AMINO19_ENTRY_ATOMS_RE.match(line)
            if m_entry:
                current_res = m_entry.group(1).upper()
                in_atoms = True
                continue

            if in_atoms:
                if line.startswith("!"):
                    in_atoms = False
                    current_res = ""
                    continue
                m_at = _AMINO19_ATOM_LINE_RE.match(line)
                if not m_at:
                    continue
                atom_name = m_at.group(1).strip().upper()
                chg = float(m_at.group(8))
                charges[(current_res, atom_name)] = chg

    return charges


def _ff19sb_charges_with_his_alias(
    raw: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    out = dict(raw)
    for (res, atom), q in raw.items():
        if res == "HIE":
            out[("HIS", atom)] = q
    return out


FF19SB_ATOM_CHARGES: Dict[Tuple[str, str], float] = _ff19sb_charges_with_his_alias(
    _parse_amino19_lib_atom_charges(_AMINO19_LIB_PATH)
)


def _resolve_histidine_ff19sb_residue(
    residue_atom_names: Optional[Set[str]],
) -> str:
    if not residue_atom_names:
        return "HIE"
    names = {n.strip().upper() for n in residue_atom_names}
    has_hd1 = "HD1" in names
    has_he2 = "HE2" in names
    if has_hd1 and has_he2:
        return "HIP"
    if has_hd1:
        return "HID"
    return "HIE"


def get_ff19sb_atom_charge_with_source(
    residue_name: str,
    atom_name: str,
    *,
    residue_atom_names: Optional[Set[str]] = None,
) -> Tuple[float, str]:
    res = (residue_name or "").strip().upper()
    atom = (atom_name or "").strip().upper()
    lookup_res = _resolve_histidine_ff19sb_residue(residue_atom_names) if res == "HIS" else res

    q = FF19SB_ATOM_CHARGES.get((lookup_res, atom))
    if q is not None:
        return float(q), "direct"

    if atom == "OXT":
        q = FF19SB_ATOM_CHARGES.get((lookup_res, "O"))
        if q is not None:
            return float(q), "fallback_oxt_to_o"

    if atom in {"H1", "H2", "H3"}:
        q = FF19SB_ATOM_CHARGES.get((lookup_res, "H"))
        if q is not None:
            return float(q), "fallback_h123_to_h"

    return 0.0, "missing"


_FF19SB_RESIDUE_REGION_CHARGE_CACHE: Dict[str, Tuple[float, float]] = {}


def get_ff19sb_residue_region_charges(
    residue_name: str,
    *,
    residue_atom_names: Optional[Set[str]] = None,
) -> Tuple[float, float]:
    res = (residue_name or "").strip().upper()
    lookup_res = _resolve_histidine_ff19sb_residue(residue_atom_names) if res == "HIS" else res
    cached = _FF19SB_RESIDUE_REGION_CHARGE_CACHE.get(lookup_res)
    if cached is not None:
        return cached

    backbone_charge = 0.0
    sidechain_charge = 0.0
    for (entry_res, atom_name), charge in FF19SB_ATOM_CHARGES.items():
        if entry_res != lookup_res:
            continue
        if atom_name in BACKBONE_ATOMS:
            backbone_charge += float(charge)
        else:
            sidechain_charge += float(charge)

    result = (float(backbone_charge), float(sidechain_charge))
    _FF19SB_RESIDUE_REGION_CHARGE_CACHE[lookup_res] = result
    return result


def ff19sb_charge_verification_rows(
    *,
    residues_atoms: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, float]]:
    if residues_atoms is None:
        residues_atoms = [
            ("ASP", "OD1"),
            ("ASP", "CG"),
            ("GLU", "OE2"),
            ("ARG", "NH1"),
            ("ARG", "CZ"),
            ("LYS", "NZ"),
            ("ALA", "CA"),
            ("ALA", "CB"),
            ("SER", "OG"),
            ("SER", "CB"),
        ]
    rows: List[Tuple[str, str, float]] = []
    for res, atom in residues_atoms:
        q, _source = get_ff19sb_atom_charge_with_source(res, atom)
        rows.append((res, atom, q))
    return rows


DONORS_ANY = frozenset({"N"})
DONOR_EXCLUDED = frozenset({("N", "PRO")})

DONOR_METADATA: Dict[Tuple[str, str], Tuple[str, int]] = {
    ("NE2", "GLN"): ("CD", 2),
    ("ND2", "ASN"): ("CG", 2),
    ("NE",  "ARG"): ("CZ", 1),
    ("NH1", "ARG"): ("CZ", 2),
    ("NH2", "ARG"): ("CZ", 2),
    ("NZ",  "LYS"): ("CE", 3),
    ("ND1", "HIS"): ("CG", 1),
    ("NE2", "HIS"): ("CD2", 1),
    ("OG",  "SER"): ("CB", 1),
    ("OG1", "THR"): ("CB", 1),
    ("OH",  "TYR"): ("CZ", 1),
    ("NE1", "TRP"): ("CD1", 1),
}
DONOR_MAX_HBONDS: Dict[Tuple[str, str], int] = {
    ("N", "ANY"): 1,
    **{key: max_hbonds for key, (_base, max_hbonds) in DONOR_METADATA.items()},
}

ACCEPTORS_ANY = frozenset({"O"})

ACCEPTOR_METADATA: Dict[Tuple[str, str], int] = {
    ("O",   "ANY"): 2,
    ("OE1", "GLN"): 1,
    ("OE1", "GLU"): 2,
    ("OE2", "GLU"): 2,
    ("OD1", "ASN"): 1,
    ("OD1", "ASP"): 2,
    ("OD2", "ASP"): 2,
    ("ND1", "HIS"): 1,
    ("NE2", "HIS"): 1,
    ("OG",  "SER"): 1,
    ("OG1", "THR"): 1,
    ("OH",  "TYR"): 1,
}
ACCEPTORS_SPECIFIC = frozenset(
    key for key in ACCEPTOR_METADATA if not (key[0] == "O" and key[1] == "ANY")
)
ACCEPTOR_MAX_HBONDS: Dict[Tuple[str, str], int] = dict(ACCEPTOR_METADATA)


BACKBONE_ATOMS = frozenset({
    "N", "CA", "C", "O",
    "H", "HA", "HA2", "HA3",
    "H1", "H2", "H3", "HN",
})


if __name__ == "__main__":
    print("ff19SB atom charges (amino19.lib) — sample rows")
    print(f"{'Residue':<8} {'Atom':<6} {'Charge':>12}")
    print("-" * 28)
    for res, atom, q in ff19sb_charge_verification_rows():
        print(f"{res:<8} {atom:<6} {q:12.6f}")
    n = len(FF19SB_ATOM_CHARGES)
    print("-" * 28)
    print(f"Total (residue, atom) entries: {n}")
