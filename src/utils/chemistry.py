from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Set, Tuple

if TYPE_CHECKING:
    from utils.parsers import Atom

# Relative side-chain SASA threshold (fraction in [0, 1]) above which a residue
# is considered surface-exposed for clustering, SASA-weighted descriptors, etc.
# Net-charge-on-surface metrics in ``run_developability`` use
# ``NET_CHARGE_EXPOSURE_REL_ASA_THRESHOLD`` instead.
EXPOSURE_REL_ASA_THRESHOLD: float = 0.20

# Rel side-chain SASA cutoff for net-charge-on-surface metrics only (run_developability:
# exposed_net_charge*, exposed_net_charge_*_simple).  Looser than
# ``EXPOSURE_REL_ASA_THRESHOLD`` so more ionizable side chains count as exposed.
NET_CHARGE_EXPOSURE_REL_ASA_THRESHOLD: float = 0.05

# DBSCAN on Cα per residue category for surface clustering (eps Å, min_samples).
DBSCAN_EPS_MIN_SAMPLES_BY_CATEGORY_DEFAULT: Dict[str, Tuple[float, int]] = {
    "negative": (10.0, 2),
    "positive": (10.0, 2),
    "polar": (10.0, 2),
    "hydrophobic": (5.0, 3),
    "aromatic": (7.0, 3),
}

# Developability surface-descriptor defaults (see ``developability.descriptors``).
SURFACE_EXPOSED_THRESHOLD_DEFAULT: float = EXPOSURE_REL_ASA_THRESHOLD

# Fine shells first (1 Å), then wide shells (2 Å). Same ``r`` can repeat; keys use width (e.g. ``3w1A`` vs ``3w2A``).
PCF_CLUSTER_BIN_STARTS_DEFAULT: Tuple[float, ...] = (3.0, 4.0, 5.0, 3.0, 5.0)
PCF_CLUSTER_BIN_WIDTHS_DEFAULT: Tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 2.0)
# Shells: [3,4), [4,5), [5,6), [3,5), [5,7) Å.
PCF_CLUSTER_N_PERMUTATIONS_DEFAULT: int = 3000

RIPLEY_K_DISTANCE: float = 8.0
RIPLEY_K_N_SAMPLES: int = 1000

ANN_INDEX_N_PERMUTATIONS_DEFAULT: int = 1000
ANN_INDEX_SASA_CUTOFF_DEFAULT: float = EXPOSURE_REL_ASA_THRESHOLD

PSH_PAIR_RADIUS: float = 7.5
CDR_VICINITY_RADIUS: float = 4.0

# Heavy-atom distance cutoff (Å) for “CDR vicinity” in SASA-weighted residue
# densities and similar: any residue with a heavy atom within this distance of
# any heavy atom of a CDR residue (CDR residues always included). This is
# intentionally 5 Å and structure-wide; ``CDR_VICINITY_RADIUS`` (4 Å) is still
# used by ``compute_surface_pair_descriptors``, which also seeds only from
# exposed CDR residues.
CDR_VICINITY_HEAVY_ATOM_CUTOFF: float = 5.0

# Surface ANN index residue filters (match PROPERMAB ``ann_index`` prop filters).
_ANN_INDEX_POSITIVE_RESIDUES = frozenset({"ARG", "LYS", "HIS"})
_ANN_INDEX_NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})
_ANN_INDEX_AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP"})

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
        # Phenolic / thiol deprotonation.
        ("OH", "TYR"),
        ("SG", "CYS"),
    }
)

# Residue sets used by the notebook
GLN_ASN_RESIDUES = frozenset({"GLN", "ASN"})

# Metrics for which we compute median / beta_sheet_median / buried_median / exposed_median
# METRICS = ["hbond_density", "salt_bridge_density", "wcn", "hbond_energy_dssp_density"]

KYTE_DOOLITTLE = {
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
KD_MIN = min(KYTE_DOOLITTLE.values())
KD_MAX = max(KYTE_DOOLITTLE.values())


def normalize_hydropathy(res_name: str) -> Optional[float]:
    """Normalize Kyte-Doolittle score to [1, 2]. Returns None if not in table."""
    score = KYTE_DOOLITTLE.get(res_name)
    if score is None:
        return None
    if KD_MAX == KD_MIN:
        return 1.5
    return 1.0 + (score - KD_MIN) / (KD_MAX - KD_MIN)


NEGATIVE_CHARGED_RESIDUES = frozenset({res for _atom, res in NEGATIVE_ATOMS})
POSITIVE_CHARGED_RESIDUES = frozenset({res for _atom, res in POSITIVE_ATOMS})

# Simple residue-count / SASA-fraction metrics (fraction_negative_*, fraction_positive_*,
# matching exposed-side-abs sums and density averages): carboxylate and primary/basic
# side chains only. HIS / TYR / CYS remain on pKa-aware paths (HH, clustering, SAP modes).
CHARGE_FRACTION_NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})
CHARGE_FRACTION_POSITIVE_RESIDUES = frozenset({"LYS", "ARG"})

POLAR_RESIDUES = frozenset(
    {"SER", "THR", "ASN", "GLN", "TYR", "GLU", "ASP", "LYS", "ARG", "HIS", "CYS"}
)
AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP"})
HYDROPHOBIC_RESIDUES = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
)

MAX_HBOND_DISTANCE = 3.2
MAX_SALT_BRIDGE_DISTANCE = 4.0
INTER_CHAIN_INTERFACE_CUTOFF = 5.0
MIN_HBOND_ANGLE = 120.0  # angle base -> donor -> acceptor
MIN_BACKBONE_SEPARATION = 3
NTERM_PKA = 8.0
CTERM_PKA = 3.1

SCM_MAIN_CHAIN_ATOMS = frozenset({"CA", "HA", "N", "C", "O", "HN", "H"})

# ── Amber ff19SB atom partial charges (amino19.lib) ────────────────────────────
# Parsed once at import. Includes hydrogens. PDB ``HIS`` maps to library ``HIE``
# (default neutral histidine in ff19SB).

_AMINO19_LIB_PATH = Path(__file__).resolve().parent.parent / "amino19.lib"
_AMINO19_ENTRY_ATOMS_RE = re.compile(r"^!entry\.([A-Za-z0-9]+)\.unit\.atoms table")
_AMINO19_ATOM_LINE_RE = re.compile(
    r'^\s*"([^"]+)"\s*"([^"]+)"\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s*$'
)


def _parse_amino19_lib_atom_charges(lib_path: Path) -> Dict[Tuple[str, str], float]:
    """
    Parse AMBER offlib ``amino19.lib`` ``!entry.<RES>.unit.atoms`` blocks.

    Returns mapping (residue_3letter_upper, atom_name_upper) -> partial charge.
    """
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
    """PDB ``HIS`` -> ff19SB default ``HIE`` charges (same atom names)."""
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
    """
    Resolve PDB ``HIS`` to ff19SB histidine microstate name.

    The ``amino19.lib`` table stores separate entries for ``HID`` (HD1),
    ``HIE`` (HE2), and ``HIP`` (both HD1 and HE2 protonated).  We infer
    microstate from atom presence within the residue.
    """
    if not residue_atom_names:
        return "HIE"  # historical default used by this project
    names = {n.strip().upper() for n in residue_atom_names}
    has_hd1 = "HD1" in names
    has_he2 = "HE2" in names
    if has_hd1 and has_he2:
        return "HIP"
    if has_hd1:
        return "HID"
    return "HIE"


def get_ff19sb_atom_charge(
    residue_name: str,
    atom_name: str,
    *,
    residue_atom_names: Optional[Set[str]] = None,
) -> float:
    """
    Amber ff19SB (``amino19.lib``) partial charge for an atom.

    Uses exact (residue, atom) lookup from ``FF19SB_ATOM_CHARGES`` with
    compatibility fallbacks for common terminal/PDB atom names:
    - ``HIS`` is resolved to ``HID/HIE/HIP`` when ``residue_atom_names`` is
      supplied.
    - ``OXT`` falls back to ``O``.
    - ``H1/H2/H3`` fall back to ``H``.

    Unknown residue or atom names return ``0.0``.
    """
    q, _source = get_ff19sb_atom_charge_with_source(
        residue_name,
        atom_name,
        residue_atom_names=residue_atom_names,
    )
    return q


def get_ff19sb_atom_charge_with_source(
    residue_name: str,
    atom_name: str,
    *,
    residue_atom_names: Optional[Set[str]] = None,
) -> Tuple[float, str]:
    """
    Charge lookup with provenance for debugging.

    Returns ``(charge, source)`` where source is one of:
    ``"direct"``, ``"fallback_oxt_to_o"``, ``"fallback_h123_to_h"``, ``"missing"``.
    """
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


def get_ff19sb_heavy_atom_charge(residue_name: str, atom_name: str) -> float:
    """Backward-compatible alias; now returns ff19SB charge for any atom."""
    return get_ff19sb_atom_charge(residue_name, atom_name)


_FF19SB_RESIDUE_REGION_CHARGE_CACHE: Dict[str, Tuple[float, float]] = {}


def get_ff19sb_residue_region_charges(
    residue_name: str,
    *,
    residue_atom_names: Optional[Set[str]] = None,
) -> Tuple[float, float]:
    """
    Sum ff19SB partial charges for one residue template, split into backbone and side chain.

    Returns ``(backbone_charge, sidechain_charge)`` using the same histidine-resolution
    logic as atom lookups. This is useful for fast residue-level approximations of
    atom-charge-based descriptors when only residue SASA partitions are available.
    """
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
        if atom_name in _BACKBONE_ATOMS:
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
    """
    Sample (residue, atom, charge) rows for sanity-checking the library parse.

    Default set: acidic / basic / neutral examples.
    """
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
        q = get_ff19sb_atom_charge(res, atom)
        rows.append((res, atom, q))
    return rows


# ── H-bond donor tables ────────────────────────────────────────────────────────

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
}
DONOR_INFO: Dict[Tuple[str, str], str] = {
    key: base for key, (base, _max_hbonds) in DONOR_METADATA.items()
}
DONOR_MAX_HBONDS: Dict[Tuple[str, str], int] = {
    ("N", "ANY"): 1,  # any residue except PRO (filtered by DONOR_EXCLUDED)
    **{key: max_hbonds for key, (_base, max_hbonds) in DONOR_METADATA.items()},
}

# ── H-bond acceptor tables ─────────────────────────────────────────────────────

ACCEPTORS_ANY = frozenset({"O"})

ACCEPTOR_METADATA: Dict[Tuple[str, str], int] = {
    ("O",   "ANY"): 2,
    ("OE1", "GLN"): 1,
    ("OE2", "GLU"): 2,
    ("OD1", "ASN"): 1,
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


def distance(atom1: Atom, atom2: Atom) -> float:
    return math.dist((atom1.x, atom1.y, atom1.z),
                     (atom2.x, atom2.y, atom2.z))


def angle_between_vectors(
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float],
) -> float:
    dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


_BACKBONE_ATOMS = frozenset({
    "N", "CA", "C", "O",
    "H", "HA", "HA2", "HA3",
    "H1", "H2", "H3", "HN",
})


def is_backbone_atom(atom) -> bool:
    name = atom.name if hasattr(atom, "name") else str(atom)
    return name in _BACKBONE_ATOMS


if __name__ == "__main__":
    # Step 2: human-readable parse check (acidic / basic / neutral).
    print("ff19SB atom charges (amino19.lib) — sample rows")
    print(f"{'Residue':<8} {'Atom':<6} {'Charge':>12}")
    print("-" * 28)
    for res, atom, q in ff19sb_charge_verification_rows():
        print(f"{res:<8} {atom:<6} {q:12.6f}")
    n = len(FF19SB_ATOM_CHARGES)
    print("-" * 28)
    print(f"Total (residue, atom) entries: {n}")
