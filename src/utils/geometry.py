from typing import Tuple
import math

from utils.parsers import Atom


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
