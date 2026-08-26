"""Fold-directory path helpers shared by OOF and nested CV.

Kept free of other ``analysis.*`` imports so those modules cannot circular-import
each other through this file.
"""

from __future__ import annotations

import json
from pathlib import Path


def resolve_fold_dir(path_raw: str | Path) -> Path:
    """Resolve fold paths after the FASTAb -> kitAb repository migration."""
    path = Path(path_raw)
    candidates = [path]
    text = str(path)
    if "/FASTAb/" in text:
        candidates.append(Path(text.replace("/FASTAb/", "/kitAb/")))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[-1]


def remap_fold_dir(path: Path, mapping: dict[str, str] | None) -> Path:
    """Rewrite ``path`` using longest matching prefix in ``mapping``."""
    if not mapping:
        return Path(path)
    text = str(path)
    for src in sorted(mapping, key=len, reverse=True):
        if not src:
            continue
        if text == src or text.startswith(src.rstrip("/") + "/"):
            return Path(mapping[src] + text[len(src) :])
    return Path(path)


def load_fold_dir_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"fold-dir map must be a JSON object: {path}")
    return {str(k): str(v) for k, v in raw.items()}
