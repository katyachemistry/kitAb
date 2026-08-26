#!/usr/bin/env python3
"""Rebuild isolated CV fold parquets for external nested CV and pooled OOF.

ProperMAb, sequence-baseline, and TAP AutoML JSONs still point at FASTAb fold
dirs (or at shared kitAb dirs whose feature set was overwritten). Nested CV
and ProperMAb ABB2 pooled OOF need local parquets with the matching feature
universe, so this writes feature-aware run dirs and a prefix map for
``--fold-dir-map``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.nested_cv import (
    RANDOM_SPLIT_STEMS,
    all_backend_pairs_mode,
    backend_yaml_key_mode,
    nested_yaml_key,
    resolve_nested_pairs,
)

SEQUENCE_INCLUDE_FEATURES = [
    "cdr_h3_length",
    "aromatic_cdr",
    "theoretical_pi",
    "n_charged_res_fv",
    "fv_charge",
    "fv_csp",
]

METHOD_BACKEND = {
    "propermab_abb2": "abb2",
    "propermab_abb3": "abb3",
    "propermab_flashabb": "flashabb",
    "propermab_sequence_baseline": "abb2",
    "tap": "abb2",
}

_RS_SUFFIX = re.compile(r"__rs\d+$")


def yaml_key_mode_for(method: str, *, structure_variant: int = 1) -> str:
    if method == "tap":
        return "stem"
    return backend_yaml_key_mode(structure_variant)


def yaml_key_suffix_for(method: str) -> str:
    return "" if method == "tap" else "_propermab"


def descriptor_dir(repo_root: Path, method: str, stem: str, yaml_key: str) -> Path:
    if method == "tap":
        return repo_root / "descriptors_tap" / stem
    backend = METHOD_BACKEND[method]
    block = _RS_SUFFIX.sub("", yaml_key)
    return repo_root / f"descriptors_propermab_{backend}" / block


def isolated_run_dir(repo_root: Path, method: str, stem: str, yaml_key: str) -> Path:
    return (
        repo_root
        / "runs"
        / f"{stem}_cv_prepare__nested_{method}__{yaml_key}"
    )


TAP_RUN_MARKER = "_cv_prepare__descriptors_tap_"


def original_tap_run_dir(repo_root: Path, stem: str, yaml_key: str) -> Path:
    return repo_root / "runs" / f"{stem}{TAP_RUN_MARKER}{yaml_key}"


def iter_existing_tap_run_dirs(repo_root: Path) -> list[tuple[str, str, Path]]:
    """Yield ``(stem, yaml_key, run_dir)`` for leftover TAP prepare dirs."""
    runs = Path(repo_root) / "runs"
    if not runs.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for path in sorted(runs.iterdir()):
        if not path.is_dir() or TAP_RUN_MARKER not in path.name:
            continue
        stem, yaml_key = path.name.split(TAP_RUN_MARKER, 1)
        if stem and yaml_key:
            out.append((stem, yaml_key, path))
    return out


def targets_from_meta(run_dir: Path) -> list[str]:
    names = []
    for dest in sorted(path for path in Path(run_dir).iterdir() if path.is_dir()):
        if (dest / "meta.json").is_file():
            names.append(dest.name)
    return names


def _load_meta_feature_cols(run_dir: Path, targets: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for target in targets:
        meta_path = Path(run_dir) / target / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cols = meta.get("feature_cols") or []
        out[target] = {str(c) for c in cols}
    return out


def _assert_feature_cols_unchanged(
    run_dir: Path,
    targets: list[str],
    previous: dict[str, set[str]],
) -> None:
    current = _load_meta_feature_cols(run_dir, targets)
    for target, old_cols in previous.items():
        new_cols = current.get(target)
        if new_cols is None:
            raise RuntimeError(f"{run_dir}/{target}: meta.json missing after restore")
        if new_cols != old_cols:
            raise RuntimeError(
                f"{run_dir}/{target}: restored feature_cols {sorted(new_cols)} "
                f"!= original {sorted(old_cols)}"
            )


def original_run_dir_from_jsons(json_dir: Path) -> Path | None:
    if not json_dir.is_dir():
        return None
    for jp in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        fold_dir = str(data.get("fold_dir") or "").strip()
        if fold_dir:
            return Path(fold_dir).parent
    return None


def fold_map_aliases(original: Path, new_run: Path) -> dict[str, str]:
    """Map FASTAb and kitAb prefixes of ``original`` onto the rebuilt run dir."""
    dest = str(Path(new_run).resolve())
    text = str(original)
    out = {text: dest}
    if "/FASTAb/" in text:
        out[text.replace("/FASTAb/", "/kitAb/")] = dest
    elif "/kitAb/" in text:
        out[text.replace("/kitAb/", "/FASTAb/")] = dest
    return out


def iter_automl_yaml_jobs(automl_root: Path) -> list[tuple[str, str, list[str]]]:
    """Yield ``(stem, yaml_key, targets)`` for every AutoML yaml_key directory."""
    root = Path(automl_root)
    if not root.is_dir():
        return []
    out: list[tuple[str, str, list[str]]] = []
    for sub in sorted(path for path in root.iterdir() if path.is_dir()):
        stem = None
        targets: set[str] = set()
        for jp in sub.glob("*.json"):
            if jp.name.endswith(".oof.json"):
                continue
            try:
                data = json.loads(jp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if stem is None:
                stem = str(data.get("dataset_stem") or "").strip() or None
            target = str(data.get("target_col") or "").strip()
            if target:
                targets.add(target)
        if stem and targets:
            out.append((stem, sub.name, sorted(targets)))
    return out


def folds_ready(run_dir: Path, targets: list[str]) -> bool:
    for target in targets:
        dest = run_dir / target
        if not (dest / "meta.json").is_file():
            return False
        if not any(dest.glob("fold_*_train.parquet")):
            return False
    return True


def _split_csv_arg(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def prepare_one(
    *,
    repo_root: Path,
    method: str,
    stem: str,
    yaml_key: str,
    targets: list[str],
    resume: bool,
    run_dir: Path | None = None,
) -> Path:
    dataset_csv = repo_root / "datasets" / f"{stem}.csv"
    if not dataset_csv.is_file():
        raise FileNotFoundError(f"Missing dataset CSV: {dataset_csv}")
    desc = descriptor_dir(repo_root, method, stem, yaml_key)
    if not desc.exists():
        raise FileNotFoundError(f"Missing descriptors for {method} {yaml_key}: {desc}")
    dest = (
        Path(run_dir)
        if run_dir is not None
        else isolated_run_dir(repo_root, method, stem, yaml_key)
    )
    if resume and folds_ready(dest, targets):
        print(f"[prepare-nested-folds] skip {method} {yaml_key} (resume)", file=sys.stderr)
        return dest

    previous_features = _load_meta_feature_cols(dest, targets)
    rs_match = re.search(r"__rs(\d+)$", yaml_key)
    random_state = int(rs_match.group(1)) if rs_match else 42
    split_col = None if stem in RANDOM_SPLIT_STEMS else "fold"
    include_features = (
        SEQUENCE_INCLUDE_FEATURES if method == "propermab_sequence_baseline" else None
    )

    env = os.environ.copy()
    pythonpath = str(_SRC_DIR)
    existing = str(env.get("PYTHONPATH") or "").strip()
    if existing:
        pythonpath = pythonpath + os.pathsep + existing
    env["PYTHONPATH"] = pythonpath
    cmd = [
        sys.executable,
        "-P",
        str(_SRC_DIR / "automl" / "prepare_run.py"),
        str(dataset_csv),
        "--name-col",
        "name",
        "--target-cols",
        *targets,
        "--developability-results",
        str(desc),
        "--output-dir",
        str(dest),
        "--n-splits",
        "5",
        "--random-state",
        str(random_state),
        "--jobs-file",
        str(dest / "parallel_jobs.txt"),
    ]
    if include_features:
        cmd.extend(["--include-features", *include_features])
    if split_col:
        cmd.extend(["--split-col", split_col])
        # Replay published CSV fold labels. Garbinski 33/21/18/14 already
        # exceeds the 2:1 assignment ratio used when creating new splits.
        cmd.append("--allow-unbalanced-splits")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[prepare-nested-folds] {method} {yaml_key} -> {dest}", file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)
    if not folds_ready(dest, targets):
        raise RuntimeError(f"Fold reconstruction failed for {method} {yaml_key}: {dest}")
    if previous_features:
        _assert_feature_cols_unchanged(dest, targets, previous_features)
    return dest


def restore_existing_tap_runs(*, repo_root: Path, resume: bool) -> int:
    rows = iter_existing_tap_run_dirs(repo_root)
    if not rows:
        print("[prepare-nested-folds] no existing TAP run dirs to restore", file=sys.stderr)
        return 1
    restored = 0
    skipped = 0
    for stem, yaml_key, run_dir in rows:
        targets = targets_from_meta(run_dir)
        if not targets:
            print(
                f"[prepare-nested-folds] skip TAP {run_dir.name}: no meta.json",
                file=sys.stderr,
            )
            continue
        before_ready = folds_ready(run_dir, targets)
        prepare_one(
            repo_root=repo_root,
            method="tap",
            stem=stem,
            yaml_key=yaml_key,
            targets=targets,
            resume=resume,
            run_dir=run_dir,
        )
        if before_ready and resume:
            skipped += 1
        else:
            restored += 1
    print(
        f"[prepare-nested-folds] TAP restore: {restored} rebuilt, "
        f"{skipped} already had parquets",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--method", choices=sorted(METHOD_BACKEND))
    p.add_argument("--automl-root", type=Path)
    p.add_argument("--map-out", type=Path)
    p.add_argument("--pairs-mode", default="all_backend_1")
    p.add_argument(
        "--structure-variant",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help=(
            "Structure minimization run suffix (_1/_2/_3). When set, overrides "
            "--pairs-mode with all_backend_N and yaml keys with backend_N."
        ),
    )
    p.add_argument("--exclude-stems", default="")
    p.add_argument("--include-stems", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--restore-existing-tap-runs",
        action="store_true",
        help=(
            "Rewrite missing train/test parquets into the original TAP fold dirs "
            "(meta.json kept; used by pooled TAP OOF)."
        ),
    )
    p.add_argument(
        "--all-automl-yaml-keys",
        action="store_true",
        help=(
            "Rebuild isolated folds for every yaml_key directory under "
            "--automl-root (all variants/seeds). Used by pooled OOF, not nested CV."
        ),
    )
    args = p.parse_args()

    if args.restore_existing_tap_runs:
        return restore_existing_tap_runs(
            repo_root=Path(args.repo_root),
            resume=bool(args.resume),
        )
    if args.method is None or args.automl_root is None or args.map_out is None:
        p.error("--method, --automl-root, and --map-out are required unless "
                "--restore-existing-tap-runs is set")

    method = str(args.method)
    backend = METHOD_BACKEND[method]
    suffix = yaml_key_suffix_for(method)
    structure_variant = args.structure_variant
    if structure_variant is None:
        pairs_mode = str(args.pairs_mode)
        yaml_key_mode = yaml_key_mode_for(method)
    else:
        pairs_mode = all_backend_pairs_mode(structure_variant)
        yaml_key_mode = yaml_key_mode_for(method, structure_variant=structure_variant)
    by_key: dict[str, tuple[str, list[str]]] = {}
    if args.all_automl_yaml_keys:
        excluded = set(_split_csv_arg(args.exclude_stems))
        included = set(_split_csv_arg(args.include_stems))
        for stem, yaml_key, targets in iter_automl_yaml_jobs(Path(args.automl_root)):
            if stem in excluded:
                continue
            if included and stem not in included:
                continue
            by_key[yaml_key] = (stem, list(targets))
    else:
        pairs = resolve_nested_pairs(
            Path(args.automl_root),
            backend=backend,
            pairs_mode=pairs_mode,
            exclude_stems=_split_csv_arg(args.exclude_stems),
            include_stems=_split_csv_arg(args.include_stems),
            yaml_key_suffix=suffix,
            yaml_key_mode=yaml_key_mode,
        )
        grouped: dict[str, list[str]] = defaultdict(list)
        for stem, target in pairs:
            yaml_key = nested_yaml_key(stem, backend, suffix=suffix, mode=yaml_key_mode)
            grouped[yaml_key].append(target)
            by_key[yaml_key] = (stem, grouped[yaml_key])
    if not by_key:
        print(f"[prepare-nested-folds] no pairs for {method}", file=sys.stderr)
        Path(args.map_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.map_out).write_text("{}\n")
        return 1

    mapping: dict[str, str] = {}
    for yaml_key, (stem, targets) in sorted(by_key.items()):
        json_dir = Path(args.automl_root) / yaml_key
        original = original_run_dir_from_jsons(json_dir)
        new_run = prepare_one(
            repo_root=Path(args.repo_root),
            method=method,
            stem=stem,
            yaml_key=yaml_key,
            targets=sorted(set(targets)),
            resume=bool(args.resume),
        )
        if original is not None:
            mapping.update(fold_map_aliases(original, new_run))

    dest = Path(args.map_out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    print(f"[prepare-nested-folds] wrote {dest} ({len(mapping)} run dir(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
