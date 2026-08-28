#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl.pipeline_defaults import (  # noqa: E402
    DEFAULT_RANDOM_STATE as DEFAULT_RANDOM_SEED,
    RANDOM_CV_SEEDS,
)
from utils.assign_sequence_folds import assign_folds as _run_seqsplit  # noqa: E402


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    return yaml


_STRUCTURE_BACKENDS = ("abb2", "abb3", "flashabb")


def _parse_structure_models(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return ["abb2"]
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.replace(",", " ").split() if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(p).strip().lower() for p in raw if str(p).strip()]
    else:
        raise SystemExit(
            "structure_prediction.model must be a backend name or a list "
            "(abb2, abb3, flashabb)"
        )
    if not parts:
        raise SystemExit(
            "structure_prediction.model must list at least one of abb2, abb3, flashabb"
        )
    seen: list[str] = []
    bad: list[str] = []
    for part in parts:
        if part not in _STRUCTURE_BACKENDS:
            bad.append(part)
        elif part not in seen:
            seen.append(part)
    if bad:
        raise SystemExit(
            "structure_prediction.model must be one or more of abb2, abb3, flashabb "
            f"(got {', '.join(repr(b) for b in bad)})"
        )
    return seen


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _filter_csv_files(
    csv_files: list[Path],
    exclude_stems: set[str],
) -> list[Path]:
    if not exclude_stems:
        return csv_files
    kept = [p for p in csv_files if p.stem not in exclude_stems]
    excluded = sorted(p.name for p in csv_files if p.stem in exclude_stems)
    if excluded:
        print(
            f"Ignoring excluded dataset(s): {', '.join(excluded)}",
            file=sys.stderr,
        )
    return kept


def _experimental_feature_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if col.startswith("feature_")]


def _allow_nan_columns_for_csv(info: dict[str, Any]) -> list[str]:
    cols = list(info["target_cols"])
    feature_cols = info.get("feature_cols")
    if feature_cols:
        cols.extend(c.strip() for c in str(feature_cols).split(",") if c.strip())
    if info.get("has_fold_col"):
        cols.append("fold")
    return cols


def _validate_input_csv(csv_path: Path) -> None:
    import pandas as pd

    from automl.dataset_validation import validate_experimental_dataset

    info = _parse_csv_info(csv_path)
    df = pd.read_csv(csv_path)
    validate_experimental_dataset(
        df,
        allow_nan_in_columns=_allow_nan_columns_for_csv(info),
        context=f"Input CSV {csv_path}",
    )


def _validate_input_csvs(csv_files: Iterable[Path]) -> None:
    seen: set[str] = set()
    for csv_path in csv_files:
        key = str(csv_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        _validate_input_csv(csv_path)


def _parse_csv_info(csv_file_path: Path) -> dict[str, Any]:
    try:
        header = csv_file_path.read_text().splitlines()[0]
        columns = [col.strip() for col in header.split(",") if col.strip()]
    except Exception as e:
        raise SystemExit(f"Failed to read CSV header from {csv_file_path}: {e}")

    target_cols = [col for col in columns if col.startswith("target_")]
    if not target_cols:
        raise SystemExit(f"No target_cols found in {csv_file_path}")

    has_fold_col = "fold" in columns
    return {
        "target_cols": target_cols,
        "feature_cols": _experimental_feature_columns(columns),
        "has_fold_col": has_fold_col,
    }


def _resolve_cv_split(
    csv_file: Path,
    stem: str,
    splits_dir: Path,
    input_csvs: str,
    repo_root: Path,
    *,
    is_scenario3: bool,
    split_randomly: set[str],
    info: dict[str, Any],
) -> dict[str, Any]:
    """Decide split_col / random_seeds and which CSV path to use for AutoML."""
    base: dict[str, Any] = {"target_cols": ",".join(info["target_cols"])}
    if info.get("feature_cols"):
        base["feature_cols"] = ",".join(info["feature_cols"])

    if is_scenario3:
        split_col = "fold" if info["has_fold_col"] else None
        return {
            **base,
            "csv_path": f"{input_csvs}/{csv_file.name}",
            "split_col": split_col,
            "random_seeds": [DEFAULT_RANDOM_SEED]
            if split_col is not None
            else list(RANDOM_CV_SEEDS),
        }

    if stem in split_randomly:
        return {
            **base,
            "csv_path": f"{input_csvs}/{csv_file.name}",
            "split_col": None,
            "random_seeds": list(RANDOM_CV_SEEDS),
        }

    if info["has_fold_col"]:
        return {
            **base,
            "csv_path": f"{input_csvs}/{csv_file.name}",
            "split_col": "fold",
            "random_seeds": [DEFAULT_RANDOM_SEED],
        }

    fold_result = _run_seqsplit(csv_file, splits_dir)
    csv_rel = (
        _rel(repo_root, fold_result["csv_path"])
        if fold_result["success"]
        else f"{input_csvs}/{csv_file.name}"
    )
    split_col = fold_result["split_col"]
    return {
        **base,
        "csv_path": csv_rel,
        "split_col": split_col,
        "random_seeds": [DEFAULT_RANDOM_SEED]
        if split_col is not None
        else list(RANDOM_CV_SEEDS),
    }


def _folder_matches_dataset_stem(folder_name: str, stem: str) -> bool:
    return folder_name == stem or folder_name.startswith(f"{stem}_")


def _dataset_stem_for_name(name: str, dataset_stems: set[str]) -> str | None:
    """Map a folder or file stem to a dataset CSV stem ({name} or {stem}_...)."""
    matches = [s for s in dataset_stems if _folder_matches_dataset_stem(name, s)]
    if not matches:
        return None
    return max(matches, key=len)


def _discover_structure_folders(structures_path: Path, stem: str) -> list[Path]:
    matched = [
        sub_dir
        for sub_dir in sorted(structures_path.iterdir())
        if sub_dir.is_dir()
        and not sub_dir.name.startswith(".")
        and _folder_matches_dataset_stem(sub_dir.name, stem)
    ]
    return matched


def _is_pipeline_structure_stem(stem: str) -> bool:
    if stem.endswith(("_H", "_L")):
        return False
    if stem.endswith("_H_chain") or stem.endswith("_L_chain"):
        return False
    if stem.endswith("_full_atom_sasa"):
        return False
    return True


def _folder_has_pipeline_structures(folder: Path) -> bool:
    for pattern in ("*.pdb", "*.cif", "*.mmcif"):
        for path in folder.glob(pattern):
            if _is_pipeline_structure_stem(path.stem):
                return True
    return False


def _discover_structure_only_jobs(structures_path: Path) -> list[Path]:
    """Discover structure folders for descriptors-only runs (no CSV pairing).

    Use subdirectories that contain PDB/mmCIF when any exist; otherwise the
    root folder itself.
    """
    subdirs = [
        sub_dir
        for sub_dir in sorted(structures_path.iterdir())
        if sub_dir.is_dir()
        and not sub_dir.name.startswith(".")
        and _folder_has_pipeline_structures(sub_dir)
    ]
    if subdirs:
        return subdirs
    if _folder_has_pipeline_structures(structures_path):
        return [structures_path]
    raise SystemExit(f"No PDB/mmCIF structures found under {structures_path}")


def _read_csv_names(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "name" not in fieldnames:
            raise SystemExit(f'CSV {csv_path} has no "name" column')
        names = [row["name"].strip() for row in reader if row.get("name", "").strip()]
    if not names:
        raise SystemExit(f'CSV {csv_path} has no rows in the "name" column')
    return names


def _structure_file_for_name(folder: Path, name: str) -> Path | None:
    for ext in (".pdb", ".cif", ".mmcif"):
        candidate = folder / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _validate_structure_folder(folder: Path, names: list[str], csv_file: Path) -> None:
    missing = [name for name in names if _structure_file_for_name(folder, name) is None]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise SystemExit(
            f"Missing structures for {csv_file.name} in {folder.name}: {preview}{suffix}"
        )


_DEFAULT_FEATURES_CSV_NAME = "features.csv"
_RESULTS_SUBDIR_NAME = "results"


def _ensure_output_dir_absent(output_dir: Path, *, config_path: Path) -> None:
    if output_dir.is_dir():
        raise SystemExit(
            f"run.output_dir already exists: {output_dir}\n"
            f"(remove or rename it before re-running; config {config_path})"
        )


def _has_matching_run_subfolders(
    folder_path: Path,
    *,
    dataset_stems: set[str],
    features_filename: str,
) -> bool:
    for sub_dir in folder_path.iterdir():
        if not sub_dir.is_dir() or sub_dir.name.startswith("."):
            continue
        if _dataset_stem_for_name(sub_dir.name, dataset_stems) is None:
            continue
        if _resolve_descriptor_source_in_run_dir(
            sub_dir, features_filename=features_filename
        ):
            return True
    return False


def _resolve_descriptor_source_in_run_dir(
    sub_dir: Path,
    *,
    features_filename: str,
) -> tuple[Path, str] | None:
    """Return (developability_results_path, kind) with kind ``csv`` or ``json``."""
    csv_path = sub_dir / features_filename
    results_dir = sub_dir / _RESULTS_SUBDIR_NAME
    if csv_path.is_file():
        return sub_dir.resolve(), "csv"
    if results_dir.is_dir() and any(results_dir.glob("*.json")):
        return results_dir.resolve(), "json"
    return None


def _build_descriptor_run_entry(
    *,
    run_key: str,
    dataset_stem: str,
    developability_path: Path,
    features_csv_path: Path | None,
    input_csvs_path: Path,
    input_csvs: str,
    splits_dir: Path,
    repo_root: Path,
    is_scenario3: bool,
    split_randomly: set[str],
    exclude_stems: set[str],
) -> dict[str, Any] | None:
    if dataset_stem in exclude_stems:
        return None
    dataset_csv = input_csvs_path / f"{dataset_stem}.csv"
    if not dataset_csv.is_file():
        return None
    _validate_input_csv(dataset_csv)
    info = _parse_csv_info(dataset_csv)
    split_info = _resolve_cv_split(
        dataset_csv,
        dataset_stem,
        splits_dir,
        str(input_csvs),
        repo_root,
        is_scenario3=is_scenario3,
        split_randomly=split_randomly,
        info=info,
    )
    entry: dict[str, Any] = {
        "key": run_key,
        "stem": dataset_stem,
        "developability_results_path": _rel(repo_root, developability_path),
        **split_info,
    }
    if features_csv_path is not None:
        entry["features_csv_path"] = _rel(repo_root, features_csv_path)
    return entry


def _load_predefined_descriptor_runs(
    pred_desc: dict[str, Any],
    *,
    input_csvs_path: Path,
    input_csvs: str,
    splits_dir: Path,
    repo_root: Path,
    is_scenario3: bool,
    split_randomly: set[str],
    exclude_stems: set[str],
) -> tuple[Path, list[dict[str, Any]]]:
    """Discover descriptor runs under one folder.

    Uses every subfolder (or flat ``*.csv``) whose name is a dataset stem or
    ``{stem}_...`` (same rule as structure folders). Each subfolder needs
    ``features.csv`` / ``features_filename`` or ``results/*.json``.
    """
    folder_name = pred_desc.get("folder")
    if not folder_name:
        raise SystemExit("predefined_descriptors.folder is required")

    features_filename = str(
        pred_desc.get("features_filename") or _DEFAULT_FEATURES_CSV_NAME
    ).strip()
    if not features_filename:
        raise SystemExit("predefined_descriptors.features_filename must be non-empty")

    folder_path = _resolve_path(repo_root, str(folder_name))
    if not folder_path.is_dir():
        raise SystemExit(f"predefined_descriptors folder not found: {folder_path}")

    dataset_stems = {
        p.stem
        for p in input_csvs_path.glob("*.csv")
        if p.stem not in exclude_stems
    }

    common_kw = dict(
        input_csvs_path=input_csvs_path,
        input_csvs=input_csvs,
        splits_dir=splits_dir,
        repo_root=repo_root,
        is_scenario3=is_scenario3,
        split_randomly=split_randomly,
        exclude_stems=exclude_stems,
    )

    runs: list[dict[str, Any]] = []
    if _has_matching_run_subfolders(
        folder_path,
        dataset_stems=dataset_stems,
        features_filename=features_filename,
    ):
        for sub_dir in sorted(folder_path.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            dataset_stem = _dataset_stem_for_name(sub_dir.name, dataset_stems)
            if not dataset_stem:
                continue
            resolved = _resolve_descriptor_source_in_run_dir(
                sub_dir, features_filename=features_filename
            )
            if resolved is None:
                continue
            developability_path, kind = resolved
            feat_csv = (sub_dir / features_filename) if kind == "csv" else None
            entry = _build_descriptor_run_entry(
                run_key=sub_dir.name,
                dataset_stem=dataset_stem,
                developability_path=developability_path,
                features_csv_path=feat_csv,
                **common_kw,
            )
            if entry is not None:
                runs.append(entry)
        layout = (
            f"subfolders ({features_filename!r} or {_RESULTS_SUBDIR_NAME}/*.json per run)"
        )
    else:
        for feat_csv in sorted(folder_path.glob("*.csv")):
            if not feat_csv.is_file():
                continue
            dataset_stem = _dataset_stem_for_name(feat_csv.stem, dataset_stems)
            if not dataset_stem:
                continue
            entry = _build_descriptor_run_entry(
                run_key=feat_csv.stem,
                dataset_stem=dataset_stem,
                developability_path=feat_csv,
                features_csv_path=feat_csv,
                **common_kw,
            )
            if entry is not None:
                runs.append(entry)
        layout = "flat *.csv files named {dataset_stem}.csv or {dataset_stem}_*.csv"

    if not runs:
        raise SystemExit(
            f"No predefined_descriptors runs under {folder_path} (tried {layout}); "
            f"need folders or files named after dataset CSVs in {input_csvs_path} "
            f"({{stem}} or {{stem}}_...)"
        )

    return folder_path, runs


def prepare_from_manifest(manifest: Any, *, resume: bool = False) -> dict[str, Any]:
    """Expand a kitAb Manifest into the internal run-config plan."""
    repo_root = Path(manifest.repo_root).resolve()
    cfg_path = Path(manifest.source_path).resolve()
    output_dir = Path(manifest.run.output_dir).resolve()
    extra_root_keys: dict[str, Any] = {}

    calc_desc = bool(manifest.descriptors.enabled)
    is_scenario3 = not bool(manifest.automl.enabled)
    split_randomly = {Path(s).stem for s in manifest.inputs.split_randomly}
    exclude_stems = {Path(s).stem for s in manifest.inputs.exclude_datasets}
    include_features = list(manifest.descriptors.include_features)
    if exclude_stems & split_randomly:
        print(
            "Warning: exclude_datasets and split_randomly overlap on "
            f"{sorted(exclude_stems & split_randomly)!r}; excluded datasets are not processed.",
            file=sys.stderr,
        )

    datasets_dir = manifest.inputs.datasets_dir
    structures_dir = manifest.inputs.structures_dir
    predefined_dir = manifest.inputs.predefined_descriptors_dir
    if datasets_dir is None and structures_dir is None:
        raise SystemExit(
            "config must set inputs.datasets_dir and/or inputs.structures_dir"
        )

    if not resume:
        _ensure_output_dir_absent(output_dir, config_path=cfg_path)

    splits_dir = output_dir / "splits"
    source_config = _rel(repo_root, cfg_path)
    output_dir_s = str(output_dir)
    result_name = _rel(repo_root, output_dir)

    if datasets_dir is None:
        if not calc_desc:
            raise SystemExit(
                "structures-only config requires descriptors.enabled "
                "(use inputs.predefined_descriptors_dir with datasets for AutoML-only runs)"
            )
        structures_path = Path(structures_dir).resolve()
        if not structures_path.is_dir():
            raise SystemExit(f"inputs.structures_dir not found: {structures_path}")

        structure_runs = [
            {
                "folder_name": folder.name,
                "structure_dir": _rel(repo_root, folder),
            }
            for folder in _discover_structure_only_jobs(structures_path)
        ]
        if manifest.automl.enabled:
            print(
                "Warning: structures-only config cannot run AutoML without CSV datasets; "
                "skipping AutoML.",
                file=sys.stderr,
            )
        return {
            "source_config": source_config,
            "output_dir": output_dir_s,
            "result_name": result_name,
            "calculate_descriptors": True,
            "structures_only": True,
            "uses_existing_structures": True,
            "input_structures_folder": _rel(repo_root, structures_path),
            "extra_root_keys": extra_root_keys,
            "structure_runs": structure_runs,
            "is_scenario3": True,
            "split_randomly": split_randomly,
            "exclude_stems": exclude_stems,
            "include_features": include_features,
        }

    input_csvs_path = Path(datasets_dir).resolve()
    if not input_csvs_path.is_dir():
        raise SystemExit(f"inputs.datasets_dir not found: {input_csvs_path}")
    input_csvs_rel = _rel(repo_root, input_csvs_path)

    if predefined_dir is not None:
        pred_desc = {"folder": _rel(repo_root, Path(predefined_dir).resolve())}
        _, predefined_runs = _load_predefined_descriptor_runs(
            pred_desc,
            input_csvs_path=input_csvs_path,
            input_csvs=input_csvs_rel,
            splits_dir=splits_dir,
            repo_root=repo_root,
            is_scenario3=is_scenario3,
            split_randomly=split_randomly,
            exclude_stems=exclude_stems,
        )
        return {
            "source_config": source_config,
            "input_csvs_folder": input_csvs_rel,
            "output_dir": output_dir_s,
            "result_name": result_name,
            "calculate_descriptors": False,
            "uses_predefined_descriptors": True,
            "extra_root_keys": extra_root_keys,
            "predefined_runs": predefined_runs,
            "is_scenario3": is_scenario3,
            "split_randomly": split_randomly,
            "exclude_stems": exclude_stems,
            "include_features": include_features,
        }

    if not calc_desc:
        raise SystemExit(
            "descriptors.enabled is false; set inputs.predefined_descriptors_dir "
            "for AutoML-only runs (subfolders with features.csv or results/*.json, "
            "or flat {dataset}{suffix}.csv)."
        )

    csv_files = _filter_csv_files(
        sorted(input_csvs_path.glob("*.csv")),
        exclude_stems,
    )
    if not csv_files:
        raise SystemExit(
            f"No *.csv files to process in {input_csvs_path} "
            f"(all datasets excluded or folder empty)"
        )

    _validate_input_csvs(csv_files)

    if structures_dir is not None:
        structures_path = Path(structures_dir).resolve()
        if not structures_path.is_dir():
            raise SystemExit(f"inputs.structures_dir not found: {structures_path}")

        structure_runs = []
        for csv_file in csv_files:
            stem = csv_file.stem
            matched_folders = _discover_structure_folders(structures_path, stem)
            if not matched_folders:
                raise SystemExit(
                    f"No structure folders for dataset {stem} under {structures_path} "
                    f"(expected a folder named {stem} or {stem}_<suffix>)"
                )

            names = _read_csv_names(csv_file)
            info = _parse_csv_info(csv_file)
            split_info = _resolve_cv_split(
                csv_file,
                stem,
                splits_dir,
                input_csvs_rel,
                repo_root,
                is_scenario3=is_scenario3,
                split_randomly=split_randomly,
                info=info,
            )
            for folder in matched_folders:
                _validate_structure_folder(folder, names, csv_file)
                structure_runs.append(
                    {
                        "stem": stem,
                        "folder_name": folder.name,
                        "structure_dir": _rel(repo_root, folder),
                        **split_info,
                    }
                )

        return {
            "source_config": source_config,
            "input_csvs_folder": input_csvs_rel,
            "output_dir": output_dir_s,
            "result_name": result_name,
            "calculate_descriptors": True,
            "uses_existing_structures": True,
            "input_structures_folder": _rel(repo_root, structures_path),
            "extra_root_keys": extra_root_keys,
            "structure_runs": structure_runs,
            "is_scenario3": is_scenario3,
            "split_randomly": split_randomly,
            "exclude_stems": exclude_stems,
            "include_features": include_features,
        }

    models = list(manifest.structure_prediction.model) or ["abb2"]
    batch_size = manifest.structure_prediction.batch_size
    if batch_size is None:
        batch_size = 50 if models == ["flashabb"] else 4
    if int(batch_size) < 1:
        raise SystemExit("structure_prediction.batch_size must be >= 1")
    device = str(manifest.structure_prediction.device or "cuda:0")

    dataset_runs = []
    for csv_file in csv_files:
        info = _parse_csv_info(csv_file)
        split_info = _resolve_cv_split(
            csv_file,
            csv_file.stem,
            splits_dir,
            input_csvs_rel,
            repo_root,
            is_scenario3=is_scenario3,
            split_randomly=split_randomly,
            info=info,
        )
        dataset_runs.append({"stem": csv_file.stem, **split_info})

    return {
        "source_config": source_config,
        "input_csvs_folder": input_csvs_rel,
        "output_dir": output_dir_s,
        "result_name": result_name,
        "calculate_descriptors": True,
        "uses_existing_structures": False,
        "extra_root_keys": extra_root_keys,
        "dataset_runs": dataset_runs,
        "is_scenario3": is_scenario3,
        "split_randomly": split_randomly,
        "exclude_stems": exclude_stems,
        "include_features": include_features,
        "structure_prediction": {
            "model": models,
            "device": device,
            "batch_size": int(batch_size),
            "runs": int(manifest.structure_prediction.runs),
        },
    }


def load_generic_config(config_path: Path, repo_root: Path, *, resume: bool = False) -> dict[str, Any]:
    from kitab.config import load_manifest

    return prepare_from_manifest(
        load_manifest(config_path, repo_root=repo_root),
        resume=resume,
    )


def build_run_config(generic: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    run_config: dict[str, Any] = {}

    for k, v in generic.get("extra_root_keys", {}).items():
        run_config[k] = v

    is_scenario3 = generic.get("is_scenario3", False)
    output_dir = Path(generic["output_dir"])

    def _apply_automl_fields(block: dict, run_info: dict) -> None:
        block["name_col"] = "name"
        block["target_cols"] = run_info["target_cols"]
        feature_cols = run_info.get("feature_cols")
        if feature_cols:
            block["feature_cols"] = feature_cols
        include_features = generic.get("include_features") or []
        if include_features:
            block["include_features"] = list(include_features)
        if run_info["split_col"]:
            block["split_col"] = run_info["split_col"]
        else:
            block["n_splits"] = 5
            block["random_seeds"] = run_info["random_seeds"]

    def _automl_root() -> str:
        return _rel(repo_root, output_dir / "automl")

    if not generic.get("calculate_descriptors", True):
        run_config["batch_result_root"] = _automl_root()
        for run in generic["predefined_runs"]:
            block: dict[str, Any] = {
                "path": run["csv_path"],
                "developability_results_path": run["developability_results_path"],
            }
            if not is_scenario3:
                _apply_automl_fields(block, run)
            run_config[run["key"]] = block
        return run_config

    descriptors_root = output_dir / "descriptors"

    if generic.get("structures_only"):
        for run_info in generic["structure_runs"]:
            key = run_info["folder_name"]
            run_config[key] = {
                "structure_dir": run_info["structure_dir"],
                "developability_results_path": _rel(
                    repo_root, descriptors_root / key / "results"
                ),
            }
        return run_config

    run_config["batch_result_root"] = _automl_root()

    if generic.get("uses_existing_structures"):
        for run_info in generic["structure_runs"]:
            key = run_info["folder_name"]
            block = {
                "path": run_info["csv_path"],
                "structure_dir": run_info["structure_dir"],
                "developability_results_path": _rel(
                    repo_root, descriptors_root / key / "results"
                ),
            }
            if not is_scenario3:
                _apply_automl_fields(block, run_info)
            run_config[key] = block
        return run_config

    models = _parse_structure_models(generic["structure_prediction"]["model"])
    structures_root = output_dir / "structures"
    n_runs = int(generic["structure_prediction"].get("runs") or 1)
    if n_runs < 1:
        raise SystemExit("structure_prediction.runs must be >= 1")

    for run_info in generic["dataset_runs"]:
        stem = run_info["stem"]
        for model in models:
            for run_i in range(1, n_runs + 1):
                key = f"{stem}_{model}_{run_i}"
                block = {
                    "path": run_info["csv_path"],
                    "structure_dir": _rel(repo_root, structures_root / key),
                    "developability_results_path": _rel(
                        repo_root, descriptors_root / key / "results"
                    ),
                }
                if not is_scenario3:
                    _apply_automl_fields(block, run_info)
                run_config[key] = block
    return run_config


def write_run_config(
    run_config: dict[str, Any],
    out_path: Path,
    source_config: str,
) -> Path:
    yaml = _require_yaml()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Generated by kitab.sh from {source_config}\n"
        + yaml.safe_dump(run_config, sort_keys=False, default_flow_style=False)
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="expand a kitAb manifest into a run-specific config"
    )
    parser.add_argument("config", type=Path, help="kitAb YAML (e.g. configs/scenario1.yaml)")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip the output_dir-already-exists guard (for resuming interrupted runs)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    generic = load_generic_config(args.config, repo_root, resume=args.resume)
    run_config = build_run_config(generic, repo_root)
    slug = Path(generic["output_dir"]).name
    out_path = write_run_config(
        run_config,
        repo_root / "run_configs" / f"{slug}.yaml",
        generic["source_config"],
    )
    print(out_path)


if __name__ == "__main__":
    main()
