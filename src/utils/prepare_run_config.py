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


def _ensure_result_folder_absent(
    repo_root: Path, result_folder: str, *, config_path: Path
) -> None:
    root = (repo_root / str(result_folder).strip()).resolve()
    if root.is_dir():
        raise SystemExit(
            f"result_folder already exists: {root}\n"
            f"(remove or rename it before re-running; config {config_path})"
        )


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


def _parse_dataset_stem_set(raw: Any, *, field_name: str) -> set[str]:
    """Parse comma-separated or YAML list of dataset file names to stem set."""
    if not raw:
        return set()
    if isinstance(raw, list):
        items = [str(s).strip() for s in raw if str(s).strip()]
    else:
        items = [s.strip() for s in str(raw).split(",") if s.strip()]
    return {Path(x).stem for x in items}


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

    If ``structures_path`` contains subdirectories with pipeline structures, each
    subdirectory is one job. Otherwise the root folder itself is a single job
    when it contains top-level PDB/mmCIF files.
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


def _match_descriptor_suffix(folder_name: str, allowed_suffixes: list[str]) -> str | None:
    for suffix in sorted({str(s) for s in allowed_suffixes}, key=len, reverse=True):
        if folder_name.endswith(suffix):
            return suffix
    return None


_DEFAULT_FEATURES_CSV_NAME = "features.csv"
_RESULTS_SUBDIR_NAME = "results"


def _predefined_descriptors_config_block(raw: dict[str, Any]) -> dict[str, Any] | None:
    block = raw.get("predefined_descriptors")
    legacy = raw.get("csv_features")
    if block is not None and legacy is not None:
        raise SystemExit(
            "Use only predefined_descriptors in the generic config "
            "(csv_features is a deprecated alias for the same block)."
        )
    if block is None and legacy is not None:
        print(
            "Note: csv_features is deprecated; use predefined_descriptors for the same options.",
            file=sys.stderr,
        )
        block = legacy
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SystemExit("predefined_descriptors must be a mapping")
    return block


def _has_matching_run_subfolders(
    folder_path: Path,
    *,
    allowed_suffixes: list[str],
) -> bool:
    for sub_dir in folder_path.iterdir():
        if sub_dir.is_dir() and not sub_dir.name.startswith("."):
            if _match_descriptor_suffix(sub_dir.name, allowed_suffixes):
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
    result_folder: str,
    splits_dir: Path,
    repo_root: Path,
    is_scenario3: bool,
    split_randomly: set[str],
    exclude_stems: set[str],
) -> tuple[Path, list[dict[str, Any]]]:
    """Discover external developability runs (any vendor) under one folder.

    Per matching run name ``{dataset_stem}{suffix}``:

    - If any such subfolders exist: each must have ``features.csv`` (or
      ``features_filename``) or ``results/*.json``.
    - Otherwise: flat ``{dataset_stem}{suffix}.csv`` files in the folder root.
    """
    folder_name = pred_desc.get("folder")
    allowed_suffixes = pred_desc.get("allowed_suffixes")
    if not folder_name:
        raise SystemExit("predefined_descriptors.folder is required")
    if not allowed_suffixes or not isinstance(allowed_suffixes, list):
        raise SystemExit("predefined_descriptors.allowed_suffixes must be a list of suffixes")

    features_filename = str(
        pred_desc.get("features_filename") or _DEFAULT_FEATURES_CSV_NAME
    ).strip()
    if not features_filename:
        raise SystemExit("predefined_descriptors.features_filename must be non-empty")

    folder_path = _resolve_path(repo_root, str(folder_name))
    if not folder_path.is_dir():
        raise SystemExit(f"predefined_descriptors folder not found: {folder_path}")

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
    if _has_matching_run_subfolders(folder_path, allowed_suffixes=allowed_suffixes):
        for sub_dir in sorted(folder_path.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            matched_suffix = _match_descriptor_suffix(sub_dir.name, allowed_suffixes)
            if not matched_suffix:
                continue
            dataset_stem = sub_dir.name[: -len(matched_suffix)]
            if not dataset_stem:
                raise SystemExit(
                    f"predefined_descriptors: folder {sub_dir.name!r} matches suffix "
                    f"{matched_suffix!r} but has no dataset stem prefix"
                )
            resolved = _resolve_descriptor_source_in_run_dir(
                sub_dir, features_filename=features_filename
            )
            if resolved is None:
                raise SystemExit(
                    f"predefined_descriptors: {sub_dir} matches suffix {matched_suffix!r} "
                    f"but has neither {features_filename!r} nor {_RESULTS_SUBDIR_NAME}/*.json"
                )
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
            run_name = feat_csv.stem
            matched_suffix = _match_descriptor_suffix(run_name, allowed_suffixes)
            if not matched_suffix:
                continue
            dataset_stem = run_name[: -len(matched_suffix)]
            if not dataset_stem:
                raise SystemExit(
                    f"predefined_descriptors: file {feat_csv.name!r} matches suffix "
                    f"{matched_suffix!r} but has no dataset stem prefix"
                )
            entry = _build_descriptor_run_entry(
                run_key=run_name,
                dataset_stem=dataset_stem,
                developability_path=feat_csv,
                features_csv_path=feat_csv,
                **common_kw,
            )
            if entry is not None:
                runs.append(entry)
        layout = "flat *.csv files named {dataset_stem}{suffix}.csv"

    if not runs:
        raise SystemExit(
            f"No predefined_descriptors runs under {folder_path} (tried {layout}); "
            f"need suffixes {allowed_suffixes!r} and matching dataset CSVs in {input_csvs_path}"
        )

    return folder_path, runs


def load_generic_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    yaml = _require_yaml()
    cfg_path = config_path.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"Config root must be a mapping: {cfg_path}")

    extra_root_keys = {}
    known_keys = {
        "input_csvs_folder",
        "result_folder",
        "structure_prediction",
        "input_structures_folder",
        "calculate_descriptors",
        "predefined_descriptors",
        "csv_features",
        "exclude_datasets",
        "split_randomly",
        "automl",
        "structures_processing",
        "n_cpu",
    }
    for k, v in raw.items():
        if k not in known_keys:
            extra_root_keys[k] = v

    calc_desc = raw.get("calculate_descriptors", True)
    is_scenario3 = (raw.get("automl") is False)
    split_randomly = _parse_dataset_stem_set(
        raw.get("split_randomly"), field_name="split_randomly"
    )
    exclude_stems = _parse_dataset_stem_set(
        raw.get("exclude_datasets"), field_name="exclude_datasets"
    )
    if exclude_stems & split_randomly:
        print(
            "Warning: exclude_datasets and split_randomly overlap on "
            f"{sorted(exclude_stems & split_randomly)!r}; excluded datasets are not processed.",
            file=sys.stderr,
        )

    for _dep_key in ("stability_targets", "stability_features"):
        if _dep_key in raw:
            print(
                f"Warning: {_dep_key} in {cfg_path.name} is ignored "
                "(all developability groups are used for every target).",
                file=sys.stderr,
            )

    input_csvs = raw.get("input_csvs_folder")
    input_structures = raw.get("input_structures_folder")
    result_folder = raw.get("result_folder")
    if not result_folder:
        raise SystemExit("result_folder is required")
    if not input_csvs and not input_structures:
        raise SystemExit(
            "config must set input_csvs_folder and/or input_structures_folder"
        )

    _ensure_result_folder_absent(repo_root, str(result_folder), config_path=cfg_path)

    if not input_csvs:
        if not calc_desc:
            raise SystemExit(
                "structures-only config requires calculate_descriptors "
                "(use predefined_descriptors with input_csvs_folder for AutoML-only runs)"
            )
        if not input_structures:
            raise SystemExit(
                "input_structures_folder is required when input_csvs_folder is omitted"
            )
        structures_path = _resolve_path(repo_root, str(input_structures))
        if not structures_path.is_dir():
            raise SystemExit(f"input_structures_folder not found: {structures_path}")

        structure_runs = [
            {
                "folder_name": folder.name,
                "structure_dir": _rel(repo_root, folder),
            }
            for folder in _discover_structure_only_jobs(structures_path)
        ]
        if raw.get("automl") is True:
            print(
                "Warning: structures-only config cannot run AutoML without CSV datasets; "
                "skipping AutoML.",
                file=sys.stderr,
            )
        return {
            "source_config": _rel(repo_root, cfg_path),
            "result_name": str(result_folder).strip(),
            "calculate_descriptors": True,
            "structures_only": True,
            "uses_existing_structures": True,
            "input_structures_folder": _rel(repo_root, structures_path),
            "extra_root_keys": extra_root_keys,
            "structure_runs": structure_runs,
            "is_scenario3": True,
            "split_randomly": split_randomly,
            "exclude_stems": exclude_stems,
        }

    input_csvs_path = _resolve_path(repo_root, str(input_csvs))
    if not input_csvs_path.is_dir():
        raise SystemExit(f"input_csvs_folder not found: {input_csvs_path}")

    splits_dir = repo_root / str(result_folder).strip() / "splits"
    pred_desc = _predefined_descriptors_config_block(raw)
    if pred_desc is not None:
        _, predefined_runs = _load_predefined_descriptor_runs(
            pred_desc,
            input_csvs_path=input_csvs_path,
            input_csvs=str(input_csvs),
            result_folder=str(result_folder).strip(),
            splits_dir=splits_dir,
            repo_root=repo_root,
            is_scenario3=is_scenario3,
            split_randomly=split_randomly,
            exclude_stems=exclude_stems,
        )
        return {
            "source_config": _rel(repo_root, cfg_path),
            "input_csvs_folder": _rel(repo_root, input_csvs_path),
            "result_name": str(result_folder).strip(),
            "calculate_descriptors": False,
            "uses_predefined_descriptors": True,
            "extra_root_keys": extra_root_keys,
            "predefined_runs": predefined_runs,
            "is_scenario3": is_scenario3,
            "split_randomly": split_randomly,
            "exclude_stems": exclude_stems,
        }

    if calc_desc:
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

        if input_structures:
            structures_path = _resolve_path(repo_root, str(input_structures))
            if not structures_path.is_dir():
                raise SystemExit(f"input_structures_folder not found: {structures_path}")

            splits_dir = repo_root / str(result_folder).strip() / "splits"
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
                    str(input_csvs),
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
                "source_config": _rel(repo_root, cfg_path),
                "input_csvs_folder": _rel(repo_root, input_csvs_path),
                "result_name": str(result_folder).strip(),
                "calculate_descriptors": True,
                "uses_existing_structures": True,
                "input_structures_folder": _rel(repo_root, structures_path),
                "extra_root_keys": extra_root_keys,
                "structure_runs": structure_runs,
                "is_scenario3": is_scenario3,
                "split_randomly": split_randomly,
                "exclude_stems": exclude_stems,
            }

        sp = raw.get("structure_prediction") or {}
        if sp is not None and not isinstance(sp, dict):
            raise SystemExit("structure_prediction must be a mapping")

        model = str(sp.get("model") or "abb3").strip().lower()
        if model not in {"abb2", "abb3"}:
            raise SystemExit(f"structure_prediction.model must be abb2 or abb3 (got {model!r})")

        runs = int(sp.get("runs") or 1)
        if runs < 1:
            raise SystemExit("structure_prediction.runs must be >= 1")

        batch_size = int(sp.get("batch_size") or 4)
        if batch_size < 1:
            raise SystemExit("structure_prediction.batch_size must be >= 1")

        _device_raw = sp.get("device")
        if _device_raw is None:
            device = "cuda:0"
        elif isinstance(_device_raw, int):
            device = f"cuda:{_device_raw}"
        else:
            _device_text = str(_device_raw).strip()
            if not _device_text:
                device = "cuda:0"
            elif _device_text.startswith("cuda:") or _device_text == "cpu":
                device = _device_text
            elif _device_text.isdigit():
                device = f"cuda:{_device_text}"
            else:
                device = _device_text

        splits_dir = repo_root / str(result_folder).strip() / "splits"
        dataset_runs = []
        for csv_file in csv_files:
            info = _parse_csv_info(csv_file)
            split_info = _resolve_cv_split(
                csv_file,
                csv_file.stem,
                splits_dir,
                str(input_csvs),
                repo_root,
                is_scenario3=is_scenario3,
                split_randomly=split_randomly,
                info=info,
            )
            dataset_runs.append({"stem": csv_file.stem, **split_info})

        return {
            "source_config": _rel(repo_root, cfg_path),
            "input_csvs_folder": _rel(repo_root, input_csvs_path),
            "result_name": str(result_folder).strip(),
            "calculate_descriptors": True,
            "uses_existing_structures": False,
            "extra_root_keys": extra_root_keys,
            "dataset_runs": dataset_runs,
            "is_scenario3": is_scenario3,
            "split_randomly": split_randomly,
            "exclude_stems": exclude_stems,
            "structure_prediction": {
                "model": model,
                "device": device,
                "runs": runs,
                "batch_size": batch_size,
            },
        }
    else:
        raise SystemExit(
            "calculate_descriptors is False; set predefined_descriptors for AutoML-only runs "
            "(subfolders with features.csv or results/*.json, or flat {dataset}{suffix}.csv)."
        )


def build_run_config(generic: dict[str, Any], repo_root: Path) -> dict[str, dict[str, str]]:
    run_config = {}

    for k, v in generic.get("extra_root_keys", {}).items():
        run_config[k] = v

    is_scenario3 = generic.get("is_scenario3", False)

    def _apply_automl_fields(block: dict, run_info: dict) -> None:
        block["name_col"] = "name"
        block["target_cols"] = run_info["target_cols"]
        feature_cols = run_info.get("feature_cols")
        if feature_cols:
            block["feature_cols"] = feature_cols
        if run_info["split_col"]:
            block["split_col"] = run_info["split_col"]
        else:
            block["n_splits"] = 5
            block["random_seeds"] = run_info["random_seeds"]

    if not generic.get("calculate_descriptors", True):
        run_config["batch_result_root"] = f"{generic['result_name']}/automl"
        for run in generic["predefined_runs"]:
            block: dict[str, Any] = {
                "path": run["csv_path"],
                "developability_results_path": run["developability_results_path"],
            }
            if not is_scenario3:
                _apply_automl_fields(block, run)
            run_config[run["key"]] = block
        return run_config

    descriptors_root = repo_root / generic["result_name"] / "descriptors"

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

    run_config["batch_result_root"] = f"{generic['result_name']}/automl"

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

    model = generic["structure_prediction"]["model"]
    runs = generic["structure_prediction"]["runs"]
    structures_root = repo_root / generic["result_name"] / "structures"

    for run_info in generic["dataset_runs"]:
        stem = run_info["stem"]
        for run in range(1, runs + 1):
            key = f"{stem}_{model}_{run}"
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
    run_config: dict[str, dict[str, str]],
    repo_root: Path,
    result_name: str,
    source_config: str,
) -> Path:
    yaml = _require_yaml()
    out_dir = repo_root / "run_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result_name}.yaml"
    out_path.write_text(
        f"# Generated by fastab.sh from {source_config}\n"
        + yaml.safe_dump(run_config, sort_keys=False, default_flow_style=False)
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="expand a generic FASTAb config into a run-specific config"
    )
    parser.add_argument("config", type=Path, help="Generic config (e.g. configs/scenario1.yaml)")
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    generic = load_generic_config(args.config, repo_root)
    run_config = build_run_config(generic, repo_root)
    out_path = write_run_config(
        run_config,
        repo_root,
        generic["result_name"],
        generic["source_config"],
    )
    print(out_path)


if __name__ == "__main__":
    main()
