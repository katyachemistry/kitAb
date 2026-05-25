#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assign_sequence_folds import assign_folds as _run_seqsplit  # noqa: E402

DEFAULT_RANDOM_SEED = 42
RANDOM_CV_SEEDS = [42, 43, 44]


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    return yaml


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

    if is_scenario3:
        split_col = "fold" if info["has_fold_col"] else None
        return {
            **base,
            "csv_path": f"{input_csvs}/{csv_file.name}",
            "split_col": split_col,
            "random_seeds": [DEFAULT_RANDOM_SEED] if split_col is not None else list(RANDOM_CV_SEEDS),
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
        "random_seeds": [DEFAULT_RANDOM_SEED] if split_col is not None else list(RANDOM_CV_SEEDS),
    }


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
        "stability_features",
        "split_randomly",
        "automl",
    }
    for k, v in raw.items():
        if k not in known_keys:
            extra_root_keys[k] = v

    calc_desc = raw.get("calculate_descriptors", True)
    is_scenario3 = (raw.get("automl") is False)
    _split_randomly_raw = raw.get("split_randomly")
    if not _split_randomly_raw:
        split_randomly: set[str] = set()
    elif isinstance(_split_randomly_raw, list):
        split_randomly = {
            Path(str(s).strip()).stem for s in _split_randomly_raw if str(s).strip()
        }
    else:
        split_randomly = {
            Path(s.strip()).stem
            for s in str(_split_randomly_raw).split(",")
            if s.strip()
        }

    stability_features_raw = raw.get("stability_features")
    stability_suffixes = []
    if stability_features_raw:
        if isinstance(stability_features_raw, list):
            stability_suffixes = [str(s).strip() for s in stability_features_raw]
        else:
            stability_suffixes = [s.strip() for s in str(stability_features_raw).split(",") if s.strip()]

    input_csvs = raw.get("input_csvs_folder")
    result_folder = raw.get("result_folder")
    if not input_csvs:
        raise SystemExit("input_csvs_folder is required")
    if not result_folder:
        raise SystemExit("result_folder is required")

    input_csvs_path = _resolve_path(repo_root, str(input_csvs))
    if not input_csvs_path.is_dir():
        raise SystemExit(f"input_csvs_folder not found: {input_csvs_path}")

    if calc_desc:
        if raw.get("input_structures_folder"):
            raise SystemExit(
                "This draft only supports CSV-only runs (no input_structures_folder). "
                "Use scenario1-style configs for now."
            )

        csv_files = sorted(input_csvs_path.glob("*.csv"))
        if not csv_files:
            raise SystemExit(f"No *.csv files in {input_csvs_path}")

        sp = raw.get("structure_prediction") or {}
        if sp is not None and not isinstance(sp, dict):
            raise SystemExit("structure_prediction must be a mapping")

        model = str(sp.get("model") or "abb3").strip().lower()
        if model not in {"abb2", "abb3"}:
            raise SystemExit(f"structure_prediction.model must be abb2 or abb3 (got {model!r})")

        runs = int(sp.get("runs") or 1)
        if runs < 1:
            raise SystemExit("structure_prediction.runs must be >= 1")

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
            "extra_root_keys": extra_root_keys,
            "dataset_runs": dataset_runs,
            "is_scenario3": is_scenario3,
            "stability_suffixes": stability_suffixes,
            "split_randomly": split_randomly,
            "structure_prediction": {
                "model": model,
                "device": device,
                "runs": runs,
            },
        }
    else:
        pred_desc = raw.get("predefined_descriptors")
        if not pred_desc or not isinstance(pred_desc, dict):
            raise SystemExit("predefined_descriptors is required as a mapping when calculate_descriptors is False")

        folder_name = pred_desc.get("folder")
        allowed_suffixes = pred_desc.get("allowed_suffixes")
        if not folder_name:
            raise SystemExit("predefined_descriptors.folder is required")
        if not allowed_suffixes or not isinstance(allowed_suffixes, list):
            raise SystemExit("predefined_descriptors.allowed_suffixes must be a list of suffixes")

        folder_path = _resolve_path(repo_root, folder_name)
        if not folder_path.is_dir():
            raise SystemExit(f"predefined_descriptors folder not found: {folder_path}")

        splits_dir = repo_root / str(result_folder).strip() / "splits"
        predefined_runs = []
        for sub_dir in sorted(folder_path.iterdir()):
            if not sub_dir.is_dir():
                continue
            matched_suffix = None
            for suffix in allowed_suffixes:
                if sub_dir.name.endswith(suffix):
                    matched_suffix = suffix
                    break

            if matched_suffix:
                stem = sub_dir.name[:-len(matched_suffix)]
                csv_file_path = input_csvs_path / f"{stem}.csv"
                if csv_file_path.is_file():
                    info = _parse_csv_info(csv_file_path)
                    split_info = _resolve_cv_split(
                        csv_file_path,
                        stem,
                        splits_dir,
                        str(input_csvs),
                        repo_root,
                        is_scenario3=is_scenario3,
                        split_randomly=split_randomly,
                        info=info,
                    )
                    predefined_runs.append({
                        "key": sub_dir.name,
                        "stem": stem,
                        "developability_results_path": f"{folder_name}/{sub_dir.name}/results",
                        **split_info,
                    })

        if not predefined_runs:
            raise SystemExit(f"No predefined subfolders matching suffixes {allowed_suffixes} found in {folder_path} with matching CSV files.")

        return {
            "source_config": _rel(repo_root, cfg_path),
            "input_csvs_folder": _rel(repo_root, input_csvs_path),
            "result_name": str(result_folder).strip(),
            "calculate_descriptors": False,
            "extra_root_keys": extra_root_keys,
            "predefined_runs": predefined_runs,
            "predefined_folder_name": Path(folder_name).name,
            "is_scenario3": is_scenario3,
            "stability_suffixes": stability_suffixes,
            "split_randomly": split_randomly,
        }


def build_run_config(generic: dict[str, Any], repo_root: Path) -> dict[str, dict[str, str]]:
    run_config = {}

    for k, v in generic.get("extra_root_keys", {}).items():
        run_config[k] = v

    is_scenario3 = generic.get("is_scenario3", False)
    stability_suffixes = generic.get("stability_suffixes", [])

    def _apply_automl_fields(block: dict, run_info: dict) -> None:
        block["name_col"] = "name"
        block["target_cols"] = run_info["target_cols"]
        if run_info["split_col"]:
            block["split_col"] = run_info["split_col"]
        else:
            block["n_splits"] = 5
            block["random_seeds"] = run_info["random_seeds"]

        if stability_suffixes:
            dev_feats = {}
            for t in run_info["target_cols"].split(","):
                t = t.strip()
                if any(t.endswith(suff) for suff in stability_suffixes):
                    dev_feats[t] = "general,surface,core,sequence_motives"
                else:
                    dev_feats[t] = "surface,general,sequence_motives"
            block["developability_features"] = dev_feats

    if not generic.get("calculate_descriptors", True):
        run_config["batch_result_root"] = (
            f"{generic['result_name']}/automl/{generic['predefined_folder_name']}_predictions"
        )
        for run in generic["predefined_runs"]:
            block: dict[str, Any] = {
                "path": run["csv_path"],
                "developability_results_path": run["developability_results_path"],
            }
            if not is_scenario3:
                _apply_automl_fields(block, run)
            run_config[run["key"]] = block
        return run_config

    model = generic["structure_prediction"]["model"]
    runs = generic["structure_prediction"]["runs"]
    descriptors_root = repo_root / generic["result_name"] / "descriptors"
    run_config["batch_result_root"] = f"{generic['result_name']}/automl/batch"

    for run_info in generic["dataset_runs"]:
        stem = run_info["stem"]
        for run in range(1, runs + 1):
            key = f"{stem}_{model}_{run}"
            block = {
                "path": run_info["csv_path"],
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
