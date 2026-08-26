"""Translate legacy scenario YAML configs into the canonical kitAb manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def translate_legacy_config(
    raw: dict[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Return (canonical dict, warnings)."""
    warnings: list[str] = []
    example = _example_snippet(raw)
    warnings.append(
        "Legacy scenario YAML detected. Prefer the new kitAb manifest format. "
        f"Example conversion for {source_path.name}:\n{example}"
    )

    inputs: dict[str, Any] = {}
    run: dict[str, Any] = {}
    descriptors: dict[str, Any] = {}
    automl: dict[str, Any] = {}
    structure_prediction: dict[str, Any] = {}
    structure_processing: dict[str, Any] = {}

    if raw.get("input_csvs_folder"):
        inputs["datasets_dir"] = raw["input_csvs_folder"]
    if raw.get("input_structures_folder"):
        inputs["structures_dir"] = raw["input_structures_folder"]

    pred_desc = raw.get("predefined_descriptors") or raw.get("csv_features")
    if raw.get("csv_features") and not raw.get("predefined_descriptors"):
        warnings.append(
            "csv_features is deprecated; use predefined_descriptors / "
            "inputs.predefined_descriptors_dir."
        )
    if isinstance(pred_desc, dict):
        inputs["predefined_descriptors_dir"] = pred_desc.get("folder")
        if pred_desc.get("allowed_suffixes"):
            warnings.append(
                "predefined_descriptors.allowed_suffixes is no longer used. "
                "Every folder under the directory named {dataset} or {dataset}_... is used."
            )

    if raw.get("exclude_datasets"):
        inputs["exclude_datasets"] = raw["exclude_datasets"]
    if raw.get("split_randomly"):
        inputs["split_randomly"] = raw["split_randomly"]

    if not raw.get("result_folder"):
        raise ValueError("Legacy config requires result_folder")
    run["output_dir"] = raw["result_folder"]
    if "n_cpu" in raw and raw["n_cpu"] is not None:
        run["n_cpu"] = raw["n_cpu"]
    if raw.get("skip_existing_results"):
        run["skip_existing_results"] = True

    if pred_desc:
        structure_prediction["enabled"] = False
        descriptors["enabled"] = False
        automl["enabled"] = True
    elif raw.get("input_structures_folder") and not raw.get("input_csvs_folder"):
        structure_prediction["enabled"] = False
        descriptors["enabled"] = True
        automl["enabled"] = False
    elif raw.get("input_structures_folder"):
        structure_prediction["enabled"] = False
        descriptors["enabled"] = True
        automl["enabled"] = raw.get("automl", True) is not False
    else:
        structure_prediction["enabled"] = True
        descriptors["enabled"] = True
        automl["enabled"] = raw.get("automl", True) is not False

    if raw.get("automl") is False:
        automl["enabled"] = False

    if isinstance(raw.get("structure_prediction"), dict):
        sp = dict(raw["structure_prediction"])
        if "runs" in sp:
            warnings.append(
                "structure_prediction.runs is no longer used; each sequence is predicted once. "
                "Use structures_processing.minimize_attempts for minimization retries."
            )
            sp.pop("runs", None)
        structure_prediction = {
            **sp,
            "enabled": structure_prediction.get("enabled", True),
        }
    if isinstance(raw.get("structures_processing"), dict):
        structure_processing = dict(raw["structures_processing"])
        structure_processing.setdefault("minimize_attempts", 5)
        structure_processing["enabled"] = bool(
            structure_processing.get("renumber_imgt")
            or structure_processing.get("minimize")
        )

    if raw.get("include_features"):
        descriptors["include_features"] = raw["include_features"]
    if raw.get("cleanup") is not None:
        descriptors["cleanup"] = raw["cleanup"]
    if raw.get("descriptor_batch_size") is not None:
        descriptors["batch_size"] = raw["descriptor_batch_size"]
    if raw.get("structures_layout"):
        warnings.append(
            "structures_layout is no longer used; structure folders are discovered automatically."
        )

    if raw.get("automl_config"):
        warnings.append(
            "automl_config is no longer used; AutoML defaults live in src/automl.yaml. "
            "Pass --techniques, --cv-mode, and --no-final-model on the CLI."
        )

    if raw.get("hyperparameter_tuning") or raw.get("tuning"):
        warnings.append(
            "hyperparameter_tuning / tuning are no longer used; kitAb fits the "
            "winning technique on all data without a hyperparameter search."
        )

    out: dict[str, Any] = {
        "inputs": inputs,
        "run": run,
        "structure_prediction": structure_prediction,
        "structure_processing": structure_processing,
        "descriptors": descriptors,
        "automl": automl,
        "_legacy_raw": True,
    }
    return out, warnings


def _example_snippet(raw: dict[str, Any]) -> str:
    lines = [
        "inputs:",
    ]
    if raw.get("input_csvs_folder"):
        lines.append(f"  datasets_dir: {raw['input_csvs_folder']}")
    if raw.get("input_structures_folder"):
        lines.append(f"  structures_dir: {raw['input_structures_folder']}")
    pred = raw.get("predefined_descriptors") or {}
    if isinstance(pred, dict) and pred.get("folder"):
        lines.append(f"  predefined_descriptors_dir: {pred['folder']}")
    lines.append("run:")
    lines.append(f"  output_dir: {raw.get('result_folder', 'runs/my_run')}")
    if raw.get("predefined_descriptors"):
        lines.append("structure_prediction:")
        lines.append("  enabled: false")
        lines.append("descriptors:")
        lines.append("  enabled: false")
        lines.append("automl:")
        lines.append("  enabled: true")
    elif raw.get("input_structures_folder") and not raw.get("input_csvs_folder"):
        lines.append("descriptors:")
        lines.append("  enabled: true")
        lines.append("automl:")
        lines.append("  enabled: false")
    elif raw.get("input_structures_folder"):
        lines.append("descriptors:")
        lines.append("  enabled: true")
        lines.append("automl:")
        lines.append("  enabled: true")
    else:
        lines.append("structure_prediction:")
        lines.append("  enabled: true")
        lines.append("descriptors:")
        lines.append("  enabled: true")
        lines.append("automl:")
        lines.append("  enabled: true")
    return "\n".join(lines)
