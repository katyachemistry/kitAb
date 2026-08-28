"""End-to-end kitAb pipeline orchestration."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from kitab.config import Manifest
from kitab.logging_state import RunLogger
from kitab import stages


def plan_text(manifest: Manifest) -> str:
    lines = [
        f"source: {manifest.source_path}",
        f"mode: {manifest.mode}",
        f"output: {manifest.run.output_dir}",
        f"stages: {' -> '.join(manifest.stage_graph())}",
        f"automl: {manifest.automl.enabled}",
        f"techniques: {', '.join(manifest.automl.techniques)}",
        f"cv_mode: {manifest.automl.cv_mode}",
        f"technique_selection: {manifest.automl.technique_selection}",
        f"save_final_model: {manifest.automl.save_final_model}",
        f"structure_prediction.runs: {manifest.structure_prediction.runs}",
        f"structure_prediction.batch_size: {manifest.structure_prediction.batch_size}",
        f"run.n_cpu: {manifest.run.n_cpu}",
        f"minimize_attempts: {manifest.structure_processing.minimize_attempts}",
        f"propka_minimize_retries: {manifest.descriptors.propka_minimize_retries}",
    ]
    if manifest.inputs.datasets_dir:
        lines.append(f"datasets_dir: {manifest.inputs.datasets_dir}")
    if manifest.inputs.structures_dir:
        lines.append(f"structures_dir: {manifest.inputs.structures_dir}")
    if manifest.inputs.predefined_descriptors_dir:
        lines.append(
            f"predefined_descriptors_dir: {manifest.inputs.predefined_descriptors_dir}"
        )
    for w in manifest.warnings:
        lines.append(f"warning: {w.splitlines()[0]}")
    return "\n".join(lines)


def _collect_input_files(manifest: Manifest) -> list[Path]:
    files: list[Path] = []
    if manifest.inputs.datasets_dir and manifest.inputs.datasets_dir.is_dir():
        files.extend(sorted(manifest.inputs.datasets_dir.glob("*.csv")))
    if manifest.inputs.structures_dir and manifest.inputs.structures_dir.is_dir():
        files.extend(sorted(manifest.inputs.structures_dir.rglob("*.pdb"))[:5000])
        files.extend(sorted(manifest.inputs.structures_dir.rglob("*.cif"))[:5000])
    return files


def run_pipeline(manifest: Manifest) -> int:
    out = manifest.run.output_dir
    if out.exists() and not (
        manifest.run.resume or manifest.run.skip_existing_results
    ):
        # Allow empty dir.
        if any(out.iterdir()):
            raise SystemExit(
                f"run.output_dir already exists and is non-empty: {out}\n"
                "Remove it, or pass --resume / set run.resume: true."
            )
    out.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out)
    for w in manifest.warnings:
        logger.info(w)

    logger.info(plan_text(manifest))

    input_hashes = stages.snapshot_input_hashes(_collect_input_files(manifest))
    logger.info(f"Snapshot of {len(input_hashes)} input file hash(es) for immutability check")

    exit_code = 0
    batch_root: Path | None = None
    completeness: dict[str, bool] = {}

    try:
        run_config = stages.prepare_internal_configs(manifest, logger)
        graph = manifest.stage_graph()

        if "predict" in graph:
            try:
                stages.run_structure_prediction(manifest, logger, run_config)
            except Exception as exc:  # noqa: BLE001
                exit_code = 1
                logger.error(f"predict stage failed: {exc}")
                logger.event(stage="predict", status="error", message=str(exc))

        if "process_structures" in graph:
            try:
                stages.process_structures(manifest, logger, run_config)
            except Exception as exc:  # noqa: BLE001
                exit_code = 1
                logger.error(f"process_structures failed: {exc}")

        if "descriptors" in graph:
            try:
                stages.run_descriptors(manifest, logger, run_config)
            except Exception as exc:  # noqa: BLE001
                exit_code = 1
                logger.error(f"descriptors failed: {exc}")

        if "completeness" in graph or "automl" in graph:
            completeness = stages.check_dataset_completeness(
                manifest, logger, run_config
            )
            if completeness and not any(completeness.values()):
                logger.error(
                    "No datasets are complete for AutoML; skipping AutoML and model export"
                )
                exit_code = 1
                graph = [s for s in graph if s != "automl"]
            elif completeness and not all(completeness.values()):
                exit_code = 1
                filtered = stages.filter_run_config_for_complete(
                    run_config,
                    completeness,
                    manifest.run.output_dir / "internal" / "run_config_complete.yaml",
                )
                run_config = filtered
                logger.info(
                    "AutoML will run only on complete datasets; incomplete ones were skipped"
                )

        if "automl" in graph and manifest.automl.enabled:
            try:
                batch_root = stages.run_automl(manifest, logger, run_config)
            except Exception as exc:  # noqa: BLE001
                exit_code = 1
                logger.error(f"automl failed: {exc}")
                logger.event(stage="automl", status="error", message=str(exc))

        stages.assert_inputs_unchanged(input_hashes)
        logger.info("Input immutability check passed")
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        logger.error(f"Pipeline aborted: {exc}")
        logger.info(traceback.format_exc())

    summary: dict[str, Any] = {
        "exit_code": exit_code,
        "stages": manifest.stage_graph(),
        "completeness": completeness,
        "output_dir": str(out),
        "techniques": list(manifest.automl.techniques),
        "cv_mode": manifest.automl.cv_mode,
        "technique_selection": manifest.automl.technique_selection,
        "models_dir": str(out / "models") if batch_root is not None else None,
    }
    logger.write_summary(summary)
    logger.info(f"Done (exit_code={exit_code}). Logs: {logger.run_log}")
    return exit_code
