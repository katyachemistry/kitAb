"""Pipeline stage adapters around existing kitAb engines (no formula changes)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from kitab.config import Manifest, write_resolved_manifest
from kitab.logging_state import RunLogger


def processing_n_cpu(manifest: Manifest) -> int:
    """CPU workers for IMGT renumber, OpenMM minimize, descriptors, AutoML."""
    return max(1, int(manifest.run.n_cpu or os.cpu_count() or 8))


def _repo_root(manifest: Manifest) -> Path:
    return manifest.repo_root


def _py_cmd() -> list[str]:
    env = os.environ.get("PY")
    if env:
        import shlex

        return shlex.split(env)
    kitab_env = os.environ.get("KITAB_ENV", "kitab")
    return ["conda", "run", "--no-capture-output", "-n", kitab_env, "python"]


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        text=True,
        capture_output=False,
        check=check,
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_input_hashes(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        if p.is_file():
            out[str(p.resolve())] = file_sha256(p)
    return out


def assert_inputs_unchanged(before: dict[str, str]) -> None:
    for path_s, digest in before.items():
        p = Path(path_s)
        if not p.is_file():
            raise RuntimeError(f"Input file disappeared during run: {p}")
        now = file_sha256(p)
        if now != digest:
            raise RuntimeError(
                f"Input file was modified during run (forbidden): {p}\n"
                f"  before={digest}\n  after={now}"
            )


def prepare_internal_configs(manifest: Manifest, logger: RunLogger) -> Path:
    """Write resolved manifest + generated run config from the canonical Manifest."""
    out = manifest.run.output_dir
    internal = out / "internal"
    internal.mkdir(parents=True, exist_ok=True)
    resolved = write_resolved_manifest(manifest, out / "resolved_manifest.yaml")
    logger.info(f"Resolved manifest: {resolved}")

    from utils.prepare_run_config import (
        build_run_config,
        prepare_from_manifest,
        write_run_config,
    )

    dest = internal / "run_config.yaml"
    logger.event(stage="prepare", status="started", command="prepare_from_manifest")
    try:
        # Pipeline owns the output-dir existence guard; always allow prepare to
        # write splits under the already-created run directory.
        plan = prepare_from_manifest(manifest, resume=True)
        run_config = build_run_config(plan, manifest.repo_root)
        write_run_config(run_config, dest, plan["source_config"])
    except SystemExit as exc:
        raise RuntimeError(str(exc) or "prepare_from_manifest failed") from exc
    logger.event(
        stage="prepare",
        status="ok",
        message=f"run_config={dest}",
        extra={"run_config": str(dest)},
    )
    return dest


def run_structure_prediction(manifest: Manifest, logger: RunLogger, run_config: Path) -> None:
    if not manifest.structure_prediction.enabled:
        logger.event(stage="predict", status="skipped", message="structure_prediction.enabled is false")
        return
    src = manifest.repo_root / "src"
    predict = src / "predict_structure.sh"
    structures_root = manifest.run.output_dir / "structures"
    structures_root.mkdir(parents=True, exist_ok=True)

    # Collect CSV paths from run config.
    import yaml

    cfg = yaml.safe_load(run_config.read_text(encoding="utf-8")) or {}
    csvs: list[Path] = []
    seen: set[str] = set()
    for key, block in cfg.items():
        if key == "batch_result_root" or not isinstance(block, dict):
            continue
        rel = block.get("path")
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = (manifest.repo_root / p).resolve()
        if str(p) in seen:
            continue
        seen.add(str(p))
        csvs.append(p)
    if not csvs:
        raise RuntimeError("No CSV datasets found for structure prediction")

    models = list(manifest.structure_prediction.model)
    device = manifest.structure_prediction.device

    for model in models:
        batch = manifest.structure_prediction.batch_size
        if batch is None:
            batch = 50 if model == "flashabb" else 4

        env = {
            "ABB3_DEVICE": device,
            "ABB2_DEVICE": device,
            "FLASHABB_DEVICE": device,
            "ABB3_BATCH_SIZE": str(batch),
            "ABB2_BATCH_SIZE": str(batch),
            "FLASHABB_BATCH_SIZE": str(batch),
        }
        args = [
            "bash",
            str(predict),
            f"--{model}",
            "--output-root",
            str(structures_root),
            "--runs",
            str(manifest.structure_prediction.runs),
            "--no-renumber",
            "--no-minimize",
        ]
        for csv_path in csvs:
            args.extend(["--csv", str(csv_path)])
        if manifest.structure_prediction.skip_existing:
            args.append("--skip-existing")

        logger.event(
            stage="predict",
            status="started",
            command=" ".join(args),
            extra={"model": model},
        )
        try:
            _run(args, cwd=manifest.repo_root, env=env, check=True)
            logger.event(stage="predict", status="ok", extra={"model": model})
        except subprocess.CalledProcessError as exc:
            logger.event(
                stage="predict",
                status="error",
                message=str(exc),
                extra={"model": model},
            )
            raise


def _structure_dirs_from_run_config(manifest: Manifest, run_config: Path) -> list[tuple[str, Path, Path | None]]:
    import yaml

    cfg = yaml.safe_load(run_config.read_text(encoding="utf-8")) or {}
    jobs: list[tuple[str, Path, Path | None]] = []
    for key, block in cfg.items():
        if key == "batch_result_root" or not isinstance(block, dict):
            continue
        struct_rel = block.get("structure_dir")
        if not struct_rel:
            continue
        struct_dir = Path(struct_rel)
        if not struct_dir.is_absolute():
            struct_dir = (manifest.repo_root / struct_dir).resolve()
        csv_path = None
        if block.get("path"):
            csv_path = Path(block["path"])
            if not csv_path.is_absolute():
                csv_path = (manifest.repo_root / csv_path).resolve()
        jobs.append((key, struct_dir, csv_path))
    return jobs


def _copy_tree_if_needed(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if src.resolve() == dest.resolve():
        return
    shutil.copytree(src, dest)


def process_structures(manifest: Manifest, logger: RunLogger, run_config: Path) -> list[Path]:
    """Copy caller structures into output (immutable inputs) and optionally minimize/renumber."""
    jobs = _structure_dirs_from_run_config(manifest, run_config)
    processed_root = manifest.run.output_dir / "structures_processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    out_dirs: list[Path] = []

    need_proc = manifest.structure_processing.enabled and (
        manifest.structure_processing.minimize or manifest.structure_processing.renumber_imgt
    )
    # For predict mode, structures already live under output_dir/structures.
    for key, struct_dir, _csv in jobs:
        if not struct_dir.is_dir():
            logger.failure(
                stage="process_structures",
                dataset=key,
                item=str(struct_dir),
                reason="structure folder missing",
            )
            continue
        dest = processed_root / key
        if manifest.structure_prediction.enabled and str(struct_dir).startswith(
            str(manifest.run.output_dir)
        ):
            # Work on a copy under structures_processed to avoid mutating prediction outputs
            # only when processing is requested; otherwise descriptors can use prediction dirs.
            if need_proc:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(struct_dir, dest)
                work = dest
            else:
                work = struct_dir
        else:
            # Existing user structures: always copy into output before any processing.
            if not dest.exists():
                logger.info(f"Copying structures (read-only inputs) {struct_dir} -> {dest}")
                shutil.copytree(struct_dir, dest)
            work = dest

        n_jobs = processing_n_cpu(manifest)
        if manifest.structure_processing.enabled and manifest.structure_processing.minimize:
            ok = minimize_directory_with_retries(
                work,
                attempts=manifest.structure_processing.minimize_attempts,
                logger=logger,
                dataset=key,
                repo_root=manifest.repo_root,
                jobs=n_jobs,
            )
            if not ok:
                logger.event(
                    stage="process_structures",
                    status="error",
                    dataset=key,
                    message=f"minimization failed after {manifest.structure_processing.minimize_attempts} attempts",
                )
        if manifest.structure_processing.enabled and manifest.structure_processing.renumber_imgt:
            predict = manifest.repo_root / "src" / "predict_structure.sh"
            cmd = [
                "bash",
                str(predict),
                "--renumber-only",
                "--in-place",
                "--structures-dir",
                str(work),
                "--renumber-jobs",
                str(n_jobs),
            ]
            logger.event(stage="renumber", status="started", dataset=key, command=" ".join(cmd))
            try:
                _run(cmd, cwd=manifest.repo_root, check=True)
                logger.event(stage="renumber", status="ok", dataset=key)
            except subprocess.CalledProcessError as exc:
                logger.failure(
                    stage="renumber",
                    dataset=key,
                    item=str(work),
                    reason=str(exc),
                )
        out_dirs.append(work)
    logger.event(stage="process_structures", status="ok", message=f"{len(out_dirs)} folder(s)")
    return out_dirs


def minimize_pdb_with_retries(
    pdb: Path,
    *,
    attempts: int,
    logger: RunLogger,
    dataset: str,
    repo_root: Path,
) -> bool:
    """Minimize one PDB/CIF with up to `attempts` retries into temp then promote."""
    predict = repo_root / "src" / "predict_structure.sh"
    ok = False
    last_err = ""
    for attempt in range(1, attempts + 1):
        tmp_dir = pdb.parent / f".minimize_tmp_{pdb.stem}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_pdb = tmp_dir / pdb.name
        shutil.copy2(pdb, tmp_pdb)
        cmd = [
            "bash",
            str(predict),
            "--minimization-only",
            "--in-place",
            "--structures-dir",
            str(tmp_dir),
            "--minimize-jobs",
            "1",
        ]
        logger.event(
            stage="minimize",
            status="started",
            dataset=dataset,
            item=pdb.name,
            attempt=attempt,
            command=" ".join(cmd),
        )
        try:
            # OpenMM in the kitab env can load a CUDA plugin that fails
            # (PTX version mismatch) even when the CPU platform is requested.
            _run(
                cmd,
                cwd=repo_root,
                check=True,
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "OPENMM_DEFAULT_PLATFORM": "CPU",
                },
            )
            promoted = tmp_dir / pdb.name
            if not promoted.is_file():
                raise RuntimeError("minimizer did not write output PDB")
            os.replace(promoted, pdb)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.event(
                stage="minimize",
                status="ok",
                dataset=dataset,
                item=pdb.name,
                attempt=attempt,
            )
            ok = True
            break
        except Exception as exc:  # noqa: BLE001 - per-item resilience
            last_err = str(exc)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.event(
                stage="minimize",
                status="retry",
                dataset=dataset,
                item=pdb.name,
                attempt=attempt,
                message=last_err,
            )
    if not ok:
        msg = f"Minimization failed after {attempts} attempts for {pdb}: {last_err}"
        logger.failure(
            stage="minimize",
            dataset=dataset,
            item=pdb.name,
            reason=msg,
            attempt=attempts,
        )
        logger.error(msg)
    return ok


def minimize_directory_with_retries(
    structure_dir: Path,
    *,
    attempts: int,
    logger: RunLogger,
    dataset: str,
    repo_root: Path,
    jobs: int | None = None,
) -> bool:
    """Minimize each PDB independently with up to `attempts` retries into temp then promote.

    PDBs in the same folder run concurrently (``jobs`` workers). Each worker is a
    one-structure ``predict_structure.sh`` subprocess so retries stay isolated.
    """
    pdbs = sorted(structure_dir.glob("*.pdb")) + sorted(structure_dir.glob("*.cif"))
    if not pdbs:
        logger.info(f"No PDB/CIF files to minimize in {structure_dir}")
        return True
    n_jobs = max(1, int(jobs or 1))
    n_jobs = min(n_jobs, len(pdbs))
    logger.event(
        stage="minimize",
        status="started",
        dataset=dataset,
        message=f"{len(pdbs)} structure(s), {n_jobs} parallel worker(s)",
    )
    all_ok = True
    if n_jobs == 1:
        for pdb in pdbs:
            if not minimize_pdb_with_retries(
                pdb,
                attempts=attempts,
                logger=logger,
                dataset=dataset,
                repo_root=repo_root,
            ):
                all_ok = False
        return all_ok

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [
            pool.submit(
                minimize_pdb_with_retries,
                pdb,
                attempts=attempts,
                logger=logger,
                dataset=dataset,
                repo_root=repo_root,
            )
            for pdb in pdbs
        ]
        for fut in as_completed(futures):
            if not fut.result():
                all_ok = False
    return all_ok


PROPKA_COVERAGE_MARKER = "PropKa coverage incomplete"


def _read_failed_structure_rows(tsv: Path) -> list[dict[str, str]]:
    if not tsv.is_file():
        return []
    with open(tsv, encoding="utf-8") as f:
        return [
            {k: str(v or "") for k, v in row.items()}
            for row in csv.DictReader(f, delimiter="\t")
        ]


def _write_failed_structure_rows(tsv: Path, rows: list[dict[str, str]]) -> None:
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "structure", "reason"], delimiter="\t"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.get("dataset", ""),
                    "structure": row.get("structure", ""),
                    "reason": row.get("reason", ""),
                }
            )


def _clear_descriptor_artifacts(desc_dataset_dir: Path, name: str) -> None:
    """Remove helper + result artifacts so a structure is fully reprocessed."""
    for path in (
        desc_dataset_dir / "dssp" / f"{name}.dssp",
        desc_dataset_dir / "propka" / f"{name}_full.pka",
        desc_dataset_dir / "propka" / f"{name}_full.log",
        desc_dataset_dir / "propka" / "tmp_structures" / f"{name}_full.pdb",
        desc_dataset_dir / "propka" / "tmp_structures" / f"{name}_full.propka_map.json",
        desc_dataset_dir / "sasa" / f"{name}_full.sasa",
        desc_dataset_dir / "sasa" / f"{name}_H_full.sasa",
        desc_dataset_dir / "sasa" / f"{name}_L_full.sasa",
        desc_dataset_dir / "results" / f"{name}.json",
    ):
        path.unlink(missing_ok=True)


def _write_names_csv(path: Path, names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for name in names:
            writer.writerow([name])
    return path


def _descriptor_cmd(
    *,
    script: Path,
    desc_root: Path,
    work: Path,
    n_cpu: str,
    names_csv: Path | None,
    cleanup: bool,
    batch_size: int | None,
    resume: bool,
) -> list[str]:
    cmd = ["bash", str(script), "--output-dir", str(desc_root), "--append-failures"]
    if names_csv is not None:
        cmd.extend(["--names-from-csv", str(names_csv)])
    if cleanup:
        cmd.append("--remove_helper_outputs")
    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
    if resume:
        cmd.extend(["--skip-existing", "--skip-failed"])
    cmd.extend([str(work), n_cpu])
    return cmd


def _retry_propka_coverage_with_minimize(
    manifest: Manifest,
    logger: RunLogger,
    *,
    desc_root: Path,
    failed_tsv: Path,
    work_dirs: dict[str, Path],
    script: Path,
    n_cpu: str,
) -> None:
    """Minimize + re-run descriptors for PropKa coverage failures (up to N cycles)."""
    max_retries = manifest.descriptors.propka_minimize_retries
    if max_retries <= 0:
        return

    for attempt in range(1, max_retries + 1):
        rows = _read_failed_structure_rows(failed_tsv)
        propka_rows = [
            r for r in rows if PROPKA_COVERAGE_MARKER in (r.get("reason") or "")
        ]
        if not propka_rows:
            break

        other_rows = [
            r for r in rows if PROPKA_COVERAGE_MARKER not in (r.get("reason") or "")
        ]
        by_dataset: dict[str, list[str]] = {}
        for row in propka_rows:
            key = (row.get("dataset") or "").strip()
            name = (row.get("structure") or "").strip()
            if not key or not name:
                continue
            by_dataset.setdefault(key, []).append(name)

        n_structs = sum(len(v) for v in by_dataset.values())
        logger.event(
            stage="descriptors",
            status="propka_minimize_retry",
            attempt=attempt,
            message=(
                f"PropKa coverage incomplete for {n_structs} structure(s); "
                f"minimizing and re-running descriptors "
                f"(attempt {attempt}/{max_retries})"
            ),
        )

        for key, names in sorted(by_dataset.items()):
            work = work_dirs.get(key)
            if work is None or not work.is_dir():
                logger.failure(
                    stage="descriptors",
                    dataset=key,
                    item=",".join(names[:5]),
                    reason="structure folder missing for PropKa minimize retry",
                )
                continue
            desc_ds = desc_root / key
            retry_jobs = max(1, min(processing_n_cpu(manifest), len(names)))
            to_min: list[Path] = []
            for name in names:
                pdb = work / f"{name}.pdb"
                if not pdb.is_file():
                    logger.failure(
                        stage="minimize",
                        dataset=key,
                        item=name,
                        reason=f"PDB missing for PropKa retry: {pdb}",
                    )
                    continue
                to_min.append(pdb)
                _clear_descriptor_artifacts(desc_ds, name)
            if to_min:
                with ThreadPoolExecutor(max_workers=retry_jobs) as pool:
                    futs = [
                        pool.submit(
                            minimize_pdb_with_retries,
                            pdb,
                            attempts=1,
                            logger=logger,
                            dataset=key,
                            repo_root=manifest.repo_root,
                        )
                        for pdb in to_min
                    ]
                    for fut in as_completed(futs):
                        fut.result()

        # Drop PropKa rows while retrying; successes stay off the TSV, new failures re-append.
        _write_failed_structure_rows(failed_tsv, other_rows)

        retry_root = manifest.run.output_dir / "internal" / "propka_minimize_retries"
        retry_root.mkdir(parents=True, exist_ok=True)
        for key, names in sorted(by_dataset.items()):
            work = work_dirs.get(key)
            if work is None or not work.is_dir():
                continue
            names_csv = _write_names_csv(
                retry_root / f"{key}_attempt{attempt}.csv", sorted(set(names))
            )
            cmd = _descriptor_cmd(
                script=script,
                desc_root=desc_root,
                work=work,
                n_cpu=n_cpu,
                names_csv=names_csv,
                cleanup=manifest.descriptors.cleanup,
                batch_size=manifest.descriptors.batch_size,
                # Always recompute this subset after minimize; do not skip-failed.
                resume=False,
            )
            logger.event(
                stage="descriptors",
                status="started",
                dataset=key,
                attempt=attempt,
                command=" ".join(cmd),
                message=f"propka_minimize_retry n={len(names)}",
            )
            try:
                _run(cmd, cwd=manifest.repo_root, check=False)
                logger.event(
                    stage="descriptors",
                    status="ok",
                    dataset=key,
                    attempt=attempt,
                    message="propka_minimize_retry",
                )
            except Exception as exc:  # noqa: BLE001
                logger.failure(
                    stage="descriptors",
                    dataset=key,
                    item=str(work),
                    reason=str(exc),
                    attempt=attempt,
                )

    remaining = [
        r
        for r in _read_failed_structure_rows(failed_tsv)
        if PROPKA_COVERAGE_MARKER in (r.get("reason") or "")
    ]
    if remaining:
        logger.event(
            stage="descriptors",
            status="propka_minimize_exhausted",
            message=(
                f"{len(remaining)} structure(s) still have PropKa coverage incomplete "
                f"after {max_retries} minimize+descriptor attempt(s)"
            ),
        )


def run_descriptors(manifest: Manifest, logger: RunLogger, run_config: Path) -> Path:
    src = manifest.repo_root / "src"
    script = src / "get_descriptors.sh"
    desc_root = manifest.run.output_dir / "descriptors"
    desc_root.mkdir(parents=True, exist_ok=True)
    failed_tsv = manifest.run.output_dir / "failed_structures.tsv"
    failed_tsv.write_text("dataset\tstructure\treason\n", encoding="utf-8")

    jobs = _structure_dirs_from_run_config(manifest, run_config)
    # Prefer processed copies when present.
    processed_root = manifest.run.output_dir / "structures_processed"
    n_cpu = str(manifest.run.n_cpu or os.cpu_count() or 4)
    work_dirs: dict[str, Path] = {}

    for key, struct_dir, csv_path in jobs:
        work = processed_root / key
        if not work.is_dir():
            work = struct_dir
        if not work.is_dir():
            logger.failure(
                stage="descriptors",
                dataset=key,
                item=str(struct_dir),
                reason="structure folder missing",
            )
            continue
        work_dirs[key] = work
        cmd = _descriptor_cmd(
            script=script,
            desc_root=desc_root,
            work=work,
            n_cpu=n_cpu,
            names_csv=csv_path,
            cleanup=manifest.descriptors.cleanup,
            batch_size=manifest.descriptors.batch_size,
            resume=manifest.run.resume,
        )
        logger.event(
            stage="descriptors",
            status="started",
            dataset=key,
            command=" ".join(cmd),
        )
        # get_descriptors.sh historically exits 0 even on partial failures; we inspect TSV later.
        try:
            _run(cmd, cwd=manifest.repo_root, check=False)
            logger.event(stage="descriptors", status="ok", dataset=key)
        except Exception as exc:  # noqa: BLE001
            logger.failure(
                stage="descriptors", dataset=key, item=str(work), reason=str(exc)
            )

    _retry_propka_coverage_with_minimize(
        manifest,
        logger,
        desc_root=desc_root,
        failed_tsv=failed_tsv,
        work_dirs=work_dirs,
        script=script,
        n_cpu=n_cpu,
    )

    logger.event(stage="descriptors", status="ok", message=f"root={desc_root}")
    return desc_root


def check_dataset_completeness(
    manifest: Manifest, logger: RunLogger, run_config: Path
) -> dict[str, bool]:
    """Return mapping dataset_key -> complete enough for AutoML."""
    import yaml

    cfg = yaml.safe_load(run_config.read_text(encoding="utf-8")) or {}
    completeness: dict[str, bool] = {}
    failed_tsv = manifest.run.output_dir / "failed_structures.tsv"
    failed_names: set[str] = set()
    if failed_tsv.is_file():
        with open(failed_tsv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                failed_names.add(str(row.get("structure") or "").strip())

    for key, block in cfg.items():
        if key == "batch_result_root" or not isinstance(block, dict):
            continue
        csv_rel = block.get("path")
        dev_rel = block.get("developability_results_path")
        if not csv_rel or not dev_rel:
            continue
        csv_path = Path(csv_rel)
        if not csv_path.is_absolute():
            csv_path = (manifest.repo_root / csv_path).resolve()
        dev_path = Path(dev_rel)
        if not dev_path.is_absolute():
            dev_path = (manifest.repo_root / dev_path).resolve()
        if not csv_path.is_file():
            completeness[key] = False
            logger.event(
                stage="completeness",
                status="skipped_incomplete",
                dataset=key,
                message=f"missing CSV {csv_path}",
            )
            continue
        with open(csv_path, encoding="utf-8") as f:
            names = [r["name"] for r in csv.DictReader(f) if r.get("name")]
        missing: list[str] = []
        for name in names:
            json_path = dev_path / f"{name}.json"
            alt = dev_path / "results" / f"{name}.json"
            if json_path.is_file() or alt.is_file():
                continue
            missing.append(name)
        if missing or (failed_names & set(names)):
            completeness[key] = False
            logger.event(
                stage="completeness",
                status="skipped_incomplete",
                dataset=key,
                message=f"missing_or_failed={len(missing)}; examples={missing[:5]}",
                extra={"missing": missing[:50]},
            )
        else:
            completeness[key] = True
            logger.event(stage="completeness", status="ok", dataset=key)
    if not completeness:
        # structures-only / no automl
        logger.event(stage="completeness", status="skipped", message="no AutoML datasets")
    return completeness


def filter_run_config_for_complete(
    run_config: Path, completeness: dict[str, bool], out_path: Path
) -> Path:
    import yaml

    cfg = yaml.safe_load(run_config.read_text(encoding="utf-8")) or {}
    filtered: dict[str, Any] = {}
    for key, block in cfg.items():
        if key == "batch_result_root":
            filtered[key] = block
            continue
        if not isinstance(block, dict):
            filtered[key] = block
            continue
        if key in completeness and not completeness[key]:
            continue
        # Keep non-automl blocks and complete ones.
        if key not in completeness or completeness[key]:
            filtered[key] = block
    out_path.write_text(
        yaml.safe_dump(filtered, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return out_path


def run_automl(manifest: Manifest, logger: RunLogger, run_config: Path) -> Path:
    """Compare the four techniques under nested CV, then fit and save the winner."""
    script = manifest.repo_root / "src" / "run_automl.sh"
    models_root = manifest.run.output_dir / "models"
    cmd = [
        "bash",
        str(script),
        "--config",
        str(run_config),
        "--techniques",
        ",".join(manifest.automl.techniques),
        "--cv-mode",
        manifest.automl.cv_mode,
        "--models-root",
        str(models_root),
    ]
    if manifest.run.n_cpu:
        cmd.extend(["--jobs", str(manifest.run.n_cpu)])
    if manifest.run.resume or manifest.run.skip_existing_results:
        cmd.append("--resume")
    if not manifest.automl.save_final_model:
        cmd.append("--no-final-model")
    logger.event(stage="automl", status="started", command=" ".join(cmd))
    _run(cmd, cwd=manifest.repo_root, check=True)
    batch_root = manifest.run.output_dir / "automl"
    if manifest.automl.save_final_model:
        _publish_model_index(models_root, manifest, logger)
    logger.event(stage="automl", status="ok", message=str(batch_root))
    return batch_root


def _publish_model_index(models_root: Path, manifest: Manifest, logger: RunLogger) -> None:
    from kitab.models import enrich_model_meta, validate_model_roundtrip, write_meta_with_checksum

    if not models_root.is_dir():
        return
    index: list[dict[str, Any]] = []
    for meta_path in sorted(models_root.glob("**/meta.json")):
        model_dir = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = enrich_model_meta(
                meta,
                model_dir=model_dir,
                manifest_checksum=manifest.checksum(),
                package_versions={"kitab": "0.2.0"},
            )
            meta = write_meta_with_checksum(model_dir, meta)
            validate_model_roundtrip(model_dir)
            index.append(
                {
                    "model_dir": str(model_dir),
                    "technique": meta.get("technique"),
                    "eval_model": meta.get("eval_model"),
                    "target_col": meta.get("target_col"),
                    "dataset_stem": meta.get("dataset_stem"),
                    "cv_spearman_pooled_oof": meta.get("cv_spearman_pooled_oof"),
                    "n_features": meta.get("n_features"),
                    "estimator_sha256": meta.get("estimator_sha256"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.failure(
                stage="automl",
                dataset=str(model_dir.name),
                item="model_index",
                reason=f"model validation failed: {exc}",
            )
    (models_root / "model_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
