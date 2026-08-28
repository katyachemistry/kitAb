from __future__ import annotations

from pathlib import Path
from unittest import mock

from kitab.logging_state import RunLogger
from kitab.stages import minimize_directory_with_retries


def test_minimize_retries_five_times(tmp_path: Path, repo_root: Path):
    struct = tmp_path / "structs"
    struct.mkdir()
    pdb = struct / "mAb1.pdb"
    pdb.write_text("ATOM\nEND\n")
    logger = RunLogger(tmp_path / "out")

    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("minimize failed")

    with mock.patch("kitab.stages._run", side_effect=boom):
        ok = minimize_directory_with_retries(
            struct,
            attempts=5,
            logger=logger,
            dataset="ab21_mini",
            repo_root=repo_root,
        )
    assert ok is False
    assert calls["n"] == 5
    failures = (tmp_path / "out" / "logs" / "failures.tsv").read_text()
    assert "Minimization failed after 5 attempts" in failures


def test_minimize_directory_runs_pdbs_in_parallel(tmp_path: Path, repo_root: Path):
    struct = tmp_path / "structs"
    struct.mkdir()
    for name in ("mAb1.pdb", "mAb2.pdb", "mAb3.pdb"):
        (struct / name).write_text("ATOM\nEND\n")
    logger = RunLogger(tmp_path / "out")
    seen: list[str] = []

    def fake_run(cmd, **kwargs):
        item = Path(cmd[cmd.index("--structures-dir") + 1]).name
        seen.append(item)
        tmp = Path(cmd[cmd.index("--structures-dir") + 1])
        pdbs = list(tmp.glob("*.pdb"))
        assert len(pdbs) == 1

    with mock.patch("kitab.stages._run", side_effect=fake_run):
        ok = minimize_directory_with_retries(
            struct,
            attempts=1,
            logger=logger,
            dataset="ab21_mini",
            repo_root=repo_root,
            jobs=3,
        )
    assert ok is True
    assert len(seen) == 3


def test_process_structures_passes_renumber_jobs(
    tmp_path: Path, repo_root: Path, fixtures_dir: Path
):
    from kitab.config import parse_manifest_dict
    from kitab.stages import process_structures

    struct = tmp_path / "structs"
    struct.mkdir()
    (struct / "mAb1.pdb").write_text("ATOM\nEND\n")
    out = tmp_path / "out"
    manifest = parse_manifest_dict(
        {
            "inputs": {
                "datasets_dir": str(fixtures_dir / "csv"),
                "structures_dir": str(struct),
            },
            "run": {"output_dir": str(out), "n_cpu": 12},
            "structure_prediction": {"enabled": False},
            "structure_processing": {
                "enabled": True,
                "minimize": False,
                "renumber_imgt": True,
            },
            "descriptors": {"enabled": True},
            "automl": {"enabled": False},
        },
        source_path=tmp_path / "t.yaml",
        repo_root=repo_root,
    )
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        "ab21_mini:\n"
        f"  path: {fixtures_dir / 'csv' / 'ab21_mini.csv'}\n"
        f"  structure_dir: {struct}\n",
        encoding="utf-8",
    )
    logger = RunLogger(out)
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)

    with mock.patch("kitab.stages._run", side_effect=fake_run):
        process_structures(manifest, logger, run_cfg)
    assert captured
    cmd = captured[0]
    assert "--renumber-only" in cmd
    assert cmd[cmd.index("--renumber-jobs") + 1] == "12"
