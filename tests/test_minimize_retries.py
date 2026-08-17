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
