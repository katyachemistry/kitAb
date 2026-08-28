"""Skip-safe structure env wiring: raw predictors + core kitab post-processing."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from unittest import mock

import pytest
import yaml

from kitab.config import load_manifest
from kitab.logging_state import RunLogger
from kitab.stages import run_structure_prediction
from structure.run_abb_batch_from_csv import process_one_dataset_abb2

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        **kwargs,
    )


def test_shell_syntax():
    for rel in ("install.sh", "src/predict_structure.sh"):
        r = _run(["bash", "-n", rel])
        assert r.returncode == 0, r.stderr


def test_install_help_does_not_need_conda(tmp_path: Path):
    fake = tmp_path / "conda"
    fake.write_text("#!/bin/sh\necho unexpected conda >&2; exit 99\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    env["CONDA_EXE"] = str(fake)
    r = subprocess.run(
        [str(REPO_ROOT / "install.sh"), "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr
    assert "--skip" in r.stdout
    assert "abb2" in r.stdout and "abb3" in r.stdout and "flashabb" in r.stdout


@pytest.mark.parametrize(
    "args, snippet",
    [
        (["--skip"], "--skip requires at least one backend"),
        (["--skip", "bogus"], "Unknown backend to skip: bogus"),
        (["--no-abb2"], "Unknown option: --no-abb2"),
        (["--skip", "abb2", "bogus"], "Unknown backend to skip: bogus"),
    ],
)
def test_install_skip_errors(args: list[str], snippet: str):
    r = _run(["./install.sh", *args])
    assert r.returncode == 1
    assert snippet in (r.stderr + r.stdout)


def test_predict_help_does_not_need_conda(tmp_path: Path):
    fake = tmp_path / "conda"
    fake.write_text("#!/bin/sh\necho unexpected conda >&2; exit 99\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    env["CONDA_EXE"] = str(fake)
    r = subprocess.run(
        [str(REPO_ROOT / "src" / "predict_structure.sh"), "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr
    assert "--renumber-only" in r.stdout
    assert "--minimization-only" in r.stdout


def test_predict_script_resolves_backends_lazily():
    src = (REPO_ROOT / "src" / "predict_structure.sh").read_text(encoding="utf-8")
    help_idx = src.find("-h|--help")
    require_idx = src.find("require_conda_python")
    abb2_assign = src.find('ABB2_PYTHON_BIN="$(require_conda_python')
    assert 0 <= help_idx < require_idx
    assert require_idx < abb2_assign
    assert "POSTPROCESS_PYTHON_CMD" in src
    assert "KITAB_PYTHON_CMD" in src


def test_core_and_backend_pins():
    core = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    assert "openmm=8.4.0" in core
    assert "pdbfixer=1.12" in core
    assert "anarci=2024.05.21" in core
    abb2 = (REPO_ROOT / "environment-abb2.yml").read_text(encoding="utf-8")
    assert "openmm=8.4.0" in abb2
    assert "pdbfixer=1.12" in abb2
    flash = (REPO_ROOT / "environment-flashabb.yml").read_text(encoding="utf-8")
    assert "openmm=8.4.0" in flash
    assert "pdbfixer=1.12" in flash


def test_predict_stage_disables_inline_postprocess(
    tmp_path: Path, repo_root: Path, fixtures_dir: Path
):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "inputs": {"datasets_dir": str(fixtures_dir / "csv")},
                "run": {"output_dir": str(tmp_path / "out")},
                "structure_prediction": {"enabled": True, "model": "abb2"},
                "descriptors": {"enabled": True},
                "automl": {"enabled": False},
                "structure_processing": {"enabled": True, "minimize": True, "renumber_imgt": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(cfg, repo_root=repo_root)
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        yaml.safe_dump(
            {
                "ab21_mini": {
                    "path": str(fixtures_dir / "csv" / "ab21_mini.csv"),
                    "structure_dir": str(tmp_path / "out" / "structures" / "ab21_mini_abb2_1"),
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger = RunLogger(tmp_path / "out")
    with mock.patch("kitab.stages._run") as run:
        run.return_value = mock.Mock()
        run_structure_prediction(manifest, logger, run_cfg)
    cmd = run.call_args[0][0]
    assert "--abb2" in cmd
    assert "--no-renumber" in cmd
    assert "--no-minimize" in cmd
    assert cmd[cmd.index("--runs") + 1] == "1"


def test_predict_stage_runs_each_model(
    tmp_path: Path, repo_root: Path, fixtures_dir: Path
):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "inputs": {"datasets_dir": str(fixtures_dir / "csv")},
                "run": {"output_dir": str(tmp_path / "out")},
                "structure_prediction": {
                    "enabled": True,
                    "model": ["abb2", "abb3"],
                    "runs": 3,
                },
                "descriptors": {"enabled": False},
                "automl": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(cfg, repo_root=repo_root)
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        yaml.safe_dump(
            {
                "ab21_mini_abb2_1": {
                    "path": str(fixtures_dir / "csv" / "ab21_mini.csv"),
                    "structure_dir": str(tmp_path / "out" / "structures" / "ab21_mini_abb2_1"),
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger = RunLogger(tmp_path / "out")
    with mock.patch("kitab.stages._run") as run:
        run.return_value = mock.Mock()
        run_structure_prediction(manifest, logger, run_cfg)
    cmds = [call.args[0] for call in run.call_args_list]
    assert len(cmds) == 2
    assert "--abb2" in cmds[0]
    assert "--abb3" in cmds[1]
    assert all("--no-renumber" in cmd and "--no-minimize" in cmd for cmd in cmds)
    assert all(cmd[cmd.index("--runs") + 1] == "3" for cmd in cmds)


def test_abb2_writes_unrefined_pdb(tmp_path: Path):
    csv = tmp_path / "one.csv"
    csv.write_text(
        "name,heavy,light\n"
        "mAb1,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDYWGQGTLVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK\n",
        encoding="utf-8",
    )

    class FakeAntibody:
        ranking = [0]

        def save_single_unrefined(self, filename, index=0):
            Path(filename).write_text(f"UNREFINED index={index}\n", encoding="utf-8")

        def save(self, *args, **kwargs):
            raise AssertionError("antibody.save() must not run OpenMM in the ABB2 env")

    class FakePredictor:
        def predict(self, sequences):
            assert "H" in sequences and "L" in sequences
            return FakeAntibody()

    process_one_dataset_abb2(
        csv,
        "mini",
        FakePredictor(),
        tmp_path / "out",
        runs=1,
        skip_existing=False,
    )
    pdb = tmp_path / "out" / "mini_abb2_1" / "mAb1.pdb"
    assert pdb.is_file()
    assert pdb.read_text(encoding="utf-8").startswith("UNREFINED")
