"""Full-pipeline e2e for ab21 + FlashABB.

Skipped by default (GPU + OpenMM). Opt in with:

    KITAB_RUN_E2E=1 pytest tests/test_e2e_ab21_flashabb_full.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from kitab.cli import main

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "e2e" / "ab21_flashabb_full.yaml"
AB21_CSV = REPO_ROOT / "tests" / "e2e" / "data" / "ab21.csv"


def test_ab21_flashabb_full_pipeline(tmp_path: Path):
    if os.environ.get("KITAB_RUN_E2E") != "1":
        pytest.skip("set KITAB_RUN_E2E=1 to run the ab21 FlashABB full pipeline")

    assert AB21_CSV.is_file()
    assert MANIFEST.is_file()

    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    datasets = list((REPO_ROOT / "tests" / "e2e" / "data").glob("*.csv"))
    assert [p.name for p in datasets] == ["ab21.csv"]

    out = tmp_path / "test_e2e_ab21_flashabb_full"
    raw["run"]["output_dir"] = str(out)
    raw["run"]["resume"] = False
    cfg = tmp_path / "ab21_flashabb_full.yaml"
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    rc = main(["run", str(cfg)])
    assert rc == 0, f"pipeline exited {rc}; see {out}"

    struct_1 = out / "structures" / "ab21_flashabb_1"
    struct_2 = out / "structures" / "ab21_flashabb_2"
    assert struct_1.is_dir() and struct_2.is_dir()
    n_pdb_1 = len(list(struct_1.glob("*.pdb")))
    n_pdb_2 = len(list(struct_2.glob("*.pdb")))
    assert n_pdb_1 == 21, n_pdb_1
    assert n_pdb_2 == 21, n_pdb_2

    processed = out / "structures_processed"
    assert (processed / "ab21_flashabb_1").is_dir()
    assert (processed / "ab21_flashabb_2").is_dir()

    for folder in ("ab21_flashabb_1", "ab21_flashabb_2"):
        results = out / "descriptors" / folder / "results"
        jsons = list(results.glob("*.json"))
        assert len(jsons) == 21, (folder, len(jsons))

    automl = out / "automl"
    assert (automl / "metrics" / "technique_comparison.csv").is_file()
    assert (out / "models" / "model_index.json").is_file()
