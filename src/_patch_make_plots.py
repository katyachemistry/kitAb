#!/usr/bin/env python3
"""One-off patch for make_plots.ipynb — abb2 (2,3,4) + abb3 (1,2,3) tracks."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "make_plots.ipynb"

CELL1 = r'''import importlib.util
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd

REPO_ROOT = Path("/storage/antibody_data/PairedStructures/FASTAb")
DESCRIPTORS_REP_ROOT = REPO_ROOT / "descriptors_reproducibility"

ABB2_REP_IDS = frozenset({2, 3, 4})
ABB3_REP_IDS = frozenset({1, 2, 3})
REPRO_TRACKS: tuple[str, ...] = ("abb2", "abb3")
Track = Literal["abb2", "abb3"]

# Load loader without importing src.utils (package __init__ expects src on PYTHONPATH).
_loader_path = REPO_ROOT / "src/utils/load_results_to_dataframe.py"
_spec = importlib.util.spec_from_file_location("load_results_to_dataframe", _loader_path)
_load_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_load_mod)
load_json_results = _load_mod.load_json_results


def _dataset_and_column_suffix(folder_name: str) -> tuple[str, str, Track] | None:
    """Map folder name -> (dataset_key, column_suffix, track).

    ABB2: ``ab21_abb2_3`` / ``..._abb2_3_propermab`` -> ``abb2_3`` (reps 2–4 only).
    ABB3: ``ab21_abb3_1`` -> ``abb3_1`` (reps 1–3 only).
    """
    m2 = re.match(r"^(.+)_abb2_(\d+)(?:_propermab)?$", folder_name)
    if m2:
        base, rep_s = m2.group(1), int(m2.group(2))
        if rep_s not in ABB2_REP_IDS:
            return None
        suf = f"abb2_{rep_s}"
        if folder_name.endswith("_propermab"):
            suf = f"{suf}_propermab"
        return base, suf, "abb2"

    m3 = re.match(r"^(.+)_abb3_(\d+)(?:_propermab)?(?:_imgt)?$", folder_name)
    if m3:
        base, rep_s = m3.group(1), int(m3.group(2))
        if rep_s not in ABB3_REP_IDS:
            return None
        suf = f"abb3_{rep_s}"
        if folder_name.endswith("_propermab"):
            suf = f"{suf}_propermab"
        return base, suf, "abb3"

    return None


def _build_by_track() -> dict[Track, dict[str, list[tuple[str, Path]]]]:
    out: dict[Track, dict[str, list[tuple[str, Path]]]] = {
        "abb2": defaultdict(list),
        "abb3": defaultdict(list),
    }
    for child in sorted(DESCRIPTORS_REP_ROOT.iterdir()):
        if not child.is_dir():
            continue
        parsed = _dataset_and_column_suffix(child.name)
        if parsed is None:
            continue
        ds_key, col_suffix, track = parsed
        results_dir = child / "results"
        if not results_dir.is_dir():
            continue
        out[track][ds_key].append((col_suffix, results_dir))
    return {t: dict(sorted(d.items())) for t, d in out.items()}


by_track = _build_by_track()
by_dataset_abb2 = by_track["abb2"]
by_dataset_abb3 = by_track["abb3"]
by_dataset = by_dataset_abb2  # default track alias used in some cells


def build_wide_reproducibility_df(run_folders: list[tuple[str, Path]]) -> pd.DataFrame:
    """One row per sample; feature columns like ``feat_abb2_2`` … ``feat_abb3_1``."""

    def _sort_key(item: tuple[str, Path]) -> tuple[int, str]:
        suf, _ = item
        for prefix in ("abb2_", "abb3_"):
            if suf.startswith(prefix):
                rest = suf.removeprefix(prefix)
                rep_part = rest.split("_", 1)[0]
                try:
                    return (0 if prefix == "abb2_" else 1, f"{int(rep_part):06d}_{suf}")
                except ValueError:
                    return (0, suf)
        return (2, suf)

    run_folders = sorted(run_folders, key=_sort_key)
    wide: pd.DataFrame | None = None
    for col_suffix, results_dir in run_folders:
        df = load_json_results(results_dir)
        rename = {c: f"{c}_{col_suffix}" for c in df.columns if c != "name"}
        df = df.rename(columns=rename)
        if wide is None:
            wide = df
        else:
            wide = wide.merge(df, on="name", how="outer")
    if wide is None:
        return pd.DataFrame()
    return wide.sort_values("name").reset_index(drop=True)


reproducibility_dfs_abb2 = {
    ds: build_wide_reproducibility_df(rows) for ds, rows in sorted(by_dataset_abb2.items())
}
reproducibility_dfs_abb3 = {
    ds: build_wide_reproducibility_df(rows) for ds, rows in sorted(by_dataset_abb3.items())
}
reproducibility_dfs = reproducibility_dfs_abb2

{
    "abb2": (list(reproducibility_dfs_abb2.keys())[:5], len(reproducibility_dfs_abb2)),
    "abb3": (list(reproducibility_dfs_abb3.keys())[:5], len(reproducibility_dfs_abb3)),
}
'''

CELL2_REPLACEMENTS = [
    (
        "def antibody_relative_variability_across_runs(\n    wide_df: pd.DataFrame,\n    descriptor: str,\n    *,\n    name_col: str = \"name\",\n    eps: float = EPS_RCV,\n) -> pd.DataFrame:",
        "def antibody_relative_variability_across_runs(\n    wide_df: pd.DataFrame,\n    descriptor: str,\n    *,\n    track: Track = \"abb2\",\n    name_col: str = \"name\",\n    eps: float = EPS_RCV,\n) -> pd.DataFrame:",
    ),
    (
        "    Expects ABB2 repeat columns ``{descriptor}_abb2_1``, ``{descriptor}_abb2_2``, …\n"
        "    (a single ``{descriptor}_abb3`` column, if present, is ignored here). Same\n"
        "    relative CV definition as in ``descriptor_variability_between_runs``.",
        "    Expects repeat columns ``{descriptor}_{track}_1``, ``{descriptor}_{track}_2``, …\n"
        "    Same relative CV definition as in ``descriptor_variability_between_runs``.",
    ),
    (
        '    pat = re.compile(rf"^{re.escape(descriptor)}_abb2_(\\d+)$")',
        '    pat = re.compile(rf"^{re.escape(descriptor)}_{track}_(\\d+)$")',
    ),
    (
        '            f"need at least two run columns matching {descriptor!r}_abb2_<int>; "',
        '            f"need at least two run columns matching {descriptor!r}_{track}_<int>; "',
    ),
]

CELL2_EXAMPLE = '''# Example: Hutchinson ABB2 — general asymmetry — highest cross-run variability first
_hutch = reproducibility_dfs_abb2["hutchinson2023enhancement_top200tm1_igg"]
asymmetry_variability_ranked_abb2 = antibody_relative_variability_across_runs(
    _hutch, "general_asymmetry_score", track="abb2"
)
asymmetry_variability_ranked_abb2.head(15)
'''

CELL3_MD = '''### Descriptor variability between runs

Uses `by_track` from the cell above: **ABB2** repeats `abb2_2`, `abb2_3`, `abb2_4` (and optional `_propermab` variants, excluded from variability); **ABB3** repeats `abb3_1`, `abb3_2`, `abb3_3`. The helpers below concatenate plain **`abb2_<n>`** or **`abb3_<n>`** folders only (no `_propermab`).

For each numeric descriptor and each antibody (`name`), **relative variability across runs** is
\\(\\mathrm{RCV}_i = \\mathrm{std}_r(x_{i,r}) / \\max(|\\mathrm{mean}_r(x_{i,r})|, \\epsilon)\\) with \\(\\epsilon\\) small (only antibodies present in every run with finite values).

The table reports **median RCV** over antibodies (most unstable descriptors at the top within each dataset), **mean RCV**. The next cell stores per-track tables in `descriptor_variability_between_runs_dfs` / `descriptor_median_across_runs_dfs`, with `descriptor_variability_between_runs_df` as the ABB2 alias for downstream cells that loop over `REPRO_TRACKS`.
'''

# For cell 4, I'll do search-replace on the loaded notebook
REPLACEMENTS_CELL4 = [
    (
        "def _abb2_concat_ok(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n    min_runs: int,\n) -> tuple[pd.DataFrame, list[str], int] | None:\n    abb2_only: list[tuple[int, Path]] = []\n    for suf, p in run_folders:\n        m = re.fullmatch(r\"abb2_(\\d+)\", suf)\n        if m:\n            abb2_only.append((int(m.group(1)), p))\n    repeat_folders = sorted(abb2_only, key=lambda x: x[0])",
        "def _repeat_concat_ok(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n    min_runs: int,\n    track: Track,\n) -> tuple[pd.DataFrame, list[str], int] | None:\n    repeats: list[tuple[int, Path]] = []\n    for suf, p in run_folders:\n        m = re.fullmatch(rf\"{track}_(\\d+)\", suf)\n        if m:\n            repeats.append((int(m.group(1)), p))\n    repeat_folders = sorted(repeats, key=lambda x: x[0])",
    ),
    ("loaded = _abb2_concat_ok(run_folders, dataset=dataset, min_runs=2)", "loaded = _repeat_concat_ok(run_folders, dataset=dataset, min_runs=2, track=track)"),
    (
        '    Uses only ``abb2_<n>`` folders (excludes ``abb2_<n>_propermab`` and ``abb3*``).',
        '    Uses only plain ``{track}_<n>`` folders (excludes ``_propermab``).',
    ),
    (
        "def descriptor_variability_between_runs(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n    eps: float = EPS,\n) -> pd.DataFrame:",
        "def descriptor_variability_between_runs(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n    track: Track,\n    eps: float = EPS,\n) -> pd.DataFrame:",
    ),
    (
        "def descriptor_median_across_abb2_runs(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n) -> pd.DataFrame:\n    \"\"\"Wide table: one row per antibody; each column is median of that descriptor across ABB2 runs.\"\"\"\n    loaded = _abb2_concat_ok(run_folders, dataset=dataset, min_runs=1)",
        "def descriptor_median_across_runs(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n    track: Track,\n) -> pd.DataFrame:\n    \"\"\"Wide table: one row per antibody; median of each descriptor across repeat runs.\"\"\"\n    loaded = _repeat_concat_ok(run_folders, dataset=dataset, min_runs=1, track=track)",
    ),
    (
        "# Full table (datasets in sorted key order; within each dataset, largest median RCV first)\n_var_parts = [\n    descriptor_variability_between_runs(rows, dataset=ds)\n    for ds, rows in sorted(by_dataset.items())\n]\ndescriptor_variability_between_runs_df = pd.concat(_var_parts, ignore_index=True)\n\n# Per-dataset wide tables (ABB2 only): median across repeat runs per antibody — compare to single-run ABB3 later\ndescriptor_median_across_abb2_runs_dfs: dict[str, pd.DataFrame] = {\n    ds: descriptor_median_across_abb2_runs(rows, dataset=ds)\n    for ds, rows in sorted(by_dataset.items())\n}\n\n# Preview: top descriptors per dataset by median relative CV (full frame: descriptor_variability_between_runs_df)\ndescriptor_variability_between_runs_df.groupby([\"dataset\", \"descriptor\"], sort=False).head(20)",
        "# Per-track tables\n_descriptor_variability_between_runs_dfs: dict[str, pd.DataFrame] = {}\n_descriptor_median_across_runs_dfs: dict[str, dict[str, pd.DataFrame]] = {}\nfor _track in REPRO_TRACKS:\n    _by = by_track[_track]\n    _parts = [\n        descriptor_variability_between_runs(rows, dataset=ds, track=_track)\n        for ds, rows in sorted(_by.items())\n    ]\n    _descriptor_variability_between_runs_dfs[_track] = pd.concat(_parts, ignore_index=True)\n    _descriptor_median_across_runs_dfs[_track] = {\n        ds: descriptor_median_across_runs(rows, dataset=ds, track=_track)\n        for ds, rows in sorted(_by.items())\n    }\n\ndescriptor_variability_between_runs_dfs = _descriptor_variability_between_runs_dfs\ndescriptor_median_across_runs_dfs = _descriptor_median_across_runs_dfs\ndescriptor_variability_between_runs_df = descriptor_variability_between_runs_dfs[\"abb2\"]\ndescriptor_median_across_abb2_runs_dfs = descriptor_median_across_runs_dfs[\"abb2\"]\n\n# Preview (ABB2 track)\ndescriptor_variability_between_runs_df.groupby([\"dataset\", \"descriptor\"], sort=False).head(20)",
    ),
]

REPLACEMENTS_CELL6 = [
    (
        "    Uses only plain ``abb2_<n>`` folders (same as ``descriptor_variability_between_runs``).",
        "    Uses only plain ``{track}_<n>`` folders (same as ``descriptor_variability_between_runs``).",
    ),
    (
        "def descriptor_relative_variation_across_antibodies(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n    eps: float = EPS,\n    normality_alpha: float = NORMALITY_ALPHA,\n) -> pd.DataFrame:",
        "def descriptor_relative_variation_across_antibodies(\n    run_folders: list[tuple[str, Path]],\n    *,\n    dataset: str,\n    track: Track,\n    eps: float = EPS,\n    normality_alpha: float = NORMALITY_ALPHA,\n) -> pd.DataFrame:",
    ),
    (
        "    abb2_runs: list[tuple[int, str, Path]] = []\n    for suf, results_dir in run_folders:\n        m = re.fullmatch(r\"abb2_(\\d+)\", suf)\n        if m:\n            abb2_runs.append((int(m.group(1)), suf, results_dir))\n    abb2_runs.sort(key=lambda x: x[0])\n\n    for run_index, run_label, results_dir in abb2_runs:",
        "    repeat_runs: list[tuple[int, str, Path]] = []\n    for suf, results_dir in run_folders:\n        m = re.fullmatch(rf\"{track}_(\\d+)\", suf)\n        if m:\n            repeat_runs.append((int(m.group(1)), suf, results_dir))\n    repeat_runs.sort(key=lambda x: x[0])\n\n    for run_index, run_label, results_dir in repeat_runs:",
    ),
    (
        "_var_ab_parts = [\n    descriptor_relative_variation_across_antibodies(rows, dataset=ds)\n    for ds, rows in sorted(by_dataset.items())\n]\ndescriptor_variation_across_antibodies_df = pd.concat(_var_ab_parts, ignore_index=True)",
        "_descriptor_variation_across_antibodies_dfs: dict[str, pd.DataFrame] = {}\nfor _track in REPRO_TRACKS:\n    _parts = [\n        descriptor_relative_variation_across_antibodies(rows, dataset=ds, track=_track)\n        for ds, rows in sorted(by_track[_track].items())\n    ]\n    _descriptor_variation_across_antibodies_dfs[_track] = pd.concat(_parts, ignore_index=True)\n\ndescriptor_variation_across_antibodies_dfs = _descriptor_variation_across_antibodies_dfs\ndescriptor_variation_across_antibodies_df = descriptor_variation_across_antibodies_dfs[\"abb2\"]",
    ),
]

CELL5_MD = '''### Relative variation across antibodies (per run)

For each **repeat run** (`abb2_2` … `abb2_4` or `abb3_1` … `abb3_3`), each dataset, and each numeric descriptor, relative variation **across antibodies** in that run is

\\(\\mathrm{RCV} = \\mathrm{std}_i(x_i) / \\max(|\\mathrm{mean}_i(x_i)|, \\epsilon)\\)

where \\(i\\) indexes antibodies (`name`) in that run's `results/` table. Complements the between-runs table above (which aggregates per-antibody run-to-run RCV).

**Normality:** Shapiro–Wilk on the antibody-level values per (dataset, run, descriptor); `is_normal` is true when `normality_pvalue` ≥ 0.05 (not enough antibodies for the test when `n_antibodies` < 3).

**Robust invariance metric:** `relative_iqr = IQR / max(abs(median), eps)` per run, then `median_relative_iqr_across_runs` per dataset × descriptor. Smaller values indicate descriptors that change less across antibodies within that dataset.
'''

CELL6_INVARIANCE = '''# Per-track robust invariance summaries
_descriptor_antibody_invariance_dfs: dict[str, pd.DataFrame] = {}
_low_antibody_variation_descriptors_dfs: dict[str, pd.DataFrame] = {}

for _track in REPRO_TRACKS:
    _var_ab = descriptor_variation_across_antibodies_dfs[_track]
    _inv = (
        _var_ab.groupby(["dataset", "descriptor"], as_index=False)
        .agg(
            median_iqr_across_runs=("iqr_across_antibodies", "median"),
            median_relative_iqr_across_runs=("relative_iqr", "median"),
            max_relative_iqr_across_runs=("relative_iqr", "max"),
            n_runs=("run", "nunique"),
            n_antibodies_min=("n_antibodies", "min"),
        )
        .sort_values(
            ["dataset", "median_relative_iqr_across_runs", "max_relative_iqr_across_runs"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    _descriptor_antibody_invariance_dfs[_track] = _inv
    descriptor_variation_across_antibodies_dfs[_track] = _var_ab.merge(
        _inv[["dataset", "descriptor", "median_relative_iqr_across_runs"]],
        on=["dataset", "descriptor"],
        how="left",
    )

descriptor_antibody_invariance_dfs = _descriptor_antibody_invariance_dfs
descriptor_antibody_invariance_df = descriptor_antibody_invariance_dfs["abb2"]

RELATIVE_IQR_INVARIANT_THRESHOLD = 0.05
_low_antibody_variation_descriptors_dfs = {
    t: descriptor_antibody_invariance_dfs[t][
        descriptor_antibody_invariance_dfs[t]["median_relative_iqr_across_runs"]
        <= RELATIVE_IQR_INVARIANT_THRESHOLD
    ].reset_index(drop=True)
    for t in REPRO_TRACKS
}
low_antibody_variation_descriptors_df = _low_antibody_variation_descriptors_dfs["abb2"]

low_antibody_variation_descriptors_df
'''


def set_cell_source(cell, text: str) -> None:
    cell["source"] = [line + "\n" for line in text.splitlines()]
    if cell["source"] and not text.endswith("\n"):
        pass  # last line already has \n from splitlines except empty
    # splitlines drops trailing newline on last line - that's fine for nbformat


def apply_replacements(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if old not in text:
            raise KeyError(f"patch fragment not found:\n{old[:120]}...")
        text = text.replace(old, new, 1)
    return text


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    set_cell_source(nb["cells"][1], CELL1)

    c2 = "".join(nb["cells"][2]["source"])
    c2 = apply_replacements(c2, CELL2_REPLACEMENTS)
    c2 = c2.replace(
        '_hutch = reproducibility_dfs["hutchinson2023enhancement_top200tm1_igg"]\n'
        'asymmetry_variability_ranked = antibody_relative_variability_across_runs(\n'
        '    _hutch, "general_asymmetry_score"\n'
        ')\n'
        'asymmetry_variability_ranked.head(15)',
        CELL2_EXAMPLE.strip(),
    )
    set_cell_source(nb["cells"][2], c2)

    set_cell_source(nb["cells"][3], CELL3_MD)

    c4 = "".join(nb["cells"][4]["source"])
    c4 = apply_replacements(c4, REPLACEMENTS_CELL4)
    set_cell_source(nb["cells"][4], c4)

    set_cell_source(nb["cells"][5], CELL5_MD)

    c6 = "".join(nb["cells"][6]["source"])
    c6 = apply_replacements(c6, REPLACEMENTS_CELL6)
    # Replace invariance block at end
    marker = "# One robust invariance summary per dataset × descriptor."
    if marker in c6:
        c6 = c6[: c6.index(marker)] + CELL6_INVARIANCE
    else:
        raise KeyError("cell 6 invariance block not found")
    set_cell_source(nb["cells"][6], c6)

    # Cell 11: loop over tracks
    c11 = """MEDIAN_RCV_THRESHOLD = 0.1

descriptors_exceeding_median_rcv_by_track: dict[str, dict[str, pd.DataFrame]] = {}
n_datasets_exceeding_median_rcv_threshold_by_track: dict[str, dict[str, int]] = {}

for _track in REPRO_TRACKS:
    _var = descriptor_variability_between_runs_dfs[_track]
    if _var.empty:
        descriptors_exceeding_median_rcv_by_track[_track] = {}
        n_datasets_exceeding_median_rcv_threshold_by_track[_track] = {}
        continue
    _high = _var[_var["median_relative_cv"] > MEDIAN_RCV_THRESHOLD].copy()
    _by_ds: dict[str, pd.DataFrame] = {}
    for ds in sorted(_var["dataset"].unique()):
        g = _high[_high["dataset"] == ds]
        _by_ds[ds] = g.sort_values(
            "median_relative_cv",
            ascending=False,
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    descriptors_exceeding_median_rcv_by_track[_track] = _by_ds
    _hits = (
        _high.groupby("descriptor", sort=False)["dataset"]
        .nunique()
        .sort_values(ascending=False, kind="mergesort")
    )
    n_datasets_exceeding_median_rcv_threshold_by_track[_track] = {
        desc: int(k) for desc, k in _hits.items() if k > 0
    }

# ABB2 aliases for cells that predate the dual-track refactor
descriptors_exceeding_median_rcv_by_dataset = descriptors_exceeding_median_rcv_by_track["abb2"]
n_datasets_exceeding_median_rcv_threshold = n_datasets_exceeding_median_rcv_threshold_by_track["abb2"]

for _track in REPRO_TRACKS:
    print(f"=== {_track.upper()} ===")
    for ds in sorted(descriptors_exceeding_median_rcv_by_track[_track].keys()):
        sub = descriptors_exceeding_median_rcv_by_track[_track][ds]
        print(
            ds,
            f"({len(sub)} descriptors with median_relative_cv > {MEDIAN_RCV_THRESHOLD})",
        )
        if sub.empty:
            print("(none)\\n")
        else:
            print(
                sub[["descriptor", "median_relative_cv", "mean_relative_cv"]].to_string(
                    index=False
                )
            )
            print()

n_datasets_exceeding_median_rcv_threshold_by_track
"""
    set_cell_source(nb["cells"][11], c11)

    # Cell 12: loop plots
    c12 = """import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── helpers ──────────────────────────────────────────────────────────────────
SHORT_DATASET = {
    ds: ds.split("_")[0]
    for ds in descriptor_variability_between_runs_dfs["abb2"]["dataset"].unique()
}

_DESCRIPTOR_PREFIXES = (
    "sequence_motives_",  # longest first
    "surface_",
    "general_",
    "core_",
)


def _strip_descriptor_prefix(name: str) -> str:
    """Drop leading category token (surface_, general_, …) for shorter plot labels."""
    s = str(name)
    for p in _DESCRIPTOR_PREFIXES:
        if s.startswith(p):
            return s[len(p) :]
    return s


_save_dir = Path("./plots")
_save_dir.mkdir(parents=True, exist_ok=True)

_THRESH_PCT = MEDIAN_RCV_THRESHOLD * 100


def _fmt_rcv_pct(x, _pos=None):
    return f"{x * 100:.0f}%"


def _plot_variability_heatmap(_track: str) -> None:
    _var = descriptor_variability_between_runs_dfs[_track]
    _n_ds = n_datasets_exceeding_median_rcv_threshold_by_track[_track]
    all_datasets = sorted(_var["dataset"].unique())
    _flagged_descs = sorted(
        _n_ds.keys(),
        key=lambda d: (
            -_n_ds[d],
            -float(_var.loc[_var["descriptor"] == d, "median_relative_cv"].max()),
        ),
    )
    if not _flagged_descs:
        print(f"{_track}: no descriptors above threshold — skipping heatmap")
        return
    _display_labels = [_strip_descriptor_prefix(d) for d in _flagged_descs]
    _pivot = (
        _var[_var["descriptor"].isin(_flagged_descs)]
        .assign(ds_short=lambda df: df["dataset"].map(SHORT_DATASET))
        .pivot_table(
            index="ds_short", columns="descriptor", values="median_relative_cv", aggfunc="first"
        )
        .reindex(columns=_flagged_descs)
        .reindex([SHORT_DATASET[ds] for ds in all_datasets])
    )
    _mat = _pivot.values.astype(float)
    _vmax = max(0.15, float(np.nanmax(_mat)))
    _cmap = mcolors.LinearSegmentedColormap.from_list("rcv_gw_cr", ["white", "crimson"], N=256)
    _norm = mcolors.Normalize(vmin=0, vmax=_vmax)
    _n_rows, n_cols = _mat.shape
    fig_heat, ax_heat = plt.subplots(
        figsize=(max(10, 0.55 * n_cols), max(5, 0.45 * _n_rows)),
        layout="constrained",
    )
    im = ax_heat.imshow(_mat, cmap=_cmap, norm=_norm, aspect="auto")
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(_display_labels, rotation=55, ha="right", fontsize=6)
    ax_heat.set_yticks(range(_n_rows))
    ax_heat.set_yticklabels(list(_pivot.index), fontsize=9)
    div_h = make_axes_locatable(ax_heat)
    cax_h = div_h.append_axes("right", size="3%", pad=0.12)
    fig_heat.colorbar(im, cax=cax_h, label="median RCV (%)", format=_fmt_rcv_pct)
    for r in range(_n_rows):
        for c in range(n_cols):
            v = _mat[r, c]
            if np.isnan(v):
                ax_heat.text(c, r, "–", ha="center", va="center", fontsize=6.5, color="#aaaaaa")
            else:
                col = "white" if v > _vmax * 0.65 else "black"
                ax_heat.text(
                    c, r, f"{v * 100:.1f}%", ha="center", va="center", fontsize=6.5, color=col
                )
    fig_heat.savefig(_save_dir / f"{_track}_variability_heatmap.pdf", bbox_inches="tight")
    fig_heat.savefig(_save_dir / f"{_track}_variability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


for _track in REPRO_TRACKS:
    _plot_variability_heatmap(_track)
"""
    set_cell_source(nb["cells"][12], c12)

    # Cell 14 - wrap in function and loop
    c14_new = '''from collections import Counter

from matplotlib.ticker import FuncFormatter

TOP_AB = 5  # top antibodies per descriptor to report

_DESCRIPTOR_PREFIXES = (
    "sequence_motives_",
    "surface_",
    "general_",
    "core_",
)


def _strip_descriptor_prefix(name: str) -> str:
    """Same as variability plot cell: drop surface_/general_/… for shorter labels."""
    s = str(name)
    for p in _DESCRIPTOR_PREFIXES:
        if s.startswith(p):
            return s[len(p) :]
    return s


def _build_run_dev_and_ab_rcv(
    track: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    run_dev_data: dict[str, pd.DataFrame] = {}
    ab_rcv_data: dict[str, pd.DataFrame] = {}
    flagged_by_ds = descriptors_exceeding_median_rcv_by_track[track]
    flagged_datasets = sorted(ds for ds, fr in flagged_by_ds.items() if not fr.empty)

    for ds in flagged_datasets:
        flagged_descs = list(flagged_by_ds[ds]["descriptor"])
        loaded = _repeat_concat_ok(by_track[track][ds], dataset=ds, min_runs=2, track=track)
        if loaded is None:
            continue
        long_c, num_cols, n_runs = loaded
        flagged_descs = [d for d in flagged_descs if d in num_cols]
        if not flagged_descs:
            continue

        dev_rows: dict[str, list] = {}
        ab_rows: list[dict] = []

        for col in flagged_descs:
            pivot = long_c.pivot_table(index="name", columns="_run", values=col, aggfunc="first")
            pivot = pivot.apply(pd.to_numeric, errors="coerce")
            run_means = pivot.mean(axis=0)
            med = run_means.median()
            mad = (run_means - med).abs().median() + EPS
            z = (run_means - med) / mad
            dev_rows[col] = z.reindex(range(n_runs)).tolist()
            m_ab = pivot.mean(axis=1)
            s_ab = pivot.std(axis=1, ddof=1)
            rcv_ab = s_ab / m_ab.abs().clip(lower=EPS)
            for name, rcv in rcv_ab.items():
                ab_rows.append({"name": str(name), "descriptor": col, "rcv": float(rcv)})

        _dev_df = pd.DataFrame(dev_rows, index=[f"run {r+1}" for r in range(n_runs)]).T
        _dev_df.index.name = "descriptor"
        run_dev_data[ds] = _dev_df
        ab_rcv_data[ds] = pd.DataFrame(ab_rows)

    return run_dev_data, ab_rcv_data, flagged_datasets


def _print_run_dev_summary(
    track: str,
    run_dev_data: dict[str, pd.DataFrame],
    ab_rcv_data: dict[str, pd.DataFrame],
    flagged_datasets: list[str],
) -> None:
    for ds in flagged_datasets:
        dev_df = run_dev_data.get(ds)
        ab_df = ab_rcv_data.get(ds)
        if dev_df is None:
            continue
        print(f"══ {track} / {ds} ══")
        outlier_run_counts = Counter(dev_df.abs().idxmax(axis=1))
        print("  Most-extreme run (by |z-score| of run mean, per descriptor):")
        for run_lbl, cnt in sorted(outlier_run_counts.items(), key=lambda x: -x[1]):
            print(f"    {run_lbl}: most extreme in {cnt}/{len(dev_df)} flagged descriptor(s)")
        print(f"  Top-{TOP_AB} antibodies per flagged descriptor:")
        for desc in dev_df.index:
            sub = ab_df[ab_df["descriptor"] == desc].nlargest(TOP_AB, "rcv")
            parts = ", ".join(f"{r['name']}(rcv={r['rcv']:.2f})" for _, r in sub.iterrows())
            print(f"    {_strip_descriptor_prefix(desc)}: {parts}")
        print()


_RIGHT_COL_DATASETS = frozenset({"jetha2019homology_RT", "pdgf38"})


def _plot_runs_and_antibodies_figure(
    track: str,
    run_dev_data: dict[str, pd.DataFrame],
    ab_rcv_data: dict[str, pd.DataFrame],
    flagged_datasets: list[str],
) -> None:
    _plot_specs = []
    for ds in flagged_datasets:
        dev_df = run_dev_data.get(ds)
        ab_df = ab_rcv_data.get(ds)
        if dev_df is None or ab_df is None:
            continue
        for desc in dev_df.index:
            sub = ab_df.loc[ab_df["descriptor"] == desc, "rcv"].dropna().values
            if len(sub) > 0:
                _plot_specs.append((ds, desc, sub))

    def _axis_specs_for_datasets(dataset_list: list[str]) -> list:
        axis_specs = []
        for ds in dataset_list:
            ds_specs = [spec for spec in _plot_specs if spec[0] == ds]
            if not ds_specs:
                continue
            if axis_specs:
                axis_specs.append(None)
            axis_specs.extend(ds_specs)
        return axis_specs

    def _last_axis_row_for_ds(axis_specs: list) -> dict[str, int]:
        return {spec[0]: idx for idx, spec in enumerate(axis_specs) if spec is not None}

    def _draw_rcv_violin(ax, spec, *, last_row_for_ds: dict[str, int], row_idx: int, prev_ds):
        if spec is None:
            ax.set_visible(False)
            return prev_ds
        ds, desc, sub = spec
        ab_df = ab_rcv_data[ds]
        ds_short = SHORT_DATASET[ds]
        parts = ax.violinplot(
            [sub], positions=[0], vert=False, widths=0.7,
            showmeans=False, showmedians=True, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("ghostwhite")
            body.set_alpha(1.0)
            body.set_edgecolor("#888888")
        if parts.get("cmedians") is not None:
            parts["cmedians"].set_color("#333333")
            parts["cmedians"].set_linewidth(0.9)
        top_names = ab_df[ab_df["descriptor"] == desc].nlargest(3, "rcv")[["name", "rcv"]]
        label_offsets = np.linspace(-0.42, 0.42, len(top_names))
        label_x_offset = 0.015 * max(ax.get_xlim()[1], MEDIAN_RCV_THRESHOLD)
        for label_offset, (_, r_row) in zip(label_offsets, top_names.iterrows()):
            ax.text(
                r_row["rcv"] + label_x_offset, label_offset, str(r_row["name"]),
                fontsize=9, fontweight="bold", va="center", color="maroon",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.3),
                clip_on=False,
            )
        ax.set_ylim(-0.85, 0.85)
        ax.set_yticks([0])
        ax.set_yticklabels([_strip_descriptor_prefix(desc)], fontsize=9)
        ax.axvline(MEDIAN_RCV_THRESHOLD, color="maroon", lw=2.2, ls="--")
        ax.set_xlabel("per-antibody RCV (%)" if row_idx == last_row_for_ds.get(ds) else "", fontsize=9)
        ax.set_title(ds_short if ds != prev_ds else "", fontsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x * 100:.0f}%"))
        ax.tick_params(axis="both", colors="black", labelsize=9)
        return ds

    _left_datasets = [ds for ds in flagged_datasets if ds not in _RIGHT_COL_DATASETS]
    _right_datasets = [ds for ds in flagged_datasets if ds in _RIGHT_COL_DATASETS]
    _left_axis_specs = _axis_specs_for_datasets(_left_datasets)
    _right_axis_specs = _axis_specs_for_datasets(_right_datasets)
    n_rows = max(len(_left_axis_specs), len(_right_axis_specs), 1)
    _n_plot_rows = max(
        sum(1 for s in _left_axis_specs if s is not None),
        sum(1 for s in _right_axis_specs if s is not None),
        1,
    )
    _n_spacer_rows = max(
        len(_left_axis_specs) - sum(1 for s in _left_axis_specs if s is not None),
        len(_right_axis_specs) - sum(1 for s in _right_axis_specs if s is not None),
        0,
    )
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(22, max(3.2, 1.08 * _n_plot_rows + 0.55 * _n_spacer_rows)),
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
        squeeze=False,
    )
    _last_row_left = _last_axis_row_for_ds(_left_axis_specs)
    _last_row_right = _last_axis_row_for_ds(_right_axis_specs)
    _prev_left = _prev_right = None
    for row in range(n_rows):
        if row < len(_left_axis_specs):
            _prev_left = _draw_rcv_violin(
                axes[row, 0], _left_axis_specs[row],
                last_row_for_ds=_last_row_left, row_idx=row, prev_ds=_prev_left,
            )
        else:
            axes[row, 0].set_visible(False)
        if row < len(_right_axis_specs):
            _prev_right = _draw_rcv_violin(
                axes[row, 1], _right_axis_specs[row],
                last_row_for_ds=_last_row_right, row_idx=row, prev_ds=_prev_right,
            )
        else:
            axes[row, 1].set_visible(False)
    plt.tight_layout(h_pad=0.75)
    plt.savefig(_save_dir / f"{track}_variability_runs_and_antibodies.png", dpi=150, bbox_inches="tight")
    plt.savefig(_save_dir / f"{track}_variability_runs_and_antibodies.pdf", bbox_inches="tight")
    plt.show()


for _track in REPRO_TRACKS:
    _rd, _ab, _fds = _build_run_dev_and_ab_rcv(_track)
    _print_run_dev_summary(_track, _rd, _ab, _fds)
    if _fds:
        _plot_runs_and_antibodies_figure(_track, _rd, _ab, _fds)
'''
    set_cell_source(nb["cells"][14], c14_new)

    NB_PATH.write_text(json.dumps(nb, indent=2) + "\n")
    print("patched", NB_PATH)


def patch_cells_12_14_only() -> None:
    """Update heatmap and violin cells only (safe after partial refactor)."""
    nb = json.loads(NB_PATH.read_text())
    c12 = """import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── helpers ──────────────────────────────────────────────────────────────────
SHORT_DATASET = {
    ds: ds.split("_")[0]
    for ds in descriptor_variability_between_runs_dfs["abb2"]["dataset"].unique()
}

_DESCRIPTOR_PREFIXES = (
    "sequence_motives_",  # longest first
    "surface_",
    "general_",
    "core_",
)


def _strip_descriptor_prefix(name: str) -> str:
    """Drop leading category token (surface_, general_, …) for shorter plot labels."""
    s = str(name)
    for p in _DESCRIPTOR_PREFIXES:
        if s.startswith(p):
            return s[len(p) :]
    return s


_save_dir = Path("./plots")
_save_dir.mkdir(parents=True, exist_ok=True)

_THRESH_PCT = MEDIAN_RCV_THRESHOLD * 100


def _fmt_rcv_pct(x, _pos=None):
    return f"{x * 100:.0f}%"


def _plot_variability_heatmap(_track: str) -> None:
    _var = descriptor_variability_between_runs_dfs[_track]
    _n_ds = n_datasets_exceeding_median_rcv_threshold_by_track[_track]
    all_datasets = sorted(_var["dataset"].unique())
    _flagged_descs = sorted(
        _n_ds.keys(),
        key=lambda d: (
            -_n_ds[d],
            -float(_var.loc[_var["descriptor"] == d, "median_relative_cv"].max()),
        ),
    )
    if not _flagged_descs:
        print(f"{_track}: no descriptors above threshold — skipping heatmap")
        return
    _display_labels = [_strip_descriptor_prefix(d) for d in _flagged_descs]
    _pivot = (
        _var[_var["descriptor"].isin(_flagged_descs)]
        .assign(ds_short=lambda df: df["dataset"].map(SHORT_DATASET))
        .pivot_table(
            index="ds_short", columns="descriptor", values="median_relative_cv", aggfunc="first"
        )
        .reindex(columns=_flagged_descs)
        .reindex([SHORT_DATASET[ds] for ds in all_datasets])
    )
    _mat = _pivot.values.astype(float)
    _vmax = max(0.15, float(np.nanmax(_mat)))
    _cmap = mcolors.LinearSegmentedColormap.from_list("rcv_gw_cr", ["white", "crimson"], N=256)
    _norm = mcolors.Normalize(vmin=0, vmax=_vmax)
    _n_rows, n_cols = _mat.shape
    fig_heat, ax_heat = plt.subplots(
        figsize=(max(10, 0.55 * n_cols), max(5, 0.45 * _n_rows)),
        layout="constrained",
    )
    im = ax_heat.imshow(_mat, cmap=_cmap, norm=_norm, aspect="auto")
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(_display_labels, rotation=55, ha="right", fontsize=6)
    ax_heat.set_yticks(range(_n_rows))
    ax_heat.set_yticklabels(list(_pivot.index), fontsize=9)
    div_h = make_axes_locatable(ax_heat)
    cax_h = div_h.append_axes("right", size="3%", pad=0.12)
    fig_heat.colorbar(im, cax=cax_h, label="median RCV (%)", format=_fmt_rcv_pct)
    for r in range(_n_rows):
        for c in range(n_cols):
            v = _mat[r, c]
            if np.isnan(v):
                ax_heat.text(c, r, "–", ha="center", va="center", fontsize=6.5, color="#aaaaaa")
            else:
                col = "white" if v > _vmax * 0.65 else "black"
                ax_heat.text(
                    c, r, f"{v * 100:.1f}%", ha="center", va="center", fontsize=6.5, color=col
                )
    fig_heat.savefig(_save_dir / f"{_track}_variability_heatmap.pdf", bbox_inches="tight")
    fig_heat.savefig(_save_dir / f"{_track}_variability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


for _track in REPRO_TRACKS:
    _plot_variability_heatmap(_track)
"""
    c14_new = '''from collections import Counter

from matplotlib.ticker import FuncFormatter

TOP_AB = 5  # top antibodies per descriptor to report

_DESCRIPTOR_PREFIXES = (
    "sequence_motives_",
    "surface_",
    "general_",
    "core_",
)


def _strip_descriptor_prefix(name: str) -> str:
    """Same as variability plot cell: drop surface_/general_/… for shorter labels."""
    s = str(name)
    for p in _DESCRIPTOR_PREFIXES:
        if s.startswith(p):
            return s[len(p) :]
    return s


def _build_run_dev_and_ab_rcv(
    track: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    run_dev_data: dict[str, pd.DataFrame] = {}
    ab_rcv_data: dict[str, pd.DataFrame] = {}
    flagged_by_ds = descriptors_exceeding_median_rcv_by_track[track]
    flagged_datasets = sorted(ds for ds, fr in flagged_by_ds.items() if not fr.empty)

    for ds in flagged_datasets:
        flagged_descs = list(flagged_by_ds[ds]["descriptor"])
        loaded = _repeat_concat_ok(by_track[track][ds], dataset=ds, min_runs=2, track=track)
        if loaded is None:
            continue
        long_c, num_cols, n_runs = loaded
        flagged_descs = [d for d in flagged_descs if d in num_cols]
        if not flagged_descs:
            continue

        dev_rows: dict[str, list] = {}
        ab_rows: list[dict] = []

        for col in flagged_descs:
            pivot = long_c.pivot_table(index="name", columns="_run", values=col, aggfunc="first")
            pivot = pivot.apply(pd.to_numeric, errors="coerce")
            run_means = pivot.mean(axis=0)
            med = run_means.median()
            mad = (run_means - med).abs().median() + EPS
            z = (run_means - med) / mad
            dev_rows[col] = z.reindex(range(n_runs)).tolist()
            m_ab = pivot.mean(axis=1)
            s_ab = pivot.std(axis=1, ddof=1)
            rcv_ab = s_ab / m_ab.abs().clip(lower=EPS)
            for name, rcv in rcv_ab.items():
                ab_rows.append({"name": str(name), "descriptor": col, "rcv": float(rcv)})

        _dev_df = pd.DataFrame(dev_rows, index=[f"run {r+1}" for r in range(n_runs)]).T
        _dev_df.index.name = "descriptor"
        run_dev_data[ds] = _dev_df
        ab_rcv_data[ds] = pd.DataFrame(ab_rows)

    return run_dev_data, ab_rcv_data, flagged_datasets


def _print_run_dev_summary(
    track: str,
    run_dev_data: dict[str, pd.DataFrame],
    ab_rcv_data: dict[str, pd.DataFrame],
    flagged_datasets: list[str],
) -> None:
    for ds in flagged_datasets:
        dev_df = run_dev_data.get(ds)
        ab_df = ab_rcv_data.get(ds)
        if dev_df is None:
            continue
        print(f"══ {track} / {ds} ══")
        outlier_run_counts = Counter(dev_df.abs().idxmax(axis=1))
        print("  Most-extreme run (by |z-score| of run mean, per descriptor):")
        for run_lbl, cnt in sorted(outlier_run_counts.items(), key=lambda x: -x[1]):
            print(f"    {run_lbl}: most extreme in {cnt}/{len(dev_df)} flagged descriptor(s)")
        print(f"  Top-{TOP_AB} antibodies per flagged descriptor:")
        for desc in dev_df.index:
            sub = ab_df[ab_df["descriptor"] == desc].nlargest(TOP_AB, "rcv")
            parts = ", ".join(f"{r['name']}(rcv={r['rcv']:.2f})" for _, r in sub.iterrows())
            print(f"    {_strip_descriptor_prefix(desc)}: {parts}")
        print()


_RIGHT_COL_DATASETS = frozenset({"jetha2019homology_RT", "pdgf38"})


def _plot_runs_and_antibodies_figure(
    track: str,
    run_dev_data: dict[str, pd.DataFrame],
    ab_rcv_data: dict[str, pd.DataFrame],
    flagged_datasets: list[str],
) -> None:
    _plot_specs = []
    for ds in flagged_datasets:
        dev_df = run_dev_data.get(ds)
        ab_df = ab_rcv_data.get(ds)
        if dev_df is None or ab_df is None:
            continue
        for desc in dev_df.index:
            sub = ab_df.loc[ab_df["descriptor"] == desc, "rcv"].dropna().values
            if len(sub) > 0:
                _plot_specs.append((ds, desc, sub))

    def _axis_specs_for_datasets(dataset_list: list[str]) -> list:
        axis_specs = []
        for ds in dataset_list:
            ds_specs = [spec for spec in _plot_specs if spec[0] == ds]
            if not ds_specs:
                continue
            if axis_specs:
                axis_specs.append(None)
            axis_specs.extend(ds_specs)
        return axis_specs

    def _last_axis_row_for_ds(axis_specs: list) -> dict[str, int]:
        return {spec[0]: idx for idx, spec in enumerate(axis_specs) if spec is not None}

    def _draw_rcv_violin(ax, spec, *, last_row_for_ds: dict[str, int], row_idx: int, prev_ds):
        if spec is None:
            ax.set_visible(False)
            return prev_ds
        ds, desc, sub = spec
        ab_df = ab_rcv_data[ds]
        ds_short = SHORT_DATASET[ds]
        parts = ax.violinplot(
            [sub], positions=[0], vert=False, widths=0.7,
            showmeans=False, showmedians=True, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("ghostwhite")
            body.set_alpha(1.0)
            body.set_edgecolor("#888888")
        if parts.get("cmedians") is not None:
            parts["cmedians"].set_color("#333333")
            parts["cmedians"].set_linewidth(0.9)
        top_names = ab_df[ab_df["descriptor"] == desc].nlargest(3, "rcv")[["name", "rcv"]]
        label_offsets = np.linspace(-0.42, 0.42, len(top_names))
        label_x_offset = 0.015 * max(ax.get_xlim()[1], MEDIAN_RCV_THRESHOLD)
        for label_offset, (_, r_row) in zip(label_offsets, top_names.iterrows()):
            ax.text(
                r_row["rcv"] + label_x_offset, label_offset, str(r_row["name"]),
                fontsize=9, fontweight="bold", va="center", color="maroon",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.3),
                clip_on=False,
            )
        ax.set_ylim(-0.85, 0.85)
        ax.set_yticks([0])
        ax.set_yticklabels([_strip_descriptor_prefix(desc)], fontsize=9)
        ax.axvline(MEDIAN_RCV_THRESHOLD, color="maroon", lw=2.2, ls="--")
        ax.set_xlabel("per-antibody RCV (%)" if row_idx == last_row_for_ds.get(ds) else "", fontsize=9)
        ax.set_title(ds_short if ds != prev_ds else "", fontsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x * 100:.0f}%"))
        ax.tick_params(axis="both", colors="black", labelsize=9)
        return ds

    _left_datasets = [ds for ds in flagged_datasets if ds not in _RIGHT_COL_DATASETS]
    _right_datasets = [ds for ds in flagged_datasets if ds in _RIGHT_COL_DATASETS]
    _left_axis_specs = _axis_specs_for_datasets(_left_datasets)
    _right_axis_specs = _axis_specs_for_datasets(_right_datasets)
    n_rows = max(len(_left_axis_specs), len(_right_axis_specs), 1)
    _n_plot_rows = max(
        sum(1 for s in _left_axis_specs if s is not None),
        sum(1 for s in _right_axis_specs if s is not None),
        1,
    )
    _n_spacer_rows = max(
        len(_left_axis_specs) - sum(1 for s in _left_axis_specs if s is not None),
        len(_right_axis_specs) - sum(1 for s in _right_axis_specs if s is not None),
        0,
    )
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(22, max(3.2, 1.08 * _n_plot_rows + 0.55 * _n_spacer_rows)),
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
        squeeze=False,
    )
    _last_row_left = _last_axis_row_for_ds(_left_axis_specs)
    _last_row_right = _last_axis_row_for_ds(_right_axis_specs)
    _prev_left = _prev_right = None
    for row in range(n_rows):
        if row < len(_left_axis_specs):
            _prev_left = _draw_rcv_violin(
                axes[row, 0], _left_axis_specs[row],
                last_row_for_ds=_last_row_left, row_idx=row, prev_ds=_prev_left,
            )
        else:
            axes[row, 0].set_visible(False)
        if row < len(_right_axis_specs):
            _prev_right = _draw_rcv_violin(
                axes[row, 1], _right_axis_specs[row],
                last_row_for_ds=_last_row_right, row_idx=row, prev_ds=_prev_right,
            )
        else:
            axes[row, 1].set_visible(False)
    plt.tight_layout(h_pad=0.75)
    plt.savefig(_save_dir / f"{track}_variability_runs_and_antibodies.png", dpi=150, bbox_inches="tight")
    plt.savefig(_save_dir / f"{track}_variability_runs_and_antibodies.pdf", bbox_inches="tight")
    plt.show()


for _track in REPRO_TRACKS:
    _rd, _ab, _fds = _build_run_dev_and_ab_rcv(_track)
    _print_run_dev_summary(_track, _rd, _ab, _fds)
    if _fds:
        _plot_runs_and_antibodies_figure(_track, _rd, _ab, _fds)
'''
    set_cell_source(nb["cells"][12], c12)
    set_cell_source(nb["cells"][14], c14_new)
    NB_PATH.write_text(json.dumps(nb, indent=2) + "\n")
    print("patched cells 12 and 14 only:", NB_PATH)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--only-12-14":
        patch_cells_12_14_only()
    else:
        main()
