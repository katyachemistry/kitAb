#!/usr/bin/env python3
"""Aggregate fold-level result JSONs from batch parallel runs into CSVs, times.csv, and plots."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from automl.pipeline_defaults import DEFAULT_FEATURES_FRAC
from automl.run_fold_pipeline_config import oof_sidecar_path

_REPO_ROOT = Path(__file__).resolve().parents[2]

TIMES_CSV_COLS: tuple[str, ...] = (
    "dataset_yaml_key",
    "Dataset_stem",
    "Developability_source",
    "Target_col",
    "total_pipeline_time_s",
)

AGG_BASE_CSV_COLS: tuple[str, ...] = (
    "Track",
    "Target-Selector-Model",
    "Target_col",
    "Dataset_stem",
    "Developability_source",
    "Spearman",
    "Pearson",
    "R2",
    "Spearman_pooled_oof",
    "Pearson_pooled_oof",
    "R2_pooled_oof",
    "n_oof",
    "n_folds_present",
    "Prediction_std_mean",
    "Spearman_relative_error_across_folds",
    "Pearson_relative_error_across_folds",
    "R2_relative_error_across_folds",
    "Prediction_std_relative_error_across_folds",
    "Pipeline_time_sum_s",
    "Pipeline_time_std_s",
    "selected_features_by_fold",
)


def _nanmean(vals: list[float]) -> float | None:
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(np.mean(a))


def _strict_mean(vals: list[float], n_expected: int) -> float | None:
    if n_expected <= 0 or len(vals) != n_expected:
        return None
    a = np.asarray(vals, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        return None
    return float(np.mean(a))


def _sort_fold_key(k: str) -> tuple[int, int | str]:
    if k.startswith("fold_") and k != "fold_unknown":
        suf = k[5:]
        try:
            return (0, int(suf))
        except ValueError:
            pass
    if k == "fold_unknown":
        return (2, 0)
    return (1, k)


def _fold_eval_lists(
    fm: dict[str, dict],
) -> tuple[list[float], list[float], list[float], list[float]]:
    sp_list: list[float] = []
    pe_list: list[float] = []
    r2_list: list[float] = []
    sd_list: list[float] = []
    for fk in sorted(fm.keys(), key=_sort_fold_key):
        d = fm.get(fk) or {}
        for key, bucket in (
            ("spearman", sp_list),
            ("pearson", pe_list),
            ("r2", r2_list),
            ("prediction_std_mean", sd_list),
        ):
            v = d.get(key)
            if v is None:
                continue
            try:
                xf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(xf):
                bucket.append(xf)
    return sp_list, pe_list, r2_list, sd_list


def _optional_float_cell(x: object) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return float(xf) if math.isfinite(xf) else None


def _relative_error_across_folds(vals: list[float]) -> float | None:
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return None
    m = float(np.mean(a))
    if abs(m) < 1e-15:
        return None
    s = float(np.std(a, ddof=1))
    return float(s / abs(m))


def _fold_eval_one_model(
    evaluation: dict | None,
    eval_model: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    if not evaluation or not isinstance(evaluation, dict):
        return None, None, None, None
    v = evaluation.get(eval_model)
    if not isinstance(v, dict) or v.get("error"):
        return None, None, None, None
    spearman = _optional_float_cell(v.get("spearman_rho"))
    pearson = _optional_float_cell(v.get("pearson_r"))
    r2 = _optional_float_cell(v.get("r2"))
    uncertainty = _optional_float_cell(v.get("prediction_std_mean"))
    return spearman, pearson, r2, uncertainty


def _fold_key_from_json(data: dict) -> str:
    raw = data.get("fold_index")
    if raw is None:
        return "fold_unknown"
    try:
        return f"fold_{int(raw) + 1}"
    except (TypeError, ValueError):
        return f"fold_{raw}"


def _get_reported_features(data: dict, eval_model: str | None = None) -> list[str] | None:
    evaluation = data.get("evaluation") or {}
    if eval_model:
        mr = evaluation.get(eval_model)
        if isinstance(mr, dict) and not mr.get("error"):
            used = mr.get("eval_features_used")
            if isinstance(used, list):
                return [str(f) for f in used]
    for model_result in evaluation.values():
        if not isinstance(model_result, dict) or model_result.get("error"):
            continue
        used = model_result.get("eval_features_used")
        if isinstance(used, list):
            return [str(f) for f in used]
    feats = data.get("selected_features")
    if isinstance(feats, list):
        return [str(f) for f in feats]
    return None


def _safe_filename_segment(s: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_")
    return t or "dataset"


def _filter_json_paths_to_dataset_yaml_key(
    paths: list[Path],
    batch_root: Path,
    dataset_yaml_key: str,
) -> list[Path]:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dataset_yaml_key)).strip("_") or "x"
    try:
        sub = (batch_root.resolve() / slug).resolve()
    except OSError:
        return []
    out: list[Path] = []
    for p in paths:
        try:
            pr = p.resolve()
        except OSError:
            continue
        try:
            pr.relative_to(sub)
        except ValueError:
            continue
        except OSError:
            continue
        out.append(p)
    return out


_HEX_HASH_RE = re.compile(r"[0-9a-f]{8,}")


def _developability_label_from_run_dir(
    run_dir: str, dev_paths: list[str] | None = None
) -> str:
    name = Path(run_dir).name.replace(".csv", "")
    if "__" in name:
        label = name.split("__", 1)[-1]
    else:
        label = name or "unknown"

    if dev_paths and _HEX_HASH_RE.search(label):
        p = Path(str(dev_paths[0]))
        parent = p.parent.name
        if parent and parent not in (".", ""):
            stem = p.stem
            new_label = f"{parent}_{stem}"
            seed_match = re.search(r"__(rs\d+)$", label)
            if seed_match:
                new_label += f"__{seed_match.group(1)}"
            return new_label

    return label


def _yaml_key_to_developability_label(manifest: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for block in manifest.get("datasets") or []:
        yk = block.get("dataset_yaml_key")
        rd = block.get("run_dir")
        if not yk:
            continue
        dev_paths: list[str] = block.get("developability_results_paths") or []
        out[str(yk)] = _developability_label_from_run_dir(str(rd or ""), dev_paths or None)
    return out


def _target_col_for_plot(row: pd.Series) -> str:
    tc = row.get("Target_col")
    if tc is not None and not (isinstance(tc, float) and math.isnan(tc)):
        s = str(tc).strip()
        if s:
            return s
    name = str(row.get("Target-Selector-Model", ""))
    parts5 = name.rsplit("-", 4)
    if len(parts5) == 5 and parts5[-1].startswith("frac"):
        return parts5[0]
    parts4 = name.rsplit("-", 3)
    if len(parts4) == 4 and parts4[-1].startswith("frac"):
        return parts4[0]
    return "unknown"


def _selector_model_frac_label(row: pd.Series) -> str:
    t = str(row.get("Target_col", "")).strip()
    full = str(row.get("Target-Selector-Model", ""))
    if t and full.startswith(t + "-"):
        base = full[len(t) + 1 :]
    else:
        base = full
    tr = str(row.get("Track", "")).strip()
    if tr:
        return f"{tr}\n{base}"
    return base


def _fmt_plot_cell(x: object) -> str:
    try:
        if x is None or pd.isna(x):
            return "n/a"
    except TypeError:
        pass
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(xf):
        return "n/a"
    return f"{xf:.4g}"


def _per_fold_metric(row: pd.Series, metric: str) -> list[float]:
    suf = f"_{metric}"
    cols = [
        c
        for c in row.index
        if str(c).startswith("fold_") and str(c).endswith(suf)
    ]

    def _base(c: object) -> str:
        return str(c)[: -len(suf)]

    out: list[float] = []
    for c in sorted(cols, key=lambda x: _sort_fold_key(_base(x))):
        v = row[c]
        try:
            if pd.isna(v):
                continue
        except TypeError:
            if v is None:
                continue
        try:
            xf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf):
            out.append(xf)
    return out


def _best_metric_slot(
    grp: pd.DataFrame, metric_col: str
) -> tuple[pd.Series, str] | None:
    g = grp.dropna(subset=[metric_col])
    if len(g) == 0:
        return None
    best = g.loc[g[metric_col].idxmax()]
    return best, _selector_model_frac_label(best)


def _suptitle_line2_single_dataset(slots: list[tuple[pd.Series, str, str]]) -> str:
    has_sp = any(s[2] == "spearman" for s in slots)
    has_pe = any(s[2] == "pearson" for s in slots)
    if has_sp and has_pe:
        return (
            "(left: best mean Spearman, ρ per fold only — "
            "right: best mean Pearson, r per fold only)"
        )
    if has_sp:
        return "(best mean Spearman: ρ per fold only)"
    return "(best mean Pearson: r per fold only)"


def _suptitle_line2_combined(slots: list[tuple[pd.Series, str, str]]) -> str:
    has_sp = any(s[2] == "spearman" for s in slots)
    has_pe = any(s[2] == "pearson" for s in slots)
    if has_sp and has_pe:
        return (
            "(all sources: Spearman-best first, then Pearson-best; "
            "dashed line between blocks; one metric per column)"
        )
    if has_sp:
        return "(Spearman-best per source; ρ per fold only)"
    return "(Pearson-best per source; r per fold only)"


def _draw_best_combo_figure(
    slots: list[tuple[pd.Series, str, str]],
    *,
    x_axis_label: str,
    suptitle_line1: str,
    suptitle_line2: str,
    out_png: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.transforms import blended_transform_factory
    except ImportError:
        print("matplotlib not installed; skipping fold scatter plots.", file=sys.stderr)
        return

    if not slots:
        return

    n = len(slots)
    fig_w = max(7.0, 2.9 * n + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 6.6))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(-0.55, max(float(n) - 0.45, 0.45))
    ax.set_xticks(list(range(n)))

    legend_sp_done = False
    legend_pe_done = False

    for i, (row, _, metric) in enumerate(slots):
        if metric == "spearman":
            vals = _per_fold_metric(row, "spearman")
            if vals:
                ax.scatter(
                    [float(i)] * len(vals),
                    vals,
                    c="C0",
                    marker="o",
                    s=44,
                    alpha=0.88,
                    zorder=5,
                    label="Spearman (fold)" if not legend_sp_done else None,
                )
                legend_sp_done = True
            mean_sp = row.get("Spearman")
            cap = f"mean ρ = {_fmt_plot_cell(mean_sp)}"
        else:
            vals = _per_fold_metric(row, "pearson")
            if vals:
                ax.scatter(
                    [float(i)] * len(vals),
                    vals,
                    c="C1",
                    marker="^",
                    s=46,
                    alpha=0.88,
                    zorder=5,
                    label="Pearson (fold)" if not legend_pe_done else None,
                )
                legend_pe_done = True
            mean_pe = row.get("Pearson")
            cap = f"mean r = {_fmt_plot_cell(mean_pe)}"
        ax.annotate(
            cap,
            xy=(float(i), 1.0),
            xycoords=trans,
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            linespacing=1.15,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.82",
                linewidth=0.8,
                alpha=0.94,
            ),
            annotation_clip=False,
            zorder=4,
        )

    first_pe = next((i for i, s in enumerate(slots) if s[2] == "pearson"), None)
    if first_pe is not None and first_pe > 0:
        ax.axvline(float(first_pe) - 0.5, color="0.82", ls="--", lw=1.0, zorder=1)

    ax.set_xticklabels([lbl for _, lbl, _ in slots], rotation=18, ha="right")
    ax.set_ylabel("Correlation (per fold): ρ or r")
    ax.set_xlabel(x_axis_label)
    ax.axhline(0.0, color="gray", lw=0.65, zorder=0)
    ax.grid(True, axis="y", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.92)
    fig.suptitle(
        f"{suptitle_line1}\n{suptitle_line2}",
        fontsize=11,
        y=0.995,
        va="top",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}", file=sys.stderr)


def _plot_best_combo_fold_scatters(
    df: pd.DataFrame,
    *,
    dataset_key: str,
    out_plot_dir: Path,
) -> None:
    pair_cols = [
        c
        for c in df.columns
        if str(c).startswith("fold_") and str(c).endswith("_spearman")
    ]
    if not pair_cols:
        print(f"No fold_*_spearman columns; skip plots for {dataset_key!r}", file=sys.stderr)
        return

    work = df.copy()
    if "Target_col" not in work.columns:
        work["Target_col"] = work.apply(_target_col_for_plot, axis=1)

    out_ds = out_plot_dir / _safe_filename_segment(dataset_key)
    out_ds.mkdir(parents=True, exist_ok=True)

    for target_col, grp in work.groupby("Target_col", sort=False):
        slots: list[tuple[pd.Series, str, str]] = []
        bs = _best_metric_slot(grp, "Spearman")
        if bs:
            slots.append((bs[0], bs[1], "spearman"))
        bp = _best_metric_slot(grp, "Pearson")
        if bp:
            slots.append((bp[0], bp[1], "pearson"))
        if not slots:
            continue
        out_png = out_ds / f"{_safe_filename_segment(str(target_col))}_best_mean_combo_fold_scatter.png"
        _draw_best_combo_figure(
            slots,
            x_axis_label="Combination (selector–model–feature frac)",
            suptitle_line1=f"{dataset_key} — {target_col}",
            suptitle_line2=_suptitle_line2_single_dataset(slots),
            out_png=out_png,
        )


def _plot_combined_by_experimental_target(
    tables: list[tuple[str, str, str, pd.DataFrame]],
    *,
    out_plot_dir: Path,
) -> None:
    if len(tables) < 2:
        return

    full = pd.concat([t[3] for t in tables], ignore_index=True)
    if "Dataset_stem" not in full.columns or "Developability_source" not in full.columns:
        return

    combined_dir = out_plot_dir / "combined"
    for (stem, target_col), grp in full.groupby(["Dataset_stem", "Target_col"], sort=False):
        if grp["Developability_source"].nunique() < 2:
            continue

        sources_sorted = sorted(grp["Developability_source"].astype(str).unique())
        slots: list[tuple[pd.Series, str, str]] = []
        for dev in sources_sorted:
            sub = grp[grp["Developability_source"] == dev]
            bs = _best_metric_slot(sub, "Spearman")
            if bs:
                slots.append((bs[0], f"{dev}\n{bs[1]}", "spearman"))
        for dev in sources_sorted:
            sub = grp[grp["Developability_source"] == dev]
            bp = _best_metric_slot(sub, "Pearson")
            if bp:
                slots.append((bp[0], f"{dev}\n{bp[1]}", "pearson"))

        if not slots:
            continue

        fn = f"{_safe_filename_segment(str(stem))}__{_safe_filename_segment(str(target_col))}_best_mean_combo_fold_scatter.png"
        out_png = combined_dir / fn
        _draw_best_combo_figure(
            slots,
            x_axis_label="Developability source + combination (selector–model–feature frac)",
            suptitle_line1=f"{stem} — {target_col}",
            suptitle_line2=_suptitle_line2_combined(slots),
            out_png=out_png,
        )


def _resolve(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _json_paths_from_master(master: Path) -> list[Path]:
    paths: list[Path] = []
    for line in master.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            print(
                f"Skipping TSV line (need ≥9 columns for output_json, got {len(parts)}): {line[:100]!r}",
                file=sys.stderr,
            )
            continue
        p = Path(parts[8])
        paths.append(p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve())
    return paths


def _remap_json_paths_batch_root(
    paths: list[Path],
    *,
    manifest_batch_root: Path,
    effective_batch_root: Path,
) -> list[Path]:
    man_r = manifest_batch_root.resolve()
    eff_r = effective_batch_root.resolve()
    if man_r == eff_r:
        return paths
    out: list[Path] = []
    for p in paths:
        pr = p.resolve()
        try:
            rel = pr.relative_to(man_r)
            out.append((eff_r / rel).resolve())
        except ValueError:
            out.append(p)
    return out


def _ingest_oof_frame(
    acc: dict,
    df: pd.DataFrame,
    *,
    ds_key: str,
    track: str,
    target: str,
    sel: str,
    mod: str,
    eval_frac: float,
    fold_index: int | None,
) -> None:
    if df is None or len(df) == 0 or "eval_model" not in df.columns:
        return
    for em, sub in df.groupby("eval_model", sort=False):
        if str(em).strip().lower() == "gpr":
            continue
        key = (ds_key, track, target, sel, mod, str(em), float(eval_frac))
        acc[key]["oof_y"].extend(pd.to_numeric(sub["y"], errors="coerce").tolist())
        acc[key]["oof_yhat"].extend(pd.to_numeric(sub["yhat"], errors="coerce").tolist())
        if fold_index is not None:
            acc[key]["oof_folds"].add(int(fold_index))
        elif "fold_index" in sub.columns:
            acc[key]["oof_folds"].update(
                int(x) for x in sub["fold_index"].dropna().unique().tolist()
            )


def _pooled_from_acc_lists(y_list: list, yhat_list: list, n_folds: int) -> dict:
    from analysis.oof_predictions import pooled_metrics_from_oof

    df = pd.DataFrame({"y": y_list, "yhat": yhat_list})
    out = pooled_metrics_from_oof(df)
    out["n_folds_present"] = int(n_folds)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate batch fold JSONs into one CSV per YAML dataset block.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="batch_manifest.json (reads batch_result_root and master_jobs_tsv).",
    )
    p.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help=(
            "Batch directory (use with --master-tsv if manifest missing). "
            "With --manifest, overrides manifest ``batch_result_root`` when this run "
            "was copied or renamed (e.g. ``…_tracks_1`` vs manifest ``…_tracks``)."
        ),
    )
    p.add_argument(
        "--master-tsv",
        type=Path,
        default=None,
        help="parallel_jobs_master.tsv (default: <batch-root>/parallel_jobs_master.tsv).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregated_*.csv (default: batch root).",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for best-combo fold scatter PNGs (default: <output-dir>/plots).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write best-combination fold scatter plots.",
    )
    p.add_argument(
        "--only-dataset-yaml-key",
        default=None,
        help=(
            "Only aggregate JSONs under this dataset's subdirectory of the batch root "
            "(same slug as prepare_parallel_from_config). Omit for the full batch."
        ),
    )
    p.add_argument(
        "--oof-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of retro *.oof.parquet files. When set, pooled OOF "
            "Spearman/Pearson/R2 are computed from these; otherwise sidecars next "
            "to each result JSON are used if present."
        ),
    )
    args = p.parse_args()

    batch_root: Path
    master_path: Path | None = None
    yaml_to_dev: dict[str, str] = {}
    manifest_batch_root: Path | None = None

    if args.manifest:
        mp = _resolve(args.manifest)
        if not mp.is_file():
            print(f"Manifest not found: {mp}", file=sys.stderr)
            sys.exit(1)
        man = json.loads(mp.read_text())
        yaml_to_dev = _yaml_key_to_developability_label(man)
        br = man.get("batch_result_root")
        if not br:
            print("Manifest missing batch_result_root", file=sys.stderr)
            sys.exit(1)
        manifest_batch_root = _resolve(Path(br))
        batch_root = manifest_batch_root
        mt = man.get("master_jobs_tsv")
        if not mt:
            print("Manifest missing master_jobs_tsv", file=sys.stderr)
            sys.exit(1)
        master_path = _resolve(Path(mt))
        if args.batch_root is not None:
            batch_root = _resolve(args.batch_root)
            if args.master_tsv is not None:
                master_path = _resolve(args.master_tsv)
            else:
                master_path = batch_root / "parallel_jobs_master.tsv"
    elif args.batch_root:
        batch_root = _resolve(args.batch_root)
        master_path = (
            _resolve(args.master_tsv)
            if args.master_tsv
            else (batch_root / "parallel_jobs_master.tsv")
        )
    else:
        print("Provide --manifest or --batch-root.", file=sys.stderr)
        sys.exit(1)

    if not batch_root.is_dir():
        print(f"Not a directory: {batch_root}", file=sys.stderr)
        sys.exit(1)

    out_dir = _resolve(args.output_dir) if args.output_dir else batch_root
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = _resolve(args.plot_dir) if args.plot_dir else (out_dir / "plots")
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    globbed_json_paths = sorted(
        x
        for x in batch_root.rglob("*.json")
        if x.is_file() and x.name != "batch_manifest.json"
    )
    if master_path is not None and master_path.is_file():
        master_json_paths = _json_paths_from_master(master_path)
        if (
            manifest_batch_root is not None
            and args.batch_root is not None
            and manifest_batch_root.resolve() != batch_root.resolve()
        ):
            master_json_paths = _remap_json_paths_batch_root(
                master_json_paths,
                manifest_batch_root=manifest_batch_root,
                effective_batch_root=batch_root,
            )
        json_paths = sorted({*master_json_paths, *globbed_json_paths})
    else:
        if master_path is not None:
            print(
                f"No master TSV at {master_path}; globbing *.json under {batch_root}",
                file=sys.stderr,
            )
        json_paths = globbed_json_paths

    only_key = (
        str(args.only_dataset_yaml_key).strip()
        if args.only_dataset_yaml_key is not None
        else ""
    )
    if only_key:
        n_before = len(json_paths)
        json_paths = _filter_json_paths_to_dataset_yaml_key(
            json_paths, batch_root, only_key
        )
        print(
            f"Restricting to dataset_yaml_key={only_key!r}: "
            f"{len(json_paths)} JSON path(s) (was {n_before})",
            file=sys.stderr,
        )
        if not json_paths:
            print(
                f"No JSON paths under batch root for {only_key!r} after filter; nothing to do.",
                file=sys.stderr,
            )
            sys.exit(1)

    stem_by_yaml: dict[str, str] = {}

    # (dataset_yaml_key, target_col) -> sum of pipeline_time_seconds (once per result JSON)
    dataset_target_pipeline_s: dict[tuple[str, str], float] = defaultdict(float)

    # (dataset_yaml_key, track, target_col, selector_name, model_type, eval_model, eval_frac) -> aggregates
    acc: dict[tuple[str, str, str, str, str, str, float], dict] = defaultdict(
        lambda: {
            "fold_metrics": {},  # fold_key -> spearman, pearson, r2, prediction_std_mean (float|None)
            "features_by_fold": {},
            "pipeline_times_by_fold": {},  # fold_key -> pipeline_time_seconds (float)
            "oof_y": [],
            "oof_yhat": [],
            "oof_folds": set(),
        }
    )

    for jp in json_paths:
        if not jp.is_file():
            print(f"Missing result JSON: {jp}", file=sys.stderr)
            continue
        try:
            data = json.loads(jp.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skip {jp}: {e}", file=sys.stderr)
            continue

        ds_key = str(
            data.get("dataset_yaml_key") or data.get("dataset_stem") or "unknown"
        )
        if ds_key not in stem_by_yaml:
            stem_by_yaml[ds_key] = str(data.get("dataset_stem") or "unknown")
        track = str(
            data.get("pipeline_track_name")
            or data.get("pipeline-track-name")
            or ""
        ).strip()
        target = str(data.get("target_col", "unknown"))
        sel = str(data.get("selector_name", "unknown"))
        mod = str(data.get("model_type", "unknown"))
        ev_raw = data.get("eval_features_frac")
        if ev_raw is None:
            ev_raw = data.get("features_frac", DEFAULT_FEATURES_FRAC)
        try:
            eval_frac = float(ev_raw)
        except (TypeError, ValueError):
            eval_frac = DEFAULT_FEATURES_FRAC

        fk = _fold_key_from_json(data)
        evaluation = data.get("evaluation")
        if isinstance(evaluation, dict) and evaluation:
            eval_models = [
                str(k)
                for k, v in evaluation.items()
                if isinstance(k, str) and isinstance(v, dict)
            ]
            if not eval_models:
                eval_models = ["none"]
        else:
            eval_models = ["none"]

        pt_raw = data.get("pipeline_time_seconds")
        pt_val: float | None = None
        if pt_raw is not None:
            try:
                pt_f = float(pt_raw)
                if math.isfinite(pt_f):
                    pt_val = pt_f
            except (TypeError, ValueError):
                pass

        if pt_val is not None:
            dataset_target_pipeline_s[(ds_key, target)] += pt_val

        for eval_model in eval_models:
            if eval_model == "none" or str(eval_model).strip().lower() == "gpr":
                continue
            ev = evaluation if isinstance(evaluation, dict) else {}
            sp, pe, r2, ps = _fold_eval_one_model(ev, eval_model)
            key = (ds_key, track, target, sel, mod, eval_model, eval_frac)
            acc[key]["fold_metrics"][fk] = {
                "spearman": sp,
                "pearson": pe,
                "r2": r2,
                "prediction_std_mean": ps,
            }

            feats = _get_reported_features(data, eval_model)
            if feats is not None:
                acc[key]["features_by_fold"][fk] = feats

            if pt_val is not None:
                acc[key]["pipeline_times_by_fold"][fk] = pt_val

        sidecar = oof_sidecar_path(jp)
        if sidecar.is_file():
            try:
                oof_df = pd.read_parquet(sidecar)
            except Exception:
                oof_df = None
            if oof_df is not None and len(oof_df):
                _ingest_oof_frame(
                    acc,
                    oof_df,
                    ds_key=ds_key,
                    track=track,
                    target=target,
                    sel=sel,
                    mod=mod,
                    eval_frac=eval_frac,
                    fold_index=data.get("fold_index"),
                )

    oof_dir = getattr(args, "oof_dir", None)
    if oof_dir is not None:
        oof_root = Path(oof_dir)
        if oof_root.is_dir():
            for pq in oof_root.rglob("*.oof.parquet"):
                try:
                    oof_df = pd.read_parquet(pq)
                except Exception:
                    continue
                if oof_df is None or len(oof_df) == 0:
                    continue
                ds_k = str(oof_df["dataset_yaml_key"].iloc[0]) if "dataset_yaml_key" in oof_df.columns else "unknown"
                tr_k = str(oof_df["pipeline_track_name"].iloc[0]) if "pipeline_track_name" in oof_df.columns else ""
                tgt_k = str(oof_df["target_col"].iloc[0]) if "target_col" in oof_df.columns else "unknown"
                sel_k = str(oof_df["selector_name"].iloc[0]) if "selector_name" in oof_df.columns else "unknown"
                mod_k = str(oof_df["model_type"].iloc[0]) if "model_type" in oof_df.columns else "unknown"
                try:
                    frac_k = float(oof_df["eval_features_frac"].iloc[0])
                except (TypeError, ValueError, KeyError, IndexError):
                    frac_k = float(DEFAULT_FEATURES_FRAC)
                _ingest_oof_frame(
                    acc,
                    oof_df,
                    ds_key=ds_k,
                    track=tr_k,
                    target=tgt_k,
                    sel=sel_k,
                    mod=mod_k,
                    eval_frac=frac_k,
                    fold_index=None,
                )

    by_ds: dict[str, list[dict]] = defaultdict(list)
    fold_keys_by_ds: dict[str, set[str]] = defaultdict(set)

    # Pre-pass: complete fold-key set per dataset (for _strict_mean).
    for (ds_key, _track, _tgt, _sel, _mod, _em, _ef), bucket in acc.items():
        fold_keys_by_ds[ds_key].update(bucket["fold_metrics"].keys())

    for (ds_key, track, target, sel, mod, eval_model, eval_frac), bucket in acc.items():
        fs = f"frac{int(round(float(eval_frac) * 100)):03d}"
        row_name = f"{target}-{sel}-{mod}-{eval_model}-{fs}"
        fbf = bucket["features_by_fold"]
        fm = bucket["fold_metrics"]
        n_expected = len(fold_keys_by_ds[ds_key])
        sp_list, pe_list, r2_list, sd_list = _fold_eval_lists(fm)
        sp_mean = _strict_mean(sp_list, n_expected)
        pe_mean = _strict_mean(pe_list, n_expected)
        r2_mean = _strict_mean(r2_list, n_expected)
        pt_fold_vals = [
            v for v in bucket.get("pipeline_times_by_fold", {}).values()
            if isinstance(v, float) and math.isfinite(v)
        ]
        pt_sum: float | None = (
            float(np.sum(pt_fold_vals)) if pt_fold_vals else None
        )
        pt_std: float | None = (
            float(np.std(pt_fold_vals, ddof=1)) if len(pt_fold_vals) >= 2 else None
        )
        pooled = _pooled_from_acc_lists(
            bucket.get("oof_y") or [],
            bucket.get("oof_yhat") or [],
            len(bucket.get("oof_folds") or set()),
        )
        by_ds[ds_key].append(
            {
                "Track": track,
                "Target-Selector-Model": row_name,
                "Target_col": target,
                "Dataset_stem": stem_by_yaml.get(ds_key, "unknown"),
                "Developability_source": yaml_to_dev.get(ds_key, ds_key),
                "Spearman": sp_mean,
                "Pearson": pe_mean,
                "R2": r2_mean,
                "Spearman_pooled_oof": pooled["Spearman_pooled_oof"],
                "Pearson_pooled_oof": pooled["Pearson_pooled_oof"],
                "R2_pooled_oof": pooled["R2_pooled_oof"],
                "n_oof": pooled["n_oof"],
                "n_folds_present": pooled["n_folds_present"],
                "Prediction_std_mean": _nanmean(sd_list),
                "Spearman_relative_error_across_folds": (
                    _relative_error_across_folds(sp_list) if sp_mean is not None else None
                ),
                "Pearson_relative_error_across_folds": (
                    _relative_error_across_folds(pe_list) if pe_mean is not None else None
                ),
                "R2_relative_error_across_folds": (
                    _relative_error_across_folds(r2_list) if r2_mean is not None else None
                ),
                "Prediction_std_relative_error_across_folds": _relative_error_across_folds(
                    sd_list
                ),
                "Pipeline_time_sum_s": pt_sum,
                "Pipeline_time_std_s": pt_std,
                "selected_features_by_fold": (
                    json.dumps(
                        {
                            k: fbf[k]
                            for k in sorted(fbf.keys(), key=_sort_fold_key)
                        },
                        ensure_ascii=False,
                    )
                    if fbf
                    else "{}"
                ),
                "_fold_metrics": fm,
                "_sort": (track, target, sel, mod, eval_model, fs),
            }
        )

    if not by_ds:
        print("No result JSONs aggregated.", file=sys.stderr)
        sys.exit(1)

    tables_for_combine: list[tuple[str, str, str, pd.DataFrame]] = []

    for ds_key, rows in sorted(by_ds.items()):
        rows.sort(key=lambda r: r["_sort"])
        sorted_folds = sorted(fold_keys_by_ds[ds_key], key=_sort_fold_key)
        fold_col_names: list[str] = []
        for fk in sorted_folds:
            fold_col_names.extend(
                [
                    f"{fk}_spearman",
                    f"{fk}_pearson",
                    f"{fk}_r2",
                    f"{fk}_prediction_std_mean",
                ]
            )
        for r in rows:
            fm = r.pop("_fold_metrics")
            for fk in sorted_folds:
                pair = fm.get(fk, {})
                r[f"{fk}_spearman"] = _optional_float_cell(pair.get("spearman"))
                r[f"{fk}_pearson"] = _optional_float_cell(pair.get("pearson"))
                r[f"{fk}_r2"] = _optional_float_cell(pair.get("r2"))
                r[f"{fk}_prediction_std_mean"] = _optional_float_cell(
                    pair.get("prediction_std_mean")
                )
        for r in rows:
            del r["_sort"]
        df = pd.DataFrame(rows)
        df = df.reindex(columns=list(AGG_BASE_CSV_COLS) + fold_col_names)
        out_csv = out_dir / f"aggregated_{_safe_filename_segment(ds_key)}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} ({len(df)} rows)", file=sys.stderr)
        if not args.no_plots:
            _plot_best_combo_fold_scatters(df, dataset_key=ds_key, out_plot_dir=plot_dir)
            stem0 = str(df["Dataset_stem"].iloc[0]) if len(df) else "unknown"
            dev0 = str(df["Developability_source"].iloc[0]) if len(df) else ds_key
            tables_for_combine.append((ds_key, stem0, dev0, df))

    if not args.no_plots and len(tables_for_combine) >= 2:
        _plot_combined_by_experimental_target(tables_for_combine, out_plot_dir=plot_dir)

    times_rows: list[dict[str, object]] = []
    for (yk, tgt), total in sorted(
        dataset_target_pipeline_s.items(), key=lambda it: (it[0][0], it[0][1])
    ):
        times_rows.append(
            {
                "dataset_yaml_key": yk,
                "Dataset_stem": stem_by_yaml.get(yk, "unknown"),
                "Developability_source": yaml_to_dev.get(yk, yk),
                "Target_col": tgt,
                "total_pipeline_time_s": float(total),
            }
        )
    times_path = out_dir / "times.csv"
    pd.DataFrame(times_rows, columns=list(TIMES_CSV_COLS)).to_csv(
        times_path, index=False
    )
    print(
        f"Wrote {times_path} ({len(times_rows)} dataset–target row(s); "
        "each total is sum of pipeline_time_seconds over distinct result JSONs)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
