#!/usr/bin/env python3
"""Per-dataset strip/box plots of per-fold Spearman."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.aggregated_csv import (
    COL_DATASET,
    COL_RUN_ID,
    COL_SOURCE,
    COL_SPEAR,
    COL_TARGET,
    expand_paths,
    fold_spearman_columns,
    is_gpr_run,
    is_our_source,
    resolve_output_dir,
)

COLOR_OUR = "#FF69B4"
ALPHA_DOT = 0.72
ALPHA_BOX = 0.55
DOT_SIZE = 28


def _best_row(sub: pd.DataFrame, *, no_gpr: bool) -> pd.Series | None:
    if sub is None or len(sub) == 0:
        return None
    if no_gpr:
        filt = sub[
            ~sub.apply(
                lambda r: is_gpr_run(str(r[COL_RUN_ID]), str(r[COL_TARGET])), axis=1
            )
        ]
        if len(filt):
            sub = filt
    s = pd.to_numeric(sub[COL_SPEAR], errors="coerce")
    return sub.loc[s.idxmax()] if s.notna().any() else None


def _fold_ys(row: pd.Series, fc: list[str]) -> list[float]:
    out = []
    for c in fc:
        try:
            v = float(row[c])
            if math.isfinite(v):
                out.append(v)
        except (TypeError, ValueError):
            pass
    return out


def plot_dataset(
    df: pd.DataFrame,
    dataset_stem: str,
    out_png: Path,
    *,
    no_gpr: bool,
    dpi: int,
    width_per_target: float,
    height: float,
    strip_offset: float,
    jitter: float,
    rng: np.random.Generator,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    sub = df[df[COL_DATASET].astype(str) == dataset_stem]
    if len(sub) == 0:
        return False
    targets = sorted(sub[COL_TARGET].astype(str).unique())
    fc = fold_spearman_columns(sub.columns)
    if not fc:
        print(f"No fold_*_spearman columns for {dataset_stem!r}; skip.", file=sys.stderr)
        return False

    n = len(targets)
    fig_w = max(3.5, width_per_target * n + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, height))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.20, top=0.58)

    trans_xdata_yaxes = blended_transform_factory(ax.transData, ax.transAxes)

    pooled_y: list[float] = []

    for ti, tgt in enumerate(targets):
        xc = float(ti)
        rows = sub[
            sub[COL_SOURCE].astype(str).map(is_our_source)
            & (sub[COL_TARGET].astype(str) == tgt)
        ]
        if rows.empty:
            continue

        sources = rows[COL_SOURCE].unique()
        all_ys: list[float] = []
        mean_spearmans: list[float] = []
        for src in sources:
            src_rows = rows[rows[COL_SOURCE] == src]
            best = _best_row(src_rows, no_gpr=no_gpr)
            if best is None:
                continue
            ys = _fold_ys(best, fc)
            all_ys.extend(ys)
            try:
                v = float(best[COL_SPEAR])
                if math.isfinite(v):
                    mean_spearmans.append(v)
            except (TypeError, ValueError):
                pass

        if not all_ys:
            continue

        pooled_y.extend(all_ys)
        xs = xc + rng.uniform(-jitter, jitter, size=len(all_ys))
        ax.scatter(
            xs, all_ys, c=COLOR_OUR, s=DOT_SIZE, alpha=ALPHA_DOT, zorder=3, edgecolors="none"
        )
        ax.boxplot(
            all_ys,
            positions=[xc],
            widths=strip_offset * 0.9,
            patch_artist=True,
            manage_ticks=False,
            zorder=4,
            notch=False,
            showcaps=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=1.8),
            boxprops=dict(facecolor=COLOR_OUR, alpha=ALPHA_BOX, linewidth=0.8),
            whiskerprops=dict(color=COLOR_OUR, linewidth=0.9, linestyle="--"),
            capprops=dict(color=COLOR_OUR, linewidth=1.0),
        )

        if mean_spearmans:
            mean_val = float(np.mean(mean_spearmans))
            ax.text(
                xc,
                1.028,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=COLOR_OUR,
                transform=trans_xdata_yaxes,
                clip_on=False,
            )

    ax.axhline(0, color="0.80", lw=0.8, zorder=0)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.set_xlim(-0.5, n - 0.5)
    if pooled_y:
        y_min_data = float(min(pooled_y))
        y_max_data = float(max(pooled_y))
        if y_min_data < 0.0:
            pad = 0.04
            y_lo = max(-1.05, y_min_data - pad)
            y_hi = min(1.05, y_max_data + pad)
            y_hi = max(y_hi, y_lo + 0.08)
            ax.set_ylim(y_lo, y_hi)
        else:
            ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ (per fold)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    no_gpr_note = "  |  GPR excluded" if no_gpr else ""
    fig.suptitle(
        f"{dataset_stem}{no_gpr_note}",
        fontsize=10,
        fontweight="medium",
        y=1.02,
        va="top",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def run_plot_fold_spearmans(
    inputs: list[str],
    out_dir: Path | None,
    *,
    no_gpr: bool = False,
    dpi: int = 150,
    width_per_target: float = 2.2,
    height: float = 4.2,
    strip_offset: float = 0.18,
    jitter: float = 0.06,
    seed: int = 42,
) -> int:
    paths = expand_paths([str(x) for x in inputs])
    if not paths:
        print("No input files found.", file=sys.stderr)
        return 1

    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except OSError as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
    if not frames:
        print("No readable CSVs.", file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)
    missing = {COL_RUN_ID, COL_TARGET, COL_DATASET, COL_SOURCE, COL_SPEAR} - set(df.columns)
    if missing:
        print(f"Missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    resolved_out = resolve_output_dir(out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df[COL_DATASET].astype(str).unique())
    n_written = 0
    for i, ds in enumerate(datasets):
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", ds).strip("_") or "dataset"
        out_png = resolved_out / f"{safe}_fold_spearman_best.png"
        rng = np.random.default_rng(seed + i * 100_003)
        ok = plot_dataset(
            df,
            ds,
            out_png,
            no_gpr=no_gpr,
            dpi=dpi,
            width_per_target=width_per_target,
            height=height,
            strip_offset=strip_offset,
            jitter=jitter,
            rng=rng,
        )
        msg = f"Wrote {out_png}" if ok else f"Skipped (no data): {ds!r}"
        print(msg, file=sys.stderr)
        if ok:
            n_written += 1

    return 0 if n_written else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Strip+box plot of per-fold Spearman per dataset.")
    p.add_argument("inputs", nargs="+", help="Aggregated CSV paths (globs OK).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./analysis_results in cwd).",
    )
    p.add_argument(
        "--no-gpr",
        "--no_gpr",
        action="store_true",
        help="Exclude GPR rows when picking best Spearman row.",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--width-per-target", type=float, default=2.2)
    p.add_argument("--height", type=float, default=4.2)
    p.add_argument("--strip-offset", type=float, default=0.18)
    p.add_argument("--jitter", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raise SystemExit(
        run_plot_fold_spearmans(
            args.inputs,
            args.out_dir,
            no_gpr=bool(args.no_gpr),
            dpi=args.dpi,
            width_per_target=args.width_per_target,
            height=args.height,
            strip_offset=args.strip_offset,
            jitter=args.jitter,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
