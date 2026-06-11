#!/usr/bin/env python3
"""Batch-run aggregated CSV analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from analysis.aggregated_csv import (
    COL_DATASET,
    COL_FEATURES,
    COL_PEAR as AGG_COL_PEAR,
    COL_RUN_ID,
    COL_SOURCE,
    COL_SPEAR as AGG_COL_SPEAR,
    COL_TARGET,
    COL_TRACK,
    eval_model_slug as _eval_model_slug,
    expand_glob_pattern,
    expand_paths as _expand_paths,
    is_our_source as _is_our_source,
    resolve_output_dir,
    row_is_gpr_eval as _row_is_gpr_eval,
    row_matches_eval_model_filter as _row_matches_eval_model_filter,
    run_suffix_after_target as _run_suffix_after_target,
    selector_model_frac as _selector_model_frac,
)
from analysis.features import (
    COL_PEAR_FEAT,
    COL_PEAR_RUN,
    COL_SPEAR_FEAT,
    COL_SPEAR_RUN,
    SUMMARY_COL_PEAR,
    SUMMARY_COL_SPEAR,
    build_feature_usage_rows,
    canonical_feature_names,
    normalize_feature_count_dict,
    normalize_feature_name,
    print_json_features_missing_from_usage,
    validate_reference_features,
    write_feature_usage_csv,
    write_global_feature_usage_from_aggregated,
)

_MAX_BOOL_WORKSPACE = 50_000_000
_DENOM_EPS = 1e-12

_FOLD_TRAIL_INT = re.compile(r"(\d+)\s*$")


def _ordered_json_fold_keys(data: dict[Any, Any]) -> list[Any]:
    def sort_key(k: Any) -> tuple[Any, ...]:
        s = str(k)
        m = _FOLD_TRAIL_INT.search(s)
        if m:
            prefix = s[: m.start()]
            return (0, prefix, int(m.group(1)))
        return (1, s)

    return sorted(data.keys(), key=sort_key)


def _observed_jaccard_from_sets(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _batch_jaccard_from_indices(
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    k_i: int,
    k_j: int,
    p: int,
) -> np.ndarray:
    b = idx_i.shape[0]
    if k_i == 0 and k_j == 0:
        return np.ones(b, dtype=np.float64)
    if k_i == 0 or k_j == 0:
        union = np.full(b, float(k_i + k_j), dtype=np.float64)
        return np.zeros(b, dtype=np.float64)

    use_bitmap = (b * p) <= _MAX_BOOL_WORKSPACE and p <= 2**24
    if use_bitmap:
        occ_i = np.zeros((b, p), dtype=bool)
        occ_j = np.zeros((b, p), dtype=bool)
        br = np.arange(b, dtype=np.int64)[:, None]
        occ_i[br, idx_i] = True
        occ_j[br, idx_j] = True
        inter = np.count_nonzero(occ_i & occ_j, axis=1).astype(np.float64)
        union = np.count_nonzero(occ_i | occ_j, axis=1).astype(np.float64)
    else:
        eq = idx_i[:, :, None] == idx_j[:, None, :]
        inter = eq.any(axis=2).sum(axis=1).astype(np.float64)
        union = (k_i + k_j - inter).astype(np.float64)

    out = np.ones(b, dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0)
    out = np.where(union > 0, out, 1.0)
    return out


def _sample_random_subsets_batched(
    rng: np.random.Generator,
    b: int,
    k: int,
    p: int,
) -> np.ndarray:
    if k == 0:
        return np.zeros((b, 0), dtype=np.int64)
    if k > p:
        raise ValueError(f"subset size k={k} exceeds universe size p={p}")
    noise = rng.random((b, p), dtype=np.float64)
    part = np.argpartition(noise, k - 1, axis=1)[:, :k]
    return part.astype(np.int64, copy=False)


def pairwise_jaccard_stability(
    feature_sets: Sequence[set[int]],
    p: int,
    *,
    n_permutations: int = 1000,
    random_seed: int = 42,
    pairwise: bool = False,
) -> dict[str, Any]:
    if p < 0 or not isinstance(p, int):
        raise ValueError("p must be a non-negative int")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    m = len(feature_sets)
    if m < 2:
        raise ValueError("need at least two folds (sets) for pairwise metrics")

    for i, s in enumerate(feature_sets):
        for x in s:
            if x < 0 or x >= p:
                raise ValueError(f"feature index {x} out of range for p={p}")
        if len(s) > p:
            raise ValueError(f"|S_{i}|={len(s)} exceeds p={p}")

    rng = np.random.default_rng(random_seed)
    pair_obs: list[float] = []
    pair_exp: list[float] = []
    pair_norm: list[float] = []
    detail: list[dict[str, Any]] = []

    for i in range(m):
        for j in range(i + 1, m):
            si = feature_sets[i]
            sj = feature_sets[j]
            k_i, k_j = len(si), len(sj)
            j_obs = _observed_jaccard_from_sets(si, sj)

            b = int(n_permutations)
            idx_i = _sample_random_subsets_batched(rng, b, k_i, p)
            idx_j = _sample_random_subsets_batched(rng, b, k_j, p)
            j_perm = _batch_jaccard_from_indices(idx_i, idx_j, k_i, k_j, p)
            e_j = float(np.mean(j_perm))

            denom = 1.0 - e_j
            if abs(denom) <= _DENOM_EPS:
                warnings.warn(
                    f"pair ({i},{j}): 1 - E[J] ≈ 0 (E[J]={e_j}); setting normalized stability to 0",
                    RuntimeWarning,
                    stacklevel=2,
                )
                j_norm = 0.0
            else:
                j_norm = (j_obs - e_j) / denom

            pair_obs.append(j_obs)
            pair_exp.append(e_j)
            pair_norm.append(j_norm)
            if pairwise:
                detail.append(
                    {
                        "i": i,
                        "j": j,
                        "k_i": k_i,
                        "k_j": k_j,
                        "jaccard_observed": j_obs,
                        "jaccard_expected": e_j,
                        "jaccard_normalized": j_norm,
                    }
                )

    return {
        "mean_jaccard": float(np.mean(pair_obs)),
        "mean_expected_jaccard": float(np.mean(pair_exp)),
        "mean_normalized_stability": float(np.mean(pair_norm)),
        **({"pairwise": detail} if pairwise else {}),
    }


def encode_feature_sets_as_ints(
    sets_of_names: Sequence[set[str]],
    universe: Sequence[str],
) -> list[set[int]]:
    if len(set(universe)) != len(universe):
        raise ValueError("universe contains duplicate feature names")
    p = len(universe)
    mp = {name: i for i, name in enumerate(universe)}
    out: list[set[int]] = []
    for t, s in enumerate(sets_of_names):
        idxs: set[int] = set()
        for name in s:
            if name not in mp:
                raise KeyError(
                    f"feature {name!r} in fold {t} not found in universe (p={p})"
                )
            idxs.add(mp[name])
        out.append(idxs)
    return out


def parse_feature_sets_by_fold_json(cell: str) -> list[set[str]] | None:
    if not cell or not str(cell).strip():
        return None
    try:
        data = json.loads(cell)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data:
        return None
    ordered_keys = _ordered_json_fold_keys(data)
    out: list[set[str]] = []
    for k in ordered_keys:
        feats = data[k]
        if not isinstance(feats, list):
            continue
        out.append({normalize_feature_name(f) for f in feats if isinstance(f, str)})
    return out if len(out) >= 2 else None


def load_universe_features(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(normalize_feature_name(s))
    return tuple(out)


def stability_from_features_cell(
    features_cell: str,
    universe: Sequence[str],
    *,
    n_permutations: int = 1000,
    random_seed: int = 42,
    pairwise: bool = False,
) -> dict[str, Any] | None:
    names = parse_feature_sets_by_fold_json(features_cell)
    if not names:
        return None
    ints = encode_feature_sets_as_ints(names, universe)
    return pairwise_jaccard_stability(
        ints,
        len(universe),
        n_permutations=n_permutations,
        random_seed=random_seed,
        pairwise=pairwise,
    )

GROUP_KEYS = (COL_DATASET, COL_SOURCE, COL_TARGET, COL_TRACK)

COL_VARIANT = "Variant"
COL_RESULT_SPEARMAN = "Spearman"
COL_RESULT_JACCARD_NORM = "Jaccard_norm"
COL_RESULT_BEST_RUN = "best_Target-Selector-Model"
COL_RESULT_BEST_FRAC = "best_selector_model_frac"

_AGGREGATED_FNAME = re.compile(r"^aggregated_(.+)\.csv$", re.IGNORECASE)
_RANDOM_SEED_SUFFIX = re.compile(r"__rs\d+$")
_VARIANT_SIDE_MARKERS = frozenset({"our"})
_ABB2_VARIANT_IN_SOURCE = re.compile(r"(abb2_\d+)")

COL_STAB_JACCARD = "best_spearman_fold_stability_mean_jaccard"
COL_STAB_EXPECTED = "best_spearman_fold_stability_mean_expected_jaccard"
COL_STAB_NORM = "best_spearman_fold_stability_mean_normalized"
STAB_COLS = (COL_STAB_JACCARD, COL_STAB_EXPECTED, COL_STAB_NORM)


def _parse_float(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _pipeline_time_s_from_aggregate_row(row: dict[str, Any]) -> float | None:
    for key in ("Pipeline_time_sum_s", "Pipeline_time_mean_s"):
        v = row.get(key)
        if v is None or v == "":
            continue
        if isinstance(v, (int, float)):
            x = float(v)
            return x if math.isfinite(x) else None
        p = _parse_float(str(v).strip())
        if p is not None:
            return p
    return None


def _feature_names_in_cell(cell: str) -> set[str]:
    fold_sets = parse_feature_sets_by_fold_json(cell)
    if not fold_sets:
        return set()
    out: set[str] = set()
    for s in fold_sets:
        out |= s
    return out


def _infer_universe_from_rows(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for r in rows:
        if not _is_our_source(str(r.get(COL_SOURCE, ""))):
            continue
        cell = str(r.get(COL_FEATURES, "") or "")
        names |= _feature_names_in_cell(cell)
    return tuple(sorted(names))


def _universe_for_source(
    source: str,
    *,
    universe_single: tuple[str, ...] | None,
    universe_our: tuple[str, ...],
) -> tuple[str, ...] | None:
    if universe_single is not None:
        return universe_single
    if _is_our_source(source):
        return universe_our if universe_our else None
    return None


def _rows_for_best_pick(
    group_rows: list[dict[str, Any]], *, no_gpr: bool
) -> tuple[list[dict[str, Any]], bool]:
    if not no_gpr:
        return group_rows, False
    filtered = [r for r in group_rows if not _row_is_gpr_eval(r)]
    if not filtered:
        return group_rows, True
    return filtered, False


def _feature_fold_counts_json(features_cell: str) -> str:
    if not features_cell or not features_cell.strip():
        return "{}"
    try:
        data = json.loads(features_cell)
    except json.JSONDecodeError:
        return "{}"
    if not isinstance(data, dict):
        return "{}"
    counts: dict[str, int] = defaultdict(int)
    for feats in data.values():
        if not isinstance(feats, list):
            continue
        in_fold = {normalize_feature_name(f) for f in feats if isinstance(f, str)}
        for f in in_fold:
            counts[f] += 1
    ordered = normalize_feature_count_dict(counts)
    return json.dumps(ordered, ensure_ascii=False)


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_file"] = p.name
                rows.append(row)
    return rows


def _track_cell(row: dict[str, Any]) -> str:
    v = row.get(COL_TRACK)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row[COL_DATASET],
        row[COL_SOURCE],
        row[COL_TARGET],
        _track_cell(row),
    )


def _group_key_best_across_tracks(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        row[COL_DATASET],
        row[COL_SOURCE],
        row[COL_TARGET],
    )


def _best_in_group(
    group_rows: list[dict[str, Any]], metric: str
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_val: float | None = None
    for r in group_rows:
        v = _parse_float(r.get(metric, ""))
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val = v
            best = r
    return best


def _stability_triple(
    features_cell: str,
    universe: tuple[str, ...],
    n_permutations: int,
    random_seed: int,
    cache: dict[tuple[str, int, int, tuple[str, ...]], dict[str, Any] | None],
) -> tuple[float | None, float | None, float | None]:
    key = (features_cell, n_permutations, random_seed, universe)
    if key not in cache:
        cache[key] = stability_from_features_cell(
            features_cell,
            universe,
            n_permutations=n_permutations,
            random_seed=random_seed,
            pairwise=False,
        )
    s = cache[key]
    if not s:
        return None, None, None
    return (
        float(s["mean_jaccard"]),
        float(s["mean_expected_jaccard"]),
        float(s["mean_normalized_stability"]),
    )


def _spearman_stability_rank_key(
    r: dict[str, Any],
    universe: tuple[str, ...],
    n_permutations: int,
    random_seed: int,
    cache: dict[tuple[str, int, int, tuple[str, ...]], dict[str, Any] | None],
    mode: str,
    stability_weight: float,
) -> tuple[Any, ...] | None:
    spe = _parse_float(r.get(AGG_COL_SPEAR, ""))
    if spe is None:
        return None
    cell = str(r.get(COL_FEATURES, "") or "")
    mj, me, mn = _stability_triple(
        cell, universe, n_permutations, random_seed, cache
    )
    stab = mn if mn is not None else float("-inf")
    if mode == "spearman":
        return (spe,)
    if mode == "lexi":
        return (spe, stab)
    if mode == "composite":
        return (spe + stability_weight * stab,)
    raise ValueError(f"unknown rank mode: {mode!r}")


def _best_spearman_row_in_group(
    group_rows: list[dict[str, Any]],
    universe: tuple[str, ...] | None,
    rank_with_stability: str,
    n_permutations: int,
    random_seed: int,
    stability_weight: float,
    cache: dict[tuple[str, int, int, tuple[str, ...]], dict[str, Any] | None],
) -> dict[str, Any] | None:
    if universe is None or rank_with_stability == "none":
        return _best_in_group(group_rows, AGG_COL_SPEAR)
    mode = rank_with_stability
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for r in group_rows:
        k = _spearman_stability_rank_key(
            r,
            universe,
            n_permutations,
            random_seed,
            cache,
            mode,
            stability_weight,
        )
        if k is None:
            continue
        if best_key is None or k > best_key:
            best_key = k
            best = r
    return best


def build_summary(
    rows: list[dict[str, Any]],
    *,
    universe_single: tuple[str, ...] | None = None,
    universe_our: tuple[str, ...] = (),
    rank_with_stability: str = "none",
    stability_permutations: int = 1000,
    stability_random_seed: int = 42,
    stability_weight: float = 0.1,
    eval_model: str | None = None,
    no_gpr: bool = False,
    best_across_tracks: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    cache: dict[tuple[str, int, int, tuple[str, ...]], dict[str, Any] | None] = {}
    em_lc = str(eval_model).strip().lower() if eval_model and str(eval_model).strip() else None
    by_g: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    gkey = _group_key_best_across_tracks if best_across_tracks else _group_key
    for r in rows:
        if not _row_matches_eval_model_filter(
            str(r.get(COL_RUN_ID, "")),
            str(r.get(COL_TARGET, "")),
            em_lc,
        ):
            continue
        by_g[gkey(r)].append(r)

    out: list[dict[str, Any]] = []
    no_gpr_fallback_groups = 0
    for key in sorted(by_g.keys()):
        g = by_g[key]
        if not g:
            continue
        ds, src, tgt = key[0], key[1], key[2]
        if best_across_tracks:
            trk_out: str = ""
        else:
            trk_out = str(key[3])
        g_spear, fb_s = _rows_for_best_pick(g, no_gpr=no_gpr)
        g_pear, fb_p = _rows_for_best_pick(g, no_gpr=no_gpr)
        if fb_s or fb_p:
            no_gpr_fallback_groups += 1
        univ = _universe_for_source(
            src,
            universe_single=universe_single,
            universe_our=universe_our,
        )
        rs = _best_spearman_row_in_group(
            g_spear,
            univ,
            rank_with_stability,
            stability_permutations,
            stability_random_seed,
            stability_weight,
            cache,
        )
        rp = _best_in_group(g_pear, AGG_COL_PEAR)
        if best_across_tracks:
            trk_out = _track_cell(rs) if rs else (_track_cell(rp) if rp else "")
        row: dict[str, Any] = {
            COL_DATASET: ds,
            COL_SOURCE: src,
            COL_TARGET: tgt,
            COL_TRACK: trk_out,
        }
        if rs:
            spe = _parse_float(rs[AGG_COL_SPEAR])
            row["best_spearman"] = spe
            row["best_spearman_Target-Selector-Model"] = rs[COL_RUN_ID]
            row["best_spearman_selector_model_frac"] = _selector_model_frac(
                rs[COL_RUN_ID], rs[COL_TARGET]
            )
            row["best_spearman_features"] = _feature_fold_counts_json(
                rs.get(COL_FEATURES, "")
            )
            if univ is not None:
                mj, me, mn = _stability_triple(
                    str(rs.get(COL_FEATURES, "") or ""),
                    univ,
                    stability_permutations,
                    stability_random_seed,
                    cache,
                )
                row[COL_STAB_JACCARD] = mj
                row[COL_STAB_EXPECTED] = me
                row[COL_STAB_NORM] = mn
            row["Pipeline_time_sum_s"] = _pipeline_time_s_from_aggregate_row(rs)
        if rp:
            pea = _parse_float(rp[AGG_COL_PEAR])
            row["best_pearson"] = pea
            row["best_pearson_Target-Selector-Model"] = rp[COL_RUN_ID]
            row["best_pearson_selector_model_frac"] = _selector_model_frac(
                rp[COL_RUN_ID], rp[COL_TARGET]
            )
            row["best_pearson_features"] = _feature_fold_counts_json(
                rp.get(COL_FEATURES, "")
            )
        out.append(row)
    return out, no_gpr_fallback_groups


def _collapse_summary_best_track_per_source(
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in summary:
        ds = str(r.get(COL_DATASET, ""))
        src = str(r.get(COL_SOURCE, ""))
        tgt = str(r.get(COL_TARGET, ""))
        by_key[(ds, src, tgt)].append(r)
    out: list[dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        rows = by_key[key]
        best_row: dict[str, Any] | None = None
        best_val: float | None = None
        for r in rows:
            spe = r.get("best_spearman")
            if not isinstance(spe, (int, float)):
                continue
            v = float(spe)
            if best_val is None or v > best_val:
                best_val = v
                best_row = r
        if best_row is not None:
            out.append(dict(best_row))
        elif rows:
            out.append(dict(rows[0]))
    return out


def _stab_normalized_from_summary_row(r: dict[str, Any]) -> float | None:
    v = r.get(COL_STAB_NORM)
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isfinite(x):
            return x
    return None


def _aggregated_slug_from_filename(name: str) -> str | None:
    m = _AGGREGATED_FNAME.match(name)
    return m.group(1) if m else None


def _strip_random_seed_suffix(slug: str) -> str:
    return _RANDOM_SEED_SUFFIX.sub("", slug)


def _variant_from_slug(slug: str, dataset_stem: str) -> str:
    slug = _strip_random_seed_suffix(slug)
    if slug == dataset_stem:
        return ""
    prefix = f"{dataset_stem}_"
    if not slug.startswith(prefix):
        return slug
    rest = slug[len(prefix) :]
    if rest.lower() in _VARIANT_SIDE_MARKERS:
        return ""
    return rest


def build_aggregated_variant_map(
    paths: list[Path], rows: list[dict[str, Any]]
) -> dict[tuple[str, str], str]:
    """Map (Dataset_stem, Developability_source) -> variant (e.g. abb2_1)."""
    dataset_by_file: dict[str, str] = {}
    for row in rows:
        fn = str(row.get("_source_file", ""))
        if fn and fn not in dataset_by_file:
            dataset_by_file[fn] = str(row.get(COL_DATASET, ""))

    file_variant: dict[str, str] = {}
    for p in paths:
        slug = _aggregated_slug_from_filename(p.name)
        if slug is None:
            continue
        ds = dataset_by_file.get(p.name, "")
        if not ds:
            continue
        file_variant[p.name] = _variant_from_slug(slug, ds)

    out: dict[tuple[str, str], str] = {}
    for row in rows:
        ds = str(row.get(COL_DATASET, ""))
        src = str(row.get(COL_SOURCE, ""))
        fn = str(row.get("_source_file", ""))
        key = (ds, src)
        if key in out:
            continue
        if fn in file_variant:
            out[key] = file_variant[fn]
    return out


def _variant_for_summary_row(
    r: dict[str, Any], variant_map: dict[tuple[str, str], str]
) -> str:
    ds = str(r.get(COL_DATASET, ""))
    src = str(r.get(COL_SOURCE, ""))
    if (ds, src) in variant_map:
        return variant_map[(ds, src)]
    m = _ABB2_VARIANT_IN_SOURCE.search(src)
    return m.group(1) if m else ""


def build_results_per_target(
    summary: list[dict[str, Any]],
    variant_map: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """One row per (dataset, variant, target): best model and mean metrics across seeds."""
    variant_map = variant_map or {}
    collapsed = _collapse_summary_best_track_per_source(summary)
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in collapsed:
        if not _is_our_source(str(r.get(COL_SOURCE, ""))):
            continue
        ds = str(r.get(COL_DATASET, ""))
        tgt = str(r.get(COL_TARGET, ""))
        variant = _variant_for_summary_row(r, variant_map)
        by_key[(ds, variant, tgt)].append(r)

    out: list[dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        ds, variant, tgt = key
        group = by_key[key]
        spears: list[float] = []
        jaccs: list[float] = []
        best_row: dict[str, Any] | None = None
        best_spear: float | None = None
        for r in group:
            spe = r.get("best_spearman")
            if isinstance(spe, (int, float)):
                v = float(spe)
                spears.append(v)
                if best_spear is None or v > best_spear:
                    best_spear = v
                    best_row = r
            jac = _stab_normalized_from_summary_row(r)
            if jac is not None:
                jaccs.append(jac)
        if best_row is None and group:
            best_row = group[0]
        out.append(
            {
                COL_DATASET: ds,
                COL_VARIANT: variant,
                COL_TARGET: tgt,
                COL_RESULT_SPEARMAN: statistics.mean(spears) if spears else None,
                COL_RESULT_JACCARD_NORM: statistics.mean(jaccs) if jaccs else None,
                COL_RESULT_BEST_RUN: str(best_row.get("best_spearman_Target-Selector-Model", ""))
                if best_row
                else "",
                COL_RESULT_BEST_FRAC: str(
                    best_row.get("best_spearman_selector_model_frac", "")
                )
                if best_row
                else "",
            }
        )
    return out


def _fmt_spearman_cell(v: float | None) -> str:
    if v is None:
        return ""
    return f"{float(v):.2f}"


def _fmt_float_cell(v: float | None) -> str:
    if v is None:
        return ""
    return f"{float(v):.17g}"


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        COL_DATASET,
        COL_VARIANT,
        COL_TARGET,
        COL_RESULT_SPEARMAN,
        COL_RESULT_JACCARD_NORM,
        COL_RESULT_BEST_RUN,
        COL_RESULT_BEST_FRAC,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    COL_DATASET: r[COL_DATASET],
                    COL_VARIANT: r.get(COL_VARIANT, ""),
                    COL_TARGET: r[COL_TARGET],
                    COL_RESULT_SPEARMAN: _fmt_spearman_cell(
                        r.get(COL_RESULT_SPEARMAN)  # type: ignore[arg-type]
                    ),
                    COL_RESULT_JACCARD_NORM: _fmt_float_cell(
                        r.get(COL_RESULT_JACCARD_NORM)  # type: ignore[arg-type]
                    ),
                    COL_RESULT_BEST_RUN: r.get(COL_RESULT_BEST_RUN, ""),
                    COL_RESULT_BEST_FRAC: r.get(COL_RESULT_BEST_FRAC, ""),
                }
            )


def run_best_metrics_from_aggregated(
    inputs: list[str],
    summary_out: Path,
    *,
    results_out: Path | None = None,
    universe_features: Path | None = None,
    universe_features_our: Path | None = None,
    rank_with_stability: str = "none",
    stability_permutations: int = 1000,
    stability_seed: int = 42,
    stability_weight: float = 0.1,
    eval_model: str | None = None,
    no_gpr: bool = False,
    best_across_tracks: bool = False,
) -> tuple[Path, Path]:
    paths = _expand_paths([str(x) for x in inputs])
    if not paths:
        raise SystemExit("No input files found.")
    rows = _read_rows(paths)
    rows = [r for r in rows if _is_our_source(str(r.get(COL_SOURCE, "")))]
    required = {COL_RUN_ID, COL_TARGET, AGG_COL_SPEAR, AGG_COL_PEAR, COL_FEATURES}
    if rows:
        missing = required - set(rows[0].keys())
        if missing:
            raise SystemExit(f"Missing required columns: {sorted(missing)}")

    def _load_universe_arg(path: Path | None, label: str) -> tuple[str, ...] | None:
        if path is None:
            return None
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
        u = load_universe_features(path)
        if len(u) == 0:
            raise SystemExit(f"{label} file is empty.")
        return u

    universe_single = _load_universe_arg(universe_features, "--universe-features")
    universe_our_arg = _load_universe_arg(universe_features_our, "--universe-features-our")
    inferred_our = _infer_universe_from_rows(rows)
    if universe_single is not None:
        universe_our = universe_single
    else:
        universe_our = universe_our_arg if universe_our_arg is not None else inferred_our
    if rank_with_stability != "none" and not universe_our:
        raise SystemExit(
            "--rank-with-stability requires a feature universe (infer from inputs or pass "
            "--universe-features / --universe-features-our)."
        )
    if stability_permutations < 1:
        raise SystemExit("--stability-permutations must be >= 1.")
    if universe_single is None and universe_our:
        print(
            f"Inferred feature universe: p={len(universe_our)}",
            file=sys.stderr,
        )

    summary, no_gpr_fallback_groups = build_summary(
        rows,
        universe_single=universe_single,
        universe_our=universe_our,
        rank_with_stability=rank_with_stability,
        stability_permutations=stability_permutations,
        stability_random_seed=stability_seed,
        stability_weight=stability_weight,
        eval_model=eval_model,
        no_gpr=no_gpr,
        best_across_tracks=best_across_tracks,
    )
    if no_gpr and no_gpr_fallback_groups:
        print(
            f"Warning: --no-gpr had {no_gpr_fallback_groups} group(s) where every candidate row "
            "was GPR (or only GPR used the new run-id shape); those groups used the full row set.",
            file=sys.stderr,
        )
    if not summary and rows and eval_model and str(eval_model).strip():
        print(
            "Warning: no summary rows after --eval-model filter "
            f"({len(rows)} input rows). Run ids must use …-selector-selmodel-<eval_model>-fracNNN.",
            file=sys.stderr,
        )
    fieldnames: list[str] = [
        COL_DATASET,
        COL_SOURCE,
        COL_TARGET,
        COL_TRACK,
        "best_spearman",
        "best_spearman_Target-Selector-Model",
        "best_spearman_selector_model_frac",
        "best_spearman_features",
        COL_STAB_JACCARD,
        COL_STAB_EXPECTED,
        COL_STAB_NORM,
        "best_pearson",
        "best_pearson_Target-Selector-Model",
        "best_pearson_selector_model_frac",
        "best_pearson_features",
        "Pipeline_time_sum_s",
        "Pipeline_time_std_s",
    ]
    stab_cols = STAB_COLS
    timing_cols = ("Pipeline_time_sum_s", "Pipeline_time_std_s")

    def _build_timing_tail_rows() -> list[dict[str, Any]]:
        times: list[float] = []
        for r in summary:
            pt_f = r.get("Pipeline_time_sum_s")
            if pt_f is None or not isinstance(pt_f, (int, float)):
                continue
            pt_f = float(pt_f)
            if math.isfinite(pt_f):
                times.append(pt_f)
        if not times:
            return []
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) >= 2 else None
        return [{
            COL_DATASET: "pipeline_time_summary",
            COL_SOURCE: "",
            COL_TARGET: "",
            COL_TRACK: "",
            "Pipeline_time_sum_s": _fmt_float_cell(mean_t),
            "Pipeline_time_std_s": _fmt_float_cell(std_t) if std_t is not None else "",
        }]

    def write_csv(dest: Any, tail_rows: list[dict[str, Any]] | None = None) -> None:
        w = csv.DictWriter(dest, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            out_row: dict[str, Any] = {}
            for k in fieldnames:
                v = r.get(k, "")
                if k in stab_cols or k in timing_cols:
                    out_row[k] = _fmt_float_cell(v) if isinstance(v, (int, float)) else ""
                else:
                    out_row[k] = v if v is not None else ""
            w.writerow(out_row)
        if tail_rows:
            for r in tail_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    timing_tail = _build_timing_tail_rows()

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", newline="", encoding="utf-8") as f:
        write_csv(f, tail_rows=timing_tail)
    print(
        f"Wrote {len(summary)} rows (+ {len(timing_tail)} pipeline-time summary) "
        f"to {summary_out}",
        file=sys.stderr,
    )

    if results_out is None:
        results_out = summary_out.parent / "results.csv"
    variant_map = build_aggregated_variant_map(paths, rows)
    result_rows = build_results_per_target(summary, variant_map=variant_map)
    write_results_csv(results_out, result_rows)
    print(
        f"Wrote {len(result_rows)} per-target result rows to {results_out}",
        file=sys.stderr,
    )
    return summary_out, results_out


COL_SPEARMAN = "best_spearman"
COL_FEAT = "best_spearman_features"
COL_TSM = "best_spearman_Target-Selector-Model"
COL_FRAC = "best_spearman_selector_model_frac"

_RANDOM_SEED_SOURCE_SUFFIX = re.compile(r"__rs\d+$")


@dataclass(frozen=True)
class FolderRowPick:
    score: float
    developability_source_json: str
    best_spearman_target_selector_model_json: str
    best_spearman_selector_model_frac_json: str
    best_spearman_features_json: str


def _is_seed_source(src: str) -> bool:
    return bool(_RANDOM_SEED_SOURCE_SUFFIX.search(src))


def _seed_suffix(src: str) -> str | None:
    m = _RANDOM_SEED_SOURCE_SUFFIX.search(src)
    return m.group(0) if m else None


def _source_base(src: str) -> str:
    return _RANDOM_SEED_SOURCE_SUFFIX.sub("", src)


def _seed_suffix_sort_key(suffix: str) -> int:
    m = re.search(r"__rs(\d+)$", suffix)
    return int(m.group(1)) if m else 0


def _iter_folded_batch_dirs(runs_dir: Path, exclude_names: frozenset[str]) -> list[Path]:
    out: list[Path] = []
    if not runs_dir.is_dir():
        return out
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("batch"):
            continue
        if "folded" not in name:
            continue
        if name in exclude_names:
            continue
        out.append(p)
    return out


def _metrics_path(batch_dir: Path) -> Path | None:
    a = batch_dir / "best_metrics.csv"
    b = batch_dir / "best_metrics_summary.csv"
    if a.is_file():
        return a
    if b.is_file():
        return b
    return None


def _parse_spearman(cell: str) -> float | None:
    if cell is None or not str(cell).strip():
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _accumulate_feature_usage_from_nested_json(
    features_json: str, totals: dict[str, int]
) -> None:
    if not features_json or not str(features_json).strip():
        return
    try:
        outer = json.loads(features_json)
    except json.JSONDecodeError:
        return
    if not isinstance(outer, dict):
        return
    for inner in outer.values():
        if not isinstance(inner, dict):
            continue
        for feat, v in inner.items():
            if not isinstance(feat, str):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if isinstance(v, float) and not v.is_integer():
                continue
            totals[normalize_feature_name(feat)] += int(v)


def _parse_feature_count_dict(cell: str) -> dict[str, int]:
    if not cell or not str(cell).strip():
        return {}
    try:
        data = json.loads(cell)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if isinstance(v, float) and not v.is_integer():
            continue
        key = normalize_feature_name(k)
        out[key] = out.get(key, 0) + int(v)
    return dict(sorted(out.items()))


def _group_score_from_rows(rows: list[tuple[str, float, str, str, str]]) -> float | None:
    seeds = [r for r in rows if _is_seed_source(r[0])]
    others = [r for r in rows if not _is_seed_source(r[0])]
    if seeds:
        return mean(r[1] for r in seeds)
    if others:
        return max(r[1] for r in others)
    return None


def _nested_pick_from_rows(rows: list[tuple[str, float, str, str, str]]) -> dict[str, str]:
    seeds = [r for r in rows if _is_seed_source(r[0])]
    if seeds:
        by_suffix: dict[str, tuple[str, float, str, str, str]] = {}
        for r in seeds:
            suf = _seed_suffix(r[0])
            if suf is not None:
                by_suffix[suf] = r
        ordered = sorted(by_suffix.items(), key=lambda item: _seed_suffix_sort_key(item[0]))
        src_m: dict[str, str] = {}
        tsm_m: dict[str, str] = {}
        frac_m: dict[str, str] = {}
        feat_m: dict[str, dict[str, int]] = {}
        for outer_key, r in ordered:
            s, _sp, feat_cell, tsm_cell, frac_cell = r
            src_m[outer_key] = s
            tsm_m[outer_key] = tsm_cell
            frac_m[outer_key] = frac_cell
            feat_m[outer_key] = _parse_feature_count_dict(feat_cell)
        return {
            "source": json.dumps(src_m, ensure_ascii=False, sort_keys=False),
            "tsm": json.dumps(tsm_m, ensure_ascii=False, sort_keys=False),
            "frac": json.dumps(frac_m, ensure_ascii=False, sort_keys=False),
            "feat": json.dumps(feat_m, ensure_ascii=False, sort_keys=False),
        }
    if not rows:
        return {"source": "{}", "tsm": "{}", "frac": "{}", "feat": "{}"}
    rep = max(rows, key=lambda r: (r[1], r[0]))
    src, _sp, feat_cell, tsm_cell, frac_cell = rep
    feat_dict = _parse_feature_count_dict(feat_cell)
    return {
        "source": json.dumps({src: src}, ensure_ascii=False, sort_keys=True),
        "tsm": json.dumps({src: tsm_cell}, ensure_ascii=False, sort_keys=True),
        "frac": json.dumps({src: frac_cell}, ensure_ascii=False, sort_keys=True),
        "feat": json.dumps({src: feat_dict}, ensure_ascii=False, sort_keys=True),
    }


def _folder_aggregate_from_csv(path: Path) -> dict[tuple[str, str], FolderRowPick]:
    raw: dict[tuple[str, str], dict[str, list[tuple[str, float, str, str, str]]]] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in (COL_DATASET, COL_SOURCE, COL_TARGET, COL_SPEARMAN):
            if col not in fields:
                print(f"Warning: {path}: missing column {col!r}; skip file.", file=sys.stderr)
                return {}

        for row in reader:
            ds = (row.get(COL_DATASET) or "").strip()
            tgt = (row.get(COL_TARGET) or "").strip()
            src = (row.get(COL_SOURCE) or "").strip()
            sp = _parse_spearman(row.get(COL_SPEARMAN, ""))
            if not ds or not tgt or src == "" or sp is None:
                continue
            if not _is_our_source(src):
                continue
            feat = row.get(COL_FEAT) or ""
            tsm = row.get(COL_TSM) or ""
            frac = row.get(COL_FRAC) or ""
            key = (ds, tgt)
            base = _source_base(src)
            raw.setdefault(key, {}).setdefault(base, []).append((src, sp, feat, tsm, frac))

    out: dict[tuple[str, str], FolderRowPick] = {}
    for key, by_base in raw.items():
        per_base: list[tuple[str, float, list[tuple[str, float, str, str, str]]]] = []
        for base, rows in by_base.items():
            sc = _group_score_from_rows(rows)
            if sc is None:
                continue
            per_base.append((base, sc, rows))
        if not per_base:
            continue
        _win_base, win_sc, win_rows = min(per_base, key=lambda t: (-t[1], t[0]))
        nested = _nested_pick_from_rows(win_rows)
        out[key] = FolderRowPick(
            score=win_sc,
            developability_source_json=nested["source"],
            best_spearman_target_selector_model_json=nested["tsm"],
            best_spearman_selector_model_frac_json=nested["frac"],
            best_spearman_features_json=nested["feat"],
        )
    return out


def run_compare_folded_batches(
    *,
    runs_dir: Path,
    out: Path,
    exclude: frozenset[str] | None = None,
    feature_usage_out: Path | None = None,
    skip_feature_usage: bool = False,
) -> None:
    exclude_list = list(exclude or ())
    runs_dir = runs_dir.resolve()
    exclude = frozenset({"batch_run_config_folded_selected_datasets", *exclude_list})

    batch_dirs = _iter_folded_batch_dirs(runs_dir, exclude)
    folder_data: list[tuple[str, Path, dict[tuple[str, str], FolderRowPick]]] = []
    for d in batch_dirs:
        mp = _metrics_path(d)
        if mp is None:
            print(f"Warning: skip {d.name}: no best_metrics.csv or best_metrics_summary.csv", file=sys.stderr)
            continue
        agg = _folder_aggregate_from_csv(mp)
        if not agg:
            print(f"Warning: skip {d.name}: no usable rows in {mp.name}", file=sys.stderr)
            continue
        folder_data.append((d.name, mp, agg))

    if not folder_data:
        print("No batch folders with metrics files found.", file=sys.stderr)
        sys.exit(1)

    all_keys: set[tuple[str, str]] = set()
    for _, _, sc in folder_data:
        all_keys.update(sc.keys())

    folder_names = [name for name, _, _ in folder_data]
    name_to_agg = {name: agg for name, _, agg in folder_data}

    rows_out: list[dict[str, str]] = []
    for key in sorted(all_keys):
        ds, tgt = key
        per_folder: list[tuple[str, FolderRowPick]] = []
        for fname, _, agg in folder_data:
            if key in agg:
                per_folder.append((fname, agg[key]))
        if not per_folder:
            continue
        best_val = max(p.score for _, p in per_folder)
        winners = sorted(fn for fn, p in per_folder if p.score == best_val)
        feat_folder = winners[0]
        pick = name_to_agg[feat_folder][key]

        row: dict[str, str] = {
            COL_DATASET: ds,
            COL_TARGET: tgt,
            "best_spearman": f"{best_val:.17g}",
            "best_batch_folder": ";".join(winners),
            "best_row_batch_folder": feat_folder,
            "n_batch_folders_compared": str(len(folder_data)),
            "n_batch_folders_with_key": str(len(per_folder)),
            COL_SOURCE: pick.developability_source_json,
            COL_TSM: pick.best_spearman_target_selector_model_json,
            COL_FRAC: pick.best_spearman_selector_model_frac_json,
            COL_FEAT: pick.best_spearman_features_json,
        }
        for fn in folder_names:
            agg = name_to_agg[fn]
            if key in agg:
                row[f"spearman_{fn}"] = f"{agg[key].score:.17g}"
            else:
                row[f"spearman_{fn}"] = ""
        rows_out.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        COL_DATASET,
        COL_TARGET,
        "best_spearman",
        "best_batch_folder",
        "best_row_batch_folder",
        "n_batch_folders_compared",
        "n_batch_folders_with_key",
        COL_SOURCE,
        COL_TSM,
        COL_FRAC,
        COL_FEAT,
        *[f"spearman_{fn}" for fn in folder_names],
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    print(
        f"Wrote {len(rows_out)} rows comparing {len(folder_data)} folders to {out}",
        file=sys.stderr,
    )

    if not skip_feature_usage:
        usage_out = feature_usage_out
        if usage_out is None:
            usage_out = out.with_name(f"{out.stem}_feature_usage.csv")
        usage_out = usage_out.resolve()
        usage_totals: dict[str, int] = defaultdict(int)
        for r in rows_out:
            _accumulate_feature_usage_from_nested_json(r.get(COL_FEAT, ""), usage_totals)
        usage_rows = sorted(usage_totals.items(), key=lambda x: (-x[1], x[0]))
        usage_out.parent.mkdir(parents=True, exist_ok=True)
        with open(usage_out, "w", newline="", encoding="utf-8") as f:
            uw = csv.writer(f)
            uw.writerow(["feature", "total_count"])
            uw.writerows(usage_rows)
        print(
            f"Wrote {len(usage_rows)} feature rows (sum of nested counts) to {usage_out}",
            file=sys.stderr,
        )

def _parse_feature_counts(cell: str) -> dict[str, int]:
    return _parse_feature_count_dict(cell)


def _float_metric(cell: str) -> float | None:
    if cell is None or (isinstance(cell, str) and not str(cell).strip()):
        return None
    try:
        x = float(cell)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def pick_best_track_rows(
    rows: list[dict[str, str]],
    *,
    metric: str,
) -> list[dict[str, str]]:
    if not rows:
        return rows
    names = set(rows[0].keys())
    if COL_TRACK not in names:
        return rows

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get(COL_DATASET, "")),
            str(row.get(COL_SOURCE, "")),
            str(row.get(COL_TARGET, "")),
        )
        groups[key].append(row)

    out: list[dict[str, str]] = []
    for grp in groups.values():
        if len(grp) == 1:
            out.append(grp[0])
            continue

        def sort_key(r: dict[str, str]) -> tuple[float, float, str]:
            sp = _float_metric(r.get(SUMMARY_COL_SPEAR, ""))
            pe = _float_metric(r.get(SUMMARY_COL_PEAR, ""))
            sp_v = sp if sp is not None else float("-inf")
            pe_v = pe if pe is not None else float("-inf")
            if metric == "pearson":
                primary = pe_v
                secondary = sp_v
            else:
                primary = sp_v
                secondary = pe_v
            tr = str(r.get(COL_TRACK, ""))
            return (primary, secondary, tr)

        best = max(grp, key=sort_key)
        out.append(best)
    return out


def aggregate_from_summary_rows(
    rows: list[dict[str, str]],
    *,
    metric: str,
    n_folds: int,
) -> tuple[dict[str, int], dict[str, set[str]]]:
    total: dict[str, int] = defaultdict(int)
    all_folds_sigs: dict[str, set[str]] = defaultdict(set)

    use_spear = metric in ("spearman", "both")
    use_pear = metric in ("pearson", "both")

    for row in rows:
        src = row.get(COL_SOURCE, "")
        if not _is_our_source(str(src)):
            continue

        ds = row.get(COL_DATASET, "")
        tgt = row.get(COL_TARGET, "")

        if use_spear:
            cell = row.get(COL_SPEAR_FEAT, "")
            run_id = row.get(COL_SPEAR_RUN, "")
            suffix = _run_suffix_after_target(str(run_id), str(tgt))
            sig = f"{ds}|{tgt}|{suffix}"
            counts = _parse_feature_counts(str(cell))
            for feat, c in counts.items():
                total[feat] += c
                if c == n_folds:
                    all_folds_sigs[feat].add(sig)

        if use_pear:
            cell = row.get(COL_PEAR_FEAT, "")
            run_id = row.get(COL_PEAR_RUN, "")
            suffix = _run_suffix_after_target(str(run_id), str(tgt))
            sig = f"{ds}|{tgt}|{suffix}"
            counts = _parse_feature_counts(str(cell))
            for feat, c in counts.items():
                total[feat] += c
                if c == n_folds:
                    all_folds_sigs[feat].add(sig)

    return dict(total), {k: v for k, v in all_folds_sigs.items()}


def write_feature_usage_from_summary(
    *,
    in_path: Path,
    out_path: Path,
    reference_json: Path,
    metric: str = "spearman",
    n_folds: int = 4,
    all_track_rows: bool = False,
) -> None:
    inp = in_path.resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    ref = reference_json.resolve()
    if not ref.is_file():
        print(f"Reference JSON not found: {ref}", file=sys.stderr)
        sys.exit(1)

    with open(inp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        need = [
            COL_SOURCE,
            COL_DATASET,
            COL_TARGET,
            COL_SPEAR_FEAT,
            COL_PEAR_FEAT,
            COL_SPEAR_RUN,
            COL_PEAR_RUN,
        ]
        missing = [c for c in need if c not in fieldnames]
        if missing:
            print(f"Missing columns in {inp}: {missing}", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    ref_rows = rows if all_track_rows else pick_best_track_rows(rows, metric=metric)

    canonical = set(canonical_feature_names(ref))
    usage, all_folds = aggregate_from_summary_rows(
        ref_rows, metric=metric, n_folds=n_folds
    )
    extras = sorted(k for k in usage if k not in canonical)
    if extras:
        print(
            f"Note: {len(extras)} feature name(s) not in reference (first few: {extras[:5]})",
            file=sys.stderr,
        )

    out_rows = build_feature_usage_rows(
        canonical=canonical, usage=usage, all_folds=all_folds
    )

    kept = sum(1 for r in rows if _is_our_source(str(r.get(COL_SOURCE, ""))))
    used = sum(1 for r in ref_rows if _is_our_source(str(r.get(COL_SOURCE, ""))))
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            write_feature_usage_csv(f, out_rows)
        mode = "all_track_rows" if all_track_rows else "pick_best_track"
        print(
            f"Wrote {len(out_rows)} rows from {len(rows)} input rows "
            f"({kept} developability-source rows); aggregated {used} row(s) after track filter "
            f"(mode={mode!r}) metric={metric!r} n_folds={n_folds} -> {out_path}",
            file=sys.stderr,
        )
    else:
        write_feature_usage_csv(sys.stdout, out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze batch-run aggregated CSVs (best metrics and feature usage)."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Aggregated CSV paths (shell glob OK). Omit when using --compare-batch-runs only.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSVs and plots (default: ./analysis_results in cwd).",
    )
    parser.add_argument(
        "--summary-name",
        default="best_metrics_summary.csv",
        help=(
            "Detailed best-metrics CSV per track (default: best_metrics_summary.csv)."
        ),
    )
    parser.add_argument(
        "--results-name",
        default="results.csv",
        help=(
            "Per-target best-model table (dataset × variant × target; "
            "default: results.csv)."
        ),
    )
    parser.add_argument(
        "--feature-usage-name",
        default="feature_usage.csv",
        help="Feature usage output filename (default: feature_usage.csv).",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=Path("pdgf38_results/AB-001.json"),
        help="Developability JSON for canonical feature_* list in feature_usage.csv.",
    )
    parser.add_argument(
        "--universe-features",
        type=Path,
        default=None,
        help="One feature name per line for both sides (fold stability).",
    )
    parser.add_argument(
        "--universe-features-our",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--rank-with-stability",
        choices=("none", "lexi", "composite"),
        default="none",
    )
    parser.add_argument(
        "--stability-permutations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--stability-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--stability-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-gpr",
        "--no_gpr",
        action="store_true",
    )
    parser.add_argument(
        "--best-across-tracks",
        action="store_true",
    )
    parser.add_argument(
        "--usage-metric",
        choices=("spearman", "pearson", "both"),
        default="spearman",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Fold count for feature_usage 'all folds' signatures.",
    )
    parser.add_argument(
        "--usage-all-track-rows",
        action="store_true",
        help="Sum feature usage across all Track rows (default: best track per group).",
    )
    parser.add_argument(
        "--skip-feature-usage",
        action="store_true",
        help="Do not write feature_usage.csv.",
    )
    parser.add_argument(
        "--compare-batch-runs",
        action="store_true",
        help="Also run cross-folder batch*folded* comparison under runs/.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
    )
    parser.add_argument(
        "--batch-compare-out",
        type=Path,
        default=Path("runs/best_batch_folder_per_dataset_target_spearman.csv"),
    )
    parser.add_argument(
        "--batch-compare-exclude",
        action="append",
        default=[],
        metavar="DIR_NAME",
    )
    parser.add_argument(
        "--skip-batch-feature-usage",
        action="store_true",
    )
    parser.add_argument(
        "--aggregate-our-glob",
        type=str,
        default=None,
        help="Glob of aggregated_*_our.csv for global feature usage (replaces summary-based usage).",
    )
    parser.add_argument(
        "--list-json-features-not-in-usage",
        type=Path,
        default=None,
        metavar="USAGE_CSV",
        help="Print feature_* names in --reference-json that are absent from this usage CSV.",
    )
    parser.add_argument(
        "--plot-fold-spearmans",
        action="store_true",
        help="Write per-dataset fold-Spearman strip plots from aggregated CSV inputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write optional analysis plots (e.g. fold-Spearman strip plots).",
    )
    parser.add_argument(
        "--plot-out-dir",
        type=Path,
        default=None,
        help="Directory for fold-Spearman plots (default: same as --out-dir).",
    )
    args = parser.parse_args()

    if args.list_json_features_not_in_usage is not None:
        try:
            validate_reference_features(args.reference_json.resolve())
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e)) from e
        raise SystemExit(
            print_json_features_missing_from_usage(
                reference_json=args.reference_json.resolve(),
                usage_csv=args.list_json_features_not_in_usage.resolve(),
            )
        )

    if args.aggregate_our_glob:
        try:
            validate_reference_features(args.reference_json.resolve())
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e)) from e
        paths = expand_glob_pattern(args.aggregate_our_glob)
        out_path = resolve_output_dir(args.out_dir) / args.feature_usage_name
        write_global_feature_usage_from_aggregated(
            reference_json=args.reference_json.resolve(),
            aggregated_paths=paths,
            out_path=out_path,
        )
        if not args.inputs and not args.compare_batch_runs:
            return

    if args.compare_batch_runs:
        run_compare_folded_batches(
            runs_dir=args.runs_dir,
            out=args.batch_compare_out,
            exclude=frozenset(args.batch_compare_exclude),
            skip_feature_usage=args.skip_batch_feature_usage,
        )

    if not args.inputs:
        if not args.compare_batch_runs:
            parser.error(
                "Provide aggregated CSV inputs, --compare-batch-runs, "
                "--aggregate-our-glob, or --list-json-features-not-in-usage."
            )
        return

    out_dir = resolve_output_dir(args.out_dir)
    summary_path = out_dir / args.summary_name
    results_path = out_dir / args.results_name
    run_best_metrics_from_aggregated(
        args.inputs,
        summary_path,
        results_out=results_path,
        universe_features=args.universe_features,
        universe_features_our=args.universe_features_our,
        rank_with_stability=args.rank_with_stability,
        stability_permutations=args.stability_permutations,
        stability_seed=args.stability_seed,
        stability_weight=args.stability_weight,
        eval_model=args.eval_model,
        no_gpr=bool(args.no_gpr),
        best_across_tracks=bool(args.best_across_tracks),
    )

    if not args.skip_feature_usage and not args.aggregate_our_glob:
        ref = args.reference_json.resolve()
        try:
            validate_reference_features(ref)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e)) from e
        write_feature_usage_from_summary(
            in_path=summary_path,
            out_path=out_dir / args.feature_usage_name,
            reference_json=ref,
            metric=args.usage_metric,
            n_folds=args.n_folds,
            all_track_rows=args.usage_all_track_rows,
        )
    elif not args.skip_feature_usage and args.aggregate_our_glob:
        ref = args.reference_json.resolve()
        try:
            validate_reference_features(ref)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(str(e)) from e
        paths = expand_glob_pattern(args.aggregate_our_glob)
        write_global_feature_usage_from_aggregated(
            reference_json=ref,
            aggregated_paths=paths,
            out_path=out_dir / args.feature_usage_name,
        )

    if args.plot_fold_spearmans and not args.no_plots:
        from analysis.plot_aggregated_fold_spearmans import run_plot_fold_spearmans

        plot_dir = resolve_output_dir(args.plot_out_dir) if args.plot_out_dir is not None else out_dir
        rc = run_plot_fold_spearmans(
            args.inputs,
            plot_dir,
            no_gpr=bool(args.no_gpr),
        )
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()

