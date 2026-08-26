#!/usr/bin/env python3
"""Nested-CV selection-bias check on leftover MMseqs2 / column folds.

For each outer fold k:
  inner train/test are the leftover original folds with outer-test names removed.
Inner AutoML ranks configurations by pooled inner OOF Spearman, then the winner
is refit on the full outer-train set and scored on the untouched outer-test fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from scipy.stats import ConstantInputWarning

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.fold_dirs import load_fold_dir_map, remap_fold_dir
from analysis.oof_predictions import (
    PAPER_BACKENDS,
    SKIP_EVAL_MODELS,
    pooled_metrics_from_oof,
)
from automl.feature_selectors import select_features_floating_sfs
from automl.pipeline_defaults import DEFAULT_FEATURES_FRAC
from automl.run_final_floating_sfs_batch import _make_scaled_fold_frames
from automl.run_fold_pipeline_config import (
    _evaluate_fold_models,
    oof_sidecar_path,
    write_oof_parquet,
)
from automl.utils import apply_minmax_to_train_test_features, parse_eval_hyperparameters_mapping

NESTED_ELIGIBLE: list[tuple[str, str]] = [
    ("ginkgo_ig_folded", "target_SEC_Monomer"),
    ("ginkgo_ig_folded", "target_SMAC"),
    ("ginkgo_ig_folded", "target_HIC"),
    ("ginkgo_ig_folded", "target_HAC"),
    ("ginkgo_ig_folded", "target_AC_SINS_pH6_0"),
    ("ginkgo_ig_folded", "target_AC_SINS_pH7_4"),
    ("ginkgo_ig_folded", "target_Tonset"),
    ("ginkgo_ig_folded", "target_Tm1"),
    ("ginkgo_ig_folded", "target_Tm2"),
    ("ginkgo_ig_folded", "target_PR_CHO"),
    ("ginkgo_ig_folded", "target_PR_Ova"),
    ("jain2017biophysical_folded_08_5", "target_ACSINS"),
    ("jain2017biophysical_folded_08_5", "target_BVPELISA"),
    ("jain2017biophysical_folded_08_5", "target_CICRT"),
    ("jain2017biophysical_folded_08_5", "target_CSIBLI"),
    ("jain2017biophysical_folded_08_5", "target_ELISA"),
    ("jain2017biophysical_folded_08_5", "target_HICRT"),
    ("jain2017biophysical_folded_08_5", "target_PSR"),
    ("jain2017biophysical_folded_08_5", "target_SAS"),
    ("jain2017biophysical_folded_08_5", "target_SGACSINS"),
    ("jain2017biophysical_folded_08_5", "target_SMACRT"),
    ("jain2017biophysical_folded_08_5", "target_Tm"),
    ("hutchinson2023enhancement_top200tm1_igg", "target_Tm1"),
    ("kraft2019herapin_relrt_folded_08_5", "target_Heparin_RT"),
]

RANDOM_SPLIT_STEMS = frozenset(
    {
        "ab21",
        "hutchinson2023enhancement_top200tm1_igg",
        "jetha2019homology_RT",
        "pdgf38",
    }
)


def _validate_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in PAPER_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of {sorted(PAPER_BACKENDS)}"
        )
    return value


STRUCTURE_VARIANTS = (1, 2, 3)
BACKEND_YAML_KEY_MODES = tuple(f"backend_{v}" for v in STRUCTURE_VARIANTS)
ALL_BACKEND_PAIRS_MODES = tuple(f"all_backend_{v}" for v in STRUCTURE_VARIANTS)


def backend_yaml_key_mode(structure_variant: int) -> str:
    variant = int(structure_variant)
    if variant not in STRUCTURE_VARIANTS:
        raise ValueError(
            f"structure_variant must be one of {STRUCTURE_VARIANTS}, got {variant}"
        )
    return f"backend_{variant}"


def all_backend_pairs_mode(structure_variant: int) -> str:
    return f"all_{backend_yaml_key_mode(structure_variant)}"


def nested_yaml_key(
    stem: str,
    backend: str = "abb2",
    *,
    suffix: str = "",
    mode: str = "backend_1",
) -> str:
    extra = str(suffix or "")
    mode_norm = str(mode or "backend_1").strip().lower()
    if mode_norm in {"backend_1", "default", ""}:
        backend = _validate_backend(backend)
        base = f"{stem}_{backend}_1"
    elif mode_norm == "stem":
        base = stem
    elif mode_norm.startswith("backend_") and mode_norm.split("_", 1)[1].isdigit():
        variant = int(mode_norm.split("_", 1)[1])
        if variant not in STRUCTURE_VARIANTS:
            raise ValueError(
                f"Unknown yaml_key_mode {mode!r}; expected one of "
                f"{BACKEND_YAML_KEY_MODES + ('stem',)}"
            )
        backend = _validate_backend(backend)
        base = f"{stem}_{backend}_{variant}"
    else:
        raise ValueError(
            f"Unknown yaml_key_mode {mode!r}; expected one of "
            f"{BACKEND_YAML_KEY_MODES + ('stem',)}"
        )
    if stem in RANDOM_SPLIT_STEMS:
        return f"{base}{extra}__rs42"
    return f"{base}{extra}"


def _split_csv_arg(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _filter_stems(
    pairs: Iterable[tuple[str, str]],
    *,
    exclude_stems: Iterable[str] = (),
    include_stems: Iterable[str] = (),
) -> list[tuple[str, str]]:
    excluded = {str(stem).strip() for stem in exclude_stems if str(stem).strip()}
    included = {str(stem).strip() for stem in include_stems if str(stem).strip()}
    out: list[tuple[str, str]] = []
    for stem, target in pairs:
        if stem in excluded:
            continue
        if included and stem not in included:
            continue
        out.append((stem, target))
    return out


def discover_all_backend_pairs(
    automl_root: Path,
    *,
    backend: str = "abb2",
    exclude_stems: Iterable[str] = (),
    include_stems: Iterable[str] = (),
    yaml_key_suffix: str = "",
    yaml_key_mode: str = "backend_1",
) -> list[tuple[str, str]]:
    """Discover canonical backend_1 dataset/target pairs from paper result JSONs."""
    backend = _validate_backend(backend)
    pairs: set[tuple[str, str]] = set()
    root = Path(automl_root)
    if not root.is_dir():
        return []
    for json_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for jp in sorted(json_dir.glob("*.json")):
            try:
                data = json.loads(jp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            stem = str(data.get("dataset_stem") or "").strip()
            target = str(data.get("target_col") or "").strip()
            yaml_key = str(data.get("dataset_yaml_key") or json_dir.name).strip()
            if (
                not stem
                or not target
                or yaml_key
                != nested_yaml_key(
                    stem,
                    backend,
                    suffix=yaml_key_suffix,
                    mode=yaml_key_mode,
                )
                or json_dir.name != yaml_key
            ):
                continue
            pairs.add((stem, target))
    return _filter_stems(
        sorted(pairs), exclude_stems=exclude_stems, include_stems=include_stems
    )


def discover_all_abb2_pairs(
    automl_root: Path,
    *,
    exclude_stems: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Backward-compatible ABB2 pair discovery."""
    return discover_all_backend_pairs(
        automl_root, backend="abb2", exclude_stems=exclude_stems
    )


def resolve_nested_pairs(
    automl_root: Path,
    *,
    backend: str = "abb2",
    pairs_mode: str = "default",
    exclude_stems: Iterable[str] = (),
    include_stems: Iterable[str] = (),
    yaml_key_suffix: str = "",
    yaml_key_mode: str = "backend_1",
) -> list[tuple[str, str]]:
    backend = _validate_backend(backend)
    mode = str(pairs_mode).strip().lower()
    if mode == "default":
        return _filter_stems(
            NESTED_ELIGIBLE,
            exclude_stems=exclude_stems,
            include_stems=include_stems,
        )
    if mode in set(ALL_BACKEND_PAIRS_MODES) | {"all_backend_1", "all_abb2_1"}:
        if mode == "all_abb2_1" and backend != "abb2":
            raise ValueError("pairs_mode='all_abb2_1' is only valid for backend='abb2'")
        resolved_yaml_mode = yaml_key_mode
        # all_backend_N selects ProperMAb structure variant N. TAP yaml keys
        # are stem-based; do not override yaml_key_mode="stem".
        requested_mode = str(yaml_key_mode or "").strip().lower()
        variant_token = mode.rsplit("_", 1)[-1]
        if (
            requested_mode != "stem"
            and mode.startswith("all_backend_")
            and variant_token.isdigit()
        ):
            resolved_yaml_mode = f"backend_{int(variant_token)}"
        return discover_all_backend_pairs(
            automl_root,
            backend=backend,
            exclude_stems=exclude_stems,
            include_stems=include_stems,
            yaml_key_suffix=yaml_key_suffix,
            yaml_key_mode=resolved_yaml_mode,
        )
    raise ValueError(
        f"Unknown pairs_mode {pairs_mode!r}; expected default, "
        f"{', '.join(ALL_BACKEND_PAIRS_MODES)}, or all_abb2_1"
    )


def write_inner_fold_dir(orig_fold_dir: Path, *, outer_k: int, dest: Path) -> Path:
    orig = Path(orig_fold_dir)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    meta = json.loads((orig / "meta.json").read_text())
    n_splits = int(meta["n_splits"])
    if outer_k < 0 or outer_k >= n_splits:
        raise ValueError(f"outer_k={outer_k} out of range for n_splits={n_splits}")

    outer_test = pd.read_parquet(orig / f"fold_{outer_k}_test.parquet")
    outer_names = set(outer_test["name"].astype(str))
    inner_indices = [i for i in range(n_splits) if i != outer_k]
    segment_sizes: list[int] = []
    for new_k, old_k in enumerate(inner_indices):
        test_df = pd.read_parquet(orig / f"fold_{old_k}_test.parquet")
        train_df = pd.read_parquet(orig / f"fold_{old_k}_train.parquet")
        train_df = train_df.loc[~train_df["name"].astype(str).isin(outer_names)].copy()
        test_df = test_df.loc[~test_df["name"].astype(str).isin(outer_names)].copy()
        if len(test_df) < 2:
            raise ValueError(
                f"inner test fold too small after dropping outer-test names: "
                f"outer_k={outer_k} old_k={old_k} n_test={len(test_df)}"
            )
        if len(train_df) < 2:
            raise ValueError(
                f"inner train fold too small after dropping outer-test names: "
                f"outer_k={outer_k} old_k={old_k} n_train={len(train_df)}"
            )
        train_df.to_parquet(dest / f"fold_{new_k}_train.parquet", index=False)
        test_df.to_parquet(dest / f"fold_{new_k}_test.parquet", index=False)
        segment_sizes.append(int(len(test_df)))

    inner_meta = dict(meta)
    inner_meta["n_splits"] = len(inner_indices)
    inner_meta["segment_sizes"] = segment_sizes
    inner_meta["nested_outer_fold"] = int(outer_k)
    inner_meta["nested_source_fold_dir"] = str(orig.resolve())
    (dest / "meta.json").write_text(json.dumps(inner_meta, indent=2))
    return dest


def _unique_configs_from_jsons(json_paths: Iterable[Path], target_col: str) -> list[dict[str, Any]]:
    seen: dict[tuple, dict[str, Any]] = {}
    for jp in json_paths:
        try:
            data = json.loads(Path(jp).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if str(data.get("target_col")) != str(target_col):
            continue
        try:
            frac = float(data.get("eval_features_frac", DEFAULT_FEATURES_FRAC))
        except (TypeError, ValueError):
            frac = float(DEFAULT_FEATURES_FRAC)
        key = (
            str(data.get("selector_name") or ""),
            str(data.get("model_type") or ""),
            frac,
            str(data.get("pipeline_track_name") or ""),
            json.dumps(data.get("selector_hyperparameters") or {}, sort_keys=True),
        )
        if key in seen:
            continue
        eval_models = [
            str(m)
            for m in (data.get("eval_models") or [])
            if str(m).strip().lower() not in SKIP_EVAL_MODELS
        ]
        if not eval_models and isinstance(data.get("evaluation"), dict):
            eval_models = [
                str(m)
                for m in data["evaluation"]
                if str(m).strip().lower() not in SKIP_EVAL_MODELS
            ]
        eval_hyperparameters = data.get("eval_hyperparameters") or {}
        if isinstance(eval_hyperparameters, dict):
            eval_hyperparameters = {
                str(model): params
                for model, params in eval_hyperparameters.items()
                if str(model).strip().lower() not in SKIP_EVAL_MODELS
            }
        seen[key] = {
            "selector_name": key[0],
            "model_type": key[1],
            "eval_features_frac": frac,
            "pipeline_track_name": key[3],
            "selector_hyperparameters": data.get("selector_hyperparameters") or {},
            "eval_hyperparameters": eval_hyperparameters,
            "eval_models": eval_models or ["linear", "elasticnet", "randomforest", "svm", "knn"],
            "correlation_min_abs_rho": data.get("correlation_min_abs_rho", "none"),
            "random_state": int(data.get("random_state", 42)),
        }
    return [seen[k] for k in sorted(seen)]


def discover_nested_jobs(
    repo_root: Path,
    out_root: Path,
    *,
    backend: str = "abb2",
    automl_root: Path | None = None,
    final_floating_only: bool = False,
    pairs: Iterable[tuple[str, str]] | None = None,
    yaml_key_suffix: str = "",
    yaml_key_mode: str = "backend_1",
    fold_dir_map: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    repo_root = Path(repo_root)
    backend = _validate_backend(backend)
    automl_root = (
        Path(automl_root)
        if automl_root is not None
        else repo_root / PAPER_BACKENDS[backend] / "automl"
    )
    jobs: list[dict[str, str]] = []
    selected_pairs = list(NESTED_ELIGIBLE if pairs is None else pairs)
    for stem, target in selected_pairs:
        yaml_key = nested_yaml_key(
            stem, backend, suffix=yaml_key_suffix, mode=yaml_key_mode
        )
        json_dir = automl_root / yaml_key
        jsons = sorted(json_dir.glob("*.json")) if json_dir.is_dir() else []
        orig_fold_dir = None
        for jp in jsons:
            try:
                data = json.loads(jp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if str(data.get("target_col")) != target:
                continue
            orig_fold_dir = remap_fold_dir(Path(str(data["fold_dir"])), fold_dir_map)
            break
        if orig_fold_dir is None or not (orig_fold_dir / "meta.json").is_file():
            continue
        meta = json.loads((orig_fold_dir / "meta.json").read_text())
        n_splits = int(meta["n_splits"])
        configs = _unique_configs_from_jsons(jsons, target)
        configs = [
            cfg
            for cfg in configs
            if (cfg["selector_name"] == "final_floating_sfs") == final_floating_only
        ]
        pair_root = Path(out_root) / yaml_key / target
        for outer_k in range(n_splits):
            inner_dir = pair_root / f"outer_{outer_k}" / "inner_folds"
            for cfg in configs:
                cfg_hash = hashlib.sha1(
                    json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:12]
                for inner_k in range(n_splits - 1):
                    out_json = (
                        pair_root
                        / f"outer_{outer_k}"
                        / "inner_results"
                        / (
                            f"inner{inner_k}__{cfg['selector_name']}__{cfg['model_type']}"
                            f"__frac{int(round(cfg['eval_features_frac'] * 100)):03d}"
                            f"__h{cfg_hash}.json"
                        )
                    )
                    jobs.append(
                        {
                            "stem": stem,
                            "target": target,
                            "yaml_key": yaml_key,
                            "orig_fold_dir": str(orig_fold_dir.resolve()),
                            "inner_fold_dir": str(inner_dir.resolve()),
                            "outer_k": str(outer_k),
                            "inner_k": str(inner_k),
                            "n_splits": str(n_splits),
                            "selector_name": cfg["selector_name"],
                            "model_type": cfg["model_type"],
                            "eval_features_frac": str(cfg["eval_features_frac"]),
                            "pipeline_track_name": cfg["pipeline_track_name"],
                            "eval_models": ",".join(cfg["eval_models"]),
                            "random_state": str(cfg["random_state"]),
                            "correlation_min_abs_rho": str(cfg.get("correlation_min_abs_rho") or "none"),
                            "selector_hyperparameters": json.dumps(
                                cfg["selector_hyperparameters"], separators=(",", ":")
                            ),
                            "eval_hyperparameters": json.dumps(
                                cfg["eval_hyperparameters"], separators=(",", ":")
                            ),
                            "output_json": str(out_json),
                        }
                    )
    return jobs


def prepare_inner_fold_dirs(jobs: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str]] = set()
    for j in jobs:
        key = (j["orig_fold_dir"], j["outer_k"])
        if key in seen:
            continue
        seen.add(key)
        write_inner_fold_dir(
            Path(j["orig_fold_dir"]),
            outer_k=int(j["outer_k"]),
            dest=Path(j["inner_fold_dir"]),
        )


def write_nested_jobs_tsv(path: Path, jobs: list[dict[str, str]]) -> None:
    """Write a GNU-parallel TSV. JSON fields must stay unquoted: csv.writer
    would double the quotes, and --colsep tab would then pass invalid JSON.
    """
    if not jobs:
        Path(path).write_text("")
        return
    cols = list(jobs[0].keys())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for j in jobs:
            f.write("\t".join(j[c] for c in cols) + "\n")


def load_nested_jobs_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [{k: str(v or "") for k, v in row.items()} for row in csv.DictReader(f, delimiter="\t")]


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _semantic_result_key(data: dict[str, Any]) -> tuple[Any, ...]:
    selector = str(data.get("selector_name") or "")
    eval_models_raw = data.get("eval_models") or []
    if isinstance(eval_models_raw, str):
        eval_models = tuple(x.strip() for x in eval_models_raw.split(",") if x.strip())
    else:
        eval_models = tuple(str(x) for x in eval_models_raw)
    core: tuple[Any, ...] = (
        str(data.get("dataset_yaml_key") or ""),
        str(data.get("target_col") or ""),
        int(data.get("fold_index", -1)),
        selector,
        str(data.get("model_type") or ""),
        float(data.get("eval_features_frac", DEFAULT_FEATURES_FRAC)),
        str(data.get("pipeline_track_name") or ""),
        eval_models,
        int(data.get("random_state", 42)),
    )
    if selector == "final_floating_sfs":
        return core
    correlation = data.get("correlation_min_abs_rho")
    correlation = "none" if correlation in (None, "", "none") else str(correlation)
    return (
        *core,
        correlation,
        json.dumps(
            _json_mapping(data.get("selector_hyperparameters")),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            _json_mapping(data.get("eval_hyperparameters")),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _semantic_job_key(job: dict[str, str]) -> tuple[Any, ...]:
    return _semantic_result_key(
        {
            "dataset_yaml_key": job.get("yaml_key"),
            "target_col": job.get("target"),
            "fold_index": job.get("inner_k"),
            "selector_name": job.get("selector_name"),
            "model_type": job.get("model_type"),
            "eval_features_frac": job.get("eval_features_frac"),
            "pipeline_track_name": job.get("pipeline_track_name"),
            "eval_models": job.get("eval_models"),
            "random_state": job.get("random_state"),
            "correlation_min_abs_rho": job.get("correlation_min_abs_rho"),
            "selector_hyperparameters": job.get("selector_hyperparameters"),
            "eval_hyperparameters": job.get("eval_hyperparameters"),
        }
    )


def pending_nested_jobs(
    jobs: list[dict[str, str]], *, require_oof: bool = False
) -> list[dict[str, str]]:
    completed_by_dir: dict[Path, set[tuple[Any, ...]]] = {}
    pending: list[dict[str, str]] = []
    for job in jobs:
        output = Path(job.get("output_json", ""))
        complete = output.is_file()
        if require_oof:
            complete = complete and oof_sidecar_path(output).is_file()
        if not complete:
            result_dir = output.parent
            if result_dir not in completed_by_dir:
                completed: set[tuple[Any, ...]] = set()
                for candidate in sorted(result_dir.glob("*.json")):
                    if require_oof and not oof_sidecar_path(candidate).is_file():
                        continue
                    try:
                        data = json.loads(candidate.read_text())
                        completed.add(_semantic_result_key(data))
                    except (json.JSONDecodeError, OSError, TypeError, ValueError):
                        continue
                completed_by_dir[result_dir] = completed
            complete = _semantic_job_key(job) in completed_by_dir[result_dir]
        if not complete:
            pending.append(job)
    return pending


def _floating_vote_map(
    inner_result_dir: Path,
    *,
    track_name: str,
    inner_k: int | None,
) -> dict[str, int]:
    """Count selected features from completed regular-grid inner results."""
    counts: defaultdict[str, int] = defaultdict(int)
    seen: set[tuple[Any, ...]] = set()
    for jp in sorted(Path(inner_result_dir).glob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if str(data.get("selector_name") or "") == "final_floating_sfs":
            continue
        if str(data.get("pipeline_track_name") or "") != str(track_name or ""):
            continue
        if inner_k is not None and int(data.get("fold_index", -1)) != int(inner_k):
            continue
        semantic_key = _semantic_result_key(data)
        if semantic_key in seen:
            continue
        seen.add(semantic_key)
        feats = data.get("selected_features")
        if not isinstance(feats, list):
            continue
        for feat in dict.fromkeys(str(f) for f in feats):
            counts[feat] += 1
    return dict(counts)


def _rank_voted_features(vote_map: dict[str, int]) -> list[str]:
    return [
        name
        for name, _ in sorted(
            ((str(name), int(votes)) for name, votes in vote_map.items()),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def run_nested_floating_sfs_job(
    *,
    inner_fold_dir: Path,
    inner_k: int,
    dataset_stem: str,
    target_col: str,
    dataset_yaml_key: str,
    selection_model: str,
    max_feature_fraction: float,
    track_name: str,
    eval_models: list[str],
    random_state: int,
    eval_hyperparameters: dict[str, Any] | None,
    output_json: Path,
) -> Path:
    """Run the post-grid floating selector for one nested inner fold."""
    fold_dir = Path(inner_fold_dir)
    output_json = Path(output_json)
    inner_result_dir = output_json.parent
    train_pq = fold_dir / f"fold_{inner_k}_train.parquet"
    test_pq = fold_dir / f"fold_{inner_k}_test.parquet"
    vote_map = _floating_vote_map(
        inner_result_dir, track_name=track_name, inner_k=inner_k
    )
    voted_features = _rank_voted_features(vote_map)
    eval_models = [
        str(m) for m in eval_models if str(m).strip().lower() not in SKIP_EVAL_MODELS
    ]
    eval_raw = eval_hyperparameters or {}
    eval_raw = {
        str(k): v
        for k, v in eval_raw.items()
        if str(k).strip().lower() not in SKIP_EVAL_MODELS
    }
    eval_hp = parse_eval_hyperparameters_mapping(eval_raw if eval_raw else None)

    payload: dict[str, Any] = {
        "fold_dir": str(fold_dir),
        "fold_index": int(inner_k),
        "random_state": int(random_state),
        "selector_name": "final_floating_sfs",
        "target_col": str(target_col),
        "feature_scaling": "minmax_train_fit_transform_test",
        "feature_selection_pipeline": "final_floating_sfs",
        "model_type": str(selection_model),
        "selection_max_features": None,
        "selected_features": [],
        "n_selected_features": 0,
        "features_after_vote_aggregation": voted_features,
        "features_after_vote_aggregation_count": len(voted_features),
        "after_step": {"final_floating_sfs": []},
        "evaluation": None,
        "eval_models": eval_models,
        "eval_features_frac": float(max_feature_fraction),
        "dataset_stem": str(dataset_stem),
        "dataset_yaml_key": str(dataset_yaml_key),
        "pipeline_track_name": str(track_name),
        "eval_hyperparameters": eval_raw,
        "final_floating_sfs_summary": {
            "max_feature_fraction": float(max_feature_fraction),
            "selection_model": str(selection_model),
            "vote_counts_by_feature": {
                name: vote_map[name] for name in voted_features
            },
            "source": "nested_inner_post_grid_vote_aggregation",
        },
    }
    oof_df = pd.DataFrame()
    try:
        train_df = pd.read_parquet(train_pq)
        test_df = pd.read_parquet(test_pq)
        candidate_cap = max(1, int(float(max_feature_fraction) * len(train_df)))
        present = [
            feat
            for feat in voted_features
            if feat in train_df.columns and feat in test_df.columns
        ]
        top_candidates = present[:candidate_cap]
        payload["selection_max_features"] = candidate_cap
        payload["features_after_vote_aggregation"] = top_candidates
        payload["features_after_vote_aggregation_count"] = len(top_candidates)
        payload["final_floating_sfs_summary"].update(
            {
                "n_train": int(len(train_df)),
                "candidate_cap": candidate_cap,
                "n_voted_features_total": len(voted_features),
                "n_candidates_present_in_fold": len(present),
                "n_candidates_used_for_sfs": len(top_candidates),
                "candidate_features_for_sfs": list(top_candidates),
            }
        )
        if not top_candidates:
            raise ValueError("No regular-grid selected features available for floating SFS")
        train_scaled, test_scaled = _make_scaled_fold_frames(
            train_df, test_df, candidate_features=top_candidates
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            selected = select_features_floating_sfs(
                train_scaled,
                target_col=str(target_col),
                candidate_features=top_candidates,
                n_features_to_select=len(top_candidates),
                random_state=int(random_state),
                n_jobs=1,
                model_type=str(selection_model),
            )
        selected = [f for f in selected if f in top_candidates]
        if not selected:
            raise ValueError("Floating SFS selected no features")
        payload["selected_features"] = selected
        payload["n_selected_features"] = len(selected)
        payload["after_step"] = {"final_floating_sfs": selected}
        payload["final_floating_sfs_summary"]["n_selected_features"] = len(selected)
        evaluation, oof_df = _evaluate_fold_models(
            train_scaled,
            test_scaled,
            target_col=str(target_col),
            feature_cols=selected,
            eval_models=eval_models,
            random_state=int(random_state),
            features_frac=float(max_feature_fraction),
            eval_hp_by_model=eval_hp,
        )
        payload["evaluation"] = evaluation
    except Exception as exc:
        payload["final_floating_sfs_summary"]["error"] = str(exc)
        payload["evaluation"] = {
            model: {"error": str(exc), "n_features": 0} for model in eval_models
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, default=str))
    if len(oof_df):
        write_oof_parquet(
            oof_sidecar_path(output_json),
            oof_df,
            extra={
                "fold_index": int(inner_k),
                "dataset_yaml_key": str(dataset_yaml_key),
                "dataset_stem": str(dataset_stem),
                "target_col": str(target_col),
                "selector_name": "final_floating_sfs",
                "model_type": str(selection_model),
                "eval_features_frac": float(max_feature_fraction),
                "pipeline_track_name": str(track_name),
            },
        )
    return output_json


def _inner_pooled_winners(inner_result_dir: Path) -> dict[str, Any]:
    by_cfg: dict[tuple, list[pd.DataFrame]] = defaultdict(list)
    folds_by_cfg: defaultdict[tuple, set[int]] = defaultdict(set)
    meta_by_cfg: dict[tuple, dict[str, Any]] = {}
    meta_path = Path(inner_result_dir).parent / "inner_folds" / "meta.json"
    expected_folds = None
    if meta_path.is_file():
        expected_folds = int(json.loads(meta_path.read_text())["n_splits"])
    seen: set[tuple[Any, ...]] = set()
    for jp in sorted(Path(inner_result_dir).glob("*.json")):
        data = json.loads(jp.read_text())
        pq = oof_sidecar_path(jp)
        if not pq.is_file():
            continue
        semantic_key = _semantic_result_key(data)
        if semantic_key in seen:
            continue
        seen.add(semantic_key)
        oof = pd.read_parquet(pq)
        if oof is None or len(oof) == 0:
            continue
        frac = float(data.get("eval_features_frac", DEFAULT_FEATURES_FRAC))
        for em, sub in oof.groupby("eval_model", sort=False):
            if str(em).strip().lower() in SKIP_EVAL_MODELS:
                continue
            key = (
                str(data.get("pipeline_track_name") or ""),
                str(data.get("selector_name") or ""),
                str(data.get("model_type") or ""),
                str(em),
                frac,
            )
            by_cfg[key].append(sub)
            folds_by_cfg[key].add(int(data.get("fold_index", -1)))
            meta_by_cfg[key] = {
                "pipeline_track_name": key[0],
                "selector_name": key[1],
                "model_type": key[2],
                "eval_model": key[3],
                "eval_features_frac": key[4],
                "selector_hyperparameters": data.get("selector_hyperparameters") or {},
                "eval_hyperparameters": data.get("eval_hyperparameters") or {},
                "random_state": int(data.get("random_state", 42)),
            }
    ranked: list[tuple[float, tuple, dict[str, Any]]] = []
    for key, parts in by_cfg.items():
        if expected_folds is not None and len(folds_by_cfg[key]) != expected_folds:
            continue
        cat = pd.concat(parts, ignore_index=True)
        metrics = pooled_metrics_from_oof(cat)
        sp = metrics["Spearman_pooled_oof"]
        if sp is None:
            continue
        ranked.append((float(sp), key, {**meta_by_cfg[key], **metrics, "inner_oof_rows": int(len(cat))}))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    if not ranked:
        return {}
    winner = ranked[0][2]
    if winner.get("selector_name") == "final_floating_sfs":
        winner["floating_vote_counts"] = _floating_vote_map(
            Path(inner_result_dir),
            track_name=str(winner.get("pipeline_track_name") or ""),
            inner_k=None,
        )
    return winner


def refit_winner_on_outer(
    orig_fold_dir: Path,
    outer_k: int,
    winner: dict[str, Any],
    *,
    dest_json: Path,
) -> dict[str, Any]:
    from automl.feature_selectors import (
        parse_selector_hyperparameters_mapping,
        run_feature_selection_on_one_fold,
    )

    orig = Path(orig_fold_dir)
    train_df = pd.read_parquet(orig / f"fold_{outer_k}_train.parquet")
    test_df = pd.read_parquet(orig / f"fold_{outer_k}_test.parquet")
    meta = json.loads((orig / "meta.json").read_text())
    target_col = str(meta["target_col"])
    candidate_features = [str(c) for c in meta["feature_cols"]]
    random_state = int(winner.get("random_state", meta.get("random_state", 42)))
    frac = float(winner["eval_features_frac"])
    sel = str(winner["selector_name"])
    selection_max_features = max(1, int(frac * len(train_df)))

    eval_raw = winner.get("eval_hyperparameters") or {}
    if isinstance(eval_raw, dict):
        eval_raw = {
            k: v for k, v in eval_raw.items() if str(k).strip().lower() not in SKIP_EVAL_MODELS
        }
    try:
        eval_hp = parse_eval_hyperparameters_mapping(eval_raw if eval_raw else None)
    except (ValueError, TypeError):
        eval_hp = parse_eval_hyperparameters_mapping(None)

    if sel == "final_floating_sfs":
        votes = winner.get("floating_vote_counts") or {}
        voted = _rank_voted_features(votes)
        present = [
            feat
            for feat in voted
            if feat in candidate_features
            and feat in train_df.columns
            and feat in test_df.columns
        ]
        top_candidates = present[:selection_max_features]
        if not top_candidates:
            raise ValueError("No voted inner-grid features available for outer floating SFS")
        train_k, test_k = _make_scaled_fold_frames(
            train_df, test_df, candidate_features=top_candidates
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            selected = select_features_floating_sfs(
                train_k,
                target_col=target_col,
                candidate_features=top_candidates,
                n_features_to_select=len(top_candidates),
                random_state=random_state,
                n_jobs=1,
                model_type=str(winner["model_type"]),
            )
        selected = [f for f in selected if f in top_candidates]
        result = {
            "selected_features": selected,
            "n_selected_features": len(selected),
            "features_after_vote_aggregation": top_candidates,
        }
    else:
        hp_obj = winner.get("selector_hyperparameters") or {}
        hp_kwargs = parse_selector_hyperparameters_mapping(sel, hp_obj) if hp_obj else {}
        train_k, test_k, result = run_feature_selection_on_one_fold(
            train_df_k=train_df,
            test_df_k=test_df,
            target_col=target_col,
            feature_selection_pipeline=sel,
            model_type=str(winner["model_type"]),
            candidate_features=candidate_features,
            random_state=random_state,
            selection_max_features=selection_max_features,
            verbose=False,
            **hp_kwargs,
        )
    evaluation, oof_df = _evaluate_fold_models(
        train_k,
        test_k,
        target_col=target_col,
        feature_cols=list(result.get("selected_features", [])),
        eval_models=[str(winner["eval_model"])],
        random_state=random_state,
        features_frac=frac,
        eval_hp_by_model=eval_hp,
    )
    dest_json = Path(dest_json)
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nested_stage": "outer_test",
        "outer_fold": int(outer_k),
        "winner": {
            k: winner[k]
            for k in winner
            if k not in {"selector_hyperparameters", "floating_vote_counts"}
        },
        "selected_features": result.get("selected_features"),
        "evaluation": evaluation,
    }
    dest_json.write_text(json.dumps(payload, indent=2, default=str))
    if len(oof_df):
        write_oof_parquet(
            oof_sidecar_path(dest_json),
            oof_df,
            extra={
                "fold_index": int(outer_k),
                "dataset_stem": "",
                "target_col": target_col,
                "selector_name": sel,
                "model_type": str(winner["model_type"]),
                "eval_features_frac": frac,
                "pipeline_track_name": str(winner.get("pipeline_track_name") or ""),
                "eval_model": str(winner["eval_model"]),
            },
        )
    ev = evaluation.get(str(winner["eval_model"])) or {}
    return {
        "outer_fold": int(outer_k),
        "winner_run": (
            f"{target_col}-{winner['selector_name']}-{winner['model_type']}-"
            f"{winner['eval_model']}-frac{int(round(frac * 100)):03d}"
        ),
        "outer_spearman": ev.get("spearman_rho"),
        "inner_spearman_pooled": winner.get("Spearman_pooled_oof"),
        "n_outer_test": ev.get("n_test"),
    }


def finish_outer_folds(pair_root: Path, orig_fold_dir: Path) -> dict[str, Any]:
    orig = Path(orig_fold_dir)
    meta = json.loads((orig / "meta.json").read_text())
    n_splits = int(meta["n_splits"])
    outer_rows: list[pd.DataFrame] = []
    winners: list[str] = []
    per_fold: list[dict[str, Any]] = []
    for outer_k in range(n_splits):
        inner_res = Path(pair_root) / f"outer_{outer_k}" / "inner_results"
        winner = _inner_pooled_winners(inner_res)
        if not winner:
            continue
        dest = Path(pair_root) / f"outer_{outer_k}" / "outer_result.json"
        info = refit_winner_on_outer(orig, outer_k, winner, dest_json=dest)
        per_fold.append(info)
        winners.append(str(info["winner_run"]))
        pq = oof_sidecar_path(dest)
        if pq.is_file():
            outer_rows.append(pd.read_parquet(pq))
    if len(per_fold) != n_splits or len(outer_rows) != n_splits:
        raise RuntimeError(
            f"Nested outer CV incomplete: completed {len(per_fold)}/{n_splits} "
            f"outer folds with {len(outer_rows)} OOF sidecars"
        )
    pooled = pooled_metrics_from_oof(pd.concat(outer_rows, ignore_index=True) if outer_rows else pd.DataFrame())
    n_unique = len(set(winners))
    return {
        "n_outer_folds_completed": len(per_fold),
        "n_unique_winners": n_unique,
        "winner_stability": (1.0 - (n_unique - 1) / max(len(winners) - 1, 1)) if winners else None,
        "winners_by_outer_fold": winners,
        "per_fold": per_fold,
        **pooled,
    }


def write_nested_report(
    nested_root: Path,
    *,
    automl_root: Path,
    dest: Path,
    backend: str = "abb2",
    flat_results_csv: Path | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    yaml_key_suffix: str = "",
    yaml_key_mode: str = "backend_1",
    flat_variant: str | None = None,
) -> pd.DataFrame:
    backend = _validate_backend(backend)
    rows: list[dict[str, Any]] = []
    automl_root = Path(automl_root)
    wanted_variant = (
        str(flat_variant).strip() if flat_variant is not None else f"{backend}_1"
    )
    flat: dict[tuple[str, str], float] = {}
    if flat_results_csv is not None and Path(flat_results_csv).is_file():
        fdf = pd.read_csv(flat_results_csv)
        if "Variant" in fdf.columns:
            variant = (
                fdf["Variant"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace({"nan": "", "None": ""})
            )
            fdf = fdf.loc[variant == wanted_variant]
        for _, r in fdf.iterrows():
            try:
                flat[(str(r.get("Dataset_stem", "")), str(r.get("Target_col", "")))] = float(r["Spearman"])
            except (TypeError, ValueError, KeyError):
                continue
    selected_pairs = list(NESTED_ELIGIBLE if pairs is None else pairs)
    for stem, target in selected_pairs:
        yaml_key = nested_yaml_key(
            stem, backend, suffix=yaml_key_suffix, mode=yaml_key_mode
        )
        pair_root = Path(nested_root) / yaml_key / target
        summary_path = pair_root / "nested_summary.json"
        if not summary_path.is_file():
            continue
        nested = json.loads(summary_path.read_text())
        nested_sp = nested.get("Spearman_pooled_oof")
        flat_sp = flat.get((stem, target))
        gap = None
        if nested_sp is not None and flat_sp is not None:
            try:
                gap = float(flat_sp) - float(nested_sp)
            except (TypeError, ValueError):
                gap = None
        rows.append(
            {
                "Dataset_stem": stem,
                "Target_col": target,
                "yaml_key": yaml_key,
                "nested_Spearman_pooled_oof": nested_sp,
                "flat_Spearman": flat_sp,
                "optimism_gap_flat_minus_nested": gap,
                "n_oof": nested.get("n_oof"),
                "n_unique_winners": nested.get("n_unique_winners"),
                "winner_stability": nested.get("winner_stability"),
                "winners_by_outer_fold": json.dumps(nested.get("winners_by_outer_fold") or []),
            }
        )
    df = pd.DataFrame(rows)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def _cmd_pending(args: argparse.Namespace) -> int:
    jobs = load_nested_jobs_tsv(Path(args.jobs_file))
    pending = pending_nested_jobs(jobs, require_oof=bool(args.require_oof))
    dest = Path(args.out)
    write_nested_jobs_tsv(dest, pending)
    print(
        f"nested resume: {len(pending)}/{len(jobs)} inner job(s) remaining -> {dest}",
        file=sys.stderr,
    )
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    automl_root = (
        Path(args.automl_root)
        if args.automl_root is not None
        else Path(args.repo_root) / PAPER_BACKENDS[args.backend] / "automl"
    )
    suffix = str(args.yaml_key_suffix or "")
    yaml_key_mode = str(args.yaml_key_mode or "backend_1")
    fold_dir_map = load_fold_dir_map(getattr(args, "fold_dir_map", None))
    pairs = resolve_nested_pairs(
        automl_root,
        backend=args.backend,
        pairs_mode=args.pairs_mode,
        exclude_stems=_split_csv_arg(args.exclude_stems),
        include_stems=_split_csv_arg(args.include_stems),
        yaml_key_suffix=suffix,
        yaml_key_mode=yaml_key_mode,
    )
    jobs = discover_nested_jobs(
        Path(args.repo_root),
        Path(args.out_dir),
        backend=args.backend,
        automl_root=automl_root,
        pairs=pairs,
        yaml_key_suffix=suffix,
        yaml_key_mode=yaml_key_mode,
        fold_dir_map=fold_dir_map,
    )
    write_nested_jobs_tsv(Path(args.jobs_file), jobs)
    floating_jobs = discover_nested_jobs(
        Path(args.repo_root),
        Path(args.out_dir),
        backend=args.backend,
        automl_root=automl_root,
        final_floating_only=True,
        pairs=pairs,
        yaml_key_suffix=suffix,
        yaml_key_mode=yaml_key_mode,
        fold_dir_map=fold_dir_map,
    )
    if args.floating_jobs_file is not None:
        write_nested_jobs_tsv(Path(args.floating_jobs_file), floating_jobs)
    pairs = {(j["stem"], j["target"]) for j in jobs}
    print(
        f"Wrote {len(jobs)} nested inner job(s) for {len(pairs)} pair(s) "
        f"(mode={args.pairs_mode}) -> {args.jobs_file}",
        file=sys.stderr,
    )
    if args.floating_jobs_file is not None:
        print(
            f"Wrote {len(floating_jobs)} nested post-grid floating-SFS job(s) "
            f"-> {args.floating_jobs_file}",
            file=sys.stderr,
        )
    if args.prepare_inner:
        prepare_inner_fold_dirs(jobs)
        print("Prepared inner fold directories.", file=sys.stderr)
    return 0 if jobs else 1


def _cmd_floating_sfs_job(args: argparse.Namespace) -> int:
    try:
        eval_models = [
            m.strip() for m in str(args.eval_models).split(",") if m.strip()
        ]
        eval_hyperparameters = json.loads(args.eval_hyperparameters or "{}")
        out = run_nested_floating_sfs_job(
            inner_fold_dir=Path(args.inner_fold_dir),
            inner_k=int(args.inner_k),
            dataset_stem=str(args.dataset_stem),
            target_col=str(args.target),
            dataset_yaml_key=str(args.yaml_key),
            selection_model=str(args.selection_model),
            max_feature_fraction=float(args.max_feature_fraction),
            track_name=str(args.track_name or ""),
            eval_models=eval_models,
            random_state=int(args.random_state),
            eval_hyperparameters=eval_hyperparameters,
            output_json=Path(args.output_json),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid nested floating-SFS job: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out}", file=sys.stderr)
    return 0


def _cmd_finish(args: argparse.Namespace) -> int:
    pair_root = Path(args.pair_root)
    orig = Path(args.orig_fold_dir)
    summary = finish_outer_folds(pair_root, orig)
    (pair_root / "nested_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in summary if k != "per_fold"}, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    suffix = str(args.yaml_key_suffix or "")
    yaml_key_mode = str(args.yaml_key_mode or "backend_1")
    pairs = resolve_nested_pairs(
        Path(args.automl_root),
        backend=args.backend,
        pairs_mode=args.pairs_mode,
        exclude_stems=_split_csv_arg(args.exclude_stems),
        include_stems=_split_csv_arg(args.include_stems),
        yaml_key_suffix=suffix,
        yaml_key_mode=yaml_key_mode,
    )
    df = write_nested_report(
        Path(args.nested_root),
        automl_root=Path(args.automl_root),
        dest=Path(args.dest),
        backend=args.backend,
        flat_results_csv=args.flat_results,
        pairs=pairs,
        yaml_key_suffix=suffix,
        yaml_key_mode=yaml_key_mode,
        flat_variant=args.flat_variant,
    )
    print(f"Wrote {args.dest} ({len(df)} rows)", file=sys.stderr)
    return 0 if len(df) else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover")
    d.add_argument("--repo-root", type=Path, required=True)
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--jobs-file", type=Path, required=True)
    d.add_argument("--floating-jobs-file", type=Path)
    d.add_argument("--backend", choices=tuple(sorted(PAPER_BACKENDS)), default="abb2")
    d.add_argument("--automl-root", type=Path, default=None)
    d.add_argument("--yaml-key-suffix", default="")
    d.add_argument(
        "--yaml-key-mode",
        choices=BACKEND_YAML_KEY_MODES + ("stem",),
        default="backend_1",
    )
    d.add_argument("--fold-dir-map", type=Path, default=None)
    d.add_argument(
        "--pairs-mode",
        choices=("default",) + ALL_BACKEND_PAIRS_MODES + ("all_abb2_1",),
        default="default",
    )
    d.add_argument("--exclude-stems", default="")
    d.add_argument("--include-stems", default="")
    d.add_argument("--prepare-inner", action="store_true")
    d.set_defaults(func=_cmd_discover)

    pend = sub.add_parser("pending")
    pend.add_argument("--jobs-file", type=Path, required=True)
    pend.add_argument("--out", type=Path, required=True)
    pend.add_argument("--require-oof", action="store_true")
    pend.set_defaults(func=_cmd_pending)

    ff = sub.add_parser("floating-sfs-job")
    ff.add_argument("--inner-fold-dir", type=Path, required=True)
    ff.add_argument("--inner-k", type=int, required=True)
    ff.add_argument("--dataset-stem", required=True)
    ff.add_argument("--target", required=True)
    ff.add_argument("--yaml-key", required=True)
    ff.add_argument("--selection-model", required=True)
    ff.add_argument("--max-feature-fraction", type=float, required=True)
    ff.add_argument("--track-name", default="")
    ff.add_argument("--eval-models", required=True)
    ff.add_argument("--random-state", type=int, default=42)
    ff.add_argument("--eval-hyperparameters", default="{}")
    ff.add_argument("--output-json", type=Path, required=True)
    ff.set_defaults(func=_cmd_floating_sfs_job)

    f = sub.add_parser("finish-pair")
    f.add_argument("--pair-root", type=Path, required=True)
    f.add_argument("--orig-fold-dir", type=Path, required=True)
    f.set_defaults(func=_cmd_finish)

    r = sub.add_parser("report")
    r.add_argument("--nested-root", type=Path, required=True)
    r.add_argument("--automl-root", type=Path, required=True)
    r.add_argument("--dest", type=Path, required=True)
    r.add_argument("--flat-results", type=Path, default=None)
    r.add_argument("--backend", choices=tuple(sorted(PAPER_BACKENDS)), default="abb2")
    r.add_argument("--yaml-key-suffix", default="")
    r.add_argument(
        "--yaml-key-mode",
        choices=BACKEND_YAML_KEY_MODES + ("stem",),
        default="backend_1",
    )
    r.add_argument("--flat-variant", default=None)
    r.add_argument(
        "--pairs-mode",
        choices=("default",) + ALL_BACKEND_PAIRS_MODES + ("all_abb2_1",),
        default="default",
    )
    r.add_argument("--exclude-stems", default="")
    r.add_argument("--include-stems", default="")
    r.set_defaults(func=_cmd_report)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
