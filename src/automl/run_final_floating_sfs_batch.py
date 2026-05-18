#!/usr/bin/env python3
"""Run a post-grid per-fold floating-SFS stage from a batch manifest.

This stage is optional and is driven by ``final_floating_sfs`` stored per dataset block
in ``batch_manifest.json``. For each dataset block / target / outer fold:

1. Read the first-stage worker JSONs listed in the master TSV.
2. Aggregate feature votes across all first-stage grid cells for that fold.
3. Keep the top voted features, capped by
   ``max(1, floor(max_feature_fraction * n_train_rows))``.
4. MinMax-scale that candidate set on the fold train split.
5. Run floating SFS on the scaled train fold using only those candidates.
6. Re-run the existing eval regressors on the resulting feature subset.

One result JSON is written per fold and per configured floating-SFS selection model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from automl.feature_selectors import select_features_floating_sfs
from automl.run_fold_pipeline_config import _evaluate_fold_models, _parse_eval_models
from automl.selection_grid_stats import aggregate_selected_feature_votes_across_grid

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(pathlike: str | Path) -> Path:
    path = Path(pathlike)
    if path.is_absolute():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_") or "x"


def _eval_frac_slug(value: float) -> str:
    return f"frac{int(round(float(value) * 100)):03d}"


def _json_paths_from_master(master_path: Path) -> list[Path]:
    """Result JSON paths from the master TSV written by prepare_parallel (≥9 columns required)."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw_line in master_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        path = _resolve(parts[8])
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _final_output_json_path(
    batch_root: Path,
    *,
    dataset_yaml_key: str,
    dataset_stem: str,
    fold_dir: Path,
    fold_index: int,
    model: str,
    max_feature_fraction: float,
    track_name: str | None = None,
) -> Path:
    subdir = batch_root / _slug(dataset_yaml_key)
    track_part = f"__{_slug(track_name)}" if track_name else ""
    fname = (
        f"{_slug(dataset_stem)}__{_slug(fold_dir.name)}__fold{fold_index}{track_part}__"
        f"final_floating_sfs__{_slug(model)}__{_eval_frac_slug(max_feature_fraction)}.json"
    )
    return (subdir / fname).resolve()


def _load_first_stage_records(json_paths: list[Path]) -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    for path in json_paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        records.append((path, data))
    return records


def _make_scaled_fold_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    candidate_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    if not candidate_features:
        return train_scaled, test_scaled
    # pandas >= 2.0 raises "Invalid value ... for dtype 'int64'" when assigning
    # float64 MinMax output into integer-typed columns via .loc.  Cast in-place first.
    for col in candidate_features:
        if col in train_scaled.columns and train_scaled[col].dtype.kind != "f":
            train_scaled[col] = train_scaled[col].astype(float)
        if col in test_scaled.columns and test_scaled[col].dtype.kind != "f":
            test_scaled[col] = test_scaled[col].astype(float)
    scaler = MinMaxScaler()
    train_scaled.loc[:, candidate_features] = scaler.fit_transform(
        train_scaled.loc[:, candidate_features]
    )
    test_scaled.loc[:, candidate_features] = scaler.transform(test_scaled.loc[:, candidate_features])
    return train_scaled, test_scaled


def _sorted_vote_items(vote_map: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(
        ((str(name), int(votes)) for name, votes in vote_map.items()),
        key=lambda item: (-item[1], item[0]),
    )


def _error_evaluation(eval_models: list[str] | None, error: str) -> dict[str, dict] | None:
    if eval_models is None:
        return None
    return {
        m: {
            "error": error,
            "n_features": 0,
        }
        for m in eval_models
    }


def _run_one_final_floating_sfs(
    *,
    batch_root: Path,
    dataset_info: dict,
    target_col: str,
    fold_index: int,
    fold_meta: dict,
    vote_map: dict[str, int],
    selection_model: str,
    eval_models: list[str] | None,
    eval_hp_by_model: dict[str, dict] | None,
    track_cfg: dict,
    track_name: str | None = None,
) -> Path:
    dataset_yaml_key = str(dataset_info["dataset_yaml_key"])
    dataset_stem = str(dataset_info["dataset_stem"])
    max_feature_fraction = float(track_cfg["max_feature_fraction"])
    fold_dir = _resolve(fold_meta["fold_dir"])
    random_state = int(fold_meta.get("random_state", dataset_info.get("random_state", 42)))
    train_pq = fold_dir / f"fold_{fold_index}_train.parquet"
    test_pq = fold_dir / f"fold_{fold_index}_test.parquet"
    out_path = _final_output_json_path(
        batch_root,
        dataset_yaml_key=dataset_yaml_key,
        dataset_stem=dataset_stem,
        fold_dir=fold_dir,
        fold_index=fold_index,
        model=selection_model,
        max_feature_fraction=max_feature_fraction,
        track_name=track_name,
    )

    vote_items = _sorted_vote_items(vote_map)
    voted_features_all = [name for name, _ in vote_items]
    voted_feature_counts = {name: votes for name, votes in vote_items}

    payload: dict = {
        "fold_dir": str(fold_dir),
        "fold_index": int(fold_index),
        "random_state": random_state,
        "selector_name": "final_floating_sfs",
        "target_col": str(target_col),
        "feature_scaling": "minmax_train_fit_transform_test",
        "feature_selection_pipeline": "final_floating_sfs",
        "model_type": str(selection_model),
        "selection_max_features": None,
        "selected_features": [],
        "n_selected_features": 0,
        "features_after_vote_aggregation": voted_features_all,
        "features_after_vote_aggregation_count": len(voted_features_all),
        "after_step": {"final_floating_sfs": []},
        "evaluation": None,
        "eval_models": eval_models,
        "eval_features_frac": max_feature_fraction,
        "dataset_stem": dataset_stem,
        "dataset_yaml_key": dataset_yaml_key,
        "final_floating_sfs_summary": {
            "max_feature_fraction": max_feature_fraction,
            "selection_model": str(selection_model),
            "vote_counts_by_feature": voted_feature_counts,
            "source": "post_grid_vote_aggregation",
        },
    }
    if track_name:
        payload["pipeline_track_name"] = track_name
    if eval_hp_by_model:
        payload["eval_hyperparameters"] = eval_hp_by_model

    try:
        if not train_pq.is_file() or not test_pq.is_file():
            raise FileNotFoundError(
                f"Missing fold parquet(s) for fold {fold_index}: {train_pq} / {test_pq}"
            )

        train_df = pd.read_parquet(train_pq)
        test_df = pd.read_parquet(test_pq)
        n_train = int(len(train_df))
        candidate_cap = max(1, int(max_feature_fraction * n_train))
        present_candidates = [
            feat
            for feat in voted_features_all
            if feat in train_df.columns and feat in test_df.columns
        ]
        top_candidates = present_candidates[:candidate_cap]

        payload["selection_max_features"] = candidate_cap
        payload["features_after_vote_aggregation"] = top_candidates
        payload["features_after_vote_aggregation_count"] = len(top_candidates)
        payload["final_floating_sfs_summary"].update(
            {
                "n_train": n_train,
                "candidate_cap": candidate_cap,
                "n_voted_features_total": len(voted_features_all),
                "n_candidates_present_in_fold": len(present_candidates),
                "n_candidates_used_for_sfs": len(top_candidates),
                "candidate_features_for_sfs": list(top_candidates),
            }
        )

        if not top_candidates:
            raise ValueError("No voted features are present in both train and test fold dataframes")

        train_scaled, test_scaled = _make_scaled_fold_frames(
            train_df, test_df, candidate_features=top_candidates
        )
        selected_features = select_features_floating_sfs(
            train_scaled,
            target_col=str(target_col),
            candidate_features=top_candidates,
            n_features_to_select=len(top_candidates),
            random_state=random_state,
            model_type=str(selection_model),
        )
        selected_features = [f for f in selected_features if f in top_candidates]

        payload["selected_features"] = list(selected_features)
        payload["n_selected_features"] = len(selected_features)
        payload["after_step"] = {"final_floating_sfs": list(selected_features)}
        payload["final_floating_sfs_summary"]["n_selected_features"] = len(selected_features)

        if eval_models is not None:
            payload["evaluation"] = _evaluate_fold_models(
                train_scaled,
                test_scaled,
                target_col=str(target_col),
                feature_cols=list(selected_features),
                eval_models=eval_models,
                random_state=random_state,
                features_frac=max_feature_fraction,
                eval_hp_by_model=eval_hp_by_model,
            )
    except Exception as e:
        payload["final_floating_sfs_summary"]["error"] = str(e)
        payload["evaluation"] = _error_evaluation(eval_models, str(e))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-fold post-grid final floating SFS from a batch manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to batch_manifest.json written by prepare_parallel_from_config.py",
    )
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read manifest {manifest_path}: {e}", file=sys.stderr)
        sys.exit(1)

    batch_root_raw = manifest.get("batch_result_root")
    master_tsv_raw = manifest.get("master_jobs_tsv")
    if not batch_root_raw or not master_tsv_raw:
        print("Manifest missing batch_result_root or master_jobs_tsv", file=sys.stderr)
        sys.exit(1)

    batch_root = _resolve(batch_root_raw)
    master_tsv = _resolve(master_tsv_raw)
    if not master_tsv.is_file():
        print(f"Master TSV not found: {master_tsv}", file=sys.stderr)
        sys.exit(1)

    json_paths = _json_paths_from_master(master_tsv)
    records = _load_first_stage_records(json_paths)

    records_by_dataset: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    fold_meta_by_group: dict[tuple[str, str, int], dict] = {}
    for path, data in records:
        ds_key = str(data.get("dataset_yaml_key") or data.get("dataset_stem") or "")
        target_col = str(data.get("target_col") or "")
        try:
            fold_index = int(data.get("fold_index"))
        except (TypeError, ValueError):
            continue
        if not ds_key or not target_col:
            continue
        records_by_dataset[ds_key].append((path, data))
        fold_meta_by_group.setdefault(
            (ds_key, target_col, fold_index),
            {
                "fold_dir": data.get("fold_dir"),
                "random_state": data.get("random_state"),
                "dataset_stem": data.get("dataset_stem"),
            },
        )

    n_written = 0
    for dataset_info in manifest.get("datasets") or []:
        raw_cfg = dataset_info.get("final_floating_sfs")
        if not raw_cfg:
            continue

        # Normalize to a list of per-track configs.
        # - No-track / legacy: a single dict → wrap as one entry with track_name=None.
        # - Track mode: a list of dicts, each having a "track_name" key.
        if isinstance(raw_cfg, dict):
            track_cfgs: list[tuple[str | None, dict]] = [(None, raw_cfg)]
        elif isinstance(raw_cfg, list):
            track_cfgs = []
            for entry in raw_cfg:
                if isinstance(entry, dict):
                    tname = entry.get("track_name") or None
                    track_cfgs.append((tname, entry))
        else:
            continue
        if not track_cfgs:
            continue

        dataset_yaml_key = str(dataset_info.get("dataset_yaml_key") or "")
        if not dataset_yaml_key:
            continue
        dataset_records = records_by_dataset.get(dataset_yaml_key, [])
        if not dataset_records:
            print(
                f"[final_floating_sfs] No first-stage JSONs found for {dataset_yaml_key}",
                file=sys.stderr,
            )
            continue

        eval_models = _parse_eval_models(str(dataset_info.get("eval_models", "all")))
        eval_hp_by_model = dataset_info.get("eval_hyperparameters") or {}

        for track_name, track_cfg in track_cfgs:
            # Filter first-stage records to those belonging to this track (or all when no track).
            if track_name is not None:
                track_records = [
                    (path, data)
                    for path, data in dataset_records
                    if (data.get("pipeline_track_name") or None) == track_name
                ]
            else:
                track_records = dataset_records

            if not track_records:
                print(
                    f"[final_floating_sfs] No first-stage JSONs for {dataset_yaml_key}"
                    + (f" track={track_name}" if track_name else ""),
                    file=sys.stderr,
                )
                continue

            paths_for_track = [path for path, _ in track_records]
            targets = sorted(
                {
                    str(data.get("target_col"))
                    for _, data in track_records
                    if str(data.get("target_col") or "").strip()
                }
            )

            for target_col in targets:
                votes_by_fold = aggregate_selected_feature_votes_across_grid(
                    paths_for_track,
                    dataset_stem=dataset_info.get("dataset_stem"),
                    target_col=target_col,
                    dataset_yaml_key=dataset_yaml_key,
                    feature_source="selected_features",
                    dedupe_features_within_combo=True,
                )
                for fold_index, vote_map in sorted(votes_by_fold.items()):
                    fold_meta = fold_meta_by_group.get((dataset_yaml_key, target_col, int(fold_index)))
                    if fold_meta is None:
                        continue
                    for selection_model in track_cfg.get("models") or []:
                        out_path = _run_one_final_floating_sfs(
                            batch_root=batch_root,
                            dataset_info=dataset_info,
                            target_col=target_col,
                            fold_index=int(fold_index),
                            fold_meta=fold_meta,
                            vote_map=vote_map,
                            selection_model=str(selection_model),
                            eval_models=eval_models,
                            eval_hp_by_model=eval_hp_by_model,
                            track_cfg=track_cfg,
                            track_name=track_name,
                        )
                        print(f"Wrote {out_path}", file=sys.stderr)
                        n_written += 1

    print(
        f"[final_floating_sfs] Wrote {n_written} fold-level result JSON(s).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
