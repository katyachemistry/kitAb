#!/usr/bin/env python3
"""Recreate fold-result JSONs from legacy aggregated CSVs.

This is used when aggregate CSVs and fold parquets were preserved but the
individual AutoML result JSONs were not (the paper TAP run). The recreated
JSONs contain the original stored fold metrics and selected features, allowing
``oof_predictions.py`` to refit evaluation models and validate them normally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _dataset_key(path: Path) -> str:
    name = path.name
    if not name.startswith("aggregated_") or not name.endswith(".csv"):
        raise ValueError(f"Not an aggregated CSV: {path}")
    return name[len("aggregated_") : -len(".csv")]


def _parse_run_id(run_id: str, target: str) -> tuple[str, str, str, float]:
    prefix = f"{target}-"
    suffix = run_id[len(prefix) :] if run_id.startswith(prefix) else run_id
    parts = suffix.rsplit("-", 3)
    if len(parts) != 4 or not parts[3].startswith("frac"):
        raise ValueError(f"Cannot parse Target-Selector-Model: {run_id!r}")
    selector, model_type, eval_model, frac_raw = parts
    return selector, model_type, eval_model, int(frac_raw[4:]) / 100.0


def _master_lookup(master_tsv: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    out: dict[tuple[str, str, int], dict[str, str]] = {}
    with Path(master_tsv).open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            fold_dir, fold_raw, stem, target, yaml_key = parts[:5]
            try:
                fold_index = int(fold_raw)
            except ValueError:
                continue
            out.setdefault(
                (yaml_key, target, fold_index),
                {
                    "fold_dir": fold_dir,
                    "dataset_stem": stem,
                    "random_state": parts[9],
                    "eval_hyperparameters": parts[13],
                },
            )
    return out


def materialize(
    *,
    aggregated_dir: Path,
    master_tsv: Path,
    pseudo_automl_root: Path,
    oof_root: Path,
    jobs_file: Path,
    method_name: str,
) -> int:
    lookup = _master_lookup(master_tsv)
    pseudo_automl_root = Path(pseudo_automl_root)
    jobs: list[tuple[str, str, str, str]] = []

    for csv_path in sorted(Path(aggregated_dir).glob("aggregated_*.csv")):
        yaml_key = _dataset_key(csv_path)
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        grouped: defaultdict[tuple[str, str, str, str, float], list[dict[str, str]]] = (
            defaultdict(list)
        )
        for row in rows:
            target = str(row.get("Target_col") or "")
            selector, model_type, eval_model, frac = _parse_run_id(
                str(row.get("Target-Selector-Model") or ""), target
            )
            if eval_model.lower() == "gpr":
                continue
            key = (
                str(row.get("Track") or ""),
                target,
                selector,
                model_type,
                frac,
            )
            row["_eval_model"] = eval_model
            grouped[key].append(row)

        for config_i, (key, config_rows) in enumerate(sorted(grouped.items())):
            track, target, selector, model_type, frac = key
            features_by_eval: dict[str, dict[str, list[str]]] = {}
            fold_keys: set[str] = set()
            for row in config_rows:
                eval_model = row["_eval_model"]
                raw = row.get("selected_features_by_fold") or "{}"
                parsed = json.loads(raw)
                features_by_eval[eval_model] = {
                    str(k): [str(x) for x in v]
                    for k, v in parsed.items()
                    if isinstance(v, list)
                }
                fold_keys.update(features_by_eval[eval_model])

            for fold_key in sorted(fold_keys):
                match = re.fullmatch(r"fold_(\d+)", fold_key)
                if not match:
                    continue
                fold_index = int(match.group(1)) - 1
                meta = lookup.get((yaml_key, target, fold_index))
                if meta is None:
                    raise KeyError(
                        f"No master job for {(yaml_key, target, fold_index)!r}"
                    )
                evaluation: dict[str, dict[str, Any]] = {}
                selected_longest: list[str] = []
                for row in config_rows:
                    eval_model = row["_eval_model"]
                    feats = features_by_eval[eval_model].get(fold_key, [])
                    if len(feats) > len(selected_longest):
                        selected_longest = feats
                    evaluation[eval_model] = {
                        "spearman_rho": _optional_float(
                            row.get(f"{fold_key}_spearman")
                        ),
                        "pearson_r": _optional_float(row.get(f"{fold_key}_pearson")),
                        "r2": _optional_float(row.get(f"{fold_key}_r2")),
                        "prediction_std_mean": _optional_float(
                            row.get(f"{fold_key}_prediction_std_mean")
                        ),
                        "eval_features_used": feats,
                        "n_features": len(feats),
                    }

                out_json = (
                    pseudo_automl_root
                    / yaml_key
                    / f"config{config_i:05d}__fold{fold_index}.json"
                )
                payload = {
                    "fold_dir": meta["fold_dir"],
                    "fold_index": fold_index,
                    "random_state": int(meta["random_state"]),
                    "selector_name": selector,
                    "target_col": target,
                    "model_type": model_type,
                    "selected_features": selected_longest,
                    "n_selected_features": len(selected_longest),
                    "evaluation": evaluation,
                    "eval_models": list(evaluation),
                    "eval_features_frac": frac,
                    "eval_hyperparameters": json.loads(
                        meta["eval_hyperparameters"] or "{}"
                    ),
                    "dataset_stem": meta["dataset_stem"],
                    "dataset_yaml_key": yaml_key,
                    "pipeline_track_name": track,
                    "materialized_from": str(csv_path.resolve()),
                }
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_json.write_text(json.dumps(payload, indent=2))
                oof_path = (
                    Path(oof_root)
                    / yaml_key
                    / f"config{config_i:05d}__fold{fold_index}.oof.parquet"
                )
                jobs.append(
                    (
                        method_name,
                        str(out_json.resolve()),
                        str(oof_path.resolve()),
                        str(pseudo_automl_root.resolve()),
                    )
                )

    jobs_file = Path(jobs_file)
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with jobs_file.open("w", encoding="utf-8") as f:
        f.write("backend\tjson_path\toof_path\tautoml_root\n")
        for row in jobs:
            f.write("\t".join(row) + "\n")
    return len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated-dir", type=Path, required=True)
    parser.add_argument("--master-tsv", type=Path, required=True)
    parser.add_argument("--pseudo-automl-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--method-name", default="tap")
    args = parser.parse_args()
    n = materialize(
        aggregated_dir=args.aggregated_dir,
        master_tsv=args.master_tsv,
        pseudo_automl_root=args.pseudo_automl_root,
        oof_root=args.oof_root,
        jobs_file=args.jobs_file,
        method_name=args.method_name,
    )
    print(f"Materialized {n} fold result JSON(s) -> {args.jobs_file}")


if __name__ == "__main__":
    main()
