"""Parse the generated run config into one record per dataset/seed to run AutoML on."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

AUTOML_CONFIG_KEYS = ("automl_config", "automl-config", "automl")
NON_DATASET_KEYS = frozenset(
    {
        "batch_result_root",
        "batch-result-root",
        "parallel_jobs",
        "parallel-jobs",
        "py",
        "skip_existing_results",
        "skip-existing-results",
        "n_cpu",
        "n-cpu",
        "hyperparameter_tuning",
        *AUTOML_CONFIG_KEYS,
    }
)


class RunConfigError(ValueError):
    """Invalid AutoML run config."""


@dataclass
class DatasetRecord:
    """One dataset (and random seed) to run the four techniques on."""

    yaml_key: str
    yaml_block_key: str
    dataset_path: Path
    dataset_stem: str
    developability_paths: tuple[Path, ...]
    name_col: str
    targets_csv: str
    features_csv: str
    developability_feature_groups_by_target: dict[str, list[str]]
    include_features: list[str]
    run_dir: Path
    jobs_file: Path
    n_splits: int
    random_state: int
    force_preprocess: bool
    max_target_nan_frac: float
    split_col: str | None
    target_cols: list[str] = field(default_factory=list)

    def descriptor_source(self) -> str:
        return developability_run_dir_suffix(self.developability_paths)


def slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_")
    return cleaned or "x"


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def developability_run_dir_suffix(dev_paths: tuple[Path, ...]) -> str:
    root = REPO_ROOT.resolve()
    tokens: list[str] = []
    for path in dev_paths:
        resolved = Path(path).resolve()
        try:
            tokens.append(resolved.relative_to(root).as_posix())
        except ValueError:
            tokens.append(str(resolved))
    token = "__".join(tokens)
    text = slug(token.replace("/", "_")) or "dev"
    if len(text) > 96:
        digest = hashlib.sha256(token.encode()).hexdigest()[:14]
        lead = Path(dev_paths[0]).name if dev_paths else "dev"
        text = slug(f"{lead}_{digest}")
    return text


def _get(block: dict, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in block and block[name] is not None:
            return block[name]
    return default


def as_bool(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _csv_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _developability_paths(block: dict, *, yaml_key: str) -> list[str]:
    single = _get(block, "developability_results_path", "developability-results-path")
    multi = _get(block, "developability_results_paths", "developability-results-paths")
    if single is None and multi is None:
        raise RunConfigError(
            f"Dataset block {yaml_key!r}: provide developability_results_path "
            "or developability_results_paths."
        )
    if single is not None and multi is not None:
        raise RunConfigError(
            f"Dataset block {yaml_key!r}: use only one of developability_results_path "
            "or developability_results_paths."
        )
    out = _csv_list(multi if multi is not None else single)
    if not out:
        raise RunConfigError(
            f"Dataset block {yaml_key!r}: developability results path list is empty."
        )
    return out


_DEV_FEATURE_KEYS = (
    "developability_features",
    "developability-features",
    "developability_feature_groups",
    "developability-feature-groups",
)


def _developability_groups_by_target(
    block: dict, target_cols: list[str], *, yaml_key: str
) -> dict[str, list[str]]:
    present = [
        (key, block.get(key))
        for key in _DEV_FEATURE_KEYS
        if key in block and block.get(key) is not None
    ]
    if len(present) > 1:
        has_mapping = any(isinstance(value, dict) for _, value in present)
        has_shared = any(not isinstance(value, dict) for _, value in present)
        if has_mapping and has_shared:
            raise RunConfigError(
                f"Dataset block {yaml_key!r}: developability features are defined both "
                f"as a shared value and as a per-target mapping "
                f"({[key for key, _ in present]!r}). Use only one form."
            )

    raw = _get(block, *_DEV_FEATURE_KEYS)
    if raw is None:
        return {str(target): [] for target in target_cols}
    if not isinstance(raw, dict):
        groups = _csv_list(raw)
        return {str(target): list(groups) for target in target_cols}

    default_groups = _csv_list(raw.get("default", raw.get("*")))
    unknown = sorted(
        key for key in raw if str(key) not in set(target_cols) | {"default", "*"}
    )
    if unknown:
        raise RunConfigError(
            f"Dataset block {yaml_key!r}: unknown developability_features mapping "
            f"key(s) {unknown!r}; expected target names from target_cols or 'default'."
        )
    return {
        target: _csv_list(raw[target]) if target in raw else list(default_groups)
        for target in target_cols
    }


def _random_seeds(raw: Any, *, yaml_key: str) -> list[int]:
    if raw is None:
        return [42]
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    seeds: list[int] = []
    for value in values:
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise RunConfigError(
                f"Dataset block {yaml_key!r}: random_seeds must be integers, "
                f"got {value!r}"
            ) from exc
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise RunConfigError(f"Dataset block {yaml_key!r}: random_seeds is empty")
    return seeds


def automl_config_path_from(root: dict) -> Path | None:
    """Pick up ``automl_config:`` from the run config, ignoring ``automl: false``."""
    for key in AUTOML_CONFIG_KEYS:
        raw = root.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        text = str(raw).strip()
        if text:
            return resolve_path(text)
    return None


def parse_dataset_records(root: dict, *, default_n_splits: int) -> list[DatasetRecord]:
    """One record per dataset block, expanded over ``random_seeds``."""
    records: list[DatasetRecord] = []
    for yaml_key, block in root.items():
        if yaml_key in NON_DATASET_KEYS or not isinstance(block, dict):
            continue
        has_dev = any(
            key in block
            for key in (
                "developability_results_path",
                "developability-results-path",
                "developability_results_paths",
                "developability-results-paths",
            )
        )
        if "path" not in block or not has_dev:
            continue

        dataset_path = resolve_path(block["path"])
        dev_paths = tuple(
            resolve_path(p) for p in _developability_paths(block, yaml_key=yaml_key)
        )
        targets_csv = str(_get(block, "target_cols", "target-cols") or "").strip()
        if not targets_csv:
            raise RunConfigError(f"Dataset block {yaml_key!r}: target_cols is required")
        target_cols = _csv_list(targets_csv)
        features_csv = str(_get(block, "feature_cols", "feature-cols") or "").strip()

        n_splits = int(_get(block, "n_splits", "n-splits", default=default_n_splits))
        split_raw = _get(block, "split_col", "split-col")
        split_col = (
            None if split_raw is None or not str(split_raw).strip() else str(split_raw).strip()
        )

        max_nan_raw = _get(block, "max_target_nan_frac", "max-target-nan-frac", default=0.7)
        try:
            max_target_nan_frac = float(max_nan_raw)
        except (TypeError, ValueError):
            raise RunConfigError(
                f"Dataset block {yaml_key!r}: max_target_nan_frac must be a float in (0, 1]"
            ) from None
        if not 0.0 < max_target_nan_frac <= 1.0:
            raise RunConfigError(
                f"Dataset block {yaml_key!r}: max_target_nan_frac must be in (0, 1], "
                f"got {max_target_nan_frac!r}"
            )

        seeds = _random_seeds(
            _get(block, "random_seeds", "random-seeds", "random_state", "random-state"),
            yaml_key=yaml_key,
        )
        if split_col and len(seeds) > 1:
            seeds = [seeds[0]]

        run_dir_user = _get(block, "run_dir", "run-dir")
        stem = dataset_path.stem
        for seed in seeds:
            effective_key = f"{yaml_key}__rs{seed}" if len(seeds) > 1 else yaml_key
            if run_dir_user:
                base = resolve_path(run_dir_user)
                run_dir = (
                    (base.parent / f"{base.name}__rs{seed}").resolve()
                    if len(seeds) > 1
                    else base.resolve()
                )
            else:
                suffix = f"__rs{seed}" if len(seeds) > 1 else ""
                run_dir = (
                    REPO_ROOT
                    / "runs"
                    / f"{stem}_cv_prepare__{developability_run_dir_suffix(dev_paths)}{suffix}"
                ).resolve()

            records.append(
                DatasetRecord(
                    yaml_key=effective_key,
                    yaml_block_key=yaml_key,
                    dataset_path=dataset_path,
                    dataset_stem=stem,
                    developability_paths=dev_paths,
                    name_col=str(_get(block, "name_col", "name-col", default="name")),
                    targets_csv=targets_csv,
                    features_csv=features_csv,
                    developability_feature_groups_by_target=(
                        _developability_groups_by_target(
                            block, target_cols, yaml_key=yaml_key
                        )
                    ),
                    include_features=_csv_list(
                        _get(block, "include_features", "include-features")
                    ),
                    run_dir=run_dir,
                    jobs_file=run_dir / "parallel_jobs.txt",
                    n_splits=n_splits,
                    random_state=seed,
                    force_preprocess=as_bool(
                        _get(block, "force_preprocess", "force-preprocess"), default=False
                    ),
                    max_target_nan_frac=max_target_nan_frac,
                    split_col=split_col,
                    target_cols=target_cols,
                )
            )

    if not records:
        raise RunConfigError(
            "No dataset blocks found (each needs path + developability_results_path(s))"
        )
    return records
