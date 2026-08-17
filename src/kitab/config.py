"""Canonical run-manifest schema and validation for kitAb."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_EVAL_MODELS = ("linear", "elasticnet", "randomforest", "svm", "knn")
STRUCTURE_BACKENDS = ("abb2", "abb3", "flashabb")

KNOWN_TOP_KEYS = frozenset(
    {
        "inputs",
        "run",
        "structure_prediction",
        "structure_processing",
        "descriptors",
        "automl",
        "tuning",
    }
)
# Accepted but unused (older YAMLs). workflow is mapped to section enabled flags.
_IGNORED_TOP_KEYS = frozenset({"schema_version", "workflow"})
KNOWN_INPUT_KEYS = frozenset(
    {
        "datasets_dir",
        "structures_dir",
        "predefined_descriptors_dir",
        "exclude_datasets",
        "split_randomly",
    }
)
KNOWN_RUN_KEYS = frozenset(
    {"output_dir", "resume", "n_cpu", "skip_existing_results", "result_folder"}
)
KNOWN_PRED_KEYS = frozenset(
    {"enabled", "model", "device", "batch_size", "skip_existing"}
)
KNOWN_PROC_KEYS = frozenset(
    {"enabled", "renumber_imgt", "minimize", "minimize_attempts"}
)
KNOWN_DESC_KEYS = frozenset(
    {
        "enabled",
        "include_features",
        "cleanup",
        "batch_size",
        "propka_minimize_retries",
    }
)
KNOWN_AUTOML_KEYS = frozenset({"enabled", "eval_models", "config_path"})
KNOWN_TUNING_KEYS = frozenset({"enabled", "margin", "max_rank", "clean_folds_after"})


class ConfigError(ValueError):
    """Invalid kitAb run manifest."""

    def __init__(self, message: str, *, issues: list[str] | None = None):
        self.issues = list(issues) if issues is not None else [message]
        super().__init__(_format_config_issues(self.issues))


def _format_config_issues(issues: list[str]) -> str:
    if len(issues) == 1:
        return issues[0]
    lines = [f"{len(issues)} errors in the manifest:"]
    lines.extend(f"  - {issue}" for issue in issues)
    return "\n".join(lines)


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required: pip install pyyaml / conda install pyyaml"
        ) from exc
    return yaml


def _as_bool(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise ConfigError(f"{field_name} is required")
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    raise ConfigError(f"{field_name} must be a boolean, got {value!r}")


def _as_int(
    value: Any,
    *,
    field_name: str,
    default: int | None = None,
    min_value: int | None = None,
) -> int:
    if value is None:
        if default is None:
            raise ConfigError(f"{field_name} is required")
        value = default
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer, got {value!r}") from exc
    if min_value is not None and n < min_value:
        raise ConfigError(f"{field_name} must be >= {min_value}, got {n}")
    return n


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(p).strip() for p in value if str(p).strip()]
    raise ConfigError(f"{field_name} must be a list or comma-separated string")


def _resolve_path(repo_root: Path, raw: str | None) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _unknown_keys(data: dict[str, Any], known: frozenset[str], section: str) -> str | None:
    unknown = sorted(set(data) - known)
    if not unknown:
        return None
    return (
        f"unknown {section} key(s): {', '.join(unknown)} "
        f"(known: {', '.join(sorted(known))})"
    )


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ConfigError(f"{field_name} must be a mapping, got {type(value).__name__}")


def _format_yaml_error(path: Path, exc: BaseException) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = str(getattr(exc, "problem", None) or exc).strip()
    loc = str(path)
    if mark is not None:
        loc = f"{path}:{mark.line + 1}:{mark.column + 1}"
    context = getattr(exc, "context", None)
    msg = f"invalid YAML at {loc}: {problem}"
    if context:
        msg += f" ({context})"
    return msg


@dataclass
class InputsConfig:
    datasets_dir: Path | None = None
    structures_dir: Path | None = None
    predefined_descriptors_dir: Path | None = None
    exclude_datasets: list[str] = field(default_factory=list)
    split_randomly: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    output_dir: Path
    resume: bool = False
    n_cpu: int | None = None
    skip_existing_results: bool = False


@dataclass
class StructurePredictionConfig:
    enabled: bool = False
    model: list[str] = field(default_factory=lambda: ["abb2"])
    device: str = "cuda:0"
    batch_size: int | None = None
    skip_existing: bool = True


@dataclass
class StructureProcessingConfig:
    enabled: bool = False
    renumber_imgt: bool = False
    minimize: bool = False
    minimize_attempts: int = 5


@dataclass
class DescriptorsConfig:
    enabled: bool = False
    include_features: list[str] = field(default_factory=list)
    cleanup: bool = False
    batch_size: int | None = None
    # On "PropKa coverage incomplete", minimize the PDB and re-run descriptors
    # for that structure (0 disables). Default matches minimize_attempts.
    propka_minimize_retries: int = 5


@dataclass
class AutomlConfig:
    enabled: bool = True
    config_path: Path | None = None
    eval_models: str = "all"


@dataclass
class TuningConfig:
    enabled: bool = False
    margin: float = 0.1
    max_rank: int = 3
    clean_folds_after: bool = True


@dataclass
class Manifest:
    source_path: Path
    legacy: bool
    repo_root: Path
    inputs: InputsConfig
    run: RunConfig
    structure_prediction: StructurePredictionConfig
    structure_processing: StructureProcessingConfig
    descriptors: DescriptorsConfig
    automl: AutomlConfig
    tuning: TuningConfig
    warnings: list[str] = field(default_factory=list)
    stages_override: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if self.structure_prediction.enabled:
            return "predict"
        if self.descriptors.enabled:
            return "structures" if self.inputs.datasets_dir is not None else "descriptors"
        if self.automl.enabled:
            return "automl"
        return "none"

    def stage_graph(self) -> list[str]:
        if self.stages_override:
            return list(self.stages_override)
        graph: list[str] = []
        if self.structure_prediction.enabled:
            graph.append("predict")
        if self.structure_prediction.enabled or self.descriptors.enabled:
            graph.append("process_structures")
        if self.descriptors.enabled:
            graph.append("descriptors")
        if self.automl.enabled:
            if self.descriptors.enabled or self.structure_prediction.enabled:
                graph.extend(["completeness", "automl", "analysis"])
            else:
                graph.extend(["automl", "analysis"])
            # Export shortlisted models; grid-search only if tuning.enabled.
            graph.append("tuning")
        return graph

    def to_resolved_dict(self) -> dict[str, Any]:
        def _path(p: Path | None) -> str | None:
            return None if p is None else str(p)

        return {
            "source_path": str(self.source_path),
            "legacy": self.legacy,
            "repo_root": str(self.repo_root),
            "warnings": list(self.warnings),
            "stage_graph": self.stage_graph(),
            "inputs": {
                "datasets_dir": _path(self.inputs.datasets_dir),
                "structures_dir": _path(self.inputs.structures_dir),
                "predefined_descriptors_dir": _path(
                    self.inputs.predefined_descriptors_dir
                ),
                "exclude_datasets": list(self.inputs.exclude_datasets),
                "split_randomly": list(self.inputs.split_randomly),
            },
            "run": {
                "output_dir": str(self.run.output_dir),
                "resume": self.run.resume,
                "n_cpu": self.run.n_cpu,
                "skip_existing_results": self.run.skip_existing_results,
            },
            "structure_prediction": asdict(self.structure_prediction),
            "structure_processing": asdict(self.structure_processing),
            "descriptors": {
                "enabled": self.descriptors.enabled,
                "include_features": list(self.descriptors.include_features),
                "cleanup": self.descriptors.cleanup,
                "batch_size": self.descriptors.batch_size,
                "propka_minimize_retries": self.descriptors.propka_minimize_retries,
            },
            "automl": {
                "enabled": self.automl.enabled,
                "config_path": _path(self.automl.config_path),
                "eval_models": self.automl.eval_models,
            },
            "tuning": asdict(self.tuning),
        }

    def checksum(self) -> str:
        payload = json.dumps(
            self.to_resolved_dict(), sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _apply_deprecated_workflow(
    workflow_raw: dict[str, Any],
    pred_raw: dict[str, Any],
    desc_raw: dict[str, Any],
    automl_raw: dict[str, Any],
    warnings: list[str],
) -> None:
    """Map old workflow.mode / enable_automl onto section enabled flags."""
    if not workflow_raw:
        return
    warnings.append(
        "The workflow: section is no longer used. Set enabled: true/false under "
        "structure_prediction, descriptors, and automl instead."
    )
    mode = str(workflow_raw.get("mode") or "").strip().lower()
    enable_automl = workflow_raw.get("enable_automl")
    if enable_automl is None:
        automl_default: bool | None = None
    else:
        automl_default = bool(enable_automl)
    if mode == "predict":
        pred_raw.setdefault("enabled", True)
        desc_raw.setdefault("enabled", True)
        automl_raw.setdefault("enabled", True if automl_default is None else automl_default)
    elif mode == "structures":
        pred_raw.setdefault("enabled", False)
        desc_raw.setdefault("enabled", True)
        automl_raw.setdefault("enabled", True if automl_default is None else automl_default)
    elif mode == "descriptors":
        pred_raw.setdefault("enabled", False)
        desc_raw.setdefault("enabled", True)
        automl_raw.setdefault("enabled", False)
    elif mode == "automl":
        pred_raw.setdefault("enabled", False)
        desc_raw.setdefault("enabled", False)
        automl_raw.setdefault("enabled", True)
    elif mode:
        warnings.append(
            f"ignored unknown workflow.mode {mode!r}; "
            "use structure_prediction.enabled / descriptors.enabled / automl.enabled"
        )


def _eval_models_spec(raw: Any) -> str:
    if raw is None or raw == "":
        return "all"
    if isinstance(raw, (list, tuple)):
        return ",".join(str(p).strip() for p in raw if str(p).strip()) or "all"
    return str(raw)


def _validate_eval_models(spec: str) -> str:
    s = str(spec).strip().lower()
    if s in ("", "all", "none", "skip", "off"):
        return s or "all"
    parts = [p for p in s.replace(",", " ").split() if p]
    for p in parts:
        if p == "gpr":
            raise ConfigError(
                "Eval model 'gpr' has been removed from kitAb. "
                f"Supported: {', '.join(SUPPORTED_EVAL_MODELS)}"
            )
        if p not in SUPPORTED_EVAL_MODELS:
            raise ConfigError(
                f"Unknown eval model {p!r}. "
                f"Supported: {', '.join(SUPPORTED_EVAL_MODELS)}, or all"
            )
    return ",".join(parts)


def _parse_structure_models(value: Any) -> list[str]:
    """Accept ``abb2``, ``abb2, abb3``, or ``[abb2, abb3]``."""
    if value is None or value == "":
        return ["abb2"]
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.replace(",", " ").split() if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip().lower() for p in value if str(p).strip()]
    else:
        raise ConfigError(
            "structure_prediction.model must be a backend name or a list "
            f"(abb2, abb3, flashabb); got {type(value).__name__}"
        )
    if not parts:
        raise ConfigError(
            "structure_prediction.model must list at least one of "
            f"{', '.join(STRUCTURE_BACKENDS)}"
        )
    seen: list[str] = []
    bad: list[str] = []
    for part in parts:
        if part not in STRUCTURE_BACKENDS:
            bad.append(part)
        elif part not in seen:
            seen.append(part)
    if bad:
        raise ConfigError(
            "structure_prediction.model must be one or more of "
            f"{', '.join(STRUCTURE_BACKENDS)}; got {', '.join(repr(b) for b in bad)}"
        )
    return seen


def parse_manifest_dict(
    raw: dict[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
    legacy: bool = False,
    warnings: list[str] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Manifest:
    if not isinstance(raw, dict):
        raise ConfigError("manifest root must be a YAML mapping (key: value)")

    issues: list[str] = []

    def note(msg: str | None) -> None:
        if msg:
            issues.append(msg)

    def collect(fn, default):
        try:
            return fn()
        except ConfigError as exc:
            issues.extend(exc.issues)
            return default

    if not legacy:
        note(
            _unknown_keys(
                raw,
                KNOWN_TOP_KEYS | {"_legacy_raw"} | _IGNORED_TOP_KEYS,
                "top-level",
            )
        )

    warnings = list(warnings or [])
    overrides = dict(cli_overrides or {})

    inputs_raw = collect(lambda: _as_mapping(raw.get("inputs"), field_name="inputs"), {})
    run_raw = collect(lambda: _as_mapping(raw.get("run"), field_name="run"), {})
    workflow_raw = collect(
        lambda: _as_mapping(raw.get("workflow"), field_name="workflow"), {}
    )
    pred_raw = collect(
        lambda: _as_mapping(
            raw.get("structure_prediction"), field_name="structure_prediction"
        ),
        {},
    )
    proc_raw = collect(
        lambda: _as_mapping(
            raw.get("structure_processing"), field_name="structure_processing"
        ),
        {},
    )
    desc_raw = collect(
        lambda: _as_mapping(raw.get("descriptors"), field_name="descriptors"), {}
    )
    automl_raw = collect(
        lambda: _as_mapping(raw.get("automl"), field_name="automl"), {}
    )
    tuning_raw = collect(
        lambda: _as_mapping(raw.get("tuning"), field_name="tuning"), {}
    )

    _apply_deprecated_workflow(
        workflow_raw, pred_raw, desc_raw, automl_raw, warnings
    )

    if pred_raw.pop("runs", None) is not None:
        warnings.append(
            "structure_prediction.runs is no longer used; each sequence is predicted once. "
            "Use structure_processing.minimize_attempts for minimization retries."
        )
    if "allowed_suffixes" in inputs_raw:
        warnings.append(
            "inputs.allowed_suffixes is no longer used. Point "
            "predefined_descriptors_dir at the descriptor folder; every subfolder "
            "named {dataset} or {dataset}_... is used."
        )
        inputs_raw.pop("allowed_suffixes")
    if desc_raw.pop("structures_layout", None) is not None:
        warnings.append(
            "descriptors.structures_layout is no longer used; "
            "structure folders are discovered automatically "
            "(subdirectories if present, otherwise the root folder)."
        )

    if not legacy:
        note(_unknown_keys(inputs_raw, KNOWN_INPUT_KEYS, "inputs"))
        note(_unknown_keys(run_raw, KNOWN_RUN_KEYS, "run"))
        note(_unknown_keys(pred_raw, KNOWN_PRED_KEYS, "structure_prediction"))
        note(_unknown_keys(proc_raw, KNOWN_PROC_KEYS, "structure_processing"))
        note(_unknown_keys(desc_raw, KNOWN_DESC_KEYS, "descriptors"))
        note(_unknown_keys(automl_raw, KNOWN_AUTOML_KEYS, "automl"))
        note(_unknown_keys(tuning_raw, KNOWN_TUNING_KEYS, "tuning"))

    if overrides.get("output_dir") is not None:
        run_raw["output_dir"] = overrides["output_dir"]
    if overrides.get("resume") is not None:
        run_raw["resume"] = overrides["resume"]
    if overrides.get("n_cpu") is not None:
        run_raw["n_cpu"] = overrides["n_cpu"]
    if overrides.get("device") is not None:
        pred_raw["device"] = overrides["device"]
    if overrides.get("enable_automl") is not None:
        automl_raw["enabled"] = overrides["enable_automl"]
    if overrides.get("enable_tuning") is not None:
        tuning_raw["enabled"] = overrides["enable_tuning"]

    stages_override = collect(
        lambda: _as_str_list(
            overrides.get("stages", workflow_raw.get("stages")),
            field_name="stages",
        ),
        [],
    )

    output_raw = run_raw.get("output_dir") or run_raw.get("result_folder")
    output_dir = _resolve_path(repo_root, str(output_raw) if output_raw else None)
    if output_dir is None:
        note("run.output_dir is required")

    datasets_dir = _resolve_path(repo_root, inputs_raw.get("datasets_dir"))
    structures_dir = _resolve_path(repo_root, inputs_raw.get("structures_dir"))
    predefined = _resolve_path(repo_root, inputs_raw.get("predefined_descriptors_dir"))

    pred_enabled = collect(
        lambda: _as_bool(
            pred_raw.get("enabled"),
            field_name="structure_prediction.enabled",
            default="structure_prediction" in raw,
        ),
        "structure_prediction" in raw,
    )
    proc_enabled = collect(
        lambda: _as_bool(
            proc_raw.get("enabled"),
            field_name="structure_processing.enabled",
            default=bool(proc_raw.get("minimize") or proc_raw.get("renumber_imgt")),
        ),
        bool(proc_raw.get("minimize") or proc_raw.get("renumber_imgt")),
    )
    desc_enabled = collect(
        lambda: _as_bool(
            desc_raw.get("enabled"),
            field_name="descriptors.enabled",
            default="descriptors" in raw,
        ),
        "descriptors" in raw,
    )
    automl_enabled = collect(
        lambda: _as_bool(
            automl_raw.get("enabled"),
            field_name="automl.enabled",
            default="automl" in raw,
        ),
        "automl" in raw,
    )

    if not pred_enabled and not desc_enabled and not automl_enabled:
        note(
            "enable at least one of structure_prediction, descriptors, or automl "
            "(set enabled: true)"
        )
    if pred_enabled and datasets_dir is None:
        note("structure_prediction.enabled requires inputs.datasets_dir")
    if desc_enabled and not pred_enabled and structures_dir is None:
        note(
            "descriptors.enabled without structure_prediction requires "
            "inputs.structures_dir"
        )
    if automl_enabled and datasets_dir is None:
        note("automl.enabled requires inputs.datasets_dir")
    if automl_enabled and not desc_enabled and predefined is None:
        note(
            "automl.enabled without descriptors requires "
            "inputs.predefined_descriptors_dir"
        )

    models = collect(
        lambda: _parse_structure_models(pred_raw.get("model")),
        ["abb2"],
    )

    device_raw = pred_raw.get("device", "cuda:0")
    if isinstance(device_raw, int) or (
        isinstance(device_raw, str) and str(device_raw).isdigit()
    ):
        device = f"cuda:{device_raw}"
    else:
        device = str(device_raw).strip() or "cuda:0"

    eval_models = collect(
        lambda: _validate_eval_models(_eval_models_spec(automl_raw.get("eval_models"))),
        "all",
    )
    automl_cfg_path = _resolve_path(repo_root, automl_raw.get("config_path"))

    exclude_datasets = collect(
        lambda: _as_str_list(
            inputs_raw.get("exclude_datasets"), field_name="inputs.exclude_datasets"
        ),
        [],
    )
    split_randomly = collect(
        lambda: _as_str_list(
            inputs_raw.get("split_randomly"), field_name="inputs.split_randomly"
        ),
        [],
    )

    resume = collect(
        lambda: _as_bool(run_raw.get("resume"), field_name="run.resume", default=False),
        False,
    )
    n_cpu = None
    if run_raw.get("n_cpu") not in (None, ""):
        n_cpu = collect(
            lambda: _as_int(run_raw.get("n_cpu"), field_name="run.n_cpu", min_value=1),
            None,
        )
    skip_existing_results = collect(
        lambda: _as_bool(
            run_raw.get("skip_existing_results"),
            field_name="run.skip_existing_results",
            default=False,
        ),
        False,
    )

    pred_batch = None
    if pred_raw.get("batch_size") not in (None, ""):
        pred_batch = collect(
            lambda: _as_int(
                pred_raw.get("batch_size"),
                field_name="structure_prediction.batch_size",
                min_value=1,
            ),
            None,
        )
    skip_existing = collect(
        lambda: _as_bool(
            pred_raw.get("skip_existing"),
            field_name="structure_prediction.skip_existing",
            default=True,
        ),
        True,
    )

    renumber_imgt = collect(
        lambda: _as_bool(
            proc_raw.get("renumber_imgt"),
            field_name="structure_processing.renumber_imgt",
            default=False,
        ),
        False,
    )
    minimize = collect(
        lambda: _as_bool(
            proc_raw.get("minimize"),
            field_name="structure_processing.minimize",
            default=False,
        ),
        False,
    )
    minimize_attempts = collect(
        lambda: _as_int(
            proc_raw.get("minimize_attempts"),
            field_name="structure_processing.minimize_attempts",
            default=5,
            min_value=1,
        ),
        5,
    )

    include_features = collect(
        lambda: _as_str_list(
            desc_raw.get("include_features"), field_name="descriptors.include_features"
        ),
        [],
    )
    cleanup = collect(
        lambda: _as_bool(
            desc_raw.get("cleanup"), field_name="descriptors.cleanup", default=False
        ),
        False,
    )
    desc_batch = None
    if desc_raw.get("batch_size") not in (None, ""):
        desc_batch = collect(
            lambda: _as_int(
                desc_raw.get("batch_size"),
                field_name="descriptors.batch_size",
                min_value=1,
            ),
            None,
        )
    propka_retries = collect(
        lambda: _as_int(
            desc_raw.get("propka_minimize_retries"),
            field_name="descriptors.propka_minimize_retries",
            default=5,
            min_value=0,
        ),
        5,
    )

    tuning_enabled = collect(
        lambda: _as_bool(
            tuning_raw.get("enabled"), field_name="tuning.enabled", default=False
        ),
        False,
    )
    try:
        margin = float(tuning_raw.get("margin", 0.1))
    except (TypeError, ValueError):
        note(f"tuning.margin must be a number, got {tuning_raw.get('margin')!r}")
        margin = 0.1
    max_rank = collect(
        lambda: _as_int(
            tuning_raw.get("max_rank"),
            field_name="tuning.max_rank",
            default=3,
            min_value=1,
        ),
        3,
    )
    clean_folds_after = collect(
        lambda: _as_bool(
            tuning_raw.get("clean_folds_after"),
            field_name="tuning.clean_folds_after",
            default=True,
        ),
        True,
    )

    if issues:
        raise ConfigError("invalid manifest", issues=issues)
    if output_dir is None:
        raise ConfigError("run.output_dir is required")

    manifest = Manifest(
        source_path=source_path.resolve(),
        legacy=legacy,
        repo_root=repo_root.resolve(),
        inputs=InputsConfig(
            datasets_dir=datasets_dir,
            structures_dir=structures_dir,
            predefined_descriptors_dir=predefined,
            exclude_datasets=exclude_datasets,
            split_randomly=split_randomly,
        ),
        run=RunConfig(
            output_dir=output_dir,
            resume=resume,
            n_cpu=n_cpu,
            skip_existing_results=skip_existing_results,
        ),
        structure_prediction=StructurePredictionConfig(
            enabled=pred_enabled,
            model=list(models),
            device=device,
            batch_size=pred_batch,
            skip_existing=skip_existing,
        ),
        structure_processing=StructureProcessingConfig(
            enabled=proc_enabled,
            renumber_imgt=renumber_imgt,
            minimize=minimize,
            minimize_attempts=minimize_attempts,
        ),
        descriptors=DescriptorsConfig(
            enabled=desc_enabled,
            include_features=include_features,
            cleanup=cleanup,
            batch_size=desc_batch,
            propka_minimize_retries=propka_retries,
        ),
        automl=AutomlConfig(
            enabled=automl_enabled,
            config_path=automl_cfg_path,
            eval_models=eval_models,
        ),
        tuning=TuningConfig(
            enabled=tuning_enabled,
            margin=margin,
            max_rank=max_rank,
            clean_folds_after=clean_folds_after,
        ),
        warnings=warnings,
        stages_override=stages_override,
    )
    manifest.stage_graph()
    return manifest


def load_yaml(path: Path) -> dict[str, Any]:
    yaml = _require_yaml()
    if not path.is_file():
        raise ConfigError(f"file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(_format_yaml_error(path, exc)) from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must be a YAML mapping of keys to values, "
            f"got {type(data).__name__}"
        )
    return data


def is_legacy_config(raw: dict[str, Any]) -> bool:
    legacy_keys = {
        "input_csvs_folder",
        "input_structures_folder",
        "result_folder",
        "predefined_descriptors",
        "csv_features",
        "structures_processing",
        "calculate_descriptors",
    }
    modern = {"inputs", "run"}
    if any(k in raw for k in modern) and "inputs" in raw:
        return False
    return any(k in raw for k in legacy_keys)


def load_manifest(
    path: Path,
    *,
    repo_root: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> Manifest:
    from kitab.legacy import translate_legacy_config

    path = path.resolve()
    raw = load_yaml(path)
    if is_legacy_config(raw):
        translated, warnings = translate_legacy_config(
            raw, source_path=path, repo_root=repo_root
        )
        return parse_manifest_dict(
            translated,
            source_path=path,
            repo_root=repo_root,
            legacy=True,
            warnings=warnings,
            cli_overrides=cli_overrides,
        )
    return parse_manifest_dict(
        raw,
        source_path=path,
        repo_root=repo_root,
        legacy=False,
        cli_overrides=cli_overrides,
    )


def write_resolved_manifest(manifest: Manifest, out_path: Path) -> Path:
    yaml = _require_yaml()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_resolved_dict()
    payload["checksum_sha256"] = manifest.checksum()
    text = (
        "# Resolved kitAb run manifest (generated; do not edit by hand)\n"
        + yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    )
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _rel_or_abs(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def manifest_to_legacy_generic(manifest: Manifest) -> dict[str, Any]:
    """Convert canonical manifest to legacy generic YAML for prepare_run_config."""
    out: dict[str, Any] = {
        "result_folder": _rel_or_abs(manifest.repo_root, manifest.run.output_dir),
    }
    if manifest.inputs.datasets_dir is not None:
        out["input_csvs_folder"] = _rel_or_abs(
            manifest.repo_root, manifest.inputs.datasets_dir
        )
    if manifest.inputs.structures_dir is not None:
        out["input_structures_folder"] = _rel_or_abs(
            manifest.repo_root, manifest.inputs.structures_dir
        )
    if manifest.inputs.predefined_descriptors_dir is not None:
        out["predefined_descriptors"] = {
            "folder": _rel_or_abs(
                manifest.repo_root, manifest.inputs.predefined_descriptors_dir
            ),
        }
        out["calculate_descriptors"] = False
    if manifest.inputs.split_randomly:
        out["split_randomly"] = list(manifest.inputs.split_randomly)
    if manifest.inputs.exclude_datasets:
        out["exclude_datasets"] = list(manifest.inputs.exclude_datasets)
    if not manifest.automl.enabled:
        out["automl"] = False
    if manifest.run.n_cpu is not None:
        out["n_cpu"] = manifest.run.n_cpu
    if manifest.run.skip_existing_results:
        out["skip_existing_results"] = True
    if manifest.descriptors.include_features:
        out["include_features"] = list(manifest.descriptors.include_features)
    if manifest.descriptors.cleanup:
        out["cleanup"] = True
    if manifest.descriptors.batch_size is not None:
        out["descriptor_batch_size"] = manifest.descriptors.batch_size
    out["structures_processing"] = {
        "renumber_imgt": manifest.structure_processing.renumber_imgt,
        "minimize": manifest.structure_processing.minimize,
    }
    if manifest.structure_prediction.enabled:
        sp: dict[str, Any] = {
            "model": manifest.structure_prediction.model,
            "device": manifest.structure_prediction.device,
            "skip_existing": manifest.structure_prediction.skip_existing,
        }
        if manifest.structure_prediction.batch_size is not None:
            sp["batch_size"] = manifest.structure_prediction.batch_size
        out["structure_prediction"] = sp
    if manifest.automl.config_path is not None:
        out["automl_config"] = _rel_or_abs(
            manifest.repo_root, manifest.automl.config_path
        )
    out["hyperparameter_tuning"] = bool(manifest.tuning.enabled)
    return out
