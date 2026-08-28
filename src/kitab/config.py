"""Canonical run-manifest schema and validation for kitAb."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STRUCTURE_BACKENDS = ("abb2", "abb3", "flashabb")
SUPPORTED_TECHNIQUES = ("elasticnet", "intercorr_svm", "sfs_svm", "sfs_knn")
CV_MODES = ("nested", "flat")

KNOWN_TOP_KEYS = frozenset(
    {
        "inputs",
        "run",
        "structure_prediction",
        "structure_processing",
        "descriptors",
        "automl",
    }
)
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
    {"output_dir", "resume", "n_cpu", "skip_existing_results"}
)
KNOWN_PRED_KEYS = frozenset(
    {"enabled", "model", "device", "batch_size", "skip_existing", "runs"}
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
KNOWN_AUTOML_KEYS = frozenset({"enabled"})


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
    runs: int = 1


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
    techniques: list[str] = field(default_factory=lambda: list(SUPPORTED_TECHNIQUES))
    cv_mode: str = "nested"
    technique_selection: str = "inner"
    save_final_model: bool = True


@dataclass
class Manifest:
    source_path: Path
    repo_root: Path
    inputs: InputsConfig
    run: RunConfig
    structure_prediction: StructurePredictionConfig
    structure_processing: StructureProcessingConfig
    descriptors: DescriptorsConfig
    automl: AutomlConfig
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
                graph.append("completeness")
            graph.append("automl")
        return graph

    def to_resolved_dict(self) -> dict[str, Any]:
        def _path(p: Path | None) -> str | None:
            return None if p is None else str(p)

        return {
            "source_path": str(self.source_path),
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
                "techniques": list(self.automl.techniques),
                "cv_mode": self.automl.cv_mode,
                "technique_selection": self.automl.technique_selection,
                "save_final_model": self.automl.save_final_model,
            },
        }

    def checksum(self) -> str:
        payload = json.dumps(
            self.to_resolved_dict(), sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_techniques(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return list(SUPPORTED_TECHNIQUES)
    names = _as_str_list(raw, field_name="automl.techniques")
    if not names:
        return list(SUPPORTED_TECHNIQUES)
    out: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key not in SUPPORTED_TECHNIQUES:
            raise ConfigError(
                f"Unknown technique {name!r}. "
                f"Supported: {', '.join(SUPPORTED_TECHNIQUES)}"
            )
        if key not in out:
            out.append(key)
    return out


def _validate_cv_mode(raw: Any) -> str:
    if raw is None or str(raw).strip() == "":
        return "nested"
    mode = str(raw).strip().lower()
    if mode not in CV_MODES:
        raise ConfigError(
            f"automl.cv_mode must be one of {', '.join(CV_MODES)}, got {mode!r}"
        )
    return mode


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

    note(_unknown_keys(raw, KNOWN_TOP_KEYS, "top-level"))

    warnings = list(warnings or [])
    overrides = dict(cli_overrides or {})

    inputs_raw = collect(lambda: _as_mapping(raw.get("inputs"), field_name="inputs"), {})
    run_raw = collect(lambda: _as_mapping(raw.get("run"), field_name="run"), {})
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
    _cli_owned_automl = (
        ("techniques", "automl.techniques is ignored; pass --techniques on the CLI."),
        ("cv_mode", "automl.cv_mode is ignored; pass --cv-mode on the CLI."),
        (
            "technique_selection",
            "automl.technique_selection is ignored; nested inner-CV selection is always used.",
        ),
        ("save_final_model", "automl.save_final_model is ignored; pass --no-final-model on the CLI."),
    )
    for key, msg in _cli_owned_automl:
        if automl_raw.pop(key, None) is not None:
            warnings.append(msg)

    note(_unknown_keys(inputs_raw, KNOWN_INPUT_KEYS, "inputs"))
    note(_unknown_keys(run_raw, KNOWN_RUN_KEYS, "run"))
    note(_unknown_keys(pred_raw, KNOWN_PRED_KEYS, "structure_prediction"))
    note(_unknown_keys(proc_raw, KNOWN_PROC_KEYS, "structure_processing"))
    note(_unknown_keys(desc_raw, KNOWN_DESC_KEYS, "descriptors"))
    note(_unknown_keys(automl_raw, KNOWN_AUTOML_KEYS, "automl"))

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

    techniques = list(SUPPORTED_TECHNIQUES)
    cv_mode = "nested"
    save_final_model = True
    if overrides.get("techniques") is not None:
        techniques = collect(
            lambda: _validate_techniques(overrides["techniques"]),
            list(SUPPORTED_TECHNIQUES),
        )
    if overrides.get("cv_mode") is not None:
        cv_mode = collect(
            lambda: _validate_cv_mode(overrides["cv_mode"]),
            "nested",
        )
    if overrides.get("no_final_model"):
        save_final_model = False

    stages_override = collect(
        lambda: _as_str_list(
            overrides.get("stages"),
            field_name="stages",
        ),
        [],
    )

    output_raw = run_raw.get("output_dir")
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
    pred_runs = collect(
        lambda: _as_int(
            pred_raw.get("runs"),
            field_name="structure_prediction.runs",
            default=1,
            min_value=1,
        ),
        1,
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

    if issues:
        raise ConfigError("invalid manifest", issues=issues)
    if output_dir is None:
        raise ConfigError("run.output_dir is required")

    manifest = Manifest(
        source_path=source_path.resolve(),
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
            runs=pred_runs,
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
            techniques=techniques,
            cv_mode=cv_mode,
            save_final_model=save_final_model,
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


def load_manifest(
    path: Path,
    *,
    repo_root: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> Manifest:
    path = path.resolve()
    raw = load_yaml(path)
    return parse_manifest_dict(
        raw,
        source_path=path,
        repo_root=repo_root,
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


