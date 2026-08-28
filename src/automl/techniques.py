"""The four kitAb AutoML techniques and the settings that parameterise them.

Every technique in :data:`DEFAULT_TECHNIQUES` is evaluated in parallel
(technique × outer fold × target). Nested CV then chooses the technique from
inner-fold Spearman, pools those mixed out-of-fold predictions as the
procedure score, and refits that inner-chosen technique on all labelled rows.
ElasticNet ``(alpha, l1_ratio)`` and the SFS eval model are taken from the
nested run (mode of the per-fold inner choices); there is no outer-test
selection and no separate hyperparameter-tuning stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl.pipeline_defaults import (
    DEFAULT_CV_MODE,
    DEFAULT_ELASTICNET_ALPHAS,
    DEFAULT_ELASTICNET_L1_RATIOS,
    DEFAULT_EVAL_HYPERPARAMETERS_RAW,
    DEFAULT_FEATURES_FRAC,
    DEFAULT_INTERCORR_IMPORTANCE_METRIC,
    DEFAULT_INTERCORR_REDUCTION_MODE,
    DEFAULT_INTERCORR_THRESHOLD,
    DEFAULT_LOW_VARIANCE_EPSILON,
    DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD,
    DEFAULT_N_SPLITS,
    DEFAULT_PERMUTATION_REPEATS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SFS_EVAL_MODELS,
    DEFAULT_TECHNIQUE_SELECTION,
    DEFAULT_SFS_INNER_CV,
    DEFAULT_SFS_MIN_IMPROVEMENT,
    DEFAULT_TECHNIQUES,
)

CV_MODES = ("nested", "flat")


class TechniqueConfigError(ValueError):
    """Invalid AutoML pipeline configuration."""


@dataclass(frozen=True)
class Technique:
    """One of the four fixed feature-selection + regression pipelines."""

    key: str
    label: str
    kind: str  # "selector" or "elasticnet"
    apply_intercorr: bool
    selector: str | None = None  # None or "sfs"
    selector_model: str | None = None
    features_frac: float | None = None
    eval_models: tuple[str, ...] = ()
    alphas: tuple[float, ...] = ()
    l1_ratios: tuple[float, ...] = ()

    @property
    def searches_eval_model(self) -> bool:
        return len(self.eval_models) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "apply_intercorr": self.apply_intercorr,
            "selector": self.selector,
            "selector_model": self.selector_model,
            "features_frac": self.features_frac,
            "eval_models": list(self.eval_models),
            "alphas": list(self.alphas),
            "l1_ratios": list(self.l1_ratios),
        }


@dataclass(frozen=True)
class SfsSettings:
    features_frac: float = DEFAULT_FEATURES_FRAC
    min_improvement: float = DEFAULT_SFS_MIN_IMPROVEMENT
    inner_cv: int = DEFAULT_SFS_INNER_CV
    eval_models: tuple[str, ...] = DEFAULT_SFS_EVAL_MODELS


@dataclass(frozen=True)
class ElasticNetSettings:
    alphas: tuple[float, ...] = DEFAULT_ELASTICNET_ALPHAS
    l1_ratios: tuple[float, ...] = DEFAULT_ELASTICNET_L1_RATIOS


@dataclass(frozen=True)
class CvSettings:
    mode: str = DEFAULT_CV_MODE
    n_splits: int = DEFAULT_N_SPLITS
    technique_selection: str = DEFAULT_TECHNIQUE_SELECTION


@dataclass(frozen=True)
class PipelineSettings:
    """Everything the technique runner needs, resolved from ``automl.yaml``."""

    techniques: tuple[str, ...] = DEFAULT_TECHNIQUES
    random_state: int = DEFAULT_RANDOM_STATE
    cv: CvSettings = field(default_factory=CvSettings)
    sfs: SfsSettings = field(default_factory=SfsSettings)
    elasticnet: ElasticNetSettings = field(default_factory=ElasticNetSettings)
    eval_hyperparameters: dict[str, dict] = field(
        default_factory=lambda: {
            model: dict(params)
            for model, params in DEFAULT_EVAL_HYPERPARAMETERS_RAW.items()
        }
    )
    low_variance_relative_std_threshold: float = (
        DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD
    )
    low_variance_epsilon: float = DEFAULT_LOW_VARIANCE_EPSILON
    intercorr_threshold: float = DEFAULT_INTERCORR_THRESHOLD
    intercorr_importance_metric: str = DEFAULT_INTERCORR_IMPORTANCE_METRIC
    intercorr_reduction_mode: str = DEFAULT_INTERCORR_REDUCTION_MODE
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS
    save_final_model: bool = True

    def build_techniques(self) -> list[Technique]:
        return [build_technique(key, self) for key in self.techniques]

    def to_dict(self) -> dict[str, Any]:
        return {
            "techniques": list(self.techniques),
            "random_state": self.random_state,
            "cv": {
                "mode": self.cv.mode,
                "n_splits": self.cv.n_splits,
                "technique_selection": self.cv.technique_selection,
            },
            "sfs": {
                "features_frac": self.sfs.features_frac,
                "min_improvement": self.sfs.min_improvement,
                "inner_cv": self.sfs.inner_cv,
                "eval_models": list(self.sfs.eval_models),
            },
            "elasticnet": {
                "alphas": list(self.elasticnet.alphas),
                "l1_ratios": list(self.elasticnet.l1_ratios),
            },
            "eval_hyperparameters": {
                model: dict(params)
                for model, params in sorted(self.eval_hyperparameters.items())
            },
            "low_variance_relative_std_threshold": (
                self.low_variance_relative_std_threshold
            ),
            "low_variance_epsilon": self.low_variance_epsilon,
            "intercorr_threshold": self.intercorr_threshold,
            "intercorr_importance_metric": self.intercorr_importance_metric,
            "intercorr_reduction_mode": self.intercorr_reduction_mode,
            "permutation_repeats": self.permutation_repeats,
            "save_final_model": self.save_final_model,
        }


def build_technique(key: str, settings: PipelineSettings) -> Technique:
    """Instantiate one technique from its key and the shared settings."""
    name = str(key).strip().lower()
    if name == "intercorr_svm":
        return Technique(
            key=name,
            label="Intercorrelation pruning + SVM",
            kind="selector",
            apply_intercorr=True,
            eval_models=("svm",),
        )
    if name in ("sfs_svm", "sfs_knn"):
        selector_model = "svm" if name == "sfs_svm" else "knn"
        return Technique(
            key=name,
            label=f"SFS ({selector_model.upper()} selector)",
            kind="selector",
            apply_intercorr=True,
            selector="sfs",
            selector_model=selector_model,
            features_frac=settings.sfs.features_frac,
            eval_models=tuple(settings.sfs.eval_models),
        )
    if name == "elasticnet":
        return Technique(
            key=name,
            label="ElasticNet",
            kind="elasticnet",
            apply_intercorr=False,
            eval_models=("elasticnet",),
            alphas=tuple(settings.elasticnet.alphas),
            l1_ratios=tuple(settings.elasticnet.l1_ratios),
        )
    raise TechniqueConfigError(
        f"Unknown technique {key!r}; supported: {', '.join(DEFAULT_TECHNIQUES)}"
    )


def _as_str_tuple(raw: Any, *, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(p).strip() for p in raw if str(p).strip()]
    else:
        raise TechniqueConfigError(
            f"{field_name} must be a list or comma-separated string, "
            f"got {type(raw).__name__}"
        )
    seen: list[str] = []
    for part in parts:
        if part.lower() not in seen:
            seen.append(part.lower())
    return tuple(seen)


def _as_float_tuple(raw: Any, *, field_name: str) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: list[Any] = [p for p in raw.replace(",", " ").split() if p]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise TechniqueConfigError(f"{field_name} must be a list of numbers")
    out: list[float] = []
    for item in items:
        try:
            out.append(float(item))
        except (TypeError, ValueError) as exc:
            raise TechniqueConfigError(
                f"{field_name} must contain numbers, got {item!r}"
            ) from exc
    if not out:
        raise TechniqueConfigError(f"{field_name} must not be empty")
    return tuple(out)


def _as_number(raw: Any, default: float, *, field_name: str) -> float:
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise TechniqueConfigError(f"{field_name} must be a number, got {raw!r}") from exc


def _as_int(raw: Any, default: int, *, field_name: str, minimum: int = 1) -> int:
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise TechniqueConfigError(
            f"{field_name} must be an integer, got {raw!r}"
        ) from exc
    if value < minimum:
        raise TechniqueConfigError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _as_bool(raw: Any, default: bool, *, field_name: str) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    raise TechniqueConfigError(f"{field_name} must be a boolean, got {raw!r}")


def _as_mapping(raw: Any, *, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    raise TechniqueConfigError(f"{field_name} must be a mapping, got {type(raw).__name__}")


_KNOWN_PIPELINE_KEYS = frozenset(
    {
        "techniques",
        "random_state",
        "cv",
        "sfs",
        "elasticnet",
        "eval_hyperparameters",
        "low_variance_relative_std_threshold",
        "low_variance_epsilon",
        "intercorr_threshold",
        "intercorr_importance_metric",
        "intercorr_reduction_mode",
        "permutation_repeats",
        "save_final_model",
    }
)
_KNOWN_CV_KEYS = frozenset({"mode", "n_splits", "technique_selection"})
_KNOWN_SFS_KEYS = frozenset({"features_frac", "min_improvement", "inner_cv", "eval_models"})
_KNOWN_ELASTICNET_KEYS = frozenset({"alphas", "l1_ratios"})


def pipeline_settings_from_block(block: dict[str, Any] | None) -> PipelineSettings:
    """Build :class:`PipelineSettings` from the ``pipeline:`` mapping."""
    raw = _as_mapping(block, field_name="pipeline")
    unknown = sorted(set(raw) - _KNOWN_PIPELINE_KEYS)
    if unknown:
        raise TechniqueConfigError(
            f"unknown pipeline key(s): {', '.join(unknown)} "
            f"(known: {', '.join(sorted(_KNOWN_PIPELINE_KEYS))})"
        )

    techniques = _as_str_tuple(raw.get("techniques"), field_name="pipeline.techniques")
    if not techniques:
        techniques = DEFAULT_TECHNIQUES
    for key in techniques:
        if key not in DEFAULT_TECHNIQUES:
            raise TechniqueConfigError(
                f"Unknown technique {key!r} in pipeline.techniques; "
                f"supported: {', '.join(DEFAULT_TECHNIQUES)}"
            )

    cv_raw = _as_mapping(raw.get("cv"), field_name="pipeline.cv")
    unknown_cv = sorted(set(cv_raw) - _KNOWN_CV_KEYS)
    if unknown_cv:
        raise TechniqueConfigError(f"unknown pipeline.cv key(s): {', '.join(unknown_cv)}")
    cv_mode = str(cv_raw.get("mode") or DEFAULT_CV_MODE).strip().lower()
    if cv_mode not in CV_MODES:
        raise TechniqueConfigError(
            f"pipeline.cv.mode must be one of {', '.join(CV_MODES)}, got {cv_mode!r}"
        )
    technique_selection = str(
        cv_raw.get("technique_selection") or DEFAULT_TECHNIQUE_SELECTION
    ).strip().lower()
    if technique_selection != "inner":
        raise TechniqueConfigError(
            "pipeline.cv.technique_selection is always nested inner-CV "
            f"(got {technique_selection!r}); outer/best-of-four selection is not supported"
        )

    sfs_raw = _as_mapping(raw.get("sfs"), field_name="pipeline.sfs")
    unknown_sfs = sorted(set(sfs_raw) - _KNOWN_SFS_KEYS)
    if unknown_sfs:
        raise TechniqueConfigError(
            f"unknown pipeline.sfs key(s): {', '.join(unknown_sfs)}"
        )
    sfs_eval_models = _as_str_tuple(
        sfs_raw.get("eval_models"), field_name="pipeline.sfs.eval_models"
    )
    features_frac = _as_number(
        sfs_raw.get("features_frac"),
        DEFAULT_FEATURES_FRAC,
        field_name="pipeline.sfs.features_frac",
    )
    if not 0.0 < features_frac <= 1.0:
        raise TechniqueConfigError(
            f"pipeline.sfs.features_frac must be in (0, 1], got {features_frac}"
        )

    enet_raw = _as_mapping(raw.get("elasticnet"), field_name="pipeline.elasticnet")
    unknown_enet = sorted(set(enet_raw) - _KNOWN_ELASTICNET_KEYS)
    if unknown_enet:
        raise TechniqueConfigError(
            f"unknown pipeline.elasticnet key(s): {', '.join(unknown_enet)}"
        )

    eval_hp = _as_mapping(
        raw.get("eval_hyperparameters"), field_name="pipeline.eval_hyperparameters"
    )
    eval_hyperparameters = {
        model: dict(params) for model, params in DEFAULT_EVAL_HYPERPARAMETERS_RAW.items()
    }
    for model, params in eval_hp.items():
        eval_hyperparameters[str(model).strip().lower()] = _as_mapping(
            params, field_name=f"pipeline.eval_hyperparameters.{model}"
        )

    return PipelineSettings(
        techniques=techniques,
        random_state=_as_int(
            raw.get("random_state"),
            DEFAULT_RANDOM_STATE,
            field_name="pipeline.random_state",
            minimum=0,
        ),
        cv=CvSettings(
            mode=cv_mode,
            n_splits=_as_int(
                cv_raw.get("n_splits"),
                DEFAULT_N_SPLITS,
                field_name="pipeline.cv.n_splits",
                minimum=2,
            ),
            technique_selection=technique_selection,
        ),
        sfs=SfsSettings(
            features_frac=features_frac,
            min_improvement=_as_number(
                sfs_raw.get("min_improvement"),
                DEFAULT_SFS_MIN_IMPROVEMENT,
                field_name="pipeline.sfs.min_improvement",
            ),
            inner_cv=_as_int(
                sfs_raw.get("inner_cv"),
                DEFAULT_SFS_INNER_CV,
                field_name="pipeline.sfs.inner_cv",
                minimum=2,
            ),
            eval_models=sfs_eval_models or DEFAULT_SFS_EVAL_MODELS,
        ),
        elasticnet=ElasticNetSettings(
            alphas=(
                _as_float_tuple(enet_raw.get("alphas"), field_name="pipeline.elasticnet.alphas")
                or DEFAULT_ELASTICNET_ALPHAS
            ),
            l1_ratios=(
                _as_float_tuple(
                    enet_raw.get("l1_ratios"), field_name="pipeline.elasticnet.l1_ratios"
                )
                or DEFAULT_ELASTICNET_L1_RATIOS
            ),
        ),
        eval_hyperparameters=eval_hyperparameters,
        low_variance_relative_std_threshold=_as_number(
            raw.get("low_variance_relative_std_threshold"),
            DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD,
            field_name="pipeline.low_variance_relative_std_threshold",
        ),
        low_variance_epsilon=_as_number(
            raw.get("low_variance_epsilon"),
            DEFAULT_LOW_VARIANCE_EPSILON,
            field_name="pipeline.low_variance_epsilon",
        ),
        intercorr_threshold=_as_number(
            raw.get("intercorr_threshold"),
            DEFAULT_INTERCORR_THRESHOLD,
            field_name="pipeline.intercorr_threshold",
        ),
        intercorr_importance_metric=str(
            raw.get("intercorr_importance_metric")
            or DEFAULT_INTERCORR_IMPORTANCE_METRIC
        ).strip().lower(),
        intercorr_reduction_mode=str(
            raw.get("intercorr_reduction_mode") or DEFAULT_INTERCORR_REDUCTION_MODE
        ).strip().lower(),
        permutation_repeats=_as_int(
            raw.get("permutation_repeats"),
            DEFAULT_PERMUTATION_REPEATS,
            field_name="pipeline.permutation_repeats",
            minimum=0,
        ),
        save_final_model=_as_bool(
            raw.get("save_final_model"), True, field_name="pipeline.save_final_model"
        ),
    )


def default_automl_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "automl.yaml"


def load_pipeline_settings(path: Path | None = None) -> PipelineSettings:
    """Read ``automl.yaml`` (or the default) into :class:`PipelineSettings`."""
    import yaml

    config_path = Path(path) if path is not None else default_automl_config_path()
    if not config_path.is_file():
        raise TechniqueConfigError(f"AutoML config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TechniqueConfigError(f"{config_path} must be a YAML mapping")
    unknown = sorted(set(raw) - {"pipeline"})
    if unknown:
        raise TechniqueConfigError(
            f"{config_path}: unknown top-level key(s): {', '.join(unknown)} "
            "(only 'pipeline' is supported)"
        )
    return pipeline_settings_from_block(raw.get("pipeline"))


def parse_technique_list(raw: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Parse a comma-separated or list of technique names."""
    if raw is None or raw == "":
        return DEFAULT_TECHNIQUES
    names = _as_str_tuple(raw, field_name="techniques")
    if not names:
        return DEFAULT_TECHNIQUES
    out: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key not in DEFAULT_TECHNIQUES:
            raise TechniqueConfigError(
                f"Unknown technique {name!r}; supported: {', '.join(DEFAULT_TECHNIQUES)}"
            )
        if key not in out:
            out.append(key)
    return tuple(out)


def apply_pipeline_cli_overrides(
    settings: PipelineSettings,
    *,
    techniques: str | list[str] | None = None,
    cv_mode: str | None = None,
    technique_selection: str | None = None,
    no_final_model: bool = False,
) -> PipelineSettings:
    """Apply ``run_automl.py`` / kitAb CLI overrides on top of ``automl.yaml``."""
    from dataclasses import replace

    updated = settings
    if techniques is not None:
        names = parse_technique_list(techniques)
        for key in names:
            build_technique(key, settings)
        updated = replace(updated, techniques=names)
    if cv_mode is not None:
        mode = str(cv_mode).strip().lower()
        if mode not in CV_MODES:
            raise TechniqueConfigError(
                f"cv_mode must be one of {', '.join(CV_MODES)}, got {mode!r}"
            )
        updated = replace(updated, cv=replace(updated.cv, mode=mode))
    if technique_selection is not None:
        selection = str(technique_selection).strip().lower()
        if selection != "inner":
            raise TechniqueConfigError(
                "technique_selection is always nested inner-CV; "
                f"got {selection!r}"
            )
    if no_final_model:
        updated = replace(updated, save_final_model=False)
    return updated
