"""Shared pipeline defaults (see ``src/automl.yaml``)."""

from __future__ import annotations

# Single seed for fixed splits, selectors, analysis permutations, descriptors, etc.
DEFAULT_RANDOM_STATE: int = 42

# Three seeds for shuffled CV only (see prepare_run_config ``random_seeds``).
RANDOM_CV_SEEDS: tuple[int, ...] = (42, 43, 44)

DEFAULT_FEATURES_FRACS: tuple[float, ...] = (0.15, 0.10, 0.08, 0.06, 0.04, 0.02)
DEFAULT_FEATURES_FRAC: float = DEFAULT_FEATURES_FRACS[0]
DEFAULT_FEATURES_FRAC_CSV: str = ",".join(str(x) for x in DEFAULT_FEATURES_FRACS)

DEFAULT_EVAL_MODELS: str = "all"

DEFAULT_EVAL_HYPERPARAMETERS_RAW: dict[str, dict] = {
    "elasticnet": {"alpha": 0.01},
    "knn": {"weights": "distance"},
    "gpr": {"alpha": 0.1},
}

DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD: float = 0.01
DEFAULT_LOW_VARIANCE_EPSILON: float = 1e-8

DEFAULT_INTERCORR_THRESHOLD: float = 0.9
DEFAULT_INTERCORR_IMPORTANCE_METRIC: str = "pearson"
DEFAULT_INTERCORR_REDUCTION_MODE: str = "pairwise"

DEFAULT_SFS_MIN_IMPROVEMENT: float = 0.02
