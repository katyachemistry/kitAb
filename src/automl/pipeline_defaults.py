"""Defaults for the kitAb AutoML pipeline (mirrors ``src/automl.yaml``)."""

from __future__ import annotations

# Single seed for fixed splits, selectors, analysis permutations, descriptors, etc.
DEFAULT_RANDOM_STATE: int = 42

# Three seeds for shuffled CV only (see prepare_run_config ``random_seeds``).
RANDOM_CV_SEEDS: tuple[int, ...] = (42, 43, 44)

# The four techniques, run one after another for every target.
DEFAULT_TECHNIQUES: tuple[str, ...] = (
    "elasticnet",
    "intercorr_svm",
    "sfs_svm",
    "sfs_knn",
)

DEFAULT_CV_MODE: str = "nested"
DEFAULT_N_SPLITS: int = 5

# Fraction of training rows used as the SFS feature budget. prepare_run.py also
# records this in fold meta.json as ``features_frac``.
DEFAULT_FEATURES_FRAC: float = 0.15

DEFAULT_EVAL_MODELS: str = "all"

# Fixed eval-model settings; the pipeline does not search hyperparameters.
DEFAULT_EVAL_HYPERPARAMETERS_RAW: dict[str, dict] = {
    "elasticnet": {"alpha": 0.01},
    "knn": {"weights": "distance"},
}

DEFAULT_EVAL_MODEL_ORDER: tuple[str, ...] = (
    "linear",
    "elasticnet",
    "randomforest",
    "svm",
    "knn",
)

DEFAULT_LOW_VARIANCE_RELATIVE_STD_THRESHOLD: float = 0.01
DEFAULT_LOW_VARIANCE_EPSILON: float = 1e-8

DEFAULT_INTERCORR_THRESHOLD: float = 0.9
DEFAULT_INTERCORR_IMPORTANCE_METRIC: str = "pearson"
DEFAULT_INTERCORR_REDUCTION_MODE: str = "pairwise"

DEFAULT_SFS_MIN_IMPROVEMENT: float = 0.02
DEFAULT_SFS_INNER_CV: int = 5
DEFAULT_SFS_EVAL_MODELS: tuple[str, ...] = ("svm", "knn", "linear", "randomforest")

DEFAULT_ELASTICNET_ALPHAS: tuple[float, ...] = (
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
)
DEFAULT_ELASTICNET_L1_RATIOS: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
)

DEFAULT_PERMUTATION_REPEATS: int = 10
