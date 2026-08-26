#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/antibody_data/PairedStructures/kitAb"
JOBS=100
OUT_DIR="${REPO_ROOT}/runs/nested_pipeline_all_methods"
RESUME=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash src/run_nested_pipeline_all_methods.sh [options]

Options:
  --jobs N          Parallel outer-fold workers (default: 100)
  --out-dir PATH    Checkpoints/results directory
  --fresh           Refuse to resume; output directory must be empty
  --dry-run         Validate roots and print task counts without fitting
  -h, --help        Show this help

Pipelines (nested CV, all prepared fold roots):
  1. intercorr_svm   low-var + intercorr prune + SVM (all survivors)
  2. sfs_svm         low-var + intercorr + SFS (SVM, frac=0.15) + eval {svm,knn,linear,rf}
  3. sfs_knn         low-var + intercorr + SFS (KNN, frac=0.15) + eval {svm,knn,linear,rf}
  4. elasticnet      nested alpha/l1 ElasticNet on all input features

Artifacts include OOF predictions, R2/Pearson/MSE, feature lineage,
coefficients/importances, permutation importance, and inner-selection tables.
target_Fab_pI is excluded. Default is resumable via checkpoints under:
  <out-dir>/checkpoints/<pipeline>/...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --fresh)
      RESUME=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

runner=(
  conda run --no-capture-output -n bio-ds
  python -P "${REPO_ROOT}/src/analysis/nested_pipeline_all_methods.py"
  --jobs "${JOBS}"
  --out-dir "${OUT_DIR}"
)
if [[ "${RESUME}" == "1" ]]; then
  runner+=(--resume)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  runner+=(--dry-run)
fi

echo "[nested-pipeline] workers=${JOBS} output=${OUT_DIR} resume=${RESUME}"
"${runner[@]}"
