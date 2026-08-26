#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/antibody_data/PairedStructures/kitAb"
JOBS=100
OUT_DIR="${REPO_ROOT}/runs/nested_elasticnet_all_methods"
RESUME=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash src/run_nested_elasticnet_all_methods.sh [options]

Options:
  --jobs N          Parallel outer-fold workers (default: 100)
  --out-dir PATH    Checkpoints/results directory
  --fresh           Refuse to resume; output directory must be empty
  --dry-run         Validate roots and print task counts without fitting
  -h, --help        Show this help

The default is resumable. Every completed outer fold is checkpointed.
After all fits finish, the publication-style averaged plot is generated as:
  <out-dir>/plots/method_comparison_spearman_elasticnet.{png,tiff,pdf}
Use analysis/plot_method_comparison_spearman_elasticnet.py --publish to copy into
analysis/plots.
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
  python -P "${REPO_ROOT}/src/analysis/nested_elasticnet_all_methods.py"
  --jobs "${JOBS}"
  --out-dir "${OUT_DIR}"
)
if [[ "${RESUME}" == "1" ]]; then
  runner+=(--resume)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  runner+=(--dry-run)
fi

echo "[nested-elasticnet] workers=${JOBS} output=${OUT_DIR} resume=${RESUME}"
"${runner[@]}"

if [[ "${DRY_RUN}" == "0" ]]; then
  conda run --no-capture-output -n bio-ds \
    python -P "${REPO_ROOT}/analysis/plot_method_comparison_spearman_elasticnet.py" \
    --run-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}/plots"
fi
