#!/usr/bin/env bash
# Run the four kitAb AutoML techniques, pick the best one and fit it on all data.
#
# Usage:
#   run_automl.sh --config PATH/run_config.yaml [--jobs N] [--resume] [--dry-run] ...
#
# Every other flag is forwarded to automl/run_automl.py (see --help there).

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_automl.sh --config PATH/run_config.yaml [options]

Options are passed through to src/automl/run_automl.py, notably:
  --automl-config PATH   pipeline YAML (default: src/automl.yaml)
  --jobs N               worker processes
  --models-root PATH     where to write estimator.joblib
  --resume               reuse outer-fold checkpoints
  --force-preprocess     rebuild fold parquets
  --cv-mode MODE        nested or flat (default: from automl.yaml)
  --no-final-model       compare techniques without fitting the final model
  --dry-run              print the work plan and exit
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/kitab.local.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/kitab.local.env"
fi

KITAB_ENV="${KITAB_ENV:-kitab}"
PY="${PY:-conda run --no-capture-output -n ${KITAB_ENV} python}"

if [[ $# -eq 0 || "${1}" == "-h" || "${1}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1}" != "--config" ]]; then
  echo "Expected --config PATH as the first argument." >&2
  usage
  exit 1
fi
shift

if [[ $# -lt 1 ]]; then
  echo "Missing YAML path after --config" >&2
  exit 1
fi
CONFIG_FILE="$1"
shift

echo >&2 "[run_automl] config: ${CONFIG_FILE}"

# shellcheck disable=SC2086
exec ${PY} "${REPO_ROOT}/src/automl/run_automl.py" "${CONFIG_FILE}" --py "${PY}" "$@"
