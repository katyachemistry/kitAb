#!/usr/bin/env bash
# Extend the completed abb2 nested CV to every abb2_1 dataset/target pair
# except AB21, PDGF38, and Jain 2024. Existing semantic results are resumed,
# not rerun.
#
# Launch:
#   tmux new -s nested-abb2-extended
#   ./src/run_nested_abb2_all_except_ab21_pdgf38.sh
#
# Optional overrides:
#   OUT_DIR=/path/to/reanalysis PARALLEL_JOBS=64 ./src/run_nested_abb2_all_except_ab21_pdgf38.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/reanalysis_20260818T113409Z}"
PARALLEL_JOBS="${PARALLEL_JOBS:-100}"
FLAT_RESULTS="${FLAT_RESULTS:-${REPO_ROOT}/our_abb2_final_set_of_features_pooled/analysis_results/results.csv}"

if [[ ! -d "${OUT_DIR}/nested_abb2" ]]; then
  echo "Existing nested abb2 run not found: ${OUT_DIR}/nested_abb2" >&2
  echo "Set OUT_DIR to the completed reanalysis directory." >&2
  exit 1
fi
if [[ ! -f "${FLAT_RESULTS}" ]]; then
  echo "Pooled flat results not found: ${FLAT_RESULTS}" >&2
  exit 1
fi
if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_JOBS must be a positive integer, got: ${PARALLEL_JOBS}" >&2
  exit 1
fi

exec "${REPO_ROOT}/src/run_reanalysis.sh" \
  --out-dir "${OUT_DIR}" \
  --resume \
  --stage nested \
  --backend abb2 \
  --parallel-jobs "${PARALLEL_JOBS}" \
  --nested-pairs-mode all_abb2_1 \
  --nested-exclude-stems ab21,pdgf38,jain2024assessment_folded_08_4 \
  --nested-flat-results "${FLAT_RESULTS}"
