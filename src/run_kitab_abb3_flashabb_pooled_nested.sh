#!/usr/bin/env bash
# Recompute pooled OOF metrics and run nested CV for the paper ABB3 and
# FlashABB kitAb feature sets. Nested CV uses every canonical backend_1
# dataset/target pair except AB21, PDGF38, and Jain 2024.
#
# Launch:
#   tmux new -s kitab-abb3-flashabb
#   ./src/run_kitab_abb3_flashabb_pooled_nested.sh --parallel-jobs 100
#
# Resume:
#   ./src/run_kitab_abb3_flashabb_pooled_nested.sh \
#     --out-dir runs/kitab_abb3_flashabb_<stamp> --parallel-jobs 100 --resume

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARALLEL_JOBS="${PARALLEL_JOBS:-100}"
OUT_DIR="${OUT_DIR:-}"
RESUME="0"
DRY_RUN="0"

usage() {
  cat >&2 <<EOF
Usage:
  $0 [--parallel-jobs N] [--out-dir DIR] [--resume] [--dry-run]

Runs pooled OOF validation/aggregation followed by nested CV for ABB3 and
FlashABB. Nested exclusions: AB21, PDGF38, and Jain 2024.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel-jobs)
      PARALLEL_JOBS="${2:?}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?}"
      shift 2
      ;;
    --resume)
      RESUME="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer, got: ${PARALLEL_JOBS}" >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  OUT_DIR="${REPO_ROOT}/runs/kitab_abb3_flashabb_${stamp}"
elif [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi

for backend in abb3 flashabb; do
  automl_root="${REPO_ROOT}/our_${backend}_final_set_of_features/automl"
  if [[ ! -d "${automl_root}" ]]; then
    echo "Missing paper AutoML results for ${backend}: ${automl_root}" >&2
    exit 1
  fi
done

common_args=(
  --out-dir "${OUT_DIR}"
  --stage all
  --parallel-jobs "${PARALLEL_JOBS}"
  --nested-pairs-mode all_backend_1
  --nested-exclude-stems ab21,pdgf38,jain2024assessment_folded_08_4
)
if [[ "${RESUME}" == "1" ]]; then
  common_args+=(--resume)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  common_args+=(--dry-run)
fi

echo "[kitab-abb3-flashabb] output: ${OUT_DIR}" >&2
for backend in abb3 flashabb; do
  echo "[kitab-abb3-flashabb] starting ${backend}" >&2
  "${REPO_ROOT}/src/run_reanalysis.sh" \
    --backend "${backend}" \
    "${common_args[@]}"
done

echo "[kitab-abb3-flashabb] complete" >&2
echo "  pooled ABB3:    ${OUT_DIR}/analysis_abb3/results.csv" >&2
echo "  nested ABB3:    ${OUT_DIR}/nested_abb3_report.csv" >&2
echo "  pooled FlashABB: ${OUT_DIR}/analysis_flashabb/results.csv" >&2
echo "  nested FlashABB: ${OUT_DIR}/nested_flashabb_report.csv" >&2
