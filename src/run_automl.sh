#!/usr/bin/env bash
# CV fold prep + parallel fold workers (GNU parallel). Usage: run_automl.sh --config YAML | legacy args (see --help).

set -euo pipefail

_stage() {
  echo >&2 "[run_automl] $*"
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 --config PATH/config.yaml [CLI overrides: --parallel-jobs, --py, --no-preprocessing-skip, ...]
  $0 [OPTIONS] DATASET DEVELOPABILITY TARGETS_CSV FEATURES_CSV [RUN_DIR]

Legacy options: --name-col, --n-splits, --random-state, --features-frac, --jobs-file,
  --parallel-jobs, --py, --selector-name (required), --model-to-use, --eval-models,
  --result-root, --no-preprocessing-skip, --no-clean-folds
  -h, --help
EOF
}

_trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

_csv_to_array() {
  local _csv=$1
  local -n _dest=$2
  _dest=()
  local IFS=,
  local -a _raw
  read -ra _raw <<< "${_csv}"
  local _p
  for _p in "${_raw[@]}"; do
    _p="$(_trim "${_p}")"
    [[ -n "${_p}" ]] && _dest+=("${_p}")
  done
}

_phase1_cache_ok() {
  local jf="$1"
  [[ -f "${jf}" ]] || return 1
  [[ -s "${jf}" ]] || return 1
  local line fold_dir k _ds _tg
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" == *$'\t'* ]] || return 1
    IFS=$'\t' read -r fold_dir k _ds _tg <<< "${line}"
    [[ -n "${fold_dir:-}" ]] || return 1
    [[ -n "${k:-}" ]] || return 1
    [[ -n "${_ds:-}" ]] || return 1
    [[ -n "${_tg:-}" ]] || return 1
    [[ -d "${fold_dir}" ]] || return 1
    [[ -f "${fold_dir}/meta.json" ]] || return 1
    [[ -f "${fold_dir}/fold_${k}_train.parquet" ]] || return 1
    [[ -f "${fold_dir}/fold_${k}_test.parquet" ]] || return 1
  done < "${jf}"
  return 0
}

_py_to_array() {
  local _py_cmd="$1"
  local -n _out=$2
  _out=()
  mapfile -t _out < <(python3 -c 'import shlex,sys
for part in shlex.split(sys.argv[1]):
    print(part)' "$_py_cmd")
  if [[ ${#_out[@]} -eq 0 ]]; then
    echo "Error: --py command is empty." >&2
    exit 1
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--config" ]]; then
  shift
  if [[ $# -lt 1 ]]; then
    echo "Missing YAML path after --config" >&2
    exit 1
  fi
  CONFIG_FILE="$1"
  shift
  _stage "Config mode: ${CONFIG_FILE}"
  PY="${PY:-conda run -n developability python}"
  run_py_cfg() {
    # shellcheck disable=SC2086
    ${PY} "$@"
  }
  run_py_cfg "${REPO_ROOT}/src/automl/prepare_parallel_from_config.py" "${CONFIG_FILE}" "$@"
  exit $?
fi

NAME_COL="name"
N_SPLITS="5"
RANDOM_STATE="42"
FEATURES_FRAC="0.1"
JOBS_FILE=""
PARALLEL_JOBS=""
PY=""
SELECTOR_NAME=""
MODEL_TO_USE="elasticnet"
EVAL_MODELS="all"
RESULT_ROOT=""
NO_PREPROCESSING_SKIP="0"
NO_CLEAN_FOLDS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --name-col)
      NAME_COL="${2:?}"
      shift 2
      ;;
    --n-splits)
      N_SPLITS="${2:?}"
      shift 2
      ;;
    --random-state)
      RANDOM_STATE="${2:?}"
      shift 2
      ;;
    --features-frac)
      FEATURES_FRAC="${2:?}"
      shift 2
      ;;
    --jobs-file)
      JOBS_FILE="${2:?}"
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="${2:?}"
      shift 2
      ;;
    --py)
      PY="${2:?}"
      shift 2
      ;;
    --selector-name)
      SELECTOR_NAME="${2:?}"
      shift 2
      ;;
    --model-to-use)
      MODEL_TO_USE="${2:?}"
      shift 2
      ;;
    --eval-models)
      EVAL_MODELS="${2:?}"
      shift 2
      ;;
    --result-root)
      RESULT_ROOT="${2:?}"
      shift 2
      ;;
    --no-preprocessing-skip)
      NO_PREPROCESSING_SKIP="1"
      shift
      ;;
    --no-clean-folds)
      NO_CLEAN_FOLDS="1"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1 (try --help)" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

PY="${PY:-conda run -n developability python}"

if [[ $# -lt 4 ]]; then
  usage
  exit 1
fi

if [[ -z "${SELECTOR_NAME}" ]]; then
  echo "Missing required --selector-name (stability or correlation)." >&2
  exit 1
fi

_stage "Legacy mode — invoking single-dataset prepare + parallel (selector=${SELECTOR_NAME}, model=${MODEL_TO_USE})"

DATASET=$1
DEVELOPABILITY=$2
TARGETS_CSV=$3
FEATURES_CSV=$4
shift 4

_stem=$(basename "${DATASET}")
_stem="${_stem%.*}"

mkdir -p "${REPO_ROOT}/tmp"
if [[ $# -ge 1 ]]; then
  RUN_DIR="${1}"
else
  RUN_DIR=$(mktemp -d "${REPO_ROOT}/tmp/cv_prepare.XXXXXX")
  echo >&2 "Using temporary RUN_DIR=${RUN_DIR} (pass RUN_DIR as last positional arg to reuse phase 1 across runs)."
fi

if [[ -z "${JOBS_FILE}" ]]; then
  JOBS_FILE="${RUN_DIR}/parallel_jobs.txt"
fi
if [[ -z "${PARALLEL_JOBS}" ]]; then
  PARALLEL_JOBS="$(nproc 2>/dev/null || echo 4)"
fi
if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --parallel-jobs must be a positive integer, got: ${PARALLEL_JOBS}" >&2
  exit 1
fi
if [[ -z "${RESULT_ROOT}" ]]; then
  RESULT_ROOT="${REPO_ROOT}/runs/${_stem}_fold_results"
fi

_stage "Paths — DATASET=${DATASET} DEVELOPABILITY=${DEVELOPABILITY} RUN_DIR=${RUN_DIR} JOBS_FILE=${JOBS_FILE} RESULT_ROOT=${RESULT_ROOT} parallel_jobs=${PARALLEL_JOBS}"

_csv_to_array "${TARGETS_CSV}" TARGET_COLS
_csv_to_array "${FEATURES_CSV}" FEATURE_COLS

if [[ ${#TARGET_COLS[@]} -eq 0 ]]; then
  echo "No targets after parsing TARGETS_CSV (got empty list)." >&2
  exit 1
fi
if [[ ${#FEATURE_COLS[@]} -eq 0 ]]; then
  echo "No features after parsing FEATURES_CSV (got empty list)." >&2
  exit 1
fi

run_py() {
  # shellcheck disable=SC2086
  ${PY} "$@"
}

mkdir -p "${RUN_DIR}"

_stage "(1/3) Phase 1 — CV folds and parallel_jobs.txt (prepare_run.py)"

RUN_PHASE1=1
if [[ "${NO_PREPROCESSING_SKIP}" == "1" ]]; then
  _stage "Phase 1: forcing run (--no-preprocessing-skip)"
elif _phase1_cache_ok "${JOBS_FILE}"; then
  RUN_PHASE1=0
  _stage "Phase 1: skipping — valid cache at JOBS_FILE=${JOBS_FILE}"
else
  _stage "Phase 1: running — no complete cache at JOBS_FILE=${JOBS_FILE}"
fi

if [[ "${RUN_PHASE1}" == "1" ]]; then
  : > "${JOBS_FILE}"
  run_py src/automl/prepare_run.py "${DATASET}" \
    --name-col "${NAME_COL}" \
    --target-cols "${TARGET_COLS[@]}" \
    --feature-cols "${FEATURE_COLS[@]}" \
    --developability-results "${DEVELOPABILITY}" \
    --output-dir "${RUN_DIR}" \
    --n-splits "${N_SPLITS}" \
    --random-state "${RANDOM_STATE}" \
    --features-frac "${FEATURES_FRAC}" \
    --jobs-file "${JOBS_FILE}" \
    2>&1 | tee "${RUN_DIR}/prepare_run.log"
fi

if ! _phase1_cache_ok "${JOBS_FILE}"; then
  echo "After phase 1, jobs file is missing or incomplete: ${JOBS_FILE}" >&2
  exit 1
fi

_stage "Phase 1 complete — $(wc -l < "${JOBS_FILE}") job line(s) in ${JOBS_FILE}"

_stage "(2/3) Phase 2 — fold workers (run_fold_pipeline_config.py via GNU parallel -> ${RESULT_ROOT})"

mkdir -p "${RESULT_ROOT}"
_py_to_array "${PY}" PY_ARR
parallel --jobs "${PARALLEL_JOBS}" --line-buffer --colsep $'\t' \
  "${PY_ARR[@]}" src/automl/run_fold_pipeline_config.py \
    --fold-dir {1} \
    --fold {2} \
    --dataset-stem {3} \
    --pipeline-target-col {4} \
    --selector-name "${SELECTOR_NAME}" \
    --model-to-use "${MODEL_TO_USE}" \
    --eval-models "${EVAL_MODELS}" \
    --result-dir "${RESULT_ROOT}" \
    --random-state "${RANDOM_STATE}" \
    --correlation-min-abs-rho none \
    --quiet \
  :::: "${JOBS_FILE}"

_stage "Phase 2 complete — result JSON under RESULT_ROOT=${RESULT_ROOT}"
_stage "Summary — JOBS_FILE=${JOBS_FILE} ($(wc -l < "${JOBS_FILE}") lines); fold data under RUN_DIR=${RUN_DIR}"

if [[ "${NO_CLEAN_FOLDS}" == "0" ]]; then
  _stage "(3/3) Phase 3 — removing fold *.parquet under RUN_DIR (pass --no-clean-folds to keep)"
  _n_pq=$(find "${RUN_DIR}" -name "*.parquet" -type f 2>/dev/null | wc -l)
  if [[ "${_n_pq}" -gt 0 ]]; then
    find "${RUN_DIR}" -name "*.parquet" -type f -delete 2>/dev/null || true
    _stage "Removed ${_n_pq} fold parquet file(s) from RUN_DIR=${RUN_DIR}"
  else
    _stage "Phase 3: no *.parquet files to remove under RUN_DIR"
  fi
else
  _stage "(3/3) Phase 3 — skipped (--no-clean-folds): keeping fold parquets under RUN_DIR=${RUN_DIR}"
fi

_stage "Done."
