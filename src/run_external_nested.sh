#!/usr/bin/env bash
# Nested CV for structural ProperMAb (ABB2/ABB3/FlashABB variant 1), the
# sequence baseline (ABB2 variant 1, Ginkgo omitted), and TAP.
#
# Ginkgo sequence baseline is handled by src/run_ginkgo_sequence_baseline.sh.
# TAP fold parquets are reconstructed first; ProperMAb/sequence-baseline folds
# are rebuilt into isolated run dirs so they do not share overwritten parquets.
#
# Launch:
#   tmux new -s external-nested
#   ./src/run_external_nested.sh --parallel-jobs 100
#
# Resume:
#   ./src/run_external_nested.sh \
#     --out-dir runs/external_nested_<stamp> --resume --parallel-jobs 100

set -euo pipefail

log_stage() {
  echo >&2 "[external-nested] $*"
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 [--method all|propermab|propermab_abb2,propermab_abb3,propermab_flashabb,propermab_sequence_baseline,tap]
     [--stage all|folds|nested]
     [--parallel-jobs N] [--out-dir DIR] [--resume] [--py CMD] [--dry-run]

Stages:
  folds    Reconstruct isolated CV fold parquets (and TAP pseudo-AutoML JSONs)
  nested   Inner AutoML + floating SFS + outer finish + report
  all      folds + nested

Nested pairs match kitAb nested CV (all backend_1 dataset/target pairs) except
AB21, PDGF38, and Jain 2024. Sequence baseline also omits Ginkgo (that nested
run is src/run_ginkgo_sequence_baseline.sh). TAP and structural ProperMAb still
include Ginkgo.

Does not launch AutoML into propermab_*/ or tap/automl/. Nested outputs go
under OUT_DIR.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/kitab.local.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/kitab.local.env"
fi

KITAB_ENV="${KITAB_ENV:-kitab}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
DEFAULT_PY="conda run --no-capture-output -n ${KITAB_ENV} env PYTHONPATH=${PYTHONPATH} python -P"

METHOD_SPEC="all"
STAGE="all"
PARALLEL_JOBS=""
OUT_DIR=""
PY=""
RESUME="0"
DRY_RUN="0"
NESTED_EXCLUDE_STEMS="ab21,pdgf38,jain2024assessment_folded_08_4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --method) METHOD_SPEC="${2:?}"; shift 2 ;;
    --stage) STAGE="${2:?}"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="${2:?}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --py) PY="${2:?}"; shift 2 ;;
    --nested-exclude-stems) NESTED_EXCLUDE_STEMS="${2:?}"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
  esac
done

PY="${PY:-$DEFAULT_PY}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc 2>/dev/null || echo 4)}"
if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer, got: ${PARALLEL_JOBS}" >&2
  exit 1
fi
case "${STAGE}" in
  all|folds|nested) ;;
  *) echo "Unknown --stage ${STAGE}" >&2; usage; exit 1 ;;
esac

ALL_METHODS=(
  propermab_abb2
  propermab_abb3
  propermab_flashabb
  propermab_sequence_baseline
  tap
)
case "${METHOD_SPEC}" in
  all) METHODS=("${ALL_METHODS[@]}") ;;
  propermab) METHODS=(propermab_abb2 propermab_abb3 propermab_flashabb) ;;
  *)
    IFS=',' read -r -a METHODS <<< "${METHOD_SPEC}"
    for method in "${METHODS[@]}"; do
      case " ${ALL_METHODS[*]} " in
        *" ${method} "*) ;;
        *) echo "Unknown method: ${method}" >&2; exit 1 ;;
      esac
    done
    ;;
esac

if [[ -z "${OUT_DIR}" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  OUT_DIR="${REPO_ROOT}/runs/external_nested_${stamp}"
elif [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
exec > >(tee -a "${OUT_DIR}/external_nested.log") 2>&1

log_stage "out-dir=${OUT_DIR} methods=${METHODS[*]} stage=${STAGE} jobs=${PARALLEL_JOBS} resume=${RESUME}"

run_py() {
  # shellcheck disable=SC2086
  ${PY} "$@"
}

py_to_array() {
  local command="$1"
  local -n output=$2
  output=()
  mapfile -t output < <(python3 -c 'import shlex,sys
for part in shlex.split(sys.argv[1]):
    print(part)' "${command}")
}

want_stage() {
  [[ "${STAGE}" == "all" || "${STAGE}" == "$1" ]]
}

method_backend() {
  case "$1" in
    propermab_abb3) echo abb3 ;;
    propermab_flashabb) echo flashabb ;;
    *) echo abb2 ;;
  esac
}

method_yaml_suffix() {
  case "$1" in
    tap) echo "" ;;
    *) echo "_propermab" ;;
  esac
}

method_yaml_mode() {
  case "$1" in
    tap) echo stem ;;
    *) echo backend_1 ;;
  esac
}

method_flat_variant() {
  case "$1" in
    propermab_abb2) echo abb2_1_propermab ;;
    propermab_abb3) echo abb3_1_propermab ;;
    propermab_flashabb) echo flashabb_1_propermab ;;
    propermab_sequence_baseline) echo abb2_1_propermab ;;
    tap) echo "" ;;
  esac
}

method_automl_root() {
  case "$1" in
    tap) echo "${OUT_DIR}/tap_pseudo_automl" ;;
    *) echo "${REPO_ROOT}/$1/automl" ;;
  esac
}

method_exclude_stems() {
  local extra="$1"
  if [[ "${extra}" == "propermab_sequence_baseline" ]]; then
    if [[ -n "${NESTED_EXCLUDE_STEMS}" ]]; then
      echo "${NESTED_EXCLUDE_STEMS},ginkgo_ig_folded"
    else
      echo "ginkgo_ig_folded"
    fi
  else
    echo "${NESTED_EXCLUDE_STEMS}"
  fi
}

materialize_tap() {
  local tap_root="${OUT_DIR}/tap_pseudo_automl"
  mkdir -p "${tap_root}" "${OUT_DIR}/tap_materialize_oof"
  if [[ "${RESUME}" == "1" && -s "${OUT_DIR}/tap_materialize_jobs.tsv" ]]; then
    local n_tap
    n_tap="$(find "${tap_root}" -name '*.json' | wc -l | tr -d ' ')"
    if [[ "${n_tap}" != "0" ]]; then
      log_stage "tap: resume skip materialize (${n_tap} JSON(s) already present)"
      return 0
    fi
  fi
  run_py "${REPO_ROOT}/src/analysis/materialize_aggregated_fold_results.py" \
    --aggregated-dir "${REPO_ROOT}/tap/automl" \
    --master-tsv "${REPO_ROOT}/tap/automl/parallel_jobs_master.tsv" \
    --pseudo-automl-root "${tap_root}" \
    --oof-root "${OUT_DIR}/tap_materialize_oof" \
    --jobs-file "${OUT_DIR}/tap_materialize_jobs.tsv" \
    --method-name tap
}

prepare_folds_for() {
  local method="$1"
  local automl_root exclude_stems map_out
  automl_root="$(method_automl_root "${method}")"
  exclude_stems="$(method_exclude_stems "${method}")"
  map_out="${OUT_DIR}/fold_maps/${method}.json"
  mkdir -p "${OUT_DIR}/fold_maps"
  if [[ "${method}" == "tap" ]]; then
    materialize_tap
  fi
  local args=(
    "${REPO_ROOT}/src/analysis/prepare_nested_folds.py"
    --repo-root "${REPO_ROOT}"
    --method "${method}"
    --automl-root "${automl_root}"
    --map-out "${map_out}"
    --pairs-mode all_backend_1
    --exclude-stems "${exclude_stems}"
  )
  if [[ "${RESUME}" == "1" ]]; then
    args+=(--resume)
  fi
  run_py "${args[@]}"
}

run_nested_for() {
  local method="$1"
  local backend yaml_suffix yaml_mode flat_variant automl_root exclude_stems
  local nested_root nested_jobs nested_floating nested_report map_out file_prefix
  backend="$(method_backend "${method}")"
  yaml_suffix="$(method_yaml_suffix "${method}")"
  yaml_mode="$(method_yaml_mode "${method}")"
  flat_variant="$(method_flat_variant "${method}")"
  automl_root="$(method_automl_root "${method}")"
  exclude_stems="$(method_exclude_stems "${method}")"
  file_prefix="nested_${method}"
  nested_root="${OUT_DIR}/${file_prefix}"
  nested_jobs="${OUT_DIR}/${file_prefix}_jobs.tsv"
  nested_floating="${OUT_DIR}/${file_prefix}_floating_jobs.tsv"
  nested_report="${OUT_DIR}/${file_prefix}_report.csv"
  map_out="${OUT_DIR}/fold_maps/${method}.json"

  [[ -d "${automl_root}" ]] || {
    echo "Missing AutoML root for ${method}: ${automl_root}" >&2
    echo "Run --stage folds first." >&2
    exit 1
  }
  [[ -f "${map_out}" ]] || {
    echo "Missing fold-dir map ${map_out}; run --stage folds first." >&2
    exit 1
  }

  mkdir -p "${nested_root}"
  local discover_args=(
    "${REPO_ROOT}/src/analysis/nested_cv.py" discover
    --repo-root "${REPO_ROOT}"
    --out-dir "${nested_root}"
    --jobs-file "${nested_jobs}"
    --floating-jobs-file "${nested_floating}"
    --backend "${backend}"
    --automl-root "${automl_root}"
    --yaml-key-suffix "${yaml_suffix}"
    --yaml-key-mode "${yaml_mode}"
    --fold-dir-map "${map_out}"
    --pairs-mode all_backend_1
    --exclude-stems "${exclude_stems}"
    --prepare-inner
  )
  run_py "${discover_args[@]}"
  if [[ ! -s "${nested_jobs}" ]]; then
    echo "Nested jobs TSV is empty for ${method}. Check folds and ${automl_root}." >&2
    exit 1
  fi

  local nested_run_jobs="${nested_jobs}"
  if [[ "${RESUME}" == "1" ]]; then
    nested_run_jobs="${OUT_DIR}/${file_prefix}_jobs_pending.tsv"
    run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
      --jobs-file "${nested_jobs}" \
      --out "${nested_run_jobs}" \
      --require-oof
  fi
  py_to_array "${PY}" PY_ARR
  local n_nj
  n_nj="$(tail -n +2 "${nested_run_jobs}" | wc -l | tr -d ' ')"
  log_stage "${method} nested inner jobs: ${n_nj}"
  if [[ "${n_nj}" != "0" ]]; then
    if ! python3 - "${nested_run_jobs}" <<'PY'
import json, sys
from pathlib import Path
fields = Path(sys.argv[1]).read_text().splitlines()[1].split("\t")
for idx in (15, 16):
    json.loads(fields[idx])
PY
    then
      echo "Nested jobs TSV JSON is not GNU-parallel-safe: ${nested_run_jobs}" >&2
      exit 1
    fi
    tail -n +2 "${nested_run_jobs}" | parallel --will-cite --jobs "${PARALLEL_JOBS}" --line-buffer --eta --colsep $'\t' \
      "${PY_ARR[@]}" "${REPO_ROOT}/src/automl/run_fold_pipeline_config.py" \
        --fold-dir {5} \
        --fold {7} \
        --dataset-stem {1} \
        --pipeline-target-col {2} \
        --dataset-yaml-key {3} \
        --selector-name {9} \
        --model-to-use {10} \
        --eval-models {13} \
        --eval-features-frac {11} \
        --output-json {18} \
        --random-state {14} \
        --correlation-min-abs-rho {15} \
        --selector-hyperparameters {16} \
        --eval-hyperparameters {17} \
        --pipeline-track-name {12} \
        --quiet \
      || {
        echo "Nested inner parallel failed for ${method}. Resume from ${nested_jobs}" >&2
        echo "  $0 --out-dir ${OUT_DIR} --resume --stage nested --method ${method}" >&2
        exit 1
      }
  fi

  local nested_regular_missing="${OUT_DIR}/${file_prefix}_jobs_missing.tsv"
  run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
    --jobs-file "${nested_jobs}" \
    --out "${nested_regular_missing}" \
    --require-oof
  local n_regular_missing
  n_regular_missing="$(tail -n +2 "${nested_regular_missing}" | wc -l | tr -d ' ')"
  if [[ "${n_regular_missing}" != "0" ]]; then
    echo "Nested regular grid incomplete for ${method}: ${n_regular_missing} job(s)." >&2
    echo "Inspect ${nested_regular_missing}; do not start floating SFS." >&2
    exit 1
  fi

  local nested_floating_run="${nested_floating}"
  local n_ffs="0"
  if [[ -s "${nested_floating}" ]]; then
    if [[ "${RESUME}" == "1" ]]; then
      nested_floating_run="${OUT_DIR}/${file_prefix}_floating_jobs_pending.tsv"
      run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
        --jobs-file "${nested_floating}" \
        --out "${nested_floating_run}" \
        --require-oof
    fi
    n_ffs="$(tail -n +2 "${nested_floating_run}" | wc -l | tr -d ' ')"
  fi
  log_stage "${method} nested post-grid floating-SFS jobs: ${n_ffs}"
  if [[ "${n_ffs}" != "0" ]]; then
    if ! python3 - "${nested_floating_run}" <<'PY'
import json, sys
from pathlib import Path
fields = Path(sys.argv[1]).read_text().splitlines()[1].split("\t")
json.loads(fields[16])
PY
    then
      echo "Nested floating jobs TSV JSON is not GNU-parallel-safe: ${nested_floating_run}" >&2
      exit 1
    fi
    tail -n +2 "${nested_floating_run}" | parallel --will-cite --jobs "${PARALLEL_JOBS}" --line-buffer --eta --colsep $'\t' \
      "${PY_ARR[@]}" "${REPO_ROOT}/src/analysis/nested_cv.py" floating-sfs-job \
        --inner-fold-dir {5} \
        --inner-k {7} \
        --dataset-stem {1} \
        --target {2} \
        --yaml-key {3} \
        --selection-model {10} \
        --max-feature-fraction {11} \
        --track-name {12} \
        --eval-models {13} \
        --random-state {14} \
        --eval-hyperparameters {17} \
        --output-json {18} \
      || {
        echo "Nested floating-SFS parallel failed for ${method}. Resume from ${nested_floating}" >&2
        echo "  $0 --out-dir ${OUT_DIR} --resume --stage nested --method ${method}" >&2
        exit 1
      }
  fi

  local nested_floating_missing="${OUT_DIR}/${file_prefix}_floating_jobs_missing.tsv"
  local n_ffs_missing="0"
  if [[ -s "${nested_floating}" ]]; then
    run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
      --jobs-file "${nested_floating}" \
      --out "${nested_floating_missing}" \
      --require-oof
    n_ffs_missing="$(tail -n +2 "${nested_floating_missing}" | wc -l | tr -d ' ')"
  fi
  if [[ "${n_ffs_missing}" != "0" ]]; then
    echo "Nested floating-SFS incomplete for ${method}: ${n_ffs_missing} job(s)." >&2
    echo "Inspect ${nested_floating_missing}; do not finish outer folds." >&2
    exit 1
  fi

  export NESTED_ROOT="${nested_root}" NESTED_JOBS="${nested_jobs}" OUT_DIR FILE_PREFIX="${file_prefix}"
  python3 - <<'PY'
import csv
import os
import sys
from pathlib import Path

jobs = Path(os.environ["NESTED_JOBS"])
nested_root = Path(os.environ["NESTED_ROOT"])
seen = set()
rows = []
with jobs.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        key = (row["yaml_key"], row["target"], row["orig_fold_dir"])
        if key in seen:
            continue
        seen.add(key)
        pair = nested_root / row["yaml_key"] / row["target"]
        rows.append((str(pair), row["orig_fold_dir"]))
print(f"finish {len(rows)} pair(s)", file=sys.stderr)
Path(os.environ["OUT_DIR"], f"{os.environ['FILE_PREFIX']}_finish_pairs.tsv").write_text(
    "pair_root\torig_fold_dir\n" + "\n".join(f"{a}\t{b}" for a, b in rows) + ("\n" if rows else "")
)
PY
  if [[ -s "${OUT_DIR}/${file_prefix}_finish_pairs.tsv" ]]; then
    while IFS=$'\t' read -r pair orig; do
      if [[ "${RESUME}" == "1" && -f "${pair}/nested_summary.json" ]]; then
        log_stage "${method} nested finish skip ${pair}"
        continue
      fi
      log_stage "${method} nested finish ${pair}"
      run_py "${REPO_ROOT}/src/analysis/nested_cv.py" finish-pair \
        --pair-root "${pair}" --orig-fold-dir "${orig}"
    done < <(tail -n +2 "${OUT_DIR}/${file_prefix}_finish_pairs.tsv")
  fi

  local report_args=(
    "${REPO_ROOT}/src/analysis/nested_cv.py" report
    --nested-root "${nested_root}"
    --automl-root "${automl_root}"
    --dest "${nested_report}"
    --flat-results "${REPO_ROOT}/${method}/analysis_results/results.csv"
    --backend "${backend}"
    --yaml-key-suffix "${yaml_suffix}"
    --yaml-key-mode "${yaml_mode}"
    --flat-variant "${flat_variant}"
    --pairs-mode all_backend_1
    --exclude-stems "${exclude_stems}"
  )
  run_py "${report_args[@]}"
  log_stage "${method} nested report: ${nested_report}"
}

if want_stage folds; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    log_stage "dry-run: skip folds"
  else
    for method in "${METHODS[@]}"; do
      log_stage "folds: ${method}"
      prepare_folds_for "${method}"
    done
  fi
fi

if want_stage nested; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    log_stage "dry-run: skip nested"
  else
    for method in "${METHODS[@]}"; do
      log_stage "nested: ${method}"
      run_nested_for "${method}"
    done
  fi
fi

log_stage "complete"
for method in "${METHODS[@]}"; do
  log_stage "  ${method}: ${OUT_DIR}/nested_${method}_report.csv"
done
log_stage "resume: $0 --out-dir ${OUT_DIR} --resume --stage ${STAGE}"
