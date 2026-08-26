#!/usr/bin/env bash
# Pooled-OOF reanalysis + nested-CV bias check. Designed to run under tmux.
#
# Example:
#   tmux new -s reanalysis
#   ./src/run_reanalysis.sh --backend all --stage all --parallel-jobs 32
#   # if interrupted:
#   ./src/run_reanalysis.sh --out-dir runs/reanalysis_<stamp> --resume --stage oof

set -euo pipefail

_stage() {
  echo >&2 "[reanalysis] $*"
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 [--backend abb2|abb3|flashabb|all] [--stage discover|oof|validate|aggregate|nested|all]
     [--parallel-jobs N] [--out-dir DIR] [--py CMD] [--resume] [--dry-run]
     [--validate-tolerance X] [--max-mismatch-rate R] [--reference-json PATH]
     [--nested-pairs-mode default|all_backend_1|all_abb2_1] [--nested-exclude-stems CSV]
     [--nested-flat-results PATH]

Suggested tmux:
  tmux new -s reanalysis
  $0 --backend all --stage all

Stages:
  discover    Write OOF jobs TSV (JSON -> parquet) from paper fold JSONs
  oof         Recompute per-sample predictions (GNU parallel)
  validate    Compare recomputed vs stored fold Spearman. Exact agreement is
              reported at 1e-9; the gate fails only on differences beyond
              --validate-tolerance (default 0.2), which absorbs randomforest
              thread-order noise and linear OLS near-tie reordering on small folds
  aggregate   Pooled OOF metrics + analysis ranking for each backend
  nested      Nested-CV check for the selected backend
  all         discover + oof + validate + aggregate + nested
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
# conda run does not forward PYTHONPATH; `env` re-injects it into the child.
DEFAULT_PY="conda run --no-capture-output -n ${KITAB_ENV} env PYTHONPATH=${PYTHONPATH} python"

BACKEND="all"
STAGE="all"
PARALLEL_JOBS=""
OUT_DIR=""
PY=""
RESUME="0"
DRY_RUN="0"
VALIDATE_TOLERANCE="0.2"
MAX_MISMATCH_RATE="0"
# Canonical feature_* list for feature_usage.csv. When unset, each backend falls
# back to a developability JSON from its own final_set_of_features descriptors.
# Pooled OOF metrics and ranking never read this.
REFERENCE_JSON="${REFERENCE_JSON:-}"
NESTED_PAIRS_MODE="default"
NESTED_EXCLUDE_STEMS=""
NESTED_FLAT_RESULTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backend)
      BACKEND="${2:?}"
      shift 2
      ;;
    --stage)
      STAGE="${2:?}"
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="${2:?}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?}"
      shift 2
      ;;
    --py)
      PY="${2:?}"
      shift 2
      ;;
    --validate-tolerance)
      VALIDATE_TOLERANCE="${2:?}"
      shift 2
      ;;
    --max-mismatch-rate)
      MAX_MISMATCH_RATE="${2:?}"
      shift 2
      ;;
    --reference-json)
      REFERENCE_JSON="${2:?}"
      shift 2
      ;;
    --nested-pairs-mode)
      NESTED_PAIRS_MODE="${2:?}"
      shift 2
      ;;
    --nested-exclude-stems)
      NESTED_EXCLUDE_STEMS="${2:?}"
      shift 2
      ;;
    --nested-flat-results)
      NESTED_FLAT_RESULTS="${2:?}"
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
    *)
      echo "Unknown option: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

PY="${PY:-$DEFAULT_PY}"
if [[ -z "${PARALLEL_JOBS}" ]]; then
  PARALLEL_JOBS="$(nproc 2>/dev/null || echo 4)"
fi
if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --parallel-jobs must be a positive integer, got: ${PARALLEL_JOBS}" >&2
  exit 1
fi
if [[ -n "${REFERENCE_JSON}" && ! -f "${REFERENCE_JSON}" ]]; then
  echo "Error: --reference-json not found: ${REFERENCE_JSON}" >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  OUT_DIR="${REPO_ROOT}/runs/reanalysis_${stamp}"
fi
OUT_DIR="$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"
LOG="${OUT_DIR}/reanalysis.log"

exec > >(tee -a "${LOG}") 2>&1

_stage "out-dir=${OUT_DIR} backend=${BACKEND} stage=${STAGE} parallel_jobs=${PARALLEL_JOBS} resume=${RESUME}"

_py_to_array() {
  local _py_cmd="$1"
  local -n _out=$2
  _out=()
  mapfile -t _out < <(python3 -c 'import shlex,sys
for part in shlex.split(sys.argv[1]):
    print(part)' "$_py_cmd")
}

run_py() {
  # shellcheck disable=SC2086
  ${PY} "$@"
}

_want() {
  local s="$1"
  [[ "${STAGE}" == "all" || "${STAGE}" == "${s}" ]]
}

JOBS_FILE="${OUT_DIR}/oof_jobs.tsv"

if _want discover || _want oof; then
  if [[ "${STAGE}" == "oof" && -s "${JOBS_FILE}" && "${RESUME}" == "1" ]]; then
    _stage "discover: reuse existing ${JOBS_FILE}"
  else
    _stage "discover: writing ${JOBS_FILE}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      _stage "dry-run: skip discover"
    else
      run_py "${REPO_ROOT}/src/analysis/oof_predictions.py" discover \
        --repo-root "${REPO_ROOT}" \
        --backend "${BACKEND}" \
        --out-dir "${OUT_DIR}/oof" \
        --jobs-file "${JOBS_FILE}"
    fi
  fi
fi

if _want oof; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    _stage "dry-run: skip oof"
  else
    if [[ ! -s "${JOBS_FILE}" ]]; then
      echo "Missing jobs file ${JOBS_FILE}; run --stage discover first." >&2
      echo "Resume with: $0 --out-dir ${OUT_DIR} --resume --stage oof" >&2
      exit 1
    fi
    n_jobs="$(tail -n +2 "${JOBS_FILE}" | wc -l | tr -d ' ')"
    _stage "oof: ${n_jobs} job(s) via GNU parallel"
    _py_to_array "${PY}" PY_ARR
    resume_flag=()
    if [[ "${RESUME}" == "1" ]]; then
      resume_flag=(--resume)
    fi
    tail -n +2 "${JOBS_FILE}" | parallel --will-cite --jobs "${PARALLEL_JOBS}" --line-buffer --eta --colsep $'\t' \
      "${PY_ARR[@]}" "${REPO_ROOT}/src/analysis/oof_predictions.py" run-one \
        --json-path {2} \
        --oof-path {3} \
        --n-jobs 1 \
        "${resume_flag[@]}" \
      || {
        echo "OOF parallel failed. Resume with:" >&2
        echo "  $0 --out-dir ${OUT_DIR} --resume --stage oof --backend ${BACKEND}" >&2
        exit 1
      }
  fi
fi

if _want validate; then
  _stage "validate: stored vs recomputed Spearman (exact at 1e-9, gate at ${VALIDATE_TOLERANCE})"
  if [[ "${DRY_RUN}" == "1" ]]; then
    _stage "dry-run: skip validate"
  else
    run_py "${REPO_ROOT}/src/analysis/oof_predictions.py" validate \
      --oof-dir "${OUT_DIR}/oof" \
      --max-mismatch-rate "${MAX_MISMATCH_RATE}" \
      --tolerance "${VALIDATE_TOLERANCE}" \
      --summary-out "${OUT_DIR}/validate_summary.json" \
      || {
        echo "Validation failed. Do not aggregate. Inspect ${OUT_DIR}/validate_summary.json" >&2
        exit 1
      }
  fi
fi

# Canonical feature_* names come from a developability JSON produced by the same
# descriptor pipeline as the runs being analysed. All three backends emit the same
# 59 keys, so any per-antibody JSON from that backend's final feature set works.
_default_reference_json() {
  local folder="$1"
  shopt -s nullglob
  local candidates=( "${REPO_ROOT}/${folder}"/descriptors/*/results/*.json )
  shopt -u nullglob
  for c in "${candidates[@]}"; do
    if [[ -f "${c}" ]]; then
      printf '%s\n' "${c}"
      return 0
    fi
  done
  return 0
}

_aggregate_one_backend() {
  local be="$1"
  local folder
  case "${be}" in
    abb2) folder="our_abb2_final_set_of_features" ;;
    abb3) folder="our_abb3_final_set_of_features" ;;
    flashabb) folder="our_flashabb_final_set_of_features" ;;
    *) echo "Unknown backend ${be}" >&2; return 1 ;;
  esac
  local batch="${REPO_ROOT}/${folder}/automl"
  local analysis_out="${OUT_DIR}/analysis_${be}"
  mkdir -p "${analysis_out}"
  _stage "aggregate ${be}: ${batch}"
  local man="${batch}/batch_manifest.json"
  if [[ -f "${man}" ]]; then
    run_py "${REPO_ROOT}/src/automl/aggregate_batch_results.py" \
      --manifest "${man}" \
      --batch-root "${batch}" \
      --output-dir "${analysis_out}" \
      --oof-dir "${OUT_DIR}/oof/${be}" \
      --no-plots \
      || run_py "${REPO_ROOT}/src/automl/aggregate_batch_results.py" \
        --batch-root "${batch}" \
        --output-dir "${analysis_out}" \
        --oof-dir "${OUT_DIR}/oof/${be}" \
        --no-plots
  else
    run_py "${REPO_ROOT}/src/automl/aggregate_batch_results.py" \
      --batch-root "${batch}" \
      --output-dir "${analysis_out}" \
      --oof-dir "${OUT_DIR}/oof/${be}" \
      --no-plots
  fi
  shopt -s nullglob
  local agg_csvs=( "${analysis_out}"/aggregated_*.csv )
  shopt -u nullglob
  if [[ ${#agg_csvs[@]} -gt 0 ]]; then
    local -a analyze_args=(
      "${agg_csvs[@]}"
      --out-dir "${analysis_out}"
      --summary-name best_metrics_summary.csv
      --results-name results.csv
      --no-plots
    )
    local ref="${REFERENCE_JSON}"
    if [[ -z "${ref}" ]]; then
      ref="$(_default_reference_json "${folder}")"
    fi
    if [[ -n "${ref}" ]]; then
      _stage "aggregate ${be}: feature-usage reference ${ref}"
      analyze_args+=( --reference-json "${ref}" )
    else
      _stage "aggregate ${be}: no reference JSON found, skipping feature usage"
      analyze_args+=( --skip-feature-usage )
    fi
    run_py "${REPO_ROOT}/src/analysis/analyze_results.py" "${analyze_args[@]}"
  fi
  if [[ -d "${OUT_DIR}/oof/${be}" ]]; then
    run_py "${REPO_ROOT}/src/analysis/random_spearman_baseline.py" \
      --selection-matched \
      --oof-dir "${OUT_DIR}/oof/${be}" \
      --out "${analysis_out}/results_random_baseline_selection_matched.csv" \
      --seed 42 \
      --repo-root "${REPO_ROOT}" || true
  fi
}

if _want aggregate; then
  backends=()
  if [[ "${BACKEND}" == "all" ]]; then
    backends=(abb2 abb3 flashabb)
  else
    backends=("${BACKEND}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    _stage "dry-run: skip aggregate"
  else
    for be in "${backends[@]}"; do
      _aggregate_one_backend "${be}"
    done
  fi
fi

if _want nested; then
  NESTED_BACKEND="${BACKEND}"
  if [[ "${NESTED_BACKEND}" == "all" ]]; then
    # Preserve the historical `--backend all --stage all` behavior. Run other
    # nested backends explicitly so each long run has an unambiguous resume key.
    NESTED_BACKEND="abb2"
    _stage "nested: --backend all retains legacy ABB2-only behavior; run ABB3/FlashABB explicitly"
  fi
  case "${NESTED_BACKEND}" in
    abb2) NESTED_AUTOML_FOLDER="our_abb2_final_set_of_features" ;;
    abb3) NESTED_AUTOML_FOLDER="our_abb3_final_set_of_features" ;;
    flashabb) NESTED_AUTOML_FOLDER="our_flashabb_final_set_of_features" ;;
    *) echo "Unknown nested backend: ${NESTED_BACKEND}" >&2; exit 1 ;;
  esac
  _stage "nested: backend=${NESTED_BACKEND} pairs_mode=${NESTED_PAIRS_MODE} exclude_stems=${NESTED_EXCLUDE_STEMS:-none}"
  NESTED_ROOT="${OUT_DIR}/nested_${NESTED_BACKEND}"
  if [[ "${NESTED_BACKEND}" == "abb2" ]]; then
    # Keep existing ABB2 resume paths compatible with the completed run.
    NESTED_JOBS="${OUT_DIR}/nested_jobs.tsv"
    NESTED_FLOATING_JOBS="${OUT_DIR}/nested_floating_jobs.tsv"
    NESTED_REPORT="${OUT_DIR}/nested_report.csv"
    NESTED_FILE_PREFIX="nested"
  else
    NESTED_JOBS="${OUT_DIR}/nested_${NESTED_BACKEND}_jobs.tsv"
    NESTED_FLOATING_JOBS="${OUT_DIR}/nested_${NESTED_BACKEND}_floating_jobs.tsv"
    NESTED_REPORT="${OUT_DIR}/nested_${NESTED_BACKEND}_report.csv"
    NESTED_FILE_PREFIX="nested_${NESTED_BACKEND}"
  fi
  mkdir -p "${NESTED_ROOT}"
  export REPO_ROOT OUT_DIR
  if [[ "${DRY_RUN}" == "1" ]]; then
    _stage "dry-run: skip nested"
  else
    run_py "${REPO_ROOT}/src/analysis/nested_cv.py" discover \
      --repo-root "${REPO_ROOT}" \
      --out-dir "${NESTED_ROOT}" \
      --jobs-file "${NESTED_JOBS}" \
      --floating-jobs-file "${NESTED_FLOATING_JOBS}" \
      --backend "${NESTED_BACKEND}" \
      --pairs-mode "${NESTED_PAIRS_MODE}" \
      --exclude-stems "${NESTED_EXCLUDE_STEMS}" \
      --prepare-inner
    if [[ -s "${NESTED_JOBS}" ]]; then
      NESTED_RUN_JOBS="${NESTED_JOBS}"
      if [[ "${RESUME}" == "1" ]]; then
        NESTED_RUN_JOBS="${OUT_DIR}/${NESTED_FILE_PREFIX}_jobs_pending.tsv"
        run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
          --jobs-file "${NESTED_JOBS}" \
          --out "${NESTED_RUN_JOBS}" \
          --require-oof
      fi
      _py_to_array "${PY}" PY_ARR
      n_nj="$(tail -n +2 "${NESTED_RUN_JOBS}" | wc -l | tr -d ' ')"
      _stage "nested inner jobs: ${n_nj}"
      if [[ "${n_nj}" != "0" ]]; then
        if ! python3 - "${NESTED_RUN_JOBS}" <<'PY'
import json, sys
from pathlib import Path
fields = Path(sys.argv[1]).read_text().splitlines()[1].split("\t")
for idx in (15, 16):
    json.loads(fields[idx])
PY
        then
          echo "Nested jobs TSV JSON is not GNU-parallel-safe: ${NESTED_RUN_JOBS}" >&2
          echo "Fields 16/17 must parse with json.loads after a raw tab split (no CSV quoting)." >&2
          exit 1
        fi
        tail -n +2 "${NESTED_RUN_JOBS}" | parallel --will-cite --jobs "${PARALLEL_JOBS}" --line-buffer --eta --colsep $'\t' \
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
            echo "Nested inner parallel failed. Resume inner jobs from ${NESTED_JOBS}" >&2
            echo "  $0 --out-dir ${OUT_DIR} --resume --stage nested --backend ${NESTED_BACKEND}" >&2
            exit 1
          }
      fi
      NESTED_REGULAR_MISSING="${OUT_DIR}/${NESTED_FILE_PREFIX}_jobs_missing.tsv"
      run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
        --jobs-file "${NESTED_JOBS}" \
        --out "${NESTED_REGULAR_MISSING}" \
        --require-oof
      n_regular_missing="$(tail -n +2 "${NESTED_REGULAR_MISSING}" | wc -l | tr -d ' ')"
      if [[ "${n_regular_missing}" != "0" ]]; then
        echo "Nested regular grid incomplete: ${n_regular_missing} job(s) lack semantic OOF results." >&2
        echo "Inspect ${NESTED_REGULAR_MISSING}; do not start floating SFS." >&2
        exit 1
      fi
      NESTED_FLOATING_RUN_JOBS="${NESTED_FLOATING_JOBS}"
      if [[ "${RESUME}" == "1" ]]; then
        NESTED_FLOATING_RUN_JOBS="${OUT_DIR}/${NESTED_FILE_PREFIX}_floating_jobs_pending.tsv"
        run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
          --jobs-file "${NESTED_FLOATING_JOBS}" \
          --out "${NESTED_FLOATING_RUN_JOBS}" \
          --require-oof
      fi
      n_ffs="$(tail -n +2 "${NESTED_FLOATING_RUN_JOBS}" | wc -l | tr -d ' ')"
      _stage "nested post-grid floating-SFS jobs: ${n_ffs}"
      if [[ "${n_ffs}" != "0" ]]; then
        if ! python3 - "${NESTED_FLOATING_RUN_JOBS}" <<'PY'
import json, sys
from pathlib import Path
fields = Path(sys.argv[1]).read_text().splitlines()[1].split("\t")
json.loads(fields[16])
PY
        then
          echo "Nested floating jobs TSV JSON is not GNU-parallel-safe: ${NESTED_FLOATING_RUN_JOBS}" >&2
          exit 1
        fi
        tail -n +2 "${NESTED_FLOATING_RUN_JOBS}" | parallel --will-cite --jobs "${PARALLEL_JOBS}" --line-buffer --eta --colsep $'\t' \
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
            echo "Nested floating-SFS parallel failed. Resume from ${NESTED_FLOATING_JOBS}" >&2
            echo "  $0 --out-dir ${OUT_DIR} --resume --stage nested --backend ${NESTED_BACKEND}" >&2
            exit 1
          }
      fi
      NESTED_FLOATING_MISSING="${OUT_DIR}/${NESTED_FILE_PREFIX}_floating_jobs_missing.tsv"
      run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
        --jobs-file "${NESTED_FLOATING_JOBS}" \
        --out "${NESTED_FLOATING_MISSING}" \
        --require-oof
      n_ffs_missing="$(tail -n +2 "${NESTED_FLOATING_MISSING}" | wc -l | tr -d ' ')"
      if [[ "${n_ffs_missing}" != "0" ]]; then
        echo "Nested floating-SFS incomplete: ${n_ffs_missing} job(s) lack OOF sidecars." >&2
        echo "Inspect ${NESTED_FLOATING_MISSING}; do not finish outer folds." >&2
        exit 1
      fi
      export NESTED_ROOT NESTED_JOBS NESTED_FILE_PREFIX
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
Path(os.environ["OUT_DIR"], f"{os.environ['NESTED_FILE_PREFIX']}_finish_pairs.tsv").write_text(
    "pair_root\torig_fold_dir\n" + "\n".join(f"{a}\t{b}" for a, b in rows) + ("\n" if rows else "")
)
PY
      if [[ -s "${OUT_DIR}/${NESTED_FILE_PREFIX}_finish_pairs.tsv" ]]; then
        tail -n +2 "${OUT_DIR}/${NESTED_FILE_PREFIX}_finish_pairs.tsv" | while IFS=$'\t' read -r pair orig; do
          if [[ "${RESUME}" == "1" && -f "${pair}/nested_summary.json" ]]; then
            _stage "nested finish skip ${pair}"
            continue
          fi
          _stage "nested finish ${pair}"
          run_py "${REPO_ROOT}/src/analysis/nested_cv.py" finish-pair --pair-root "${pair}" --orig-fold-dir "${orig}"
        done
      fi
      run_py "${REPO_ROOT}/src/analysis/nested_cv.py" report \
        --nested-root "${NESTED_ROOT}" \
        --automl-root "${REPO_ROOT}/${NESTED_AUTOML_FOLDER}/automl" \
        --dest "${NESTED_REPORT}" \
        --flat-results "${NESTED_FLAT_RESULTS:-${OUT_DIR}/analysis_${NESTED_BACKEND}/results.csv}" \
        --backend "${NESTED_BACKEND}" \
        --pairs-mode "${NESTED_PAIRS_MODE}" \
        --exclude-stems "${NESTED_EXCLUDE_STEMS}"
    else
      _stage "nested: no jobs discovered (check ${NESTED_BACKEND} paper run paths)"
    fi
  fi
fi

_stage "Done. Log: ${LOG}"
_stage "Resume: $0 --out-dir ${OUT_DIR} --resume --stage ${STAGE} --backend ${BACKEND}"
