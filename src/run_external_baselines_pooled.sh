#!/usr/bin/env bash
# Recompute pooled OOF metrics for PROPERMAB, its sequence baseline, and TAP.
# Ginkgo sequence-baseline OOF is excluded here; that dataset is rebuilt by
# src/run_ginkgo_sequence_baseline.sh.
#
# Typical use:
#   tmux new -s external-pooled
#   ./src/run_external_baselines_pooled.sh --parallel-jobs 100
#
# Resume after interruption (skips jobs whose OOF parquet already exists):
#   ./src/run_external_baselines_pooled.sh \
#     --out-dir runs/external_pooled_<stamp> --resume --parallel-jobs 100

set -euo pipefail

log_stage() {
  echo >&2 "[external-pooled] $*"
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 [--method all|propermab|propermab_abb2,propermab_abb3,propermab_flashabb,propermab_sequence_baseline,tap]
     [--stage all|discover|oof|validate|aggregate|publish]
     [--parallel-jobs N] [--out-dir DIR] [--resume] [--py CMD] [--dry-run]
     [--validate-tolerance X] [--max-mismatch-rate R]

Default: all five runs, all stages, and all detected CPUs.
Ginkgo is omitted from propermab_sequence_baseline (use src/run_ginkgo_sequence_baseline.sh).

Outputs are published to:
  propermab_abb2_pooled/analysis_results
  propermab_abb3_pooled/analysis_results
  propermab_flashabb_pooled/analysis_results
  propermab_sequence_baseline_pooled/analysis_results
  tap_pooled/analysis_results
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
# -P keeps the top-level analysis/ directory from shadowing src/analysis/.
DEFAULT_PY="conda run --no-capture-output -n ${KITAB_ENV} env PYTHONPATH=${PYTHONPATH} python -P"

METHOD_SPEC="all"
STAGE="all"
PARALLEL_JOBS=""
OUT_DIR=""
PY=""
RESUME="0"
DRY_RUN="0"
VALIDATE_TOLERANCE="0.2"
MAX_MISMATCH_RATE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --method) METHOD_SPEC="${2:?}"; shift 2 ;;
    --stage) STAGE="${2:?}"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="${2:?}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --py) PY="${2:?}"; shift 2 ;;
    --validate-tolerance) VALIDATE_TOLERANCE="${2:?}"; shift 2 ;;
    --max-mismatch-rate) MAX_MISMATCH_RATE="${2:?}"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
  esac
done

PY="${PY:-$DEFAULT_PY}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc 2>/dev/null || echo 4)}"
if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --parallel-jobs must be a positive integer." >&2
  exit 1
fi

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
  OUT_DIR="${REPO_ROOT}/runs/external_pooled_${stamp}"
fi
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
exec > >(tee -a "${OUT_DIR}/external_pooled.log") 2>&1

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

contains_method() {
  local wanted="$1"
  local method
  for method in "${METHODS[@]}"; do
    [[ "${method}" == "${wanted}" ]] && return 0
  done
  return 1
}

JOBS_FILE="${OUT_DIR}/oof_jobs.tsv"
TAP_PSEUDO_ROOT="${OUT_DIR}/tap_pseudo_automl"
REUSE_DISCOVER="0"
if [[ "${RESUME}" == "1" && -s "${JOBS_FILE}" ]]; then
  REUSE_DISCOVER="1"
fi

if { want_stage discover && [[ "${REUSE_DISCOVER}" == "1" ]]; }; then
  log_stage "discover: reuse existing ${JOBS_FILE}"
elif want_stage discover || { want_stage oof && [[ ! -s "${JOBS_FILE}" ]]; }; then
  log_stage "discover: writing ${JOBS_FILE}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    regular_methods=()
    for method in "${METHODS[@]}"; do
      [[ "${method}" != "tap" ]] && regular_methods+=("${method}")
    done
    run_py - "${REPO_ROOT}" "${OUT_DIR}" "${JOBS_FILE}" "${regular_methods[@]}" <<'PY'
import sys
from pathlib import Path

repo, out, jobs_file = map(Path, sys.argv[1:4])
jobs = []
skipped_ginkgo = 0
for method in sys.argv[4:]:
    automl = repo / method / "automl"
    result_jsons = []
    if automl.is_dir():
        for subdir in sorted(path for path in automl.iterdir() if path.is_dir()):
            result_jsons.extend(
                path for path in sorted(subdir.glob("*.json"))
                if not path.name.endswith(".oof.json")
            )
    for result_json in result_jsons:
        posix = result_json.as_posix()
        if method == "propermab_sequence_baseline" and "ginkgo_ig_folded" in posix:
            skipped_ginkgo += 1
            continue
        relative = result_json.resolve().relative_to(automl.resolve())
        oof_path = (
            out / "oof" / method / relative.with_suffix("")
        ).with_name(result_json.stem + ".oof.parquet")
        jobs.append({
            "backend": method,
            "json_path": str(result_json.resolve()),
            "oof_path": str(oof_path.resolve()),
            "automl_root": str(automl.resolve()),
        })
jobs_file.parent.mkdir(parents=True, exist_ok=True)
with jobs_file.open("w", encoding="utf-8") as handle:
    handle.write("backend\tjson_path\toof_path\tautoml_root\n")
    for job in jobs:
        handle.write(
            f"{job['backend']}\t{job['json_path']}\t"
            f"{job['oof_path']}\t{job['automl_root']}\n"
        )
print(
    f"Wrote {len(jobs)} preserved-result OOF job(s) "
    f"(skipped {skipped_ginkgo} Ginkgo sequence-baseline JSON(s))"
)
PY

    if contains_method tap; then
      tap_jobs="${OUT_DIR}/tap_oof_jobs.tsv"
      run_py "${REPO_ROOT}/src/analysis/materialize_aggregated_fold_results.py" \
        --aggregated-dir "${REPO_ROOT}/tap/automl" \
        --master-tsv "${REPO_ROOT}/tap/automl/parallel_jobs_master.tsv" \
        --pseudo-automl-root "${TAP_PSEUDO_ROOT}" \
        --oof-root "${OUT_DIR}/oof/tap" \
        --jobs-file "${tap_jobs}" \
        --method-name tap
      tail -n +2 "${tap_jobs}" >> "${JOBS_FILE}"
    fi
  fi
fi

if contains_method tap && want_stage oof; then
  if [[ "${DRY_RUN}" == "0" ]]; then
    log_stage "folds: restore TAP train/test parquets into original run dirs"
    restore_args=(
      "${REPO_ROOT}/src/analysis/prepare_nested_folds.py"
      --repo-root "${REPO_ROOT}"
      --restore-existing-tap-runs
    )
    [[ "${RESUME}" == "1" ]] && restore_args+=(--resume)
    run_py "${restore_args[@]}"
  else
    log_stage "dry-run: skip TAP fold restore"
  fi
fi

if want_stage oof; then
  [[ -s "${JOBS_FILE}" ]] || { echo "Missing ${JOBS_FILE}" >&2; exit 1; }
  if awk -F'\t' 'NR > 1 && $1 == "propermab_sequence_baseline" && $2 ~ /ginkgo_ig_folded/ { found=1; exit } END { exit !found }' "${JOBS_FILE}"; then
    log_stage "oof: dropping Ginkgo sequence-baseline jobs from ${JOBS_FILE}"
    awk -F'\t' 'NR == 1 || !($1 == "propermab_sequence_baseline" && $2 ~ /ginkgo_ig_folded/)' \
      "${JOBS_FILE}" > "${JOBS_FILE}.tmp"
    mv "${JOBS_FILE}.tmp" "${JOBS_FILE}"
  fi
  OOF_RUN_JOBS="${JOBS_FILE}"
  if [[ "${RESUME}" == "1" ]]; then
    OOF_RUN_JOBS="${OUT_DIR}/oof_jobs_pending.tsv"
    log_stage "oof: filtering jobs whose parquet already exists"
    run_py "${REPO_ROOT}/src/analysis/oof_predictions.py" pending \
      --jobs-file "${JOBS_FILE}" \
      --out "${OOF_RUN_JOBS}"
  fi
  n_jobs="$(awk 'NR > 1 {n++} END {print n+0}' "${OOF_RUN_JOBS}")"
  log_stage "oof: ${n_jobs} fold-result job(s)"
  if [[ "${DRY_RUN}" == "0" ]]; then
    if [[ "${n_jobs}" == "0" ]]; then
      log_stage "oof: nothing pending"
    else
      py_to_array "${PY}" PY_ARRAY
      resume_args=()
      [[ "${RESUME}" == "1" ]] && resume_args=(--resume)
      tail -n +2 "${OOF_RUN_JOBS}" | \
        parallel --will-cite --jobs "${PARALLEL_JOBS}" --line-buffer --eta --colsep $'\t' \
          "${PY_ARRAY[@]}" "${REPO_ROOT}/src/analysis/oof_predictions.py" run-one \
            --json-path {2} --oof-path {3} --n-jobs 1 "${resume_args[@]}"
    fi
  fi
fi

if want_stage validate; then
  log_stage "validate: stored versus recomputed fold Spearman"
  if [[ "${DRY_RUN}" == "0" ]]; then
    run_py "${REPO_ROOT}/src/analysis/oof_predictions.py" validate \
      --oof-dir "${OUT_DIR}/oof" \
      --tolerance "${VALIDATE_TOLERANCE}" \
      --max-mismatch-rate "${MAX_MISMATCH_RATE}" \
      --summary-out "${OUT_DIR}/validate_summary.json" || {
        echo "Validation failed; do not aggregate. Inspect ${OUT_DIR}/validate_summary.json" >&2
        exit 1
      }
  fi
fi

reference_csv_for() {
  case "$1" in
    propermab_abb2) printf '%s\n' "${REPO_ROOT}/descriptors_propermab_abb2/ab21_abb2_1_propermab/features.csv" ;;
    propermab_abb3) printf '%s\n' "${REPO_ROOT}/descriptors_propermab_abb3/ab21_abb3_1_propermab/features.csv" ;;
    propermab_flashabb) printf '%s\n' "${REPO_ROOT}/descriptors_propermab_flashabb/ab21_flashabb_1_propermab/features.csv" ;;
    propermab_sequence_baseline) printf '%s\n' "${OUT_DIR}/propermab_sequence_reference.csv" ;;
    tap) printf '%s\n' "${REPO_ROOT}/descriptors_tap/ab21/features.csv" ;;
  esac
}

aggregate_one() {
  local method="$1"
  local source_batch="${REPO_ROOT}/${method}/automl"
  local effective_batch="${source_batch}"
  [[ "${method}" == "tap" ]] && effective_batch="${TAP_PSEUDO_ROOT}"
  local analysis_out="${OUT_DIR}/analysis_${method}"
  mkdir -p "${analysis_out}"
  log_stage "aggregate ${method}"

  aggregate_args=(
    --manifest "${source_batch}/batch_manifest.json"
    --batch-root "${effective_batch}"
    --output-dir "${analysis_out}"
    --oof-dir "${OUT_DIR}/oof/${method}"
    --no-plots
  )
  if [[ "${effective_batch}" == "${source_batch}" ]]; then
    aggregate_args+=(--master-tsv "${source_batch}/parallel_jobs_master.tsv")
  fi
  run_py "${REPO_ROOT}/src/automl/aggregate_batch_results.py" "${aggregate_args[@]}"

  shopt -s nullglob
  aggregated_csvs=("${analysis_out}"/aggregated_*.csv)
  shopt -u nullglob
  if [[ "${method}" == "propermab_sequence_baseline" ]]; then
    filtered_csvs=()
    for csv in "${aggregated_csvs[@]}"; do
      [[ "$(basename "${csv}")" == aggregated_ginkgo_ig_folded* ]] && continue
      filtered_csvs+=("${csv}")
    done
    aggregated_csvs=("${filtered_csvs[@]}")
  fi
  [[ ${#aggregated_csvs[@]} -gt 0 ]] || {
    echo "No aggregated CSVs produced for ${method}" >&2
    return 1
  }
  local reference_csv
  reference_csv="$(reference_csv_for "${method}")"
  run_py "${REPO_ROOT}/src/analysis/analyze_results.py" \
    "${aggregated_csvs[@]}" \
    --out-dir "${analysis_out}" \
    --summary-name best_metrics_summary.csv \
    --results-name results.csv \
    --reference-json "${reference_csv}" \
    --no-plots
  run_py "${REPO_ROOT}/src/analysis/random_spearman_baseline.py" \
    --selection-matched \
    --oof-dir "${OUT_DIR}/oof/${method}" \
    --out "${analysis_out}/results_random_baseline_selection_matched.csv" \
    --seed 42 \
    --repo-root "${REPO_ROOT}"
}

if want_stage aggregate; then
  if [[ "${DRY_RUN}" == "0" ]]; then
    printf '%s\n' \
      "name,cdr_h3_length,aromatic_cdr,theoretical_pi,n_charged_res_fv,fv_charge,fv_csp" \
      "reference,0,0,0,0,0,0" \
      > "${OUT_DIR}/propermab_sequence_reference.csv"
    for method in "${METHODS[@]}"; do
      aggregate_one "${method}"
    done
  fi
fi

if want_stage publish; then
  for method in "${METHODS[@]}"; do
    analysis_out="${OUT_DIR}/analysis_${method}"
    destination="${REPO_ROOT}/${method}_pooled/analysis_results"
    [[ -d "${analysis_out}" ]] || {
      echo "Missing ${analysis_out}; run --stage aggregate first." >&2
      exit 1
    }
    log_stage "publish ${method}: ${destination}"
    if [[ "${DRY_RUN}" == "0" ]]; then
      mkdir -p "${destination}"
      cp -a "${analysis_out}/." "${destination}/"
    fi
  done
fi

log_stage "complete"
