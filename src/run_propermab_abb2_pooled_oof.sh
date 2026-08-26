#!/usr/bin/env bash
# Rebuild isolated structural ProperMAb ABB2 folds and recompute pooled OOF.
#
# Does not overwrite the shared sequence-baseline 6-feature dirs, and does not
# touch ABB3 / FlashABB / TAP / sequence-baseline OOF. Writes into the existing
# external pooled run by default so later aggregate/publish can use it.
#
# Launch:
#   tmux new -s abb2-pooled-oof
#   ./src/run_propermab_abb2_pooled_oof.sh --parallel-jobs 100
#
# Resume after interruption (keeps ABB2 OOF parquets already written this rerun):
#   ./src/run_propermab_abb2_pooled_oof.sh \
#     --out-dir runs/external_pooled_20260820T152340Z --resume --parallel-jobs 100

set -euo pipefail

log_stage() {
  echo >&2 "[abb2-pooled-oof] $*"
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 [--stage all|folds|oof|validate]
     [--parallel-jobs N] [--out-dir DIR] [--resume] [--py CMD] [--dry-run]
     [--validate-tolerance X] [--max-mismatch-rate R]

Default out-dir: runs/external_pooled_20260820T152340Z if it exists.
Without --resume, existing oof/propermab_abb2 parquets are deleted first.
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

STAGE="all"
PARALLEL_JOBS=""
OUT_DIR=""
PY=""
RESUME="0"
DRY_RUN="0"
VALIDATE_TOLERANCE="0.2"
MAX_MISMATCH_RATE="0"
DEFAULT_POOLED="${REPO_ROOT}/runs/external_pooled_20260820T152340Z"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
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

case "${STAGE}" in
  all|folds|oof|validate) ;;
  *) echo "Unknown --stage ${STAGE}" >&2; usage; exit 1 ;;
esac

PY="${PY:-$DEFAULT_PY}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc 2>/dev/null || echo 4)}"
if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --parallel-jobs must be a positive integer." >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  if [[ -d "${DEFAULT_POOLED}" ]]; then
    OUT_DIR="${DEFAULT_POOLED}"
  else
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT_DIR="${REPO_ROOT}/runs/external_pooled_${stamp}"
  fi
elif [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
exec > >(tee -a "${OUT_DIR}/abb2_pooled_oof.log") 2>&1

log_stage "out-dir=${OUT_DIR} stage=${STAGE} jobs=${PARALLEL_JOBS} resume=${RESUME}"

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

AUTOML_ROOT="${REPO_ROOT}/propermab_abb2/automl"
MAP_OUT="${OUT_DIR}/fold_maps/propermab_abb2.json"
JOBS_FILE="${OUT_DIR}/oof_jobs_abb2.tsv"
OOF_DIR="${OUT_DIR}/oof/propermab_abb2"

if [[ ! -d "${AUTOML_ROOT}" ]]; then
  echo "Missing AutoML root ${AUTOML_ROOT}" >&2
  exit 1
fi

if want_stage folds; then
  if [[ "${DRY_RUN}" == "0" ]]; then
    log_stage "folds: isolated structural ABB2 parquets for every yaml_key"
    restore_args=(
      "${REPO_ROOT}/src/analysis/prepare_nested_folds.py"
      --repo-root "${REPO_ROOT}"
      --method propermab_abb2
      --automl-root "${AUTOML_ROOT}"
      --map-out "${MAP_OUT}"
      --all-automl-yaml-keys
    )
    [[ "${RESUME}" == "1" ]] && restore_args+=(--resume)
    run_py "${restore_args[@]}"
  else
    log_stage "dry-run: skip ABB2 fold rebuild"
  fi
fi

if want_stage oof && [[ "${DRY_RUN}" == "0" ]]; then
  if [[ ! -s "${MAP_OUT}" ]]; then
    echo "Missing fold-dir map ${MAP_OUT}; run --stage folds first." >&2
    exit 1
  fi
fi

if want_stage oof; then
  if [[ "${RESUME}" != "1" && "${DRY_RUN}" == "0" && -d "${OOF_DIR}" ]]; then
    log_stage "oof: removing stale ${OOF_DIR}"
    rm -rf "${OOF_DIR}"
  fi
  log_stage "oof: writing ${JOBS_FILE}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    run_py - "${REPO_ROOT}" "${OUT_DIR}" "${JOBS_FILE}" "${AUTOML_ROOT}" <<'PY'
import sys
from pathlib import Path

repo, out, jobs_file, automl = map(Path, sys.argv[1:5])
jobs = []
if automl.is_dir():
    for subdir in sorted(path for path in automl.iterdir() if path.is_dir()):
        for result_json in sorted(subdir.glob("*.json")):
            if result_json.name.endswith(".oof.json"):
                continue
            relative = result_json.resolve().relative_to(automl.resolve())
            oof_path = (
                out / "oof" / "propermab_abb2" / relative.with_suffix("")
            ).with_name(result_json.stem + ".oof.parquet")
            jobs.append(
                (
                    "propermab_abb2",
                    str(result_json.resolve()),
                    str(oof_path.resolve()),
                    str(automl.resolve()),
                )
            )
jobs_file.parent.mkdir(parents=True, exist_ok=True)
with jobs_file.open("w", encoding="utf-8") as handle:
    handle.write("backend\tjson_path\toof_path\tautoml_root\n")
    for backend, json_path, oof_path, automl_root in jobs:
        handle.write(f"{backend}\t{json_path}\t{oof_path}\t{automl_root}\n")
print(f"Wrote {len(jobs)} ProperMAb ABB2 OOF job(s)")
PY
  fi
  [[ -s "${JOBS_FILE}" ]] || { echo "Missing ${JOBS_FILE}" >&2; exit 1; }
  OOF_RUN_JOBS="${JOBS_FILE}"
  if [[ "${RESUME}" == "1" ]]; then
    OOF_RUN_JOBS="${OUT_DIR}/oof_jobs_abb2_pending.tsv"
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
            --json-path {2} --oof-path {3} --n-jobs 1 \
            --fold-dir-map "${MAP_OUT}" "${resume_args[@]}"
    fi
  fi
fi

if want_stage validate; then
  log_stage "validate: stored versus recomputed fold Spearman (ABB2 only)"
  if [[ "${DRY_RUN}" == "0" ]]; then
    run_py "${REPO_ROOT}/src/analysis/oof_predictions.py" validate \
      --oof-dir "${OOF_DIR}" \
      --tolerance "${VALIDATE_TOLERANCE}" \
      --max-mismatch-rate "${MAX_MISMATCH_RATE}" \
      --summary-out "${OUT_DIR}/validate_summary_abb2.json" || {
        echo "ABB2 validation failed. Inspect ${OUT_DIR}/validate_summary_abb2.json" >&2
        exit 1
      }
  fi
fi

log_stage "complete"
log_stage "ABB2 OOF: ${OOF_DIR}"
log_stage "validate: ${OUT_DIR}/validate_summary_abb2.json"
log_stage "resume: $0 --out-dir ${OUT_DIR} --resume --stage all --parallel-jobs ${PARALLEL_JOBS}"
