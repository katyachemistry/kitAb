#!/usr/bin/env bash
# Recalculate the Ginkgo ProperMAb sequence baseline from scratch:
# isolated 13-feature folds, flat AutoML, pooled-OOF Spearman, nested CV.
#
# Does not write into propermab_sequence_baseline/. The stopped external pooled
# OOF run never reached that method; Ginkgo sequence-baseline OOF parquets were
# not created there.
#
# Launch:
#   tmux new -s ginkgo-seq-baseline
#   ./src/run_ginkgo_sequence_baseline.sh --parallel-jobs 100
#
# Resume:
#   ./src/run_ginkgo_sequence_baseline.sh \
#     --out-dir runs/ginkgo_sequence_baseline_<stamp> --resume --parallel-jobs 100

set -euo pipefail

log_stage() {
  echo >&2 "[ginkgo-seq-baseline] $*"
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 [--stage all|automl|oof|validate|aggregate|nested]
     [--parallel-jobs N] [--out-dir DIR] [--resume] [--py CMD]
     [--dry-run] [--validate-tolerance X] [--max-mismatch-rate R]
     [--variants 1,2,3]

Stages:
  automl     Rebuild isolated Ginkgo folds (6 sequence + HC/LC one-hots) and
             run flat AutoML + post-grid floating SFS into OUT_DIR/automl
  oof        Recompute per-sample OOF predictions from the new result JSONs
  validate   Compare stored vs recomputed fold Spearman
  aggregate  Pooled-OOF Spearman + analysis CSVs
  nested     Nested CV on ABB2 variant 1 (all 11 Ginkgo targets, including HAC)
  all        automl + oof + validate + aggregate + nested

Fold directories use feature-aware names and are not shared with structural
ProperMAb:
  runs/ginkgo_ig_folded_cv_prepare__propermab_sequence_baseline__descriptors_propermab_abb2_ginkgo_ig_folded_abb2_{1,2,3}_propermab
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

STAGE="all"
PARALLEL_JOBS=""
OUT_DIR=""
PY=""
RESUME="0"
DRY_RUN="0"
VALIDATE_TOLERANCE="0.2"
MAX_MISMATCH_RATE="0"
VARIANTS_SPEC="1,2,3"

GINKGO_TARGETS="target_SEC_Monomer,target_SMAC,target_HIC,target_HAC,target_PR_CHO,target_PR_Ova,target_AC_SINS_pH6_0,target_AC_SINS_pH7_4,target_Tonset,target_Tm1,target_Tm2"
GINKGO_CSV="${REPO_ROOT}/datasets/ginkgo_ig_folded.csv"
NESTED_FLAT_VARIANT="abb2_1_propermab"
YAML_KEY_SUFFIX="_propermab"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --stage) STAGE="${2:?}"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="${2:?}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --py) PY="${2:?}"; shift 2 ;;
    --validate-tolerance) VALIDATE_TOLERANCE="${2:?}"; shift 2 ;;
    --max-mismatch-rate) MAX_MISMATCH_RATE="${2:?}"; shift 2 ;;
    --variants) VARIANTS_SPEC="${2:?}"; shift 2 ;;
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
case "${STAGE}" in
  all|automl|oof|validate|aggregate|nested) ;;
  *) echo "Unknown --stage ${STAGE}" >&2; usage; exit 1 ;;
esac

IFS=',' read -r -a VARIANTS <<< "${VARIANTS_SPEC}"
for v in "${VARIANTS[@]}"; do
  case "${v}" in
    1|2|3) ;;
    *) echo "Unknown variant ${v}; expected 1, 2, and/or 3." >&2; exit 1 ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  OUT_DIR="${REPO_ROOT}/runs/ginkgo_sequence_baseline_${stamp}"
elif [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
exec > >(tee -a "${OUT_DIR}/ginkgo_sequence_baseline.log") 2>&1

log_stage "out-dir=${OUT_DIR} stage=${STAGE} jobs=${PARALLEL_JOBS} resume=${RESUME} variants=${VARIANTS[*]}"

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

fold_dir_for() {
  local variant="$1"
  printf '%s\n' \
    "${REPO_ROOT}/runs/ginkgo_ig_folded_cv_prepare__propermab_sequence_baseline__descriptors_propermab_abb2_ginkgo_ig_folded_abb2_${variant}_propermab"
}

descriptor_dir_for() {
  local variant="$1"
  printf '%s\n' \
    "${REPO_ROOT}/descriptors_propermab_abb2/ginkgo_ig_folded_abb2_${variant}_propermab"
}

for v in "${VARIANTS[@]}"; do
  desc="$(descriptor_dir_for "${v}")"
  if [[ ! -d "${desc}" ]]; then
    echo "Missing ProperMAb ABB2 descriptors for Ginkgo variant ${v}: ${desc}" >&2
    echo "Sequence-baseline folds subset those JSONs; they must already exist." >&2
    exit 1
  fi
done
if [[ ! -f "${GINKGO_CSV}" ]]; then
  echo "Missing ${GINKGO_CSV}" >&2
  exit 1
fi

write_run_config() {
  local dest="$1"
  local skip_existing="$2"
  local automl_yaml="$3"
  {
    echo "skip_existing_results: ${skip_existing}"
    echo "batch_result_root: ${OUT_DIR}/automl"
    echo "automl_config: ${automl_yaml}"
    echo
    for v in "${VARIANTS[@]}"; do
      cat <<EOF
ginkgo_ig_folded_abb2_${v}_propermab:
  path: ${GINKGO_CSV}
  developability_results_path: $(descriptor_dir_for "${v}")
  name_col: name
  target_cols: ${GINKGO_TARGETS}
  feature_cols: feature_hc_subtype,feature_lc_subtype
  include_features:
    - cdr_h3_length
    - aromatic_cdr
    - theoretical_pi
    - n_charged_res_fv
    - fv_charge
    - fv_csp
  split_col: fold
  run_dir: $(fold_dir_for "${v}")

EOF
    done
  } > "${dest}"
}

write_automl_yaml() {
  local dest="$1"
  python3 - "${REPO_ROOT}/src/automl.yaml" "${dest}" <<'PY'
from pathlib import Path
import sys

src, dest = map(Path, sys.argv[1:3])
text = src.read_text()
old_linear = """    # final_floating_sfs:
    #   max_feature_fraction: 0.15
    #   model: elasticnet,svm
"""
new_linear = """    final_floating_sfs:
      max_feature_fraction: 0.15
      model: elasticnet,svm
"""
old_nl = """    # final_floating_sfs:
    #   max_feature_fraction: 0.15
    #   model: randomforest,knn
"""
new_nl = """    final_floating_sfs:
      max_feature_fraction: 0.15
      model: randomforest,knn
"""
if old_linear not in text or old_nl not in text:
    raise SystemExit("Could not enable final_floating_sfs in src/automl.yaml")
dest.write_text(text.replace(old_linear, new_linear, 1).replace(old_nl, new_nl, 1))
PY
}

AUTOML_ROOT="${OUT_DIR}/automl"
RUN_CONFIG="${OUT_DIR}/ginkgo_sequence_baseline.yaml"
AUTOML_YAML="${OUT_DIR}/automl.yaml"
JOBS_FILE="${OUT_DIR}/oof_jobs.tsv"
ANALYSIS_OUT="${OUT_DIR}/analysis"
NESTED_ROOT="${OUT_DIR}/nested_abb2"
NESTED_JOBS="${OUT_DIR}/nested_jobs.tsv"
NESTED_FLOATING_JOBS="${OUT_DIR}/nested_floating_jobs.tsv"
NESTED_REPORT="${OUT_DIR}/nested_report.csv"

if want_stage automl; then
  expected_ffs=$(( ${#VARIANTS[@]} * 11 * 5 * 4 ))
  n_ffs=0
  if [[ -d "${AUTOML_ROOT}" ]]; then
    n_ffs="$(find "${AUTOML_ROOT}" -name '*final_floating_sfs*.json' | wc -l | tr -d ' ')"
  fi
  if [[ "${RESUME}" == "1" && "${n_ffs}" -ge "${expected_ffs}" ]]; then
    log_stage "automl: resume skip (${n_ffs} floating-SFS JSON(s) already present)"
  else
    log_stage "automl: isolated Ginkgo folds + flat CV"
    if [[ "${DRY_RUN}" == "0" ]]; then
      write_automl_yaml "${AUTOML_YAML}"
      skip_existing="false"
      [[ "${RESUME}" == "1" ]] && skip_existing="true"
      write_run_config "${RUN_CONFIG}" "${skip_existing}" "${AUTOML_YAML}"
      automl_args=(
        "${REPO_ROOT}/src/automl/prepare_parallel_from_config.py"
        "${RUN_CONFIG}"
        --parallel-jobs "${PARALLEL_JOBS}"
        --py "${PY}"
        --no-aggregate
        --no-preprocessing-skip
      )
      run_py "${automl_args[@]}"
    else
      log_stage "dry-run: skip automl"
    fi
  fi
fi

if want_stage oof; then
  if [[ "${DRY_RUN}" == "0" ]]; then
    [[ -d "${AUTOML_ROOT}" ]] || {
      echo "Missing ${AUTOML_ROOT}; run --stage automl first." >&2
      exit 1
    }
    log_stage "oof: discover jobs -> ${JOBS_FILE}"
    run_py - "${AUTOML_ROOT}" "${OUT_DIR}/oof" "${JOBS_FILE}" <<'PY'
import sys
from pathlib import Path

automl, oof_root, jobs_file = map(Path, sys.argv[1:4])
jobs = []
for subdir in sorted(path for path in automl.iterdir() if path.is_dir()):
    for result_json in sorted(subdir.glob("*.json")):
        if result_json.name.endswith(".oof.json"):
            continue
        relative = result_json.resolve().relative_to(automl.resolve())
        oof_path = (oof_root / relative.with_suffix("")).with_name(
            result_json.stem + ".oof.parquet"
        )
        jobs.append((str(result_json.resolve()), str(oof_path.resolve())))
jobs_file.parent.mkdir(parents=True, exist_ok=True)
with jobs_file.open("w", encoding="utf-8") as handle:
    handle.write("json_path\toof_path\n")
    for json_path, oof_path in jobs:
        handle.write(f"{json_path}\t{oof_path}\n")
print(f"Wrote {len(jobs)} OOF job(s)")
PY
    n_jobs="$(awk 'NR > 1 {n++} END {print n+0}' "${JOBS_FILE}")"
    log_stage "oof: ${n_jobs} fold-result job(s)"
    py_to_array "${PY}" PY_ARRAY
    resume_args=()
    [[ "${RESUME}" == "1" ]] && resume_args=(--resume)
    tail -n +2 "${JOBS_FILE}" | \
      parallel --will-cite --jobs "${PARALLEL_JOBS}" --line-buffer --eta --colsep $'\t' \
        "${PY_ARRAY[@]}" "${REPO_ROOT}/src/analysis/oof_predictions.py" run-one \
          --json-path {1} --oof-path {2} --n-jobs 1 "${resume_args[@]}"
  else
    log_stage "dry-run: skip oof"
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
  else
    log_stage "dry-run: skip validate"
  fi
fi

if want_stage aggregate; then
  log_stage "aggregate pooled OOF Spearman"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "${ANALYSIS_OUT}"
    printf '%s\n' \
      "name,cdr_h3_length,aromatic_cdr,theoretical_pi,n_charged_res_fv,fv_charge,fv_csp" \
      "reference,0,0,0,0,0,0" \
      > "${OUT_DIR}/propermab_sequence_reference.csv"
    run_py "${REPO_ROOT}/src/automl/aggregate_batch_results.py" \
      --manifest "${AUTOML_ROOT}/batch_manifest.json" \
      --batch-root "${AUTOML_ROOT}" \
      --master-tsv "${AUTOML_ROOT}/parallel_jobs_master.tsv" \
      --output-dir "${ANALYSIS_OUT}" \
      --oof-dir "${OUT_DIR}/oof" \
      --no-plots
    shopt -s nullglob
    aggregated_csvs=("${ANALYSIS_OUT}"/aggregated_*.csv)
    shopt -u nullglob
    [[ ${#aggregated_csvs[@]} -gt 0 ]] || {
      echo "No aggregated CSVs produced" >&2
      exit 1
    }
    run_py "${REPO_ROOT}/src/analysis/analyze_results.py" \
      "${aggregated_csvs[@]}" \
      --out-dir "${ANALYSIS_OUT}" \
      --summary-name best_metrics_summary.csv \
      --results-name results.csv \
      --reference-json "${OUT_DIR}/propermab_sequence_reference.csv" \
      --no-plots
    run_py "${REPO_ROOT}/src/analysis/random_spearman_baseline.py" \
      --selection-matched \
      --oof-dir "${OUT_DIR}/oof" \
      --out "${ANALYSIS_OUT}/results_random_baseline_selection_matched.csv" \
      --seed 42 \
      --repo-root "${REPO_ROOT}" || true
  else
    log_stage "dry-run: skip aggregate"
  fi
fi

if want_stage nested; then
  log_stage "nested: Ginkgo ABB2 variant 1 only (${NESTED_FLAT_VARIANT})"
  if [[ "${DRY_RUN}" == "0" ]]; then
    [[ -d "${AUTOML_ROOT}" ]] || {
      echo "Missing ${AUTOML_ROOT}; run --stage automl first." >&2
      exit 1
    }
    mkdir -p "${NESTED_ROOT}"
    run_py "${REPO_ROOT}/src/analysis/nested_cv.py" discover \
      --repo-root "${REPO_ROOT}" \
      --out-dir "${NESTED_ROOT}" \
      --jobs-file "${NESTED_JOBS}" \
      --floating-jobs-file "${NESTED_FLOATING_JOBS}" \
      --backend abb2 \
      --automl-root "${AUTOML_ROOT}" \
      --yaml-key-suffix "${YAML_KEY_SUFFIX}" \
      --pairs-mode default \
      --include-stems ginkgo_ig_folded \
      --prepare-inner
    if [[ ! -s "${NESTED_JOBS}" ]]; then
      echo "Nested jobs TSV is empty. Check ${AUTOML_ROOT}/ginkgo_ig_folded_abb2_1_propermab" >&2
      exit 1
    fi
    NESTED_RUN_JOBS="${NESTED_JOBS}"
    if [[ "${RESUME}" == "1" ]]; then
      NESTED_RUN_JOBS="${OUT_DIR}/nested_jobs_pending.tsv"
      run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
        --jobs-file "${NESTED_JOBS}" \
        --out "${NESTED_RUN_JOBS}" \
        --require-oof
    fi
    py_to_array "${PY}" PY_ARR
    n_nj="$(tail -n +2 "${NESTED_RUN_JOBS}" | wc -l | tr -d ' ')"
    log_stage "nested inner jobs: ${n_nj}"
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
          echo "Nested inner parallel failed. Resume from ${NESTED_JOBS}" >&2
          echo "  $0 --out-dir ${OUT_DIR} --resume --stage nested" >&2
          exit 1
        }
    fi
    NESTED_REGULAR_MISSING="${OUT_DIR}/nested_jobs_missing.tsv"
    run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
      --jobs-file "${NESTED_JOBS}" \
      --out "${NESTED_REGULAR_MISSING}" \
      --require-oof
    n_regular_missing="$(tail -n +2 "${NESTED_REGULAR_MISSING}" | wc -l | tr -d ' ')"
    if [[ "${n_regular_missing}" != "0" ]]; then
      echo "Nested regular grid incomplete: ${n_regular_missing} job(s) lack OOF results." >&2
      echo "Inspect ${NESTED_REGULAR_MISSING}; do not start floating SFS." >&2
      exit 1
    fi
    NESTED_FLOATING_RUN_JOBS="${NESTED_FLOATING_JOBS}"
    if [[ "${RESUME}" == "1" ]]; then
      NESTED_FLOATING_RUN_JOBS="${OUT_DIR}/nested_floating_jobs_pending.tsv"
      run_py "${REPO_ROOT}/src/analysis/nested_cv.py" pending \
        --jobs-file "${NESTED_FLOATING_JOBS}" \
        --out "${NESTED_FLOATING_RUN_JOBS}" \
        --require-oof
    fi
    n_ffs="$(tail -n +2 "${NESTED_FLOATING_RUN_JOBS}" | wc -l | tr -d ' ')"
    log_stage "nested post-grid floating-SFS jobs: ${n_ffs}"
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
          echo "  $0 --out-dir ${OUT_DIR} --resume --stage nested" >&2
          exit 1
        }
    fi
    NESTED_FLOATING_MISSING="${OUT_DIR}/nested_floating_jobs_missing.tsv"
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
    export NESTED_ROOT NESTED_JOBS OUT_DIR
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
Path(os.environ["OUT_DIR"], "nested_finish_pairs.tsv").write_text(
    "pair_root\torig_fold_dir\n" + "\n".join(f"{a}\t{b}" for a, b in rows) + ("\n" if rows else "")
)
PY
    if [[ -s "${OUT_DIR}/nested_finish_pairs.tsv" ]]; then
      tail -n +2 "${OUT_DIR}/nested_finish_pairs.tsv" | while IFS=$'\t' read -r pair orig; do
        if [[ "${RESUME}" == "1" && -f "${pair}/nested_summary.json" ]]; then
          log_stage "nested finish skip ${pair}"
          continue
        fi
        log_stage "nested finish ${pair}"
        run_py "${REPO_ROOT}/src/analysis/nested_cv.py" finish-pair \
          --pair-root "${pair}" --orig-fold-dir "${orig}"
      done
    fi
    FLAT_RESULTS="${ANALYSIS_OUT}/results.csv"
    run_py "${REPO_ROOT}/src/analysis/nested_cv.py" report \
      --nested-root "${NESTED_ROOT}" \
      --automl-root "${AUTOML_ROOT}" \
      --dest "${NESTED_REPORT}" \
      --flat-results "${FLAT_RESULTS}" \
      --backend abb2 \
      --yaml-key-suffix "${YAML_KEY_SUFFIX}" \
      --flat-variant "${NESTED_FLAT_VARIANT}" \
      --pairs-mode default \
      --include-stems ginkgo_ig_folded
  else
    log_stage "dry-run: skip nested"
  fi
fi

log_stage "complete"
log_stage "flat pooled Spearman: ${ANALYSIS_OUT}/results.csv"
log_stage "nested pooled Spearman: ${NESTED_REPORT}"
log_stage "resume: $0 --out-dir ${OUT_DIR} --resume --stage ${STAGE}"
