#!/usr/bin/env bash
# FASTAb pipeline: generic config -> run config -> structures -> descriptors -> AutoML -> analysis.
#
# Usage:
#   ./fastab.sh configs/scenario1.yaml
#
# Step 1: read generic config from configs/, write run-specific config to run_configs/
#         (one top-level block per dataset; paths under <result_folder>/).
# Step 2: structure prediction (src/predict_structure.sh) when calculate_descriptors is True.
# Step 3: developability descriptors (src/get_descriptors.sh).
# Step 4: AutoML (src/run_automl.sh) unless automl: false in the generic config.
# Step 5: analysis (src/analysis/analyze_results.py) on aggregated AutoML CSVs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$REPO_ROOT/src"
PREPARE_SCRIPT="$SRC_DIR/utils/prepare_run_config.py"
PREDICT_SCRIPT="$SRC_DIR/predict_structure.sh"
DESCRIPTORS_SCRIPT="$SRC_DIR/get_descriptors.sh"
AUTOML_SCRIPT="$SRC_DIR/run_automl.sh"

if [[ -f "$REPO_ROOT/fastab.local.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/fastab.local.env"
fi

PY="${PY:-python3}"

_stage() {
    echo "[fastab] $*" >&2
}

_die() {
    echo "[fastab] ERROR: $*" >&2
    exit 1
}

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

# Read a scalar from the generic config YAML.
read_generic() {
    "$PY" -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg[sys.argv[2]])' "$GENERIC_CONFIG" "$1"
}

read_generic_nested() {
    "$PY" -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg[sys.argv[2]][sys.argv[3]])' "$GENERIC_CONFIG" "$1" "$2"
}

read_generic_optional() {
    "$PY" -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); v=cfg.get(sys.argv[2]); print("" if v is None else v)' "$GENERIC_CONFIG" "$1"
}

# batch_result_root from the generated run config (repo-relative or absolute).
read_run_config_batch_root() {
    "$PY" -c '
import yaml, sys
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1]))
root = Path(sys.argv[2])
br = cfg.get("batch_result_root")
if br:
    p = Path(br)
    print(str(p if p.is_absolute() else (root / p).resolve()))
' "$RUN_CONFIG" "$REPO_ROOT"
}

# Structure folder for descriptors: only {stem}_{model}_{run} (no _imgt / _minimized).
structure_dir_for_descriptors() {
    local stem="$1" model="$2" run_idx="$3"
    local base="$STRUCTURES_ROOT/${stem}_${model}_${run_idx}"
    if [[ -d "$base" ]]; then
        echo "$base"
    else
        echo ""
    fi
}

if [[ $# -lt 1 ]]; then
    usage >&2
    _die "Missing config YAML path"
fi

GENERIC_CONFIG="$1"
if [[ ! -f "$GENERIC_CONFIG" ]]; then
    _die "Config not found: $GENERIC_CONFIG"
fi

# ---------------------------------------------------------------------------
# Step 1: generic config -> run config
# ---------------------------------------------------------------------------

_stage "Preparing run config from $GENERIC_CONFIG"
RUN_CONFIG="$("$PY" "$PREPARE_SCRIPT" "$GENERIC_CONFIG" --repo-root "$REPO_ROOT")"
_stage "Run config: $RUN_CONFIG"

CALCULATE_DESCRIPTORS="$("$PY" -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('calculate_descriptors', True))" "$GENERIC_CONFIG")"
SKIP_AUTOML="$("$PY" -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('automl', True) is False)" "$GENERIC_CONFIG")"
N_CPU="$(read_generic_optional n_cpu)"

PARALLEL_JOBS="${N_CPU:-$(nproc)}"

run_automl_pipeline() {
    local -a extra=()
    if [[ -n "$N_CPU" ]]; then
        extra=(--parallel-jobs "$N_CPU")
    fi
    _stage "AutoML (run config: $RUN_CONFIG)"
    bash "$AUTOML_SCRIPT" --config "$RUN_CONFIG" "${extra[@]}"
}

run_analysis() {
    local batch_root="$1"
    local analysis_out="$2"

    shopt -s nullglob
    local -a agg_files=("$batch_root"/aggregated_*.csv)
    shopt -u nullglob

    if [[ ${#agg_files[@]} -eq 0 ]]; then
        _die "No aggregated_*.csv under $batch_root (AutoML aggregation may have failed)"
    fi

    local ref_json=""
    shopt -s nullglob
    local -a sample_jsons=("$REPO_ROOT"/"$RESULT_FOLDER"/descriptors/*/results/*.json)
    shopt -u nullglob
    if [[ ${#sample_jsons[@]} -gt 0 ]]; then
        ref_json="${sample_jsons[0]}"
    else
        ref_json="$REPO_ROOT/descriptors_reproducibility_finalization/pdgf38_abb2_1/results/AB-001.json"
        if [[ ! -f "$ref_json" ]]; then
            _die "No descriptor JSON for --reference-json (run descriptors first or set a valid path)"
        fi
    fi

    mkdir -p "$analysis_out"
    _stage "Analysis (${#agg_files[@]} aggregated CSVs -> $analysis_out)"
    (
        cd "$SRC_DIR"
        export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        "$PY" -m analysis.analyze_results \
            "${agg_files[@]}" \
            --out-dir "$analysis_out" \
            --reference-json "$ref_json" \
            --plot-fold-spearmans \
            --plot-out-dir "$analysis_out"
    )
}

# ---------------------------------------------------------------------------
# Scenario 4: descriptors already computed; AutoML + analysis only
# ---------------------------------------------------------------------------

if [[ "$CALCULATE_DESCRIPTORS" == "False" ]]; then
    _stage "Skipping structure prediction and descriptor calculation."
    if [[ "$SKIP_AUTOML" == "True" ]]; then
        _die "calculate_descriptors: false with automl: false leaves nothing to run"
    fi
    run_automl_pipeline
    RESULT_FOLDER="$(read_generic result_folder)"
    BATCH_ROOT="$(read_run_config_batch_root)"
    if [[ -z "$BATCH_ROOT" ]]; then
        BATCH_ROOT="$REPO_ROOT/runs/batch_$(basename "$RUN_CONFIG" .yaml)"
    fi
    run_analysis "$BATCH_ROOT" "$REPO_ROOT/$RESULT_FOLDER/analysis_results"
    _stage "Done. Run config: $RUN_CONFIG"
    exit 0
fi

# ---------------------------------------------------------------------------
# Scenarios 1–3: predict structures + descriptors
# ---------------------------------------------------------------------------

RESULT_FOLDER="$(read_generic result_folder)"
INPUT_CSVS="$REPO_ROOT/$(read_generic input_csvs_folder)"
STRUCTURES_ROOT="$REPO_ROOT/$RESULT_FOLDER/structures"
DESCRIPTORS_ROOT="$REPO_ROOT/$RESULT_FOLDER/descriptors"
MODEL="$(read_generic_nested structure_prediction model)"
RUNS="$(read_generic_nested structure_prediction runs)"

mkdir -p "$STRUCTURES_ROOT" "$DESCRIPTORS_ROOT"

run_structure_prediction() {
    local device
    device="$(read_generic_nested structure_prediction device)"
    if [[ "$device" =~ ^[0-9]+$ ]]; then
        device="cuda:$device"
    fi

    export ABB3_DEVICE="$device"
    export ABB2_DEVICE="$device"

    local -a args=(
        --data-dir "$INPUT_CSVS"
        --output-root "$STRUCTURES_ROOT"
        --runs "$RUNS"
    )

    if [[ "$MODEL" == "abb3" ]]; then
        args=(--abb3 "${args[@]}")
    else
        args=(--abb2 "${args[@]}")
    fi

    _stage "Structure prediction ($MODEL, device=$device, runs=$RUNS)"
    _stage "  CSVs:       $INPUT_CSVS"
    _stage "  Structures: $STRUCTURES_ROOT"

    bash "$PREDICT_SCRIPT" "${args[@]}"
}

run_descriptor_calculation() {
    local -a structure_dirs=()
    local csv stem run_idx dir

    shopt -s nullglob
    local csv_files=("$INPUT_CSVS"/*.csv)
    shopt -u nullglob

    if [[ ${#csv_files[@]} -eq 0 ]]; then
        _die "No CSV files in $INPUT_CSVS"
    fi

    for csv in "${csv_files[@]}"; do
        stem="$(basename "$csv" .csv)"
        for (( run_idx=1; run_idx<=RUNS; run_idx++ )); do
            dir="$(structure_dir_for_descriptors "$stem" "$MODEL" "$run_idx")"
            if [[ -z "$dir" ]]; then
                _stage "  [descriptors] skip missing structures: ${stem}_${MODEL}_${run_idx}" >&2
                continue
            fi
            structure_dirs+=("$dir")
        done
    done

    if [[ ${#structure_dirs[@]} -eq 0 ]]; then
        _die "No structure directories found under $STRUCTURES_ROOT for descriptor calculation"
    fi

    _stage "Descriptor calculation (${#structure_dirs[@]} structure folders -> $DESCRIPTORS_ROOT)"
    bash "$DESCRIPTORS_SCRIPT" \
        --output-dir "$DESCRIPTORS_ROOT" \
        "${structure_dirs[@]}" \
        "$PARALLEL_JOBS"
}

run_structure_prediction
run_descriptor_calculation

if [[ "$SKIP_AUTOML" == "True" ]]; then
    _stage "automl: false — skipping AutoML and analysis (scenario 3)"
    _stage "Done. Descriptors under $DESCRIPTORS_ROOT"
    exit 0
fi

run_automl_pipeline

BATCH_ROOT="$(read_run_config_batch_root)"
if [[ -z "$BATCH_ROOT" ]]; then
    BATCH_ROOT="$REPO_ROOT/runs/batch_$(basename "$RUN_CONFIG" .yaml)"
fi
run_analysis "$BATCH_ROOT" "$REPO_ROOT/$RESULT_FOLDER/analysis_results"

_stage "Done."
_stage "  Run config:   $RUN_CONFIG"
_stage "  Structures:   $STRUCTURES_ROOT"
_stage "  Descriptors:  $DESCRIPTORS_ROOT"
_stage "  AutoML batch: $BATCH_ROOT"
_stage "  Analysis:     $REPO_ROOT/$RESULT_FOLDER/analysis_results"
