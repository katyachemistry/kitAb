#!/usr/bin/env bash
# FASTAb pipeline: generic config -> run config -> structures -> descriptors -> AutoML -> analysis.
#
# Usage:
#   ./fastab.sh configs/scenario1.yaml
#
# Step 1: read generic config from configs/, write run-specific config to run_configs/
#         (one top-level block per dataset; paths under <result_folder>/).
# Step 2: structure prediction (src/predict_structure.sh) when calculate_descriptors is True
#         and input_structures_folder is not set (scenario 1).
# Step 2b: validate existing structures when input_structures_folder is set (scenario 2).
# Step 3: developability descriptors (src/get_descriptors.sh).
# Step 4: AutoML (src/run_automl.sh) unless automl: false in the generic config.
# Step 5: analysis (src/analysis/analyze_results.py) on aggregated AutoML CSVs.
# Step 6: eval hyperparameter grid search (models under <automl>/tuned_models/).
# Step 6b: after tuning succeeds, remove fold parquets from tuned run_dir(s) only.
# Step 7: remove per-dataset subdirs under <automl>/ that hold fold result JSON
#         (keeps aggregated_*.csv, batch_manifest.json, etc.). Pass --no-clean-batch to keep them.
# Optional: --clean-external-outputs removes dssp/, propka/, and sasa/ under each descriptor dataset
#         after descriptor JSONs are written (default: keep those helper dirs).
#
# Scenario 4 (predefined_descriptors): skip structure/descriptor calc; AutoML + analysis only.

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

FASTAB_ENV="${FASTAB_ENV:-fastab}"
PY="${PY:-conda run -n ${FASTAB_ENV} python}"

_stage() {
    echo "[fastab] $*" >&2
}

_die() {
    echo "[fastab] ERROR: $*" >&2
    exit 1
}

_py_to_array() {
    local _py_cmd="$1"
    local -n _out=$2
    _out=()
    mapfile -t _out < <(python3 -c 'import shlex,sys
for part in shlex.split(sys.argv[1]):
    print(part)' "$_py_cmd")
    if [[ ${#_out[@]} -eq 0 ]]; then
        _die "PY command is empty"
    fi
}
_py_to_array "$PY" PY_ARR

run_py() {
    "${PY_ARR[@]}" "$@"
}

usage() {
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    echo "Options:"
    echo "  --no-clean-batch          Keep per-dataset JSON subdirs under <result>/automl after analysis"
    echo "  --clean-external-outputs  After descriptors, remove dssp/, propka/, and sasa/ under each dataset"
}

read_generic() {
    run_py -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg[sys.argv[2]])' "$GENERIC_CONFIG" "$1"
}

read_generic_nested() {
    run_py -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg[sys.argv[2]][sys.argv[3]])' "$GENERIC_CONFIG" "$1" "$2"
}

read_generic_nested_optional() {
    run_py -c 'import yaml,sys
cfg=yaml.safe_load(open(sys.argv[1]))
block=cfg.get(sys.argv[2]) or {}
v=block.get(sys.argv[3])
print(sys.argv[4] if v is None else v)' "$GENERIC_CONFIG" "$1" "$2" "$3"
}

read_generic_optional() {
    run_py -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); v=cfg.get(sys.argv[2]); print("" if v is None else v)' "$GENERIC_CONFIG" "$1"
}

read_generic_has_key() {
    run_py -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(sys.argv[2] in cfg)' "$GENERIC_CONFIG" "$1"
}

reference_json_from_run_config() {
    run_py -c '
import sys
from pathlib import Path
import yaml

repo_root = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(open(sys.argv[2]))

def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path

for _key, block in cfg.items():
    if _key == "batch_result_root" or not isinstance(block, dict):
        continue
    dev = block.get("developability_results_path")
    if not dev:
        continue
    dev_dir = _resolve(dev)
    jsons = sorted(dev_dir.glob("*.json"))
    if jsons:
        print(jsons[0])
        raise SystemExit(0)
    results_dir = dev_dir / "results"
    if results_dir.is_dir():
        jsons = sorted(results_dir.glob("*.json"))
        if jsons:
            print(jsons[0])
            raise SystemExit(0)
    features_csv = block.get("features_csv_path")
    if features_csv:
        csv_path = _resolve(features_csv)
        if csv_path.is_file():
            print(csv_path)
            raise SystemExit(0)
    for csv_name in ("features.csv",):
        csv_path = dev_dir / csv_name
        if csv_path.is_file():
            print(csv_path)
            raise SystemExit(0)
raise SystemExit(
    "No descriptor reference found in run config developability_results_path entries "
    "(expected *.json, results/*.json, or features.csv)"
)
' "$REPO_ROOT" "$RUN_CONFIG"
}

# batch_result_root from the generated run config (repo-relative or absolute).
read_run_config_batch_root() {
    run_py -c '
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

# Resolve structure folder + CSV pairs from run config (scenarios 1 and 2).
descriptor_jobs_from_run_config() {
    run_py -c '
import sys
from pathlib import Path
import yaml

repo_root = Path(sys.argv[1]).resolve()
run_config = yaml.safe_load(open(sys.argv[2]))
result_folder = sys.argv[3]

for key, block in run_config.items():
    if key == "batch_result_root" or not isinstance(block, dict):
        continue
    struct_rel = block.get("structure_dir") or f"{result_folder}/structures/{key}"
    struct_dir = (repo_root / struct_rel).resolve()
    if not struct_dir.is_dir():
        raise SystemExit(f"Structure folder not found for {key!r}: {struct_dir}")
    csv_rel = block.get("path")
    if csv_rel:
        csv_path = (repo_root / csv_rel).resolve()
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found for {key!r}: {csv_path}")
        print(f"{struct_dir}\t{csv_path}")
    else:
        print(f"{struct_dir}\t")
' "$REPO_ROOT" "$RUN_CONFIG" "$RESULT_FOLDER"
}

structure_csvs_from_run_config() {
    run_py -c '
import sys
from pathlib import Path
import yaml

repo_root = Path(sys.argv[1]).resolve()
run_config = yaml.safe_load(open(sys.argv[2]))
seen: set[str] = set()
for _key, block in run_config.items():
    if _key == "batch_result_root" or not isinstance(block, dict):
        continue
    rel = block.get("path")
    if not rel:
        continue
    csv_path = (repo_root / rel).resolve()
    key = str(csv_path)
    if key in seen:
        continue
    seen.add(key)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found for structure prediction: {csv_path}")
    print(csv_path)
' "$REPO_ROOT" "$RUN_CONFIG"
}

NO_CLEAN_BATCH="${FASTAB_NO_CLEAN_BATCH:-0}"
CLEAN_EXTERNAL_OUTPUTS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-clean-batch)
            NO_CLEAN_BATCH=1
            shift
            ;;
        --clean-external-outputs)
            CLEAN_EXTERNAL_OUTPUTS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            _die "Unknown option: $1 (see --help)"
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    usage >&2
    _die "Missing config YAML path"
fi

GENERIC_CONFIG="$1"
if [[ ! -f "$GENERIC_CONFIG" ]]; then
    _die "Config not found: $GENERIC_CONFIG"
fi

RESULT_FOLDER="$(read_generic result_folder)"
if [[ -z "$RESULT_FOLDER" ]]; then
    _die "result_folder is required in $GENERIC_CONFIG"
fi
RESULT_ROOT="$REPO_ROOT/$RESULT_FOLDER"
if [[ -d "$RESULT_ROOT" ]]; then
    _die "result_folder already exists: $RESULT_ROOT (remove or rename it before re-running)"
fi

# ---------------------------------------------------------------------------
# Step 1: generic config -> run config
# ---------------------------------------------------------------------------

_stage "Preparing run config from $GENERIC_CONFIG"
RUN_CONFIG="$(run_py "$PREPARE_SCRIPT" "$GENERIC_CONFIG" --repo-root "$REPO_ROOT")"
_stage "Run config: $RUN_CONFIG"

CALCULATE_DESCRIPTORS="$(run_py -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('calculate_descriptors', True))" "$GENERIC_CONFIG")"
SKIP_AUTOML="$(run_py -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('automl', True) is False)" "$GENERIC_CONFIG")"
USE_EXTERNAL_DESCRIPTORS=0
if [[ "$(read_generic_has_key predefined_descriptors)" == "True" ]] \
    || [[ "$(read_generic_has_key csv_features)" == "True" ]]; then
    USE_EXTERNAL_DESCRIPTORS=1
fi
N_CPU="$(read_generic_optional n_cpu)"

PARALLEL_JOBS="${N_CPU:-$(nproc)}"

run_automl_pipeline() {
    local -a extra=()
    if [[ -n "$N_CPU" ]]; then
        extra+=(--parallel-jobs "$N_CPU")
    fi
    _stage "AutoML (run config: $RUN_CONFIG)"
    bash "$AUTOML_SCRIPT" --config "$RUN_CONFIG" "${extra[@]}"
}

run_hyperparameter_tuning() {
    local batch_root="$1"
    local analysis_out="$2"
    local summary_csv="$analysis_out/best_metrics_summary.csv"
    local manifest="$batch_root/batch_manifest.json"

    if [[ ! -f "$summary_csv" ]]; then
        _die "Missing $summary_csv (run analysis before hyperparameter tuning)"
    fi
    if [[ ! -f "$manifest" ]]; then
        _die "Missing batch manifest: $manifest"
    fi

    shopt -s nullglob
    local -a agg_files=("$batch_root"/aggregated_*.csv)
    shopt -u nullglob
    if [[ ${#agg_files[@]} -eq 0 ]]; then
        _die "No aggregated_*.csv under $batch_root for hyperparameter tuning"
    fi

    local models_root
    models_root="$batch_root/tuned_models"

    _stage "Hyperparameter tuning (shortlist -> grid search -> $models_root)"
    (
        cd "$SRC_DIR"
        export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        run_py -m automl.tune_eval_hyperparameters \
            --batch-root "$batch_root" \
            --best-metrics-summary "$summary_csv" \
            --aggregated-glob "$batch_root/aggregated_*.csv" \
            --out-dir "$analysis_out" \
            --manifest "$manifest" \
            --models-root "$models_root" \
            --clean-folds
    )
}

# Remove direct children of batch_root that contain fold-result *.json (not batch_manifest.json).
cleanup_batch_json_subdirs() {
    local batch_root="$1"
    if [[ "${NO_CLEAN_BATCH:-${FASTAB_NO_CLEAN_BATCH:-0}}" == "1" ]]; then
        _stage "Skipping AutoML batch JSON subdir cleanup (--no-clean-batch / FASTAB_NO_CLEAN_BATCH=1)"
        return 0
    fi
    if [[ ! -d "$batch_root" ]]; then
        return 0
    fi

    local removed=0
    local d
    for d in "$batch_root"/*/; do
        [[ -d "$d" ]] || continue
        shopt -s nullglob
        local -a jsons=("$d"*.json)
        shopt -u nullglob
        if [[ ${#jsons[@]} -eq 0 ]]; then
            continue
        fi
        _stage "Removing batch result JSON dir: ${d%/}"
        rm -rf "$d"
        removed=$((removed + 1))
    done
    if [[ "$removed" -gt 0 ]]; then
        _stage "Cleaned $removed AutoML batch subdir(s) under $batch_root (kept aggregated CSVs and manifest)"
    else
        _stage "No JSON subdirs to clean under $batch_root"
    fi
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
    if [[ "${USE_EXTERNAL_DESCRIPTORS:-0}" -eq 1 ]]; then
        ref_json="$(reference_json_from_run_config)"
    else
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
    fi

    mkdir -p "$analysis_out"
    _stage "Analysis (${#agg_files[@]} aggregated CSVs -> $analysis_out)"
    (
        cd "$SRC_DIR"
        export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        run_py -m analysis.analyze_results \
            "${agg_files[@]}" \
            --out-dir "$analysis_out" \
            --reference-json "$ref_json" \
            --stability-seed 42 \
            --no-plots
    )
}

# ---------------------------------------------------------------------------
# Scenario 4: predefined_descriptors (CSV or JSON); AutoML + analysis only
# ---------------------------------------------------------------------------

if [[ "$USE_EXTERNAL_DESCRIPTORS" -eq 1 ]]; then
    _stage "Scenario 4: predefined descriptors (skipping structure prediction and descriptor calculation)"
    if [[ "$SKIP_AUTOML" == "True" ]]; then
        _die "external descriptor config with automl: false leaves nothing to run"
    fi
    run_automl_pipeline
    BATCH_ROOT="$(read_run_config_batch_root)"
    if [[ -z "$BATCH_ROOT" ]]; then
        BATCH_ROOT="$REPO_ROOT/runs/batch_$(basename "$RUN_CONFIG" .yaml)"
    fi
    run_analysis "$BATCH_ROOT" "$REPO_ROOT/$RESULT_FOLDER/analysis_results"
    run_hyperparameter_tuning "$BATCH_ROOT" "$REPO_ROOT/$RESULT_FOLDER/analysis_results"
    cleanup_batch_json_subdirs "$BATCH_ROOT"
    _stage "Done. Run config: $RUN_CONFIG"
    exit 0
fi

if [[ "$CALCULATE_DESCRIPTORS" == "False" ]]; then
    _die "calculate_descriptors: false requires predefined_descriptors (scenario 4)"
fi

# ---------------------------------------------------------------------------
# Scenarios 1–3: predict structures + descriptors (or existing structures + descriptors)
# ---------------------------------------------------------------------------

INPUT_CSVS="$(read_generic_optional input_csvs_folder)"
if [[ -n "$INPUT_CSVS" ]]; then
    INPUT_CSVS="$REPO_ROOT/$INPUT_CSVS"
fi
INPUT_STRUCTURES_FOLDER="$(read_generic_optional input_structures_folder)"
STRUCTURES_ONLY_CFG=0
if [[ -z "$INPUT_CSVS" && -n "$INPUT_STRUCTURES_FOLDER" ]]; then
    STRUCTURES_ONLY_CFG=1
    SKIP_AUTOML="True"
fi
USE_EXISTING_STRUCTURES=0
if [[ -n "$INPUT_STRUCTURES_FOLDER" ]]; then
    USE_EXISTING_STRUCTURES=1
fi

DESCRIPTORS_ROOT="$REPO_ROOT/$RESULT_FOLDER/descriptors"
mkdir -p "$DESCRIPTORS_ROOT"

if [[ "$USE_EXISTING_STRUCTURES" -eq 1 ]]; then
    INPUT_STRUCTURES_ROOT="$REPO_ROOT/$INPUT_STRUCTURES_FOLDER"
    STRUCTURES_ROOT="$INPUT_STRUCTURES_ROOT"
    if [[ ! -d "$INPUT_STRUCTURES_ROOT" ]]; then
        _die "input_structures_folder not found: $INPUT_STRUCTURES_ROOT"
    fi
    if [[ "$STRUCTURES_ONLY_CFG" -eq 1 ]]; then
        _stage "Scenario 2a: descriptors only on existing structures from $INPUT_STRUCTURES_ROOT"
    else
        _stage "Scenario 2: using existing structures from $INPUT_STRUCTURES_ROOT (skipping prediction)"
    fi
else
    STRUCTURES_ROOT="$REPO_ROOT/$RESULT_FOLDER/structures"
    MODEL="$(read_generic_nested structure_prediction model)"
    RUNS="$(read_generic_nested structure_prediction runs)"
    mkdir -p "$STRUCTURES_ROOT"
fi

run_structures_processing() {
    local -a structure_dirs=("$@")
    local renumber_imgt minimize

    renumber_imgt="$(read_generic_nested_optional structures_processing renumber_imgt false)"
    minimize="$(read_generic_nested_optional structures_processing minimize false)"

    if [[ "$renumber_imgt" != "True" && "$minimize" != "True" ]]; then
        return 0
    fi

    local dir
    for dir in "${structure_dirs[@]}"; do
        if [[ "$minimize" == "True" ]]; then
            _stage "Minimize (in place): $dir"
            bash "$PREDICT_SCRIPT" --minimization-only --in-place --structures-dir "$dir" || _die "Minimize failed: $dir"
        fi
        if [[ "$renumber_imgt" == "True" ]]; then
            _stage "IMGT renumber (in place): $dir"
            bash "$PREDICT_SCRIPT" --renumber-only --in-place --structures-dir "$dir" || _die "Renumber failed: $dir"
        fi
    done
}

run_structure_prediction() {
    local device
    device="$(read_generic_nested structure_prediction device)"
    if [[ "$device" =~ ^[0-9]+$ ]]; then
        device="cuda:$device"
    fi

    export ABB3_DEVICE="$device"
    export ABB2_DEVICE="$device"

    local batch_size
    batch_size="$(read_generic_nested_optional structure_prediction batch_size 4)"
    export ABB3_BATCH_SIZE="$batch_size"
    export ABB2_BATCH_SIZE="$batch_size"

    local skip_existing
    skip_existing="$(read_generic_nested_optional structure_prediction skip_existing false)"

    local -a args=(
        --output-root "$STRUCTURES_ROOT"
        --runs "$RUNS"
    )

    local -a csv_paths=()
    mapfile -t csv_paths < <(structure_csvs_from_run_config)
    if [[ ${#csv_paths[@]} -eq 0 ]]; then
        _die "No CSV datasets in run config for structure prediction: $RUN_CONFIG"
    fi
    local csv_path
    for csv_path in "${csv_paths[@]}"; do
        args+=(--csv "$csv_path")
    done

    if [[ "$skip_existing" == "True" ]]; then
        args+=(--skip-existing)
    fi

    if [[ "$MODEL" == "abb3" ]]; then
        args=(--abb3 "${args[@]}")
    else
        args=(--abb2 "${args[@]}")
    fi

    _stage "Structure prediction ($MODEL, device=$device, batch_size=$batch_size, runs=$RUNS)"
    _stage "  CSVs (${#csv_paths[@]}): $(printf '%s ' "${csv_paths[@]##*/}")"
    _stage "  Structures: $STRUCTURES_ROOT"

    bash "$PREDICT_SCRIPT" "${args[@]}"
}

run_descriptor_calculation() {
    local -a structure_dirs=()
    local -a csv_paths=()
    local struct_dir csv_path

    while IFS=$'\t' read -r struct_dir csv_path; do
        [[ -z "$struct_dir" ]] && continue
        structure_dirs+=("$struct_dir")
        csv_paths+=("$csv_path")
    done < <(descriptor_jobs_from_run_config)

    if [[ ${#structure_dirs[@]} -eq 0 ]]; then
        _die "No descriptor jobs resolved from run config: $RUN_CONFIG"
    fi

    if [[ "$USE_EXISTING_STRUCTURES" -eq 1 ]]; then
        run_structures_processing "${structure_dirs[@]}"
    fi

    local -a desc_extra=()
    if [[ "$CLEAN_EXTERNAL_OUTPUTS" -eq 1 ]]; then
        desc_extra+=(--clean-external-outputs)
        _stage "  Descriptor helper cleanup: dssp/, propka/, sasa/ removed after each dataset"
    fi

    local i
    for i in "${!structure_dirs[@]}"; do
        struct_dir="${structure_dirs[$i]}"
        csv_path="${csv_paths[$i]}"
        local -a desc_cmd=(
            bash "$DESCRIPTORS_SCRIPT"
            --output-dir "$DESCRIPTORS_ROOT"
        )
        if [[ -n "$csv_path" ]]; then
            _stage "  [descriptors] $struct_dir  <-  $(basename "$csv_path")"
            desc_cmd+=(--names-from-csv "$csv_path")
        else
            _stage "  [descriptors] $struct_dir"
        fi
        "${desc_cmd[@]}" \
            "${desc_extra[@]}" \
            "$struct_dir" \
            "$PARALLEL_JOBS"
    done

    _stage "Descriptor calculation done (${#structure_dirs[@]} structure folder(s) -> $DESCRIPTORS_ROOT)"
}

if [[ "$USE_EXISTING_STRUCTURES" -eq 0 ]]; then
    run_structure_prediction
fi
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
run_hyperparameter_tuning "$BATCH_ROOT" "$REPO_ROOT/$RESULT_FOLDER/analysis_results"
cleanup_batch_json_subdirs "$BATCH_ROOT"

_stage "Done."
_stage "  Run config:   $RUN_CONFIG"
_stage "  Structures:   $STRUCTURES_ROOT"
_stage "  Descriptors:  $DESCRIPTORS_ROOT"
_stage "  AutoML batch: $BATCH_ROOT"
_stage "  Analysis:     $REPO_ROOT/$RESULT_FOLDER/analysis_results"
_stage "  Tuned models: $BATCH_ROOT/tuned_models"
