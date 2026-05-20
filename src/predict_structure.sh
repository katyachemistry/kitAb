#!/usr/bin/env bash
# Run AbodyBuilder3 or ABodyBuilder2 on every top-level CSV in data/.
# Required columns: name, heavy, light -> <name>.pdb (| in name -> '_').
# Output: structures/<dataset>_abb3_<run>/ or structures/<dataset>_abb2_<run>/ for run in 1..N (--runs).
#
# Usage:
#   ./run_structure_prediction.sh [--abb3|--abb2] [--data-dir DIR] [--output-root DIR]
#       [--runs N] [--skip-existing] [--no-minimize] [--minimize-jobs N]
#   ./run_structure_prediction.sh --minimization-only \
#       --input-dir structures/ds_abb3_1 [--input-dir structures/ds_abb3_2 ...] \
#       [--minimize-jobs N]
#   ABB3_CHECKPOINT=/path/to.ckpt ./run_structure_prediction.sh --abb3 --runs 3
# Default backend: --abb3. Default --runs 1. PYTHON=… for interpreter.
# Default --data-dir: <repo>/data. Default --output-root: <repo>/structures.
#
# Device / batch (ABB3): ABB3_DEVICE (default cuda:1), ABB3_BATCH_SIZE (default 4).
# ABB2 uses the same defaults via ABB2_DEVICE / ABB2_BATCH_SIZE when unset
# (they fall back to ABB3_*). --batch-size is passed through for CLI parity; ABB2
# still predicts one sequence at a time (see run_abb2_batch_from_csv.py).
#
# IMGT renumbering (ANARCI, abb2 conda env):
#   After ABB3, each output folder is renumbered into a sibling <folder>_imgt/
#   by default.  Pass --no-renumber to skip.
#
# Minimization (OpenMM, amber14, same pipeline as ABB2 ImmuneBuilder refinement):
#   After renumbering (or directly after ABB3 if --no-renumber), each folder is
#   minimized into a sibling <folder>_minimized/ by default.
#   Pass --no-minimize to skip.  --minimization-only runs only this step on
#   explicitly provided --input-dir folders (abb2 conda env used).
#   ABB2_PYTHON overrides the python used for both steps (default: conda run -n abb2 python3).
#
# All datasets in data/ (one model load):
#   ./run_structure_prediction.sh --abb3 --runs 5
#   ./run_structure_prediction.sh --abb2 --runs 5
# Or several CSVs explicitly (ABB3 example):
#   python3 structure/run_abb3_batch_from_csv.py --csv data/a.csv --csv data/b.csv --output-root structures

set -euo pipefail

ABB3_CHECKPOINT="${ABB3_CHECKPOINT:-/home/kb/abodybuilder3/output/plddt-loss/best_second_stage.ckpt}"
ABB3_DEVICE="${ABB3_DEVICE:-cuda:1}"
ABB3_BATCH_SIZE="${ABB3_BATCH_SIZE:-4}"
ABB2_DEVICE="${ABB2_DEVICE:-$ABB3_DEVICE}"
ABB2_BATCH_SIZE="${ABB2_BATCH_SIZE:-$ABB3_BATCH_SIZE}"
PYTHON="${PYTHON:-python3}"
# Python used for minimization — must have openmm/pdbfixer/scipy (abb2 env).
ABB2_PYTHON="${ABB2_PYTHON:-conda run -n abb2 python3}"

BACKEND=abb3
HAVE_ABB3=0
HAVE_ABB2=0
RUNS=1
SKIP_EXISTING=0
DATA_DIR_CLI=""
OUT_DIR_CLI=""
RENUMBER=1           # IMGT-renumber after ABB3 by default
MINIMIZE=1           # minimize after ABB3 (after renumbering) by default
MINIMIZATION_ONLY=0  # skip prediction; only minimize --input-dir folders
MINIMIZE_INPUT_DIRS=()
MINIMIZE_JOBS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --abb3)
            BACKEND=abb3
            HAVE_ABB3=1
            shift
            ;;
        --abb2)
            BACKEND=abb2
            HAVE_ABB2=1
            shift
            ;;
        --runs)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --runs" >&2
                exit 1
            fi
            RUNS="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --data-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --data-dir" >&2
                exit 1
            fi
            DATA_DIR_CLI="$2"
            shift 2
            ;;
        --output-root)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --output-root" >&2
                exit 1
            fi
            OUT_DIR_CLI="$2"
            shift 2
            ;;
        --no-renumber)
            RENUMBER=0
            shift
            ;;
        --no-minimize)
            MINIMIZE=0
            shift
            ;;
        --minimization-only)
            MINIMIZATION_ONLY=1
            shift
            ;;
        --input-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --input-dir" >&2
                exit 1
            fi
            MINIMIZE_INPUT_DIRS+=("$2")
            shift 2
            ;;
        --minimize-jobs)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --minimize-jobs" >&2
                exit 1
            fi
            MINIMIZE_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ "$HAVE_ABB3" -eq 1 && "$HAVE_ABB2" -eq 1 ]]; then
    echo "Use only one of --abb3 or --abb2" >&2
    exit 1
fi

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--runs must be a positive integer (got: $RUNS)" >&2
    exit 1
fi

if ! [[ "$MINIMIZE_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--minimize-jobs must be a positive integer (got: $MINIMIZE_JOBS)" >&2
    exit 1
fi

if [[ "$MINIMIZATION_ONLY" -eq 1 && ${#MINIMIZE_INPUT_DIRS[@]} -eq 0 ]]; then
    echo "--minimization-only requires at least one --input-dir DIR" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRUCTURE_DIR="$SCRIPT_DIR/structure"

# ---------------------------------------------------------------------------
# Helper: minimize one folder -> sibling folder with _minimized suffix
# ---------------------------------------------------------------------------

run_minimization() {
    local input_dir="$1"
    local skip_flag="${2:-}"   # "--skip-existing" or ""

    if [[ ! -d "$input_dir" ]]; then
        echo "  [minimize] Skipping missing directory: $input_dir" >&2
        return
    fi

    local parent
    parent="$(cd "$(dirname "$input_dir")" && pwd)"
    local base
    base="$(basename "$input_dir")"
    local output_dir="${parent}/${base}_minimized"

    echo ""
    echo "--- Minimizing: $input_dir -> $output_dir ---"

    local skip_args=()
    if [[ "$skip_flag" == "--skip-existing" ]]; then
        skip_args=(--skip-existing)
    fi

    $ABB2_PYTHON "$STRUCTURE_DIR/minimize_structures_batch.py" \
        --input-dir  "$input_dir" \
        --output-dir "$output_dir" \
        --jobs       "$MINIMIZE_JOBS" \
        "${skip_args[@]}"
}

# ---------------------------------------------------------------------------
# --minimization-only: no prediction, just minimize the given folders
# ---------------------------------------------------------------------------

if [[ "$MINIMIZATION_ONLY" -eq 1 ]]; then
    SKIP_FLAG=""
    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
        SKIP_FLAG="--skip-existing"
    fi

    for dir in "${MINIMIZE_INPUT_DIRS[@]}"; do
        abs_dir="$(cd "$dir" && pwd)"
        run_minimization "$abs_dir" "$SKIP_FLAG"
    done

    echo ""
    echo "Minimization complete."
    exit 0
fi

# ---------------------------------------------------------------------------
# Normal prediction path
# ---------------------------------------------------------------------------

if [[ -n "$DATA_DIR_CLI" ]]; then
    if [[ ! -d "$DATA_DIR_CLI" ]]; then
        echo "Not a directory: $DATA_DIR_CLI" >&2
        exit 1
    fi
    DATA_DIR="$(cd "$DATA_DIR_CLI" && pwd)"
else
    DATA_DIR="$SCRIPT_DIR/data"
fi

if [[ -n "$OUT_DIR_CLI" ]]; then
    mkdir -p "$OUT_DIR_CLI"
    OUT_DIR="$(cd "$OUT_DIR_CLI" && pwd)"
else
    mkdir -p "$SCRIPT_DIR/structures"
    OUT_DIR="$(cd "$SCRIPT_DIR/structures" && pwd)"
fi

SKIP_ARGS=()
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
    SKIP_ARGS=(--skip-existing)
fi

shopt -s nullglob
csv_files=("$DATA_DIR"/*.csv)

if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No CSV files found in $DATA_DIR" >&2
    exit 1
fi

if [[ "$BACKEND" == "abb3" ]]; then
    if [[ ! -f "$ABB3_CHECKPOINT" ]]; then
        echo "Checkpoint not found: $ABB3_CHECKPOINT (set ABB3_CHECKPOINT)" >&2
        exit 1
    fi

    "$PYTHON" "$STRUCTURE_DIR/run_abb3_batch_from_csv.py" \
        --data-dir "$DATA_DIR" \
        --output-root "$OUT_DIR" \
        --runs "$RUNS" \
        --checkpoint "$ABB3_CHECKPOINT" \
        --device "$ABB3_DEVICE" \
        --batch-size "$ABB3_BATCH_SIZE" \
        "${SKIP_ARGS[@]}"

    # Post-processing: IMGT renumber and/or minimize.
    # All intermediate directories are temporary; the final result always lands
    # in stem_abb3_N/ (same name as the raw ABB3 output, overwriting it).
    if [[ "$RENUMBER" -eq 1 || "$MINIMIZE" -eq 1 ]]; then
        minimize_skip_args=()
        [[ "$SKIP_EXISTING" -eq 1 ]] && minimize_skip_args=(--skip-existing)

        for csv in "${csv_files[@]}"; do
            stem="$(basename "$csv" .csv)"
            for (( run=1; run<=RUNS; run++ )); do
                final_dir="$OUT_DIR/${stem}_abb3_${run}"

                if [[ ! -d "$final_dir" ]]; then
                    echo "  [post-process] Skipping missing: $final_dir" >&2
                    continue
                fi

                # Move raw ABB3 output aside so we can write the final result
                # back to the same directory name.
                raw_tmp="${final_dir}_tmp_${$}_raw"
                mv "$final_dir" "$raw_tmp"
                current="$raw_tmp"

                if [[ "$RENUMBER" -eq 1 ]]; then
                    echo ""
                    echo "--- IMGT renumbering: ${stem}_abb3_${run} ---"
                    if [[ "$MINIMIZE" -eq 1 ]]; then
                        # Another step follows; renumber into a second temp dir.
                        imgt_tmp="${final_dir}_tmp_${$}_imgt"
                        $ABB2_PYTHON "$STRUCTURE_DIR/renumber_abb3_imgt.py" \
                            "$current" --out-dir "$imgt_tmp"
                        rm -rf "$current"
                        current="$imgt_tmp"
                    else
                        # Last step; write directly to final_dir.
                        $ABB2_PYTHON "$STRUCTURE_DIR/renumber_abb3_imgt.py" \
                            "$current" --out-dir "$final_dir"
                        rm -rf "$current"
                        current=""
                    fi
                fi

                if [[ "$MINIMIZE" -eq 1 ]]; then
                    echo ""
                    echo "--- Minimizing: ${stem}_abb3_${run} ---"
                    $ABB2_PYTHON "$STRUCTURE_DIR/minimize_structures_batch.py" \
                        --input-dir  "$current" \
                        --output-dir "$final_dir" \
                        --jobs       "$MINIMIZE_JOBS" \
                        "${minimize_skip_args[@]}"
                    rm -rf "$current"
                fi
            done
        done
    fi

else
    export ABB2_DEVICE="$ABB2_DEVICE"
    "$PYTHON" "$STRUCTURE_DIR/run_abb2_batch_from_csv.py" \
        --data-dir "$DATA_DIR" \
        --output-root "$OUT_DIR" \
        --runs "$RUNS" \
        --device "$ABB2_DEVICE" \
        --batch-size "$ABB2_BATCH_SIZE" \
        "${SKIP_ARGS[@]}"
fi

echo ""
echo "All datasets processed. Structures under $OUT_DIR"
