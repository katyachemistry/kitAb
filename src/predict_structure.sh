#!/usr/bin/env bash
# Predict structures from CSVs (name, heavy, light); output structures/<stem>_abb3_<n>, _abb2_<n>, or _flashabb_<n>.
# Usage: predict_structure.sh [--abb3|--abb2|--flashabb] [--data-dir DIR] [--output-root DIR] [--runs N]
#   [--skip-existing] [--no-renumber] [--no-minimize] [--minimize-jobs N]
#   [--renumber-only|--minimization-only] [--structures-dir DIR] [--allow-partial-domain]

set -euo pipefail

# tmux keeps env vars from when the tmux *server* was first started. Older FASTAb
# setups exported PYTHON="conda run python", which breaks conda run even in new sessions.
unset PYTHON ABB3_PYTHON 2>/dev/null || true
CONDA_RUN_BIN="${CONDA_EXE:-$(type -P conda || true)}"
CONDA_RUN_BIN="${CONDA_RUN_BIN:-conda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$REPO_ROOT/fastab.local.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/fastab.local.env"
fi
FASTAB_ENV_ABB2="${FASTAB_ENV_ABB2:-fastab-abb2}"
FASTAB_ENV_ABB3="${FASTAB_ENV_ABB3:-fastab-abb3}"
FASTAB_ENV_FLASHABB="${FASTAB_ENV_FLASHABB:-fastab-flashabb}"
ABB3_PYTHON_BIN="$("$CONDA_RUN_BIN" run -n "$FASTAB_ENV_ABB3" python -c 'import sys; print(sys.executable)')"
ABB2_PYTHON_BIN="$("$CONDA_RUN_BIN" run -n "$FASTAB_ENV_ABB2" python -c 'import sys; print(sys.executable)')"
# FlashABB env resolution is fault-tolerant: if the env is absent and --flashabb is not
# passed, the script continues. The binary is checked only when backend=flashabb.
FLASHABB_PYTHON_BIN="$("$CONDA_RUN_BIN" run -n "$FASTAB_ENV_FLASHABB" python -c 'import sys; print(sys.executable)' 2>/dev/null || echo "")"
ABB2_ENV_BIN_DIR="$(dirname -- "$ABB2_PYTHON_BIN")"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

FASTAB_ENV_ABB2="${FASTAB_ENV_ABB2:-fastab-abb2}"
FASTAB_ENV_ABB3="${FASTAB_ENV_ABB3:-fastab-abb3}"
FASTAB_ENV_FLASHABB="${FASTAB_ENV_FLASHABB:-fastab-flashabb}"
ABB3_DIR="${ABB3_DIR:-$REPO_ROOT/vendor/abodybuilder3}"
FLASHABB_DIR="${FLASHABB_DIR:-$REPO_ROOT/FlashABB}"

ABB3_CHECKPOINT="${ABB3_CHECKPOINT:-$ABB3_DIR/output/plddt-loss/best_second_stage.ckpt}"
ABB3_DEVICE="${ABB3_DEVICE:-cuda:0}"
ABB3_BATCH_SIZE="${ABB3_BATCH_SIZE:-4}"
ABB2_DEVICE="${ABB2_DEVICE:-$ABB3_DEVICE}"
ABB2_BATCH_SIZE="${ABB2_BATCH_SIZE:-$ABB3_BATCH_SIZE}"
FLASHABB_DEVICE="${FLASHABB_DEVICE:-$ABB3_DEVICE}"
FLASHABB_BATCH_SIZE="${FLASHABB_BATCH_SIZE:-50}"
ABB3_PYTHON_CMD=("$ABB3_PYTHON_BIN")
ABB2_PYTHON_CMD=("$ABB2_PYTHON_BIN")
FLASHABB_PYTHON_CMD=("$FLASHABB_PYTHON_BIN")
# Minimization always uses the Python from the backend's own env (all have openmm).
# Defaults to ABB3; overridden to FLASHABB_PYTHON_CMD when backend is flashabb.
MINIMIZE_PYTHON_CMD=("${ABB3_PYTHON_CMD[@]}")

BACKEND=abb3
HAVE_ABB3=0
HAVE_ABB2=0
HAVE_FLASHABB=0
RUNS=1
SKIP_EXISTING=0
DATA_DIR_CLI=""
CSV_FILES=()
OUT_DIR_CLI=""
RENUMBER=1
MINIMIZE=1
MINIMIZATION_ONLY=0
RENUMBER_ONLY=0
STRUCTURES_DIRS=()
ALLOW_PARTIAL_DOMAIN=0
MINIMIZE_JOBS=8
IN_PLACE=0

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
        --flashabb)
            BACKEND=flashabb
            HAVE_FLASHABB=1
            shift
            ;;
        --runs)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --runs" >&2
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
                echo "missing value for --data-dir" >&2
                exit 1
            fi
            DATA_DIR_CLI="$2"
            shift 2
            ;;
        --csv)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --csv" >&2
                exit 1
            fi
            CSV_FILES+=("$2")
            shift 2
            ;;
        --output-root)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --output-root" >&2
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
        --renumber-only)
            RENUMBER_ONLY=1
            shift
            ;;
        --allow-partial-domain)
            ALLOW_PARTIAL_DOMAIN=1
            shift
            ;;
        --structures-dir)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --structures-dir" >&2
                exit 1
            fi
            STRUCTURES_DIRS+=("$2")
            shift 2
            ;;
        --minimize-jobs)
            if [[ $# -lt 2 ]]; then
                echo "missing value for --minimize-jobs" >&2
                exit 1
            fi
            MINIMIZE_JOBS="$2"
            shift 2
            ;;
        --in-place)
            IN_PLACE=1
            shift
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: $1 (try --help)" >&2
            exit 1
            ;;
    esac
done

if (( HAVE_ABB3 + HAVE_ABB2 + HAVE_FLASHABB > 1 )); then
    echo "Use only one of --abb3, --abb2, or --flashabb" >&2
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

if [[ "$MINIMIZATION_ONLY" -eq 1 && "$RENUMBER_ONLY" -eq 1 ]]; then
    echo "pick --minimization-only or --renumber-only, not both" >&2
    exit 1
fi

if [[ "$MINIMIZATION_ONLY" -eq 1 && ${#STRUCTURES_DIRS[@]} -eq 0 ]]; then
    echo "--minimization-only needs at least one --structures-dir" >&2
    exit 1
fi

if [[ "$RENUMBER_ONLY" -eq 1 && ${#STRUCTURES_DIRS[@]} -eq 0 ]]; then
    echo "--renumber-only needs at least one --structures-dir" >&2
    exit 1
fi

STRUCTURE_DIR="$SCRIPT_DIR/structure"

run_renumber() {
    local input_dir="$1"
    local inplace="${2:-0}"

    if [[ ! -d "$input_dir" ]]; then
        echo "  skip renumber, no dir: $input_dir" >&2
        return 1
    fi

    local parent base out_dir
    parent="$(cd "$(dirname "$input_dir")" && pwd)"
    base="$(basename "$input_dir")"
    if [[ "$inplace" -eq 1 ]]; then
        out_dir="$input_dir"
    else
        out_dir="${parent}/${base}_imgt"
    fi

    echo ""
    if [[ "$inplace" -eq 1 ]]; then
        echo "IMGT renumber (in place): $input_dir"
    else
        echo "IMGT renumber: $input_dir -> $out_dir"
    fi
    local extra=(--out-dir "$out_dir" --overwrite)
    if [[ "$ALLOW_PARTIAL_DOMAIN" -eq 1 ]]; then
        extra+=(--allow-partial-domain)
    fi

    PATH="$ABB2_ENV_BIN_DIR:${PATH}" \
        "${ABB2_PYTHON_CMD[@]}" "$STRUCTURE_DIR/postprocess_structures.py" renumber "$input_dir" "${extra[@]}"
}

run_minimization() {
    local input_dir="$1"
    local skip_flag="${2:-}"
    local inplace="${3:-0}"

    if [[ ! -d "$input_dir" ]]; then
        echo "  [minimize] Skipping missing directory: $input_dir" >&2
        return 1
    fi

    local parent base output_dir
    parent="$(cd "$(dirname "$input_dir")" && pwd)"
    base="$(basename "$input_dir")"
    if [[ "$inplace" -eq 1 ]]; then
        output_dir="$input_dir"
    else
        if [[ "$base" == *_imgt ]]; then
            base="${base%_imgt}"
        fi
        output_dir="${parent}/${base}_minimized"
    fi

    echo ""
    if [[ "$inplace" -eq 1 ]]; then
        echo "--- Minimizing (in place): $input_dir ---"
    else
        echo "--- Minimizing: $input_dir -> $output_dir ---"
    fi

    local skip_args=()
    if [[ "$skip_flag" == "--skip-existing" ]]; then
        skip_args=(--skip-existing)
    fi

    "${MINIMIZE_PYTHON_CMD[@]}" "$STRUCTURE_DIR/postprocess_structures.py" minimize \
        --input-dir  "$input_dir" \
        --output-dir "$output_dir" \
        --jobs       "$MINIMIZE_JOBS" \
        "${skip_args[@]}"
}

run_postprocess_predict_dirs() {
    local backend_tag="$1"

    for csv in "${csv_files[@]}"; do
        local stem predict_dir
        stem="$(basename "$csv" .csv)"
        for (( run=1; run<=RUNS; run++ )); do
            predict_dir="$OUT_DIR/${stem}_${backend_tag}_${run}"

            if [[ ! -d "$predict_dir" ]]; then
                echo "  [post-process] Skipping missing: $predict_dir" >&2
                continue
            fi

            if [[ "$MINIMIZE" -eq 1 ]]; then
                # ABB3 raw outputs can contain locally strained peptide bonds that
                # OpenMM fixes cleanly before IMGT insertion-code renumbering.
                run_minimization "$predict_dir" "" 1
            fi

            if [[ "$RENUMBER" -eq 1 ]]; then
                run_renumber "$predict_dir" 1 || exit 1
            fi
        done
    done
}

if [[ "$RENUMBER_ONLY" -eq 1 ]]; then
    for dir in "${STRUCTURES_DIRS[@]}"; do
        abs_dir="$(cd "$dir" && pwd)"
        run_renumber "$abs_dir" "$IN_PLACE" || exit 1
    done
    echo ""
    echo "done"
    exit 0
fi

if [[ "$MINIMIZATION_ONLY" -eq 1 ]]; then
    SKIP_FLAG=""
    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
        SKIP_FLAG="--skip-existing"
    fi

    for dir in "${STRUCTURES_DIRS[@]}"; do
        abs_dir="$(cd "$dir" && pwd)"
        run_minimization "$abs_dir" "$SKIP_FLAG" "$IN_PLACE"
    done

    echo ""
    echo "done"
    exit 0
fi

if [[ -n "$DATA_DIR_CLI" && ${#CSV_FILES[@]} -gt 0 ]]; then
    echo "Use either --data-dir or --csv, not both" >&2
    exit 1
fi

if [[ -n "$DATA_DIR_CLI" ]]; then
    if [[ ! -d "$DATA_DIR_CLI" ]]; then
        echo "Not a directory: $DATA_DIR_CLI" >&2
        exit 1
    fi
    DATA_DIR="$(cd "$DATA_DIR_CLI" && pwd)"
elif [[ ${#CSV_FILES[@]} -gt 0 ]]; then
    DATA_DIR=""
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
csv_files=()
if [[ -n "$DATA_DIR" ]]; then
    csv_files=("$DATA_DIR"/*.csv)
    if [[ ${#csv_files[@]} -eq 0 ]]; then
        echo "No CSV files found in $DATA_DIR" >&2
        exit 1
    fi
elif [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "No CSV files provided (use --data-dir or one or more --csv paths)" >&2
    exit 1
else
    for _csv_f in "${CSV_FILES[@]}"; do
        [[ -n "$_csv_f" ]] && csv_files+=("$_csv_f")
    done
fi

# When using FlashABB, its own env has OpenMM for minimization.
if [[ "$BACKEND" == "flashabb" ]]; then
    if [[ -z "$FLASHABB_PYTHON_BIN" ]]; then
        echo "FlashABB conda env '$FASTAB_ENV_FLASHABB' not found." >&2
        echo "Run ./install.sh (or ./install.sh --no-abb2) to create it." >&2
        exit 1
    fi
    MINIMIZE_PYTHON_CMD=("${FLASHABB_PYTHON_CMD[@]}")
fi

if [[ "$BACKEND" == "abb3" ]]; then
    if [[ ! -f "$ABB3_CHECKPOINT" ]]; then
        echo "Checkpoint not found: $ABB3_CHECKPOINT (set ABB3_CHECKPOINT)" >&2
        exit 1
    fi

    batch_csv_args=()
    if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
        for csv_path in "${CSV_FILES[@]}"; do
            [[ -z "$csv_path" ]] && continue
            batch_csv_args+=(--csv "$csv_path")
        done
    fi

    echo "ABB3 Python: ${ABB3_PYTHON_CMD[*]}"
    "${ABB3_PYTHON_CMD[@]}" "$STRUCTURE_DIR/run_abb_batch_from_csv.py" abb3 \
        ${DATA_DIR:+--data-dir "$DATA_DIR"} \
        "${batch_csv_args[@]}" \
        --output-root "$OUT_DIR" \
        --runs "$RUNS" \
        --checkpoint "$ABB3_CHECKPOINT" \
        --device "$ABB3_DEVICE" \
        --batch-size "$ABB3_BATCH_SIZE" \
        "${SKIP_ARGS[@]}"

    if [[ "$RENUMBER" -eq 1 || "$MINIMIZE" -eq 1 ]]; then
        run_postprocess_predict_dirs abb3
    fi

elif [[ "$BACKEND" == "flashabb" ]]; then
    batch_csv_args=()
    if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
        for csv_path in "${CSV_FILES[@]}"; do
            [[ -z "$csv_path" ]] && continue
            batch_csv_args+=(--csv "$csv_path")
        done
    fi

    echo "FlashABB Python: ${FLASHABB_PYTHON_CMD[*]}"
    FLASHABB_DIR="$FLASHABB_DIR" \
        "${FLASHABB_PYTHON_CMD[@]}" "$STRUCTURE_DIR/run_abb_batch_from_csv.py" flashabb \
        ${DATA_DIR:+--data-dir "$DATA_DIR"} \
        "${batch_csv_args[@]}" \
        --output-root "$OUT_DIR" \
        --runs "$RUNS" \
        --device "$FLASHABB_DEVICE" \
        --batch-size "$FLASHABB_BATCH_SIZE" \
        "${SKIP_ARGS[@]}"

    if [[ "$RENUMBER" -eq 1 || "$MINIMIZE" -eq 1 ]]; then
        run_postprocess_predict_dirs flashabb
    fi

else
    batch_csv_args=()
    if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
        for csv_path in "${CSV_FILES[@]}"; do
            [[ -z "$csv_path" ]] && continue
            batch_csv_args+=(--csv "$csv_path")
        done
    fi

    "${ABB2_PYTHON_CMD[@]}" "$STRUCTURE_DIR/run_abb_batch_from_csv.py" abb2 \
        ${DATA_DIR:+--data-dir "$DATA_DIR"} \
        "${batch_csv_args[@]}" \
        --output-root "$OUT_DIR" \
        --runs "$RUNS" \
        --device "$ABB2_DEVICE" \
        --batch-size "$ABB2_BATCH_SIZE" \
        "${SKIP_ARGS[@]}"
fi

echo ""
echo "All datasets processed. Structures under $OUT_DIR"
