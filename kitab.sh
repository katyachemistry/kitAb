#!/usr/bin/env bash
# kitAb entrypoint.
#
#   ./kitab.sh validate CONFIG.yaml
#   ./kitab.sh CONFIG.yaml [--resume] [--techniques ...] ...
#   ./kitab.sh resume RUN_DIR

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$REPO_ROOT/src"

if [[ -f "$REPO_ROOT/kitab.local.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/kitab.local.env"
fi

KITAB_ENV="${KITAB_ENV:-kitab}"
PY="${PY:-conda run --no-capture-output -n ${KITAB_ENV} python}"

_die() {
    echo "[kitab] ERROR: $*" >&2
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

usage() {
    cat <<EOF
Usage:
  ./kitab.sh CONFIG.yaml [options]
  ./kitab.sh validate|run|resume ...

Options for CONFIG.yaml mode:
  --resume              Continue an interrupted run
  --disable-automl      Skip AutoML even if config enables it
  --enable-automl       Force-enable AutoML
  --techniques LIST     Comma-separated: elasticnet,intercorr_svm,sfs_svm,sfs_knn
  --cv-mode MODE        nested (default) or flat
  --no-final-model      Compare techniques without saving estimator.joblib
  --output-dir DIR      Override run.output_dir
  --cpus N              Override CPU count (AutoML worker pool)
  --device DEV          Override structure_prediction.device
  -h, --help
EOF
}

export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# If first arg is a subcommand, delegate fully.
if [[ "${1:-}" =~ ^(validate|run|resume)$ ]]; then
    exec "${PY_ARR[@]}" -m kitab "$@"
fi

CONFIG=""
RESUME=0
DISABLE_AUTOML=0
ENABLE_AUTOML=0
TECHNIQUES=""
CV_MODE=""
NO_FINAL_MODEL=0
OUTPUT_DIR=""
CPUS=""
DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --disable-automl)
            DISABLE_AUTOML=1
            shift
            ;;
        --enable-automl)
            ENABLE_AUTOML=1
            shift
            ;;
        --techniques)
            TECHNIQUES="${2:?}"
            shift 2
            ;;
        --cv-mode)
            CV_MODE="${2:?}"
            shift 2
            ;;
        --no-final-model)
            NO_FINAL_MODEL=1
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?}"
            shift 2
            ;;
        --cpus)
            CPUS="${2:?}"
            shift 2
            ;;
        --device)
            DEVICE="${2:?}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            _die "Unknown option: $1 (see --help)"
            ;;
        *)
            if [[ -n "$CONFIG" ]]; then
                _die "Unexpected argument: $1"
            fi
            CONFIG="$1"
            shift
            ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    usage >&2
    _die "Missing config YAML path"
fi
if [[ ! -f "$CONFIG" ]]; then
    _die "Config not found: $CONFIG"
fi

CMD=("${PY_ARR[@]}" -m kitab run "$CONFIG")
if [[ "$RESUME" -eq 1 ]]; then
    CMD+=(--resume)
fi
if [[ "$DISABLE_AUTOML" -eq 1 ]]; then
    CMD+=(--disable-automl)
fi
if [[ "$ENABLE_AUTOML" -eq 1 ]]; then
    CMD+=(--enable-automl)
fi
if [[ -n "$TECHNIQUES" ]]; then
    CMD+=(--techniques "$TECHNIQUES")
fi
if [[ -n "$CV_MODE" ]]; then
    CMD+=(--cv-mode "$CV_MODE")
fi
if [[ "$NO_FINAL_MODEL" -eq 1 ]]; then
    CMD+=(--no-final-model)
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$CPUS" ]]; then
    CMD+=(--cpus "$CPUS")
fi
if [[ -n "$DEVICE" ]]; then
    CMD+=(--device "$DEVICE")
fi

echo "[kitab] Invoking: ${CMD[*]}" >&2
exec "${CMD[@]}"
