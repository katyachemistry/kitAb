#!/usr/bin/env bash
# kitAb entrypoint.
#
#   ./kitab.sh validate CONFIG.yaml
#   ./kitab.sh CONFIG.yaml [--resume] [--enable-tuning] ...
#   ./kitab.sh resume RUN_DIR
#
# Hyperparameter tuning is OFF by default. Enable with --enable-tuning
# or tuning.enabled: true in the manifest.

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
  --enable-tuning       Opt-in hyperparameter optimization / model export
  --disable-automl      Skip AutoML/analysis even if config enables it
  --enable-automl       Force-enable AutoML
  --output-dir DIR      Override run.output_dir / result_folder
  --cpus N              Override CPU count
  --device DEV          Override structure_prediction.device
  --no-clean-batch      Kept for compatibility (no longer required)
  --clean-external-outputs  Kept for compatibility (use descriptors.cleanup)
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
ENABLE_TUNING=0
DISABLE_AUTOML=0
ENABLE_AUTOML=0
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
        --enable-tuning)
            ENABLE_TUNING=1
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
        --no-clean-batch|--clean-external-outputs)
            echo "[kitab] note: $1 is accepted for compatibility; prefer manifest fields." >&2
            shift
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
if [[ "$ENABLE_TUNING" -eq 1 ]]; then
    CMD+=(--enable-tuning)
fi
if [[ "$DISABLE_AUTOML" -eq 1 ]]; then
    CMD+=(--disable-automl)
fi
if [[ "$ENABLE_AUTOML" -eq 1 ]]; then
    CMD+=(--enable-automl)
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
