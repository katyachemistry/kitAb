#!/bin/bash
# Wrapper: delegates to run_parallel.sh --parallel propka
# Usage: ./run_parallel_propka3.sh STRUCTURES_DIR [STRUCTURES_DIR ...] [num_jobs]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/src/run_parallel.sh" --parallel propka "$@"
