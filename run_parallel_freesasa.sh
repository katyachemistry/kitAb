#!/bin/bash
# Wrapper: delegates to run_parallel.sh --parallel freesasa
# Usage: ./run_parallel_freesasa.sh STRUCTURES_DIR [STRUCTURES_DIR ...] [num_jobs]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/src/run_parallel.sh" --parallel freesasa "$@"
