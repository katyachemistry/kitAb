#!/bin/bash
# Wrapper: delegates to run_parallel.sh --parallel developability
# Usage: ./run_parallel_developability.sh STRUCTURES_DIR [STRUCTURES_DIR ...] [num_jobs] [options]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/src/run_parallel.sh" --parallel developability "$@"
