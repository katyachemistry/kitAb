#!/bin/bash
# Parallel freesasa execution for structure folders
# Usage: ./run_parallel_freesasa_GINKGO.sh STRUCTURES_DIR [num_jobs]
#
# STRUCTURES_DIR: path to folder containing PDB files (e.g. ./garbinski2023 or ./GINKGO_structures)
# Output is saved to same parent dir with suffix _sasa (e.g. ./garbinski2023_sasa)
# Naming is consistent for use with run_parallel_thermostability.sh (expects *_full.sasa per structure)

set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 STRUCTURES_DIR [num_jobs]"
    echo "  STRUCTURES_DIR  Path to folder containing PDB files"
    echo "  num_jobs       Optional. Default: nproc"
    exit 1
fi

STRUCTURES_DIR="$(cd "$1" && pwd)"
PARENT_DIR="$(dirname "$STRUCTURES_DIR")"
BASE_NAME="$(basename "$STRUCTURES_DIR")"
SASA_DIR="${PARENT_DIR}/${BASE_NAME}_sasa"

# Number of parallel jobs (default: number of CPU cores)
NUM_JOBS=${2:-$(nproc)}

# Create output directory
mkdir -p "$SASA_DIR"

echo "Running freesasa on structures..."
echo "Input directory:  $STRUCTURES_DIR"
echo "Output directory: $SASA_DIR"
echo "Number of parallel jobs: $NUM_JOBS"
echo ""

# Count total files
TOTAL=$(find "$STRUCTURES_DIR" -name "*.pdb" -type f | wc -l)
echo "Found $TOTAL PDB files"
echo ""

# Run freesasa in parallel. Output *_full.sasa for thermostability pipeline.
find "$STRUCTURES_DIR" -name "*.pdb" -type f | parallel -j "$NUM_JOBS" '
    pdbfile={}
    filename=$(basename "$pdbfile" .pdb)
    sasa_file="'"$SASA_DIR"'/${filename}_full.sasa"

    # Skip if already exists
    if [ -f "$sasa_file" ]; then
        echo "⏭  $filename (already exists)"
    else
        # Run freesasa
        if freesasa --shrake-rupley --format=rsa --depth=residue "$pdbfile" > "$sasa_file" 2>&1; then
            echo "✓  $filename"
        else
            echo "✗  $filename (failed)"
            rm -f "$sasa_file"
        fi
    fi
'

echo ""
echo "All SASA files saved to: $SASA_DIR"
