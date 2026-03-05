#!/bin/bash
# Parallel execution script for DSSP secondary structure calculation
# Usage: ./run_parallel_dssp.sh STRUCTURES_DIR [num_jobs]
#
# STRUCTURES_DIR: path to folder containing PDB files (e.g. ./garbinski2023 or ./GINKGO_structures)
# Output is saved to same parent dir with suffix _dssp (e.g. ./garbinski2023_dssp)

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
OUTPUT_DIR="${PARENT_DIR}/${BASE_NAME}_dssp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Number of parallel jobs (default: number of CPU cores)
NUM_JOBS=${2:-$(nproc)}

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR"

echo "Running DSSP on structures..."
echo "Input directory:  $STRUCTURES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of parallel jobs: $NUM_JOBS"
echo ""

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with $NUM_JOBS jobs..."

    # Create a function to process one file
    process_file() {
        local pdb_file="$1"
        local basename=$(basename "$pdb_file" .pdb)
        local filename=$(basename "$pdb_file")
        local output_file="${OUTPUT_DIR}/${basename}.dssp"
        local temp_pdb=$(mktemp)

        # Preprocess PDB: remove REMARK and CRYST1 lines, then add header and CRYST1 explicitly
        {
            printf "REMARK    @%s (1-2)\n" "$filename"
            echo "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          "
            awk '!/^REMARK/ && !/^CRYST1/' "$pdb_file"
        } > "$temp_pdb"

        if dssp "$temp_pdb" "$output_file" 2>&1; then
            echo "✓ $basename"
        else
            echo "✗ $basename (failed)"
        fi

        rm -f "$temp_pdb"
    }

    export -f process_file
    export OUTPUT_DIR

    # Find all PDB files and process in parallel
    find "$STRUCTURES_DIR" -name "*.pdb" -type f | \
        parallel -j "$NUM_JOBS" process_file {}

else
    echo "GNU parallel not found. Using Python multiprocessing..."
    export STRUCTURES_DIR OUTPUT_DIR SCRIPT_DIR NUM_JOBS
    python3 << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile

STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def process_file(pdb_path):
    """Process a single PDB file."""
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    filename = pdb_file.name
    output_file = Path(OUTPUT_DIR) / f"{basename}.dssp"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_pdb:
            temp_pdb_path = temp_pdb.name
            temp_pdb.write(f"REMARK    @{filename} (1-2)\n")
            temp_pdb.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n")
            with open(pdb_file, 'r') as f:
                for line in f:
                    if not line.startswith('REMARK') and not line.startswith('CRYST1'):
                        temp_pdb.write(line)

        result = subprocess.run(
            ["dssp", temp_pdb_path, str(output_file)],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )

        Path(temp_pdb_path).unlink()

        if result.returncode == 0:
            return f"✓ {basename}"
        else:
            return f"✗ {basename} (failed: {result.stderr[:50]})"
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:50]})"

if __name__ == "__main__":
    pdb_files = sorted(Path(STRUCTURES_DIR).rglob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Using {num_jobs} parallel jobs")
    print(f"Output directory: {OUTPUT_DIR}\n")

    with Pool(num_jobs) as pool:
        results = pool.map(process_file, pdb_files)

    for result in results:
        print(result)

    successful = sum(1 for r in results if r.startswith("✓"))
    print(f"\nCompleted: {successful}/{len(pdb_files)} successful")
PYTHON_SCRIPT
fi

echo ""
echo "All results saved to: $OUTPUT_DIR"
