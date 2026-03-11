#!/bin/bash
# Parallel execution script for thermostability calculation (H-bonds, salt bridges, aromatic residues, and WCN)
# Usage: ./run_parallel_thermostability.sh STRUCTURES_DIR [num_jobs] [options]
#
# STRUCTURES_DIR: path to folder containing PDB files (e.g. ./garbinski2023 or ./GINKGO_structures).
#   SASA, DSSP, propka and results are taken from/saved to same parent dir with suffixes _sasa, _dssp, _propka, _results.
#
# Options (passed to calculate_thermostability.py):
#   --average              Calculate only averages
#   --hbonds-only          Calculate only H-bonds
#   --salt-bridges-only    Calculate only salt bridges
#   --aromatic-only        Calculate only aromatic residues
#   --wcn-only             Calculate only Weighted Contact Number (WCN, no SASA needed)
#   --unweighted           Use unweighted calculations (no SASA needed)
#   --weighting {inverse|negative_linear}  Weighting strategy (default: inverse)
#   --format {table|csv|json}  Output format (default: csv)
#   --pka-file <path>      Path to pKa file (optional, will auto-detect if not provided)
#   --pH <value>           pH value for salt bridge charge state determination (default: 7.4)
#
# Examples:
#   ./run_parallel_thermostability.sh ./garbinski2023 8
#   ./run_parallel_thermostability.sh ./GINKGO_structures 8 --unweighted
#   ./run_parallel_thermostability.sh ./garbinski2023 --hbonds-only --format json

set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 STRUCTURES_DIR [num_jobs] [options]" >&2
    echo "  STRUCTURES_DIR  Path to folder containing PDB files (required)" >&2
    exit 1
fi

STRUCTURES_DIR="$(cd "$1" && pwd)"
PARENT_DIR="$(dirname "$STRUCTURES_DIR")"
BASE_NAME="$(basename "$STRUCTURES_DIR")"
SASA_DIR="${PARENT_DIR}/${BASE_NAME}_sasa"
DSSP_DIR="${PARENT_DIR}/${BASE_NAME}_dssp"
PKA_DIR="${PARENT_DIR}/${BASE_NAME}_propka"
OUTPUT_DIR="${PARENT_DIR}/${BASE_NAME}_results"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

shift || true

# Parse arguments
NUM_JOBS=""
EXTRA_ARGS=()
USE_UNWEIGHTED=false
OUTPUT_FORMAT="csv"
WEIGHTING="inverse"
PH_VALUE="7.4"

while [[ $# -gt 0 ]]; do
    case $1 in
        --unweighted)
            USE_UNWEIGHTED=true
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --average|--hbonds-only|--salt-bridges-only|--aromatic-only|--wcn-only)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --weighting)
            WEIGHTING="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --pka-file)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --pH)
            PH_VALUE="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        [0-9]*)
            NUM_JOBS="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 STRUCTURES_DIR [num_jobs] [--average] [--hbonds-only] [--salt-bridges-only] [--aromatic-only] [--wcn-only] [--unweighted] [--weighting {inverse|negative_linear}] [--format {table|csv|json}] [--pka-file <path>] [--pH <value>]" >&2
            exit 1
            ;;
    esac
done

# Number of parallel jobs (default: number of CPU cores)
NUM_JOBS=${NUM_JOBS:-$(nproc)}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to script directory
cd "$SCRIPT_DIR"

# Export variables for Python script and parallel
export NUM_JOBS USE_UNWEIGHTED OUTPUT_FORMAT WEIGHTING PH_VALUE
export STRUCTURES_DIR SASA_DIR DSSP_DIR PKA_DIR OUTPUT_DIR SCRIPT_DIR
export EXTRA_ARGS_STR="${EXTRA_ARGS[*]}"

# Display configuration
echo "Configuration:"
echo "  Structures: $STRUCTURES_DIR"
echo "  SASA:       $SASA_DIR"
echo "  DSSP:       $DSSP_DIR"
echo "  pKa:        $PKA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Parallel jobs: $NUM_JOBS"
echo "  Output format: $OUTPUT_FORMAT"
echo "  Unweighted: $USE_UNWEIGHTED"
if [ "$USE_UNWEIGHTED" = false ]; then
    echo "  Weighting: $WEIGHTING"
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Extra options: ${EXTRA_ARGS[*]}"
fi
echo ""

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with $NUM_JOBS jobs..."
    
    # Create a function to process one file pair
    process_file() {
        local pdb_file="$1"
        local basename=$(basename "$pdb_file" .pdb)
        local sasa_file="${SASA_DIR}/${basename}_full.sasa"
        local dssp_file="${DSSP_DIR}/${basename}.dssp"
        local pka_file="${PKA_DIR}/${basename}_full.pka"
        
        # Determine output file extension based on format
        case "$OUTPUT_FORMAT" in
            json)
                local output_file="${OUTPUT_DIR}/${basename}.json"
                ;;
            table)
                local output_file="${OUTPUT_DIR}/${basename}.txt"
                ;;
            *)
                local output_file="${OUTPUT_DIR}/${basename}.csv"
                ;;
        esac
        
        # Build command
        local cmd=("python3" "thermostability/calculate_thermostability.py" "$pdb_file")
        
        # Reconstruct extra arguments from string (for GNU parallel compatibility)
        local extra_args_array=($EXTRA_ARGS_STR)
        
        # Check if --wcn-only is in extra args (WCN doesn't require SASA file)
        local has_wcn_only=false
        for arg in "${extra_args_array[@]}"; do
            if [ "$arg" = "--wcn-only" ]; then
                has_wcn_only=true
                break
            fi
        done
        
        # Add SASA file when needed for calculations or for output columns (so all runs have identical CSV columns)
        if [ "$has_wcn_only" = false ]; then
            if [ -f "$sasa_file" ]; then
                cmd+=("$sasa_file")
            elif [ "$USE_UNWEIGHTED" = false ]; then
                # Weighted run requires SASA; fail if missing
                echo "✗ $basename (SASA file not found)"
                return
            fi
        fi
        
        # Add DSSP file if it exists
        if [ -f "$dssp_file" ]; then
            cmd+=("--dssp-file" "$dssp_file")
        fi
        
        # Add pKa file if it exists (unless --pka-file is already in extra args)
        local has_pka_file=false
        for arg in "${extra_args_array[@]}"; do
            if [ "$arg" = "--pka-file" ]; then
                has_pka_file=true
                break
            fi
        done
        
        if [ "$has_pka_file" = false ] && [ -f "$pka_file" ]; then
            cmd+=("--pka-file" "$pka_file")
        fi
        
        # Add pH if not already in extra args
        local has_ph=false
        for arg in "${extra_args_array[@]}"; do
            if [ "$arg" = "--pH" ]; then
                has_ph=true
                break
            fi
        done
        
        if [ "$has_ph" = false ]; then
            cmd+=("--pH" "$PH_VALUE")
        fi
        
        cmd+=("${extra_args_array[@]}")
        
        # Add format and output
        cmd+=("--format" "$OUTPUT_FORMAT" "--output" "$output_file")
        
        # Run command
        if "${cmd[@]}" 2>&1; then
            echo "✓ $basename"
        else
            echo "✗ $basename (failed)"
        fi
    }
    
    export -f process_file
    export SASA_DIR DSSP_DIR PKA_DIR OUTPUT_DIR SCRIPT_DIR USE_UNWEIGHTED OUTPUT_FORMAT WEIGHTING PH_VALUE EXTRA_ARGS_STR
    
    # Find all PDB files and process in parallel
    find "$STRUCTURES_DIR" -name "*.pdb" -type f | \
        parallel -j "$NUM_JOBS" process_file {}
    
else
    echo "GNU parallel not found. Using Python multiprocessing..."
    python3 << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess

STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
SASA_DIR = os.environ.get("SASA_DIR", ".")
DSSP_DIR = os.environ.get("DSSP_DIR", ".")
PKA_DIR = os.environ.get("PKA_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")

# Get configuration from environment
num_jobs = int(os.environ.get('NUM_JOBS', cpu_count()))
use_unweighted = os.environ.get('USE_UNWEIGHTED', 'false').lower() == 'true'
output_format = os.environ.get('OUTPUT_FORMAT', 'csv')
weighting = os.environ.get('WEIGHTING', 'inverse')
ph_value = os.environ.get('PH_VALUE', '7.4')

# Parse extra arguments from environment
extra_args_str = os.environ.get('EXTRA_ARGS_STR', '')
extra_args = extra_args_str.split() if extra_args_str else []

def process_file(pdb_path):
    """Process a single PDB file."""
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    sasa_file = Path(SASA_DIR) / f"{basename}_full.sasa"
    dssp_file = Path(DSSP_DIR) / f"{basename}.dssp"
    pka_file = Path(PKA_DIR) / f"{basename}_full.pka"
    
    # Determine output file extension based on format
    if output_format == 'json':
        output_file = Path(OUTPUT_DIR) / f"{basename}.json"
    elif output_format == 'table':
        output_file = Path(OUTPUT_DIR) / f"{basename}.txt"
    else:
        output_file = Path(OUTPUT_DIR) / f"{basename}.csv"
    
    # Build command
    cmd = [
        "python3",
        str(Path(SCRIPT_DIR) / "thermostability" / "calculate_thermostability.py"),
        str(pdb_file)
    ]
    
    # Add SASA file if not using unweighted and not using WCN-only
    # WCN doesn't require SASA file
    if not use_unweighted:
        # Check if --wcn-only is in extra args
        has_wcn_only = '--wcn-only' in extra_args
        
        if not has_wcn_only:
            if not sasa_file.exists():
                return f"✗ {basename} (SASA file not found)"
            cmd.append(str(sasa_file))
    
    # Add DSSP file if it exists
    if dssp_file.exists():
        cmd.extend(["--dssp-file", str(dssp_file)])
    
    # Add pKa file if it exists (unless --pka-file is already in extra args)
    has_pka_file = '--pka-file' in extra_args
    if not has_pka_file and pka_file.exists():
        cmd.extend(["--pka-file", str(pka_file)])
    
    # Add pH if not already in extra args
    has_ph = '--pH' in extra_args
    if not has_ph:
        cmd.extend(["--pH", ph_value])
    
    # Add extra arguments
    cmd.extend(extra_args)
    
    # Add format and output
    cmd.extend(["--format", output_format, "--output", str(output_file)])
    
    try:
        # Run the calculation script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode == 0:
            return f"✓ {basename}"
        else:
            return f"✗ {basename} (failed: {result.stderr[:50]})"
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:50]})"

if __name__ == "__main__":
    # Find all PDB files
    pdb_files = sorted(Path(STRUCTURES_DIR).rglob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Using {num_jobs} parallel jobs")
    print(f"Output format: {output_format}")
    print(f"Unweighted: {use_unweighted}")
    if not use_unweighted:
        print(f"Weighting: {weighting}")
    print(f"pH: {ph_value}")
    if extra_args:
        print(f"Extra options: {' '.join(extra_args)}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Process in parallel
    with Pool(num_jobs) as pool:
        results = pool.map(process_file, pdb_files)
    
    # Print results
    for result in results:
        print(result)
    
    # Summary
    successful = sum(1 for r in results if r.startswith("✓"))
    print(f"\nCompleted: {successful}/{len(pdb_files)} successful")
PYTHON_SCRIPT
fi

echo ""
echo "All results saved to: $OUTPUT_DIR"

