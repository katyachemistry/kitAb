#!/bin/bash
# Parallel execution script for propka3 pKa calculation
# Usage: ./run_parallel_propka3.sh STRUCTURES_DIR [num_jobs]
#
# STRUCTURES_DIR: path to folder containing PDB files (e.g. ./garbinski2023 or ./GINKGO_structures)
# Output is saved to same parent dir with suffix _propka (e.g. ./garbinski2023_propka)

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
OUTPUT_DIR="${PARENT_DIR}/${BASE_NAME}_propka"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Number of parallel jobs (default: number of CPU cores)
NUM_JOBS=${2:-$(nproc)}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to script directory for any relative paths
cd "$SCRIPT_DIR"

echo "Running propka3 on structures..."
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
        local output_dir="$2"

        # Activate conda environment
        source /home/kb/miniforge3/bin/activate developability

        # Change to output directory (propka3 writes to current directory)
        cd "$output_dir"

        echo "Processing $basename..."

        # Step 1: Run propka3 on full PDB
        propka3 "$pdb_file" > "${basename}_full.log" 2>&1

        # Rename the output .pka file if it exists
        if [ -f "${basename}.pka" ]; then
            mv "${basename}.pka" "${basename}_full.pka"
            echo "✓ $basename (full)"
        else
            echo "✗ $basename (full - failed)"
        fi

        # Step 2: Extract H chain and run propka3
        grep -E "^REMARK|^HEADER|^TITLE|^COMPND|^SOURCE" "$pdb_file" > "${basename}_H_chain.pdb.tmp"
        awk '/^ATOM/ || /^HETATM/ { if (substr($0, 22, 1) == "H") print }' "$pdb_file" >> "${basename}_H_chain.pdb.tmp"
        echo "END" >> "${basename}_H_chain.pdb.tmp"
        mv "${basename}_H_chain.pdb.tmp" "${basename}_H_chain.pdb"

        propka3 "${basename}_H_chain.pdb" > "${basename}_H.log" 2>&1

        if [ -f "${basename}_H_chain.pka" ]; then
            mv "${basename}_H_chain.pka" "${basename}_H.pka"
            echo "✓ $basename (H chain)"
        else
            echo "✗ $basename (H chain - failed)"
        fi

        # Step 3: Extract L chain and run propka3
        grep -E "^REMARK|^HEADER|^TITLE|^COMPND|^SOURCE" "$pdb_file" > "${basename}_L_chain.pdb.tmp"
        awk '/^ATOM/ || /^HETATM/ { if (substr($0, 22, 1) == "L") print }' "$pdb_file" >> "${basename}_L_chain.pdb.tmp"
        echo "END" >> "${basename}_L_chain.pdb.tmp"
        mv "${basename}_L_chain.pdb.tmp" "${basename}_L_chain.pdb"

        propka3 "${basename}_L_chain.pdb" > "${basename}_L.log" 2>&1

        if [ -f "${basename}_L_chain.pka" ]; then
            mv "${basename}_L_chain.pka" "${basename}_L.pka"
            echo "✓ $basename (L chain)"
        else
            echo "✗ $basename (L chain - failed)"
        fi

        # Clean up temporary chain PDB files
        rm -f "${basename}_H_chain.pdb" "${basename}_L_chain.pdb"
    }

    export -f process_file
    export OUTPUT_DIR

    # Find all PDB files and process in parallel
    find "$STRUCTURES_DIR" -name "*.pdb" -type f | \
        parallel -j "$NUM_JOBS" process_file {} "$OUTPUT_DIR"

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
    output_dir = Path(OUTPUT_DIR)

    results = []

    try:
        # Activate conda environment and run propka3
        conda_activate = "source /home/kb/miniforge3/bin/activate developability"

        # Step 1: Run propka3 on full PDB
        result = subprocess.run(
            ["bash", "-c", f"{conda_activate} && propka3 '{pdb_file}'"],
            capture_output=True,
            text=True,
            cwd=output_dir
        )

        # Save log
        with open(output_dir / f"{basename}_full.log", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Rename output file if it exists
        pka_output = output_dir / f"{basename}.pka"
        if pka_output.exists():
            pka_output.rename(output_dir / f"{basename}_full.pka")
            results.append(f"✓ {basename} (full)")
        else:
            results.append(f"✗ {basename} (full - failed)")

        # Step 2: Extract H chain
        h_chain_pdb = output_dir / f"{basename}_H_chain.pdb"
        with open(h_chain_pdb, "w") as f:
            with open(pdb_file, "r") as pdb:
                for line in pdb:
                    if any(line.startswith(x) for x in ["REMARK", "HEADER", "TITLE", "COMPND", "SOURCE"]):
                        f.write(line)
            with open(pdb_file, "r") as pdb:
                for line in pdb:
                    if line.startswith(("ATOM", "HETATM")):
                        if len(line) > 21 and line[21] == "H":
                            f.write(line)
            f.write("END\n")

        result = subprocess.run(
            ["bash", "-c", f"{conda_activate} && propka3 '{h_chain_pdb}'"],
            capture_output=True,
            text=True,
            cwd=output_dir
        )

        with open(output_dir / f"{basename}_H.log", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        h_pka_output = output_dir / f"{basename}_H_chain.pka"
        if h_pka_output.exists():
            h_pka_output.rename(output_dir / f"{basename}_H.pka")
            results.append(f"✓ {basename} (H chain)")
        else:
            results.append(f"✗ {basename} (H chain - failed)")

        # Step 3: Extract L chain
        l_chain_pdb = output_dir / f"{basename}_L_chain.pdb"
        with open(l_chain_pdb, "w") as f:
            with open(pdb_file, "r") as pdb:
                for line in pdb:
                    if any(line.startswith(x) for x in ["REMARK", "HEADER", "TITLE", "COMPND", "SOURCE"]):
                        f.write(line)
            with open(pdb_file, "r") as pdb:
                for line in pdb:
                    if line.startswith(("ATOM", "HETATM")):
                        if len(line) > 21 and line[21] == "L":
                            f.write(line)
            f.write("END\n")

        result = subprocess.run(
            ["bash", "-c", f"{conda_activate} && propka3 '{l_chain_pdb}'"],
            capture_output=True,
            text=True,
            cwd=output_dir
        )

        with open(output_dir / f"{basename}_L.log", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        l_pka_output = output_dir / f"{basename}_L_chain.pka"
        if l_pka_output.exists():
            l_pka_output.rename(output_dir / f"{basename}_L.pka")
            results.append(f"✓ {basename} (L chain)")
        else:
            results.append(f"✗ {basename} (L chain - failed)")

        h_chain_pdb.unlink(missing_ok=True)
        l_chain_pdb.unlink(missing_ok=True)

        return "\n".join(results)

    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:100]})"

if __name__ == "__main__":
    pdb_files = sorted(Path(STRUCTURES_DIR).rglob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Using {num_jobs} parallel jobs")
    print(f"Output directory: {OUTPUT_DIR}\n")

    with Pool(num_jobs) as pool:
        results = pool.map(process_file, pdb_files)

    for result in results:
        print(result)

    successful = sum(1 for r in results for line in r.split("\n") if line.startswith("✓"))
    total = len(pdb_files) * 3
    print(f"\nCompleted: {successful}/{total} successful")
PYTHON_SCRIPT
fi

echo ""
echo "All results saved to: $OUTPUT_DIR"
