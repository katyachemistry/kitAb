#!/bin/bash

set -e

usage() {
    echo "Usage: $0 (--ginkgo | --flab) DATASET_DIR"
    echo "  --ginkgo     Use GINKGO columns: antibody_name, vh_protein_sequence, vl_protein_sequence"
    echo "  --flab       Use FLab columns: heavy, light (names = dataset_prefix + index)"
    echo "  DATASET_DIR  Path to folder containing CSV files"
    echo ""
    echo "Finds all CSV files in DATASET_DIR that:"
    echo "  - have 40–10000 data rows"
    echo "  - contain the required columns (heavy+light for --flab, or GINKGO columns for --ginkgo)"
    echo "For datasets sharing the same base name (before first '_'), uses the CSV with the most rows."
    echo "Output: for each dataset, a folder named by the base (e.g. jain2023identifying) in the current directory."
    exit 1
}

MODE=""
DATASET_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ginkgo)
            MODE="ginkgo"
            shift
            ;;
        --flab)
            MODE="flab"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -z "$DATASET_DIR" ]]; then
                DATASET_DIR="$1"
            else
                echo "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODE" ]] || [[ -z "$DATASET_DIR" ]]; then
    echo "Error: must specify --ginkgo or --flab and DATASET_DIR"
    usage
fi

DATASET_DIR="${DATASET_DIR%/}"
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Error: directory not found: $DATASET_DIR"
    exit 1
fi

echo "Mode: $MODE"
echo "Dataset directory: $DATASET_DIR"

# Discover valid CSVs: 40–10000 rows, required columns; group by base (name before first _), pick largest per group
SELECTED_CSVS=$(python3 << PYTHON_DISCOVER
import csv
import os
import sys

dataset_dir = "$DATASET_DIR"
mode = "$MODE"

if mode == "flab":
    required = {"heavy", "light"}
else:
    required = {"antibody_name", "vh_protein_sequence", "vl_protein_sequence"}

min_rows = 40
max_rows = 10000

def get_base(filename):
    stem = os.path.splitext(filename)[0]
    return stem.split("_", 1)[0] if "_" in stem else stem

candidates = []
for f in sorted(os.listdir(dataset_dir)):
    if not f.endswith(".csv"):
        continue
    path = os.path.join(dataset_dir, f)
    if not os.path.isfile(path):
        continue
    try:
        with open(path, "r") as fp:
            reader = csv.DictReader(fp)
            headers = reader.fieldnames or []
            if not required.issubset(set(headers)):
                continue
            nrows = sum(1 for _ in reader)
        if min_rows <= nrows <= max_rows:
            base = get_base(f)
            candidates.append((base, nrows, f))
    except Exception as e:
        sys.stderr.write(f"Skip {f}: {e}\n")
        continue

# Group by base, keep (base, nrows, filename) with max nrows per base
by_base = {}
for base, nrows, filename in candidates:
    if base not in by_base or nrows > by_base[base][1]:
        by_base[base] = (base, nrows, filename)

selected = sorted(by_base.values(), key=lambda x: x[0])
for base, nrows, filename in selected:
    print(os.path.join(dataset_dir, filename))
PYTHON_DISCOVER
)

if [[ -z "$SELECTED_CSVS" ]]; then
    echo "No CSV files found with 40–10000 rows and required columns."
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="0,1"

while IFS= read -r CSV_PATH; do
    [[ -z "$CSV_PATH" ]] && continue
    CSV_FILENAME=$(basename "$CSV_PATH")

    # Output folder = current directory + prefix from CSV filename (everything before first "_")
    BASENAME="${CSV_FILENAME%.*}"
    PREFIX="${BASENAME%%_*}"
    STRUCTURE_DIR="$(pwd)/${PREFIX}"

    if [[ -d "$STRUCTURE_DIR" ]]; then
        echo "Output directory already exists: $STRUCTURE_DIR (will add only missing structures)"
    fi

    echo "--- Processing: $CSV_FILENAME -> $STRUCTURE_DIR ---"

    mkdir -p "$STRUCTURE_DIR"

    TMP_CSV=$(mktemp)

    if [[ "$MODE" == "ginkgo" ]]; then
        python3 << PYTHON_SCRIPT
import csv
import os
import sys

csv_path = "$CSV_PATH"
structure_dir = "$STRUCTURE_DIR"
tmp_csv = "$TMP_CSV"

count = 0
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    with open(tmp_csv, 'w', newline='') as out:
        writer = csv.writer(out)
        for row in reader:
            name = row['antibody_name']
            heavy = row['vh_protein_sequence']
            light = row['vl_protein_sequence']

            filename = name.replace('|', '_') + '.pdb'
            filepath = os.path.join(structure_dir, filename)

            if not os.path.exists(filepath):
                writer.writerow([name, heavy, light])
                count += 1

print(f"Found {count} entries to process", file=sys.stderr)
PYTHON_SCRIPT
    else
        python3 << PYTHON_SCRIPT
import csv
import os
import sys

csv_path = "$CSV_PATH"
structure_dir = "$STRUCTURE_DIR"
tmp_csv = "$TMP_CSV"
prefix = "$PREFIX"

count = 0
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    with open(tmp_csv, 'w', newline='') as out:
        writer = csv.writer(out)
        for i, row in enumerate(reader):
            heavy = row['heavy']
            light = row['light']
            name = f"{prefix}_{i}"

            filename = name.replace('|', '_') + '.pdb'
            filepath = os.path.join(structure_dir, filename)

            if not os.path.exists(filepath):
                writer.writerow([name, heavy, light])
                count += 1

print(f"Found {count} entries to process", file=sys.stderr)
PYTHON_SCRIPT
    fi

    if [[ ! -s "$TMP_CSV" ]]; then
        echo "No new entries to process for $CSV_FILENAME (all structures already exist)."
        rm -f "$TMP_CSV"
        continue
    fi

    wc -l "$TMP_CSV"
    cat "$TMP_CSV" | parallel -j 50 --colsep ',' python3 "$SCRIPT_DIR/abb2_single_thread.py" {1} {2} {3} --output-dir "$STRUCTURE_DIR" >> logs.txt 2>&1
    rm -f "$TMP_CSV"
    echo "Done. Structures saved to $STRUCTURE_DIR"
done <<< "$SELECTED_CSVS"

echo "All datasets processed."
