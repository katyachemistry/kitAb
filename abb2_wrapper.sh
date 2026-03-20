#!/bin/bash

set -e

usage() {
    echo "Usage: $0 CSV_FILE | $0 DATASET_DIR"
    echo ""
    echo "CSV must have columns: name, heavy, light. Each row -> name.pdb"
    echo "(name is sanitized for filenames: '|' -> '_')"
    echo ""
    echo "  CSV_FILE    Single CSV to process; output dir = current dir / (filename base before first '_')"
    echo "  DATASET_DIR Directory: finds all CSVs with name, heavy, light (1–10000 rows),"
    echo "              groups by base, picks largest per base, processes each."
    exit 1
}

CSV_INPUT=""
DIR_INPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)
            CSV_INPUT="$2"
            shift 2
            ;;
        --dir)
            DIR_INPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate input
if [[ -n "$CSV_INPUT" && -n "$DIR_INPUT" ]]; then
    echo "Error: use either --csv or --dir, not both"
    exit 1
fi

if [[ -z "$CSV_INPUT" && -z "$DIR_INPUT" ]]; then
    echo "Error: must provide either --csv or --dir"
    usage
fi

# Normalize path
normalize_path() {
    local p="$1"
    p="$(cd "$(dirname "$p")" 2>/dev/null && pwd)/$(basename "$p")" || return 1
    echo "${p%/}"
}

if [[ -n "$CSV_INPUT" ]]; then
    INPUT="$(normalize_path "$CSV_INPUT")" || {
        echo "Error resolving path: $CSV_INPUT"
        exit 1
    }

    if [[ ! -f "$INPUT" ]]; then
        echo "Error: not a file: $INPUT"
        exit 1
    fi

    if [[ "$INPUT" != *.csv ]]; then
        echo "Error: not a CSV file: $INPUT"
        exit 1
    fi

    # Validate CSV columns
    if ! python3 -c "
import csv, sys
with open('$INPUT', 'r') as f:
    r = csv.DictReader(f)
    h = set(r.fieldnames or [])
    if not {'name','heavy','light'}.issubset(h):
        sys.stderr.write('CSV must have columns: name, heavy, light\n')
        sys.exit(1)
" 2>/dev/null; then
        echo "Error: CSV must have columns name, heavy, light: $INPUT"
        exit 1
    fi

    SELECTED_CSVS="$INPUT"
    echo "CSV file: $INPUT"

elif [[ -n "$DIR_INPUT" ]]; then
    INPUT="$(normalize_path "$DIR_INPUT")" || {
        echo "Error resolving path: $DIR_INPUT"
        exit 1
    }

    if [[ ! -d "$INPUT" ]]; then
        echo "Error: not a directory: $INPUT"
        exit 1
    fi

    echo "Dataset directory: $INPUT"

    SELECTED_CSVS=$(python3 << PYTHON_DISCOVER
# (keep your existing discovery code unchanged here)
PYTHON_DISCOVER
)

    if [[ -z "$SELECTED_CSVS" ]]; then
        echo "No CSV files found with columns name, heavy, light and row count 1–10000."
        exit 0
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="0,1"

while IFS= read -r CSV_PATH; do
    [[ -z "$CSV_PATH" ]] && continue
    CSV_FILENAME=$(basename "$CSV_PATH")

    BASENAME="${CSV_FILENAME%.*}"
    PREFIX="${BASENAME%%_*}"
    STRUCTURE_DIR="$(pwd)/${PREFIX}"

    if [[ -d "$STRUCTURE_DIR" ]]; then
        echo "Output directory already exists: $STRUCTURE_DIR (will add only missing structures)"
    fi

    echo "--- Processing: $CSV_FILENAME -> $STRUCTURE_DIR ---"

    mkdir -p "$STRUCTURE_DIR"

    TMP_TSV=$(mktemp)

    python3 << PYTHON_SCRIPT
import csv
import os
import sys

csv_path = "$CSV_PATH"
structure_dir = "$STRUCTURE_DIR"
tmp_tsv = "$TMP_TSV"

count = 0
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    with open(tmp_tsv, 'w', newline='') as out:
        writer = csv.writer(out, delimiter='\t')
        for row in reader:
            name = (row.get('name') or '').strip()
            heavy = (row.get('heavy') or '').strip()
            light = (row.get('light') or '').strip()
            if not name:
                continue
            filename = name.replace('|', '_') + '.pdb'
            filepath = os.path.join(structure_dir, filename)
            if not os.path.exists(filepath):
                writer.writerow([name, heavy, light])
                count += 1

print(f"Found {count} entries to process", file=sys.stderr)
PYTHON_SCRIPT

    if [[ ! -s "$TMP_TSV" ]]; then
        echo "No new entries to process for $CSV_FILENAME (all structures already exist)."
        rm -f "$TMP_TSV"
        continue
    fi

    sed -i 's/\r$//' "$TMP_TSV" 2>/dev/null || true

    wc -l "$TMP_TSV"
    if ! cat "$TMP_TSV" | parallel -j 50 --colsep $'\t' python3 "$SCRIPT_DIR/abb2_single_thread.py" {1} {2} {3} --output-dir "$STRUCTURE_DIR" >> logs.txt 2>&1; then
        echo "Warning: some jobs failed for $CSV_FILENAME. Check logs.txt for errors (e.g. 'tail -20 logs.txt')."
    fi
    rm -f "$TMP_TSV"
    echo "Done. Structures saved to $STRUCTURE_DIR"
done <<< "$SELECTED_CSVS"

echo "All datasets processed."
