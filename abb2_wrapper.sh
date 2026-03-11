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

INPUT="${1:-}"

if [[ -z "$INPUT" ]]; then
    echo "Error: CSV file or dataset directory required"
    usage
fi

INPUT="$(cd "$(dirname "$INPUT")" 2>/dev/null && pwd)/$(basename "$INPUT")" || true
INPUT="${INPUT%/}"

if [[ -f "$INPUT" ]]; then
    if [[ "$INPUT" != *.csv ]]; then
        echo "Error: not a CSV file: $INPUT"
        exit 1
    fi
    # Validate single CSV has name, heavy, light
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
elif [[ -d "$INPUT" ]]; then
    echo "Dataset directory: $INPUT"
    SELECTED_CSVS=$(python3 << PYTHON_DISCOVER
import csv
import os
import sys

dataset_dir = "$INPUT"
required = {"name", "heavy", "light"}
min_rows = 1
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
        echo "No CSV files found with columns name, heavy, light and row count 1–10000."
        exit 0
    fi
else
    echo "Error: not a file or directory: $INPUT"
    exit 1
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
