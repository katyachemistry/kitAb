#!/bin/bash
# mkdssp, propka, freesasa -> developability (parallel over PDB folders).
# Usage: get_descriptors.sh [--output-dir DIR] [--parent-dir DIR ...] STRUCTURES_DIR ... [num_jobs] [--pH VALUE] [--sanity_check_abb2] [--remove_helper_outputs] [--clean-external-outputs]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/fastab.local.env" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/fastab.local.env"
fi

FASTAB_ENV="${FASTAB_ENV:-fastab}"
PY="${PY:-conda run -n ${FASTAB_ENV} python}"
DSSP_BIN="${DSSP_BIN:-mkdssp}"

_py_to_array() {
    local _py_cmd="$1"
    local -n _out=$2
    _out=()
    mapfile -t _out < <(python3 -c 'import shlex,sys
for part in shlex.split(sys.argv[1]):
    print(part)' "$_py_cmd")
    if [[ ${#_out[@]} -eq 0 ]]; then
        echo "Error: PY command is empty." >&2
        exit 1
    fi
}
_py_to_array "$PY" PY_ARR

_fastab_run() {
    if command -v mamba &>/dev/null; then
        mamba run -n "$FASTAB_ENV" "$@"
    elif command -v conda &>/dev/null; then
        conda run -n "$FASTAB_ENV" "$@"
    else
        "$@"
    fi
}

EXPLICIT_STRUCTURES=()
PARENT_DIR_INPUTS=()
NUM_JOBS=""
PH_VALUE="7.5"
USER_OUTPUT_DESCRIPTOR_ROOT=""
NAMES_FROM_CSV=""
SANITY_CHECK_ABB2=false
REMOVE_HELPER_OUTPUTS=false
CLEAN_EXTERNAL_OUTPUTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 [--output-dir DIR] ..." >&2
                echo "  Error: --output-dir requires a directory path." >&2
                exit 1
            fi
            USER_OUTPUT_DESCRIPTOR_ROOT="$2"
            shift 2
            ;;
        --parent-dir)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 [--output-dir DIR] [--parent-dir DIR ...] STRUCTURES_DIR [STRUCTURES_DIR ...] [num_jobs] [options]" >&2
                echo "  Error: --parent-dir requires a directory path." >&2
                exit 1
            fi
            PARENT_DIR_INPUTS+=("$2")
            shift 2
            ;;
        --pH)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 ... [--pH VALUE]" >&2
                echo "  Error: --pH requires a value." >&2
                exit 1
            fi
            PH_VALUE="$2"
            shift 2
            ;;
        --names-from-csv)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 ... [--names-from-csv CSV]" >&2
                echo "  Error: --names-from-csv requires a CSV path." >&2
                exit 1
            fi
            NAMES_FROM_CSV="$2"
            shift 2
            ;;
        --sanity_check_abb2)
            SANITY_CHECK_ABB2=true
            shift
            ;;
        --remove_helper_outputs)
            REMOVE_HELPER_OUTPUTS=true
            shift
            ;;
        --clean-external-outputs)
            CLEAN_EXTERNAL_OUTPUTS=true
            shift
            ;;
        [0-9]*)
            NUM_JOBS="$1"
            shift
            ;;
        *)
            EXPLICIT_STRUCTURES+=("$1")
            shift
            ;;
    esac
done

STRUCTURES_DIRS=()
for arg in "${EXPLICIT_STRUCTURES[@]}"; do
    if [[ -z "$arg" ]]; then
        continue
    fi
    if [[ -d "$arg" ]]; then
        STRUCTURES_DIRS+=("$arg")
        continue
    fi
    if [[ "$arg" != /* && "$arg" != *[\*\?\[]* && -d "$PROJECT_ROOT/$arg" ]]; then
        STRUCTURES_DIRS+=("$PROJECT_ROOT/$arg")
        continue
    fi
    if [[ "$arg" != *[\*\?\[]* ]]; then
        echo "Error: not a directory: $arg" >&2
        exit 1
    fi
    _n_before="${#STRUCTURES_DIRS[@]}"
    while IFS= read -r m; do
        [[ -z "$m" ]] && continue
        if [[ -d "$m" ]]; then
            STRUCTURES_DIRS+=("$m")
        elif [[ -e "$m" ]]; then
            echo "Error: matched path is not a directory (use folders of PDBs only): $m" >&2
            exit 1
        fi
    done < <(compgen -G "$arg" | LC_ALL=C sort -u)
    if [[ "${#STRUCTURES_DIRS[@]}" -eq "$_n_before" && "$arg" != /* ]]; then
        while IFS= read -r m; do
            [[ -z "$m" ]] && continue
            if [[ -d "$m" ]]; then
                STRUCTURES_DIRS+=("$m")
            elif [[ -e "$m" ]]; then
                echo "Error: matched path is not a directory (use folders of PDBs only): $m" >&2
                exit 1
            fi
        done < <(compgen -G "$PROJECT_ROOT/$arg" | LC_ALL=C sort -u)
    fi
    if [[ "${#STRUCTURES_DIRS[@]}" -eq "$_n_before" ]]; then
        echo "Error: pattern matched no directories: $arg (cwd: $PWD, also tried under repo: $PROJECT_ROOT)" >&2
        exit 1
    fi
    unset _n_before
done

for pd in "${PARENT_DIR_INPUTS[@]}"; do
    if [[ ! -d "$pd" ]]; then
        echo "Error: --parent-dir is not a directory: $pd" >&2
        exit 1
    fi
    pd_abs="$(cd "$pd" && pwd)"
    while IFS= read -r -d '' entry; do
        base="$(basename "$entry")"
        [[ "$base" == .* ]] && continue
        if [[ -n "$(find "$entry" -name '*.pdb' -type f -print -quit 2>/dev/null)" ]]; then
            STRUCTURES_DIRS+=("$entry")
        fi
    done < <(find "$pd_abs" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | LC_ALL=C sort -z)
done

if [[ ${#STRUCTURES_DIRS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--output-dir DIR] [--parent-dir DIR ...] STRUCTURES_DIR [STRUCTURES_DIR ...] [num_jobs] [options]" >&2
    echo "  Runs mkdssp -> propka -> freesasa -> developability for each folder (default: ./developability_descriptors/<basename>/)." >&2
    echo "  --output-dir DIR: write per-dataset outputs under DIR/<basename>/ instead of ./developability_descriptors/<basename>/." >&2
    echo "  --parent-dir: run on every immediate subdirectory of DIR that contains at least one .pdb file." >&2
    echo "  --sanity_check_abb2: skip PDBs whose ATOM B-factors are all 0.00; log skipped names under the output dir." >&2
    echo "  --remove_helper_outputs: delete dssp/propka/freesasa files after a structure's descriptor JSON is written." >&2
    echo "  --clean-external-outputs: after all descriptors for a dataset, remove dssp/, propka/, and sasa/ subdirs." >&2
    exit 1
fi

declare -A _seen_structure_dirs=()
_deduped_structure_dirs=()
for d in "${STRUCTURES_DIRS[@]}"; do
    if [[ ! -d "$d" ]]; then
        echo "Error: not a directory: $d" >&2
        exit 1
    fi
    resolved="$(cd "$d" && pwd)"
    if [[ -n "${_seen_structure_dirs[$resolved]:-}" ]]; then
        continue
    fi
    _seen_structure_dirs[$resolved]=1
    _deduped_structure_dirs+=("$resolved")
done
STRUCTURES_DIRS=("${_deduped_structure_dirs[@]}")
unset _seen_structure_dirs
unset _deduped_structure_dirs

NUM_JOBS=${NUM_JOBS:-$(nproc)}
if ! [[ "$NUM_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_jobs must be a positive integer, got: ${NUM_JOBS:-<empty>}" >&2
    exit 1
fi
RUN_INVOCATION_DIR="$(pwd -P 2>/dev/null || pwd)"
if [[ -n "$USER_OUTPUT_DESCRIPTOR_ROOT" ]]; then
    if [[ "$USER_OUTPUT_DESCRIPTOR_ROOT" == /* ]]; then
        mkdir -p "$USER_OUTPUT_DESCRIPTOR_ROOT"
        DESCRIPTOR_ROOT="$(cd "$USER_OUTPUT_DESCRIPTOR_ROOT" && pwd -P 2>/dev/null || pwd)"
    else
        mkdir -p "$RUN_INVOCATION_DIR/$USER_OUTPUT_DESCRIPTOR_ROOT"
        DESCRIPTOR_ROOT="$(cd "$RUN_INVOCATION_DIR/$USER_OUTPUT_DESCRIPTOR_ROOT" && pwd -P 2>/dev/null || pwd)"
    fi
else
    DESCRIPTOR_ROOT="${RUN_INVOCATION_DIR}/developability_descriptors"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NUM_JOBS SCRIPT_DIR
export PH_VALUE
export SANITY_CHECK_ABB2
export REMOVE_HELPER_OUTPUTS
export CLEAN_EXTERNAL_OUTPUTS

_remove_helper_outputs_for_stem() {
    local basename="$1"
    [[ "$REMOVE_HELPER_OUTPUTS" == true ]] || return 0
    rm -f \
        "${DSSP_DIR}/${basename}.dssp" \
        "${PKA_DIR}/${basename}_full.pka" \
        "${PKA_DIR}/${basename}_full.log" \
        "${SASA_DIR}/${basename}_full.sasa" \
        "${SASA_DIR}/${basename}_H_full.sasa" \
        "${SASA_DIR}/${basename}_L_full.sasa"
}

_clean_external_output_dirs() {
    [[ "$CLEAN_EXTERNAL_OUTPUTS" == true ]] || return 0
    rm -rf "$DSSP_OUTPUT_DIR" "$PROPKA_OUTPUT_DIR" "$SASA_OUTPUT_DIR"
    echo "Removed external helper dirs for ${BASE_NAME}: dssp/, propka/, sasa/"
}

# PropKa writes {input_stem}.pka in cwd. Temp renumbered inputs use the prepare stem
# (e.g. mAb1_full.pdb -> mAb1_full.pka), then rename to {basename}_full.pka.
_finalize_propka_pka() {
    local generated="$1"
    local target="$2"
    if [ ! -f "$generated" ]; then
        return 1
    fi
    if [ "$generated" != "$target" ]; then
        mv "$generated" "$target"
    fi
    return 0
}

_abb2_bfactor_all_zero() {
    local pdb_path="$1"
    # PDB B-factor columns 61-66; skip when every ATOM/HETATM value is 0.00.
    awk '
        /^ATOM|^HETATM/ {
            if (length($0) < 66) next
            n++
            if ((substr($0, 61, 6) + 0) != 0) found_nonzero = 1
        }
        END {
            if (n == 0) exit 1
            exit (found_nonzero ? 1 : 0)
        }
    ' "$pdb_path"
}

_log_abb2_sanity_skip() {
    local stem="$1"
    local log_file="$2"
    echo "$stem" >> "$log_file"
}

declare -A ABB2_SKIP_LOOKUP=()
ABB2_SKIP_STEMS_CSV=""

_build_abb2_sanity_skip_set() {
    local dir="$1"
    local log_file="$2"
    ABB2_SKIP_LOOKUP=()
    ABB2_SKIP_STEMS_CSV=""
    : > "$log_file"
    local pdb stem
    while IFS= read -r -d '' pdb; do
        stem="${pdb##*/}"
        stem="${stem%.pdb}"
        if ! _is_pipeline_pdb_stem "$stem"; then
            continue
        fi
        if _abb2_bfactor_all_zero "$pdb"; then
            ABB2_SKIP_LOOKUP[$stem]=1
            _log_abb2_sanity_skip "$stem" "$log_file"
        fi
    done < <(find "$dir" -maxdepth 1 -name "*.pdb" -type f ! -path '*/.*/*' ! -name "*_H.pdb" ! -name "*_L.pdb" -print0)
    if [[ ${#ABB2_SKIP_LOOKUP[@]} -gt 0 ]]; then
        local stems=()
        for stem in "${!ABB2_SKIP_LOOKUP[@]}"; do
            stems+=("$stem")
        done
        ABB2_SKIP_STEMS_CSV="$(printf '%s,' "${stems[@]}")"
    fi
}

_is_abb2_sanity_skipped_stem() {
    local stem="$1"
    [[ -n "${ABB2_SKIP_LOOKUP[$stem]:-}" ]]
}

_validate_descriptor_outputs_for_csv() {
    local csv_path="$1"
    local results_dir="$2"
    if [[ -z "$csv_path" || ! -f "$csv_path" ]]; then
        return 0
    fi
    if [[ ! -d "$results_dir" ]]; then
        echo "Error: descriptor results directory not found: $results_dir" >&2
        exit 1
    fi

    local skip_csv="${ABB2_SKIP_STEMS_CSV:-}"
    if ! "${PY_ARR[@]}" - "$csv_path" "$results_dir" "$skip_csv" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
skip_names = {
    name.strip()
    for name in sys.argv[3].split(",")
    if name.strip()
}

with csv_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    if "name" not in (reader.fieldnames or []):
        raise SystemExit(f"CSV has no name column: {csv_path}")
    expected = [
        (row.get("name") or "").strip()
        for row in reader
        if (row.get("name") or "").strip()
    ]

missing = []
for name in expected:
    if name in skip_names:
        continue
    json_path = results_dir / f"{name}.json"
    if not json_path.is_file() or json_path.stat().st_size == 0:
        missing.append(name)

if missing:
    preview = ", ".join(missing[:10])
    extra = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
    raise SystemExit(
        f"Missing descriptor JSON for {len(missing)} CSV name(s) in {results_dir}: "
        f"{preview}{extra}"
    )
PY
    then
        exit 1
    fi
}

ALLOWED_NAMES=()
if [[ -n "$NAMES_FROM_CSV" ]]; then
    if [[ ! -f "$NAMES_FROM_CSV" ]]; then
        echo "Error: --names-from-csv file not found: $NAMES_FROM_CSV" >&2
        exit 1
    fi
    mapfile -t ALLOWED_NAMES < <("${PY_ARR[@]}" -c 'import csv, sys
with open(sys.argv[1], newline="") as handle:
    reader = csv.DictReader(handle)
    if "name" not in (reader.fieldnames or []):
        raise SystemExit(f"CSV has no name column: {sys.argv[1]}")
    for row in reader:
        name = (row.get("name") or "").strip()
        if name:
            print(name)' "$NAMES_FROM_CSV")
    if [[ ${#ALLOWED_NAMES[@]} -eq 0 ]]; then
        echo "Error: no names found in --names-from-csv: $NAMES_FROM_CSV" >&2
        exit 1
    fi
fi
export ALLOWED_NAMES
if [[ ${#ALLOWED_NAMES[@]} -gt 0 ]]; then
    ALLOWED_NAMES_CSV="$(printf '%s,' "${ALLOWED_NAMES[@]}")"
else
    ALLOWED_NAMES_CSV=""
fi
export ALLOWED_NAMES_CSV

_is_pipeline_pdb_stem() {
    local stem="$1"
    case "$stem" in
        *_full_atom_sasa|*_H_chain|*_L_chain|*_H|*_L) return 1 ;;
    esac
    if [[ -n "$ALLOWED_NAMES_CSV" ]]; then
        local name
        for name in "${ALLOWED_NAMES[@]}"; do
            [[ "$stem" == "$name" ]] && return 0
        done
        return 1
    fi
    return 0
}

pipeline_pdb_paths() {
    local dir="$1"
    # Top-level antibody PDBs only; skip hidden staging dirs and pipeline artifacts.
    find "$dir" -maxdepth 1 -name "*.pdb" -type f ! -path '*/.*/*' ! -name "*_H.pdb" ! -name "*_L.pdb" -print0 | while IFS= read -r -d '' pdb; do
        local stem="${pdb##*/}"
        stem="${stem%.pdb}"
        if _is_pipeline_pdb_stem "$stem"; then
            if [[ "$SANITY_CHECK_ABB2" == true ]] && _is_abb2_sanity_skipped_stem "$stem"; then
                continue
            fi
            printf '%s\0' "$pdb"
        fi
    done
}

count_pipeline_pdbs() {
    local n=0 _p
    while IFS= read -r -d '' _p; do
        n=$((n + 1))
    done < <(pipeline_pdb_paths "$1")
    echo "$n"
}

USE_GNU_PARALLEL=false
command -v parallel &>/dev/null && USE_GNU_PARALLEL=true

RUN_START=$(date +%s)
TOTAL_STRUCTURES=0

PIPELINE_ORDER=(dssp propka freesasa developability)

for STRUCTURES_DIR in "${STRUCTURES_DIRS[@]}"; do
    STRUCTURES_DIR="$(cd "$STRUCTURES_DIR" && pwd)"
    BASE_NAME="$(basename "$STRUCTURES_DIR")"

    DATASET_DESCRIPTOR_DIR="${DESCRIPTOR_ROOT}/${BASE_NAME}"
    DSSP_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/dssp"
    PROPKA_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/propka"
    SASA_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/sasa"
    DEV_JSON_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/results"
    mkdir -p "$DSSP_OUTPUT_DIR" "$PROPKA_OUTPUT_DIR" "$SASA_OUTPUT_DIR" "$DEV_JSON_OUTPUT_DIR"

    if [[ "$SANITY_CHECK_ABB2" == true ]]; then
        ABB2_SANITY_SKIP_LOG="${DATASET_DESCRIPTOR_DIR}/abb2_sanity_skip.log"
        _build_abb2_sanity_skip_set "$STRUCTURES_DIR" "$ABB2_SANITY_SKIP_LOG"
        export ABB2_SANITY_SKIP_LOG ABB2_SKIP_STEMS_CSV
    else
        ABB2_SKIP_LOOKUP=()
        ABB2_SKIP_STEMS_CSV=""
        unset ABB2_SANITY_SKIP_LOG
    fi

    N_PDB=$(count_pipeline_pdbs "$STRUCTURES_DIR")
    TOTAL_STRUCTURES=$((TOTAL_STRUCTURES + N_PDB))

    for MODE in "${PIPELINE_ORDER[@]}"; do
    case "$MODE" in
        dssp)
            OUTPUT_DIR="$DSSP_OUTPUT_DIR"
            export OUTPUT_DIR
            if [ "$USE_GNU_PARALLEL" = true ]; then
                process_file() {
                    local pdb_file="$1"
                    local basename=$(basename "$pdb_file" .pdb)
                    local filename=$(basename "$pdb_file")
                    local output_file="${OUTPUT_DIR}/${basename}.dssp"
                    local temp_pdb=$(mktemp)
                    {
                        printf "REMARK    @%s (1-2)\n" "$filename"
                        echo "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          "
                        awk '!/^REMARK/ && !/^CRYST1/' "$pdb_file"
                    } > "$temp_pdb"
                    if ! _fastab_run "$DSSP_BIN" "$temp_pdb" "$output_file" &>/dev/null; then
                        echo "✗ $basename (failed)"
                        rm -f "$output_file"
                    elif [[ ! -s "$output_file" ]]; then
                        echo "✗ $basename (no DSSP output)"
                        rm -f "$output_file"
                    fi
                    rm -f "$temp_pdb"
                }
                export -f process_file _fastab_run
                export DSSP_BIN FASTAB_ENV
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel -0 -j "$NUM_JOBS" process_file {}
            else
                export STRUCTURES_DIR DSSP_BIN FASTAB_ENV
                "${PY_ARR[@]}" << 'DSSP_PY'
import os
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")
DSSP_BIN = os.environ.get("DSSP_BIN", "mkdssp")
FASTAB_ENV = os.environ.get("FASTAB_ENV", "fastab")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def run_in_fastab(args, **kwargs):
    for conda in ("mamba", "conda"):
        if shutil.which(conda):
            return subprocess.run([conda, "run", "-n", FASTAB_ENV, *args], **kwargs)
    return subprocess.run(args, **kwargs)

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    filename = pdb_file.name
    output_file = Path(OUTPUT_DIR) / f"{basename}.dssp"
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tf:
            tf.write(f"REMARK    @{filename} (1-2)\n")
            tf.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n")
            with open(pdb_file) as f:
                for line in f:
                    if not line.startswith('REMARK') and not line.startswith('CRYST1'):
                        tf.write(line)
            path = tf.name
        r = run_in_fastab([DSSP_BIN, path, str(output_file)], capture_output=True, text=True, cwd=SCRIPT_DIR)
        Path(path).unlink(missing_ok=True)
        if r.returncode != 0:
            output_file.unlink(missing_ok=True)
            return f"✗ {basename} (failed)"
        if not output_file.is_file() or output_file.stat().st_size == 0:
            output_file.unlink(missing_ok=True)
            return f"✗ {basename} (no DSSP output)"
        return None
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:50]})"

allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
            continue
        if allowed_names and stem not in allowed_names:
            continue
        if stem in abb2_skip_stems:
            continue
        out.append(p)
    return out

pdb_files = pipeline_pdb_files(STRUCTURES_DIR)
with Pool(num_jobs) as pool:
    results = pool.map(process_file, pdb_files)
for r in results:
    if r:
        print(r)
failed = sum(1 for r in results if r)
if failed:
    print(f"Completed: {len(pdb_files) - failed}/{len(pdb_files)} successful ({failed} failed)")
DSSP_PY
            fi
            ;;

        freesasa)
            SASA_DIR="$SASA_OUTPUT_DIR"
            export SASA_DIR
            if [ "$USE_GNU_PARALLEL" = true ]; then
                process_file() {
                    local pdbfile="$1"
                    local filename
                    filename=$(basename "$pdbfile" .pdb)
                    local sasa_full="${SASA_DIR}/${filename}_full.sasa"
                    local sasa_H="${SASA_DIR}/${filename}_H_full.sasa"
                    local sasa_L="${SASA_DIR}/${filename}_L_full.sasa"
                    local tmp_H tmp_L
                    tmp_H=$(mktemp)
                    tmp_L=$(mktemp)
                    awk -v h="$tmp_H" -v l="$tmp_L" '{
                        if ($1 == "ATOM" || $1 == "HETATM") {
                            chain = substr($0, 22, 1);
                            if (chain == "H") print > h;
                            if (chain == "L") print > l;
                        } else {
                            print > h;
                            print > l;
                        }
                    }' "$pdbfile"

                    if ! _fastab_run freesasa --shrake-rupley --format=rsa --depth=residue "$pdbfile" > "$sasa_full" 2>/dev/null; then
                        echo "✗ $filename (full failed)"
                        rm -f "$sasa_full"
                    fi

                    if grep -q "^ATOM" "$tmp_H"; then
                        if ! _fastab_run freesasa --shrake-rupley --format=rsa --depth=residue "$tmp_H" > "$sasa_H" 2>/dev/null; then
                            echo "✗ $filename (H-only failed)"
                            rm -f "$sasa_H"
                        fi
                    fi

                    if grep -q "^ATOM" "$tmp_L"; then
                        if ! _fastab_run freesasa --shrake-rupley --format=rsa --depth=residue "$tmp_L" > "$sasa_L" 2>/dev/null; then
                            echo "✗ $filename (L-only failed)"
                            rm -f "$sasa_L"
                        fi
                    fi

                    rm -f "$tmp_H" "$tmp_L"
                }
                export -f process_file _fastab_run
                export SASA_DIR FASTAB_ENV
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel -0 -j "$NUM_JOBS" process_file {}
            else
                export STRUCTURES_DIR SASA_DIR FASTAB_ENV
                "${PY_ARR[@]}" << 'FREESASA_PY'
import os
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
SASA_DIR = os.environ.get("SASA_DIR", ".")
FASTAB_ENV = os.environ.get("FASTAB_ENV", "fastab")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def run_in_fastab(args, **kwargs):
    for conda in ("mamba", "conda"):
        if shutil.which(conda):
            return subprocess.run([conda, "run", "-n", FASTAB_ENV, *args], **kwargs)
    return subprocess.run(args, **kwargs)

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    sasa_full = Path(SASA_DIR) / f"{basename}_full.sasa"
    sasa_H = Path(SASA_DIR) / f"{basename}_H_full.sasa"
    sasa_L = Path(SASA_DIR) / f"{basename}_L_full.sasa"

    with open(pdb_file) as src, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp_H, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp_L:
        tmp_H_path = Path(tmp_H.name)
        tmp_L_path = Path(tmp_L.name)
        for line in src:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                chain = line[21]
                if chain == "H":
                    tmp_H.write(line)
                if chain == "L":
                    tmp_L.write(line)
            else:
                tmp_H.write(line)
                tmp_L.write(line)

    errors = []

    r = run_in_fastab(
        ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(pdb_file)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        errors.append(f"✗ {basename} (full failed)")
        sasa_full.unlink(missing_ok=True)
    else:
        sasa_full.write_text(r.stdout)
        if sasa_full.stat().st_size == 0:
            errors.append(f"✗ {basename} (full failed)")
            sasa_full.unlink(missing_ok=True)

    has_H_atoms = any(l.startswith("ATOM") for l in tmp_H_path.read_text().splitlines())
    if has_H_atoms:
        r = run_in_fastab(
            ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(tmp_H_path)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            errors.append(f"✗ {basename} (H-only failed)")
        else:
            sasa_H.write_text(r.stdout)

    has_L_atoms = any(l.startswith("ATOM") for l in tmp_L_path.read_text().splitlines())
    if has_L_atoms:
        r = run_in_fastab(
            ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(tmp_L_path)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            errors.append(f"✗ {basename} (L-only failed)")
        else:
            sasa_L.write_text(r.stdout)

    try:
        tmp_H_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        tmp_L_path.unlink(missing_ok=True)
    except Exception:
        pass

    return "; ".join(errors) if errors else None

allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
            continue
        if allowed_names and stem not in allowed_names:
            continue
        if stem in abb2_skip_stems:
            continue
        out.append(p)
    return out

pdb_files = pipeline_pdb_files(STRUCTURES_DIR)
with Pool(num_jobs) as pool:
    results = pool.map(process_file, pdb_files)
for r in results:
    if r:
        print(r)
failed = sum(1 for r in results if r)
if failed:
    print(f"Completed: {len(pdb_files) - failed}/{len(pdb_files)} successful ({failed} failed)")
FREESASA_PY
            fi
            ;;

        propka)
            OUTPUT_DIR="$PROPKA_OUTPUT_DIR"
            export OUTPUT_DIR
            PROPKA_TMP_STRUCTURES="${OUTPUT_DIR}/tmp_structures"
            mkdir -p "$PROPKA_TMP_STRUCTURES"
            export PROPKA_TMP_STRUCTURES
            if [ "$USE_GNU_PARALLEL" = true ]; then
                process_file() {
                    local pdb_file="$1"
                    local basename=$(basename "$pdb_file" .pdb)
                    local output_dir="$2"
                    local tmp_structures="${output_dir}/tmp_structures"
                    mkdir -p "$tmp_structures"
                    cd "$output_dir"
                    local propka_input propka_stem
                    propka_input=$(cd "$SCRIPT_DIR" && _fastab_run python utils/prepare_propka_input.py "$pdb_file" \
                        --tmp-dir "$tmp_structures" --stem "${basename}_full" 2>/dev/null | head -1)
                    propka_stem=$(basename "$propka_input" .pdb)
                    _fastab_run propka3 "$propka_input" > "${basename}_full.log" 2>&1
                    if ! _finalize_propka_pka "${propka_stem}.pka" "${basename}_full.pka"; then echo "✗ $basename (full - failed)"; fi
                }
                export -f process_file _fastab_run _finalize_propka_pka
                export FASTAB_ENV SCRIPT_DIR
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel -0 -j "$NUM_JOBS" process_file {} "$OUTPUT_DIR"
            else
                export STRUCTURES_DIR FASTAB_ENV PY SCRIPT_DIR
                "${PY_ARR[@]}" << 'PROPKA_PY'
import os
import shutil
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
FASTAB_ENV = os.environ.get("FASTAB_ENV", "fastab")
SCRIPT_DIR = Path(os.environ.get("SCRIPT_DIR", "."))
TMP_STRUCTURES = Path(OUTPUT_DIR) / "tmp_structures"
TMP_STRUCTURES.mkdir(parents=True, exist_ok=True)
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def run_in_fastab(args, **kwargs):
    for conda in ("mamba", "conda"):
        if shutil.which(conda):
            return subprocess.run([conda, "run", "-n", FASTAB_ENV, *args], **kwargs)
    return subprocess.run(args, **kwargs)

def prepare_propka_input(source_pdb, stem):
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "utils" / "prepare_propka_input.py"),
        str(source_pdb),
        "--tmp-dir",
        str(TMP_STRUCTURES),
        "--stem",
        stem,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout or f"prepare_propka_input failed for {source_pdb}")
    propka_input = r.stdout.strip().splitlines()[0].strip()
    if not propka_input:
        raise RuntimeError(f"prepare_propka_input returned no PDB path for {source_pdb}")
    return Path(propka_input)

def run_propka_variant(pdb_file, basename, label, stem):
    out = []
    output_dir = Path(OUTPUT_DIR)
    propka_input = prepare_propka_input(pdb_file, stem)
    propka_stem = propka_input.stem
    log_out = output_dir / f"{basename}_{label}.log"
    pka_out = output_dir / f"{basename}_{label}.pka"
    r = run_in_fastab(["propka3", str(propka_input)], capture_output=True, text=True, cwd=output_dir)
    with open(log_out, "w") as f:
        f.write(r.stdout)
        f.write(r.stderr)
    generated = output_dir / f"{propka_stem}.pka"
    if generated.exists():
        if generated != pka_out:
            generated.rename(pka_out)
    else:
        out.append(f"✗ {basename} ({label} - failed)")
    return out

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    try:
        out = run_propka_variant(pdb_file, basename, "full", f"{basename}_full")
        return "\n".join(out) if out else None
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:80]})"

allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
            continue
        if allowed_names and stem not in allowed_names:
            continue
        if stem in abb2_skip_stems:
            continue
        out.append(p)
    return out

pdb_files = pipeline_pdb_files(STRUCTURES_DIR)
with Pool(num_jobs) as pool:
    results = pool.map(process_file, pdb_files)
for r in results:
    if r:
        for line in r.split("\n"):
            if line.strip():
                print(line)
propka_failed = sum(1 for r in results if r)
if propka_failed:
    print(f"Completed: {len(pdb_files) - propka_failed}/{len(pdb_files)} successful ({propka_failed} failed)")
PROPKA_PY
            fi
            ;;

        developability)
            SASA_DIR="$SASA_OUTPUT_DIR"
            DSSP_DIR="$DSSP_OUTPUT_DIR"
            PKA_DIR="$PROPKA_OUTPUT_DIR"
            OUTPUT_DIR="$DEV_JSON_OUTPUT_DIR"
            export STRUCTURES_DIR SASA_DIR DSSP_DIR PKA_DIR OUTPUT_DIR

            if [ "$USE_GNU_PARALLEL" = true ]; then
                process_file() {
                    local pdb_file="$1"
                    local basename=$(basename "$pdb_file" .pdb)
                    local sasa_file="${SASA_DIR}/${basename}_full.sasa"
                    local dssp_file="${DSSP_DIR}/${basename}.dssp"
                    local pka_file="${PKA_DIR}/${basename}_full.pka"
                    local output_file="${OUTPUT_DIR}/${basename}.json"
                    local -a cmd=(python developability/calculate_descriptors.py "$pdb_file" "$sasa_file")
                    if [ ! -f "$sasa_file" ]; then
                        echo "✗ $basename (SASA file not found)"
                        return
                    fi
                    [ -f "$dssp_file" ] && cmd+=("--dssp-file" "$dssp_file")
                    [ -f "$pka_file" ] && cmd+=("--pka-file" "$pka_file")
                    cmd+=("--pH" "$PH_VALUE")
                    cmd+=("--output" "$output_file")
                    if ! output=$(cd "$SCRIPT_DIR" && _fastab_run "${cmd[@]}" 2>&1); then
                        echo "✗ $basename (failed)"
                        printf '%s\n' "$output"
                    elif [[ -s "$output_file" ]]; then
                        _remove_helper_outputs_for_stem "$basename"
                    fi
                }
                export -f process_file _fastab_run _remove_helper_outputs_for_stem
                export SASA_DIR DSSP_DIR PKA_DIR OUTPUT_DIR PH_VALUE SCRIPT_DIR FASTAB_ENV REMOVE_HELPER_OUTPUTS
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel -0 -j "$NUM_JOBS" process_file {}
            else
                export PY REMOVE_HELPER_OUTPUTS
                "${PY_ARR[@]}" << 'DEV_PY'
import os
import shlex
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
SASA_DIR = os.environ.get("SASA_DIR", ".")
DSSP_DIR = os.environ.get("DSSP_DIR", ".")
PKA_DIR = os.environ.get("PKA_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")
num_jobs = int(os.environ.get('NUM_JOBS', cpu_count()))
ph_value = os.environ.get('PH_VALUE', '7.5')
py_cmd = shlex.split(os.environ.get("PY", "python3"))
remove_helper_outputs = os.environ.get("REMOVE_HELPER_OUTPUTS", "").lower() in ("1", "true", "yes")

def remove_helper_outputs_for_stem(basename):
    if not remove_helper_outputs:
        return
    for path in (
        Path(DSSP_DIR) / f"{basename}.dssp",
        Path(PKA_DIR) / f"{basename}_full.pka",
        Path(PKA_DIR) / f"{basename}_full.log",
        Path(SASA_DIR) / f"{basename}_full.sasa",
        Path(SASA_DIR) / f"{basename}_H_full.sasa",
        Path(SASA_DIR) / f"{basename}_L_full.sasa",
    ):
        path.unlink(missing_ok=True)

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    sasa_file = Path(SASA_DIR) / f"{basename}_full.sasa"
    dssp_file = Path(DSSP_DIR) / f"{basename}.dssp"
    pka_file = Path(PKA_DIR) / f"{basename}_full.pka"
    output_file = Path(OUTPUT_DIR) / f"{basename}.json"
    if not sasa_file.exists():
        return f"✗ {basename} (SASA file not found)"
    cmd = py_cmd + ["developability/calculate_descriptors.py", str(pdb_file), str(sasa_file)]
    if dssp_file.exists():
        cmd.extend(["--dssp-file", str(dssp_file)])
    if pka_file.exists():
        cmd.extend(["--pka-file", str(pka_file)])
    cmd.extend(["--pH", ph_value])
    cmd.extend(["--output", str(output_file)])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
        if r.returncode != 0:
            return f"✗ {basename} (failed)\n{r.stderr}"
        if output_file.is_file() and output_file.stat().st_size > 0:
            remove_helper_outputs_for_stem(basename)
        return None
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:200]})"

allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
            continue
        if allowed_names and stem not in allowed_names:
            continue
        if stem in abb2_skip_stems:
            continue
        out.append(p)
    return out

pdb_files = pipeline_pdb_files(STRUCTURES_DIR)
with Pool(num_jobs) as pool:
    results = pool.map(process_file, pdb_files)
for r in results:
    if r:
        print(r)
failed = sum(1 for r in results if r)
if failed:
    print(f"Completed: {len(pdb_files) - failed}/{len(pdb_files)} successful ({failed} failed)")
DEV_PY
            fi
            ;;
    esac
    done

    if [[ -n "$NAMES_FROM_CSV" ]]; then
        _validate_descriptor_outputs_for_csv "$NAMES_FROM_CSV" "$DEV_JSON_OUTPUT_DIR"
    fi

    _clean_external_output_dirs
done

ELAPSED=$(( $(date +%s) - RUN_START ))
echo ""
echo "--- Summary ---"
echo "Structures processed: $TOTAL_STRUCTURES"
printf "Total time: %02d:%02d:%02d\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
