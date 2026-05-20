#!/bin/bash
# Full developability descriptor pipeline (parallel over PDBs): DSSP -> PropKa -> freesasa -> developability.
# Usage: ./run_parallel.sh [--output-dir DIR] STRUCTURES_DIR [STRUCTURES_DIR ...] [num_jobs] [options]
#    or: ./run_parallel.sh [--output-dir DIR] --parent-dir DIR [more --parent-dir DIR ...] [STRUCTURES_DIR ...] [num_jobs] [options]
#
# STRUCTURES_DIR    One or more folders containing PDB files (processed in sequence; each folder parallelized internally).
#                   Relative paths and globs (e.g. structures/*abb3*) resolve from your cwd first; if that finds nothing,
#                   the script retries under the repository root (parent of src/), so running from src/ still works.
# --parent-dir DIR  Each immediate subdirectory of DIR that contains at least one *.pdb is treated as a STRUCTURES_DIR
#                   (non-recursive: only DIR/<name>/). Hidden directories (name starting with .) are skipped.
# num_jobs          Optional. Default: nproc
#
# Output layout (default: under your shell cwd when you launched the script, before cd to repo root):
#   ./developability_descriptors/${BASE}/{dssp,propka,sasa,results}/
#   Override the parent of per-dataset folders with --output-dir DIR -> DIR/${BASE}/{dssp,propka,sasa,results}/
#   dssp/propka/sasa: tool outputs (same filenames as before, e.g. *.dssp, *_full.sasa, *_full.pka).
#   results/: developability JSON per structure (same as former ${BASE}_results/*.json).
#
# Developability-only options (after num_jobs): --pH <value>
# --force-sasa   Recompute freesasa outputs even if they already exist (default: skip when all 4 outputs present)
#
# Examples:
#   ./run_parallel.sh ./ab21 ./pdgf38 8
#   ./run_parallel.sh ./garbinski2023 4
#   ./run_parallel.sh --parent-dir ./structures
#   ./run_parallel.sh --parent-dir /storage/antibody_data/PairedStructures/FASTAb/structures 16
#   ./run_parallel.sh --output-dir ./my_descriptor_runs ./ab21
#   bash src/run_parallel.sh 'structures/*abb3*'   # quoted glob: expand here if shell did not

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

EXPLICIT_STRUCTURES=()
PARENT_DIR_INPUTS=()
NUM_JOBS=""
EXTRA_ARGS=()
PH_VALUE="7.4"
USER_OUTPUT_DESCRIPTOR_ROOT=""
FORCE_SASA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-sasa)
            FORCE_SASA=true
            shift
            ;;
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
            PH_VALUE="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
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
    if [[ -d "$arg" ]]; then
        STRUCTURES_DIRS+=("$arg")
        continue
    fi
    # Relative path without glob: allow repo-root resolution (e.g. run from src/).
    if [[ "$arg" != /* && "$arg" != *[\*\?\[]* && -d "$PROJECT_ROOT/$arg" ]]; then
        STRUCTURES_DIRS+=("$PROJECT_ROOT/$arg")
        continue
    fi
    # Unmatched shell glob is passed literally; expand with compgen from cwd, then from repo root.
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
    echo "  Runs dssp -> propka -> freesasa -> developability for each folder (default: ./developability_descriptors/<basename>/)." >&2
    echo "  --output-dir DIR: write per-dataset outputs under DIR/<basename>/ instead of ./developability_descriptors/<basename>/." >&2
    echo "  --parent-dir: run on every immediate subdirectory of DIR that contains at least one .pdb file." >&2
    exit 1
fi

# Deduplicate by resolved path (preserve first-seen order).
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
# Outputs default next to your shell cwd (not next to each PDB tree); captured before cd to repo root.
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
export NUM_JOBS SCRIPT_DIR PROJECT_ROOT RUN_INVOCATION_DIR DESCRIPTOR_ROOT
export PH_VALUE
export EXTRA_ARGS_STR="${EXTRA_ARGS[*]}"
export FORCE_SASA

USE_GNU_PARALLEL=false
command -v parallel &>/dev/null && USE_GNU_PARALLEL=true

# PropKa conda environment (edit if needed)
PROPKA_CONDA_ACTIVATE="${PROPKA_CONDA_ACTIVATE:-source /home/kb/miniforge3/bin/activate developability}"

RUN_START=$(date +%s)
TOTAL_STRUCTURES=0

# Fixed pipeline order (developability consumes outputs from the same run).
PIPELINE_ORDER=(dssp propka freesasa developability)

for STRUCTURES_DIR in "${STRUCTURES_DIRS[@]}"; do
    STRUCTURES_DIR="$(cd "$STRUCTURES_DIR" && pwd)"
    BASE_NAME="$(basename "$STRUCTURES_DIR")"
    N_PDB=$(find "$STRUCTURES_DIR" -name "*.pdb" -type f | wc -l)
    TOTAL_STRUCTURES=$((TOTAL_STRUCTURES + N_PDB))

    DATASET_DESCRIPTOR_DIR="${DESCRIPTOR_ROOT}/${BASE_NAME}"
    DSSP_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/dssp"
    PROPKA_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/propka"
    SASA_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/sasa"
    DEV_JSON_OUTPUT_DIR="${DATASET_DESCRIPTOR_DIR}/results"
    mkdir -p "$DSSP_OUTPUT_DIR" "$PROPKA_OUTPUT_DIR" "$SASA_OUTPUT_DIR" "$DEV_JSON_OUTPUT_DIR"

    for MODE in "${PIPELINE_ORDER[@]}"; do
    case "$MODE" in
        dssp)
            OUTPUT_DIR="$DSSP_OUTPUT_DIR"
            mkdir -p "$OUTPUT_DIR"
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
                    if ! dssp "$temp_pdb" "$output_file" &>/dev/null; then
                        echo "✗ $basename (failed)"
                    fi
                    rm -f "$temp_pdb"
                }
                export -f process_file
                # Only full-structure PDBs (exclude single-chain *_H.pdb, *_L.pdb)
                find "$STRUCTURES_DIR" -name "*.pdb" -type f ! -name "*_H.pdb" ! -name "*_L.pdb" | parallel -j "$NUM_JOBS" process_file {}
            else
                export STRUCTURES_DIR
                python3 << 'DSSP_PY'
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

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
        r = subprocess.run(["dssp", path, str(output_file)], capture_output=True, text=True, cwd=SCRIPT_DIR)
        Path(path).unlink(missing_ok=True)
        if r.returncode != 0:
            return f"✗ {basename} (failed)"
        return None
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:50]})"

# Only full-structure PDBs (exclude single-chain *_H.pdb, *_L.pdb)
pdb_files = sorted(p for p in Path(STRUCTURES_DIR).rglob("*.pdb") if not (p.stem.endswith("_H") or p.stem.endswith("_L")))
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
            mkdir -p "$SASA_DIR"
            export SASA_DIR
            if [ "$USE_GNU_PARALLEL" = true ]; then
                # Full PDBs only: H/L *_full.sasa for inter_chain_buried_sasa are built inside each job (tmp split), not from *_H.pdb on disk.
                find "$STRUCTURES_DIR" -name "*.pdb" -type f ! -name "*_H.pdb" ! -name "*_L.pdb" | parallel -j "$NUM_JOBS" '
                    pdbfile={}
                    filename=$(basename "$pdbfile" .pdb)
                    sasa_full="'"$SASA_DIR"'/${filename}_full.sasa"
                    atom_sasa_full="'"$SASA_DIR"'/${filename}_full_atom_sasa.pdb"
                    sasa_H="'"$SASA_DIR"'/${filename}_H_full.sasa"
                    sasa_L="'"$SASA_DIR"'/${filename}_L_full.sasa"

                    # If all required outputs already exist, skip work for this PDB
                    # (unless --force-sasa was passed).
                    if [ "$FORCE_SASA" != "true" ] && [ -f "$sasa_full" ] && [ -f "$atom_sasa_full" ] && [ -f "$sasa_H" ] && [ -f "$sasa_L" ]; then
                        exit 0
                    fi

                    tmp_H=$(mktemp)
                    tmp_L=$(mktemp)

                    # Build H-only and L-only PDBs once (preserve non-ATOM/HETATM records)
                    awk '"'"'{
                        if ($1 == "ATOM" || $1 == "HETATM") {
                            chain = substr($0, 22, 1);
                            if (chain == "H") print > "'"'"'"$tmp_H"'"'"'";
                            if (chain == "L") print > "'"'"'"$tmp_L"'"'"'";
                        } else {
                            print > "'"'"'"$tmp_H"'"'"'";
                            print > "'"'"'"$tmp_L"'"'"'";
                        }
                    }'"'"' "$pdbfile"

                    if [ ! -f "$sasa_full" ]; then
                        if ! freesasa --shrake-rupley --format=rsa --depth=residue "$pdbfile" > "$sasa_full" 2>/dev/null; then
                            echo "✗ $filename (full failed)"
                            rm -f "$sasa_full"
                        fi
                    fi

                    # Full complex atom-level SASA in compact PDB form:
                    # keep only what we need later (atom serial -> absolute SASA from B-factor).
                    # Include hydrogens because the future exposed-atom electrostatic
                    # patch proxy should account for real proton/terminus atoms too.
                    if [ ! -f "$atom_sasa_full" ]; then
                        if ! freesasa --shrake-rupley --format=pdb --depth=atom --hydrogen "$pdbfile" > "$atom_sasa_full" 2>/dev/null; then
                            echo "✗ $filename (full atom SASA failed)"
                            rm -f "$atom_sasa_full"
                        fi
                    fi

                    # Heavy-chain-only SASA (if we actually have any H atoms)
                    if [ ! -f "$sasa_H" ] && grep -q "^ATOM" "$tmp_H"; then
                        if ! freesasa --shrake-rupley --format=rsa --depth=residue "$tmp_H" > "$sasa_H" 2>/dev/null; then
                            echo "✗ $filename (H-only failed)"
                            rm -f "$sasa_H"
                        fi
                    fi

                    # Light-chain-only SASA (if we actually have any L atoms)
                    if [ ! -f "$sasa_L" ] && grep -q "^ATOM" "$tmp_L"; then
                        if ! freesasa --shrake-rupley --format=rsa --depth=residue "$tmp_L" > "$sasa_L" 2>/dev/null; then
                            echo "✗ $filename (L-only failed)"
                            rm -f "$sasa_L"
                        fi
                    fi

                    rm -f "$tmp_H" "$tmp_L"
                '
            else
                export STRUCTURES_DIR
                python3 << 'FREESASA_PY'
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
SASA_DIR = os.environ.get("SASA_DIR", ".")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    sasa_full = Path(SASA_DIR) / f"{basename}_full.sasa"
    atom_sasa_full = Path(SASA_DIR) / f"{basename}_full_atom_sasa.pdb"
    sasa_H = Path(SASA_DIR) / f"{basename}_H_full.sasa"
    sasa_L = Path(SASA_DIR) / f"{basename}_L_full.sasa"

    force_sasa = os.environ.get("FORCE_SASA", "false").lower() == "true"
    if not force_sasa and sasa_full.exists() and atom_sasa_full.exists() and sasa_H.exists() and sasa_L.exists():
        return None

    # Build temporary H-only and L-only PDBs once
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

    if not sasa_full.exists():
        r = subprocess.run(
            ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(pdb_file)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            errors.append(f"✗ {basename} (full failed)")
        else:
            sasa_full.write_text(r.stdout)

    # Full complex atom-level SASA in compact PDB form:
    # keep only what we need later (atom serial -> absolute SASA from B-factor).
    # Include hydrogens because the future exposed-atom electrostatic
    # patch proxy should account for real proton/terminus atoms too.
    if not atom_sasa_full.exists():
        r = subprocess.run(
            ["freesasa", "--shrake-rupley", "--format=pdb", "--depth=atom", "--hydrogen", str(pdb_file)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            errors.append(f"✗ {basename} (full atom SASA failed)")
        else:
            atom_sasa_full.write_text(r.stdout)

    # Heavy-chain-only SASA (only if there are any ATOM records for H)
    has_H_atoms = False
    with open(tmp_H_path) as f:
        for l in f:
            if l.startswith("ATOM"):
                has_H_atoms = True
                break
    if not sasa_H.exists() and has_H_atoms:
        r = subprocess.run(
            ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(tmp_H_path)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            errors.append(f"✗ {basename} (H-only failed)")
        else:
            sasa_H.write_text(r.stdout)

    # Light-chain-only SASA (only if there are any ATOM records for L)
    has_L_atoms = False
    with open(tmp_L_path) as f:
        for l in f:
            if l.startswith("ATOM"):
                has_L_atoms = True
                break
    if not sasa_L.exists() and has_L_atoms:
        r = subprocess.run(
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

pdb_files = sorted(p for p in Path(STRUCTURES_DIR).rglob("*.pdb") if not (p.stem.endswith("_H") or p.stem.endswith("_L")))
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
            mkdir -p "$OUTPUT_DIR"
            export OUTPUT_DIR PROPKA_CONDA_ACTIVATE
            if [ "$USE_GNU_PARALLEL" = true ]; then
                process_file() {
                    local pdb_file="$1"
                    local basename=$(basename "$pdb_file" .pdb)
                    local output_dir="$2"
                    eval "$PROPKA_CONDA_ACTIVATE"
                    cd "$output_dir"
                    propka3 "$pdb_file" > "${basename}_full.log" 2>&1
                    if [ -f "${basename}.pka" ]; then mv "${basename}.pka" "${basename}_full.pka"; else echo "✗ $basename (full - failed)"; fi
                    grep -E "^REMARK|^HEADER|^TITLE|^COMPND|^SOURCE" "$pdb_file" > "${basename}_H_chain.pdb.tmp"
                    awk '/^ATOM/ || /^HETATM/ { if (substr($0, 22, 1) == "H") print }' "$pdb_file" >> "${basename}_H_chain.pdb.tmp"
                    echo "END" >> "${basename}_H_chain.pdb.tmp"
                    mv "${basename}_H_chain.pdb.tmp" "${basename}_H_chain.pdb"
                    propka3 "${basename}_H_chain.pdb" > "${basename}_H.log" 2>&1
                    if [ -f "${basename}_H_chain.pka" ]; then mv "${basename}_H_chain.pka" "${basename}_H.pka"; else echo "✗ $basename (H chain - failed)"; fi
                    grep -E "^REMARK|^HEADER|^TITLE|^COMPND|^SOURCE" "$pdb_file" > "${basename}_L_chain.pdb.tmp"
                    awk '/^ATOM/ || /^HETATM/ { if (substr($0, 22, 1) == "L") print }' "$pdb_file" >> "${basename}_L_chain.pdb.tmp"
                    echo "END" >> "${basename}_L_chain.pdb.tmp"
                    mv "${basename}_L_chain.pdb.tmp" "${basename}_L_chain.pdb"
                    propka3 "${basename}_L_chain.pdb" > "${basename}_L.log" 2>&1
                    if [ -f "${basename}_L_chain.pka" ]; then mv "${basename}_L_chain.pka" "${basename}_L.pka"; else echo "✗ $basename (L chain - failed)"; fi
                    rm -f "${basename}_H_chain.pdb" "${basename}_L_chain.pdb"
                }
                export -f process_file
                # Only full-structure PDBs (exclude single-chain *_H.pdb, *_L.pdb)
                find "$STRUCTURES_DIR" -name "*.pdb" -type f ! -name "*_H.pdb" ! -name "*_L.pdb" | parallel -j "$NUM_JOBS" process_file {} "$OUTPUT_DIR"
            else
                export STRUCTURES_DIR NUM_JOBS
                python3 << 'PROPKA_PY'
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))
conda_activate = os.environ.get("PROPKA_CONDA_ACTIVATE", "source /home/kb/miniforge3/bin/activate developability")

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    output_dir = Path(OUTPUT_DIR)
    out = []
    try:
        for label, pdb_in, pka_out, log_out in [
            ("full", pdb_file, output_dir / f"{basename}_full.pka", output_dir / f"{basename}_full.log"),
        ]:
            r = subprocess.run(["bash", "-c", f"{conda_activate} && propka3 '{pdb_in}'"], capture_output=True, text=True, cwd=output_dir)
            with open(log_out, "w") as f:
                f.write(r.stdout)
                f.write(r.stderr)
            pka = output_dir / f"{basename}.pka"
            if pka.exists():
                pka.rename(pka_out)
            else:
                out.append(f"✗ {basename} ({label} - failed)")
        for chain, ch in [("H", "H"), ("L", "L")]:
            chain_pdb = output_dir / f"{basename}_{chain}_chain.pdb"
            with open(chain_pdb, "w") as f:
                with open(pdb_file) as pdb:
                    for line in pdb:
                        if any(line.startswith(x) for x in ["REMARK", "HEADER", "TITLE", "COMPND", "SOURCE"]):
                            f.write(line)
                with open(pdb_file) as pdb:
                    for line in pdb:
                        if line.startswith(("ATOM", "HETATM")) and len(line) > 21 and line[21] == ch:
                            f.write(line)
                f.write("END\n")
            r = subprocess.run(["bash", "-c", f"{conda_activate} && propka3 '{chain_pdb}'"], capture_output=True, text=True, cwd=output_dir)
            with open(output_dir / f"{basename}_{chain}.log", "w") as f:
                f.write(r.stdout)
                f.write(r.stderr)
            pka = output_dir / f"{basename}_{chain}_chain.pka"
            if pka.exists():
                pka.rename(output_dir / f"{basename}_{chain}.pka")
            else:
                out.append(f"✗ {basename} ({chain} chain - failed)")
            chain_pdb.unlink(missing_ok=True)
        return "\n".join(out) if out else None
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:80]})"

# Only full-structure PDBs (exclude single-chain *_H.pdb, *_L.pdb)
pdb_files = sorted(p for p in Path(STRUCTURES_DIR).rglob("*.pdb") if not (p.stem.endswith("_H") or p.stem.endswith("_L")))
with Pool(num_jobs) as pool:
    results = pool.map(process_file, pdb_files)
for r in results:
    if r:
        for line in r.split("\n"):
            if line.strip():
                print(line)
failed = sum(1 for r in results if r)
if failed:
    print(f"Completed: {len(pdb_files)*3 - sum(r and r.count('✗') or 0 for r in results)}/{len(pdb_files)*3} successful")
PROPKA_PY
            fi
            ;;

        developability)
            SASA_DIR="$SASA_OUTPUT_DIR"
            DSSP_DIR="$DSSP_OUTPUT_DIR"
            PKA_DIR="$PROPKA_OUTPUT_DIR"
            OUTPUT_DIR="$DEV_JSON_OUTPUT_DIR"
            mkdir -p "$OUTPUT_DIR"
            export STRUCTURES_DIR SASA_DIR DSSP_DIR PKA_DIR OUTPUT_DIR
            export PH_VALUE EXTRA_ARGS_STR

            if [ "$USE_GNU_PARALLEL" = true ]; then
                process_file() {
                    local pdb_file="$1"
                    local basename=$(basename "$pdb_file" .pdb)
                    local sasa_file="${SASA_DIR}/${basename}_full.sasa"
                    local dssp_file="${DSSP_DIR}/${basename}.dssp"
                    local pka_file="${PKA_DIR}/${basename}_full.pka"
                    local output_file="${OUTPUT_DIR}/${basename}.json"
                    # Use relative path and run from SCRIPT_DIR so we don't duplicate "src/src/..."
                    local cmd=("python3" "developability/run_developability.py" "$pdb_file" "$sasa_file")
                    local extra_args_array=($EXTRA_ARGS_STR)
                    if [ ! -f "$sasa_file" ]; then
                        echo "✗ $basename (SASA file not found)"
                        return
                    fi
                    [ -f "$dssp_file" ] && cmd+=("--dssp-file" "$dssp_file")
                    [ -f "$pka_file" ] && cmd+=("--pka-file" "$pka_file")
                    has_ph=false
                    for arg in "${extra_args_array[@]}"; do
                        [ "$arg" = "--pH" ] && has_ph=true && break
                    done
                    [ "$has_ph" = false ] && cmd+=("--pH" "$PH_VALUE")
                    cmd+=("${extra_args_array[@]}")
                    cmd+=("--output" "$output_file")
                    if ! output=$(cd "$SCRIPT_DIR" && "${cmd[@]}" 2>&1 >/dev/null); then
                        echo "✗ $basename (failed)"
                        printf '%s\n' "$output"
                    fi
                }
                export -f process_file
                export SASA_DIR DSSP_DIR PKA_DIR OUTPUT_DIR PH_VALUE EXTRA_ARGS_STR
                find "$STRUCTURES_DIR" -name "*.pdb" -type f ! -name "*_H.pdb" ! -name "*_L.pdb" | parallel -j "$NUM_JOBS" process_file {}
            else
                python3 << 'DEV_PY'
import os
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
ph_value = os.environ.get('PH_VALUE', '7.4')
extra_args = (os.environ.get('EXTRA_ARGS_STR') or '').split()

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    sasa_file = Path(SASA_DIR) / f"{basename}_full.sasa"
    dssp_file = Path(DSSP_DIR) / f"{basename}.dssp"
    pka_file = Path(PKA_DIR) / f"{basename}_full.pka"
    output_file = Path(OUTPUT_DIR) / f"{basename}.json"
    # Use relative path; we run with cwd=SCRIPT_DIR to avoid src/src duplication
    if not sasa_file.exists():
        return f"✗ {basename} (SASA file not found)"
    cmd = ["python3", "developability/run_developability.py", str(pdb_file), str(sasa_file)]
    if dssp_file.exists():
        cmd.extend(["--dssp-file", str(dssp_file)])
    if pka_file.exists():
        cmd.extend(["--pka-file", str(pka_file)])
    if '--pH' not in extra_args:
        cmd.extend(["--pH", ph_value])
    cmd.extend(extra_args)
    cmd.extend(["--output", str(output_file)])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
        if r.returncode != 0:
            return f"✗ {basename} (failed)\n{r.stderr}"
        return None
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:200]})"

pdb_files = sorted(p for p in Path(STRUCTURES_DIR).rglob("*.pdb") if not (p.stem.endswith("_H") or p.stem.endswith("_L")))
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
done

ELAPSED=$(( $(date +%s) - RUN_START ))
echo ""
echo "--- Summary ---"
echo "Structures processed: $TOTAL_STRUCTURES"
printf "Total time: %02d:%02d:%02d\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
