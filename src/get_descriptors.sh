#!/bin/bash
# mkdssp, propka, freesasa -> developability (parallel over PDB folders).
# Usage: get_descriptors.sh [--output-dir DIR] [--parent-dir DIR ...] STRUCTURES_DIR ... [num_jobs] [--pH VALUE] [--sanity_check_abb2] [--remove_helper_outputs] [--clean-external-outputs] [--append-failures]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/kitab.local.env" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/kitab.local.env"
fi

KITAB_ENV="${KITAB_ENV:-kitab}"
PY="${PY:-conda run -n ${KITAB_ENV} python}"
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

_kitab_run() {
    if command -v mamba &>/dev/null; then
        mamba run -n "$KITAB_ENV" "$@"
    elif command -v conda &>/dev/null; then
        conda run -n "$KITAB_ENV" "$@"
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
SKIP_EXISTING=false
SKIP_FAILED=false
BATCH_SIZE=0
APPEND_FAILURES=false

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
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --skip-failed)
            SKIP_FAILED=true
            shift
            ;;
        --batch-size)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --batch-size requires a positive integer." >&2
                exit 1
            fi
            BATCH_SIZE="$2"
            shift 2
            ;;
        --append-failures)
            APPEND_FAILURES=true
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
  echo "  --skip-existing: skip any PDB whose descriptor JSON already exists (non-empty) in the results/ dir." >&2
  echo "  --skip-failed: skip structures with unresolved failures in pipeline.log and still lack JSON." >&2
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
RUN_FAILED_TSV="${DESCRIPTOR_ROOT%/*}/failed_structures.tsv"
RUN_TOTAL_FAILURES=0
mkdir -p "$(dirname "$RUN_FAILED_TSV")"
if [[ "$APPEND_FAILURES" == true ]]; then
    if [[ ! -f "$RUN_FAILED_TSV" ]]; then
        printf 'dataset\tstructure\treason\n' > "$RUN_FAILED_TSV"
    fi
else
    printf 'dataset\tstructure\treason\n' > "$RUN_FAILED_TSV"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NUM_JOBS SCRIPT_DIR
export PH_VALUE
export SANITY_CHECK_ABB2
export REMOVE_HELPER_OUTPUTS
export CLEAN_EXTERNAL_OUTPUTS
export SKIP_EXISTING

_remove_helper_outputs_for_stem() {
    local basename="$1"
    [[ "$REMOVE_HELPER_OUTPUTS" == true ]] || return 0
    rm -f \
        "${DSSP_DIR}/${basename}.dssp" \
        "${PKA_DIR}/${basename}_full.pka" \
        "${PKA_DIR}/${basename}_full.log" \
        "${PKA_DIR}/tmp_structures/${basename}_full.pdb" \
        "${PKA_DIR}/tmp_structures/${basename}_full.propka_map.json" \
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

_truncate_for_terminal() {
    local text="${1//$'\n'/ }"
    text="${text//  / }"
    local max="${2:-120}"
    if ((${#text} > max)); then
        printf '%s...' "${text:0:max}"
    else
        printf '%s' "$text"
    fi
}

_extract_failure_reason() {
    local raw="$1"
    local reason=""
    reason=$(printf '%s\n' "$raw" | grep -m1 -E 'PropKa coverage incomplete|pKa data is empty|SASA file not found|empty output|did not produce|No such file|Error calculating' || true)
    if [[ -z "$reason" ]]; then
        reason=$(printf '%s\n' "$raw" | grep -m1 -E 'ERROR|Error |RuntimeError|Traceback|failed' | sed -E 's/^ERROR:[^:]+://; s/^[[:space:]]+//' || true)
    fi
    if [[ -z "$reason" ]]; then
        reason=$(printf '%s\n' "$raw" | tr '\n' ' ' | sed 's/  */ /g')
    fi
    reason=$(printf '%s' "$reason" | sed -E 's|Error calculating developability descriptors for [^:]+: ||')
    reason=$(printf '%s' "$reason" | sed -E 's/^ERROR:[^:]+://')
    _truncate_for_terminal "$reason" 500
}

_emit_failure() {
    local stage="$1" stem="$2" reason="$3"
    local short
    short=$(_truncate_for_terminal "$reason" 120)
    echo "✗ ${stem} (${stage}: ${short})"
    echo "[$(date -Iseconds)] ${stage} ✗ ${stem}: ${reason}" >> "$PIPELINE_LOG"
}

# Parse pipeline.log failures. Modes: stems (one stem per line) or records (stem<TAB>stage<TAB>reason).
_pipeline_failure_records() {
    local log_file="$1"
    local json_dir="${2:-}"
    local since_session="${3:-since_session}"
    local mode="${4:-records}"
    local parser_script parser_rc
    parser_script="$(mktemp)"
    cat > "$parser_script" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
json_dir = Path(sys.argv[2]) if sys.argv[2] else None
since_session = sys.argv[3] == "since_session"
mode = sys.argv[4]

lines = log_path.read_text(errors="replace").splitlines() if log_path.is_file() else []

start_idx = 0
for i, line in enumerate(lines):
    if "=== Pipeline session START" in line or "] Pipeline started for" in line:
        start_idx = i

scan = lines[start_idx:] if since_session else lines

fail_pat = re.compile(r"\] (INPUT|DSSP|SASA|PROPKA|DESCRIPTORS) ✗ ([^:]+):?\s*(.*)$")
failures = {}
for line in scan:
    m = fail_pat.search(line)
    if m:
        stage, stem, reason = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
        failures[stem] = (stage, reason)

err_pat = re.compile(
    r"Error calculating developability descriptors for (.+?\.pdb): (.+)$"
)
for line in scan:
    m = err_pat.search(line)
    if m:
        stem, reason = Path(m.group(1)).stem, m.group(2).strip()
        failures.setdefault(stem, ("DESCRIPTORS", reason))

if json_dir and json_dir.is_dir():
    for stem in list(failures):
        jp = json_dir / f"{stem}.json"
        if jp.is_file() and jp.stat().st_size > 0:
            del failures[stem]

for stem in sorted(failures):
    stage, reason = failures[stem]
    if mode == "stems":
        print(stem)
    else:
        print(f"{stem}\t{stage}\t{reason.replace(chr(9), ' ')}")
PY
    if "${PY_ARR[@]}" "$parser_script" "$log_file" "$json_dir" "$since_session" "$mode"; then
        parser_rc=0
    else
        parser_rc=$?
    fi
    rm -f "$parser_script"
    return "$parser_rc"
}

_pipeline_session_end_report() {
    local base_name="$1"
    local log_file="$2"
    local json_dir="$3"
    local -a fail_lines=()
    mapfile -t fail_lines < <(_pipeline_failure_records "$log_file" "$json_dir" since_session records)
    local -a _nonempty_fail_lines=()
    local _fail_line
    for _fail_line in "${fail_lines[@]}"; do
        [[ -z "$_fail_line" ]] && continue
        _nonempty_fail_lines+=("$_fail_line")
    done
    fail_lines=("${_nonempty_fail_lines[@]}")
    local n=${#fail_lines[@]}
    echo "[$(date -Iseconds)] === Pipeline session END (${n} unresolved failure(s)) ===" >> "$log_file"
    if [[ $n -eq 0 ]]; then
        echo "  ${base_name}: all structures OK this session"
        return 0
    fi
    RUN_TOTAL_FAILURES=$((RUN_TOTAL_FAILURES + n))
    echo ""
    echo "  Failures in ${base_name} (${n}):"
    local line stem stage reason short
    for line in "${fail_lines[@]}"; do
        IFS=$'\t' read -r stem stage reason <<< "$line"
        short=$(_truncate_for_terminal "$reason" 100)
        echo "    ✗ ${stem} (${stage}: ${short})"
        printf '%s\t%s\t%s\n' "$base_name" "$stem" "$reason" >> "$RUN_FAILED_TSV"
    done
    echo "  Log: ${log_file}"
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
    log_hint = results_dir.parent / "pipeline.log"
    raise SystemExit(
        f"Missing descriptor JSON for {len(missing)} CSV name(s) in {results_dir}: "
        f"{preview}{extra}\n"
        f"See {log_hint} for per-structure failure reasons."
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

_is_valid_descriptor_pdb() {
    local pdb="$1"
    local stem="$2"
    [[ -f "$pdb" ]] || return 0
    if [[ ! -s "$pdb" ]]; then
        _emit_failure "INPUT" "$stem" "empty PDB file"
        return 1
    fi
    if ! grep -qE '^(ATOM  |HETATM)' "$pdb"; then
        _emit_failure "INPUT" "$stem" "PDB file has no ATOM/HETATM records"
        return 1
    fi
    return 0
}

pipeline_pdb_paths() {
    local dir="$1"
    if [[ ${#ALLOWED_NAMES[@]} -gt 0 ]]; then
        # Fast-path: allowed names are known — stat each file directly, no directory scan.
        local name pdb
        for name in "${ALLOWED_NAMES[@]}"; do
            pdb="$dir/${name}.pdb"
            [[ -f "$pdb" ]] || continue
            if [[ "$SANITY_CHECK_ABB2" == true ]] && _is_abb2_sanity_skipped_stem "$name"; then
                continue
            fi
            printf '%s\0' "$pdb"
        done
    else
        # Full scan: enumerate every top-level pipeline PDB.
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
    fi
}

count_pipeline_pdbs() {
    local n=0 _p
    while IFS= read -r -d '' _p; do
        n=$((n + 1))
    done < <(pipeline_pdb_paths "$1")
    echo "$n"
}

declare -A PIPELINE_FAILED_STEMS=()

_load_pipeline_failed_stems() {
    local log_file="$1"
    local json_dir="${2:-}"
    unset PIPELINE_FAILED_STEMS
    declare -gA PIPELINE_FAILED_STEMS
    [[ -f "$log_file" ]] || return 0
    while IFS= read -r stem; do
        [[ -z "$stem" ]] && continue
        PIPELINE_FAILED_STEMS["$stem"]=1
    done < <(_pipeline_failure_records "$log_file" "$json_dir" all_sessions stems)
    return 0
}

_is_pipeline_failed_stem() {
    local stem="$1"
    [[ -n "$stem" && -n "${PIPELINE_FAILED_STEMS[$stem]+x}" ]]
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

    PIPELINE_LOG="${DATASET_DESCRIPTOR_DIR}/pipeline.log"
    export PIPELINE_LOG
    echo "[$(date -Iseconds)] === Pipeline session START (${BASE_NAME}) ===" >> "$PIPELINE_LOG"

    if [[ "$SKIP_FAILED" == true ]]; then
        _load_pipeline_failed_stems "$PIPELINE_LOG" "$DEV_JSON_OUTPUT_DIR"
    fi

    if [[ "$SANITY_CHECK_ABB2" == true ]]; then
        ABB2_SANITY_SKIP_LOG="${DATASET_DESCRIPTOR_DIR}/abb2_sanity_skip.log"
        _build_abb2_sanity_skip_set "$STRUCTURES_DIR" "$ABB2_SANITY_SKIP_LOG"
        export ABB2_SANITY_SKIP_LOG ABB2_SKIP_STEMS_CSV
    else
        ABB2_SKIP_LOOKUP=()
        ABB2_SKIP_STEMS_CSV=""
        unset ABB2_SANITY_SKIP_LOG
    fi

    # ----------------------------------------------------------------
    # _run_pipeline_stages: runs the 4 pipeline stages (DSSP, FreeSASA,
    # PropKa, Developability) for the ALLOWED_NAMES / ALLOWED_NAMES_CSV
    # currently set in the global scope.  Defined per STRUCTURES_DIR
    # iteration so all per-dataset globals are in scope at call time.
    # ----------------------------------------------------------------
    _run_pipeline_stages() {
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
                    if [[ ! -s "$pdb_file" ]]; then
                        _emit_failure "DSSP" "$basename" "empty PDB file"
                        rm -f "$output_file"
                        return 0
                    fi
                    if ! grep -qE '^(ATOM  |HETATM)' "$pdb_file"; then
                        _emit_failure "DSSP" "$basename" "PDB file has no ATOM/HETATM records"
                        rm -f "$output_file"
                        return 0
                    fi
                    local temp_pdb=$(mktemp)
                    {
                        printf "REMARK    @%s (1-2)\n" "$filename"
                        echo "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          "
                        awk '!/^REMARK/ && !/^CRYST1/' "$pdb_file"
                    } > "$temp_pdb"
                    local dssp_err
                    if ! dssp_err=$(_kitab_run "$DSSP_BIN" "$temp_pdb" "$output_file" 2>&1 >/dev/null); then
                        _emit_failure "DSSP" "$basename" "$dssp_err"
                        rm -f "$output_file"
                    elif [[ ! -s "$output_file" ]]; then
                        _emit_failure "DSSP" "$basename" "empty output"
                        rm -f "$output_file"
                    fi
                    rm -f "$temp_pdb"
                }
                export -f process_file _kitab_run _emit_failure _truncate_for_terminal
                export DSSP_BIN KITAB_ENV PIPELINE_LOG
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel --will-cite -0 -j "$NUM_JOBS" process_file {}
            else
                export STRUCTURES_DIR DSSP_BIN KITAB_ENV
                "${PY_ARR[@]}" << 'DSSP_PY'
import os
import shutil
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")
DSSP_BIN = os.environ.get("DSSP_BIN", "mkdssp")
KITAB_ENV = os.environ.get("KITAB_ENV", "kitab")
PIPELINE_LOG = os.environ.get("PIPELINE_LOG", "")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def run_in_kitab(args, **kwargs):
    for conda in ("mamba", "conda"):
        if shutil.which(conda):
            return subprocess.run([conda, "run", "-n", KITAB_ENV, *args], **kwargs)
    return subprocess.run(args, **kwargs)

def _log(msg):
    if PIPELINE_LOG:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(PIPELINE_LOG, "a") as lf:
            lf.write(f"[{ts}] {msg}\n")

def process_file(pdb_path):
    pdb_file = Path(pdb_path)
    basename = pdb_file.stem
    filename = pdb_file.name
    output_file = Path(OUTPUT_DIR) / f"{basename}.dssp"
    try:
        if not pdb_file.stat().st_size:
            output_file.unlink(missing_ok=True)
            _log(f"DSSP ✗ {basename}: empty PDB file")
            return f"✗ {basename} (empty PDB file)"
        with open(pdb_file) as f:
            has_atoms = any(line.startswith(("ATOM  ", "HETATM")) for line in f)
        if not has_atoms:
            output_file.unlink(missing_ok=True)
            _log(f"DSSP ✗ {basename}: PDB file has no ATOM/HETATM records")
            return f"✗ {basename} (no ATOM/HETATM records)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tf:
            tf.write(f"REMARK    @{filename} (1-2)\n")
            tf.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n")
            with open(pdb_file) as f:
                for line in f:
                    if not line.startswith('REMARK') and not line.startswith('CRYST1'):
                        tf.write(line)
            path = tf.name
        r = run_in_kitab([DSSP_BIN, path, str(output_file)], capture_output=True, text=True, cwd=SCRIPT_DIR)
        Path(path).unlink(missing_ok=True)
        if r.returncode != 0:
            output_file.unlink(missing_ok=True)
            _log(f"DSSP ✗ {basename}: {r.stderr.strip()}")
            return f"✗ {basename} (failed)"
        if not output_file.is_file() or output_file.stat().st_size == 0:
            output_file.unlink(missing_ok=True)
            _log(f"DSSP ✗ {basename}: empty output")
            return f"✗ {basename} (no DSSP output)"
        return None
    except Exception as e:
        _log(f"DSSP ✗ {basename}: {e}")
        return f"✗ {basename} (error: {str(e)[:50]})"

_anf = os.environ.get("ALLOWED_NAMES_FILE", "")
if _anf and Path(_anf).is_file():
    allowed_names = set(n for n in Path(_anf).read_text().splitlines() if n.strip())
else:
    allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    if allowed_names:
        # Fast-path: names are known — stat each file directly, no glob.
        for name in sorted(allowed_names):
            if name in abb2_skip_stems:
                continue
            p = root / f"{name}.pdb"
            if p.is_file():
                out.append(p)
        return out
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
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

                    local sasa_err
                    if ! sasa_err=$(_kitab_run freesasa --shrake-rupley --format=rsa --depth=residue "$pdbfile" > "$sasa_full" 2>&1); then
                        _emit_failure "SASA" "$filename" "$sasa_err"
                        rm -f "$sasa_full"
                    fi

                    if grep -q "^ATOM" "$tmp_H"; then
                        if ! sasa_err=$(_kitab_run freesasa --shrake-rupley --format=rsa --depth=residue "$tmp_H" > "$sasa_H" 2>&1); then
                            _emit_failure "SASA" "$filename" "H-chain: ${sasa_err}"
                            rm -f "$sasa_H"
                        fi
                    fi

                    if grep -q "^ATOM" "$tmp_L"; then
                        if ! sasa_err=$(_kitab_run freesasa --shrake-rupley --format=rsa --depth=residue "$tmp_L" > "$sasa_L" 2>&1); then
                            _emit_failure "SASA" "$filename" "L-chain: ${sasa_err}"
                            rm -f "$sasa_L"
                        fi
                    fi

                    rm -f "$tmp_H" "$tmp_L"
                }
                export -f process_file _kitab_run _emit_failure _truncate_for_terminal
                export SASA_DIR KITAB_ENV PIPELINE_LOG
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel --will-cite -0 -j "$NUM_JOBS" process_file {}
            else
                export STRUCTURES_DIR SASA_DIR KITAB_ENV PIPELINE_LOG
                "${PY_ARR[@]}" << 'FREESASA_PY'
import os
import shutil
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
SASA_DIR = os.environ.get("SASA_DIR", ".")
KITAB_ENV = os.environ.get("KITAB_ENV", "kitab")
PIPELINE_LOG = os.environ.get("PIPELINE_LOG", "")
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def run_in_kitab(args, **kwargs):
    for conda in ("mamba", "conda"):
        if shutil.which(conda):
            return subprocess.run([conda, "run", "-n", KITAB_ENV, *args], **kwargs)
    return subprocess.run(args, **kwargs)

def _log(msg):
    if PIPELINE_LOG:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(PIPELINE_LOG, "a") as lf:
            lf.write(f"[{ts}] {msg}\n")

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

    r = run_in_kitab(
        ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(pdb_file)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        errors.append(f"✗ {basename} (full failed)")
        _log(f"SASA ✗ {basename} full: {r.stderr.strip()}")
        sasa_full.unlink(missing_ok=True)
    else:
        sasa_full.write_text(r.stdout)
        if sasa_full.stat().st_size == 0:
            errors.append(f"✗ {basename} (full failed)")
            _log(f"SASA ✗ {basename} full: empty output")
            sasa_full.unlink(missing_ok=True)

    has_H_atoms = any(l.startswith("ATOM") for l in tmp_H_path.read_text().splitlines())
    if has_H_atoms:
        r = run_in_kitab(
            ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(tmp_H_path)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            errors.append(f"✗ {basename} (H-only failed)")
            _log(f"SASA ✗ {basename} H-chain: {r.stderr.strip()}")
        else:
            sasa_H.write_text(r.stdout)

    has_L_atoms = any(l.startswith("ATOM") for l in tmp_L_path.read_text().splitlines())
    if has_L_atoms:
        r = run_in_kitab(
            ["freesasa", "--shrake-rupley", "--format=rsa", "--depth=residue", str(tmp_L_path)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            errors.append(f"✗ {basename} (L-only failed)")
            _log(f"SASA ✗ {basename} L-chain: {r.stderr.strip()}")
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

_anf = os.environ.get("ALLOWED_NAMES_FILE", "")
if _anf and Path(_anf).is_file():
    allowed_names = set(n for n in Path(_anf).read_text().splitlines() if n.strip())
else:
    allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    if allowed_names:
        # Fast-path: names are known — stat each file directly, no glob.
        for name in sorted(allowed_names):
            if name in abb2_skip_stems:
                continue
            p = root / f"{name}.pdb"
            if p.is_file():
                out.append(p)
        return out
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
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
                    propka_input=$(cd "$SCRIPT_DIR" && _kitab_run python utils/prepare_propka_input.py "$pdb_file" \
                        --tmp-dir "$tmp_structures" --stem "${basename}_full" 2>/dev/null | head -1)
                    propka_stem=$(basename "$propka_input" .pdb)
                    _kitab_run propka3 "$propka_input" > "${basename}_full.log" 2>&1
                    if ! _finalize_propka_pka "${propka_stem}.pka" "${basename}_full.pka"; then
                        local propka_reason="PropKa did not produce .pka output"
                        if [[ -s "${basename}_full.log" ]]; then
                            propka_reason=$(grep -m1 -E 'failed protonation|Missing atoms|Error|WARNING' "${basename}_full.log" || echo "$propka_reason")
                        fi
                        _emit_failure "PROPKA" "$basename" "$propka_reason"
                    fi
                }
                export -f process_file _kitab_run _finalize_propka_pka _emit_failure _truncate_for_terminal
                export KITAB_ENV SCRIPT_DIR
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel --will-cite -0 -j "$NUM_JOBS" process_file {} "$OUTPUT_DIR"
            else
                export STRUCTURES_DIR KITAB_ENV PY SCRIPT_DIR
                "${PY_ARR[@]}" << 'PROPKA_PY'
import os
import shutil
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
STRUCTURES_DIR = os.environ.get("STRUCTURES_DIR", ".")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
KITAB_ENV = os.environ.get("KITAB_ENV", "kitab")
SCRIPT_DIR = Path(os.environ.get("SCRIPT_DIR", "."))
TMP_STRUCTURES = Path(OUTPUT_DIR) / "tmp_structures"
TMP_STRUCTURES.mkdir(parents=True, exist_ok=True)
num_jobs = int(os.environ.get("NUM_JOBS", str(cpu_count())))

def run_in_kitab(args, **kwargs):
    for conda in ("mamba", "conda"):
        if shutil.which(conda):
            return subprocess.run([conda, "run", "-n", KITAB_ENV, *args], **kwargs)
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
    r = run_in_kitab(["propka3", str(propka_input)], capture_output=True, text=True, cwd=output_dir)
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

_anf = os.environ.get("ALLOWED_NAMES_FILE", "")
if _anf and Path(_anf).is_file():
    allowed_names = set(n for n in Path(_anf).read_text().splitlines() if n.strip())
else:
    allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    if allowed_names:
        # Fast-path: names are known — stat each file directly, no glob.
        for name in sorted(allowed_names):
            if name in abb2_skip_stems:
                continue
            p = root / f"{name}.pdb"
            if p.is_file():
                out.append(p)
        return out
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
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
                        _emit_failure "DESCRIPTORS" "$basename" "SASA file not found: ${sasa_file}"
                        return
                    fi
                    [ -f "$dssp_file" ] && cmd+=("--dssp-file" "$dssp_file")
                    [ -f "$pka_file" ] && cmd+=("--pka-file" "$pka_file")
                    cmd+=("--pH" "$PH_VALUE")
                    cmd+=("--output" "$output_file")
                    cmd+=("--log-file" "$PIPELINE_LOG")
                    local output reason
                    if ! output=$(cd "$SCRIPT_DIR" && _kitab_run "${cmd[@]}" 2>&1); then
                        reason=$(_extract_failure_reason "$output")
                        _emit_failure "DESCRIPTORS" "$basename" "$reason"
                    elif [[ -s "$output_file" ]]; then
                        _remove_helper_outputs_for_stem "$basename"
                    fi
                }
                export -f process_file _kitab_run _remove_helper_outputs_for_stem _emit_failure _extract_failure_reason _truncate_for_terminal
                export SASA_DIR DSSP_DIR PKA_DIR OUTPUT_DIR PH_VALUE SCRIPT_DIR KITAB_ENV REMOVE_HELPER_OUTPUTS PIPELINE_LOG
                pipeline_pdb_paths "$STRUCTURES_DIR" | parallel --will-cite -0 -j "$NUM_JOBS" process_file {}
            else
                export PY REMOVE_HELPER_OUTPUTS PIPELINE_LOG
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
PIPELINE_LOG = os.environ.get("PIPELINE_LOG", "")
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
        Path(PKA_DIR) / "tmp_structures" / f"{basename}_full.pdb",
        Path(PKA_DIR) / "tmp_structures" / f"{basename}_full.propka_map.json",
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
    if PIPELINE_LOG:
        cmd.extend(["--log-file", PIPELINE_LOG])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
        if r.returncode != 0:
            return f"✗ {basename} (failed)\n{r.stderr}"
        if output_file.is_file() and output_file.stat().st_size > 0:
            remove_helper_outputs_for_stem(basename)
        return None
    except Exception as e:
        return f"✗ {basename} (error: {str(e)[:200]})"

_anf = os.environ.get("ALLOWED_NAMES_FILE", "")
if _anf and Path(_anf).is_file():
    allowed_names = set(n for n in Path(_anf).read_text().splitlines() if n.strip())
else:
    allowed_names = set(filter(None, os.environ.get("ALLOWED_NAMES_CSV", "").split(",")))
abb2_skip_stems = set(filter(None, os.environ.get("ABB2_SKIP_STEMS_CSV", "").split(",")))

def pipeline_pdb_files(structures_dir):
    root = Path(structures_dir)
    out = []
    if allowed_names:
        # Fast-path: names are known — stat each file directly, no glob.
        for name in sorted(allowed_names):
            if name in abb2_skip_stems:
                continue
            p = root / f"{name}.pdb"
            if p.is_file():
                out.append(p)
        return out
    for p in sorted(root.glob("*.pdb")):
        stem = p.stem
        if stem.endswith(("_H", "_L", "_full_atom_sasa", "_H_chain", "_L_chain")):
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
    done  # for MODE
    }  # _run_pipeline_stages

    # ----------------------------------------------------------------
    # Dispatch: stream find output into batches when BATCH_SIZE>0 and no
    # pre-set name filter (no --names-from-csv).  The first batch fires
    # as soon as BATCH_SIZE stems arrive — no full upfront enumeration.
    # Fall back to a single pass for BATCH_SIZE=0 or when names are
    # already known from --names-from-csv.
    # ----------------------------------------------------------------
    if [[ "$BATCH_SIZE" -gt 0 && ${#ALLOWED_NAMES[@]} -eq 0 ]]; then
        _batch_stems=()
        _BATCH_NUM=0
        _batch_total=0
        _sp_skipped_done=0
        _sp_skipped_failed=0
        _sp_skipped_invalid=0
        while IFS= read -r -d '' _pdb; do
            _stem="${_pdb##*/}"
            _stem="${_stem%.pdb}"
            # Inline filter matching _is_pipeline_pdb_stem (no ALLOWED_NAMES yet).
            case "$_stem" in *_full_atom_sasa|*_H_chain|*_L_chain|*_H|*_L) continue ;; esac
            if [[ "$SANITY_CHECK_ABB2" == true ]] && _is_abb2_sanity_skipped_stem "$_stem"; then
                continue
            fi
            if [[ "$SKIP_EXISTING" == true ]] && [[ -s "$DEV_JSON_OUTPUT_DIR/${_stem}.json" ]]; then
                _sp_skipped_done=$((_sp_skipped_done + 1))
                continue
            fi
            if [[ "$SKIP_FAILED" == true ]] && _is_pipeline_failed_stem "$_stem"; then
                _sp_skipped_failed=$((_sp_skipped_failed + 1))
                continue
            fi
            if ! _is_valid_descriptor_pdb "$_pdb" "$_stem"; then
                _sp_skipped_invalid=$((_sp_skipped_invalid + 1))
                continue
            fi
            _batch_stems+=("$_stem")
            if [[ ${#_batch_stems[@]} -ge "$BATCH_SIZE" ]]; then
                _BATCH_NUM=$(( _BATCH_NUM + 1 ))
                ALLOWED_NAMES=("${_batch_stems[@]}")
                # Write names to a temp file so the env stays small.
                _ALLOWED_NAMES_FILE=$(mktemp)
                printf '%s\n' "${ALLOWED_NAMES[@]}" > "$_ALLOWED_NAMES_FILE"
                ALLOWED_NAMES_CSV=""
                export ALLOWED_NAMES_CSV ALLOWED_NAMES_FILE="$_ALLOWED_NAMES_FILE"
                echo "  [Batch $_BATCH_NUM] ${#ALLOWED_NAMES[@]} structures ..."
                echo "[$(date -Iseconds)] Batch $_BATCH_NUM start (${#ALLOWED_NAMES[@]} structures)" >> "$PIPELINE_LOG"
                _run_pipeline_stages
                rm -f "$_ALLOWED_NAMES_FILE"
                unset ALLOWED_NAMES_FILE
                _batch_total=$(( _batch_total + ${#ALLOWED_NAMES[@]} ))
                _batch_stems=()
            fi
        done < <(find "$STRUCTURES_DIR" -maxdepth 1 -name "*.pdb" -type f \
                      ! -path '*/.*/*' ! -name "*_H.pdb" ! -name "*_L.pdb" -print0)
        # Remaining partial batch (last batch, may be smaller than BATCH_SIZE).
        if [[ ${#_batch_stems[@]} -gt 0 ]]; then
            _BATCH_NUM=$(( _BATCH_NUM + 1 ))
            ALLOWED_NAMES=("${_batch_stems[@]}")
            _ALLOWED_NAMES_FILE=$(mktemp)
            printf '%s\n' "${ALLOWED_NAMES[@]}" > "$_ALLOWED_NAMES_FILE"
            ALLOWED_NAMES_CSV=""
            export ALLOWED_NAMES_CSV ALLOWED_NAMES_FILE="$_ALLOWED_NAMES_FILE"
            echo "  [Batch $_BATCH_NUM (final)] ${#ALLOWED_NAMES[@]} structures ..."
            echo "[$(date -Iseconds)] Batch $_BATCH_NUM final (${#ALLOWED_NAMES[@]} structures)" >> "$PIPELINE_LOG"
            _run_pipeline_stages
            rm -f "$_ALLOWED_NAMES_FILE"
            unset ALLOWED_NAMES_FILE
            _batch_total=$(( _batch_total + ${#ALLOWED_NAMES[@]} ))
        fi
        TOTAL_STRUCTURES=$(( TOTAL_STRUCTURES + _batch_total ))
        if [[ "$SKIP_EXISTING" == true && $_sp_skipped_done -gt 0 ]]; then
            echo "  skip-existing: $_sp_skipped_done already-done structure(s) skipped"
        fi
        if [[ "$SKIP_FAILED" == true && $_sp_skipped_failed -gt 0 ]]; then
            echo "  skip-failed: $_sp_skipped_failed structure(s) skipped (unresolved failure in pipeline.log, no JSON yet; see ${PIPELINE_LOG})"
        fi
        if [[ "$_sp_skipped_invalid" -gt 0 ]]; then
            echo "  invalid-input: $_sp_skipped_invalid invalid PDB file(s) skipped (see ${PIPELINE_LOG})"
        fi
        echo "  Batch mode done: $_batch_total structure(s) in $_BATCH_NUM batch(es)"
        ALLOWED_NAMES=()
        ALLOWED_NAMES_CSV=""
        export ALLOWED_NAMES_CSV
    else
        # Single-pass: BATCH_SIZE=0, or names already set from --names-from-csv.

        # Remove invalid inputs, and optionally remove finished or previously failed stems.
        if true; then
            _sp_filtered=()
            _sp_skipped_done=0
            _sp_skipped_failed=0
            _sp_skipped_invalid=0
            if [[ ${#ALLOWED_NAMES[@]} -gt 0 ]]; then
                # Pre-set name list (e.g. from --names-from-csv): filter in-memory.
                for _sp_name in "${ALLOWED_NAMES[@]}"; do
                    if [[ "$SKIP_EXISTING" == true ]] && [[ -s "$DEV_JSON_OUTPUT_DIR/${_sp_name}.json" ]]; then
                        _sp_skipped_done=$((_sp_skipped_done + 1))
                        continue
                    fi
                    if [[ "$SKIP_FAILED" == true ]] && _is_pipeline_failed_stem "$_sp_name"; then
                        _sp_skipped_failed=$((_sp_skipped_failed + 1))
                        continue
                    fi
                    if ! _is_valid_descriptor_pdb "$STRUCTURES_DIR/${_sp_name}.pdb" "$_sp_name"; then
                        _sp_skipped_invalid=$((_sp_skipped_invalid + 1))
                        continue
                    fi
                    _sp_filtered+=("$_sp_name")
                done
                [[ "$SKIP_EXISTING" == true && $_sp_skipped_done -gt 0 ]] && echo "  skip-existing: $_sp_skipped_done already-done structure(s) skipped"
                [[ "$SKIP_FAILED" == true && $_sp_skipped_failed -gt 0 ]] && echo "  skip-failed: $_sp_skipped_failed structure(s) skipped (unresolved failure in pipeline.log, no JSON yet; see ${PIPELINE_LOG})"
                [[ "$_sp_skipped_invalid" -gt 0 ]] && echo "  invalid-input: $_sp_skipped_invalid invalid PDB file(s) skipped (see ${PIPELINE_LOG})"
                ALLOWED_NAMES=("${_sp_filtered[@]}")
                ALLOWED_NAMES_CSV="$(printf '%s,' "${ALLOWED_NAMES[@]}")"
                export ALLOWED_NAMES_CSV
            else
                # No pre-set names: scan PDB files, exclude done and/or previously failed.
                _SP_NAMES_FILE=$(mktemp)
                while IFS= read -r -d '' _sp_pdb; do
                    _sp_stem="${_sp_pdb##*/}"; _sp_stem="${_sp_stem%.pdb}"
                    case "$_sp_stem" in *_full_atom_sasa|*_H_chain|*_L_chain|*_H|*_L) continue ;; esac
                    if [[ "$SANITY_CHECK_ABB2" == true ]] && _is_abb2_sanity_skipped_stem "$_sp_stem"; then continue; fi
                    if [[ "$SKIP_EXISTING" == true ]] && [[ -s "$DEV_JSON_OUTPUT_DIR/${_sp_stem}.json" ]]; then
                        _sp_skipped_done=$((_sp_skipped_done + 1))
                        continue
                    fi
                    if [[ "$SKIP_FAILED" == true ]] && _is_pipeline_failed_stem "$_sp_stem"; then
                        _sp_skipped_failed=$((_sp_skipped_failed + 1))
                        continue
                    fi
                    if ! _is_valid_descriptor_pdb "$_sp_pdb" "$_sp_stem"; then
                        _sp_skipped_invalid=$((_sp_skipped_invalid + 1))
                        continue
                    fi
                    _sp_filtered+=("$_sp_stem")
                    echo "$_sp_stem" >> "$_SP_NAMES_FILE"
                done < <(find "$STRUCTURES_DIR" -maxdepth 1 -name "*.pdb" -type f \
                              ! -path '*/.*/*' ! -name "*_H.pdb" ! -name "*_L.pdb" -print0)
                [[ "$SKIP_EXISTING" == true && $_sp_skipped_done -gt 0 ]] && echo "  skip-existing: $_sp_skipped_done already-done structure(s) skipped"
                [[ "$SKIP_FAILED" == true && $_sp_skipped_failed -gt 0 ]] && echo "  skip-failed: $_sp_skipped_failed structure(s) skipped (unresolved failure in pipeline.log, no JSON yet; see ${PIPELINE_LOG})"
                [[ "$_sp_skipped_invalid" -gt 0 ]] && echo "  invalid-input: $_sp_skipped_invalid invalid PDB file(s) skipped (see ${PIPELINE_LOG})"
                if [[ ${#_sp_filtered[@]} -gt 0 ]]; then
                    ALLOWED_NAMES=("${_sp_filtered[@]}")
                    ALLOWED_NAMES_CSV=""
                    export ALLOWED_NAMES_CSV ALLOWED_NAMES_FILE="$_SP_NAMES_FILE"
                else
                    rm -f "$_SP_NAMES_FILE"
                    unset _SP_NAMES_FILE
                fi
            fi
        fi

        if [[ "$BATCH_SIZE" -eq 0 ]]; then
            N_PDB=$(count_pipeline_pdbs "$STRUCTURES_DIR")
            TOTAL_STRUCTURES=$(( TOTAL_STRUCTURES + N_PDB ))
        else
            # BATCH_SIZE>0 but ALLOWED_NAMES already known from --names-from-csv.
            TOTAL_STRUCTURES=$(( TOTAL_STRUCTURES + ${#ALLOWED_NAMES[@]} ))
        fi
        _run_pipeline_stages
        if [[ -n "${_SP_NAMES_FILE:-}" ]]; then
            rm -f "$_SP_NAMES_FILE"
            unset ALLOWED_NAMES_FILE _SP_NAMES_FILE
        fi
    fi

    if [[ -n "$NAMES_FROM_CSV" ]]; then
        _validate_descriptor_outputs_for_csv "$NAMES_FROM_CSV" "$DEV_JSON_OUTPUT_DIR"
    fi

    _pipeline_session_end_report "$BASE_NAME" "$PIPELINE_LOG" "$DEV_JSON_OUTPUT_DIR"

    _clean_external_output_dirs
done

ELAPSED=$(( $(date +%s) - RUN_START ))
echo ""
echo "--- Summary ---"
echo "Structures processed: $TOTAL_STRUCTURES"
printf "Total time: %02d:%02d:%02d\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
if [[ "$RUN_TOTAL_FAILURES" -gt 0 ]]; then
    echo "Failures this run: $RUN_TOTAL_FAILURES (details: $RUN_FAILED_TSV)"
else
    echo "All descriptor runs succeeded this session."
fi
