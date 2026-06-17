#!/usr/bin/env bash
# FASTAb one-shot installer: conda envs for descriptors, ABB2 (ImmuneBuilder), and ABB3.
#
# Usage:
#   ./install.sh
#   ./install.sh --fastab-only        # descriptors / automl env only
#   ./install.sh --no-abb2            # fastab + abb3 + flashabb (skip abb2)
#   ./install.sh --no-flashabb        # fastab + abb2 + abb3 (skip flashabb)
#
# Prerequisites: git, wget or curl, mamba or conda (Miniforge/Mambaforge recommended).
# Optional: NVIDIA GPU + driver for ABB2/ABB3 CUDA wheels.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_FASTAB="${FASTAB_ENV:-fastab}"
ENV_ABB2="${FASTAB_ENV_ABB2:-fastab-abb2}"
ENV_ABB3="${FASTAB_ENV_ABB3:-fastab-abb3}"
ENV_FLASHABB="${FASTAB_ENV_FLASHABB:-fastab-flashabb}"

ABB3_REPO="${ABB3_REPO:-https://github.com/Exscientia/abodybuilder3.git}"
ABB3_DIR="${ABB3_DIR:-$REPO_ROOT/vendor/abodybuilder3}"
FLASHABB_DIR="${FLASHABB_DIR:-$REPO_ROOT/FlashABB}"
IMMUNEBUILDER_VERSION="${IMMUNEBUILDER_VERSION:-1.2}"
INSTALL_ABB2=1
INSTALL_ABB3=1
INSTALL_FLASHABB=1
INSTALL_MODE="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fastab-only)
            INSTALL_ABB2=0
            INSTALL_ABB3=0
            INSTALL_FLASHABB=0
            INSTALL_MODE="fastab-only"
            shift
            ;;
        --no-abb2)
            INSTALL_ABB2=0
            INSTALL_ABB3=1
            INSTALL_MODE="no-abb2"
            shift
            ;;
        --no-flashabb)
            INSTALL_FLASHABB=0
            shift
            ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)" >&2
            exit 1
            ;;
    esac
done

log() { echo "[install] $*" >&2; }
warn() { echo "[install] WARNING: $*" >&2; }
die() { echo "[install] ERROR: $*" >&2; exit 1; }

pick_conda() {
    if command -v mamba &>/dev/null; then
        echo mamba
        return
    fi
    if command -v conda &>/dev/null; then
        echo conda
        return
    fi
    die "mamba or conda not found on PATH. Install Miniforge: https://github.com/conda-forge/miniforge"
}

CONDA_CMD="$(pick_conda)"
log "Using $CONDA_CMD"

eval "$("$CONDA_CMD" shell.bash hook)"

env_exists() {
    "$CONDA_CMD" env list | awk 'NR>2 && NF {print $1}' | grep -qx "$1"
}

create_env_from_yml() {
    local name="$1"
    local yml="$2"
    [[ -f "$yml" ]] || die "Missing $yml"
    if env_exists "$name"; then
        log "Env $name already exists — skipping conda create"
        return 0
    fi
    log "Creating conda env: $name from $yml"
    PIP_REQUIRE_VIRTUALENV=false "$CONDA_CMD" env create -f "$yml" -n "$name" -y
}

conda_run() {
    "$CONDA_CMD" run -n "$1" "${@:2}"
}

# propka 3.5.1 (conda-forge) breaks on Python 3.14: Parameters.parse_line reads
# self.__annotations__ (instances have none). Fixed upstream in propka#202.
patch_propka_py314() {
    local env_name="$1"
    env_exists "$env_name" || return 0
    local params broken fixed
    params="$(conda_run "$env_name" python -c "import propka.parameters as m; print(m.__file__)" 2>/dev/null)" || return 0
    [[ -f "$params" ]] || return 0
    broken='self.__annotations__.get(words[0])'
    fixed='type(self).__annotations__.get(words[0])'
    if grep -qF "$broken" "$params"; then
        log "Patching propka parameters.py for Python 3.14 compatibility"
        sed -i "s/${broken}/${fixed}/" "$params"
    fi
}

# Pick PyTorch pip index from nvidia-smi (driver/GPU present → CUDA wheel; else CPU).
detect_torch_index() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local cuda_ver
        cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
        if [[ -n "$cuda_ver" ]]; then
            log "ABB2: detected CUDA $cuda_ver (nvidia-smi) — using cu130 wheels"
        else
            log "ABB2: NVIDIA GPU detected — using cu130 wheels"
        fi
        echo "https://download.pytorch.org/whl/cu130"
    else
        warn "No NVIDIA GPU detected; installing CPU PyTorch for ABB2"
        echo "https://download.pytorch.org/whl/cpu"
    fi
}

install_fastab() {
    create_env_from_yml "$ENV_FASTAB" "$REPO_ROOT/environment.yml"
    ensure_conda_pkg "$ENV_FASTAB" mmseqs mmseqs2 "-c conda-forge -c bioconda"
    ensure_conda_pkg "$ENV_FASTAB" parallel parallel "-c conda-forge"
    patch_propka_py314 "$ENV_FASTAB"
}

# Backfill a package when reusing an env created before it was added to environment.yml.
ensure_conda_pkg() {
    local env_name="$1"
    local cmd="$2"
    local pkg="$3"
    local ch="$4"
    env_exists "$env_name" || return 0
    if conda_run "$env_name" which "$cmd" &>/dev/null; then
        return 0
    fi
    log "Installing $pkg into $env_name (missing from existing env; $ch)"
    # shellcheck disable=SC2086
    PIP_REQUIRE_VIRTUALENV=false "$CONDA_CMD" install -n "$env_name" $ch "$pkg" -y
}

install_abb2() {
    create_env_from_yml "$ENV_ABB2" "$REPO_ROOT/environment-abb2.yml"
    if ! env_exists "$ENV_ABB2"; then
        die "Env $ENV_ABB2 missing after create"
    fi
    local torch_index
    torch_index="$(detect_torch_index)"
    log "ABB2: installing PyTorch from $torch_index"
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_ABB2" pip install --upgrade pip
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_ABB2" pip install torch torchvision --index-url "$torch_index"
    log "ABB2: installing ImmuneBuilder==$IMMUNEBUILDER_VERSION"
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_ABB2" pip install "ImmuneBuilder==${IMMUNEBUILDER_VERSION}"
    if ! conda_run_with_env_lib "$ENV_ABB2" python -c "from ImmuneBuilder import ABodyBuilder2; import openmm" 2>/dev/null; then
        warn "ABB2 import check failed (often fixed by: source fastab.local.env — prepends conda lib to LD_LIBRARY_PATH)"
    fi
}

clone_abb3_repo() {
    if [[ -d "$ABB3_DIR/.git" ]]; then
        log "ABB3 repo present: $ABB3_DIR"
        return 0
    fi
    mkdir -p "$(dirname "$ABB3_DIR")"
    log "Cloning ABodyBuilder3: $ABB3_REPO -> $ABB3_DIR"
    git clone --depth 1 "$ABB3_REPO" "$ABB3_DIR"
}

install_abb3() {
    clone_abb3_repo
    [[ -f "$ABB3_DIR/environment_gpu.yml" ]] || die "Not an ABodyBuilder3 tree: $ABB3_DIR"

    local yml="environment_gpu.yml"
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null 2>&1; then
        warn "No GPU detected — using ABB3 CPU environment (slow for inference)"
        yml="environment_cpu.yml"
    else
        log "ABB3: GPU detected — using environment_gpu.yml"
    fi

    if env_exists "$ENV_ABB3"; then
        log "Env $ENV_ABB3 already exists — skipping conda create"
    else
        log "Creating conda env: $ENV_ABB3 from $ABB3_DIR/$yml"
        PIP_REQUIRE_VIRTUALENV=false "$CONDA_CMD" env create -f "$ABB3_DIR/$yml" -n "$ENV_ABB3" -y
    fi

    log "ABB3: pip install editable package (pinned versions)"
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_ABB3" pip install --upgrade pip
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_ABB3" pip install -e "$ABB3_DIR" \
        --constraint "$ABB3_DIR/pinned-versions.txt"

    ensure_abb3_lightning
    install_abb3_weights
}

# ABB3 imports ``lightning.pytorch``; pip can leave only ``pytorch-lightning`` installed.
ensure_abb3_lightning() {
    env_exists "$ENV_ABB3" || return 0
    local py_check='import lightning.pytorch as pl; print(pl.__version__)'
    if conda_run "$ENV_ABB3" python -c "$py_check" &>/dev/null; then
        log "ABB3: lightning.pytorch import OK"
        return 0
    fi
    warn "ABB3: missing 'lightning' package (pytorch-lightning alone is not enough)"
    log "ABB3: installing lightning==2.1.2"
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_ABB3" python -m pip install 'lightning==2.1.2'
    conda_run "$ENV_ABB3" python -c "$py_check" \
        || die "ABB3: lightning.pytorch still not importable — try: conda run -n $ENV_ABB3 python -m pip install 'lightning==2.1.2'"
}

install_abb3_weights() {
    local ckpt="$ABB3_DIR/output/plddt-loss/best_second_stage.ckpt"
    if [[ -f "$ckpt" ]]; then
        log "ABB3 checkpoint already present: $ckpt"
        return 0
    fi
    log "ABB3: downloading inference weights from Zenodo (output.tar.gz) …"
    mkdir -p "$ABB3_DIR/zenodo" "$ABB3_DIR/output"
    local url="https://zenodo.org/records/11354577/files/output.tar.gz"
    if command -v wget &>/dev/null; then
        wget -O "$ABB3_DIR/zenodo/output.tar.gz" "$url"
    elif command -v curl &>/dev/null; then
        curl -fsSL -o "$ABB3_DIR/zenodo/output.tar.gz" "$url"
    else
        die "wget or curl required to download ABB3 weights"
    fi
    (cd "$ABB3_DIR" && tar -xzf zenodo/output.tar.gz -C output/)
    if [[ ! -f "$ckpt" ]]; then
        die "Expected checkpoint not found at $ckpt after extract"
    fi
    log "ABB3 checkpoint ready: $ckpt"
}

install_flashabb() {
    [[ -d "$FLASHABB_DIR" ]] || die "FlashABB directory not found: $FLASHABB_DIR (expected vendored clone at FlashABB/)"
    create_env_from_yml "$ENV_FLASHABB" "$REPO_ROOT/environment-flashabb.yml"
    if ! env_exists "$ENV_FLASHABB"; then
        die "Env $ENV_FLASHABB missing after create"
    fi
    local torch_index
    torch_index="$(detect_torch_index)"
    log "FlashABB: installing PyTorch from $torch_index"
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_FLASHABB" pip install --upgrade pip
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_FLASHABB" pip install torch --index-url "$torch_index"
    log "FlashABB: installing flash-abb from $FLASHABB_DIR"
    PIP_REQUIRE_VIRTUALENV=false conda_run "$ENV_FLASHABB" pip install -e "$FLASHABB_DIR"
    if ! conda_run_with_env_lib "$ENV_FLASHABB" python -c "from flash_abb import pretrained; import openmm" 2>/dev/null; then
        warn "FlashABB import check failed (often fixed by: source fastab.local.env)"
    fi
}

conda_env_bin_dir() {
    local env_name="$1"
    conda_run "$env_name" python -c "import sys; from pathlib import Path; print(Path(sys.executable).parent)" 2>/dev/null || true
}

conda_env_lib_dir() {
    local env_name="$1"
    conda_run "$env_name" python -c "import sys; from pathlib import Path; print(Path(sys.executable).resolve().parent.parent / 'lib')" 2>/dev/null || true
}

conda_run_with_env_lib() {
    local env_name="$1"
    shift
    local lib_dir
    lib_dir="$(conda_env_lib_dir "$env_name")"
    if [[ -n "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" conda_run "$env_name" "$@"
    else
        conda_run "$env_name" "$@"
    fi
}

write_fastab_env() {
    local out="$REPO_ROOT/fastab.local.env"
    local fastab_bin abb2_lib abb3_lib flashabb_lib ld_add=""
    fastab_bin="$(conda_env_bin_dir "$ENV_FASTAB")"
    abb2_lib="$(conda_env_lib_dir "$ENV_ABB2")"
    abb3_lib="$(conda_env_lib_dir "$ENV_ABB3")"
    flashabb_lib="$(conda_env_lib_dir "$ENV_FLASHABB")"
    [[ -n "$abb2_lib" ]] && ld_add="$abb2_lib"
    [[ -n "$abb3_lib" ]] && ld_add="${ld_add:+$ld_add:}$abb3_lib"
    [[ -n "$flashabb_lib" ]] && ld_add="${ld_add:+$ld_add:}$flashabb_lib"
    cat >"$out" <<EOF
# Generated by install.sh — source before running FASTAb pipelines:
#   source fastab.local.env
# If you conda activate fastab-abb3 or fastab-abb2 afterward, source this file
# *before* activating (PATH below prepends the fastab env for mkdssp/propka3/etc.).

FASTAB_ROOT="$REPO_ROOT"

export FASTAB_ENV=$ENV_FASTAB
export FASTAB_ENV_ABB2=$ENV_ABB2
export FASTAB_ENV_ABB3=$ENV_ABB3
export FASTAB_ENV_FLASHABB=$ENV_FLASHABB

export DSSP_BIN=mkdssp

export ABB3_SRC="$ABB3_DIR/src"
export ABB3_CHECKPOINT="$ABB3_DIR/output/plddt-loss/best_second_stage.ckpt"
export ABB3_DEVICE=\${ABB3_DEVICE:-cuda:0}
export ABB3_BATCH_SIZE=\${ABB3_BATCH_SIZE:-4}

export ABB2_DEVICE=\${ABB2_DEVICE:-cuda:0}
export ABB2_BATCH_SIZE=\${ABB2_BATCH_SIZE:-4}
export ABB2_PYTHON="conda run -n ${ENV_ABB2} python"

export FLASHABB_DEVICE=\${FLASHABB_DEVICE:-cuda:0}
export FLASHABB_BATCH_SIZE=\${FLASHABB_BATCH_SIZE:-50}

export PY="conda run -n ${ENV_FASTAB} python"
EOF
    if [[ -n "$fastab_bin" ]]; then
        cat >>"$out" <<EOF

# mkdssp, propka3, freesasa, mmseqs, GNU parallel (install.sh)
export PATH="$fastab_bin:\${PATH}"
EOF
    fi
    if [[ -n "$ld_add" ]]; then
        cat >>"$out" <<EOF

# OpenMM (ABB2/ABB3/FlashABB) needs conda's libstdc++ when CUDA/system paths are on LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$ld_add:\${LD_LIBRARY_PATH:-}"
EOF
    fi
    log "Wrote $out"
}

doctor() {
    local ok=0
    echo ""
    echo "=== FASTAb doctor ==="

    check_env() {
        local name="$1"
        if env_exists "$name"; then
            echo "  OK   conda env: $name"
        else
            echo "  FAIL conda env: $name (missing)"
            ok=1
        fi
    }

    check_cmd_in_env() {
        local env="$1"
        local cmd="$2"
        if env_exists "$env" && conda_run "$env" which "$cmd" &>/dev/null; then
            echo "  OK   $env: $(conda_run "$env" which "$cmd")"
        else
            echo "  FAIL $env: $cmd not on PATH"
            ok=1
        fi
    }

    check_env "$ENV_FASTAB"
    env_exists "$ENV_FASTAB" && {
        check_cmd_in_env "$ENV_FASTAB" mkdssp
        check_cmd_in_env "$ENV_FASTAB" freesasa
        check_cmd_in_env "$ENV_FASTAB" propka3
        check_cmd_in_env "$ENV_FASTAB" mmseqs
        check_cmd_in_env "$ENV_FASTAB" parallel
        conda_run "$ENV_FASTAB" python -c "import pandas, sklearn, scipy" 2>/dev/null \
            && echo "  OK   $ENV_FASTAB: pandas/sklearn/scipy import" \
            || { echo "  FAIL $ENV_FASTAB: Python imports"; ok=1; }
    }

    if [[ "$INSTALL_ABB2" -eq 1 ]]; then
        check_env "$ENV_ABB2"
        env_exists "$ENV_ABB2" && {
            if conda_run_with_env_lib "$ENV_ABB2" python -c "from ImmuneBuilder import ABodyBuilder2; import openmm" 2>/dev/null; then
                echo "  OK   $ENV_ABB2: ImmuneBuilder + OpenMM import"
            else
                echo "  FAIL $ENV_ABB2: ImmuneBuilder/OpenMM import" >&2
                conda_run_with_env_lib "$ENV_ABB2" python -c "from ImmuneBuilder import ABodyBuilder2; import openmm" 2>&1 \
                    | sed 's/^/         /' >&2 || true
                ok=1
            fi
        }
    else
        echo "  SKIP abb2 checks ($INSTALL_MODE)"
        env_exists "$ENV_ABB2" && echo "  NOTE $ENV_ABB2 exists (not installed this run)"
    fi

    if [[ "$INSTALL_ABB3" -eq 1 ]]; then
        check_env "$ENV_ABB3"
        [[ -d "$ABB3_DIR/src/abodybuilder3" ]] && echo "  OK   ABB3_SRC: $ABB3_DIR/src" \
            || { echo "  FAIL ABB3_SRC: $ABB3_DIR/src"; ok=1; }
        local ckpt="$ABB3_DIR/output/plddt-loss/best_second_stage.ckpt"
        [[ -f "$ckpt" ]] && echo "  OK   checkpoint: $ckpt" \
            || { echo "  FAIL checkpoint missing: $ckpt"; ok=1; }
        env_exists "$ENV_ABB3" && {
            if conda_run_with_env_lib "$ENV_ABB3" python -c \
                "import lightning.pytorch; import sys; sys.path.insert(0,'$ABB3_DIR/src'); from abodybuilder3.lightning_module import LitABB3; import torch; print('torch', torch.__version__)" 2>/dev/null; then
                echo "  OK   $ENV_ABB3: lightning + LitABB3 + torch import"
            else
                echo "  FAIL $ENV_ABB3: abodybuilder3 import" >&2
                ok=1
            fi
        }
    else
        echo "  SKIP abb3 checks ($INSTALL_MODE)"
        env_exists "$ENV_ABB3" && echo "  NOTE $ENV_ABB3 exists (not installed this run)"
    fi

    if [[ "$INSTALL_FLASHABB" -eq 1 ]]; then
        check_env "$ENV_FLASHABB"
        env_exists "$ENV_FLASHABB" && {
            if conda_run_with_env_lib "$ENV_FLASHABB" python -c \
                "from flash_abb import pretrained; import openmm; import torch; print('torch', torch.__version__)" 2>/dev/null; then
                echo "  OK   $ENV_FLASHABB: flash_abb + openmm + torch import"
            else
                echo "  FAIL $ENV_FLASHABB: flash_abb/openmm import" >&2
                conda_run_with_env_lib "$ENV_FLASHABB" python -c \
                    "from flash_abb import pretrained; import openmm" 2>&1 \
                    | sed 's/^/         /' >&2 || true
                ok=1
            fi
        }
    else
        echo "  SKIP flashabb checks ($INSTALL_MODE)"
        env_exists "$ENV_FLASHABB" && echo "  NOTE $ENV_FLASHABB exists (not installed this run)"
    fi

    command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -3 \
        || echo "  NOTE nvidia-smi not available (CPU-only inference)"

    echo "=== end doctor ==="
    return "$ok"
}

main() {
    log "FASTAb install (root: $REPO_ROOT)"
    if [[ "$INSTALL_MODE" != "full" ]]; then
        log "Mode: --$INSTALL_MODE"
    fi

    install_fastab
    if [[ "$INSTALL_ABB2" -eq 1 ]]; then
        install_abb2
    fi
    if [[ "$INSTALL_ABB3" -eq 1 ]]; then
        install_abb3
    fi
    if [[ "$INSTALL_FLASHABB" -eq 1 ]]; then
        install_flashabb
    fi

    write_fastab_env

    echo ""
    log "Done. Next steps:"
    echo "  source fastab.local.env"
    echo "  # optional: conda activate $ENV_ABB3  (source fastab.local.env first)"
    echo "  cd src && ./predict_structure.sh --help"
    echo "  conda activate $ENV_FASTAB && ./src/get_descriptors.sh --help"

    doctor || die "Doctor checks failed — review output above"
}

main "$@"
