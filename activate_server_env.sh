#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -n "${TR3D_VENV:-}" ]; then
    VENV_PATH="$TR3D_VENV"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    VENV_PATH="$SCRIPT_DIR/venv"
else
    VENV_PATH=""
fi

if [ -n "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    echo "Activated virtualenv: $VENV_PATH"
else
    echo "No virtualenv auto-detected. Set TR3D_VENV=/path/to/venv if needed."
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export H="$HOME"

# These defaults describe the layout this codebase was developed against
# (see MinkowskiEngine/MinkowskiEngine_Installation_Guide.md). Override any of
# them for your machine, or skip this script entirely if your CUDA toolchain is
# already on PATH.
export CUDA_HOME="${CUDA_HOME:-$H/local/cuda-11.3}"
export OPENBLAS_PREFIX="${OPENBLAS_PREFIX:-$H/local/openblas}"
export OPT_GCC9_ROOT="${OPT_GCC9_ROOT:-$H/opt/gcc-9}"

for var in CUDA_HOME OPENBLAS_PREFIX OPT_GCC9_ROOT; do
    if [ ! -d "${!var}" ]; then
        echo "warning: $var=${!var} does not exist; set it for your machine." >&2
    fi
done

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$OPENBLAS_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$OPT_GCC9_ROOT/ROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "Repo root: $SCRIPT_DIR"
echo "Python: $(command -v python || true)"
echo "PYTHONPATH: $PYTHONPATH"
