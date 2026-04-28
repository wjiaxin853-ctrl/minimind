#!/usr/bin/env bash

# Usage:
#   source /media/admin204/2tbdisk2/wjx/minimind/env.remote.sh
# This will:
# 1. switch to the remote project directory
# 2. deactivate conda base if active
# 3. export project-local cache directories on the 2TB disk
# 4. activate this project's .venv

PROJECT_ROOT=/media/admin204/2tbdisk2/wjx/minimind

cd "$PROJECT_ROOT"

# Exit conda environment if one is active.
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  conda deactivate || true
fi

# Project-local cache locations on the remote 2TB disk.
export HF_HOME=/media/admin204/2tbdisk2/wjx/minimind-cache/hf
export HF_DATASETS_CACHE=/media/admin204/2tbdisk2/wjx/minimind-cache/hf/datasets
export UV_CACHE_DIR=/media/admin204/2tbdisk2/wjx/minimind-cache/uv
export PIP_CACHE_DIR=/media/admin204/2tbdisk2/wjx/minimind-cache/pip

source "$PROJECT_ROOT/.venv/bin/activate"

echo "Project environment ready:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  HF_HOME=$HF_HOME"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  VIRTUAL_ENV=$VIRTUAL_ENV"
