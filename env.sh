#!/usr/bin/env bash

# Shared LeRobot paths. This file intentionally contains no credentials.
export LEROBOT_WORKSPACE="/home/kk/kk_data/lerobot"
export LEROBOT_CONDA_BASE="${LEROBOT_WORKSPACE}/miniforge3"
export HF_HOME="${LEROBOT_WORKSPACE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_LEROBOT_HOME="${LEROBOT_WORKSPACE}/datasets"
export PIP_CACHE_DIR="${LEROBOT_WORKSPACE}/pip_cache"
export WANDB_DIR="${LEROBOT_WORKSPACE}/logs/wandb"

# Stay local by default. Override with `export WANDB_MODE=online` after login.
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_DISABLE_TELEMETRY="1"
export TOKENIZERS_PARALLELISM="false"

