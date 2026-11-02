#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用: source /home/kk/kk_data/lerobot/activate-train.sh"
    exit 1
fi

source /home/kk/kk_data/lerobot/env.sh
source "${LEROBOT_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lerobot-train

