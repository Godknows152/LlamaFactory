#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_BUILDER="${SCRIPT_DIR}/build_expert_sft_dataset.py"
DATA_PYTHON="${DATA_PYTHON:-/home/LXJ/anaconda3/envs/agent-lightning/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ "${REBUILD_DATA:-0}" == "1" || ! -s "${PROJECT_DIR}/data/manifest.json" ]]; then
  echo "========== Build RL-aligned SFT datasets =========="
  "${DATA_PYTHON}" "${DATA_BUILDER}"
fi

for expert in fog snow rain low_light; do
  echo "========== SFT: ${expert} expert =========="
  bash "${SCRIPT_DIR}/run_expert_sft.sh" "${expert}"
done

echo "========== All four expert SFT runs completed =========="
