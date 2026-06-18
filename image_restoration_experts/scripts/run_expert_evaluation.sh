#!/usr/bin/env bash

set -euo pipefail

LLAMAFACTORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="${LLAMAFACTORY_DIR}/image_restoration_experts"
PYTHON="${PYTHON:-/home/LXJ/anaconda3/envs/llamafactory/bin/python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${PROJECT_DIR}/log/expert_evaluation_${TIMESTAMP}.log"
LATEST_LOG="${PROJECT_DIR}/log/expert_evaluation_latest.log"

mkdir -p "${PROJECT_DIR}/log"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Start time: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
echo "Log file: ${LOG_FILE}"
echo

cd "${LLAMAFACTORY_DIR}"
PYTHONUNBUFFERED=1 "${PYTHON}" \
  image_restoration_experts/scripts/evaluate_expert_agents.py \
  --gpus "${CUDA_DEVICES}" \
  --batch-size "${BATCH_SIZE:-32}" \
  "$@"
