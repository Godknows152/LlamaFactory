#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMAFACTORY_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/configs/unified_expert_4gpu/qwen35_unified_expert_lora_sft.yaml"
DATASET_PATH="${PROJECT_DIR}/data/unified_expert_first_token_actions_train.jsonl"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/LXJ/anaconda3/envs/llamafactory/bin/llamafactory-cli}"
# This unified run is intentionally pinned to physical GPUs 0 and 1. Ignore
# any inherited CUDA_VISIBLE_DEVICES/CUDA_DEVICES value so no other GPU is used.
CUDA_DEVICES="0,1"
LOG_DIR="${LLAMAFACTORY_DIR}/logs"
PID_FILE="${LOG_DIR}/unified_expert_sft_2gpu.pid"

mkdir -p "${LOG_DIR}"

if [[ "${SFT_RUN_IN_FOREGROUND:-0}" != "1" && "${SFT_BACKGROUND_CHILD:-0}" != "1" ]]; then
  if [[ -s "${PID_FILE}" ]]; then
    EXISTING_PID="$(<"${PID_FILE}")"
    if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
      echo "Unified expert SFT is already running (PID ${EXISTING_PID})." >&2
      echo "PID file: ${PID_FILE}" >&2
      exit 1
    fi
    unlink "${PID_FILE}"
  fi

  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  MAIN_LOG="${LOG_DIR}/unified_expert_sft_2gpu_${TIMESTAMP}.log"
  nohup env \
    SFT_BACKGROUND_CHILD=1 \
    SFT_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!

  echo "Started unified expert SFT in the background (PID ${BACKGROUND_PID})."
  echo "GPU devices: ${CUDA_DEVICES}"
  echo "Log file: ${MAIN_LOG}"
  echo "PID file: ${PID_FILE}"
  exit 0
fi

if [[ "${SFT_BACKGROUND_CHILD:-0}" == "1" ]]; then
  MAIN_LOG="${SFT_MAIN_LOG:?SFT_MAIN_LOG is required for the background child}"
  printf '%s\n' "$$" >"${PID_FILE}"
  cleanup_pid_file() {
    if [[ -f "${PID_FILE}" ]] && [[ "$(<"${PID_FILE}")" == "$$" ]]; then
      unlink "${PID_FILE}"
    fi
  }
  trap cleanup_pid_file EXIT
else
  MAIN_LOG="/dev/stdout"
fi

echo "========== Unified four-expert SFT =========="
echo "Start time: $(date --iso-8601=seconds)"
echo "PID: $$"
echo "Configuration: ${CONFIG_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
echo "Epochs: 3"
echo "Batch configuration: read from ${CONFIG_PATH}"
echo "Log file: ${MAIN_LOG}"
echo

LLAMAFACTORY_BIN_DIR="$(dirname "${LLAMAFACTORY_CLI}")"
export PATH="${LLAMAFACTORY_BIN_DIR}:${PATH}"
export PYTHONPATH="${LLAMAFACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf 'CUDA_VISIBLE_DEVICES=%q FORCE_TORCHRUN=1 PYTHONUNBUFFERED=1 %q train %q\n' \
  "${CUDA_DEVICES}" "${LLAMAFACTORY_CLI}" "${CONFIG_PATH}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: command validation passed; training was not started."
  exit 0
fi

cd "${LLAMAFACTORY_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
FORCE_TORCHRUN=1 \
PYTHONUNBUFFERED=1 \
"${LLAMAFACTORY_CLI}" train "${CONFIG_PATH}"

echo
echo "End time: $(date --iso-8601=seconds)"
