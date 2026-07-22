#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMAFACTORY_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
DATA_BUILDER="${SCRIPT_DIR}/build_expert_sft_dataset.py"
DATA_PYTHON="${DATA_PYTHON:-/home/LXJ/anaconda3/envs/agent-lightning/bin/python}"
MANIFEST_PATH="${PROJECT_DIR}/data/manifest.json"
LOG_DIR="${LLAMAFACTORY_DIR}/logs"
PID_FILE="${LOG_DIR}/four_experts_sft.pid"
EXPERTS=(fog snow rain low_light)

mkdir -p "${LOG_DIR}"

if [[ "${SFT_RUN_IN_FOREGROUND:-0}" != "1" && "${SFT_BACKGROUND_CHILD:-0}" != "1" ]]; then
  if [[ -s "${PID_FILE}" ]]; then
    EXISTING_PID="$(<"${PID_FILE}")"
    if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
      echo "Four-expert SFT is already running (PID ${EXISTING_PID})." >&2
      exit 1
    fi
    unlink "${PID_FILE}"
  fi

  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  MAIN_LOG="${LOG_DIR}/four_experts_sft_${TIMESTAMP}.log"
  nohup env \
    SFT_BACKGROUND_CHILD=1 \
    SFT_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!

  echo "Started four-expert SFT in the background (PID ${BACKGROUND_PID})."
  echo "GPU devices: 0,1"
  echo "Log file: ${MAIN_LOG}"
  exit 0
fi

if [[ "${SFT_BACKGROUND_CHILD:-0}" == "1" ]]; then
  MAIN_LOG="${SFT_MAIN_LOG:?SFT_MAIN_LOG is required for the background child}"
  printf '%s\n' "$$" >"${PID_FILE}"
  cleanup_pid_file() {
    if [[ -f "${PID_FILE}" ]]; then
      unlink "${PID_FILE}"
    fi
  }
  trap cleanup_pid_file EXIT
else
  MAIN_LOG="/dev/stdout"
fi

# This workflow is intentionally pinned to physical GPU 0 and GPU 1. Ignore
# any inherited CUDA_VISIBLE_DEVICES value so no other GPU can be selected.
export CUDA_VISIBLE_DEVICES="0,1"

echo "========== Four-expert thinking-enabled first-turn SFT started =========="
echo "Start time: $(date --iso-8601=seconds)"
echo "PID: $$"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Epochs per expert: 3"
echo "Output root: ${PROJECT_DIR}/outputs/qwen3_5_0721"
echo "SwanLab project: image-restoration-expert-sft"
echo "Log file: ${MAIN_LOG}"

manifest_is_current() {
  "${DATA_PYTHON}" -c '
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
valid = (
    data.get("version") == 4
    and data.get("purpose") == "first_turn_only_rl_aligned_four_expert_sft_with_short_reasoning_targets"
    and data.get("enable_thinking") is True
)
sys.exit(0 if valid else 1)
' "${MANIFEST_PATH}" >/dev/null 2>&1
}

if [[ "${REBUILD_DATA:-0}" == "1" ]] || ! manifest_is_current; then
  echo "========== Build thinking-enabled first-turn-only RL-aligned SFT datasets =========="
  "${DATA_PYTHON}" "${DATA_BUILDER}"
else
  echo "========== Reuse validated thinking-enabled first-turn-only SFT datasets =========="
fi

for expert in "${EXPERTS[@]}"; do
  echo "========== SFT: ${expert} expert =========="
  bash "${SCRIPT_DIR}/run_expert_sft.sh" "${expert}"
done

echo "========== All four expert SFT runs completed =========="
echo "End time: $(date --iso-8601=seconds)"
