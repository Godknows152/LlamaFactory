#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMAFACTORY_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
DATA_BUILDER="${SCRIPT_DIR}/build_expert_sft_dataset_first_token.py"
DATA_PYTHON="${DATA_PYTHON:-/home/LXJ/anaconda3/envs/agent-lightning/bin/python}"
MANIFEST_PATH="${PROJECT_DIR}/data/first_token_actions/manifest.json"
LOG_DIR="${LLAMAFACTORY_DIR}/logs/first_token_actions_4gpu"
PID_FILE="${LOG_DIR}/four_experts_first_token_sft_4gpu.pid"
EXPERTS=(fog snow rain low_light)

mkdir -p "${LOG_DIR}"

if [[ "${SFT_RUN_IN_FOREGROUND:-0}" != "1" && "${SFT_BACKGROUND_CHILD:-0}" != "1" ]]; then
  if [[ -s "${PID_FILE}" ]]; then
    EXISTING_PID="$(<"${PID_FILE}")"
    if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
      echo "Four-expert first-token SFT is already running (PID ${EXISTING_PID})." >&2
      exit 1
    fi
    unlink "${PID_FILE}"
  fi

  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  MAIN_LOG="${LOG_DIR}/four_experts_first_token_sft_4gpu_${TIMESTAMP}.log"
  nohup env \
    SFT_BACKGROUND_CHILD=1 \
    SFT_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!

  echo "Started four-expert first-token SFT in the background (PID ${BACKGROUND_PID})."
  echo "GPU devices per expert: 0,1,2,3"
  echo "Experts are trained serially."
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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo "========== Four-expert first-token-distinct SFT started =========="
echo "Start time: $(date --iso-8601=seconds)"
echo "PID: $$"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Epochs per expert: 3"
echo "Effective batch size: 4 GPUs x 4 samples x 2 accumulation = 32"
echo "Output root: ${PROJECT_DIR}/outputs/qwen3_5_0731"
echo "Log file: ${MAIN_LOG}"

manifest_is_current() {
  "${DATA_PYTHON}" -c '
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
experts = data.get("experts", {})
valid = (
    data.get("version") == 5
    and data.get("purpose") == "first_turn_four_expert_sft_with_first_token_distinct_action_aliases"
    and data.get("enable_thinking") is True
    and all(experts.get(name, {}).get("num_samples") == 1000 for name in ("fog", "snow", "rain", "low_light"))
)
sys.exit(0 if valid else 1)
' "${MANIFEST_PATH}" >/dev/null 2>&1
}

if [[ "${REBUILD_DATA:-0}" == "1" ]] || ! manifest_is_current; then
  echo "========== Build 4 x 1,000 first-token-distinct SFT samples =========="
  "${DATA_PYTHON}" "${DATA_BUILDER}"
else
  echo "========== Reuse validated first-token-distinct SFT datasets =========="
fi

for expert in "${EXPERTS[@]}"; do
  echo "========== 4-GPU SFT: ${expert} expert =========="
  bash "${SCRIPT_DIR}/run_expert_sft_first_token_4gpu.sh" "${expert}"
done

echo "========== All four first-token SFT runs completed =========="
echo "End time: $(date --iso-8601=seconds)"
