#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {fog|snow|rain|low_light}" >&2
  exit 2
fi

EXPERT="$1"
case "${EXPERT}" in
  fog|snow|rain|low_light) ;;
  *)
    echo "Unsupported expert: ${EXPERT}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMAFACTORY_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/configs/qwen35_${EXPERT}_expert_lora_sft.yaml"
DATASET_PATH="${PROJECT_DIR}/data/${EXPERT}_expert_rl_aligned_train.jsonl"
LOG_DIR="${LLAMAFACTORY_DIR}/logs"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/LXJ/anaconda3/envs/llamafactory/bin/llamafactory-cli}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOG_FILE="${LOG_DIR}/${EXPERT}_0716.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "Start time: $(date --iso-8601=seconds)"
echo "Expert: ${EXPERT}"
echo "Configuration: ${CONFIG_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
echo "Log file: ${LOG_FILE}"
echo

if [[ ! -x "${LLAMAFACTORY_CLI}" ]]; then
  echo "LlamaFactory CLI is not executable: ${LLAMAFACTORY_CLI}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training configuration does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -s "${DATASET_PATH}" ]]; then
  echo "Training dataset does not exist or is empty: ${DATASET_PATH}" >&2
  exit 1
fi

export PATH="$(dirname "${LLAMAFACTORY_CLI}"):${PATH}"
export PYTHONPATH="${LLAMAFACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf 'CUDA_VISIBLE_DEVICES=%q FORCE_TORCHRUN=1 PYTHONUNBUFFERED=1 %q train %q\n' \
  "${CUDA_DEVICES}" "${LLAMAFACTORY_CLI}" "${CONFIG_PATH}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: preflight checks passed; training was not started."
  exit 0
fi

cd "${LLAMAFACTORY_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
FORCE_TORCHRUN=1 \
PYTHONUNBUFFERED=1 \
"${LLAMAFACTORY_CLI}" train "${CONFIG_PATH}"

echo
echo "End time: $(date --iso-8601=seconds)"
