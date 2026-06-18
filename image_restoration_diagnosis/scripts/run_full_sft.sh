#!/usr/bin/env bash

set -euo pipefail

LLAMAFACTORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${LLAMAFACTORY_DIR}/image_restoration_diagnosis/configs/qwen35_diagnosis_lora_sft.yaml"
LOG_DIR="${LLAMAFACTORY_DIR}/log"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/LXJ/anaconda3/envs/llamafactory/bin/llamafactory-cli}"
LLAMAFACTORY_ENV_BIN="$(dirname "${LLAMAFACTORY_CLI}")"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/diagnosis_sft_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/diagnosis_sft_latest.log"

mkdir -p "${LOG_DIR}"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

# Capture every subsequent stdout/stderr line, including preflight failures.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Start time: $(date --iso-8601=seconds)"
echo "Script PID: $$"
echo "Working directory: ${LLAMAFACTORY_DIR}"
echo "Configuration: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
echo "LlamaFactory environment bin: ${LLAMAFACTORY_ENV_BIN}"
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

if [[ ! -x "${LLAMAFACTORY_ENV_BIN}/python" || ! -x "${LLAMAFACTORY_ENV_BIN}/torchrun" ]]; then
  echo "Python or torchrun is missing from: ${LLAMAFACTORY_ENV_BIN}" >&2
  exit 1
fi

export PATH="${LLAMAFACTORY_ENV_BIN}:${PATH}"
export PYTHONPATH="${LLAMAFACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Python executable: $(command -v python)"
echo "Torchrun executable: $(command -v torchrun)"
python -c "import llamafactory; print('LlamaFactory import:', llamafactory.__file__)"
echo

echo "Training command:"
printf 'CUDA_VISIBLE_DEVICES=%q FORCE_TORCHRUN=1 PYTHONUNBUFFERED=1 %q train %q\n' \
  "${CUDA_DEVICES}" "${LLAMAFACTORY_CLI}" "${CONFIG_PATH}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: preflight checks passed; training was not started."
  echo "End time: $(date --iso-8601=seconds)"
  echo "Training exit code: 0"
  exit 0
fi

cd "${LLAMAFACTORY_DIR}"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
FORCE_TORCHRUN=1 \
PYTHONUNBUFFERED=1 \
"${LLAMAFACTORY_CLI}" train "${CONFIG_PATH}"
TRAIN_STATUS=$?
set -e

echo
echo "End time: $(date --iso-8601=seconds)"
echo "Training exit code: ${TRAIN_STATUS}"

exit "${TRAIN_STATUS}"
