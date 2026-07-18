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
MANIFEST_PATH="${PROJECT_DIR}/data/manifest.json"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/LXJ/anaconda3/envs/llamafactory/bin/llamafactory-cli}"
DATA_PYTHON="${DATA_PYTHON:-/home/LXJ/anaconda3/envs/agent-lightning/bin/python}"
CUDA_DEVICES="0,1"

echo "Start time: $(date --iso-8601=seconds)"
echo "Expert: ${EXPERT}"
echo "Configuration: ${CONFIG_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
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
if [[ ! -s "${MANIFEST_PATH}" ]]; then
  echo "Dataset manifest does not exist or is empty: ${MANIFEST_PATH}" >&2
  echo "Run scripts/run_all_expert_sft.sh or rebuild the data first." >&2
  exit 1
fi
if [[ ! -x "${DATA_PYTHON}" ]]; then
  echo "Dataset validation Python is not executable: ${DATA_PYTHON}" >&2
  exit 1
fi

"${DATA_PYTHON}" - "${MANIFEST_PATH}" "${EXPERT}" "${DATASET_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expert = sys.argv[2]
dataset_path = Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

expected_purpose = "first_turn_only_rl_aligned_four_expert_sft_without_reasoning_targets"
if manifest.get("version") != 2 or manifest.get("purpose") != expected_purpose:
    raise SystemExit(
        "The SFT manifest is stale: expected first-turn-only version 2 data. "
        "Run run_all_expert_sft.sh with REBUILD_DATA=1."
    )

entry = manifest.get("experts", {}).get(expert)
if not isinstance(entry, dict):
    raise SystemExit(f"The SFT manifest has no entry for expert {expert!r}.")
if entry.get("file_name") != dataset_path.name:
    raise SystemExit(f"Manifest file name does not match {dataset_path.name}.")
if entry.get("samples_per_input_image") != 1 or entry.get("num_samples") != entry.get("source_images"):
    raise SystemExit("Manifest does not describe exactly one first-turn sample per input image.")

line_count = sum(1 for line in dataset_path.open(encoding="utf-8") if line.strip())
if line_count != entry.get("num_samples"):
    raise SystemExit(f"Dataset row count {line_count} does not match manifest count {entry.get('num_samples')}.")
digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
if digest != entry.get("sha256"):
    raise SystemExit("Dataset checksum does not match the manifest; rebuild the SFT data.")
PY

LLAMAFACTORY_BIN_DIR="$(dirname "${LLAMAFACTORY_CLI}")"
export PATH="${LLAMAFACTORY_BIN_DIR}:${PATH}"
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
