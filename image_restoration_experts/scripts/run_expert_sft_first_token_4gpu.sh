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
CONFIG_PATH="${PROJECT_DIR}/configs/first_token_actions_4gpu/qwen35_${EXPERT}_expert_lora_sft.yaml"
DATA_DIR="${PROJECT_DIR}/data/first_token_actions"
DATASET_PATH="${DATA_DIR}/${EXPERT}_expert_first_token_actions_train.jsonl"
MANIFEST_PATH="${DATA_DIR}/manifest.json"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/LXJ/anaconda3/envs/llamafactory/bin/llamafactory-cli}"
DATA_PYTHON="${DATA_PYTHON:-/home/LXJ/anaconda3/envs/agent-lightning/bin/python}"
CUDA_DEVICES="0,1,2,3"

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
if [[ ! -f "${CONFIG_PATH}" || ! -s "${DATASET_PATH}" || ! -s "${MANIFEST_PATH}" ]]; then
  echo "First-token SFT configuration, dataset, or manifest is missing." >&2
  echo "Run scripts/run_all_expert_sft_first_token_4gpu.sh or rebuild the data first." >&2
  exit 1
fi
if [[ ! -x "${DATA_PYTHON}" ]]; then
  echo "Dataset validation Python is not executable: ${DATA_PYTHON}" >&2
  exit 1
fi

"${DATA_PYTHON}" - "${MANIFEST_PATH}" "${EXPERT}" "${DATASET_PATH}" "${CONFIG_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
expert = sys.argv[2]
dataset_path = Path(sys.argv[3])
config_path = Path(sys.argv[4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
train_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

expected_purpose = "first_turn_four_expert_sft_with_first_token_distinct_action_aliases"
if (
    manifest.get("version") != 5
    or manifest.get("purpose") != expected_purpose
    or manifest.get("enable_thinking") is not True
):
    raise SystemExit("Expected version 5 first-token-distinct SFT data; rebuild the dataset.")

runtime_to_model = manifest.get("action_vocabulary", {}).get("runtime_to_model", {})
if len(runtime_to_model) != 17 or len({alias[0] for alias in runtime_to_model.values()}) != 17:
    raise SystemExit("The manifest does not contain 17 unique action initials.")
if runtime_to_model.get("stop") != "stop":
    raise SystemExit("The stop action alias is missing or unexpected.")

entry = manifest.get("experts", {}).get(expert)
if not isinstance(entry, dict) or entry.get("file_name") != dataset_path.name:
    raise SystemExit(f"The manifest has no valid entry for expert {expert!r}.")
if entry.get("num_samples") != 1000 or entry.get("source_images") != 1000:
    raise SystemExit("Each expert dataset must contain exactly 1,000 first-turn samples.")
if sum(1 for line in dataset_path.open(encoding="utf-8") if line.strip()) != 1000:
    raise SystemExit("Dataset row count does not match the required 1,000 samples.")
if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != entry.get("sha256"):
    raise SystemExit("Dataset checksum does not match the manifest; rebuild the SFT data.")

with dataset_path.open(encoding="utf-8") as file:
    first_row = json.loads(next(line for line in file if line.strip()))
metadata = first_row.get("metadata", {})
alias = metadata.get("selected_action_alias")
canonical = metadata.get("selected_action")
if runtime_to_model.get(canonical) != alias:
    raise SystemExit("The first row has an inconsistent canonical-to-model action mapping.")
target = first_row.get("messages", [])[-1].get("content", "")
if target.count(alias) != 2 or not target.startswith("<think>\n"):
    raise SystemExit("The action alias must appear once in thinking and once in the tool call.")
schema_actions = json.loads(first_row["tools"])[0]["function"]["parameters"]["properties"]["action"]["enum"]
if len(schema_actions) != 16 or "stop" in schema_actions or alias not in schema_actions:
    raise SystemExit("The first-turn tool schema does not contain the 16 expected aliases.")

project_dir = config_path.parents[2]
expected_output_dir = project_dir / "outputs/qwen3_5_0731/format_cold_start" / expert
expected_swanlab_logdir = project_dir / "outputs/qwen3_5_0731/swanlab" / expert
expected_dataset = f"image_restoration_{expert}_expert_first_token_actions_train"
checks = {
    "dataset": train_config.get("dataset") == expected_dataset,
    "dataset_dir": Path(train_config.get("dataset_dir", "")).resolve() == dataset_path.parent.resolve(),
    "output_dir": Path(train_config.get("output_dir", "")).resolve() == expected_output_dir.resolve(),
    "swanlab_logdir": Path(train_config.get("swanlab_logdir", "")).resolve() == expected_swanlab_logdir.resolve(),
    "enable_thinking": train_config.get("enable_thinking") is True,
    "epochs": train_config.get("num_train_epochs") == 3,
    "per_device_batch": train_config.get("per_device_train_batch_size") == 4,
    "gradient_accumulation": train_config.get("gradient_accumulation_steps") == 2,
    "swanlab": train_config.get("use_swanlab") is True,
    "swanlab_project": train_config.get("swanlab_project") == "v4.1.1SFT",
    "swanlab_run_name": train_config.get("swanlab_run_name") == expert,
}
failed = [name for name, valid in checks.items() if not valid]
if failed:
    raise SystemExit(f"Invalid 4-GPU first-token SFT configuration fields: {failed}")
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
