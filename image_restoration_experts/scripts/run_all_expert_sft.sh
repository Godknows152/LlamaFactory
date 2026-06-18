#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for expert in fog snow rain low_light; do
  echo "========== Training ${expert} expert =========="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    bash "${SCRIPT_DIR}/run_expert_sft.sh" "${expert}"
done
