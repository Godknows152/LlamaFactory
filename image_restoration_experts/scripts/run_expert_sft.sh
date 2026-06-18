#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "run_expert_sft.sh now delegates to the Qwen3.5 format cold-start SFT entrypoint."
exec bash "${SCRIPT_DIR}/run_expert_format_sft.sh" "$@"
