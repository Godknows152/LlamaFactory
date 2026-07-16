# Image Restoration Expert SFT

This directory builds four independent Qwen3.5 LoRA adapters for fog, snow,
rain, and low-light restoration. The SFT inputs are aligned with the current
old-VERL restoration decisions:

- the current image and current expert prompt;
- the complete `restore_image` schema used at that decision;
- compact historical action and IQA aggregate feedback;
- no `stop` enum before three completed restoration calls;
- both continue and stop targets after `stop` becomes available.

The datasets do not contain reasoning ground truth. Training uses the
`qwen3_5` template with `enable_thinking: false`; the template places an empty
thinking block in the prompt and masks it from loss. Every supervised response
is one structured `restore_image` function call, rendered in the native
Qwen3.5 XML format expected by the RL `qwen3_coder` parser.

This setting is SFT-only. The old-VERL GRPO configuration is intentionally
unchanged, so GRPO rollouts retain the model's current thinking mode and can
learn their reasoning policy from reward instead of synthetic reasoning labels.

## Build Data

From the Agent Lightning repository root:

```bash
/home/LXJ/anaconda3/envs/agent-lightning/bin/python \
  LlamaFactory/image_restoration_experts/scripts/build_expert_sft_dataset.py
```

Each of the 1,000 source images per expert produces five decision-local rows:
initial action, continuation after improvement, recovery after regression,
continuation when stop is available but quality is insufficient, and stop after
sufficient quality gain.

## Train All Experts

The launcher trains Fog, Snow, Rain, and Low-light serially on GPUs 0 and 1 and
stops immediately if one run fails:

```bash
bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=2,3 DRY_RUN=1 \
  bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft.sh

REBUILD_DATA=1 \
  bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft.sh
```

Adapters are written to
`outputs/qwen3_5/format_cold_start/{fog,snow,rain,low_light}`, which preserves
the paths already consumed by the four RL configurations.

## Included Files

- `scripts/build_expert_sft_dataset.py`: stages images and builds validated RL-aligned ShareGPT datasets.
- `scripts/run_expert_sft.sh`: launches one expert with preflight checks and timestamped logging.
- `scripts/run_all_expert_sft.sh`: optionally rebuilds data and trains all four experts serially.
- `configs/qwen35_<expert>_expert_lora_sft.yaml`: independent no-thinking LoRA configurations.
- `data/dataset_info.json`: LLaMA-Factory dataset registration generated with the datasets.
- `data/manifest.json`: generated sample counts, action distributions, checksums, and alignment metadata.
