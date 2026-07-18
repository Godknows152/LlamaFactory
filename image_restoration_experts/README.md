# Image Restoration Expert SFT

This directory builds four independent Qwen3.5 LoRA adapters for fog, snow,
rain, and low-light restoration. Every SFT row is aligned exclusively with the
first restoration turn in the current old-VERL rollout:

- one original input image;
- the exact first-turn system and user prompts from
  `build_expert_single_step_sft_*`;
- the first-turn `restore_image` schema containing all 16 restoration actions
  and no `stop`;
- exactly one supervised `restore_image` action.

There are no later-turn samples, synthetic action histories, synthetic IQA
feedback, or stop targets. The datasets also do not contain reasoning ground
truth. Training uses the `qwen3_5` template with `enable_thinking: false`; the
template places an empty thinking block in the prompt and masks it from loss.
Every supervised response is stored as one structured `function_call`, which
the template renders in the native Qwen3.5 XML form expected by the RL
`qwen3_coder` parser:

```text
<tool_call>
<function=restore_image>
<parameter=action>
<ACTION>
</parameter>
</function>
</tool_call>
```

This setting is SFT-only. The old-VERL GRPO configuration is intentionally
unchanged, so a later GRPO run can enable thinking and learn its reasoning
policy from reward instead of synthetic reasoning labels.

## Build Data

From the Agent Lightning repository root:

```bash
/home/LXJ/anaconda3/envs/agent-lightning/bin/python \
  LlamaFactory/image_restoration_experts/scripts/build_expert_sft_dataset.py
```

Each of the 1,000 source images per expert produces exactly one first-turn row,
for 1,000 samples per expert and 4,000 samples in total. Targets use a
deterministic round-robin assignment over all 16 registered restoration tools.
Because 1,000 is not divisible by 16, each tool appears either 62 or 63 times
in every expert dataset; the maximum difference is one.

## Train All Experts

The launcher validates the data manifest, rebuilds automatically when an older
multi-turn manifest is found, and starts one detached background process. That
process trains Fog, Snow, Rain, and Low-light serially, stops immediately if
one run fails, and is hard-pinned to physical GPUs 0 and 1:

```bash
bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft.sh
```

The command returns immediately after printing the background PID and log
path. All four experts' stdout and stderr are written to one timestamped file:

```text
LlamaFactory/logs/four_experts_sft_<timestamp>.log
```

SwanLab and other reporting backends are disabled, so this terminal log is the
only runtime log. The active background PID is recorded in
`LlamaFactory/logs/four_experts_sft.pid` and removed when the serial run exits.

Useful commands:

```bash
REBUILD_DATA=1 \
  bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft.sh

tail -f LlamaFactory/logs/four_experts_sft_<timestamp>.log

SFT_RUN_IN_FOREGROUND=1 DRY_RUN=1 \
  bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft.sh
```

Adapters are written to
`outputs/qwen3_5/format_cold_start/{fog,snow,rain,low_light}`, which preserves
the paths already consumed by the four RL configurations.

Every expert is configured for exactly two training epochs.

## Included Files

- `scripts/build_expert_sft_dataset.py`: stages images and builds validated,
  first-turn-only RL-aligned ShareGPT datasets.
- `scripts/run_expert_sft.sh`: rejects stale or modified datasets, then runs
  one expert on GPU 0 and 1.
- `scripts/run_all_expert_sft.sh`: starts the validated four-expert serial
  workflow in the background and owns its single terminal log.
- `configs/qwen35_<expert>_expert_lora_sft.yaml`: independent no-thinking LoRA configurations.
- `data/dataset_info.json`: LLaMA-Factory dataset registration generated with the datasets.
- `data/manifest.json`: generated sample counts, action distributions, checksums, and alignment metadata.
