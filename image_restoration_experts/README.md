# Image Restoration Expert SFT

This directory builds four independent Qwen3.5 LoRA adapters for fog, snow,
rain, and low-light restoration. Every SFT row is aligned exclusively with the
first restoration turn in the current old-VERL rollout:

- one original input image;
- the exact first-turn system and user prompts from
  `build_expert_single_step_sft_*`;
- the first-turn `restore_image` schema containing all 16 restoration actions
  and no `stop`;
- one short supervised reasoning block selected from five alternatives written
  specifically for the target restoration action;
- exactly one supervised `restore_image` action.

There are no later-turn samples, synthetic action histories, synthetic IQA
feedback, or stop targets. Training uses the `qwen3_5` template with
`enable_thinking: true`. Every supervised assistant response contains the
selected reasoning text and the native Qwen3.5 XML form expected by the RL
`qwen3_coder` parser:

```text
<think>
<SHORT ACTION-SPECIFIC REASONING>
</think>

<tool_call>
<function=restore_image>
<parameter=action>
<ACTION>
</parameter>
</function>
</tool_call>
```

## First-Token-Distinct 4-GPU Variant

The new SFT variant keeps the same images, first-turn prompts, 80 reasoning
templates, uniform target distribution, LoRA settings, and three training
epochs. It changes only the model-facing action vocabulary and the GPU
topology. The legacy 2-GPU data, launchers, configurations, and outputs remain
independent.

Every model-facing action begins with a stable unique letter:

```text
A_real_esrgan                 -> real_esrgan
B_scunet                      -> scunet
C_retinexformer_fivek         -> retinexformer_fivek
D_hvicidnet                   -> hvicidnet
E_lightdiff                   -> lightdiff
F_turbo_rain                  -> turbo_rain
G_s2former                    -> s2former
H_idt                         -> idt
I_ridcp                       -> ridcp
J_kanet                       -> kanet
K_turbo_snow                  -> turbo_snow
L_snowmaster                  -> snowmaster
M_nafnet_denoise              -> nafnet_denoise
N_focalnet_dehaze             -> focalnet_dehaze
O_focalnet_desnow             -> focalnet_desnow
P_mb_taylorformer_dehaze      -> mb_taylorformer_dehaze
stop                          -> stop
```

The first-turn SFT data contains aliases A through P; native `stop` is reserved
for later GRPO turns and is not a first-turn target. Keeping its native name
preserves the base model's existing stop behavior while its first token remains
distinct from all A-through-P action tokens. In every training row, the
same alias appears once in the reasoning text and once as the XML `action`
argument. The supplied tool schema also uses the 16 aliases, so the target is
schema-valid. The manifest preserves both mapping directions.

The Qwen3.5-9B tokenizer was checked for standalone action strings, action
names preceded by a reasoning-space, and XML action names preceded by a
newline. All three contexts produce 17 distinct branch tokens.

Build the four independent 1,000-row datasets:

```bash
/home/LXJ/anaconda3/envs/agent-lightning/bin/python \
  LlamaFactory/image_restoration_experts/scripts/build_expert_sft_dataset_first_token.py
```

Run all experts serially. Each expert uses physical GPUs 0, 1, 2, and 3 with
DDP; the next expert starts only after the previous one finishes:

```bash
bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft_first_token_4gpu.sh
```

Useful validation and control commands:

```bash
SFT_RUN_IN_FOREGROUND=1 DRY_RUN=1 \
  bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft_first_token_4gpu.sh

REBUILD_DATA=1 \
  bash LlamaFactory/image_restoration_experts/scripts/run_all_expert_sft_first_token_4gpu.sh

tail -f LlamaFactory/logs/first_token_actions_4gpu/four_experts_first_token_sft_4gpu_<timestamp>.log
```

The 4-GPU configuration uses per-device batch size 4 and gradient accumulation
2, preserving the legacy effective batch size of 32. New artifacts are kept
under independent paths:

```text
data:    image_restoration_experts/data/first_token_actions/
config:  image_restoration_experts/configs/first_token_actions_4gpu/
output:  image_restoration_experts/outputs/qwen3_5_0731/
log:     LlamaFactory/logs/first_token_actions_4gpu/
```

The GRPO tool schemas, response parsers, decision-point detector, and
model-visible action history use this model-facing vocabulary. The tool runtime
translates aliases back to canonical actions before restoration, reward, and
cache logic runs.

## Unified Four-Expert SFT

The unified variant directly concatenates the latest first-token-distinct fog,
snow, rain, and low-light datasets in that order. It contains 4,000 rows total,
with 1,000 rows from each expert, and preserves each row's original
`degradation_type`, `expert_name`, image, thinking target, action alias, and tool
schema. No row is resampled or shuffled during dataset construction.

Build the merged dataset and run one unified Qwen3.5 LoRA adapter on GPUs 0 and 1:

```bash
SFT_RUN_IN_FOREGROUND=1 DRY_RUN=1 \
  bash LlamaFactory/image_restoration_experts/scripts/run_unified_expert_sft_first_token_4gpu.sh

bash LlamaFactory/image_restoration_experts/scripts/run_unified_expert_sft_first_token_4gpu.sh
```

The launcher starts training in the background by default and writes all output
to a timestamped log under `LlamaFactory/logs`:

```text
LlamaFactory/logs/unified_expert_sft_2gpu_<timestamp>.log
LlamaFactory/logs/unified_expert_sft_2gpu.pid
```

Use `SFT_RUN_IN_FOREGROUND=1` when an attached foreground run is preferred.

The unified run uses three epochs and an effective batch size of 128 (2 GPUs ×
16 samples × 4 gradient accumulation). Its files
are isolated from the four existing expert adapters:

```text
data:     image_restoration_experts/data/unified_expert_first_token_actions_train.jsonl
manifest: image_restoration_experts/data/unified_expert_manifest.json
config:   image_restoration_experts/configs/unified_expert_4gpu/qwen35_unified_expert_lora_sft.yaml
output:   image_restoration_experts/outputs/qwen3_5_0813/format_cold_start/unified
```

The reasoning catalog contains exactly five distinct texts for each of the 16
non-stop tools, for 80 texts in total. Selection uses the dataset seed, so the
assignment is random across samples but byte-for-byte reproducible when the
same seed and source images are used.

## Legacy 2-GPU Build Data

From the Agent Lightning repository root:

```bash
/home/LXJ/anaconda3/envs/agent-lightning/bin/python \
  LlamaFactory/image_restoration_experts/scripts/build_expert_sft_dataset.py
```

Each of the 1,000 source images per expert produces exactly one first-turn row,
for 1,000 samples per expert and 4,000 samples in total. Tool targets use a
deterministic round-robin assignment over all 16 registered restoration tools,
while one of that tool's five reasoning texts is selected by a seeded RNG.
Because 1,000 is not divisible by 16, each tool appears either 62 or 63 times
in every expert dataset; the maximum difference is one.

## Legacy 2-GPU Train All Experts

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

The 2026-07-18 adapters are archived under
`outputs/qwen3_5_0718old/format_cold_start/{fog,snow,rain,low_light}`. New
adapters are written to
`outputs/qwen3_5_0721/format_cold_start/{fog,snow,rain,low_light}`.

Each expert reports Trainer metrics to the SwanLab cloud project
`image-restoration-expert-sft`, using run names `fog_0721`, `snow_0721`,
`rain_0721`, and `low_light_0721`. Local SwanLab records are stored under
`outputs/qwen3_5_0721/swanlab/<expert>`.

Every expert is configured for exactly three training epochs.

## Included Files

- `scripts/build_expert_sft_dataset.py`: stages images and builds validated,
  thinking-enabled first-turn-only RL-aligned ShareGPT datasets.
- `scripts/build_expert_sft_dataset_first_token.py`: builds the independent
  first-token-distinct datasets and records the runtime mapping.
- `scripts/first_token_action_vocabulary.py`: defines and validates the stable
  A-through-P plus native-stop model-facing action vocabulary.
- `scripts/restoration_thinking_templates.py`: defines and validates five short
  reasoning alternatives for every non-stop restoration tool.
- `scripts/run_expert_sft.sh`: rejects stale or modified datasets, then runs
  one expert on GPU 0 and 1.
- `scripts/run_all_expert_sft.sh`: starts the validated four-expert serial
  workflow in the background and owns its single terminal log.
- `scripts/run_expert_sft_first_token_4gpu.sh`: validates and runs one new SFT
  expert on GPUs 0 through 3.
- `scripts/run_all_expert_sft_first_token_4gpu.sh`: builds and trains all four
  new experts serially in one detached 4-GPU workflow.
- `configs/qwen35_<expert>_expert_lora_sft.yaml`: independent three-epoch,
  thinking-enabled LoRA configurations.
- `configs/first_token_actions_4gpu/`: independent 4-GPU configurations for
  the first-token-distinct adapters.
- `data/dataset_info.json`: LLaMA-Factory dataset registration generated with the datasets.
- `data/manifest.json`: generated sample counts, action distributions, checksums, and alignment metadata.
