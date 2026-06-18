# Image Restoration Expert SFT Data

This project builds four independent single-step SFT datasets for the fog,
snow, rain, and low-light restoration experts.

Each expert receives 1000 images from its own degradation category:

- 500 images are labeled with the highest-scoring action after executing all
  16 restoration tools and applying `config/iqa_reward_v1.json`.
- 500 images are assigned uniformly across all 16 restoration actions to
  provide Hermes format and tool coverage.

The model input contains only the current image, expert identity, full shared
tool schema, and the instruction to select one action. The assistant target is
one raw Hermes `restore_image` call. IQA scores and feedback are not exposed to
the model.

Run from the Agent_Lightning repository root:

```bash
/home/LXJ/anaconda3/envs/verl/bin/python \
  LlamaFactory/image_restoration_experts/scripts/build_expert_sft_dataset.py \
  --gpus 0,1 \
  --restoration-workers-per-action 2 \
  --iqa-workers 6 \
  --resume
```

`--restoration-workers-per-action 2` splits each unfinished restoration action
between GPU 0 and GPU 1. Existing unsharded and sharded progress files are
combined automatically when resuming.

`--iqa-workers 6` launches six IQA workers round-robin across the listed GPUs.
All workers read the global completed-image set, so changing worker count while
resuming does not duplicate scores from an earlier sharding layout.

Restoration and IQA progress is persisted below `cache/iqa_best/`; rerunning
the command resumes completed actions and images. Final ShareGPT JSONL files,
`dataset_info.json`, staged image links, and a label manifest are written to
`data/`.

## Expert SFT

Each expert has an independent Qwen3.5 LoRA configuration in `configs/`.
Train one expert on GPU 0 and GPU 1 with:

```bash
bash image_restoration_experts/scripts/run_expert_sft.sh fog
```

Run all four experts sequentially, stopping on the first failure, with:

```bash
bash image_restoration_experts/scripts/run_all_expert_sft.sh
```

Evaluate all four LoRA adapters on their SFT datasets with two-GPU batched
inference:

```bash
bash image_restoration_experts/scripts/run_expert_evaluation.sh
```

Reports are written to `outputs/qwen3_5/evaluation/format_cold_start/<expert>/`, with aggregate
metrics in `outputs/qwen3_5/evaluation/format_cold_start/metrics_all_experts.json`.

## GRPO-format cold start

The original datasets only teach the first restoration turn. Build the
synthetic format-cold-start datasets with:

```bash
/home/LXJ/anaconda3/envs/agent-lightning/bin/python \
  image_restoration_experts/scripts/build_expert_format_sft_dataset.py
```

No restoration or IQA model is executed. Each source image produces four
independent samples matching the formal GRPO transition prompts:

- initial action selection;
- continuation after a synthetic improvement;
- tool switching after a synthetic decline;
- `stop` after a synthetic sufficient quality gain.

The targets are strict raw Hermes `restore_image` calls. Non-stop actions are
balanced across all 16 tools. Qwen3.5 expert adapters are created from the base
model and written below `outputs/qwen3_5/format_cold_start/<expert>/`.

Train all four experts sequentially:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash image_restoration_experts/scripts/run_all_expert_format_sft.sh
```

Evaluate the resulting adapters on the synthetic state-format dataset:

```bash
/home/LXJ/anaconda3/envs/llamafactory/bin/python \
  image_restoration_experts/scripts/evaluate_expert_agents.py \
  --dataset-profile format \
  --adapter-root image_restoration_experts/outputs/qwen3_5/format_cold_start \
  --output-dir image_restoration_experts/outputs/qwen3_5/evaluation/format_cold_start \
  --gpus 0,1
```
