# Degradation Diagnosis SFT Dataset

This builder creates a class-balanced multimodal SFT dataset for the degradation
diagnosis agent. It maps the source directories as follows:

```text
image_restoration_diagnosis/
├── configs/   # Full and smoke-test LLaMA-Factory configurations
├── data/      # Generated JSONL files and staged image links
├── outputs/   # LoRA adapters, checkpoints, and training logs
└── scripts/   # Dataset construction scripts
```

`data/` and `outputs/` are generated locally and ignored by Git. All dataset
construction and SFT output paths remain inside this project directory.

| Source directory | Training label |
|---|---|
| `fog_series/fog` | `fog` |
| `rain_series/rain` | `rain` |
| `snow_series/snow` | `snow` |
| `night_series/night` | `low_light` |

Run from the project directory:

```bash
cd /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory/image_restoration_diagnosis
python scripts/build_diagnosis_sft_dataset.py \
  --train-root "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/train/images" \
  --test-root "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/test/images" \
  --output-dir /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory/image_restoration_diagnosis/data \
  --train-samples-per-class 1000 \
  --test-samples-per-class 200 \
  --image-mode symlink
```

The output contains:

- `diagnosis_train.jsonl`: 4000 training samples, 1000 per class.
- `diagnosis_test.jsonl`: 800 held-out test samples, 200 per class.
- `dataset_info.json`: LLaMA-Factory dataset registration.
- `manifest.json`: sampling parameters, class counts, and dataset checksum.
- `images/<split>/<class>/`: links, hard links, or copies of selected images.

Each degradation class defines 20 unique `visual_evidence` annotations. The
builder assigns them by sample index after deterministic sampling. Every
training annotation is used exactly 50 times per class, and every test
annotation is used exactly 10 times per class.

Use these dataset settings in LLaMA-Factory:

```yaml
dataset: image_restoration_diagnosis_train
dataset_dir: /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory/image_restoration_diagnosis/data
media_dir: /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory/image_restoration_diagnosis/data
template: qwen3_5_nothink
enable_thinking: false
resize_vocab: false
```

Do not set `tool_format`. The assistant response is deliberately stored as raw
Hermes text:

```text
<tool_call>
{"name":"diagnose_degradation","arguments":{"primary_type":"fog","visual_evidence":[...]}}
</tool_call>
```

Using the ShareGPT `function_call` role would cause LLaMA-Factory to convert the
target to model-native function-call formatting.
Setting `enable_thinking: false` prevents the reasoning template from injecting
an empty `<think>...</think>` block into the supervised target. The checkpoint
already contains all template stop-token IDs inside its padded 151552-row
embedding table, and tokenization does not increase the tokenizer length, so
`resize_vocab` remains disabled. Enabling it would unnecessarily save the full
embedding and language-head matrices alongside the LoRA adapter.

The source dataset provides folder-level class labels but no per-image evidence
annotations. By default, `visual_evidence` therefore uses deterministic
class-level templates as weak supervision. The 20 variants reduce direct
single-phrase memorization, but they should not be treated as human-verified
image-specific evidence.

The full and smoke-test configurations are:

```text
configs/qwen35_diagnosis_lora_sft.yaml
configs/qwen35_diagnosis_lora_smoke.yaml
```

The full configuration is intended for two GPUs. With
`per_device_train_batch_size=2`, `gradient_accumulation_steps=2`, and two
processes, its global effective batch size is eight. SwanLab cloud logging is
enabled under project `image-restoration-multi-agent` with run name
`qwen3.5-diagnosis-lora-sft`.

Run the full training job on GPU 0 and GPU 1:

```bash
cd /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory
bash image_restoration_diagnosis/scripts/run_full_sft.sh
```

The script displays stdout and stderr in the terminal while also writing them
to `log/diagnosis_sft_<timestamp>.log`. The symlink
`log/diagnosis_sft_latest.log` always points to the latest run. It preserves
the actual training process exit code, so command failures remain visible to
shell scripts and job schedulers.

Verify logging and all preflight checks without loading the model:

```bash
DRY_RUN=1 bash image_restoration_diagnosis/scripts/run_full_sft.sh
cat log/diagnosis_sft_latest.log
```

Run the smoke test on GPU 0:

```bash
cd /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory
CUDA_VISIBLE_DEVICES=0 \
  /home/LXJ/anaconda3/envs/llamafactory/bin/llamafactory-cli train \
  /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory/image_restoration_diagnosis/configs/qwen35_diagnosis_lora_smoke.yaml
```

## Evaluate the diagnosis agent

The evaluation script loads one complete base-model and LoRA replica on each
selected GPU, splits the test set evenly across the workers, and performs real
batched multimodal generation. The initial batch size is specified per GPU. If
a worker runs out of memory, it halves its own batch size and retries the same
samples automatically.

By default, the script reads
`/home/LXJ/Python_Projects/JarvisIR/data/inference/images/cleanbench_PaperTest`,
maps the `night` directory to `low_light`, and samples 100 images from each of
the four classes using seed 42. Run this 400-image evaluation on GPU 0 and GPU
1 with:

```bash
cd /home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory
CUDA_VISIBLE_DEVICES=0,1 \
  /home/LXJ/anaconda3/envs/llamafactory/bin/python \
  image_restoration_diagnosis/scripts/evaluate_diagnosis_agent.py \
  --gpus 0,1 \
  --batch-size 32 \
  --samples-per-class 100 \
  --sample-seed 42
```

The script reproduces LLaMA-Factory's `enable_thinking: false` inference
prefix, uses deterministic greedy decoding, and validates both the
`diagnose_degradation` argument schema and its Hermes envelope. Use `--resume`
to continue an interrupted run from the per-GPU prediction files.

Reports are written to
`outputs/qwen3_5/evaluation/cleanbench_papertest_100_per_class/`:

- `predictions.jsonl`: ordered predictions, raw generations, parse states, and
  per-sample labels.
- `sampled_images.jsonl`: the deterministic sampling manifest.
- `misclassified_predictions.jsonl`: only the incorrectly classified samples.
- `metrics.json`: aggregate metrics, per-class metrics, timing, memory, and
  worker statistics.
- `classification_report.csv`: precision, recall, and F1 for every class.
- `confusion_matrix.csv` and `confusion_matrix.png`: classification errors.
- `summary.md`: concise run summary.

The CleanBench PaperTest evaluation performed on 2026-06-13 produced:

- 400 test samples, with 100 samples from each class.
- 98.25% accuracy and 98.25% Macro-F1.
- 100% schema-valid, Hermes-valid, and pure-Hermes outputs.
- Fog, rain, and low-light recall of 100%; snow recall of 93%.
- All seven errors were snow images classified as rain.
- 6.10 samples/second end-to-end throughput in 65.58 seconds.
- Batch size 32 per GPU without OOM fallback; peak allocated memory was
  approximately 25.48 GiB per GPU.

## Earlier smoke-test result

Before this directory relocation, the two-step smoke test completed
successfully on 2026-06-13:

- Loaded `Qwen3.5-9B` and all multimodal processors.
- Tokenized eight image-text training samples.
- Trained 47,595,520 LoRA parameters, 0.4603% of the model.
- Completed two optimizer steps with losses 2.8258 and 2.4476.
- Reported an average training loss of 2.6367 in 8.72 seconds.
- Saved the final adapter and two resumable checkpoints.
- Reloaded the final adapter and generated 16 new tokens from one held-out test
  image, confirming that the saved LoRA and multimodal inference path work.

The relocated smoke-test configuration now writes new artifacts to
`outputs/smoke`.

The smoke-test generation did not yet follow the required Hermes format. This
is expected after only two optimization steps over eight samples. The smoke
test verifies pipeline feasibility, not classification accuracy or output
format convergence. The 800 test samples were not loaded during training and
remain reserved for later evaluation.
