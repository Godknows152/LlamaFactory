#!/usr/bin/env python3
"""Evaluate the degradation-diagnosis LoRA on the held-out image test set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("fog", "snow", "rain", "low_light")
TOOL_NAME = "diagnose_degradation"
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
PURE_HERMES_PATTERN = re.compile(r"^\s*<tool_call>\s*.*?\s*</tool_call>\s*$", re.DOTALL)
THINK_HERMES_PATTERN = re.compile(
    r"^\s*<think>(.*?)</think>\s*<tool_call>\s*.*?\s*</tool_call>\s*$",
    re.DOTALL,
)
UNCLOSED_THINK_HERMES_PATTERN = re.compile(
    r"^\s*<think>\s*<tool_call>\s*.*?\s*</tool_call>\s*$",
    re.DOTALL,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
DEFAULT_BASE_MODEL = Path("/home/LXJ/Python_Projects/Models/Qwen3.5-9B")
DEFAULT_ADAPTER = PROJECT_DIR / "outputs" / "qwen3_5" / "full"
DEFAULT_TEST_FILE = PROJECT_DIR / "data" / "diagnosis_test.jsonl"
DEFAULT_IMAGE_ROOT = Path(
    "/home/LXJ/Python_Projects/JarvisIR/data/inference/images/cleanbench_PaperTest"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "qwen3_5" / "evaluation" / "cleanbench_papertest_100_per_class"
CLASS_DIRECTORIES = {
    "fog": "fog",
    "snow": "snow",
    "rain": "rain",
    "low_light": "night",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Class-directory image root containing fog, snow, rain, and night subdirectories.",
    )
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU IDs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Initial batch size per GPU.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=262_144)
    parser.add_argument("--image-min-pixels", type=int, default=1_024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prefill-empty-think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally prepend an empty thinking block for compatible reasoning templates.",
    )
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_rows(
    path: Path,
    max_samples: int | None = None,
    image_root: Path | None = None,
    samples_per_class: int = 100,
    sample_seed: int = 42,
) -> list[dict[str, Any]]:
    source_rows = load_jsonl_rows(path)
    if image_root is None:
        rows = source_rows
    else:
        if not source_rows:
            raise ValueError(f"Prompt template dataset is empty: {path}")
        if samples_per_class < 1:
            raise ValueError("--samples-per-class must be positive.")

        template = source_rows[0]
        user_message = next(message for message in template["messages"] if message["role"] == "user")
        random_generator = random.Random(sample_seed)
        rows = []
        for label, directory_name in CLASS_DIRECTORIES.items():
            class_directory = image_root / directory_name
            candidates = sorted(
                candidate.resolve()
                for candidate in class_directory.iterdir()
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
            if len(candidates) < samples_per_class:
                raise ValueError(
                    f"Class {directory_name!r} contains {len(candidates)} images, "
                    f"but {samples_per_class} were requested."
                )
            selected = random_generator.sample(candidates, samples_per_class)
            for image_path in selected:
                rows.append(
                    {
                        "messages": [user_message],
                        "system": template["system"],
                        "images": [str(image_path)],
                        "sample_id": f"cleanbench-{label}-{image_path.stem}",
                        "split": "test",
                        "degradation_type": label,
                        "source_image": str(image_path),
                        "annotation_provenance": "cleanbench_directory_label",
                    }
                )
        random_generator.shuffle(rows)

    if max_samples is not None:
        rows = rows[:max_samples]

    for index, row in enumerate(rows):
        row["dataset_index"] = index
    return rows


def load_evaluation_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return load_rows(
        args.test_file,
        max_samples=args.max_samples,
        image_root=args.image_root,
        samples_per_class=args.samples_per_class,
        sample_seed=args.sample_seed,
    )


def preprocess_image(path: Path, max_pixels: int, min_pixels: int):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        pixels = width * height
        if pixels > max_pixels:
            factor = math.sqrt(max_pixels / pixels)
            image = image.resize((max(1, int(width * factor)), max(1, int(height * factor))))
        elif pixels < min_pixels:
            factor = math.sqrt(min_pixels / pixels)
            image = image.resize((max(1, int(width * factor)), max(1, int(height * factor))))
        return image.copy()


def build_conversation(row: dict[str, Any], image) -> list[dict[str, Any]]:
    user_text = row["messages"][0]["content"]
    user_text = re.sub(r"^\s*<image>\s*", "", user_text, count=1)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": row["system"]}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def parse_prediction(text: str) -> dict[str, Any]:
    matches = TOOL_CALL_PATTERN.findall(text)
    result: dict[str, Any] = {
        "parse_status": "valid",
        "schema_valid": False,
        "hermes_valid": False,
        "pure_hermes": False,
        "think_hermes": False,
        "empty_think_hermes": False,
        "unclosed_think_hermes": False,
        "predicted_type": None,
        "visual_evidence": None,
    }
    if not matches:
        result["parse_status"] = "missing_tool_call"
        return result
    if len(matches) != 1:
        result["parse_status"] = "multiple_tool_calls"
        return result

    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        result["parse_status"] = "invalid_json"
        return result

    if not isinstance(payload, dict) or payload.get("name") != TOOL_NAME:
        result["parse_status"] = "invalid_tool_name"
        return result
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != {"primary_type", "visual_evidence"}:
        result["parse_status"] = "invalid_arguments"
        return result

    predicted_type = arguments.get("primary_type")
    evidence = arguments.get("visual_evidence")
    if predicted_type not in LABELS:
        result["parse_status"] = "invalid_primary_type"
        return result
    if not isinstance(evidence, list) or not evidence or not all(isinstance(item, str) and item.strip() for item in evidence):
        result["parse_status"] = "invalid_visual_evidence"
        return result

    pure_match = PURE_HERMES_PATTERN.fullmatch(text)
    think_match = THINK_HERMES_PATTERN.fullmatch(text)
    unclosed_think_match = UNCLOSED_THINK_HERMES_PATTERN.fullmatch(text)
    result.update(
        {
            "schema_valid": True,
            "hermes_valid": pure_match is not None or think_match is not None,
            "pure_hermes": pure_match is not None,
            "think_hermes": think_match is not None,
            "empty_think_hermes": think_match is not None and not think_match.group(1).strip(),
            "unclosed_think_hermes": unclosed_think_match is not None,
            "predicted_type": predicted_type,
            "visual_evidence": evidence,
        }
    )
    if not result["hermes_valid"]:
        result["parse_status"] = "valid_schema_with_extra_text"
    return result


def move_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def worker_main(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    torch.set_grad_enabled(False)

    rows = load_evaluation_rows(args)
    shard = rows[args.worker_rank :: args.world_size]
    output_file = args.output_dir / f"predictions_rank{args.worker_rank}.jsonl"
    stats_file = args.output_dir / f"worker_stats_rank{args.worker_rank}.json"
    completed_ids: set[str] = set()
    if args.resume and output_file.exists():
        with output_file.open(encoding="utf-8") as file:
            completed_ids = {json.loads(line)["sample_id"] for line in file if line.strip()}
    pending = [row for row in shard if row["sample_id"] not in completed_ids]

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    chat_template_path = args.adapter / "chat_template.jinja"
    if chat_template_path.exists():
        processor.chat_template = chat_template_path.read_text(encoding="utf-8")
    if args.prefill_empty_think:
        generation_marker = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- if enable_thinking is defined and enable_thinking is false %}"
        if generation_marker in processor.chat_template:
            generation_prefix = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {{- '<think>\\n\\n</think>\\n\\n' }}\n{%- if enable_thinking is defined and enable_thinking is false %}"
            processor.chat_template = processor.chat_template.replace(generation_marker, generation_prefix)
        else:
            # Fallback: inject enable_thinking=False at template level
            pass
    # Always pass enable_thinking=False to suppress the open <think> tag
    if "enable_thinking" not in processor.chat_template:
        # Template doesn't have the enable_thinking guard; modify the generation prompt block
        gen_block = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {{- '<think>\\n' }}\n{%- endif %}"
        gen_block_fixed = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {{- '<think>\\n\\n</think>\\n\\n' }}\n{%- endif %}"
        if gen_block in processor.chat_template:
            processor.chat_template = processor.chat_template.replace(gen_block, gen_block_fixed)
    chat_template_kwargs = {"enable_thinking": False}

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda:0")
    model.eval()
    model.config.use_cache = True
    load_seconds = time.perf_counter() - load_start

    current_batch_size = max(1, args.batch_size)
    oom_reductions = 0
    processed = 0
    inference_seconds = 0.0
    mode = "a" if args.resume else "w"
    with output_file.open(mode, encoding="utf-8") as output:
        cursor = 0
        while cursor < len(pending):
            batch_rows = pending[cursor : cursor + current_batch_size]
            images = []
            inputs = None
            try:
                for row in batch_rows:
                    image_path = Path(row["images"][0])
                    if not image_path.is_absolute():
                        image_path = args.test_file.parent / image_path
                    images.append(preprocess_image(image_path, args.image_max_pixels, args.image_min_pixels))
                conversations = [build_conversation(row, image) for row, image in zip(batch_rows, images)]
                inputs = processor.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    **chat_template_kwargs,
                )
                inputs = move_to_device(inputs, "cuda:0")
                prompt_length = inputs["input_ids"].shape[1]
                batch_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=model.generation_config.eos_token_id,
                    )
                torch.cuda.synchronize()
                batch_seconds = time.perf_counter() - batch_start
                inference_seconds += batch_seconds
                generated_ids = output_ids[:, prompt_length:]
                texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            except torch.OutOfMemoryError:
                if inputs is not None:
                    del inputs
                gc.collect()
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                oom_reductions += 1
                print(
                    f"[rank {args.worker_rank}] OOM: reducing batch size to {current_batch_size}",
                    flush=True,
                )
                continue
            finally:
                for image in images:
                    image.close()

            per_sample_seconds = batch_seconds / len(batch_rows)
            for row, text in zip(batch_rows, texts):
                parsed = parse_prediction(text)
                record = {
                    "dataset_index": row["dataset_index"],
                    "sample_id": row["sample_id"],
                    "image": row["images"][0],
                    "ground_truth": row["degradation_type"],
                    "raw_output": text,
                    "seconds": per_sample_seconds,
                    **parsed,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            cursor += len(batch_rows)
            processed += len(batch_rows)
            print(
                f"[rank {args.worker_rank}] {processed}/{len(pending)} "
                f"batch={len(batch_rows)} active_batch_size={current_batch_size}",
                flush=True,
            )
            del inputs, output_ids, generated_ids
            gc.collect()

    stats = {
        "rank": args.worker_rank,
        "samples": len(shard),
        "new_samples": len(pending),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "samples_per_second": len(pending) / inference_seconds if inference_seconds else 0.0,
        "initial_batch_size": args.batch_size,
        "final_batch_size": current_batch_size,
        "oom_reductions": oom_reductions,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(records: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    confusion = {truth: dict.fromkeys((*LABELS, "invalid"), 0) for truth in LABELS}
    parse_statuses = Counter()
    prediction_counts = Counter()
    schema_valid = hermes_valid = pure_hermes = empty_think = unclosed_think = correct = 0
    evidence_items = 0

    for record in records:
        truth = record["ground_truth"]
        prediction = record["predicted_type"] if record["schema_valid"] else "invalid"
        confusion[truth][prediction] += 1
        parse_statuses[record["parse_status"]] += 1
        prediction_counts[prediction] += 1
        schema_valid += int(record["schema_valid"])
        hermes_valid += int(record["hermes_valid"])
        pure_hermes += int(record["pure_hermes"])
        empty_think += int(record["empty_think_hermes"])
        unclosed_think += int(record["unclosed_think_hermes"])
        correct += int(prediction == truth)
        evidence_items += len(record["visual_evidence"] or [])

    per_class = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label][prediction] for prediction in (*LABELS, "invalid") if prediction != label)
        false_positive = sum(confusion[truth][label] for truth in LABELS if truth != label)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        support = sum(confusion[label].values())
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = len(records)
    return {
        "total_samples": total,
        "accuracy": safe_divide(correct, total),
        "macro_precision": sum(item["precision"] for item in per_class.values()) / len(LABELS),
        "macro_recall": sum(item["recall"] for item in per_class.values()) / len(LABELS),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABELS),
        "schema_valid_rate": safe_divide(schema_valid, total),
        "hermes_valid_rate": safe_divide(hermes_valid, total),
        "pure_hermes_rate": safe_divide(pure_hermes, total),
        "empty_think_hermes_rate": safe_divide(empty_think, total),
        "unclosed_think_hermes_rate": safe_divide(unclosed_think, total),
        "accuracy_on_schema_valid": safe_divide(correct, schema_valid),
        "average_evidence_items": safe_divide(evidence_items, schema_valid),
        "wall_seconds": wall_seconds,
        "samples_per_second": safe_divide(total, wall_seconds),
        "parse_status_counts": dict(sorted(parse_statuses.items())),
        "prediction_counts": {key: prediction_counts[key] for key in (*LABELS, "invalid")},
        "per_class": per_class,
        "confusion_matrix": confusion,
        "worker_stats": [],
    }


def write_reports(records: list[dict[str, Any]], metrics: dict[str, Any], output_dir: Path) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (output_dir / "sampled_images.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            sample = {
                "dataset_index": record["dataset_index"],
                "sample_id": record["sample_id"],
                "image": record["image"],
                "ground_truth": record["ground_truth"],
            }
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with (output_dir / "misclassified_predictions.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            if record["predicted_type"] != record["ground_truth"]:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (output_dir / "classification_report.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for label, values in metrics["per_class"].items():
            writer.writerow({"class": label, **values})

    columns = [*LABELS, "invalid"]
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ground_truth\\prediction", *columns])
        for truth in LABELS:
            writer.writerow([truth, *(metrics["confusion_matrix"][truth][prediction] for prediction in columns)])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[metrics["confusion_matrix"][truth][prediction] for prediction in columns] for truth in LABELS]
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Ground-truth label")
    axis.set_title("Degradation diagnosis confusion matrix")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = f"""# Diagnosis Agent Evaluation

- Samples: {metrics['total_samples']}
- Accuracy: {metrics['accuracy']:.4%}
- Macro-F1: {metrics['macro_f1']:.4%}
- Schema-valid rate: {metrics['schema_valid_rate']:.4%}
- Hermes-valid rate: {metrics['hermes_valid_rate']:.4%}
- Pure Hermes rate: {metrics['pure_hermes_rate']:.4%}
- Empty-think Hermes rate: {metrics['empty_think_hermes_rate']:.4%}
- Unclosed-think tool-call rate: {metrics['unclosed_think_hermes_rate']:.4%}
- Wall time: {metrics['wall_seconds']:.2f} seconds
- Throughput: {metrics['samples_per_second']:.2f} samples/second
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def coordinator_main(args: argparse.Namespace) -> int:
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    if args.output_dir.exists() and not args.resume:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_evaluation_rows(args)
    start = time.perf_counter()
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(args.adapter),
            "--test-file",
            str(args.test_file),
            "--image-root",
            str(args.image_root),
            "--samples-per-class",
            str(args.samples_per_class),
            "--sample-seed",
            str(args.sample_seed),
            "--output-dir",
            str(args.output_dir),
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--image-max-pixels",
            str(args.image_max_pixels),
            "--image-min-pixels",
            str(args.image_min_pixels),
            "--worker-rank",
            str(rank),
            "--world-size",
            str(len(gpu_ids)),
        ]
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        if args.resume:
            command.append("--resume")
        if not args.prefill_empty_think:
            command.append("--no-prefill-empty-think")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        print(f"Launching rank {rank} on physical GPU {gpu_id}", flush=True)
        processes.append(subprocess.Popen(command, env=environment))

    return_codes = [process.wait() for process in processes]
    if any(return_code != 0 for return_code in return_codes):
        raise RuntimeError(f"Evaluation worker failure: return codes={return_codes}")

    records = []
    worker_stats = []
    for rank in range(len(gpu_ids)):
        prediction_file = args.output_dir / f"predictions_rank{rank}.jsonl"
        with prediction_file.open(encoding="utf-8") as file:
            records.extend(json.loads(line) for line in file if line.strip())
        worker_stats.append(json.loads((args.output_dir / f"worker_stats_rank{rank}.json").read_text()))
    records.sort(key=lambda item: item["dataset_index"])
    if len(records) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} predictions, found {len(records)}.")

    metrics = compute_metrics(records, time.perf_counter() - start)
    metrics["worker_stats"] = worker_stats
    metrics["base_model"] = str(args.base_model)
    metrics["adapter"] = str(args.adapter)
    metrics["test_file"] = str(args.test_file)
    metrics["image_root"] = str(args.image_root)
    metrics["samples_per_class"] = args.samples_per_class
    metrics["sample_seed"] = args.sample_seed
    metrics["initial_batch_size_per_gpu"] = args.batch_size
    metrics["gpu_ids"] = gpu_ids
    metrics["prefill_empty_think"] = args.prefill_empty_think
    write_reports(records, metrics, args.output_dir)
    print(json.dumps({key: metrics[key] for key in (
        "total_samples",
        "accuracy",
        "macro_f1",
        "schema_valid_rate",
        "hermes_valid_rate",
        "wall_seconds",
        "samples_per_second",
    )}, indent=2))
    print(f"Reports saved to: {args.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_rank is not None:
        if args.world_size is None:
            raise ValueError("--world-size is required for workers.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return worker_main(args)
    return coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
