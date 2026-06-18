#!/usr/bin/env python3
"""Evaluate the four single-step restoration-expert LoRA adapters."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
DEFAULT_BASE_MODEL = Path("/home/LXJ/Python_Projects/Models/Qwen3.5-9B")
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_ADAPTER_ROOT = PROJECT_DIR / "outputs/qwen3_5/format_cold_start"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs/qwen3_5/evaluation/format_cold_start"
DEFAULT_TOOLS_CONFIG = SCRIPT_PATH.parents[3] / "examples/image_restoration_multi_agent/config/tools.yaml"
EXPERTS = ("fog", "snow", "rain", "low_light")
TOOL_NAME = "restore_image"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tools-config", type=Path, default=DEFAULT_TOOLS_CONFIG)
    parser.add_argument("--experts", nargs="+", choices=EXPERTS, default=list(EXPERTS))
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU IDs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Initial batch size per GPU.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=262_144)
    parser.add_argument("--image-min-pixels", type=int, default=1_024)
    parser.add_argument("--max-samples-per-expert", type=int, default=None)
    parser.add_argument(
        "--dataset-profile",
        choices=("policy", "format"),
        default="policy",
        help="Evaluate the original single-step policy data or synthetic GRPO-format data.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prefill-empty-think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally prepend an empty thinking block for compatible reasoning templates.",
    )
    parser.add_argument("--expert", choices=EXPERTS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_actions(path: Path) -> tuple[str, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(tool["name"] for tool in payload["tools"])


def load_rows(args: argparse.Namespace, expert: str) -> list[dict[str, Any]]:
    suffix = "_expert_format_train.jsonl" if args.dataset_profile == "format" else "_expert_train.jsonl"
    path = args.data_dir / f"{expert}{suffix}"
    rows = read_jsonl(path)
    if args.max_samples_per_expert is not None:
        rows = rows[: args.max_samples_per_expert]
    for index, row in enumerate(rows):
        row["dataset_index"] = index
    return rows


def preprocess_image(path: Path, max_pixels: int, min_pixels: int):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        pixels = image.width * image.height
        if pixels > max_pixels:
            factor = math.sqrt(max_pixels / pixels)
            image = image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))))
        elif pixels < min_pixels:
            factor = math.sqrt(min_pixels / pixels)
            image = image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))))
        return image.copy()


def build_conversation(row: dict[str, Any], image) -> list[dict[str, Any]]:
    user_message = next(message for message in row["messages"] if message["role"] == "user")
    user_text = re.sub(r"^\s*<image>\s*", "", user_message["content"], count=1)
    return [
        {"role": "system", "content": [{"type": "text", "text": row["system"]}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def parse_prediction(text: str, actions: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parse_status": "valid",
        "schema_valid": False,
        "hermes_valid": False,
        "pure_hermes": False,
        "think_hermes": False,
        "empty_think_hermes": False,
        "unclosed_think_hermes": False,
        "predicted_action": None,
    }
    matches = TOOL_CALL_PATTERN.findall(text)
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
    if not isinstance(arguments, dict) or set(arguments) != {"action"}:
        result["parse_status"] = "invalid_arguments"
        return result
    action = arguments.get("action")
    if action not in actions:
        result["parse_status"] = "invalid_action"
        return result

    pure_match = PURE_HERMES_PATTERN.fullmatch(text)
    think_match = THINK_HERMES_PATTERN.fullmatch(text)
    unclosed_match = UNCLOSED_THINK_HERMES_PATTERN.fullmatch(text)
    result.update(
        {
            "schema_valid": True,
            "hermes_valid": pure_match is not None or think_match is not None,
            "pure_hermes": pure_match is not None,
            "think_hermes": think_match is not None,
            "empty_think_hermes": think_match is not None and not think_match.group(1).strip(),
            "unclosed_think_hermes": unclosed_match is not None,
            "predicted_action": action,
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

    if args.expert is None or args.worker_rank is None or args.world_size is None:
        raise ValueError("Worker mode requires --expert, --worker-rank, and --world-size.")

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    torch.set_grad_enabled(False)

    rows = load_rows(args, args.expert)
    shard = rows[args.worker_rank :: args.world_size]
    expert_output = args.output_dir / args.expert
    expert_output.mkdir(parents=True, exist_ok=True)
    output_file = expert_output / f"predictions_rank{args.worker_rank}.jsonl"
    stats_file = expert_output / f"worker_stats_rank{args.worker_rank}.json"
    completed_ids = set()
    if args.resume and output_file.exists():
        completed_ids = {row["sample_id"] for row in read_jsonl(output_file)}
    pending = [row for row in shard if row["sample_id"] not in completed_ids]

    if not pending:
        stats_file.write_text(
            json.dumps({"rank": args.worker_rank, "samples": len(shard), "new_samples": 0}, indent=2),
            encoding="utf-8",
        )
        return 0

    adapter = args.adapter_root / args.expert
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    chat_template_path = adapter / "chat_template.jinja"
    if chat_template_path.exists():
        processor.chat_template = chat_template_path.read_text(encoding="utf-8")
    if args.prefill_empty_think:
        marker = "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        prefix = "{% if add_generation_prompt %}<|assistant|>\n<think>\n\n</think>\n\n{% endif %}"
        if marker not in processor.chat_template:
            raise ValueError("Could not add the empty-think prefix to the active chat template.")
        processor.chat_template = processor.chat_template.replace(marker, prefix)

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda:0")
    model.eval()
    model.config.use_cache = True
    load_seconds = time.perf_counter() - load_start

    actions = set(load_actions(args.tools_config))
    current_batch_size = max(1, args.batch_size)
    oom_reductions = 0
    inference_seconds = 0.0
    processed = 0
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
                        image_path = args.data_dir / image_path
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
                )
                inputs = move_to_device(inputs, "cuda:0")
                prompt_length = inputs["input_ids"].shape[1]
                started = time.perf_counter()
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
                batch_seconds = time.perf_counter() - started
                inference_seconds += batch_seconds
                texts = processor.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
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
                    f"[{args.expert} rank {args.worker_rank}] OOM: batch -> {current_batch_size}",
                    flush=True,
                )
                continue
            finally:
                for image in images:
                    image.close()

            for row, text in zip(batch_rows, texts):
                parsed = parse_prediction(text, actions)
                output.write(
                    json.dumps(
                        {
                            "dataset_index": row["dataset_index"],
                            "sample_id": row["sample_id"],
                            "expert": args.expert,
                            "image": row["images"][0],
                            "label_source": row["metadata"]["label_source"],
                            "state_variant": row["metadata"].get("state_variant"),
                            "target_action": row["metadata"]["selected_action"],
                            "raw_output": text,
                            "seconds": batch_seconds / len(batch_rows),
                            **parsed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            output.flush()
            cursor += len(batch_rows)
            processed += len(batch_rows)
            print(
                f"[{args.expert} rank {args.worker_rank}] {processed}/{len(pending)} "
                f"batch={len(batch_rows)} active_batch_size={current_batch_size}",
                flush=True,
            )
            del inputs, output_ids
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


def compute_metrics(records: list[dict[str, Any]], actions: tuple[str, ...]) -> dict[str, Any]:
    total = len(records)
    correct = sum(record["predicted_action"] == record["target_action"] for record in records)
    schema_valid = sum(record["schema_valid"] for record in records)
    hermes_valid = sum(record["hermes_valid"] for record in records)
    pure_hermes = sum(record["pure_hermes"] for record in records)
    parse_statuses = Counter(record["parse_status"] for record in records)
    predicted = Counter(record["predicted_action"] or "invalid" for record in records)
    targets = Counter(record["target_action"] for record in records)
    source_metrics = {}
    sources = sorted({record["label_source"] for record in records})
    for source in sources:
        source_records = [record for record in records if record["label_source"] == source]
        source_metrics[source] = {
            "samples": len(source_records),
            "target_action_accuracy": safe_divide(
                sum(record["predicted_action"] == record["target_action"] for record in source_records),
                len(source_records),
            ),
            "schema_valid_rate": safe_divide(sum(record["schema_valid"] for record in source_records), len(source_records)),
            "hermes_valid_rate": safe_divide(sum(record["hermes_valid"] for record in source_records), len(source_records)),
        }
    confusion = {
        target: dict.fromkeys((*actions, "invalid"), 0)
        for target in actions
    }
    for record in records:
        confusion[record["target_action"]][record["predicted_action"] or "invalid"] += 1
    return {
        "total_samples": total,
        "target_action_accuracy": safe_divide(correct, total),
        "accuracy_on_schema_valid": safe_divide(correct, schema_valid),
        "schema_valid_rate": safe_divide(schema_valid, total),
        "hermes_valid_rate": safe_divide(hermes_valid, total),
        "pure_hermes_rate": safe_divide(pure_hermes, total),
        "parse_status_counts": dict(sorted(parse_statuses.items())),
        "target_action_counts": {action: targets[action] for action in actions},
        "predicted_action_counts": {action: predicted[action] for action in (*actions, "invalid")},
        "source_metrics": source_metrics,
        "confusion_matrix": confusion,
    }


def write_reports(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    actions: tuple[str, ...],
    output_dir: Path,
) -> None:
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "incorrect_predictions.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            if record["predicted_action"] != record["target_action"]:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    columns = [*actions, "invalid"]
    with (output_dir / "action_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target\\prediction", *columns])
        for target in actions:
            writer.writerow([target, *(metrics["confusion_matrix"][target][prediction] for prediction in columns)])
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# {records[0]['expert']} Expert Evaluation",
                "",
                f"- Samples: {metrics['total_samples']}",
                f"- Target-action accuracy: {metrics['target_action_accuracy']:.4%}",
                f"- Schema-valid rate: {metrics['schema_valid_rate']:.4%}",
                f"- Hermes-valid rate: {metrics['hermes_valid_rate']:.4%}",
                f"- Pure-Hermes rate: {metrics['pure_hermes_rate']:.4%}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_one_expert(args: argparse.Namespace, expert: str, gpu_ids: list[str]) -> dict[str, Any]:
    expert_output = args.output_dir / expert
    if expert_output.exists() and not args.resume:
        shutil.rmtree(expert_output)
    expert_output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args, expert)
    adapter = args.adapter_root / expert
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter}")

    started = time.perf_counter()
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-model",
            str(args.base_model),
            "--data-dir",
            str(args.data_dir),
            "--adapter-root",
            str(args.adapter_root),
            "--output-dir",
            str(args.output_dir),
            "--tools-config",
            str(args.tools_config),
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dataset-profile",
            args.dataset_profile,
            "--image-max-pixels",
            str(args.image_max_pixels),
            "--image-min-pixels",
            str(args.image_min_pixels),
            "--expert",
            expert,
            "--worker-rank",
            str(rank),
            "--world-size",
            str(len(gpu_ids)),
        ]
        if args.max_samples_per_expert is not None:
            command.extend(["--max-samples-per-expert", str(args.max_samples_per_expert)])
        if args.resume:
            command.append("--resume")
        if not args.prefill_empty_think:
            command.append("--no-prefill-empty-think")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        print(f"Launching {expert} rank {rank} on physical GPU {gpu_id}", flush=True)
        processes.append(subprocess.Popen(command, env=environment))
    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"{expert} evaluation worker failure: return codes={return_codes}")

    records = []
    worker_stats = []
    for rank in range(len(gpu_ids)):
        records.extend(read_jsonl(expert_output / f"predictions_rank{rank}.jsonl"))
        worker_stats.append(json.loads((expert_output / f"worker_stats_rank{rank}.json").read_text()))
    records.sort(key=lambda row: row["dataset_index"])
    if len(records) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} {expert} predictions, found {len(records)}.")

    actions = load_actions(args.tools_config)
    metrics = compute_metrics(records, actions)
    metrics.update(
        {
            "expert": expert,
            "wall_seconds": time.perf_counter() - started,
            "adapter": str(adapter),
            "dataset": str(
                args.data_dir
                / (
                    f"{expert}_expert_format_train.jsonl"
                    if args.dataset_profile == "format"
                    else f"{expert}_expert_train.jsonl"
                )
            ),
            "worker_stats": worker_stats,
        }
    )
    write_reports(records, metrics, actions, expert_output)
    print(
        json.dumps(
            {
                "expert": expert,
                "target_action_accuracy": metrics["target_action_accuracy"],
                "schema_valid_rate": metrics["schema_valid_rate"],
                "hermes_valid_rate": metrics["hermes_valid_rate"],
            },
            indent=2,
        ),
        flush=True,
    )
    return metrics


def coordinator_main(args: argparse.Namespace) -> int:
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_expert = {}
    for expert in args.experts:
        metrics_by_expert[expert] = run_one_expert(args, expert, gpu_ids)
    aggregate = {
        "experts": list(args.experts),
        "total_samples": sum(metrics["total_samples"] for metrics in metrics_by_expert.values()),
        "mean_target_action_accuracy": sum(
            metrics["target_action_accuracy"] for metrics in metrics_by_expert.values()
        )
        / len(metrics_by_expert),
        "mean_schema_valid_rate": sum(metrics["schema_valid_rate"] for metrics in metrics_by_expert.values())
        / len(metrics_by_expert),
        "mean_hermes_valid_rate": sum(metrics["hermes_valid_rate"] for metrics in metrics_by_expert.values())
        / len(metrics_by_expert),
        "per_expert": metrics_by_expert,
    }
    (args.output_dir / "metrics_all_experts.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"All expert evaluation reports saved to: {args.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    args.base_model = args.base_model.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.adapter_root = args.adapter_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tools_config = args.tools_config.expanduser().resolve()
    if args.worker_rank is not None:
        return worker_main(args)
    return coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
