#!/usr/bin/env python3
"""Build four single-step image-restoration expert SFT datasets.

Each class contributes 1000 source images. Half receive the action with the
highest calibrated four-metric IQA score after running all restoration tools;
the other half receive deterministic uniformly distributed tool labels.

Run from the Agent_Lightning repository root:

    /home/LXJ/anaconda3/envs/verl/bin/python \
      LlamaFactory/image_restoration_experts/scripts/build_expert_sft_dataset.py \
      --gpus 0,1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
EXAMPLE_DIR = REPOSITORY_ROOT / "examples/image_restoration_multi_agent"
CALIBRATION_SCRIPT = EXAMPLE_DIR / "calibrate_iqa_reward.py"
TOOLS_CONFIG = EXAMPLE_DIR / "config/tools.yaml"
REWARD_CONFIG = EXAMPLE_DIR / "config/iqa_reward_v1.json"
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / "LlamaFactory/image_restoration_diagnosis/data/images/train"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data"
DEFAULT_CACHE_DIR = PROJECT_DIR / "cache/iqa_best"
DEFAULT_PYTHON = Path("/home/LXJ/anaconda3/envs/verl/bin/python")
LABELS = ("fog", "snow", "rain", "low_light")
EXPERT_NAMES = {label: f"{label}_expert" for label in LABELS}
SAMPLES_PER_CLASS = 1000
IQA_BEST_SAMPLES_PER_CLASS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--restoration-workers-per-action",
        type=int,
        default=1,
        help="Shard each restoration action across this many GPUs.",
    )
    parser.add_argument(
        "--iqa-workers",
        type=int,
        default=None,
        help="Total IQA workers. Defaults to one worker per GPU.",
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_actions() -> list[str]:
    payload = yaml.safe_load(TOOLS_CONFIG.read_text(encoding="utf-8"))
    return [tool["name"] for tool in payload["tools"]]


def load_prompt_builders():
    sys.path.insert(0, str(EXAMPLE_DIR))
    from agents.prompts import (
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
    )
    from schemas import ExpertName
    from tool_registry import ToolRegistry

    return (
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
        ExpertName,
        ToolRegistry,
    )


def create_partition(args: argparse.Namespace, actions: list[str]) -> list[dict[str, Any]]:
    partition_path = args.cache_dir / "partition_manifest.jsonl"
    if args.resume and partition_path.exists():
        rows = read_jsonl(partition_path)
        if len(rows) != SAMPLES_PER_CLASS * len(LABELS):
            raise ValueError(f"Existing partition contains {len(rows)} rows instead of 4000.")
        return rows

    rows = []
    for class_index, label in enumerate(LABELS):
        class_rng = random.Random(args.seed + class_index)
        images = sorted(path.resolve() for path in (args.source_root / label).iterdir() if path.is_file())
        if len(images) != SAMPLES_PER_CLASS:
            raise ValueError(f"Expected exactly 1000 images for {label}, found {len(images)}.")
        class_rng.shuffle(images)
        iqa_images = images[:IQA_BEST_SAMPLES_PER_CLASS]
        coverage_images = images[IQA_BEST_SAMPLES_PER_CLASS:]
        repetitions, remainder = divmod(len(coverage_images), len(actions))
        extra_start = class_index * remainder
        extras = [actions[(extra_start + index) % len(actions)] for index in range(remainder)]
        coverage_actions = actions * repetitions + extras
        class_rng.shuffle(coverage_actions)
        for path in iqa_images:
            rows.append(
                {
                    "sample_id": f"{label}-{path.stem}",
                    "degradation_type": label,
                    "source_path": str(path),
                    "label_source": "iqa_best",
                    "selected_action": None,
                }
            )
        for path, action in zip(coverage_images, coverage_actions):
            rows.append(
                {
                    "sample_id": f"{label}-{path.stem}",
                    "degradation_type": label,
                    "source_path": str(path),
                    "label_source": "uniform_tool_coverage",
                    "selected_action": action,
                }
            )
    rows.sort(key=lambda row: (row["degradation_type"], row["sample_id"]))
    write_jsonl(partition_path, rows)
    return rows


def create_iqa_manifest(args: argparse.Namespace, partition: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = args.cache_dir / "sample_manifest.jsonl"
    rows = [
        {
            "sample_id": row["sample_id"],
            "degradation_type": row["degradation_type"],
            "source_path": row["source_path"],
            "original_path": row["source_path"],
        }
        for row in partition
        if row["label_source"] == "iqa_best"
    ]
    if len(rows) != IQA_BEST_SAMPLES_PER_CLASS * len(LABELS):
        raise ValueError(f"Expected 2000 IQA-best rows, found {len(rows)}.")
    if args.resume and manifest_path.exists():
        existing = read_jsonl(manifest_path)
        if existing != rows:
            raise ValueError("Existing IQA manifest does not match the deterministic partition.")
    else:
        write_jsonl(manifest_path, rows)
    return rows


def run_process_pool(
    commands: list[tuple[str, list[str], dict[str, str]]],
    max_parallel: int,
    max_parallel_per_gpu: int = 1,
) -> None:
    pending = list(commands)
    running: list[tuple[str, str, subprocess.Popen[bytes]]] = []
    while pending or running:
        gpu_loads = Counter(gpu_id for _, gpu_id, _ in running)
        while pending and len(running) < max_parallel:
            next_index = next(
                (
                    index
                    for index, (_, _, environment) in enumerate(pending)
                    if gpu_loads[environment["CUDA_VISIBLE_DEVICES"]] < max_parallel_per_gpu
                ),
                None,
            )
            if next_index is None:
                break
            name, command, environment = pending.pop(next_index)
            gpu_id = environment["CUDA_VISIBLE_DEVICES"]
            print(f"Launching {name}", flush=True)
            running.append((name, gpu_id, subprocess.Popen(command, env=environment)))
            gpu_loads[gpu_id] += 1
        time.sleep(2)
        still_running = []
        for name, gpu_id, process in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((name, gpu_id, process))
            elif return_code != 0:
                for _, _, other_process in running:
                    if other_process.poll() is None:
                        other_process.terminate()
                raise RuntimeError(f"Worker {name} exited with code {return_code}.")
        running = still_running


def run_restoration_and_iqa(args: argparse.Namespace, actions: list[str]) -> None:
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU is required.")
    if not 1 <= args.restoration_workers_per_action <= len(gpu_ids):
        raise ValueError("--restoration-workers-per-action must be between 1 and the number of GPUs.")
    common = [
        "--output-dir",
        str(args.cache_dir),
        "--tools-config",
        str(TOOLS_CONFIG),
        "--external-tools-root",
        str(REPOSITORY_ROOT / "External_Tools"),
    ]
    restoration_commands = []
    worker_count = args.restoration_workers_per_action
    for action_index, action in enumerate(actions):
        for rank in range(worker_count):
            gpu_id = gpu_ids[(action_index * worker_count + rank) % len(gpu_ids)]
            command = [str(args.python), str(CALIBRATION_SCRIPT), "--mode", "restore", "--action", action]
            if worker_count > 1:
                command.extend(["--rank", str(rank), "--world-size", str(worker_count)])
            command.extend(common)
            if args.resume:
                command.append("--resume")
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONUNBUFFERED": "1"})
            restoration_commands.append(
                (f"restoration:{action}:rank{rank}@gpu{gpu_id}", command, environment)
            )
    run_process_pool(restoration_commands, len(gpu_ids))

    iqa_workers = args.iqa_workers or len(gpu_ids)
    if iqa_workers < 1:
        raise ValueError("--iqa-workers must be positive.")
    workers_per_gpu = (iqa_workers + len(gpu_ids) - 1) // len(gpu_ids)
    score_commands = []
    for rank in range(iqa_workers):
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        command = [
            str(args.python),
            str(CALIBRATION_SCRIPT),
            "--mode",
            "score",
            "--rank",
            str(rank),
            "--world-size",
            str(iqa_workers),
            *common,
        ]
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONUNBUFFERED": "1"})
        score_commands.append((f"iqa:rank{rank}@gpu{gpu_id}", command, environment))
    run_process_pool(score_commands, iqa_workers, max_parallel_per_gpu=workers_per_gpu)


def load_iqa_scores(args: argparse.Namespace, expected: int) -> dict[str, dict[str, dict[str, float]]]:
    rows = []
    for path in sorted((args.cache_dir / "progress/iqa").glob("rank*.jsonl")):
        rows.extend(read_jsonl(path))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} IQA score rows, found {len(rows)}.")
    by_sample: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["action"] == "original":
            continue
        by_sample.setdefault(row["sample_id"], {})[row["action"]] = row["oriented_scores"]
    return by_sample


def assign_iqa_best_labels(
    args: argparse.Namespace,
    partition: list[dict[str, Any]],
    actions: list[str],
) -> dict[str, dict[str, Any]]:
    reward = json.loads(REWARD_CONFIG.read_text(encoding="utf-8"))
    metric_configs = reward["metrics"]
    iqa_count = IQA_BEST_SAMPLES_PER_CLASS * len(LABELS)
    by_sample = load_iqa_scores(args, iqa_count * (len(actions) + 1))
    audit: dict[str, dict[str, Any]] = {}
    for row in partition:
        if row["label_source"] != "iqa_best":
            continue
        action_scores = by_sample.get(row["sample_id"], {})
        if set(action_scores) != set(actions):
            missing = sorted(set(actions) - set(action_scores))
            raise RuntimeError(f"Missing IQA actions for {row['sample_id']}: {missing}")
        aggregates = {}
        for action, scores in action_scores.items():
            aggregates[action] = sum(
                config["weight"] * ((scores[metric] - config["mean"]) / config["std"])
                for metric, config in metric_configs.items()
            )
        ranked = sorted(aggregates.items(), key=lambda item: (-item[1], actions.index(item[0])))
        selected_action = ranked[0][0]
        row["selected_action"] = selected_action
        audit[row["sample_id"]] = {
            "selected_action": selected_action,
            "ranked_action_scores": [{"action": action, "aggregate_score": score} for action, score in ranked],
        }
    write_jsonl(args.cache_dir / "iqa_best_action_audit.jsonl", [
        {"sample_id": sample_id, **payload} for sample_id, payload in sorted(audit.items())
    ])
    return audit


def stage_images(args: argparse.Namespace, partition: list[dict[str, Any]]) -> dict[str, str]:
    relative_paths = {}
    for row in partition:
        source = Path(row["source_path"])
        destination = args.output_dir / "images" / row["degradation_type"] / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            if destination.resolve() != source.resolve():
                destination.unlink()
            else:
                relative_paths[row["sample_id"]] = str(destination.relative_to(args.output_dir))
                continue
        destination.symlink_to(source.resolve())
        relative_paths[row["sample_id"]] = str(destination.relative_to(args.output_dir))
    return relative_paths


def build_datasets(
    args: argparse.Namespace,
    partition: list[dict[str, Any]],
    actions: list[str],
    audit: dict[str, dict[str, Any]],
) -> None:
    build_system_prompt, build_user_prompt, ExpertName, ToolRegistry = load_prompt_builders()
    registry = ToolRegistry.from_yaml(TOOLS_CONFIG)
    relative_images = stage_images(args, partition)
    dataset_info = {}
    manifest = {
        "version": 1,
        "seed": args.seed,
        "samples_per_expert": SAMPLES_PER_CLASS,
        "iqa_best_samples_per_expert": IQA_BEST_SAMPLES_PER_CLASS,
        "uniform_tool_coverage_samples_per_expert": SAMPLES_PER_CLASS - IQA_BEST_SAMPLES_PER_CLASS,
        "actions": actions,
        "reward_config": str(REWARD_CONFIG),
        "experts": {},
    }
    expert_enum = {
        "fog": ExpertName.FOG,
        "snow": ExpertName.SNOW,
        "rain": ExpertName.RAIN,
        "low_light": ExpertName.LOW_LIGHT,
    }
    for label in LABELS:
        rows = [row for row in partition if row["degradation_type"] == label]
        random.Random(args.seed + 100 + LABELS.index(label)).shuffle(rows)
        system_prompt = build_system_prompt(expert_enum[label], registry)
        user_prompt = build_user_prompt()
        output_rows = []
        for row in rows:
            action = row["selected_action"]
            if action not in actions:
                raise ValueError(f"Invalid selected action for {row['sample_id']}: {action}")
            assistant = (
                '<tool_call>\n'
                + json.dumps(
                    {"name": "restore_image", "arguments": {"action": action}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n</tool_call>"
            )
            metadata = {
                "label_source": row["label_source"],
                "selected_action": action,
                "reward_config_version": "iqa_reward_v1" if row["label_source"] == "iqa_best" else None,
            }
            if row["label_source"] == "iqa_best":
                metadata["best_aggregate_score"] = audit[row["sample_id"]]["ranked_action_scores"][0][
                    "aggregate_score"
                ]
            output_rows.append(
                {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "system": system_prompt,
                    "images": [relative_images[row["sample_id"]]],
                    "sample_id": row["sample_id"],
                    "split": "train",
                    "degradation_type": label,
                    "expert_name": EXPERT_NAMES[label],
                    "metadata": metadata,
                }
            )
        file_name = f"{label}_expert_train.jsonl"
        write_jsonl(args.output_dir / file_name, output_rows)
        dataset_name = f"image_restoration_{label}_expert_train"
        dataset_info[dataset_name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "system": "system", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        action_counts = Counter(row["selected_action"] for row in rows)
        source_counts = Counter(row["label_source"] for row in rows)
        manifest["experts"][label] = {
            "file_name": file_name,
            "num_samples": len(rows),
            "label_source_counts": dict(sorted(source_counts.items())),
            "action_counts": {action: action_counts[action] for action in actions},
            "sha256": hashlib.sha256((args.output_dir / file_name).read_bytes()).hexdigest(),
        }
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    actions = load_actions()
    partition = create_partition(args, actions)
    create_iqa_manifest(args, partition)
    if not args.skip_inference:
        run_restoration_and_iqa(args, actions)
    audit = assign_iqa_best_labels(args, partition, actions)
    build_datasets(args, partition, actions, audit)
    print(f"Expert SFT datasets saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
