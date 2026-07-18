#!/usr/bin/env python3
"""Build first-turn SFT datasets for the four image-restoration experts.

Each source image produces exactly one sample. Its system prompt, user prompt,
tool schema, and tool-call target match the first restoration decision made by
`examples/image_restoration_multi_agent/old_verl_grpo`. No synthetic
multi-turn state, IQA history, later restoration decision, or stop target is
included.

Run from the Agent Lightning repository root:

    /home/LXJ/anaconda3/envs/agent-lightning/bin/python \
      LlamaFactory/image_restoration_experts/scripts/build_expert_sft_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
EXAMPLE_DIR = REPOSITORY_ROOT / "examples/image_restoration_multi_agent"
TOOLS_CONFIG = EXAMPLE_DIR / "config/tools.yaml"
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / "LlamaFactory/image_restoration_diagnosis/data/images/train"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"

LABELS = ("fog", "snow", "rain", "low_light")
EXPERT_NAMES = {label: f"{label}_expert" for label in LABELS}
EXPECTED_RESTORATION_ACTIONS = 16
MANIFEST_VERSION = 2
PROMPT_ALIGNMENT = "old_verl_grpo_first_restoration_turn_v1"
PROMPT_VERSION = "expert-single-step-sft-hermes-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--samples-per-expert",
        type=int,
        default=None,
        help="Limit source images per expert for local validation; default uses every image.",
    )
    return parser.parse_args()


def load_project_components() -> tuple[Any, Any, Any, Any]:
    """Load the exact prompt and tool builders used by the RL first turn."""
    sys.path.insert(0, str(EXAMPLE_DIR))
    from agents.prompts import (  # noqa: PLC0415
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
    )
    from schemas import ExpertName  # noqa: PLC0415
    from tool_registry import ToolRegistry  # noqa: PLC0415

    return (
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
        ExpertName,
        ToolRegistry,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stage_images(source_root: Path, data_dir: Path, label: str, limit: int | None) -> list[Path]:
    """Stage stable relative symlinks for one degradation class."""
    source_dir = source_root / label
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source image directory does not exist: {source_dir}")

    source_paths = sorted(path.resolve() for path in source_dir.iterdir() if path.is_file())
    if limit is not None:
        if limit <= 0:
            raise ValueError("--samples-per-expert must be positive")
        source_paths = source_paths[:limit]
    if not source_paths:
        raise ValueError(f"no source images found for {label}")

    staged_dir = data_dir / "images" / label
    shutil.rmtree(staged_dir, ignore_errors=True)
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_paths: list[Path] = []
    used_names: set[str] = set()
    for index, source_path in enumerate(source_paths):
        name = source_path.name
        if name in used_names:
            name = f"{index:06d}_{name}"
        used_names.add(name)
        staged_path = staged_dir / name
        staged_path.symlink_to(source_path)
        staged_paths.append(staged_path)
    return staged_paths


def function_call(action: str) -> str:
    """Build the structured target consumed by LLaMA-Factory's tool formatter."""
    return json.dumps(
        {"name": "restore_image", "arguments": {"action": action}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_row(
    *,
    label: str,
    image_path: Path,
    sample_index: int,
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    action: str,
) -> dict[str, Any]:
    """Build one image-to-first-action supervision pair."""
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "function_call", "content": function_call(action)},
        ],
        "system": system,
        "tools": json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
        "images": [str(image_path.relative_to(image_path.parents[2]))],
        "sample_id": f"{label}-{sample_index:06d}-first-turn",
        "degradation_type": label,
        "expert_name": EXPERT_NAMES[label],
        "metadata": {
            "prompt_alignment": PROMPT_ALIGNMENT,
            "prompt_version": PROMPT_VERSION,
            "turn_index": 0,
            "selected_action": action,
            "enable_thinking": False,
            "synthetic_state": False,
        },
    }


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    data_dir: Path,
    registry: Any,
    expected_system: str,
    expected_user: str,
    expected_tools: list[dict[str, Any]],
) -> None:
    """Reject any row that is not an exact RL-aligned first-turn sample."""
    expected_actions = [action for action in registry.actions if action != "stop"]
    expected_tools_text = json.dumps(expected_tools, ensure_ascii=False, separators=(",", ":"))
    seen_images: set[str] = set()
    seen_sample_ids: set[str] = set()

    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_sample_ids.add(sample_id)

        if len(row["messages"]) != 2:
            raise ValueError(f"{sample_id}: expected one user/function_call pair")
        user, assistant = row["messages"]
        if user != {"role": "user", "content": expected_user}:
            raise ValueError(f"{sample_id}: user prompt differs from the RL first-turn prompt")
        if row["system"] != expected_system:
            raise ValueError(f"{sample_id}: system prompt differs from the RL first-turn prompt")
        if row["tools"] != expected_tools_text:
            raise ValueError(f"{sample_id}: tool schema differs from the RL first-turn schema")
        if assistant["role"] != "function_call":
            raise ValueError(f"{sample_id}: assistant target must be a structured function call")
        if "<think>" in assistant["content"] or "</think>" in assistant["content"]:
            raise ValueError(f"{sample_id}: thinking content is forbidden")

        call = json.loads(assistant["content"])
        if call.get("name") != "restore_image" or set(call.get("arguments", {})) != {"action"}:
            raise ValueError(f"{sample_id}: invalid restore_image call")
        action = call["arguments"]["action"]
        if action == "stop":
            raise ValueError(f"{sample_id}: stop is forbidden on the first restoration turn")
        registry.validate_action(action)
        schema_actions = json.loads(row["tools"])[0]["function"]["parameters"]["properties"]["action"]["enum"]
        if schema_actions != expected_actions or action not in schema_actions:
            raise ValueError(f"{sample_id}: invalid first-turn action schema")

        metadata = row["metadata"]
        if (
            metadata.get("turn_index") != 0
            or metadata.get("synthetic_state") is not False
            or metadata.get("enable_thinking") is not False
        ):
            raise ValueError(f"{sample_id}: metadata does not describe a no-thinking first-turn sample")

        images = row.get("images", [])
        if len(images) != 1:
            raise ValueError(f"{sample_id}: expected exactly one input image")
        if images[0] in seen_images:
            raise ValueError(f"{sample_id}: input image appears in more than one SFT row")
        seen_images.add(images[0])
        image_path = data_dir / images[0]
        if not image_path.is_file():
            raise FileNotFoundError(f"{sample_id}: staged image does not resolve: {image_path}")


def dataset_definition(file_name: str) -> dict[str, Any]:
    """Build one LLaMA-Factory ShareGPT dataset registration."""
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "system": "system", "tools": "tools", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "observation",
            "function_tag": "function_call",
            "system_tag": "system",
        },
    }


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    build_initial_system, build_initial_user, ExpertName, ToolRegistry = load_project_components()
    registry = ToolRegistry.from_yaml(TOOLS_CONFIG)
    restoration_actions = tuple(action for action in registry.actions if action != "stop")
    if len(restoration_actions) != EXPECTED_RESTORATION_ACTIONS:
        raise ValueError(
            f"expected {EXPECTED_RESTORATION_ACTIONS} non-stop restoration actions, found {len(restoration_actions)}"
        )

    experts = {
        "fog": ExpertName.FOG,
        "snow": ExpertName.SNOW,
        "rain": ExpertName.RAIN,
        "low_light": ExpertName.LOW_LIGHT,
    }
    initial_user = build_initial_user()
    initial_tools = [registry.build_tool_schema(include_stop=False)]

    dataset_info: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "purpose": "first_turn_only_rl_aligned_four_expert_sft_without_reasoning_targets",
        "seed": args.seed,
        "source_root": str(source_root),
        "prompt_alignment": PROMPT_ALIGNMENT,
        "prompt_version": PROMPT_VERSION,
        "sample_scope": {
            "samples_per_input_image": 1,
            "turn_indices": [0],
            "contains_synthetic_history": False,
            "contains_stop_targets": False,
        },
        "assistant_format": {
            "dataset_role": "function_call",
            "dataset_content": '{"name":"restore_image","arguments":{"action":"<ACTION>"}}',
            "rendered_by_template": "qwen3_5_xml_tool_call",
            "rendered_example": (
                "<tool_call>\n<function=restore_image>\n<parameter=action>\n"
                "<ACTION>\n</parameter>\n</function>\n</tool_call>"
            ),
        },
        "enable_thinking": False,
        "target_distribution": {
            "strategy": "round_robin_uniform_over_all_non_stop_restoration_tools_per_expert",
            "restoration_actions": list(restoration_actions),
            "maximum_count_difference": 1,
        },
        "experts": {},
    }

    for label_index, label in enumerate(LABELS):
        staged_images = stage_images(source_root, data_dir, label, args.samples_per_expert)
        initial_system = build_initial_system(experts[label], registry)
        rows = [
            make_row(
                label=label,
                image_path=image_path,
                sample_index=sample_index,
                system=initial_system,
                user=initial_user,
                tools=initial_tools,
                action=restoration_actions[sample_index % len(restoration_actions)],
            )
            for sample_index, image_path in enumerate(staged_images)
        ]
        random.Random(args.seed + label_index).shuffle(rows)
        validate_rows(
            rows,
            data_dir=data_dir,
            registry=registry,
            expected_system=initial_system,
            expected_user=initial_user,
            expected_tools=initial_tools,
        )

        action_counts = Counter(row["metadata"]["selected_action"] for row in rows)
        counts = [action_counts[action] for action in restoration_actions]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"{label}: restoration targets are not uniformly distributed")
        if "stop" in action_counts:
            raise ValueError(f"{label}: first-turn dataset unexpectedly contains stop")
        if len(rows) != len(staged_images):
            raise ValueError(f"{label}: expected exactly one SFT row per source image")

        file_name = f"{label}_expert_rl_aligned_train.jsonl"
        output_path = data_dir / file_name
        write_jsonl(output_path, rows)
        dataset_name = f"image_restoration_{label}_expert_rl_aligned_train"
        dataset_info[dataset_name] = dataset_definition(file_name)
        manifest["experts"][label] = {
            "dataset_name": dataset_name,
            "file_name": file_name,
            "source_images": len(staged_images),
            "num_samples": len(rows),
            "samples_per_input_image": 1,
            "turn_counts": {"0": len(rows)},
            "action_counts": dict(sorted(action_counts.items())),
            "action_count_range": [min(counts), max(counts)],
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    (data_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
