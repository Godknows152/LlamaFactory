#!/usr/bin/env python3
"""Build RL-aligned, no-thinking SFT datasets for four restoration experts.

Each source image produces five independent decision samples that mirror the
current restoration rollout prompts: an initial action, two pre-stop decisions,
a decision that should continue after stop becomes available, and a decision
that should stop after sufficient IQA improvement. The assistant targets are
structured function calls; LLaMA-Factory renders them with the native Qwen3.5
XML tool-call format.

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
MIN_STOP_TOOL_CALLS = 3
VARIANTS = (
    "initial_action",
    "continue_after_improvement",
    "recover_after_regression",
    "continue_when_stop_available",
    "stop_after_sufficient_gain",
)
NON_STOP_VARIANT_COUNT = 4
EXPECTED_RESTORATION_ACTIONS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--samples-per-expert",
        type=int,
        default=None,
        help="Limit source images per expert for local validation; default uses every image.",
    )
    return parser.parse_args()


def load_project_components():
    sys.path.insert(0, str(EXAMPLE_DIR))
    from agents.prompts import (  # noqa: PLC0415
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
        build_expert_state_prompt,
        build_expert_system_prompt,
    )
    from schemas import ExpertName  # noqa: PLC0415
    from tool_registry import ToolRegistry  # noqa: PLC0415

    return (
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
        build_expert_state_prompt,
        build_expert_system_prompt,
        ExpertName,
        ToolRegistry,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stage_images(source_root: Path, data_dir: Path, label: str, limit: int | None) -> list[Path]:
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


def history_entry(step_index: int, action: str, aggregate_score: float) -> str:
    return f"Step {step_index}: selected action {action}; IQA aggregate_score={aggregate_score:.4f}."


def history_feedback(entries: list[str]) -> str:
    if not entries:
        return "No historical restoration actions have been executed yet."
    return "Historical tool feedback: " + " ".join(entries)


def state_user_prompt(build_state, entries: list[str], *, include_stop: bool) -> str:
    prompt = build_state(history_feedback=history_feedback(entries))
    if include_stop:
        prompt += (
            "\n\nThe stop action is now available. Use action stop only when further processing is unlikely "
            "to improve the historical best image."
        )
    else:
        remaining = max(0, MIN_STOP_TOOL_CALLS - len(entries))
        prompt += (
            "\n\nThe stop action is not available yet. Continue with one non-stop restoration action; "
            f"{remaining} more restoration tool call(s) are required before stop becomes available."
        )
    # SFT rows are decision-local, so each row explicitly binds its current image.
    return "<image>\n" + prompt


def function_call(action: str) -> str:
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
    variant: str,
    step_index: int,
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    action: str,
    history: list[str],
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "function_call", "content": function_call(action)},
        ],
        "system": system,
        "tools": json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
        "images": [str(image_path.relative_to(image_path.parents[2]))],
        "sample_id": f"{label}-{sample_index:06d}-{variant}",
        "degradation_type": label,
        "expert_name": EXPERT_NAMES[label],
        "metadata": {
            "prompt_alignment": "old_verl_current_restoration_decision_v1",
            "state_variant": variant,
            "step_index": step_index,
            "selected_action": action,
            "history_feedback": history_feedback(history),
            "enable_thinking": False,
            "synthetic_state": step_index > 0,
        },
    }


def build_image_rows(
    *,
    label: str,
    image_path: Path,
    sample_index: int,
    restoration_actions: tuple[str, ...],
    registry,
    expert,
    build_initial_system,
    build_initial_user,
    build_state,
    build_system,
) -> list[dict[str, Any]]:
    action_stride = len(restoration_actions) // NON_STOP_VARIANT_COUNT
    first_action = restoration_actions[sample_index % len(restoration_actions)]
    second_action = restoration_actions[(sample_index + action_stride) % len(restoration_actions)]
    third_action = restoration_actions[(sample_index + 2 * action_stride) % len(restoration_actions)]
    fourth_action = restoration_actions[(sample_index + 3 * action_stride) % len(restoration_actions)]

    first = history_entry(0, first_action, 0.3000)
    regressed = history_entry(1, second_action, 0.1200)
    weak_recovery = history_entry(2, third_action, 0.1800)
    improved = history_entry(1, second_action, 0.4200)
    sufficient = history_entry(2, third_action, 0.6500)

    no_stop_tools = [registry.build_tool_schema(include_stop=False)]
    with_stop_tools = [registry.build_tool_schema(include_stop=True)]
    initial_system = build_initial_system(expert, registry)
    continue_system = build_system(
        expert,
        registry,
        allow_stop=False,
        min_stop_tool_calls=MIN_STOP_TOOL_CALLS,
    )
    stop_system = build_system(expert, registry, allow_stop=True)

    definitions = [
        (
            "initial_action",
            0,
            initial_system,
            build_initial_user(),
            no_stop_tools,
            first_action,
            [],
        ),
        (
            "continue_after_improvement",
            1,
            continue_system,
            state_user_prompt(build_state, [first], include_stop=False),
            no_stop_tools,
            second_action,
            [first],
        ),
        (
            "recover_after_regression",
            2,
            continue_system,
            state_user_prompt(build_state, [first, regressed], include_stop=False),
            no_stop_tools,
            third_action,
            [first, regressed],
        ),
        (
            "continue_when_stop_available",
            3,
            stop_system,
            state_user_prompt(build_state, [first, regressed, weak_recovery], include_stop=True),
            with_stop_tools,
            fourth_action,
            [first, regressed, weak_recovery],
        ),
        (
            "stop_after_sufficient_gain",
            3,
            stop_system,
            state_user_prompt(build_state, [first, improved, sufficient], include_stop=True),
            with_stop_tools,
            "stop",
            [first, improved, sufficient],
        ),
    ]

    return [
        make_row(
            label=label,
            image_path=image_path,
            sample_index=sample_index,
            variant=variant,
            step_index=step_index,
            system=system,
            user=user,
            tools=tools,
            action=action,
            history=history,
        )
        for variant, step_index, system, user, tools, action, history in definitions
    ]


def validate_rows(rows: list[dict[str, Any]], data_dir: Path, registry) -> None:
    expected_variants = set(VARIANTS)
    for row in rows:
        sample_id = row["sample_id"]
        if len(row["messages"]) != 2:
            raise ValueError(f"{sample_id}: expected one user/function_call pair")
        user, assistant = row["messages"]
        if user["role"] != "user" or not user["content"].startswith("<image>\n"):
            raise ValueError(f"{sample_id}: user message is not image aligned")
        if assistant["role"] != "function_call":
            raise ValueError(f"{sample_id}: assistant target must be a structured function call")
        if "<think>" in assistant["content"] or "</think>" in assistant["content"]:
            raise ValueError(f"{sample_id}: thinking content is forbidden")
        call = json.loads(assistant["content"])
        if call.get("name") != "restore_image" or set(call.get("arguments", {})) != {"action"}:
            raise ValueError(f"{sample_id}: invalid restore_image call")
        action = call["arguments"]["action"]
        registry.validate_action(action)
        tool_definitions = json.loads(row["tools"])
        schema_actions = tool_definitions[0]["function"]["parameters"]["properties"]["action"]["enum"]
        if action not in schema_actions:
            raise ValueError(f"{sample_id}: target action is absent from its tool schema")
        variant = row["metadata"]["state_variant"]
        if variant not in expected_variants:
            raise ValueError(f"{sample_id}: unknown state variant {variant}")
        stop_available = "stop" in schema_actions
        if stop_available != (row["metadata"]["step_index"] >= MIN_STOP_TOOL_CALLS):
            raise ValueError(f"{sample_id}: stop availability does not match the RL step")
        image_path = data_dir / row["images"][0]
        if not image_path.is_file():
            raise FileNotFoundError(f"{sample_id}: staged image does not resolve: {image_path}")


def dataset_definition(file_name: str) -> dict[str, Any]:
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

    (
        build_initial_system,
        build_initial_user,
        build_state,
        build_system,
        ExpertName,
        ToolRegistry,
    ) = load_project_components()
    registry = ToolRegistry.from_yaml(TOOLS_CONFIG)
    restoration_actions = tuple(action for action in registry.actions if action != "stop")
    if len(restoration_actions) != EXPECTED_RESTORATION_ACTIONS:
        raise ValueError(
            f"expected {EXPECTED_RESTORATION_ACTIONS} non-stop restoration actions, "
            f"found {len(restoration_actions)}"
        )
    if len(restoration_actions) % NON_STOP_VARIANT_COUNT != 0:
        raise ValueError("the restoration action count must be divisible by the non-stop variant count")

    experts = {
        "fog": ExpertName.FOG,
        "snow": ExpertName.SNOW,
        "rain": ExpertName.RAIN,
        "low_light": ExpertName.LOW_LIGHT,
    }

    dataset_info: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "version": 1,
        "purpose": "rl_aligned_four_expert_sft_without_reasoning_targets",
        "seed": args.seed,
        "source_root": str(source_root),
        "prompt_alignment": "old_verl_current_restoration_decision_v1",
        "assistant_format": "qwen3.5_xml_restore_image_function_call",
        "enable_thinking": False,
        "min_stop_tool_calls": MIN_STOP_TOOL_CALLS,
        "variants": list(VARIANTS),
        "target_distribution": {
            "strategy": "uniform_over_non_stop_restoration_tools_per_expert",
            "restoration_actions": list(restoration_actions),
            "stop_is_separate_termination_target": True,
        },
        "experts": {},
    }

    for label_index, label in enumerate(LABELS):
        staged_images = stage_images(source_root, data_dir, label, args.samples_per_expert)
        rows: list[dict[str, Any]] = []
        for sample_index, image_path in enumerate(staged_images):
            rows.extend(
                build_image_rows(
                    label=label,
                    image_path=image_path,
                    sample_index=sample_index,
                    restoration_actions=restoration_actions,
                    registry=registry,
                    expert=experts[label],
                    build_initial_system=build_initial_system,
                    build_initial_user=build_initial_user,
                    build_state=build_state,
                    build_system=build_system,
                )
            )
        random.Random(args.seed + label_index).shuffle(rows)
        validate_rows(rows, data_dir, registry)
        action_counts = Counter(row["metadata"]["selected_action"] for row in rows)
        non_stop_counts = [action_counts[action] for action in restoration_actions]
        if max(non_stop_counts) - min(non_stop_counts) > 1:
            raise ValueError(f"{label}: non-stop restoration targets are not uniformly distributed")
        if len(staged_images) * NON_STOP_VARIANT_COUNT % len(restoration_actions) == 0:
            expected_count = len(staged_images) * NON_STOP_VARIANT_COUNT // len(restoration_actions)
            if any(count != expected_count for count in non_stop_counts):
                raise ValueError(f"{label}: expected exactly {expected_count} targets per restoration action")
        if action_counts["stop"] != len(staged_images):
            raise ValueError(f"{label}: expected one stop target per source image")

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
            "variant_counts": dict(sorted(Counter(row["metadata"]["state_variant"] for row in rows).items())),
            "action_counts": dict(sorted(action_counts.items())),
            "non_stop_action_count_range": [min(non_stop_counts), max(non_stop_counts)],
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
