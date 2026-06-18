#!/usr/bin/env python3
"""Build synthetic GRPO-format SFT data for the four restoration experts.

The dataset is a format cold start, not a restoration-policy oracle. It reuses
the existing expert images and creates independent visual transitions matching
the prompts used by formal GRPO:

1. initial image-only action selection;
2. action selection after a synthetic IQA improvement;
3. action switching after a synthetic IQA decline;
4. stopping after a synthetic sufficient quality gain.

No restoration model or IQA model is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
EXAMPLE_DIR = REPOSITORY_ROOT / "examples/image_restoration_multi_agent"
TOOLS_CONFIG = EXAMPLE_DIR / "config/tools.yaml"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
LABELS = ("fog", "snow", "rain", "low_light")
EXPERT_NAMES = {label: f"{label}_expert" for label in LABELS}
VARIANTS = (
    "initial_action",
    "continue_after_improvement",
    "switch_after_decline",
    "stop_after_sufficient_gain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--samples-per-expert",
        type=int,
        default=None,
        help="Limit source images per expert. By default all existing images are used.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_prompt_builders():
    sys.path.insert(0, str(EXAMPLE_DIR))
    from agents.prompts import (
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
        build_expert_state_prompt,
        build_expert_system_prompt,
    )
    from schemas import ExpertName
    from tool_registry import ToolRegistry

    return (
        build_expert_single_step_sft_system_prompt,
        build_expert_single_step_sft_user_prompt,
        build_expert_state_prompt,
        build_expert_system_prompt,
        ExpertName,
        ToolRegistry,
    )


def load_actions() -> list[str]:
    payload = yaml.safe_load(TOOLS_CONFIG.read_text(encoding="utf-8"))
    return [tool["name"] for tool in payload["tools"]]


def hermes_call(action: str) -> str:
    payload = {"name": "restore_image", "arguments": {"action": action}}
    return "<tool_call>\n" + json.dumps(payload, separators=(",", ":")) + "\n</tool_call>"


def synthetic_scores(aggregate_score: float) -> dict[str, float]:
    return {
        "maniqa": aggregate_score,
        "niqe": aggregate_score,
        "clipiqa": aggregate_score,
        "topiq_nr": aggregate_score,
    }


def history_step(
    *,
    step_index: int,
    action: str,
    aggregate_score: float,
    previous_score: float,
    original_score: float,
    best_before: float,
) -> dict[str, Any]:
    delta_previous = aggregate_score - previous_score
    delta_original = aggregate_score - original_score
    delta_best = aggregate_score - best_before
    improved = aggregate_score > best_before
    return {
        "step_index": step_index,
        "action": action,
        "success": True,
        "raw_scores": synthetic_scores(aggregate_score),
        "normalized_scores": synthetic_scores(aggregate_score),
        "aggregate_score": aggregate_score,
        "delta_from_previous": delta_previous,
        "delta_from_original": delta_original,
        "delta_from_best": delta_best,
        "is_new_best": improved,
        "step_reward": delta_previous,
        "feedback": (
            f"IQA aggregate {'improved' if improved else 'did not improve'}; "
            f"aggregate_score={aggregate_score:.4f}."
        ),
        "error": None,
    }


def build_state_user_prompt(
    build_expert_state_prompt,
    *,
    step_index: int,
    current_score: float,
    best_score: float,
    original_score: float,
    consecutive_no_improvement: int,
    history: list[dict[str, Any]],
) -> str:
    latest_feedback = history[-1]["feedback"]
    state_prompt = build_expert_state_prompt(
        step_index=step_index,
        remaining_steps=6 - step_index,
        current_score=current_score,
        best_score=best_score,
        original_score=original_score,
        consecutive_no_improvement=consecutive_no_improvement,
        history_json=json.dumps(history, ensure_ascii=False, separators=(",", ":")),
        latest_feedback=latest_feedback,
    )
    return "<image>\n" + state_prompt


def build_variants(
    *,
    image_row: dict[str, Any],
    image_index: int,
    actions: list[str],
    build_single_system,
    build_single_user,
    build_state_prompt,
    build_full_system,
    expert,
    registry,
) -> list[dict[str, Any]]:
    action_0 = actions[image_index % len(actions)]
    action_1 = actions[(image_index + 5) % len(actions)]
    action_2 = actions[(image_index + 11) % len(actions)]
    original_score = 0.0

    history_1 = [
        history_step(
            step_index=0,
            action=action_0,
            aggregate_score=0.08,
            previous_score=original_score,
            original_score=original_score,
            best_before=original_score,
        )
    ]
    history_2 = [
        *history_1,
        history_step(
            step_index=1,
            action=action_1,
            aggregate_score=0.03,
            previous_score=0.08,
            original_score=original_score,
            best_before=0.08,
        ),
    ]
    history_3 = [
        *history_2,
        history_step(
            step_index=2,
            action=action_2,
            aggregate_score=0.14,
            previous_score=0.03,
            original_score=original_score,
            best_before=0.08,
        ),
    ]

    full_system = build_full_system(expert, registry)
    definitions = [
        (
            "initial_action",
            build_single_system(expert, registry),
            build_single_user(),
            action_0,
            0,
        ),
        (
            "continue_after_improvement",
            full_system,
            build_state_user_prompt(
                build_state_prompt,
                step_index=1,
                current_score=0.08,
                best_score=0.08,
                original_score=original_score,
                consecutive_no_improvement=0,
                history=history_1,
            ),
            action_1,
            1,
        ),
        (
            "switch_after_decline",
            full_system,
            build_state_user_prompt(
                build_state_prompt,
                step_index=2,
                current_score=0.03,
                best_score=0.08,
                original_score=original_score,
                consecutive_no_improvement=1,
                history=history_2,
            ),
            action_2,
            2,
        ),
        (
            "stop_after_sufficient_gain",
            full_system,
            build_state_user_prompt(
                build_state_prompt,
                step_index=3,
                current_score=0.14,
                best_score=0.14,
                original_score=original_score,
                consecutive_no_improvement=0,
                history=history_3,
            ),
            "stop",
            3,
        ),
    ]

    output = []
    for variant, system_prompt, user_prompt, target_action, step_index in definitions:
        output.append(
            {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": hermes_call(target_action)},
                ],
                "system": system_prompt,
                "images": image_row["images"],
                "sample_id": f"{image_row['sample_id']}-format-step{step_index}",
                "split": "train",
                "degradation_type": image_row["degradation_type"],
                "expert_name": image_row["expert_name"],
                "metadata": {
                    "label_source": "synthetic_grpo_format",
                    "state_variant": variant,
                    "step_index": step_index,
                    "selected_action": target_action,
                    "synthetic_state": True,
                    "source_sample_id": image_row["sample_id"],
                },
            }
        )
    return output


def validate_rows(rows: list[dict[str, Any]], actions: set[str]) -> None:
    expected_prefix = "<tool_call>\n"
    for row in rows:
        if len(row["messages"]) != 2:
            raise ValueError(f"{row['sample_id']} does not contain one user/assistant pair")
        if not row["messages"][0]["content"].startswith("<image>\n"):
            raise ValueError(f"{row['sample_id']} is missing the image placeholder")
        assistant = row["messages"][1]["content"]
        if not assistant.startswith(expected_prefix) or not assistant.endswith("\n</tool_call>"):
            raise ValueError(f"{row['sample_id']} is not a pure Hermes response")
        payload = json.loads(assistant.removeprefix(expected_prefix).removesuffix("\n</tool_call>"))
        if payload.get("name") != "restore_image":
            raise ValueError(f"{row['sample_id']} uses an invalid function name")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) != {"action"}:
            raise ValueError(f"{row['sample_id']} uses invalid arguments")
        if arguments["action"] not in actions | {"stop"}:
            raise ValueError(f"{row['sample_id']} uses an unknown action")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    actions = load_actions()
    (
        build_single_system,
        build_single_user,
        build_state_prompt,
        build_full_system,
        ExpertName,
        ToolRegistry,
    ) = load_prompt_builders()
    registry = ToolRegistry.from_yaml(TOOLS_CONFIG)
    expert_enum = {
        "fog": ExpertName.FOG,
        "snow": ExpertName.SNOW,
        "rain": ExpertName.RAIN,
        "low_light": ExpertName.LOW_LIGHT,
    }

    dataset_info_path = data_dir / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "version": 2,
        "purpose": "synthetic_grpo_format_cold_start",
        "seed": args.seed,
        "variants": list(VARIANTS),
        "actions": [*actions, "stop"],
        "experts": {},
    }

    for label in LABELS:
        source_path = data_dir / f"{label}_expert_train.jsonl"
        source_rows = read_jsonl(source_path)
        random.Random(args.seed + LABELS.index(label)).shuffle(source_rows)
        if args.samples_per_expert is not None:
            source_rows = source_rows[: args.samples_per_expert]

        output_rows: list[dict[str, Any]] = []
        for index, source_row in enumerate(source_rows):
            output_rows.extend(
                build_variants(
                    image_row=source_row,
                    image_index=index,
                    actions=actions,
                    build_single_system=build_single_system,
                    build_single_user=build_single_user,
                    build_state_prompt=build_state_prompt,
                    build_full_system=build_full_system,
                    expert=expert_enum[label],
                    registry=registry,
                )
            )
        random.Random(args.seed + 100 + LABELS.index(label)).shuffle(output_rows)
        validate_rows(output_rows, set(actions))

        file_name = f"{label}_expert_format_train.jsonl"
        output_path = data_dir / file_name
        write_jsonl(output_path, output_rows)
        dataset_name = f"image_restoration_{label}_expert_format_train"
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
        action_counts = Counter(row["metadata"]["selected_action"] for row in output_rows)
        variant_counts = Counter(row["metadata"]["state_variant"] for row in output_rows)
        manifest["experts"][label] = {
            "file_name": file_name,
            "source_images": len(source_rows),
            "num_samples": len(output_rows),
            "variant_counts": dict(sorted(variant_counts.items())),
            "action_counts": dict(sorted(action_counts.items())),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (data_dir / "format_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
