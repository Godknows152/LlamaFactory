#!/usr/bin/env python3
"""Build the four expert SFT datasets with first-token-distinct actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from build_expert_sft_dataset import (
    DEFAULT_SOURCE_ROOT,
    EXPERT_NAMES,
    LABELS,
    TOOLS_CONFIG,
    dataset_definition,
    load_project_components,
    stage_images,
    write_jsonl,
)
from first_token_action_vocabulary import (
    alias_thinking_text,
    validate_action_aliases,
)
from restoration_thinking_templates import (
    RESTORATION_THINKING_TEMPLATES,
    THINKING_TEMPLATES_PER_ACTION,
    validate_thinking_templates,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "first_token_actions"
EXPECTED_RESTORATION_ACTIONS = 16
MANIFEST_VERSION = 5
MANIFEST_PURPOSE = "first_turn_four_expert_sft_with_first_token_distinct_action_aliases"
PROMPT_ALIGNMENT = "old_verl_grpo_first_restoration_turn_with_first_token_actions"
PROMPT_VERSION = "expert-single-step-sft-hermes-thinking-v4"


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


def build_model_tool_schema(registry: Any, *, include_stop: bool) -> dict[str, Any]:
    """Return the shared model-facing restore_image schema."""

    return registry.build_tool_schema(include_stop=include_stop)


def render_tool_call(action_alias: str) -> str:
    return (
        "<tool_call>\n<function=restore_image>\n<parameter=action>\n"
        f"{action_alias}\n"
        "</parameter>\n</function>\n</tool_call>"
    )


def assistant_target(thinking_text: str, action_alias: str) -> str:
    return f"<think>\n{thinking_text}\n</think>\n\n{render_tool_call(action_alias)}"


def make_row(
    *,
    label: str,
    image_path: Path,
    sample_index: int,
    system: str,
    user: str,
    tools: list[dict[str, Any]],
    canonical_action: str,
    action_aliases: dict[str, str],
    thinking_text: str,
    thinking_template_index: int,
) -> dict[str, Any]:
    action_alias = action_aliases[canonical_action]
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_target(thinking_text, action_alias)},
        ],
        "system": system,
        "tools": json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
        "images": [str(image_path.relative_to(image_path.parents[2]))],
        "sample_id": f"{label}-{sample_index:06d}-first-token-action-first-turn",
        "degradation_type": label,
        "expert_name": EXPERT_NAMES[label],
        "metadata": {
            "prompt_alignment": PROMPT_ALIGNMENT,
            "prompt_version": PROMPT_VERSION,
            "turn_index": 0,
            "selected_action": canonical_action,
            "selected_action_alias": action_alias,
            "enable_thinking": True,
            "thinking_template_index": thinking_template_index,
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
    action_aliases: dict[str, str],
) -> None:
    expected_actions = list(registry.actions[:-1])
    expected_aliases = [action_aliases[action] for action in expected_actions]
    expected_tools_text = json.dumps(expected_tools, ensure_ascii=False, separators=(",", ":"))
    seen_images: set[str] = set()
    seen_sample_ids: set[str] = set()

    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_sample_ids.add(sample_id)

        user, assistant = row["messages"]
        if user != {"role": "user", "content": expected_user}:
            raise ValueError(f"{sample_id}: user prompt differs from the RL first-turn prompt")
        if row["system"] != expected_system or row["tools"] != expected_tools_text:
            raise ValueError(f"{sample_id}: system prompt or tool schema is inconsistent")

        metadata = row["metadata"]
        canonical_action = metadata["selected_action"]
        action_alias = metadata["selected_action_alias"]
        template_index = metadata["thinking_template_index"]
        if canonical_action not in expected_actions:
            raise ValueError(f"{sample_id}: invalid canonical action {canonical_action!r}")
        if action_alias != action_aliases[canonical_action]:
            raise ValueError(f"{sample_id}: canonical action and model alias disagree")
        registry.validate_action(canonical_action)

        original_thinking = RESTORATION_THINKING_TEMPLATES[canonical_action][template_index]
        thinking_text = alias_thinking_text(canonical_action, original_thinking, action_aliases)
        if assistant != {
            "role": "assistant",
            "content": assistant_target(thinking_text, action_alias),
        }:
            raise ValueError(f"{sample_id}: assistant reasoning/tool-call target is malformed")
        if assistant["content"].count(action_alias) != 2:
            raise ValueError(f"{sample_id}: alias must appear once in thinking and once in the tool call")

        schema_actions = json.loads(row["tools"])[0]["function"]["parameters"]["properties"]["action"]["enum"]
        if schema_actions != expected_aliases or action_alias not in schema_actions:
            raise ValueError(f"{sample_id}: model-facing first-turn action schema is invalid")
        if action_aliases["stop"] in schema_actions:
            raise ValueError(f"{sample_id}: stop alias is forbidden on the first turn")

        if (
            metadata.get("turn_index") != 0
            or metadata.get("synthetic_state") is not False
            or metadata.get("enable_thinking") is not True
            or metadata.get("prompt_version") != PROMPT_VERSION
        ):
            raise ValueError(f"{sample_id}: metadata does not describe the new first-turn SFT")

        images = row.get("images", [])
        if len(images) != 1 or images[0] in seen_images:
            raise ValueError(f"{sample_id}: expected one unique input image")
        seen_images.add(images[0])
        if not (data_dir / images[0]).is_file():
            raise FileNotFoundError(f"{sample_id}: staged image does not resolve")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    build_system, build_user, ExpertName, ToolRegistry = load_project_components()
    registry = ToolRegistry.from_yaml(TOOLS_CONFIG)
    canonical_actions = tuple(registry.actions[:-1])
    if len(canonical_actions) != EXPECTED_RESTORATION_ACTIONS:
        raise ValueError(f"expected {EXPECTED_RESTORATION_ACTIONS} restoration actions")
    validate_thinking_templates(canonical_actions)
    action_aliases = {action: registry.to_model_action(action) for action in registry.actions}
    validate_action_aliases(action_aliases, registry.actions)

    experts = {
        "fog": ExpertName.FOG,
        "snow": ExpertName.SNOW,
        "rain": ExpertName.RAIN,
        "low_light": ExpertName.LOW_LIGHT,
    }
    initial_user = build_user()
    initial_tools = [build_model_tool_schema(registry, include_stop=False)]
    alias_templates = {
        action: tuple(alias_thinking_text(action, text, action_aliases) for text in texts)
        for action, texts in RESTORATION_THINKING_TEMPLATES.items()
    }
    alias_catalog_json = json.dumps(alias_templates, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    dataset_info: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "purpose": MANIFEST_PURPOSE,
        "seed": args.seed,
        "source_root": str(source_root),
        "prompt_alignment": PROMPT_ALIGNMENT,
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": True,
        "action_vocabulary": {
            "model_to_runtime": {
                alias: action for action, alias in action_aliases.items()
            },
            "runtime_to_model": action_aliases,
            "unique_first_characters": True,
            "includes_stop_action": True,
            "stop_uses_native_name": True,
        },
        "sample_scope": {
            "samples_per_input_image": 1,
            "turn_indices": [0],
            "contains_synthetic_history": False,
            "contains_stop_targets": False,
        },
        "assistant_format": {
            "dataset_role": "assistant",
            "dataset_content": "<think>\\n<REASONING_WITH_ACTION_ALIAS>\\n</think>\\n\\n<TOOL_CALL_WITH_ACTION_ALIAS>",
            "rendered_by_template": "qwen3_5_reasoning_with_native_xml_tool_call",
        },
        "thinking_targets": {
            "selection": "seeded_uniform_random_per_sample",
            "templates_per_action": THINKING_TEMPLATES_PER_ACTION,
            "num_actions": len(RESTORATION_THINKING_TEMPLATES),
            "num_templates": len(RESTORATION_THINKING_TEMPLATES) * THINKING_TEMPLATES_PER_ACTION,
            "catalog_sha256": hashlib.sha256(alias_catalog_json.encode("utf-8")).hexdigest(),
        },
        "target_distribution": {
            "strategy": "round_robin_uniform_over_all_non_stop_restoration_tools_per_expert",
            "canonical_actions": list(canonical_actions),
            "model_actions": [action_aliases[action] for action in canonical_actions],
            "maximum_count_difference": 1,
        },
        "experts": {},
    }

    for label_index, label in enumerate(LABELS):
        staged_images = stage_images(source_root, data_dir, label, args.samples_per_expert)
        initial_system = build_system(experts[label], registry)
        if f"Prompt version: {PROMPT_VERSION}" not in initial_system:
            raise ValueError(f"{label}: SFT prompt version does not match dataset metadata")

        rng = random.Random(args.seed + label_index)
        rows = []
        for sample_index, image_path in enumerate(staged_images):
            canonical_action = canonical_actions[sample_index % len(canonical_actions)]
            template_index = rng.randrange(THINKING_TEMPLATES_PER_ACTION)
            rows.append(
                make_row(
                    label=label,
                    image_path=image_path,
                    sample_index=sample_index,
                    system=initial_system,
                    user=initial_user,
                    tools=initial_tools,
                    canonical_action=canonical_action,
                    action_aliases=action_aliases,
                    thinking_text=alias_templates[canonical_action][template_index],
                    thinking_template_index=template_index,
                )
            )
        rng.shuffle(rows)
        validate_rows(
            rows,
            data_dir=data_dir,
            registry=registry,
            expected_system=initial_system,
            expected_user=initial_user,
            expected_tools=initial_tools,
            action_aliases=action_aliases,
        )

        action_counts = Counter(row["metadata"]["selected_action"] for row in rows)
        counts = [action_counts[action] for action in canonical_actions]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"{label}: restoration targets are not uniformly distributed")

        file_name = f"{label}_expert_first_token_actions_train.jsonl"
        output_path = data_dir / file_name
        write_jsonl(output_path, rows)
        dataset_name = f"image_restoration_{label}_expert_first_token_actions_train"
        dataset_info[dataset_name] = dataset_definition(file_name)
        manifest["experts"][label] = {
            "dataset_name": dataset_name,
            "file_name": file_name,
            "source_images": len(staged_images),
            "num_samples": len(rows),
            "samples_per_input_image": 1,
            "action_counts": dict(sorted(action_counts.items())),
            "action_count_range": [min(counts), max(counts)],
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    (data_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
