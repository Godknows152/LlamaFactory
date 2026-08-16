#!/usr/bin/env python3
"""Merge the four first-token-distinct expert datasets without reshuffling."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SOURCE_DIR = DATA_DIR / "first_token_actions"
SOURCE_MANIFEST_PATH = SOURCE_DIR / "manifest.json"
OUTPUT_PATH = DATA_DIR / "unified_expert_first_token_actions_train.jsonl"
OUTPUT_MANIFEST_PATH = DATA_DIR / "unified_expert_manifest.json"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"

EXPERTS = ("fog", "snow", "rain", "low_light")
DATASET_NAME = "image_restoration_unified_expert_first_token_actions_train"
EXPECTED_SOURCE_PURPOSE = "first_turn_four_expert_sft_with_first_token_distinct_action_aliases"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_registration() -> dict[str, Any]:
    return {
        "file_name": OUTPUT_PATH.name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "system": "system",
            "tools": "tools",
            "images": "images",
        },
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


def main() -> None:
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    if source_manifest.get("version") != 5 or source_manifest.get("purpose") != EXPECTED_SOURCE_PURPOSE:
        raise SystemExit("The source manifest is not the expected version 5 first-token-distinct dataset.")

    source_entries: dict[str, Any] = source_manifest.get("experts", {})
    rows: list[str] = []
    expert_counts: Counter[str] = Counter()
    degradation_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    source_summary: dict[str, Any] = {}

    for expert in EXPERTS:
        entry = source_entries.get(expert)
        if not isinstance(entry, dict):
            raise SystemExit(f"Missing source manifest entry for {expert!r}.")

        source_path = SOURCE_DIR / entry["file_name"]
        if not source_path.is_file():
            raise SystemExit(f"Missing source dataset: {source_path}")
        if sha256(source_path) != entry.get("sha256"):
            raise SystemExit(f"Source checksum mismatch for {source_path.name}.")

        source_rows = 0
        with source_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise SystemExit(f"Missing sample_id in {source_path.name}:{line_number}.")
                if sample_id in sample_ids:
                    raise SystemExit(f"Duplicate sample_id across expert datasets: {sample_id}")
                sample_ids.add(sample_id)

                expected_expert_name = f"{expert}_expert"
                if row.get("expert_name") != expected_expert_name or row.get("degradation_type") != expert:
                    raise SystemExit(f"Expert metadata mismatch in {source_path.name}:{line_number}.")

                metadata = row.get("metadata", {})
                alias = metadata.get("selected_action_alias")
                canonical_action = metadata.get("selected_action")
                runtime_to_model = source_manifest["action_vocabulary"]["runtime_to_model"]
                if runtime_to_model.get(canonical_action) != alias:
                    raise SystemExit(f"Action alias mismatch in {source_path.name}:{line_number}.")

                rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                source_rows += 1
                expert_counts[expert] += 1
                degradation_counts[row["degradation_type"]] += 1
                action_counts[canonical_action] += 1

        if source_rows != 1000 or source_rows != entry.get("num_samples"):
            raise SystemExit(f"Expected 1,000 rows for {expert}, found {source_rows}.")
        source_summary[expert] = {
            "file_name": source_path.name,
            "num_samples": source_rows,
            "sha256": sha256(source_path),
        }

    if len(rows) != 4000 or any(expert_counts[name] != 1000 for name in EXPERTS):
        raise SystemExit(f"Expected four balanced 1,000-row datasets, got {dict(expert_counts)}.")

    OUTPUT_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset_info = load_json(DATASET_INFO_PATH) if DATASET_INFO_PATH.exists() else {}
    dataset_info[DATASET_NAME] = dataset_registration()
    DATASET_INFO_PATH.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_manifest = {
        "version": 1,
        "purpose": "unified_four_expert_first_token_distinct_sft",
        "dataset_name": DATASET_NAME,
        "file_name": OUTPUT_PATH.name,
        "merge_strategy": "direct_concatenation_without_shuffle",
        "expert_order": list(EXPERTS),
        "num_samples": len(rows),
        "expert_counts": dict(expert_counts),
        "degradation_counts": dict(degradation_counts),
        "action_counts": dict(sorted(action_counts.items())),
        "sha256": sha256(OUTPUT_PATH),
        "source_manifest": {
            "file_name": str(SOURCE_MANIFEST_PATH.relative_to(DATA_DIR)),
            "sha256": sha256(SOURCE_MANIFEST_PATH),
            "version": source_manifest["version"],
            "purpose": source_manifest["purpose"],
        },
        "sources": source_summary,
        "action_vocabulary": source_manifest["action_vocabulary"],
        "enable_thinking": source_manifest["enable_thinking"],
        "prompt_alignment": source_manifest["prompt_alignment"],
        "prompt_version": source_manifest["prompt_version"],
    }
    OUTPUT_MANIFEST_PATH.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Created {OUTPUT_PATH} with {len(rows)} rows.")
    print(f"Expert counts: {dict(expert_counts)}")
    print(f"SHA256: {output_manifest['sha256']}")


if __name__ == "__main__":
    main()
