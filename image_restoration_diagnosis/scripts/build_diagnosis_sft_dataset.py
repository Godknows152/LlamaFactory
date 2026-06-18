#!/usr/bin/env python3
"""Build a class-balanced LLaMA-Factory SFT dataset for degradation diagnosis.

The source dataset is expected to contain the following directories:

    fog_series/fog
    rain_series/rain
    snow_series/snow
    night_series/night

The generated samples use LLaMA-Factory's ShareGPT multimodal format. The
assistant target is kept as raw Hermes text instead of a ``function_call`` role
so that LLaMA-Factory does not convert it to a model-native tool-call format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

TRAIN_DATASET_NAME = "image_restoration_diagnosis_train"
TEST_DATASET_NAME = "image_restoration_diagnosis_test"
TRAIN_DATASET_FILE_NAME = "diagnosis_train.jsonl"
TEST_DATASET_FILE_NAME = "diagnosis_test.jsonl"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
ImageMode = Literal["reference", "symlink", "hardlink", "copy"]
EvidenceMode = Literal["class_template", "empty"]
PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ClassSpec:
    source_subdir: str
    degradation_type: str
    evidence_templates: tuple[tuple[str, ...], ...]


CLASS_SPECS = (
    ClassSpec(
        source_subdir="fog_series/fog",
        degradation_type="fog",
        evidence_templates=(
            ("global contrast is reduced", "distant regions appear washed out"),
            ("a veil-like haze covers the scene", "colors and edges look muted"),
            (
                "background details are obscured by haze",
                "the scene has a pale low-contrast appearance",
            ),
            (
                "atmospheric haze softens object boundaries",
                "remote structures lack clear detail",
            ),
            (
                "a grayish veil is visible across the scene",
                "depth cues are weakened by fog",
            ),
            (
                "foreground and background have compressed contrast",
                "fine textures fade with distance",
            ),
            (
                "the air appears dense and opaque",
                "scene visibility decreases toward the background",
            ),
            (
                "dark regions look lifted by scattered light",
                "distant edges blend into the haze",
            ),
            (
                "the image has a milky atmospheric cast",
                "background colors appear desaturated",
            ),
            (
                "haze reduces separation between scene layers",
                "faraway objects are difficult to distinguish",
            ),
            (
                "a broad fog layer lowers overall clarity",
                "object contours become faint at long range",
            ),
            (
                "the scene appears washed out rather than sharply exposed",
                "atmospheric scattering hides detail",
            ),
            (
                "visibility is limited by a uniform pale veil",
                "background contrast is notably weak",
            ),
            (
                "distant content merges with the bright atmosphere",
                "local details are softened by haze",
            ),
            (
                "the image shows depth-dependent fading",
                "remote areas lose color saturation",
            ),
            (
                "fog produces a smooth low-frequency veil",
                "high-frequency details are suppressed",
            ),
            (
                "scene radiance is dominated by atmospheric light",
                "true object colors appear diluted",
            ),
            (
                "the horizon and background are poorly defined",
                "haze creates a flat visual appearance",
            ),
            (
                "contrast attenuation increases with scene depth",
                "distant objects look increasingly pale",
            ),
            (
                "a translucent mist covers the view",
                "background textures are partially concealed",
            ),
        ),
    ),
    ClassSpec(
        source_subdir="rain_series/rain",
        degradation_type="rain",
        evidence_templates=(
            (
                "visible rain streaks cross the image",
                "scene details are blurred by rainfall",
            ),
            (
                "bright elongated streaks appear in the scene",
                "wet-weather artifacts reduce local clarity",
            ),
            (
                "rain patterns partially occlude objects",
                "streak-like precipitation disturbs local contrast",
            ),
            (
                "thin diagonal rain traces are visible",
                "repeated streaks interfere with object textures",
            ),
            (
                "rainfall creates linear bright artifacts",
                "background detail is partially masked",
            ),
            (
                "the scene contains dense precipitation streaks",
                "local edges are disrupted by rain",
            ),
            (
                "water-related marks cover parts of the image",
                "fine scene structure is harder to recognize",
            ),
            (
                "multiple narrow streaks follow a similar direction",
                "rain artifacts overlay the underlying content",
            ),
            (
                "wet-weather veiling lowers scene clarity",
                "rain traces are distributed across the frame",
            ),
            (
                "elongated droplets create transient bright lines",
                "objects behind the rainfall appear less distinct",
            ),
            (
                "rain streaks overlap high-frequency textures",
                "the precipitation introduces visual clutter",
            ),
            (
                "the image shows repeated linear weather artifacts",
                "rain reduces clean separation of object boundaries",
            ),
            (
                "falling precipitation is visible as slanted lines",
                "background regions are obscured by rainfall",
            ),
            (
                "rain produces directional streak patterns",
                "small details are hidden beneath the streaks",
            ),
            (
                "water streaks interrupt otherwise continuous surfaces",
                "the scene has a noticeable rainy-weather veil",
            ),
            (
                "bright and dark rain traces cross scene content",
                "rainfall lowers local visibility",
            ),
            (
                "precipitation artifacts extend across several regions",
                "object appearance is contaminated by rain streaks",
            ),
            (
                "the frame contains numerous narrow rainfall marks",
                "rain introduces blur and partial occlusion",
            ),
            (
                "directional rain structures are prominent",
                "underlying textures lose clarity",
            ),
            (
                "weather-induced streaks are spread across the view",
                "scene details appear degraded by active rainfall",
            ),
        ),
    ),
    ClassSpec(
        source_subdir="snow_series/snow",
        degradation_type="snow",
        evidence_templates=(
            (
                "white snow particles cover parts of the image",
                "snowy veiling obscures scene details",
            ),
            (
                "bright flakes and clusters are visible",
                "snow-related occlusion reduces local clarity",
            ),
            (
                "scattered white particles appear across the scene",
                "dense snow hides background detail",
            ),
            (
                "irregular white flakes overlap scene objects",
                "snow particles conceal local textures",
            ),
            (
                "the image contains many bright particle-like occlusions",
                "snow reduces visibility in multiple regions",
            ),
            (
                "snowflakes of different sizes are distributed across the frame",
                "foreground particles block background content",
            ),
            (
                "white translucent blobs interrupt object boundaries",
                "falling snow creates visual clutter",
            ),
            (
                "dense bright flakes dominate portions of the scene",
                "underlying details are partially hidden",
            ),
            (
                "snow particles produce scattered high-intensity spots",
                "the weather layer masks fine structure",
            ),
            (
                "large and small flakes overlap the background",
                "snowfall weakens scene clarity",
            ),
            (
                "the scene is covered by irregular snowy artifacts",
                "objects are occluded by bright precipitation",
            ),
            (
                "clustered snowflakes create a nonuniform veil",
                "background textures are difficult to see",
            ),
            (
                "white particle patterns appear at varying depths",
                "snowfall obscures both near and distant content",
            ),
            (
                "bright snow traces are spread throughout the view",
                "local contrast is disturbed by particle occlusion",
            ),
            (
                "falling flakes form a dense weather layer",
                "scene elements behind the snow are less distinct",
            ),
            (
                "snow introduces numerous irregular bright regions",
                "fine edges are hidden by overlapping particles",
            ),
            (
                "the image shows widespread flake-shaped occlusions",
                "snowy artifacts interfere with object recognition",
            ),
            (
                "particle density is high across the scene",
                "bright flakes conceal portions of the background",
            ),
            (
                "snowfall creates scattered opaque and translucent marks",
                "visual details are blocked by the weather layer",
            ),
            (
                "multiple white flakes overlap important scene regions",
                "snow-related occlusion lowers overall readability",
            ),
        ),
    ),
    ClassSpec(
        source_subdir="night_series/night",
        degradation_type="low_light",
        evidence_templates=(
            (
                "the scene is strongly underexposed",
                "shadow regions contain little visible detail",
            ),
            ("overall illumination is very low", "dark areas dominate the image"),
            (
                "brightness is insufficient across the scene",
                "objects are difficult to distinguish in shadows",
            ),
            (
                "most pixels have low luminance",
                "fine structures disappear in dark regions",
            ),
            (
                "the image has a compressed dark tonal range",
                "shadow details are poorly resolved",
            ),
            (
                "ambient lighting is weak throughout the scene",
                "object boundaries blend into darkness",
            ),
            (
                "large regions are nearly black",
                "only a few illuminated areas remain clearly visible",
            ),
            ("the scene lacks adequate exposure", "dark surfaces show limited texture"),
            (
                "low illumination suppresses color and detail",
                "background objects are hard to recognize",
            ),
            (
                "the image appears dim with deep shadows",
                "important content is hidden by insufficient light",
            ),
            (
                "brightness is concentrated in a few small areas",
                "the remaining scene is severely dark",
            ),
            (
                "weak lighting causes poor visibility",
                "low-luminance regions lose structural information",
            ),
            (
                "the scene has a night-time underexposed appearance",
                "darkness masks fine object features",
            ),
            (
                "most scene content falls below a useful brightness level",
                "shadow contrast is difficult to perceive",
            ),
            (
                "the exposure level is too low for clear observation",
                "textures vanish in the darkest regions",
            ),
            (
                "objects are visible only as dark silhouettes",
                "illumination is inadequate across the frame",
            ),
            (
                "the image contains broad low-intensity regions",
                "details are buried in shadows",
            ),
            (
                "limited available light produces a very dim scene",
                "colors appear muted in dark areas",
            ),
            (
                "the scene is dominated by deep black and near-black tones",
                "low-light noise and lost detail reduce clarity",
            ),
            (
                "insufficient lighting makes the background indistinct",
                "shadowed content lacks readable features",
            ),
        ),
    ),
)

for class_spec in CLASS_SPECS:
    if len(class_spec.evidence_templates) != 20:
        raise ValueError(
            f"{class_spec.degradation_type} must define exactly 20 evidence templates"
        )
    if len(set(class_spec.evidence_templates)) != 20:
        raise ValueError(
            f"{class_spec.degradation_type} evidence templates must be unique"
        )


DIAGNOSIS_TOOL_SCHEMA = {
    "name": "diagnose_degradation",
    "description": "Classify the single primary image degradation and record visible evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "primary_type": {
                "type": "string",
                "enum": ["fog", "snow", "rain", "low_light"],
            },
            "visual_evidence": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["primary_type", "visual_evidence"],
        "additionalProperties": False,
    },
}


def _build_diagnosis_system_prompt() -> str:
    serialized_schema = json.dumps(
        DIAGNOSIS_TOOL_SCHEMA, ensure_ascii=False, separators=(",", ":")
    )
    example_payload = {
        "name": "diagnose_degradation",
        "arguments": {
            "primary_type": "fog",
            "visual_evidence": [
                "global contrast is reduced",
                "distant regions appear washed out",
            ],
        },
    }
    serialized_example = json.dumps(
        example_payload, ensure_ascii=False, separators=(",", ":")
    )
    return f"""Prompt version: diagnosis-hermes-v1

You are the degradation diagnosis agent. Inspect the input image and identify exactly one
primary degradation from fog, snow, rain, or low_light. Base the decision only on visible
image evidence. Do not use a filename, path, hidden label, or tool-name heuristic.

You are provided with one function signature inside <tools></tools> tags:
<tools>
{serialized_schema}
</tools>

Return exactly one Hermes tool call and no other actionable output:
<tool_call>
{serialized_example}
</tool_call>

The example category and evidence illustrate syntax only. Determine the category from the
actual image.

Rules:
1. Emit exactly one <tool_call></tool_call> block.
2. The name field must be exactly diagnose_degradation.
3. The arguments object must contain exactly primary_type and visual_evidence.
4. primary_type must be exactly one of fog, snow, rain, low_light.
5. visual_evidence must be a JSON array of concise observations visible in the image.
6. Do not output confidence or route_to; the controller owns the fixed category-to-expert mapping.
7. Do not wrap the tool call in a Markdown code fence or emit bare JSON outside the tags.
8. Do not call a restoration tool and do not propose a restoration sequence."""


DIAGNOSIS_SYSTEM_PROMPT = _build_diagnosis_system_prompt()
USER_PROMPT = "<image>\nDiagnose this image using exactly one diagnose_degradation Hermes tool call."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path(
            "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/train/images"
        ),
        help="Training root containing the four degradation-series directories.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path(
            "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/test/images"
        ),
        help="Test root containing the four degradation-series directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "data",
        help="Directory for JSONL, dataset_info.json, manifest, and staged images.",
    )
    parser.add_argument(
        "--train-samples-per-class",
        type=int,
        default=1000,
        help="Number of training images selected independently from each class.",
    )
    parser.add_argument(
        "--test-samples-per-class",
        type=int,
        default=200,
        help="Number of test images selected independently from each class.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260613, help="Deterministic sampling seed."
    )
    parser.add_argument(
        "--image-mode",
        choices=("reference", "symlink", "hardlink", "copy"),
        default="symlink",
        help="How selected images are exposed below the output directory.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=("class_template", "empty"),
        default="class_template",
        help="Use weak class-level evidence templates or emit an empty evidence list.",
    )
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Do not ask Pillow to verify selected image files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory.",
    )
    return parser.parse_args()


def _class_seed(seed: int, degradation_type: str) -> int:
    digest = hashlib.sha256(f"{seed}:{degradation_type}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _iter_image_files(directory: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def select_images(
    source_dir: Path,
    *,
    count: int,
    seed: int,
    verify_images: bool,
) -> tuple[list[Path], list[Path], int]:
    """Select valid images deterministically and return selection diagnostics."""
    candidates = list(_iter_image_files(source_dir))
    random.Random(seed).shuffle(candidates)
    selected: list[Path] = []
    invalid: list[Path] = []
    for candidate in candidates:
        if verify_images and not _is_valid_image(candidate):
            invalid.append(candidate)
            continue
        selected.append(candidate)
        if len(selected) == count:
            break

    if len(selected) < count:
        raise ValueError(
            f"{source_dir} contains only {len(selected)} usable images; "
            f"{count} were requested ({len(invalid)} invalid images skipped)."
        )
    return selected, invalid, len(candidates)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _stage_image(source: Path, destination: Path, mode: ImageMode) -> str:
    if mode == "reference":
        return str(source.resolve())

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")
    return destination.as_posix()


def _visual_evidence(
    spec: ClassSpec, sample_index: int, evidence_mode: EvidenceMode
) -> tuple[int | None, list[str]]:
    if evidence_mode == "empty":
        return None, []
    template_index = sample_index % len(spec.evidence_templates)
    return template_index, list(spec.evidence_templates[template_index])


def _assistant_target(degradation_type: str, visual_evidence: list[str]) -> str:
    payload = {
        "name": "diagnose_degradation",
        "arguments": {
            "primary_type": degradation_type,
            "visual_evidence": visual_evidence,
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"<tool_call>\n{serialized}\n</tool_call>"


def _dataset_info() -> dict[str, object]:
    def dataset_entry(file_name: str) -> dict[str, object]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "system": "system",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

    return {
        TRAIN_DATASET_NAME: dataset_entry(TRAIN_DATASET_FILE_NAME),
        TEST_DATASET_NAME: dataset_entry(TEST_DATASET_FILE_NAME),
    }


def _build_split(
    *,
    split: str,
    input_root: Path,
    output_dir: Path,
    samples_per_class: int,
    seed: int,
    image_mode: ImageMode,
    evidence_mode: EvidenceMode,
    verify_images: bool,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    if samples_per_class <= 0:
        raise ValueError(f"{split} samples per class must be positive.")
    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    records: list[dict[str, object]] = []
    class_summary: dict[str, dict[str, object]] = {}

    for spec in CLASS_SPECS:
        source_dir = input_root / spec.source_subdir
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Required class directory does not exist: {source_dir}"
            )
        selected, invalid, candidate_count = select_images(
            source_dir,
            count=samples_per_class,
            seed=_class_seed(seed, f"{split}:{spec.degradation_type}"),
            verify_images=verify_images,
        )
        for index, source_path in enumerate(selected):
            staged_path = (
                output_dir / "images" / split / spec.degradation_type / source_path.name
            )
            image_reference = _stage_image(source_path, staged_path, image_mode)
            if image_mode != "reference":
                image_reference = staged_path.relative_to(output_dir).as_posix()
            evidence_template_index, evidence = _visual_evidence(
                spec, index, evidence_mode
            )
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": USER_PROMPT},
                        {
                            "role": "assistant",
                            "content": _assistant_target(
                                spec.degradation_type, evidence
                            ),
                        },
                    ],
                    "system": DIAGNOSIS_SYSTEM_PROMPT,
                    "images": [image_reference],
                    "sample_id": f"{split}-{spec.degradation_type}-{index:04d}-{source_path.stem}",
                    "split": split,
                    "degradation_type": spec.degradation_type,
                    "source_image": str(source_path),
                    "evidence_template_index": evidence_template_index,
                    "annotation_provenance": (
                        "folder_label_and_balanced_class_level_evidence_template"
                        if evidence_mode == "class_template"
                        else "folder_label_only"
                    ),
                }
            )
        class_summary[spec.degradation_type] = {
            "source_dir": str(source_dir),
            "candidate_count": candidate_count,
            "selected_count": len(selected),
            "invalid_skipped": len(invalid),
            "evidence_template_count": len(spec.evidence_templates)
            if evidence_mode == "class_template"
            else 0,
            "evidence_template_usage": (
                {
                    str(template_index): sum(
                        1
                        for sample_index in range(len(selected))
                        if sample_index % len(spec.evidence_templates) == template_index
                    )
                    for template_index in range(len(spec.evidence_templates))
                }
                if evidence_mode == "class_template"
                else {}
            ),
        }

    random.Random(_class_seed(seed, split)).shuffle(records)
    return records, class_summary


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(
    *,
    train_root: Path,
    test_root: Path,
    output_dir: Path,
    train_samples_per_class: int,
    test_samples_per_class: int,
    seed: int,
    image_mode: ImageMode,
    evidence_mode: EvidenceMode,
    verify_images: bool,
    overwrite: bool,
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, overwrite)
    train_records, train_summary = _build_split(
        split="train",
        input_root=train_root,
        output_dir=output_dir,
        samples_per_class=train_samples_per_class,
        seed=seed,
        image_mode=image_mode,
        evidence_mode=evidence_mode,
        verify_images=verify_images,
    )
    test_records, test_summary = _build_split(
        split="test",
        input_root=test_root,
        output_dir=output_dir,
        samples_per_class=test_samples_per_class,
        seed=seed,
        image_mode=image_mode,
        evidence_mode=evidence_mode,
        verify_images=verify_images,
    )
    train_path = output_dir / TRAIN_DATASET_FILE_NAME
    test_path = output_dir / TEST_DATASET_FILE_NAME
    train_sha256 = _write_jsonl(train_path, train_records)
    test_sha256 = _write_jsonl(test_path, test_records)

    dataset_info_path = output_dir / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(_dataset_info(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "datasets": {
            "train": {
                "dataset_name": TRAIN_DATASET_NAME,
                "dataset_file": TRAIN_DATASET_FILE_NAME,
                "dataset_sha256": train_sha256,
                "input_root": str(train_root.expanduser().resolve()),
                "samples_per_class": train_samples_per_class,
                "total_samples": len(train_records),
                "classes": train_summary,
            },
            "test": {
                "dataset_name": TEST_DATASET_NAME,
                "dataset_file": TEST_DATASET_FILE_NAME,
                "dataset_sha256": test_sha256,
                "input_root": str(test_root.expanduser().resolve()),
                "samples_per_class": test_samples_per_class,
                "total_samples": len(test_records),
                "classes": test_summary,
            },
        },
        "output_dir": str(output_dir),
        "seed": seed,
        "image_mode": image_mode,
        "verify_images": verify_images,
        "evidence_mode": evidence_mode,
        "evidence_warning": (
            "visual_evidence is weak supervision selected from class-level templates; "
            "it is not a per-image human annotation."
            if evidence_mode == "class_template"
            else "visual_evidence is intentionally empty."
        ),
        "llamafactory": {
            "dataset": TRAIN_DATASET_NAME,
            "eval_dataset": TEST_DATASET_NAME,
            "dataset_dir": str(output_dir),
            "media_dir": str(output_dir),
            "template": "qwen3_5_nothink",
            "enable_thinking": False,
            "resize_vocab": False,
            "tool_format": None,
            "assistant_target_format": "raw Hermes text",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        train_root=args.train_root,
        test_root=args.test_root,
        output_dir=args.output_dir,
        train_samples_per_class=args.train_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        seed=args.seed,
        image_mode=args.image_mode,
        evidence_mode=args.evidence_mode,
        verify_images=not args.skip_image_verification,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        "\nLLaMA-Factory dataset arguments:\n"
        f"  dataset: {TRAIN_DATASET_NAME}\n"
        f"  eval_dataset: {TEST_DATASET_NAME}\n"
        f"  dataset_dir: {Path(manifest['output_dir'])}\n"
        f"  media_dir: {Path(manifest['output_dir'])}\n"
        "  template: qwen3_5_nothink\n"
        "  enable_thinking: false\n"
        "  resize_vocab: false\n"
        "  # Do not set tool_format: the assistant target already contains raw Hermes tags."
    )


if __name__ == "__main__":
    main()
