#!/usr/bin/env python3
"""Count text + image tokens and estimate costs for run_28_general.

Hardcoded for:
- Dataset: output/run_28_general/test_run_28_general.json
- Image size: 562x1000
- Models: GPT-5.2, Gemini3-Pro, and Gemini3-Flash (input-only costs)
- Requires tiktoken encodings (fails fast if unavailable)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
import tiktoken

DATASET_PATH = Path(
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/"
    "output/run_28_general/test_run_28_general.json"
)

IMAGE_WIDTH = 562
IMAGE_HEIGHT = 1000
IMAGE_PLACEHOLDER = "<image>"

MODELS = [
    {
        "label": "chatgpt5.2",
        "tokenizer": "gpt-5.2",
        "encoding": "o200k_base",
        "type": "gpt",
        "input_price_per_million": 1.75,
        "image_tokens_per_image":  1536,
    },
    {
        "label": "GEMINI_3",
        "tokenizer": "gemini-3",
        "encoding": "o200k_base",
        "type": "gemini",
        "input_price_per_million": 2.00,
        "input_price_per_million_high": 4.00,
        "tier_threshold": 200_000,
        "image_tokens_per_image": 1548,
    },
    {
        "label": "GEMINI_3_FLASH",
        "tokenizer": "gemini-3-flash",
        "encoding": "o200k_base",
        "type": "gemini",
        "input_price_per_million": 0.25,
        "input_price_per_million_high": 0.25,
        "tier_threshold": 200_000,
        "image_tokens_per_image": 1548,
    },
]


def load_entries(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of entries, received {type(data).__name__}")
    return data


def resolve_encoding(model_cfg: dict) -> tiktoken.Encoding:
    encoding_name = model_cfg.get("encoding")
    if encoding_name:
        return tiktoken.get_encoding(encoding_name)
    return tiktoken.encoding_for_model(model_cfg["tokenizer"])


def normalize_file_names(file_name_field) -> Iterable[str]:
    if file_name_field is None:
        return []
    if isinstance(file_name_field, str):
        return [file_name_field]
    if isinstance(file_name_field, Iterable):
        return [str(path) for path in file_name_field]
    return []


def compute_image_tokens(width: int, height: int) -> int:
    return math.ceil(width / 32) * math.ceil(height / 32)


def count_for_model(entries: Iterable[dict], model_cfg: dict) -> dict:
    try:
        encoding = resolve_encoding(model_cfg)
    except Exception as exc:
        raise RuntimeError(
            "tiktoken encoding unavailable. Install the encoding files locally or "
            f"update the model encoding mapping. Model={model_cfg['label']} "
            f"tokenizer={model_cfg['tokenizer']} encoding={model_cfg.get('encoding')}. "
            f"Error: {exc}"
        ) from exc
    if model_cfg.get("image_tokens_per_image") is None:
        tokens_per_image = compute_image_tokens(IMAGE_WIDTH, IMAGE_HEIGHT)
    else:
        tokens_per_image = model_cfg["image_tokens_per_image"]
    total_text_tokens = 0
    total_image_tokens = 0
    total_entries = 0
    total_images = 0
    total_tokens = 0
    max_total_tokens = 0
    image_count_hist = Counter()

    # Cost tracking
    total_cost = 0.0
    tier_counts = Counter()

    for entry in entries:
        question = entry.get("question")
        if not isinstance(question, str):
            continue

        cleaned_question = question.replace(IMAGE_PLACEHOLDER, " ").strip()

        image_paths = list(normalize_file_names(entry.get("file_name")))
        num_images = len(image_paths)

        text_tokens = len(encoding.encode(cleaned_question))

        image_tokens = num_images * tokens_per_image
        entry_total_tokens = text_tokens + image_tokens

        total_text_tokens += text_tokens
        total_image_tokens += image_tokens
        total_tokens += entry_total_tokens
        total_entries += 1
        total_images += num_images
        max_total_tokens = max(max_total_tokens, entry_total_tokens)
        image_count_hist[num_images] += 1

        if model_cfg["type"] == "gemini":
            if entry_total_tokens > model_cfg["tier_threshold"]:
                rate = model_cfg["input_price_per_million_high"]
                tier_counts[">200k"] += 1
            else:
                rate = model_cfg["input_price_per_million"]
                tier_counts["<=200k"] += 1
            total_cost += entry_total_tokens * rate / 1_000_000

    if model_cfg["type"] == "gpt":
        total_cost = total_tokens * model_cfg["input_price_per_million"] / 1_000_000

    return {
        "model": model_cfg["label"],
        "tokenizer": model_cfg["tokenizer"],
        "encoding": encoding.name,
        "entries": total_entries,
        "total_text_tokens": total_text_tokens,
        "total_image_tokens": total_image_tokens,
        "total_tokens": total_tokens,
        "max_total_tokens": max_total_tokens,
        "total_images": total_images,
        "image_tokens_per_image": tokens_per_image,
        "image_count_hist": dict(sorted(image_count_hist.items())),
        "total_cost": total_cost,
        "batched_cost": total_cost * 0.5,
        "tier_counts": dict(tier_counts),
    }


def print_report(summary: dict) -> None:
    print(f"Model: {summary['model']} ({summary['tokenizer']})")
    print(f"  Encoding: {summary['encoding']}")
    print(f"  Total text tokens: {summary['total_text_tokens']}")
    print(f"  Total image tokens: {summary['total_image_tokens']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Max total tokens per question: {summary['max_total_tokens']}")
    print(f"  Total images referenced: {summary['total_images']}")
    print("  Image count histogram (num_images: questions):")
    for num_images, count in summary["image_count_hist"].items():
        print(f"    {num_images}: {count}")

    if summary["tier_counts"]:
        print("  Gemini price tiers (questions):")
        for tier, count in summary["tier_counts"].items():
            print(f"    {tier}: {count}")

    print(f"  Estimated input cost: ${summary['total_cost']:.6f}")
    print(f"  Batched cost (50%): ${summary['batched_cost']:.6f}")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(str(DATASET_PATH))

    entries = load_entries(DATASET_PATH)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Entries processed: {len(entries)}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print()

    for model_cfg in MODELS:
        summary = count_for_model(entries, model_cfg)
        print_report(summary)
        print()


if __name__ == "__main__":
    main()
