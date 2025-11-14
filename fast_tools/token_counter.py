"""Utility to report token usage statistics for VQA question prompts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import tiktoken

DEFAULT_MODEL = "gpt-5-preview"
FALLBACK_ENCODING = "o200k_base"
IMAGE_PLACEHOLDER = "<image>"
DEFAULT_IMAGE_WIDTH = 1000
DEFAULT_IMAGE_HEIGHT = 500


def load_entries(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of entries, received {type(data).__name__}")

    return data


def resolve_encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


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


def count_tokens(
    entries: Iterable[dict],
    model_name: str,
    image_width: int,
    image_height: int,
):
    encoding = resolve_encoding(model_name)
    text_token_counts = []
    image_token_counts = []
    image_counts = []
    metadata = []
    tokens_per_image = compute_image_tokens(image_width, image_height)

    for entry in entries:
        question = entry.get("question")
        if not isinstance(question, str):
            continue

        cleaned_question = question.replace(IMAGE_PLACEHOLDER, " ").strip()
        placeholder_count = question.count(IMAGE_PLACEHOLDER)
        text_tokens = len(encoding.encode(cleaned_question))

        image_paths = list(normalize_file_names(entry.get("file_name")))
        reference_count = len(image_paths)
        num_images = max(placeholder_count, reference_count)

        text_token_counts.append(text_tokens)
        image_counts.append(num_images)
        image_token_counts.append(num_images * tokens_per_image)
        metadata.append(entry.get("idx") or entry.get("question_id") or "unknown")

    return {
        "encoding": encoding.name,
        "text_token_counts": text_token_counts,
        "image_token_counts": image_token_counts,
        "image_counts": image_counts,
        "metadata": metadata,
        "image_tokens_per_image": tokens_per_image,
        "image_width": image_width,
        "image_height": image_height,
    }


def summarize(counts: dict) -> dict:
    total_entries = len(counts["text_token_counts"])
    total_text_tokens = sum(counts["text_token_counts"])
    total_image_tokens = sum(counts["image_token_counts"])
    total_images = sum(counts["image_counts"])
    total_tokens = total_text_tokens + total_image_tokens
    max_tokens = max(counts["text_token_counts"], default=0)
    max_idx = (
        counts["metadata"][counts["text_token_counts"].index(max_tokens)]
        if max_tokens and total_entries
        else None
    )

    return {
        "entries": total_entries,
        "encoding": counts["encoding"],
        "total_text_tokens": total_text_tokens,
        "avg_text_tokens": (total_text_tokens / total_entries) if total_entries else 0.0,
        "max_text_tokens": max_tokens,
        "max_text_tokens_entry": max_idx,
        "total_image_tokens": total_image_tokens,
        "total_images": total_images,
        "avg_images_per_question": (total_images / total_entries) if total_entries else 0.0,
        "image_tokens_per_image": counts["image_tokens_per_image"],
        "image_width": counts["image_width"],
        "image_height": counts["image_height"],
        "avg_total_tokens_per_question": (total_tokens / total_entries) if total_entries else 0.0,
        "total_tokens": total_tokens,
    }


def print_report(summary: dict, dataset_path: Path, model_name: str) -> None:
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Encoding used: {summary['encoding']}")
    print(f"Questions processed: {summary['entries']}")
    print(f"Total text tokens: {summary['total_text_tokens']}")
    print(f"Average text tokens per question: {summary['avg_text_tokens']:.2f}")
    if summary["max_text_tokens_entry"] is not None:
        print(
            "Max text tokens: "
            f"{summary['max_text_tokens']} "
            f"(entry {summary['max_text_tokens_entry']})"
        )
    else:
        print(f"Max text tokens: {summary['max_text_tokens']}")
    print(
        "Tokens per image "
        f"({summary['image_width']}x{summary['image_height']}): "
        f"{summary['image_tokens_per_image']}"
    )
    print(f"Total image tokens: {summary['total_image_tokens']}")
    print(f"Total images referenced: {summary['total_images']}")
    print(f"Average images per question: {summary['avg_images_per_question']:.2f}")
    print(f"Total tokens (text + image): {summary['total_tokens']}")
    print(f"Average total tokens per question: {summary['avg_total_tokens_per_question']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count textual tokens for ChatGPT-style prompts and report image usage stats."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("output/test_run01_1K.json"),
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Image width in pixels used to estimate Vision tokens.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Image height in pixels used to estimate Vision tokens.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model name passed to tiktoken.encoding_for_model. "
            f"Falls back to '{FALLBACK_ENCODING}' if unavailable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.dataset)
    counts = count_tokens(entries, args.model, args.image_width, args.image_height)
    summary = summarize(counts)
    print_report(summary, args.dataset, args.model)


if __name__ == "__main__":
    main()
