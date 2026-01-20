"""Count total tokens for multiple VLMs on a TinyVQA-style dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import tiktoken

DEFAULT_DATASET = Path(
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/"
    "run_16_general/test_run_16_general_10K.json"
)
DEFAULT_MODELS = ["gpt-5.2", "gemini-3", "grok-fast"]
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
        return tiktoken.encoding_for_model(model_name), False
    except KeyError:
        pass
    except Exception:
        return None, True

    try:
        return tiktoken.get_encoding(FALLBACK_ENCODING), True
    except Exception:
        return None, True


def heuristic_token_count(text: str) -> int:
    # Rough approximation when tokenizers are unavailable offline.
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


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
) -> dict:
    encoding, used_fallback = resolve_encoding(model_name)
    tokens_per_image = compute_image_tokens(image_width, image_height)
    total_text_tokens = 0
    total_image_tokens = 0
    total_entries = 0
    image_count_hist = Counter()

    for entry in entries:
        question = entry.get("question")
        if not isinstance(question, str):
            continue

        cleaned_question = question.replace(IMAGE_PLACEHOLDER, " ").strip()
        placeholder_count = question.count(IMAGE_PLACEHOLDER)

        image_paths = list(normalize_file_names(entry.get("file_name")))
        reference_count = len(image_paths)
        num_images = max(placeholder_count, reference_count)

        if encoding is None:
            total_text_tokens += heuristic_token_count(cleaned_question)
        else:
            total_text_tokens += len(encoding.encode(cleaned_question))
        total_image_tokens += num_images * tokens_per_image
        total_entries += 1
        image_count_hist[num_images] += 1

    return {
        "model": model_name,
        "encoding": encoding.name if encoding is not None else "heuristic",
        "used_fallback": used_fallback,
        "entries": total_entries,
        "total_text_tokens": total_text_tokens,
        "total_image_tokens": total_image_tokens,
        "total_tokens": total_text_tokens + total_image_tokens,
        "image_tokens_per_image": tokens_per_image,
        "image_count_hist": dict(sorted(image_count_hist.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report total text + image tokens for multiple VLMs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of model names.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Image width in pixels used to estimate vision tokens.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Image height in pixels used to estimate vision tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.dataset)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Dataset: {args.dataset}")
    print(f"Entries processed: {len(entries)}")
    print(
        "Image tokens per image "
        f"({args.image_width}x{args.image_height}): "
        f"{compute_image_tokens(args.image_width, args.image_height)}"
    )
    print()

    for model_name in model_names:
        summary = count_tokens(entries, model_name, args.image_width, args.image_height)
        if summary["used_fallback"] and summary["encoding"] == "heuristic":
            fallback_note = " (heuristic tokens)"
        elif summary["used_fallback"]:
            fallback_note = " (fallback encoding)"
        else:
            fallback_note = ""
        print(f"Model: {summary['model']}{fallback_note}")
        print(f"  Encoding: {summary['encoding']}")
        print(f"  Total text tokens: {summary['total_text_tokens']}")
        print(f"  Total image tokens: {summary['total_image_tokens']}")
        print(f"  Total tokens: {summary['total_tokens']}")
        print("  Image count histogram (num_images: questions):")
        for num_images, count in summary["image_count_hist"].items():
            print(f"    {num_images}: {count}")
        print()


if __name__ == "__main__":
    main()
