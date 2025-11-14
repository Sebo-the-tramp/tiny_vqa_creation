#!/usr/bin/env python3
"""
Populate VQA question categories by matching questions to template prompts.

The script reads questions from ``../output_single/test.json`` and infers the
corresponding template entry from ``../simple_vqa.json`` by removing object
names (e.g. ``LEGO_City_Advent_Calendar``) and looking for the template string
with placeholder tokens (e.g. ``<OBJECT>``). The resulting category key from
``simple_vqa.json`` is appended to each sample under the ``\"category\"`` field.

By default, the updated annotations are written to
``../output_single/test_with_category.json`` so the original file remains
untouched. Use ``--overwrite`` to update the input file in place.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Matches object names that use the canonical underscore-separated format.
OBJECT_TOKEN = re.compile(r"\b(?!<object>)[a-z0-9]+(?:_[a-z0-9-]+)+\b", re.IGNORECASE)
OPTION_LINE = re.compile(r"^[A-E]\.\s*")


def normalise_template(question: str) -> str:
    """Return a normalised template string with unified placeholders."""
    text = " ".join(question.lower().split())
    text = re.sub(r"<object_\d>", "<object>", text)
    return text


def load_templates(template_path: Path) -> Dict[str, Tuple[str, str, dict]]:
    """Load and flatten the template map from ``simple_vqa.json``."""
    with template_path.open("r", encoding="utf-8") as handle:
        tree = json.load(handle)

    flat: Dict[str, Tuple[str, str, dict]] = {}
    for domain, entries in tree.items():
        for key, payload in entries.items():
            normalised = normalise_template(payload["question"])
            flat[normalised] = (domain, key, payload)
    return flat


def normalise_question(raw: str) -> str:
    """Strip boilerplate (images/options) and replace object names."""
    lines: List[str] = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped or "<image>" in stripped:
            continue
        if OPTION_LINE.match(stripped):
            continue
        lines.append(line.replace("<image>", " "))

    text = " ".join(lines)
    text = OBJECT_TOKEN.sub("<OBJECT>", text)
    text = " ".join(text.lower().split())
    return text


def match_template(question: str, templates: Dict[str, Tuple[str, str, dict]]) -> Tuple[str, str, dict]:
    """Find the template whose normalised text is contained in the question."""
    normalised = normalise_question(question)
    for template, meta in templates.items():
        if template in normalised:
            return meta
    raise ValueError(f"Could not match question: {question!r}")


def process_samples(
    samples: Iterable[dict],
    templates: Dict[str, Tuple[str, str, dict]],
) -> List[dict]:
    """Attach a ``category`` field to each question sample."""
    processed: List[dict] = []
    for sample in samples:
        domain, key, payload = match_template(sample["question"], templates)
        enriched = dict(sample)
        # enriched["category"] = key
        enriched["category"] = domain
        enriched["sub_category"] = payload.get("sub_category")
        processed.append(enriched)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("..") / "output_multi" / "test_og.json",
        help="Path to the JSON file with exported questions.",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("..") / "simple_vqa.json",
        help="Path to the template definition JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("..") / "output_single" / "test_with_category.json",
        help="Output path for the enriched annotations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file instead of writing a sibling file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    templates = load_templates(args.templates)

    with args.input.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)

    updated = process_samples(samples, templates)

    target_path = args.input if args.overwrite else args.output
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(updated, handle, indent=4)
        handle.write("\n")

    unmatched_count = sum(1 for sample in updated if "category" not in sample)
    print(f"Wrote {len(updated)} samples to {target_path}")
    if unmatched_count:
        print(f"Warning: {unmatched_count} samples missing category tags")


if __name__ == "__main__":
    main()
