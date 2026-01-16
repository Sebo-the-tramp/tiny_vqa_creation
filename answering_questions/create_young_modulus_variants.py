#!/usr/bin/env python3
"""Create metric/scientific label variants for each VQA question."""

from __future__ import annotations

import argparse
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, List


CHOICE_PATTERN = re.compile(r"^(?P<letter>[A-Z])\.\s+(?P<label>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create metric/scientific variants by rewriting the answer labels in a "
            "VQA-style JSON file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the augmented JSON.",
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Do not include the original records in the output.",
    )
    return parser.parse_args()


def to_scientific(label: str) -> str:
    value_str = label.replace(" Pa", "")
    value = float(value_str)
    if value == 0:
        return "0 x 10^0 Pa"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f}x10^{exponent} Pa"


def to_metric(label: str) -> str:
    value_str = label.replace(" Pa", "")
    value = float(value_str)
    if value >= 1e9:
        new_value = value / 1e9
        return f"{new_value:.2f} GPa"
    if value >= 1e6:
        new_value = value / 1e6
        return f"{new_value:.2f} MPa"
    if value >= 1e3:
        new_value = value / 1e3
        return f"{new_value:.2f} kPa"
    return f"{value:.2f} Pa"


def rewrite_choices(lines: Iterable[str], formatter) -> List[str]:
    updated: List[str] = []
    for line in lines:
        match = CHOICE_PATTERN.match(line)
        if not match:
            updated.append(line)
            continue
        letter = match.group("letter")
        label = match.group("label")
        new_label = formatter(label)
        updated.append(f"{letter}. {new_label}")
    return updated


def make_variant(record: dict[str, Any], suffix: str, formatter) -> dict[str, Any]:
    new_record = deepcopy(record)
    question = new_record.get("question")
    if isinstance(question, str):
        lines = question.splitlines()
        lines = rewrite_choices(lines, formatter)
        new_record["question"] = "\n".join(lines)
    if "question_id" in new_record and isinstance(new_record["question_id"], str):
        new_record["question_id"] = f"{new_record['question_id']}{suffix}"
    if "idx" in new_record and isinstance(new_record["idx"], str):
        new_record["idx"] = f"{new_record['idx']}{suffix}"
    return new_record


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise SystemExit("Expected the input JSON to be a list of records.")

    output: List[dict[str, Any]] = []
    for record in data:
        if not isinstance(record, dict):
            raise SystemExit("Each record in the input JSON must be an object.")
        if not args.no_original:
            output.append(record)
        output.append(make_variant(record, "_scientific", to_scientific))
        output.append(make_variant(record, "_metric", to_metric))

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
