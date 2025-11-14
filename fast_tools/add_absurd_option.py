#!/usr/bin/env python
"""Append a deliberately absurd multiple-choice option to every question entry.

The script looks for question-like dictionaries inside the JSON structure,
adds a new option with a nonsensical physics-defying answer, and updates the
question text so the new option appears at the end of the prompt.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_SUBJECTS = [
    "asdohasoidja",
    "oasdjoaishdasda",
    "asodijaoisjd",
]

# _SUBJECTS = [
#     "time-traveling toaster",
#     "quantum unicorn",
#     "banana-powered warp drive",
#     "gravity-resistant jellyfish",
#     "cosmic rubber duck",
#     "anti-gravity marshmallow",
#     "sentient equation scribble",
#     "thermodynamic hamster",
#     "fourth-dimensional skateboard",
#     "plasma-fueled cupcake",
# ]

# _VERBS = [
#     "counteracts",
#     "moonwalks across",
#     "reverses",
#     "teleports past",
#     "evaporates",
#     "dissolves",
#     "outwits",
#     "ignites",
#     "juggles",
#     "serenades",
# ]

# _OBJECTS = [
#     "Newton's third apple",
#     "Schrödinger's trampoline",
#     "the conservation of socks",
#     "entropy flavored ice cream",
#     "the theory of everything bagels",
#     "Heisenberg's rubber chicken",
#     "Planck-scale bowling balls",
#     "relativistic lemonade",
#     "dark-matter bubble bath",
#     "plasma-filled snow globe",
# ]

# _ENDINGS = [
#     "during a cosmic tea party",
#     "inside a black hole made of glitter",
#     "while orbiting a confused penguin",
#     "at absolute zero karaoke night",
#     "on a frictionless rollercoaster",
#     "while balancing on a photon surfboard",
#     "during intergalactic taco hour",
#     "inside a wormhole full of waffles",
#     "at a zero-gravity pillow fight",
#     "inside Schrödinger's picnic basket",
# ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the JSON file to augment (e.g. test.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="./test_absurd.json",
        help="Optional path for the augmented JSON. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def generate_nonsense() -> str:
    """Create a physics-defying nonsense option."""
    return " ".join(
        [
            random.choice(_SUBJECTS),
            # random.choice(_VERBS),
            # random.choice(_OBJECTS),
            # random.choice(_ENDINGS),
        ]
    )


def determine_next_label(existing: Iterable[str]) -> str:
    """Return the next available uppercase option label."""
    letters = [
        key
        for key in existing
        if isinstance(key, str) and len(key) == 1 and key.isalpha() and key.isupper()
    ]
    if not letters:
        return "A"
    max_letter = max(letters)
    idx = _LETTERS.find(max_letter)
    if idx == -1 or idx + 1 >= len(_LETTERS):
        return f"Option {len(letters) + 1}"
    return _LETTERS[idx + 1]


def determine_label_from_list(option_list: List[Any]) -> str:
    length = len(option_list)
    if length < len(_LETTERS):
        return _LETTERS[length]
    return f"Option {length + 1}"


def determine_label_from_question(question: Optional[str]) -> str:
    if not question:
        return "A"
    letters = [
        part.strip().split(".", 1)[0]
        for part in question.split("\n")
        if part.strip().startswith(tuple(f"{ch}." for ch in _LETTERS))
    ]
    if not letters:
        return "A"
    return determine_next_label(letters)


def append_to_question(question: Optional[str], label: str, option: str) -> str:
    base = (question or "").rstrip()
    if base:
        return f"{base}\n{label}. {option}\n"
    return f"{label}. {option}\n"


def update_entry(entry: MutableMapping[str, Any]) -> None:
    options = entry.get("options")
    label: str

    if isinstance(options, dict):
        label = determine_next_label(options.keys())
        if label in options:
            return
        options[label] = generate_nonsense()
    elif isinstance(options, list):
        label = determine_label_from_list(options)
        options.append(generate_nonsense())
    else:
        label = determine_label_from_question(entry.get("question"))
        entry["options"] = {label: generate_nonsense()}

    new_option = (
        entry["options"][-1] if isinstance(entry["options"], list) else entry["options"][label]
    )
    entry["question"] = append_to_question(entry.get("question"), label, new_option)

    option_files = entry.get("option_files")
    if isinstance(option_files, dict) and label not in option_files:
        option_files[label] = None


def process_json(obj: Any) -> None:
    if isinstance(obj, list):
        for item in obj:
            process_json(item)
    elif isinstance(obj, dict):
        if "question" in obj:
            update_entry(obj)
        for value in obj.values():
            process_json(value)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    with args.input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    process_json(data)

    output_path = args.output or args.input_path
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


if __name__ == "__main__":
    main()

# python utils/add_absurd_option.py ./output/test.json -o ./output/test_absurd.json