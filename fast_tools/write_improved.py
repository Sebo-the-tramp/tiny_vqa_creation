#!/usr/bin/env python3
import json
import re
from pathlib import Path

from check_regex import (
    ANSWER_KEY,
    FIRST_REGEX,
    INPUT_JSON_PATH,
    SAFE_BEST_REGEX,
    VALID_CHOICES,
)

OUTPUT_PATH = Path("improved.txt")


def extract_all_choices(answer: str, pattern: re.Pattern[str]) -> list[tuple[str, int, int]]:
    hits: list[tuple[str, int, int]] = []
    for match in pattern.finditer(answer):
        for group_index, group_value in enumerate(match.groups(), start=1):
            if group_value is None:
                continue
            choice = group_value.upper()
            if choice in VALID_CHOICES:
                start, end = match.span(group_index)
                hits.append((choice, start, end))
                break
    return hits


def highlight_all_hits(answer: str, hits: list[tuple[str, int, int]]) -> str:
    ordered_hits = sorted(hits, key=lambda x: x[1])
    chunks: list[str] = []
    cursor = 0
    for choice, start, end in ordered_hits:
        if start < cursor:
            continue
        chunks.append(answer[cursor:start])
        chunks.append(f"-> {choice} <-")
        cursor = end
    chunks.append(answer[cursor:])
    return "".join(chunks)


def format_hit_list(hits: list[tuple[str, int, int]]) -> str:
    return ", ".join(f"{choice}@{start}" for choice, start, _ in hits)


def main() -> None:
    rows = json.loads(INPUT_JSON_PATH.read_text(encoding="utf-8"))

    first_pattern = re.compile(FIRST_REGEX)
    improved_pattern = re.compile(SAFE_BEST_REGEX, re.IGNORECASE | re.MULTILINE)

    selected: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        answer = row[ANSWER_KEY]
        first_hits = extract_all_choices(answer, first_pattern)
        improved_hits = extract_all_choices(answer, improved_pattern)
        if not improved_hits or first_hits:
            continue

        unique_letters = sorted({choice for choice, _, _ in improved_hits})
        flags: list[str] = []
        if len(improved_hits) > 1:
            flags.append("MULTI_HIT")
        if len(unique_letters) > 1:
            flags.append("MULTI_LETTER")

        flag_text = "NONE" if not flags else "|".join(flags)
        highlighted_answer = highlight_all_hits(answer, improved_hits)
        hit_list = format_hit_list(improved_hits)
        selected.append(
            (
                str(row.get("idx", "")),
                unique_letters[0],
                "/".join(unique_letters),
                flag_text,
                f"hits={len(improved_hits)} [{hit_list}]",
            )
        )
        selected.append(("", "", "", "", highlighted_answer))

    lines: list[str] = []
    lines.append(f"first regex: {FIRST_REGEX}")
    lines.append(f"improved regex (core_plus_paren_plus_multiline): {SAFE_BEST_REGEX}")
    lines.append("flags: MULTI_HIT => multiple extracted occurrences, MULTI_LETTER => conflicting letters")
    lines.append(f"total improved-only answers: {len(selected) // 2}")
    lines.append("")

    item_idx = 1
    for meta, answer_line in zip(selected[0::2], selected[1::2]):
        idx, first_letter, unique_letters, flag_text, hit_summary = meta
        highlighted_answer = answer_line[4]
        lines.append(
            f"[{item_idx}] idx={idx} first_extracted={first_letter} "
            f"letters={unique_letters} flags={flag_text} {hit_summary}"
        )
        lines.append(highlighted_answer)
        lines.append("")
        item_idx += 1

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.resolve()} with {item_idx - 1} improved-only answers")


if __name__ == "__main__":
    main()
