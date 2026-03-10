#!/usr/bin/env python3
import json
import random
import re
from pathlib import Path
from typing import Pattern

INPUT_JSON_PATH = Path(
    "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general/results_run_28_general-all/MiniCPM-V2.5_val.json"
)
ANSWER_KEY = "answer"
VALID_CHOICES = {"A", "B", "C", "D"}
ENABLE_SUBSAMPLE = False
SUBSAMPLE_SIZE = 10_000
SUBSAMPLE_SEED = 42
DIFF_LEFT_REGEX_NAME = "experimental_answerword_window"
DIFF_RIGHT_REGEX_NAME = "first"
DIFF_PREVIEW_COUNT = 10
DIFF_PREVIEW_MAX_CHARS = 220
REGEX_DISPLAY_MAX_CHARS = 100
HIGHLIGHT_TOP_ROW_GREEN = True
GREEN = "\033[32m"
RESET = "\033[0m"

FIRST_REGEX = r"\b([ABCD])\.|^\s*([ABCD])\s*$"
LAST_REGEX = r"\banswer\s*(?:is|:)\s*([ABCD])[\.\)]?"
CORRECT_ANSWER_REGEX = r"\bcorrect\s+answer\s*(?:is|:)\s*([ABCD])[\.\)]?"
CORRECT_OPTION_REGEX = r"\bcorrect\s+option\s*(?:is|:)\s*([ABCD])[\.\)]?"
CORRECT_CHOICE_REGEX = r"\bcorrect\s+choice\s*(?:is|:)\s*([ABCD])[\.\)]?"
FINAL_ANSWER_REGEX = r"\bfinal\s+answer\s*(?:is|:)\s*([ABCD])[\.\)]?"
THE_ANSWER_IS_REGEX = r"\bthe\s+answer\s+is\s*([ABCD])[\.\)]?"
OPTION_LETTER_REGEX = r"\boption\s*([ABCD])\b"
PAREN_LETTER_REGEX = r"\(([ABCD])\)"
MULTILINE_SINGLE_LETTER_REGEX = r"^\s*([ABCD])\s*$"
EXPERIMENTAL_CLOSE_PAREN_LOOSE_REGEX = r"\b([ABCD])\)"
EXPERIMENTAL_ANSWERWORD_WINDOW_REGEX = r"\b(?:answer|option|choice|correct|final)\b[^\n]{0,30}\b([ABCD])\b"
EXPERIMENTAL_SINGLE_TOKEN_REGEX = r"\b([ABCD])\b"
SANITIZE_ANSWER_REGEX = r"(?:^([A-D])\b|\b([A-D])\b\s*[\.\,\:\)]?$)"

CORE_REGEX = (
    rf"(?:{FIRST_REGEX})|(?:{LAST_REGEX})|(?:{CORRECT_ANSWER_REGEX})|"
    rf"(?:{CORRECT_OPTION_REGEX})|(?:{CORRECT_CHOICE_REGEX})|"
    rf"(?:{FINAL_ANSWER_REGEX})|(?:{THE_ANSWER_IS_REGEX})|(?:{OPTION_LETTER_REGEX})"
)
SAFE_BEST_REGEX = (
    rf"(?:{CORE_REGEX})|(?:{PAREN_LETTER_REGEX})|(?:{MULTILINE_SINGLE_LETTER_REGEX})"
)

REGEX_PATTERNS: list[tuple[str, str, int]] = [
    ("first", FIRST_REGEX, 0),
    ("sanitize_answer_regex", SANITIZE_ANSWER_REGEX, re.IGNORECASE),
    ("first_or_answer", rf"(?:{FIRST_REGEX})|(?:{LAST_REGEX})", re.IGNORECASE),
    ("first_or_correct_answer", rf"(?:{FIRST_REGEX})|(?:{CORRECT_ANSWER_REGEX})", re.IGNORECASE),
    ("first_or_correct_option", rf"(?:{FIRST_REGEX})|(?:{CORRECT_OPTION_REGEX})", re.IGNORECASE),
    ("first_or_correct_choice", rf"(?:{FIRST_REGEX})|(?:{CORRECT_CHOICE_REGEX})", re.IGNORECASE),
    ("first_or_final_answer", rf"(?:{FIRST_REGEX})|(?:{FINAL_ANSWER_REGEX})", re.IGNORECASE),
    ("first_or_the_answer_is", rf"(?:{FIRST_REGEX})|(?:{THE_ANSWER_IS_REGEX})", re.IGNORECASE),
    ("first_or_option_letter", rf"(?:{FIRST_REGEX})|(?:{OPTION_LETTER_REGEX})", re.IGNORECASE),
    ("first_or_paren_letter", rf"(?:{FIRST_REGEX})|(?:{PAREN_LETTER_REGEX})", re.IGNORECASE),
    ("core", CORE_REGEX, re.IGNORECASE),
    ("core_plus_paren", rf"(?:{CORE_REGEX})|(?:{PAREN_LETTER_REGEX})", re.IGNORECASE),
    (
        "core_plus_paren_plus_multiline",
        SAFE_BEST_REGEX,
        re.IGNORECASE | re.MULTILINE,
    ),
    (
        "experimental_close_paren_loose",
        rf"(?:{SAFE_BEST_REGEX})|(?:{EXPERIMENTAL_CLOSE_PAREN_LOOSE_REGEX})",
        re.IGNORECASE | re.MULTILINE,
    ),
    (
        "experimental_answerword_window",
        rf"(?:{SAFE_BEST_REGEX})|(?:{EXPERIMENTAL_ANSWERWORD_WINDOW_REGEX})",
        re.IGNORECASE | re.MULTILINE,
    ),
    # (
    #     "experimental_single_token_case_sensitive",
    #     rf"(?i:(?:{SAFE_BEST_REGEX}))|(?:{EXPERIMENTAL_SINGLE_TOKEN_REGEX})",
    #     re.MULTILINE,
    # ),
    # (
    #     "experimental_single_token_case_insensitive",
    #     rf"(?:{SAFE_BEST_REGEX})|(?:{EXPERIMENTAL_SINGLE_TOKEN_REGEX})",
    #     re.IGNORECASE | re.MULTILINE,
    # ),
]


def load_answers_and_indices(path: Path) -> tuple[list[str], list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    answers = [row[ANSWER_KEY] for row in rows]
    assert all(isinstance(answer, str) for answer in answers)
    indices = [str(row.get("idx", "")) for row in rows]
    return answers, indices


def extract_all_choices(answer: str, pattern: Pattern[str]) -> list[str]:
    choices: list[str] = []
    for match in pattern.finditer(answer):
        for group in match.groups():
            if group is None:
                continue
            choice = group.upper()
            if choice in VALID_CHOICES:
                choices.append(choice)
                break
    return choices


def count_stats(answers: list[str], pattern: Pattern[str]) -> tuple[int, int]:
    extracted_count = 0
    multi_hit_count = 0
    for answer in answers:
        choices = extract_all_choices(answer, pattern)
        if not choices:
            continue
        extracted_count += 1
        if len(choices) > 1:
            multi_hit_count += 1
    return extracted_count, multi_hit_count


def analyze_pattern(answers: list[str], pattern: Pattern[str]) -> tuple[int, int, list[list[str]]]:
    extracted_count = 0
    multi_hit_count = 0
    all_choices: list[list[str]] = []
    for answer in answers:
        choices = extract_all_choices(answer, pattern)
        all_choices.append(choices)
        if not choices:
            continue
        extracted_count += 1
        if len(choices) > 1:
            multi_hit_count += 1
    return extracted_count, multi_hit_count, all_choices


def build_subsample(answers: list[str]) -> list[str]:
    sample_size = min(SUBSAMPLE_SIZE, len(answers))
    rng = random.Random(SUBSAMPLE_SEED)
    sample_indices = rng.sample(range(len(answers)), sample_size)
    return [answers[i] for i in sample_indices]


def flags_to_source(flags: int) -> str:
    names: list[str] = []
    if flags & re.IGNORECASE:
        names.append("re.IGNORECASE")
    if flags & re.MULTILINE:
        names.append("re.MULTILINE")
    if flags & re.DOTALL:
        names.append("re.DOTALL")
    if flags & re.VERBOSE:
        names.append("re.VERBOSE")
    if flags & re.ASCII:
        names.append("re.ASCII")
    if not names:
        return "0"
    return " | ".join(names)


def print_table(rows: list[tuple[str, str, str, str, str | None]], include_subsample: bool) -> None:
    regex_header = "regex used"
    result_header = "extracted"
    multi_hit_header = "multi-hit"
    sure_header = "sure"
    sample_header = "subsample"
    value_header = "(count/total, %)"

    display_rows = []
    for regex_used, result, multi_hit_result, sure_result, sample_result in rows:
        display_regex = regex_used
        if len(display_regex) > REGEX_DISPLAY_MAX_CHARS:
            display_regex = f"{display_regex[:REGEX_DISPLAY_MAX_CHARS - 3]}..."
        display_rows.append((display_regex, result, multi_hit_result, sure_result, sample_result))

    regex_width = max(len(regex_header), *(len(regex_used) for regex_used, _, _, _, _ in display_rows))
    result_width = max(len(result_header), len(value_header), *(len(result) for _, result, _, _, _ in rows))
    multi_hit_width = max(
        len(multi_hit_header),
        len(value_header),
        *(len(multi_hit_result) for _, _, multi_hit_result, _, _ in rows),
    )
    sure_width = max(len(sure_header), len(value_header), *(len(sure_result) for _, _, _, sure_result, _ in rows))
    if include_subsample:
        sample_width = max(
            len(sample_header),
            len(value_header),
            *(len(sample_result or "") for _, _, _, _, sample_result in rows),
        )
        sep = (
            f"+-{'-' * regex_width}-+-{'-' * result_width}-+-{'-' * multi_hit_width}"
            f"-+-{'-' * sure_width}-+-{'-' * sample_width}-+"
        )
        print(sep)
        print(
            f"| {regex_header.ljust(regex_width)} | {result_header.ljust(result_width)} | "
            f"{multi_hit_header.ljust(multi_hit_width)} | {sure_header.ljust(sure_width)} | "
            f"{sample_header.ljust(sample_width)} |"
        )
        print(
            f"| {' '.ljust(regex_width)} | {value_header.ljust(result_width)} | "
            f"{value_header.ljust(multi_hit_width)} | {value_header.ljust(sure_width)} | "
            f"{value_header.ljust(sample_width)} |"
        )
        print(sep)
        for i, (regex_used, result, multi_hit_result, sure_result, sample_result) in enumerate(display_rows):
            line = (
                f"| {regex_used.ljust(regex_width)} | {result.ljust(result_width)} | "
                f"{multi_hit_result.ljust(multi_hit_width)} | {sure_result.ljust(sure_width)} | "
                f"{(sample_result or '').ljust(sample_width)} |"
            )
            if HIGHLIGHT_TOP_ROW_GREEN and i == 0:
                print(f"{GREEN}{line}{RESET}")
            else:
                print(line)
        print(sep)
        return

    sep = (
        f"+-{'-' * regex_width}-+-{'-' * result_width}-+-{'-' * multi_hit_width}"
        f"-+-{'-' * sure_width}-+"
    )
    print(sep)
    print(
        f"| {regex_header.ljust(regex_width)} | {result_header.ljust(result_width)} | "
        f"{multi_hit_header.ljust(multi_hit_width)} | {sure_header.ljust(sure_width)} |"
    )
    print(
        f"| {' '.ljust(regex_width)} | {value_header.ljust(result_width)} | "
        f"{value_header.ljust(multi_hit_width)} | {value_header.ljust(sure_width)} |"
    )
    print(sep)
    for i, (regex_used, result, multi_hit_result, sure_result, _) in enumerate(display_rows):
        line = (
            f"| {regex_used.ljust(regex_width)} | {result.ljust(result_width)} | "
            f"{multi_hit_result.ljust(multi_hit_width)} | {sure_result.ljust(sure_width)} |"
        )
        if HIGHLIGHT_TOP_ROW_GREEN and i == 0:
            print(f"{GREEN}{line}{RESET}")
        else:
            print(line)
    print(sep)


def print_diff_summary(
    answers: list[str],
    indices: list[str],
    left_name: str,
    right_name: str,
    choices_by_name: dict[str, list[list[str]]],
) -> None:
    left_choices = choices_by_name[left_name]
    right_choices = choices_by_name[right_name]

    left_set = {i for i, choices in enumerate(left_choices) if choices}
    right_set = {i for i, choices in enumerate(right_choices) if choices}
    overlap = len(left_set & right_set)
    left_only = sorted(left_set - right_set)
    right_only = sorted(right_set - left_set)

    print(f"\ncomparison: {left_name} vs {right_name}")
    print(f"{left_name}: {len(left_set)}")
    print(f"{right_name}: {len(right_set)}")
    print(f"overlap: {overlap}")
    print(f"{left_name}_only: {len(left_only)}")
    print(f"{right_name}_only: {len(right_only)}")

    print(f"\n{left_name}_only preview (first {min(DIFF_PREVIEW_COUNT, len(left_only))}):")
    for pos in left_only[:DIFF_PREVIEW_COUNT]:
        hit_text = "/".join(left_choices[pos])
        preview = " ".join(answers[pos].split())[:DIFF_PREVIEW_MAX_CHARS]
        print(f"- idx={indices[pos]} hits={hit_text} text={preview}")

    if right_only:
        print(f"\n{right_name}_only preview (first {min(DIFF_PREVIEW_COUNT, len(right_only))}):")
        for pos in right_only[:DIFF_PREVIEW_COUNT]:
            hit_text = "/".join(right_choices[pos])
            preview = " ".join(answers[pos].split())[:DIFF_PREVIEW_MAX_CHARS]
            print(f"- idx={indices[pos]} hits={hit_text} text={preview}")


def main() -> None:
    answers, answer_indices = load_answers_and_indices(INPUT_JSON_PATH)
    total_questions = len(answers)
    sample_answers: list[str] = []
    sample_total = 0
    if ENABLE_SUBSAMPLE:
        sample_answers = build_subsample(answers)
        sample_total = len(sample_answers)

    scored_rows: list[tuple[float, tuple[str, str, str, str, str | None]]] = []
    diff_choices: dict[str, list[list[str]]] = {}
    for name, regex_pattern, flags in REGEX_PATTERNS:
        pattern = re.compile(regex_pattern, flags)
        extracted, multi_hit, full_choices = analyze_pattern(answers, pattern)
        percentage = extracted / total_questions * 100
        multi_hit_percentage = multi_hit / total_questions * 100
        sure_answers = extracted - multi_hit
        sure_percentage = sure_answers / total_questions * 100
        sample_result = None
        if ENABLE_SUBSAMPLE:
            sample_extracted, _ = count_stats(sample_answers, pattern)
            sample_percentage = sample_extracted / sample_total * 100
            sample_result = f"{sample_extracted}/{sample_total} ({sample_percentage:.2f}%)"
        if name in {DIFF_LEFT_REGEX_NAME, DIFF_RIGHT_REGEX_NAME}:
            diff_choices[name] = full_choices
        scored_rows.append(
            (
                percentage,
                (
                    f"{name} :: {regex_pattern}",
                    f"{extracted}/{total_questions} ({percentage:.2f}%)",
                    f"{multi_hit}/{total_questions} ({multi_hit_percentage:.2f}%)",
                    f"{sure_answers}/{total_questions} ({sure_percentage:.2f}%)",
                    sample_result,
                ),
            )
        )

    rows = [row for _, row in sorted(scored_rows, key=lambda x: x[0], reverse=True)]
    print_table(rows, ENABLE_SUBSAMPLE)
    print_diff_summary(
        answers=answers,
        indices=answer_indices,
        left_name=DIFF_LEFT_REGEX_NAME,
        right_name=DIFF_RIGHT_REGEX_NAME,
        choices_by_name=diff_choices,
    )

    print("\ncopy-paste Python snippets:")
    print("import re")
    for target_name in ("first", "experimental_close_paren_loose"):
        target_pattern, target_flags = next(
            (regex_pattern, flags)
            for name, regex_pattern, flags in REGEX_PATTERNS
            if name == target_name
        )
        print()
        print(f"{target_name}_pattern = {target_pattern!r}")
        print(
            f"{target_name} = re.compile("
            f"{target_name}_pattern, {flags_to_source(target_flags)})"
        )


if __name__ == "__main__":
    main()
