#!/usr/bin/env python3
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(
    "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general/results_run_28_general-all"
)
FILE_GLOB = "*.json"
ANSWER_KEY = "answer"
IDX_KEY = "idx"
VALID_ANSWERS = ["A", "B", "C", "D"]
MAX_WORKERS = os.cpu_count() or 1
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"
NOT_FOUND_DIR = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/fast_tools/not_found")
DIFFERENCE_DIR = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/fast_tools/difference")

ANSWER_PATTERN = r"(?:^([A-D])\b|\b([A-D])\b\s*[\.\,\:\)]?$)" # Raoul simple
# ALT_ANSWER_PATTERN = r"\b(?:[Oo]ption|[Aa]nswer is)\s*([A-D])\b|^\s*([A-D])\b|\b([A-D])\."
ALT_ANSWER_PATTERN = r"\b(?:(?:[Tt]he|[Cc]orrect|[Ff]inal)\s+)?(?:[Aa]nswer|[Oo]ption|[Cc]hoice)\s*(?:[Ii]s|:)?\s*([ABCD])\b|^\s*\(?([ABCD])\)?\b|\b([ABCD])[\.\,\)]|\b([ABCD])\:?\s*$" # 99.27%
# ALT_ANSWER_PATTERN = r"\b([ABCD])[\.\,]|^\s*([ABCD])\b|\b([ABCD])\s*$" # 99.21
# ALT_ANSWER_PATTERN = r"\b([ABCD])\.|^\s*([ABCD])\s*" # Sebastian Simple -> 99.08

_ANSWER_RE = re.compile(r"(?:^([A-D])\b|\b([A-D])\b\s*[\.\,\:\)]?$)", re.IGNORECASE)

_ANSWER_RE = re.compile(r"\b(?:(?:[Tt]he|[Cc]orrect|[Ff]inal)\s+)?(?:[Aa]nswer|[Oo]ption|[Cc]hoice)\s*(?:[Ii]s|:)?\s*([ABCD])\b|^\s*\(?([ABCD])\)?\b|\b([ABCD])[\.\,\)]|\b([ABCD])\:?\s*$", re.IGNORECASE)
_ALT_ANSWER_RE = re.compile(ALT_ANSWER_PATTERN, re.IGNORECASE)


def _sanitize_answer_with_regex(answer: object, pattern: re.Pattern[str]) -> str | None:
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return ""
    match = pattern.search(str(answer))
    if not match:
        return "?"
    answer = next((group for group in match.groups() if group), None)
    return answer.upper()


def _sanitize_answer(answer: object) -> str | None:
    return _sanitize_answer_with_regex(answer, _ANSWER_RE)


def _sanitize_answer_alt(answer: object) -> str | None:
    return _sanitize_answer_with_regex(answer, _ALT_ANSWER_RE)


def _is_valid_answer(answer: str | None) -> bool:
    return answer in VALID_ANSWERS


def _count_file(path: Path) -> tuple[str, int, int, int]:
    frame = pd.read_json(path)
    answers = frame[ANSWER_KEY]
    sanitized_answers = answers.map(_sanitize_answer)
    sanitized_answers_alt = answers.map(_sanitize_answer_alt)
    total_answers = int((sanitized_answers != "").sum())
    regex_hits = int(sanitized_answers.map(_is_valid_answer).sum())
    regex_hits_alt = int(sanitized_answers_alt.map(_is_valid_answer).sum())
    return path.name, regex_hits, regex_hits_alt, total_answers


def _model_name(path: Path) -> str:
    if path.name.endswith("_val.json"):
        return path.name[: -len("_val.json")]
    return path.stem


def _write_missing_lines(path: Path) -> None:
    frame = pd.read_json(path)
    answers = frame[ANSWER_KEY]
    sanitized_answers = answers.map(_sanitize_answer)
    missing_mask = sanitized_answers == "?"
    missing_rows = answers.index[missing_mask]
    idx_values = frame[IDX_KEY] if IDX_KEY in frame.columns else pd.Series(frame.index, index=frame.index)

    lines: list[str] = []
    for row_index in missing_rows:
        idx_value = str(idx_values.at[row_index])
        answer_text = str(answers.at[row_index]).replace("\n", "\\n")
        lines.append(f"{row_index}\t{idx_value}\t{answer_text}")

    model_dir = NOT_FOUND_DIR / _model_name(path)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = "" if not lines else "\n".join(lines) + "\n"
    (model_dir / "missing_lines.txt").write_text(payload, encoding="utf-8")


def _write_difference_lines(path: Path) -> tuple[str, int]:
    frame = pd.read_json(path)
    answers = frame[ANSWER_KEY]
    sanitized_answers = answers.map(_sanitize_answer)
    sanitized_answers_alt = answers.map(_sanitize_answer_alt)
    diff_mask = sanitized_answers != sanitized_answers_alt
    diff_rows = answers.index[diff_mask]
    idx_values = frame[IDX_KEY] if IDX_KEY in frame.columns else pd.Series(frame.index, index=frame.index)

    lines: list[str] = []
    for row_index in diff_rows:
        answer_1 = sanitized_answers.at[row_index]
        answer_2 = sanitized_answers_alt.at[row_index]
        found_1 = _is_valid_answer(answer_1)
        found_2 = _is_valid_answer(answer_2)
        diff_type = "found_status_diff" if found_1 != found_2 else "choice_diff"
        idx_value = str(idx_values.at[row_index])
        answer_text = str(answers.at[row_index]).replace("\n", "\\n")
        lines.append(
            f"{row_index}\t{idx_value}\t{answer_1}\t{answer_2}\t"
            f"{int(found_1)}\t{int(found_2)}\t{diff_type}\t{answer_text}"
        )

    model_name = _model_name(path)
    if not lines:
        return model_name, 0
    model_dir = DIFFERENCE_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    header = "row_index\tidx\tregex1\tregex2\tregex1_found\tregex2_found\tdiff_type\tanswer_text\n"
    payload = header + "\n".join(lines) + "\n"
    (model_dir / "different_lines.txt").write_text(payload, encoding="utf-8")
    return model_name, len(lines)


def main() -> None:
    paths = sorted(RESULTS_DIR.glob(FILE_GLOB))
    assert paths

    workers = min(MAX_WORKERS, len(paths))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_count_file, paths))

    total_hits = 0
    total_hits_alt = 0
    total_answers = 0
    bad_paths: list[Path] = []
    display_rows: list[tuple[str, str, str, float]] = []
    path_by_name = {path.name: path for path in paths}
    for name, regex_hits, regex_hits_alt, answers in rows:
        total_hits += regex_hits
        total_hits_alt += regex_hits_alt
        total_answers += answers
        ratio = 0.0 if answers == 0 else regex_hits / answers * 100
        ratio_alt = 0.0 if answers == 0 else regex_hits_alt / answers * 100
        delta_pp = round(ratio_alt - ratio, 2)
        raoul_text = f"{regex_hits}/{answers} ({ratio:.2f}%)"
        new_text = f"{regex_hits_alt}/{answers} ({ratio_alt:.2f}%, {delta_pp:+.2f}%)"
        display_rows.append((name, raoul_text, new_text, delta_pp))
        if regex_hits != answers:
            bad_paths.append(path_by_name[name])

    if NOT_FOUND_DIR.exists():
        shutil.rmtree(NOT_FOUND_DIR)
    NOT_FOUND_DIR.mkdir(parents=True, exist_ok=True)
    if bad_paths:
        writer_workers = min(MAX_WORKERS, len(bad_paths))
        with ProcessPoolExecutor(max_workers=writer_workers) as pool:
            list(pool.map(_write_missing_lines, bad_paths))

    if DIFFERENCE_DIR.exists():
        shutil.rmtree(DIFFERENCE_DIR)
    DIFFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        difference_rows = list(pool.map(_write_difference_lines, paths))
    difference_rows = [(model_name, count) for model_name, count in difference_rows if count > 0]

    total_ratio = 0.0 if total_answers == 0 else total_hits / total_answers * 100
    total_ratio_alt = 0.0 if total_answers == 0 else total_hits_alt / total_answers * 100
    total_delta_pp = round(total_ratio_alt - total_ratio, 2)
    total_raoul_text = f"{total_hits}/{total_answers} ({total_ratio:.2f}%)"
    total_new_text = f"{total_hits_alt}/{total_answers} ({total_ratio_alt:.2f}%, {total_delta_pp:+.2f}%)"

    model_header = "model"
    raoul_header = "raoul"
    new_header = "new count (% , delta)"
    name_width = max(len(model_header), len("TOTAL"), *(len(name) for name, _, _, _ in display_rows))
    raoul_width = max(len(raoul_header), len(total_raoul_text), *(len(raoul) for _, raoul, _, _ in display_rows))
    new_width = max(len(new_header), len(total_new_text), *(len(new) for _, _, new, _ in display_rows))
    sep = f"+-{'-' * name_width}-+-{'-' * raoul_width}-+-{'-' * new_width}-+"

    print(sep)
    print(
        f"| {model_header.ljust(name_width)} | {raoul_header.ljust(raoul_width)} | "
        f"{new_header.ljust(new_width)} |"
    )
    print(sep)
    for name, raoul_text, new_text, delta_pp in display_rows:
        line = f"| {name.ljust(name_width)} | {raoul_text.ljust(raoul_width)} | {new_text.ljust(new_width)} |"
        if delta_pp < 0:
            print(f"{RED}{line}{RESET}")
            continue
        if delta_pp > 0:
            print(f"{GREEN}{line}{RESET}")
            continue
        print(line)
    print(sep)
    total_line = (
        f"| {'TOTAL'.ljust(name_width)} | {total_raoul_text.ljust(raoul_width)} | "
        f"{total_new_text.ljust(new_width)} |"
    )
    if total_delta_pp < 0:
        print(f"{RED}{total_line}{RESET}")
    elif total_delta_pp > 0:
        print(f"{GREEN}{total_line}{RESET}")
    else:
        print(total_line)
    print(sep)
    if difference_rows:
        print(f"difference files: {len(difference_rows)} models in {DIFFERENCE_DIR}")
    else:
        print(f"difference files: 0 models in {DIFFERENCE_DIR}")


if __name__ == "__main__":
    main()
