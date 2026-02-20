#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_ORANGE = "\033[38;5;208m"
ANSI_GREY = "\033[90m"
ANSI_BLUE = "\033[94m"
ANSI_PURPLE = "\033[95m"
ANSI_RESET = "\033[0m"


def _stacked_progress_bar(data, width=32):
    total = (
        data.get("created", 0)
        + data.get("impossible", 0)
        + data.get("errors", 0)
        + data.get("missing", 0)
    )
    if total <= 0:
        return "[" + "-" * width + "]"
    created_len = int(round((data.get("created", 0) / total) * width))
    impossible_len = int(round((data.get("impossible", 0) / total) * width))
    errors_len = int(round((data.get("errors", 0) / total) * width))
    missing_len = width - (created_len + impossible_len + errors_len)
    if missing_len < 0:
        missing_len = 0
    return (
        "["
        + f"{ANSI_GREEN}{'#' * created_len}{ANSI_RESET}"
        + f"{ANSI_ORANGE}{'#' * impossible_len}{ANSI_RESET}"
        + f"{ANSI_RED}{'#' * errors_len}{ANSI_RESET}"
        + f"{ANSI_GREY}{'#' * missing_len}{ANSI_RESET}"
        + "]"
    )


def _load_items(path: Path):
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("questions", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise SystemExit("Unsupported JSON structure; expected a list of question objects.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print distribution of questions by category and sub-category."
    )
    parser.add_argument(
        "--input",
        default=(
            "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/"
            "run_28_general/test_run_28_general.json"
        ),
        help="Path to the questions JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    items = _load_items(input_path)

    counts = defaultdict(int)
    category_counts = defaultdict(int)
    sub_category_counts = defaultdict(int)
    unique_question_ids = set()

    for item in items:
        category = item.get("category") or "unknown_category"
        sub_category = item.get("sub_category") or "unknown_sub_category"
        category = str(category)
        sub_category = str(sub_category)
        counts[(category, sub_category)] += 1
        category_counts[category] += 1
        sub_category_counts[sub_category] += 1
        qid = item.get("question_id") or item.get("question_key")
        if qid is not None:
            unique_question_ids.add(str(qid))

    rows = []
    max_key_len = 0
    max_sub_len = 0
    max_c_len = 0
    max_i_len = 0
    max_e_len = 0
    max_m_len = 0
    max_a_len = 0
    max_p_len = 0

    total_created = 0
    total_impossible = 0
    total_errors = 0
    total_missing = 0
    total_attempted = 0

    for (category, sub_category) in sorted(counts.keys()):
        created = counts[(category, sub_category)]
        data = {
            "created": created,
            "impossible": 0,
            "errors": 0,
            "missing": 0,
            "attempted": created,
        }
        rows.append((category, sub_category, data))

        max_key_len = max(max_key_len, len(category))
        max_sub_len = max(max_sub_len, len(sub_category))
        max_c_len = max(max_c_len, len(str(created)))
        max_i_len = max(max_i_len, len("0"))
        max_e_len = max(max_e_len, len("0"))
        max_m_len = max(max_m_len, len("0"))
        max_a_len = max(max_a_len, len(str(created)))
        max_p_len = max(max_p_len, len("100.00%"))

        total_created += created
        total_attempted += created

    print("\nSummary by category and sub-category:")
    legend = (
        f"{ANSI_GREEN}C=created{ANSI_RESET}, "
        f"{ANSI_ORANGE}I=impossible{ANSI_RESET}, "
        f"{ANSI_RED}E=errors{ANSI_RESET}, "
        f"{ANSI_BLUE}M=missing{ANSI_RESET}, "
        f"{ANSI_PURPLE}A=attempted{ANSI_RESET}"
    )
    print(f"Legend:\t{legend}")

    for category, sub_category, data in rows:
        bar = _stacked_progress_bar(data)
        key_field = category.ljust(max_key_len)
        sub_field = sub_category.ljust(max_sub_len)
        c_val = str(data["created"]).rjust(max_c_len)
        i_val = str(data["impossible"]).rjust(max_i_len)
        e_val = str(data["errors"]).rjust(max_e_len)
        m_val = str(data["missing"]).rjust(max_m_len)
        a_val = str(data["attempted"]).rjust(max_a_len)
        pct = (data["created"] / total_created * 100.0) if total_created > 0 else 0.0
        p_val = f"{pct:.2f}%".rjust(max_p_len)
        line = (
            f"{bar}\t{key_field}\t{sub_field}\t"
            f"{ANSI_GREEN}C={c_val}{ANSI_RESET}\t"
            f"{ANSI_ORANGE}I={i_val}{ANSI_RESET}\t"
            f"{ANSI_RED}E={e_val}{ANSI_RESET}\t"
            f"{ANSI_BLUE}M={m_val}{ANSI_RESET}\t"
            f"{ANSI_PURPLE}A={a_val}{ANSI_RESET}\t"
            f"{ANSI_GREY}P={p_val}{ANSI_RESET}"
        )
        print(line)

    print("")
    print("Summary by category (percent of total):")
    max_cat_len = max((len(k) for k in category_counts), default=0)
    max_cat_c_len = max((len(str(v)) for v in category_counts.values()), default=1)
    max_cat_p_len = len("100.00%")
    for category in sorted(category_counts.keys()):
        created = category_counts[category]
        pct = (created / total_created * 100.0) if total_created > 0 else 0.0
        cat_field = category.ljust(max_cat_len)
        c_val = str(created).rjust(max_cat_c_len)
        p_val = f"{pct:.2f}%".rjust(max_cat_p_len)
        print(
            f"{cat_field}\t{ANSI_GREEN}C={c_val}{ANSI_RESET}\t{ANSI_GREY}P={p_val}{ANSI_RESET}"
        )

    print("")
    print("Summary by sub-category (percent of total):")
    max_subcat_len = max((len(k) for k in sub_category_counts), default=0)
    max_subcat_c_len = max(
        (len(str(v)) for v in sub_category_counts.values()), default=1
    )
    max_subcat_p_len = len("100.00%")
    for sub_category in sorted(sub_category_counts.keys()):
        created = sub_category_counts[sub_category]
        pct = (created / total_created * 100.0) if total_created > 0 else 0.0
        sub_field = sub_category.ljust(max_subcat_len)
        c_val = str(created).rjust(max_subcat_c_len)
        p_val = f"{pct:.2f}%".rjust(max_subcat_p_len)
        print(
            f"{sub_field}\t{ANSI_GREEN}C={c_val}{ANSI_RESET}\t{ANSI_GREY}P={p_val}{ANSI_RESET}"
        )

    print("-" * 12)
    total_data = {
        "created": total_created,
        "impossible": total_impossible,
        "errors": total_errors,
        "missing": total_missing,
    }
    total_bar = _stacked_progress_bar(total_data)
    total_key = "TOTAL".ljust(max_key_len)
    total_sub = "-".ljust(max_sub_len)
    total_c = str(total_created).rjust(max_c_len)
    total_i = str(total_impossible).rjust(max_i_len)
    total_e = str(total_errors).rjust(max_e_len)
    total_m = str(total_missing).rjust(max_m_len)
    total_a = str(total_attempted).rjust(max_a_len)
    total_unique = str(len(unique_question_ids))
    total_line = (
        f"{total_bar}\t{total_key}\t{total_sub}\t"
        f"{ANSI_GREEN}C={total_c}{ANSI_RESET}\t"
        f"{ANSI_ORANGE}I={total_i}{ANSI_RESET}\t"
        f"{ANSI_RED}E={total_e}{ANSI_RESET}\t"
        f"{ANSI_BLUE}M={total_m}{ANSI_RESET}\t"
        f"{ANSI_PURPLE}A={total_a}{ANSI_RESET}\t"
        f"{ANSI_GREY}Q={total_unique}{ANSI_RESET}"
    )
    print(total_line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
