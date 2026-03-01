import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_questions(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of questions in {path}")
    return data


def count_by(questions, key):
    counts = Counter()
    for item in questions:
        counts[item.get(key, "__MISSING__")] += 1
    return counts


def format_bar(count, total, width, cap):
    if total == 0:
        return ""
    if cap <= 0:
        cap = max(count, 1)
    scaled = min(count, cap) / cap
    bar_len = int(round(scaled * width))
    return "#" * bar_len


def print_counts(title, counts, width, cap, label_width):
    total = sum(counts.values())
    print(f"\n=== {title} (n={total}) ===")
    if total == 0:
        return
    for value, count in counts.most_common():
        pct = count / total
        label = str(value)[:label_width]
        bar = format_bar(count, total, width, cap)
        print(f"{label:<{label_width}} | {bar:<{width}} {count} ({pct:.1%})")


def normalize_question_id(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"[^\w]+$", "", text)
    return text or None


def count_question_id_suffixes(questions):
    counts = Counter()
    for item in questions:
        idx_value = normalize_question_id(item.get("idx"))
        if idx_value is None:
            counts["__MISSING__"] += 1
            continue
        if idx_value.endswith("_g"):
            counts["_g"] += 1
        elif idx_value.endswith("_i"):
            counts["_i"] += 1
        else:
            counts["other"] += 1
    return counts


def debug_idx_summary(questions, width, cap, label_width, limit):
    type_counts = Counter()
    raw_tail2 = Counter()
    raw_tail3 = Counter()
    cleaned_tail2 = Counter()
    cleaned_tail3 = Counter()
    cleaned_changes = Counter()

    for item in questions:
        raw = item.get("idx")
        if raw is None:
            type_counts["__MISSING__"] += 1
            continue

        if isinstance(raw, (list, tuple)):
            type_counts["list"] += 1
            raw_value = raw[0] if raw else None
        else:
            raw_value = raw

        if raw_value is None:
            type_counts["__EMPTY__"] += 1
            continue

        type_counts[type(raw_value).__name__] += 1
        raw_text = str(raw_value)
        if raw_text:
            if len(raw_text) >= 2:
                raw_tail2[raw_text[-2:]] += 1
            if len(raw_text) >= 3:
                raw_tail3[raw_text[-3:]] += 1

        cleaned = normalize_question_id(raw_value)
        if cleaned is None:
            type_counts["__CLEANED_EMPTY__"] += 1
            continue
        if cleaned != raw_text.strip():
            cleaned_changes["cleaned_changed"] += 1
        lower = cleaned.lower()
        if len(lower) >= 2:
            cleaned_tail2[lower[-2:]] += 1
        if len(lower) >= 3:
            cleaned_tail3[lower[-3:]] += 1

    print_counts("Idx type summary", type_counts, width, cap, label_width)
    if limit and limit > 0:
        raw_tail2 = Counter(dict(raw_tail2.most_common(limit)))
        raw_tail3 = Counter(dict(raw_tail3.most_common(limit)))
        cleaned_tail2 = Counter(dict(cleaned_tail2.most_common(limit)))
        cleaned_tail3 = Counter(dict(cleaned_tail3.most_common(limit)))
    print_counts("Idx raw tail-2 summary", raw_tail2, width, cap, label_width)
    print_counts("Idx raw tail-3 summary", raw_tail3, width, cap, label_width)
    print_counts("Idx cleaned tail-2 summary", cleaned_tail2, width, cap, label_width)
    print_counts("Idx cleaned tail-3 summary", cleaned_tail3, width, cap, label_width)
    if cleaned_changes:
        print_counts(
            "Idx cleaned changes",
            cleaned_changes,
            width,
            cap,
            label_width,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize question counts by category, sub_category, and question_id "
            "with ASCII bars scaled to a cap."
        )
    )
    parser.add_argument("path", help="Path to output JSON file")
    parser.add_argument("--width", type=int, default=50, help="Bar width in characters")
    parser.add_argument("--cap", type=int, default=500, help="Cap for bar scaling")
    parser.add_argument(
        "--label-width", type=int, default=28, help="Label width in characters"
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=0,
        help="Limit question_id rows (0 = show all)",
    )
    parser.add_argument(
        "--debug-idx",
        action="store_true",
        help="Show aggregated debug summaries for idx values (no per-idx output).",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=20,
        help="Limit rows in debug summaries (0 = show all).",
    )
    args = parser.parse_args()

    questions = load_questions(args.path)
    print(f"Loaded {len(questions)} questions from {args.path}")
    print(f"Bars scaled to cap={args.cap}, width={args.width}")

    print_counts(
        "Category distribution",
        count_by(questions, "category"),
        args.width,
        args.cap,
        args.label_width,
    )
    print_counts(
        "Sub-category distribution",
        count_by(questions, "sub_category"),
        args.width,
        args.cap,
        args.label_width,
    )

    question_counts = count_by(questions, "idx")
    if args.question_limit and args.question_limit > 0:
        question_counts = Counter(dict(question_counts.most_common(args.question_limit)))
    print_counts(
        "Idx distribution",
        question_counts,
        args.width,
        args.cap,
        args.label_width,
    )
    print_counts(
        "Idx suffix distribution",
        count_question_id_suffixes(questions),
        args.width,
        args.cap,
        args.label_width,
    )
    if args.debug_idx:
        debug_idx_summary(
            questions,
            args.width,
            args.cap,
            args.label_width,
            args.debug_limit,
        )


if __name__ == "__main__":
    main()
