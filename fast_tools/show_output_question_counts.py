import argparse
import json
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

    question_counts = count_by(questions, "question_id")
    if args.question_limit and args.question_limit > 0:
        question_counts = Counter(dict(question_counts.most_common(args.question_limit)))
    print_counts(
        "Question ID distribution",
        question_counts,
        args.width,
        args.cap,
        args.label_width,
    )


if __name__ == "__main__":
    main()
