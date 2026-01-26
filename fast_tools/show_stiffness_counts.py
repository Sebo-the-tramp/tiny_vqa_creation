import argparse
import json
from collections import Counter
from pathlib import Path


STIFFNESS_KEYS = ("stiff", "medium", "soft")
GROUP_KEYS = ("category", "sub_category", "question_id")


def load_questions(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of questions in {path}")
    return data


def extract_stiffness(simulation_id):
    if not simulation_id:
        return "__MISSING__"
    sim = str(simulation_id).lower()
    for key in STIFFNESS_KEYS:
        if key in sim:
            return key
    return "__OTHER__"


def summarize_by_group(questions, group_key):
    groups = {}
    for q in questions:
        group_value = q.get(group_key, "__MISSING__")
        stiffness = extract_stiffness(q.get("simulation_id"))
        if group_value not in groups:
            groups[group_value] = Counter()
        groups[group_value][stiffness] += 1
    return groups


def print_group_summary(title, groups):
    print(f"\n=== {title} ===")
    for group_value in sorted(groups.keys(), key=lambda v: str(v)):
        counts = groups[group_value]
        total = sum(counts.values())
        parts = []
        for key in STIFFNESS_KEYS:
            cnt = counts.get(key, 0)
            pct = cnt / total if total else 0
            parts.append(f"{key}={cnt} ({pct:.1%})")
        other = counts.get("__OTHER__", 0)
        missing = counts.get("__MISSING__", 0)
        if other:
            parts.append(f"__OTHER__={other}")
        if missing:
            parts.append(f"__MISSING__={missing}")
        print(f"{group_value}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Count questions by stiffness inferred from simulation_id."
    )
    parser.add_argument("path", help="Path to output JSON file")
    args = parser.parse_args()

    questions = load_questions(args.path)
    counts = Counter(extract_stiffness(q.get("simulation_id")) for q in questions)

    total = sum(counts.values())
    print(f"Loaded {len(questions)} questions from {args.path}")
    print("Counts by stiffness:")
    for key in STIFFNESS_KEYS + ("__OTHER__", "__MISSING__"):
        if key in counts:
            pct = counts[key] / total if total else 0
            print(f"{key:<10} {counts[key]} ({pct:.1%})")

    for group_key in GROUP_KEYS:
        groups = summarize_by_group(questions, group_key)
        print_group_summary(f"By {group_key}", groups)


if __name__ == "__main__":
    main()
