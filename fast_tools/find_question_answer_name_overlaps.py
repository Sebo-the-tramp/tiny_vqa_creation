#!/usr/bin/env python3
import argparse
import json
import re
import sys


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_question_and_options(question: str):
    parts = re.split(r"\b([A-D])\.\s*", question)
    if len(parts) <= 1:
        return question.strip(), []
    question_part = parts[0].strip()
    options = []
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        letter = parts[i]
        text = parts[i + 1].strip()
        options.append((letter, text))
    return question_part, options


def extract_quoted_names(text: str):
    return [m.group(1).strip() for m in re.finditer(r'"([^"]+)"', text)]


def find_overlaps(items):
    results = []
    for item in items:
        question = item.get("question", "")
        question_part, options = split_question_and_options(question)
        names = extract_quoted_names(question_part)
        if not names or not options:
            continue
        name_norms = {normalize(n): n for n in names if normalize(n)}
        if not name_norms:
            continue
        hits = []
        for letter, opt_text in options:
            opt_norm = normalize(opt_text)
            for name_norm, name in name_norms.items():
                if name_norm and name_norm in opt_norm:
                    hits.append(
                        {
                            "letter": letter,
                            "option": opt_text,
                            "name": name,
                        }
                    )
        if hits:
            results.append(
                {
                    "question_id": item.get("question_id"),
                    "idx": item.get("idx"),
                    "question": question,
                    "hits": hits,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find quoted names in questions that also appear in answer options."
    )
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument(
        "--max-print",
        type=int,
        default=25,
        help="Maximum number of matches to print (default: 25)",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Expected a JSON list at top level.", file=sys.stderr)
        return 2

    results = find_overlaps(data)
    print(f"Matches found: {len(results)}")
    for entry in results[: args.max_print]:
        qid = entry.get("question_id")
        idx = entry.get("idx")
        sim_path = entry.get("simulation_id")
        file_name = entry.get("file_name")
        print("-" * 80)
        print(f"question_id: {qid} | idx: {idx}")
        if sim_path:
            print(f"simulation_id: {sim_path}")
        if file_name:
            if isinstance(file_name, list):
                for p in file_name:
                    print(f"image_path: {p}")
            else:
                print(f"image_path: {file_name}")
        print(entry["question"])
        for hit in entry["hits"]:
            print(f"  {hit['letter']}. {hit['option']}  [name: {hit['name']}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
