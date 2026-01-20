#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_object_name(obj):
    desc = obj.get("description") or {}
    name = desc.get("object_name")
    if name:
        return name
    # Fallbacks in case object_name is missing.
    return obj.get("model") or obj.get("name") or "unknown"


def extract_quoted_phrases(text):
    if not text:
        return []
    # Capture phrases in double or single quotes.
    phrases = re.findall(r'"([^"]+)"', text)
    phrases += re.findall(r"'([^']+)'", text)
    return phrases


def parse_mcq_options(question_text):
    # Extract lines like "A. option text"
    options = {}
    if not question_text:
        return options
    for line in question_text.splitlines():
        line = line.strip()
        if re.match(r"^[A-D]\\.", line):
            letter = line[0]
            option = line[2:].strip()
            options[letter] = option
    return options


def normalize_text(s):
    return re.sub(r"\\s+", " ", s.strip().lower())


def find_duplicates_in_sim(sim_path):
    sim = load_json(sim_path)
    objects = sim.get("objects", {})
    name_to_ids = defaultdict(list)
    model_to_ids = defaultdict(list)
    for obj_id, obj in objects.items():
        name = extract_object_name(obj).strip()
        name_to_ids[name.lower()].append(obj_id)
        model = (obj.get("model") or "").strip()
        if model:
            model_to_ids[model.lower()].append(obj_id)
    dup_names = {n: ids for n, ids in name_to_ids.items() if len(ids) > 1}
    dup_models = {m: ids for m, ids in model_to_ids.items() if len(ids) > 1}
    return dup_names, dup_models


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find questions that reference object names appearing multiple times "
            "in a simulation."
        )
    )
    parser.add_argument("questions_path", help="Path to questions JSON (list).")
    parser.add_argument(
        "--answers-path",
        help="Optional path to answers JSON (list with idx/answer).",
    )
    parser.add_argument(
        "--include-non-simulation",
        action="store_true",
        help="Include questions where source != 'simulation'.",
    )
    args = parser.parse_args()

    questions = load_json(args.questions_path)
    answers = {}
    if args.answers_path:
        answer_rows = load_json(args.answers_path)
        for row in answer_rows:
            idx = row.get("idx")
            ans = row.get("answer")
            if idx and ans:
                answers[idx] = ans
    sim_cache = {}
    sim_errors = {}
    matches = []
    for q in questions:
        if not args.include_non_simulation and q.get("source") != "simulation":
            continue
        sim_path = q.get("simulation_id")
        if not sim_path:
            continue
        if sim_path not in sim_cache and sim_path not in sim_errors:
            try:
                sim_cache[sim_path] = find_duplicates_in_sim(sim_path)
            except Exception as exc:
                sim_errors[sim_path] = str(exc)
        if sim_path in sim_errors:
            continue
        dup_names, dup_models = sim_cache[sim_path]
        qtext = q.get("question", "")
        phrases = [normalize_text(p) for p in extract_quoted_phrases(qtext)]
        if not phrases:
            hit_names = []
            hit_models = []
        else:
            hit_names = [(n, ids) for n, ids in dup_names.items() if n in phrases]
            hit_models = [(m, ids) for m, ids in dup_models.items() if m in phrases]

        answer_match = []
        idx = q.get("idx")
        if idx in answers:
            options = parse_mcq_options(qtext)
            ans_letter = answers[idx]
            ans_text = options.get(ans_letter)
            if ans_text:
                ans_norm = normalize_text(ans_text)
                for name, ids in dup_names.items():
                    if ans_norm == name:
                        answer_match.append(("name", name, ids, ans_text))
                for model, ids in dup_models.items():
                    if ans_norm == model:
                        answer_match.append(("model", model, ids, ans_text))

        if hit_names or hit_models or answer_match:
            matches.append(
                {
                    "idx": q.get("idx"),
                    "question_id": q.get("question_id"),
                    "question": qtext,
                    "simulation_id": sim_path,
                    "matches": hit_names,
                    "model_matches": hit_models,
                    "answer_matches": answer_match,
                }
            )

    if sim_errors:
        for path, err in sorted(sim_errors.items()):
            print(f"warning: failed to read {path}: {err}")
        print()

    for m in matches:
        print(
            f"idx={m['idx']} question_id={m['question_id']} simulation_id={m['simulation_id']}"
        )
        for name, ids in m["matches"]:
            print(f"  name='{name}' count={len(ids)} object_ids={ids}")
        for model, ids in m["model_matches"]:
            print(f"  model='{model}' count={len(ids)} object_ids={ids}")
        for kind, value, ids, ans_text in m["answer_matches"]:
            print(
                f"  answer_{kind}='{value}' count={len(ids)} object_ids={ids} answer_text='{ans_text}'"
            )
        print(f"  question={m['question']}")
        print()

    if not matches:
        print("No questions reference duplicate object names/models, and no answers match duplicates.")


if __name__ == "__main__":
    main()


# python check_duplicate_object_names.py /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_23_general/test_run_23_general_10K.json --answers-path /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_23_general/val_answer_run_23_general.json