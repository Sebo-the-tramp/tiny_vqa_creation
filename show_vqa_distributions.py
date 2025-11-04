import json
from pathlib import Path

import pandas as pd

SIMPLE_VQA_PATH = Path("simple_vqa.json")


def load_simple_vqa(path):
    path = Path(path)
    data = json.loads(path.read_text())
    records = []

    for category, questions in data.items():
        if not isinstance(questions, dict):
            continue
        for question_id, meta in questions.items():
            meta = meta or {}
            if not isinstance(meta, dict):
                meta = {"value": meta}
            record = {"category": category, "question_id": question_id}
            record.update(meta)
            records.append(record)

    if not records:
        raise ValueError(f"No questions found in {path}")

    return pd.DataFrame.from_records(records)


def print_histogram(series, title=None, width=40, label_width=24):
    counts = series.value_counts(dropna=False)
    total = int(counts.sum())
    if total == 0:
        return
    if title:
        print(f"\n=== {title} ===")
    for val, cnt in counts.items():
        pct = cnt / total
        bar = "█" * int(pct * width)
        label = str(val)[:label_width]
        print(f"{label:<{label_width}} | {bar:<{width}} {cnt} ({pct:.1%})")


df = load_simple_vqa(SIMPLE_VQA_PATH)

print(f"Loaded {len(df)} question templates from {SIMPLE_VQA_PATH}")

if "sub_category" not in df.columns:
    print("No 'sub_category' field found; nothing to summarize.")
else:
    print_histogram(df["sub_category"], title="Overall sub_category distribution")
    print_histogram(df["category"], title="Overall category distribution")

    if "category" in df.columns:
        for category, subset in df.groupby("category"):
            print_histogram(
                subset["sub_category"],
                title=f"Sub_category distribution — category: {category}",
            )



