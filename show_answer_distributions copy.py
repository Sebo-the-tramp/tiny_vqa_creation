import json
import pandas as pd

# PATH = "/mnt/proj1/eu-25-92/tiny_vqa_creation/output/"
PATH = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
RUN_NAME = "test_run04_full_images.json"

with open(PATH + RUN_NAME, "r") as f:
    data = json.load(f)

eval_df = pd.json_normalize(data)

def print_histogram(series, title=None):
    """Print a 40-char wide normalized histogram for a given pandas Series."""
    answer_counts = series.value_counts()
    total = answer_counts.sum()
    if title:
        print(f"\n=== {title} ===")
    for answer, count in answer_counts.items():
        bar = '█' * int((count / total) * 40)
        print(f"{answer:20} | {bar} {count} ({count/total:.1%})")

# Overall distribution
print_histogram(eval_df['answer'], title="Overall Answer Distribution")

# By category
if 'category' in eval_df.columns:
    for cat, sub_df in eval_df.groupby('category'):
        print_histogram(sub_df['answer'], title=f"Category: {cat}")

# By sub_category
if 'sub_category' in eval_df.columns:
    for subcat, sub_df in eval_df.groupby('sub_category'):
        print_histogram(sub_df['answer'], title=f"Sub-category: {subcat}")

# By question_ID
if 'question_ID' in eval_df.columns:
    for qid, sub_df in eval_df.groupby('question_ID'):
        print_histogram(sub_df['answer'], title=f"Question ID: {qid}")
