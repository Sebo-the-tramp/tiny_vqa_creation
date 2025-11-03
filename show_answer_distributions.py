import json
import pandas as pd

PATH = "/mnt/proj1/eu-25-92/tiny_vqa_creation/output/"
RUN_NAME = "val_answer_test_imbalance.json"

with open(PATH + RUN_NAME, "r") as f:
    data = json.load(f)

eval_df = pd.json_normalize(data)
answer_counts = eval_df['answer'].value_counts()

# Normalize and print textual histogram
total = answer_counts.sum()
for answer, count in answer_counts.items():
    bar = '█' * int((count / total) * 40)  # 40-char wide bar
    print(f"{answer:10} | {bar} {count} ({count/total:.1%})")
