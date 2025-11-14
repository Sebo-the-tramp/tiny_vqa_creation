import os
import json

file_1_path = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_08_contour/test_run_08_contour.json"
file_2_path = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_08_general/test_run_08_general.json"

with open(file_1_path, 'r') as f1, open(file_2_path, 'r') as f2:
    data_1 = json.load(f1)
    data_2 = json.load(f2)

dict_questions_as_id_1 = {
    item['question'] for item in data_1
}

for item in data_2:
    question = item['question']
    if question not in dict_questions_as_id_1:
        print(f"Question not found in file 1: {question}")
    else:
        print(f"Question found in both files: {question}")
