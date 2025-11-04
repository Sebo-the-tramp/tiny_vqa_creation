import os
import json

file_1_path = "../output/test_reproducibility_00.json"
file_2_path = "../output/test_reproducibility_01.json"

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
