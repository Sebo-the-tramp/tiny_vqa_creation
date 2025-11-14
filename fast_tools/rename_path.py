import os
import sys
import json

if len(sys.argv) != 2:
    print("Usage: python rename_path.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'r') as f:
    data = json.load(f)

for record in data:
    if 'file_name' in record:
        file_names = record['file_name']
        updated_file_names = []
        for fn in file_names:
            if("/render/" in fn):
                new_fn = fn.replace("/mnt/proj1/eu-25-92/tiny_vqa_creation/data/simulation_v3_augmented", "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3")
                updated_file_names.append(new_fn)
            else:
                updated_file_names.append(fn)
        record['file_name'] = updated_file_names

with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)
