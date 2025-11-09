import os 
import json

root_out_dir = "../output"
if not os.path.exists(root_out_dir):
    root_out_dir = "/data/sebastian.cavada/compositional-physics/tiny_vqa_creation/output"        


metadata_llm_path = "../analysis_example/utils/metadata.json"
with open(metadata_llm_path, 'r') as f:
    metadata = json.load(f)

for run_name in os.listdir(root_out_dir):
    print("-----"*5)
    print(f"Checking run: {run_name}")
    print("-----"*5)
    result_folder = os.path.join(root_out_dir, run_name, f"results_{run_name}")
    if os.path.exists(result_folder):
        print(f"Results folder exists for {run_name}: {result_folder}")
    else:
        print(f"Results folder does NOT exist for {run_name}: {result_folder}")

    files_in_folder = os.listdir(result_folder) if os.path.exists(result_folder) else []
    json_files = [f for f in files_in_folder if f.endswith('.json')]
    model_names = [f.replace("_val.json", "") for f in json_files]

    if not json_files:
        print(f"No JSON result files found in {result_folder}")
        continue    
    
    total_missing = 0
    for model in metadata:
        model_name = model['id']
        if model_name in model_names:
            pass
            # print(f"Model {model_name} results found in {result_folder}")
        else:
            if(model['license'] == ""):
                continue
            print(f"Missing result for {model_name}")
            total_missing += 1

    print(f"Total missing results for {run_name}: {total_missing}")




