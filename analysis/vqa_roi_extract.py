import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import sys

RUN_NAME = "run_28_general"
VQA_SET = sys.argv[1] if len(sys.argv) > 1 else "150K"
results_json = Path(f"../output/{RUN_NAME}/test_{RUN_NAME}_{VQA_SET}.json")
answers_json = Path(f"../output/{RUN_NAME}/val_answer_{RUN_NAME}.json")

SIM_ROOT = "../../physics-sim/output/sims/"

def cleanup_sim_id(sim_id):
    return sim_id.replace("/scratch/project/eu-25-92/composite_physics/dataset/simulation_", "")

def load_sim(sim_id):
    with Path(SIM_ROOT + sim_id).open("r", encoding="utf-8") as f:
        sim_data = json.load(f)

    sim_properties = {
        "num_objects": len(sim_data["objects"]),
        "objects": {},
        "objects_sims": []
    }

    for o_id in sim_data["objects"]:
        sim_properties["objects"][o_id] = {
            "name": sim_data["objects"][o_id]["description"]["object_name"],
            "props": sim_data["objects"][o_id]["props"],
            "scale": sim_data["objects"][o_id]["scale"],
            "volume": sim_data["objects"][o_id]["volume"],
            "material": sim_data["objects"][o_id]["description"]["material"],
            "mass": sim_data["objects"][o_id]["mass"]
        }

    sim_properties["objects_sims"] = np.unique([sim_properties["objects"][o_id]["props"]["name"] for o_id in sim_properties["objects"]]).tolist()

    return sim_properties


with results_json.open("r", encoding="utf-8") as f:
    data = json.load(f)
run_test_data = pd.DataFrame(data).set_index("idx")

simulation_props_path = Path(f"output/{RUN_NAME}/{VQA_SET}/sim_properties.json")
if simulation_props_path.exists():
    print(f"Loading simulation properties from: {simulation_props_path}")
    with simulation_props_path.open("r", encoding="utf-8") as f:
        sims_props = json.load(f)
else:
    sims_props = {}
    for i, row in tqdm(run_test_data.iterrows(), total=len(run_test_data), desc="Simulation loop"):
        sim_id = cleanup_sim_id(row["simulation_id"])

        if sim_id in sims_props:
            continue

        sim_props = load_sim(sim_id)
        sims_props[sim_id] = sim_props

    json.dump(sims_props, simulation_props_path.open("w", encoding="utf-8"), indent=4)
    print(f"Saved simulation properties to: {simulation_props_path}")
print(f"{len(sims_props)} unique simulations.")


# ##################
with answers_json.open("r", encoding="utf-8") as f:
    data = json.load(f)
answers_data = pd.DataFrame(data).set_index("idx")

run_data = run_test_data.merge(answers_data, left_on="idx", right_index=True, how="left")

vqa_roi = []
for idx, row in tqdm(run_data.iterrows(), total=len(run_data), desc="Question loop"):
    sim_id = cleanup_sim_id(row["simulation_id"])
    assert sim_id in sims_props, f"Simulation ID {sim_id} not found in loaded simulations."
    sim_props = sims_props[sim_id]
    objects_to_id = {sim_props["objects"][o_id]["name"]: o_id for o_id in sim_props["objects"]}

    question = row["question"]

    # Capture in 2nd line, all quoted objects in the question (e.g., "red cube") and answers (e.g., "A. red cube") using regex
    pattern = r'.*\n[^"]*"([^"\\]*(?:\\.[^"\\\n]*)*)"'
    question_objects = re.findall(pattern, question)

    # Capture correct answer
    correct_answer = row["answer"]
    pattern = rf"\n{correct_answer}\. (.+?)(?=\n|$)"
    answers = re.findall(pattern, question)

    # Extract objects IDs for question and answers IF they are actual objects in the simulation
    question_objects_ids = [objects_to_id[obj] for obj in question_objects if obj in objects_to_id]
    answers_objects_ids = [objects_to_id[obj] for obj in answers if obj in objects_to_id]
    roi_objects_ids = sorted(question_objects_ids + answers_objects_ids)

    for o_id in roi_objects_ids:
        # assert o_id in sim_props["objects"], f"Object ID {o_id} not found in simulation properties for sim_id {sim_id}."
        vqa_roi.append({
            "idx": idx,
            "roi_object": o_id,
            "roi_object_props": sim_props["objects"][o_id],
        })

# vqa_roi_path = results_json.parent / (results_json.stem + "_roi.json")
vqa_roi_path = Path(f"output/{RUN_NAME}/{VQA_SET}/vqa_roi.json")
json.dump(vqa_roi, vqa_roi_path.open("w", encoding="utf-8"), indent=4)
print(f"Saved VQA data for ROI objects to: {vqa_roi_path}")