import os
import json

path = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output"


for experiment in sorted(os.listdir(path)):

    if(experiment != "run_06_general"):
        continue

    experiment_path = os.path.join(path, experiment)
    if not os.path.isdir(experiment_path):
        continue

    # print(f"Processing ./{experiment}/test_{experiment}.json")
    print(f"Processing ./{experiment}/test_{experiment}_10K_copy.json")

    with open(os.path.join(experiment_path, f"test_{experiment}_10K_copy.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

        for question in data:

            file_names = question['file_name']
            wrong_indexes = [i for i, file in enumerate(file_names) if "//render" in file]                        
            count_wrong = len(wrong_indexes)
            file_names = [file for i, file in enumerate(file_names) if i not in wrong_indexes]
            question['file_name'] = file_names

            count_images = len(file_names)
            question['question'] = question['question'].replace("<image>"*count_images, "<image>"*(count_images - count_wrong))

            # if len(question['file_name']) == 12:
            #     print(len(question['file_name']))
            #     print("Fixing 12 images bug")
            #     question['file_name'] = question['file_name'][4:8]
            #     question["question"] = question["question"].replace("<image><image><image><image><image><image><image><image>", "<image><image><image><image>")                

                # question = question.replace('"num_images": 12', '"num_images": 10')

            # with open(os.path.join(experiment_path, f"test_{experiment}_fixed.json"), "a") as f_out:
            #     f_out.write(question)

    print(f"Saving fixed ./{experiment}/test_{experiment}_10K.json")
    with open(os.path.join(experiment_path, f"test_{experiment}_10K.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
