
# !/bin/bash

# syncing with Karo for test.json
# simulation folder on Karo:
# /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation

# I need to copy from let's say test_karo.json to test.json locally

test_file_to_sync="test_run_05_10K.json"
val_answer_run_05_full="val_answer_run_05_10K.json"

rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/$file_to_sync ./output
rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/val_answer_run_05_full.json ./output/val_answer_run_05_10K.json

sed -i "s#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#/data0/sebastian.cavada/datasets/simulations_v3g" ./output/$test_file_to_sync