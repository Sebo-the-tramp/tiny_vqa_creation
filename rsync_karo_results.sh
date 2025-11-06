# !/bin/bash

# syncing with Karo for test.json
# simulation folder on Karo:
# /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation

# I need to copy from let's say test_karo.json to test.json locally

RUN_NAME="results_run_05_10K"

# should sync back and forth the results
# rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" it4i-thvu@login3.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/$RUN_NAME ./output/$RUN_NAME
rsync -avzu -e "ssh -i ~/.ssh/id_rsa_karolina" ./output/$RUN_NAME it4i-thvu@login3.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/$RUN_NAME