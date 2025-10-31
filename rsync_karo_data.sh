
# !/bin/bash

# syncing with Karo for test.json
# simulation folder on Karo:
# /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation

# I need to copy from let's say test_karo.json to test.json locally
rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" ./output/test_karo.json it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/test.json