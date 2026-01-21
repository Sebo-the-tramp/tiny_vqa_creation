# !/bin/bash

# syncing with Karo for test.json
# simulation folder on Karo:
# /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation

# copy results from karo to local
rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" \
  --include="*/" \
  --include="*run_23**" \
  --exclude="*" \
  it4i-thvu@login3.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/ \
  ./output/

# https://rank.opencompass.org.cn/leaderboard-multimodal
