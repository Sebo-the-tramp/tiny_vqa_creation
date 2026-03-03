# !/bin/bash

# syncing with Karo for test.json
# simulation folder on Karo:
# /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation

# copy results from karo to local
rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" \
  --include="*run_24_general_yms_variations**" \
  --include="*run_26_general_levels**" \
  --include="*run_28**" \
  --exclude="*" \
  it4i-thvu@login3.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/ \
  ./output_tmp/

# https://rank.opencompass.org.cn/leaderboard-multimodal
