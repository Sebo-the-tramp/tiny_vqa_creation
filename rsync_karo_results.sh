# !/bin/bash

# syncing with Karo for test.json
# simulation folder on Karo:
# /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation

# I need to copy from let's say test_karo.json to test.json locally



# should sync back and forth the results
# rsync -av --dry-run -e "ssh -i ~/.ssh/id_rsa_karolina" it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/$RUN_NAME/results_$RUN_NAME ./output/$RUN_NAME/results_$RUN_NAME/
# rsync -av --dry-run -e "ssh -i ~/.ssh/id_rsa_karolina" ./output/$RUN_NAME/results_$RUN_NAME/ it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/$RUN_NAME/results_$RUN_NAME
# rsync -av -e "ssh -i ~/.ssh/id_rsa_karolina" ./output/ it4i-thvu@login3.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/


# only copy results_general_10K

COUNT_RUN=09

# RUN_NAME="run_${COUNT_RUN}_general"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_10K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json

# RUN_NAME="run_${COUNT_RUN}_soft"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_1K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json

    
# RUN_NAME="run_${COUNT_RUN}_medium"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_1K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json


# RUN_NAME="run_${COUNT_RUN}_stiff"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_1K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json


# RUN_NAME="run_${COUNT_RUN}_roi_circling_no_text"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_10K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/mnt/proj1/eu-25-92/tiny_vqa_creation/data/simulation_v3_augmented#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json
# python ./utils/rename_path.py ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# RUN_NAME="run_${COUNT_RUN}_roi_circling_yes_text"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_10K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/mnt/proj1/eu-25-92/tiny_vqa_creation/data/simulation_v3_augmented#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json
# python ./utils/rename_path.py ./output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# RUN_NAME="run_${COUNT_RUN}_contour"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_1K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/mnt/proj1/eu-25-92/tiny_vqa_creation/data/simulation_v3_augmented#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json


# RUN_NAME="run_${COUNT_RUN}_scene_context"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_1K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json


# RUN_NAME="run_${COUNT_RUN}_textual_context"
# cp ./output/$RUN_NAME/test_${RUN_NAME}_1K.json ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json


# copy results from local to karo
rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" \
  --include="*run_09_*/***" \
  --exclude="*" \
  ./output/ \
  it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/ \


# # # copy results from karo to local
# rsync -avz --dry-run -e "ssh -i ~/.ssh/id_rsa_karolina" \
#   --include="*/" \
#   --include="*run_09_general/**" \
#   --exclude="*" \
#   it4i-thvu@login3.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/ \
#   ./output/

# https://rank.opencompass.org.cn/leaderboard-multimodal


# rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" ./output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json \
#     it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/$RUN_NAME/test_${RUN_NAME}_karo_1K.json
