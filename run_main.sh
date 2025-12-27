#!/bin/bash

if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
    source "/data0/sebastian.cavada/.telegram_bot.env"
    BASE_PATH="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv"
    BASE_PATH_CF="/data0/sebastian.cavada/datasets/simulations_v3/dl3dv-counterfact"
    DESTINATION_SIMULATION_PATH="/data0/sebastian.cavada/datasets/physbench/simulations"
else
    source "/home/it4i-thvu/seb_dev/.telegram_bot.env"
    BASE_PATH="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv"
    BASE_PATH_CF="/scratch/project/eu-25-92/composite_physics/datasets/simulations_v4/dl3dv-counterfact"
    DESTINATION_SIMULATION_PATH="/scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation_v4"
fi

cd answering_questions

# This are the runs I need to create

GENERAL_RUN_COUNT=11

# # 10K general # text - no circling
# python main_parallel.py --simulation_path "${BASE_PATH}/random/" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_general_test_persistance" \
#     --include_categories "persistence" \
#     --n_scenes 700

####################

# 10K general # text - no circling
# python main_parallel.py --simulation_path "${BASE_PATH}/random/" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_general" \
#     --n_scenes 1400 \
#     --exclude_simulations_file "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/answering_questions/problematic_paths.txt"

# python ./subsample_questions_percentage.py \
#     --count 10000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_general/test_run_${GENERAL_RUN_COUNT}_general.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_general/test_run_${GENERAL_RUN_COUNT}_general_10K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# RUN_NAME="run_${GENERAL_RUN_COUNT}_general"
# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# -------------------------------------------------------------
# # 10K general - yms variations 
# RUN_NAME="run_${GENERAL_RUN_COUNT}_general_yms-variations"

# # python main_parallel.py --simulation_path "${BASE_PATH}/yms-variations/" \
# #     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
# #     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_general_yms-variations" \
# #     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 10000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# cp ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json ../output/${RUN_NAME}/test_${RUN_NAME}_karo_10K.json


# -------------------------------------------------------------
# 10K general 
# python main_parallel.py --simulation_path "${BASE_PATH}/random-cam-stationary/" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_general_cam-stationary" \
#     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 10000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_general/test_run_${GENERAL_RUN_COUNT}_general_cam-stationary.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_general/test_run_${GENERAL_RUN_COUNT}_general_cam-stationary_10K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# RUN_NAME="run_${GENERAL_RUN_COUNT}_general"
# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_cam-stationary_karo_10K.json
# sed -i "s#/data0/sebastian.cavada/datasets/simulations_v3#/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3#g" ../output/$RUN_NAME/test_${RUN_NAME}_cam-stationary_karo_10K.json


# -------------------------------------------------------------
# rsync -avz -e "ssh -i ~/.ssh/id_rsa_karolina" \
#   --include="*run_${GENERAL_RUN_COUNT}_*/***" \
#   --exclude="*" \
#   ../output/ \
#   it4i-thvu@login2.karolina.it4i.cz:/mnt/proj1/eu-25-92/tiny_vqa_creation/output/ \

# -------------------------------------------------------------
# COUNTERFACTUALS 
# -------------------------------------------------------------

# -------------------------------------------------------------
1K general # text counterfactual shift
python main_parallel_counterfactual.py --simulation_path "${BASE_PATH_CF}/shift-x" "${BASE_PATH_CF}/shift-z" \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_shift" \
    --counterfactual_type "shift" \
    --n_scenes 1000
    
# -------------------------------------------------------------
# 1K general # text counterfactual gravity
python main_parallel_counterfactual.py --simulation_path "${BASE_PATH_CF}/low-gravity" \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_gravity" \
    --counterfactual_type "gravity" \
    --n_scenes 1000

# -------------------------------------------------------------
# 1K general # text counterfactual gravity
python main_parallel_counterfactual.py --simulation_path "${BASE_PATH_CF}/2xsmaller" \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_smaller" \
    --counterfactual_type "volume" \
    --n_scenes 1000

# -------------------------------------------------------------
# ABLATION STUDYs
# -------------------------------------------------------------

# -------------------------------------------------------------
# 1K roi circling - no text

# RUN_NAME="run_${GENERAL_RUN_COUNT}_roi_circling_no_text"

# # python main_parallel.py --simulation_path "${BASE_PATH}/random" \
# #     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
# #     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_roi_circling_no_text" \
# #     --augmentation "roi_circling_no_text" \
# #     --include_categories "material_understanding" \
# #     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json

# # # -------------------------------------------------------------
# # # 1K roi circling - layout position - no text 

# RUN_NAME="run_${GENERAL_RUN_COUNT}_roi_circling_no_text_layout_position"

# # python main_parallel.py --simulation_path "${BASE_PATH}/random" \
# #     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
# #     --export_format json --run_name "${RUN_NAME}" \
# #     --augmentation "roi_circling_no_text_layout_position" \
# #     --include_categories "material_understanding" \
# #     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# # # -------------------------------------------------------------
# # # 1K roi circling - text

# RUN_NAME="run_${GENERAL_RUN_COUNT}_roi_circling_text"

# # python main_parallel.py --simulation_path "${BASE_PATH}/random" \
# #     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
# #     --export_format json --run_name "${RUN_NAME}" \
# #     --augmentation "roi_circling_text" \
# #     --include_categories "material_understanding" \
# #     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# # # -------------------------------------------------------------
# # # 1K roi circling - layout position - text

# RUN_NAME="run_${GENERAL_RUN_COUNT}_roi_circling_text_layout_position"

# # python main_parallel.py --simulation_path "${BASE_PATH}/random" \
# #     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
# #     --export_format json --run_name "${RUN_NAME}" \
# #     --augmentation "roi_circling_text_layout_position" \
# #     --include_categories "material_understanding" \
# #     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json

# # # -------------------------------------------------------------
# # # 1K roi circling - layout position - text

# RUN_NAME_PREVIOUS="run_${GENERAL_RUN_COUNT}_roi_circling_no_text"
# RUN_NAME="run_${GENERAL_RUN_COUNT}_black"

# image_path="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv/common/black.png"

# mkdir -p ../output/${RUN_NAME}
# cp ../output/${RUN_NAME_PREVIOUS}/test_${RUN_NAME_PREVIOUS}_10K.json ../output/${RUN_NAME}/test_${RUN_NAME}_1K.json
# sed -E -i "s|\"[^\"]+\.png\"|\"${image_path}\"|g" ../output/${RUN_NAME}/test_${RUN_NAME}_1K.json

# -------------------------------------------------------------
# Send Telegram notification when done

# curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
#      -d chat_id="${TELEGRAM_CHAT_ID}" \
#      --data-urlencode text="VQA_creation_done" >/dev/null &