#!/bin/bash

if [ -d "/data0/sebastian.cavada/datasets/simulations_v3" ]; then
    source "/data0/sebastian.cavada/.telegram_bot.env"
    BASE_PATH="/data0/sebastian.cavada/datasets/simulations_v3/dl3dv"
    DESTINATION_SIMULATION_PATH="/data0/sebastian.cavada/datasets/physbench/simulations"
else
    source "/home/it4i-thvu/seb_dev/.telegram_bot.env"
    BASE_PATH="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3/dl3dv"
    DESTINATION_SIMULATION_PATH="/scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation_v3"
fi

cd answering_questions

# This are the runs I need to create

GENERAL_RUN_COUNT=09

# # 10K general # text - no circling
# python main_parallel.py --simulation_path "${BASE_PATH}/random" "${BASE_PATH}/random-cam-stationary" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_general" \
#     --n_scenes 2000

# # 1K roi circling - text
# python main_parallel.py --simulation_path "${BASE_PATH}/random" "${BASE_PATH}/random-cam-stationary" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_roi_circling" \
#     --augmentation "roi_circling" \
#     --n_scenes 2000

# I need another for no text + just roi circling

# # 1K soft
# python main_parallel.py --simulation_path ${BASE_PATH}/yms-variations/soft \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_soft_testttt"\
#     --n_scenes 1000

# # 1K medium
# python main_parallel.py --simulation_path ${BASE_PATH}/yms-variations/medium \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_medium" \
#     --n_scenes 1000

# # 1K stiff
# python main_parallel.py --simulation_path ${BASE_PATH}/yms-variations/stiff \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_stiff" \
#     --n_scenes 1000

# 

# # 1K contour
# python main_parallel.py --simulation_path "${BASE_PATH}/random" "${BASE_PATH}/random-cam-stationary" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_contour_full" \
#     --augmentation "contour" \
#     --n_scenes 2000

# # 1K scene context
# python main_parallel.py --simulation_path "${BASE_PATH}/random" "${BASE_PATH}/random-cam-stationary" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_scene_context" \
#     --augmentation "scene_context" \
#     --n_scenes 2000

# # 1K textual context
# python main_parallel.py --simulation_path "${BASE_PATH}/random" "${BASE_PATH}/random-cam-stationary" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_textual_context" \
#     --augmentation "textual_context" \
#     --n_scenes 2000


### BALANCING THE RUNS ###

# general 10K
# python ./subsample_questions_percentage.py \
#     --count 10000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_general/test_run_${GENERAL_RUN_COUNT}_general.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_general/test_run_${GENERAL_RUN_COUNT}_general_10K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# # 1K roi circling
# python subsample_questions_percentage.py \
#     --count 10000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_roi_circling/test_run_${GENERAL_RUN_COUNT}_roi_circling.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_roi_circling/test_run_${GENERAL_RUN_COUNT}_roi_circling_10K.json \
#     --percentage-map balancing_sub_categories.json \
#     --seed 42 \


# # soft 1K
# python subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_soft/test_run_${GENERAL_RUN_COUNT}_soft.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_soft/test_run_${GENERAL_RUN_COUNT}_soft_1K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42


# # 1K_medium
# python subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_medium/test_run_${GENERAL_RUN_COUNT}_medium.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_medium/test_run_${GENERAL_RUN_COUNT}_medium_1K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# # 1K stiff
# python subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_stiff/test_run_${GENERAL_RUN_COUNT}_stiff.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_stiff/test_run_${GENERAL_RUN_COUNT}_stiff_1K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# # 1K roi circling
# python subsample_questions_percentage_subset.py \
#     --count 10000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_roi_circling_bigger/test_run_${GENERAL_RUN_COUNT}_roi_circling_bigger.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_roi_circling_bigger/test_run_${GENERAL_RUN_COUNT}_roi_circling_bigger_10K.json \
#     --percentage-map balancing_sub_categories.json \
#     --seed 42 \
#     # --subset /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_08_general/test_run_08_general_10K.json \

# # 1K contour
# python subsample_questions_percentage_subset.py \
#     --count 1000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_contour/test_run_${GENERAL_RUN_COUNT}_contour.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_contour/test_run_${GENERAL_RUN_COUNT}_contour_1K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# 1K scene context
# python subsample_questions_percentage_subset.py \
#     --count 1000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_scene_context/test_run_${GENERAL_RUN_COUNT}_scene_context.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_scene_context/test_run_${GENERAL_RUN_COUNT}_scene_context_1K_test.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# python ../utils/check_questions_have_answers.py --question-path ../output/run_${GENERAL_RUN_COUNT}_scene_context/test_run_${GENERAL_RUN_COUNT}_scene_context_1K.json \
#     --answer-path /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_08_general/test_run_08_general_10K.json

# # 1K textual context
# python subsample_questions_percentage_subset.py \
#     --count 1000 \
#     --input ../output/run_${GENERAL_RUN_COUNT}_textual_context/test_run_${GENERAL_RUN_COUNT}_textual_context.json \
#     --output ../output/run_${GENERAL_RUN_COUNT}_textual_context/test_run_${GENERAL_RUN_COUNT}_textual_context_1K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# python ../utils/check_questions_have_answers.py --question-path ../output/run_${GENERAL_RUN_COUNT}_textual_context/test_run_${GENERAL_RUN_COUNT}_textual_context_1K.json \
#     --answer-path /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_08_general/test_run_08_general_10K.json


## OFFICIAL RUNS DONE

# # 1K slope 1
# python main_parallel.py --simulation_path ${BASE_PATH} \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_slope_1" \
#     --n_scenes 1000 --slope 1

# # 1K slope 2
# python main_parallel.py --simulation_path ${BASE_PATH} \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_slope_2" \
#     --n_scenes 1000 --slope 2

# # 1K slope 4
# python main_parallel.py --simulation_path ${BASE_PATH} \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_slope_4" \
#     --n_scenes 1000 --slope 4

# python main.py --simulation_path /data0/sebastian.cavada/datasets/simulations/dl3dv \
#     --destination_simulation_path /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation/dl3dv

# python main.py --simulation_path /scratch/project/eu-25-92/composite_physics/dataset/simulation_v3 \
#     --export_format json 

# python main_parallel.py --simulation_path /scratch/project/eu-25-92/composite_physics/dataset/simulation_v3 \
#     --export_format json --run_name run_test_save_config

# python main_parallel.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v3 \
#     --export_format json --run_name test_seed_00

# python main_parallel.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v3 \
#     --export_format json --run_name test_seed_01

# python main.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v3 \
#     --export_format json

# curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
#      -d chat_id="${TELEGRAM_CHAT_ID}" \
#      --data-urlencode text="VQA_creation_done" >/dev/null &