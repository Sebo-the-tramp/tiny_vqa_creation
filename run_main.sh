#!/bin/bash

if [ -d "/data0/sebastian.cavada/datasets/simulations_v3" ]; then
    source "/data0/sebastian.cavada/.telegram_bot.env"
    BASE_PATH="/data0/sebastian.cavada/datasets/simulations_v3/dl3dv"
    DESTINATION_SIMULATION_PATH="/data0/sebastian.cavada/datasets/physbench/simulations_v3"
else
    source "/home/it4i-thvu/seb_dev/.telegram_bot.env"
    BASE_PATH="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v3/dl3dv"
    DESTINATION_SIMULATION_PATH="/scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation_v3"
fi

cd answering_questions

# This are the runs I need to create

GENERAL_RUN_COUNT=06

# # 10K general
# python main_parallel.py --simulation_path ${BASE_PATH} \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --export_format json --run_name "run_${GENERAL_RUN_COUNT}_10K_general"

# 1K soft
python main_parallel.py --simulation_path ${BASE_PATH}/yms-variations/soft \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_soft"\
    --n_scenes 100 --slope 2

# 1K medium
python main_parallel.py --simulation_path ${BASE_PATH}/yms-variations/medium \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_medium" \
    --n_scenes 100 --slope 2

# 1K stiff
python main_parallel.py --simulation_path ${BASE_PATH}/yms-variations/stiff \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_stiff" \
    --n_scenes 100 --slope 2    

# 1K slope 1
python main_parallel.py --simulation_path ${BASE_PATH} \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_slope_1" \
    --n_scenes 100 --slope 1

# 1K slope 2
python main_parallel.py --simulation_path ${BASE_PATH} \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_slope_2" \
    --n_scenes 100 --slope 2

# 1K slope 4
python main_parallel.py --simulation_path ${BASE_PATH} \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_slope_4" \
    --n_scenes 100 --slope 4

# 1K roi circling
python main_parallel.py --simulation_path ${BASE_PATH} \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_roi_circling" \
    --augmentation "roi_circling" --slope 2 --n_scenes 100

# 1K masking
python main_parallel.py --simulation_path ${BASE_PATH} \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_masking" \
    --augmentation "masking" --slope 2 --n_scenes 100

# 1K scene context
python main_parallel.py --simulation_path ${BASE_PATH} \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_scene_context" \
    --augmentation "scene_context" --slope 2 --n_scenes 100

# 1K textual context
python main_parallel.py --simulation_path ${BASE_PATH} \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --export_format json --run_name "run_${GENERAL_RUN_COUNT}_1K_textual_context" \
    --augmentation "textual_context" --slope 2 --n_scenes 100


## OFFICIAL RUNS DONE
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

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="VQA_creation_done" >/dev/null &