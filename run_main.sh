#!/bin/bash

# codex resume 019b8e96-5e52-74c2-826d-482eb2baca4c -> tqdm and everything

if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
    source "/data0/sebastian.cavada/.telegram_bot.env"
    BASE_PATH="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv"
    BASE_PATH_CF="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv-counterfact"
    DESTINATION_SIMULATION_PATH="/data0/sebastian.cavada/datasets/physbench/simulations"
    CPUS="32"
else
    source "/home/it4i-thvu/seb_dev/.telegram_bot.env"
    BASE_PATH="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv"
    BASE_PATH_CF="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv-counterfact"
    DESTINATION_SIMULATION_PATH="/scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation_v4"
    CPUS="128"
fi

cd answering_questions

GENERAL_RUN_COUNT=16

####################

# python main_parallel.py --simulation_path "${BASE_PATH}/random/" "${BASE_PATH}/random-cam-stationary/" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --run_name "run_${GENERAL_RUN_COUNT}_general" \
#     --n_scenes 5000 \
#     --exclude_simulations_file "problematic_paths.txt" \
#     --n_proc $CPUS \
#     --timeit \
# #    --include_categories "mechanics" \

# RUN_NAME="run_${GENERAL_RUN_COUNT}_general"
# python ./subsample_questions_percentage.py \
#     --count 10000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# RUN_NAME="run_${GENERAL_RUN_COUNT}_general_obj_num"
# python ./subsample_questions_numbers.py \
#     --input ../output/run_${GENERAL_RUN_COUNT}_general/test_run_${GENERAL_RUN_COUNT}_general.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --count 15000 \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# -------------------------------------------------------------
# # 10K general - yms variations 
# RUN_NAME="run_${GENERAL_RUN_COUNT}_general_yms-variations"

# python main_parallel.py --simulation_path "${BASE_PATH}/yms-variations/" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --run_name "run_${GENERAL_RUN_COUNT}_general_yms-variations" \
#     --n_scenes 2000

# python ./subsample_questions_yms_variations.py \
#     --input ../output/run_12_general_yms-variations/test_run_12_general_yms-variations.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --subcategory-map ./balancing_sub_categories.json \
#     --total 10000 \

# cp ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json ../output/${RUN_NAME}/test_${RUN_NAME}_karo_10K.json


# -------------------------------------------------------------
# 10K general 
# python main_parallel.py --simulation_path "${BASE_PATH}/random-cam-stationary/" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
#     --run_name "run_${GENERAL_RUN_COUNT}_general_cam-stationary" \
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
# 1K general # text counterfactual shift
python main_parallel_counterfactual.py --simulation_path "${BASE_PATH_CF}/shift-x" "${BASE_PATH_CF}/shift-z" \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_shift" \
    --counterfactual_type "shift" \
    --timeit \
    --n_scenes 2000
    
# -------------------------------------------------------------
# # 1K general # text counterfactual gravity
python main_parallel_counterfactual.py --simulation_path "${BASE_PATH_CF}/low-gravity" \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_gravity" \
    --counterfactual_type "gravity" \
    --n_scenes 1000

# # -------------------------------------------------------------
# 1K general # text counterfactual volume
python main_parallel_counterfactual.py --simulation_path "${BASE_PATH_CF}/2xsmaller" \
    --destination_simulation_path ${DESTINATION_SIMULATION_PATH} \
    --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_smaller" \
    --counterfactual_type "volume" \
    --n_scenes 1000

# -------------------------------------------------------------
# ABLATION STUDYs
# -------------------------------------------------------------

# -------------------------------------------------------------
# 1K roi circling - no text

# RUN_NAME="run_${GENERAL_RUN_COUNT}_roi_circling_no_text"

# python main_parallel.py --simulation_path "${BASE_PATH}/random" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
#     --run_name "run_${GENERAL_RUN_COUNT}_roi_circling_no_text" \
#     --augmentation "roi_circling_no_text" \
#     --include_categories "material_understanding" \
#     --exclude_question_ids "F_MASS_HEAVIEST_OBJECT" "F_MASS_LIGHTEST_OBJECT" "F_PHYSICS_PROPERTY_DENSITY_OBJECT_RELATIVE" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL" "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL" "F_MATERIAL_IDENTIFICATION_SIMILAR_OBJECT" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST" "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL" \
#     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# # # # -------------------------------------------------------------
# # # # 1K roi circling - layout position - no text 

# RUN_NAME="run_${GENERAL_RUN_COUNT}_roi_circling_no_text_layout_position"

# python main_parallel.py --simulation_path "${BASE_PATH}/random" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
#     --run_name "${RUN_NAME}" \
#     --augmentation "roi_circling_no_text_layout_position" \
#     --include_categories "material_understanding" \
#     --exclude_question_ids "F_MASS_HEAVIEST_OBJECT" "F_MASS_LIGHTEST_OBJECT" "F_PHYSICS_PROPERTY_DENSITY_OBJECT_RELATIVE" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL" "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL" "F_MATERIAL_IDENTIFICATION_SIMILAR_OBJECT" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST" "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL" \
#     --n_scenes 700

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

# python main_parallel.py --simulation_path "${BASE_PATH}/random" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
#     --run_name "${RUN_NAME}" \
#     --augmentation "roi_circling_text" \
#     --include_categories "material_understanding" \
#     --exclude_question_ids "F_MASS_HEAVIEST_OBJECT" "F_MASS_LIGHTEST_OBJECT" "F_PHYSICS_PROPERTY_DENSITY_OBJECT_RELATIVE" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL" "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL" "F_MATERIAL_IDENTIFICATION_SIMILAR_OBJECT" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST" "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL" \
#     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json


# # # # -------------------------------------------------------------
# # # # 1K roi circling - layout position - text

# RUN_NAME="run_${GENERAL_RUN_COUNT}_roi_circling_text_layout_position"

# python main_parallel.py --simulation_path "${BASE_PATH}/random" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
#     --run_name "${RUN_NAME}" \
#     --augmentation "roi_circling_text_layout_position" \
#     --include_categories "material_understanding" \
#     --exclude_question_ids "F_MASS_HEAVIEST_OBJECT" "F_MASS_LIGHTEST_OBJECT" "F_PHYSICS_PROPERTY_DENSITY_OBJECT_RELATIVE" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL" "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL" "F_MATERIAL_IDENTIFICATION_SIMILAR_OBJECT" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST" "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL" \
#     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json



# # # -------------------------------------------------------------
# # # 1K roi circling - BASELINE

# RUN_NAME="run_${GENERAL_RUN_COUNT}_ablation_baseline"

# python main_parallel.py --simulation_path "${BASE_PATH}/random" \
#     --destination_simulation_path ${DESTINATION_SIMULATION_PATH}_modified_images \
#     --run_name "${RUN_NAME}" \
#     --include_categories "material_understanding" \
#     --augmentation "ablation" \
#     --exclude_question_ids "F_MASS_HEAVIEST_OBJECT" "F_MASS_LIGHTEST_OBJECT" "F_PHYSICS_PROPERTY_DENSITY_OBJECT_RELATIVE" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL" \
#     "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST" "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL" "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL" "F_MATERIAL_IDENTIFICATION_SIMILAR_OBJECT" \
#     "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST" "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL" \
#     --n_scenes 700

# python ./subsample_questions_percentage.py \
#     --count 1000 \
#     --input ../output/${RUN_NAME}/test_${RUN_NAME}.json \
#     --output ../output/${RUN_NAME}/test_${RUN_NAME}_10K.json \
#     --percentage-map ./balancing_sub_categories_material_only.json \
#     --seed 42

# cp ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/test_${RUN_NAME}_karo_10K.json






# python run_parallel.py --model-size "VLMEval" --run-name "run_11_general" --quantity "10K"


















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
