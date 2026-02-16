#!/bin/bash

set -euo pipefail

source ../tiny_vqa_creation/.venv/bin/activate

GENERAL_RUN_COUNT=26
PER_OBJECT_COUNT=300
MATERIAL_SUBSAMPLE_COUNT=2000
MATERIAL_OBJECTS_PER_COUNT=300

# Select creations by uncommenting names below.
# This is the only place where toggling is needed.
SELECTED_CREATIONS=(
    # GENERAL
    "general_generate"
    # "general_subsample_30k"
    # "general_obj_numbers_10k"
    # "general_yms_variations_10k"

    # ABLATIONS
    # THIS IS FIRST SO THAT WE CAN CREATE THE IMAGEs
    # "ablation_roi_circling_text"

    # "ablation_baseline"

    # "ablation_roi_circling_no_text"
    # "ablation_roi_circling_no_text_layout_position"
    # "ablation_roi_circling_text_layout_position"
    # "ablation_no_roi_no_text_layout_position"
    # "ablation_no_roi_text_layout_position"

    # "ablation_physics_mass_text"
    # "ablation_physics_duration_text"

    # COUNTERFACTUALS
    # "counterfactual_shift"
    # "counterfactual_gravity"
    # "counterfactual_volume"

    # LEVELS
    # "levels_general_5k"
)

if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
    source "/data0/sebastian.cavada/.telegram_bot.env"
    BASE_PATH="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv"
    BASE_PATH_CF="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv-counterfact"
    DESTINATION_SIMULATION_PATH="/data0/sebastian.cavada/datasets/physbench/simulations_augmented"
    CPUS="44"
elif [ -d "/Volumes/Extreme SSD" ]; then
    BASE_PATH="/Volumes/Extreme SSD/simulation_v4/dl3dv"
    BASE_PATH_CF="/Volumes/Extreme SSD/simulation_v4/dl3dv-counterfact"
    DESTINATION_SIMULATION_PATH="/Volumes/Extreme SSD/simulation_v4_augmented"
    CPUS="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"

    if [ -f "${HOME}/.telegram_bot.env" ]; then
        source "${HOME}/.telegram_bot.env"
    fi
else
    source "/home/it4i-thvu/seb_dev/.telegram_bot.env"
    BASE_PATH="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv"
    BASE_PATH_CF="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv-counterfact"
    DESTINATION_SIMULATION_PATH="/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4_augmented"
    CPUS="128"

    if [ -n "${TELEGRAM_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            --data-urlencode text="VQA creation_started" >/dev/null &
    fi
fi

cd answering_questions

MATERIAL_ABLATION_EXCLUDE_IDS=(
    "F_MASS_HEAVIEST_OBJECT"
    "F_MASS_LIGHTEST_OBJECT"
    "F_PHYSICS_PROPERTY_DENSITY_OBJECT_RELATIVE"
    "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR"
    "F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR_NON_TECHNICAL"
    "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST"
    "F_PHYSICS_PROPERTY_YOUNG_MODULUS_HIGHEST_NON_TECHNICAL"
    "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR"
    "F_PHYSICS_PROPERTY_POISSON_RATIO_OBJECT_SIMILAR_NON_TECHNICAL"
    "F_MATERIAL_IDENTIFICATION_SIMILAR_OBJECT"
    "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST"
    "F_PHYSICS_PROPERTY_POISSON_RATIO_HIGHEST_NON_TECHNICAL"
)

run_general_generate() {
    local simulation_paths=("${BASE_PATH}/random/")
    if [ -d "${BASE_PATH}/random-cam-stationary" ]; then
        simulation_paths+=("${BASE_PATH}/random-cam-stationary/")
    fi

    python main_parallel.py --simulation_paths "${simulation_paths[@]}" \
        --destination_simulation_path "${DESTINATION_SIMULATION_PATH}" \
        --run_name "run_${GENERAL_RUN_COUNT}_general" \
        --n_scenes 5000 \
        --exclude_simulations_file "problematic_paths.txt" \
        --n_proc "${CPUS}" \
        --timeit \
        --print_errors

    local run_name="run_${GENERAL_RUN_COUNT}_general"
    ./slice_json.py ../output/${run_name}/test_${run_name}.json 10000
}

run_general_subsample_30k() {
    local run_name="run_${GENERAL_RUN_COUNT}_general"
    python ./subsample_questions_uniform_question_id.py \
        --count-per-sub-category 2000 \
        --input "../output/${run_name}/test_${run_name}.json" \
        --output "../output/${run_name}/test_${run_name}_30K.json" \
        --seed 42
    
    ./slice_json.py ../output/${run_name}/test_${run_name}.json 10000
}

run_general_obj_numbers_10k() {
    local run_name="run_${GENERAL_RUN_COUNT}_general"
    local run_name_obj="run_${GENERAL_RUN_COUNT}_general_obj_num"

    python ./subsample_questions_numbers.py \
        --input "../output/${run_name}/test_${run_name}.json" \
        --output "../output/${run_name_obj}/test_${run_name_obj}_10K.json" \
        --count 15000 \
        --seed 42

    cp "../output/${run_name_obj}/test_${run_name_obj}_10K.json" "../output/${run_name_obj}/test_${run_name_obj}_karo_10K.json"
    cp "../output/${run_name}/val_answer_${run_name}.json" "../output/${run_name_obj}/val_answer_${run_name_obj}.json"
}

run_general_yms_variations_10k() {
    local run_name="run_${GENERAL_RUN_COUNT}_general_yms-variations"

    python main_parallel.py --simulation_paths "${BASE_PATH}/yms-variations/" \
        --destination_simulation_path "${DESTINATION_SIMULATION_PATH}" \
        --run_name "${run_name}" \
        --n_scenes 3000 \
        --per_object_count "${PER_OBJECT_COUNT}" \
        --n_proc "${CPUS}" \
        --timeit

    python ./subsample_questions_yms_variations.py \
        --input "../output/${run_name}/test_${run_name}.json" \
        --output "../output/${run_name}/test_${run_name}_10K.json" \
        --subcategory-map ./balancing_sub_categories.json \
        --total 10000

    cp "../output/${run_name}/test_${run_name}_10K.json" "../output/${run_name}/test_${run_name}_karo_10K.json"
}

run_material_ablation() {
    local run_suffix="$1"
    local augmentation="$2"
    local run_name="run_${GENERAL_RUN_COUNT}_${run_suffix}"

    python main_parallel.py --simulation_paths "${BASE_PATH}/random" \
        --per_object_count "${PER_OBJECT_COUNT}" \
        --destination_simulation_path "${DESTINATION_SIMULATION_PATH}_modified_images" \
        --run_name "${run_name}" \
        --augmentation "${augmentation}" \
        --include_categories "material_understanding" \
        --exclude_question_ids "${MATERIAL_ABLATION_EXCLUDE_IDS[@]}" \
        --n_proc "${CPUS}" \
        --print_errors

    python ./subsample_questions_percentage.py \
        --count "${MATERIAL_SUBSAMPLE_COUNT}" \
        --input "../output/${run_name}/test_${run_name}.json" \
        --output "../output/${run_name}/test_${run_name}_10K.json" \
        --percentage-map ./balancing_sub_categories_material_only.json \
        --mode "general" \
        --objects-per-count "${MATERIAL_OBJECTS_PER_COUNT}" \
        --soft-objects-per-count \
        --percentages-within-objects \
        --seed 42

    cp "../output/${run_name}/test_${run_name}_10K.json" "../output/${run_name}/test_${run_name}_karo_10K.json"
}

run_ablation_roi_circling_no_text() {
    run_material_ablation "roi_circling_no_text" "roi_circling_no_text"
}

run_ablation_roi_circling_no_text_layout_position() {
    run_material_ablation "roi_circling_no_text_layout_position" "roi_circling_no_text_layout_position"
}

run_ablation_roi_circling_text() {
    run_material_ablation "roi_circling_text" "roi_circling_text"
}

run_ablation_roi_circling_text_layout_position() {
    run_material_ablation "roi_circling_text_layout_position" "roi_circling_text_layout_position"
}

run_ablation_no_roi_no_text_layout_position() {
    run_material_ablation "no_roi_circling_no_text_layout_position" "ablation_no_text_layout_position"
}

run_ablation_no_roi_text_layout_position() {
    run_material_ablation "no_roi_circling_yes_text_layout_position" "ablation_text_layout_position"
}

run_ablation_baseline() {
    run_material_ablation "roi_ablation_baseline" "ablation"
}

run_ablation_physics_mass_text() {
    run_material_ablation "ablation_physics_mass_text" "ablation_physics_mass_text"
}

run_ablation_physics_duration_text() {
    run_material_ablation "ablation_physics_duration_text" "ablation_physics_duration_text"
}

run_counterfactual_shift() {
    python main_parallel_counterfactual_new.py --simulation_paths "${BASE_PATH_CF}/jitter-xy" "${BASE_PATH_CF}/jitter-z" \
        --destination_simulation_path "${DESTINATION_SIMULATION_PATH}" \
        --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_shift" \
        --counterfactual_type "shift" \
        --n_scenes 2000 \
        --n_proc "${CPUS}"
}

run_counterfactual_gravity() {
    python main_parallel_counterfactual_new.py --simulation_paths "${BASE_PATH_CF}/low-gravity" \
        --destination_simulation_path "${DESTINATION_SIMULATION_PATH}" \
        --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_gravity" \
        --counterfactual_type "gravity" \
        --n_scenes 1000 \
        --n_proc "${CPUS}"

    ## ADD conversion to 1OK_karo
}

run_counterfactual_volume() {
    python main_parallel_counterfactual_new.py --simulation_paths "${BASE_PATH_CF}/rescale" \
        --destination_simulation_path "${DESTINATION_SIMULATION_PATH}" \
        --run_name "run_${GENERAL_RUN_COUNT}_counterfactual_smaller" \
        --counterfactual_type "volume" \
        --n_scenes 1000 \
        --n_proc "${CPUS}"
}

run_levels_general_5k() {
    local run_tag="run_${GENERAL_RUN_COUNT}_general"
    local levels_tag="${run_tag}_levels"
    local input_file="test_${run_tag}"
    local base_input_path=""
    local input_path=""
    local output_path=""

    if [ -d "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4" ]; then
        echo "Directory exists. I AM on KARO"
        base_input_path="/mnt/proj1/eu-25-92/tiny_vqa_creation/output"
        input_path="${base_input_path}/${run_tag}/${input_file}_10K.json"
        output_path="${base_input_path}/${levels_tag}/${input_file}_levels_karo_5K.json"
    elif [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
        echo "Directory exists. I AM on CavadaLAB"
        base_input_path="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output"
        input_path="${base_input_path}/${run_tag}/${input_file}.json"
        output_path="${base_input_path}/${levels_tag}/${input_file}_levels_karo_5K.json"
    else
        echo "Directory exists. I AM on local/macOS"
        base_input_path="../output"
        if [ -f "${base_input_path}/${run_tag}/${input_file}_10K.json" ]; then
            input_path="${base_input_path}/${run_tag}/${input_file}_10K.json"
        else
            input_path="${base_input_path}/${run_tag}/${input_file}.json"
        fi
        output_path="${base_input_path}/${levels_tag}/${input_file}_levels_karo_5K.json"
    fi

    if [ ! -f "${input_path}" ]; then
        echo "Levels input file not found: ${input_path}" >&2
        return 1
    fi

    python ./level_difficulty/generate_levels_questions.py \
        --input "${input_path}" \
        --output "${output_path}" \
        --max-questions 1000

    cp "${base_input_path}/${run_tag}/val_answer_${run_tag}.json" \
        "${base_input_path}/${levels_tag}/val_answer_${levels_tag}.json"
}

run_creation() {
    local creation="$1"
    case "${creation}" in
        general_generate) run_general_generate ;;
        general_subsample_30k) run_general_subsample_30k ;;
        general_obj_numbers_10k) run_general_obj_numbers_10k ;;
        general_yms_variations_10k) run_general_yms_variations_10k ;;
        ablation_roi_circling_no_text) run_ablation_roi_circling_no_text ;;
        ablation_roi_circling_no_text_layout_position) run_ablation_roi_circling_no_text_layout_position ;;
        ablation_roi_circling_text) run_ablation_roi_circling_text ;;
        ablation_roi_circling_text_layout_position) run_ablation_roi_circling_text_layout_position ;;
        ablation_no_roi_no_text_layout_position) run_ablation_no_roi_no_text_layout_position ;;
        ablation_no_roi_text_layout_position) run_ablation_no_roi_text_layout_position ;;
        ablation_baseline) run_ablation_baseline ;;
        ablation_physics_mass_text) run_ablation_physics_mass_text ;;
        ablation_physics_duration_text) run_ablation_physics_duration_text ;;
        counterfactual_shift) run_counterfactual_shift ;;
        counterfactual_gravity) run_counterfactual_gravity ;;
        counterfactual_volume) run_counterfactual_volume ;;
        levels_general_5k) run_levels_general_5k ;;
        *)
            echo "Unknown creation name: ${creation}" >&2
            exit 1
            ;;
    esac
}

if [ "${#SELECTED_CREATIONS[@]}" -eq 0 ]; then
    echo "No creations selected. Uncomment at least one entry in SELECTED_CREATIONS." >&2
    exit 0
fi

for creation in "${SELECTED_CREATIONS[@]}"; do
    echo "============================================================"
    echo "Running creation: ${creation}"
    echo "============================================================"
    run_creation "${creation}"
done
