#!/usr/bin/env bash

BASE_PATH=""
INPUT_FILE="test_run_11_general"

if [ -d "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4" ]; then
    echo "Directory exists. I AM on KARO"
    BASE_INPUT_PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output"
    INPUT_PATH="${BASE_INPUT_PATH}/run_11_general/${INPUT_FILE}_10K.json"
    OUTPUT_PATH="${BASE_INPUT_PATH}/run_11_general_levels/${INPUT_FILE}_levels_karo_5K.json"
fi

if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
    echo "Directory exists. I AM on CavadaLAB"
    BASE_INPUT_PATH="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output"
    INPUT_PATH="${BASE_INPUT_PATH}/run_11_general/${INPUT_FILE}_10K.json"
    OUTPUT_PATH="${BASE_INPUT_PATH}/run_11_general_levels/${INPUT_FILE}_levels_karo_5K.json"
fi

python generate_levels_questions.py \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --max-questions 1000

cp "${BASE_INPUT_PATH}/run_11_general/val_answer_run_11_general.json" \
   "${BASE_INPUT_PATH}/run_11_general_levels/val_answer_run_11_general_levels.json"