#!/usr/bin/env bash

BASE_PATH=""
RUN_ID="25"
RUN_TAG="run_${RUN_ID}_general"
LEVELS_TAG="${RUN_TAG}_levels"
INPUT_FILE="test_${RUN_TAG}"

if [ -d "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4" ]; then
    echo "Directory exists. I AM on KARO"
    BASE_INPUT_PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output"
    INPUT_PATH="${BASE_INPUT_PATH}/${RUN_TAG}/${INPUT_FILE}_10K.json"
    OUTPUT_PATH="${BASE_INPUT_PATH}/${LEVELS_TAG}/${INPUT_FILE}_levels_karo_5K.json"
fi

if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
    echo "Directory exists. I AM on CavadaLAB"
    BASE_INPUT_PATH="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
    INPUT_PATH="${BASE_INPUT_PATH}/${RUN_TAG}/${INPUT_FILE}.json"
    OUTPUT_PATH="${BASE_INPUT_PATH}/${LEVELS_TAG}/${INPUT_FILE}_levels_karo_5K.json"
fi

python generate_levels_questions.py \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --max-questions 1000

cp "${BASE_INPUT_PATH}/${RUN_TAG}/val_answer_${RUN_TAG}.json" \
   "${BASE_INPUT_PATH}/${LEVELS_TAG}/val_answer_${LEVELS_TAG}.json"
