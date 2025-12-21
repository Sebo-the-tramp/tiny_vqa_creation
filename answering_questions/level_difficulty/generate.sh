#!/usr/bin/env bash

if [ -d "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4" ]; then
    echo "Directory exists. I AM on KARO"
    PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output/run_11_general/test_run_10_general_10K.json"
    OUTPUT_PATH="/mnt/proj1/eu-25-92/tiny_vqa_creation/output/run_11_general/"
fi

if [ -d "/data0/sebastian.cavada/datasets/simulations_v4" ]; then
    echo "Directory exists. I AM on CavadaLAB"
    PATH="--input /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_10_general/test_run_10_general_10K.json"
    OUTPUT_PATH="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_10_general/"
fi

python generate_levels_questions.py \
  --input "$PATH" \
  --output "$OUTPUT_PATH"
