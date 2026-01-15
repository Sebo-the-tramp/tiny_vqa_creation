#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="run_17_general"

BASE_DIR="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/${RUN_NAME}"
INPUT_DIR="${BASE_DIR}/results_${RUN_NAME}"
OUTPUT_DIR="${BASE_DIR}/results_${RUN_NAME}_sanitized"

python sanitize_answers.py "${INPUT_DIR}" "${OUTPUT_DIR}"
