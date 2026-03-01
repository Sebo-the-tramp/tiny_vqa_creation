#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-run_28_}"
MAX_PREFIX_CHARS="${2:--1}"
# OUTPUT_ROOT="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output"
OUTPUT_ROOT="/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
run_dirs=("${OUTPUT_ROOT}/${PREFIX}"*)
shopt -u nullglob

if [ ${#run_dirs[@]} -eq 0 ]; then
  echo "Warning: no run folders found for prefix '${PREFIX}' in ${OUTPUT_ROOT}" >&2
  exit 0
fi

for run_dir in "${run_dirs[@]}"; do
  if [ ! -d "${run_dir}" ]; then
    continue
  fi
  run_name="$(basename "${run_dir}")"
  input_dir="${run_dir}/results_${run_name}"
  output_dir="${run_dir}/results_${run_name}_sanitized"

  if [ ! -d "${input_dir}" ]; then
    echo "Skipping ${run_name}: missing ${input_dir}" >&2
    continue
  fi

  echo "Sanitizing ${run_name}"
  if ! python "${SCRIPT_DIR}/sanitize_answers.py" \
    "${input_dir}" \
    "${output_dir}" \
    --max-prefix-chars "${MAX_PREFIX_CHARS}"; then
    echo "Warning: sanitize failed for ${run_name}; continuing" >&2
    continue
  fi
done
