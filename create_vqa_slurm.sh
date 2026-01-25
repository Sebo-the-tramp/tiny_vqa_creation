#!/bin/bash
#SBATCH -A EU-25-92
#SBATCH -p qgpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -t 0:30:00
#SBATCH -J interactive_gpu

source "/home/it4i-thvu/seb_dev/.telegram_bot.env"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="🚀 GPU session started for GENERAL_SMALL_MODELS on $(hostname) at $(date)" >/dev/null &

source /mnt/proj1/eu-25-92/tiny_vqa_creation/.venv/bin/activate

./run_main.sh

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="✅ GPU session completed for GENERAL_SMALL_MODELS different CHMOD on $(hostname) at $(date)" >/dev/null &