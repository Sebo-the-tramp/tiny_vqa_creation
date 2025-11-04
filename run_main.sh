source "/home/it4i-thvu/seb_dev/.telegram_bot.env"

cd answering_questions
# python main.py --simulation_path /data0/sebastian.cavada/datasets/simulations/dl3dv \
#     --destination_simulation_path /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation/dl3dv

python main.py --simulation_path /scratch/project/eu-25-92/composite_physics/dataset/simulation_v3 \
    --export_format json --run_name run_05_full

# python main_parallel.py --simulation_path /scratch/project/eu-25-92/composite_physics/dataset/simulation_v3 \
#     --export_format json

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="VQA_creation_done" >/dev/null &

# python main_parallel.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v2 \
#     --export_format json

# python main.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v3 \
#     --export_format json