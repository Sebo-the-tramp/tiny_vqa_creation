# source .venv/bin/activate

cd answering_questions
# python main.py --simulation_path /data0/sebastian.cavada/datasets/simulations/dl3dv \
#     --destination_simulation_path /scratch/project/eu-25-92/composite_physics/dataset/physbench/simulation/dl3dv

# python main.py --simulation_path /scratch/project/eu-25-92/composite_physics/dataset/simulation_v3 \
#     --export_format json

# python main_parallel.py --simulation_path /scratch/project/eu-25-92/composite_physics/dataset/simulation_v3 \
#     --export_format json

python main_parallel.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v3 \
    --export_format json

# python main.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v3 \
#     --export_format json