# folder dl3dv-counterfact
if [ -d "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4" ]; then
    echo "Directory exists. I AM on KARO"
    cd ./answering_questions
    python augment_kinematics_v3.py /scratch/project/eu-25-92/composite_physics/dataset/simulation_v4 --max-workers 128
fi

if [ -d "/data0/sebastian.cavada/datasets/simulations_v3" ]; then
    echo "Directory exists. I AM on CavadaLAB"
    cd ./answering_questions
    python augment_kinematics_v3.py /data0/sebastian.cavada/datasets/simulations_v4 --max-workers 36
fi

# folder dl3dv-counterfact
if [ -d "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4" ]; then
    echo "Directory exists. I AM on KARO"
    cd ./answering_questions
    python augment_kinematics_v3.py /scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv-counterfact --max-workers 128
fi

if [ -d "/data0/sebastian.cavada/datasets/simulations_v3" ]; then
    echo "Directory exists. I AM on CavadaLAB"
    cd ./answering_questions
    python augment_kinematics_v3.py /data0/sebastian.cavada/datasets/simulations_v4/dl3dv-counterfact --max-workers 36
fi
