import os
import json
import glob

simulation_roots = ["/data0/sebastian.cavada/datasets/simulations_v3/dl3dv/"]

list_simulations = []

for simulation_root in simulation_roots:
    pattern = os.path.join(simulation_root, '**', 'simulation_kinematics.json')    
    print("Searching for simulation files with pattern:", pattern)        
    for sim_file in glob.glob(pattern, recursive=True):
        list_simulations.append(sim_file)

print(f"Found {len(list_simulations)} simulation files.")

for sim_file in list_simulations:
    with open(sim_file, "r") as f:
        simulation_data = json.load(f)

        # check if there is a simulation_data['encoding']['classes']
        if 'encoding' in simulation_data and 'classes' in simulation_data['encoding']:
            # print("Found classes in:", sim_file)
            continue
        else:
            print("No classes found in:", sim_file)