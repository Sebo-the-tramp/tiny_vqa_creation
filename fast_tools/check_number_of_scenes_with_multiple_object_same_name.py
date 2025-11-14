import os
import ijson
import glob
import re 

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

root_dir = ["/data0/sebastian.cavada/datasets/simulations_v3/dl3dv/random", "/data0/sebastian.cavada/datasets/simulations_v3/dl3dv/random-cam-stationary"]

simulation_roots = root_dir
list_simulations = []

for simulation_root in simulation_roots:
    pattern = os.path.join(simulation_root, '**', 'simulation.json')    

    print("Searching for simulation files with pattern:", pattern)        
    for sim_file in glob.glob(pattern, recursive=True):
        list_simulations.append(sim_file)

list_simulations.sort(key=natural_key)

print(f"Found {len(list_simulations)} simulation files.")


# a fare bene dovrei controllare se nell'oggetto della domanda ci sono oggetti nella simulazione con lo stesso nome

simulation_cache = {}

for sim_file in list_simulations:
    with open(sim_file, 'r') as f:
        # print(f"Processing simulation file: {sim_file}")
        # iterate over each key-value pair in "objects"
        duplicated_names = {}
        for key, obj in ijson.kvitems(f, 'objects'):
            name = obj.get('description', {}).get('object_name')
            if name:
                # print(f"Object {key}: {name}")
                if name in duplicated_names:
                    duplicated_names[name].append(key)
                else:
                    duplicated_names[name] = [key]

        # Store the duplicated names for this simulation
        simulation_cache[sim_file] = duplicated_names
        for name, keys in duplicated_names.items():
            if len(keys) > 1:
                print(f"Simulation file {sim_file} has multiple objects with the same name '{name}': {keys}")

print('number of simulations with duplicated object names:')
count = 0
for sim_file, duplicated_names in simulation_cache.items():
    has_duplicates = any(len(keys) > 1 for keys in duplicated_names.values())
    if has_duplicates:
        count += 1
        print(f"- {sim_file}")
print(f"Total: {count}")
print("Percentage:", (count / len(list_simulations)) * 100)