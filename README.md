Tiny VQA Deterministic
======================

This repository builds a deterministic VQA (visual question answering) dataset from
physics simulations. It parses simulation metadata, generates question/answer pairs
from category-specific logic, optionally augments images, and exports datasets as JSON.

If you need guidance, ask; in general, you will find everything here.
Some refactoring will be needed, and I will do it in those days.

Overview
--------
- Input: simulation folders containing `simulation.json`, `simulation_kinematics.json`,
  and rendered frames.
- Logic: category modules in `answering_questions/categories/` compute answers based
  on the simulation and question templates.
- Output: `output/<run_name>/test_<run_name>.json` (questions) and
  `output/<run_name>/val_answer_<run_name>.json` (answers), plus a config snapshot.

> [!NOTE]
> Recent Run Summary
> ------------------
> RUN SUMMARY: questions=84850 wall=374.12s cpu=4.80s rss=867008KB
>
> That run clocks in at ~6m 14s for 84,850 questions, down from ~35 minutes on the
> previous version (around a 5.6x speedup). Not bad.

Repository Layout
-----------------
- `answering_questions/`
  - `main_parallel.py`: main dataset generation entry point for standard runs.
  - `main_parallel_counterfactual.py`: generation for counterfactual runs.
  - `categories/`: per-category question generation and answer logic.
  - `utils/`: shared helpers (config, geometry, augmentation, saving).
- `fast_tools/`: utilities to inspect, debug, and sanity-check outputs.
- `run_main.sh`: example run script (multiple blocks, mostly commented).
- `run_augment.sh`: batch augmentation driver for simulation folders.
- `simple_vqa*.json`: question templates.

Expected Simulation Folder Structure
------------------------------------
Each simulation is assumed to look like:

```
<simulation_root>/
  .../seed-XYZ_.../
    simulation.json
    simulation_kinematics.json
    render/
      000000.png
      000001.png
      ...
    instances/
      000000.png
      ...
```

Notes:
- `render/` frames are referenced by index in the questions.
- `instances/` frames are used for ROI-circling augmentation.
- Counterfactual runs expect `dl3dv-counterfact/<variant>/...` folders and map them
  back to the original `dl3dv` seed folder.

> [!NOTE]
> For faster runs, create a minimized simulation folder by running:
> ```bash
> python minify_simulation_v1.py /data0/sebastian.cavada/datasets/simulations_v4/dl3dv \
>   --input-name simulation_kinematics.json
> ```

Install / Setup (uv)
--------------------
Python 3.11 is required (`pyproject.toml`).

1) Create the virtual environment and install dependencies with uv:

```bash
uv venv .venv
uv pip install -e .
```

2) Activate the environment (optional if you always use `uv run`):

```bash
source .venv/bin/activate
```

3) Additional dependencies used in code (not listed in `pyproject.toml` yet):
- `opencv-python` (used in `utils/augment_VQA.py`)
- `numpy`
- `matplotlib` (used in `fast_tools/check_VQA.py`)

You can install them with uv:

```bash
uv pip install opencv-python numpy matplotlib
```

4) Verify config defaults:
- `answering_questions/utils/json/config.json` controls sampling, thresholds, and
  other physics-related constants.

How the Repository Works
------------------------
1) Templates
   - Question templates live in `simple_vqa.json` and the counterfactual variants.
   - Each template belongs to a category and sub-category.

2) Category Resolvers
   - Each category module exposes functions to get a handler by question key.
   - Example: `categories/mechanics/mechanics.py` routes a question key to a handler.

3) Main Pipeline
   - `main_parallel.py` loads templates and dispatches work across simulations using
     a `ProcessPoolExecutor`.
   - Each worker loads the same templates and processes one simulation at a time.
   - The dataset entries are normalized and saved in `utils/saving_utils.py`.

4) Augmentation
   - Image augmentation happens in `utils/augment_VQA.py`.
   - Supported augmentation modes include:
     - `roi_circling_text`
     - `roi_circling_no_text`
     - `roi_circling_text_layout_position`
     - `roi_circling_no_text_layout_position`
     - `ablation`
     - `grounding_physics`

5) Outputs
   - Questions: `output/<run_name>/test_<run_name>.json`
   - Answers: `output/<run_name>/val_answer_<run_name>.json`
   - Config snapshot: `output/<run_name>/test_<run_name>_config_used.json`

How to Run (Standard)
---------------------
Minimal example:

```bash
cd answering_questions
python main_parallel.py \
  --simulation_paths /path/to/simulations/dl3dv/random \
  --destination_simulation_path /path/to/simulations \
  --run_name run_001_general \
  --n_scenes 100 \
  --image_output path \
  --seed 1337
```

Key arguments:
- `--simulation_paths`: one or more roots; globbing finds `**/simulation.json`.
- `--destination_simulation_path`: used to build paths for image references.
- `--include_categories`: optional list to filter categories.
- `--exclude_question_ids`: skip specific question ids.
- `--exclude_simulations_file`: list of simulation folders to skip.
- `--augmentation`: apply image modifications (see Augmentation).

How to Run (Counterfactual)
---------------------------
Counterfactual runs use a different template and link to the original seed.

```bash
cd answering_questions
python main_parallel_counterfactual.py \
  --simulation_path /path/to/dl3dv-counterfact/shift-x /path/to/dl3dv-counterfact/shift-z \
  --destination_simulation_path /path/to/simulations \
  --run_name run_001_counterfactual_shift \
  --counterfactual_type shift \
  --n_scenes 100 \
  --image_output path \
  --seed 1337
```

How to Use the Provided Run Scripts
-----------------------------------
- `run_main.sh` contains example runs with environment-specific paths and
  commented blocks. It is the best reference for full production runs.
- `run_augment.sh` runs the kinematics augmentation for base and counterfactual
  datasets on specific machines.

How to Generate Your Own Run Script
-----------------------------------
Create a copy of `run_main.sh` and keep only the blocks you need:

```bash
cp run_main.sh run_my_experiment.sh
chmod +x run_my_experiment.sh
```

Then edit:
- `BASE_PATH`, `BASE_PATH_CF`, `DESTINATION_SIMULATION_PATH`
- `GENERAL_RUN_COUNT`
- The specific Python block you want to execute.

For a minimal script:

```bash
#!/bin/bash
set -euo pipefail

BASE_PATH="/path/to/simulations/dl3dv"
DESTINATION_SIMULATION_PATH="/path/to/simulations"
RUN_NAME="run_001_general"

cd answering_questions
python main_parallel.py \
  --simulation_paths "${BASE_PATH}/random" \
  --destination_simulation_path "${DESTINATION_SIMULATION_PATH}" \
  --export_format json \
  --run_name "${RUN_NAME}" \
  --n_scenes 1000 \
  --image_output path
```

Debugging
---------
General tips:
- Use small `--n_scenes` and `--include_categories` for quick iteration.
- Set `--verbose` to log per-question behavior.
- For a single simulation, point `--simulation_paths` to a specific seed folder.

Useful tools:
- `fast_tools/check_VQA.py`: visualize questions, images, and answers.
- `fast_tools/check_VQA_with_answers.py`: checks consistency with answers.
- `fast_tools/check_simulations_correctness.py`: sanity checks for sim data.

Clean Commits (Formatting / Lint)
---------------------------------
Ruff is recommended for quick formatting and linting before a commit.

```bash
uv pip install ruff
ruff format answering_questions fast_tools
ruff check answering_questions fast_tools
```

If you want pre-commit hooks:

```bash
uv pip install pre-commit
pre-commit install
```

Then run:

```bash
pre-commit run --all-files
```

VSCode `launch.json` (example)
------------------------------
Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "VQA: main_parallel",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/answering_questions/main_parallel.py",
      "cwd": "${workspaceFolder}/answering_questions",
      "console": "integratedTerminal",
      "args": [
        "--simulation_paths",
        "/path/to/simulations/dl3dv/random",
        "--destination_simulation_path",
        "/path/to/simulations",
        "--run_name",
        "run_debug",
        "--n_scenes",
        "5",
        "--image_output",
        "path",
        "--verbose"
      ]
    },
    {
      "name": "VQA: main_parallel_counterfactual",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/answering_questions/main_parallel_counterfactual.py",
      "cwd": "${workspaceFolder}/answering_questions",
      "console": "integratedTerminal",
      "args": [
        "--simulation_path",
        "/path/to/simulations/dl3dv-counterfact/shift-x",
        "--destination_simulation_path",
        "/path/to/simulations",
        "--run_name",
        "run_debug_cf",
        "--counterfactual_type",
        "shift",
        "--n_scenes",
        "5",
        "--image_output",
        "path",
        "--verbose"
      ]
    }
  ]
}
```

How to Generate the Output JSONs
--------------------------------
The main scripts export three files to `output/<run_name>/`:
- `test_<run_name>.json` (question payloads)
- `val_answer_<run_name>.json` (answers)
- `test_<run_name>_config_used.json` (config snapshot)

This export is performed by `utils/saving_utils.save_questions_answers_json` inside
`main_parallel.py` and `main_parallel_counterfactual.py`.

Common Issues and Fixes
-----------------------
- Missing images: ensure `render/` exists and paths in `destination_simulation_path`
  match the file layout used in the simulations.
- ROI circling fails: make sure `instances/` masks exist for the same frames.
- Counterfactual mapping fails: verify `dl3dv-counterfact` and `dl3dv` have matching
  seed naming and folder structure.

Contributing / Adding New VQA
-----------------------------
Contributions are welcome. If you want to add a new VQA type:

1) Add or update templates:
   - Extend `simple_vqa.json` or create a new `simple_vqa_*.json`.
   - Ensure each question has a unique `question_key`, plus `sub_category`.

2) Implement the logic:
   - Add a handler in the appropriate category module under
     `answering_questions/categories/`.
   - If it's a new category, add a new module and wire it into the resolver maps in
     `answering_questions/main_parallel.py` (and
     `answering_questions/main_parallel_counterfactual.py` if needed).

3) Connect to the resolver:
   - Add a mapping in `get_function_by_name_<category>` and, if needed, the
     `get_result_by_name_<category>` counterpart.

4) Validate locally:
   - Run a small job with `--n_scenes 5` and `--include_categories` for the new
     category.
   - Use `fast_tools/check_VQA.py` to visualize questions and image paths.

5) Use pre-commit before opening a PR:
   - Install and enable the hooks (see "Clean Commits (Formatting / Lint)").
   - Run `pre-commit run --all-files` to catch lint/format issues.

Status / Notes
--------------
- Some refactoring is needed; I will do it in those days.
- PRs are welcome, and optimization is much needed.
