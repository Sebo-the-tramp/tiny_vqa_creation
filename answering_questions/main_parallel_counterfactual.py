import os
import re
import json
import glob
import argparse

import multiprocessing
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from utils.config import set_config
from utils.augment_VQA import get_counterfactual_image_paths

from copy import deepcopy

from utils.saving_utils import (
    save_questions_answers_json,
)
from utils.my_exception import ImpossibleToAnswer
from utils import seed_utils

# Import categories - alphabetically
from categories.spatial_reasoning.spatial_reasoning import (
    get_function_by_name_spatial_reasoning,
)

from categories.mechanics.mechanics import (
    get_function_by_name_mechanics,
)

from categories.material_understanding.material_understanding import (
    get_function_by_name_material_understanding,
)

from categories.persistence.persistence import (
    get_function_by_name_persistence,
)

from categories.viewpoint.viewpoint import (
    get_function_by_name_viewpoint,
)


# Globals initialized in worker processes
QUESTIONS = None
DEST_ROOT = None
VERBOSE = None

ANSI_RED = "\033[91m"
ANSI_ORANGE = "\033[38;5;208m"
ANSI_GREEN = "\033[92m"
ANSI_GREY = "\033[90m"
ANSI_RESET = "\033[0m"


def _init_worker(vqa_path, dest_root, verbose, base_seed, counterfactual_type):
    """Runs once per worker process."""
    import os

    global QUESTIONS, DEST_ROOT, VERBOSE
    counterfactual_type = counterfactual_type.lower()
    counterfactual_file = f"simple_vqa_counterfactual_{counterfactual_type}.json"
    QUESTIONS = read_questions(os.path.join(vqa_path, counterfactual_file))
    DEST_ROOT = dest_root
    VERBOSE = verbose
    seed_utils.seed_everything(base_seed)
    try:
        worker_name = multiprocessing.current_process().name
        worker_idx = int(worker_name.rsplit("_", 1)[-1])
    except Exception:
        worker_idx = 0
    seed_utils.reseed_for_context(f"worker::{worker_idx}")


def _process_one(sim_file, args):
    """Process a single simulation.json path and return its VQA list and stats."""
    try:
        if not os.path.isfile(sim_file):
            if VERBOSE:
                print("Skipping non-file:", sim_file)
            return [], {}
        simulation_id_path = sim_file.replace("simulation.json", "")
        destination_simulation_id_path = os.path.join(DEST_ROOT, simulation_id_path)
        print("Processing simulation:", sim_file)
        simulation_steps_modified = read_simulation(
            os.path.join(simulation_id_path, "simulation_kinematics.json")
        )

        simulation_id_path_og = re.sub(
            r"/dl3dv-counterfact/[^/]+/", "/dl3dv/", simulation_id_path
        )
        # I need to check the folder that contains the original simulation
        base_dir = simulation_id_path_og.split("seed-")[0]
        seed = simulation_id_path_og.split("seed-")[1].split("_")[0]

        matches = glob.glob(base_dir + "seed-" + seed + "_*")
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Original simulation folder not found for {simulation_id_path_og}"
            )
        simulation_id_path_og = matches[0]

        simulation_steps_og = read_simulation(
            os.path.join(simulation_id_path_og, "simulation_kinematics.json")
        )

        return create_vqa(
            QUESTIONS,
            simulation_steps_og,
            simulation_steps_modified,
            sim_file,
            destination_simulation_id_path,
            verbose=VERBOSE,
            config=args,
        )
    except Exception as e:
        # Keep the pool running even if one simulation fails
        # if VERBOSE:
        print("\033[91mWorker error on", simulation_id_path, "->", repr(e), "\033[0m")
        print(e.with_traceback())
        return [], {}


# ----- UTILS FUNCTIONS
def read_questions(vqa_path):
    with open(vqa_path, "r") as f:
        questions = json.load(f)
    return questions


def read_simulation(simulation_path):
    with open(simulation_path, "r") as f:
        simulation_steps = json.load(f)
    return simulation_steps


# ----- FUNCTION TO GET ANSWER FROM SIMULATION
resolver = {
    "spatial_reasoning": get_function_by_name_spatial_reasoning,
    "mechanics": get_function_by_name_mechanics,
    "material_understanding": get_function_by_name_material_understanding,
    "persistence": get_function_by_name_persistence,
    "view_point": get_function_by_name_viewpoint,
}


def get_answer(question_key, question_category):
    return resolver[question_category](question_key)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ----- MAIN VQA CREATION LOGIC
def create_vqa(
    questions,
    simulation_steps_og,
    simulation_steps_modified,
    simulation_id,
    destination_simulation_id_path,
    verbose=False,
    config=None,
):
    seed_utils.reseed_for_context(simulation_id)
    impossible_to_answer = 0

    print("Starting VQA creation...")

    all_vqa = []
    stats = {}

    for category_key, category in questions.items():
        # current category development
        # if (
        #     category_key != "mechanics"
        #     # category_key != "material_understanding"
        #     # category_key != "persistence"
        #     # category_key != "view_point"
        # ):
        #     continue

        if verbose:
            print("###" * 10, f"Processing category: {category_key}", "###" * 10)
            print(f"questions: \n{category}")
            print("###" * 20)

        for question_key, question_data in category.items():
            question_payload = deepcopy(question_data)
            question_payload["_question_key"] = question_key
            question_payload["_simulation_id"] = simulation_id
            sub_category = question_payload.get("sub_category", "unknown_sub_category")
            stats_key = (question_key, sub_category, category_key)
            if stats_key not in stats:
                stats[stats_key] = {
                    "created": 0,
                    "impossible": 0,
                    "errors": 0,
                    "non_interesting": 0,
                }

            fn_to_answer_question_modified_data_factual = get_answer(
                # we remove the first C -> to get the factual version
                question_key[1:],
                category_key,
            )

            try:
                answer_list_modified_data_factual = (
                    fn_to_answer_question_modified_data_factual(
                        simulation_steps_modified,
                        question_payload,
                        destination_simulation_id_path,
                    )
                )
            except ImpossibleToAnswer:
                impossible_to_answer += 1
                stats[stats_key]["impossible"] += 1
                continue
            except Exception:
                stats[stats_key]["errors"] += 1
                continue

            fn_to_check_answer_original_data_cf = get_answer(question_key, category_key)

            try:
                answer_list_original_data_cf = fn_to_check_answer_original_data_cf(
                    simulation_steps_og,
                    simulation_steps_modified,
                    answer_list_modified_data_factual,
                    question_payload,
                    destination_simulation_id_path,
                )
            except ImpossibleToAnswer:
                impossible_to_answer += 1
                stats[stats_key]["impossible"] += 1
                continue
            except Exception:
                stats[stats_key]["errors"] += 1
                continue

            # checking that the counterfactual answer is different from the factual one
            # so we have an interesting question
            correct_answer_counterfactual = answer_list_modified_data_factual[0][1][
                answer_list_modified_data_factual[0][2]
            ]
            correct_answer_factual = answer_list_original_data_cf[0][1][
                answer_list_original_data_cf[0][2]
            ]

            if str(correct_answer_counterfactual) == str(correct_answer_factual):
                print("NOT INTERESTING AT ALL!")
                stats[stats_key]["non_interesting"] += 1
                continue
            else:
                print("INTERESTING!")

            # we need to change the question from factual to counterfactual
            for question_factual, question_counterfactual in zip(
                answer_list_modified_data_factual, answer_list_original_data_cf
            ):
                str_question = question_counterfactual[0]["question"]
                question_factual[0]["question"] = str_question

            for (
                question,
                labels,
                correct_idx,
                imgs_idx,
                world_state,
                resolved_attributes,
            ) in answer_list_modified_data_factual:
                # changing from image_paths to image_paths
                file_names_to_augment = [
                    destination_simulation_id_path + f"render/{int(frame_idx):06d}.png"
                    for frame_idx in imgs_idx
                ]

                # regex to check if in the label we have an image
                pattern = re.compile(r"^\d{6}$")

                for _, label in enumerate(labels):
                    if pattern.match(label):
                        # do a smart replacement
                        new_image_path = (
                            destination_simulation_id_path + f"/render/{label}.png"
                        )
                        file_names_to_augment.append(new_image_path)

                file_names = get_counterfactual_image_paths(
                    file_names_to_augment.copy()
                )

                all_vqa.append(
                    {
                        "scene": simulation_steps_modified["scene"].get(
                            "scene", "unknown_scene"
                        ),
                        "simulation_id": simulation_id,
                        "question": question,
                        "category": category_key,
                        "sub_category": question_payload["sub_category"],
                        "question_key": question_key,
                        "image_paths": file_names,
                        "labels": labels,
                        "answer_index": correct_idx,
                        "mode": (
                            "image-only"
                            if question["task_splits"] == "single"
                            else "general"
                        ),
                        "choice": question["choice"],
                    }
                )

                if verbose:
                    print(f"  Question: {question}")
                    print(f"  Labels: {labels}")
                    print(f"  Correct Index: {correct_idx}")
                    print(f"  Images Indexes: {imgs_idx}")
                stats[stats_key]["created"] += 1

    return all_vqa, stats


def _merge_stats(target, incoming):
    for stats_key, data in incoming.items():
        if stats_key not in target:
            target[stats_key] = {
                "created": 0,
                "impossible": 0,
                "errors": 0,
                "non_interesting": 0,
            }
        for field in ("created", "impossible", "errors", "non_interesting"):
            target[stats_key][field] += data.get(field, 0)


def _stacked_progress_bar(data, width=32):
    total = (
        data.get("created", 0)
        + data.get("impossible", 0)
        + data.get("errors", 0)
        + data.get("non_interesting", 0)
    )
    if total <= 0:
        return "[" + "-" * width + "]"
    created_len = int(round((data.get("created", 0) / total) * width))
    impossible_len = int(round((data.get("impossible", 0) / total) * width))
    errors_len = int(round((data.get("errors", 0) / total) * width))
    non_interesting_len = width - (created_len + impossible_len + errors_len)
    if non_interesting_len < 0:
        non_interesting_len = 0
    return (
        "["
        + f"{ANSI_GREEN}{'#' * created_len}{ANSI_RESET}"
        + f"{ANSI_ORANGE}{'#' * impossible_len}{ANSI_RESET}"
        + f"{ANSI_RED}{'#' * errors_len}{ANSI_RESET}"
        + f"{ANSI_GREY}{'#' * non_interesting_len}{ANSI_RESET}"
        + "]"
    )


def _print_summary(stats):
    if not stats:
        print("No summary stats available.")
        return
    rows = []
    max_key_len = 0
    max_sub_len = 0
    max_c_len = 0
    max_i_len = 0
    max_e_len = 0
    max_n_len = 0
    total_created = 0
    total_impossible = 0
    total_errors = 0
    total_non_interesting = 0
    for (question_key, sub_category, category_key), data in sorted(
        stats.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
    ):
        display_key = question_key
        if display_key.startswith("CF_"):
            display_key = display_key[3:]
            if "_" in display_key:
                display_key = display_key.split("_", 1)[1]
        max_key_len = max(max_key_len, len(display_key))
        max_sub_len = max(max_sub_len, len(sub_category))
        max_c_len = max(max_c_len, len(str(data["created"])))
        max_i_len = max(max_i_len, len(str(data["impossible"])))
        max_e_len = max(max_e_len, len(str(data["errors"])))
        max_n_len = max(max_n_len, len(str(data["non_interesting"])))
        total_created += data["created"]
        total_impossible += data["impossible"]
        total_errors += data["errors"]
        total_non_interesting += data["non_interesting"]
        rows.append((category_key, display_key, sub_category, data))
    print("\nSummary by question_id and sub-category:")
    legend = (
        f"{ANSI_GREEN}C=created{ANSI_RESET}, "
        f"{ANSI_ORANGE}I=impossible{ANSI_RESET}, "
        f"{ANSI_RED}E=errors{ANSI_RESET}, "
        f"{ANSI_GREY}N=non-interesting{ANSI_RESET}"
    )
    print(f"Legend:\t{legend}")
    current_category = None
    for category_key, display_key, sub_category, data in rows:
        if category_key != current_category:
            print(f"---- {category_key.upper()} ----")
            current_category = category_key
        bar = _stacked_progress_bar(data)
        key_field = display_key.ljust(max_key_len)
        sub_field = sub_category.ljust(max_sub_len)
        c_val = str(data["created"]).rjust(max_c_len)
        i_val = str(data["impossible"]).rjust(max_i_len)
        e_val = str(data["errors"]).rjust(max_e_len)
        n_val = str(data["non_interesting"]).rjust(max_n_len)
        line = (
            f"{bar}\t{key_field}\t{sub_field}\t"
            f"{ANSI_GREEN}C={c_val}{ANSI_RESET}\t"
            f"{ANSI_ORANGE}I={i_val}{ANSI_RESET}\t"
            f"{ANSI_RED}E={e_val}{ANSI_RESET}\t"
            f"{ANSI_GREY}N={n_val}{ANSI_RESET}"
        )
        print(line)
    print("-" * 12)
    total_data = {
        "created": total_created,
        "impossible": total_impossible,
        "errors": total_errors,
        "non_interesting": total_non_interesting,
    }
    total_bar = _stacked_progress_bar(total_data)
    total_key = "TOTAL".ljust(max_key_len)
    total_sub = "-".ljust(max_sub_len)
    total_c = str(total_created).rjust(max_c_len)
    total_i = str(total_impossible).rjust(max_i_len)
    total_e = str(total_errors).rjust(max_e_len)
    total_n = str(total_non_interesting).rjust(max_n_len)
    print(
        f"{total_bar}\t{total_key}\t{total_sub}\t"
        f"{ANSI_GREEN}C={total_c}{ANSI_RESET}\t"
        f"{ANSI_ORANGE}I={total_i}{ANSI_RESET}\t"
        f"{ANSI_RED}E={total_e}{ANSI_RESET}\t"
        f"{ANSI_GREY}N={total_n}{ANSI_RESET}"
    )


def main(args):
    # first changing some global variables that would affect the whole run
    set_config("slope_bins", args.slope)

    # create output folder if it does not exist
    if not os.path.exists(args.output_path + f"/{args.run_name}/"):
        os.makedirs(args.output_path + f"/{args.run_name}/", exist_ok=True)

    # then seeding everything
    seed_utils.seed_everything(args.seed)

    # ready to go
    all_vqa = []
    all_stats = {}

    simulation_roots = args.simulation_paths
    list_simulations = []

    for simulation_root in simulation_roots:
        pattern = os.path.join(simulation_root, "**", "simulation.json")

        number_simulations = args.n_scenes

        print("Searching for simulation files with pattern:", pattern)
        for sim_file in glob.glob(pattern, recursive=True):
            list_simulations.append(sim_file)

    list_simulations.sort(key=natural_key)

    # Parallel execution across simulations
    if not list_simulations:
        print("No simulation files found.")
        return

    print("Found", len(list_simulations), "simulation files.")
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=12,
        initializer=_init_worker,
        initargs=(
            args.vqa_path,
            args.destination_simulation_path,
            args.verbose,
            args.seed,
            args.counterfactual_type,
        ),
        mp_context=ctx,
    ) as ex:
        max_simulations = min(number_simulations, len(list_simulations))
        print(f"Processing {max_simulations} simulations...")
        for sim_vqa, sim_stats in ex.map(
            _process_one, list_simulations[:max_simulations], [args] * max_simulations
        ):  # limit to 100s for now
            all_vqa.extend(sim_vqa)
            _merge_stats(all_stats, sim_stats)

    print(f"Saved {len(all_vqa)} questions and answers.")

    save_questions_answers_json(
        all_vqa,
        args.output_path,
        run_name=args.run_name,
    )
    print(f"Saved questions and answers to {args.output_path}")

    print("VQA creation completed.")
    print("LIST OF SIMULATIONS PROCESSED:", len(list_simulations))
    _print_summary(all_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vqa_path",
        type=str,
        default="../",
        help="Path to simpler.json file or similar that contain all the vqa templates.",
    )
    parser.add_argument(
        "--simulation_paths",
        nargs="+",
        type=str,
        default="/data0/sebastian.cavada/datasets/simulations_v3/dl3dv",
        help="Path to the simulation file containing the scenes.",
    )
    parser.add_argument(
        "--destination_simulation_path",
        type=str,
        default="/data0/sebastian.cavada/simulations/dl3dv",
        help="Path where the simulation files are stored (on same or different computer).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../output",
        help="Path to save the questions.json and answers.json files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging.",
    )
    parser.add_argument(
        "--number_of_images_max",
        type=int,
        default=8,
        help="Maximum number of images to save for VQA.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="test_seed_00",
        help="Name of the run for saving outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Global random seed used for all stochastic operations.",
    )

    parser.add_argument(
        "--n_scenes",
        type=int,
        default=4000,
        help="Number of scenes to process.",
    )

    # changing the slope
    parser.add_argument(
        "--slope",
        type=float,
        default=4,
        help="Slope value to be used in the simulation.",
    )

    # different tests to run
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        help="Type of augmentation to use (roi_circling, masking, scene_context, textual_context, etc).",
    )
    parser.add_argument(
        "--counterfactual_type",
        choices=["gravity", "shift", "volume"],
        default="shift",
        help="Select which counterfactual JSON template (gravity, shift, volume) is loaded.",
    )

    args = parser.parse_args()
    timestart = os.times()
    main(args)

#  python main_parallel.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v2
