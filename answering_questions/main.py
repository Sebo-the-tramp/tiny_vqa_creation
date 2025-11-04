import os
import re
import json
import glob
import argparse

from utils.saving_utils import (
    save_questions_answers_json,
    save_questions_answers_tsv,
)
from utils.my_exception import ImpossibleToAnswer

# Import categories - alphabetically

from categories.spatial_reasoning.spatial_reasoning import (
    get_function_by_name_spatial_reasoning,
    get_result_by_name_spatial_reasoning,
)

from categories.mechanics.mechanics import (
    get_function_by_name_mechanics,
    get_result_by_name_mechanics,
)

from categories.material_understanding.material_understanding import (
    get_function_by_name_material_understanding,
    get_result_by_name_material_understanding,
)

from categories.temporal.temporal import (
    get_function_by_name_temporal,
    get_result_by_name_temporal,
)

from categories.viewpoint.viewpoint import (
    get_function_by_name_viewpoint,
    get_result_by_name_viewpoint,
)


# ----- UTILS FUNCTIONS
def read_questions(vqa_path):
    with open(vqa_path, "r") as f:
        questions = json.load(f)
    return questions


def read_simulation(simulation_path):
    with open(simulation_path, "r") as f:
        simulation_steps = json.load(f)
    return simulation_steps


# ----- FUNCTION TO GET ANSWER FROM SIMULTAION

resolver_gt = {
    "spatial_reasoning": get_result_by_name_spatial_reasoning,
    "mechanics": get_result_by_name_mechanics,
    "material_understanding": get_result_by_name_material_understanding,
    "temporal": get_result_by_name_temporal,
    "view_point": get_result_by_name_viewpoint,
}

resolver = {
    "spatial_reasoning": get_function_by_name_spatial_reasoning,
    "mechanics": get_function_by_name_mechanics,
    "material_understanding": get_function_by_name_material_understanding,
    "temporal": get_function_by_name_temporal,
    "view_point": get_function_by_name_viewpoint,
}


def get_answer(question_key, question_category, mock=False):
    return resolver[question_category](question_key, mock=mock)


def get_gt(question_key, question_category, mock=False):
    return resolver_gt[question_category](question_key, mock=mock)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# ----- MAIN VQA CREATION LOGIC
def create_vqa(
    questions,
    simulation_steps,
    simulation_id,
    destination_simulation_id_path,
    arg_mock,
    verbose=False,
):
    
    answered = 0
    failed = 0

    print("Starting VQA creation...")

    all_vqa = []

    for category_key, category in questions.items():
        # current category dev
        if (
            category_key != "mechanics"
        ):
            continue

        if verbose:
            print("###" * 10, f"Processing category: {category_key}", "###" * 10)
            print(f"questions: \n{category}")
            print("###" * 20)
        total_questions_in_category = len(category)
        total_correct_answers = 0
        not_implemented = 0
        for question_key, question_data in category.items():            
            fn_to_answer_question = get_answer(
                question_key, category_key, mock=arg_mock
            )

            try:
                # answer_list = question, labels, correct_idx, imgs_idx
                answer_list = fn_to_answer_question(simulation_steps, question_data, destination_simulation_id_path)
            except ImpossibleToAnswer:
                not_implemented += 1
                print(f"  Question Key: {question_key}")
                continue
            except Exception as e:
                print(f"  ERROR processing question {question_key}: {e}")
                exit(1)

            for question, labels, correct_idx, imgs_idx in answer_list:
                # changing from image_paths to image_paths
                file_names = [
                    destination_simulation_id_path + f"render/{int(frame_idx):06d}.png"
                    for frame_idx in imgs_idx
                ]

                all_vqa.append(
                    {
                        "scene": simulation_steps.get("scene", {}).get(
                            "scene", "unknown_scene"
                        ),
                        "simulation_id": simulation_id,
                        "question": question,
                        "category": category_key,
                        "sub_category": question_data['sub_category'],
                        "question_key": question_key,
                        "image_paths": file_names,
                        "labels": labels,
                        "answer_index": correct_idx,
                        "mode": "image-only"
                        if question["task_splits"] == "single"
                        else "general",
                        "choice": question["choice"],
                    }
                )

                if verbose:
                    print(f"  Question: {question}")
                    print(f"  Labels: {labels}")
                    print(f"  Correct Index: {correct_idx}")
                    print(f"  Images Indexes: {imgs_idx}")

                gt = get_gt(question_key, category_key, mock=arg_mock)
                if verbose:
                    print(
                        f"  Answer from function: {labels[correct_idx]}\n  Should match GT: {gt}"
                    )

                # Just for development, the rng function given more or less functions will break the integration test
                if str(labels[correct_idx]) != str(gt) and verbose:
                    print(
                        "\033[93m  WARNING: Answer does not match Ground Truth!\033[0m"
                    )
                    # exit(1)
                else:
                    if str(labels[correct_idx]) == "not_implemented":
                        not_implemented += 1
                    else:
                        total_correct_answers += 1
                if verbose:
                    print("===" * 20)

        # total_correct_per_category[category_key] = (
        #     total_correct_answers,
        #     not_implemented,
        #     total_questions_in_category,
        # )

        total_answered = total_questions_in_category - not_implemented
        # print(
        #     f"Category '{category_key}': {total_answered}/{total_questions_in_category} correct answers."
        # )

        answered += total_answered
        failed += not_implemented


    return all_vqa, answered, failed


def main(args):
    all_vqa = []
    total_answered = 0
    total_failed = 0

    simulation_root = args.simulation_path
    pattern = os.path.join(simulation_root, '**', 'simulation.json')

    print("Searching for simulation files with pattern:", pattern)

    list_simulations = []
    for sim_file in glob.glob(pattern, recursive=True):
        simulation_id = os.path.dirname(sim_file)  # folder containing the file

        print(simulation_id)
        print(sim_file)

        list_simulations.append(sim_file)

    list_simulations.sort(key=natural_key)

    # limit to 100s for now
    for simulation_id in list_simulations:
        print(simulation_id)
        if not os.path.isfile(simulation_id):
            print("not folder found")
            continue
        else:
            print("Found simulation folder:", simulation_id)

            questions_raw = read_questions(args.vqa_path + "simple_vqa.json")

            # questions = split_questions_by_task_splits(questions_raw)
            questions = questions_raw

            simulation_id_path = simulation_id.replace("simulation.json", "")
            destination_simulation_id_path = os.path.join(
                args.destination_simulation_path, simulation_id_path
            )

            simulation_steps = read_simulation(
                os.path.join(simulation_id_path, "simulation_kinematics.json")
            )

            simulation_vqa, answered, failed = create_vqa(
                questions,
                simulation_steps,
                simulation_id,
                destination_simulation_id_path,
                args.mock,
                verbose=args.verbose,
            )
            all_vqa.extend(simulation_vqa)
            total_answered += answered
            total_failed += failed 

    print(
        f"Total answered questions: {total_answered}, Total failed questions: {total_failed}"
    )

    # Finally save the questions and answers
    print(f"Saved {len(all_vqa)} questions and answers.")

    if args.export_format in ["json", "both"]:
        questions_path, answers_path = save_questions_answers_json(
            all_vqa,
            args.output_path,
            export_format=args.export_format,
            image_output=args.image_output,
            number_of_images_max=args.number_of_images_max,
        )
        print(
            f"Saved questions and answers to {questions_path} and {answers_path} ({args.export_format})"
        )

    if args.export_format in ["tsv", "both"]:
        save_questions_answers_tsv(
            all_vqa,
            args.output_path,
            export_format=args.export_format,
            image_output=args.image_output,
            number_of_images_max=args.number_of_images_max,
        )
        print(
            f"Saved questions and answers to {args.output_path} ({args.export_format})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vqa_path",
        type=str,
        default="../",
        help="Path to simpler.json file or similar that contain all the vqa templates.",
    )
    parser.add_argument(
        "--simulation_path",
        type=str,
        # default="./sample_simulation_1000_steps_v2_kinematics.json
        # default="/Users/sebastiancavada/Desktop/tmp_Paris/vqa/data/output/sims/dl3dv-hf-gso2/3-cg/c-0_no-3_d-3_s-dl3dv-1bef58393fffbf6e34cac11d0b03dd22f65954a1668b7b9dec548f6ad44f29b5_models-hf-gso_MLP-10_smooth_h-10-40_seed-0_dbgsub-1_20251016_013244",
        # default="/mnt/proj1/eu-25-92/tiny_vqa_creation/data/simulations/dl3dv/random/7-cg",
        # default="/Users/sebastiancavada/Desktop/tmp_Paris/vqa/answering_questions/",
        default="/data0/sebastian.cavada/datasets/simulations/dl3dv",
        help="Path to the simulation file containing the scenes.",
    )
    parser.add_argument(
        "--destination_simulation_path",
        type=str,
        default="/data0/sebastian.cavada/simulations/dl3dv",
        # default="/mnt/proj1/eu-25-92/tiny_vqa_creation/data/simulations/dl3dv/random/7-cg",
        # default="/Users/sebastiancavada/Desktop/tmp_Paris/vqa/answering_questions/",
        help="Path where the simulation files are stored (on same or different computer).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../output",
        help="Path to save the questions.json and answers.json files.",
    )
    parser.add_argument(
        "--export_format",
        choices=["json", "tsv", "both"],
        default="json",
        help="Output format for generated questions and answers.",
    )
    parser.add_argument(
        "--image_output",
        choices=["base64", "path"],
        default="base64",
        help="Select whether exported questions reference images via base64 or filesystem paths (TSV always uses paths).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock implementations for testing.",
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
    args = parser.parse_args()

    timestart = os.times()

    main(args)


# python main.py --simulation_path /data0/sebastian.cavada/datasets/simulations_v2
