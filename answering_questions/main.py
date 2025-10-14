import os
import json
import argparse

# Import categories - alphabetically

from categories.collision.collision import (
    get_function_by_name_collision,
    get_result_by_name_collision,
)

from categories.forces.forces import (
    get_function_by_name_forces,
    get_result_by_name_forces,
)

from categories.kinematics.kinematics import (
    get_function_by_name_kinematics,
    get_result_by_name_kinematics,
)

from categories.mass.mass import (
    get_function_by_name_mass,
    get_result_by_name_mass,
)

from categories.material.material import (
    get_function_by_name_material,
    get_result_by_name_material,
)

from categories.meta.meta import (
    get_function_by_name_meta,
    get_result_by_name_meta,
)

from categories.spatial.spatial import (
    get_function_by_name_spatial,
    get_result_by_name_spatial,
)

from categories.temporal.temporal import (
    get_function_by_name_temporal,
    get_result_by_name_temporal,    
)

from categories.visibility.visibility import (
    get_function_by_name_visibility,
    get_result_by_name_visibility,
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


def save_questions_answers(questions, answers, output_path):
    os.makedirs(output_path, exist_ok=True)
    questions_path = os.path.join(output_path, "questions.json")
    answers_path = os.path.join(output_path, "answers.json")

    with open(questions_path, "w") as f:
        json.dump(questions, f, indent=4)

    with open(answers_path, "w") as f:
        json.dump(answers, f, indent=4)


# ----- FUNCTION TO GET ANSWER FROM SIMULTAION

resolver_gt = {
    "collision": get_result_by_name_collision,
    "forces": get_result_by_name_forces,
    "kinematics": get_result_by_name_kinematics,    
    "mass": get_result_by_name_mass,
    "material": get_result_by_name_material,
    "meta": get_result_by_name_meta,
    "spatial": get_result_by_name_spatial,    
    "temporal": get_result_by_name_temporal,
    "visibility": get_result_by_name_visibility,
}

resolver = {
    "collision": get_function_by_name_collision,
    "forces": get_function_by_name_forces,
    "kinematics": get_function_by_name_kinematics,    
    "mass": get_function_by_name_mass,
    "material": get_function_by_name_material,
    "meta": get_function_by_name_meta,
    "spatial": get_function_by_name_spatial,        
    "temporal": get_function_by_name_temporal,
    "visibility": get_function_by_name_visibility,
}


def get_answer(question_key, question_category, mock=False):
    return resolver[question_category](question_key, mock=mock)


def get_gt(question_key, question_category, mock=False):
    return resolver_gt[question_category](question_key, mock=mock)


# ----- MAIN VQA CREATION LOGIC
def create_vqa(questions, simulation_steps, arg_mock):
    total_correct_per_category = {}

    for category_key, category in questions.items():
        # current category dev
        if category_key != "collision":
            continue

        print("###" * 10, f"Processing category: {category_key}", "###" * 10)
        print(f"questions: \n{category}")
        print("###" * 20)
        total_questions_in_category = len(category)
        total_correct_answers = 0
        not_implemented = 0
        for question_key, question_data in category.items():
            print(f"  Question Key: {question_key}")
            fn_to_answer_question = get_answer(
                question_key, category_key, mock=arg_mock
            )
            question, labels, correct_idx = fn_to_answer_question(
                simulation_steps, question_data
            )
            print(f"  Question: {question}")
            print(f"  Labels: {labels}")
            print(f"  Correct Index: {correct_idx}")

            gt = get_gt(question_key, category_key, mock=arg_mock)
            print(f"  Ground Truth: {gt}")
            print(
                f"  Answer from function: {labels[correct_idx]}\n  Should match GT: {gt}"
            )

            if str(labels[correct_idx]) != str(gt):
                print("  WARNING: Answer does not match Ground Truth!")
                exit(1)
            else:
                if str(labels[correct_idx]) == "not_implemented":
                    not_implemented += 1
                else:
                    total_correct_answers += 1
            print("===" * 20)
        total_correct_per_category[category_key] = (
            total_correct_answers,
            not_implemented,
            total_questions_in_category,
        )

    print("Summary of correct answers per category:")
    for category, (
        correct,
        not_implemented,
        total,
    ) in total_correct_per_category.items():
        print(
            f"Category '{category}': {correct}/{total - not_implemented} correct answers, {not_implemented} not implemented"
        )

    all_questions = []
    all_answers = []

    return all_questions, all_answers


def main(args):
    questions = read_questions(args.vqa_path)
    simulation_steps = read_simulation(args.simulation_path)

    all_questions, all_answers = create_vqa(questions, simulation_steps, args.mock)
    print(
        f"Saved {len(all_questions)} questions and {len(all_answers)} answers to {args.output_path}"
    )

    save_questions_answers(all_questions, all_answers, args.output_path)
    print(f"Saved questions and answers to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vqa_path",
        type=str,
        default="../simpler_example.json",
        help="Path to simpler.json file or similar that contain all the vqa templates.",
    )
    parser.add_argument(
        "--simulation_path",
        type=str,
        default="./sample_simulation_1000_steps_v2_kinematics.json",
        help="Path to the simulation file containing the scenes.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../output",
        help="Path to save the questions.json and answers.json files.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock implementations for testing.",
    )
    args = parser.parse_args()

    main(args)
