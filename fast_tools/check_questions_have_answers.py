import os 
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Check if all questions have corresponding answers')
    parser.add_argument('--question-path', 
                       default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/test_run04_1K_balanced.json",
                       help='Path to questions JSON file')
    parser.add_argument('--answer-path',
                       default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/val_answer_run04_1K_balanced.json", 
                       help='Path to answers JSON file')
    
    args = parser.parse_args()
    
    with open(args.question_path, "r") as f:
        questions = json.load(f)
    with open(args.answer_path, "r") as f:
        answers = json.load(f)

    question_ids = set(q['idx'] for q in questions)
    answer_ids = set(a['idx'] for a in answers)

    missing_answers = question_ids - answer_ids
    if missing_answers:
        print("FALSE")
        for qid in missing_answers:
            print(f" - {qid}")
    else:
        print("TRUE")

if __name__ == "__main__":
    main()


# python utils/check_questions_have_answers.py --question-path /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_06_general/test_run_06_general_10K.json \
#  --answer-path /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output_invalid/run_06_general_wrong/val_answer_run_06_general.json