RUN_NAME="run_13_general"
# RUN_NAME="test_seed_00"

# python check_VQA.py ../output/test$RUN_NAME.json ../output/val_answer$RUN_NAME.json --random --limit 500  --question-ids F_CAMERA_MOTION_DIRECTION

# python check_VQA_with_answers.py ../output/$RUN_NAME/test_${RUN_NAME}_mega.json ../output/$RUN_NAME/val_answer_${RUN_NAME}.json \
#  --results-path ../output/$RUN_NAME/results_${RUN_NAME} --limit 1000 --random --question-ids F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_1 F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_2 F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_3

python check_VQA_with_answers.py ../output/$RUN_NAME/test_${RUN_NAME}.json ../output/$RUN_NAME/val_answer_${RUN_NAME}.json \
 --results-path ../output/$RUN_NAME/results_${RUN_NAME} --limit 1000 --random --question-ids F_PHYSICS_PROPERTY_YOUNG_MODULUS_OBJECT_SIMILAR

# python check_VQA_with_answers.py ../output/test$RUN_NAME.json ../output/val_answer$RUN_NAME.json  --results-path ../output/results$RUN_NAME --random