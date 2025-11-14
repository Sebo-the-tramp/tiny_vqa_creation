RUN_NAME="run_08_general"

# python check_VQA.py ../output/test$RUN_NAME.json ../output/val_answer$RUN_NAME.json --random --limit 500  --question-ids F_CAMERA_MOTION_DIRECTION

python check_VQA_with_answers.py ../output/$RUN_NAME/test_${RUN_NAME}_10K.json ../output/$RUN_NAME/val_answer_${RUN_NAME}.json \
 --results-path ../output/$RUN_NAME/results_${RUN_NAME} --limit 1000  --question-ids F_VISIBILITY_OBJECT 
 
# python check_VQA_with_answers.py ../output/test$RUN_NAME.json ../output/val_answer$RUN_NAME.json  --results-path ../output/results$RUN_NAME --random