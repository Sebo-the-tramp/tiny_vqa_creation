RUN_NAME="_run_06_1K_roi_circling"

# python check_VQA.py ../output/test$RUN_NAME.json ../output/val_answer$RUN_NAME.json --random --limit 500  --question-ids F_CAMERA_MOTION_DIRECTION

python check_VQA_with_answers.py ../output/test$RUN_NAME.json ../output/val_answer$RUN_NAME.json --random 
# python check_VQA_with_answers.py ../output/test$RUN_NAME.json ../output/val_answer$RUN_NAME.json  --results-path ../output/results$RUN_NAME --randomi