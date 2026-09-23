RUN_NAME="run_28_general"
VQA_SET="${1:-150K-s17000-s6}"

BASE_PATH="../output"

RUN_IN_BACKGROUND=0
run_cmd() {
    if [[ "$RUN_IN_BACKGROUND" -eq 1 ]]; then
        "$@" 2>&1 &
    else
        "$@" 2>&1
    fi
}

set -x

# All analyses main run
run_cmd python ./analysis_vqaset.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode all
run_cmd python ./analysis_categorical.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode all
# run_cmd python ./analysis_numobj.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode all --unbalanced
# run_cmd python ./analysis_numobj.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode all --balanced
# run_cmd python ./analysis_commonsense.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET
# run_cmd python ./analysis_commonsense.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --family InternVLChat2

# Variance study with ranging
# for num in $(seq 2 1 10); do
#     run_cmd python ./vqa_subsample.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode range --num $num --skip-existing
# done
# run_cmd python ./analysis_variance.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode range