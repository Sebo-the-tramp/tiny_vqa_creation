RUN_NAME="run_28_general"
VQA_SET="150K"
# RUN_NAME="run_26_general"
# VQA_SET="30K"
VARIANCE_SPLITS=3

YMS_RUN_NAME="run_24_general_yms_variations"
YMS_VQA_SET="10K"

LEVELS_RUN_NAME="run_26_general_levels"
LEVELS_VQA_SET="10K"

ABLATIONS_RUN_NAME="run_28"
ABLATIONS_VQA_SET="10K"

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
run_cmd python ./analysis_numobj.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode all --unbalanced
run_cmd python ./analysis_numobj.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode all --balanced
run_cmd python ./analysis_commonsense.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET
run_cmd python ./analysis_commonsense.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --family InternVLChat2

# YMS
run_cmd python ./analysis_vqaset.py --base-path $BASE_PATH --run-name $YMS_RUN_NAME --vqa-set $YMS_VQA_SET --mode all
run_cmd python ./analysis_material_yms.py --base-path $BASE_PATH --run-name $YMS_RUN_NAME --vqa-set $YMS_VQA_SET

# Levels
run_cmd python ./analysis_vqaset.py --base-path $BASE_PATH --run-name $LEVELS_RUN_NAME --vqa-set $LEVELS_VQA_SET --mode all
run_cmd python ./analysis_levels.py --base-path $BASE_PATH --run-name $LEVELS_RUN_NAME --vqa-set $LEVELS_VQA_SET

# Ablations
run_cmd python ./analysis_ablation_rel.py --base-path $BASE_PATH --run-name $ABLATIONS_RUN_NAME --vqa-set $ABLATIONS_VQA_SET

# Variance study with sampling
for sampling in $(seq 0.03 0.03 0.30); do  #  sampling roughly 5k (~0.03)
    run_cmd python ./vqa_subsample.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode sample --sampling $sampling --num $VARIANCE_SPLITS --skip-existing
done
run_cmd python ./analysis_variance.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode sample

# Variance study with ranging
# for num in $(seq 2 1 10); do
#     run_cmd python ./vqa_subsample.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --mode range --num $num --skip-existing
# done
# run_cmd python ./analysis_variance.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode range