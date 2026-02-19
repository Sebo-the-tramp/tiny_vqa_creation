RUN_NAME="run_26_general"
VQA_SET="30K"
VARIANCE_SPLITS=5

OBJNUM_RUN_NAME="run_24_general_obj_num"
OBJNUM_VQA_SET="10K"

YMS_RUN_NAME="run_24_general_yms-variations"
YMS_VQA_SET="10K"

LEVELS_RUN_NAME="${RUN_NAME}_levels"
LEVELS_VQA_SET="10K"

set -x
python ./analysis_correlation.py --base-path ../output --run-name $OBJNUM_RUN_NAME --vqa-set $OBJNUM_VQA_SET
python ./analysis_vqaset.py --base-path ../output/ --run-name $RUN_NAME --vqa-set $VQA_SET --mode mixed
python ./analysis_category.py --base-path ../output/ --run-name $RUN_NAME --vqa-set $VQA_SET --mode mixed
python ./analysis_commonsense.py --base-path ../output/ --run-name $RUN_NAME --vqa-set $VQA_SET

python ./analysis_material_yms.py --base-path ../output --run-name $YMS_RUN_NAME --vqa-set $YMS_VQA_SET
python ./analysis_5_levels.py --base-path ../output --run-name $LEVELS_RUN_NAME --vqa-set $LEVELS_VQA_SET

# Variance study with sampling
for sampling in $(seq 0.1 0.1 0.7); do
    python ./vqa_subsample.py --base-path ../output/ --run-name $RUN_NAME --vqa-set $VQA_SET --num $VARIANCE_SPLITS --mode sample --sampling $sampling --skip-existing
done
python ./analysis_variance.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode sample

# # Variance study with ranging
# for num in $(seq 2 1 10); do
#     python ./vqa_subsample.py --base-path ../output/ --run-name $RUN_NAME --vqa-set $VQA_SET --mode range --num $num --skip-existing
# done
python ./analysis_variance.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode range