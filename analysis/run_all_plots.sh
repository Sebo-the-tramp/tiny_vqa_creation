RUN_NAME="run_28_general"
VQA_SET="150K"
# RUN_NAME="run_26_general"
# VQA_SET="30K"
VARIANCE_SPLITS=3

# OBJNUM_RUN_NAME=${RUN_NAME}
# OBJNUM_VQA_SET=${VQA_SET}

# YMS_RUN_NAME="run_24_general_yms-variations"
# YMS_VQA_SET="10K"

# LEVELS_RUN_NAME="${RUN_NAME}_levels"
# LEVELS_VQA_SET="10K"

set -x
python ./analysis_vqaset.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --mode all
python ./analysis_category.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --mode all
python ./analysis_correlation.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --mode all --unbalanced
python ./analysis_correlation.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --mode all --balanced
python ./analysis_commonsense.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET
python ./analysis_commonsense.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --family InternVLChat2

# python ./analysis_material_yms.py --base-path ../output --run-name $YMS_RUN_NAME --vqa-set $YMS_VQA_SET
# python ./analysis_5_levels.py --base-path ../output --run-name $LEVELS_RUN_NAME --vqa-set $LEVELS_VQA_SET

# Variance study with sampling (roughly 5k to 50k questions per split)
for sampling in $(seq 0.03 0.03 0.30); do
    python ./vqa_subsample.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode sample --sampling $sampling --num $VARIANCE_SPLITS --skip-existing
done
python ./analysis_variance.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode sample

# Variance study with ranging
# for num in $(seq 2 1 10); do
#     python ./vqa_subsample.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --mode range --num $num --skip-existing
# done
# python ./analysis_variance.py --base-path ../output --run-name $RUN_NAME --vqa-set $VQA_SET --vqa-split-mode range