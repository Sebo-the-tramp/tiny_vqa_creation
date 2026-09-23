VQA_SET="${1:-150K}"

# 
python vqa_roi_extract.py ${VQA_SET}

#
python ./analysis_roi_physics_shift.py --vqa-set ${VQA_SET}