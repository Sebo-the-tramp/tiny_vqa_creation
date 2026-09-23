RUN_NAME="run_28_general"
BASE_PATH="../output"

SAMPLING="17000"


python ./vqa_subsample.py --base-path $BASE_PATH --run-name $RUN_NAME --vqa-set "150K" --vqa-split-mode sample --sampling $SAMPLING --num 10 --skip-existing

for i in $(seq 0 9); do
    ./rebut_tinysubset.sh 150K-s$SAMPLING-s$i &
done
