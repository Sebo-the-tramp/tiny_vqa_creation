# cd answering_questions
# python subsample_questions.py \
#     --input ../output/test_run04_full_images.json \
#     --output ../output/test_run04_1K.json \
#     --count 1000 \
#     --seed 42

cd answering_questions
python subsample_questions_balanced.py \
    --input ../output/test_run04_full_images.json \
    --output ../output/test_run04_1K_balanced.json \
    --count 1000 \
    --seed 42