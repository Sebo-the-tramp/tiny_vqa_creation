cd answering_questions
python subsample_questions.py \
    --input ../output/test_og.json \
    --output ../output/test_image-only.json \
    --count 10000 \
    --mode image-only \
    --seed 42
