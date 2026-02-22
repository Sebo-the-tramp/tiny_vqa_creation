# visualization_2

Simple FastAPI + static frontend viewer for:

- questions: `/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_28_general/test_run_28_general.json`
- correct answers: `/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_28_general/val_answer_run_26_general.json`

## What it supports

- server-side paging
- exact filters:
  - `idx`
  - `question_id` (dropdown with all IDs)
  - `object_count` (parsed from image path)
  - scenes (multi-select from `scenes.json`)
- startup default scene selection = all scene IDs except `exclude_scenes.txt`
- single-image view: image left, question right, correct answer in bold
- multi-image view: thumbnail carousel + large selected image, question and answer on right
- per-question download button:
  - downloads `folder_<idx>.zip`
  - zip contains `folder_<idx>/question.json` and `folder_<idx>/images/*`
- image loading optimization:
  - frontend delays image requests until image is in viewport for 1 second
  - `/api/image` can return compressed previews (webp/jpeg) with server-side cache

## Run

```bash
cd /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/visualization_2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
chmod +x run.sh
./run.sh
```

Open: `http://localhost:8086`

## Optional environment variables

- `HOST` (default `0.0.0.0`)
- `PORT` (default `8086`)
- `PYTHON_BIN` (optional explicit interpreter path)
- `VQA_QUESTION_FILE`
- `VQA_ANSWER_FILE`
- `VQA_SCENES_FILE`
- `VQA_EXCLUDE_FILE`
- `VQA_MAX_PAGE_SIZE` (default `200`)
- `VQA_FILTER_CACHE_SIZE` (default `64`)
- `VQA_IMAGE_PREVIEW_MAX_WIDTH` (default `1280`)
- `VQA_IMAGE_PREVIEW_QUALITY` (default `70`)
- `VQA_IMAGE_PREVIEW_FORMAT` (`webp`, `jpeg`, `orig`; default `webp`)
- `VQA_IMAGE_PREVIEW_CACHE_SIZE` (default `256`)

## API extra endpoint

- `GET /api/download?idx=<question_idx>`
  - returns a zip bundle for that question
