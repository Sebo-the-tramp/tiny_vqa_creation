#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


QUESTIONS_DIR = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/supp_material_scripts/questions_paper")
SIMPLE_VQA_PATH = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/simple_vqa.json")
OUTPUT_DIR = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/supp_material_scripts/questions_paper_site")
MEDIA_DIR_NAME = "media"
GIF_DURATION_MS = 350
VQA_PAGE_NAME = "vqa.html"
MAPPING_CAT_COLORS = {
    "mechanics": "#FF5733",
    "spatial_reasoning": "#3498DB",
    "permanence": "#F43FC7",
    "persistence": "#F43FC7",
    "temporal": "#0DA792",
    "view_point": "#EEAC32",
    "material_understanding": "#2BAE27",
    "visual_percetion": "#9B59B6",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}


def title_case(text: str) -> str:
    return text.replace("_", " ").title()


def ask_override(path: Path) -> None:
    if not path.exists() or not any(path.iterdir()):
        return
    answer = input(f"{path} exists. Override it? [y/N] ").strip().lower()
    assert answer == "y", "Aborted."
    shutil.rmtree(path)


def load_qid_order() -> tuple[list[str], dict[str, str], dict[str, str]]:
    data = json.loads(SIMPLE_VQA_PATH.read_text())
    qid_order: list[str] = []
    qid_to_category: dict[str, str] = {}
    qid_to_sub_category: dict[str, str] = {}
    for category, questions in data.items():
        if not isinstance(questions, dict):
            continue
        for qid, question_data in questions.items():
            qid_str = str(qid)
            qid_order.append(qid_str)
            qid_to_category[qid_str] = str(category)
            qid_to_sub_category[qid_str] = str(question_data["sub_category"])
    return qid_order, qid_to_category, qid_to_sub_category


def parse_question_block(text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prompt_lines: list[str] = []
    choices: list[tuple[str, str]] = []
    for line in lines:
        if len(line) > 2 and line[0] in "ABCD" and line[1] in ".)":
            choices.append((line[0], line[2:].strip()))
        else:
            prompt_lines.append(line)
    prompt = " ".join(prompt_lines).replace("<image>", " ")
    prompt = " ".join(prompt.split())
    return prompt, choices


def media_entries(question_dir: Path, data: dict[str, Any]) -> list[Path]:
    media: list[Path] = []
    for item in data.get("images", []):
        rel = item if isinstance(item, str) else item["filename"]
        direct_path = (question_dir / rel).resolve()
        images_path = (question_dir / "images" / rel).resolve()
        media.append(images_path if images_path.exists() else direct_path)
    if media:
        return media
    images_dir = question_dir / "images"
    assert images_dir.exists(), f"Missing images dir in {question_dir}"
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES)


def create_gif(image_paths: list[Path], gif_path: Path) -> None:
    frames = [Image.open(path).convert("RGB") for path in image_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )


def load_entries() -> list[dict[str, Any]]:
    qid_order, qid_to_category, qid_to_sub_category = load_qid_order()
    qid_rank = {qid: i for i, qid in enumerate(qid_order)}
    entries: list[dict[str, Any]] = []
    for question_file in QUESTIONS_DIR.glob("folder_*/question.json"):
        data = json.loads(question_file.read_text())
        idx = str(data["idx"])
        qid = str(data["question_id"])
        prompt, choices = parse_question_block(str(data["question"]))
        item_type = "single" if idx.endswith("_i") else "multi" if idx.endswith("_g") else str(data.get("item_type", "unknown"))
        entries.append(
            {
                "idx": idx,
                "qid": qid,
                "rank": qid_rank[qid],
                "category": qid_to_category[qid],
                "sub_category": qid_to_sub_category[qid],
                "item_type": item_type,
                "prompt": prompt,
                "choices": choices,
                "correct_answer": str(data["correct_answer"]),
                "media": media_entries(question_file.parent, data),
            }
        )
    return sorted(entries, key=lambda entry: (entry["rank"], entry["idx"]))


def copy_media(entries: list[dict[str, Any]]) -> None:
    media_root = OUTPUT_DIR / MEDIA_DIR_NAME
    for entry in entries:
        copied: list[str] = []
        copied_paths: list[Path] = []
        entry_dir = media_root / entry["idx"]
        entry_dir.mkdir(parents=True, exist_ok=True)
        for source in entry["media"]:
            assert source.exists(), f"Missing source media: {source}"
            destination = entry_dir / source.name
            shutil.copy2(source, destination)
            assert destination.exists(), f"Failed to copy media: {destination}"
            copied.append(f"{MEDIA_DIR_NAME}/{entry['idx']}/{source.name}")
            copied_paths.append(destination)
        gif_path = None
        if entry["item_type"] == "multi" and len(copied_paths) > 1:
            gif_path = entry_dir / "animation.gif"
            create_gif(copied_paths, gif_path)
        entry["copied_media"] = copied
        entry["copied_gif"] = None if gif_path is None else f"{MEDIA_DIR_NAME}/{entry['idx']}/{gif_path.name}"


def render_media(entry: dict[str, Any]) -> str:
    paths = entry["copied_media"]
    gif_path = entry["copied_gif"]
    if gif_path is not None:
        image_id = f"gif-{entry['idx']}"
        still_path = html.escape(paths[0])
        anim_path = html.escape(gif_path)
        return (
            '<div class="media-box">'
            f'<img class="media-main" id="{html.escape(image_id)}" src="{still_path}" alt="" />'
            "</div>"
            f'<div style="margin-top:6px;"><button class="media-button" type="button" data-state="still" data-still="{still_path}" data-anim="{anim_path}" '
            f'onclick="toggleGif(this, \'{html.escape(image_id)}\')">Play</button></div>'
        )
    blocks: list[str] = []
    for rel_path in paths:
        suffix = Path(rel_path).suffix.lower()
        safe_path = html.escape(rel_path)
        if suffix in VIDEO_SUFFIXES:
            blocks.append(f'<video class="media-main" controls preload="metadata" src="{safe_path}"></video>')
        else:
            blocks.append(f'<img class="media-main" src="{safe_path}" alt="" />')
    if len(blocks) == 1:
        return '<div class="media-box">' + blocks[0] + "</div>"
    return '<div class="media-grid">' + "".join(f'<div class="media-box">{block}</div>' for block in blocks) + "</div>"


def render_choices(choices: list[tuple[str, str]], correct_answer: str) -> str:
    items: list[str] = []
    for label, text in choices:
        safe_text = html.escape(text)
        if label == correct_answer:
            safe_text = f"<b>{safe_text}</b>"
        items.append(f"<div>{label}) {safe_text}</div>")
    return "".join(items)


def render_html(entries: list[dict[str, Any]]) -> str:
    categories = sorted({entry["category"] for entry in entries})
    sub_categories = sorted({entry.get("sub_category", "") for entry in entries})
    category_options = "".join(
        f'<option value="{html.escape(category)}">{html.escape(title_case(category))}</option>'
        for category in categories
    )
    sub_category_options = "".join(
        f'<option value="{html.escape(sub_category)}">{html.escape(title_case(sub_category))}</option>'
        for sub_category in sub_categories
    )
    rows: list[str] = []
    for entry in entries:
        color = MAPPING_CAT_COLORS[entry["category"]]
        rows.append(
            f'<tr data-category="{html.escape(entry["category"])}" data-sub-category="{html.escape(entry["sub_category"])}">'
            f'<td class="compact"><div class="rot">{html.escape(entry["idx"])}</div></td>'
            f'<td class="compact" style="background:{html.escape(color)}22;"><div class="rot">{html.escape(title_case(entry["category"]))}</div></td>'
            f'<td class="compact" style="background:{html.escape(color)}22;"><div class="rot">{html.escape(title_case(entry["sub_category"]))}</div></td>'
            f'<td class="compact" style="background:{html.escape(color)}22;"><div class="rot">{html.escape(entry["qid"])}</div></td>'
            f'<td style="vertical-align: top; white-space: nowrap;">{html.escape(entry["item_type"])}</td>'
            f'<td style="vertical-align: top; min-width: 420px;">{render_media(entry)}</td>'
            f'<td style="vertical-align: top;"><div style="margin-bottom:8px;"><b>{html.escape(entry["prompt"])}</b></div>{render_choices(entry["choices"], entry["correct_answer"])}</td>'
            "</tr>"
        )
    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>Questions Paper</title>"
        "<style>"
        "body{font-family:serif;margin:16px;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #000;padding:8px;}th{text-align:left;}img,video{display:block;margin:auto;}"
        ".compact{width:34px;padding:4px;text-align:center;vertical-align:middle;}"
        ".rot{writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;font-size:12px;}"
        ".media-grid{display:grid;grid-template-columns:repeat(2, 1fr);gap:6px;}"
        ".media-box{width:100%;height:320px;display:flex;align-items:center;justify-content:center;overflow:hidden;}"
        ".media-main{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;}"
        ".media-button{width:100%;padding:8px 0;}"
        ".filters{display:flex;gap:12px;align-items:center;margin:12px 0;}"
        "</style>"
        "<script>"
        "function toggleGif(button, imageId){"
        "const img=document.getElementById(imageId);"
        "const state=button.dataset.state;"
        "if(state==='still'){img.src=button.dataset.anim;button.dataset.state='anim';button.textContent='Stop';return;}"
        "img.src=button.dataset.still;button.dataset.state='still';button.textContent='Play';"
        "}"
        "function applyFilters(){"
        "const category=document.getElementById('category-filter').value;"
        "const subCategory=document.getElementById('sub-category-filter').value;"
        "for(const row of document.querySelectorAll('tbody tr')){"
        "const okCategory=!category||row.dataset.category===category;"
        "const okSubCategory=!subCategory||row.dataset.subCategory===subCategory;"
        "row.style.display=okCategory&&okSubCategory?'':'none';"
        "}"
        "}"
        "</script></head><body>"
        "<p><a href='index.html'>Back to index</a></p>"
        "<h1>Questions Paper</h1>"
        "<p>* GIFs may show small color differences due to website optimization. This visualization is only for browsing and is not exactly what the model sees.</p>"
        "<div class='filters'>"
        "<label>Category "
        f"<select id='category-filter' onchange='applyFilters()'><option value=''>All</option>{category_options}</select>"
        "</label>"
        "<label>Sub-category "
        f"<select id='sub-category-filter' onchange='applyFilters()'><option value=''>All</option>{sub_category_options}</select>"
        "</label>"
        "</div>"
        "<table>"
        "<thead><tr><th class='compact'><div class='rot'>ID</div></th><th class='compact'><div class='rot'>Category</div></th><th class='compact'><div class='rot'>Sub-category</div></th><th class='compact'><div class='rot'>Question ID</div></th><th>Type</th><th>Visual Input</th><th>Question & Choices</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def render_home_html() -> str:
    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>Questions Paper Site</title>"
        "<style>body{font-family:serif;margin:24px;}ul{line-height:1.8;}</style>"
        "</head><body>"
        "<h1>Questions Paper Site</h1>"
        "<ul>"
        f"<li><a href='{VQA_PAGE_NAME}'>VQA</a></li>"
        "<li><a href='video_examples.html'>Video examples</a></li>"
        "<li><a href='other_things.html'>Other things</a></li>"
        "</ul>"
        "</body></html>"
    )


def render_placeholder_html(title: str) -> str:
    return (
        "<!doctype html>"
        f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:serif;margin:24px;}</style>"
        "</head><body>"
        "<p><a href='index.html'>Back to index</a></p>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>Sample page.</p>"
        "</body></html>"
    )


def main() -> None:
    ask_override(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_entries()
    copy_media(entries)
    (OUTPUT_DIR / "index.html").write_text(render_home_html())
    (OUTPUT_DIR / VQA_PAGE_NAME).write_text(render_html(entries))
    (OUTPUT_DIR / "video_examples.html").write_text(render_placeholder_html("Video examples"))
    (OUTPUT_DIR / "other_things.html").write_text(render_placeholder_html("Other things"))
    print(f"Wrote {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
