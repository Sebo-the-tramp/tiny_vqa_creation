#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any


QUESTIONS_DIR = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/supp_material_scripts/questions_paper")
SIMPLE_VQA_PATH = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/simple_vqa.json")
OUTPUT_DIR = Path("/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/supp_material_scripts/questions_paper_site")
MEDIA_DIR_NAME = "media"
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


def load_qid_order() -> tuple[list[str], dict[str, str]]:
    data = json.loads(SIMPLE_VQA_PATH.read_text())
    qid_order: list[str] = []
    qid_to_category: dict[str, str] = {}
    for category, questions in data.items():
        if not isinstance(questions, dict):
            continue
        for qid in questions:
            qid_str = str(qid)
            qid_order.append(qid_str)
            qid_to_category[qid_str] = str(category)
    return qid_order, qid_to_category


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


def multi_video_path(data: dict[str, Any]) -> Path | None:
    raw = data.get("raw", {})
    simulation_id = data.get("simulation_id") or raw.get("simulation_id")
    if simulation_id:
        path = Path(str(simulation_id)).with_name("render.mp4")
        if path.exists():
            return path
    for key in ("original_file_name", "file_name"):
        files = data.get(key) or raw.get(key) or []
        if files:
            path = Path(str(files[0])).parent.parent / "render.mp4"
            if path.exists():
                return path
    return None


def load_entries() -> list[dict[str, Any]]:
    qid_order, qid_to_category = load_qid_order()
    qid_rank = {qid: i for i, qid in enumerate(qid_order)}
    entries: list[dict[str, Any]] = []
    for question_file in QUESTIONS_DIR.glob("folder_*/question.json"):
        data = json.loads(question_file.read_text())
        idx = str(data["idx"])
        qid = str(data["question_id"])
        prompt, choices = parse_question_block(str(data["question"]))
        item_type = "single" if idx.endswith("_i") else "multi" if idx.endswith("_g") else str(data.get("item_type", "unknown"))
        video_path = multi_video_path(data) if item_type == "multi" else None
        entries.append(
            {
                "idx": idx,
                "qid": qid,
                "rank": qid_rank[qid],
                "category": qid_to_category[qid],
                "item_type": item_type,
                "prompt": prompt,
                "choices": choices,
                "correct_answer": str(data["correct_answer"]),
                "video": video_path,
                "media": media_entries(question_file.parent, data),
            }
        )
    return sorted(entries, key=lambda entry: (entry["rank"], entry["idx"]))


def copy_media(entries: list[dict[str, Any]]) -> None:
    media_root = OUTPUT_DIR / MEDIA_DIR_NAME
    for entry in entries:
        copied: list[str] = []
        entry_dir = media_root / entry["idx"]
        entry_dir.mkdir(parents=True, exist_ok=True)
        video = entry["video"]
        if video is not None:
            destination = entry_dir / video.name
            shutil.copy2(video, destination)
            entry["copied_video"] = f"{MEDIA_DIR_NAME}/{entry['idx']}/{video.name}"
        else:
            entry["copied_video"] = None
        for source in entry["media"]:
            destination = entry_dir / source.name
            shutil.copy2(source, destination)
            copied.append(f"{MEDIA_DIR_NAME}/{entry['idx']}/{source.name}")
        entry["copied_media"] = copied


def render_media(paths: list[str], video_path: str | None) -> str:
    if video_path is not None:
        safe_path = html.escape(video_path)
        return f'<video controls preload="metadata" src="{safe_path}" style="max-width: 100%; max-height: 320px;"></video>'
    blocks: list[str] = []
    for rel_path in paths:
        suffix = Path(rel_path).suffix.lower()
        safe_path = html.escape(rel_path)
        if suffix in VIDEO_SUFFIXES:
            blocks.append(f'<video controls preload="metadata" src="{safe_path}" style="max-width: 100%; max-height: 260px;"></video>')
        else:
            blocks.append(f'<img src="{safe_path}" alt="" style="max-width: 100%; max-height: 260px;" />')
    if len(blocks) == 1:
        return blocks[0]
    return '<div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:6px;">' + "".join(blocks) + "</div>"


def render_choices(choices: list[tuple[str, str]], correct_answer: str) -> str:
    items: list[str] = []
    for label, text in choices:
        safe_text = html.escape(text)
        if label == correct_answer:
            safe_text = f"<b>{safe_text}</b>"
        items.append(f"<div>{label}) {safe_text}</div>")
    return "".join(items)


def render_html(entries: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for entry in entries:
        color = MAPPING_CAT_COLORS[entry["category"]]
        rows.append(
            "<tr>"
            f'<td class="compact"><div class="rot">{html.escape(entry["idx"])}</div></td>'
            f'<td class="compact" style="background:{html.escape(color)}22;"><div class="rot">{html.escape(entry["qid"])}</div></td>'
            f'<td style="vertical-align: top; white-space: nowrap;">{html.escape(entry["item_type"])}</td>'
            f'<td style="vertical-align: top; min-width: 420px;">{render_media(entry["copied_media"], entry["copied_video"])}</td>'
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
        "</style></head><body>"
        "<h1>Questions Paper</h1>"
        "<table>"
        "<tr><th class='compact'><div class='rot'>ID</div></th><th class='compact'><div class='rot'>Question ID</div></th><th>Type</th><th>Visual Input</th><th>Question & Choices</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )


def main() -> None:
    ask_override(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_entries()
    copy_media(entries)
    (OUTPUT_DIR / "index.html").write_text(render_html(entries))
    print(f"Wrote {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
