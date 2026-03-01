#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
import shutil
import subprocess
import unicodedata


def normalize_text(text: str) -> str:
    repl = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2011": "-",
        "\u00a0": " ",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def format_entry_id(entry_id: str, max_len: int = 30) -> str:
    escaped = latex_escape(entry_id)
    if len(entry_id) <= max_len:
        return escaped
    parts = entry_id.split("_")
    if len(parts) <= 1:
        return escaped
    mid = len(parts) // 2
    return latex_escape("_".join(parts[:mid])) + r"\_\allowbreak " + latex_escape(
        "_".join(parts[mid:])
    )


def title_from_key(key: str | None) -> str:
    if not key:
        return "Unknown"
    key = key.replace("_", " ")
    return re.sub(r"\b([a-z])", lambda m: m.group(1).upper(), key)


TEMPORAL_T5_T8_QIDS = {
    "F_TEMPORAL_PREDICTION_NEXT_IMAGE_GRANULARITY_1",
    "F_TEMPORAL_PREDICTION_NEXT_IMAGE_GRANULARITY_2",
    "F_TEMPORAL_PREDICTION_NEXT_IMAGE_GRANULARITY_5",
    "F_TEMPORAL_PREDICTION_PREVIOUS_IMAGE",
    "F_TEMPORAL_PREDICTION_MISSING_IMAGE",
}

TEMPORAL_T5_T8_LABELS = {
    "A": "t5",
    "B": "t6",
    "C": "t7",
    "D": "t8",
}

TEMPORAL_STRIP_FOUR_QIDS = {
    "F_TEMPORAL_SEQUENCE_IMAGES",
}


def apply_temporal_question_overrides(qid: str, question_text: str) -> str:
    if qid not in TEMPORAL_T5_T8_QIDS:
        return question_text
    text = question_text
    text = re.sub(
        r"\(\s*A\s*,\s*B\s*,\s*C\s*(?:,\s*or\s*|,\s*)D\s*\)",
        "(t5, t6, t7, or t8)",
        text,
    )
    if "t1" not in text.lower():
        updated = re.sub(
            r"(provided frame sequence)",
            r"\1 (t1, t2, t3, t4)",
            text,
            flags=re.IGNORECASE,
        )
        if updated == text:
            updated = re.sub(
                r"(frame sequence)",
                r"\1 (t1, t2, t3, t4)",
                text,
                flags=re.IGNORECASE,
            )
        text = updated
    return text


def parse_question_and_choices(question_text: str) -> tuple[str, dict[str, str], bool]:
    lines = [line.strip() for line in question_text.splitlines() if line.strip()]

    prompt_lines: list[str] = []
    raw_choices: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^([A-Da-d])[\.\)]\s*(.*)$", line)
        if match:
            key = match.group(1).upper()
            raw_choices[key] = match.group(2).strip()
        else:
            prompt_lines.append(line)

    question = " ".join(prompt_lines).replace("<image>", " ")
    question = " ".join(question.split()).strip()

    choices = {"A": "", "B": "", "C": "", "D": ""}
    for key in choices:
        value = raw_choices.get(key, "")
        value = value.replace("<image>", "").strip()
        choices[key] = value

    non_empty = [v for v in raw_choices.values() if v.strip()]
    options_are_images = bool(non_empty) and all(v.strip() == "<image>" for v in non_empty)

    return question, choices, options_are_images


def build_question_tex(
    question_text: str,
    correct_answer: str | None,
    options_are_images: bool,
    choice_labels: dict[str, str] | None = None,
) -> str:
    question, choices, _ = parse_question_and_choices(question_text)
    question = normalize_text(question)
    for k in list(choices.keys()):
        choices[k] = normalize_text(choices[k])

    label_overrides = choice_labels or {}

    if correct_answer not in choices:
        correct_answer = None

    def fmt_choice(key: str) -> str:
        if key in label_overrides:
            text = label_overrides[key]
        else:
            text = choices.get(key, "") or key
        text = latex_escape(text)
        if correct_answer == key:
            return r"\textbf{" + text + "}"
        return text

    question = latex_escape(question)
    if options_are_images:
        # Image options: render A/B/C/D as labels for image choices.
        def fmt_label(key: str) -> str:
            text = latex_escape(label_overrides.get(key, key))
            if correct_answer == key:
                return r"\textbf{" + text + "}"
            return text

        opts = (
            r"\opts{"
            + fmt_label("A")
            + "}{"
            + fmt_label("B")
            + "}{"
            + fmt_label("C")
            + "}{"
            + fmt_label("D")
            + "}"
        )
        if question:
            return f"{question} {opts}"
        return opts

    choices_present = any(v for v in choices.values())
    if not choices_present:
        return question

    opts = (
        r"\opts{"
        + fmt_choice("A")
        + "}{"
        + fmt_choice("B")
        + "}{"
        + fmt_choice("C")
        + "}{"
        + fmt_choice("D")
        + "}"
    )
    if question:
        return f"{question} {opts}"
    return opts


def unzip_all(base_dir: Path) -> list[Path]:
    extracted = []
    for zip_path in sorted(base_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_dir)
        extracted.append(zip_path)
    return extracted


def find_question_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("question*.json"))


def resolve_images(folder: Path, question_data: dict) -> list[Path]:
    images_dir = folder / "images"
    images: list[Path] = []

    img_entries = question_data.get("images")
    if isinstance(img_entries, list) and img_entries:
        for item in img_entries:
            if isinstance(item, dict) and item.get("filename"):
                images.append(images_dir / item["filename"])

    if not images:
        file_names = question_data.get("file_name") or []
        for item in file_names:
            if not isinstance(item, str):
                continue
            images.append(images_dir / Path(item).name)

    if not images and images_dir.exists():
        images = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

    images = [p for p in images if p.exists()]
    return images


def _pad_images(image_paths: list[str], target: int) -> list[str]:
    if not image_paths:
        return [""] * target
    imgs = image_paths[:target]
    while len(imgs) < target:
        imgs.append(imgs[-1])
    return imgs


def format_strip_four(image_paths: list[str]) -> str:
    imgs = _pad_images(image_paths, 4)
    return r"\stripFour{" + "}{".join(imgs) + "}"


def format_strip_eight(image_paths: list[str]) -> str:
    imgs = _pad_images(image_paths, 8)
    return (
        r"\stripEight{"
        + "}{".join(imgs[:4])
        + "}\n"
        + "                {"
        + "}{".join(imgs[4:])
        + "}"
    )


def format_grid_four(image_paths: list[str]) -> str:
    imgs = _pad_images(image_paths, 4)
    return r"\gridFour{" + "}{".join(imgs) + "}"


def decide_layout(
    image_paths: list[str],
    split: str | None,
    options_are_images: bool,
) -> str:
    if split == "single":
        return format_grid_four(image_paths)
    if options_are_images and len(image_paths) == 4:
        return format_strip_four(image_paths)
    return format_strip_eight(image_paths)


def latex_rel_path(path: Path, base_dir: Path, prefix: str) -> str:
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        rel = path
    rel_posix = rel.as_posix()
    if prefix:
        rel_posix = f"{prefix}/{rel_posix}"
    return rel_posix


def folder_to_idx(folder: Path) -> str:
    name = folder.name
    if name.startswith("folder_"):
        return name[len("folder_") :]
    return name


def sort_key_for_idx(idx: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)(?:_([a-zA-Z]+))?$", idx)
    if not match:
        return (10**18, idx)
    num = int(match.group(1))
    suffix = match.group(2) or ""
    return (num, suffix)


def determine_split(data: dict, raw: dict) -> str | None:
    idx = data.get("idx") or raw.get("idx") or ""
    if isinstance(idx, str):
        if idx.endswith("_i"):
            return "single"
        if idx.endswith("_g"):
            return "multi"
    mode = str(data.get("mode") or raw.get("mode") or "").lower()
    if "image" in mode or "single" in mode:
        return "single"
    if "general" in mode or "multi" in mode:
        return "multi"
    return None


def build_variant_from_folder(
    folder: Path,
    base_dir: Path,
    path_prefix: str,
    simple_meta: dict[str, dict[str, str]] | None = None,
) -> tuple[dict | None, list[Path]]:
    question_files = find_question_files(folder)
    if not question_files:
        return None, []

    data = json.loads(question_files[0].read_text())
    raw = data.get("raw") or {}
    qid = data.get("question_id") or raw.get("question_id") or ""
    if not qid:
        return None, []
    split = determine_split(data, raw)
    if split is None:
        return None, []

    category = title_from_key(raw.get("category") or data.get("category"))
    sub_category = title_from_key(raw.get("sub_category") or data.get("sub_category"))
    if simple_meta:
        meta = simple_meta.get(str(qid))
        if meta:
            if category == "Unknown":
                category = meta.get("category", category)
            if sub_category == "Unknown":
                sub_category = meta.get("sub_category", sub_category)

    images = resolve_images(folder, data)
    image_paths = [latex_rel_path(p, base_dir, path_prefix) for p in images]
    preview_image = image_paths[0] if image_paths else ""

    q_text = data.get("question") or raw.get("question") or ""
    q_text = apply_temporal_question_overrides(str(qid), q_text)
    _, _, options_are_images = parse_question_and_choices(q_text)
    correct = data.get("correct_answer")
    choice_labels = TEMPORAL_T5_T8_LABELS if str(qid) in TEMPORAL_T5_T8_QIDS else None
    question_tex = build_question_tex(
        q_text, correct, options_are_images, choice_labels=choice_labels
    )
    layout = decide_layout(image_paths, split, options_are_images)
    if str(qid) in TEMPORAL_STRIP_FOUR_QIDS:
        layout = format_strip_four(image_paths)

    variant = {
        "question_id": str(qid),
        "split": split,
        "idx": str(data.get("idx") or raw.get("idx") or ""),
        "category": category,
        "sub_category": sub_category,
        "preview_image": preview_image,
        "layout": layout,
        "question_tex": question_tex,
    }

    return variant, images


def copy_images(
    image_paths: list[Path], base_dir: Path, dest_dir: Path
) -> tuple[int, list[Path]]:
    copied = 0
    dest_paths: list[Path] = []
    for src in image_paths:
        try:
            rel = src.relative_to(base_dir)
        except ValueError:
            rel = src.name
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1
        dest_paths.append(dest)
    return copied, dest_paths


def optimize_images_with_oxipng(image_paths: list[Path], level: int = 4) -> int:
    if not image_paths:
        return 0
    if shutil.which("oxipng") is None:
        print("oxipng not found; skipping image optimization.")
        return 0
    optimized = 0
    for path in image_paths:
        if path.suffix.lower() != ".png":
            continue
        result = subprocess.run(
            ["oxipng", f"-o{level}", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            optimized += 1
    return optimized


def load_simple_vqa(path: Path) -> dict[str, dict[str, bool]]:
    data = json.loads(path.read_text())
    required: dict[str, dict[str, bool]] = {}
    for cat in data.values():
        if not isinstance(cat, dict):
            continue
        for qid, item in cat.items():
            if not isinstance(item, dict):
                continue
            task_splits = str(item.get("task_splits") or "")
            tokens = [t.strip().lower() for t in task_splits.split("+") if t.strip()]
            required[str(qid)] = {
                "single": "single" in tokens,
                "multi": "multi" in tokens,
            }
    return required


def load_simple_vqa_metadata(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    data = json.loads(path.read_text())
    qid_order: list[str] = []
    metadata: dict[str, dict[str, str]] = {}
    for cat_key, cat in data.items():
        if not isinstance(cat, dict):
            continue
        for qid, item in cat.items():
            if not isinstance(item, dict):
                continue
            qid_str = str(qid)
            qid_order.append(qid_str)
            sub = item.get("sub_category")
            metadata[qid_str] = {
                "category": title_from_key(str(cat_key)),
                "sub_category": title_from_key(str(sub)) if sub else "Unknown",
            }
    return qid_order, metadata


def _is_emoji(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1F300 <= code <= 0x1FAFF
        or 0x1F1E6 <= code <= 0x1F1FF
        or 0x2600 <= code <= 0x26FF
        or 0x2700 <= code <= 0x27BF
    )


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def display_width(text: str) -> int:
    text = ANSI_RE.sub("", text)
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        if _is_emoji(ch) or unicodedata.east_asian_width(ch) in {"W", "F"}:
            width += 2
        else:
            width += 1
    return width


def pad_right(text: str, width: int) -> str:
    pad = max(0, width - display_width(text))
    return text + (" " * pad)


def pad_center(text: str, width: int) -> str:
    pad = max(0, width - display_width(text))
    left = pad // 2
    right = pad - left
    return (" " * left) + text + (" " * right)


def collect_available_splits(
    base_dir: Path,
) -> dict[str, dict[str, dict[str, set[str]]]]:
    available: dict[str, dict[str, dict[str, set[str]]]] = {}
    for folder in base_dir.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        for qpath in find_question_files(folder):
            data = json.loads(qpath.read_text())
            raw = data.get("raw") or {}
            qid = data.get("question_id") or raw.get("question_id")
            if not qid:
                continue
            idx = data.get("idx") or raw.get("idx") or ""
            split = None
            if isinstance(idx, str):
                if idx.endswith("_i"):
                    split = "single"
                elif idx.endswith("_g"):
                    split = "multi"
            if split is None:
                mode = str(data.get("mode") or raw.get("mode") or "").lower()
                if "image" in mode or "single" in mode:
                    split = "single"
                elif "general" in mode or "multi" in mode:
                    split = "multi"
            if split is None:
                continue
            qid_key = str(qid)
            split_entry = available.setdefault(qid_key, {}).setdefault(
                split, {"idxs": set(), "scenes": set()}
            )
            split_entry["idxs"].add(str(idx))
            scene_id = data.get("scene") or raw.get("scene") or ""
            if scene_id:
                split_entry["scenes"].add(str(scene_id))
    return available


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate vqaSplitEntry LaTeX blocks from question folders."
    )
    parser.add_argument(
        "--base-dir",
        default="/Users/sebastiancavada/Downloads/questions_paper",
        help="Directory containing question folders and optional zip files.",
    )
    parser.add_argument(
        "--out-txt",
        default="vqa_split_entries.txt",
        help="Output .txt file with vqaSplitEntry blocks.",
    )
    parser.add_argument(
        "--path-prefix",
        "--image-prefix",
        default="figures/supp/questions_paper",
        help="Prefix for image paths in LaTeX output.",
    )
    parser.add_argument(
        "--placeholder",
        default=r"\textbf{No question provided/possible}",
        help="Placeholder for missing second question.",
    )
    parser.add_argument(
        "--copy-images-dir",
        default="figures/supp/questions_paper",
        help="Destination folder to copy images for Overleaf upload.",
    )
    parser.add_argument(
        "--optimize-images",
        action="store_true",
        help="Run lossless PNG optimization with oxipng on copied images.",
    )
    parser.add_argument(
        "--simple-vqa-json",
        default="/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/simple_vqa.json",
        help="Path to simple_vqa.json for missing question_id summary.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    if args.path_prefix and not args.path_prefix.startswith("figures/supp/"):
        args.path_prefix = f"figures/supp/{args.path_prefix.lstrip('/')}"

    simple_path = Path(args.simple_vqa_json)
    simple_qid_order: list[str] = []
    simple_meta: dict[str, dict[str, str]] = {}
    if simple_path.exists():
        simple_qid_order, simple_meta = load_simple_vqa_metadata(simple_path)

    extracted = unzip_all(base_dir)

    folders = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: sort_key_for_idx(folder_to_idx(p)),
    )

    entries: list[str] = []
    all_images: list[Path] = []
    variants_by_qid: dict[str, dict[str, dict]] = {}
    qid_order: list[str] = []
    seen_qids: set[str] = set()
    for folder in folders:
        variant, images = build_variant_from_folder(
            folder=folder,
            base_dir=base_dir,
            path_prefix=args.path_prefix,
            simple_meta=simple_meta,
        )
        if variant:
            qid = variant["question_id"]
            split = variant["split"]
            variants_by_qid.setdefault(qid, {})[split] = variant
            if qid not in seen_qids:
                seen_qids.add(qid)
                qid_order.append(qid)
        all_images.extend(images)

    if simple_qid_order:
        known = [qid for qid in simple_qid_order if qid in variants_by_qid]
        extras = [qid for qid in qid_order if qid not in set(known)]
        qid_order = known + extras

    for qid in qid_order:
        splits = variants_by_qid[qid]
        variant_multi = splits.get("multi")
        variant_single = splits.get("single")
        base_variant = variant_multi or variant_single
        if not base_variant:
            continue

        q_single = (
            variant_single["question_tex"] if variant_single else args.placeholder
        )
        q_multi = variant_multi["question_tex"] if variant_multi else args.placeholder
        preview_image = variant_single["preview_image"] if variant_single else ""
        layout = variant_multi["layout"] if variant_multi else ""

        entry = [
            f"% --- ENTRY {qid} ---",
            rf"\vqaSplitEntry{{{format_entry_id(qid)}}}",
            f"    {{{latex_escape(base_variant['category'])}}}",
            f"    {{{latex_escape(base_variant['sub_category'])}}}",
            f"    {{{preview_image}}}",
            f"    {{{layout}}}",
            f"    {{{q_single}}}",
            f"    {{{q_multi}}}",
            "",
        ]
        entries.append("\n".join(entry))

    out_path = Path(args.out_txt)
    out_path.write_text("\n".join(entries).rstrip() + "\n", encoding="utf-8")

    copied = 0
    copied_paths: list[Path] = []
    if args.copy_images_dir:
        dest_dir = Path(args.copy_images_dir)
        copied, copied_paths = copy_images(all_images, base_dir, dest_dir)
        if args.optimize_images:
            optimized = optimize_images_with_oxipng(copied_paths)
            print(f"Images optimized (oxipng): {optimized}")

    table_rows: list[tuple[str, str, str, str, str, str]] = []
    completed = 0
    total = 0
    scene_raw_values: list[str] = []
    if simple_path.exists():
        required = load_simple_vqa(simple_path)
        available = collect_available_splits(base_dir)
        for qid, needed in sorted(required.items()):
            single_expected = needed.get("single", False)
            multi_expected = needed.get("multi", False)
            avail = available.get(qid, {})
            single_entry = avail.get("single", {"idxs": set(), "scenes": set()})
            multi_entry = avail.get("multi", {"idxs": set(), "scenes": set()})
            single_idxs = sorted(single_entry.get("idxs", set()))
            multi_idxs = sorted(multi_entry.get("idxs", set()))
            single_scenes = sorted(single_entry.get("scenes", set()))
            multi_scenes = sorted(multi_entry.get("scenes", set()))

            if single_expected:
                total += 1
                single_mark = "✅" if single_idxs else "❌"
                if single_idxs:
                    completed += 1
            else:
                single_mark = "—"

            if multi_expected:
                total += 1
                multi_mark = "✅" if multi_idxs else "❌"
                if multi_idxs:
                    completed += 1
            else:
                multi_mark = "—"

            def fmt_idx(idxs: list[str], expected: bool) -> str:
                if not expected:
                    return "—"
                if not idxs:
                    return ""
                return ", ".join(idxs)

            scene_parts: list[str] = []
            if single_expected:
                if single_scenes:
                    scene_parts.extend(single_scenes)
            if multi_expected:
                if multi_scenes:
                    scene_parts.extend(multi_scenes)

            unique_scenes = sorted(set(scene_parts))
            if not unique_scenes:
                scene_cell = "—"
                scene_raw = ""
            elif len(unique_scenes) == 1:
                scene_cell = unique_scenes[0]
                scene_raw = unique_scenes[0]
            else:
                scene_cell = " | ".join(unique_scenes)
                scene_raw = ""

            scene_raw_values.append(scene_raw)

            table_rows.append(
                (
                    qid,
                    scene_cell,
                    single_mark if single_expected else "—",
                    ", ".join(single_idxs) if single_expected else "—",
                    multi_mark if multi_expected else "—",
                    ", ".join(multi_idxs) if multi_expected else "—",
                )
            )

    print(f"Extracted zips: {len(extracted)}")
    print(f"Question folders: {len(folders)}")
    print(f"Entries written: {len(entries)}")
    print(f"Wrote: {out_path}")
    if args.copy_images_dir:
        print(f"Images copied: {copied}")
        print(f"Copy dir: {args.copy_images_dir}")
    if simple_path.exists():
        scene_counts: dict[str, int] = {}
        for scene in scene_raw_values:
            if scene:
                scene_counts[scene] = scene_counts.get(scene, 0) + 1

        def color_scene(text: str) -> str:
            if not text or " | " in text:
                return text
            count = scene_counts.get(text, 0)
            if count > 1:
                return f"\033[31m{text}\033[0m"
            return f"\033[32m{text}\033[0m"

        header = ("question_id", "scene_id", "single", "single_idx", "multi", "multi_idx")
        widths = []
        for i in range(len(header)):
            header_w = display_width(header[i])
            if i == 1:
                row_w = max(
                    [display_width(color_scene(r[1])) for r in table_rows], default=0
                )
            else:
                row_w = max([display_width(r[i]) for r in table_rows], default=0)
            widths.append(max(header_w, row_w, 1))

        header_line = (
            f"{pad_right(header[0], widths[0])} | "
            f"{pad_right(header[1], widths[1])} | "
            f"{pad_right(header[2], widths[2])} | "
            f"{pad_right(header[3], widths[3])} | "
            f"{pad_right(header[4], widths[4])} | "
            f"{pad_right(header[5], widths[5])} |"
        )
        row_sep = (
            f"{'-' * widths[0]}-+-{'-' * widths[1]}-+-{'-' * widths[2]}-+-"
            f"{'-' * widths[3]}-+-{'-' * widths[4]}-+-{'-' * widths[5]}-+"
        )
        print(header_line)
        print(row_sep)
        for qid, scene_cell, s_mark, s_idx, m_mark, m_idx in table_rows:
            row_line = (
                f"{pad_right(qid, widths[0])} | "
                f"{pad_right(color_scene(scene_cell), widths[1])} | "
                f"{pad_center(s_mark, widths[2])} | "
                f"{pad_right(s_idx, widths[3])} | "
                f"{pad_center(m_mark, widths[4])} | "
                f"{pad_right(m_idx, widths[5])} |"
            )
            print(row_line)
            print(row_sep)
        if total:
            print(f"Completed: {completed}/{total}")
        used_scenes = sorted({s for s in scene_raw_values if s})
        print(f"Scene IDs used: {len(used_scenes)}")
        if used_scenes:
            print("Scene IDs:")
            for scene_id in used_scenes:
                print(f"  {scene_id}")


if __name__ == "__main__":
    main()
