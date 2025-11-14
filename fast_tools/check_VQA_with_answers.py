import argparse
import glob
import json
import math
import os
import random
import textwrap
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    import imageio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    imageio = None


def load_json(path: str) -> Iterable:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_question_parts(question: str) -> Tuple[str, List[Tuple[str, str]]]:
    lines = [line.strip() for line in question.splitlines()]
    question_lines: List[str] = []
    options: List[Tuple[str, str]] = []

    for raw_line in lines:
        if not raw_line or raw_line.lower().startswith("<image"):
            continue
        if len(raw_line) > 2 and raw_line[1] == "." and raw_line[0].isalpha():
            letter = raw_line[0].upper()
            option_text = raw_line[2:].strip()
            options.append((letter, option_text))
        else:
            question_lines.append(raw_line)

    joined_question = " ".join(question_lines).strip()
    return joined_question, options


def ensure_answer_map(entries: Iterable) -> Dict[str, str]:
    answer_map: Dict[str, str] = {}
    if isinstance(entries, dict):
        # Already a mapping from idx to answer.
        return {key: str(value).strip().upper() for key, value in entries.items()}

    for item in entries:
        if not isinstance(item, dict):
            continue
        idx = str(item.get("idx", "")).strip()
        ans = str(item.get("answer", "")).strip().upper()
        if idx and ans:
            answer_map[idx] = ans
    return answer_map


def normalize_answer_letter(raw_answer: object) -> str:
    """Return the first alphabetical character (upper-cased) found in the answer."""
    answer = str(raw_answer or "").strip().upper()
    for char in answer:
        if char.isalpha():
            return char
    return ""


def load_model_answers(paths: Iterable[str]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[Tuple[str, str]]]]:
    """
    Load per-model answers from one or more files/directories.

    Returns a tuple with:
        * mapping from question idx -> model name -> normalized answer letter.
        * mapping from question idx -> list of (model name, raw answer) for entries that
          could not be normalized to a single choice letter.
    """
    model_answers: Dict[str, Dict[str, str]] = {}
    invalid_answers: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def _iter_json_files(path_item: str) -> Iterable[Tuple[str, str]]:
        if os.path.isdir(path_item):
            for file_path in sorted(glob.glob(os.path.join(path_item, "*.json"))):
                yield file_path, os.path.splitext(os.path.basename(file_path))[0]
        elif os.path.isfile(path_item):
            yield path_item, os.path.splitext(os.path.basename(path_item))[0]

    for path in paths:
        for file_path, model_name in _iter_json_files(path):
            try:
                entries = load_json(file_path)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Warning: could not load {file_path}: {exc}")
                continue

            for item in entries:
                if not isinstance(item, dict):
                    continue
                idx = str(item.get("idx", "")).strip()
                if not idx:
                    continue
                answer_letter = normalize_answer_letter(item.get("answer", ""))
                if not answer_letter:
                    invalid_answers[idx].append((model_name, str(item.get("answer", ""))))
                    continue
                model_answers.setdefault(idx, {})[model_name] = answer_letter

    return model_answers, invalid_answers


def resolve_image_path(image_path: str, search_dirs: Iterable[str]) -> str:
    """Return an existing path for the given image if it can be resolved."""
    if not image_path:
        return image_path
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path

    for base in search_dirs:
        if not base:
            continue
        candidate = os.path.join(base, image_path)
        if os.path.exists(candidate):
            return candidate

    absolute_candidate = os.path.abspath(image_path)
    if os.path.exists(absolute_candidate):
        return absolute_candidate

    return image_path


def display_entry(
    entry: Dict,
    answer_map: Dict[str, str],
    imgs_per_row: int,
    model_answers: Optional[Dict[str, Dict[str, str]]] = None,
    invalid_model_answers: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> None:
    idx = entry.get("idx", "unknown")
    question_raw = entry.get("question", "")
    media_paths = entry.get("file_name") or []
    if isinstance(media_paths, str):
        media_paths = [media_paths]

    question_image_paths: List[str] = []
    question_video_paths: List[str] = []
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    for path in media_paths:
        if not path:
            continue
        extension = os.path.splitext(str(path))[1].lower()
        if extension in video_extensions:
            question_video_paths.append(str(path))
        else:
            question_image_paths.append(str(path))

    question_text, options = parse_question_parts(question_raw)
    correct_letter = answer_map.get(idx, "")

    model_answers_for_idx = model_answers.get(idx, {}) if model_answers else {}
    invalid_answers_for_idx = (
        invalid_model_answers.get(idx, []) if invalid_model_answers else []
    )
    models_by_letter: Dict[str, List[str]] = defaultdict(list)
    if model_answers_for_idx:
        for model_name, letter in model_answers_for_idx.items():
            models_by_letter[letter].append(model_name)
        for letter_models in models_by_letter.values():
            letter_models.sort()

    entry_options = entry.get("options")
    normalized_options = {}
    if isinstance(entry_options, dict):
        normalized_options = {
            str(letter).upper(): str(text).strip()
            for letter, text in entry_options.items()
        }
        if options:
            options = [
                (letter, normalized_options.get(letter, text))
                for letter, text in options
            ]
        else:
            options = [
                (letter, normalized_options[letter])
                for letter in sorted(normalized_options)
            ]

    wrapped_question = "\n".join(
        textwrap.wrap(question_text, width=90)
    ) if question_text else "No question text available."

    search_roots: List[str] = []
    potential_video_roots: set[str] = set()

    def add_search_root(root: Optional[str]) -> None:
        if not root:
            return
        if os.path.isdir(root) and root not in search_roots:
            search_roots.append(root)

    def register_video_root(root: Optional[str]) -> None:
        if not root:
            return
        if os.path.isdir(root):
            if root not in potential_video_roots:
                potential_video_roots.add(root)
            add_search_root(root)

    for candidate in question_image_paths + question_video_paths:
        directory = os.path.dirname(candidate)
        register_video_root(directory)
        if directory and os.path.basename(directory) == "render":
            register_video_root(os.path.dirname(directory))

    simulation_path = entry.get("simulation_id")
    if isinstance(simulation_path, str) and simulation_path:
        register_video_root(os.path.dirname(simulation_path))

    question_image_paths = [
        resolve_image_path(path, search_roots) for path in question_image_paths
    ]
    question_video_paths = [
        resolve_image_path(path, search_roots) for path in question_video_paths
    ]

    for resolved_path in question_image_paths + question_video_paths:
        register_video_root(os.path.dirname(resolved_path))

    option_files_raw = entry.get("option_files")
    option_image_items = []
    if isinstance(option_files_raw, dict):
        normalized_option_files = {
            str(letter).upper(): str(path)
            for letter, path in option_files_raw.items()
        }
        option_order = [letter for letter, _ in options] if options else sorted(
            normalized_option_files
        )
        seen_letters = set()
        for letter in option_order:
            seen_letters.add(letter)
            path = normalized_option_files.get(letter, "")
            if not path:
                continue
            resolved = resolve_image_path(path, search_roots)
            register_video_root(os.path.dirname(resolved))
            option_text = next(
                (text for opt_letter, text in options if opt_letter == letter),
                normalized_options.get(letter, ""),
            )
            option_image_items.append((letter, option_text, resolved))
        for letter, path in normalized_option_files.items():
            letter_upper = str(letter).upper()
            if letter_upper in seen_letters or not path:
                continue
            resolved = resolve_image_path(path, search_roots)
            register_video_root(os.path.dirname(resolved))
            option_text = normalized_options.get(letter_upper, "")
            option_image_items.append((letter_upper, option_text, resolved))

    video_patterns = (
        "*_fps-25_render.mp4",
    )
    discovered_videos: List[str] = []
    for root in sorted(potential_video_roots):
        for pattern in video_patterns:
            for candidate in sorted(glob.glob(os.path.join(root, pattern))):
                if candidate not in discovered_videos:
                    discovered_videos.append(candidate)

    all_video_paths: List[str] = []
    for path in question_video_paths + discovered_videos:
        if path and path not in all_video_paths:
            all_video_paths.append(path)

    question_media_items: List[Tuple[str, str]] = [
        ("image", path) for path in question_image_paths
    ]
    question_media_items.extend(("video", path) for path in all_video_paths)

    n_question_media = len(question_media_items)
    n_option_images = len(option_image_items)
    cols = min(
        imgs_per_row,
        max(n_question_media, n_option_images, 1),
    )
    cols = max(1, cols)

    question_rows = math.ceil(n_question_media / cols) if n_question_media else 0
    option_rows = math.ceil(n_option_images / cols) if n_option_images else 0
    total_rows = 1 + question_rows + option_rows  # Extra row for text.

    fig_width = max(16.0, cols * 4.0)
    fig_height = max(9.0, question_rows * 3.5 + option_rows * 3.0 + 2.5)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(total_rows, cols)

    text_ax = fig.add_subplot(gs[0, :])
    text_ax.axis("off")
    text_ax.text(
        0.01,
        0.95,
        f"ID: {idx}",
        fontsize=11,
        color="gray",
        ha="left",
        va="top",
        transform=text_ax.transAxes,
    )
    text_ax.text(
        0.01,
        0.8,
        wrapped_question,
        fontsize=12,
        ha="left",
        va="top",
        transform=text_ax.transAxes,
    )

    primary_img_path = question_image_paths[0] if question_image_paths else None
    info_cursor = 0.65
    if primary_img_path:
        wrapped_path = textwrap.fill(f"Image path: {primary_img_path}", width=95)
        text_ax.text(
            0.01,
            info_cursor,
            wrapped_path,
            fontsize=10,
            ha="left",
            va="top",
            color="gray",
            transform=text_ax.transAxes,
        )
        info_cursor -= 0.12

    if all_video_paths:
        for vid_index, video_path in enumerate(all_video_paths, start=1):
            label = f"Video {vid_index}: {video_path}"
            wrapped_video = textwrap.fill(label, width=95)
            text_ax.text(
                0.01,
                info_cursor,
                wrapped_video,
                fontsize=10,
                ha="left",
                va="top",
                color="gray",
                transform=text_ax.transAxes,
            )
            info_cursor -= 0.12

    if options:
        option_y = info_cursor if (primary_img_path or all_video_paths) else 0.5
        option_y = max(option_y, 0.15)
        option_letters = {letter for letter, _ in options}
        for letter, opt_text in options:
            color = "green" if letter == correct_letter else "black"
            models_for_option = models_by_letter.get(letter, [])
            models_preview = ""
            if models_for_option:
                total = len(models_for_option)
                preview = ", ".join(models_for_option[:3])
                if total > 3:
                    preview += ", ..."
                plural = "s" if total != 1 else ""
                models_preview = f" ({total} model{plural}: {preview})"
            wrapped_option = textwrap.fill(
                f"{letter}. {opt_text}{models_preview}",
                width=95,
            )
            text_ax.text(
                0.01,
                option_y,
                wrapped_option,
                fontsize=11,
                ha="left",
                va="top",
                color=color,
                transform=text_ax.transAxes,
            )
            option_y -= 0.12
        off_option_answers = {
            letter: models
            for letter, models in models_by_letter.items()
            if letter not in option_letters
        }
        if off_option_answers:
            for letter, models in sorted(off_option_answers.items()):
                preview = ", ".join(models[:3])
                if len(models) > 3:
                    preview += ", ..."
                warning_text = textwrap.fill(
                    f"Models answering non-listed option {letter}: {preview}",
                    width=95,
                )
                text_ax.text(
                    0.01,
                    option_y,
                    warning_text,
                    fontsize=10,
                    ha="left",
                    va="top",
                    color="firebrick",
                    transform=text_ax.transAxes,
                )
                option_y -= 0.1
        if invalid_answers_for_idx:
            for model_name, raw_answer in invalid_answers_for_idx[:3]:
                warning_text = textwrap.fill(
                    f"{model_name} provided an unrecognized answer: {raw_answer}",
                    width=95,
                )
                text_ax.text(
                    0.01,
                    option_y,
                    warning_text,
                    fontsize=10,
                    ha="left",
                    va="top",
                    color="firebrick",
                    transform=text_ax.transAxes,
                )
                option_y -= 0.1
            remaining = len(invalid_answers_for_idx) - 3
            if remaining > 0:
                text_ax.text(
                    0.01,
                    option_y,
                    f"... and {remaining} more invalid answers.",
                    fontsize=10,
                    ha="left",
                    va="top",
                    color="firebrick",
                    transform=text_ax.transAxes,
                )
                option_y -= 0.1
    elif correct_letter:
        text_ax.text(
            0.01,
            0.5,
            f"Answer: {correct_letter}",
            fontsize=11,
            ha="left",
            va="top",
                color="green",
                transform=text_ax.transAxes,
        )

    cleanup_callbacks: List[Callable[[], None]] = []
    cleanup_state = {"done": False}
    video_animations: List[animation.FuncAnimation] = []

    question_axes = [
        fig.add_subplot(gs[1 + row, col])
        for row in range(question_rows)
        for col in range(cols)
    ]
    for idx_media, ax in enumerate(question_axes):
        ax.axis("off")
        if idx_media >= len(question_media_items):
            continue

        media_type, media_path = question_media_items[idx_media]
        if media_type == "image":
            ax.set_title(os.path.basename(media_path), fontsize=10, pad=6, color="gray")
            if not os.path.exists(media_path):
                ax.text(
                    0.5,
                    0.5,
                    f"Missing image:\n{os.path.basename(media_path)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
                continue
            try:
                image = mpimg.imread(media_path)
                ax.imshow(image)
            except Exception as exc:  # pylint: disable=broad-except
                ax.text(
                    0.5,
                    0.5,
                    f"Error loading image:\n{exc}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
        else:
            ax.set_title(os.path.basename(media_path), fontsize=10, pad=6, color="gray")
            if not os.path.exists(media_path):
                ax.text(
                    0.5,
                    0.5,
                    f"Missing video:\n{os.path.basename(media_path)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
                continue

            if imageio is None:
                ax.text(
                    0.5,
                    0.5,
                    "Video playback requires imageio",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
                continue

            try:
                reader = imageio.get_reader(media_path)
                meta = reader.get_meta_data()
                fps = float(meta.get("fps", 24.0) or 24.0)
                frame_iterator = reader.iter_data()
                first_frame = next(frame_iterator)
            except Exception as exc:  # pylint: disable=broad-except
                ax.text(
                    0.5,
                    0.5,
                    f"Error loading video:\n{exc}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
                try:
                    reader.close()  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - best effort
                    pass
                continue

            image_artist = ax.imshow(first_frame)
            ax.set_xlabel(os.path.basename(media_path), fontsize=8, color="gray")

            def _cleanup_reader(target=reader) -> None:
                try:
                    target.close()
                except Exception:  # pragma: no cover - best effort
                    pass

            cleanup_callbacks.append(_cleanup_reader)

            state = {
                "reader": reader,
                "iterator": frame_iterator,
            }

            def _advance(_frame_index: int, artist=image_artist, state=state, ax=ax) -> tuple:
                iterator = state["iterator"]
                try:
                    frame = next(iterator)
                except StopIteration:
                    try:
                        iterator = state["reader"].iter_data()
                        frame = next(iterator)
                        state["iterator"] = iterator
                    except Exception as exc_inner:  # pylint: disable=broad-except
                        ax.text(
                            0.5,
                            0.5,
                            f"Playback error:\n{exc_inner}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="red",
                        )
                        return (artist,)
                artist.set_data(frame)
                return (artist,)

            interval_ms = max(1, int(1000.0 / fps))
            anim = animation.FuncAnimation(
                fig,
                _advance,
                interval=interval_ms,
                blit=True,
                repeat=True,
            )
            video_animations.append(anim)

    if option_image_items:
        option_offset = 1 + question_rows
        option_axes = [
            fig.add_subplot(gs[option_offset + row, col])
            for row in range(option_rows)
            for col in range(cols)
        ]
        for opt_idx, (letter, opt_text, img_path) in enumerate(option_image_items):
            if opt_idx >= len(option_axes):
                break
            ax = option_axes[opt_idx]
            ax.axis("off")
            title_color = "green" if letter == correct_letter else "gray"
            caption = "\n".join(textwrap.wrap(f"{letter}. {opt_text}" if opt_text else letter, width=40))
            try:
                image = mpimg.imread(img_path)
                ax.imshow(image)
            except Exception as exc:  # pylint: disable=broad-except
                ax.text(
                    0.5,
                    0.5,
                    f"Error loading option image:\n{exc}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
            ax.set_title(caption, fontsize=10, color=title_color)
            if not os.path.exists(img_path):
                ax.text(
                    0.5,
                    0.05,
                    f"Missing image:\n{os.path.basename(img_path)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="red",
                    transform=ax.transAxes,
                )
            else:
                ax.set_xlabel(
                    os.path.basename(img_path),
                    fontsize=8,
                    color="gray",
                )
            if letter == correct_letter:
                for spine in ax.spines.values():
                    spine.set_edgecolor("green")
                    spine.set_linewidth(2.0)

    fig._video_animations = video_animations  # type: ignore[attr-defined]
    fig.suptitle(entry.get("question_id", ""), fontsize=10, color="gray")
    plt.tight_layout()

    def _run_cleanups() -> None:
        if cleanup_state["done"]:
            return
        cleanup_state["done"] = True
        for callback in cleanup_callbacks:
            try:
                callback()
            except Exception:  # pragma: no cover - best effort
                pass

    def _close_on_click(_event) -> None:
        _run_cleanups()
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _close_on_click)
    fig.canvas.mpl_connect("close_event", lambda _evt: _run_cleanups())

    plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizer for Tiny VQA questions and answers."
    )
    parser.add_argument("test_json", help="Path to the test JSON file.")
    parser.add_argument("val_answers_json", help="Path to the answers JSON file.")
    parser.add_argument(
        "--results-paths",
        nargs="+",
        help=(
            "Optional list of JSON files or directories containing per-model VLM "
            "answers (e.g. ../output/results_run04_1K)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of questions shown."
    )
    parser.add_argument(
        "--imgs-per-row",
        type=int,
        default=3,
        help="Maximum number of images per row in the figure (default: 3).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Only display questions whose category matches one of the provided values.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Shuffle the questions before displaying them.",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Only display entries whose question_id contains one of the provided values.",
    )

    args = parser.parse_args()

    test_entries = load_json(args.test_json)
    answer_entries = load_json(args.val_answers_json)
    # print(test_entries)

    answer_map = ensure_answer_map(answer_entries)
    if not answer_map:
        print("Warning: No answers could be read from the answers file.")

    model_answers_map: Dict[str, Dict[str, str]] = {}
    invalid_model_answers: Dict[str, List[Tuple[str, str]]] = {}
    if args.results_paths:
        model_answers_map, invalid_model_answers = load_model_answers(
            args.results_paths
        )
        if not model_answers_map:
            print(
                "Warning: No model answers were loaded from the provided results paths."
            )

    count = 0
    entries = test_entries[:]
    # print(entries)
    if args.categories:
        requested = {category.lower() for category in args.categories}
        all_entries = entries = [
            entry
            for entry in entries
        ]
        # print(all_entries)
        # print(requested)
        entries = [
            entry
            for entry in entries
            if str(entry.get("category", "")).lower() in requested
        ]
        if not entries:
            print("Warning: No entries matched the requested categories.")

    if args.question_ids:
        requested_id_fragments = [str(qid) for qid in args.question_ids]
        entries = [
            entry
            for entry in entries
            if any(
                fragment in str(entry.get("question_id", ""))
                for fragment in requested_id_fragments
            )
        ]
        if not entries:
            print("Warning: No entries matched the requested question IDs.")

    if args.random:
        random.shuffle(entries)

    for entry in entries:
        display_entry(
            entry,
            answer_map,
            imgs_per_row=max(1, args.imgs_per_row),
            model_answers=model_answers_map if model_answers_map else None,
            invalid_model_answers=(
                invalid_model_answers if invalid_model_answers else None
            ),
        )
        count += 1
        if args.limit is not None and count >= args.limit:
            break


if __name__ == "__main__":
    main()


# python check_VQA.py ../output/test.json ../output/val_answer_run02_100.json --limit 5
# python check_VQA.py ../output/test_test_imbalance.json ../output/val_answer_test_imbalance.json --cat temporal
