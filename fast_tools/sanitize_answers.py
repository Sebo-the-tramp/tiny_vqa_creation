import argparse
import json
import re
from pathlib import Path


_ANSWER_RE = re.compile(r"(?:^([A-D])\b|\b([A-D])\s*[\.\,\:\)]|\b([A-D])\b$)")


def sanitize_answer(answer, max_prefix_chars=None):
    if max_prefix_chars is None or max_prefix_chars < 0:
        text = str(answer)
    else:
        text = str(answer)[:max_prefix_chars]
    match = _ANSWER_RE.search(text)
    if not match:
        return "?"
    extracted = next((group for group in match.groups() if group), None)
    return extracted.upper()


def sanitize_in_place(obj, max_prefix_chars=None):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "answer":
                obj[key] = sanitize_answer(value, max_prefix_chars=max_prefix_chars)
            else:
                sanitize_in_place(value, max_prefix_chars=max_prefix_chars)
    elif isinstance(obj, list):
        for item in obj:
            sanitize_in_place(item, max_prefix_chars=max_prefix_chars)


def sanitize_folder(input_dir, output_dir, max_prefix_chars=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    for src in json_files:
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sanitize_in_place(data, max_prefix_chars=max_prefix_chars)
        dst = output_path / src.name
        with dst.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize answer fields in JSON files from input folder into output folder."
    )
    parser.add_argument("input_dir", help="Folder containing JSON files.")
    parser.add_argument("output_dir", help="Folder to write sanitized JSON files.")
    parser.add_argument(
        "--max-prefix-chars",
        type=int,
        default=None,
        help="Truncate answers to this many chars before matching. Use -1 for no limit.",
    )
    args = parser.parse_args()
    sanitize_folder(
        args.input_dir, args.output_dir, max_prefix_chars=args.max_prefix_chars
    )


if __name__ == "__main__":
    main()


# python fast_tools/sanitize_answers.py /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_23_general/results_run_23_general /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_23_general/results_run_23_general_sanitized --max-prefix-chars -1