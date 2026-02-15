#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def k_label(n: int) -> str:
    return f"{n / 1000:g}"


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def chunk_dict(data, size):
    items = list(data.items())
    for i in range(0, len(items), size):
        yield dict(items[i:i + size])


def main():
    parser = argparse.ArgumentParser(description="Slice a JSON file into multiple chunked JSON files.")
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("x", type=int, help="Number of question IDs (entries) per output file")
    args = parser.parse_args()

    if args.x <= 0:
        raise SystemExit("x must be > 0")

    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stem = input_path.stem
    parent = input_path.parent
    label = k_label(args.x)

    if isinstance(data, list):
        chunks = chunk_list(data, args.x)
    elif isinstance(data, dict):
        chunks = chunk_dict(data, args.x)
    else:
        raise SystemExit("Input JSON must be a list or dict.")

    count = 0
    for i, chunk in enumerate(chunks, start=1):
        out_path = parent / f"{stem}_karo_{label}K_{i}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False)
        count += 1

    print(f"Wrote {count} files in {parent}")


if __name__ == "__main__":
    main()
