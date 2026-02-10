#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Canonical ablation names mapped to directory suffix candidates.
VARIANT_ALIASES = {
    "ablation_roi_circling_text": ["roi_circling_text"],
    "ablation_baseline": ["roi_ablation_baseline", "ablation_baseline"],
    "ablation_roi_circling_no_text": ["roi_circling_no_text"],
    "ablation_roi_circling_no_text_layout_position": [
        "roi_circling_no_text_layout_position"
    ],
    "ablation_roi_circling_text_layout_position": [
        "roi_circling_text_layout_position"
    ],
    "ablation_no_roi_no_text_layout_position": [
        "no_roi_circling_no_text_layout_position",
        "no_roi_circling_no_text_yes_layout_position",
    ],
    "ablation_no_roi_text_layout_position": [
        "no_roi_circling_yes_text_layout_position",
        "no_roi_text_layout_position",
    ],
}

RUN_DIR_RE = re.compile(r"^run_(\d+)_(.+)$")


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _frame_token_from_name(name: str) -> str:
    m = re.match(r"^(\d+)", name)
    if m:
        return m.group(1)
    return Path(name).stem


def file_signature(file_name_value, mode: str) -> str:
    if isinstance(file_name_value, list):
        names = [Path(str(x)).name for x in file_name_value]
        if mode == "frame":
            names = [_frame_token_from_name(n) for n in names]
        return "|".join(names)
    if file_name_value is None:
        return "__MISSING__"
    name = Path(str(file_name_value)).name
    if mode == "frame":
        return _frame_token_from_name(name)
    return name


def detect_kind(path: Path) -> str:
    name = path.name
    if name.endswith("_karo_10K.json"):
        return "test_karo_10k"
    if name.endswith("_10K.json") and name.startswith("test_run_"):
        return "test_10k"
    if name.startswith("val_answer_") and name.endswith(".json"):
        return "val_answer"
    if name.endswith("_config_used.json"):
        return "config"
    if name.startswith("test_run_") and name.endswith(".json"):
        return "test_full"
    return "other"


@dataclass
class CompareResult:
    ref_total: int
    other_total: int
    common_idx: int
    missing_in_other: int
    extra_in_other: int
    qid_match: int
    file_match: int
    qid_and_file_match: int
    question_exact: int
    question_similar: int
    sample_mismatches: List[str]


def load_json_list(path: Path) -> List[dict]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def index_by_idx(rows: Iterable[dict]) -> Dict[str, dict]:
    out = {}
    for pos, row in enumerate(rows):
        key = str(row.get("idx", pos))
        out[key] = row
    return out


def compare_records(
    ref_rows: List[dict],
    other_rows: List[dict],
    similarity_threshold: float,
    samples: int,
    file_compare_mode: str,
) -> CompareResult:
    ref = index_by_idx(ref_rows)
    other = index_by_idx(other_rows)

    ref_keys = set(ref.keys())
    other_keys = set(other.keys())
    common = sorted(ref_keys & other_keys, key=lambda x: int(x) if x.isdigit() else x)

    qid_match = 0
    file_match = 0
    both_match = 0
    question_exact = 0
    question_similar = 0
    sample_mismatches: List[str] = []

    for key in common:
        r = ref[key]
        o = other[key]

        r_qid = str(r.get("question_id", ""))
        o_qid = str(o.get("question_id", ""))
        qid_ok = r_qid == o_qid
        if qid_ok:
            qid_match += 1

        r_file = file_signature(r.get("file_name"), file_compare_mode)
        o_file = file_signature(o.get("file_name"), file_compare_mode)
        file_ok = r_file == o_file
        if file_ok:
            file_match += 1

        if qid_ok and file_ok:
            both_match += 1

        r_q = normalize_text(str(r.get("question", "")))
        o_q = normalize_text(str(o.get("question", "")))
        if r_q == o_q:
            question_exact += 1
            question_similar += 1
        else:
            score = SequenceMatcher(None, r_q, o_q).ratio()
            if score >= similarity_threshold:
                question_similar += 1

        if (not qid_ok or not file_ok) and len(sample_mismatches) < samples:
            sample_mismatches.append(
                f"idx={key} qid_equal={qid_ok} file_equal={file_ok} "
                f"ref_qid={r_qid} other_qid={o_qid} "
                f"ref_file={r_file[:80]} other_file={o_file[:80]}"
            )

    return CompareResult(
        ref_total=len(ref_rows),
        other_total=len(other_rows),
        common_idx=len(common),
        missing_in_other=len(ref_keys - other_keys),
        extra_in_other=len(other_keys - ref_keys),
        qid_match=qid_match,
        file_match=file_match,
        qid_and_file_match=both_match,
        question_exact=question_exact,
        question_similar=question_similar,
        sample_mismatches=sample_mismatches,
    )


def collect_run_dirs(output_dir: Path, variants: List[str]) -> Dict[int, Dict[str, Path]]:
    result: Dict[int, Dict[str, Path]] = defaultdict(dict)
    desired = {
        variant: VARIANT_ALIASES.get(variant, [variant]) for variant in variants
    }

    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        m = RUN_DIR_RE.match(item.name)
        if not m:
            continue
        run_num = int(m.group(1))
        suffix = m.group(2)
        for canonical, aliases in desired.items():
            if suffix in aliases:
                result[run_num][canonical] = item
    return result


def choose_file_by_kind(run_dir: Path, kind: str) -> Path:
    files = sorted(p for p in run_dir.glob("*.json") if detect_kind(p) == kind)
    if len(files) == 1:
        return files[0]
    if len(files) > 1:
        for p in files:
            if "_sanitized" not in p.name:
                return p
        return files[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare question alignment across ablation run folders using idx/question_id/"
            "file_name and question similarity."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output",
        help="Path to output folder containing run_* directories.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_ALIASES.keys()),
        help="Canonical variant names to compare.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["test_10k"],
        choices=["test_10k", "test_karo_10k", "test_full", "val_answer"],
        help="JSON file kinds to compare from each run folder.",
    )
    parser.add_argument(
        "--reference-variant",
        default="ablation_baseline",
        help="Variant used as comparison reference inside each run number.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.92,
        help="Question text similarity threshold (0-1).",
    )
    parser.add_argument(
        "--sample-mismatches",
        type=int,
        default=3,
        help="How many mismatch examples per pair to print.",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help="Only compare this run number (e.g., 26).",
    )
    parser.add_argument(
        "--file-compare-mode",
        choices=["frame", "basename"],
        default="frame",
        help=(
            "How to compare file_name values. "
            "'frame' compares leading frame numbers (000003...), "
            "'basename' requires exact basename match."
        ),
    )
    return parser.parse_args()


def print_pair_report(run_num: int, kind: str, ref_variant: str, other_variant: str, r: CompareResult):
    denom = max(r.common_idx, 1)
    print(
        f"  {other_variant} vs {ref_variant} [{kind}] "
        f"common_idx={r.common_idx} missing={r.missing_in_other} extra={r.extra_in_other}"
    )
    print(
        f"    qid_match={r.qid_match}/{denom} ({r.qid_match/denom:.1%}) "
        f"file_match={r.file_match}/{denom} ({r.file_match/denom:.1%}) "
        f"qid+file={r.qid_and_file_match}/{denom} ({r.qid_and_file_match/denom:.1%})"
    )
    print(
        f"    question_exact={r.question_exact}/{denom} ({r.question_exact/denom:.1%}) "
        f"question_similar={r.question_similar}/{denom} ({r.question_similar/denom:.1%})"
    )
    for line in r.sample_mismatches:
        print(f"    sample_mismatch: {line}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    run_dirs = collect_run_dirs(output_dir, args.variants)

    if not run_dirs:
        raise SystemExit(f"No matching run directories found under {output_dir}")

    print(f"Output dir: {output_dir}")
    print(f"Variants: {', '.join(args.variants)}")
    print(f"Kinds: {', '.join(args.kinds)}")
    print(f"Reference variant: {args.reference_variant}")
    print("")

    found_any = False
    for run_num in sorted(run_dirs):
        if args.run_number is not None and run_num != args.run_number:
            continue

        variant_dirs = run_dirs[run_num]
        if args.reference_variant not in variant_dirs:
            continue

        compared_here = False
        ref_dir = variant_dirs[args.reference_variant]
        for kind in args.kinds:
            ref_file = choose_file_by_kind(ref_dir, kind)
            if ref_file is None:
                continue

            ref_rows = load_json_list(ref_file)
            for other_variant in args.variants:
                if other_variant == args.reference_variant:
                    continue
                other_dir = variant_dirs.get(other_variant)
                if other_dir is None:
                    continue
                other_file = choose_file_by_kind(other_dir, kind)
                if other_file is None:
                    continue
                other_rows = load_json_list(other_file)
                if not compared_here:
                    print(f"Run {run_num}")
                    compared_here = True
                r = compare_records(
                    ref_rows,
                    other_rows,
                    similarity_threshold=args.similarity_threshold,
                    samples=args.sample_mismatches,
                    file_compare_mode=args.file_compare_mode,
                )
                print_pair_report(run_num, kind, args.reference_variant, other_variant, r)
                found_any = True

        if compared_here:
            print("")

    if not found_any:
        raise SystemExit("No comparable file pairs found. Check --kinds and --variants.")


if __name__ == "__main__":
    main()
