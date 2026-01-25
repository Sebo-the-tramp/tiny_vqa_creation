#!/usr/bin/env python3
"""Compute the maximum sample size that can satisfy a target distribution."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from decimal import Decimal, ROUND_FLOOR
from fractions import Fraction
from pathlib import Path
from typing import Any, DefaultDict, Dict, Mapping, Sequence, Tuple

from subsample_questions_balanced import load_questions, make_balance_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report the maximum number of questions that can be sampled while "
            "respecting a target distribution."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("test.json"),
        help="Path to the source JSON file (default: test.json).",
    )
    parser.add_argument(
        "--percentage-map",
        type=str,
        required=True,
        metavar="JSON",
        help=(
            "Mapping from balance keys to percentages. Provide either an inline JSON string "
            '(e.g., \'{"distance": 0.4, "occlusion": 0.6}\') or a path to a JSON file.'
        ),
    )
    parser.add_argument(
        "--default-percentage",
        type=float,
        default=None,
        help=(
            "Fallback percentage (0-1 or 0-100) applied to balance groups missing from the map. "
            "If omitted, encountering an unmapped group raises an error."
        ),
    )
    parser.add_argument(
        "--balance-on",
        nargs="+",
        default=["sub_category"],
        metavar="FIELD",
        help=(
            "Fields used to define balance groups (default: sub_category). Use '-' to skip grouping "
            "and apply a single percentage to the entire dataset."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Filter questions by this mode before computing totals (e.g., 'general').",
    )
    parser.add_argument(
        "--integer-counts",
        action="store_true",
        help=(
            "Restrict the maximum total to a value that yields integer per-group counts "
            "for the provided percentages."
        ),
    )
    parser.add_argument(
        "--show-breakdown",
        action="store_true",
        help="Print per-group availability and limiting totals.",
    )
    return parser.parse_args()


def _load_percentage_map(raw: str) -> Tuple[Dict[str, Decimal], Dict[str, Fraction]]:
    path = Path(raw)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = raw

    try:
        payload = json.loads(text, parse_float=Decimal, parse_int=Decimal)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "--percentage-map must be valid JSON (either inline or path)."
        ) from exc

    if not isinstance(payload, Mapping):
        raise SystemExit(
            f"Percentage map must be a JSON object, received {type(payload).__name__}."
        )

    decimals: Dict[str, Decimal] = {}
    fractions: Dict[str, Fraction] = {}
    for key, value in payload.items():
        if not isinstance(value, (int, float, Decimal)):
            raise SystemExit(
                f"Percentage for '{key}' must be numeric, received {type(value).__name__}."
            )
        pct = _normalise_percentage(Decimal(str(value)), context=f"map entry '{key}'")
        decimals[str(key)] = pct
        fractions[str(key)] = _decimal_to_fraction(pct)

    if not decimals:
        raise SystemExit("Percentage map is empty; nothing to compute.")

    return decimals, fractions


def _normalise_percentage(value: Decimal, context: str = "value") -> Decimal:
    if value < 0:
        raise SystemExit(f"Percentage for {context} must be non-negative.")
    if value <= 1:
        return value
    if value <= 100:
        return value / Decimal("100")
    raise SystemExit(f"Percentage for {context} exceeds 100%: {value}.")


def _decimal_to_fraction(value: Decimal) -> Fraction:
    if value.is_zero():
        return Fraction(0, 1)
    value = value.normalize()
    sign, digits, exponent = value.as_tuple()
    if exponent >= 0:
        numerator = int(value)
        return Fraction(numerator, 1)
    scale = 10 ** (-exponent)
    numerator = int(value * Decimal(scale))
    if sign:
        numerator = -numerator
    return Fraction(numerator, scale)


def _format_group_key(key: Tuple[Any, ...]) -> str:
    if not key:
        return "<all>"
    if len(key) == 1:
        return str(key[0])
    return "|".join(str(part) for part in key)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else abs(a or b)


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)
    if args.mode is not None:
        questions = [record for record in questions if record.get("mode") == args.mode]

    grouped = make_balance_groups(questions, args.balance_on)
    available: Dict[str, int] = {
        _format_group_key(key): len(records) for key, records in grouped.items()
    }

    pct_map, frac_map = _load_percentage_map(args.percentage_map)
    default_pct = (
        None
        if args.default_percentage is None
        else _normalise_percentage(
            Decimal(str(args.default_percentage)), context="--default-percentage"
        )
    )

    if not available:
        raise SystemExit("No questions available after filtering; nothing to compute.")

    missing_required = [key for key in pct_map if key not in available]
    if missing_required:
        print(
            "Warning: percentage map includes keys with no available questions: "
            + ", ".join(sorted(missing_required))
        )

    unmapped = [key for key in available if key not in pct_map]
    if unmapped and default_pct is None:
        raise SystemExit(
            "Found balance groups missing from the percentage map and no default was provided: "
            + ", ".join(sorted(unmapped))
        )

    limits: Dict[str, Decimal] = {}
    for label in sorted(set(available) | set(pct_map)):
        count = available.get(label, 0)
        pct = pct_map.get(label, default_pct)
        if pct is None:
            raise SystemExit(f"No percentage specified for balance group '{label}'.")
        if pct == 0:
            continue
        limits[label] = Decimal(count) / pct

    if not limits:
        raise SystemExit("All provided percentages are zero; no feasible total.")

    max_total = min(limits.values())
    max_total_floor = int(max_total.to_integral_value(rounding=ROUND_FLOOR))

    total_available = sum(available.values())
    print(f"Total available questions: {total_available}")
    print(f"Maximum feasible total (continuous): {max_total:.6f}")
    print(f"Maximum feasible total (floor): {max_total_floor}")

    if args.integer_counts:
        denominators = [frac.denominator for frac in frac_map.values() if frac > 0]
        if default_pct is not None:
            denominators.append(_decimal_to_fraction(default_pct).denominator)
        base_multiple = 1
        for denom in denominators:
            base_multiple = _lcm(base_multiple, denom)
        if base_multiple <= 0:
            raise SystemExit("Unable to compute integer-count multiple from percentages.")
        max_total_int = (max_total_floor // base_multiple) * base_multiple
        print(
            "Maximum feasible total (integer counts): "
            f"{max_total_int} (multiple of {base_multiple})"
        )

    if args.show_breakdown:
        print("Per-group availability and limiting totals:")
        for label in sorted(set(available) | set(pct_map)):
            count = available.get(label, 0)
            pct = pct_map.get(label, default_pct)
            if pct is None:
                pct_str = "n/a"
                limit = "n/a"
            else:
                pct_str = f"{pct:.6f}"
                if pct == 0:
                    limit = "n/a"
                else:
                    limit = f"{(Decimal(count) / pct):.6f}"
            print(f"  {label:28s} available={count:6d} pct={pct_str:>8s} limit={limit}")


if __name__ == "__main__":
    main()
