from __future__ import annotations
from typing import List, Tuple, Optional, Union, Iterable, Dict
import math
import random
import re


# ---------- helpers ----------
_ws = re.compile(r"\s+")


def _round_sig(x: float, sig: int = 3) -> float:
    sig = max(1, sig)
    if x == 0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def _decimals_for_sig(x: float, sig: int = 3) -> int:
    if x == 0:
        return max(0, sig - 1)
    return max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))


def norm(name: str) -> str:
    s = _ws.sub(" ", name.strip().lower())
    # ultra-light singularization (good enough for most dataset labels)
    if s.endswith("ies") and len(s) > 3:
        s = s[:-3] + "y"
    elif s.endswith("sses") or s.endswith("shes") or s.endswith("ches"):
        pass
    elif s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def title_label(name: str) -> str:
    specials = {"tv": "TV", "tshirt": "T-shirt"}
    n = norm(name)
    return specials.get(n, " ".join(w.capitalize() for w in n.split()))


def bigram_sim(a: str, b: str) -> float:
    def grams(s: str) -> set[str]:
        s = f" {s} "
        return {s[i : i + 2] for i in range(len(s) - 1)}

    A, B = grams(a), grams(b)
    return (len(A & B) / len(A | B)) if A and B else 0.0


# ---------- integer mode (counting) ----------
def create_mc_options_integer(
    gt: int,
    num_answers: int = 4,
    lo: int = 0,
    hi: int = 9,
    *,
    seed: Optional[int] = None,
) -> Tuple[List[int], int]:
    if lo > hi:
        raise ValueError("lo cannot be > hi")
    if not (lo <= gt <= hi):
        raise ValueError(f"gt={gt} is outside [{lo}, {hi}]")
    domain_size = hi - lo + 1
    if domain_size < num_answers:
        raise ValueError(
            f"Need {num_answers} unique ints but only {domain_size} available."
        )

    rng = random.Random(seed)
    needed = num_answers - 1

    neighbors: List[int] = []
    d = 1
    while len(neighbors) < domain_size - 1:
        for sgn in (-1, 1):
            v = gt + sgn * d
            if lo <= v <= hi and v != gt and v not in neighbors:
                neighbors.append(v)
        d += 1
        if gt - d < lo and gt + d > hi:
            break

    band = neighbors[: max(needed + 2, min(5, len(neighbors)))]
    rng.shuffle(band)

    distractors = band[:needed]
    if len(distractors) < needed:
        remaining = [v for v in neighbors if v not in distractors]
        rng.shuffle(remaining)
        distractors += remaining[: needed - len(distractors)]

    options = [gt] + distractors[:needed]
    rng.shuffle(options)
    correct_idx = options.index(gt)
    return options, correct_idx


# ---------- float mode (continuous, now display-aware) ----------
def create_mc_options_around_gt(
    gt: float,
    num_answers: int = 4,
    *,
    seed: Optional[int] = None,
    sig_digits: int = 3,
    lo: Optional[float] = None,  # e.g., 0.0 for speed
    hi: Optional[float] = None,  # upper bound if needed
    display_decimals: Optional[
        int
    ] = None,  # enforce distinct labels at this resolution
) -> Tuple[List[float], int]:
    """
    Create MC options around GT with a 'moderate' difficulty.

    New: if `display_decimals` is provided, we ensure that when numbers are rounded
    to that many decimals, every label is distinct (no more [0, 0, 1, 0]).
    """
    if num_answers < 2:
        raise ValueError("num_answers must be at least 2.")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("lo cannot be > hi")

    rng = random.Random(seed)

    spread_rel = 0.25
    min_rel_gap = 0.05
    attempts_limit = 6000

    rounded_gt = _round_sig(gt, sig_digits)
    min_abs_gap = (
        10 ** (math.floor(math.log10(abs(rounded_gt))) - 2) if rounded_gt != 0 else 1e-3
    )

    # Display resolution (size of one label bin)
    display_unit = (10 ** (-display_decimals)) if display_decimals is not None else None

    # If domain bounds + resolution make it impossible, fail fast with a helpful error.
    if display_unit is not None and lo is not None and hi is not None:
        # number of distinct on-screen labels possible in [lo, hi]
        max_labels = int(math.floor((hi - lo) / display_unit)) + 1
        if max_labels < num_answers:
            raise ValueError(
                f"At display_decimals={display_decimals}, interval [{lo}, {hi}] "
                f"supports only {max_labels} distinct labels; need {num_answers}."
            )

    def in_bounds(v: float) -> bool:
        if lo is not None and v < lo:
            return False
        if hi is not None and v > hi:
            return False
        return True

    def too_close_to_any(v: float, vals: List[float]) -> bool:
        # avoid hugging GT and other distractors in value space
        if abs(v - rounded_gt) <= max(
            min_abs_gap, min_rel_gap * max(abs(rounded_gt), abs(v), 1e-12)
        ):
            return True
        for o in vals:
            if abs(v - o) <= max(
                min_abs_gap, 0.5 * min_rel_gap * max(abs(o), abs(v), 1e-12)
            ):
                return True
        return False

    # Track label bins we've already used at the chosen display resolution
    def label_key(v: float) -> Optional[float]:
        return round(v, display_decimals) if display_decimals is not None else None

    used_label_keys = set()
    if display_decimals is not None:
        used_label_keys.add(label_key(rounded_gt))

    distractors: List[float] = []
    needed = num_answers - 1
    attempts = 0

    while len(distractors) < needed and attempts < attempts_limit:
        attempts += 1

        # multiplicative jitter around GT, with near-zero additive fallback
        if abs(gt) > 1e-12:
            f = rng.uniform(1.0 - spread_rel, 1.0 + spread_rel)
            if abs(f - 1.0) < 0.8 * min_rel_gap:
                continue
            candidate = gt * f
        else:
            step = 10 ** (-max(2, _decimals_for_sig(1.0, sig_digits) + 1))  # ~0.01
            candidate = gt + rng.choice([-3, -2, -1, 1, 2, 3]) * step

        rounded_cand = float(_round_sig(candidate, sig_digits))
        if not in_bounds(rounded_cand):
            continue
        if rounded_cand == rounded_gt:
            continue
        if too_close_to_any(rounded_cand, distractors):
            continue

        if display_decimals is not None:
            lk = label_key(rounded_cand)
            if lk in used_label_keys:
                continue
            used_label_keys.add(lk)

        distractors.append(rounded_cand)

    # Fallback: even-spaced steps outward; ensure we cross label bins when display_decimals is set.
    if len(distractors) < needed:
        base_step = max(min_abs_gap * 2.0, max(1e-12, abs(rounded_gt)) * 0.1)
        if display_unit is not None:
            base_step = max(
                base_step, display_unit
            )  # guarantees a new rounded label at given resolution
        k = 1
        while len(distractors) < needed and k <= 200:
            for sgn in (-1, 1):
                v = float(_round_sig(rounded_gt + sgn * k * base_step, sig_digits))
                if not in_bounds(v):
                    continue
                if v == rounded_gt or v in distractors:
                    continue
                if too_close_to_any(v, distractors):
                    continue
                if display_decimals is not None:
                    lk = label_key(v)
                    if lk in used_label_keys:
                        continue
                    used_label_keys.add(lk)
                distractors.append(v)
                if len(distractors) >= needed:
                    break
            k += 1

    options = [float(rounded_gt)] + distractors[:needed]
    rng.shuffle(options)
    correct_idx = options.index(float(rounded_gt))
    return options, correct_idx


# ---------- uniform display labels (UI layer) ----------
def uniform_labels(
    options: List[Union[int, float]],
    *,
    integer: bool = False,
    decimals: Optional[int] = None,
    sig_digits: int = 3,
) -> List[str]:
    """
    Return string labels that look uniform.
    - integer=True  -> '7', '3', ...
    - integer=False -> fixed decimals for all (e.g., '0.8', '3.0').
    If decimals is None, choose automatically based on scale (and ensure uniqueness).
    If decimals is not None, we respect it (no auto-bump).
    """
    if integer:
        return [str(int(v)) for v in options]

    if decimals is None:
        max_abs = max(abs(float(v)) for v in options) if options else 1.0
        decimals = max(0, _decimals_for_sig(max_abs, sig_digits))
        decimals = min(decimals, 3)

        fmt = "{:." + str(decimals) + "f}"
        labels = [fmt.format(float(v)) for v in options]
        # if rounding collides, bump decimals up to 6
        if len(set(labels)) < len(labels):
            for d in range(decimals + 1, 7):
                fmt = "{:." + str(d) + "f}"
                tmp = [fmt.format(float(v)) for v in options]
                if len(set(tmp)) == len(tmp):
                    return tmp
        return labels
    else:
        fmt = "{:." + str(decimals) + "f}"
        return [fmt.format(float(v)) for v in options]


# ---------- convenience wrapper ----------
def create_bins_from_single_gt(
    gt: float,
    num_answer: int = 4,
    *,
    integer: bool = False,
    lo: Optional[Union[int, float]] = None,
    hi: Optional[Union[int, float]] = None,
    seed: Optional[int] = None,
    return_labels: bool = False,
    label_decimals: Optional[int] = None,
) -> Union[List[Union[int, float]], List[str]]:
    """
    - integer=True  : integer mode (no decimals), options in [lo, hi] if provided.
    - integer=False : float mode, rounded to sig figs; if `label_decimals` is provided,
                      the generator ensures distinct on-screen labels at that resolution.
    """
    if integer:
        if lo is None:
            lo = 0
        if hi is None:
            hi = 9
        opts, _ = create_mc_options_integer(
            int(round(gt)), num_answers=num_answer, lo=int(lo), hi=int(hi), seed=seed
        )
        return uniform_labels(opts, integer=True) if return_labels else opts
    else:
        opts, _ = create_mc_options_around_gt(
            float(gt),
            num_answers=num_answer,
            seed=seed,
            display_decimals=label_decimals,
            lo=(float(lo) if lo is not None else None),
            hi=(float(hi) if hi is not None else None),
        )
        return (
            uniform_labels(opts, integer=False, decimals=label_decimals)
            if return_labels
            else opts
        )


# ---------- main helper ----------
def create_mc_object_names_from_dataset(
    gt: str,
    present_objects: Iterable[str],
    dataset_labels: Iterable[str],
    num_answers: int = 4,
    *,
    seed: Optional[int] = None,
    category_map: Optional[
        Dict[str, str]
    ] = None,  # optional: label->category (normalized keys)
    prefer_same_category: bool = True,
    min_name_sim_if_uncat: float = 0.15,  # when no category info, require some name similarity
) -> Tuple[List[str], int]:
    """
    Make multiple-choice options for object names.

    Priority:
      1) Distractors from present_objects (excluding GT), preferring same category if provided.
      2) Top up from dataset_labels (excluding GT and any already used), again preferring same category.
      3) If no category info, fall back to simple name similarity to keep distractors plausible.

    Returns:
      (labels, correct_index) with consistent Title Case labeling.
    """
    if num_answers < 2:
        raise ValueError("num_answers must be at least 2")

    rng = random.Random(seed)

    gt_n = norm(gt)
    if not gt_n:
        raise ValueError("Ground-truth object name becomes empty after normalization.")

    # Normalize datasets
    ds_norm_set = {norm(x) for x in dataset_labels if norm(x)}
    if gt_n not in ds_norm_set:
        # Not required, but nice to know; we still proceed.
        pass

    # Present objects (normalized, unique, excluding GT)
    seen_present = set()
    present_n = []
    for o in present_objects:
        n = norm(o)
        if n and n != gt_n and n not in seen_present:
            seen_present.add(n)
            present_n.append(n)

    # Category lookup helper
    def cat_of(n: str) -> Optional[str]:
        if not category_map:
            return None
        # category_map keys should be normalized
        return category_map.get(n)

    gt_cat = cat_of(gt_n)

    # Split present by category (if available)
    if prefer_same_category and gt_cat is not None:
        present_same = [x for x in present_n if cat_of(x) == gt_cat]
        present_other = [x for x in present_n if x not in present_same]
    else:
        present_same, present_other = [], present_n

    rng.shuffle(present_same)
    rng.shuffle(present_other)

    distractors: List[str] = []
    needed = num_answers - 1

    def take_from(lst: List[str], k: int):
        nonlocal distractors
        for x in lst:
            if len(distractors) >= k:
                break
            if x != gt_n and x not in distractors:
                distractors.append(x)

    # 1) Fill from present objects
    take_from(present_same, needed)
    if len(distractors) < needed:
        take_from(present_other, needed)

    # 2) Top up from dataset labels
    if len(distractors) < needed:
        # candidate pool = dataset minus GT and already used/present
        cand = [
            x
            for x in ds_norm_set
            if x != gt_n and x not in distractors and x not in seen_present
        ]

        # rank: same category first (if available), then name similarity to GT
        def rank_key(x: str):
            same_cat = (
                1
                if (prefer_same_category and gt_cat is not None and cat_of(x) == gt_cat)
                else 0
            )
            sim = bigram_sim(gt_n, x)
            return (same_cat, sim)

        cand.sort(key=rank_key, reverse=True)

        # If no category info at all, enforce a tiny similarity floor so we don't pick wild, unrelated labels
        if gt_cat is None and not category_map:
            filtered = [x for x in cand if bigram_sim(gt_n, x) >= min_name_sim_if_uncat]
            cand = filtered or cand  # if nothing passes, keep original to avoid failure

        for x in cand:
            distractors.append(x)
            if len(distractors) >= needed:
                break

    # 3) Final sanity check
    if len(distractors) < needed:
        have = 1 + len(distractors)
        raise ValueError(
            f"Not enough unique object names to make {num_answers} options. "
            f"Only {have} available (including the correct answer). "
            f"Add more dataset_labels or lower num_answers."
        )

    # Assemble + shuffle; format labels uniformly
    options_n = [gt_n] + distractors[:needed]
    rng.shuffle(options_n)
    labels = [title_label(x) for x in options_n]
    correct_idx = options_n.index(gt_n)
    return labels, correct_idx
