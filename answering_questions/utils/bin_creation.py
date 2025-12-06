from __future__ import annotations
from typing import List, Tuple, Optional, Union, Iterable, Dict
import math
import random
import re

from utils.my_exception import ImpossibleToAnswer

from utils.config import get_config
from utils import seed_utils

# ---------- helpers ----------
_ws = re.compile(r"\s+")


def _select_rng(seed: Optional[int]) -> Optional[random.Random]:
    return random.Random(seed) if seed is not None else None


def _shuffle_inplace(items: List, rng: Optional[random.Random]) -> None:
    if rng is None:
        random.shuffle(items)
    else:
        rng.shuffle(items)


def _deterministic_uniform(lo: float, hi: float, material: str) -> float:
    """Return a deterministic uniform sample in [lo, hi] using the global seed."""
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        return lo
    local_seed = seed_utils.seed_from_material(material)
    return random.Random(local_seed).uniform(lo, hi)


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
    return name.lower()


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

    rng = _select_rng(seed)
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
    _shuffle_inplace(band, rng)

    distractors = band[:needed]
    if len(distractors) < needed:
        remaining = [v for v in neighbors if v not in distractors]
        _shuffle_inplace(remaining, rng)
        distractors += remaining[: needed - len(distractors)]

    options = [gt] + distractors[:needed]
    _shuffle_inplace(options, rng)
    correct_idx = options.index(gt)
    return options, correct_idx


# improved version after Raoul's feedback
# https://chatgpt.com/c/6906646f-be44-8325-a42e-98ddbf72eec8 -> to improve probably with slope bins
def create_mc_options_around_gt(
    gt: float,
    num_answers: int = 4,
    lo: Optional[float] = None,  # e.g., 0.0 for speed
    hi: Optional[float] = None,  # upper bound if needed
    display_decimals: int = 2,
    min_threshold: float = 0.04,  # minimum value for the options
) -> Tuple[List[float], int]:
    if abs(gt) < min_threshold and gt >= -min_threshold:
        # I mean the speed is 0, we should understand that
        gt = round(gt, display_decimals) if display_decimals is not None else gt

        options = [gt, 2.50, 5.00, 10.00]  # some arbitrary distractors
        random.shuffle(options)
        correct_idx = options.index(gt)
        return options, correct_idx

    slope_cfg = get_config().get("slope_bins", 4.0)
    slope_min = 1.0
    slope_max = 4.0
    if isinstance(slope_cfg, dict):
        slope_min = float(slope_cfg.get("min", slope_min))
        slope_max = float(slope_cfg.get("max", slope_max))
    elif isinstance(slope_cfg, (list, tuple)) and len(slope_cfg) == 2:
        slope_min = float(min(slope_cfg))
        slope_max = float(max(slope_cfg))
    elif isinstance(slope_cfg, (int, float)):
        slope_max = float(slope_cfg)
        if slope_max < slope_min:
            slope_min = slope_max
    else:
        slope_max = 4.0
    current_slope_bin = _deterministic_uniform(
        slope_min,
        slope_max,
        material=f"slope_bin::{gt}:{num_answers}:{lo}:{hi}:{display_decimals}",
    )

    if abs(gt) < 1.0:
        current_slope_bin = 0.4

    gt = round(gt, display_decimals) if display_decimals is not None else gt

    intervals = [i for i in range(-3, 4) if i != 0]

    options_raw = [(x + current_slope_bin) / current_slope_bin for x in intervals]

    options = [
        round(opt * gt, display_decimals)
        for opt in options_raw
        if (lo is None or round(opt * gt, display_decimals) >= lo)
        and (hi is None or round(opt * gt, display_decimals) <= hi)
    ]
    options = [opt for opt in options if opt > min_threshold]

    options = options[: num_answers - 1]
    options.append(gt)
    random.shuffle(options)
    correct_idx = options.index(gt)

    return options, correct_idx

# logarithmic spacing version for young modulus
# https://www.researchgate.net/figure/Material-property-chart-plotting-Youngs-modulus-E-against-density-r-The-heavy-envelopes_fig3_311498694
def create_mc_options_around_gt_log(
    gt: float,
    num_answers: int = 4,
    lo: Optional[float] = None,  # e.g., 0.0 for speed
    hi: Optional[float] = None,  # upper bound if needed
    display_decimals: int = 2,
    min_threshold: float = 0.04,  # minimum value for the options
) -> Tuple[List[float], int]:
    f = math.log  # natural log
    finv = math.exp

    if abs(gt) < min_threshold:
        raise ImpossibleToAnswer("GT too close to zero for meaningful options.")
    if gt <= 0:
        raise ImpossibleToAnswer("GT must be positive for logarithmic spacing.")

    gt = round(gt, display_decimals) if display_decimals is not None else gt

    offsets = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

    z_gt = f(gt)
    step = math.log(10)
    options = [finv(z_gt + (k * step)) for k in offsets]

    # Filter & round
    options = [round(opt, display_decimals) for opt in options]
    # print("Options before filtering by min threshold:", options)
    options = [opt for opt in options if opt > min_threshold]
    if lo is not None:
        options = [opt for opt in options if opt >= lo]
    if hi is not None:
        options = [opt for opt in options if opt <= hi]

    options = options[: num_answers - 1]
    options.append(gt)
    random.shuffle(options)
    correct_idx = options.index(gt)

    return options, correct_idx


def create_mc_options_around_gt_poisson_ratio(
    gt: float,
    num_answers: int = 4,
    lo: Optional[float] = None,  # e.g., 0.0 for speed
    hi: Optional[float] = None,  # upper bound if needed
    display_decimals: int = 2,
    min_threshold: float = 0.04,  # minimum value for the options
) -> Tuple[List[float], int]:
    if abs(gt) < min_threshold and gt >= -min_threshold:
        # I mean the speed is 0, we should understand that
        options = [gt, 2.5, 5.0, 10.0]  # some arbitrary distractors
        random.shuffle(options)
        correct_idx = options.index(gt)
        return options, correct_idx

        # raise ImpossibleToAnswer("GT too close to zero for meaningful options.")

    gt = round(gt, display_decimals) if display_decimals is not None else gt

    intervals = [i for i in range(-3, 4) if i != 0]

    options_raw = [(x + 3) / 3 for x in intervals]

    options = [
        round(opt * gt, display_decimals)
        for opt in options_raw
        if (lo is None or round(opt * gt, display_decimals) >= lo)
        and (hi is None or round(opt * gt, display_decimals) <= hi)
    ]
    options = [opt for opt in options if opt > min_threshold]

    if len(options) < num_answers - 1:
        raise ImpossibleToAnswer("Not enough options could be generated around GT.")

    options = options[: num_answers - 1]
    options.append(gt)
    random.shuffle(options)
    correct_idx = options.index(gt)

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

    rng = _select_rng(seed)

    gt_n = norm(gt)
    if not gt_n:
        raise ValueError("Ground-truth object name becomes empty after normalization.")

    # Normalize datasets
    ds_norm_set = {
        norm(x) for x in dataset_labels if (norm(x) != "" and norm(x) != gt_n)
    }
    if gt_n in ds_norm_set:
        raise ValueError("Ground-truth object name found in dataset labels.")

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

    _shuffle_inplace(present_same, rng)
    _shuffle_inplace(present_other, rng)

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
    _shuffle_inplace(options_n, rng)
    labels = [x for x in options_n]
    correct_idx = options_n.index(gt_n)
    return labels, correct_idx
