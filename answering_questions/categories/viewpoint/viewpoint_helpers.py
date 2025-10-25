"""
Mock visibility reasoning resolvers.

These helpers extract best-effort visibility answers from the provided world state.
They operate on lightweight metadata (positions, orientations, region tags, etc.)
and fall back to sensible defaults when information is missing.
"""

from __future__ import annotations


from typing import (
    Any,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import math
import random

Number = Union[int, float]
Vector = Tuple[float, float, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[str, float, Vector, Mapping[str, Any], Sequence[str]]

## --- viewpoint helpers functions -- ##


def _normalize(v):
    n = math.sqrt(sum(x*x for x in v))
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [x/n for x in v]

def infer_world_up(world_state, cam_up):
    """
    Pick a world up axis. Preference order:
    1) explicit world_up in world_state['metadata'] if present
    2) the canonical axis (Z, Y, X) that is closest to cam_up, with sign aligned
    """
    md = world_state.get("metadata", {}) if isinstance(world_state, dict) else {}
    if isinstance(md, dict) and "world_up" in md:
        wu = md["world_up"]  # expect [x,y,z]
        return _normalize(wu)

    # choose the unit axis with the largest absolute dot to cam_up
    candidates = [[0,0,1], [0,1,0], [1,0,0]]
    cam_up_n = _normalize(cam_up)
    best = max(candidates, key=lambda a: abs(sum(a[i]*cam_up_n[i] for i in range(3))))
    # align sign so cam_up • world_up >= 0
    sign = 1.0 if sum(best[i]*cam_up_n[i] for i in range(3)) >= 0 else -1.0
    return [sign*best[0], sign*best[1], sign*best[2]]

def forward(eye, at):
    return _normalize([at[0]-eye[0], at[1]-eye[1], at[2]-eye[2]])

def pitch_deg(forward_vec, world_up):
    """
    Pitch is the angle above (+) or below (-) the horizontal plane.
    Using sin(pitch) = forward • world_up. Range [-90, +90] degrees.
    """
    dot_fu = sum(forward_vec[i]*world_up[i] for i in range(3))
    dot_fu = max(-1.0, min(1.0, dot_fu))
    return math.degrees(math.asin(dot_fu))

def _ensure_radians(fov_value):
    # If someone passes degrees by mistake (e.g., > pi), convert to radians.
    return fov_value if fov_value <= math.pi else math.radians(fov_value)

def horizontal_fov_rad(fov_value, width, height, fov_axis="vertical"):
    """
    Compute horizontal FOV from a given FOV and aspect.
    - fov_axis: "vertical" (default), "horizontal"
    """
    fov_rad = _ensure_radians(fov_value)
    aspect = float(width) / float(height)
    if fov_axis.lower().startswith("h"):  # already a horizontal FOV
        return fov_rad
    # assume vertical FOV otherwise
    return 2.0 * math.atan(math.tan(fov_rad * 0.5) * aspect)

def classify_camera_angle_index(pitch_deg):
    """
    Thresholds (inclusive on upper bound where applicable):
      bird's-eye   <= -60
      high angle   -60 < pitch <= -15
      eye level    -15 < pitch < 15
      low angle     15 <= pitch < 60
      worm's-eye    >= 60
    Labels (fixed order):
      ["low angle","eye level","high angle","bird's-eye","worm's-eye"]
    """
    if pitch_deg <= -60.0:
        label = "bird's-eye (<=-60 degrees)"
    elif pitch_deg <= -15.0:
        label = "high angle (-60 to -15 degrees)"
    elif pitch_deg < 15.0:
        label = "eye level (-15 to 15 degrees)"
    elif pitch_deg < 60.0:
        label = "low angle (15 to 60 degrees)"
    else:
        label = "worm's-eye (>=60 degrees)"

    labels = ["low angle (15 to 60 degrees)","eye level (-15 to 15 degrees)","high angle (-60 to -15 degrees)","bird's-eye (<=-60 degrees)","worm's-eye (>=60 degrees)"]
    # Ensure the correct label is included, then add 3 random others
    other_labels = [l for l in labels if l != label]
    random.shuffle(other_labels)
    labels = [label] + other_labels[:3]
    random.shuffle(labels)
    idx = labels.index(label)
    return labels, idx

def classify_focal_length_index(hfov_deg):
    """
    Horizontal FOV thresholds:
      ultra-wide      >= 90
      wide            63 to < 90
      normal          40 to < 63
      telephoto       < 40
    Labels (fixed order):
      ["ultra-wide","wide","normal","short telephoto","telephoto"]
    """
    if hfov_deg >= 90.0:
        label = "ultra-wide (>=90)"
    elif hfov_deg >= 63.0:
        label = "wide (63-90)"
    elif hfov_deg >= 40.0:
        label = "normal (40-63)"
    else:
        label = "telephoto (<40)"

    labels = ["ultra-wide (>=90)","wide (63-90)","normal (40-63)","telephoto (<40)"]
    idx = labels.index(label)
    return labels, idx