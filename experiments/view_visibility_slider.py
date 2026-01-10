#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ANSWERING_QUESTIONS_DIR = os.path.join(REPO_ROOT, "answering_questions")
for path in (REPO_ROOT, ANSWERING_QUESTIONS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ANSWERING_QUESTIONS_DIR)

from answering_questions.categories.persistence.persistence_helpers import (  # noqa: E402
    get_visibility_mask,
)
from answering_questions.utils.helpers import (  # noqa: E402
    is_object_visible,
    is_object_visible_v3,
    get_visibility_ratio_v3,
)
from answering_questions.utils.geometry import project_obb, external_points_2d  # noqa: E402
from answering_questions.utils.all_objects import get_gso_mapping

from answering_questions.utils.geometry import project_obb

gso_mapping = get_gso_mapping()

#decorator
def add_obj_id(func):
    def wrapper(world_state: Mapping) -> Mapping:
        for obj_id, object in world_state["objects"].items():
            object["id"] = obj_id
            object["name"] = gso_mapping[object["model"]]["name"]

        return func(world_state)

    return wrapper

def read_json(path: str) -> Mapping:
    with open(path, "r") as f:
        return json.load(f)


def natural_key(s: str) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in __import__("re").split(r"(\d+)", s)]


def list_render_frames(render_dir: str) -> List[str]:
    frames = glob.glob(os.path.join(render_dir, "*.png")) + glob.glob(
        os.path.join(render_dir, "*.jpg")
    )
    frames.sort(key=natural_key)
    return frames

@add_obj_id
def build_frame_timestep_map(world_state: Mapping) -> Dict[int, str]:
    frame_to_timestep: Dict[int, str] = {}
    for timestep, step_data in world_state.get("simulation", {}).items():
        frame_idx = step_data.get("frame_idx")
        if frame_idx is None:
            continue
        frame_to_timestep[int(frame_idx)] = str(timestep)
    return frame_to_timestep


def resolve_instance_dir(instances_root: str, obj_id: str) -> Optional[str]:
    direct = os.path.join(instances_root, f"obj_{obj_id}")
    if os.path.isdir(direct):
        return direct
    return None


def visible_objects_for_timestep(world_state: Mapping, timestep: str) -> List[str]:
    objects = world_state.get("simulation", {}).get(str(timestep), {}).get("objects", {})
    visible_ids: List[str] = []
    for obj_id, obj_state in objects.items():
        if is_object_visible(obj_state):
            visible_ids.append(str(obj_id))
    return visible_ids


def visibility_percent(
    timestep_index: int,
    obj_index: int,
    visibility_percentage_mask: np.ndarray,
) -> float:
    if (
        obj_index < 0
        or obj_index >= visibility_percentage_mask.shape[0]
        or timestep_index < 0
        or timestep_index >= visibility_percentage_mask.shape[1]
    ):
        return 0.0
    return float(visibility_percentage_mask[obj_index, timestep_index])


def visibility_stats(world_state: Mapping, timestep: str, obj_id: str) -> Dict[str, float]:
    obj_state = (
        world_state.get("simulation", {})
        .get(str(timestep), {})
        .get("objects", {})
        .get(str(obj_id), {})
    )
    pixels_void = float(obj_state.get("infov_pixels_void", 0.0))
    pixels_visible = float(obj_state.get("infov_pixels_visible", 0.0))
    infov_pixels = float(obj_state.get("infov_pixels", 0.0))
    onscreen_pixels = pixels_visible + pixels_void
    true_visibility_ratio = (onscreen_pixels / infov_pixels) if infov_pixels > 0 else 0.0
    return {
        "pixels_void": pixels_void,
        "pixels_visible": pixels_visible,
        "infov_pixels": infov_pixels,
        "onscreen_pixels": onscreen_pixels,
        "true_visibility_ratio": true_visibility_ratio,
    }


def load_mask(mask_path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(mask_path).convert("L")
    return np.array(img)


def overlay_masks(
    base: np.ndarray, masks: Sequence[np.ndarray], colors: Sequence[Tuple[int, int, int]]
) -> np.ndarray:
    out = base.astype(np.float32) / 255.0
    for mask, color in zip(masks, colors):
        if mask is None:
            continue
        alpha = (mask > 0).astype(np.float32) * 0.45
        if alpha.max() == 0:
            continue
        color_arr = np.array(color, dtype=np.float32) / 255.0
        out = out * (1 - alpha[..., None]) + color_arr[None, None, :] * alpha[..., None]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def color_palette(n: int) -> List[Tuple[int, int, int]]:
    base = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    ]
    if n <= len(base):
        return base[:n]
    colors = list(base)
    rng = np.random.RandomState(1337)
    while len(colors) < n:
        colors.append(tuple(int(x) for x in rng.randint(30, 225, size=3)))
    return colors


def get_simulation_dir(simulation_path: str) -> str:
    if simulation_path.endswith("simulation.json"):
        return os.path.dirname(simulation_path)
    return simulation_path


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon_to_rect(
    poly: np.ndarray, width: int, height: int
) -> np.ndarray:
    def clip_edge(points: np.ndarray, edge_fn):
        if len(points) == 0:
            return points
        clipped = []
        for i in range(len(points)):
            curr = points[i]
            prev = points[i - 1]
            curr_in = edge_fn(curr)
            prev_in = edge_fn(prev)
            if curr_in:
                if not prev_in:
                    clipped.append(intersect(prev, curr, edge_fn))
                clipped.append(curr)
            elif prev_in:
                clipped.append(intersect(prev, curr, edge_fn))
        return np.array(clipped, dtype=float)

    def intersect(p1: np.ndarray, p2: np.ndarray, edge_fn):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if edge_fn == left:
            t = (0 - x1) / dx if dx != 0 else 0
        elif edge_fn == right:
            t = ((width - 1) - x1) / dx if dx != 0 else 0
        elif edge_fn == top:
            t = (0 - y1) / dy if dy != 0 else 0
        else:  # bottom
            t = ((height - 1) - y1) / dy if dy != 0 else 0
        return np.array([x1 + t * dx, y1 + t * dy], dtype=float)

    def left(p):  # x >= 0
        return p[0] >= 0

    def right(p):  # x <= width-1
        return p[0] <= width - 1

    def top(p):  # y >= 0
        return p[1] >= 0

    def bottom(p):  # y <= height-1
        return p[1] <= height - 1

    clipped = poly.astype(float)
    for edge in (left, right, top, bottom):
        clipped = clip_edge(clipped, edge)
        if len(clipped) == 0:
            break
    return clipped


def obb_inside_ratio(
    obb: Mapping, cam: Mapping, width: int, height: int
) -> Tuple[float, Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
    uv, z = project_obb(obb, cam)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z)
    if not np.any(valid):
        return 0.0, None, None
    uv = uv[valid]
    hull = external_points_2d(uv)
    total_area = polygon_area(hull)
    if total_area <= 0:
        return 0.0, None, None
    clipped = clip_polygon_to_rect(hull, width, height)
    inside_area = polygon_area(clipped)
    inside_ratio = max(0.0, min(1.0, inside_area / total_area))

    min_x, min_y = np.min(uv, axis=0)
    max_x, max_y = np.max(uv, axis=0)
    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    return inside_ratio, hull, bbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation_dir",
        type=str,
        required=True,
        help="Path to a simulation folder (contains simulation.json).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial frame index for the slider.",
    )
    args = parser.parse_args()

    simulation_dir = get_simulation_dir(args.simulation_dir)
    kinematics_path = os.path.join(simulation_dir, "simulation_kinematics.json")
    render_dir = os.path.join(simulation_dir, "render")
    instances_root = os.path.join(simulation_dir, "instances")

    if not os.path.isfile(kinematics_path):
        raise FileNotFoundError(f"Missing {kinematics_path}")
    if not os.path.isdir(render_dir):
        raise FileNotFoundError(f"Missing {render_dir}")
    if not os.path.isdir(instances_root):
        raise FileNotFoundError(f"Missing {instances_root}")

    world_state = read_json(kinematics_path)
    frame_to_timestep = build_frame_timestep_map(world_state)
    all_timesteps = list(world_state.get("simulation", {}).keys())
    timestep_to_index = {timestep: idx for idx, timestep in enumerate(all_timesteps)}
    visibility_mask, visibility_percentage_mask, = get_visibility_mask(world_state)

    frame_paths = list_render_frames(render_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No render frames in {render_dir}")

    object_ids = sorted(
        [str(obj_id) for obj_id in world_state.get("objects", {}).keys()],
        key=natural_key,
    )
    obj_dirs = {obj_id: resolve_instance_dir(instances_root, obj_id) for obj_id in object_ids}
    colors = color_palette(len(object_ids))
    obj_color = {obj_id: colors[i] for i, obj_id in enumerate(object_ids)}

    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    def render_frame(frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= len(frame_paths):
            frame_idx = 0
        frame_path = frame_paths[frame_idx]
        frame_name = os.path.basename(frame_path)
        base = np.array(Image.open(frame_path).convert("RGB"))

        timestep = frame_to_timestep.get(frame_idx)
        if timestep is None:
            return base

        masks: List[np.ndarray] = []
        mask_colors: List[Tuple[int, int, int]] = []
        for obj_id in object_ids:
            inst_dir = obj_dirs.get(obj_id)
            if not inst_dir:
                continue
            mask_path = os.path.join(inst_dir, frame_name)
            if not os.path.isfile(mask_path):
                continue
            masks.append(load_mask(mask_path))
            mask_colors.append(obj_color.get(obj_id, (255, 255, 255)))

        return overlay_masks(base, masks, mask_colors) if masks else base

    initial_idx = max(0, min(args.start_frame, len(frame_paths) - 1))
    fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.35, wspace=0.05)
    left_img = ax_left.imshow(np.array(Image.open(frame_paths[initial_idx]).convert("RGB")))
    inst_path = os.path.join(instances_root, os.path.basename(frame_paths[initial_idx]))
    if os.path.isfile(inst_path):
        mid_img = ax_mid.imshow(np.array(Image.open(inst_path).convert("RGB")))
    else:
        mid_img = ax_mid.imshow(np.zeros_like(left_img.get_array()))
    right_img = ax_right.imshow(render_frame(initial_idx))
    ax_left.set_axis_off()
    ax_mid.set_axis_off()
    ax_right.set_axis_off()
    ax_left.set_title("RGB")
    ax_mid.set_title("Instances")
    ax_right.set_title("RGB + visible masks")
    legend = None
    patches: List[object] = []

    ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="frame_idx",
        valmin=0,
        valmax=len(frame_paths) - 1,
        valinit=initial_idx,
        valstep=1,
    )

    def update(val: float) -> None:
        nonlocal legend, patches
        idx = int(val)
        frame_name = os.path.basename(frame_paths[idx])
        left_img.set_data(np.array(Image.open(frame_paths[idx]).convert("RGB")))
        inst_path = os.path.join(instances_root, frame_name)
        if os.path.isfile(inst_path):
            mid_img.set_data(np.array(Image.open(inst_path).convert("RGB")))
        else:
            mid_img.set_data(np.zeros_like(left_img.get_array()))
        right_img.set_data(render_frame(idx))
        ax_left.set_title(f"RGB (frame {idx})")
        ax_mid.set_title(f"Instances (frame {idx})")
        ax_right.set_title("RGB + visible masks")
        for patch in patches:
            patch.remove()
        patches = []
        timestep = frame_to_timestep.get(idx)
        if timestep is None:
            if legend is not None:
                legend.remove()
                legend = None
        else:
            handles = []
            for obj_id in object_ids:
                t_idx = timestep_to_index.get(timestep, -1)
                obj_idx = int(obj_id) - 1 if obj_id.isdigit() else -1
                percent = visibility_percent(t_idx, obj_idx, visibility_percentage_mask)
                is_visible = 1 if is_object_visible_v3(world_state, obj_id, timestep) else 0
                obj_state = (
                    world_state.get("simulation", {})
                    .get(str(timestep), {})
                    .get("objects", {})
                    .get(str(obj_id), {})
                )
                cam = world_state.get("simulation", {}).get(str(timestep), {}).get("camera", {})
                inside_ratio = 0.0
                hull = None
                bbox = None
                if obj_state and cam and "obb" in obj_state:
                    inside_ratio, hull, bbox = obb_inside_ratio(
                        obj_state["obb"],
                        cam,
                        width=right_img.get_array().shape[1],
                        height=right_img.get_array().shape[0],
                    )
                outside_ratio = 1.0 - inside_ratio
                total_visibility = (
                    get_visibility_ratio_v3(world_state, obj_id, timestep)
                    if obj_state
                    else 0.0
                )
                fov_visibility = float(obj_state.get("fov_visibility", 0.0)) if obj_state else 0.0
                pixels_visible = float(obj_state.get("infov_pixels_visible", 0.0)) if obj_state else 0.0
                color = np.array(obj_color.get(obj_id, (255, 255, 255))) / 255.0
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="s",
                        color="none",
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markersize=8,
                        label=(
                            f"obj {obj_id}: visible {is_visible} | "
                            f"fov {fov_visibility*100:.1f}% | "
                            f"obb in {inside_ratio*100:.1f}% out {outside_ratio*100:.1f}% | "
                            f"pix {pixels_visible:.0f} | total {total_visibility*100:.1f}%"
                        ),
                    )
                )
                if hull is not None:
                    from matplotlib.patches import Polygon

                    poly = Polygon(
                        hull,
                        closed=True,
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax_right.add_patch(poly)
                    patches.append(poly)
            if legend is not None:
                legend.remove()
            if handles:
                legend = ax_right.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.2),
                    framealpha=0.6,
                    fontsize=9,
                )
            else:
                legend = None
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(initial_idx)
    plt.show()


if __name__ == "__main__":
    main()
