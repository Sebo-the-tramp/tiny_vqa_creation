#!/usr/bin/env python3
# viz_sim_rerun.py
# Visualize the JSON simulation in the Rerun Viewer.
#
# Usage:
#   pip install rerun-sdk numpy
#   python viz_sim_rerun.py path/to/simulation.json --spawn
#
# Viewer tips:
#   • Press Play in the timeline to animate.
#   • Toggle the 3D grid, change background, etc. from the right panel.
#   • Use the entity tree to hide/show velocity arrows or trails.
#
# Notes:
#   - Supports two per-step formats:
#       • Rigid transform: [x,y,z, rx,ry,rz] or {"p":[...], "q":[x,y,z,w]}
#       • Oriented bounding box (OBB): {"obb": {"R": 3x3, "center": [3], "extents": [3]}}
#   - If 'kinematics' exists, it draws velocity arrows (else estimates from positions).
#   - Geometry: sphere via Ellipsoids3D, box via Boxes3D, cylinder via Cylinders3D; OBB uses Boxes3D with half_sizes=extents.
#   - World coordinates assumed right-handed with +Y up, +Z forward.
#
import argparse
import json
import os
from collections import deque


import numpy as np
import rerun as rr
import open3d as o3d

from utils.load_pointclouds import load_scene_pointcloud

from scipy.spatial.transform import Rotation as R

# ---------------------------- Math helpers ----------------------------
def euler_to_quat_xyz(rx, ry, rz):
    """Return quaternion [x,y,z,w] from intrinsic XYZ Euler angles (radians)."""
    cx, sx = np.cos(0.5 * rx), np.sin(0.5 * rx)
    cy, sy = np.cos(0.5 * ry), np.sin(0.5 * ry)
    cz, sz = np.cos(0.5 * rz), np.sin(0.5 * rz)
    # q = qx * qy * qz
    qx = np.array([sx, 0.0, 0.0, cx])
    qy = np.array([0.0, sy, 0.0, cy])
    qz = np.array([0.0, 0.0, sz, cz])

    # quaternion multiply
    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ]
        )

    q = qmul(qmul(qx, qy), qz)
    # normalize
    n = np.linalg.norm(q)
    return (q / n) if n > 0 else np.array([0, 0, 0, 1])


def rotmat_to_quat(R):
    """Convert a 3x3 rotation matrix to quaternion [x,y,z,w]."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3 rotation matrix")
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        S = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=float)
    # normalize
    n = np.linalg.norm(q)
    return (q / n) if n > 0 else np.array([0, 0, 0, 1], dtype=float)


def stress_color(stress):
    """Map stress in [0,1] to a color between green->yellow->red."""
    s = float(np.clip(stress, 0.0, 1.0))
    # simple piecewise: [0,0.5]-> green to yellow, [0.5,1]-> yellow to red
    if s <= 0.5:
        t = s / 0.5
        r, g, b = int(255 * t), 255, 0
    else:
        t = (s - 0.5) / 0.5
        r, g, b = 255, int(255 * (1.0 - t)), 0
    return [r, g, b]


MATERIAL_COLORS = {
    "rubber": [240, 90, 60],  # warm
    "plastic": [70, 160, 255],  # blue-ish
    "metal": [190, 190, 190],  # gray
    "wood": [120, 90, 50],  # brown (table)
    "concrete": [110, 110, 110],  # ground
}


# ---------------------------- JSON parsing ----------------------------
def get_pose(obj):
    """Return (p[3], q[4]) for an object entry at a given step."""
    tr = obj.get("transform", None)
    if isinstance(tr, dict):
        if "p" in tr and "q" in tr:
            p, q = tr["p"], tr["q"]
            return np.array(p, float), np.array(q, float)
        if (
            "pose" in tr
            and isinstance(tr["pose"], dict)
            and "p" in tr["pose"]
            and "q" in tr["pose"]
        ):
            return np.array(tr["pose"]["p"], float), np.array(tr["pose"]["q"], float)
        if "position" in tr and "rotation_quat" in tr:
            return np.array(tr["position"], float), np.array(tr["rotation_quat"], float)
    if isinstance(tr, (list, tuple)) and len(tr) >= 6:
        x, y, z, rx, ry, rz = tr[:6]
        q = euler_to_quat_xyz(rx, ry, rz)
        return np.array([x, y, z], float), q
    # fallback: no rotation
    if isinstance(tr, (list, tuple)) and len(tr) >= 3:
        return np.array(tr[:3], float), np.array([0, 0, 0, 1], float)
    raise ValueError(
        "Could not parse transform: expected [x,y,z,rx,ry,rz] or dict with p/q."
    )


def get_obb_pose_and_dims(ent):
    """If entity has an OBB, return (p[3], q[4], half_sizes[3]); else None.

    - OBB dict has keys: R (3x3 rotation), center (3), extents (3).
    - Extents describe full side lengths; convert to half-sizes for rr.Boxes3D.
    """
    obb = ent.get("obb")
    if not isinstance(obb, dict):
        return None
    R = np.asarray(obb.get("R"), dtype=float)
    center = np.asarray(obb.get("center"), dtype=float)
    extents = np.asarray(obb.get("extents"), dtype=float)
    if R.shape != (3, 3) or center.shape != (3,) or extents.shape != (3,):
        return None

    half_sizes = extents * 0.5

    # Simulation OBBs use Z-up with swapped Y/Z axes; align with viewer basis.
    basis_fix = np.array(
        [
            [1.0, 0.0, 0.0],  # keep X axis
            [0.0, 0.0, 1.0],  # new Y axis from old Z
            [0.0, 1.0, 0.0],  # new Z axis from old Y
        ],
        dtype=float,
    )
    R_world = R @ basis_fix
    half_sizes = half_sizes[[0, 2, 1]]

    q = rotmat_to_quat(R_world)
    return center, q, half_sizes


def get_velocity(obj, prev_p, dt):
    """Return velocity vector if available; else finite-difference from prev_p."""
    kin = obj.get("kinematics")
    if isinstance(kin, dict) and "linear_velocity_world" in kin:
        return np.array(kin["linear_velocity_world"], float)
    if prev_p is None or dt <= 0.0:
        return np.zeros(3, float)
    # finite difference
    # We'll keep the caller responsible for passing the current position as p and prev position.
    return None  # we compute outside to avoid re-parsing twice


# ---------------------------- Visualization ----------------------------
def quat_from_two_vectors(a, b):
    """Quaternion [x,y,z,w] that rotates vector a to vector b.

    If a or b is near-zero, returns identity. Handles opposite vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        # 180-degree rotation: find orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(a, axis)
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=float)
    s = np.sqrt((1.0 + c) * 2.0)
    invs = 1.0 / s
    qx, qy, qz = v * invs
    qw = 0.5 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    # normalize
    return q / np.linalg.norm(q)


def camera_quat_from_lookat(eye, target, up):
    """Quaternion aligning camera RUB axes so that -Z looks at target."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = target - eye
    f_norm = np.linalg.norm(forward)
    if f_norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    forward = forward / f_norm

    up_norm = np.linalg.norm(up)
    if up_norm < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        up = up / up_norm

    right = np.cross(forward, up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        # Fall back to an arbitrary orthogonal up vector.
        alt_up = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(forward[0]) > 0.9:
            alt_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, alt_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    right = right / r_norm

    true_up = np.cross(right, forward)
    u_norm = np.linalg.norm(true_up)
    if u_norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    true_up = true_up / u_norm

    back = -forward
    R = np.column_stack((right, true_up, back))
    return rotmat_to_quat(R)


def log_static_scene(sim, has_step_camera=False):
    # Coordinate system: right-handed, +Y up (viewer convention)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Derive ground orientation from gravity
    gravity = np.array(
        sim.get("config", {}).get("scene", {}).get("gravity", [0.0, -9.81, 0.0]),
        dtype=float,
    )
    # Ground normal points opposite gravity ("up")
    if np.linalg.norm(gravity) > 1e-8:
        ground_normal = -gravity / np.linalg.norm(gravity)
    else:
        ground_normal = np.array([0.0, 1.0, 0.0], dtype=float)  # default Y-up

    # Default box's local +Y is its normal; rotate +Y to ground_normal
    q_ground = quat_from_two_vectors([0.0, 1.0, 0.0], ground_normal)

    # Ground plane centered at origin, thin thickness along its normal
    rr.log(
        "world/ground",
        rr.Transform3D(translation=[0.0, 0.0, 0.0], quaternion=q_ground.tolist()),
        static=True,
    )
    rr.log(
        "world/ground/geom",
        rr.Boxes3D(
            half_sizes=[[10.0, 0.01, 10.0]],
            centers=[[0.0, 0.0, 0.0]],
            colors=[MATERIAL_COLORS.get("concrete", [120, 120, 120])],
        ),
        static=True,
    )

    # Camera, if provided
    cam = sim.get("camera", {})
    if not has_step_camera and isinstance(cam, dict):
        tr = cam.get("initial_transform", [])
        if isinstance(tr, (list, tuple)) and len(tr) >= 6:
            tx, ty, tz, rx, ry, rz = tr[:6]
            q = euler_to_quat_xyz(rx, ry, rz)
            rr.log(
                "world/camera",
                rr.Transform3D(translation=[tx, ty, tz], quaternion=q),
                static=True,
            )

        intr = cam.get("intrinsics", {})
        width, height = intr.get("width"), intr.get("height")
        fx = intr.get("fx", None)
        if width and height:
            # Pinhole draws a camera frustum; focal_length is enough for a visual.
            # If fx is present we use it; else fall back to a rough fov.
            if fx is not None:
                rr.log(
                    "world/camera/frustum",
                    rr.Pinhole(
                        width=int(width),
                        height=int(height),
                        focal_length=float(fx),
                        camera_xyz=rr.ViewCoordinates.RUB,
                    ),
                    static=True,
                )
            else:
                rr.log(
                    "world/camera/frustum",
                    rr.Pinhole(
                        fov_y=0.8,
                        aspect_ratio=width / height,
                        camera_xyz=rr.ViewCoordinates.RUB,
                    ),
                    static=True,
                )
        elif fx is not None:
            rr.log(
                "world/camera/frustum",
                rr.Pinhole(
                    focal_length=float(fx),
                    camera_xyz=rr.ViewCoordinates.RUB,
                ),
                static=True,
            )


def log_scene_pointcloud(sim, json_path):
    """Load and log the static scene point cloud if available."""
    scene_info = sim.get("scene")
    if not isinstance(scene_info, dict):
        return

    json_dir = os.path.dirname(os.path.abspath(json_path))

    def _extract_points(entry):
        pts = entry.get("points")
        if isinstance(pts, list) and pts and isinstance(pts[0], (list, tuple)):
            pts_arr = np.asarray(pts, dtype=float)
            cols = entry.get("colors")
            if isinstance(cols, list) and len(cols) == len(pts):
                cols_arr = np.asarray(cols, dtype=float)
            else:
                cols_arr = None
            return pts_arr, cols_arr
        return None

    points = None
    colors = None

    candidate_entries = [scene_info]
    pc_entry = scene_info.get("pointcloud")
    if isinstance(pc_entry, dict):
        candidate_entries.append(pc_entry)

    for entry in candidate_entries:
        result = _extract_points(entry)
        if result is not None:
            points, colors = result
            break

    if points is None:
        candidate_paths = []
        for entry in candidate_entries:
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("path")
            if isinstance(raw_path, str) and raw_path:
                expanded = os.path.expanduser(raw_path)
                candidate_paths.append(expanded)
                candidate_paths.append(
                    os.path.normpath(os.path.join(json_dir, expanded))
                )

        pointcloud_path = next(
            (p for p in candidate_paths if os.path.isfile(p)), None
        )
        if pointcloud_path and o3d is not None:
            try:
                pcd = o3d.io.read_point_cloud(pointcloud_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to load point cloud {pointcloud_path}: {exc}")
            else:
                if not pcd.is_empty():
                    points = np.asarray(pcd.points, dtype=float)
                    colors = (
                        np.asarray(pcd.colors, dtype=float) if pcd.has_colors() else None
                    )

    if points is None and load_scene_pointcloud is not None:
        scene_id = scene_info.get("scene")
        if isinstance(scene_id, str) and scene_id:
            try:
                pcd = load_scene_pointcloud(scene_id)['pcd']
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Warning: failed to load point cloud for scene '{scene_id}': {exc}"
                )
            else:
                if not pcd.is_empty():
                    points = np.asarray(pcd.points, dtype=float)
                    colors = (
                        np.asarray(pcd.colors, dtype=float) if pcd.has_colors() else None
                    )

    if points is None or points.size == 0:
        return

    if colors is not None and len(colors) == len(points):
        if colors.dtype != np.uint8:
            if colors.max() <= 1.0:
                colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            else:
                colors = np.clip(colors, 0, 255).astype(np.uint8)
        else:
            colors = colors.copy()
    else:
        colors = None

    rr.log(
        "world/scene/pointcloud",
        rr.Points3D(points, colors=colors, radii=0.01),
        static=True,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Visualize simulation JSON in Rerun Viewer."
    )
    ap.add_argument("input", help="Path to simulation JSON")
    ap.add_argument(
        "--spawn", action="store_true", help="Spawn the Rerun Viewer window"
    )
    ap.add_argument(
        "--timeline",
        default="sim_time",
        help="Timeline name to use (default: sim_time)",
    )
    ap.add_argument(
        "--trail", action="store_true", help="Draw motion trails (line strips)"
    )
    ap.add_argument(
        "--arrows",
        action="store_true",
        help="Draw velocity arrows if kinematics present (or estimate)",
    )
    ap.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Step stride for logging (default: 1 means every step)",
    )
    args = ap.parse_args()

    rr.init("simulation_view")
    # rr.send_blueprint(rrb.Spatial3DView())

    server_uri = rr.serve_grpc()

    # Connect the web viewer to the gRPC server and open it in the browser
    rr.serve_web_viewer(connect_to=server_uri)

    json_path = os.path.abspath(args.input)
    with open(json_path, "r") as f:
        sim = json.load(f)

    steps = sim.get("simulation", {})
    has_step_camera = any(
        isinstance(frame, dict) and "camera" in frame for frame in steps.values()
    )

    log_static_scene(sim, has_step_camera=has_step_camera)
    log_scene_pointcloud(sim, json_path=json_path)

    obj_meta = sim.get("objects", {})  # may be empty or missing shape info
    time_keys = sorted(steps.keys(), key=lambda s: float(s))

    # Gravity vector for force visualization and ground alignment
    gravity = np.array(
        sim.get("config", {}).get("scene", {}).get("gravity", [0.0, -9.81, 0.0]),
        dtype=float,
    )

    # Mass map per object id, default 1.0 if missing
    mass_map = {}
    for oid, meta in obj_meta.items():
        m = meta.get("mass")
        try:
            mass_map[oid] = float(m) if m is not None else 1.0
        except Exception:
            mass_map[oid] = 1.0

    # Determine per-object base color (fallback palette when no material)
    palette = [
        [230, 57, 70],  # red-ish
        [29, 161, 242],  # blue
        [87, 187, 138],  # green
        [255, 196, 0],  # yellow
        [149, 76, 233],  # purple
        [255, 99, 72],  # orange
        [16, 172, 132],  # teal
        [72, 149, 239],  # light blue
        [238, 82, 83],  # coral
        [52, 172, 224],  # cyan
    ]

    def default_color_for_oid(oid: str):
        try:
            idx = int(oid) % len(palette)
        except Exception:
            idx = sum(ord(c) for c in oid) % len(palette)
        return palette[idx]

    # Precompute static shape info only if metadata contains shape/scale.
    shape_info = {}
    for oid, meta in obj_meta.items():
        color = MATERIAL_COLORS.get(
            meta.get("material", ""), default_color_for_oid(oid)
        )
        shape = str(meta.get("shape", "")).lower()
        scale = meta.get("scale")
        if shape == "sphere" and scale is not None:
            r = float(scale[0])
            shape_info[oid] = ("sphere", [r, r, r], color)
        elif shape == "cube" and scale is not None:
            half = [float(scale[0]) / 2.0, float(scale[1]) / 2.0, float(scale[2]) / 2.0]
            shape_info[oid] = ("box", half, color)
        elif shape == "cylinder" and scale is not None:
            radius = float(scale[0])
            height = float(scale[2])
            shape_info[oid] = ("cylinder", [radius, height], color)
        else:
            # Unknown shape in metadata: defer to per-step info (e.g., OBB) or box fallback.
            shape_info[oid] = ("unknown", None, color)

    trails = {}
    prev_positions = {}
    prev_times = {}
    prev_velocities = {}
    pos_hist = {}  # oid -> deque of last 3 positions
    time_hist = {}  # oid -> deque of last 3 times

    for i, tk in enumerate(time_keys):
        if (i % args.skip) != 0:
            # still advance time so timeline plays evenly
            rr.set_time_seconds(args.timeline, float(tk))
            continue

        t = float(tk)
        rr.set_time_seconds(args.timeline, t)

        frame = steps[tk]
        objects = frame.get("objects", {})

        if has_step_camera:
            cam = frame.get("camera", {})
            eye = cam.get("eye")
            at = cam.get("at")
            up = cam.get("up", [0.0, 1.0, 0.0])
            if (
                isinstance(cam, dict)
                and isinstance(eye, (list, tuple))
                and isinstance(at, (list, tuple))
                and isinstance(up, (list, tuple))
                and len(eye) == 3
                and len(at) == 3
                and len(up) == 3
            ):
                eye_arr = np.asarray(eye, dtype=float)
                at_arr = np.asarray(at, dtype=float)
                up_arr = np.asarray(up, dtype=float)
                q_cam = camera_quat_from_lookat(eye_arr, at_arr, up_arr).tolist()
                rr.log(
                    "world/camera",
                    rr.Transform3D(
                        translation=eye_arr.tolist(),
                        quaternion=q_cam,
                    ),
                )

                pinhole_kwargs = {"camera_xyz": rr.ViewCoordinates.RUB}
                width = cam.get("width")
                height = cam.get("height")
                if width is not None and height is not None:
                    try:
                        w = int(width)
                        h = int(height)
                        if w > 0 and h > 0:
                            pinhole_kwargs["width"] = w
                            pinhole_kwargs["height"] = h
                            pinhole_kwargs["aspect_ratio"] = float(w) / float(h)
                    except Exception:
                        pass
                fov = cam.get("fov")
                if fov is not None:
                    try:
                        pinhole_kwargs["fov_y"] = float(fov)
                    except Exception:
                        pass
                if len(pinhole_kwargs) > 1:
                    rr.log(
                        "world/camera/frustum",
                        rr.Pinhole(**pinhole_kwargs),
                    )

        # Collisions (draw a pulse point at each contact)
        for oid, ent in objects.items():
            for hit in ent.get("collide", []):
                if isinstance(hit, dict) and hit:
                    info = list(hit.values())[0]
                    pos = info.get("pos", None)
                    if pos:
                        rr.log(
                            "world/collisions",
                            rr.Points3D([pos], radii=0.02, colors=[[255, 80, 80]]),
                        )
        # Also support step-level collisions as list of positions if present
        step_collisions = frame.get("collisions", [])
        if isinstance(step_collisions, list):
            for c in step_collisions:
                if isinstance(c, dict):
                    pos = c.get("pos") or c.get("position")
                    if pos is not None:
                        rr.log(
                            "world/collisions",
                            rr.Points3D([pos], radii=0.02, colors=[[255, 80, 80]]),
                        )

        # Objects
        for oid, ent in objects.items():
            # Ensure dynamic containers exist
            if oid not in trails:
                trails[oid] = []
            if oid not in prev_positions:
                prev_positions[oid] = None
            if oid not in prev_times:
                prev_times[oid] = None
            if oid not in prev_velocities:
                prev_velocities[oid] = None
            if oid not in pos_hist:
                pos_hist[oid] = deque(maxlen=3)
            if oid not in time_hist:
                time_hist[oid] = deque(maxlen=3)

            obb_info = get_obb_pose_and_dims(ent)
            if obb_info is not None:
                p, q, half_sizes = obb_info
                kind = "box"
                dims = half_sizes.tolist()
            else:
                p, q = get_pose(ent)
                # Fallback dims from metadata or default small box
                kind, dims, _ = shape_info.get(oid, ("box", [0.05, 0.05, 0.05], None))

            trails[oid].append(p.tolist())

            # Transform: attach geometry to object frame
            rr.log(
                f"world/objects/{oid}",
                rr.Transform3D(translation=p.tolist(), quaternion=q.tolist()),
            )

            # Color by material, modulated by stress if present
            base_color = shape_info.get(oid, (None, None, None))[2]
            if base_color is None:
                # Try metadata if exists; else default palette
                meta = obj_meta.get(oid, {})
                base_color = MATERIAL_COLORS.get(
                    meta.get("material", ""), default_color_for_oid(oid)
                )
            stress = ent.get("stress", None)
            if stress is not None:
                # blend material color toward stress heat color
                heat = np.array(stress_color(stress), float)
                base = np.array(base_color, float)
                col = (0.5 * base + 0.5 * heat).astype(np.uint8).tolist()
            else:
                col = base_color

            if kind == "sphere":
                # ellipsoid centered at local origin
                rr.log(
                    f"world/objects/{oid}/geom",
                    rr.Ellipsoids3D(
                        centers=[[0, 0, 0]], half_sizes=[dims], colors=[col]
                    ),
                )
            elif kind == "box":
                rr.log(
                    f"world/objects/{oid}/geom",
                    rr.Boxes3D(centers=[[0, 0, 0]], half_sizes=[dims], colors=[col]),
                )
            elif kind == "cylinder":
                radius, height = dims
                rr.log(
                    f"world/objects/{oid}/geom",
                    rr.Cylinders3D(
                        centers=[[0, 0, 0]],
                        lengths=[height],
                        radii=[radius],
                        colors=[col],
                    ),
                )
            else:
                rr.log(
                    f"world/objects/{oid}/geom",
                    rr.Boxes3D(
                        centers=[[0, 0, 0]],
                        half_sizes=[[0.05, 0.05, 0.05]],
                        colors=[col],
                    ),
                )

            # Velocity arrows in world space
            if args.arrows:
                kin = ent.get("kinematics", {})
                # Velocity: prefer provided value; else finite difference using per-object prev time
                if "linear_velocity_world" in kin:
                    v = np.array(kin["linear_velocity_world"], float)
                else:
                    v = np.zeros(3, float)
                    prev_p = prev_positions.get(oid)
                    pt_prev = prev_times.get(oid)
                    if prev_p is not None and pt_prev is not None:
                        dt = t - pt_prev
                        if dt > 0:
                            v = (p - prev_p) / dt

                # Velocity arrow (reddish)
                vel_scale = 1.0
                rr.log(
                    f"world/velocity/{oid}",
                    rr.Arrows3D(
                        vectors=[(v * vel_scale).tolist()],
                        origins=[p.tolist()],
                        radii=0.01,
                        colors=[[255, 120, 120]],
                    ),
                )

                # Acceleration: central difference with one-frame delay
                # Maintain sliding window of last 3 samples
                time_hist[oid].append(t)
                pos_hist[oid].append(p.copy())
                if len(time_hist[oid]) == 3:
                    t0, t1, t2 = time_hist[oid][0], time_hist[oid][1], time_hist[oid][2]
                    p0, p1, p2 = pos_hist[oid][0], pos_hist[oid][1], pos_hist[oid][2]
                    dtb = t1 - t0
                    dtf = t2 - t1
                    a = np.zeros(3, float)
                    if dtb > 0 and dtf > 0:
                        v_b = (p1 - p0) / dtb
                        v_f = (p2 - p1) / dtf
                        denom = t2 - t0
                        if denom > 0:
                            a = 2.0 * (v_f - v_b) / denom
                    # Log at the middle timestamp t1, with origin at p1
                    # Temporarily switch timeline to t1 to avoid visual lag
                    rr.set_time_seconds(args.timeline, float(t1))
                    acc_scale = 0.1
                    rr.log(
                        f"world/acceleration/{oid}",
                        rr.Arrows3D(
                            vectors=[(a * acc_scale).tolist()],
                            origins=[p1.tolist()],
                            radii=0.01,
                            colors=[[80, 220, 80]],
                        ),
                    )
                    # Switch back to current time t
                    rr.set_time_seconds(args.timeline, t)

                # Update history for velocity backward-diff reference
                prev_velocities[oid] = v.copy()
                prev_positions[oid] = p.copy()
                prev_times[oid] = t

            # Force arrows (gravity by default; use provided force fields if present)
            # Try common keys in case the JSON contains explicit forces
            f_vec = None
            for key in ("force_world", "force", "net_force_world", "net_force"):
                if key in ent:
                    try:
                        f_vec = np.array(ent[key], dtype=float)
                        break
                    except Exception:
                        pass
            if f_vec is None:
                # fallback to gravity force
                mass = mass_map.get(oid, 1.0)
                f_vec = mass * gravity

            # Visual scale for readability (tune as needed)
            force_scale = 0.05
            rr.log(
                f"world/forces/{oid}",
                rr.Arrows3D(
                    vectors=[(f_vec * force_scale).tolist()],
                    origins=[p.tolist()],
                    radii=0.012,
                    colors=[[70, 120, 255]],
                ),
            )

        # Trails
        if args.trail:
            for oid, pts in trails.items():
                if len(pts) >= 2:
                    rr.log(f"world/trails/{oid}", rr.LineStrips3D([pts], radii=0.003))

    # Done
    print("Done streaming to Rerun. Use the Viewer timeline to play the animation.")


if __name__ == "__main__":
    main()

    # Keep server running. If we cancel it too early, data may never arrive in the browser.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server…")


# python answering_questions/viz_sim_rerun.py /data0/sebastian.cavada/datasets/simulations_test/3/c-1_no-3_d-3_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-2_20251031_185938/simulation_kinematics.json --spawn

# python answering_questions/viz_sim_rerun.py /data0/sebastian.cavada/datasets/simulations_v3/dl3dv/random-cam-stationary/5/c-1_no-5_d-4_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-14_20251102_195058/simulation_kinematics.json --spawn
