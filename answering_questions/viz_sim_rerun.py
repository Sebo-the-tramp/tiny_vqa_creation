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
#   - Expects per-step transforms in either [x,y,z, rx,ry,rz] (radians, XYZ order)
#     or a dict {"p":[...], "q":[...]} where q is a quaternion [x,y,z,w].
#   - If 'kinematics' exists (from the augmenter), it will visualize velocity arrows.
#   - Cylinder is drawn using Cylinders3D (axis/length), sphere via Ellipsoids3D, cube via Boxes3D.
#   - World coordinates assumed right-handed with +Y up, +Z forward.
#
import argparse, json, numpy as np
import rerun as rr

# ---------------------------- Math helpers ----------------------------
def euler_to_quat_xyz(rx, ry, rz):
    """Return quaternion [x,y,z,w] from intrinsic XYZ Euler angles (radians)."""
    cx, sx = np.cos(0.5*rx), np.sin(0.5*rx)
    cy, sy = np.cos(0.5*ry), np.sin(0.5*ry)
    cz, sz = np.cos(0.5*rz), np.sin(0.5*rz)
    # q = qx * qy * qz
    qx = np.array([sx, 0.0, 0.0, cx])
    qy = np.array([0.0, sy, 0.0, cy])
    qz = np.array([0.0, 0.0, sz, cz])
    # quaternion multiply
    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz
        ])
    q = qmul(qmul(qx,qy), qz)
    # normalize
    n = np.linalg.norm(q)
    return (q / n) if n > 0 else np.array([0,0,0,1])

def stress_color(stress):
    """Map stress in [0,1] to a color between green->yellow->red."""
    s = float(np.clip(stress, 0.0, 1.0))
    # simple piecewise: [0,0.5]-> green to yellow, [0.5,1]-> yellow to red
    if s <= 0.5:
        t = s/0.5
        r, g, b = int(255*t), 255, 0
    else:
        t = (s-0.5)/0.5
        r, g, b = 255, int(255*(1.0 - t)), 0
    return [r, g, b]

MATERIAL_COLORS = {
    "rubber":  [240, 90,  60],   # warm
    "plastic": [70,  160, 255],  # blue-ish
    "metal":   [190, 190, 190],  # gray
    "wood":    [120, 90,  50],   # brown (table)
    "concrete":[110, 110, 110],  # ground
}

# ---------------------------- JSON parsing ----------------------------
def get_pose(obj):
    """Return (p[3], q[4]) for an object entry at a given step."""
    tr = obj.get("transform", None)
    if isinstance(tr, dict):
        if "p" in tr and "q" in tr:
            p, q = tr["p"], tr["q"]
            return np.array(p, float), np.array(q, float)
        if "pose" in tr and isinstance(tr["pose"], dict) and "p" in tr["pose"] and "q" in tr["pose"]:
            return np.array(tr["pose"]["p"], float), np.array(tr["pose"]["q"], float)
        if "position" in tr and "rotation_quat" in tr:
            return np.array(tr["position"], float), np.array(tr["rotation_quat"], float)
    if isinstance(tr, (list, tuple)) and len(tr) >= 6:
        x,y,z, rx,ry,rz = tr[:6]
        q = euler_to_quat_xyz(rx,ry,rz)
        return np.array([x,y,z], float), q
    # fallback: no rotation
    if isinstance(tr, (list, tuple)) and len(tr) >= 3:
        return np.array(tr[:3], float), np.array([0,0,0,1], float)
    raise ValueError("Could not parse transform: expected [x,y,z,rx,ry,rz] or dict with p/q.")

def get_velocity(obj, prev_p, dt):
    """Return velocity vector if available; else finite-difference from prev_p."""
    kin = obj.get("kinematics")
    if isinstance(kin, dict) and "linear_velocity_world" in kin:
        return np.array(kin["linear_velocity_world"], float)
    if prev_p is None or dt <= 0.0:
        return np.zeros(3, float)
    # finite difference
    # We'll keep the caller responsible for passing the current position as p and prev position.
    return (None)  # we compute outside to avoid re-parsing twice

# ---------------------------- Visualization ----------------------------
def log_static_scene(sim):
    # Coordinate system: right-handed, +Y up
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Draw a ground plane (thin box) and a table slab for context
    ground_y = 0.0
    table_h = 0.76  # default if not provided elsewhere
    rr.log(
        "world/ground",
        rr.Boxes3D(half_sizes=[[10.0, 0.01, 10.0]], centers=[[0.0, -0.01, 0.0]], colors=[MATERIAL_COLORS.get("concrete",[120,120,120])]),
        static=True,
    )
    rr.log(
        "world/pingpong_table",
        rr.Boxes3D(half_sizes=[[2.0, 0.015, 1.2]], centers=[[0.0, table_h - 0.015, 0.0]], colors=[MATERIAL_COLORS.get("wood",[120,90,50])]),
        static=True,
    )

    # Camera, if provided
    cam = sim.get("camera", {})
    tr = cam.get("initial_transform", [])
    if isinstance(tr, (list,tuple)) and len(tr) >= 6:
        tx,ty,tz, rx,ry,rz = tr[:6]
        q = euler_to_quat_xyz(rx,ry,rz)
        rr.log("world/camera", rr.Transform3D(translation=[tx,ty,tz], quaternion=q), static=True)

    intr = cam.get("intrinsics", {})
    width, height = intr.get("width"), intr.get("height")
    fx = intr.get("fx", None)
    if width and height:
        # Pinhole draws a camera frustum; focal_length is enough for a visual.
        # If fx is present we use it; else fall back to a rough fov.
        if fx is not None:
            rr.log("world/camera/frustum", rr.Pinhole(width=int(width), height=int(height), focal_length=float(fx), camera_xyz=rr.ViewCoordinates.RUB), static=True)
        else:
            rr.log("world/camera/frustum", rr.Pinhole(fov_y=0.8, aspect_ratio=width/height, camera_xyz=rr.ViewCoordinates.RUB), static=True)

def main():
    ap = argparse.ArgumentParser(description="Visualize simulation JSON in Rerun Viewer.")
    ap.add_argument("input", help="Path to simulation JSON")
    ap.add_argument("--spawn", action="store_true", help="Spawn the Rerun Viewer window")
    ap.add_argument("--timeline", default="sim_time", help="Timeline name to use (default: sim_time)")
    ap.add_argument("--trail", action="store_true", help="Draw motion trails (line strips)")
    ap.add_argument("--arrows", action="store_true", help="Draw velocity arrows if kinematics present (or estimate)")
    ap.add_argument("--skip", type=int, default=1, help="Step stride for logging (default: 1 means every step)")
    args = ap.parse_args()

    rr.init("simulation_view", spawn=args.spawn)

    with open(args.input, "r") as f:
        sim = json.load(f)

    log_static_scene(sim)

    obj_meta = sim.get("objects", {})
    steps = sim.get("simulation_steps", {})
    time_keys = sorted(steps.keys(), key=lambda s: float(s))

    # Precompute shape geometry per object
    shape_info = {}
    for oid, meta in obj_meta.items():
        shape = meta.get("shape", "sphere").lower()
        scale = meta.get("scale", [0.1,0.1,0.1])
        color = MATERIAL_COLORS.get(meta.get("material",""), [200,200,200])
        if shape == "sphere":
            r = float(scale[0])
            shape_info[oid] = ("sphere", [r,r,r], color)
        elif shape == "cube":
            half = [float(scale[0])/2.0, float(scale[1])/2.0, float(scale[2])/2.0]
            shape_info[oid] = ("box", half, color)
        elif shape == "cylinder":
            radius = float(scale[0])
            height = float(scale[2])
            shape_info[oid] = ("cylinder", [radius, height], color)
        else:
            # default to box using the scale as half-sizes
            half = [float(scale[0])/2.0, float(scale[1])/2.0, float(scale[2])/2.0]
            shape_info[oid] = ("box", half, color)

    trails = {oid: [] for oid in obj_meta.keys()}
    prev_positions = {oid: None for oid in obj_meta.keys()}

    for i, tk in enumerate(time_keys):
        if (i % args.skip) != 0:
            # still advance time so timeline plays evenly
            rr.set_time_seconds(args.timeline, float(tk))
            continue

        t = float(tk)
        rr.set_time_seconds(args.timeline, t)

        frame = steps[tk]
        objects = frame.get("objects", {})

        # Collisions (draw a pulse point at each contact)
        for oid, ent in objects.items():
            for hit in ent.get("collide", []):
                # hit is like {"object_ID_3": {"pos": [...]}} or {"pingpong_table": {"pos":[...]}}
                # Extract the first (and only) value dict
                if isinstance(hit, dict) and hit:
                    info = list(hit.values())[0]
                    pos = info.get("pos", None)
                    if pos:
                        rr.log("world/collisions", rr.Points3D([pos], radii=0.02, colors=[[255, 80, 80]]))

        # Objects
        for oid, ent in objects.items():
            p, q = get_pose(ent)
            trails[oid].append(p.tolist())

            # Transform: attach geometry to object frame
            rr.log(f"world/objects/{oid}", rr.Transform3D(translation=p.tolist(), quaternion=q.tolist()))

            # Color by material, modulated by stress if present
            base_color = shape_info[oid][2]
            stress = ent.get("stress", None)
            if stress is not None:
                # blend material color toward stress heat color
                heat = np.array(stress_color(stress), float)
                base = np.array(base_color, float)
                col = (0.5*base + 0.5*heat).astype(np.uint8).tolist()
            else:
                col = base_color

            kind, dims, _ = shape_info[oid]
            if kind == "sphere":
                # ellipsoid centered at local origin
                rr.log(f"world/objects/{oid}/geom", rr.Ellipsoids3D(centers=[[0,0,0]], half_sizes=[dims], colors=[col]))
            elif kind == "box":
                rr.log(f"world/objects/{oid}/geom", rr.Boxes3D(centers=[[0,0,0]], half_sizes=[dims], colors=[col]))
            elif kind == "cylinder":
                radius, height = dims
                rr.log(f"world/objects/{oid}/geom", rr.Cylinders3D(centers=[[0,0,0]], lengths=[height], radii=[radius], colors=[col]))
            else:
                rr.log(f"world/objects/{oid}/geom", rr.Boxes3D(centers=[[0,0,0]], half_sizes=[[0.05,0.05,0.05]], colors=[col]))

            # Velocity arrows in world space
            if args.arrows:
                kin = ent.get("kinematics", {})
                if "linear_velocity_world" in kin:
                    v = np.array(kin["linear_velocity_world"], float)
                else:
                    # estimate from previous position
                    v = np.zeros(3, float)
                    prev = prev_positions[oid]
                    # Try to derive dt from neighbor keys if uniform; else 0
                    if prev is not None:
                        # compute dt from time keys (works even with --skip)
                        prev_t = float(time_keys[max(0, i-args.skip)])
                        dt = t - prev_t
                        if dt > 0:
                            v = (p - prev) / dt
                prev_positions[oid] = p.copy()

                # scale arrow length for readability
                scale = 1.0
                rr.log("world/velocity", rr.Arrows3D(vectors=[(v*scale).tolist()], origins=[p.tolist()], radii=0.01))

        # Trails
        if args.trail:
            for oid, pts in trails.items():
                if len(pts) >= 2:
                    rr.log(f"world/trails/{oid}", rr.LineStrips3D([pts], radii=0.003))

    # Done
    print("Done streaming to Rerun. Use the Viewer timeline to play the animation.")

if __name__ == "__main__":
    main()
