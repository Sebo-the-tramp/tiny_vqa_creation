#!/usr/bin/env python3
"""
augment_kinematics.py
Second-pass augmentation for JSON simulations. Computes per-object kinematics
(velocities, accelerations, jerks), angular velocities/accelerations, and
derived scalars (speed, curvature, tangential/normal accel, momentum, KE).

Works with these transform formats per object per step:
1) List of 6: [x,y,z, rx,ry,rz] (Euler angles, radians; default order=XYZ)
2) Dict {"p":[...], "q":[...]} or {"position":[...], "rotation_quat":[...]}
3) Dict {"pose":{"p":[...], "q":[...]}}  or {"state":{"pose":{"p":...,"q":...}}}

Angular velocity is computed from quaternions if available (recommended).
If only Euler is present, each axis is unwrapped and differentiated (approx).

Usage:
  python augment_kinematics.py input.json --out-json out.json --out-csv out.csv
  # optional flags:
  --euler-order XYZ|ZYX (default XYZ)
  --csv-float 6     (decimal places)
  --no-csv          (skip csv)
"""

import json
import math
import argparse
import sys
import csv
import os
from typing import Dict, Any, Tuple
import numpy as np


def clamp(x, a, b):
    return a if x < a else (b if x > b else x)


def norm(v):
    return float(np.linalg.norm(v))


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([0, 0, 0, 1], dtype=float)
    return (q / n).astype(float)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def quat_conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=float)


def quat_inv(q):
    q = np.asarray(q, dtype=float)
    return quat_conj(q) / np.dot(q, q)


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([0, 0, 0, 1], dtype=float)
    a = axis / n
    s = math.sin(0.5 * angle)
    c = math.cos(0.5 * angle)
    return np.array([a[0] * s, a[1] * s, a[2] * s, c], dtype=float)


def quat_to_axis_angle(q):
    q = quat_normalize(q)
    x, y, z, w = q
    if w < 0:  # shortest path
        x, y, z, w = -x, -y, -z, -w
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    axis = np.array([x / s, y / s, z / s], dtype=float)
    angle = 2.0 * math.acos(clamp(w, -1.0, 1.0))
    return axis, angle


def euler_to_quat(rx, ry, rz, order="XYZ"):
    # Intrinsic rotations: q = qx(rx) ⊗ qy(ry) ⊗ qz(rz) for "XYZ"
    # For "ZYX": q = qz(rz) ⊗ qy(ry) ⊗ qx(rx)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    qx = np.array([sx, 0, 0, cx], dtype=float)
    qy = np.array([0, sy, 0, cy], dtype=float)
    qz = np.array([0, 0, sz, cz], dtype=float)

    if order == "XYZ":
        q = quat_mul(quat_mul(qx, qy), qz)
    elif order == "ZYX":
        q = quat_mul(quat_mul(qz, qy), qx)
    else:
        raise ValueError(f"Unsupported euler order: {order}")
    return quat_normalize(q)


def mat3_to_quat(R):
    # Robust conversion from rotation matrix to quaternion
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return quat_normalize(np.array([qx, qy, qz, qw], dtype=float))


def ensure_quat_continuity(qs: np.ndarray) -> np.ndarray:
    # Flip sign if dot product negative between consecutive quats
    out = qs.copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0:
            out[i] = -out[i]
    return out


def extract_pose_quat(
    obj_step: Dict[str, Any], euler_order="XYZ"
) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (position[3], quaternion[4])
    # Try state->pose
    if "state" in obj_step and isinstance(obj_step["state"], dict):
        pose = obj_step["state"].get("pose", {})
        p = pose.get("p")
        q = pose.get("q")
        if p is not None and q is not None:
            return np.array(p, dtype=float), quat_normalize(np.array(q, dtype=float))

    # Try transform as dict with p/q or position/rotation_quat or pose{}
    tr = obj_step.get("transform", None)
    if isinstance(tr, dict):
        if "p" in tr and "q" in tr:
            return np.array(tr["p"], dtype=float), quat_normalize(
                np.array(tr["q"], dtype=float)
            )
        if (
            "pose" in tr
            and isinstance(tr["pose"], dict)
            and "p" in tr["pose"]
            and "q" in tr["pose"]
        ):
            return np.array(tr["pose"]["p"], dtype=float), quat_normalize(
                np.array(tr["pose"]["q"], dtype=float)
            )
        if "position" in tr and "rotation_quat" in tr:
            return np.array(tr["position"], dtype=float), quat_normalize(
                np.array(tr["rotation_quat"], dtype=float)
            )
        # transform as 4x4 matrix (flat list length 16) or nested list 4x4
        if "matrix" in tr:
            M = np.array(tr["matrix"], dtype=float).reshape(4, 4)
            R = M[:3, :3]
            p = M[:3, 3]
            return p, mat3_to_quat(R)

    # If transform is a list [x,y,z, rx,ry,rz]
    if isinstance(tr, list) and (len(tr) == 6):
        x, y, z, rx, ry, rz = tr
        q = euler_to_quat(rx, ry, rz, order=euler_order)
        return np.array([x, y, z], dtype=float), q

    # As a last resort: try "position"/"rotation_euler" keys
    if isinstance(tr, dict) and "position" in tr and "rotation_euler" in tr:
        rx, ry, rz = tr["rotation_euler"]
        q = euler_to_quat(rx, ry, rz, order=euler_order)
        return np.array(tr["position"], dtype=float), q

    raise ValueError(
        "Could not parse pose/quaternion from object step; expected transform or state.pose."
    )


def unwrap_euler_series(eulers: np.ndarray) -> np.ndarray:
    # Unwrap each axis to avoid 2π jumps
    u = np.unwrap(eulers, axis=0)
    return u


def central_quat_omega(q_prev, q_next, dt):
    # Angular velocity using quaternion delta across 2 frames (central difference).
    # ω = axis * (angle / dt), where delta = q_prev^{-1} ⊗ q_next
    if dt <= 0:
        return np.zeros(3, dtype=float)
    d = quat_mul(quat_inv(q_prev), q_next)
    # ensure shortest path
    if d[3] < 0:
        d = -d
    axis, angle = quat_to_axis_angle(d)
    return axis * (angle / dt)


def gradient_with_time(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Vectorized gradient with non-uniform time support
    # numpy.gradient handles variable spacing for 1D; for 2D we loop columns
    if arr.ndim == 1:
        return np.gradient(arr, t, edge_order=2)
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = np.gradient(arr[:, j], t, edge_order=2)
    return out


def compute_kinematics(
    times: np.ndarray,
    positions: np.ndarray,
    quats: np.ndarray,
    masses: float,
    have_quat: bool,
) -> Dict[str, np.ndarray]:
    # Linear kinematics
    v = gradient_with_time(positions, times)  # m/s
    a = gradient_with_time(v, times)  # m/s^2
    j = gradient_with_time(a, times)  # m/s^3

    speed = np.linalg.norm(v, axis=1)
    v_hat = np.where(speed[:, None] > 1e-12, v / speed[:, None], 0.0)
    a_t = np.sum(a * v_hat, axis=1)  # tangential accel
    a_n = np.sqrt(np.maximum(0.0, np.sum(a * a, axis=1) - a_t * a_t))
    cross = np.cross(v, a)
    curvature = np.where(speed > 1e-12, np.linalg.norm(cross, axis=1) / (speed**3), 0.0)
    momentum = masses * v
    ke_trans = 0.5 * masses * (speed**2)

    # Angular kinematics
    if have_quat:
        qs = ensure_quat_continuity(quats)
        # central difference: ω_i from q_{i-1} to q_{i+1}
        w = np.zeros_like(positions)
        for i in range(len(times)):
            if i == 0:
                dt = times[i + 1] - times[i]
                w[i] = central_quat_omega(qs[i], qs[i + 1], dt)
            elif i == len(times) - 1:
                dt = times[i] - times[i - 1]
                w[i] = central_quat_omega(qs[i - 1], qs[i], dt)
            else:
                dt = times[i + 1] - times[i - 1]
                w[i] = central_quat_omega(qs[i - 1], qs[i + 1], dt)
        alpha = gradient_with_time(w, times)
    else:
        # Fallback: from unwrapped Euler (approximate)
        # Placeholder: we don't have the raw Euler series here. Users who only
        # store Euler should convert to quats first. We will return zeros.
        w = np.zeros_like(positions)
        alpha = np.zeros_like(positions)

    return {
        "v": v,
        "a": a,
        "j": j,
        "speed": speed,
        "v_hat": v_hat,
        "a_t": a_t,
        "a_n": a_n,
        "curvature": curvature,
        "momentum": momentum,
        "ke_trans": ke_trans,
        "w": w,
        "alpha": alpha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input simulation JSON")
    ap.add_argument("--out-json", default=None, help="Path to write augmented JSON")
    ap.add_argument("--out-csv", default=None, help="Path to write long-form CSV")
    ap.add_argument("--euler-order", default="XYZ", choices=["XYZ", "ZYX"])
    ap.add_argument("--csv-float", type=int, default=6)
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        sim = json.load(f)

    steps = sim.get("simulation_steps", {})
    if not steps:
        print("No simulation_steps found.", file=sys.stderr)
        sys.exit(2)

    # Time vector
    time_keys = sorted(steps.keys(), key=lambda s: float(s))
    times = np.array([float(k) for k in time_keys], dtype=float)

    obj_ids = list(sim.get("objects", {}).keys())
    if not obj_ids:
        # try to infer from first step
        first_step = sim["simulation_steps"][time_keys[0]]
        obj_ids = list(first_step.get("objects", {}).keys())

    # Collect per-object arrays
    per_obj = {}
    for oid in obj_ids:
        P = []
        Q = []
        have_quat = True
        for tk in time_keys:
            step = sim["simulation_steps"][tk]
            obj = step["objects"].get(oid, None)
            if obj is None:
                raise ValueError(f"Missing object {oid} at time {tk}")
            try:
                p, q = extract_pose_quat(obj, euler_order=args.euler_order)
            except Exception:
                # If completely missing orientation, assume identity quaternion
                tr = obj.get("transform", None)
                if isinstance(tr, list) and len(tr) >= 3:
                    p = np.array(tr[:3], dtype=float)
                    q = np.array([0, 0, 0, 1], dtype=float)
                    have_quat = False
                else:
                    raise
            P.append(p)
            Q.append(q)
        P = np.vstack(P)  # (N,3)
        Q = np.vstack(Q)  # (N,4)
        mass = float(sim["objects"].get(oid, {}).get("mass", 1.0))
        kin = compute_kinematics(times, P, Q, mass, have_quat=have_quat)
        per_obj[oid] = {"P": P, "Q": Q, "mass": mass, "kin": kin}

    # Inject kinematics back into JSON
    for idx, tk in enumerate(time_keys):
        st = sim["simulation_steps"][tk]
        for oid in obj_ids:
            kin = per_obj[oid]["kin"]
            v = kin["v"][idx]
            a = kin["a"][idx]
            j = kin["j"][idx]
            w = kin["w"][idx]
            alpha = kin["alpha"][idx]
            rec = {
                "linear_velocity_world": np.round(v, args.csv_float).tolist(),
                "linear_accel_world": np.round(a, args.csv_float).tolist(),
                "linear_jerk_world": np.round(j, args.csv_float).tolist(),
                "speed": round(float(kin["speed"][idx]), args.csv_float),
                "tangential_accel": round(float(kin["a_t"][idx]), args.csv_float),
                "normal_accel": round(float(kin["a_n"][idx]), args.csv_float),
                "curvature": round(float(kin["curvature"][idx]), args.csv_float),
                "momentum_world": np.round(
                    kin["momentum"][idx], args.csv_float
                ).tolist(),
                "kinetic_energy_trans": round(
                    float(kin["ke_trans"][idx]), args.csv_float
                ),
                "angular_velocity_world": np.round(w, args.csv_float).tolist(),
                "angular_accel_world": np.round(alpha, args.csv_float).tolist(),
                "frame": "world",
            }
            st["objects"][oid]["kinematics"] = rec

    # Add a tiny conventions block if missing
    sim.setdefault("conventions", {})
    sim["conventions"].setdefault("axes", "right-handed, +Y up, +Z forward")
    sim["conventions"].setdefault("transform_direction", "world_T_object")
    sim["conventions"].setdefault("units", {"length": "m", "angle": "rad", "time": "s"})
    sim["conventions"]["augmentation"] = {
        "version": "kinematics-1.0",
        "method": "finite-difference (central), quaternion log for ω",
        "euler_order_assumed": args.euler_order,
    }

    # Write outputs
    out_json = args.out_json or (os.path.splitext(args.input)[0] + "_kinematics.json")
    with open(out_json, "w") as f:
        json.dump(sim, f, indent=2)
    print("Wrote augmented JSON:", out_json)

    if not args.no_csv:
        out_csv = args.out_csv or (os.path.splitext(args.input)[0] + "_kinematics.csv")
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            header = [
                "time",
                "object_id",
                "px",
                "py",
                "pz",
                "vx",
                "vy",
                "vz",
                "ax",
                "ay",
                "az",
                "jx",
                "jy",
                "jz",
                "speed",
                "a_t",
                "a_n",
                "curvature",
                "mx",
                "my",
                "mz",
                "ke_trans",
                "wx",
                "wy",
                "wz",
                "alphax",
                "alphay",
                "alphaz",
            ]
            wcsv.writerow(header)
            for i, tk in enumerate(time_keys):
                t = float(tk)
                for oid in obj_ids:
                    P = per_obj[oid]["P"][i]
                    kin = per_obj[oid]["kin"]
                    v = kin["v"][i]
                    a = kin["a"][i]
                    j = kin["j"][i]
                    w = kin["w"][i]
                    alpha = kin["alpha"][i]
                    speed = kin["speed"][i]
                    at = kin["a_t"][i]
                    an = kin["a_n"][i]
                    curv = kin["curvature"][i]
                    mom = kin["momentum"][i]
                    ke = kin["ke_trans"][i]
                    row = [
                        t,
                        oid,
                        P[0],
                        P[1],
                        P[2],
                        v[0],
                        v[1],
                        v[2],
                        a[0],
                        a[1],
                        a[2],
                        j[0],
                        j[1],
                        j[2],
                        speed,
                        at,
                        an,
                        curv,
                        mom[0],
                        mom[1],
                        mom[2],
                        ke,
                        w[0],
                        w[1],
                        w[2],
                        alpha[0],
                        alpha[1],
                        alpha[2],
                    ]
                    wcsv.writerow(row)
        print("Wrote CSV:", out_csv)


if __name__ == "__main__":
    main()
