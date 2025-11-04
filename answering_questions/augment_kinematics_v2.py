#!/usr/bin/env python3
import json, sys, math

# Reads:
# { "simulation": { "0000.000": { "objects": { "1": { "obb": { "center": [x,y,z] } } } } } }
# Writes the same structure plus per-object "motion" with v, speed, a, accel for each timestep.

def get_center(obj):
    try:
        c = obj["obb"]["center"]
        if isinstance(c, list) and len(c) >= 3:
            return float(c[0]), float(c[1]), float(c[2])
    except Exception:
        pass
    return None

def main():
    if len(sys.argv) < 2:
        print("usage: python augment_kinematics_v2.py input.json [output.json]", file=sys.stderr)
        sys.exit(2)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    data = json.load(open(in_path, "r", encoding="utf-8"))
    sim = data["simulation"]
    times = sorted(sim.keys(), key=lambda s: float(s))

    last = {}  # oid -> (t, px, py, pz, vx, vy, vz or None)

    for t_str in times:
        t = float(t_str)
        frame = sim[t_str]
        for oid, obj in frame.get("objects", {}).items():
            p = get_center(obj)
            if p is None:
                continue
            vx = vy = vz = speed = ax = ay = az = accel = None
            if oid in last:
                t_prev, px_prev, py_prev, pz_prev, vx_prev, vy_prev, vz_prev = last[oid]
                dt = t - t_prev
                if dt != 0:
                    vx = (p[0]-px_prev)/dt
                    vy = (p[1]-py_prev)/dt
                    vz = (p[2]-pz_prev)/dt
                    speed = (vx*vx + vy*vy + vz*vz) ** 0.5
                    if vx_prev is not None:
                        ax = (vx - vx_prev)/dt
                        ay = (vy - vy_prev)/dt
                        az = (vz - vz_prev)/dt
                        accel = (ax*ax + ay*ay + az*az) ** 0.5
            obj["kinematics"] = {"v":[vx,vy,vz], "speed":speed, "a":[ax,ay,az], "accel":accel}
            last[oid] = (t, p[0], p[1], p[2], vx, vy, vz)

    out_text = json.dumps(data, indent=2)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        sys.stdout.write(out_text)

if __name__ == "__main__":
    main()


# Example usage:
# python augment_kinematics_v2.py /data0/sebastian.cavada/datasets/simulations_v2/dl3dv/random/1/c-1_no-1_d-4_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-0_20251101_021957/simulation.json /data0/sebastian.cavada/datasets/simulations_v2/dl3dv/random/1/c-1_no-1_d-4_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-0_20251101_021957/simulation_kinematics_2.json