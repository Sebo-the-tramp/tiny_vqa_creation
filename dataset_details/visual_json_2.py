import json
import math
import plotly.graph_objects as go

# --- Paths ---
PATH_DATA = "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general/test_run_28_general.json"

# --- Configuration ---
PERCENT_MODE = "parent"  # "parent" or "entry"
THRESHOLD = 0.05
# Add labels here to hide them from automatic rendering
EXCLUDE_LABELS = ["Spatial<br>Reasoning", "Mechanics", "Temporal<br>Reasoning", "Viewpoint"]

mapping_sub = {
    "visibility": "Visibility",
    "material_identification": "Material<br>Identification",
    "size": "Size",
    "camera_characteristics": "Camera<br>Characteristics",
    "physics_property": "Physics<br>Property",
    "kinematics": "Kinematics",
    "collision": "Collision",
    "mass": "Mass",
    "density": "Density",
    "young_modulus": "Young's<br>Modulus",
    "poisson_ratio": "Poisson<br>Ratio",
    "camera_motion": "Camera Motion",
    "layout": "Layout",
    "distance": "Distance",
    "event_ordering": "Event<br>Ordering",
    "object_count": "Object<br>Count",
    "object_identity": "Object<br>Identity"
}

mapping_cat = {
    "mechanics": "Mechanics",
    "spatial_reasoning": "Spatial<br>Reasoning",
    "visual_percetion": "Visual<br>Perception",
    "temporal": "Temporal<br>Reasoning",
    "view_point": "Viewpoint",
    "material_understanding": "Material<br>Understanding",
    "permanence": "Permanence",
}

mapping_cat_colors = {
    "mechanics": "#FF5733",              
    "spatial_reasoning": "#3498DB",       
    "permanence": "#F43FC7",             
    "temporal": "#0DA792",               
    "view_point": "#EEAC32",             
    "material_understanding": "#2BAE27",   
    "visual_percetion": "#9B59B6"         
}

def _format_sig_floor(value, sig=3):
    if value == 0: return "0"
    sign = -1 if value < 0 else 1
    value = abs(value)
    exp = math.floor(math.log10(value))
    decimals = max(sig - exp - 1, 0)
    factor = 10 ** decimals
    floored = math.floor(value * factor) / factor
    s = str(int(floored)) if decimals == 0 else f"{floored:.{decimals}f}".rstrip("0").rstrip(".")
    return f"-{s}" if sign < 0 else s

def format_compact_count_floor(n):
    abs_n = abs(n)
    suffixes = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]
    for threshold, suffix in suffixes:
        if abs_n >= threshold:
            return f"{_format_sig_floor(n / threshold, 3)}{suffix}"
    return str(n)

# --- Data Loading & Processing ---
try:
    with open(PATH_DATA) as f:
        data = json.load(f)
except FileNotFoundError:
    print("File not found.")
    raise SystemExit(1)

total_count = len(data)
CENTER_TEXT = f"<b>{format_compact_count_floor(total_count)}<br>VQA</b>"

root_id = "root"
ids, labels, parents, values = [root_id], [CENTER_TEXT], [""], [0.0]
id_to_maincat = {root_id: None}
main_totals, subs_for_main = {}, {}

for item in data:
    main_cat = item.get("category")
    sub_cat = item.get("sub_category")
    if not main_cat or not sub_cat: continue

    main_totals[main_cat] = main_totals.get(main_cat, 0) + 1
    subs_for_main.setdefault(main_cat, {})[sub_cat] = subs_for_main[main_cat].get(sub_cat, 0) + 1

# Sorting keys to keep visual positions stable
for main_cat in sorted(main_totals.keys(), key=lambda x: main_totals[x], reverse=True):
    cat_id = f"cat::{main_cat}"
    ids.append(cat_id)
    labels.append(mapping_cat.get(main_cat, main_cat))
    parents.append(root_id)
    values.append(main_totals[main_cat])
    id_to_maincat[cat_id] = main_cat

    for sub_cat, count in subs_for_main[main_cat].items():
        sub_id = f"{cat_id}::sub::{sub_cat}"
        ids.append(sub_id)
        labels.append(mapping_sub.get(sub_cat, sub_cat))
        parents.append(cat_id)
        values.append(count)
        id_to_maincat[sub_id] = main_cat

values[0] = sum(main_totals.values())

# --- Colors ---
color_list = ["white"] + [mapping_cat_colors.get(id_to_maincat[ids[i]], "#D3D3D3") for i in range(1, len(ids))]

# --- Chart ---
percent_token = "%{percentParent:.1%}" if PERCENT_MODE == "parent" else "%{percentEntry:.1%}"
parent_ids = set(parents) - {""}

fig = go.Figure(go.Sunburst(
    ids=ids, labels=labels, parents=parents, values=values,
    branchvalues="total",
    texttemplate=[
        "" if (i == 0 or labels[i] in EXCLUDE_LABELS or values[i]/values[0] < THRESHOLD)
        else ("%{label}" if ids[i] in parent_ids else "%{label}<br>" + percent_token)
        for i in range(len(labels))
    ],
    insidetextorientation="horizontal",
    textfont=dict(size=16, family="Arial Black"),
    marker=dict(colors=color_list, line=dict(color="white", width=2)),
))

fig.update_layout(
    width=700, height=700/1.6,
    margin=dict(t=20, l=20, r=20, b=40),
    paper_bgcolor="white"
)

# --- Manual Annotations ---
fig.add_annotation(text=CENTER_TEXT, x=0.5, y=0.5, showarrow=False, font=dict(size=30))

# Small Slices (External)
small_label_font = dict(size=12, color="#444444")
fig.add_annotation(text="Layout 2%", x=0.53, y=-0.05, showarrow=False, font=small_label_font)
fig.add_annotation(text="Camera<br>Motion 2%", x=0.85, y=0.23, showarrow=False, font=small_label_font)

# Your Custom Placements (Internal)
fig.add_annotation(text="<b>Spatial<br>Reasoning</b>", x=0.35, y=0.45, showarrow=False, font=dict(color="white", size=14))
fig.add_annotation(text="<b>Mechanics</b>", x=0.62, y=0.55, showarrow=False, font=dict(color="white", size=12))
fig.add_annotation(text="<b>Temporal<br>Reasoning</b>", x=0.60, y=0.38, showarrow=False, font=dict(color="white", size=11))
fig.add_annotation(text="<b>Viewpoint</b>", x=0.68, y=0.48, showarrow=False, font=dict(color="#444444", size=12))

fig.show()