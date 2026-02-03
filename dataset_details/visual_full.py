import json
import plotly.graph_objects as go

# Paths
PATH_DATA = "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_24_general/test_run_24_general.json"

PERCENT_MODE = "entry"  # change to "parent" if you prefer

# Label mappings
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
    "visual_percetion": "Visual<br>Perception", # Note: Check spelling in your data
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
    "visual_percetion": "#9B59B6"         # Added a color for visual perception
}

# --- Data Loading ---
try:
    with open(PATH_DATA) as f:
        data = json.load(f)
except FileNotFoundError:
    print("File not found. Please check path.")
    exit()

# Expecting a list of question entries
total_count = len(data) if isinstance(data, list) else 0
CENTER_TEXT = f"{total_count}<br>VQA"
root_id = "root"
ids, labels, parents, values = [root_id], [CENTER_TEXT], [""], [0.0]
id_to_maincat = {root_id: None}
main_totals = {}
subs_for_main = {}
qids_for_sub = {}

for item in data:
    if not isinstance(item, dict):
        continue
    main_cat = item.get("category")
    sub_cat = item.get("sub_category")
    qid = item.get("question_id")
    if not main_cat or not sub_cat or not qid:
        continue
    main_totals.setdefault(main_cat, 0)
    subs_for_main.setdefault(main_cat, {})
    subs_for_main[main_cat].setdefault(sub_cat, 0)
    qids_for_sub.setdefault((main_cat, sub_cat), {})

    main_totals[main_cat] += 1
    subs_for_main[main_cat][sub_cat] += 1
    qids_for_sub[(main_cat, sub_cat)].setdefault(qid, 0)
    qids_for_sub[(main_cat, sub_cat)][qid] += 1

# Add Main Categories
for main_cat, total in main_totals.items():
    cat_id = f"cat::{main_cat}"
    ids.append(cat_id)
    labels.append(mapping_cat.get(main_cat, main_cat))
    parents.append(root_id)
    values.append(total)
    id_to_maincat[cat_id] = main_cat

# Add Subcategories
for main_cat, sub_counts in subs_for_main.items():
    parent_id = f"cat::{main_cat}"
    for sub_cat, total in sub_counts.items():
        sub_id = f"{parent_id}::sub::{sub_cat}"
        ids.append(sub_id)
        labels.append(mapping_sub.get(sub_cat, sub_cat))
        parents.append(parent_id)
        values.append(total)
        id_to_maincat[sub_id] = main_cat

# Add Question IDs (3rd level)
for (main_cat, sub_cat), qid_counts in qids_for_sub.items():
    parent_id = f"cat::{main_cat}::sub::{sub_cat}"
    for qid, total in qid_counts.items():
        qid_id = f"{parent_id}::qid::{qid}"
        ids.append(qid_id)
        labels.append(qid)
        parents.append(parent_id)
        values.append(total)
        id_to_maincat[qid_id] = main_cat

values[0] = sum(main_totals.values())

# --- Color Processing ---
color_list = []
for i, label in enumerate(labels):
    if parents[i] == "":
        color_list.append("white")
        continue

    current_cat_key = id_to_maincat.get(ids[i])
    color_list.append(mapping_cat_colors.get(current_cat_key, "#D3D3D3"))

# --- Chart Formatting ---
if PERCENT_MODE == "parent":
    percent_token = "%{percentParent:.1%}"
    textinfo_mode = "label+percent parent"
else:
    percent_token = "%{percentEntry:.1%}"
    textinfo_mode = "label+percent entry"

total_sum = values[0]
threshold = 0.03
parent_nodes = set(parents) - {""}

fig = go.Figure(
    go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo=textinfo_mode,
        texttemplate=[
            "" if i == 0
            else (
                "" if (values[i] / total_sum < threshold) or labels[i] in [] # "Viewpoint", "Temporal<br>Reasoning"
                else "%{label}<br>" + percent_token
            )
            for i in range(len(labels))
        ],
        insidetextorientation="horizontal", 
        textfont=dict(size=16),
        marker=dict(
            colors=color_list, 
            line=dict(color="white", width=2)
        ),
    ),
)

fig.add_annotation(
    text=CENTER_TEXT,
    showarrow=False,
    font=dict(size=48),
    x=0.5, y=0.5,
)

fig.update_layout(
    width=1000, 
    height=1000,
    margin=dict(t=20, l=20, r=20, b=40),
    paper_bgcolor="white",
)

# External annotations for small slices
small_label_font = dict(size=12, color="#444444", family="Arial")
fig.add_annotation(text="Layout 2%", x=0.53, y=-0.05, showarrow=True, ax=0, ay=-30, arrowwidth=1, arrowhead=2, font=small_label_font)
fig.add_annotation(text="Camera<br>Motion 2%", x=0.85, y=0.23, showarrow=True, ax=30, ay=0, arrowwidth=1, arrowhead=2, font=small_label_font)
fig.add_annotation(text="Camera<br>Characteristics<br>1%", x=0.92, y=0.5, showarrow=True, ax=35, ay=0, arrowwidth=1, arrowhead=2, font=small_label_font)

# fig.add_annotation(text="Viewpoint", x=0.685, y=0.46, showarrow=False, font=small_label_font)
# fig.add_annotation(text="Temporal<br>Reasoning", x=0.625, y=0.36, showarrow=False, font= dict(size=9.510, color="#FFFFFF", family="Arial"))

fig.show()
