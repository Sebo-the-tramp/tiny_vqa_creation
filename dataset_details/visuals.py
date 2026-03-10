import json
import plotly.graph_objects as go

# Paths
PATH_BAL_SUB = "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/answering_questions/balancing_sub_categories.json"
PATH_DATA = "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/simple_vqa.json"

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


def _format_sig_floor(value, sig=3):
    if value == 0:
        return "0"
    import math

    sign = -1 if value < 0 else 1
    value = abs(value)
    exp = math.floor(math.log10(value))
    decimals = max(sig - exp - 1, 0)
    factor = 10 ** decimals
    floored = math.floor(value * factor) / factor
    if decimals == 0:
        s = str(int(floored))
    else:
        s = f"{floored:.{decimals}f}".rstrip("0").rstrip(".")
    return f"-{s}" if sign < 0 else s


def format_compact_count_floor(n):
    abs_n = abs(n)
    suffixes = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]
    for threshold, suffix in suffixes:
        if abs_n >= threshold:
            value = n / threshold
            s = _format_sig_floor(value, 3)
            return f"{s}{suffix}"
    return str(n)

# --- Data Loading ---
# Mocking data for structure if files are missing; replace with your open() calls
try:
    with open(PATH_BAL_SUB) as f:
        balancing = json.load(f)
    with open(PATH_DATA) as f:
        data = json.load(f)
except FileNotFoundError:
    print("Files not found. Please check paths.")
    exit()

main_totals, subs_for_main = {}, {}

for main_cat, items in data.items():
    seen = set()
    for _, props in items.items():
        if isinstance(props, dict):
            sc = props.get("sub_category")
            if sc and sc in balancing:
                seen.add(sc)
    if not seen:
        continue
    total = sum(balancing[sc] for sc in seen)
    if total <= 0:
        continue
    main_totals[main_cat] = total
    subs_for_main[main_cat] = sorted(seen)

CENTER_TEXT = f"<b>{format_compact_count_floor(sum(main_totals.values()))}<br>VQA</b>"
labels, parents, values = [CENTER_TEXT], [""], [0.0]

# Add Main Categories
for main_cat, total in main_totals.items():
    labels.append(mapping_cat.get(main_cat, main_cat))
    parents.append(CENTER_TEXT)
    values.append(total)

# Add Subcategories
for main_cat, sc_list in subs_for_main.items():
    parent_label = mapping_cat.get(main_cat, main_cat)
    for sc in sc_list:
        labels.append(mapping_sub.get(sc, sc))
        parents.append(parent_label)
        values.append(balancing[sc])

values[0] = sum(main_totals.values())

# --- Color Processing ---
color_list = []
for i, label in enumerate(labels):
    if parents[i] == "":
        color_list.append("white")
        continue
    
    # Determine which main category this label (or its parent) belongs to
    current_cat_key = None
    # Check if the current label is a main category
    for key, mapped_val in mapping_cat.items():
        if mapped_val == label:
            current_cat_key = key
            break
            
    # If not found, it's a subcategory; check the parent's mapping
    if current_cat_key is None:
        parent_label = parents[i]
        for key, mapped_val in mapping_cat.items():
            if mapped_val == parent_label:
                current_cat_key = key
                break
    
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
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo=textinfo_mode,
        texttemplate=[
            "" if (i == 0 or values[i] / total_sum < threshold) or labels[i] in ["Viewpoint", "Temporal<br>Reasoning"]
            else ("%{label}" if labels[i] in parent_nodes else "%{label}<br>" + percent_token)
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
    showarrow=False,    font=dict(size=30),
    x=0.5, y=0.5,
)

ratio = 4/2.5
fig.update_layout(
    width=700, 
    height=700/ratio,
    margin=dict(t=20, l=20, r=20, b=40),
    paper_bgcolor="white",
)

# External annotations for small slices
small_label_font = dict(size=12, color="#444444", family="Arial")
fig.add_annotation(text="Layout 2%", x=0.53, y=-0.05, showarrow=False, font=small_label_font)
fig.add_annotation(text="Camera<br>Motion 2%", x=0.85, y=0.23, showarrow=False, font=small_label_font)
fig.add_annotation(text="Camera<br>Characteristics<br>1%", x=0.92, y=0.5, showarrow=False, font=small_label_font)

fig.add_annotation(text="Viewpoint", x=0.685, y=0.46, showarrow=False, font=small_label_font)
fig.add_annotation(text="Temporal<br>Reasoning", x=0.625, y=0.36, showarrow=False, font= dict(size=9.510, color="#FFFFFF", family="Arial"))

fig.show()
