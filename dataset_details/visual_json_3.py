import json
import plotly.graph_objects as go

# Paths
PATH_DATA = "/Users/sebastiancavada/Desktop/tmp_paris/tiny_vqa_creation/output/run_28_general/test_run_28_general.json"

# Options
PERCENT_MODE = "root"  # "root", "parent", or "entry"
INCLUDE_QID_LAYER = False  # Set False to remove the 3rd layer (question_id)
THRESHOLD = 0.05
EXCLUDE_LABELS = ["Material<br>Identification", "Mechanics", "Spatial<br>Reasoning"]  # e.g. ["Viewpoint", "Temporal<br>Reasoning"]
EXCLUDED_QUESTION_IDS: list[str] = [
    "F_OCCLUSION_PERCENTAGE_OBJECT",
    "F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_3",
    "F_CAMERA_ZOOM_BEHAVIOR",
    "F_FOCAL_LENGTH_CLASS",
]
OBJECT_PERMANENCE_LABEL_X = 0.24
OBJECT_PERMANENCE_LABEL_Y = 0.84
OBJECT_PERMANENCE_SUBLABEL_X = 0.24
OBJECT_PERMANENCE_SUBLABEL_Y = 0.75
PERSISTENCE_MAIN_CATEGORIES: list[str] = ["permanence", "persistence"]
PERSISTENCE_COUNT_QIDS: list[str] = [
    "F_PERSISTENCE_OBJECT_TOTAL_COUNT",
    "F_PERSISTENCE_OBJECT_TOTAL_COUNT_HIDDEN",
]
PERSISTENCE_IDENTITY_QIDS: list[str] = [
    "F_PERSISTENCE_OBJECT_PRESENT",
    "F_PERSISTENCE_OBJECT_DISAPPEAR",
]
# EXCLUDE_LABELS = ["Spatial<br>Reasoning", "Mechanics", "Temporal<br>Reasoning", "Viewpoint"]
SAVE_IMAGE = True
OUTPUT_IMAGE_PATH = "sunburst_high_res.png"
OUTPUT_IMAGE_SCALE = 8
FIG_SIZE = 850
LABEL_FONT_SIZE = 25
CENTER_FONT_SIZE = 55
EXTERNAL_LABEL_FONT_SIZE = 25
TOP_MARGIN = 20
BASE_LEFT_MARGIN = 20
EXTRA_LEFT_MARGIN = 0
LEFT_MARGIN = BASE_LEFT_MARGIN + EXTRA_LEFT_MARGIN
RIGHT_MARGIN = 150
BOTTOM_MARGIN = 90
RIGHT_DOMAIN_PADDING = 0.05
BOTTOM_DOMAIN_PADDING = 0.05
REFERENCE_LEFT_MARGIN_FOR_RIGHT_LABELS = BASE_LEFT_MARGIN
SUNBURST_X0 = 0.0
SUNBURST_X1 = 1.0 - RIGHT_DOMAIN_PADDING
SUNBURST_Y0 = BOTTOM_DOMAIN_PADDING
SUNBURST_Y1 = 1.0
CENTER_X = (SUNBURST_X0 + SUNBURST_X1) / 2.0
CENTER_Y = (SUNBURST_Y0 + SUNBURST_Y1) / 2.0

# Keep a deterministic clockwise layout that matches the reference design.
MAIN_CATEGORY_ORDER = [
    "material_understanding",
    "spatial_reasoning",
    "temporal",
    "view_point",
    "visual_percetion",
    "permanence",
    "mechanics",
    "persistence",
]

SUBCATEGORY_ORDER = {
    "material_understanding": [
        "young_modulus",
        "poisson_ratio",
        "material_identification",
        "mass",
        "density",
    ],
    "visual_percetion": ["camera_characteristics", "physics_property"],
    "view_point": ["visibility"],
    "temporal": ["event_ordering", "camera_motion"],
    "permanence": ["object_count", "object_identity"],
    "persistence": ["object_count", "object_identity"],
    "spatial_reasoning": ["distance", "size", "layout"],
    "mechanics": ["collision", "kinematics"],
}

# Label mappings
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
    "object_identity": "Object<br>Identity",
    "object_persistence": "Object<br>Permanence",
}

mapping_cat = {
    "mechanics": "Mechanics",
    "spatial_reasoning": "Spatial<br>Reasoning",
    "visual_percetion": "Visual<br>Perception", # Note: Check spelling in your data
    "temporal": "Temporal<br>Reasoning",
    "view_point": "Viewpoint",
    "material_understanding": "Material<br>Understanding",
    "permanence": "Permanence",
    "persistence": "Permanence",
}

mapping_cat_colors = {
    "mechanics": "#FF5733",              
    "spatial_reasoning": "#3498DB",       
    "permanence": "#F43FC7",             
    "temporal": "#0DA792",               
    "view_point": "#EEAC32",             
    "material_understanding": "#2BAE27",   
    "persistence": "#F43FC7",
    "visual_percetion": "#9B59B6"         # Added a color for visual perception
}

def _format_sig_floor(value: float, sig: int = 3) -> str:
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


def format_compact_count_floor(n: int) -> str:
    abs_n = abs(n)
    suffixes = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]
    for threshold, suffix in suffixes:
        if abs_n >= threshold:
            value = n / threshold
            s = _format_sig_floor(value, 3)
            return f"{s}{suffix}"
    return str(n)


# --- Data Loading ---
with open(PATH_DATA) as f:
    data = json.load(f)
assert isinstance(data, list), "Expected a list of question entries."
data = [item for item in data if item.get("question_id") not in EXCLUDED_QUESTION_IDS]

total_count = len(data)
CENTER_TEXT = f"<b>{format_compact_count_floor(total_count)}<br>VQA</b>"

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
    if not main_cat or not sub_cat:
        continue

    if main_cat in PERSISTENCE_MAIN_CATEGORIES and sub_cat == "object_persistence":
        assert qid in PERSISTENCE_COUNT_QIDS + PERSISTENCE_IDENTITY_QIDS
        if qid in PERSISTENCE_COUNT_QIDS:
            sub_cat = "object_count"
        else:
            sub_cat = "object_identity"

    main_totals.setdefault(main_cat, 0)
    subs_for_main.setdefault(main_cat, {})
    subs_for_main[main_cat].setdefault(sub_cat, 0)

    main_totals[main_cat] += 1
    subs_for_main[main_cat][sub_cat] += 1

    if INCLUDE_QID_LAYER:
        if qid:
            qids_for_sub.setdefault((main_cat, sub_cat), {})
            qids_for_sub[(main_cat, sub_cat)].setdefault(qid, 0)
            qids_for_sub[(main_cat, sub_cat)][qid] += 1

assert all(sum(subs.values()) == main_totals[main_cat] for main_cat, subs in subs_for_main.items())

main_order_index = {name: idx for idx, name in enumerate(MAIN_CATEGORY_ORDER)}

# Add Main Categories
for main_cat in sorted(
    main_totals.keys(),
    key=lambda x: (main_order_index.get(x, len(main_order_index)), x),
):
    total = main_totals[main_cat]
    cat_id = f"cat::{main_cat}"
    ids.append(cat_id)
    labels.append(mapping_cat.get(main_cat, main_cat))
    parents.append(root_id)
    values.append(total)
    id_to_maincat[cat_id] = main_cat

# Add Subcategories
for main_cat in sorted(
    subs_for_main.keys(),
    key=lambda x: (main_order_index.get(x, len(main_order_index)), x),
):
    sub_counts = subs_for_main[main_cat]
    parent_id = f"cat::{main_cat}"
    sub_order_index = {
        name: idx for idx, name in enumerate(SUBCATEGORY_ORDER.get(main_cat, []))
    }
    for sub_cat in sorted(
        sub_counts.keys(),
        key=lambda x: (sub_order_index.get(x, len(sub_order_index)), x),
    ):
        total = sub_counts[sub_cat]
        sub_id = f"{parent_id}::sub::{sub_cat}"
        ids.append(sub_id)
        labels.append(mapping_sub.get(sub_cat, sub_cat))
        parents.append(parent_id)
        values.append(total)
        id_to_maincat[sub_id] = main_cat

# Add Question IDs (3rd level)
if INCLUDE_QID_LAYER:
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

def _pct_text(count: int, total: int) -> str:
    return f"{(count / total) * 100:.1f}%"

def _right_label_x(reference_x: float) -> float:
    reference_plot_width = (
        FIG_SIZE - REFERENCE_LEFT_MARGIN_FOR_RIGHT_LABELS - RIGHT_MARGIN
    )
    current_plot_width = FIG_SIZE - LEFT_MARGIN - RIGHT_MARGIN
    absolute_x = (
        REFERENCE_LEFT_MARGIN_FOR_RIGHT_LABELS + reference_x * reference_plot_width
    )
    return (absolute_x - LEFT_MARGIN) / current_plot_width


def _add_connector_line(
    fig: go.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
) -> None:
    fig.add_shape(
        type="line",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        xref="paper",
        yref="paper",
        line=dict(color=color, width=2),
    )


layout_count = subs_for_main["spatial_reasoning"]["layout"]
collision_count = subs_for_main["mechanics"]["collision"]
camera_motion_count = subs_for_main["temporal"]["camera_motion"]
camera_char_count = subs_for_main["view_point"]["camera_characteristics"]
persistence_key = "permanence" if "permanence" in main_totals else "persistence"
object_count_count = subs_for_main[persistence_key]["object_count"]
object_identity_count = subs_for_main[persistence_key]["object_identity"]
object_permanence_total_count = object_count_count + object_identity_count
material_identification_count = subs_for_main["material_understanding"]["material_identification"]

# --- Color Processing ---
color_list = []
for i in range(len(labels)):
    if parents[i] == "":
        color_list.append("white")
        continue
    current_cat_key = id_to_maincat.get(ids[i])
    color_list.append(mapping_cat_colors.get(current_cat_key, "#D3D3D3"))

# --- Chart Formatting ---
if PERCENT_MODE == "parent":
    percent_token = "%{percentParent:.1%}"
    textinfo_mode = "label+percent parent"
elif PERCENT_MODE == "entry":
    percent_token = "%{percentEntry:.1%}"
    textinfo_mode = "label+percent entry"
else:
    percent_token = "%{percentRoot:.1%}"
    textinfo_mode = "label+percent root"

total_sum = values[0]
parent_ids = set(parents) - {""}

fig = go.Figure(
    go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        domain=dict(
            x=[SUNBURST_X0, SUNBURST_X1],
            y=[SUNBURST_Y0, SUNBURST_Y1],
        ),
        sort=False,
        branchvalues="total",
        textinfo=textinfo_mode,
        texttemplate=[
            "" if i == 0
            else (
                "" if (values[i] / total_sum < THRESHOLD) or labels[i] in EXCLUDE_LABELS
                else ("%{label}" if ids[i] in parent_ids else "%{label}<br>" + percent_token)
            )
            for i in range(len(labels))
        ],
        insidetextorientation="horizontal",
        textfont=dict(size=LABEL_FONT_SIZE),
        marker=dict(
            colors=color_list,
            line=dict(color="white", width=2),
        ),
    ),
)

fig.add_annotation(
    text=CENTER_TEXT,
    showarrow=False,
    font=dict(size=CENTER_FONT_SIZE),
    xref="paper",
    yref="paper",
    x=CENTER_X, y=CENTER_Y,
)

fig.update_layout(
    width=FIG_SIZE,
    height=FIG_SIZE,
    margin=dict(t=TOP_MARGIN, l=LEFT_MARGIN, r=RIGHT_MARGIN, b=BOTTOM_MARGIN),
    paper_bgcolor="white",
)

# External annotations for small slices
small_label_font = dict(size=EXTERNAL_LABEL_FONT_SIZE, color="#444444", family="Arial")
small_label_font_internal = dict(size=EXTERNAL_LABEL_FONT_SIZE, color="#ffffff", family="Arial")


fig.add_annotation(
    text="Mechanics",
    x=_right_label_x(0.78),
    y=0.465,
    showarrow=False,
    font=dict(size=20, color="#ffffff", family="Arial"),
    xref="paper",
    yref="paper",
)

fig.add_annotation(
    text="Spatial<br>Reasoning",
    x=_right_label_x(0.2),
    y=0.42,
    showarrow=False,
    font=dict(size=23, color="#ffffff", family="Arial"),
    xref="paper",
    yref="paper",
)

fig.add_annotation(
    text=f"Material<br>Identification<br>{_pct_text(material_identification_count, total_count)}",
    x=0.39,
    y=0.93,
    showarrow=False,
    font=dict(size=22, color="#ffffff", family="Arial"),
    xref="paper",
    yref="paper",
)

layout_label_x = 0.225
layout_label_y = 0.01
fig.add_annotation(
    text=f"Layout<br>{_pct_text(layout_count, total_count)}",
    x=layout_label_x,
    y=layout_label_y,
    showarrow=False,
    font=dict(size=EXTERNAL_LABEL_FONT_SIZE, color=mapping_cat_colors["spatial_reasoning"], family="Arial"),
    xref="paper",
    yref="paper",
)
height_layout=0.125
_add_connector_line(
    fig=fig,
    x0=0.28,
    y0=height_layout,
    x1=0.28,
    y1=0.09,
    color=mapping_cat_colors["spatial_reasoning"],
)


# Camera Characteristics (Viewpoint - Yellow/Gold)
# Note: In your script, this falls under "view_point" mapping
camera_char_label_x = 0.622
camera_char_label_y = -0.01
fig.add_annotation(
    text=f"Camera<br>Motion {_pct_text(camera_motion_count, total_count)}",
    x=camera_char_label_x,
    y=camera_char_label_y,
    showarrow=False,
    font=dict(size=EXTERNAL_LABEL_FONT_SIZE, color=mapping_cat_colors["temporal"], family="Arial"),
    xref="paper",
    yref="paper",
)
# height_camera_char=0.00
width_camera_char=camera_char_label_x
_add_connector_line(
    fig=fig,
    x0=0.61,
    y0=0.07,
    x1=width_camera_char,
    y1=0.106,
    color=mapping_cat_colors["temporal"],
)

# Camera Motion (Temporal Reasoning - Teal)
camera_motion_label_x = _right_label_x(1.06)
camera_motion_label_y = 0.1
fig.add_annotation(
    text=f"Camera<br>Characteristics<br>{_pct_text(camera_char_count, total_count)}",
    x=camera_motion_label_x,
    y=camera_motion_label_y,
    showarrow=False,
    font=dict(size=EXTERNAL_LABEL_FONT_SIZE, color=mapping_cat_colors["view_point"], family="Arial"),
    xref="paper",
    yref="paper",
    )
height_camera_motion = 0.21
_add_connector_line(
    fig=fig,
    x0=0.835,
    y0=0.235,
    x1=0.865,
    y1=height_camera_motion,
    color=mapping_cat_colors["view_point"],
)


# THIS WHEN TEMPORAL REASONING AND VIEWPOINT ARE SWAPPED
# # Camera Characteristics (Viewpoint - Yellow/Gold)
# # Note: In your script, this falls under "view_point" mapping
# camera_char_label_x = 0.56
# camera_char_label_y = -0.05
# fig.add_annotation(
#     text=f"Camera<br>Characteristics<br>{_pct_text(camera_char_count, total_count)}",
#     x=camera_char_label_x,
#     y=camera_char_label_y,
#     showarrow=False,
#     font=dict(size=EXTERNAL_LABEL_FONT_SIZE, color=mapping_cat_colors["view_point"], family="Arial"),
#     xref="paper",
#     yref="paper",
# )
# # height_camera_char=0.00
# width_camera_char=0.56
# _add_connector_line(
#     fig=fig,
#     x0=width_camera_char,
#     y0=0.07,
#     x1=width_camera_char,
#     y1=0.095,
#     color=mapping_cat_colors["view_point"],
# )

# # Camera Motion (Temporal Reasoning - Teal)
# camera_motion_label_x = _right_label_x(1.06)
# camera_motion_label_y = 0.2
# fig.add_annotation(
#     text=f"Camera<br>Motion {_pct_text(camera_motion_count, total_count)}",
#     x=camera_motion_label_x,
#     y=camera_motion_label_y,
#     showarrow=False,
#     font=dict(size=EXTERNAL_LABEL_FONT_SIZE, color=mapping_cat_colors["temporal"], family="Arial"),
#     xref="paper",
#     yref="paper",
#     )
# height_camera_motion = 0.27
# _add_connector_line(
#     fig=fig,
#     x0=0.86,
#     y0=height_camera_motion,
#     x1=0.885,
#     y1=height_camera_motion,
#     color=mapping_cat_colors["temporal"],
# )

# Collision (Mechanics - Red/Orange)
collision_label_x = _right_label_x(1.15)
collision_label_y = 0.29
fig.add_annotation(
    text=f"Collision {_pct_text(collision_count, total_count)}",
    x=collision_label_x,
    y=collision_label_y,
    showarrow=False,
    font=dict(size=EXTERNAL_LABEL_FONT_SIZE, color=mapping_cat_colors["mechanics"], family="Arial"),
    xref="paper",
    yref="paper",
)
_add_connector_line(
    fig=fig,
    x0=0.89,
    y0=0.31,
    x1=0.91,
    y1=0.31,
    color=mapping_cat_colors["mechanics"],
)


# Define coordinates for the new position and the connector line
# The box is placed at a new top-right position
# and the slice position is defined for the pink "Permanence" sliver.
NEW_BOX_X = 1.2
NEW_BOX_Y = 0.51
# The reference point on the Sunburst for the pink "Permanence" slice
SLICE_X = 0.95
SLICE_Y = 0.5074

# 1. Add the Connector Line (as a shape)
# This creates a line from the pink slice to the callout box.
fig.add_shape(
    type="line",
    # Start of the line at the slice's approximate position
    x0=SLICE_X, y0=SLICE_Y, 
    # End of the line at the box's top-right corner
    x1=0.97, y1=0.5074,
    xref="paper", yref="paper",
    # Set the line color to the specific 'permanence' pink and adjust width
    line=dict(color=mapping_cat_colors["permanence"], width=1.8),
)

# 2. Add the Styled Callout Box
# This box contains the detailed information about Object Permanence.
fig.add_annotation(
    text=(
        # Apply the 'permanence' pink color to the title within the box
        f"Object<br>Permanence</b><br>"
        # List the sub-categories with their calculated percentages
        f"• Count: {_pct_text(object_count_count, total_count)}</b><br>"
        f"• Identity: {_pct_text(object_identity_count, total_count)}</b>"
    ),
    # Set the box's position to the new top-right coordinates
    x=NEW_BOX_X,
    y=NEW_BOX_Y,
    # Align the text within the box to the left
    align="left",
    # Hide the default arrow from the annotation
    showarrow=False,
    xref="paper", yref="paper",
    # Set the font family and make the sub-text color slightly less dark
    font=dict(size=22, color=mapping_cat_colors["permanence"], family="Arial"),
    # Define the box's styling: white background, the specific 'permanence' pink outline, and spacing
    # bgcolor="white",
    # bordercolor=mapping_cat_colors["permanence"], # Outline color is now the pink
    # borderwidth=2.5, # Slightly thicker border for definition
    # borderpad=12,   # More inner padding for a cleaner look
)


if SAVE_IMAGE:
    fig.write_image(OUTPUT_IMAGE_PATH, scale=OUTPUT_IMAGE_SCALE)

fig.show()
