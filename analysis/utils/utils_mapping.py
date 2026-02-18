from __future__ import annotations

from pathlib import Path

import matplotlib.markers as mmarkers
import numpy as np
import pandas as pd
import seaborn as sns

import utils.utils_read

mapping_sub = {
    "visibility": "Visibility",
    "material_identification": "Material\nIdentification",
    "size": "Size",
    "camera_characteristics": "Camera Characteristics",
    "physics_property": "Physics Property",
    "kinematics": "Kinematics",
    "collision": "Collision",
    "mass": "Mass",
    "camera_motion": "Camera Motion",
    "layout": "Layout",
    "distance": "Distance",
    "event_ordering": "Event Ordering",
    "poisson_ratio": "Poisson's ratio",
    "young_modulus": "Young's modulus",
    "density": "Density",
    "persistence": "Persistence",
}

mapping_sub_cat_id = {
    "visibility": "view_point",
    "material_identification": "visual_percetion",
    "size": "spatial_reasoning",
    "camera_characteristics": "view_point",
    "physics_property": "material_understanding",
    "kinematics": "mechanics",
    "collision": "mechanics",
    "mass": "material_understanding",
    "camera_motion": "temporal",
    "layout": "spatial_reasoning",
    "distance": "spatial_reasoning",
    "event_ordering": "temporal",    
}

mapping_sub_cat_name = {
    "Visibility": "view_point",
    "Material Identification": "visual_percetion",
    "Size": "spatial_reasoning",
    "Camera Characteristics": "view_point",
    "Physics Property": "material_understanding",
    "Kinematics": "mechanics",
    "Collision": "mechanics",
    "Mass": "material_understanding",
    "Camera Motion": "temporal",
    "Layout": "spatial_reasoning",
    "Distance": "spatial_reasoning",
    "Event Ordering": "temporal"
}

mapping_cat = {
    "mechanics": "Mechanics",
    "spatial_reasoning": "Spatial Reasoning",
    "persistence": "Permanence",
    "temporal": "Temporal Reasoning",
    "view_point": "Viewpoint",
    "material_understanding": "Material Understanding"
}

mapping_cat_order = {
    "material_understanding": 0,
    "mechanics": 1,
    "spatial_reasoning": 2,
    "view_point": 3,
    "persistence": 4,
    "temporal": 5,
}

mapping_cat_colors = {
    # More vivid pastel-like colors
    "mechanics": "#FF5733",              # vivid orange-red
    "spatial_reasoning": "#3498DB",       # vivid blue
    "persistence": "#F43FC7",        # vivid turquoise
    "temporal": "#0DA792",                # vivid orange
    "view_point": "#EEAC32",              # vivid yellow
    "material_understanding": "#2BAE27",   # vivid green
}

mapping_cat_short = {
    "mechanics": mapping_cat.get("mechanics"),
    "spatial_reasoning": mapping_cat.get("spatial_reasoning"),
    "persistence": mapping_cat.get("persistence"),
    "temporal": mapping_cat.get("temporal"),
    "view_point": mapping_cat.get("view_point"),
    "material_understanding": "Material Underst."
}

family_marker = {
    "AquilaVL":		    "o",  # circle
    "InternVLChat":	    ">",  # triangle right
    "InternVLChat":	    "H",  # hexagon (flat)
    "LLaVAVideo":		"h",  # hexagon (pointy)
    "Mantis":			"D",  # diamond
    "MiniCPMV":		    "d",  # thin diamond
    "MolmoE":			".",  # point
    "Phi":			    "1",  # tri-down tick (approx from image)
    "QwenVLChat":		"3",  # tri-left tick (approx)
    "XinyuanVL":		"P",  # filled plus
    "BLIP2":			"s",  # square
    "Cambrian":		    "^",  # triangle up
    "DeepSeekVL":		"v",  # triangle down
    "InstructBlip":	    "<",  # triangle left
    "LLaVA":			"*",  # star
    "LLaVAInterleave":	"8",  # octagon
    "Owl3":			    "1",  # tick-style (approx)
    "PaliGemma2":		"3",  # tick-style (approx)
    "VILAModel":		"o",  # circle
}

_DEFAULT_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "<",
    ">",
    "p",
    "*",
    "h",
    "H",
    "D",
    "d",
    ".",
    "1",
    "2",
    "3",
    "4",
    "8",
    "P",
    "X",
]

def _build_model_style(
    metadata_path: str | Path | None,
    *,
    group_by: str = "model_id",
    family_marker_mode: str = "distinct",
) -> tuple[dict[str, tuple[str, object, float]], dict[str, str]]:
    group_ids = []
    palette = []

    metadata_df = None
    if metadata_path is not None:
        metadata_df = utils.utils_read._load_model_metadata(metadata_path)

    families = []
    params = []
    family_map = {}
    for model_id in pd.unique(metadata_df["model_id"]):
        if metadata_df is not None and model_id in set(metadata_df["model_id"]):
            row = metadata_df[metadata_df["model_id"] == model_id].iloc[0]
            family = row["family"]
            params_b = pd.to_numeric(row.get("params_b", np.nan), errors="coerce")
        else:
            family = "Other"
            params_b = np.nan
        family_map[str(model_id)] = family
        if group_by == "model_id":
            families.append(family)
            params.append(params_b)

    if group_by == "model_family":
        group_ids = pd.unique(pd.Series(list(family_map.values())))
        families = list(group_ids)
        params = [np.nan] * len(group_ids)
    else:
        group_ids = pd.unique(metadata_df["model_id"])
    
    unique_families = list(dict.fromkeys(families)) if families else ["Other"]
    unique_families = sorted(unique_families)
    palette = sns.color_palette("tab20", len(group_ids))
    if family_marker_mode not in {"distinct", "rotated"}:
        raise ValueError(
            f"family_marker_mode must be 'distinct' or 'rotated', got {family_marker_mode}"
        )

    if family_marker_mode == "rotated":
        angle_step = 360.0 / max(1, len(unique_families))
        family_markers = {
            fam: mmarkers.MarkerStyle("^").rotated(angle_step * i)
            for i, fam in enumerate(unique_families)
        }
    else:
        markers = list(_DEFAULT_MARKERS)
        # for marker in list(mmarkers.MarkerStyle.markers.keys()):
        #     if marker not in markers:
        #         markers.append(marker)
        markers = [
            marker
            for marker in markers
            if marker not in (None, "", " ", "None", "none")
        ]
        family_markers = {
            fam: family_marker.get(fam, 's')
                 for i, fam in enumerate(unique_families)
        }
    

    params = np.array(params, dtype=float)
    valid_params = params[~np.isnan(params)]
    fallback = float(np.nanmedian(valid_params)) if valid_params.size else 5.0
    params = np.where(np.isnan(params), fallback, params)
    params = np.clip(params, 2.0, 15.0)
    sizes = (params - 2.0) / (15.0 - 2.0)

    model_style = {}
    for group_id, color, fam, size in zip(group_ids, palette, families, sizes):
        model_style[str(group_id)] = (color, family_markers.get(fam, "o"), float(size))

    # Alphabetically sort family_map and model_style by keys for consistent ordering
    family_map = {k: family_map[k] for k in sorted(family_map)}
    model_style = {k: model_style[k] for k in sorted(model_style)}

    return model_style, family_map