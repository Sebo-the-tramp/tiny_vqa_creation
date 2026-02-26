from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.markers as mmarkers
import numpy as np
import pandas as pd
import seaborn as sns

categories = {
    "material_understanding": "Material Understanding",
    "mechanics": "Mechanics",
    "spatial_reasoning": "Spatial Reasoning",
    "view_point": "Viewpoint",
    "persistence": "Permanence",
    "temporal": "Temporal Reasoning",
}

subcategories = {
    # Material understanding
    'density': "Density", 
    'mass': "Mass", 
    'material_identification': "Material Identification", 
    'poisson_ratio': "Poisson's ratio", 
    'young_modulus': "Young's modulus", 

    # Mechanics
    'collision': "Collision",
    'kinematics': "Kinematics",

    # Spatial reasoning
    'distance': "Distance",
    'layout': "Layout",
    'size': "Size",
    
    # Viewpoint
    'camera_characteristics': "Camera Characteristics",
    'visibility': "Visibility",

    # Permanence
    'object_identity': "Object Identity",
    'object_count': "Object Count",
    
    # Temporal reasoning
    'camera_motion': "Camera Motion",
    'event_ordering': "Event Ordering"
}

subcat_to_cat = {}  # Automatically populated during loading of the results
question_to_subcat = {}  # Automatically populated during loading of the results

questions = {}

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
    "mechanics": categories.get("mechanics"),
    "spatial_reasoning": categories.get("spatial_reasoning"),
    "persistence": categories.get("persistence"),
    "temporal": categories.get("temporal"),
    "view_point": categories.get("view_point"),
    "material_understanding": "Material Underst."
}

family_marker = {
    "AquilaVL":		    "o",  # circle
    "InternVLChat":	    ">",  # triangle right
    "InternVLChat2":    "H",  # hexagon (flat)
    "LLaVAVideo":		"h",  # hexagon (pointy)
    "Mantis":			"D",  # diamond
    "MiniCPMV":		    "d",  # thin diamond
    "Molmo":			".",  # point
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

def sort_categories(cats: List[str] = None) -> List[str]:
    if cats is None:
        cats = list(categories.keys())
    
    cat_order = {k: i for i, k in enumerate(categories.keys())}
    return sorted(cats, key=lambda sub: cat_order.get(sub, float("inf")))

def sort_subcategories(subcats: List[str] = None, flatten: bool = False) -> List[str]:
    if subcats is None:
        subcats = list(subcategories.keys())
    
    categories = sort_categories()
    subcats_out = {}
    for cat in categories:
        cat_subcats = [subcat for subcat, cat_subcat in subcat_to_cat.items() if cat_subcat == cat and subcat in subcats]
        sub_order = {k: i for i, k in enumerate(subcategories.keys())}
        cat_subcats_sorted = sorted(cat_subcats, key=lambda sub: sub_order.get(sub, float("inf")))
        
        if len(cat_subcats_sorted) > 0:
            subcats_out[cat] = cat_subcats_sorted
    
    if flatten:
        return [subcat for cat_subcats in subcats_out.values() for subcat in cat_subcats]
    else:
        return subcats_out

def sort_questions(triplets: List[List[str]], quests: List[str] = None, flatten: bool = False) -> List[str]:
    # To avoid storing all questions ID in mapping, the strategy differs here.
    # triplets is a list of (category, sub_category, question_id) triplets, 
    # which is used to infer the mapping between question_id and subcategories.
    if quests is None:
        quests = [qid for cat, subcat, qid in triplets]

    triplets_subcats = [subcat for cat, subcat, qid in triplets]
    subcats = sort_subcategories(triplets_subcats)  # This will also do the sanity check on categories and subcategories

    qs = {}
    for cat in subcats:
        qs[cat] = {}
        for subcat in subcats[cat]:
            subcat_qids = [qid for c, s, qid in triplets if s == subcat]
            subcat_qids_sorted = sorted(np.unique(subcat_qids).tolist())  # Assuming question IDs can be sorted lexicographically

            if len(subcat_qids_sorted) > 0:
                qs[cat][subcat] = subcat_qids_sorted

        if len(qs[cat]) == 0:
            del qs[cat]

    if flatten:
        return [qid for cat_subcats in qs.values() for subcat_qids in cat_subcats.values() for qid in subcat_qids]
    else:
        return qs

def _build_model_style(
    metadata_path: str | Path | None,
    *,
    group_by: str = "model_id",
    family_marker_mode: str = "distinct",
) -> tuple[dict[str, tuple[str, object, float]], dict[str, str]]:
    assert group_by in {"model_id", "model_family"}, "group_by must be 'model_id' or 'model_family'"

    import utils.utils_read

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

    params = np.clip(params, 1.0, 20.0)
    sizes = (params - 1.0) / (30.0 - 1.0)
    sizes = 8 + 7 * sizes  # Scale to range [8, 22] for better visibility

    model_style = {}
    for group_id, color, fam, size in zip(group_ids, palette, families, sizes):
        model_style[str(group_id)] = (color, family_markers.get(fam, "o"), float(size))

    # Alphabetically sort family_map and model_style by keys for consistent ordering
    family_map = {k: family_map[k] for k in sorted(family_map)}
    model_style = {k: model_style[k] for k in sorted(model_style)}

    return model_style, family_map