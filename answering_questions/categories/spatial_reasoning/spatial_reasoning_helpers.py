from __future__ import annotations

import itertools
import random

import numpy as np

from typing import Any, Mapping, Optional, Tuple, Union, List

from shapely.geometry import Polygon

from utils.helpers import as_vector
from utils.my_exception import ImpossibleToAnswer
from utils.geometry import (OBB_to_eight_points, polygon_area, project_points, external_points_2d)

# set random seed for reproducibility
rng = random.Random(42)

Number = Union[int, float]
WorldState = Mapping[str, Any]
QuestionPayload = Mapping[str, Any]
Answer = Union[int, float, str]


from utils.config import get_config
from PIL import Image
from pathlib import Path


AXIS_TO_NUM = {"X": 0, "Y": 1, "Z": 2}
WORLD_UP_AXIS = get_config()["world_up_axis"]  # 0=X, 1=Y, 2=Z for World Up
WORLD_UP_AXIS_NUM = AXIS_TO_NUM[WORLD_UP_AXIS]


def point_to_plane_distance(point, normal, d):
    """
    Compute the signed distance from a point to a plane.

    Plane: a*x + b*y + c*z = d
    point: (x, y, z)
    normal: (a, b, c)
    """
    point = np.array(point)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # ensure unit normal
    return np.dot(normal, point) - d


def get_position(
    world_state: Mapping[str, Any], object_id: str, timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["objects"][object_id]["obb"][
        "center"
    ]
    return as_vector(current_timestep_involved_object)


def get_position_camera(
    world_state: Mapping[str, Any], timestep: str
) -> Optional[Tuple[float, ...]]:
    timestep_world = world_state["simulation"][timestep]
    current_timestep_involved_object = timestep_world["camera"]["eye"]
    return as_vector(current_timestep_involved_object)


def get_max_height_from_obb(obb: Mapping[str, Any]) -> float:
    """
    Given an oriented bounding box (obb), return the maximum height (z coordinate) of the object.
    """
    center = np.array(obb["center"])
    extents = np.array(obb["extents"])
    axes = np.array(obb["R"])  # 3x3 rotation matrix
    up = np.array([0.0, 0.0, 1.0])

    # Fast path: choose the sign of each extent by the up-dot for each axis
    signs = np.sign(axes.T @ up)  # shape (3,)
    p_high = center + axes @ (signs * extents)

    return p_high[2]  # return the z coordinate since z is up


def get_min_height_from_obb(obb: Mapping[str, Any]) -> float:
    """
    Given an oriented bounding box (obb), return the minimum height (z coordinate) of the object.
    """
    center = np.array(obb["center"])
    extents = np.array(obb["extents"])
    axes = np.array(obb["R"])  # 3x3 rotation matrix
    up = np.array([0.0, 0.0, 1.0])

    # Fast path: choose the sign of each extent by the up-dot for each axis
    signs = -np.sign(axes.T @ up)  # shape (3,)
    p_low = center + axes @ (signs * extents)

    return max(
        0, p_low[2]
    )  # return the z coordinate since z is up # should not be negative


def get_min_distance_pointcloud_to_obb(
    pointcloud: Any, obb: Mapping[str, Any], iso_ratio: float = 3.0, m: int = 5
) -> float:
    """
    Compute the minimum distance from a point cloud to an oriented bounding box (obb).
    """    
    center = np.array(obb["center"])

    cnt, idx, d2 = pointcloud['kd_tree'].search_knn_vector_3d(center, 20)    
    if cnt == 0:
        return float('inf')
    d = np.sqrt(np.asarray(d2))
    m = min(int(m), cnt)
    if d[m - 1] > iso_ratio * d[0]:
        return float(np.median(d[:m]))

    return float(d[0])


def get_closest_object(
    world_state: Mapping[str, Any],
    object_id: str,
    object_position_at_time: List[float],
    timestep: str,
) -> str:
    min_distance = float("inf")
    closest_object = None

    for obj_id, obj_data in world_state["objects"].items():
        if obj_id == object_id:
            continue
        obj_position = get_position(world_state, obj_id, timestep)
        if obj_position is None:
            continue
        distance = np.linalg.norm(
            np.array(object_position_at_time) - np.array(obj_position)
        )
        if distance < min_distance:
            min_distance = distance
            closest_object = obj_data
        
    if closest_object is None:
        raise ImpossibleToAnswer("No other visbile objects found in the scene.")

    return closest_object

def bbox_from_points(uv):
    xymin = uv.min(axis=0)
    xymax = uv.max(axis=0)
    return np.array([xymin[0], xymin[1], xymax[0], xymax[1]])


def get_spatial_relationship_camera_view(obj_1_state, obj_2_state, camera, timestep) -> str:
    

    eight_points_1 = OBB_to_eight_points(obj_1_state["obb"])
    center_1 = obj_1_state["obb"]["center"]
    eight_points_2 = OBB_to_eight_points(obj_2_state["obb"])
    center_2 = obj_2_state["obb"]["center"]

    # use fake photo to load and check projection correcteness
    fake_photo = Image.open(f"/data0/sebastian.cavada/datasets/simulations_v3/dl3dv/random/3/c-1_no-3_d-4_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-9_20251102_063341/render/{str(timestep).zfill(6)}.png")  # dummy image just to get width and height
    numpy_image = np.array(fake_photo)

    project_center_1_uv, z1 = project_points(np.array([center_1]), camera)
    u1, v1 = int(project_center_1_uv[0][0]), int(project_center_1_uv[0][1])
    project_center_2_uv, z2 = project_points(np.array([center_2]), camera)
    u2, v2 = int(project_center_2_uv[0][0]), int(project_center_2_uv[0][1])
    projected_eight_points_1_uv, z1 = project_points(np.array(eight_points_1), camera)    
    hull1 = external_points_2d(projected_eight_points_1_uv)
    projected_eight_points_2_uv, z2 = project_points(np.array(eight_points_2), camera)
    hull2 = external_points_2d(projected_eight_points_2_uv)    

    # add points to the image, red dots for object 1, blue dots for object 2
    for point in hull1:
        u, v = int(point[0]), int(point[1])
        if 0 <= u < numpy_image.shape[1] and 0 <= v < numpy_image.shape[0]:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= v+i < numpy_image.shape[0] and 0 <= u+j < numpy_image.shape[1]:
                        numpy_image[v+i, u+j] = [255, 0, 0]  # red dot

    for point in hull2:
        u, v = int(point[0]), int(point[1])
        if 0 <= u < numpy_image.shape[1] and 0 <= v < numpy_image.shape[0]:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= v+i < numpy_image.shape[0] and 0 <= u+j < numpy_image.shape[1]:
                        numpy_image[v+i, u+j] = [0, 0, 255]  # blue dot

    print("DONE PROJECTION PLOTTING")

    polygon1 = Polygon(hull1)
    polygon2 = Polygon(hull2)
    intersection_polygon = polygon1.intersection(polygon2)
    intersection_area = intersection_polygon.area
    intersection_points = np.array(intersection_polygon.exterior.coords)

    for point in intersection_points:
        u, v = int(point[0]), int(point[1])
        if 0 <= u < numpy_image.shape[1] and 0 <= v < numpy_image.shape[0]:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= v+i < numpy_image.shape[0] and 0 <= u+j < numpy_image.shape[1]:
                        numpy_image[v+i, u+j] = [0, 255, 0]  # green dot for intersection

    print(f"Intersection area between object 1 and object 2 projections: {intersection_area}")

    area_polygon1 = polygon1.area
    area_polygon2 = polygon2.area
    union_area = area_polygon1 + area_polygon2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.

    horizontal = ""
    vertical = ""
    depth = ""

    if iou > 0.6:
        if z1 < z2:
            depth = "in front"
        else:
            depth = "behind"

    if abs(u2 - u1) > abs(v2 - v1):
        if u2 > u1:
            horizontal = "to the right"
        else:
            horizontal = "to the left"
    else:
        if v2 > v1:
            vertical = "below"
        else:
            vertical = "above"


    return "to be implemented"


def get_all_relational_positional_adjectives():
    directions = ["behind", "in front", "to the right", "to the left", "below", "above", \
        "horizontally aligned", "vertically aligned", "same_depth"]
    return directions
