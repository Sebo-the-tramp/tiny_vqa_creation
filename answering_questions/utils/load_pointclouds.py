import os
import open3d as o3d

SCENE_POINTCLOUD_PATH = "/data0/sebastian.cavada/datasets/gs_dl3dv/dataset_dl3dv/dl3dv/1K"

point_cloud_cache = {}

def load_scene_pointcloud(scene_id):
    global point_cloud_cache
    if scene_id in point_cloud_cache:
        return point_cloud_cache[scene_id]

    pointcloud_file = os.path.join(SCENE_POINTCLOUD_PATH, f"{scene_id}/gs_scene/point_cloud/iteration_30000/point_cloud.ply")
    pcd = o3d.io.read_point_cloud(pointcloud_file)
    point_cloud_cache[scene_id] = {
        "pcd": pcd,
        "kd_tree": o3d.geometry.KDTreeFlann(pcd)
    }

    print("Loaded point cloud for scene:", scene_id)

    return point_cloud_cache[scene_id]