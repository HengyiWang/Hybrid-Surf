import numpy as np
import math
from copy import deepcopy

import torch


def focal_to_fov(focal, height):
    return math.degrees(2.0 * math.atan(height / (2.0 * focal)))

def fov_to_focal(fov, height):
    return height / (2.0 * math.tan(math.radians(0.5 * fov)))

def get_projection_matrix(fov, aspect, near, far):
    f = 1.0 / math.tan(0.5 * math.radians(fov))

    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ])

def cull_from_one_pose(points, pose, H, W, K, rendered_depth=None, depth_gt=None, remove_missing_depth=True, remove_occlusion=True):
    # rotation = np.transpose(pose[:3, :3])
    # translation = -pose[:3, 3:4]
    # shouldn't be this?

    c2w = deepcopy(pose)
    # to OpenCV
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: inside frustum
    in_frustum = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)
    eps = 0.02
    obs_mask = in_frustum
    # step 2: not occluded
    if remove_occlusion:
        obs_mask = in_frustum & (pz < (rendered_depth[v, u] + eps))  # & (depth_gt[v, u] > 0.)

    # step 3: valid depth in gt
    if remove_missing_depth:
        invalid_mask = in_frustum & (depth_gt[v, u] <= 0.)
    else:
        invalid_mask = np.zeros_like(obs_mask)

    return obs_mask.astype(np.int32), invalid_mask.astype(np.int32)


def get_grid_culling_pattern(points, poses, H, W, K, rendered_depth_list=None, depth_gt_list=None, remove_missing_depth=True, remove_occlusion=True, verbose=False):

    obs_mask = np.zeros(points.shape[0])
    invalid_mask = np.zeros(points.shape[0])
    for i, pose in enumerate(poses):
        if verbose:
            print('Processing pose ' + str(i + 1) + ' out of ' + str(len(poses)))
        rendered_depth = rendered_depth_list[i] if rendered_depth_list is not None else None
        depth_gt = depth_gt_list[i] if depth_gt_list is not None else None
        obs, invalid = cull_from_one_pose(points, pose, H, W, K, rendered_depth=rendered_depth, depth_gt=depth_gt, remove_missing_depth=remove_missing_depth, remove_occlusion=remove_occlusion)
        obs_mask = obs_mask + obs
        invalid_mask = invalid_mask + invalid

    return obs_mask, invalid_mask

