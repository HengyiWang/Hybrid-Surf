import re
import math
import numpy as np
import torch


def tum2matrix(pose):
    """Return homogeneous rotation matrix from quaternion.
    """
    t = pose[:3]
    # under TUM format q is in the order of [x, y, z, w], need change to [w, x, y, z]
    quaternion = [pose[6], pose[3], pose[4], pose[5]]
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(np.float64).eps:
        return np.identity(4)

    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], t[0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], t[1]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], t[2]],
        [0., 0., 0., 1.]])

def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type is  'OpenGL':
        dirs = torch.stack([(i + 0.5 - cx)/fx, -(j + 0.5 - cy)/fy, -torch.ones_like(i)], -1)
    elif type is 'OpenCV':
        dirs = torch.stack([(i + 0.5 - cx)/fx, (j + 0.5 - cy)/fy, torch.ones_like(i)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d

def load_poses(posefile):
    
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if 'nan' in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid

def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def get_scene_bounds(scene_name, new_bound=False):
    if scene_name == 'scene0000_00':
        x_min, x_max = -0.2, 8.6
        y_min, y_max = -0.2, 8.9
        z_min, z_max = -0.2, 3.2

    elif scene_name == 'scene0002_00':
        x_min, x_max = 0.6, 5.0
        y_min, y_max = -0.2, 5.8
        z_min, z_max = 0.0, 3.5

    elif scene_name == 'scene0005_00':
        x_min, x_max = -0.24, 5.55
        y_min, y_max = 0.30, 5.55
        z_min, z_max = -0.24, 2.65

    elif scene_name == 'scene0006_00':
        x_min, x_max = -0.08, 4.13
        y_min, y_max = -0.18, 7.40
        z_min, z_max = -0.06, 2.65

    elif scene_name == 'scene0012_00':
        x_min, x_max = -0.20, 5.60
        y_min, y_max = -0.20, 5.50
        z_min, z_max = -0.20, 2.70

    elif scene_name == 'scene0024_00':
        x_min, x_max = -0.20, 7.38
        y_min, y_max = -0.20, 8.19
        z_min, z_max = -0.20, 2.65

    elif scene_name == 'scene0050_00':
        x_min, x_max = 0.80, 6.70
        y_min, y_max = 0.10, 4.60
        z_min, z_max = -0.20, 2.90

    elif scene_name == 'scene0054_00':
        x_min, x_max = -1.4, 1.4
        y_min, y_max = -0.3, 1.4
        z_min, z_max = -1.4, 1.4

    elif scene_name == 'whiteroom':
        if new_bound:
            # scene_bounds = np.array([[-2.46, -0.1, 0.36],
            #                          [3.06, 3.3, 8.2]])
            x_min, x_max = -2.46, 3.06
            y_min, y_max = -0.3, 3.5
            z_min, z_max = 0.36, 8.2
        else:
            x_min, x_max = -1.2, 1.0
            y_min, y_max = -1.3, 0.9
            z_min, z_max = -0.8, 0.8

    elif scene_name == 'kitchen':
        if new_bound:
            # scene_bounds = np.array([[-3.12, -0.1, -3.18],
            #                          [3.75, 3.3, 5.45]])
            x_min, x_max = -3.20, 3.80
            y_min, y_max = -0.2, 3.20
            z_min, z_max = -3.20, 5.50
        else:
            x_min, x_max = -1.0, 1.4
            y_min, y_max = -1.4, 1.0
            z_min, z_max = -0.8, 1.0

    elif scene_name == 'breakfast_room':
        if new_bound:
            x_min, x_max = -2.23, 1.85
            y_min, y_max = -0.5, 2.77
            z_min, z_max = -1.7, 3.0
        else:
            x_min, x_max = -1.0, 1.0
            y_min, y_max = -1.0, 1.0
            z_min, z_max = -1.0, 1.1
            # x_min, x_max = -1.0, 1.0
            # y_min, y_max = -0.9, 0.9
            # z_min, z_max = -1.0, 1.1

    elif scene_name == 'staircase':
        if new_bound:
            # scene_bounds = np.array([[-4.14, -0.1, -5.25],
            #                          [2.52, 3.43, 1.08]])
            x_min, x_max = -4.20, 2.60
            y_min, y_max = -0.2, 3.5
            z_min, z_max = -5.3, 1.2
        else:
            x_min, x_max = -1.2, 1.1
            y_min, y_max = -1.1, 1.2
            z_min, z_max = -0.8, 1.2

    elif scene_name == 'icl_living_room':
        if new_bound:
            x_min, x_max = -2.6, 2.7
            y_min, y_max = -0.1, 2.8
            z_min, z_max = -2.2, 3.2
        else:
            x_min, x_max = -1.1, 1.1
            y_min, y_max = -1.1, 1.1
            z_min, z_max = -0.6, 0.5

    elif scene_name == 'complete_kitchen':
        if new_bound:
            # scene_bounds = np.array([[-5.55, 0.0, -6.45],
            #                          [3.65, 3.1, 3.5]])
            x_min, x_max = -5.60, 3.70
            y_min, y_max = -0.1, 3.2
            z_min, z_max = -6.50, 3.50
        else:
            x_min, x_max = -1.2, 1.2
            y_min, y_max = -0.9, 0.9
            z_min, z_max = -0.6, 0.6

    elif scene_name == 'green_room':
        if new_bound:
            # scene_bounds = np.array([[-2.5, -0.1, 0.4],
            #                          [5.4, 2.8, 5.0]])
            x_min, x_max = -2.5, 5.5
            y_min, y_max = -0.2, 2.9
            z_min, z_max = 0.3, 5.0
        else:
            x_min, x_max = -0.85, 0.65
            y_min, y_max = -1.1, 1.1
            z_min, z_max = -0.8, 0.6

    elif scene_name == 'grey_white_room':
        if new_bound:
            # scene_bounds = np.array([[-0.55, -0.1, -3.75],
            #                          [5.3, 3.0, 0.65]])
            x_min, x_max = -0.55, 5.3
            y_min, y_max = -0.1, 3.0
            z_min, z_max = -3.75, 0.65
        else:
            x_min, x_max = -0.62, 0.62
            y_min, y_max = -0.83, 0.83
            z_min, z_max = -0.56, 0.6

    elif scene_name == 'morning_apartment':
        if new_bound:
            x_min, x_max = -1.38, 2.10
            y_min, y_max = -0.20, 2.10
            z_min, z_max = -2.20, 1.75
        else:
            x_min, x_max = -0.86, 0.86
            y_min, y_max = -1.0, 0.94
            z_min, z_max = -0.75, 0.75

    elif scene_name == 'thin_geometry':
        if new_bound:
            x_min, x_max = -2.35, 1.00
            y_min, y_max = -0.20, 1.00
            z_min, z_max = 0.20, 3.80
        else:
            x_min, x_max = -0.25, 1.45
            y_min, y_max = 0.1, 1.7
            z_min, z_max = -1.25, 0.0

    else:
        raise NotImplementedError

    return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])