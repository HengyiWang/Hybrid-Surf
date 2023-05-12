import pyrender
import math
import numpy as np
from copy import deepcopy


def render_depth_maps(mesh, poses, H, W, K, yfov=60.0, far=2.0):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    # camera = pyrender.PerspectiveCamera(yfov=math.radians(yfov), aspectRatio=W/H, znear=0.01, zfar=far)
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for pose in poses:
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)
    

    return depth_maps


# For meshes with backward-facing faces. For some reasong the no_culling flag in pyrender doesn't work for depth maps
def render_depth_maps_doublesided(mesh, poses, H, W, K, yfov=60.0, far=10.0):
    # depth_maps_1 = render_depth_maps(mesh, poses, H, W, yfov, far)
    depth_maps_1 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    # depth_maps_2 = render_depth_maps(mesh, poses, H, W, yfov, far)
    depth_maps_2 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in range(len(depth_maps_1)):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where((depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map), depth_maps_2[i], depth_map)
        depth_maps.append(depth_map)

    return depth_maps
