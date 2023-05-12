import os
from tools.frustum_culling import get_grid_culling_pattern
from tools import rendering
import trimesh
import numpy as np
import open3d
import torch
import imageio
import sys
sys.path.append("/")
from datasets.rgbd_dataset import RGBDDataset


def cull_by_bounds(points, scene_bounds):
    eps = 0.02
    inside_mask = np.all(points >= (scene_bounds[0] - eps), axis=1) & np.all(points <= (scene_bounds[1] + eps), axis=1)
    return inside_mask

def cull_mesh(data_dir, mesh_path, save_path, gt_pose=True, 
              remove_missing_depth=True, remove_occlusion=True, 
              scene_bounds=None, subdivide=True, max_edge=0.015,
              chkpt_path=None, silent=False, platform='egl'):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = mesh.vertices
    triangles = mesh.faces
    # print(remove_occlusion)
    if mesh_path.startswith("gt"):
        max_edge = 0.05

    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, max_edge=max_edge, max_iter=10)

    # Cull with the bounding box first
    inside_mask = None
    if scene_bounds is not None:
        inside_mask = cull_by_bounds(vertices, scene_bounds)

    # triangles_in_bounds = []
    # for triangle in triangles:
    #     if inside_mask[triangle[0]] or inside_mask[triangle[1]] or inside_mask[triangle[2]]:
    #         triangles_in_bounds.append(triangle)
    #
    # triangles = np.array(triangles_in_bounds)
    inside_mask = inside_mask[triangles[:, 0]] | inside_mask[triangles[:, 1]] | inside_mask[triangles[:, 2]]
    triangles = triangles[inside_mask, :]

    print("Processed culling by bound")
    os.environ['PYOPENGL_PLATFORM'] = platform
    # load poses
    # state = torch.load(chkpt_path)
    # c2w_tensor = state["poses"]
    # init_gt_c2w = state["gt_poses"][0]
    # align_matrix = init_gt_c2w @ torch.inverse(c2w_tensor[0, :, :])
    # load dataset
    dataset = RGBDDataset(os.path.join(data_dir), trainskip=1, load=False)
    # TODO: optimized poses only work when trainskip=1
    # assert len(dataset) == c2w_tensor.shape[0], "Number of images mismatch!!!"
    H, W, K = dataset.H, dataset.W, dataset.K
    c2w_list = []
    depth_gt_list = []
    step = len(dataset) // 300
    for i, frame_id in enumerate(dataset.frame_ids):
        if i % step != 0:
            continue
        # TODO: should we use gt_poses or estimated poses?
        if gt_pose:
            c2w = np.array(dataset.all_gt_poses[frame_id]).astype(np.float32)
        # else:
        #     c2w = (align_matrix @ c2w_tensor[i, :, :]).detach().cpu().numpy()
        depth_gt = imageio.imread(os.path.join(dataset.basedir, 'depth', dataset.gt_depth_files[frame_id]))
        depth_gt = (np.array(depth_gt) / 1000.0).astype(np.float32)
        c2w_list.append(c2w)
        depth_gt_list.append(depth_gt)
        if not silent:
            print("Load frame: {}".format(i))

    del dataset
    # rendered_depth_maps = rendering.render_depth_maps_doublesided(mesh, c2w_list, H, W, K, far=10.0)
    rendered_depth_maps = rendering.render_depth_maps(mesh, c2w_list, H, W, K, far=10.0)

    # we don't need subdivided mesh to render depth
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()

    # Cull faces
    points = vertices[:, :3]
    # proj = get_projection_matrix(fov, W / H, near=0.01, far=10.0)
    obs_mask, invalid_mask = get_grid_culling_pattern(points, c2w_list, H, W, K,
                                                      rendered_depth_list=rendered_depth_maps,
                                                      depth_gt_list=depth_gt_list,
                                                      remove_missing_depth=remove_missing_depth,
                                                      remove_occlusion=remove_occlusion,
                                                      verbose=silent)
    obs1 = obs_mask[triangles[:, 0]]
    obs2 = obs_mask[triangles[:, 1]]
    obs3 = obs_mask[triangles[:, 2]]
    th1 = 3
    obs_mask = (obs1 > th1) | (obs2 > th1) | (obs3 > th1)
    inv1 = invalid_mask[triangles[:, 0]]
    inv2 = invalid_mask[triangles[:, 1]]
    inv3 = invalid_mask[triangles[:, 2]]
    invalid_mask = (inv1 > 0.7 * obs1) & (inv2 > 0.7 * obs2) & (inv3 > 0.7 * obs3)
    valid_mask = obs_mask & (~invalid_mask)
    triangles_in_frustum = triangles[valid_mask, :]
    # triangles_in_frustum = []
    # for triangle in triangles:
    #     obs1, obs2, obs3 = obs_mask[triangle[0]], obs_mask[triangle[1]], obs_mask[triangle[2]]
    #     if obs1 > 3 or obs2 > 3 or obs3 > 3:
    #         inv1, inv2, inv3 = invalid_mask[triangle[0]], invalid_mask[triangle[1]], invalid_mask[triangle[2]]
    #         if (inv1 > 0.7 * obs1) and (inv2 > 0.7 * obs2) and (inv3 > 0.7 * obs3):
    #             continue
    #         triangles_in_frustum.append(triangle)
    #
    # triangles_in_frustum = np.array(triangles_in_frustum)

    mesh = trimesh.Trimesh(vertices, triangles_in_frustum, process=False)
    mesh.remove_unreferenced_vertices()

    mesh.export(save_path)

def test_rendered_depth(data_dir, chkpt_path, mesh_path, save_path, gt_pose=True, scene_bounds=None, subdivide=True, max_edge=0.015):

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = mesh.vertices
    triangles = mesh.faces
    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, max_edge=max_edge, max_iter=10)

    # Cull with the bounding box first
    inside_mask = None
    if scene_bounds is not None:
        inside_mask = cull_by_bounds(vertices, scene_bounds)

    triangles_in_bounds = []
    for triangle in triangles:
        if inside_mask[triangle[0]] or inside_mask[triangle[1]] or inside_mask[triangle[2]]:
            triangles_in_bounds.append(triangle)

    triangles = np.array(triangles_in_bounds)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()

    dataset = RGBDDataset(os.path.join(data_dir), load=False, device=torch.device("cpu"), new_bound=True)
    H, W, K = dataset.H, dataset.W, dataset.K
    poses = []
    step = len(dataset) // 200
    for i, frame_id in enumerate(dataset.frame_ids):
        if i % step != 0:
            continue
        # TODO: should we use gt_poses or estimated poses?
        pose = np.array(dataset.all_gt_poses[frame_id]).astype(np.float32)
        poses.append(pose)
        print("Load frame: {}".format(i))

    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()
    # depth_maps = rendering.render_depth_maps(mesh, poses, H, W, fov, 10.0)
    # depth_maps = rendering.render_depth_maps_doublesided(mesh, poses, H, W, fov, 10.0)
    depth_maps = rendering.render_depth_maps_doublesided(mesh, poses, H, W, K, 10.0)

    K = open3d.camera.PinholeCameraIntrinsic(640, 480, 554.2562584220408, 554.2562584220408, 319.5, 239.5)
    voxel_length = 0.01
    volume = open3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=0.04,
                                                             color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(depth_maps)):
        rgb = np.ones((int(H), int(W), 3))
        rgb = rgb.astype(np.uint8)
        rgb = open3d.geometry.Image(rgb)
        depth = depth_maps[i]
        depth = open3d.geometry.Image(depth)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0,  depth_trunc=10.0,
                                                                     convert_rgb_to_intensity=False)
        c2w = poses[i]
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        w2c = np.linalg.inv(c2w)
        # requires w2c
        volume.integrate(rgbd, K, w2c)
        print("Processed frame: {}".format(i))

    print("Extract a triangle mesh from the volume and visualize it.")
    cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print("Writing file:", "mesh.ply")
    open3d.io.write_triangle_mesh("mesh_fusion.ply", mesh)

    print("Writing file:", "pcd.ply")
    open3d.io.write_point_cloud("pcd_fusion.ply", cloud)

    # np.savetxt(os.path.join(basedir, "pcd.txt"), cloud)

    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])

    open3d.visualization.draw_geometries([cloud, mesh_frame])
    open3d.visualization.draw_geometries([mesh, mesh_frame])

def get_scene_bound(scene):
    if scene == "whiteroom":
        scene_bounds = np.array([[-2.46, -0.1, 0.36],
                                 [3.06, 3.3, 8.2]])
    elif scene == "kitchen":
        scene_bounds = np.array([[-3.12, -0.1, -3.18],
                                [3.75, 3.3, 5.45]])
    elif scene == "breakfast_room":
        scene_bounds = np.array([[-2.23, -0.5, -1.7],
                                [1.85, 2.77, 3.0]])
    elif scene == "staircase":
        scene_bounds = np.array([[-4.14, -0.1, -5.25],
                                [2.52, 3.43, 1.08]])
    elif scene == "complete_kitchen":
        scene_bounds = np.array([[-5.55, 0.0, -6.45],
                                [3.65, 3.1, 3.5]])
    elif scene == "green_room":
        scene_bounds = np.array([[-2.5, -0.1, 0.4],
                                [5.4, 2.8, 5.0]])
    elif scene == "grey_white_room":
        scene_bounds = np.array([[-0.55, -0.1, -3.75],
                                [5.3, 3.0, 0.65]])
    elif scene == "morning_apartment":
        scene_bounds = np.array([[-1.38, -0.1, -2.2],
                                [2.1, 2.1, 1.75]])
    elif scene == "thin_geometry":
        scene_bounds = np.array([[-2.15, 0.0, 0.0],
                                 [0.77, 0.75, 3.53]])
    elif scene == "icl_living_room":
        scene_bounds = np.array([[-2.5, -0.1, -2.1],
                                 [2.6, 2.7, 3.1]])
    else:
        raise NotImplementedError

    return scene_bounds

if __name__ == '__main__':
    import argparse
    from tools.mesh_metrics import compute_metrics
    parser = argparse.ArgumentParser(
        description='Arguments to cull the mesh.'
    )

    parser.add_argument('--scene', type=str, default="morning_apartment")


    args = parser.parse_args()
    scene = args.scene
    
    'tools'

    data_dir = "./data/neural_rgbd_data/{}".format(scene)
    mesh_path = os.path.join(data_dir, "gt_mesh.ply")
    save_path = os.path.join(data_dir, "gt_mesh_culled_ours.ply")
    scene_bounds = get_scene_bound(scene)
    remove_depth = True
    if 'thin_geometry' in scene or 'staircase' in scene:
        remove_depth = False
    cull_mesh(data_dir, mesh_path, save_path, gt_pose=True, scene_bounds=scene_bounds,
              remove_missing_depth=remove_depth, subdivide=True, max_edge=0.015)
    #gt_mesh_path = os.path.join(args.datadir, 'gt_mesh_culled_ours.ply')
    #rst, meshes = compute_metrics(cull_save_path, gt_mesh_path)
    # test_rendered_depth(data_dir, chkpt_path, mesh_path, save_path, scene_bounds=scene_bounds, subdivide=True, max_edge=0.015)
