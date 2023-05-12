import json
import numpy as np
import pdb
import torch
import random
import os
import trimesh
import datasets.scene_bounds as scene_bounds
import marching_cubes as mcubes
from torch.cuda.amp import autocast as autocast
from kornia import create_meshgrid

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle


BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    dir_bounds = directions.view(-1, 3)
    # print("Directions ", directions[0,0,:], directions[H-1,0,:], directions[0,W-1,:], directions[H-1, W-1, :])
    # print("Directions ", dir_bounds[0], dir_bounds[W-1], dir_bounds[H*W-W], dir_bounds[H*W-1])

    return directions

def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates (Bs, 8, 3)
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

def get_rays_(directions, c2w, test=False):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate

    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape) # (H, W, 3)

    if test:
        print('rays_o', rays_o)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_bbox3d_for_scanNet(poses, hwf, near=0.0, far=2.0):
    '''
    Params:
        poses: all camera poses
        hwf: intrinsic parameters
    Return:
        bounding_box: tuple (bound_min, bound_min)
    '''
    H, W, focal = hwf
    H, W = int(H), int(W)
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]
    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays_(directions, pose[:3, :4])
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)
    
    bound_min = min(min_bound)
    bound_max = max(max_bound)

    if abs(bound_min) > bound_max:
        bound_arg = abs(bound_min)
    else:
        bound_arg = bound_max
    
    assert bound_arg > 0, 'Bound should be larger than 0'

    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.1]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.1])), bound_arg

def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        #print("ALERT: some points are outside bounding box. Clipping them!")
        #pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution  # The size of each grid
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

    # hashed_voxel_indices = [] # B x 8 ... 000,001,010,011,100,101,110,111
    # for i in [0, 1]:
    #     for j in [0, 1]:
    #         for k in [0, 1]:
    #             vertex_idx = bottom_left_idx + torch.tensor([i,j,k])
    #             # vertex = bottom_left + torch.tensor([i,j,k])*grid_size
    #             hashed_voxel_indices.append(hash(vertex_idx, log2_hashmap_size))

    # BOX_OFFSETS (1, 8, 3)
    # bottom_left_idx.unsqueeze(1) (Bs, 1, 3)
    # voxel_indices (Bs, 8, 3)
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_camera_rays_np(H, W, focal):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------
    dirs = np.stack([(i + 0.5 - W*.5)/focal, -(j + 0.5 - H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = dirs
    return rays_d

def get_rays_rgbd(H, W, focal, poses, images, depth_images, normal=None, shuffle=False, mask=None):
    print('get rays')
    # get_camera_rays_np() returns rays_direction=[H, W, 3]
    # origin is at center, coordinate system is shown above
    # rays: [N, H, W, 3]
    # rays is in camera space
    rays = np.stack([get_camera_rays_np(H, W, focal) for _ in range(poses.shape[0])], 0)  # [N, H, W, 3]
    print('done, concats')

    # Concatenate color and depth
    # rays = view direction (3) + RGB (3) + Depth (1)
    rays = np.concatenate([rays, images], -1)  # [N, H, W, 6]
    rays = np.concatenate([rays, depth_images], -1)  # [N, H, W, 7]

    # Concatenate frame ids
    ids = np.arange(rays.shape[0], dtype=np.float32)
    ids = ids[:, np.newaxis, np.newaxis, np.newaxis]
    ids = np.tile(ids, [1, rays.shape[1], rays.shape[2], 1])

    # rays = view direction (3) + RGB (3) + Depth (1) + Frame id (1)
    rays = np.concatenate([rays, ids], -1)  # [N, H, W, 8]

    if normal is not None:
        rays = np.concatenate([rays, normal], -1)  # [N, H, W, 11]
    
    if mask is not None:
        rays = np.concatenate([rays, mask], -1)  # [N, H, W, 11]


    rays = rays.reshape([-1, rays.shape[-1]])  # [N_rays, 8 or 11]

    if shuffle:
        print('Shuffle rays')
        np.random.shuffle(rays)

    return rays

def coordinates(voxel_dim, device: torch.device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def get_batch_query_fn(query_fn):

    fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :])

    return fn

def extract_mesh(query_fn, config, voxel_size=0.01, isolevel=0.0, scene_name='', mesh_savepath=''):
    # TODO: TCNN encoding mesh extraction need to be changed
    # Query network on dense 3d grid of points
    volume = config.bounding_box[1] - config.bounding_box[0]
    center = config.bounding_box[0] + volume / 2

    voxel_size *= config.sc_factor  # in "network space"
    x_min, y_min, z_min = config.bounding_box[0]
    x_max, y_max, z_max = config.bounding_box[1]

    tx, ty, tz = scene_bounds.getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size)

    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)

    
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3]).to(config.bounding_box[0])

    if config.tcnn_encoding:
        flat = (flat - config.bounding_box[0]) / (config.bounding_box[1] - config.bounding_box[0])

    fn = get_batch_query_fn(query_fn)

    chunk = 1024 * 64
    with autocast(False):
        raw = [fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    

    print('Running Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(raw.squeeze(), isolevel, truncation=3.0)
    print('done', vertices.shape, triangles.shape)

    # voxel_sizes = volume.cpu().numpy() / (np.array([len(tx), len(ty), len(tz)])-1)
    # vertices *= voxel_sizes

    # vertices += center.cpu().numpy()

    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    tx = tx.cpu().data.numpy()
    ty = ty.cpu().data.numpy()
    tz = tz.cpu().data.numpy()
    
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / config.sc_factor - config.translation

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    # Transform the mesh to Scannet's coordinate system
    # gl_to_scannet = np.array([[1, 0, 0, 0],
    #                           [0, 0, -1, 0],
    #                           [0, 1, 0, 0],
    #                           [0, 0, 0, 1]]).astype(np.float32).reshape([4, 4])

    # mesh.apply_transform(gl_to_scannet)

    if mesh_savepath == '':
        mesh_savepath = os.path.join(config.basedir, config.expname, f"mesh_vs{voxel_size / config.sc_factor.ply}")
    mesh.export(mesh_savepath)

    print('Mesh saved')

def rmse(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean(dim=[-1, -2])).mean()

def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

