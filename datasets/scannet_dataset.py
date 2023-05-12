import os
import cv2
import torch
import imageio
import numpy as np

from torch.utils.data import Dataset
from .utils import get_scene_bounds, get_camera_rays

class ScannetDataset(Dataset):
    def __init__(self,
                 basedir,
                 start: int = -1,
                 end: int = -1,
                 trainskip: int = 1,
                 near: float = 0.01,
                 far: float = 5.0,
                 normal=False,
                 load=True,
                 ):
        super(ScannetDataset).__init__()
        self.basedir = basedir
        self.rgb_dir = os.path.join(basedir, 'color')
        self.depth_dir = os.path.join(basedir, 'depth')
        self.pose_dir = os.path.join(basedir, 'pose') 

        self.rgb_pattern = "{:d}.jpg"
        self.depth_pattern = "{:d}.png"
        self.pose_pattern = "{:d}.txt"
        intri_rgb = np.loadtxt(os.path.join(basedir, "intrinsic/intrinsic_color.txt")).astype(np.float32)[:3, :3]
        intri_depth = np.loadtxt(os.path.join(basedir, "intrinsic/intrinsic_depth.txt")).astype(np.float32)[:3, :3]
        self.intrinsics_rgb = torch.from_numpy(intri_rgb)
        n_frames = len(os.listdir(self.pose_dir))

        self.H = 480
        self.W = 640
        self.intri_depth = intri_depth
        self.intrinsics_depth = torch.from_numpy(intri_depth)
        self.near = near
        self.far = far
        self.norm_rgb = norm_rgb
        self.normal = normal
        self.frame_ids = []
        self.load = load

        if self.load:
            self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F)
            self.normal_estimator = cv2.rgbd.RgbdNormals_create(self.H, self.W, cv2.CV_32F, intri_depth, 5)
        
        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.K_list = []

        for i in range(n_frames):
            if i < start:
                continue

            if i == end:
                break

            if (i - start) % trainskip != 0:
                continue

            c2w = np.loadtxt(os.path.join(self.pose_dir, self.pose_pattern.format(i))).astype(np.float32).reshape(4, 4)
            c2w = torch.from_numpy(c2w)

            if torch.isnan(c2w).any() or torch.isinf(c2w).any():
                continue

            self.frame_ids.append(i)
            self.c2w_list.append(c2w)

        if load:
            self.get_all_frames()
    
    def get_all_frames(self):
        for i, frame_id in enumerate(self.frame_ids):
            rgb_path = os.path.join(self.rgb_dir, self.rgb_pattern.format(frame_id))
            depth_path = os.path.join(self.depth_dir, self.depth_pattern.format(frame_id))

            rgb = np.array(imageio.imread(rgb_path)).astype(np.float32)

            h_orig = rgb.shape[0]
            rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rgb = torch.as_tensor(rgb)
            s = float(h_orig) / float(self.H)
            intri_rgb = self.intrinsics_rgb.clone()
            intri_rgb[0, :] /= s
            intri_rgb[1, :] /= s

            rgb /= 255.
            
            depth = cv2.imread(depth_path, -1).astype(np.float32)
            depth_filtered = self.depth_cleaner.apply(depth) / 1000.

            depth_filtered[depth == 0.] = 0.
            depth_filtered = np.nan_to_num(depth_filtered)
            depth = torch.from_numpy(depth_filtered)
            depth[depth < self.near] = 0.
            depth[depth > self.far] = 0.

            self.rgb_list.append(rgb)
            self.depth_list.append(depth)
            self.K_list.append(self.intrinsics_depth)

        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)


        fx, fy = self.K_list[0, 0, 0], self.K_list[0, 1, 1]
        cx, cy = self.K_list[0, 0, 2], self.K_list[0, 1, 2]
        
        self.ray_d = get_camera_rays(self.H, self.W, fx, fy, cx, cy, type='OpenCV')
    
    def get_frame(self, id):
        ret = {
            "frame_id": self.frame_ids[id],
            "sample_id": id,
            "c2w": self.c2w_list[id],
            "rgb": self.rgb_list[id],
            "depth": self.depth_list[id],
            "K": self.intrinsics_depth,
            'direction': self.ray_d
        }
        return ret
    
    def __len__(self):
        return len(self.frame_ids)
    
    def get_bounds(self):
        # mesh_gt = o3d.io.read_triangle_mesh(self.dataset_dir + "/" + self.dataset_dir.split('/')[-1] + "_vh_clean_2.ply")
        # return torch.from_numpy(compute_scene_bounds(mesh_gt))
        return torch.from_numpy(get_scene_bounds(self.basedir.split('/')[-1])).float()

    def __getitem__(self, id):
        return self.get_frame(id)


class ScannetRaysDataset(ScannetDataset):
    def __init__(self, *args, **kwargs):
        super(ScannetRaysDataset, self).__init__(*args, **kwargs)
        self.load_rays()
    
    def load_rays(self):
        print(self.ray_d.shape)
        print(self.rgb_list.shape)
        print(self.depth_list.shape)
        idx = torch.arange(0,len(self.frame_ids))
        rays = torch.cat([self.ray_d.unsqueeze(0).repeat(len(self.frame_ids),1,1,1),
                          self.rgb_list,
                          self.depth_list[...,None],
                          idx[...,None,None,None].expand(-1, self.H, self.W, 1)
                          ],
                          dim=-1)
        if self.normal:
            rays = torch.cat([rays, self.normal_list], dim=-1)

        
        self.rays = rays.reshape([-1, rays.shape[-1]])


        # Shuffle rays
        print('Shuffle rays')
        indice = torch.randperm(self.rays.shape[0])
        self.rays = self.rays[indice]
    
    def __len__(self):
        return self.rays.shape[0]
    
    def get_rays(self, id):
        return self.rays[id]
    
    def __getitem__(self, id):
        return self.get_rays(id)




    