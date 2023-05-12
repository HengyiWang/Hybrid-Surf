import os
import cv2
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset
from .utils import alphanum_key,load_poses,load_focal_length,get_scene_bounds,get_camera_rays

class RGBDDataset(Dataset):
    def __init__(self, basedir, align=True, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0, normals=False, load=True, world_coordinate=False):
        super(RGBDDataset).__init__()

        # Config
        self.basedir = basedir
        self.align = align
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.normals = normals
        self.world_coordinate = world_coordinate

        # Files 
        # Get image filenames, poses and intrinsics
        self.img_files = [f for f in sorted(os.listdir(os.path.join(self.basedir, 'images')), key=alphanum_key) if f.endswith('png')]
        self.depth_files = [f for f in sorted(os.listdir(os.path.join(self.basedir, 'depth_filtered')), key=alphanum_key) if f.endswith('png')]
        self.gt_depth_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'depth')), key=alphanum_key) if f.endswith('png')]

        # If Pose is NaN, then valid=false, initialise as 4x4 identity matrix
        self.all_poses, valid_poses = load_poses(os.path.join(self.basedir, 'trainval_poses.txt'))
        self.all_gt_poses, valid_gt_poses = load_poses(os.path.join(basedir, 'poses.txt'))

        # get transformation between first bundle-fusion pose and first gt pose
        init_pose = np.array(self.all_poses[0]).astype(np.float32)
        init_gt_pose = np.array(self.all_gt_poses[0]).astype(np.float32)
        self.align_matrix = init_gt_pose @ np.linalg.inv(init_pose)

        depth = imageio.imread(os.path.join(self.basedir, 'depth_filtered', self.depth_files[0]))
        self.H, self.W = depth.shape[:2]
        focal = load_focal_length(os.path.join(self.basedir, 'focal.txt'))
        self.K = np.array([[focal, 0., (self.W - 1) / 2],
                          [0., focal, (self.H - 1) / 2],
                          [0., 0., 1.]])
        
        if self.normals:
            self.normal_estimator = cv2.rgbd.RgbdNormals_create(self.H, self.W, cv2.CV_32F, self.K, 5)
        
        # Train, val and test split
        num_frames = len(self.img_files)
        train_frame_ids = list(range(0, num_frames, trainskip))

        self.frame_ids = []
        for id in train_frame_ids:
            if valid_poses[id]:
                self.frame_ids.append(id)
        
        self.c2w_gt_list = []
        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.depth_gt_list = []
        self.K_list = []
        self.normal_list = []

        if load:
            self.load_data()

    
    def load_data(self):
        for i, frame_id in enumerate(self.frame_ids):
            c2w_gt = np.array(self.all_gt_poses[frame_id]).astype(np.float32)
            c2w_gt = torch.from_numpy(c2w_gt)
            c2w = np.array(self.all_poses[frame_id]).astype(np.float32)
            if self.align:
                c2w = self.align_matrix @ c2w
                c2w[:3, 3] *= self.sc_factor
                
            else:
                c2w[:3, 3] += self.translation
                c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w)
            rgb = imageio.imread(os.path.join(self.basedir, 'images', self.img_files[frame_id]))
            depth = imageio.imread(os.path.join(self.basedir, 'depth_filtered', self.depth_files[frame_id]))
            H, W = depth.shape[:2]
            focal = load_focal_length(os.path.join(self.basedir, 'focal.txt'))
            rgb = (np.array(rgb) / 255.).astype(np.float32)
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            depth = (np.array(depth) / 1000.0).astype(np.float32)  * self.sc_factor

            # load normal


            if self.normals:
                pts_3d = cv2.rgbd.depthTo3d(depth, self.K)
                normals_np = self.normal_estimator.apply(pts_3d)
                normals_np = np.nan_to_num(normals_np)
                normals = torch.from_numpy(normals_np)
                normals[1:, ...] *= -1  # convert to OpenGL convention

            # load gt_depth
            depth_gt = imageio.imread(os.path.join(self.basedir, 'depth', self.gt_depth_files[frame_id]))
            depth_gt = (np.array(depth_gt) / 1000.0).astype(np.float32)  * self.sc_factor

            # Crop the undistortion artifacts
            if self.crop > 0:
                rgb = rgb[:, self.crop:-self.crop, self.crop:-self.crop, :]
                depth = depth[:, self.crop:-self.crop, self.crop:-self.crop, :]
                depth_gt = depth_gt[:, self.crop:-self.crop, self.crop:-self.crop, :]
                H, W = depth.shape[:2]

            if self.downsample_factor > 1:
                H = H // self.downsample_factor
                W = W // self.downsample_factor
                focal = focal // self.downsample_factor
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)

            rgb = torch.from_numpy(rgb)
            depth = torch.from_numpy(depth)
            depth_gt = torch.from_numpy(depth_gt)
            K = torch.tensor([[focal, 0., (W - 1) / 2],
                              [0., focal, (H - 1) / 2],
                              [0., 0., 1.]], dtype=torch.float32)

            self.c2w_gt_list.append(c2w_gt)
            self.c2w_list.append(c2w)
            self.rgb_list.append(rgb)
            self.depth_list.append(depth)
            self.K_list.append(K)
            self.depth_gt_list.append(depth_gt)

            if self.normals:
                self.normal_list.append(normals)
            
        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)
        self.depth_gt_list = torch.stack(self.depth_gt_list, dim=0)

        if self.normals:
            self.normal_list = torch.stack(self.normal_list, dim=0)

        self.ray_d = get_camera_rays(H, W, focal)

        self.H = self.H
        self.W = self.W
        self.focal = focal
    

    def get_frame(self, id):
        ret = {
            "frame_id": self.frame_ids[id],
            "sample_id": torch.tensor([id])[...,None,None,None].expand(-1, self.H, self.W, 1),
            "c2w": self.c2w_list[id],
            "c2w_gt": self.c2w_gt_list[id],
            "rgb": self.rgb_list[id],
            "depth": self.depth_list[id],
            "depth_gt": self.depth_gt_list[id],
            "K": self.K_list[id],
            'direction': self.ray_d
        }

        if self.normals:
            ret['normal'] = self.normal_list[id]
        
        if self.world_coordinate:
            pos_cam = self.ray_d * self.depth_list[id][..., None] # H, W, 3

            # print('self.ray_d', self.ray_d.shape)
            # print('self.depth_list[id][..., None]', self.depth_list[id][..., None].shape)
            # print('pos_cam', pos_cam.shape)
            pos_world = torch.sum(pos_cam[..., None, :] * self.c2w_list[id][None, None, :3, :3], -1) + self.c2w_list[id][None, None, :3, 3]
            ret['position'] = pos_world[self.depth_list[id] > 0]






        return ret

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, id):
        return self.get_frame(id)
    
    def get_bounds(self):
        new_bound = self.align and self.sc_factor==1
        return torch.from_numpy(get_scene_bounds(self.basedir.split('/')[-1], new_bound)).float()
    
class RGBDRaysDataset(RGBDDataset):
    def __init__(self, *args, **kwargs):
        super(RGBDRaysDataset, self).__init__(*args, **kwargs)
        self.load_rays()


    def load_rays(self):
        idx = torch.arange(0,len(self.frame_ids))
        rays = torch.cat([self.ray_d.unsqueeze(0).repeat(len(self.frame_ids),1,1,1),
                          self.rgb_list,
                          self.depth_list[...,None],
                          idx[...,None,None,None].expand(-1, self.H, self.W, 1)
                          ],
                          dim=-1)
        if self.normals:
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

    

        

    

    
