import os
import numpy as np
import imageio
import torch
import cv2

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .utils import tum2matrix, get_camera_rays, alphanum_key

class iPhoneDataset(Dataset):
    def __init__(self, basedir, trainskip=1, near=0, far=2,
                 downsample_factor=1, normals=False, load=True, world_coordinate=False):
        super(iPhoneDataset).__init__()

        # Config
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.normals = normals
        self.world_coordinate = world_coordinate
        self.bound = None
        self.near = near
        self.far = far

        self.poses = self.load_pose(basedir)

        # Files 
        # Get image filenames, poses and intrinsics
        self.video_path = os.path.join(self.basedir, 'rgb.mp4')
        if not os.path.exists(os.path.join(basedir, 'images')):
            os.makedirs(os.path.join(basedir, 'images'))
            self.process_video()
        
        self.img_files = [f for f in sorted(os.listdir(os.path.join(self.basedir, 'images')), key=alphanum_key) if f.endswith('png')]
        self.depth_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'depth')), key=alphanum_key) if f.endswith('png')]

        self.K = self.get_intrinsics()
        
        if self.normals:
            self.normal_estimator = cv2.rgbd.RgbdNormals_create(self.H, self.W, cv2.CV_32F, self.K, 5)
        
        # Train, val and test split
        num_frames = len(self.img_files)
        train_frame_ids = list(range(0, num_frames, trainskip))

        self.frame_ids = []
        for id in train_frame_ids:
            self.frame_ids.append(id)
        
        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.K_list = []
        self.normal_list = []

        if load:
            self.load_data()

    def _resize_camera_matrix(self, camera_matrix, scale_x, scale_y):
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        return np.array([[fx * scale_x, 0.0, cx * scale_x],
            [0., fy * scale_y, cy * scale_y],
            [0., 0., 1.0]])
    
    def get_intrinsics(self):
        """
        Scales the intrinsics matrix to be of the appropriate scale for the depth maps.
        """
        depth = imageio.imread(os.path.join(self.basedir, 'depth', self.depth_files[0]))
        H, W = depth.shape[:2]
        intrinsics = np.loadtxt(os.path.join(self.basedir, 'camera_matrix.csv'), delimiter=',')
        intrinsics_scaled = self._resize_camera_matrix(intrinsics, W / 1920, H / 1440)
        return intrinsics_scaled
    
    def qTomatrix(self, pose):
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        T_WC[:3, 3] = pose[:3]

        return T_WC
    
    def load_pose(self, basedir):
        path_to_pose = os.path.join(basedir, 'odometry.csv')
        pose_data = np.loadtxt(path_to_pose, delimiter=',', skiprows=1)
        poses = [self.qTomatrix(pose_data[i][2:]) for i in range(pose_data.shape[0])]

        return poses
    
    def process_video(self):
        print('processing video')
        # frames = imageio.v3.imread(self.video_path, plugin="pyav")
        # for i, frame in enumerate(frames):
        #     imageio.imsave(os.path.join(self.basedir, 'images', "{:06d}.png".format(i)), frame) 
        vidcap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        print('num_frames:', num_frames)
        while(frame_count < num_frames):
            success,image = vidcap.read()
            cv2.imwrite(os.path.join(self.basedir, 'images', "{:06d}.png".format(frame_count)), image)     # save frame as JPEG file      
            frame_count += 1
        
        print('processing video... done!')
  
    def load_data(self):
        for i, frame_id in tqdm(enumerate(self.frame_ids)):
            c2w = np.array(self.poses[frame_id]).astype(np.float32)
            
            c2w = torch.from_numpy(c2w)
            # rgb = imageio.v3.imread(
            #     self.video_path,
            #     index=frame_id,
            #     plugin="pyav",
            #     )
            rgb = imageio.imread(os.path.join(self.basedir, 'images', self.img_files[frame_id]))
            depth = imageio.imread(os.path.join(self.basedir, 'depth', self.depth_files[frame_id]))
            H, W = depth.shape[:2]
            rgb = (np.array(rgb) / 255.).astype(np.float32)
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            depth = (np.array(depth) / 1000.0).astype(np.float32)

            K = self.K


            if self.normals:
                pts_3d = cv2.rgbd.depthTo3d(depth, self.K)
                normals_np = self.normal_estimator.apply(pts_3d)
                normals_np = np.nan_to_num(normals_np)
                normals = torch.from_numpy(normals_np)
                normals[1:, ...] *= -1  # convert to OpenGL convention


            if self.downsample_factor > 1:
                H = H // self.downsample_factor
                W = W // self.downsample_factor
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                K = self.K // self.downsample_factor

            rgb = torch.from_numpy(rgb)
            depth = torch.from_numpy(depth)

            self.c2w_list.append(c2w)
            self.rgb_list.append(rgb)
            self.depth_list.append(depth)
            self.K_list.append(torch.from_numpy(K))

            if self.normals:
                self.normal_list.append(normals)
            
        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)

        if self.normals:
            self.normal_list = torch.stack(self.normal_list, dim=0)

        fx, fy = self.K_list[0, 0, 0], self.K_list[0, 1, 1]
        cx, cy = self.K_list[0, 0, 2], self.K_list[0, 1, 2]
        print(fx, fy, cx, cy)
        
        self.ray_d = get_camera_rays(H, W, fx, fy, cx, cy, type='OpenCV')

        self.H = H
        self.W = W

        print('Finish loading')
    
    def get_frame(self, id):
        ret = {
            "frame_id": self.frame_ids[id],
            "sample_id": torch.tensor(id),
            "c2w": self.c2w_list[id],
            "rgb": self.rgb_list[id],
            "depth": self.depth_list[id],
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
        if self.bound is None:
            self.bound = torch.from_numpy(self.get_bbox3d_for_iphone(self.near, self.far)).float()

        return self.bound
    
    def get_bbox3d_for_iphone(self, near=0.0, far=5.0):
        '''
        Params:
            poses: all camera poses
            hwf: intrinsic parameters
        Return:
            bounding_box: tuple (bound_min, bound_min)
        '''

        min_bound = [100, 100, 100]
        max_bound = [-100, -100, -100]
        points = []
        poses = torch.FloatTensor(self.poses)
        for pose in poses:
            rays_d = torch.sum(self.ray_d[..., None, :] * pose[:3,:3], -1).reshape((-1, 3))
            rays_o = pose[:3,-1].expand(rays_d.shape)
            
            def find_min_max(pt):
                for i in range(3):
                    if(min_bound[i] > pt[i]):
                        min_bound[i] = pt[i]
                    if(max_bound[i] < pt[i]):
                        max_bound[i] = pt[i]
                return

            for i in [0, self.W-1, self.H*self.W-self.W, self.H*self.W-1]:
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

        return np.array([[min_bound[0], max_bound[0]], [min_bound[1], max_bound[1]], [min_bound[2], max_bound[2]]])


class iPhoneRaysDataset(iPhoneDataset):
    def __init__(self, *args, **kwargs):
        super(iPhoneRaysDataset, self).__init__(*args, **kwargs)
        self.load_rays()
    
    def get_all_rays(self):
        idx = torch.arange(0,len(self.frame_ids))
        rays = torch.cat([self.ray_d.unsqueeze(0).repeat(len(self.frame_ids),1,1,1),
                          self.rgb_list,
                          self.depth_list[...,None],
                          idx[...,None,None,None].expand(-1, self.H, self.W, 1)
                          ],
                          dim=-1)
        return rays

    def get_one_frame_ray(self, id):
        rays = self.get_all_rays()
        rays = rays[id:id+1]
        return rays.reshape([-1, rays.shape[-1]])

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

    

        