'''
Original version, using SH encoding for viewdirs
'''

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import imageio
import parser_utils
import torch
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from model.hash_surf import HashSurface
from torch.utils.data import DataLoader

from optimization.radam import RAdam
from datasets.rgbd_dataset import RGBDRaysDataset
from cull_mesh import cull_mesh, get_scene_bound
from tools.mesh_metrics import compute_metrics

from utils import coordinates, seed_everything, get_rays
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def get_batch_query_fn(query_fn):

    fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :])

    return fn

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def extract_mesh(query_fn, config, voxel_size=0.01, isolevel=0.0, scene_name='', mesh_savepath='', model=None, color=False):
    import trimesh
    import datasets.scene_bounds as scene_bounds
    import marching_cubes as mcubes
    from torch.cuda.amp import autocast as autocast
    volume = config.bounding_box[1] - config.bounding_box[0]
    center = config.bounding_box[0] + volume / 2

    voxel_size *= config.sc_factor  # in "network space"
    x_min, y_min, z_min = config.bounding_box[0]
    x_max, y_max, z_max = config.bounding_box[1]

    tx, ty, tz = scene_bounds.getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size)

    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)

    R = model.pose_array.get_rotation_matrices([0])
    t = model.pose_array.get_translations([0])

    transformation = np.eye(4)
    transformation[:3, :3] = R.cpu().data.numpy().squeeze().T
    transformation[:3, 3] = -t.cpu().data.numpy().squeeze()
    
    
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3]).to(config.bounding_box[0])

    if config.tcnn_encoding:
        flat = (flat - config.bounding_box[0]) / (config.bounding_box[1] - config.bounding_box[0])

    
    if color:
        fn = get_batch_query_fn(model.query_color_sdf)
    else:
        fn = get_batch_query_fn(query_fn)

    chunk = 1024 * 64
    with autocast(False):
        raw = [fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])

    print('Running Marching Cubes')

    if color:
        vertices, triangles = mcubes.marching_cubes(raw[...,-1], isolevel, truncation=3.0)
        rgb= sigmoid(raw[...,:3])
        xyz_min= config.bounding_box[0].cpu().data.numpy()
        verts_ind = np.floor((vertices - xyz_min[None, :])).astype(np.int32)-1
        color_vals = rgb[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]

    else:
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
    if color:
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color_vals)
    else:
        mesh = trimesh.Trimesh(vertices, triangles, process=False)


    if model is not None:
        mesh.apply_transform(transformation)
        

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

def smoothness(args, model, optimizer, sample_points=256):
    volume = args.bounding_box[1] - args.bounding_box[0]
    center = args.bounding_box[0] + volume / 2

    coords = coordinates(sample_points - 1, device, flatten=False).float().to(volume)
    pts = (coords + torch.rand((1,1,1,3)).to(volume)) * volume / sample_points + args.bounding_box[0]

    if args.tcnn_encoding:
        pts_tcnn = (pts - args.bounding_box[0]) / (args.bounding_box[1] - args.bounding_box[0])
    

    sdf = model.query_sdf(pts_tcnn, embed=True)
    tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
    tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
    tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

    return (tv_x + tv_y + tv_z)/ (sample_points**3)

def experiment_setup(args):
    basedir = args.basedir
    if args.init is not None:
        args.expname += args.init
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    args.expname += "_TV" + str(args.tv_loss_weight)
    if args.i_embed==1:
        args.expname += "_hashXYZ"
    elif args.i_embed==0:
        args.expname += "_posXYZ"
    if args.i_embed_views==2:
        args.expname += "_sphereVIEW"
    elif args.i_embed_views==0:
        args.expname += "_posVIEW"
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    args.expname += "_skip"+str(args.trainskip)
    args.expname += "_RAdam"
    if args.sparse_loss_weight > 0:
        args.expname += "_sparse" + str(args.sparse_loss_weight)
    if args.tcnn:
        args.expname += "_TCNN"
    # args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')
    expname = args.expname   
 
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    return basedir, expname

def geometric_init(args, model, sample_points=128, chunk=1024*2048):

    volume = args.bounding_box[1] - args.bounding_box[0]
    center = args.bounding_box[0] + volume / 2
    radius = volume.min() / 2
    print('Volume:', volume)
    print('Center:', center)

    optimizer1 = RAdam([
                            {'params': model.model.parameters(), 'weight_decay': 1e-6},
                            {'params': model.embed_fn.parameters(), 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))

    print('geometric initialisation')
    ckpts = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if 'geomet' in f]
    if len(ckpts) > 0:
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
    else:
        loss = 0
        scaler = GradScaler()
        pbar = tqdm(range(args.geometric_init))
        for _ in pbar:
            optimizer1.zero_grad()
            coords = coordinates(sample_points - 1, device).float().t()
            pts = (coords + torch.rand_like(coords)) * volume / sample_points + args.bounding_box[0]

            if args.tcnn_encoding:
                pts_tcnn = (pts - args.bounding_box[0]) / (args.bounding_box[1] - args.bounding_box[0])

            with autocast(False):
                for i in range(0, pts_tcnn.shape[0], 128*128*128):
                    sdf = model.query_sdf(pts_tcnn[i:i+128*128*128]).squeeze()
                    target_sdf = radius - (center - pts[i:i+128*128*128]).norm(dim=-1)
                    loss = torch.nn.functional.mse_loss(sdf, target_sdf)
                    pbar.set_postfix({'loss': loss.cpu().item()})

                    scaler.scale(loss).backward()
                    scaler.step(optimizer1)
                    scaler.update()
                
                if loss.item() < 2e-5:
                    break

def config_parser():
    return parser_utils.get_parser()

def get_bounding_box(args, rgbd_data):
    bounding_box = rgbd_data.get_bounds() # tensor (3, 2)
    return (bounding_box[:, 0].to(device) - 0.1, bounding_box[:, 1].to(device)+ 0.1) # add a small margin

def train():
    print('Load config')
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    scene = args.expname

    print('Load data')

    rgbd_data = RGBDRaysDataset(args.datadir, 
                                align=True, 
                                trainskip=args.trainskip, 
                                downsample_factor=args.factor, 
                                translation=args.translation, 
                                sc_factor=args.sc_factor, 
                                crop=args.crop, 
                                normals=False)
    
    args.num_training_frames = len(rgbd_data.frame_ids)
    
    rgbd_loader = DataLoader(rgbd_data, num_workers=2, batch_size=args.N_rand)

    args.bounding_box = get_bounding_box(args, rgbd_data)
    print('Bounding box:', args.bounding_box)

    # Create model and optimizer
    poses = torch.stack(rgbd_data.c2w_list).to(device)
    poses_gt = torch.stack(rgbd_data.c2w_gt_list)
    model = HashSurface(args, poses, num_frames=args.num_training_frames).to(device)
    optimizer = RAdam([
                            {'params': model.model.parameters(), 'weight_decay': 1e-6},
                            {'params': model.embed_fn.parameters(), 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    pose_optimizer = RAdam([
                            {'params': model.pose_array.parameters(), 'weight_decay': 1e-6}
                        ], lr=args.lrate/10, betas=(0.9, 0.99))
    stop_pose = False

    # Create log dir and copy the config file
    basedir, expname = experiment_setup(args)

    for params in model.pose_array.parameters():
        params.requires_grad = False

    #scaler = GradScaler()

    # to cuda
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')


    geometric_init(args, model, sample_points=256)

    # mesh_savepath = os.path.join(args.basedir, args.expname, 'init.ply')
    # extract_mesh(model.query_sdf, args,
    #                                   isolevel=0, mesh_savepath=mesh_savepath, model=model,color=False)

    print('Begin')

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    for i, batch in tqdm(enumerate(rgbd_loader)):
        batch_rays = torch.stack([torch.zeros_like(batch[:, :3]), batch[:, :3]], 0).to(device)
        target_s = batch[:, 3:6].to(device)
        target_d = batch[:, 6:7].to(device)
        frame_ids = batch[:, 7:8].to(torch.int64).to(device)

        optimizer.zero_grad()
        model.train()

        with autocast(False):

            ret = model.forward(batch_rays, frame_ids, target_s, target_d, global_step=i)
            
            loss = args.rgb_weight * ret['rgb_loss']+\
            args.depth_weight * ret['depth_loss'] +\
            args.fs_weight * ret["fs_loss"] +\
            args.trunc_weight * ret["sdf_loss"] +\
            args.eikonal_weight * ret["eikonal_loss"] #+\
            #args.tv_loss_weight * ret["tv_loss"]

            if i>1000:
                args.tv_loss_weight = 0.0

            #tv_reg = smoothness(args, model, optimizer, sample_points=128)

            tv_reg = torch.tensor([0])
            #loss += 1e-1 * tv_reg
        
            
        # scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        loss.backward()
        # pdb.set_trace()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()


        if i > 500:
            for params in model.pose_array.parameters():
                params.requires_grad = True
            
            if i % 5 == 0:
                if stop_pose == True:
                    stop_pose = False
                else:
                    stop_pose = True
                    pose_optimizer.step()
                pose_optimizer.zero_grad()

                
        else:
            pose_optimizer.zero_grad()


        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################


        # Rest is logging
        if i%args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': i,
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            
            print('Saved checkpoints at', path)

        if i % args.i_print == 0 or i < 10:
            frame_ids_all = torch.arange(0, args.num_training_frames)
            with torch.no_grad():
                R = model.pose_array.get_rotation_matrices(frame_ids_all)
                t = model.pose_array.get_translations(frame_ids_all)
                R_refine = torch.sum(R[...,None] * model.poses[..., None, :3, :3], dim=2)
                t_refine = model.poses[..., :3, 3]+ t
                R0 = model.pose_array.get_rotation_matrices([0]).permute(0, 2, 1)
                t0 = model.pose_array.get_translations([0])

                R_refine = torch.sum(R0[...,None] * R_refine[..., None, :3, :3], dim=2)
                t_refine = t_refine - t0
                
                dir_cos = torch.einsum('ij,ij->i', poses_gt[:, :3, 2], R_refine[:, :3, 2].cpu())
                dir_cos = torch.clip(dir_cos, 0, 1)
                r_error = torch.rad2deg(torch.mean(torch.arccos(dir_cos)))

                t_error = torch.linalg.norm(poses_gt[..., :3, 3] - t_refine.cpu(), dim=-1).mean()

            print("{}: loss:{:.4e}, FS:{:.4e}, SDF:{:.4e}, Depth:{:.4e}, PSNR:{:.4f} Rot:{:.4f}, Trans:{:.4f} TV{:.4e}, ek{:.4e}".format(
                i, loss.cpu().data.numpy().item(), 
                ret['fs_loss'].cpu().data.numpy().item(), 
                ret['sdf_loss'].cpu().data.numpy().item(), 
                ret['depth_loss'].cpu().data.numpy().item(),
                ret['psnr'].cpu().data.numpy().item(),
                r_error.data.numpy().item(), 
                t_error.data.numpy().item(),
                #ret['tv_loss'].cpu().data.numpy().item(),
                tv_reg.cpu().data.numpy().item(),
                ret['eikonal_loss'].cpu().data.numpy().item()
            ))

            writer.add_scalar('loss', loss, i)
            writer.add_scalar('img_loss', ret['rgb_loss'], i)
            writer.add_scalar('depth_loss', ret['depth_loss'], i)
            writer.add_scalar('free_space_loss', ret['fs_loss'], i)
            writer.add_scalar('sdf_loss', ret['sdf_loss'], i)
            writer.add_scalar('psnr', ret['psnr'], i)
            writer.add_scalar('tv', ret['tv_loss'], i)
            writer.add_scalar('eikonal', ret['eikonal_loss'], i)

        if i % args.i_img == 0 and i > 0:

            def get_logging_images(img_i):
                pose = torch.eye(4, 4)

                render_height = rgbd_data.H // args.render_factor
                render_width = rgbd_data.W // args.render_factor
                render_focal = rgbd_data.focal / args.render_factor

                K = np.array([
                    [render_focal, 0, 0.5*render_width],
                    [0, render_focal, 0.5*render_height],
                    [0, 0, 1]
                ])

                ids = img_i * torch.ones((render_height * render_width, 1), dtype=torch.float32)
                with autocast(False):
                    rays_o, rays_d = get_rays(render_height, render_width, K, pose)
                    batch_rays = torch.stack([rays_o.reshape([-1, 3]), rays_d.reshape([-1, 3])], 0)

                    model.eval()
                    rgb = []
                    depth = []
                    with torch.no_grad():
                        for i in range(0, batch_rays.shape[1], args.chunk):
                            ret = model.forward(batch_rays[:,i:i+args.chunk].to(device), ids[i:i+args.chunk].to(device), None, None)
                        
                            rgb.append(ret['rgb'].detach().cpu())
                            depth.append(ret['depth'].detach().cpu())
                        
                        rgb = torch.cat(rgb, dim=0).reshape([render_height, render_width, 3]).numpy()
                        depth = torch.cat(depth, dim=0).reshape([render_height, render_width, 1]).numpy()

                return rgb, depth

            # Save a rendered training view to disk
            img_i = np.random.choice(args.num_training_frames)
            rgb, depth = get_logging_images(img_i)
            frame_idx = rgbd_data.frame_ids[img_i]

            trainimgdir = os.path.join(basedir, expname, 'tboard_train_imgs')
            os.makedirs(trainimgdir, exist_ok=True)
            imageio.imwrite(os.path.join(trainimgdir, 'rgb_{:06d}_{:04d}.png'.format(i, frame_idx)), to8b(rgb))
            imageio.imwrite(os.path.join(trainimgdir, 'depth_{:06d}_{:04d}.png'.format(i, frame_idx)),
                            to8b(depth / np.max(depth)))

        if i % args.i_mesh == 0 and i > 0:
            network_fn = model
            isolevel = 0.0 if args.mode == 'sdf' else 20.0
            mesh_savepath = os.path.join(args.basedir, args.expname, f'mesh_{i:06}.ply')
            extract_mesh(network_fn.query_sdf, args,
                                      isolevel=isolevel, mesh_savepath=mesh_savepath, model=model,color=False)

            cull_save_path = os.path.join(args.basedir, args.expname, f'cull_mesh_{i:06}.ply')

            # culling mesh
            remove_depth = True
            if 'thin_geometry' in scene or 'staircase' in scene:
                remove_depth = False
            
            cull_mesh(args.datadir, mesh_savepath, cull_save_path, silent=True, 
                      scene_bounds = get_scene_bound(scene), remove_missing_depth=remove_depth)
            gt_mesh_path = os.path.join(args.datadir, 'gt_mesh_culled_ours.ply')
            rst, meshes = compute_metrics(cull_save_path, gt_mesh_path)
            rst['rot'] = r_error.data.numpy().item()
            rst['trans'] = t_error
            print(rst, file=open(os.path.join(args.basedir, args.expname, "output.txt"), "a"))

        if i > args.N_iters:
            break

if __name__=='__main__':

    train()




