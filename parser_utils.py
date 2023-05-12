import configargparse

def get_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_iters", type=int, default=1000000,
                        help='number of iterations for which to train the network')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    
    parser.add_argument("--rgb_weight", type=float,
                        default=1.0, help='weight of the img loss')
   
                        
    parser.add_argument("--depth_weight", type=float,
                        default=1.0, help='weight of the depth loss')
    parser.add_argument("--fs_weight", type=float,
                        default=1.0, help='weight of the free-space loss')
    parser.add_argument("--trunc_weight", type=float,
                        default=1.0, help='weight of the truncation loss')
    parser.add_argument("--share_coarse_fine", action='store_true',
                        help='use the same network for both coarse and fine samples')
    parser.add_argument("--rgb_loss_type", type=str, default='l2',
                        help='which RGB loss to use - l1/l2 are currently supported')
    parser.add_argument("--sdf_loss_type", type=str, default='l2',
                        help='which SDF loss to use - l1/l2 are currently supported')
    parser.add_argument("--frame_features", type=int, default=0,
                        help='number of channels of the learnable per-frame features')
    parser.add_argument("--optimize_poses", action='store_true',
                        help='optimize a pose refinement for the initial poses')
    parser.add_argument("--use_deformation_field", action='store_true',
                        help='use a deformation field to account for inaccuracies in intrinsic parameters')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1, 
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2, 
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--mode", type=str, default='sdf',
                        help='whether the network predicts density or SDF values')
    parser.add_argument("--trunc", type=float, default=0.05,
                        help='length of the truncation region in meters')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    parser.add_argument("--trainskip", type=int, default=1,
                        help='will load 1/N images from the training set, useful for large datasets like deepvoxels')
    parser.add_argument("--factor", type=int, default=1,
                        help='downsample factor for depth images')
    parser.add_argument("--sc_factor", type=float, default=1.0,
                        help='factor by which to scale the camera translation and the depth maps')
    parser.add_argument("--translation", action="append", default=None, required=False, type=float,
                        help='translation vector for the camera poses')
    parser.add_argument("--crop", type=int, default=0,
                        help='number of pixels by which to crop the image edges (e.g. due to undistortion artifacts')
    parser.add_argument("--near", type=float, default=0.0, help='distance to the near plane')
    parser.add_argument("--far", type=float, default=1.0, help='distance to the far plane')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_mesh", type=int, default=200000,
                        help='frequency of mesh extraction')

    parser.add_argument("--finest_res",   type=int, default=512, 
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19, 
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    parser.add_argument("--tcnn", action='store_true', help='Using tiny cuda nn')
    parser.add_argument("--tcnn_encoding", action='store_true', help='Using tiny cuda nn')
    parser.add_argument("--tcnn_network", action='store_true', help='Using tiny cuda nn')

    parser.add_argument("--init", type=str, default=None, help='Grid intialisation')
    parser.add_argument("--init_grid", type=str, default=None, help='Grid intialisation')
    parser.add_argument("--num_layers",   type=int, default=2, 
                        help='num of layers for SDF net')
    parser.add_argument("--hidden_dim",   type=int, default=64, 
                        help='num of hidden dims')
    parser.add_argument("--geo_feat_dim",   type=int, default=15, 
                        help='num of geometric features')
    parser.add_argument("--num_layers_color",   type=int, default=2, 
                        help='num of layers for color net')
    parser.add_argument("--hidden_dim_color",   type=int, default=64, 
                        help='num of hidden dims')
    # rendering config
    parser.add_argument("--n_samples", type=int, default=128, 
                        help='number of coarse samples per ray')
    parser.add_argument("--n_importance", type=int, default=36,
                        help='number of additional fine samples per ray')
    parser.add_argument("--truncation", type=float, default=0.05,
                        help='length of the truncation region in meters')
    parser.add_argument("--geometric_init", type=int, default=1000, 
                        help='iterations for training the model')
    parser.add_argument("--eikonal_weight", type=float, default=0, 
                        help='weight for eikonal loss')
                        
    parser.add_argument("--normal_weight", type=float, default=0.01, 
                        help='weight for normal loss')
    parser.add_argument("--n_samples_d", type=int, default=128, 
                        help='Depth sampling')

    parser.add_argument("--base_resolution", type=int, default=16, 
                        help='base_resolution')
                        

                        

                        

 
    return parser