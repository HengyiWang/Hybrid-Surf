'''
This file contains the implementation of joint encoding of scene representation
Base model: OneBlob + Hash encoding; Spherical Harmonics + Geometric features
rendering: neuralRGBD
Loss : RGB, depth, sdf, fs, eikonal (Not working very well)

This is the original version of the model, which is used in my thesis
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from optimization.pose_array import PoseArray
from .embedding import get_embedder
from .networks import ColorSDFNet, ColorSDFNet_v2
from .utils import total_variation_loss, sample_pdf, compute_grads, batchify, get_sdf_loss, mse2psnr
from torch.distributions import Categorical


def query_sdf(query_points, embed_fn, sdf_net, return_geo=True, bound=None):
    inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
    if bound is not None:
        inputs_flat = (inputs_flat + bound[:,0]) / (bound[:,1]+ bound[:,0])
    embedded = embed_fn(inputs_flat)
    out = sdf_net(embedded)
    sdf, geo_feat = out[..., :1], out[..., 1:]

    sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
    if not return_geo:
        return sdf
    geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

    return sdf, geo_feat

def query_color(view_dirs, geo_feat, embeddirs_fn, color_net):
    view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])
    geo_feat_flat = torch.reshape(geo_feat, [-1, geo_feat.shape[-1]])
    embedded_dirs = embeddirs_fn(view_dirs_flat)

    input_feat = torch.cat([embedded_dirs, geo_feat_flat], dim=-1)
    rgb_flat = color_net(input_feat)

    rgb = torch.reshape(rgb_flat, list(view_dirs.shape[:-1]) + [rgb_flat.shape[-1]])
    return torch.sigmoid(rgb)

def sdf2weights(sdf, z_vals, args=None):
    weights = torch.sigmoid(sdf / args.truncation) * torch.sigmoid(-sdf / args.truncation)

    signs = sdf[:, 1:] * sdf[:, :-1]
    mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
    inds = torch.argmax(mask, axis=1)
    inds = inds[..., None]
    z_min = torch.gather(z_vals, 1, inds) # The first surface
    mask = torch.where(z_vals < z_min + args.sc_factor * args.truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))

    weights = weights * mask
    return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)

class HashSurface(nn.Module):
    '''
    Positional encoding + Sparse parametric encoding for scene representation
    '''
    def __init__(self, config, poses, nabla=True, num_frames=None):
        super(HashSurface, self).__init__()
        self.config = config
        self.embedpos_fn, self.input_ch_pos = get_embedder(config, multires=config.multires, i_embed=0)
        self.embed_fn, self.input_ch = get_embedder(config, multires=config.multires, i_embed=config.i_embed)
        self.embeddirs_fn, self.input_ch_views = get_embedder(config, config.multires_views, i_embed=config.i_embed_views)
        self.model = ColorSDFNet_v2(config, 
                    input_ch=self.input_ch, 
                    input_ch_views=self.input_ch_views,
                    input_ch_pos=self.input_ch_pos)
        
        self.color_net = batchify(self.model.color_net, None)
        self.sdf_net = batchify(self.model.sdf_net, None)
        self.poses = poses
        self.nabla = nabla

        if config.tcnn_network:
            # TODO: Tiny cuda nn does not seem to be faster
            self.nabla=False

        if num_frames is not None:
            self.pose_array = PoseArray(num_frames)
        else:
            self.pose_array = None
    
    def query_sdf(self, query_points, return_geo=False, embed=False):
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat
    
    def query_color_sdf(self, query_points, view_dir=None):
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        if view_dir is None:
            view_dir = torch.zeros_like(inputs_flat).to(inputs_flat)
        
            view_dir += 1
            view_dir/=2

        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)
        embedded_dirs = self.embeddirs_fn(view_dir)
        return self.model(embed, embe_pos, embedded_dirs=embedded_dirs)

    def run_network(self, inputs, viewdirs, frame_ids, bound=None):
        """Prepares inputs and applies network 'fn'.
        """
        if frame_ids is not None:
            frame_ids = frame_ids.expand(inputs.shape[:-1])
            frame_ids = torch.reshape(frame_ids, [-1]).to(torch.int64)

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        # Transform points to world space only for evaluation
        c2w = self.poses[frame_ids]
        inputs_flat = torch.sum(inputs_flat[..., None, :] * c2w[..., :3, :3], -1) + c2w[..., :3, 3]
            

        # Apply pose correction
        if self.pose_array is not None:
            R = self.pose_array.get_rotation_matrices(frame_ids)
            t = self.pose_array.get_translations(frame_ids)
            inputs_flat = torch.sum(inputs_flat[..., None, :] * R, -1) + t
        
        if self.config.tcnn_encoding:
            # TODO: not exactly 0, 1
            inputs_flat = (inputs_flat - self.config.bounding_box[0]) / (self.config.bounding_box[1] - self.config.bounding_box[0])
            # inputs_flat = (inputs_flat + 2.5) / 5
        

            


        if viewdirs is not None:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

            input_dirs_flat = torch.sum(input_dirs_flat[..., None, :] * c2w[..., :3, :3], -1)
            if self.pose_array is not None:
                input_dirs_flat = torch.sum(input_dirs_flat[..., None, :] * R, -1)
        
            if self.config.tcnn_encoding:
                input_dirs_flat = (input_dirs_flat + 1) / 2


        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat, input_dirs_flat)
        
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        if self.training and self.nabla and self.config.eikonal_weight>0:
            grads = compute_grads(outputs[..., -1], inputs_flat)
            grads = torch.reshape(grads, list(inputs.shape[:-1]) + [3])
        else:
            grads = torch.zeros_like(outputs)
        return outputs, grads

    def render_rays(self, rays_o, rays_d, frame_ids, target_d=None):
        def raw2outputs(raw, z_vals, white_bkgd=False):
            rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
            weights = sdf2weights(raw[..., 3], z_vals, args=self.config)
            rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

            depth_map = torch.sum(weights * z_vals, -1)
            disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
            acc_map = torch.sum(weights, -1)

            if white_bkgd:
                rgb_map = rgb_map + (1.-acc_map[...,None])

            # Calculate weights sparsity loss
            mask = weights.sum(-1) > 0.5
            entropy = Categorical(probs = weights+1e-5).entropy()
            sparsity_loss = entropy * mask

            return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss
        n_rays = rays_o.shape[0]

        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3])

        if target_d is not None:
            z_samples = torch.linspace(-0.1, 0.1, steps=11).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d

            if self.config.n_samples_d > 0:
                z_vals = torch.linspace(self.config.near, self.config.far, self.config.n_samples_d)[None, :].repeat(n_rays, 1).to(rays_o)
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config.near, self.config.far, self.config.n_samples).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        if self.config.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        # (self, inputs, viewdirs, pose_array, frame_ids, c2w_array, bound=None, eval_mode=False)
        # TODO: TCNN compatability bound=None
        raw, grads = self.run_network(pts, viewdirs, frame_ids, bound=None)
        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, self.config.white_bkgd)

        if self.config.n_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, sparsity_loss_0 = rgb_map, disp_map, acc_map, depth_map, sparsity_loss

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config.n_importance, det=(self.config.perturb==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

    #         raw = run_network(pts, fn=run_fn)
            raw, grads = self.run_network(pts, viewdirs, frame_ids, bound=None)
            rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, self.config.white_bkgd)

        ret = {'rgb' : rgb_map, 'depth' :depth_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss, 'grad':grads}
        ret = {**ret, 'z_vals': z_vals}

        ret['raw'] = raw

        if self.config.n_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['sparsity_loss0'] = sparsity_loss_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        # for k in ret:
        #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
        #         print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret
    
    def sample_ray(self, sdfs, N_samples, thresh=1e-3):
        '''
        sdfs: [N_rays]
        '''
        #sdfs[sdfs<1e-3] = 0
        #sdfs = torch.exp(sdfs)

        pdf = sdfs / torch.sum(sdfs)
        mask = torch.multinomial(pdf, N_samples)
        # cdf = torch.cumsum(pdf, 0) 
        # u = torch.rand([N_samples]).to(sdfs)
        # mask = torch.searchsorted(cdf, u) # right=True
        return mask
    
    def select_rays(self, rays_o, rays_d, frame_ids, target_rgb, target_d):
        if not self.training:
            return rays_o, rays_d, frame_ids, target_rgb, target_d
        
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3])
        
        pts = rays_o + rays_d * target_d 
        pts = pts[...,None,:] # [N_rays, 1, 3]
        with torch.no_grad():
            raw, grads = self.run_network(pts, viewdirs, frame_ids, bound=None)

        sdf = raw[..., -1].squeeze() # [N_rays, 1]
        sdf_abs = torch.abs(sdf)
        # sdf_mean = torch.mean(sdf_abs)
        mask = self.sample_ray(sdf_abs, 1024, thresh=1e-3)

        # mask = sdf_abs>sdf_mean

        return rays_o[mask], rays_d[mask], frame_ids[mask], target_rgb[mask], target_d[mask]
    
    def train_select_rays(self, rays_o, rays_d, frame_ids, target_rgb, target_d):
        valid_depth_mask = target_d.squeeze() > 0.
        rays_o = rays_o[valid_depth_mask]
        rays_d = rays_d[valid_depth_mask]
        frame_ids = frame_ids[valid_depth_mask]
        target_rgb = target_rgb[valid_depth_mask]
        target_d = target_d[valid_depth_mask]

        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3])
        
        pts = rays_o + rays_d * target_d 
        pts = pts[...,None,:] # [N_rays, 1, 3]
        with torch.no_grad():
            raw, grads = self.run_network(pts, viewdirs, frame_ids, bound=None)

        sdf = raw[..., -1].squeeze() # [N_rays, 1]
        sdf_abs = torch.abs(sdf)
        sdf_mean = torch.mean(sdf_abs)

        mask = sdf_abs>sdf_mean

        rays_o[mask], rays_d[mask], frame_ids[mask], target_rgb[mask], target_d[mask]
    
    def forward(self, rays, frame_ids, target_rgb, target_d, global_step=0):
        '''
        Params:
            rays: batch rays (2, Bs, 3) rays_o + rays_d
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)

        '''
        rays_o, rays_d = rays

        # After initial training, we only select rays that have high uncertainty
        if global_step > 500:
            rays_o, rays_d, frame_ids, target_rgb, target_d = self.select_rays(rays_o, rays_d, frame_ids, target_rgb, target_d)

        rend_dict = self.render_rays(rays_o, rays_d, frame_ids, target_d=target_d)

        if not self.training:
            return rend_dict

        rgb_loss = F.mse_loss(rend_dict["rgb"], target_rgb)
        psnr = mse2psnr(rgb_loss)


        valid_depth_mask = target_d.squeeze() > 0.
        depth_loss = F.mse_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        sparsity_loss = rend_dict["sparsity_loss"].sum()
        
        z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = rend_dict['raw'][..., -1]

        truncation = self.config.trunc * self.config.sc_factor
        fs_loss, sdf_loss, eikonal_loss = get_sdf_loss(z_vals, target_d, sdf, truncation, self.config.sdf_loss_type, grad=rend_dict['grad'])
        
        if 'rgb0' in rend_dict:
            rgb_loss += F.mse_loss(rend_dict["rgb0"], target_rgb)
            sparsity_loss += rend_dict["sparsity_loss0"].sum()
            depth_loss += F.mse_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        TV_loss = torch.tensor([0])
        if not self.config.tcnn_encoding:
            n_levels = self.embed_fn.n_levels
            min_res = self.embed_fn.base_resolution
            max_res = self.embed_fn.finest_resolution
            log2_hashmap_size = self.embed_fn.log2_hashmap_size
            TV_loss = sum(total_variation_loss(self.embed_fn.embeddings[i], \
                                            min_res, max_res, \
                                            i, log2_hashmap_size, \
                                            n_levels=n_levels) for i in range(n_levels))
        

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "eikonal_loss": eikonal_loss,
            "psnr": psnr,
            "tv_loss": TV_loss
        }

        return ret

        



