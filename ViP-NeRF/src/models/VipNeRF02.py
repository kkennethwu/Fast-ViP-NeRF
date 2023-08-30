# Shree KRISHNAya Namaha
# NeRF that supports predicting visibility.
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

import numpy
import torch
import torch.nn.functional as F

from models.tensorBase import MLPRender_Vis, MLPRender_Fea, raw2alpha
from models.tensoRF import TensorVMSplit

class VipNeRF(torch.nn.Module):
    def __init__(self, configs: dict, model_configs: dict):
        super().__init__()
        self.configs = configs
        self.model_configs = model_configs
        self.ndc = self.configs['data_loader']['ndc']
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        self.predict_visibility = self.configs['model']['coarse_mlp']['predict_visibility']# or self.configs['model']['fine_mlp']['predict_visibility']

        self.coarse_model = None
        self.fine_model = None
        self.build_nerf()
        return

    def build_nerf(self):
        if self.coarse_mlp_needed:
            self.coarse_model = MLP(self.configs, self.configs['model']['coarse_mlp'])
            
        if self.fine_mlp_needed:
            self.fine_model = MLP(self.configs, self.configs['model']['fine_mlp'])
        return

    def forward(self, input_batch: dict, retraw: bool = False, sec_views_vis: bool = False):
        if 'common_data' in input_batch.keys():
            # unpack common data
            for key in input_batch['common_data'].keys():
                if isinstance(input_batch['common_data'][key], torch.Tensor):
                    input_batch['common_data'][key] = input_batch['common_data'][key][0]
        ##### _STEP 1: input_batch into "self.render" #####
        render_output_dict = self.render(input_batch, retraw=retraw or self.training, sec_views_vis=sec_views_vis or self.training)
        return render_output_dict

    def render(self, input_dict: dict, retraw: bool, sec_views_vis: bool):
        ##### _STEP2: input_dict into "batchify_rays" #####
        all_ret = self.batchify_rays(input_dict, retraw, sec_views_vis)
        return all_ret

    def batchify_rays(self, input_dict: dict, retraw, sec_views_vis):
        """
        Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        num_rays = input_dict['rays_o'].shape[0]
        chunk = self.configs['model']['chunk']
        for i in range(0, num_rays, chunk):
            render_rays_dict = {}
            for key in input_dict:
                if isinstance(input_dict[key], torch.Tensor) and (input_dict[key].shape[0] == num_rays):
                    render_rays_dict[key] = input_dict[key][i:i+chunk]
                elif isinstance(input_dict[key], numpy.ndarray) and (input_dict[key].shape[0] == num_rays):  # indices
                    render_rays_dict[key] = input_dict[key][i:i+chunk]
                else:
                    render_rays_dict[key] = input_dict[key]

            
            # ret = self.render_rays(rays_flat[i:i+chunk], **kwargs)
            ##### _STEP 3: render_ray_dict into "self.render_rays" #####
            ret = self.render_rays(render_rays_dict, retraw, sec_views_vis)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = self.merge_mini_batch_data(all_ret)
        return all_ret

    def render_rays(self, input_dict: dict, retraw, sec_views_vis):
        rays_o = input_dict['rays_o']
        rays_d = input_dict['rays_d']
        if self.ndc:
            rays_o_ndc = input_dict['rays_o_ndc']
            rays_d_ndc = input_dict['rays_d_ndc']
        if self.configs['model']['coarse_mlp']['use_view_dirs'] or self.configs['model']['fine_mlp']['use_view_dirs']:
            # provide ray directions as input
            view_dirs = input_dict['view_dirs']

        if self.predict_visibility and sec_views_vis:
            if 'rays_o2' in input_dict:
                rays_o2 = input_dict['rays_o2']  # (nr, nf-1, 3)
            else:
                poses = input_dict['common_data']['poses']
                pixel_id = input_dict['pixel_id']
                image_id = pixel_id[:, 0].long()
                num_frames = input_dict['num_frames']
                rays_o2 = []
                for i in range(num_frames-1):
                    other_image_id = i + (i >= image_id).long()
                    poses2i = poses[other_image_id]  # (n, 3, 4)
                    rays_o2i = poses2i[:, :3, 3]  # (n, 3)
                    rays_o2.append(rays_o2i)
                rays_o2 = torch.stack(rays_o2, dim=1)  # (nr, nf-1, 3)

        return_dict = {}
        ##### coarse model #####
        if self.coarse_mlp_needed:
            z_vals_coarse = self.get_z_vals_coarse(input_dict)

            if not self.ndc:
                pts_coarse = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_coarse[...,:,None] # [num_rays, num_samples, 3]
            else:
                pts_coarse = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_coarse[..., :, None]  # [num_rays, num_samples, 3]
            network_input_coarse = {
                'pts': pts_coarse,
            }

            if self.configs['model']['coarse_mlp']['use_view_dirs']:
                network_input_coarse['view_dirs'] = view_dirs

            if self.coarse_model.predict_visibility and sec_views_vis:
                view_dirs2 = self.compute_other_view_dirs(z_vals_coarse, rays_o, rays_d, rays_o2)
                network_input_coarse['view_dirs2'] = view_dirs2
            ##### _STEP 4: input the into MLP ##### 
            outputs_coarse, network_output_coarse = self.run_network(network_input_coarse, self.coarse_model, z_vals_ndc=z_vals_coarse,
                                                       rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d,
                                                       sec_views_vis=sec_views_vis)
            ##### Do volume rendering #####
            # if not self.ndc:
            #     outputs_coarse = self.volume_rendering(network_output_coarse, z_vals=z_vals_coarse, rays_d=rays_d,
            #                                            sec_views_vis=sec_views_vis)
            # else:
            #     outputs_coarse = self.volume_rendering(network_output_coarse, z_vals_ndc=z_vals_coarse,
            #                                            rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d,
            #                                            sec_views_vis=sec_views_vis)
            weights_coarse = outputs_coarse['weights']
            return_dict['z_vals_coarse'] = z_vals_coarse
            for key in outputs_coarse:
                return_dict[f'{key}_coarse'] = outputs_coarse[key]
            if retraw:
                for key in network_output_coarse.keys():
                    return_dict[f'raw_{key}_coarse'] = network_output_coarse[key]
        ##### fine model #####
        if self.fine_mlp_needed:
            z_vals_fine = self.get_z_vals_fine(z_vals_coarse, weights_coarse)
            if not self.ndc:
                pts_fine = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_fine[...,:,None]  # [num_rays, num_samples, 3]
            else:
                pts_fine = rays_o_ndc[..., None, :] + rays_d_ndc[..., None, :] * z_vals_fine[..., :, None]  # [num_rays, num_samples, 3]

            network_input_fine = {
                'pts': pts_fine,
            }
            if self.configs['model']['fine_mlp']['use_view_dirs']:
                network_input_fine['view_dirs'] = view_dirs

            if self.fine_model.predict_visibility and sec_views_vis:
                view_dirs2 = self.compute_other_view_dirs(z_vals_fine, rays_o, rays_d, rays_o2)
                network_input_fine['view_dirs2'] = view_dirs2

            network_output_fine = self.run_network(network_input_fine, self.fine_model)
            if not self.ndc:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals=z_vals_fine, rays_d=rays_d,
                                                     sec_views_vis=sec_views_vis)
            else:
                outputs_fine = self.volume_rendering(network_output_fine, z_vals_ndc=z_vals_fine, rays_d_ndc=rays_d_ndc,
                                                     rays_o=rays_o, rays_d=rays_d, sec_views_vis=sec_views_vis)
            # weights_fine = outputs_fine['weights']

            return_dict['z_vals_fine'] = z_vals_fine
            for key in outputs_fine:
                return_dict[f'{key}_fine'] = outputs_fine[key]
            if retraw:
                for key in network_output_fine.keys():
                    return_dict[f'raw_{key}_fine'] = network_output_fine[key]

        if not retraw:
            if self.coarse_mlp_needed:
                del return_dict['z_vals_coarse'], return_dict['visibility_coarse'], return_dict['weights_coarse']
            if self.fine_mlp_needed:
                del return_dict['z_vals_fine'], return_dict['visibility_fine'], return_dict['weights_fine']
        return return_dict

    def get_z_vals_coarse(self, input_dict: dict):
        num_rays = input_dict['rays_o'].shape[0]
        if not self.ndc:
            near, far = input_dict['near'], input_dict['far']
        else:
            near, far = input_dict['near_ndc'], input_dict['far_ndc']

        perturb = self.configs['model']['perturb']
        if not self.training:
            perturb = False
        lindisp = self.configs['model']['lindisp']

        num_samples_coarse = self.configs['model']['coarse_mlp']['num_samples']
        t_vals = torch.linspace(0., 1., steps=num_samples_coarse).to(input_dict['rays_o'].device)
        if not lindisp:
            z_vals_coarse = near * (1.-t_vals) + far * t_vals
        else:
            z_vals_coarse = 1./(1. / near * (1.-t_vals) + 1. / far * t_vals)

        z_vals_coarse = z_vals_coarse.expand([num_rays, num_samples_coarse])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            upper = torch.cat([mids, z_vals_coarse[..., -1:]], -1)
            lower = torch.cat([z_vals_coarse[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals_coarse.shape).to(input_dict['rays_o'].device)

            z_vals_coarse = lower + (upper - lower) * t_rand
        return z_vals_coarse

    def get_z_vals_fine(self, z_vals_coarse, weights_coarse):
        num_samples_fine = self.configs['model']['fine_mlp']['num_samples']
        perturb = self.configs['model']['perturb']
        if not self.training:
            perturb = False

        z_vals_mid = .5 * (z_vals_coarse[...,1:] + z_vals_coarse[...,:-1])
        z_samples = self.sample_pdf(z_vals_mid, weights_coarse[...,1:-1], num_samples_fine, det=(not perturb))
        z_samples = z_samples.detach()

        z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
        return z_vals_fine

    def compute_other_view_dirs(self, z_vals, rays_o, rays_d, rays_o2):
        if self.ndc:
            near = 1  # TODO: do not hard-code
            tn = -(near + rays_o[..., 2]) / rays_d[..., 2]
            z_vals = (((rays_o[..., None, 2] + tn[..., None] * rays_d[..., None, 2]) / (1 - z_vals + 1e-6)) - rays_o[..., None, 2]) / rays_d[..., None, 2]
        pts = rays_o[..., None, :] + z_vals[..., None] * rays_d[..., None, :]
        view_dirs_other = (pts[:, :, None] - rays_o2[..., None, :, :])  # (nr, ns, nf-1, 3)
        view_dirs_other = view_dirs_other / torch.norm(view_dirs_other, dim=-1, keepdim=True)
        return view_dirs_other

    # Hierarchical sampling (section 5.2)
    @staticmethod
    def sample_pdf(bins, weights, N_samples, det=False):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples).to(weights.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights.device)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    def run_network(self, input_dict, nerf_mlp, z_vals=None, rays_o=None, rays_d=None, z_vals_ndc=None,
                         rays_d_ndc=None, sec_views_vis=False):
        ##### In the MLP, we need to change 'nerf_mlp' into 'TenoRF decomposition' #####
        """
        Prepares inputs and applies network 'nerf_mlp'.
        """
        # pts_flat = torch.reshape(input_dict['pts'], [-1, input_dict['pts'].shape[-1]])
        network_input_dict = {
            'pts': input_dict['pts'],
            'z_vals': z_vals_ndc if self.ndc else z_vals,
        }

        if nerf_mlp.mlp_configs['use_view_dirs']:
            viewdirs = input_dict['view_dirs']
            if viewdirs.ndim == 2:
                viewdirs = viewdirs[:,None].expand(input_dict['pts'].shape)
            # viewdirs_flat = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]])
            network_input_dict['view_dirs'] = viewdirs

            if nerf_mlp.predict_visibility and ('view_dirs2' in input_dict):
                view_dirs2 = input_dict['view_dirs2']
                view_dirs2 = view_dirs2[:, : , 0, :]
                # view_dirs2_flat = torch.reshape(view_dirs2, [-1, view_dirs2.shape[-2], view_dirs2.shape[-1]])  # (nr*ns, nf-1, 3)
                network_input_dict['view_dirs2'] = view_dirs2
        
        # nerf_mlp = nerf_mlp.to(pts_flat.device)
        ##### _STEP 5: input to the nerf_mlp #####
        network_output_dict = self.batchify(nerf_mlp)(network_input_dict)
    
        # for k, v in network_output_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         network_output_dict[k] = torch.reshape(v, list(input_dict['pts'].shape[:-1]) + list(v.shape[1:]))
        #     else:
        #         raise NotImplementedError
            
        ##### Do volume rendering #####
        if not self.ndc:
            outputs_coarse = self.volume_rendering(network_output_dict, z_vals=z_vals, rays_d=rays_d,
                                                   sec_views_vis=sec_views_vis)
        else:
            outputs_coarse = self.volume_rendering(network_output_dict, z_vals_ndc=z_vals_ndc,
                                                   rays_d_ndc=rays_d_ndc, rays_o=rays_o, rays_d=rays_d,
                                                   sec_views_vis=sec_views_vis)
            
        return outputs_coarse, network_output_dict

    def batchify(self, nerf_mlp):
        """Constructs a version of 'nerf_mlp' that applies to smaller batches.
        """
        chunk = self.configs['model']['netchunk']
        if chunk is None:
            return nerf_mlp

        def ret(input_dict: dict):
            num_pts = input_dict['pts'].shape[0]
            network_output_chunks = {}
            for i in range(0, num_pts, chunk):
                network_input_chunk = {}
                for key in input_dict:
                    if isinstance(input_dict[key], torch.Tensor):
                        network_input_chunk[key] = input_dict[key][i:i+chunk]
                    else:
                        raise RuntimeError(key)

                ##### _STEP 6: input chunks into nerf_mlp #####
                network_output_chunk = nerf_mlp(network_input_chunk)
                for k in network_output_chunk.keys():
                    if k not in network_output_chunks:
                        network_output_chunks[k] = []
                    if isinstance(network_output_chunk[k], torch.Tensor):
                        network_output_chunks[k].append(network_output_chunk[k])
                    else:
                        raise RuntimeError
            for k in network_output_chunks:
                if isinstance(network_output_chunks[k][0], torch.Tensor):
                    network_output_chunks[k] = torch.cat(network_output_chunks[k], dim=0)
                else:
                    raise NotImplementedError
            return network_output_chunks
        return ret

    def volume_rendering(self, network_output_dict, z_vals=None, rays_o=None, rays_d=None, z_vals_ndc=None,
                         rays_d_ndc=None, sec_views_vis=False):
        if not self.ndc:
            inf_depth = torch.Tensor([1e10]).to(rays_d.device)
            z_vals1 = torch.cat([z_vals, inf_depth.expand(z_vals[...,:1].shape)], -1)
            z_dists = z_vals1[...,1:] - z_vals1[...,:-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d[...,None,:], dim=-1)
        else:
            inf_depth = torch.Tensor([1]).to(rays_d.device)
            z_vals1 = torch.cat([z_vals_ndc, inf_depth.expand(z_vals_ndc[...,:1].shape)], -1)
            z_dists = z_vals1[...,1:] - z_vals1[...,:-1]  # [N_rays, N_samples]
            delta = z_dists * torch.norm(rays_d_ndc[...,None,:], dim=-1)

        rgb = network_output_dict['rgb']  # [N_rays, N_samples, 3]
        sigma = network_output_dict['sigma']  # [N_rays, N_samples]

        alpha = 1. - torch.exp(-sigma * delta * 25)  # [N_rays, N_samples]

        visibility = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(rays_d.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * visibility
        # app_mask = weights < 0.0001
        # rgb[app_mask] = 0
        rgb_map = torch.sum(weights[...,None] * rgb, dim=-2)  # [N_rays, 3]
        acc_map = torch.sum(weights, dim=-1)

        rgb_map = rgb_map.clamp(min=0., max=1.)
        
        if not self.ndc:
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)
        else:
            depth_map_ndc = torch.sum(weights * z_vals_ndc, dim=-1) / (acc_map + 1e-6)
            depth_var_map_ndc = torch.sum(weights * torch.square(z_vals_ndc - depth_map_ndc[..., None]), dim=-1)
            z_vals = self.convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d)
            depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-6)
            depth_var_map = torch.sum(weights * torch.square(z_vals - depth_map[..., None]), dim=-1)

        if self.configs['model']['white_bkgd']:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return_dict = {
            'rgb': rgb_map,
            'acc': acc_map,
            'alpha': alpha,
            'visibility': visibility,
            'weights': weights,
            'depth': depth_map,
            'depth_var': depth_var_map,
        }

        if self.ndc:
            return_dict['depth_ndc'] = depth_map_ndc
            return_dict['depth_var_ndc'] = depth_var_map_ndc

        if self.predict_visibility and sec_views_vis and ('visibility2' in network_output_dict):
            vis2_point3d = network_output_dict['visibility2']
            vis2_pixel = torch.sum(weights[..., None] * vis2_point3d, dim=-2) / (acc_map[..., None] + 1e-6)  # (nr, nf-1)
            return_dict['visibility2'] = vis2_pixel
        return return_dict

    @staticmethod
    def convert_depth_from_ndc(z_vals_ndc, rays_o, rays_d):
        """
        Converts depth in ndc to actual values
        From ndc write up, t' is z_vals_ndc and t is z_vals.
        t' = 1 - oz / (oz + t * dz)
        t = (oz / dz) * (1 / (1-t') - 1)
        But due to the final trick, oz is shifted. So, the actual oz = oz + tn * dz
        Overall t_act = t + tn = ((oz + tn * dz) / dz) * (1 / (1 - t') - 1) + tn
        """
        near = 1  # TODO: do not hard-code
        oz = rays_o[..., 2:3]
        dz = rays_d[..., 2:3]
        tn = -(near + oz) / dz
        constant = torch.where(z_vals_ndc == 1., 1e-3, 0.)
        # depth = (((oz + tn * dz) / (1 - z_vals_ndc + constant)) - oz) / dz
        depth = (oz + tn * dz) / dz * (1 / (1 - z_vals_ndc + constant) - 1) + tn
        return depth

    @staticmethod
    def merge_mini_batch_data(data_chunks: dict):
        merged_data = {}
        for key in data_chunks:
            if isinstance(data_chunks[key][0], torch.Tensor):
                merged_data[key] = torch.cat(data_chunks[key], dim=0)
            else:
                raise NotImplementedError
        return merged_data

class MLP(torch.nn.Module):
    def __init__(self, configs, mlp_configs):
        """
        """
        super(MLP, self).__init__()
        self.configs = configs
        self.mlp_configs = mlp_configs
        self.view_dep_rgb = self.mlp_configs['view_dependent_rgb']
        self.predict_visibility = self.mlp_configs['predict_visibility']
        self.view_dep_outputs = self.view_dep_rgb or self.predict_visibility
        self.device = "cuda:1"
        #### For TensoRF #####
        density_n_comp = [16, 4, 4]
        app_n_comp = [48, 12, 12]
        gridSize = [400, 400, 400]
        self.aabb = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]).to(self.device)
        kwargs = {'appearance_n_comp' : app_n_comp, 
                  'density_n_comp' : density_n_comp, 
                  'shadingMode' : 'MLP_Vis',
                  'view_pe' : 0,
                  'fea_pe' : 0,
                  'fea2denseAct' : 'relu',}
        
        
        self.tensorf = TensorVMSplit(aabb=self.aabb, gridSize=gridSize, device=self.device, **kwargs)

        return

    def forward(self, input_batch):
        ##### _STEP 7: lots of '3d points' and thers 'view_dir' finally start going into nerf_mlp structure #####
        input_pts = input_batch['pts']
        input_z_vals = input_batch['z_vals']
        dists = torch.cat((input_z_vals[:, 1:] - input_z_vals[:, :-1], torch.zeros_like(input_z_vals[:, :1])), dim=-1)
        output_batch = {}

        # TENSORF IMPLEMENTATION #
        input_pts = self.tensorf.normalize_coord(input_pts)
        
        ########## TODO: decomposite 'input_pts' and then output 'sigma_feature' & 'sigma' ########## 
        view_independent_output_dcit = self.compute_view_independent_output(input_pts)
        output_batch.update(view_independent_output_dcit)
        
        # if not self.view_dep_rgb:
        #     rgb = pts_outputs['rgb_view_independent']
        ########## TODO: decomposite 'input_pts' and output 'sigma_feature' ########## 
        alpha, weight, bg_weight = raw2alpha(output_batch['sigma'], dists * 25)

        app_mask = weight > 0.0001

        app_features = self.tensorf.compute_appfeature(input_pts[app_mask])
        
        ########## TODO: input "app_feature" and "primary_viewing_dir" to MLP ##########
        primary_view = input_batch["view_dirs"]
        primary_output_dict = self.compute_view_dependent_output(input_pts, primary_view, app_features, app_mask=app_mask)
        output_batch.update(primary_output_dict)

        if self.view_dep_rgb: 
            rgb = primary_output_dict['rgb_view_dependent']
        output_batch['rgb'] = rgb
        
        ########## TODO: input "app_feature" and "secondary_viewing_dir" to MLP ##########
        if 'view_dirs2' in input_batch.keys():
            secondary_view = input_batch['view_dirs2']
            secondary_output_dict = self.compute_view_dependent_output(input_pts, secondary_view, app_features, app_mask=app_mask)
            output_batch['visibility2'] = secondary_output_dict['visibility']
        
        
        if 'feature' in output_batch:
            del output_batch['feature']
        return output_batch


    def compute_view_independent_output(self, input_pts):
        output_dict = {}
        sigma = torch.zeros(input_pts.shape[:-1], device=self.device)
        mask_outbbox = ~((self.aabb[0] > input_pts) | (input_pts > self.aabb[1])).any(dim=-1)
        sigma_feature = self.tensorf.compute_densityfeature(input_pts[mask_outbbox])


        valid_sigma = self.tensorf.feature2density(sigma_feature)
        sigma[mask_outbbox] = valid_sigma
        output_dict['sigma'] = sigma

        return output_dict
        
    
    def compute_view_dependent_output(self, input_pts, view_dir, app_features, app_mask=None):
        output_dict = {}
        rgb = torch.zeros((*input_pts.shape[:2], 3), device=input_pts.device)
        visibility = torch.zeros((*input_pts.shape[:2], 1), device=input_pts.device)
        # if view_dir.ndim == 4:
            # For viewdirs2
            # nf = view_dir.shape[2] + 1
            # app_features = app_features[:, None, :].repeat([1, nf-1, 1])  # (nc, nf-1, cv)

        
        rgb[app_mask], visibility[app_mask] = self.tensorf.renderModule(input_pts[app_mask], view_dir[app_mask], app_features)
        if self.view_dep_rgb:
            output_dict['rgb_view_dependent'] = rgb
        if self.predict_visibility:           
            output_dict['visibility'] = visibility

        return output_dict
