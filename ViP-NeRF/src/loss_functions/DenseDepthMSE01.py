# Shree KRISHNAya Namaha
# MSE loss function on dense depth
# Author: Nagabhushan S N
# Last Modified: 29/03/2023

from pathlib import Path

import torch

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DenseDepthMSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        total_loss = 0
        loss_maps = {}

        if 'dense_depth_values' not in input_dict:
            return {'loss_value': torch.Tensor([0]).to(input_dict['rays_o']).mean()}
        
        indices_mask = input_dict['indices_mask_nerf']
        gt_depth = input_dict['dense_depth_values'][:, 0]

        if self.coarse_mlp_needed:
            pred_depth_coarse = output_dict['depth_coarse']
            depth_mse_coarse = self.compute_mse(pred_depth_coarse, gt_depth, indices_mask, return_loss_maps)
            total_loss += depth_mse_coarse['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, depth_mse_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed:
            pred_depth_fine = output_dict['depth_fine']
            depth_mse_fine = self.compute_mse(pred_depth_fine, gt_depth, indices_mask, return_loss_maps)
            total_loss += depth_mse_fine['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, depth_mse_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        
        return loss_dict

    @staticmethod
    def compute_mse(pred_depth, gt_depth, indices_mask, return_loss_maps: bool):
        def depth_normalization(depth):
            t_d = torch.median(depth)
            s_d = torch.mean(torch.abs(depth - t_d))
            depth_norm = (depth - t_d) / (s_d + 1e-10)
            return depth_norm
        
        pred_depth = pred_depth[indices_mask] 
        gt_depth = gt_depth[indices_mask]
        pred_depth = depth_normalization(1.0 / pred_depth + 1e-6)
        gt_depth = depth_normalization(gt_depth)

        error = pred_depth - gt_depth
        ray_se = torch.square(error)
        depth_mse = torch.mean(ray_se) if pred_depth.numel() > 0 else 0
        loss_dict = {
            'loss_value': depth_mse,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = {
                this_filename: ray_se
            }

        return loss_dict
