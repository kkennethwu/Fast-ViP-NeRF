# TV_loss_density
# Author: Chung-Ho-Wu
# Last Modified: 25/08/2023
# reference: TensoRF

from pathlib import Path

import torch

from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem

class TV_loos_density(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        return
    
    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        total_loss = 0
        loss_maps = {}
        
        if self.coarse_mlp_needed:
            pred_vis_coarse = output_dict['raw_visibility_coarse'][..., 0]
            # Stop gradients from flowing through sigma.
            target_vis_coarse = output_dict['visibility_coarse']
            loss_coarse = self.compute_visibility_loss(pred_vis_coarse, target_vis_coarse, return_loss_maps)
            total_loss += loss_coarse['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_coarse['loss_maps'], suffix='coarse')

        if self.fine_mlp_needed:
            pred_vis_fine = output_dict['raw_visibility_fine'][..., 0]
            # Stop gradients from flowing through sigma.
            target_vis_fine = output_dict['visibility_fine']
            loss_fine = self.compute_visibility_loss(pred_vis_fine, target_vis_fine, return_loss_maps)
            total_loss += loss_fine['loss_value']
            if return_loss_maps:
                loss_maps = LossUtils01.update_loss_map_dict(loss_maps, loss_fine['loss_maps'], suffix='fine')

        loss_dict = {
            'loss_value': total_loss,
        }
        if return_loss_maps:
            loss_dict['loss_maps'] = loss_maps
        return loss_dict
    
    