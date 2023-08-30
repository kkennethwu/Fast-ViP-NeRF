import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load
from loss_functions import LossUtils01
from loss_functions.LossFunctionParent01 import LossFunctionParent
parent_dir = os.path.dirname(os.path.abspath(__file__))
sources = [
        os.path.join(parent_dir, path)
        for path in ['cuda/segment_cumsum.cpp', 'cuda/segment_cumsum_kernel.cu']]
__CUDA_FIRSTTIME__ = True


class FlattenEffDistLoss(torch.autograd.Function):
        @staticmethod
        def forward(w, m, interval, ray_id):
            '''
            w:        Float tensor in shape [N]. Volume rendering weights of each point.
            m:        Float tensor in shape [N]. Midpoint distance to camera of each point.
            interval: Scalar or float tensor in shape [N]. The query interval of each point.
            ray_id:   Long tensor in shape [N]. The ray index of each point.
            '''
            global __CUDA_FIRSTTIME__
            segment_cumsum_cuda = load(
                    name='segment_cumsum_cuda',
                    sources=sources,
                    verbose=__CUDA_FIRSTTIME__)
            __CUDA_FIRSTTIME__ = False

            n_rays = ray_id.max()+1
            w_prefix, w_total, wm_prefix, wm_total = segment_cumsum_cuda.segment_cumsum(w, m, ray_id)
            loss_uni = (1/3) * interval * w.pow(2)
            loss_bi = 2 * w * (m * w_prefix - wm_prefix)
            return (loss_bi.sum() + loss_uni.sum()) / n_rays
    

    

class DistortionLoss():
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.coarse_mlp_needed = 'coarse_mlp' in self.configs['model']
        self.fine_mlp_needed = 'fine_mlp' in self.configs['model']
        return
    def compute_loss(self, input_dict: dict, output_dict: dict, return_loss_maps: bool = False):
        #B rays=>batch 1941
        B = input_dict['rays_o'].shape[0]
        device = torch.device('cuda:1')
        eff_dist_loss = FlattenEffDistLoss()
        total_loss = 0
        #w weights=>
        #N samples=>
        #loss_coarse=0.0
        if self.coarse_mlp_needed:

            Interval_coarse = output_dict['z_vals_coarse']#[B,N]=>[1941,64]
            last_column = Interval_coarse[:, -1:]

            N = output_dict['visibility_coarse'].shape[1]
            w = output_dict['weights_coarse']#[B,N]=>[1941,64]
  
            m = (Interval_coarse [:, 1:] + Interval_coarse [:,:-1]) * 0.5
            last_column = m[:, -1:]
            m = torch.cat((m, last_column), dim=1)
            ray_id_coarse = torch.arange(0, B-1, dtype=torch.int64,device=w.device)
            ray_id_coarse = ray_id_coarse.view(-1, 1) 
            ray_id_coarse = ray_id_coarse.repeat(1,N)
            ray_id_coarse= ray_id_coarse.flatten()
            loss_coarse = eff_dist_loss.forward(w, m, Interval_coarse, ray_id_coarse)
            
            total_loss = total_loss+loss_coarse
        if self.fine_mlp_needed:
            Interval_fine  = output_dict['z_vals_fine']
            N = output_dict['visibility_fine'].shape[1]
            w = output_dict['weights_fine']#[B,N]=>[1941,64]
            m = (Interval_fine [:, 1:] + Interval_fine [:, :-1]) * 0.5
            last_column = m[:, -1:]
            m = torch.cat((m, last_column), dim=1)
            ray_id_fine = torch.arange(0, B-1, dtype=torch.int64,device=w.device)
            ray_id_fine = ray_id_fine.repeat(N,1)
            ray_id_fine = ray_id_fine.flatten()
            
            
            loss_fine = eff_dist_loss.forward(w, m, Interval_fine, ray_id_fine)
            total_loss += loss_fine
         
        loss_dict = {
            'loss_value': total_loss,
        }
        print ("loss:",total_loss)
        #if return_loss_maps:
            #loss_dict['loss_maps'] = loss_maps
        return loss_dict



