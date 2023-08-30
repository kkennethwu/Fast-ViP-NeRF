import datetime
import os
import time
import traceback
from pathlib import Path

import numpy
import pandas
import skimage.io
import skvideo.io

import Tester01 as Tester
import Trainer01 as Trainer

this_filepath = Path(__file__)
this_filename = this_filepath.stem

from NerfLlffTrainerTester01 import *


def DLP_demo_all_views_wo_sparse_depth():
    train_num = 13+3
    test_num = 13+3
    # scene_names = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
    scene_names = ['fern']
    
    for scene_name in scene_names:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'NeRF_LLFF',
            'database_dirpath': 'databases/NeRF_LLFF/data',
            'data_loader': {
                'data_loader_name': 'NerfLlffDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 1,
                'scene_names': [scene_name],
                'resolution_suffix': '_down4',
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 1024,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_prior': {
                    'load_masks': False,
                    'load_weights': False,
                    'masks_dirname': 'VW04',
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'coarse_mlp': {
                    'num_samples':128,
                    'netdepth': 8,
                    'netwidth': 128,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': True,
                },
                # 'fine_mlp': {
                #     'num_samples': 128,
                #     'netdepth': 8,
                #     'netwidth': 256,
                #     'points_positional_encoding_degree': 10,
                #     'views_positional_encoding_degree': 4,
                #     'use_view_dirs': True,
                #     'view_dependent_rgb': True,
                #     'predict_visibility': True,
                # },
                'chunk': 512*1024,
                'lindisp': False,
                'netchunk': 1024*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE01',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss01',
                    'weight': 0.1,
                },
                # {
                #     'name': 'VisibilityPriorLoss01',
                #     'iter_weights': {
                #         '0': 0.1, '30000': 0.01,
                #     },
                # },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 25000,
            'validation_interval': 25000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 25000,
            'mixed_precision_training': False,
            # 'seed': numpy.random.randint(1000),
            'seed': 0,
            'device': [0],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 1,
            'train_num': train_num,
            'model_name': 'Model_Iter025000.tar',
            'database_name': 'NeRF_LLFF',
            'database_dirpath': 'NeRF_LLFF/data',
            'resolution_suffix': train_configs['data_loader']['resolution_suffix'],
            'scene_names': [scene_name],
            'device': [0],
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return

def DLP_demo_4_views():
    train_num = 13
    test_num = 13
    scene_names = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']

    for scene_name in scene_names:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'NeRF_LLFF',
            'database_dirpath': 'databases/NeRF_LLFF/data',
            'data_loader': {
                'data_loader_name': 'NerfLlffDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 4,
                'scene_names': [scene_name],
                'resolution_suffix': '_down4',
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 2048,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_prior': {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VW04',
                },
                'sparse_depth': {
                    'dirname': 'DE04',
                    'num_rays': 2048,
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'coarse_mlp': {
                    'num_samples': 64,
                    'netdepth': 8,
                    'netwidth': 256,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': True,
                },
                'fine_mlp': {
                    'num_samples': 128,
                    'netdepth': 8,
                    'netwidth': 256,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': True,
                },
                'chunk': 4*1024,
                'lindisp': False,
                'netchunk': 16*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE01',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss01',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityPriorLoss01',
                    'iter_weights': {
                        '0': 0, '30000': 0.001,
                    },
                },
                {
                    "name": "SparseDepthMSE01",
                    "weight": 0.1,
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 200000,
            'validation_interval': 10000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            # 'seed': numpy.random.randint(1000),
            'seed': 0,
            'device': [1],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 4,
            'train_num': train_num,
            'model_name': 'Model_Iter200000.tar',
            'database_name': 'NeRF_LLFF',
            'database_dirpath': 'NeRF_LLFF/data',
            'resolution_suffix': train_configs['data_loader']['resolution_suffix'],
            'scene_names': [scene_name],
            'device': [1],
        }
        start_training(train_configs)
        start_testing(test_configs)
        # start_testing_videos(test_configs)
        # start_testing_static_videos(test_configs)
    return

def DLP_demo_2_views():
    train_num = 11
    test_num = 11
    # scene_names = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
    scene_names = ['fern']
    
    for scene_name in scene_names:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'NeRF_LLFF',
            'database_dirpath': 'databases/NeRF_LLFF/data',
            'data_loader': {
                'data_loader_name': 'NerfLlffDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 2,
                'scene_names': [scene_name],
                'resolution_suffix': '_down4',
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 1024,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_prior': {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VW02',
                },
                'sparse_depth': {
                    'dirname': 'DE02',
                    'num_rays': 1024,
                },
            },
            'model': {
                'name': 'VipNeRF01',
                'coarse_mlp': {
                    'num_samples': 256,
                    'netdepth': 8,
                    'netwidth': 256,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': True,
                },
                # 'fine_mlp': {
                #     'num_samples': 128,
                #     'netdepth': 8,
                #     'netwidth': 256,
                #     'points_positional_encoding_degree': 10,
                #     'views_positional_encoding_degree': 4,
                #     'use_view_dirs': True,
                #     'view_dependent_rgb': True,
                #     'predict_visibility': True,
                # },
                'chunk': 512*1024,
                'lindisp': False,
                'netchunk': 1024*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE01',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss01',
                    'weight': 0.2,
                },
                {
                    'name': 'VisibilityPriorLoss01',
                    'iter_weights': {
                        '0': 0.2, '3000': 0.2,
                    },
                },
                {
                    "name": "SparseDepthMSE01",
                    "weight": 0.2,
                },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 20000,
            'validation_interval': 2000,
            'validation_chunk_size': 512 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 10000,
            'mixed_precision_training': False,
            # 'seed': numpy.random.randint(1000),
            'seed': 0,
            'device': [0],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 2,
            'train_num': train_num,
            'model_name': 'Model_Iter020000.tar',
            'database_name': 'NeRF_LLFF',
            'database_dirpath': 'NeRF_LLFF/data',
            'resolution_suffix': train_configs['data_loader']['resolution_suffix'],
            'scene_names': [scene_name],
            'device': [0],
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return
    
def demo_best_2_view():
    train_num = 11
    test_num = 11
    # scene_names = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']

    for scene_name in scene_names:
        train_configs = {
            'trainer': f'{this_filename}/{Trainer.this_filename}',
            'train_num': train_num,
            'database': 'NeRF_LLFF',
            'database_dirpath': 'databases/NeRF_LLFF/data',
            'data_loader': {
                'data_loader_name': 'NerfLlffDataLoader01',
                'data_preprocessor_name': 'DataPreprocessor01',
                'train_set_num': 2,
                'scene_names': [scene_name],
                'resolution_suffix': '_down4',
                'recenter_camera_poses': True,
                'bd_factor': 0.75,
                'spherify': False,
                'ndc': True,
                'batching': True,
                'downsampling_factor': 1,
                'num_rays': 1024,
                'precrop_fraction': 1,
                'precrop_iterations': -1,
                'visibility_prior': {
                    'load_masks': True,
                    'load_weights': False,
                    'masks_dirname': 'VW02',
                },
                'sparse_depth': {
                    'dirname': 'DE02',
                    'num_rays': 1024,
                },
                # 'dense_depth': {
                #     'dirname': 'DE02',
                #     'num_rays': 1024,
                # },
            },
            'N_voxel_init': 2097156, # 128**3
            'N_voxel_final': 262144000, # 640**3
            # 'N_voxel_final': 27000000, # 300**3'
            # 'upsamp_list': [2000,3000,4000,5500],
            'upsamp_list': [2000,3000,4000,5500],
            'aabb': [[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]], # define scene_box
            'step_ratio': 0.5,
            'lr_upsample_reset': True,
            'lr_decay_target_ratio': 0.1,
            'lr_decay_iters': -1,
            'lr_initial_voxel': 0.2,
            'lr_initial_mlp': 0.001,
            'model': {
                'name': 'VipNeRF01',
                'coarse_mlp': {
                    'num_samples': 128,
                    'max_nSamples': 2000,
                    'netdepth': 8,
                    'netwidth': 256,
                    'points_positional_encoding_degree': 10,
                    'views_positional_encoding_degree': 4,
                    'use_view_dirs': True,
                    'view_dependent_rgb': True,
                    'predict_visibility': True,
                    'gridSize': [128, 128, 128], # the actaul gridSize is based on N_voxel_init. This config just help transfer parameter
                },
                # 'fine_mlp': {
                #     'num_samples': 128,
                #     'netdepth': 8,
                #     'netwidth': 256,
                #     'points_positional_encoding_degree': 10,
                #     'views_positional_encoding_degree': 4,
                #     'use_view_dirs': True,
                #     'view_dependent_rgb': True,
                #     'predict_visibility': True,
                # },
                'chunk': 64*1024,
                'lindisp': False,
                'netchunk': 128*1024,
                'perturb': True,
                'raw_noise_std': 1.0,
                'white_bkgd': False,
            },
            'losses': [
                {
                    'name': 'MSE01',
                    'weight': 1,
                },
                {
                    'name': 'VisibilityLoss01',
                    'weight': 0.1,
                },
                {
                    'name': 'VisibilityPriorLoss01',
                    'iter_weights': {
                        '0': 0, '3000': 0.001
                    },
                },
                {
                    "name": "SparseDepthMSE01",
                    "weight": 0.1,
                },
                # {
                #     "name": "DenseDepthMSE01",
                #     "weight": 0.2,
                # },
            ],
            'optimizer': {
                'lr_decayer_name': 'NeRFLearningRateDecayer01',
                'lr_initial': 5e-4,
                'lr_decay': 250,
                'beta1': 0.9,
                'beta2': 0.999,
            },
            'resume_training': True,
            'num_iterations': 30000,
            'validation_interval': 3000,
            'validation_chunk_size': 64 * 1024,
            'validation_save_loss_maps': False,
            'model_save_interval': 30000,
            'mixed_precision_training': False,
            # 'seed': numpy.random.randint(1000),
            'seed': 0,
            'device': [0],
        }
        test_configs = {
            'Tester': f'{this_filename}/{Tester.this_filename}',
            'test_num': test_num,
            'test_set_num': 2,
            'train_num': train_num,
            'model_name': 'Model_Iter030000.tar',
            'database_name': 'NeRF_LLFF',
            'database_dirpath': 'NeRF_LLFF/data',
            'resolution_suffix': train_configs['data_loader']['resolution_suffix'],
            'scene_names': [scene_name],
            'device': [0],
        }
        start_training(train_configs)
        start_testing(test_configs)
        start_testing_videos(test_configs)
        start_testing_static_videos(test_configs)
    return


def main():
    # DLP_demo_all_views_wo_sparse_depth()x
    # DLP_demo_4_views()
    # DLP_demo_2_views()
    # DLP_demo_all_views_wo_sparse_depth()
    demo_best_2_view()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))