import argparse
import os
import sys
import random
import time

import math
import pickle
import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from core import models
from core.datasets.dataset import BVHDataset, create_data_from_processed_npy
from core.utils.config import Config
from core.visualize.save_video import save_video 
from core.utils.motion_utils import reconstruct_v_trajectory
import core.utils.motion_utils as motion_utils

import pathlib

def RotationMatrix(theta):
    x = theta[..., 0]
    y = theta[..., 1]
    z = theta[..., 2]
    cos = torch.cos
    sin = torch.sin
    R1 = torch.stack([cos(y) * cos(z), sin(x) * sin(y) * cos(z) - cos(x) * sin(z), cos(x) * sin(y) * cos(z) + sin(x) * sin(z)], dim=-1)
    R2 = torch.stack([cos(y) * sin(z), sin(x) * sin(y) * sin(z) + cos(x) * cos(z), cos(x) * sin(y) * sin(z) - sin(x) * cos(z)], dim=-1)
    R3 = torch.stack([-sin(y),         sin(x) * cos(y), cos(x) * cos(y)], dim=-1)
    R = torch.stack([R1, R2, R3], dim=-2)
    return R



# from train import rotate
def rotate(_motion, rotation, bias, device='cpu'):
    R = RotationMatrix(rotation).to(device)
    shape = _motion.shape
    motion = torch.matmul(_motion.reshape(*shape[:-1], 68, 3), R)
    motion = motion + bias.unsqueeze(-2).to(device)
    motion = motion.view(shape)
    return _motion



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description='MotionGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------
def test():
    global args, cfg, device

    args = parse_args()
    cfg = Config.from_file(args.config)


    # Set ?PU device
    cuda = torch.cuda.is_available()
    if cuda:
        print('\033[1m\033[91m' + '# cuda available!' + '\033[0m')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'


    #####################################################    
    ## Prepare for test 
    #####################################################  



    pca_root = pathlib.Path(cfg.train.dataset.pca_root)
    pca_mean = np.load(pca_root / "mean.npy")
    pca_components = np.load(pca_root / "components.npy")
    pca_mean = torch.Tensor(pca_mean).to(device)
    pca_components = torch.Tensor(pca_components).to(device)




    # Set up generator network
    num_class = len(cfg.train.dataset.class_list)
    gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, num_class, pca_mean, pca_components).to(device)

    total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f'Total parameter amount : \033[1m{total_params}\033[0m')


    # Load weight
    if args.weight is not None:
        checkpoint_path = args.weight
    else:
        checkpoint_path = os.path.join(cfg.test.out, 'gen.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = sorted(glob.glob(os.path.join(cfg.test.out, 'checkpoint', 'iter_*.pth.tar')))[-1]

    if not os.path.exists(checkpoint_path):
        print('Generator weight not found!')
    else:
        print(f'Loading generator model from \033[1m{checkpoint_path}\033[0m')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'gen_state_dict' in checkpoint:
            gen.load_state_dict(checkpoint['gen_state_dict'])
            iteration = checkpoint['iteration']
        else:
            gen.load_state_dict(checkpoint)
            iteration = cfg.train.total_iterations
    gen.eval()


    # Set up dataset
    # test_dataset = BVHDataset(cfg.test.dataset, mode='test')
    # test_dataset_name = os.path.split(cfg.test.dataset.data_root.replace('*', ''))[1]

    source_path = pathlib.Path(args.source)

    data = create_data_from_processed_npy(cfg.test.dataset, source_path, data_path=None)

    # # Set standard bvh
    # standard_bvh = cfg.test.dataset.standard_bvh if hasattr(cfg.test.dataset, 'standard_bvh') else 'core/datasets/CMU_standard.bvh'


    # Create output directory
    result_dir = pathlib.Path(f'{cfg.test.out}/sample_generation/iter_{iteration}/')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # from IPython import embed; embed()
    #####################################################    
    ## Test start
    #####################################################  
    x_data = data["motion"]
    x_data = x_data[:x_data.shape[0]-x_data.shape[0]%(cfg.test.dataset.frame_step*16):cfg.test.dataset.frame_step,:]
    if list(data['spline_length_map'].keys())[-1] * 0.1 < x_data.shape[0] * cfg.test.dataset.frame_step:
        x_data = x_data[:-16,:]

    trajectory_part = x_data[:,:3] 
    control_data = motion_utils.sampling(trajectory_part, data['spline_f'], data['spline_length_map'], 0.1, startT=0, endT=x_data.shape[0]*cfg.test.dataset.frame_step, step=cfg.test.dataset.frame_step, with_noise=False)



    # if x_data.shape[0] < cfg.train.dataset.frame_nums // cfg.train.dataset.frame_step:
    #     continue

    # Motion and control signal data
    x_data = torch.from_numpy(x_data).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor)
    x_real = Variable(x_data).to(device)

    control_data = torch.from_numpy(control_data).unsqueeze(0).unsqueeze(1).type(torch.FloatTensor) 
    control = control_data.to(device)

    n_joints = 68
    # Convert root trajectory to velocity
    gt_rotation = x_data[:,:,:,0:3]
    gt_bias = x_data[:,:,:,3:6]
    gt_trajectory = x_data[:,:,:,6:9]
    control = gt_trajectory.to(device)#.unsqueeze(1).type(torch.FloatTensor)
    gt_motion = x_real[:,:,:,6:]
    gt_motion = torch.matmul(gt_motion, pca_components) + pca_mean


    # Convert control curve to velocity 
    v_control = control[:,:,1:,] - control[:,:,:-1,:]
    v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
    v_control = Variable(v_control).to(device)

    results_list = []
    # source_label = cfg.train.dataset.class_list[int(source_path.stem[7:8])-1]
    source_label = source_path.stem.split("_")[0]
    # from IPython import embed; embed()
    # x_motion = rotate(gt_motion.to("cpu"), x_data[:,:,:,3:6].to("cpu"), x_data[:,:,:,6:9].to("cpu"), device="cpu")
    x_motion = gt_motion.to("cpu").detach()
    results_list.append({'caption': f'real({source_label})', 'motion': x_motion, 'control': control.data.cpu(), 'raw':x_data.data.cpu()})
    x_motion = x_motion.detach().numpy().squeeze().T
    # x_motion = x_motion.reshape(x_motion[0], -1, 3) 
    np.save(result_dir / (source_path.stem + "_real.npy"), x_motion)
    start_time = time.time()


    # Generate fake sample
    for fake_label in range(len(cfg.train.dataset.class_list)):
        # Generate noize z
        z = gen.make_hidden(1, x_data.shape[2]).to(device) if cfg.models.generator.use_z else None
        fake_label = torch.tensor([fake_label]).type(torch.LongTensor).to(device)
        fake_rot, fake_bias, x_fake = gen(v_control, z, fake_label)
        fake_motion = torch.matmul(x_fake, pca_components) + pca_mean
        fake_trajectory = x_fake[:,:,:,:3]

        # fake_trajectory = reconstruct_v_trajectory(fake_v_trajectory.data.cpu(), torch.zeros(1,1,1,3))
        # fake_motion = rotate(fake_motion.to("cpu"), fake_rot.to("cpu"), fake_bias.to("cpu"), device="cpu")
        fake_motion = fake_motion.to("cpu").detach()
        caption = f'{cfg.train.dataset.class_list[fake_label]}'
        results_list.append({'caption': caption, 'motion': fake_motion.data.cpu(), 'control': control.data.cpu(), 'raw': torch.cat([fake_rot, fake_bias, x_fake], dim=3).data.cpu()})
        fake_motion = fake_motion.detach().numpy().squeeze().T
    # x_motion = x_motion.reshape(x_motion[0], -1, 3) 
        np.save(result_dir / (source_path.stem + f"_fake_{caption}.npy"), fake_motion)
    start_time = time.time()


    avg_time = (time.time() - start_time) / len(cfg.train.dataset.class_list)

    # Save results
    result_path = result_dir / f'{source_path.stem}_transfered.pkl'
    print(f'\nInference : {str(result_path)} ({v_control.shape[2]} frames) Time: {avg_time:.05f}') 
    pickle.dump(results_list, open(result_path, "wb"))

    # save_video(result_path, results_list, cfg.test)



if __name__ == '__main__':
    test()
