#!/usr/bin/env python
import argparse
import os
import sys
import time
import shutil
import pickle
import glob

import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import tensorboardX as tbx
from core.datasets.dataset import BVHDataset, create_data_from_processed_npy


from core import models
from core.datasets.dataset import BVHDataset
from core.utils.config import Config
from core.utils.gradient_penalty import gradient_penalty
from core.utils.motion_utils import reconstruct_v_trajectory, get_bones_norm
from core.utils.bvh_to_joint import collect_bones
from core.visualize.save_video import save_video

from train import rotate
import pandas as pd
import pathlib

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description='EqualledCycleGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--weight', type=str)
    parser.add_argument('--target', type=str)    
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Restart from the checkpoint.')
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------
def eval_classification():
    global args, cfg, device
    
    args = parse_args()
    cfg = Config.from_file(args.config)

    #======================================================================   
    #
    ### Set up training
    #
    #======================================================================

    # Set ?PU device
    cuda = torch.cuda.is_available()
    if cuda:
        print('\033[1m\033[91m' + '# cuda available!' + '\033[0m')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cpu'

    # set start iteration
    iteration = 0

    # Set up discriminator
    num_class = len(cfg.evalc.dataset.class_list)
    #gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, num_class).to(device)
    dis = getattr(models, cfg.models.discriminator.model)(cfg.models.discriminator, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step, num_class, True).to(device)
    networks = {'dis': dis}
    target_dir = pathlib.Path(args.target)
    files = target_dir.glob("*fake*.npy")

    # Load weight
    if args.weight is not None:
        checkpoint_path = args.weight
    else:
        checkpoint_path = os.path.join(cfg.evalc.out, 'dis.pth')
    
    if not os.path.exists(checkpoint_path):
        print('\033[31m' + 'discriminator weight not found!' + '\033[0m')
    else:
        print('\033[33m' + 'loading discriminator model from ' + '\033[1m' + checkpoint_path + '\033[0m')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(checkpoint)
        if 'gen_state_dict' in checkpoint:
            dis.load_state_dict(checkpoint['dis_state_dict'])
            iteration = checkpoint['iteration']
        else:
            dis.load_state_dict(checkpoint)
            iteration = cfg.train.total_iterations

    dis.eval()
           
    # Set up dataset
    source_dir = pathlib.Path(cfg.test.dataset.data_root)
    npy_source_paths = list(source_dir.glob(f"*.npy"))
    # test_dataset = BVHDataset(cfg.test.dataset, mode='test', from_paths=n)
    
    print(f'Data root \033[1m\"{source_dir.name}\"\033[0m contains \033[1m{len(npy_source_paths)}\033[0m samples.')

    correct = 0

    prediction = []
    for k in range(len(npy_source_paths)):
        data = create_data_from_processed_npy(cfg.test.dataset, npy_source_paths[k], data_path=None)
        x_data = data["motion"]
        label = int(npy_source_paths[k].stem[7:8]) - 1
        x_data = x_data[:x_data.shape[0]-x_data.shape[0]%(cfg.test.dataset.frame_step*16):cfg.test.dataset.frame_step,:]
        if list(data['spline_length_map'].keys())[-1] * 0.1 < x_data.shape[0] * cfg.test.dataset.frame_step:
            x_data = x_data[:-16,:]

        
        # Motion and control signal data
        x_data = torch.from_numpy(x_data)
        x_data = x_data.unsqueeze(0).type(torch.FloatTensor)
        x_data = x_data.unsqueeze(0).type(torch.FloatTensor)
        print(x_data.shape)

        x_real = Variable(x_data).to(device)
        gt_motion = x_real[:,:,:32,9:]


        
        d_real_adv, d_real_cls = dis(gt_motion)
        d_real_cls = d_real_cls.cpu().detach().numpy()

        if label == np.argmax(d_real_cls):
            correct += 1
        
        print('Data %d: GT label %d (%s), Estimate %d (%s): %d / %d (%f)'%(k, label, cfg.evalc.dataset.class_list[label], np.argmax(d_real_cls), cfg.evalc.dataset.class_list[np.argmax(d_real_cls)], correct, k+1, (correct+1e-10)/(k+1)), end="")
        print(d_real_cls)
        prediction.append({"class": cfg.evalc.dataset.class_list[label], "prediction": cfg.evalc.dataset.class_list[np.argmax(d_real_cls)]})
    df = pd.DataFrame(prediction)
    df.to_csv(target_dir / "gt_prediction.csv")
    print("\nstart fake classification")
    prediction = []
    for path in files:
        x_data = np.load(path)
        
        # Motion and control signal data
        x_data = torch.from_numpy(x_data.T[:32])
        x_data = x_data.unsqueeze(0).type(torch.FloatTensor)
        x_data = x_data.unsqueeze(0).type(torch.FloatTensor)        
        #print(x_data.shape)
        x_real = Variable(x_data).to(device)

        d_real_adv, d_real_cls = dis(x_real)
        d_real_cls = d_real_cls.cpu().detach().numpy()
        emotion = path.stem.split("_")[-1]
        label = cfg.evalc.dataset.class_list.index(emotion)
        if label == np.argmax(d_real_cls):
            correct += 1
        
        print('Data %d: GT label %d (%s), Estimate %d (%s): %d / %d (%f)'%(k, label, cfg.evalc.dataset.class_list[label], np.argmax(d_real_cls), cfg.evalc.dataset.class_list[np.argmax(d_real_cls)], correct, k+1, (correct+1e-10)/(k+1)), end="")
        print(d_real_cls)
        prediction.append({"class": emotion, "prediction": cfg.evalc.dataset.class_list[np.argmax(d_real_cls)]})
    df = pd.DataFrame(prediction)
    df.to_csv(args.target + "/fake_prediction.csv")

def label2onehot(label, class_list):
    """ Convert label scalar to onehot vector
    Arguments:
        label <Tensor (batchsize,1)> 
        class_list <List>
    Outputs:
        label_onehot <Tensor (batchsize, n_class)>
    """
    num_class = len(class_list)
    label_array = np.zeros((label.shape[0],num_class))
    for i in range(label.shape[0]):
        label_array[i,label[i]] += 1
    label_onehot = torch.from_numpy(label_array)
    return label_onehot


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

if __name__ == '__main__':
    eval_classification()
