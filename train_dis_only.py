#!/usr/bin/env python
import argparse
import os
import sys
import time
import shutil
import pickle

import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import tensorboardX as tbx

from core import models
from core.datasets.dataset import BVHDataset
from core.utils.config import Config
from core.utils.gradient_penalty import gradient_penalty
from core.utils.motion_utils import reconstruct_v_trajectory, get_bones_norm
from core.utils.bvh_to_joint import collect_bones
from core.visualize.save_video import save_video
from sklearn.model_selection import train_test_split



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description='EqualledCycleGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Restart from the checkpoint.')
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------


def train():
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

    # Set up dataset
    dataset = BVHDataset(cfg.train.dataset, mode='train')
    rand_indice = np.arange(len(dataset))
    np.random.shuffle(rand_indice)
    train_num = int(len(dataset) * 0.9)
    print(f"split train {train_num}: val {len(dataset)-train_num}")
    train_indices, val_indices = rand_indice[:train_num], rand_indice[train_num:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    # from IPython import embed ; embed()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = cfg.train.batchsize,
        num_workers = cfg.train.num_workers,
        shuffle=True,
        drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = cfg.train.batchsize,
        shuffle=True, 
        num_workers =cfg.train.num_workers,
        drop_last=True
    )
    print(f'Data root \033[1m\"{cfg.train.dataset.data_root}\"\033[0m contains \033[1m{len(train_dataset)}\033[0m samples.')
    # from IPython import embed; embed()
    
    # Set up networks to train
    num_class = len(cfg.train.dataset.class_list)
    # from IPython import embed; embed()
    n_joints = (train_dataset[0][0].shape[1]-6)//3

    dis = getattr(models, cfg.models.discriminator.model)(cfg.models.discriminator, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step, num_class, True).to(device)
    networks = {'dis': dis}

    
    # Load resume state_dict (to restart training)
    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            print(f'loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            for name, model in networks.items():
                model.load_state_dict(checkpoint[f'{name}_state_dict'])
            iteration = checkpoint['iteration']


    # Set up an optimizer
    gen_lr = cfg.train.parameters.g_lr
    dis_lr = cfg.train.parameters.d_lr
    opts = {}
    opts['dis'] = torch.optim.Adam(dis.parameters(), lr=dis_lr, betas=(0.5, 0.999))

    # Load resume state_dict
    if args.resume:

        opts['dis'].load_state_dict(checkpoint['opt_dis_state_dict'])
           



    # Save scripts and command
    if not os.path.exists(cfg.train.out):
        os.makedirs(cfg.train.out)
    shutil.copy(args.config, f'./{cfg.train.out}')
    shutil.copy('./core/models/MotionGAN.py', f'./{cfg.train.out}')
    shutil.copy('./train.py', f'./{cfg.train.out}')

    commands = sys.argv
    with open(f'./{cfg.train.out}/command.txt', 'w') as f:
        f.write(f'python {commands[0]} ')
        for command in commands[1:]:
            f.write(command + ' ') 

    # Set Criterion
    if cfg.train.GAN_type == 'normal':
         GAN_criterion = torch.nn.BCELoss().to(device)
    elif cfg.train.GAN_type == 'ls':
         GAN_criterion = torch.nn.MSELoss().to(device)
    else:
         GAN_criterion = None
    BCE_criterion = torch.nn.BCELoss().to(device)
    base_criterion = torch.nn.MSELoss().to(device)


    # Tensorboard Summary Writer
    writer = tbx.SummaryWriter(log_dir=os.path.join(cfg.train.out, 'log'))


    # train
    print('\033[1m\033[93m## Start Training!! ###\033[0m')
    while iteration < cfg.train.total_iterations:
        iteration = train_loop(train_loader,
                               train_dataset,
                               val_data_loader,
                               networks,
                               opts,
                               iteration,
                               cfg.train.total_iterations,
                               GAN_criterion,
                               BCE_criterion,
                               base_criterion,
                               writer)

    # Save final model
    state = {'iteration':iteration, 'config':dict(cfg)}
    state[f'dis_state_dict'] = dis.state_dict()
    state['opt_dis_state_dict'] = opts['dis'].state_dict()
    
    path = os.path.join(os.path.join(cfg.train.out,'checkpoint'), f'checkpoint.pth.tar')
    torch.save(state, path)
    torch.save(dis.state_dict(), os.path.join(cfg.train.out,f'dis.pth'))
    print(f'trained model saved!')

    writer.close()


#======================================================================
# 
### Train epoch
# 
#======================================================================

def train_loop(train_loader,
          train_dataset,
          val_loader,
          networks,
          opts,
          iteration,
          total_iteration,
          GAN_criterion,
          BCE_criterion,
          base_criterion,
          writer):
    # Time Keeper
    batch_time = AverageMeter()

    #####################################################    
    ### Set up train option
    #####################################################    

    # Standard skelton
    standard_bvh = cfg.train.dataset.standard_bvh if hasattr(cfg.train.dataset, 'standard_bvh') else 'core/datasets/CMU_standard.bvh'
    class_list = cfg.train.dataset.class_list
    
    # Cofficients of training loss
    _lam_g_adv = cfg.train.parameters.lam_g_adv
    _lam_g_trj = cfg.train.parameters.lam_g_trj
    _lam_g_cls = cfg.train.parameters.lam_g_cls
    _lam_g_cons = cfg.train.parameters.lam_g_cons
    _lam_g_bone = cfg.train.parameters.lam_g_bone if hasattr(cfg.train.parameters, 'lam_g_bone') else 0
    _lam_d_adv = cfg.train.parameters.lam_d_adv
    _lam_d_gp = cfg.train.parameters.lam_d_gp if cfg.train.GAN_type in ['wgan-gp', 'r1'] else 0
    _lam_d_drift = cfg.train.parameters.lam_d_drift if cfg.train.GAN_type == 'wgan-gp' else 0
    _lam_d_cls = cfg.train.parameters.lam_d_cls
   
    # Target tensor of adversarial loss
    real_target = Variable(torch.ones(1,1)*0.9).to(device)
    fake_target = Variable(torch.ones(1,1)*0.1).to(device)

    # Prepare for bone loss
    if _lam_g_bone > 0:
        bones = collect_bones(standard_bvh)
        standard_frame = torch.from_numpy(train_dataset[0][0][None,None,:,:]).type(torch.FloatTensor).to(device)
        target_bones_norm = get_bones_norm(standard_frame[:,:,:1,3:], bones)


    ## Get model
    dis = networks['dis']
    
    opt_dis = opts['dis']

    end = time.time()

    # Switch model mode to train
    dis.train()
 

    #####################################################    
    ## Training iteration
    #####################################################    
    for i, (x_data, control_data, label) in enumerate(train_loader):

        #---------------------------------------------------
        #  Prepare model input 
        #---------------------------------------------------

        # Motion and control signal data
        x_data = x_data.unsqueeze(1).type(torch.FloatTensor)
        x_real = Variable(x_data).to(device)
        gt_motion = x_real[:,:,:,9:]
        gt_motion = Variable(gt_motion).to(device)

        control_data = control_data.unsqueeze(1).type(torch.FloatTensor)
        control = control_data.to(device)
        loss_collector = {}
        
        #---------------------------------------------------
        #  Update Discriminator
        #---------------------------------------------------
        if _lam_d_adv > 0:
            # Forward Discriminator
            _x_real = gt_motion - torch.mean(gt_motion, 2, keepdim=True)#rotate(gt_motion, gt_rotation, gt_bias, device=device)
            d_real_adv, d_real_cls = dis(_x_real)


            # Class loss
            real_label_onehot = label2onehot(label, class_list).type(torch.FloatTensor).to(device) 

            d_cls_loss = _lam_d_cls * BCE_criterion(d_real_cls, real_label_onehot)
            d_loss = d_cls_loss
            loss_collector['d_cls_loss'] = d_cls_loss.item()
            opt_dis.zero_grad()
            d_loss.backward()
            # torch.nn.utils.clip_grad_norm_(dis.parameters(), 5)
            opt_dis.step()

        #---------------------------------------------------
        #  Update generator
        #---------------------------------------------------



        # Measure batch_time
        batch_time.update(time.time() - end)
        end = time.time()

        

        #---------------------------------------------------
        # Print Log
        #---------------------------------------------------
        if (iteration + i + 1) % cfg.train.display_interval == 0:
            total = 0
            correct = 0
            for (x_data, control_data, label) in val_loader:

                # label = label.cpu().detach().numpy()
                x_data = x_data.unsqueeze(1).type(torch.FloatTensor)
                x_real = x_data[:,:,:,9:].to(device)
                x_real = x_real -  torch.mean(x_real, 2, keepdim=True)
                d_real_adv, d_real_cls = dis(x_real.detach())
                real_label_onehot = label2onehot(label, class_list).type(torch.FloatTensor).to(device) 
                d_cls_loss = _lam_d_cls * BCE_criterion(d_real_cls, real_label_onehot)
                d_real_cls = d_real_cls.cpu().detach().numpy()
                total += d_real_cls.shape[0]
                correct += np.sum(label.cpu().detach().numpy() == np.argmax(d_real_cls, axis=1))
                loss_collector['val_cls_loss'] = d_cls_loss.item()
            
            total_time = batch_time.val * (total_iteration - (iteration + i))
            mini, sec = divmod(total_time, 60)
            hour, mini = divmod(mini, 60)
            loss_summary = ''.join([f'{name}:{val:.5f}  ' for name, val in loss_collector.items()])

            print((f'Iteration:[{iteration+i}][{total_iteration}]\t'
                   f'Time {batch_time.val:.3f} (Total {int(hour)}:{int(mini)}:{sec:.02f} )\t'
                   f'Loss {loss_summary}\t'
                   ))

            writer.add_scalars('train/loss', loss_collector, iteration+i+1)
            print(f'val: {correct*1.0 / total}')


        #---------------------------------------------------
        # Save checkpoint
        #---------------------------------------------------
        if (iteration+i+1) % cfg.train.save_interval == 0:
            if not os.path.exists(os.path.join(cfg.train.out,'checkpoint')):
                os.makedirs(os.path.join(cfg.train.out,'checkpoint'))
            path = os.path.join(os.path.join(cfg.train.out,'checkpoint'), f'iter_{iteration + i:04d}.pth.tar')
            state = {'iteration':iteration+i+1, 'config':dict(cfg)}
            state[f'dis_state_dict'] = dis.state_dict()
            state['opt_dis_state_dict'] = opt_dis.state_dict()
            torch.save(state, path)

        # Finish training
        if iteration+i+1 > total_iteration:
            return iteration+i+1
    return iteration+i+1




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
    train()
