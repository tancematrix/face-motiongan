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

def rotate(_motion, rotation, bias, device='cpu'):
    R = RotationMatrix(rotation).to(device)
    shape = _motion.shape
    motion = torch.matmul(_motion.reshape(*shape[:-1], 68, 3), R)
    motion = motion + bias.unsqueeze(-2).to(device)
    motion = motion.view(shape)
    return motion


def rotate_and_parallel(motion, homo_vec):
    R = homo_vec[..., 1:4].view(-1, 3, 3)
    pararell = homo_vec[..., :1]
    shape = motion.shape
    motion = torch.matmul(motion.reshape(*shape[:-1], 68, 3), R)
    motion = motion + pararell
    return motion


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
    train_dataset = BVHDataset(cfg.train.dataset, mode='train')
    # from IPython import embed ; embed()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = cfg.train.batchsize,
        num_workers = cfg.train.num_workers,
        shuffle=True,
        drop_last=True)
    print(f'Data root \033[1m\"{cfg.train.dataset.data_root}\"\033[0m contains \033[1m{len(train_dataset)}\033[0m samples.')
    # from IPython import embed; embed()
    
    # Set up networks to train
    num_class = len(cfg.train.dataset.class_list)
    # from IPython import embed; embed()
    n_joints = (train_dataset[0][0].shape[1]-6)//3

    gen = getattr(models, cfg.models.generator.model)(cfg.models.generator, num_class).to(device)
    dis = getattr(models, cfg.models.discriminator.model)(cfg.models.discriminator, cfg.train.dataset.frame_nums//cfg.train.dataset.frame_step, num_class, True).to(device)
    networks = {'gen': gen, 'dis': dis}

    
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
    opts['gen'] = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    opts['dis'] = torch.optim.Adam(dis.parameters(), lr=dis_lr, betas=(0.5, 0.999))

    # Load resume state_dict
    if args.resume:
        opts['gen'].load_state_dict(checkpoint['opt_gen_state_dict'])
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
    state[f'gen_state_dict'] = gen.state_dict()
    state[f'dis_state_dict'] = dis.state_dict()
    state['opt_gen_state_dict'] = opts['gen'].state_dict()
    state['opt_dis_state_dict'] = opts['dis'].state_dict()
    
    path = os.path.join(os.path.join(cfg.train.out,'checkpoint'), f'checkpoint.pth.tar')
    torch.save(state, path)
    torch.save(gen.state_dict(), os.path.join(cfg.train.out,f'gen.pth'))
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
    gen = networks['gen']
    dis = networks['dis']
    
    opt_gen = opts['gen']
    opt_dis = opts['dis']

    end = time.time()

    # Switch model mode to train
    gen.train()
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
        control_data = control_data.unsqueeze(1).type(torch.FloatTensor)
        control = control_data.to(device)

        batchsize = x_data.shape[0]
        n_joints = (x_data.shape[3]-9)//3

        # Convert root trajectory to velocity
        gt_rotation = x_data[:,:,:,3:6]
        gt_bias = x_data[:,:,:,6:9]
        gt_trajectory = x_data[:,:,:,0:3]
        gt_motion = x_real[:,:,:,9:]
        gt_v_trajectory = gt_trajectory[:,:,1:,:] - gt_trajectory[:,:,:-1,:]
        gt_v_trajectory = F.pad(gt_v_trajectory, (0,0,1,0), mode='reflect')
        gt_v_trajectory = Variable(gt_v_trajectory).to(device)
        gt_motion = Variable(gt_motion).to(device)
        # import pdb; pdb.set_trace()

        # Convert control curve to velociry
        # v_control = control[:,:,1:,] - control[:,:,:-1,:]
        # v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
        # v_control = Variable(v_control).to(device) 

        input_dim = cfg.models.generator.input_dim if hasattr(cfg.models.generator, 'input_dim') else 3
        if input_dim == 3:
            control = gt_trajectory.to(device)#.unsqueeze(1).type(torch.FloatTensor)
            v_control = control[:,:,1:,] - control[:,:,:-1,:]
            v_control = F.pad(v_control, (0,0,1,0), mode='reflect')
            v_control = Variable(v_control).to(device) 
        elif input_dim == 27:
            v_control = x_real[:,:,:,6:]
        elif input_dim == 204:
            v_control = gt_motion
        else:
            raise ValueError("invalid input ndim")

        # Make style label
        real_label = label
        fake_label = torch.randint(0, len(class_list), size=(batchsize,)).type(torch.LongTensor)

        real_label_onehot = label2onehot(real_label, class_list).type(torch.FloatTensor).to(device) 
        fake_label_onehot = label2onehot(fake_label, class_list).type(torch.FloatTensor).to(device)
        fake_label = fake_label.to(device)
        real_label = real_label.to(device)


        # Generate noize z
        z = Variable(gen.make_hidden(batchsize, x_data.shape[2])).to(device) if cfg.models.generator.use_z else None
       
        ### Forward Generator
        fake_rot, fake_bias, x_fake = gen(v_control, z, fake_label)

        
        loss_collector = {}
        
        #---------------------------------------------------
        #  Update Discriminator
        #---------------------------------------------------
        if _lam_d_adv > 0:
            # Forward Discriminator
            # np.save("../face-transform/face_analysis/train_example.npy", np.array(x_data.cpu()))
            _x_real = gt_motion #rotate(gt_motion, gt_rotation, gt_bias, device=device)
            # np.save("../face-transform/face_analysis/train_example_rot.npy", np.array(_x_real.cpu()))
            # raise ValueError("????????????")
            # _x_fake = rotate(x_fake, fake_rot, fake_bias, device=device)
            # np.save("../face-transform/face_analysis/train_example_fake.npy", x_fake.detach().cpu().numpy())
            # from IPython import embed; embed()
            _x_fake = x_fake
            d_fake_adv, d_fake_cls = dis(_x_fake.detach())
            d_real_adv, d_real_cls = dis(_x_real)

            # GAN loss
            if cfg.train.GAN_type == 'ls' or cfg.train.GAN_type == 'normal':
                fake_target = fake_target.expand_as(d_fake_adv)
                real_target = real_target.expand_as(d_real_adv)
                d_adv_loss = _lam_d_adv * (GAN_criterion(d_fake_adv, fake_target) + GAN_criterion(d_real_adv, real_target))
                d_loss = d_adv_loss
            elif cfg.train.GAN_type == 'wgan-gp':
                d_adv_loss = _lam_d_adv * (torch.mean(d_fake_adv) - torch.mean(d_real_adv))
                # calucurate gradinet penalty
                d_gp_loss = _lam_d_gp * gradient_penalty(input_d_fake, input_d_real, dis, device) 
                d_drift_loss = _lam_d_drift * torch.mean(d_real_adv * d_real_adv)
                d_loss = d_adv_loss + d_gp_loss + d_drift_loss 
                loss_collector['d_gp_loss'] = d_gp_loss.item()
                loss_collector['d_drift_loss'] = d_drift_loss.item()
            elif cfg.train.GAN_type == 'hinge':
                d_adv_loss = _lam_d_adv * (torch.mean(torch.relu(1. - d_real_adv)) + torch.mean(torch.relu(1. + d_fake_adv)))
                d_loss = d_adv_loss
            else:
                raise ValueError(f'Invalid loss type!! ({self.GAN_type})')  
            loss_collector['d_adv_loss'] = d_adv_loss.item()


            # Class loss
            d_cls_loss = _lam_d_cls * BCE_criterion(d_real_cls, real_label_onehot)
            d_loss += d_cls_loss
            loss_collector['d_cls_loss'] = d_cls_loss.item()


            opt_dis.zero_grad()
            d_loss.backward()
            # torch.nn.utils.clip_grad_norm_(dis.parameters(), 5)
            opt_dis.step()

        #---------------------------------------------------
        #  Update generator
        #---------------------------------------------------
        g_loss = 0

        if _lam_g_cons > 0:
            # gt_residual = gt_motion - torch.mean(gt_motion, -2, keepdim=True)
            # fake_residual = x_fake - torch.mean(gt_motion, -2, keepdim=True)
            # _lam_patience = 5.
            # g_cons_loss = torch.var(fake_residual) - _lam_patience*torch.var(gt_residual)
            # g_cons_loss = _lam_g_cons * torch.relu(g_cons_loss)
            # loss_collector["g_cons_loss"] = g_cons_loss.item()

            # gt_residual_rot = _x_real - torch.mean(gt_motion, -2, keepdim=True)
            # fake_residual_rot = _x_fake - torch.mean(gt_motion, -2, keepdim=True)
            # _lam_patience = 5.
            # g_cons_loss_rot = torch.var(fake_residual_rot) - _lam_patience*torch.var(gt_residual_rot)
            # g_cons_loss_rot = _lam_g_cons * torch.relu(g_cons_loss_rot)
            # loss_collector["g_cons_loss_rot"] = g_cons_loss_rot.item()
            # g_loss += g_cons_loss_rot
            # from IPython import embed; embed()
            cons_rot, cons_bias, x_cons = gen(v_control, z, real_label)
            g_cons_loss = _lam_g_cons * base_criterion(x_cons, gt_motion)
            loss_collector['g_cons_loss'] = g_cons_loss.item()
            g_loss += g_cons_loss


        # GAN loss
        if _lam_g_adv > 0:
            _x_fake = x_fake
            d_fake_adv, d_fake_cls = dis(_x_fake)

            if cfg.train.GAN_type == 'ls' or cfg.train.GAN_type == 'normal':
                g_adv_loss = _lam_g_adv * GAN_criterion(d_fake_adv, real_target)
            elif cfg.train.GAN_type =='wgan-gp':
                g_adv_loss = - _lam_g_adv * d_fake_adv.mean()
            elif cfg.train.GAN_type == 'hinge':
                g_adv_loss = _lam_g_adv * (- torch.mean(d_fake_adv))
            else:
                raise ValueError(f'Invalid loss type!! ({self.GAN_type})')  
            loss_collector['g_adv_loss'] = g_adv_loss.item()
            g_loss += g_adv_loss


        ### Trajectory reconstrunction loss
        if _lam_g_trj > 0:
            g_trj_loss = 0
            trjloss_sampling_points = cfg.train.trjloss_sampling_points if hasattr(cfg.train, 'trjloss_sampling_points') else 1
            sampling_interval = fake_v_trajectory.shape[2]//trjloss_sampling_points
             # error at each sampling point
            for tp in range(trjloss_sampling_points):
                fake_trajectory_point = control[:,:,0,:] + torch.sum(fake_v_trajectory[:,:,:(tp+1)*sampling_interval,:], dim=2)
                g_trj_loss += _lam_g_trj * 0.5 * (base_criterion(fake_trajectory_point[:,:,0], control[:,:,(tp+1)*sampling_interval-1,0]) + base_criterion(fake_trajectory_point[:,:,2], control[:,:,(tp+1)*sampling_interval-1,2])) # except y-axis

            loss_collector['g_trj_loss'] = g_trj_loss.item()
            g_loss += g_trj_loss

        ### Class loss
        g_cls_loss = _lam_g_cls * BCE_criterion(d_fake_cls, fake_label_onehot)
        g_loss += g_cls_loss
        loss_collector['g_cls_loss'] = g_cls_loss.item()


        ### Bone loss
        # Note that bone loss cannot be used with meanstd normalization because its break constraint
        if _lam_g_bone > 0:
            fake_bones_norm = get_bones_norm(x_fake, bones)
            g_bone_loss = _lam_g_bone * base_criterion(fake_bones_norm, target_bones_norm.expand_as(fake_bones_norm))
            g_loss += g_bone_loss
            loss_collector['g_bone_loss'] = g_bone_loss.item()

            real_bones_norm = get_bones_norm(x_real[:,:,:,3:], bones)
        
             
        opt_gen.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5)

        opt_gen.step()


        # Measure batch_time
        batch_time.update(time.time() - end)
        end = time.time()

        

        #---------------------------------------------------
        # Print Log
        #---------------------------------------------------
        if (iteration + i + 1) % cfg.train.display_interval == 0:
            total_time = batch_time.val * (total_iteration - (iteration + i))
            mini, sec = divmod(total_time, 60)
            hour, mini = divmod(mini, 60)
            loss_summary = ''.join([f'{name}:{val:.5f}  ' for name, val in loss_collector.items()])

            print((f'Iteration:[{iteration+i}][{total_iteration}]\t'
                   f'Time {batch_time.val:.3f} (Total {int(hour)}:{int(mini)}:{sec:.02f} )\t'
                   f'Loss {loss_summary}\t'
                   ))

            writer.add_scalars('train/loss', loss_collector, iteration+i+1)


        #---------------------------------------------------
        # Save checkpoint
        #---------------------------------------------------
        if (iteration+i+1) % cfg.train.save_interval == 0:
            if not os.path.exists(os.path.join(cfg.train.out,'checkpoint')):
                os.makedirs(os.path.join(cfg.train.out,'checkpoint'))
            path = os.path.join(os.path.join(cfg.train.out,'checkpoint'), f'iter_{iteration + i:04d}.pth.tar')
            state = {'iteration':iteration+i+1, 'config':dict(cfg)}
            state[f'gen_state_dict'] = gen.state_dict()
            state[f'dis_state_dict'] = dis.state_dict()
            state['opt_gen_state_dict'] = opt_gen.state_dict()
            state['opt_dis_state_dict'] = opt_dis.state_dict()
            torch.save(state, path)

        #---------------------------------------------------
        # Save preview image
        #---------------------------------------------------
        if (iteration+i+1) % cfg.train.preview_interval == 0:
            gen.eval()
            # Generate multiple samples
            preview_list = []
            preview_list.append({'caption': 'real', 'motion': _x_real[:1,:,:].cpu(), 'control': control.data.cpu()[:1,:,:,:], 'rotation': x_data[:1,:,:,3:6]})
            for k in range(3):
                z = Variable(gen.make_hidden(1, x_data.shape[2])).to(device) if cfg.models.generator.use_z else None
                fake_label = torch.randint(0, len(class_list), size=(1,)).type(torch.LongTensor).to(device)

                fake_rot, fake_bias, x_fake = gen(v_control[:1,:,:,:], z, fake_label)
                caption = cfg.train.dataset.class_list[fake_label[0].cpu().numpy()]
                # x_fake = rotate(x_fake, fake_rot, fake_bias, device=device)
                preview_list.append({'caption': caption, 'motion':x_fake.detach().cpu()[:1,:,:,:].numpy(), 'control': control.detach().cpu()[:1,:,:,:].numpy(), 'rotation':fake_rot.detach().cpu()[:1,:,:,:].numpy()})
                
            preview_path = os.path.join(cfg.train.out, 'preview', f'iter_{iteration+i+1}.pkl')
            # print(preview_path)
            if not os.path.exists(os.path.split(preview_path)[0]):
                os.makedirs(os.path.split(preview_path)[0])
            pickle.dump(preview_list, open(preview_path, "wb"))
            # print("save preview!")
            # save_video(preview_path, preview_list, cfg.train)
            # gen.train()

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
