import os, sys, glob
import random
import pickle
import math

import numpy as np
import torch
from torch.utils.data import Dataset

import core.utils.motion_utils as motion_utils
import core.utils.bvh_to_joint as btoj




class BVHDataset(Dataset):
    def __init__(self, cfg, mode='train', from_paths=None):
        self.cfg = cfg
        self.datalist, self.indexes = collect_motion_datalist(cfg, mode=mode, npy_paths=from_paths)
        self.mode = mode

    def __len__(self):
        return len(self.indexes)


    def __getitem__(self, i):
        motion_id, start_frame = self.indexes[i]
        data = self.datalist[motion_id]
        label = data['label'] if self.cfg.class_list else None
        
        if self.mode == 'train' and self.cfg.augment_fps:
            # Augument fps with choice from [x0.5, x1.0, x1.5]
            fps = min(np.random.choice([0.5,1.0,1.5]), (data['motion'].shape[0]-start_frame-1)/self.cfg.frame_nums)
            frame_step = int(self.cfg.frame_step * fps)
            frame_nums = int(self.cfg.frame_nums * (frame_step / self.cfg.frame_step))

            # Cut fixed length sequence from motion
            motion_part = data['motion'][start_frame:start_frame+frame_nums:frame_step, :]

            if len(motion_part) < frame_nums // frame_step:
                motion_part = np.pad(motion_part, (0, frame_nums - len(motion_part)), "edge")
            # Get trajectory from motion
            trajectory = data['motion'][start_frame:start_frame+frame_nums,:3].copy()
            if len(trajectory) < frame_nums:
                trajectory = np.pad(trajectory, (0, frame_nums - len(trajectory)), "edge")

            # Re-calcurate spline f(t) with modified fps
            spline_f = motion_utils.interpolate_spline(trajectory, control_point_interval=self.cfg.control_point_interval)
            spline_length_map = motion_utils.get_spline_length(trajectory, spline_f, 1.0)

            # Sampling control signal
            if list(spline_length_map.keys())[-1] + 1 < frame_nums:
                # Avoid error by interpolating bilinearly
                xp = list(spline_length_map.keys())[-1] + 1
                control_part = motion_utils.sampling(trajectory, spline_f, spline_length_map, 1.0, startT=0, endT=xp, step=frame_step, with_noise=True)
                control_part = np.stack([np.interp(np.arange(frame_nums), np.arange(control_part.shape[0]), control_part[:,0]),
                                        np.interp(np.arange(frame_nums), np.arange(control_part.shape[0]), control_part[:,1]),
                                        np.interp(np.arange(frame_nums), np.arange(control_part.shape[0]), control_part[:,2])], axis=1)
            else:
                control_part = motion_utils.sampling(trajectory, spline_f, spline_length_map, 1.0, startT=0, endT=frame_nums, step=frame_step, with_noise=True)
            # control_part[:,1] = control_part[:,1] - control_part[:,1]

        elif self.mode == 'test':
            # Get maximum length motion
            motion = data['motion']
            motion_part = motion[:motion.shape[0]-motion.shape[0]%(self.cfg.frame_step*16):self.cfg.frame_step,:]
            
            if list(data['spline_length_map'].keys())[-1] * 0.1 < motion_part.shape[0] * self.cfg.frame_step:
                motion_part = motion_part[:-16,:]
            trajectory_part = motion_part[:,:3] 
            control_part = motion_utils.sampling(trajectory_part, data['spline_f'], data['spline_length_map'], 0.1, startT=0, endT=motion_part.shape[0]*self.cfg.frame_step, step=self.cfg.frame_step, with_noise=False)
            # control_part[:,1] = control_part[:,1] - control_part[:,1]

        else:
            # Cut fixed length sequence from motion
            motion_part = data['motion'][start_frame:start_frame+self.cfg.frame_nums:self.cfg.frame_step, :]
            # Get trajectory from motion
            trajectory = data['motion'][start_frame:start_frame+self.cfg.frame_nums,:3].copy()
            # Sampling control signal
            control_part = motion_utils.sampling(trajectory, data['spline_f'], data['spline_length_map'], 1.0, startT=0, endT=self.cfg.frame_nums, step=self.cfg.frame_step, with_noise=True)
            # control_part[:,1] = control_part[:,1] - control_part[:,1]


        # rotation
        if self.mode == 'train' and self.cfg.rotate:
            theta = np.random.rand() * 2. * math.pi
            motion_part = motion_utils.rotate(motion_part, theta)
            control_part = motion_utils.rotate(control_part, theta)

        # scaling normalization
        motion_part /= self.cfg.scale
        control_part /= self.cfg.scale 

        return motion_part, control_part, label



def collect_motion_datalist(cfg, sampling_interval=3, mode='train', npy_paths=None):

    # Collect .npy path
    if npy_paths is None:
        npy_paths = []
        for (root, dirs, files) in os.walk(cfg.data_root):
            for npy_dir in dirs:
                if npy_dir.find('spline') > -1:
                    continue
                npy_paths.extend(glob.glob(os.path.join(root, npy_dir, '*.npy')))
            if not npy_paths:
                for npy_file in files:
                    if npy_file.endswith('.npy'):
                        npy_paths.append(os.path.join(root, npy_file))
        if mode == 'test':
            npy_paths.sort()


    datalist = []
    indexes = []
    motion_count = 0

    # Register label
    label_dict = {}
    for l,name in enumerate(cfg.class_list):
        label_dict[name] = l

    # Prepare skeleton
    if cfg.standard_bvh is not None:
        skeleton, non_end_bones, joints_to_index, permute_xyz_order = btoj.get_standard_format(cfg.standard_bvh)
        _, non_zero_joint_to_index = btoj.cut_zero_length_bone(skeleton, joints_to_index)

    # Create data
    for i, npy_path in enumerate(npy_paths):
        # Create path to save inclusive data (.pkl)
        top, name = os.path.split(npy_path)
        name, ext = os.path.splitext(name)
        data_path = os.path.join(top, f'processed_cp{cfg.control_point_interval}', name+'.pkl')
        if not os.path.exists(os.path.split(data_path)[0]): os.makedirs(os.path.split(data_path)[0]) 

        # Get label information from path
        style = os.path.splitext(os.path.split(npy_path)[1])[0].split('_')[0]
        if style in label_dict:
            label = label_dict[style]
        elif not cfg.class_list:
            label = []
        else:
            continue

        # pickle file contains {motion, trajectory, spline_f, spline_length_map}
        # if pickle file exists, load it, otherwise create new pickle file
        if os.path.exists(data_path):
            with open(data_path, mode='rb') as f:
                data = pickle.load(f)
        elif cfg.standard_bvh is not None:
            data = create_data_from_npy(cfg, npy_path, data_path, skeleton, joints_to_index)
            if data is None: continue
        else:
            data = create_data_from_processed_npy(cfg, npy_path, data_path)
            if data is None:
                print("ignore one file for some err.")
                continue

        # Register label info
        data['label'] = label

        # Add data to list
        if mode == 'train':
            if data['motion'].shape[0] > cfg.frame_nums:
                datalist.append(data)
                for j in range((data['motion'].shape[0] - cfg.frame_nums) // sampling_interval):
                    indexes.append(tuple((motion_count, j * sampling_interval)))
            else:
                continue
        else:
            datalist.append(data)
            indexes.append(tuple((motion_count, 0)))
        motion_count += 1


    if mode == 'train':
        random.shuffle(indexes)
    return datalist, indexes



def create_data_from_npy(cfg, npy_path, data_path, skeleton, joints_to_index):
    # load motion
    motion = np.load(npy_path)
    if len(motion.shape) != 2:
        motion = np.reshape(motion, (motion.shape[0], motion.shape[1]*3))
    # Cut zero from motion
    _, motion = btoj.cut_zero_length_bone_frames(motion, skeleton, joints_to_index) 

    motion = motion[cfg.start_offset:,:]
    # In advance, calcurate spline function and hasmap between t and length of each motion
    trajectory = motion[:,:3].copy()
    trajectory = np.concatenate([np.concatenate([trajectory[:,0:1], np.zeros((trajectory.shape[0],1))], axis=1), trajectory[:,2:3]], axis=1)
    spline_f = motion_utils.interpolate_spline(trajectory, control_point_interval=cfg.control_point_interval)
    if spline_f is None:
        return None 
         
    # Get length map for sampling
    spline_length_map = motion_utils.get_spline_length(trajectory, spline_f, dt=0.1)
    data = {'motion':motion, 'trajectory':trajectory, 'spline_f':spline_f, 'spline_length_map':spline_length_map}

    # Create new pickle file
    with open(data_path, mode='wb') as f:
        pickle.dump(data, f)
        print(f'Create {data_path}  {motion.shape}')

    return data

def create_data_from_processed_npy(cfg, npy_path, data_path):
    # load motion
    motion = np.load(npy_path).T
    # motion = motion[cfg.start_offset:,:]

    # In advance, calcurate spline function and hasmap between t and length of each motion
    trajectory = motion[:,:3].copy()
    spline_f = motion_utils.interpolate_spline(trajectory, control_point_interval=cfg.control_point_interval)
    if spline_f is None:
        return None 
         
    # Get length map for sampling
    spline_length_map = motion_utils.get_spline_length(trajectory, spline_f, dt=0.1)
    data = {'motion':motion, 'trajectory':trajectory, 'spline_f':spline_f, 'spline_length_map':spline_length_map}

    # Create new pickle file
    if data_path is not None:
        with open(data_path, mode='wb') as f:
            pickle.dump(data, f)
            print(f'Create {data_path}  {motion.shape}')
    return data

