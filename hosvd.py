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

class Hosvid:
    def __init__(self):
        self.Ulist = []
        self.S = None
    
    def fit(self, A):
        S,Ulist = self.hosvd(A)
        self.Ulist = Ulist
        self.S = S
        return None
    
        
    def unfolding(self, n,A):
        shape = A.shape
        size = np.prod(shape)
        lsize = size // shape[n]
        sizelist = list(range(len(shape)))
        sizelist.pop(n)
        sizelist = [n] + sizelist
        return A.transpose(sizelist).reshape(shape[n],lsize)

    def modalsvd(self, n,A):
        nA = self.unfolding(n,A)
        return np.linalg.svd(nA, full_matrices=False)

    def modalsvd_trunc(self, n,A, rank):
        nA = self.unfolding(n,A)
        return randomized_svd(nA, n_components=rank, n_iter=5, random_state=None)

    def hosvd(self, A):
        Ulist = []
        S = A
        for i,ni in enumerate(A.shape):
            u,s,vh = self.modalsvd(i,A)
            Ulist.append(u)
            S = np.tensordot(S, u.T, ([0],[0]))
        return S,Ulist

    def build_T(self, axes, S=None):
        if S is None:
            S = self.S
        T = S
        for i in range(self.S.ndim):
            if i != axes:
                ind = list(range(1, self.S.ndim)) + [0]
                T = np.transpose(T, ind)
            else:
                U = self.Ulist[i]
                T =  np.tensordot(T, U, ([0],[0]))
        return T
    
    def approx(self, faces, ranks):
        Ulist = []
        S = faces
        for i,ni in enumerate(S.shape):
            u,s,vh = self.modalsvd_trunc(i,faces, ranks[i])
            u=u.T

            Ulist.append(u)
            S = np.tensordot(S, u.T, ([0],[0]))
        print(S.shape)
        T = S
        for i, rank in enumerate(ranks):
            U = Ulist[i]
            T = np.tensordot(T, U, ([0],[0]))
        self.S = S
        self.Ulist = Ulist
        return T

        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Argument Parser 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_args():
    parser = argparse.ArgumentParser(description='MotionGAN')
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--source', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    return args


#%---------------------------------------------------------------------------------------
def test():
    args = parse_args()
    source_dir = pathlib.Path(args.data_dir)
    train_dir = source_dir / "train"
    test_dir = source_dir / "test"
    result_dir = pathlib.Path(args.result_dir)
    if not result_dir.exists():
        result_dir.mkdir()
    npy_train_paths = source_dir.glob(f"01-01-0{args.source}-*.npy")
    emotion_list = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    source_emotion = args.source

    frame_step = 2
    print("loading train files...")
    train_motion = []
    for i in range(75):
        person_data = []
        for emo in emotion_list:
            data = np.load(train_dir / f"{emo}_{i}.npy")[:, :96:frame_step]
            if data.shape[1] != (96/frame_step):
                break
            person_data.append(data)
        else:
            train_motion.append(person_data)
    train_motion = np.array(train_motion)
    train_motion= train_motion[:, :,9:,:]
    train_motion = train_motion.transpose(1,0, 3, 2) * 0.01
    print("train data size: ", train_motion.shape)
    test_motion = np.array([[[np.load(test_dir / f"{emotion_list[source_emotion]}_01-01-0{source_emotion+1}-01-01-0{content}-{i}.npy")[:, :96:frame_step] for i in range(20, 25)]] for content in [1, 2]])
    test_motion= test_motion[:, :,:,9:]
    print("test data size: ", test_motion.shape)

    test_motion = test_motion.transpose(1,0,2, 4, 3).reshape(1, -1, int(96/frame_step), 204) * 0.01
    print("test data size: ", test_motion.shape)

    hosvd = Hosvid()
    hosvd.fit(train_motion)
    S, Ulist = hosvd.S, hosvd.Ulist
    T = S.copy()
    for i in range(S.ndim):
        if i == 1: # person
            ind = list(range(1, S.ndim)) + [0]
            T = np.transpose(T, ind)
        elif i == 0: # emotion
            U = Ulist[i] # neutral
            T =  np.tensordot(T, U, ([0],[0]))
        else: # 
            U = Ulist[i]
            T =  np.tensordot(T, U, ([0],[0]))
    T_exp = T #(1,19,100,204)
    T_exp_unf = T_exp[:1]
    T_exp_inv = np.linalg.pinv(hosvd.unfolding(1, T_exp_unf))
    u_p = np.dot(hosvd.unfolding(1, test_motion), T_exp_inv)
    Trans = np.tensordot(T_exp, u_p, ([1],[1]))
    Trans = Trans.reshape(*Trans.shape[:-1], 2,5)
    for content in [1,2]:
        for p_id, person in enumerate(range(20, 25)):
            for i, emotion in enumerate(emotion_list):
                fake_motion = Trans[i, :,:,content-1, p_id].T
                path = result_dir / f"01-01-0{source_emotion+1}-01-01-0{content}-{person}_fake_{emotion}.npy"
                np.save(path, fake_motion)
    print("done")


if __name__ == '__main__':
    test()
