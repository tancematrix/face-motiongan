#!/usr/bin/env python
# coding: utf-8

#################################################################################
# このスクリプトはもともとjupyter notebookに書き捨てたものを若干整理しただけのものです
# 読みにくくてすみません。あと間違いがあるかもしれません、すみません。  竹内
#################################################################################

import pathlib 
import numpy as np
import os
import pandas as pd
import math
from scipy.spatial.transform import Rotation
# import imageio
import os
import glob
from matplotlib import pyplot as plt
import pickle as pkl
from sklearn.decomposition import PCA 
import re
import matplotlib.animation as anm

emotion_list = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# trainファイルを読み込んでPCAにかける
target_dir = pathlib.Path("/home/takeuchi/data/RAVDESS_processed/full_av/") # RAVDESSのデータ（三次元npyデータ）
pca_dir = pathlib.Path("/home/takeuchi/data/RAVDESS_processed/content_split/PCAdata_normalize/") # PCAの結果を格納するディレクトリ
rotation_dir = pathlib.Path("/home/takeuchi/data/RAVDESS_processed/Homography2/") # 正面を向いたmotionのデータと、回転・移動のデータ

files = list(target_dir.glob("train/*.npy"))
faces = []
for f in files:
    npy = np.load(f)
    npy = npy * 0.01 # 値を-1~1におさめるため
    faces.append(npy)
faces = np.concatenate(faces, axis=1)


pca = PCA(n_components=27) # n_componentsは適当な値をいれてください
pca.fit(faces.T)

transformed = []
motion_list = [[] for e in emotion_list]
content_list = [[] for e in emotion_list] # contentはRAVDESSデータセットの発話内容。1と2がある。

# fitさせたPCの空間にmotionを変換する
for f in emotion_train_files:
    motion = np.load(f) * 0.01
    rot = np.load(rotation_dir / f.name)
    rot[:, 3:] =- rot[:, 3:].mean(axis=0)
    rot[:, 3:] = rot[:, 3:] * 0.01
    coded = pca.transform(motion.T).T
    coded = np.concatenate([rot.T, coded]) # あまりいい設計とはいえないが、[回転, PCAで変換したmotion]というデータにしてGANの入力とする
    transformed.append(pca.transform(motion.T))
    for i in range(len(emotion_list)):
        if f.name.startswith(f"01-01-0{i+1}"):
            motion_list[i].append(coded)
            content_list[i].append(int(f.stem[-7:-6]))


transformed = np.concatenate(transformed)
transformed = transformed.reshape(-1, 27)
m, std = transformed.mean(axis=0), transformed.std(axis=0)




for motions, emotion, content in zip(motion_list, emotion_list, content_list):
    for i, m in enumerate(motions):
        m[6:] = m[6:] / std[:, None]
        if content[i] == 1:
            np.save(pca_dir / "train_1" / f"{emotion}_{i}", m) # 分散を揃える



for f in emotion_test_files:
    motion = np.load(f)
    motion = motion * 0.01
    rot = np.load(rotation_dir / f.name)
    rot[:, 3:] =- rot[:, 3:].mean(axis=0)
    rot[:, 3:] = rot[:, 3:] * 0.01
    coded = pca.transform(motion.T).T /  std[:, None]
    coded = np.concatenate([rot.T, coded])
    content = f.stem[-7:-6]
    np.save(pca_dir / f"test_{content}"/ f.name , coded)



np.save(pca_dir / "mean.npy", pca.mean_)
np.save(pca_dir / "components.npy", pca.components_ * std[:, None])
