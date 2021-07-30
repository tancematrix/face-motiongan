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
from fastdtw import fastdtw
import seaborn as sns
import scipy
import matplotlib
from sklearn.utils.extmath import randomized_svd
import json
import argparse

def adjust_alignment(align):
    res = []
    ind = 0
    for (l, r) in align:
        if l == ind:
            res.append(r)
            ind += 1
    return res

def deriv(arr):
    firstderiv = arr[2:] - arr[1:-1]
    secondderiv = (arr[2:] - arr[:-2]) / 2.
    return (firstderiv + secondderiv) / 2.

def load_gt_data(path):
    frame_step = 2
    motions_gt = np.load(path)[:, :96:frame_step]

def calcurate_dist(gt_motion, fake_motion, emotion_list):
    data = []
    for content_id in [0,1]:
        for person_id in range(5):
            for gt_i, gt_emotion in enumerate(emotion_list):
                for fake_i, fake_emotion in enumerate(emotion_list):
                    size = np.prod(gt_motion.shape[-2:])
                    gt_mot = gt_motion[content_id, gt_i,person_id ].reshape(-1, size)
                    fake_mot = fake_motion[content_id, fake_i, person_id ].reshape(-1, size)
                    result_dic = {}
                    result_dic["content"] = content_id
                    result_dic["p_id"] = person_id
                    result_dic["gt"] = gt_emotion
                    result_dic["trans"] = fake_emotion
                    distance, _ = fastdtw(gt_mot.T, fake_mot.T, dist=2)
                    result_dic["dtw"] = distance
                    gt_d = deriv(gt_mot.T)
                    fake_d = deriv(fake_mot.T)
                    
                    distance, align = fastdtw(gt_d, fake_d, dist=2)
                    result_dic["ddtw"] = distance
                    l2 = np.linalg.norm(gt_mot - fake_mot, axis=-1).mean()
                    result_dic["l2"] = l2
                    result_dic["dl2"] = np.linalg.norm(gt_d - fake_d)
                    data.append(result_dic)        
    df = pd.DataFrame(data)
    return df

def eval_dist_matrix(df, metrics):
    dist_mat = pd.pivot_table(df, values=metrics, columns='gt', index='trans').to_numpy()
    dist_diag = np.diag(dist_mat)
    dist_others = dist_mat.sum(axis=-1) - dist_diag
    dist_others = dist_others.mean() / (dist_mat.shape[0]-1)
    return dist_diag.mean(), dist_others

def rank_dist_matrix(df, metrics):
    dist_mat = pd.pivot_table(df, values=metrics, columns='gt', index='trans')
    mean_rank = np.diag(dist_mat.rank(axis=1).to_numpy()).mean()
    return mean_rank

def get_lip_motion(face_motion): 
    height = face_motion[:,:,:,:, 61:64] - face_motion[:,:,:,:, 65:68]
    # print(height.shape)
    height = np.abs(height.mean(axis=4, keepdims=True))
    width = face_motion[:,:,:,:, 60:61] - face_motion[:,:,:,:, 64:65]
    # width = width[:,0]
    return np.concatenate([height, width], axis=-2)



def main(args):
    path_to_gt = pathlib.Path("/home/takeuchi/data/RAVDESS_processed/Homography/test/")
    transfer_paths = json.load(open(args.result_list, "r"))
    emotion_list = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    frame_step = 2
    gt_motion = np.array([[[np.load(path_to_gt / f"{emotion_list[emo_id]}_01-01-0{emo_id+1}-01-01-0{content}-{i}.npy")[:, :96:frame_step] \
                            for i in range(20, 25)] \
                                for emo_id in range(len(emotion_list))] \
                                    for content in [1,2]])
    gt_motion= gt_motion[:, :,:,9:]
    gt_motion = gt_motion.transpose(0,1, 2, 4,3).reshape(2,8,5,48,68,3) * 0.01
    gt_motion = get_lip_motion(gt_motion)

    methods = ["neutral", "content_swap"] + transfer_paths
    for method in methods:
        if method == "neutral":
            neutral_gt = gt_motion[:, :1]
            fake_motion = np.repeat(gt_motion[:, :1], 8, axis=1)
        elif method == "content_swap":
            fake_motion = gt_motion[[1,0]]
        else:
            path_to_preview = pathlib.Path(method)
            fake_motion = np.array([[[np.load(path_to_preview / f"01-01-01-01-01-0{content}-{i}_fake_{emo}.npy") for i in range(20, 25)] for emo in emotion_list] for content in [1,2]])
            fake_motion = fake_motion.reshape(2, 8,5,68,3,48).transpose(0,1, 2, 5, 3, 4)
            fake_motion = get_lip_motion(fake_motion)
        df = calcurate_dist(gt_motion, fake_motion, emotion_list)
        if method == "neutral":
            df = df.query('gt != "neutral" & trans != "neutral"')
        print(method)
        eval_data = []
        for c_id in [0, 1]:
            mean_df = df.query(f"content == {c_id}").groupby(["gt", "trans"]).agg("mean")
            dic1 = {}
            dic2 = {}
            dic3 = {}
            dic1["content"] = c_id
            dic2["content"] = c_id
            dic1["style"] = "target"
            dic2["style"] = "other"
            # dic3["content"] = c_id
            # dic3["style"] = "rank"

            for metric in ["l2", "dtw", "ddtw"]:
                dic1[metric] , dic2[metric]= eval_dist_matrix(mean_df, metric)
                # dic3[metric] = rank_dist_matrix(mean_df, metric)
                
            eval_data += [dic1, dic2]#, dic3]
        eval_data = pd.DataFrame(eval_data)
        print(eval_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_list", action="store", required=True)
    parser.add_argument("--out", action="store", default="evalations.txt")
    args = parser.parse_args()
    main(args)

