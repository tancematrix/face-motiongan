import argparse
from sys import path

import cv2
import pathlib
import pandas as pd
import numpy as np
from warp_face import animate_face, animate_faces, search_best_frame, npy_to_gif, clop_face, rotate, face_warp, fit_dst_lms_to_src

parser = argparse.ArgumentParser()
parser.add_argument("--mp4", type=str, action="store")
parser.add_argument("--out_dir", type=str, action="store")
args = parser.parse_args()

source_csv_path = pathlib.Path(f"/home/takeuchi/research/nishi_lab/RAVDESS/csv/")
source_mp4_dir = f"/home/takeuchi/research/nishi_lab/RAVDESS/Actor_20/"
gt_dir = pathlib.Path("/home/takeuchi/research/nishi_lab/pytorch_results/Homography/test/")
out_dir = pathlib.Path(args.out_dir)

cap = cv2.VideoCapture(args.mp4)
frame_step = 2
stop = 48
pid = 20
z0 = 4

DEBUG = False

emotion_list = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

for emo_id in range(8):
    emo = emotion_list[emo_id]
    data = pd.read_csv(source_csv_path / f"01-01-0{emo_id+1}-01-01-01-{pid}.csv", sep=',+\s+', engine='python')
    cood_cols_2d = np.logical_or(data.columns.str.startswith("x_"),  data.columns.str.startswith("y_"))

    source_lms = np.array(data.loc[:, cood_cols_2d]).reshape(len(data),2,68).transpose(0,2,1)[:stop*2:frame_step]

    target_lms = np.load(gt_dir / f"{emo}_01-01-0{emo_id+1}-01-01-01-{pid}.npy")[9:, :stop*2:frame_step]
    target_lms = target_lms.reshape(68, 3, -1).transpose(2,0,1) * 0.01

    target_lms = target_lms / (target_lms[:,:,2:] + z0)
    target_lms = target_lms[:,:,:2]
    if DEBUG:
        stop = 2
        target_lms = target_lms[:2]
        source_lms = source_lms[:2]
    

    for frame_id in range(stop):
        cap = cv2.VideoCapture(source_mp4_dir + f"01-01-0{emo_id+1}-01-01-01-{pid}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id*frame_step)
        ret, source_image = cap.read()
        
        source_image, src_face_lm = clop_face(source_image, source_lms[frame_id], mergin=0)
        h, w, _ = source_image.shape
        target_lm = fit_dst_lms_to_src(target_lms, src_face_lm)[frame_id]
    
        wrap_lm_ = np.array([[-1,-1], [-1,h+1], [w+1,-1], [h+1,w+1]])
        src_face_lm = np.concatenate([src_face_lm, wrap_lm_])
        target_lm = np.concatenate([target_lm, wrap_lm_])
    
        out_img = face_warp(source_image, src_face_lm, target_lm)[10:-10,10:-10]
        cv2.imwrite(str(out_dir / f"01-01-0{emo_id+1}-01-01-01-{pid}_frame_{frame_id}.jpg"), out_img)
    break