import argparse
import pathlib
import numpy as np
import pandas as pd
import cv2
from warp_face import animate_face, search_best_frame, npy_to_gif, clop_face





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lm_csv", "-s", type=str, action="store", required=True)
    parser.add_argument("--target_lm_npy", "-t", type=str, action="store", required=True)
    parser.add_argument("--source_mp4", "-m", type=str, action="store", required=True)
    parser.add_argument("--z", "-z", type=float, action="store", default=None)
    parser.add_argument("--title", type=str, action="store", default=None)
    parser.add_argument("--out", type=str, action="store")
    args = parser.parse_args()


    source_csv_path = pathlib.Path(args.source_lm_csv)
    target_lm = np.load(args.target_lm_npy).reshape(68, 3, -1).transpose(2,0,1) # (T, 68, 3)
    z0 = args.z
    data = pd.read_csv(source_csv_path, sep=',+\s+', engine='python')
    frame_step = 2

    # source image のじゅんび
    cood_cols_2d = np.logical_or(data.columns.str.startswith("x_"),  data.columns.str.startswith("y_"))
    mot2d = np.array(data.loc[:, cood_cols_2d]).reshape(len(data),2,68) # T, 2, 68

    best_frame = search_best_frame(mot2d)

    cap = cv2.VideoCapture(args.source_mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
    ret, source_image = cap.read()
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    source_lm = mot2d[best_frame].T

    # target_lm の下処理
    tx = data['pose_Tx']
    ty = data['pose_Ty']
    tz = data['pose_Tz']
    rx = data['pose_Rx']
    ry = data['pose_Ry']
    rz = data['pose_Rz']
    tx = tx - tx.mean()
    ty = ty - ty.mean()
    tx = tz - tz.mean()
    rotation = np.array([rx, ry, rz])
    len_d = rotation.shape[1]
    rotation = rotation[:, :len_d-len_d%(frame_step*16):frame_step]

    target_lm = rotate(target_lm, rotation)

    if z0 is not None:
        target_lm = target_lm / (target_lm[:,:,2:] + z0)
    target_lm = target_lm[:,:,:2]

    source_image, source_lm = clop_face(source_image, source_lm, mergin=50)
    movie_arr = animate_face(src_face=source_image, src_face_lm=source_lm, dst_face_lm=target_lm, wrap=True)

    if args.title is not None:
        movie_arr = [
            cv2.putText(frame, args.title, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 1, 4)
        for frame in movie_arr]
    npy_to_gif(movie_arr, 'trans.gif')

    