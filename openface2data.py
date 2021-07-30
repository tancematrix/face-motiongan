import glob
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
import time

from tqdm import tqdm
import argparse


def RotationMatrix(theta) :
    x = theta[0]
    y = theta[1]
    z = theta[2]
    cos = np.cos
    sin = np.sin
    R = np.array([[cos(y) * cos(z), sin(x) * sin(y) * cos(z) - cos(x) * sin(z), cos(x) * sin(y) * cos(z) + sin(x) * sin(z)],
                    [cos(y) * sin(z), sin(x) * sin(y) * sin(z) + cos(x) * cos(z), cos(x) * sin(y) * sin(z) - sin(x) * cos(z)],
                    [-sin(y),         sin(x) * cos(y), cos(x) * cos(y)]
                    ])          
    return R.T

def convert_data(fname):
    data = pd.read_csv(fname, sep=',+\s+', engine='python')
    tx = data['pose_Tx']
    ty = data['pose_Ty']
    tz = data['pose_Tz']
    rx = data['pose_Rx']
    ry = data['pose_Ry']
    rz = data['pose_Rz']
    # tx = tx - tx.mean()
    # ty = ty - ty.mean()
    # tx = tz - tz.mean()
    cood_cols = np.logical_or(np.logical_or(data.columns.str.startswith("X_"),  data.columns.str.startswith("Y_")), data.columns.str.startswith("Z_"))
    data_arr = np.array(data.loc[:, cood_cols]).reshape(len(data),3,68).transpose(0,2,1)
    
    rotation_mat = RotationMatrix([rx, ry, rz])
    data_arr -= np.array([tx,ty,tz]).T[:, np.newaxis]
    mot = np.matmul(data_arr, np.linalg.inv(rotation_mat))
   
    return mot, np.array([rx, ry, rz])

def convert_data2d(fname):
    data = pd.read_csv(fname, sep=',+\s+', engine='python')
    cood_cols = np.logical_or(data.columns.str.startswith("x_"),  data.columns.str.startswith("y_"))
    data_arr = np.array(data.loc[:, cood_cols]).reshape(len(data),2,68).transpose(0,2,1)
    return data_arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, action="store")
    parser.add_argument("--target", type=str, action="store")
    parser.add_argument("--use2d", action="store_true", help="二次元でのlandmark")
    args = parser.parse_args()

    source_path = pathlib.Path(args.source)
    target_dir = pathlib.Path(args.target)
    
    if source_path.is_dir():
        files = list(source_path.glob("*.csv"))
    elif source_path.is_file() and source_path.suffix == ".csv":
        files = [source_path]
    else:
        print("入力が間違っています。csvかディレクトリを指定してください")
    print(f"{len(files)} files proceccing...\n")
    for fname in tqdm(files):
        if args.use2d:
            mot = convert_data2d(fname)
            np.save(target_dir / (fname.stem+".npy"), mot)
        else:
            mot, rot = convert_data(fname)
            np.save(target_dir / (fname.stem+".npy"), mot)
            np.save(target_dir / (fname.stem+"_rot.npy"), rot)


if __name__ == "__main__" :
    main()