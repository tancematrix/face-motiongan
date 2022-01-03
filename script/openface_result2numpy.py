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

import matplotlib.pyplot as plt
import matplotlib.animation as anm
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

def extract_face3d_forward(data):
    
    cood_cols = np.logical_or(np.logical_or(data.columns.str.startswith("X_"),  data.columns.str.startswith("Y_")), data.columns.str.startswith("Z_"))

    data_arr = np.array(data.loc[:, cood_cols]).reshape(len(data),3,68).transpose(0,2,1)

    tx = data['pose_Tx']
    ty = data['pose_Ty']
    tz = data['pose_Tz']
    rx = data['pose_Rx']
    ry = data['pose_Ry']
    rz = data['pose_Rz']
    tx = tx - tx.mean()
    ty = ty - ty.mean()
    tx = tz - tz.mean()
    rot = np.array([rx, ry, rz, tx, ty, tz]).T

    data_arr -= np.array([tx,ty,tz]).T[:, np.newaxis]

    rotation_mat = RotationMatrix([rx, ry, rz])
    mot = np.matmul(data_arr, np.linalg.inv(rotation_mat))
    return mot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "i", type=str, required=True, action="store")
    parser.add_argument("--outdir", "o", type=str, required=True, action="store")
    args = parser.parse_args()
    source_path = pathlib.Path(args.input)
    target_path = pathlib.Path(args.outdir)

    data = pd.read_csv(source_path, sep=',+\s+', engine='python')

    mot = extract_face3d_forward(data)
    # homo_vec = np.concatenate([rotation_mat.reshape(-1, 3), np.array([tx,ty,tz]).T.reshape(-1, 1)], axis=1)
    np.save(target_path /  f"{source_path.stem}.npy", mot)
