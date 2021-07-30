import glob
import pathlib 
import numpy as np
import os
import pandas as pd

# import imageio
import os
import glob
from matplotlib import pyplot as plt
import pickle as pkl
import time
from sklearn.linear_model import LinearRegression

from tqdm import tqdm
import argparse


class PCA():
    def __init__(self, pca_component, pca_mean):
        self.norm = np.linalg.norm(pca_component, axis=1)
        self.pca_component = pca_component
        self.pca_inverse = np.linalg.pinv(pca_component)# pca_component.T /  (self.norm**2)[None]
        self.pca_mean = pca_mean
        

    def transform(self, x):
        # x: (T, 204)
        # print(x.mean(axis=0), self.pca_mean)
        # from IPython import embed; embed()
        # lr.fit(x.mean(axis=0).reshape(68,3), self.pca_mean.reshape(68, 3))
        # bias, scale = lr.intercept_, lr.coef_[0]
        x = x.reshape(-1, 68,3)
        bias = x.mean(axis=(0,1)) - self.pca_mean.reshape(68, 3).mean(axis=0)
        x = x - bias
        scale = self.pca_mean.reshape(68,3) / x
        x = x * scale.mean()
        # # from IPython import embed; embed()
        # # print(bias, scale)
        
        x = x.reshape(-1, 204)
        return np.matmul(x - self.pca_mean, self.pca_inverse)
    
    def inverse_transform(self, x):
        return np.matmul(x, self.pca_component) + self.pca_mean


def convert_motion_to_pca(motion, pca):
    return pca.transform(motion.reshape(-1, 204))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, action="store")
    parser.add_argument("--target", type=str, action="store")
    parser.add_argument("--pca_dir", type=str, action="store")
    args = parser.parse_args()

    source_path = pathlib.Path(args.source)
    target_dir = pathlib.Path(args.target)
    pca_dir = pathlib.Path(args.pca_dir)
    
    if source_path.is_dir():
        files = list([f for f in source_path.glob("*.npy") if "rot" not in str(f)])
    elif source_path.is_file() and source_path.suffix == ".":
        files = [source_path]
    else:
        print("入力が間違っています。npyかディレクトリを指定してください")
    print(f"{len(files)} files proceccing...\n")

    pca_component = np.load(pca_dir / "components.npy")
    pca_mean = np.load(pca_dir / "mean.npy")
    # print(np.linalg.norm(pca_component, axis=0))
    # print(np.linalg.norm(pca_component, axis=1))

    pca = PCA(pca_component, pca_mean)


    for fname in tqdm(files):
        motion = np.load(fname)
        print(motion.shape)
        coded = convert_motion_to_pca(motion.reshape(-1, 204), pca)
        # np.save(target_dir / ("abetmp.npy"), coded)
        # break
        rec = np.matmul(coded, pca_component) + pca_mean
        # print(motion.reshape(-1, 204)*0.01 - rec)
        np.save(target_dir / (fname.stem+".npy"), coded)
        np.save(target_dir / (fname.stem+"rec.npy"), rec)


if __name__ == "__main__" :
    main()