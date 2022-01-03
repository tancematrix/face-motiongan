import argparse
import numpy as np
import PIL.Image
import cv2
from warp_face import animate_face
import moviepy.editor as mpy

def RotationMatrix(theta):
    x, y, z = theta
    cos = np.cos
    sin = np.sin
    R1 = np.array([cos(y) * cos(z), sin(x) * sin(y) * cos(z) - cos(x) * sin(z), cos(x) * sin(y) * cos(z) + sin(x) * sin(z)])
    R2 = np.array([cos(y) * sin(z), sin(x) * sin(y) * sin(z) + cos(x) * cos(z), cos(x) * sin(y) * sin(z) - sin(x) * cos(z)])
    R3 = np.array([-sin(y),         sin(x) * cos(y), cos(x) * cos(y)])
    R = np.stack([R1, R2, R3])
    return R

def rotate(motion, rotation):
    R = RotationMatrix(rotation)
    motion = np.matmul(motion, R.T)
    return motion



def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)
    clip.write_gif(filename)

def clop_face(source_image, source_lm, mergin=50):
    x_min, x_max = source_lm[:,0].min(), source_lm[:,0].max()
    y_min, y_max = source_lm[:,1].min(), source_lm[:,1].max()
    x_min = int(max(0, x_min-mergin))
    y_min = int(max(0, y_min-mergin))
    x_max = int(min(source_image.shape[1], x_max+mergin))
    y_max = int(min(source_image.shape[0], y_max+mergin))
    source_image = source_image[y_min:y_max,x_min:x_max]
    source_lm[:,0] = source_lm[:,0] - x_min
    source_lm[:,1] = source_lm[:,1] - y_min
    return source_image, source_lm

FPS = 30

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image", type=str, action="store")
    parser.add_argument("--source_lm", type=str, action="store")
    parser.add_argument("--rotation", type=str, action="store", default="")
    parser.add_argument("--target_lm", type=str, action="store")
    parser.add_argument("--out", type=str, action="store")
    args = parser.parse_args()

    fps = FPS

    source_image = np.array(PIL.Image.open(args.source_image).resize([512, 256]))
    print(source_image.shape)
    source_lm = np.load(args.source_lm).T # (68, 2)
    target_lm = np.load(args.target_lm).reshape(68, 3, -1).transpose(2,0,1) # (T, 68, 2)

    size = np.array([source_image.shape[1], source_image.shape[0]])
    source_lm = source_lm * size[None]
    source_image, source_lm = clop_face(source_image, source_lm, mergin=50)

    if args.rotation:
        rotation = np.load(args.rotation)
        target_lm = rotate(target_lm, rotation)[:,:,:2]
    else:
        target_lm = target_lm[:,:,:2]
    movie_arr = animate_face(src_face=source_image, src_face_lm=source_lm, dst_face_lm=target_lm)

    npy_to_gif(movie_arr, 'trans.gif')