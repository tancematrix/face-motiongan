"""
original code from https://github.com/marsbroshok/face-replace
"""

import skimage
import skimage.transform
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import PIL
import PIL.Image
import PIL.ImageFilter
import cv2
from matplotlib import pyplot as plt
import moviepy.editor as mpy


def _merge_images(img_top, img_bottom, mask=0, radius=0):
    """
    Function to combine two images with mask by replacing all pixels of img_bottom which
    equals to mask by pixels from img_top.
    :param img_top: greyscale image which will replace masked pixels
    :param img_bottom: greyscale image which pixels will be replace
    :param mask: pixel value to be used as mask (int)
    :return: combined greyscale image
    """


    img_top = skimage.img_as_ubyte(img_top)
    img_bottom = skimage.img_as_ubyte(img_bottom)
    merge_layer = (img_top == mask).astype(np.uint8)
    
    img_top = img_top * (1 - merge_layer)
    img_bottom = img_bottom * merge_layer
    return (img_top + img_bottom).astype(np.uint8)

    merge_layer = (np.prod(img_top == mask, axis=-1)*255).astype(np.uint8)
    # merge_layer = cv2.dilate(merge_layer, 5, 1)
    merge_layer = PIL.Image.fromarray(merge_layer).convert('L')
    # merge_layer = merge_layer.filter(PIL.ImageFilter.MaxFilter(3))
    # merge_layer = merge_layer.filter(PIL.ImageFilter.MaxFilter(7))
    img_top = PIL.Image.fromarray(img_top)
    img_bottom = PIL.Image.fromarray(img_bottom)
    # merge_layer = merge_layer.filter(PIL.ImageFilter.GaussianBlur(3))
    # img = PIL.Image.composite(img_bottom,img_top, merge_layer)
    return np.array(img_top)



def face_warp(src_face, src_face_lm, dst_face_lm, composite=True):
    """
    Function takes two faces and landmarks and warp one face around another according to
    the face landmarks.
    :param src_face: grayscale image (np.array of int) of face which will warped around second face
    :param src_face_lm: landmarks for the src_face
    :param dst_face: grayscale image (np.array of int) which will be replaced by src_face.
                     Landmarks to the `dst_face` will be calculated each time from the image.
    :return: image with warped face
    """
    # Helpers
    output_shape = src_face.shape  # dimensions of our final image (from webcam eg)

    # Get the landmarks/parts for the face.
    # try:
    warp_trans = skimage.transform.PiecewiseAffineTransform()
    
    warp_trans.estimate(dst_face_lm, src_face_lm)
    warped_face = skimage.transform.warp(src_face, warp_trans, output_shape=output_shape)
    mouth_pts = dst_face_lm[60:67]
    cv2.fillPoly(warped_face, [mouth_pts.reshape((-1,1,2)).astype(np.int32)], (0.05, 0.02, 0.02))

    # Merge two images: new warped face and background of dst image
    # through using a mask (pixel level value is 0)
    if composite:
        warped_face = _merge_images(warped_face, src_face)
    else:
        warped_face = skimage.img_as_ubyte(warped_face)
    # from IPython import embed; embed()
    return warped_face

def fit_dst_lms_to_src(dst_face_lm, src_face_lm):
    num_frame = dst_face_lm.shape[0]
    lr = LinearRegression(positive=True)
    for i in range(2):
        X = dst_face_lm[:,:,i].flatten()[:, None]
        Y = np.repeat(src_face_lm[None,:,i], num_frame, axis=0).flatten()[:, None]
        lr.fit(X, Y)
        coef = lr.coef_[0]
        shift = lr.intercept_
        dst_face_lm[:,:,i] = dst_face_lm[:,:,i] * coef + shift
    # print(f"coef: {coef}, shift: {shift}")
    return dst_face_lm

def wrap_lm(lm_2d):
    wrap_lm_ = (lm_2d[[0, 4, 8, 12, 16, 26, 17]] + lm_2d[[4, 8, 12, 16, 26, 17, 0]]) / 2.
    diff_lm = (lm_2d[[0, 4, 8, 12, 16, 26, 17]] - lm_2d[[4, 8, 12, 16, 26, 17, 0]]) / 2.
    wrap_lm_[:,0] += diff_lm[:,1]
    wrap_lm_[:,1] -= diff_lm[:,0]
    return wrap_lm_

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

"""
目が空いていて口もある程度開いているようなフレームを探してくる
"""
def search_best_frame(data_arr):
    eye_th = (data_arr[:, 1, 40]-data_arr[:, 1, 38]).max() * 0.8

    for i in range(data_arr.shape[0]):
        if (data_arr[i, 1, 40]-data_arr[i, 1, 38]) > eye_th and (data_arr[i, 1, 46]-data_arr[i, 1, 44])> eye_th and 5 < (data_arr[i, 1, 66]-data_arr[i, 1, 62]) < 10:
            return i
    print("not found, continue..")
    for i in range(data_arr.shape[0]):
        print((data_arr[i, 1, 40]-data_arr[i, 1, 38]), (data_arr[i, 1, 46]-data_arr[i, 1, 44]), (data_arr[i, 1, 66]-data_arr[i, 1, 62]) )
        if (data_arr[i, 1, 40]-data_arr[i, 1, 38]) > eye_th and (data_arr[i, 1, 46]-data_arr[i, 1, 44])> eye_th and 5 < (data_arr[i, 1, 66]-data_arr[i, 1, 62]) < 15:
            return i
    raise ValueError("not found")

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

def animate_face(src_face, src_face_lm, dst_face_lm, wrap=False):
    # src_face: image
    # dst_face_lm: (T, 68, 2) :landmark sequence
    # src_face_lm: (68, 2) : lamdmark
    # size = np.array([src_face.shape[1], src_face.shape[0]])
    # src_face_lm = src_face_lm * size[None]
    num_frame = dst_face_lm.shape[0]
    out_movie_arr = np.zeros([num_frame, *src_face.shape], dtype=np.uint8)
    dst_face_lm = fit_dst_lms_to_src(dst_face_lm, src_face_lm)
    if wrap:
        wrap_lm_ = wrap_lm(src_face_lm)
        src_face_lm = np.concatenate([src_face_lm, wrap_lm_])
        dst_face_lm = np.concatenate([dst_face_lm, np.tile(wrap_lm_, (num_frame, 1, 1))], axis=1)
    
    for frame_id in tqdm(range(num_frame)):
        out_movie_arr[frame_id] = face_warp(src_face, src_face_lm, dst_face_lm[frame_id], composite=True)
    
    return out_movie_arr

def animate_faces(src_faces, src_face_lms, dst_face_lm, wrap=False):
    # src_face: images
    # dst_face_lm: (T, 68, 2) :landmark sequence
    # src_face_lms: (T, 68, 2) : lamdmark
    # size = np.array([src_face.shape[1], src_face.shape[0]])
    # src_face_lm = src_face_lm * size[None]
    num_frame, h, w, _ = src_faces.shape
    out_movie_arr = np.zeros([num_frame, *src_faces[0].shape], dtype=np.uint8)
    dst_face_lm = fit_dst_lms_to_src(dst_face_lm, src_face_lms[0])
    if wrap:
        # wrap_lm_ = wrap_lm(src_face_lms[0])
        # wrap_lm_ = np.concatenate([wrap_lm_, np.array([[0,0], [0,h], [w,0], [h,w]])])
        wrap_lm_ = np.zeros([16, 2], dtype=np.int)
        wrap_lm_[4:8, 1] = np.linspace(0,h,num=4,dtype=np.int)
        wrap_lm_[12:16, 1] = np.linspace(0,h,num=4,dtype=np.int)
        wrap_lm_[12:16, 0] = w
        wrap_lm_[0:4, 0] = np.linspace(0,w,num=4,dtype=np.int)
        wrap_lm_[8:12, 0] = np.linspace(0,w,num=4,dtype=np.int)
        wrap_lm_[8:12, 1] = h
        src_face_lms = np.concatenate([src_face_lms, np.tile(wrap_lm_, (num_frame, 1, 1))], axis=1)
        dst_face_lm = np.concatenate([dst_face_lm, np.tile(wrap_lm_, (num_frame, 1, 1))], axis=1)
    
    for frame_id in tqdm(range(num_frame)):
        out_movie_arr[frame_id] = face_warp(src_faces[frame_id], src_face_lms[frame_id], dst_face_lm[frame_id], composite=False)
    
    return out_movie_arr