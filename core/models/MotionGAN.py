import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

from core.models.layers import conv_layer, deconv_layer
from core.models.norm import PixelNormalizationLayer, AdaIN


#=================================================================================
#
###   Generator 
#
#=================================================================================

class MotionGAN_generator(nn.Module):
    def __init__(self, cfg, num_class):
        super(MotionGAN_generator, self).__init__()

        # Parameters for model initialization
        top = cfg.top
        padding_mode = cfg.padding_mode
        kw = cfg.kw
        w_dim = cfg.w_dim

        # Other settings
        self.cfg = cfg
        self.num_class = num_class
        self.z_dim = cfg.z_dim
        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.n_rows = 204
        self.num_adain = 14


        input_points = cfg.input_dim if hasattr(cfg, 'input_dim') else 3
        if input_points == 3:
            self.skip_n = [12,51]
            self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        elif input_points == 27:
            self.skip_n = [12,12]
            self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        elif input_points == 204:
            # TODO e1 > d1なのでrepeatが0.3とかにしないといけない
            self.skip_n = [2,2]
            self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_0 = conv_layer(top, top*2, ksize=(kw,7), stride=(2,6), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,6), stride=(2,5), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
            self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,6), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        else:
            raise ValueError("input points num must in [3, 27, 204]")

        self.dc_bottom = deconv_layer(top*8, top*8, ksize=(kw,3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 

        self.dc3_1 = deconv_layer(top*8, top*8, ksize=(kw,5), stride=(1,4), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc3_0 = conv_layer(top*8, top*4, ksize=(kw,2), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.dc2_1 = deconv_layer(top*8, top*4, ksize=(kw,6), stride=(1,4), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc2_0 = conv_layer(top*4, top*2, ksize=(kw,2), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc1_1 = deconv_layer(top*4, top*2, ksize=(kw,6), stride=(1,4), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc1_0 = conv_layer(top*2, top, ksize=(kw,2), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.dc0_1 = deconv_layer(top, top, ksize=(kw, 6), stride=(1,4), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.dc0_0 = conv_layer(top, 1, ksize=(kw,2), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_traj = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_rot = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_bias = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)

        # style generator
        self.latent_transform = LatentTransformation(cfg, num_class)

        self.adain_dc_bototm = AdaIN(top*8, w_dim)
        self.adain_dc3_1 = AdaIN(top*8, w_dim)
        self.adain_dc3_0 = AdaIN(top*4, w_dim)
        self.adain_dc2_1 = AdaIN(top*4, w_dim)
        self.adain_dc2_0 = AdaIN(top*2, w_dim)
        self.adain_dc1_1 = AdaIN(top*2, w_dim)
        self.adain_dc1_0 = AdaIN(top, w_dim)

        self.adain_ec3_1 = AdaIN(top*8, w_dim)
        self.adain_ec3_0 = AdaIN(top*8, w_dim)
        self.adain_ec2_1 = AdaIN(top*4, w_dim)
        self.adain_ec2_0 = AdaIN(top*4, w_dim)
        self.adain_ec1_1 = AdaIN(top*2, w_dim)
        self.adain_ec1_0 = AdaIN(top*2, w_dim)
        self.adain_ec0_0 = AdaIN(top, w_dim)
        self.num_adain = 14
            

    # Generate noise
    def make_hidden(self, size, frame_nums, y=None):
        mode = self.cfg.use_z
        if not mode:
            z = None
        elif mode == 'transform':
            z = torch.randn(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_const':
            z = torch.zeros(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_uniform':
            z = torch.rand(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor) * 2 - 1.0
        else:
            raise ValueError(f'Invalid noise mode \"{mode}\"!!')
        return z

    def inference_adain(self, z, labels, layer_name):
        w = self.latent_transform(z, labels)
        adain_layer = getattr(self, layer_name)
        adain_params = adain_layer.eval_params(w) 
        return adain_params 

    def homography(self, motion, rotation, bias):
        shape = motion.shape
        R = rotation.reshape(*shape[:-1], 3, 3)
        motion = torch.matmul(motion.reshape(*shape[:-1], 68, 3), R).view(shape)
        motion = motion + bias
        return motion


    def forward(self, control, z=None, labels=None, w=None):
        bs, _, ts, _ = control.shape 
        h = control 

        # transform z
        if w is None:
            w = self.latent_transform(z, labels)
        # duplicate w to input each AdaIN layer
        w = w.view(bs, 1, -1, 1, 1).expand(-1, self.num_adain, -1, -1, -1)

        ## Encoder
        e0 = self.activation(self.ec0_0(h))
        e0 = self.adain_ec0_0(e0, w[:,13])

        e1 = self.activation(self.ec1_0(e0))
        e1 = self.adain_ec1_0(e1, w[:,12])
        e1 = self.activation(self.ec1_1(e1))
        e1 = self.adain_ec1_1(e1, w[:,11])

        e2 = self.activation(self.ec2_0(e1))
        e2 = self.adain_ec2_0(e2, w[:,10])
        e2 = self.activation(self.ec2_1(e2))
        e2 = self.adain_ec2_1(e2, w[:,9])

        e3 = self.activation(self.ec3_0(e2))
        e3 = self.adain_ec3_0(e3, w[:,8])
        e3 = self.activation(self.ec3_1(e3))
        e3 = self.adain_ec3_1(e3, w[:,7])

        e3 = self.activation(self.dc_bottom(e3))
        e3 = self.adain_dc_bototm(e3, w[:,6])

        ## Decoder
        d2 = self.activation(self.dc3_1(F.interpolate(e3, scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d2 = self.adain_dc3_1(d2, w[:,5])
        d2 = self.activation(self.dc3_0(d2))
        d2 = self.adain_dc3_0(d2, w[:,4])
        # print(f"d2:{d2.shape}, e2: {e2.shape}")
        d1 = self.activation(self.dc2_1(F.interpolate(torch.cat((e2.repeat(1,1,1,self.skip_n[0])[:,:,:,:d2.shape[3]], d2), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d1 = self.adain_dc2_1(d1, w[:,3])
        d1 = self.activation(self.dc2_0(d1))
        d1 = self.adain_dc2_0(d1, w[:,2])
        # print(f"d1:{d1.shape}, e1: {e1.shape}")
        d0 = self.activation(self.dc1_1(F.interpolate(torch.cat((e1.repeat(1,1,1,self.skip_n[1])[:,:,:,:d1.shape[3]], d1), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d0 = self.adain_dc1_1(d0, w[:,1])
        d0 = self.activation(self.dc1_0(d0))
        d0 = self.adain_dc1_0(d0, w[:,0])
        # print(f"d0:{d0.shape}")

        ## Get trajectory
        traj = self.c_traj(d0)
        traj = traj.transpose(1,3)

        ## Get rotation
        rot = self.c_rot(d0)
        # rot = self.tanh(rot)
        rot = rot.transpose(1,3)

        ## Get bias
        bias = self.c_bias(d0)
        bias = bias.transpose(1,3)

        ## Get motion 
        motion = self.dc0_0(d0)
        # print(f"motion:{motion.shape}")

        # motion = self.homography(motion, rot, bias)
 
        return rot, bias, motion



#=================================================================================
#
###    Discriminator
#
#=================================================================================

class MotionGAN_discriminator(nn.Module):
    def __init__(self, cfg, frame_nums, num_class, pca=False):
        super(MotionGAN_discriminator, self).__init__()

        # Parameters for model initialization
        top = cfg.top
        padding_mode = cfg.padding_mode
        kw = cfg.kw
        norm = cfg.norm

        # Other settings
        self.num_class = num_class
        self.activation = nn.LeakyReLU()

        # If use normal GAN loss, last layer if FC and use sigmoid
        self.use_sigmoid = cfg.use_sigmoid if hasattr(cfg, 'use_sigmoid') else False
        pca = cfg.use_pca if hasattr(cfg, 'use_pca') else False
        # Layer structure
        ch = 1
        if pca:
            self.c0_0 = conv_layer(ch, top, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
            self.c1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,2), pad=(kw//2,1), normalize=norm, padding_mode=padding_mode)
            self.c2_0 = conv_layer(top*2, top*4, ksize=(kw,3), stride=(2,2), pad=(kw//2,1), normalize=norm, padding_mode=padding_mode)
            self.c3_0 = conv_layer(top*4, top*8, ksize=(kw,3), stride=(2,2), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
        else:
            self.c0_0 = conv_layer(ch, top, ksize=(kw,4), stride=(2,4), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
            self.c1_0 = conv_layer(top, top*2, ksize=(kw,4), stride=(2,4), pad=(kw//2,1), normalize=norm, padding_mode=padding_mode)
            self.c2_0 = conv_layer(top*2, top*4, ksize=(kw,4), stride=(2,4), pad=(kw//2,2), normalize=norm, padding_mode=padding_mode)
            self.c3_0 = conv_layer(top*4, top*8, ksize=(kw,4), stride=(2,4), pad=(kw//2,0), normalize=norm, padding_mode=padding_mode)
        if self.use_sigmoid:
            self.l_last = nn.Linear(top*frame_nums//2, 1) 
        else:
            if norm is not None and norm.startswith('spectral'):
                self.c_last = conv_layer(top*8, 1, ksize=(1,1), stride=(1,1), pad=(0,0), normalize='spectral', padding_mode=padding_mode)
            else:
                self.c_last = conv_layer(top*8, 1, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=None, padding_mode=padding_mode)

        self.l_cls = nn.Linear(top*frame_nums//2, num_class)
        self.softmax = nn.Softmax(dim=1)

        # Spectral normalization
        if norm is not None and norm.startswith('spectral'):
            if num_class > 0:
                self.l_cls = spectral_norm(self.l_cls)
            if self.use_sigmoid:
                self.l_last = spectral_norm(self.l_last)

        # Initialize linear layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.02)
       

    def forward(self, x, remove_softmax=False):
        bs = x.shape[0]
 
        h = x
        # print(h.shape)
        h = self.activation(self.c0_0(h))
        # print(h.shape)
        h = self.activation(self.c1_0(h))
        # print(h.shape)
        h = self.activation(self.c2_0(h))
        # print(h.shape)
        h = self.activation(self.c3_0(h))
        # print(h.shape)
        # raise ValueError("終わる")
        if self.use_sigmoid:
            out = torch.sigmoid(self.l_last(h.view(bs, -1)))
        else:
            out = self.c_last(h)

        if remove_softmax:
            cls = self.l_cls(h.view(bs, -1))
        else:
            cls = self.softmax(self.l_cls(h.view(bs, -1)))

        return out, cls


    def inference(self, x):
        h = x
        h = self.activation(self.c0_0(h))
        h = self.activation(self.c1_0(h))
        return h




class LatentTransformation(nn.Module):
    def __init__(self, cfg, num_class):
        super().__init__()
    
        self.z_dim = cfg.z_dim
        self.w_dim = cfg.w_dim
        self.normalize_z = PixelNormalizationLayer() if cfg.normalize_z else None

        activation = nn.LeakyReLU()

        self.latent_transform = nn.Sequential(
                conv_layer(self.z_dim*2, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation,
                conv_layer(self.w_dim, self.w_dim, ksize=(1,1), stride=(1,1), pad=(0,0)), 
                activation
        )

        self.label_embed = nn.Embedding(num_class, self.z_dim)


    def forward(self, z, labels):
        labels_embed = self.label_embed(labels).view([-1, self.z_dim, 1, 1])
        z = torch.cat([z.view([-1,self.z_dim,1,1]), labels_embed], dim=1)

        if self.normalize_z is not None:
            z = self.normalize_z(z)
     
        w = self.latent_transform(z)
        return w 


class PCAParam_generator(nn.Module):
    def __init__(self, cfg, num_class, pca_mean, pca_components):
        super(PCAParam_generator, self).__init__()

        # Parameters for model initialization
        top = cfg.top
        padding_mode = cfg.padding_mode
        kw = cfg.kw
        w_dim = cfg.w_dim

        # Other settings
        self.cfg = cfg
        self.num_class = num_class
        self.z_dim = cfg.z_dim
        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.n_rows = 27
        self.num_adain = 14
        input_points = cfg.input_dim if hasattr(cfg, 'input_dim') else 3

        if input_points == 3:
            self.skip_n = [6,13]
            self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        elif input_points == 27:
            self.skip_n = [2,2]
            self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        elif input_points == 204:
            # TODO e1 > d1なのでrepeatが0.3とかにしないといけない
            self.skip_n = [1,1]
            self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_0 = conv_layer(top, top*2, ksize=(kw,7), stride=(2,6), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,6), stride=(2,5), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
            self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,6), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
            self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        else:
            raise ValueError("input points num must in [3, 27, 204]")

        self.dc_bottom = deconv_layer(top*8, top*8, ksize=(kw,3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 

        self.dc3_1 = deconv_layer(top*8, top*8, ksize=(kw,2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc3_0 = conv_layer(top*8, top*4, ksize=(kw,3), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc2_1 = deconv_layer(top*8, top*4, ksize=(kw, 2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc2_0 = conv_layer(top*4, top*2, ksize=(kw,2), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc1_1 = deconv_layer(top*4, top*2, ksize=(kw, 2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc1_0 = conv_layer(top*2, top, ksize=(kw,2), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc0_1 = deconv_layer(top, top, ksize=(kw, 2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.dc0_0 = conv_layer(top, 1, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_rot = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_bias = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)


        # style generator
        self.latent_transform = LatentTransformation(cfg, num_class)

        self.adain_dc_bototm = AdaIN(top*8, w_dim)
        self.adain_dc3_1 = AdaIN(top*8, w_dim)
        self.adain_dc3_0 = AdaIN(top*4, w_dim)
        self.adain_dc2_1 = AdaIN(top*4, w_dim)
        self.adain_dc2_0 = AdaIN(top*2, w_dim)
        self.adain_dc1_1 = AdaIN(top*2, w_dim)
        self.adain_dc1_0 = AdaIN(top, w_dim)

        self.adain_ec3_1 = AdaIN(top*8, w_dim)
        self.adain_ec3_0 = AdaIN(top*8, w_dim)
        self.adain_ec2_1 = AdaIN(top*4, w_dim)
        self.adain_ec2_0 = AdaIN(top*4, w_dim)
        self.adain_ec1_1 = AdaIN(top*2, w_dim)
        self.adain_ec1_0 = AdaIN(top*2, w_dim)
        self.adain_ec0_0 = AdaIN(top, w_dim)
        self.num_adain = 14
            

    # Generate noise
    def make_hidden(self, size, frame_nums, y=None):
        mode = self.cfg.use_z
        if not mode:
            z = None
        elif mode == 'transform':
            z = torch.randn(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_const':
            z = torch.zeros(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_uniform':
            z = torch.rand(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor) * 2 - 1.0
        else:
            raise ValueError(f'Invalid noise mode \"{mode}\"!!')
        return z

    def inference_adain(self, z, labels, layer_name):
        w = self.latent_transform(z, labels)
        adain_layer = getattr(self, layer_name)
        adain_params = adain_layer.eval_params(w) 
        return adain_params 

    def homography(self, motion, rotation, bias):
        shape = motion.shape
        R = rotation.reshape(*shape[:-1], 3, 3)
        motion = torch.matmul(motion.reshape(*shape[:-1], 68, 3), R).view(shape)
        motion = motion + bias
        return motion


    def forward(self, control, z=None, labels=None, w=None):
        bs, _, ts, _ = control.shape 
        h = control 

        # transform z
        if w is None:
            w = self.latent_transform(z, labels)
        # duplicate w to input each AdaIN layer
        w = w.view(bs, 1, -1, 1, 1).expand(-1, self.num_adain, -1, -1, -1)

        ## Encoder
        e0 = self.activation(self.ec0_0(h))
        e0 = self.adain_ec0_0(e0, w[:,13])
        # print("e0", e0.shape)

        e1 = self.activation(self.ec1_0(e0))
        e1 = self.adain_ec1_0(e1, w[:,12])
        e1 = self.activation(self.ec1_1(e1))
        e1 = self.adain_ec1_1(e1, w[:,11])
        # print("e1", e1.shape)

        e2 = self.activation(self.ec2_0(e1))
        e2 = self.adain_ec2_0(e2, w[:,10])
        e2 = self.activation(self.ec2_1(e2))
        e2 = self.adain_ec2_1(e2, w[:,9])
        # print("e2", e2.shape)

        e3 = self.activation(self.ec3_0(e2))
        e3 = self.adain_ec3_0(e3, w[:,8])
        e3 = self.activation(self.ec3_1(e3))
        e3 = self.adain_ec3_1(e3, w[:,7])
        # print("e3", e3.shape)

        e3 = self.activation(self.dc_bottom(e3))
        e3 = self.adain_dc_bototm(e3, w[:,6])
        # print("e3", e3.shape)

        ## Decoder
        d2 = self.activation(self.dc3_1(F.interpolate(e3, scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d2 = self.adain_dc3_1(d2, w[:,5])
        d2 = self.activation(self.dc3_0(d2))
        d2 = self.adain_dc3_0(d2, w[:,4])
        # print(f"d2:{d2.shape}")
        d1 = self.activation(self.dc2_1(F.interpolate(torch.cat((e2.repeat(1,1,1,self.skip_n[0])[:,:,:,:d2.shape[3]], d2), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d1 = self.adain_dc2_1(d1, w[:,3])
        d1 = self.activation(self.dc2_0(d1))
        d1 = self.adain_dc2_0(d1, w[:,2])
        # print(f"d1:{d1.shape}")
        d0 = self.activation(self.dc1_1(F.interpolate(torch.cat((e1.repeat(1,1,1,self.skip_n[1])[:,:,:,:d1.shape[3]], d1), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d0 = self.adain_dc1_1(d0, w[:,1])
        d0 = self.activation(self.dc1_0(d0))
        d0 = self.adain_dc1_0(d0, w[:,0])
        # print(f"d0:{d0.shape}")


        ## Get rotation
        rot = self.c_rot(d0)
        rot = self.tanh(rot)
        rot = rot.transpose(1,3)

        ## Get bias
        bias = self.c_bias(d0)
        bias = bias.transpose(1,3)

        ## Get motion 
        motion = self.dc0_0(d0)
        
        # motion = self.homography(motion, rot, bias)
 
        return rot, bias, motion 


class restricted_generator(nn.Module):
    def __init__(self, cfg, num_class):
        super(restricted_generator, self).__init__()

        # Parameters for model initialization
        top = cfg.top
        padding_mode = cfg.padding_mode
        kw = cfg.kw
        w_dim = cfg.w_dim

        # Other settings
        self.cfg = cfg
        self.num_class = num_class
        self.z_dim = cfg.z_dim
        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.n_rows = 27
        self.num_adain = 14


        self.eclin = nn.Linear(204, 3)
        self.ec0_0 = conv_layer(1, top, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec1_0 = conv_layer(top, top*2, ksize=(kw,3), stride=(2,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec1_1 = conv_layer(top*2, top*2, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec2_0 = conv_layer(top*2, top*4, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec2_1 = conv_layer(top*4, top*4, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec3_0 = conv_layer(top*4, top*8, ksize=(kw,1), stride=(2,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.ec3_1 = conv_layer(top*8, top*8, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)

        self.dc_bottom = deconv_layer(top*8, top*8, ksize=(kw,3), stride=(1,3), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 

        self.dc3_1 = deconv_layer(top*8, top*8, ksize=(kw,2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc3_0 = conv_layer(top*8, top*4, ksize=(kw,3), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc2_1 = deconv_layer(top*8, top*4, ksize=(kw, 2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc2_0 = conv_layer(top*4, top*2, ksize=(kw,2), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc1_1 = deconv_layer(top*4, top*2, ksize=(kw, 2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode) 
        self.dc1_0 = conv_layer(top*2, top, ksize=(kw,2), stride=(1,1), pad=(kw//2,1), normalize=None, padding_mode=padding_mode)
        self.dc0_1 = deconv_layer(top, top, ksize=(kw, 2), stride=(1,2), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.dc0_0 = conv_layer(top, 1, ksize=(kw,1), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_rot = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.c_bias = conv_layer(top, 3, ksize=(kw,self.n_rows), stride=(1,1), pad=(kw//2,0), normalize=None, padding_mode=padding_mode)
        self.dclin = nn.Linear(27, 204)

        # style generator
        self.latent_transform = LatentTransformation(cfg, num_class)

        self.adain_dc_bototm = AdaIN(top*8, w_dim)
        self.adain_dc3_1 = AdaIN(top*8, w_dim)
        self.adain_dc3_0 = AdaIN(top*4, w_dim)
        self.adain_dc2_1 = AdaIN(top*4, w_dim)
        self.adain_dc2_0 = AdaIN(top*2, w_dim)
        self.adain_dc1_1 = AdaIN(top*2, w_dim)
        self.adain_dc1_0 = AdaIN(top, w_dim)

        self.adain_ec3_1 = AdaIN(top*8, w_dim)
        self.adain_ec3_0 = AdaIN(top*8, w_dim)
        self.adain_ec2_1 = AdaIN(top*4, w_dim)
        self.adain_ec2_0 = AdaIN(top*4, w_dim)
        self.adain_ec1_1 = AdaIN(top*2, w_dim)
        self.adain_ec1_0 = AdaIN(top*2, w_dim)
        self.adain_ec0_0 = AdaIN(top, w_dim)
        self.num_adain = 14
            

    # Generate noise
    def make_hidden(self, size, frame_nums, y=None):
        mode = self.cfg.use_z
        if not mode:
            z = None
        elif mode == 'transform':
            z = torch.randn(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_const':
            z = torch.zeros(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor)
        elif mode == 'transform_uniform':
            z = torch.rand(size, self.cfg.z_dim, 1, 1).type(torch.FloatTensor) * 2 - 1.0
        else:
            raise ValueError(f'Invalid noise mode \"{mode}\"!!')
        return z

    def inference_adain(self, z, labels, layer_name):
        w = self.latent_transform(z, labels)
        adain_layer = getattr(self, layer_name)
        adain_params = adain_layer.eval_params(w) 
        return adain_params 

    def homography(self, motion, rotation, bias):
        shape = motion.shape
        R = rotation.reshape(*shape[:-1], 3, 3)
        motion = torch.matmul(motion.reshape(*shape[:-1], 68, 3), R).view(shape)
        motion = motion + bias
        return motion


    def forward(self, control, z=None, labels=None, w=None):
        bs, _, ts, _ = control.shape 
        h = control 

        # transform z
        if w is None:
            w = self.latent_transform(z, labels)
        # duplicate w to input each AdaIN layer
        w = w.view(bs, 1, -1, 1, 1).expand(-1, self.num_adain, -1, -1, -1)

        ## Encoder
        e0 = self.eclin(h)
        # print(f"e0:{e0.shape}")
        e0 = self.activation(self.ec0_0(e0))
        e0 = self.adain_ec0_0(e0, w[:,13])

        e1 = self.activation(self.ec1_0(e0))
        e1 = self.adain_ec1_0(e1, w[:,12])
        e1 = self.activation(self.ec1_1(e1))
        e1 = self.adain_ec1_1(e1, w[:,11])

        e2 = self.activation(self.ec2_0(e1))
        e2 = self.adain_ec2_0(e2, w[:,10])
        e2 = self.activation(self.ec2_1(e2))
        e2 = self.adain_ec2_1(e2, w[:,9])

        e3 = self.activation(self.ec3_0(e2))
        e3 = self.adain_ec3_0(e3, w[:,8])
        e3 = self.activation(self.ec3_1(e3))
        e3 = self.adain_ec3_1(e3, w[:,7])

        e3 = self.activation(self.dc_bottom(e3))
        e3 = self.adain_dc_bototm(e3, w[:,6])

        ## Decoder
        d2 = self.activation(self.dc3_1(F.interpolate(e3, scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d2 = self.adain_dc3_1(d2, w[:,5])
        d2 = self.activation(self.dc3_0(d2))
        d2 = self.adain_dc3_0(d2, w[:,4])
        # print(f"d2:{d2.shape}")
        d1 = self.activation(self.dc2_1(F.interpolate(torch.cat((e2.repeat(1,1,1,6), d2), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d1 = self.adain_dc2_1(d1, w[:,3])
        d1 = self.activation(self.dc2_0(d1))
        d1 = self.adain_dc2_0(d1, w[:,2])
        # print(f"d1:{d1.shape}")
        d0 = self.activation(self.dc1_1(F.interpolate(torch.cat((e1.repeat(1,1,1,13), d1), dim=1), scale_factor=(2,1), mode='bilinear', align_corners=False)))
        d0 = self.adain_dc1_1(d0, w[:,1])
        d0 = self.activation(self.dc1_0(d0))
        d0 = self.adain_dc1_0(d0, w[:,0])
        # print(f"d0:{d0.shape}")

        ## Get bias
        bias = self.c_bias(d0)
        bias = bias.transpose(1,3)

        ## Get motion 
        motion = self.dc0_0(d0) 
        motion = self.dclin(motion)
        ## Get rotation
        rot = self.c_rot(d0)
        rot = self.tanh(rot)
        rot = rot.transpose(1,3)

        # print(f"motion:{motion.shape}")

        # motion = self.homography(motion, rot, bias)
        return rot, bias, motion 


    def RotationMatrix(self, theta):
        x = theta[..., 0]
        y = theta[..., 1]
        z = theta[..., 2]
        cos = torch.cos
        sin = torch.sin
        R1 = torch.stack([cos(y) * cos(z), sin(x) * sin(y) * cos(z) - cos(x) * sin(z), cos(x) * sin(y) * cos(z) + sin(x) * sin(z)], dim=-1)
        R2 = torch.stack([cos(y) * sin(z), sin(x) * sin(y) * sin(z) + cos(x) * cos(z), cos(x) * sin(y) * sin(z) - sin(x) * cos(z)], dim=-1)
        R3 = torch.stack([-sin(y),         sin(x) * cos(y), cos(x) * cos(y)], dim=-1)
        R = torch.stack([R1, R2, R3], dim=-2)
        return R

    def rotate(self, _motion, rotation, bias, device='cpu'):
        R = self.RotationMatrix(rotation).to(device)
        shape = _motion.shape
        motion = torch.matmul(_motion.reshape(*shape[:-1], 68, 3), R)
        motion = motion + bias.unsqueeze(-2).to(device)
        motion = motion.view(shape)
        return motion

