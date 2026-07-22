#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

from utils.sh_utils import eval_sh

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, map=False):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, map)

def _ssim(img1, img2, window, window_size, channel, size_average=True, map=False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average and not map:
        return ssim_map.mean()
    elif map:
        return ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


###############################################
# Predictive photometric uncertainty: Bayesian regularizer over the
# uncertainty SH coefficients (used by train_errors.py, --lambda_reg > 0,
# recommended for sparse-view captures).
###############################################
def set_up_sh_matrix(sh_degree, dtype=torch.float32, device="cpu"):
    # Reimplemented Gauss-Legendre / equiangular sampling to avoid the heavy and
    # fragile s2fft + jax dependency; dense-capture users who leave lambda_reg=0
    # still don't hit this path.
    sampling = "gl"
    L = sh_degree + 1

    # GL theta samples: arccos of flipped Legendre nodes on (-1, 1)
    theta_array = np.flip(np.arccos(np.polynomial.legendre.leggauss(L)[0]))
    # Equiangular phi samples on [0, 2pi)
    phi_array = 2 * np.pi * np.arange(2 * L - 1) / (2 * L - 1)
    theta, phi = np.meshgrid(theta_array, phi_array, indexing="ij")
    dirs = np.stack(
        [np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
         np.cos(theta)], axis=0
    )
    dirs = torch.tensor(dirs, dtype=dtype, device=device).reshape(3, -1).permute(1, 0)
    N = dirs.shape[0]

    D = L**2
    shs_view = torch.zeros((D,1,D), dtype=dtype, device=device)
    for i in range(D):
        shs_view[i,0,i] = 1.0

    B = torch.zeros((N,D), dtype=dtype, device=device)
    for i in range(N):
        sh2uq = eval_sh(sh_degree, shs_view, dirs[i:i+1].expand((D,3))).reshape(-1)
        B[i] = sh2uq

    return B


def l2_reg_loss_direct_on_shs(features, B):
    """
    Regularize the uncertainty SH coefficients towards a value leading to one for all
    viewing directions (i.e. towards isotropic uncertainty).
    """
    X = torch.matmul(features.reshape(-1, B.shape[1]), B.T)
    return l2_loss(X, 1.0)
