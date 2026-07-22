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
#
# Plug-and-play predictive photometric uncertainty training.
#
# This script assumes a Gaussian Splatting scene has already been trained (see train.py)
# and fits an additional per-Gaussian "uncertainty" SH channel post-hoc, by regressing the
# photometric residual between the trained splat and the training images (least squares),
# optionally regularized with a Bayesian prior that pulls the uncertainty channel towards
# an isotropic value for all viewing directions (`--lambda_reg`).
#
# Dense captures (default): lambda_reg=0.0, i.e. the Bayesian regularizer is disabled.
# Sparse-view captures: pass --sparseTrainingViews and set --lambda_reg > 0 to enable it.

import os
import sys
import random
from random import randint
from argparse import ArgumentParser

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_errors as render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.loss_utils import l2_loss, l2_reg_loss_direct_on_shs, set_up_sh_matrix, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_sparse_training_views(scene, num_init_views=4):
    """Pick a small, spatially-diverse subset of training views: start with view 0, then
    greedily add the view furthest (in camera-center distance) from the current set."""
    all_cams = scene.train_cameras[1.0]
    init_views = [0]
    candidate_views = [i for i in range(len(all_cams)) if i not in init_views]
    train_cams = [all_cams[i] for i in init_views]
    candidate_cams = [all_cams[i] for i in candidate_views]

    for _ in range(num_init_views - len(init_views)):
        trainT = torch.stack([c.camera_center.cpu() for c in train_cams])
        candidateT = torch.stack([c.camera_center.cpu() for c in candidate_cams])
        dist_mat = torch.cdist(candidateT, trainT)
        candidate_min_dist = dist_mat.min(dim=1).values

        selected_idx = candidate_min_dist.argmax().item()
        init_views.append(candidate_views.pop(selected_idx))
        train_cams.append(candidate_cams.pop(selected_idx))

    return init_views


def render_and_save_outputs(dataset, gaussians, pipe, viewpoint_stack, folder_n="renders/test",
                             uncertain_background=False):
    """Render and save original/rendered/uncertainty outputs (as .npy, for uncertainty_metrics.py),
    plus a quick sanity-check combined figure per view."""
    print(f"\nRendering and saving outputs to {os.path.join(dataset.model_path, folder_n)} ...")

    for sub in ("original", "render", "error_masks", "combined"):
        os.makedirs(os.path.join(dataset.model_path, folder_n, sub), exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    error_mask_bg_color = [1, 1, 1] if uncertain_background else [0, 0, 0]
    error_mask_background = torch.tensor(error_mask_bg_color, dtype=torch.float32, device="cuda")

    for i in tqdm(range(len(viewpoint_stack)), desc="Rendering viewpoints"):
        viewpoint = viewpoint_stack[i]
        gt_image = viewpoint.original_image.to("cuda")

        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipe, background, error_mask_bg=error_mask_background)
        rendered_image = render_pkg["render"]
        error_mask = render_pkg["error_mask"]

        original_clipped = torch.clamp(gt_image, 0, 1)
        rendered_clipped = torch.clamp(rendered_image, 0, 1)

        np.save(os.path.join(dataset.model_path, folder_n, "original", f"{i:05d}.npy"), original_clipped.cpu().numpy())
        np.save(os.path.join(dataset.model_path, folder_n, "render", f"{i:05d}.npy"), rendered_clipped.cpu().numpy())

        error_mask_np = error_mask.cpu().numpy()
        if error_mask_np.ndim == 3:
            error_mask_np = error_mask_np.mean(axis=0)
        np.save(os.path.join(dataset.model_path, folder_n, "error_masks", f"{i:05d}.npy"), error_mask_np)

        error_mask_norm = ((error_mask_np - error_mask_np.min()) /
                            (error_mask_np.max() - error_mask_np.min() + 1e-8) * 255).astype(np.uint8)
        heatmap = cv2.cvtColor(cv2.applyColorMap(error_mask_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_clipped.permute(1, 2, 0).cpu().numpy()); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(rendered_clipped.permute(1, 2, 0).cpu().numpy()); axes[1].set_title('Rendered'); axes[1].axis('off')
        axes[2].imshow(heatmap); axes[2].set_title('Uncertainty'); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dataset.model_path, folder_n, "combined", f"{i:05d}_combined.png"),
                     dpi=150, bbox_inches='tight')
        plt.close()


def training(dataset, opt, pipe, iterations, save_iterations, error_sh_degree, lambda_reg, load_iteration,
             save_name="error", renders_folder="renders", skip_training=False,
             train_uncertain_background=False, uncertain_background=False):

    gaussians = GaussianModel(dataset.sh_degree)
    error_ply = os.path.join(dataset.model_path, "point_cloud", f"{save_name}_{iterations}", "point_cloud.ply")

    if skip_training and os.path.isfile(error_ply):
        print(f"Found existing uncertainty checkpoint at {error_ply}, skipping fitting.")
        scene = Scene(dataset, gaussians, load_iteration=iterations, load_type=save_name, shuffle=False)
        if dataset.sparseTrainingViews:
            print("Use sparse training views!")
            scene.train_idxs = select_sparse_training_views(scene)
        render_and_save_outputs(dataset, gaussians, pipe, scene.getTestCameras().copy(),
                                 f"{renders_folder}/test", uncertain_background)
        return

    scene = Scene(dataset, gaussians, load_iteration=load_iteration, load_type="iteration", shuffle=False)
    gaussians.init_change_feature(error_sh_degree)

    if dataset.sparseTrainingViews:
        print("Use sparse training views!")
        scene.train_idxs = select_sparse_training_views(scene)

    gaussians.training_setup_uncertainty(opt)

    tb_writer = SummaryWriter(dataset.model_path) if TENSORBOARD_FOUND else None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    error_bg_color = [1, 1, 1] if train_uncertain_background else bg_color
    error_background = torch.tensor(error_bg_color, dtype=torch.float32, device="cuda")

    train_cameras = scene.getTrainCameras()
    viewpoint_stack = train_cameras.copy()

    if lambda_reg != 0.0:
        B = set_up_sh_matrix(error_sh_degree, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(1, iterations + 1), desc="Fitting uncertainty")
    iteration = 0
    for iteration in range(1, iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if viewpoint_cam.error_map is None:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                 error_sh_degree=error_sh_degree, error_mask_bg=error_background)
            image, error_rendered = render_pkg["render"], render_pkg["error_mask"]

            gt_image = viewpoint_cam.original_image.to("cuda")
            error = torch.abs(gt_image - image).mean(dim=0) * (1. - opt.lambda_dssim) + \
                opt.lambda_dssim * (1. - ssim(gt_image.unsqueeze(0), image.unsqueeze(0), map=True).squeeze(0).mean(dim=0))
            viewpoint_cam.error_map = error.detach()
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, error_background,
                                 error_sh_degree=error_sh_degree, compute_render=False)
            error_rendered = render_pkg["error_mask"]
            error = viewpoint_cam.error_map

        loss_reg = l2_reg_loss_direct_on_shs(gaussians.get_change_feature, B) if lambda_reg != 0.0 else 0.0
        loss_least_squares = l2_loss(error, error_rendered)
        loss = loss_least_squares + lambda_reg * loss_reg
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if tb_writer:
                tb_writer.add_scalar('uncertainty/total_loss', loss.item(), iteration)
                tb_writer.add_scalar('uncertainty/least_squares_loss', loss_least_squares.item(), iteration)

            if iteration in save_iterations:
                print(f"\n[ITER {iteration}] Saving uncertainty-fitted Gaussians")
                scene.save_error(iteration, save_name)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
    progress_bar.close()

    render_and_save_outputs(dataset, gaussians, pipe, scene.getTestCameras().copy(),
                             f"{renders_folder}/test", uncertain_background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Uncertainty fitting parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.set_defaults(iterations=3000)
    parser.add_argument("--load_iteration", type=int, default=-1,
                        help="Iteration of the trained 3DGS model to load (-1: highest available).")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[],
                        help="Iterations at which to save the uncertainty-fitted checkpoint.")
    parser.add_argument("--error_sh_degree", type=int, default=3,
                        help="Spherical Harmonics degree for the uncertainty channel.")
    parser.add_argument("--lambda_reg", type=float, default=0.0,
                        help="Weight of the Bayesian regularizer over the uncertainty SH coefficients. "
                             "Defaults to 0.0 (disabled) for dense captures; set > 0 for sparse-view captures.")
    parser.add_argument("--save_name", type=str, default="error",
                        help="Name to save the Gaussian Splatting checkpoint with learned uncertainty channel.")
    parser.add_argument("--renders_folder", type=str, default="renders", help="Folder name to store renders.")
    parser.add_argument("--skip_training", action="store_true", default=False,
                        help="If a matching checkpoint already exists, skip fitting and only re-render/evaluate.")
    parser.add_argument("--train_uncertain_background", action="store_true", default=False,
                        help="Treat the background as high-uncertainty during training.")
    parser.add_argument("--uncertain_background", action="store_true", default=False,
                        help="Treat the background as high-uncertainty when rendering final outputs.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Fitting uncertainty for " + args.model_path)

    safe_state(args.quiet)
    set_seed(42)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.iterations, args.save_iterations,
              args.error_sh_degree, args.lambda_reg, args.load_iteration, args.save_name, args.renders_folder,
              args.skip_training, args.train_uncertain_background, args.uncertain_background)

    print("\nUncertainty fitting complete.")
