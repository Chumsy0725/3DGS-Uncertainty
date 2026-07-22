import json
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch

from lpipsPyTorch import lpips_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def ause_torch(error: torch.Tensor, uncertainty: torch.Tensor) -> float:
    """
    AUSE (area under specification error)

    The area between two different AUSC curves (area under specification curve), which are defined as the mean error after filtering out a fraction of pixels.
    The two different AUSC curves are fist the one using the true pixel error to filter out pixels and the second uses an uncertainty measure.
    """
    err_vec = error.reshape(-1)
    unc_vec = uncertainty.reshape(-1)

    vec_mask = torch.logical_and(err_vec != 0.0, unc_vec != 0.0)
    if not torch.any(vec_mask):
        ause_err = torch.zeros(size=100, device=error.device)
        ause_err_by_var = torch.zeros(size=100, device=error.device)
        ause = 0.0
        return ause, ause_err, ause_err_by_var
    err_vec = err_vec[vec_mask]
    unc_vec = unc_vec[vec_mask]

    ratio_removed = torch.linspace(0, 0.999, 100, device=error.device)

    # AUSC for error
    err_vec_sorted, _ = torch.sort(err_vec)
    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ratio_idx = ((1-ratio_removed)*n_valid_pixels).to(int)[:-1]

    err_slices = torch.cumsum(err_vec_sorted, dim=0)[ratio_idx-1] / ratio_idx

    # AUSC for uncertainty
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]

    err_by_var_slices = torch.cumsum(err_vec_sorted_by_var, dim=0)[ratio_idx-1] / ratio_idx

    # Normalize and append
    # (normalize by start value and not by max value
    # to avoid low AUSE value due to a large tail of the AUSC for uncertainty)
    start_val = err_slices[0]
    ause_err = err_slices / start_val

    ause_err_by_var = err_by_var_slices / start_val

    ause = torch.trapz(ause_err_by_var - ause_err, ratio_removed[:len(ause_err)])

    return ause, ause_err, ause_err_by_var


def compute_uncertainty_metrics(uncertainty, error):
    # flatten
    uncertainty_vec = uncertainty.reshape(-1)
    error_vec = error.reshape(-1)

    # filter out nan values of error
    error_vec_mask = error_vec == error_vec
    error_vec = error_vec[error_vec_mask]
    uncertainty_vec = uncertainty_vec[error_vec_mask]

    # pearson correlation
    pearson = sp.stats.pearsonr(uncertainty_vec.detach().cpu().numpy(), error_vec.detach().cpu().numpy())

    # AUSE (area under specification error)
    ause, ause_err, ause_err_by_var = ause_torch(error_vec, uncertainty_vec)
    ause = ause.item()

    eval_dict = {
        "pearson": pearson.statistic,
        "AUSE": ause
    }
    return eval_dict, ause_err, ause_err_by_var


def save_rgb_uncertainty_plot(rgb_image: torch.Tensor, uncertainty: torch.Tensor, path: Path,
                               unc_vmin=None, unc_vmax=None):
    """Side-by-side visualization: rendered RGB image | uncertainty heatmap."""
    if unc_vmin is None or unc_vmax is None:
        valid = uncertainty[uncertainty == uncertainty]
        unc_vmin = valid.min().item() if unc_vmin is None else unc_vmin
        unc_vmax = valid.max().item() if unc_vmax is None else unc_vmax

    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    axs[0].imshow(torch.clamp(rgb_image, 0, 1).permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("RGB")
    axs[0].axis("off")

    im = axs[1].imshow(uncertainty.cpu().numpy(), cmap="turbo", vmin=unc_vmin, vmax=unc_vmax)
    axs[1].set_title("Uncertainty")
    axs[1].axis("off")
    fig.colorbar(im, ax=axs[1], orientation="horizontal")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    parser = ArgumentParser(description="Uncertainty Evaluation Parameters")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="path to the model output folder (contains 'renders/')")
    parser.add_argument("--renders_folder", type=str, default="renders", help="Folder name to store renders.")
    parser.add_argument("-s", "--split", type=str, default="test", help="data split to evaluate")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="creates side-by-side RGB+uncertainty visualizations for all eval views.")
    parser.add_argument("--unc_vmin", type=float, default=None, help="minimum color value for plot of uncertainty map")
    parser.add_argument("--unc_vmax", type=float, default=None, help="maximum color value for plot of uncertainty map")
    parser.add_argument("-f", "--uncertainty_folder", type=str, help="folder name for uncertainty maps, i.e. input/renders/split/uncertainty_folder", default="error_masks")
    args = parser.parse_args()

    base_path = Path(args.input)
    assert base_path.is_dir()

    path_gt = base_path / args.renders_folder / args.split / "original"
    path_pred = base_path / args.renders_folder / args.split / "render"
    path_uq = base_path / args.renders_folder / args.split / args.uncertainty_folder

    files_gt = sorted(path_gt.glob("*.npy"))
    files_pred = sorted([f for f in path_pred.glob("*.npy") if path_gt/f.name in files_gt])
    files_uq = sorted([f for f in path_uq.glob("*.npy") if path_gt/f.name in files_gt])
    assert len(files_gt) == len(files_pred) and len(files_gt) == len(files_uq)

    rgb_me_list_l1 = []
    metrics_rgb_l1 = []
    rgb_me_list_dssim = []
    metrics_rgb_dssim = []

    # GS quality metrics
    lpips = lpips_func("cuda", net_type='vgg')
    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0

    for i in tqdm(range(len(files_gt))):
        with open(files_gt[i], "rb") as f:
            gt_img = torch.tensor(np.load(f)).to("cuda")
        with open(files_pred[i], "rb") as f:
            pred_img = torch.tensor(np.load(f)).to("cuda")
        with open(files_uq[i], "rb") as f:
            uq_map = torch.tensor(np.load(f)).to("cuda")

        # compute GS quality metrics
        l1_test += l1_loss(pred_img, gt_img).mean().double()
        psnr_test += psnr(pred_img, gt_img).mean().double()
        ssim_test += ssim(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).mean().double()
        lpips.to(gt_img.device)
        lpips_test += lpips(pred_img, gt_img).mean().double()

        # Compute L1 error
        rgb_error_l1 = torch.abs(gt_img - pred_img).mean(dim=0)
        rgb_me_list_l1.append(float(rgb_error_l1[rgb_error_l1 == rgb_error_l1].mean().item()))

        metrics_dict_rgb_l1, _, _ = compute_uncertainty_metrics(uncertainty=uq_map, error=rgb_error_l1)
        metrics_rgb_l1.append(metrics_dict_rgb_l1)

        # Compute DSSIM error
        rgb_error_dssim = (1. - ssim(gt_img.unsqueeze(0), pred_img.unsqueeze(0), map=True).squeeze(0).mean(dim=0))
        rgb_me_list_dssim.append(float(rgb_error_dssim[rgb_error_dssim == rgb_error_dssim].mean().item()))

        metrics_dict_rgb_dssim, _, _ = compute_uncertainty_metrics(uncertainty=uq_map, error=rgb_error_dssim)
        metrics_rgb_dssim.append(metrics_dict_rgb_dssim)

        if args.plot:
            # Side-by-side RGB + uncertainty visualization (once per view)
            rgb_unc_plot_path = base_path / args.renders_folder / "eval_plots" / args.uncertainty_folder / args.split / f"{i:04d}_rgb_uncertainty.png"
            save_rgb_uncertainty_plot(pred_img, uq_map, path=rgb_unc_plot_path, unc_vmin=args.unc_vmin, unc_vmax=args.unc_vmax)

    # mean GS quality metrics
    psnr_test /= len(files_gt)
    l1_test /= len(files_gt)
    ssim_test /= len(files_gt)
    lpips_test /= len(files_gt)
    quality_metrics = {
        "l1": l1_test,
        "psnr": psnr_test,
        "ssim": ssim_test,
        "lpips": lpips_test
    }

    # Compute mean of metrics for L1
    metrics_dict_l1 = {"rgb_mean-error": np.mean(rgb_me_list_l1), "rgb_mean-error_std": np.std(rgb_me_list_l1)}
    metrics_dict_l1["uncertainty_metrics"] = {}
    for met in metrics_rgb_l1[0]:
        met_rgb_values_l1 = [met_dict[met] for met_dict in metrics_rgb_l1]
        metrics_dict_l1["uncertainty_metrics"][met] = np.mean(met_rgb_values_l1)
        metrics_dict_l1["uncertainty_metrics"][f"{met}_std"] = np.std(met_rgb_values_l1)

    # Compute mean of metrics for DSSIM
    metrics_dict_dssim = {"rgb_mean-error": np.mean(rgb_me_list_dssim), "rgb_mean-error_std": np.std(rgb_me_list_dssim)}
    metrics_dict_dssim["uncertainty_metrics"] = {}
    for met in metrics_rgb_dssim[0]:
        met_rgb_values_dssim = [met_dict[met] for met_dict in metrics_rgb_dssim]
        metrics_dict_dssim["uncertainty_metrics"][met] = np.mean(met_rgb_values_dssim)
        metrics_dict_dssim["uncertainty_metrics"][f"{met}_std"] = np.std(met_rgb_values_dssim)

    benchmark_info = {
        "experiment_name": base_path.name,
        "results": {
            "quality_metrics": quality_metrics,
            "L1": metrics_dict_l1,
            "DSSIM": metrics_dict_dssim
        }
    }

    # Convert all values in benchmark_info to standard Python types
    benchmark_info = json.loads(json.dumps(benchmark_info, default=lambda o: float(o) if isinstance(o, (np.floating, torch.Tensor)) else o))

    # Write output
    output_path = base_path / args.renders_folder / "eval" / args.split / f"uncertainty_metrics__{args.uncertainty_folder}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
    print(f"Save results to: {output_path}")
