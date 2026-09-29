import os
import sys
import argparse
import shutil
import warnings
import multiprocessing

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import wandb

from utils import (
    get_config,
    prepare_sub_folder,
    get_data_loader,
    save_image_3d,
    save_image_3d_slices,
    _save_raw_and_npy_from_tensor,
)
from Line_geometry_projector import LineCT3DProjector
from skimage.metrics import structural_similarity as compare_ssim
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_psnr(pred, target, data_range=None):
    """
    Compute PSNR using 10 * log10(MAX^2 / MSE).

    Args:
        pred: Predicted tensor or NumPy array.
        target: Reference tensor or NumPy array.
        data_range: Signal dynamic range. If None, it is estimated from target.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()
    else:
        pred = torch.as_tensor(pred)
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu()
    else:
        target = torch.as_tensor(target)

    if data_range is None:
        data_range = float(target.max() - target.min())
    data_range = max(data_range, 1e-8)

    mse = torch.mean((pred - target) ** 2).item()
    psnr = 10 * np.log10((data_range ** 2) / (mse + 1e-10))
    return psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phantom4_simupcb_Line_gaussian.yaml',
        help='Path to the config file.',
    )
    parser.add_argument('--output_path', type=str, default='.', help='Output path.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to a checkpoint for resuming training.',
    )
    parser.add_argument(
        '--real_projs',
        type=str,
        default=None,
        help='Path to real projections (.npy or .npz).',
    )
    parser.add_argument(
        '--real_projs_key',
        type=str,
        default=None,
        help='Array key for .npz input. The first array is used by default.',
    )
    return parser.parse_args()


def weighted_proj_mse(pred, target, bg_thr=0.03, bg_weight=0.5, eps=1e-8):
    """
    Weighted projection-domain MSE.

    Args:
        pred: Predicted projections with shape [B, P, H, W].
        target: Target projections with shape [B, P, H, W].
        bg_thr: Background threshold.
        bg_weight: Weight applied to background regions.
    """
    with torch.no_grad():
        weight = torch.ones_like(target)
        weight[target < bg_thr] = bg_weight
    loss_map = weight * (pred - target) ** 2
    return loss_map.sum() / (weight.sum() + eps)


class OptimizationParams:
    def __init__(self):
        self.position_lr_init = 0.005
        self.position_lr_final = 0.00002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.intensity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

    def as_dict(self):
        return {
            "position_lr_init": self.position_lr_init,
            "position_lr_final": self.position_lr_final,
            "position_lr_delay_mult": self.position_lr_delay_mult,
            "position_lr_max_steps": self.position_lr_max_steps,
            "intensity_lr": self.intensity_lr,
            "scaling_lr": self.scaling_lr,
            "rotation_lr": self.rotation_lr,
            "percent_dense": self.percent_dense,
        }


def line_awtv_regularization(
    image,
    tau,
    alpha_h=1.0,
    alpha_w=1.0,
    alpha_d=1.0,
    delta_cl=0.0,
):
    """Geometry-adaptive TV regularization for linear CL."""
    tau = max(float(tau), 1e-8)
    alpha_h = float(alpha_h)
    alpha_w = float(alpha_w)
    alpha_d = float(alpha_d)
    delta_cl = float(delta_cl)

    # image shape: [B, D, H, W, C]
    grad_h = torch.zeros_like(image)
    grad_w = torch.zeros_like(image)
    grad_d = torch.zeros_like(image)

    grad_h[:, :, 1:, :, :] = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
    grad_w[:, :, :, 1:, :] = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
    grad_d[:, 1:, :, :, :] = image[:, 1:, :, :, :] - image[:, :-1, :, :, :]

    def g_func(t_abs):
        g0 = (
            torch.cos(torch.pi * torch.clamp(t_abs / tau, max=1.0)) + 1.0
        ) * 0.5
        return delta_cl + (1.0 - delta_cl) * g0

    gh = torch.abs(grad_h)
    gw = torch.abs(grad_w)
    gd = torch.abs(grad_d)

    wh = alpha_h * g_func(gh)
    ww = alpha_w * g_func(gw)
    wd = alpha_d * g_func(gd)

    tv_map = wh * gh + ww * gw + wd * gd
    return torch.mean(tv_map)


def projection_dssim_loss(pred_projs, gt_projs, window_size=7):
    """
    Projection-domain DSSIM loss.

    Args:
        pred_projs: Predicted projections with shape [B, P, U, V].
        gt_projs: Target projections with shape [B, P, U, V].
        window_size: Local averaging window size.

    Returns:
        Scalar loss equal to 1 - SSIM.
    """
    if window_size % 2 == 0:
        window_size += 1

    padding = window_size // 2
    pred = pred_projs.reshape(
        -1, 1, pred_projs.shape[-2], pred_projs.shape[-1]
    )
    gt = gt_projs.reshape(
        -1, 1, gt_projs.shape[-2], gt_projs.shape[-1]
    )

    mu_pred = F.avg_pool2d(
        pred, kernel_size=window_size, stride=1, padding=padding
    )
    mu_gt = F.avg_pool2d(
        gt, kernel_size=window_size, stride=1, padding=padding
    )

    sigma_pred = (
        F.avg_pool2d(
            pred * pred,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )
        - mu_pred * mu_pred
    )
    sigma_gt = (
        F.avg_pool2d(
            gt * gt,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )
        - mu_gt * mu_gt
    )
    sigma_pred_gt = (
        F.avg_pool2d(
            pred * gt,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )
        - mu_pred * mu_gt
    )

    data_range = torch.clamp(gt.max() - gt.min(), min=1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = (
        (2 * mu_pred * mu_gt + c1)
        * (2 * sigma_pred_gt + c2)
        / (
            (mu_pred * mu_pred + mu_gt * mu_gt + c1)
            * (sigma_pred + sigma_gt + c2)
            + 1e-8
        )
    )
    ssim_value = torch.clamp(ssim_map.mean(), min=0.0, max=1.0)
    return 1.0 - ssim_value


@torch.no_grad()
def add_poisson_noise_to_projs(projs, n0=1e6, eps=1e-8):
    """
    Add Poisson noise in the transmission domain.

    Args:
        projs: Line-integral projections with shape [B, P, U, V].
        n0: Incident photon count.

    The model is I = I0 * exp(-p), counts ~ Poisson(I), and
    p_noisy = -log(counts / I0).
    """
    n0 = max(float(n0), 1.0)
    p = torch.clamp(projs.to(dtype=torch.float32), min=0.0)
    intensity = torch.exp(-p)
    lam = torch.clamp(intensity * n0, min=0.0)
    counts = torch.poisson(lam)
    intensity_noisy = torch.clamp(counts / n0, min=eps)
    return -torch.log(intensity_noisy)


def main():
    opts = parse_args()

    # When resuming, prefer the config saved with the checkpoint experiment.
    if opts.config is None or (
        opts.checkpoint is not None and os.path.exists(opts.checkpoint)
    ):
        if opts.checkpoint is not None:
            checkpoint_dir = os.path.dirname(os.path.abspath(opts.checkpoint))
            candidate_config = os.path.join(
                os.path.dirname(checkpoint_dir), 'config.yaml'
            )
            if os.path.exists(candidate_config):
                opts.config = candidate_config
                print(f"[Config] Auto use checkpoint config: {opts.config}")
            else:
                opts.config = 'configs/Line_gaussian.yaml'
                print(
                    "[Config] config.yaml was not found beside the checkpoint; "
                    f"fall back to: {opts.config}"
                )
        else:
            opts.config = 'configs/Line_gaussian.yaml'

    config = get_config(opts.config)

    max_iter = config['max_iter']
    checkpoint_save_iter = config.get('checkpoint_save_iter', 1000)

    # Line-TAAwTV parameters.
    line_awtv_tau = float(config.get('line_awtv_tau', 0.02))
    line_awtv_alpha_h = config.get('line_awtv_alpha_h', None)
    line_awtv_alpha_w = config.get('line_awtv_alpha_w', None)
    line_awtv_alpha_d = config.get('line_awtv_alpha_d', None)
    line_awtv_beta_h = float(config.get('line_awtv_beta_h', 0.0))
    line_awtv_beta_w = float(config.get('line_awtv_beta_w', 0.0))
    line_awtv_beta_d = float(config.get('line_awtv_beta_d', 0.0))
    line_awtv_weight = float(config.get('line_awtv_weight', 1.0))
    line_awtv_weight_schedule = bool(
        config.get('line_awtv_weight_schedule', True)
    )
    line_awtv_weight_start = float(
        config.get('line_awtv_weight_start', line_awtv_weight)
    )
    line_awtv_weight_end = float(
        config.get('line_awtv_weight_end', line_awtv_weight)
    )
    line_awtv_weight_iters = int(
        config.get('line_awtv_weight_iters', 500)
    )
    line_awtv_delta_cl = float(config.get('line_awtv_delta_cl', 0.0))
    line_awtv_delta_schedule = bool(
        config.get('line_awtv_delta_schedule', False)
    )
    line_awtv_delta_start = float(
        config.get('line_awtv_delta_start', line_awtv_delta_cl)
    )
    line_awtv_delta_end = float(
        config.get('line_awtv_delta_end', line_awtv_delta_cl)
    )
    line_awtv_delta_iters = int(
        config.get('line_awtv_delta_iters', line_awtv_weight_iters)
    )

    lambda_dssim = float(config.get('lambda_dssim', 0.0))
    dssim_window = int(config.get('dssim_window', 7))

    # Projection-domain background weighting.
    proj_bg_weight_enabled = bool(
        config.get('proj_bg_weight_enabled', False)
    )
    proj_bg_thr = float(config.get('proj_bg_thr', 0.03))
    proj_bg_weight = float(config.get('proj_bg_weight', 0.5))

    # Optional Poisson noise for simulated projections.
    simu_poisson_enabled = bool(
        config.get('simu_poisson_enabled', False)
    )
    simu_poisson_n0 = float(config.get('simu_poisson_n0', 1e6))
    simu_poisson_seed = config.get('simu_poisson_seed', None)

    # Sampling-aware Gaussian scale-space filtering.
    smooth_filter_enabled = bool(
        config.get('smooth_filter_enabled', True)
    )
    smooth_filter_s = float(config.get('smooth_filter_s', 0.2))
    smooth_filter_interval = int(
        config.get('smooth_filter_interval', 100)
    )
    smooth_filter_margin = float(
        config.get('smooth_filter_margin', 0.15)
    )
    smooth_filter_min_depth = float(
        config.get('smooth_filter_min_depth', 1e-4)
    )
    smooth_filter_last_iters = int(
        config.get('smooth_filter_last_iters', 0)
    )

    if smooth_filter_last_iters < 0:
        smooth_filter_enabled = False
        smooth_filter_start_iter = max_iter
    elif 'smooth_filter_start_iter' in config:
        smooth_filter_start_iter = int(
            config.get('smooth_filter_start_iter')
        )
    elif smooth_filter_last_iters > 0:
        smooth_filter_start_iter = max(
            0, max_iter - smooth_filter_last_iters
        )
    else:
        smooth_filter_start_iter = 0

    smooth_filter_start_iter = max(
        0, min(smooth_filter_start_iter, max_iter)
    )

    cudnn.benchmark = True

    # Reuse the original output directory when resuming from a checkpoint.
    checkpoint_output_directory = None
    if opts.checkpoint is not None and os.path.exists(opts.checkpoint):
        checkpoint_dir = os.path.dirname(os.path.abspath(opts.checkpoint))
        checkpoint_output_directory = os.path.dirname(checkpoint_dir)

    output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    output_subfolder = config['data']
    dataset_name = config.get('dataset_name', '')

    if checkpoint_output_directory is not None:
        output_directory = checkpoint_output_directory
    elif str(config.get('data', '')).lower() == 'realdata':
        output_directory = os.path.join(
            opts.output_path,
            'realdata_outputs',
            output_folder,
            dataset_name,
        )
    else:
        output_directory = os.path.join(
            opts.output_path,
            'simu_outputs',
            output_folder,
            dataset_name,
            output_subfolder,
        )

    checkpoint_directory, image_directory = prepare_sub_folder(
        output_directory
    )

    # Save terminal output to a log file.
    log_path = os.path.join(output_directory, 'train_log.txt')
    log_file = open(log_path, 'a', encoding='utf-8')

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"[Log] Save terminal log to: {log_path}")
    print(
        f"[GSF] enabled={smooth_filter_enabled}, "
        f"start_iter={smooth_filter_start_iter}, "
        f"s={smooth_filter_s}, interval={smooth_filter_interval}"
    )
    print(
        f"[SimuPoisson] enabled={simu_poisson_enabled}, "
        f"n0={simu_poisson_n0}"
    )
    print(
        f"[Line-TAAwTV] weight={line_awtv_weight}, "
        f"tau={line_awtv_tau}, "
        f"beta_h={line_awtv_beta_h}, "
        f"beta_w={line_awtv_beta_w}, "
        f"beta_d={line_awtv_beta_d}"
    )

    if simu_poisson_seed is not None:
        try:
            seed_val = int(simu_poisson_seed)
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)
            np.random.seed(seed_val)
            print(f"[SimuPoisson] use seed={seed_val}")
        except Exception as e:
            print(f"[SimuPoisson] invalid seed {simu_poisson_seed}: {e}")

    dst_config_path = os.path.join(output_directory, 'config.yaml')
    src_config_path = os.path.abspath(opts.config)
    dst_config_abs = os.path.abspath(dst_config_path)
    if src_config_path != dst_config_abs:
        shutil.copy(src_config_path, dst_config_path)
    else:
        print(
            "[Config] Source and target config are the same file; "
            f"skip copy: {dst_config_abs}"
        )

    wandb.init(
        project="",
        name="",
        config=config,
        mode=os.environ.get("WANDB_MODE", "disabled"),
    )
    # Projection file can be specified by CLI or by the 'realproj_npy' config key.
    real_projs_path = opts.real_projs if opts.real_projs is not None else config.get('realproj_npy', None)
    real_projs_key = opts.real_projs_key if hasattr(opts, 'real_projs_key') and opts.real_projs_key is not None else config.get('real_proj_key', None)

    print('Load image: {}'.format(config['img_path']))
    data_loader = get_data_loader(
        config['data'],
        config['img_path'],
        config['img_size'],
        img_slice=None,
        train=True,
        batch_size=config['batch_size'],
        num_workers=0
    )

    # Load real projections if provided. Expected shape: [P, U, V].
    real_projs = None
    if real_projs_path is not None and os.path.exists(real_projs_path):
        print(f"[RealProjs] Loading real projections from: {real_projs_path}")
        try:
            loaded = np.load(real_projs_path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                # Select the requested array key.
                if real_projs_key and real_projs_key in loaded:
                    arr = loaded[real_projs_key]
                else:
                    # Otherwise use the first array in the .npz file.
                    first_key = list(loaded.files)[0]
                    arr = loaded[first_key]
            else:
                arr = loaded

            arr = np.asarray(arr)
            # Flip detector columns when the measured horizontal direction is opposite to ASTRA u.
            if bool(config.get('real_proj_flip_horizontal', False)):
                arr = arr[:, :, ::-1].copy()  # [P, det_rows, det_cols]

            if arr.ndim == 3:
                real_projs = arr.astype(np.float32)
            else:
                raise RuntimeError(f"Real projections array must be 3D (P,U,V), got shape {arr.shape}")
            print(f"[RealProjs] Loaded array shape: {real_projs.shape}")
        except Exception as e:
            print(f"[RealProjs] Failed to load projections: {e}")
            real_projs = None

    config['img_size'] = (config['img_size'], config['img_size'], config['img_size']) if type(config['img_size']) == int else tuple(config['img_size'])
    slice_idx = list(range(0, config['img_size'][0], int(config['img_size'][0]/config['display_image_num'])))
    all_slice_idx = list(range(0, config['img_size'][0]))
    if config['num_proj'] > config['display_image_num']:
        proj_idx = list(range(0, config['num_proj'], int(config['num_proj']/config['display_image_num'])))
    else:
        proj_idx = list(range(0, config['num_proj']))
    all_proj_idx = list(range(0, config['num_proj']))

    # Read linear CT geometry parameters
    source_det = config.get('source_det', 800)
    source_origin = config.get('source_origin', 100)
    dis_step = config.get('dis_step', 0.59)
    det_spacing = config.get('det_spacing', 2.0)
    voxel_spacing = config.get('voxel_spacing', 0.15)

    # Compute geometry-adaptive Line-TAAwTV directional weights.
    source_origin_safe = max(float(source_origin), 1e-8)
    num_proj = int(config['num_proj'])
    # Compute the maximum half coverage angle using the source trajectory.
    s_max = ((num_proj - 1) / 2.0) * float(dis_step)
    theta = np.arctan(s_max / source_origin_safe)
    s_theta = theta / (np.pi / 2.0)
    # Geometry-based alpha for H: alpha_h = 1 + beta_h*(1 - s_theta)
    alpha_h_geom = 1.0 + float(line_awtv_beta_h) * (1.0 - s_theta)
    alpha_h_final = float(alpha_h_geom if line_awtv_alpha_h is None else line_awtv_alpha_h)
    # Geometry-based alpha for W: same formula as H, but controlled by beta_w
    alpha_w_geom = 1.0 + float(line_awtv_beta_w) * (1.0 - s_theta)
    alpha_w_final = float(alpha_w_geom if line_awtv_alpha_w is None else line_awtv_alpha_w)

    i_arr = np.arange(num_proj, dtype=np.float64)
    s_i = (i_arr - (num_proj - 1) / 2.0) * float(dis_step)

    phi_arr = np.arctan(np.abs(s_i) / source_origin_safe)
    s_phi_arr = phi_arr / (np.pi / 2.0)

    alpha_d_arr = 1.0 + float(line_awtv_beta_d) * (1.0 - s_phi_arr)
    alpha_d_geom = float(np.mean(alpha_d_arr))
    alpha_d_final = float(alpha_d_geom if line_awtv_alpha_d is None else line_awtv_alpha_d)
    
    print(f'Linear CT geometry: dis_step={dis_step}, source_det={source_det}, source_origin={source_origin}')
    
    # Create linear CT projectors
    line_projector_low_reso = LineCT3DProjector(
        image_size=(config['img_size'][0]//2, config['img_size'][1]//2, config['img_size'][2]//2),
        proj_size=(config['proj_size'][0]//2, config['proj_size'][1]//2),
        num_proj=config['num_proj'],
        source_det=source_det,
        source_origin=source_origin,
        dis_step=dis_step, 
        det_spacing=det_spacing*2.0,
        voxel_spacing=voxel_spacing*2.0
    )
    
    line_projector = LineCT3DProjector(
        image_size=config['img_size'],
        proj_size=config['proj_size'],
        num_proj=config['num_proj'],
        source_det=source_det,
        source_origin=source_origin,
        dis_step=dis_step,
        det_spacing=det_spacing,
        voxel_spacing=voxel_spacing
    )

    for it, (grid, image) in enumerate(data_loader):
        grid = grid.cuda()
        image = image.cuda()

        # prepare init folder early so it can be used by downstream save calls
        init_folder = os.path.join(image_directory, "init_data")
        # If real projections were provided, use them as target projections.
        # Expected shape in file: [num_proj, det_rows, det_cols]
        if real_projs is not None:
            projs_np = real_projs
            # Validate num_proj and proj_size
            if projs_np.shape[0] != config['num_proj']:
                print(f"[Warning] real proj num {projs_np.shape[0]} != config num_proj {config['num_proj']}")
            if tuple(projs_np.shape[1:3]) != tuple(config['proj_size']):
                print(f"[Warning] real proj size {projs_np.shape[1:3]} != config proj_size {config['proj_size']}")

            projs = torch.from_numpy(projs_np).unsqueeze(0).to(device='cuda', dtype=torch.float32)
            projs = projs.contiguous()
        else:
            projs = line_projector.forward_project(image.transpose(1, 4).squeeze(1))
            if simu_poisson_enabled:
                projs = add_poisson_noise_to_projs(projs, n0=simu_poisson_n0)

        image_low_resos = []
        projs_low_resos = []
        # If we have a ground-truth image, generate low-res image/proj pairs from it.
        if real_projs is None:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        image_low_reso = image[:, i::2, j::2, k::2, :]
                        image_low_resos.append(image_low_reso)
                        low_proj = line_projector_low_reso.forward_project(image_low_reso.transpose(1, 4).squeeze(1))
                        if simu_poisson_enabled:
                            low_proj = add_poisson_noise_to_projs(low_proj, n0=simu_poisson_n0)
                        projs_low_resos.append(low_proj)
        else:
            # For real projections: create low-res projection targets by detector binning (no angle subsampling)
            def bin_projs_tensor(
                projs_tensor,
                bin_factor=2,
                row_offset=0,
                col_offset=0,
                pad_mode='reflect',
            ):
                # projs_tensor: [B, P, U, V] [batch, num_proj, det_rows, det_cols]
                # ensure dtype and memory layout
                projs_tensor = projs_tensor.to(dtype=torch.float32).contiguous()
                B, P, U, V = projs_tensor.shape

                # offsets must be within [0, bin_factor-1]
                assert 0 <= row_offset < bin_factor and 0 <= col_offset < bin_factor, \
                    f"offsets must be in [0,{bin_factor-1}]"

                target_U = config['proj_size'][0] // bin_factor
                target_V = config['proj_size'][1] // bin_factor

                # compute minimal pixels needed for this offset
                need_U = row_offset + target_U * bin_factor
                need_V = col_offset + target_V * bin_factor

                pad_r = max(0, need_U - U)
                pad_c = max(0, need_V - V)

                x = projs_tensor.view(B * P, 1, U, V)

                if pad_r > 0 or pad_c > 0:
                    # pad = (left, right, top, bottom)
                    x = F.pad(x, (0, pad_c, 0, pad_r), mode=pad_mode)

                x = x[
                    :,
                    :,
                    row_offset:row_offset + target_U * bin_factor,
                    col_offset:col_offset + target_V * bin_factor,
                ]

                pooled = F.avg_pool2d(x, kernel_size=bin_factor, stride=bin_factor)

                pooled = pooled.view(B, P, target_U, target_V)

                assert pooled.shape[-2:] == (target_U, target_V), \
                    f"low-res proj shape mismatch: got {pooled.shape[-2:]}, expected {(target_U, target_V)}"

                return pooled.contiguous()

            # compute 4 detector-phase offsets (0,0),(0,1),(1,0),(1,1)
            projs_low_resos = []
            shapes = []
            for r_off in (0, 1):
                for c_off in (0, 1):
                    low = bin_projs_tensor(projs, bin_factor=2, row_offset=r_off, col_offset=c_off, pad_mode='reflect')
                    projs_low_resos.append(low)
                    shapes.append(tuple(low.shape))
            # duplicate to length 8 to match existing indexing elsewhere
            projs_low_resos = projs_low_resos + [x.clone() for x in projs_low_resos]
            # save one sample for inspection and print per-offset shapes
            try:
                os.makedirs(init_folder, exist_ok=True)
                for idx, s in enumerate(shapes):
                    print(f"[InitSave] projs_low_resos offset idx {idx} shape {s}")
                sample_np = projs_low_resos[0].squeeze(0).contiguous().cpu().numpy()
                np.save(os.path.join(init_folder, 'projs_low_res_sample.npy'), sample_np.astype(np.float32))
                print(f"[InitSave] Saved projs_low_res_sample.npy shape={sample_np.shape}")
            except Exception as e:
                print(f"[InitSave] Failed saving low-res sample: {e}")

        # BP-guided Gaussian initialization.
        projs = projs.contiguous()
        bp_recon = line_projector.backward_project(projs)

        # Build training and test data tuples. For real projections, the image
        # returned by the loader is used only as a placeholder for visualization.
        test_data = (grid, image)
        train_data = (grid, projs)

        save_image_3d(
            test_data[1],
            slice_idx,
            os.path.join(image_directory, "test.png"),
        )
        save_image_3d(
            train_data[1].unsqueeze(-1),
            proj_idx,
            os.path.join(image_directory, "train.png"),
        )

        # Convert the projector's HWD layout to DHW for comparison with truth.
        bp_recon_np = np.transpose(
            bp_recon.squeeze().cpu().numpy(), (2, 0, 1)
        )
        try:
            test_np = test_data[1].squeeze().cpu().numpy()
            data_range = max(
                float(test_np.max() - test_np.min()), 1e-8
            )
            bp_recon_ssim = compare_ssim(
                bp_recon_np,
                test_np,
                data_range=data_range,
                channel_axis=None,
            )
        except Exception:
            bp_recon_ssim = 0.0

        # Convert HWD to [B, D, H, W, 1] for Gaussian initialization.
        bp_recon = bp_recon.unsqueeze(1).transpose(1, 4)

        bp_recon_psnr = calculate_psnr(
            bp_recon,
            test_data[1],
            data_range=(data_range if 'data_range' in locals() else None),
        )
        print(
            f"[InitMetric] BP PSNR={bp_recon_psnr:.6f} dB, "
            f"SSIM={bp_recon_ssim:.6f}, "
            f"min={float(bp_recon.min()):.6g}, "
            f"max={float(bp_recon.max()):.6g}"
        )

        save_image_3d(
            bp_recon,
            slice_idx,
            os.path.join(
                image_directory,
                f"bp_recon_{bp_recon_psnr:.4g}dB_"
                f"ssim{bp_recon_ssim:.4g}.png",
            ),
        )

        init_folder = os.path.join(image_directory, "init_data")
        os.makedirs(init_folder, exist_ok=True)
        with open(
            os.path.join(init_folder, 'bp_guide_metrics.txt'),
            'w',
            encoding='utf-8',
        ) as metric_file:
            metric_file.write("guide: bp\n")
            metric_file.write(f"PSNR_dB: {bp_recon_psnr:.8f}\n")
            metric_file.write(f"SSIM: {bp_recon_ssim:.8f}\n")
            metric_file.write(f"min: {float(bp_recon.min()):.8g}\n")
            metric_file.write(f"max: {float(bp_recon.max()):.8g}\n")

        save_image_3d_slices(
            test_data[1],
            all_slice_idx,
            os.path.join(init_folder, "test"),
            prefix='test',
        )
        save_image_3d_slices(
            train_data[1].unsqueeze(-1),
            all_proj_idx,
            os.path.join(init_folder, "train_proj"),
            prefix='proj',
        )
        save_image_3d_slices(
            bp_recon,
            all_slice_idx,
            os.path.join(init_folder, "bp_recon"),
            prefix='bp',
        )

        # Save the complete BP guide before Gaussian sampling.
        try:
            guide_dhw = (
                bp_recon.detach().squeeze().cpu().numpy().astype(np.float32)
            )
            guide_hwd = np.ascontiguousarray(
                np.transpose(guide_dhw, (1, 2, 0))
            )
            _save_raw_and_npy_from_tensor(
                guide_hwd,
                os.path.join(init_folder, 'bp_guide'),
                reorder_for_name=(0, 1, 2),
                rotate90=False,
                target_shape=None,
            )
        except Exception as exc:
            print(f"[InitSave] Failed saving BP guide volume: {exc}")

        from models.gaussian_model_Line import GaussianModelLine

        op = OptimizationParams()
        print("[OptimizationParams]")
        for key, value in op.as_dict().items():
            print(f"  {key}: {value}")

        gaussians = GaussianModelLine()
        gaussians.configure_smooth_filter(
            enabled=False,
            smooth_filter_s=smooth_filter_s,
            update_interval=smooth_filter_interval,
            margin=smooth_filter_margin,
            min_depth=smooth_filter_min_depth,
        )

        grid_low_resos = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    grid_low_resos.append(
                        grid[:, i::2, j::2, k::2, :]
                    )
        bp_recon_low_reso = bp_recon[:, ::2, ::2, ::2, :]

        # For real projections, use the BP guide only for low-resolution metrics.
        if real_projs is not None:
            image_low_resos = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        image_low_resos.append(
                            bp_recon[:, i::2, j::2, k::2, :]
                        )

        start_iter = 0

        def initialize_gaussians():
            gaussians.create_from_realBP(
                bp_recon_low_reso,
                air_threshold=config['air_threshold'],
                ini_intensity=config['ini_intensity'],
                ini_sigma=config['ini_sigma'],
                spatial_lr_scale=config['spatial_lr_scale'],
                num_samples=config['num_gaussian'],
                start=config['start'],
                mix_fb=float(config.get('mix_fb', 0.0)),
                mix_grad=float(config.get('mix_grad', 1.0)),
            )

        # Resume from a checkpoint if provided.
        if opts.checkpoint is not None and os.path.exists(opts.checkpoint):
            print(f"[Resume] Loading checkpoint from: {opts.checkpoint}")
            checkpoint = torch.load(
                opts.checkpoint,
                map_location='cuda',
                weights_only=False,
            )
            if 'gaussians' in checkpoint:
                gaussians.restore(
                    checkpoint['gaussians'],
                    training_args=op,
                )
                start_iter = checkpoint.get('iteration', 0)
                print(
                    "[Checkpoint] Restored model and optimizer, "
                    f"start from iter {start_iter}"
                )
            else:
                print(
                    "[Checkpoint] Missing 'gaussians'; "
                    "fall back to BP-guided initialization."
                )
                initialize_gaussians()
                gaussians.training_setup(op)
        else:
            print(
                "[Init] BP-guided Gaussian initialization, "
                f"guide_shape={tuple(bp_recon_low_reso.shape)}"
            )
            initialize_gaussians()
            gaussians.training_setup(op)

        # Render and save the initialized Gaussian volume.
        try:
            os.makedirs(init_folder, exist_ok=True)
            gauss_vol = gaussians.grid_sample(grid)
            gauss_arr = (
                gauss_vol.detach().squeeze().cpu().numpy().astype(np.float32)
            )

            if gauss_arr.ndim != 3:
                raise ValueError(
                    "Expected a 3D volume after squeeze, "
                    f"but got shape {gauss_arr.shape}"
                )

            print(
                "[InitSave] Gaussian-rendered volume shape "
                f"(original) = {gauss_arr.shape}"
            )

            gauss_dhw = np.ascontiguousarray(gauss_arr)
            D, H, W = gauss_dhw.shape
            print(f"[InitSave] Save as [D,H,W] = [{D}, {H}, {W}]")

            gauss_hwd = np.ascontiguousarray(
                np.transpose(gauss_dhw, (1, 2, 0))
            )
            base = os.path.join(init_folder, 'gaussian_from_bp')

            try:
                _save_raw_and_npy_from_tensor(
                    gauss_hwd,
                    base,
                    reorder_for_name=(0, 1, 2),
                    rotate90=False,
                    target_shape=None,
                )
                shape_str = f"{H}x{W}x{D}"
                npy_path = f"{base}_{shape_str}.npy"
                raw_path = f"{base}_{shape_str}.raw"
                print(f"[InitSave] Saved NPY: {npy_path}")
                print(f"[InitSave] Saved RAW: {raw_path}")
                print("[InitSave] ImageJ/Fiji RAW import settings:")
                print(
                    f"           Width={W}, Height={H}, "
                    f"Number of Images={D}, 32-bit Real, Little-endian"
                )
            except Exception as e:
                print(f"[InitSave] Failed saving Gaussian volume: {e}")

        except Exception as e:
            print(
                "[InitSave] Failed to save Gaussian-rendered "
                f"initialization volume: {e}"
            )

        for iteration in range(start_iter, max_iter):
            gaussians.update_learning_rate(iteration)

            # Enable GSF after the configured start iteration.
            smooth_filter_active = bool(
                smooth_filter_enabled
                and iteration >= smooth_filter_start_iter
            )

            if smooth_filter_active:
                gaussians.smooth_filter_enabled = True
                gaussians.maybe_update_3D_filter(
                    iteration, line_projector
                )
            else:
                gaussians.smooth_filter_enabled = False
            if iteration < config['low_reso_stage']:
                train_output = gaussians.grid_sample(
                    grid_low_resos[iteration % 8]
                )
            else:
                train_output = gaussians.grid_sample(grid)

            if iteration < config['low_reso_stage']:
                train_projs = line_projector_low_reso.forward_project(
                    train_output.transpose(1, 4).squeeze(1)
                )
            else:
                train_projs = line_projector.forward_project(
                    train_output.transpose(1, 4).squeeze(1)
                )

            if iteration < config['low_reso_stage']:
                target_projs = projs_low_resos[iteration % 8]
            else:
                target_projs = projs

            if proj_bg_weight_enabled:
                loss_mse = weighted_proj_mse(
                    train_projs,
                    target_projs,
                    bg_thr=proj_bg_thr,
                    bg_weight=proj_bg_weight,
                )
            else:
                loss_mse = F.mse_loss(train_projs, target_projs)

            if line_awtv_weight_schedule:
                if line_awtv_weight_iters <= 0:
                    line_awtv_weight_cur = line_awtv_weight_end
                else:
                    n = min(iteration, line_awtv_weight_iters)
                    line_awtv_weight_cur = (
                        line_awtv_weight_start
                        - (
                            line_awtv_weight_start
                            - line_awtv_weight_end
                        )
                        * (n / line_awtv_weight_iters)
                    )
            else:
                line_awtv_weight_cur = line_awtv_weight

            if line_awtv_delta_schedule:
                if line_awtv_delta_iters <= 0:
                    line_awtv_delta_cur = line_awtv_delta_end
                else:
                    n_delta = min(iteration, line_awtv_delta_iters)
                    line_awtv_delta_cur = (
                        line_awtv_delta_start
                        + (
                            line_awtv_delta_end
                            - line_awtv_delta_start
                        )
                        * (n_delta / line_awtv_delta_iters)
                    )
            else:
                line_awtv_delta_cur = line_awtv_delta_cl

            line_taawtv_term = line_awtv_regularization(
                train_output,
                tau=line_awtv_tau,
                alpha_h=alpha_h_final,
                alpha_w=alpha_w_final,
                alpha_d=alpha_d_final,
                delta_cl=line_awtv_delta_cur,
            )
            loss = (
                loss_mse
                + line_awtv_weight_cur * line_taawtv_term
            )

            if lambda_dssim > 0:
                loss_dssim = projection_dssim_loss(
                    train_projs,
                    target_projs,
                    window_size=dssim_window,
                )
                loss = loss + lambda_dssim * loss_dssim
            else:
                loss_dssim = torch.tensor(
                    0.0,
                    device=train_projs.device,
                )

            loss.backward()
            gaussians.optimizer.step()

            if config['do_density_control']:
                with torch.no_grad():
                    if (
                        gaussians.get_xyz.shape[-2]
                        < config['max_gaussians']
                        and iteration < config['densify_until_iter']
                    ):
                        if (
                            iteration > config['densify_from_iter']
                            and iteration
                            % config['densification_interval']
                            == 0
                        ):
                            gaussians.densify_and_prune(
                                config['max_grad'],
                                config['min_intensity'],
                                sigma_extent=config['sigma_extent'],
                            )

            gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration + 1) % checkpoint_save_iter == 0:
                checkpoint_path = os.path.join(
                    checkpoint_directory,
                    f"gaussians_iter_{iteration + 1}.pth",
                )
                torch.save(
                    {
                        'iteration': iteration + 1,
                        'gaussians': gaussians.capture(),
                    },
                    checkpoint_path,
                )
                print(f"[Checkpoint] Saved: {checkpoint_path}")

            if (iteration + 1) % config['log_iter'] == 0:
                if iteration < config['low_reso_stage']:
                    train_psnr = calculate_psnr(
                        train_projs,
                        projs_low_resos[iteration % 8],
                    )
                else:
                    train_psnr = calculate_psnr(
                        train_projs,
                        projs,
                    )

                train_loss = loss.item()

                print(
                    "[Iteration: {}/{}] Train loss: {:.4g} | "
                    "Train psnr: {:.4g}".format(
                        iteration + 1,
                        max_iter,
                        train_loss,
                        train_psnr,
                    )
                )
                wandb.log(
                    {
                        "Iteration": iteration + 1,
                        "Train Loss": train_loss,
                        "Train PSNR": train_psnr,
                        "Train MSE": loss_mse.item(),
                        "Train DSSIM": loss_dssim.item(),
                        "Line-TAAwTV Term": line_taawtv_term.item(),
                        "Line-TAAwTV Weight": line_awtv_weight_cur,
                        "Line-TAAwTV Delta": line_awtv_delta_cur,
                        "GSF Active": int(smooth_filter_active),
                        "GSF Start Iter": smooth_filter_start_iter,
                    }
                )

            if (
                iteration == 0
                or (iteration + 1) % config['val_iter'] == 0
            ):
                with torch.no_grad():
                    test_output = gaussians.grid_sample(test_data[0])
                    test_loss = 0.5 * torch.mean(
                        (test_output - test_data[1]) ** 2
                    )
                    test_loss = test_loss.item()

                    test_output_np = (
                        test_output.transpose(1, 4)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    test_target_np = (
                        test_data[1]
                        .transpose(1, 4)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    test_data_range = max(
                        float(
                            test_target_np.max()
                            - test_target_np.min()
                        ),
                        1e-8,
                    )
                    test_psnr = calculate_psnr(
                        test_output,
                        test_data[1],
                        data_range=test_data_range,
                    )
                    test_ssim = compare_ssim(
                        test_output_np,
                        test_target_np,
                        data_range=test_data_range,
                        channel_axis=None,
                    )

                    test_output_low_reso = gaussians.grid_sample(
                        grid_low_resos[iteration % 8]
                    )
                    test_loss_low_reso = 0.5 * torch.mean(
                        (
                            test_output_low_reso
                            - image_low_resos[iteration % 8]
                        )
                        ** 2
                    )
                    test_loss_low_reso = (
                        test_loss_low_reso.item()
                    )
                    test_output_low_np = (
                        test_output_low_reso.transpose(1, 4)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    image_low_np = (
                        image_low_resos[iteration % 8]
                        .transpose(1, 4)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    low_data_range = max(
                        float(
                            image_low_np.max()
                            - image_low_np.min()
                        ),
                        1e-8,
                    )
                    test_psnr_low_reso = calculate_psnr(
                        test_output_low_reso,
                        image_low_resos[iteration % 8],
                        data_range=low_data_range,
                    )
                    test_ssim_low_reso = compare_ssim(
                        test_output_low_np,
                        image_low_np,
                        data_range=low_data_range,
                        channel_axis=None,
                    )

                save_image_3d(
                    test_output,
                    slice_idx,
                    os.path.join(
                        image_directory,
                        "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(
                            iteration + 1,
                            test_psnr,
                            test_ssim,
                        ),
                    ),
                )

                iter_folder_name = (
                    "recon_{}_{:.4g}dB_ssim{:.4g}".format(
                        iteration + 1,
                        test_psnr,
                        test_ssim,
                    )
                )
                iter_folder = os.path.join(
                    image_directory,
                    iter_folder_name,
                )
                save_image_3d_slices(
                    test_output,
                    all_slice_idx,
                    iter_folder,
                    prefix='recon',
                )

                wandb.log(
                    {
                        "Iteration": iteration + 1,
                        "Test Loss": test_loss,
                        "Test PSNR": test_psnr,
                        "Test SSIM": test_ssim,
                        "Test Loss-low_reso": test_loss_low_reso,
                        "Test PSNR-low_reso": test_psnr_low_reso,
                        "Test SSIM-low_reso": test_ssim_low_reso,
                    }
                )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    set_seed(42)
    main()
