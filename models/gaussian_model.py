import torch
import numpy as np
from gs_utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
# from plyfile import PlyData, PlyElement
# from simple_knn._C import distCUDA2
from gs_utils.general_utils import strip_symmetric, build_scaling_rotation, build_rotation
import time
import torch.nn.functional as F

import sys

from gs_utils.Compute_intensity import compute_intensity


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)

            return actual_covariance

        self._scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.intensity_activation = torch.sigmoid
        self.inverse_intensity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)                                               
        self._intensity = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._intensity,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (
        self._xyz, 
        self._scaling,
        self._rotation,
        self._intensity,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self._scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_intensity(self):
        return self.intensity_activation(self._intensity)
    
    @property
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    @property
    def get_inv_covariance(self, scaling_modifier=1):
        scaling = self.get_scaling
        rotation = self.get_rotation
        scaling_inv_squared = 1.0 / (scaling * scaling_modifier) ** 2
        S_inv_squared = torch.diag_embed(scaling_inv_squared)
        R = build_rotation(rotation)
        R_transpose = R.transpose(1, 2)
        covariance_inv = torch.matmul(R, torch.matmul(S_inv_squared, R_transpose))
        return covariance_inv

    def create_from_fbp(self, fbp_image, air_threshold=0.05, ini_intensity=0.04, ini_sigma=0.01, spatial_lr_scale=1, num_samples=150000, 
                                start = 0.15,
                                ):
        print(f"[GaussianModel] create_from_fbp called: fbp_image.shape={tuple(fbp_image.shape)} dtype={fbp_image.dtype} device={fbp_image.device if isinstance(fbp_image, torch.Tensor) else 'numpy'}")
        try:
            minv = float(fbp_image.min())
            maxv = float(fbp_image.max())
            print(f"[GaussianModel] fbp_image min={minv:.6g}, max={maxv:.6g}")
        except Exception:
            pass

        self.spatial_lr_scale = spatial_lr_scale
        bs, D, H, W, _ = fbp_image.shape 
        start = int(start * D * H * W)

        fbp_image[fbp_image < air_threshold] = 0
        fbp_image = fbp_image.permute(0, 4, 1, 2, 3) # [bs, 1, z, x, y]
        # [bs, 1, z, x, y]
        grad_x = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 1:-1, 2:])
        grad_y = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 2:, 1:-1])
        grad_z = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 2:, 1:-1, 1:-1])
  
        grad_x_padded = F.pad(grad_x, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_y_padded = F.pad(grad_y, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_z_padded = F.pad(grad_z, (1, 1, 1, 1, 1, 1), "constant", 0)

        grad_norm = torch.sqrt(grad_x_padded ** 2 + grad_y_padded ** 2 + grad_z_padded ** 2)
        grad_norm = grad_norm.reshape(-1)

        total_candidates = grad_norm.numel()
        requested_k = start + num_samples
        if requested_k > total_candidates:
            print(
                f"[GaussianModel] Warning: start + num_samples = {requested_k} "
                f"exceeds available voxels {total_candidates}; clipping."
            )
        start = min(start, max(total_candidates - 1, 0))
        topk_k = min(requested_k, total_candidates)
        _, indices = torch.topk(grad_norm, topk_k)
        indices = indices[start:]
        num_samples = int(indices.numel())
        if num_samples <= 0:
            raise RuntimeError(
                f"No Gaussian samples selected: start={start}, topk_k={topk_k}, "
                f"available={total_candidates}"
            )
        coords = torch.stack(torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W)), dim=-1).reshape(-1, 3).cuda()

        sampled_coords = coords[indices]

        grid = torch.zeros((D, H, W), dtype=torch.int32, device="cuda")
        # Increase the count in the grid at the location of each sampled point
        indices_3d = sampled_coords.long()
        grid[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]] += 1
        # Apply a 3D convolution to count neighbours
        kernel_size = 5  # Define the size of the neighbourhood
        padding = kernel_size // 2
        conv_kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device="cuda", dtype=torch.float32)
        neighbours_count = F.conv3d(grid.unsqueeze(0).unsqueeze(0).float(), conv_kernel, padding=padding).squeeze()
        # Retrieve the number of neighbours for each sampled point
        num_neighbours = neighbours_count[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]].float()
        num_neighbours = torch.clamp(num_neighbours, min=1.0)
        # Adjust scaling based on the number of neighbours, with lower bound to avoid instant pruning
        scaling = ini_sigma / num_neighbours
        scaling = torch.clamp(scaling, min=0.004, max=ini_sigma)
    
        fbp_image[fbp_image<air_threshold] = 0
        init_vals = fbp_image.reshape(-1)[indices]
        init_vals = torch.nan_to_num(init_vals, nan=0.0, posinf=0.0, neginf=0.0)
        intensities = ini_intensity * init_vals + 0.001
        intensities = torch.clamp(intensities, min=1e-6, max=1.0 - 1e-6)
     
        sampled_coords = sampled_coords.float()

        sampled_coords = sampled_coords / torch.tensor([D, H, W], dtype=torch.float, device="cuda")

        intensities = inverse_sigmoid(intensities).unsqueeze(1)
        scaling = torch.log(scaling).unsqueeze(1).repeat(1, 3)
        rotation = torch.zeros((num_samples, 4), device="cuda")
        rotation[:, 0] = 1

        self._xyz = nn.Parameter(sampled_coords.requires_grad_(True))
        self._intensity = nn.Parameter(intensities.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))

        print("[Gaussian init] num_points={}, intensity(min/max)=({:.4g}, {:.4g}), scaling(min/max)=({:.4g}, {:.4g})".format(
            self._xyz.shape[0],
            float(self.get_intensity.min().item()),
            float(self.get_intensity.max().item()),
            float(self.get_scaling.min().item()),
            float(self.get_scaling.max().item())
        ))

    def create_from_realBP(self, fbp_image, air_threshold=0.05, ini_intensity=0.04, ini_sigma=0.01, spatial_lr_scale=1, num_samples=150000,
                            start=0.15, mix_fb=0, mix_grad=1):
        """
        初始化用于真实投影的采样：使用强度与梯度的加权混合分数进行采样。
        分数 score = mix_fb * fbp_norm + mix_grad * grad_normed
        """
        print(f"[GaussianModel] create_from_realBP called: fbp_image.shape={tuple(fbp_image.shape)} dtype={fbp_image.dtype} device={fbp_image.device if isinstance(fbp_image, torch.Tensor) else 'numpy'}")
        try:
            minv = float(fbp_image.min())
            maxv = float(fbp_image.max())
            print(f"[GaussianModel] fbp_image min={minv:.6g}, max={maxv:.6g}")
        except Exception:
            pass

        self.spatial_lr_scale = spatial_lr_scale
        bs, D, H, W, _ = fbp_image.shape
        start = int(start * D * H * W)

        # mask air
        fbp_image[fbp_image < air_threshold] = 0
        # permute to [bs, 1, D, H, W]
        fbp_perm = fbp_image.permute(0, 4, 1, 2, 3)

        # compute gradients (same as create_from_fbp)
        grad_x = torch.abs(fbp_perm[:, :, 1:-1, 1:-1, 1:-1] - fbp_perm[:, :, 1:-1, 1:-1, 2:])
        grad_y = torch.abs(fbp_perm[:, :, 1:-1, 1:-1, 1:-1] - fbp_perm[:, :, 1:-1, 2:, 1:-1])
        grad_z = torch.abs(fbp_perm[:, :, 1:-1, 1:-1, 1:-1] - fbp_perm[:, :, 2:, 1:-1, 1:-1])

        grad_x_padded = F.pad(grad_x, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_y_padded = F.pad(grad_y, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_z_padded = F.pad(grad_z, (1, 1, 1, 1, 1, 1), "constant", 0)

        grad_norm = torch.sqrt(grad_x_padded ** 2 + grad_y_padded ** 2 + grad_z_padded ** 2)

        # flatten both fbp and grad to compute mixed score
        fbp_flat = fbp_perm.reshape(-1)
        grad_flat = grad_norm.reshape(-1)

        # robust percentile normalization for fbp_flat: clip to [p1, p99]
        # this reduces sensitivity to extreme outliers in real FBP volumes
        try:
            p_low = float(torch.quantile(fbp_flat, 0.01))
            p_high = float(torch.quantile(fbp_flat, 0.99))
        except Exception:
            p_low = float(fbp_flat.min())
            p_high = float(fbp_flat.max())

        if p_high <= p_low:
            p_low = float(fbp_flat.min())
            p_high = float(fbp_flat.max())

        denom = (p_high - p_low) + 1e-8
        fbp_clipped = torch.clamp(fbp_flat, min=p_low, max=p_high)
        fbp_norm = (fbp_clipped - p_low) / denom

        # normalize grad_flat to [0,1] by max
        grad_max = float(grad_flat.max())
        grad_normed = grad_flat / (grad_max + 1e-8)

        # mixed score
        score = mix_fb * fbp_norm + mix_grad * grad_normed

        # top-k selection on score
        total_candidates = score.numel()
        requested_k = start + num_samples
        if requested_k > total_candidates:
            print(
                f"[GaussianModel] Warning: start + num_samples = {requested_k} "
                f"exceeds available voxels {total_candidates}; clipping."
            )
        start = min(start, max(total_candidates - 1, 0))
        topk_k = min(requested_k, total_candidates)
        _, indices = torch.topk(score, topk_k)
        indices = indices[start:]
        num_samples = int(indices.numel())
        if num_samples <= 0:
            raise RuntimeError(
                f"No Gaussian samples selected: start={start}, topk_k={topk_k}, "
                f"available={total_candidates}"
            )

        # compute coords and sampled coords
        coords = torch.stack(torch.meshgrid(torch.arange(D, device=fbp_image.device), torch.arange(H, device=fbp_image.device), torch.arange(W, device=fbp_image.device)), dim=-1).reshape(-1, 3)
        sampled_coords = coords[indices]

        grid = torch.zeros((D, H, W), dtype=torch.int32, device=fbp_image.device)
        indices_3d = sampled_coords.long()
        grid[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]] += 1

        # neighbor counting (same as create_from_fbp)
        kernel_size = 5
        padding = kernel_size // 2
        conv_kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=fbp_image.device, dtype=torch.float32)
        neighbours_count = F.conv3d(grid.unsqueeze(0).unsqueeze(0).float(), conv_kernel, padding=padding).squeeze()
        num_neighbours = neighbours_count[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]].float()
        num_neighbours = torch.clamp(num_neighbours, min=1.0)
        scaling = ini_sigma / num_neighbours
        scaling = torch.clamp(scaling, min=0.004, max=ini_sigma)

        # init values from fbp_perm
        fbp_perm[fbp_perm < air_threshold] = 0
        init_vals = fbp_perm.reshape(-1)[indices]
        init_vals = torch.nan_to_num(init_vals, nan=0.0, posinf=0.0, neginf=0.0)
        intensities = ini_intensity * init_vals + 0.001
        intensities = torch.clamp(intensities, min=1e-6, max=1.0 - 1e-6)

        sampled_coords = sampled_coords.float()
        sampled_coords = sampled_coords / torch.tensor([D, H, W], dtype=torch.float, device=fbp_image.device)

        intensities = inverse_sigmoid(intensities).unsqueeze(1)
        scaling = torch.log(scaling).unsqueeze(1).repeat(1, 3)
        rotation = torch.zeros((num_samples, 4), device=fbp_image.device)
        rotation[:, 0] = 1

        self._xyz = nn.Parameter(sampled_coords.requires_grad_(True))
        self._intensity = nn.Parameter(intensities.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))

        print("[Gaussian init - realBP] num_points={}, intensity(min/max)=({:.4g}, {:.4g}), scaling(min/max)=({:.4g}, {:.4g})".format(
            self._xyz.shape[0],
            float(self.get_intensity.min().item()),
            float(self.get_intensity.max().item()),
            float(self.get_scaling.min().item()),
            float(self.get_scaling.max().item())
        ))

    def _create_ablation_gaussians(self, sampled_coords, volume_shape, ini_intensity,
                                   ini_sigma, spatial_lr_scale, mode_name):
        """Create ablation Gaussians with constant intensity and isotropic scale."""
        D, H, W = (int(v) for v in volume_shape)
        device = sampled_coords.device
        num_samples = int(sampled_coords.shape[0])
        if num_samples <= 0:
            raise RuntimeError(f"No Gaussian samples selected for {mode_name} initialization")

        # Random and gradient-only ablations use one fixed isotropic scale.
        # The BP initializer keeps its local-density scale rule unchanged.
        ini_sigma = float(ini_sigma)
        if ini_sigma <= 0:
            raise ValueError(f"ini_sigma must be positive, got {ini_sigma}")
        scaling = torch.full(
            (num_samples,), ini_sigma, dtype=torch.float32, device=device
        )

        intensity_value = float(np.clip(ini_intensity, 1e-6, 1.0 - 1e-6))
        intensities = torch.full((num_samples, 1), intensity_value, device=device)
        intensities = inverse_sigmoid(intensities)
        scaling = torch.log(scaling)[:, None].repeat(1, 3)
        rotation = torch.zeros((num_samples, 4), device=device)
        rotation[:, 0] = 1.0
        xyz = sampled_coords.float() / torch.tensor([D, H, W], dtype=torch.float32, device=device)

        self.spatial_lr_scale = float(spatial_lr_scale)
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._intensity = nn.Parameter(intensities.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))

        print("[Gaussian init - {}] num_points={}, constant_intensity={:.4g}, "
              "scaling(min/max)=({:.4g}, {:.4g})".format(
                  mode_name,
                  num_samples,
                  intensity_value,
                  float(self.get_scaling.min().item()),
                  float(self.get_scaling.max().item()),
              ))

    def create_from_random(self, volume_shape, ini_intensity=0.001, ini_sigma=0.01,
                           spatial_lr_scale=1, num_samples=150000, seed=42):
        """Uniformly sample Gaussian positions without using BP values or gradients."""
        D, H, W = (int(v) for v in volume_shape)
        total = D * H * W
        num_samples = min(int(num_samples), total)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        device = self._xyz.device if self._xyz.is_cuda else torch.device("cuda")
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        flat_indices = torch.randperm(total, device=device, generator=generator)[:num_samples]
        d = torch.div(flat_indices, H * W, rounding_mode='floor')
        h = torch.div(flat_indices % (H * W), W, rounding_mode='floor')
        w = flat_indices % W
        sampled_coords = torch.stack([d, h, w], dim=1)
        self._create_ablation_gaussians(
            sampled_coords, (D, H, W), ini_intensity, ini_sigma,
            spatial_lr_scale, "random",
        )

    def create_from_gradient_only(self, fbp_image, air_threshold=0.05,
                                  ini_intensity=0.001, ini_sigma=0.01,
                                  spatial_lr_scale=1, num_samples=150000,
                                  start=0.15):
        """Use only BP-gradient magnitude for positions; initialize intensity uniformly."""
        fbp = fbp_image.detach().clone().to(dtype=torch.float32)
        if fbp.ndim != 5 or fbp.shape[-1] != 1:
            raise ValueError(f"Expected BP shape [B,D,H,W,1], got {tuple(fbp.shape)}")
        _, D, H, W, _ = fbp.shape
        fbp[fbp < float(air_threshold)] = 0
        fbp = fbp.permute(0, 4, 1, 2, 3)

        grad_x = torch.abs(fbp[:, :, 1:-1, 1:-1, 1:-1] - fbp[:, :, 1:-1, 1:-1, 2:])
        grad_y = torch.abs(fbp[:, :, 1:-1, 1:-1, 1:-1] - fbp[:, :, 1:-1, 2:, 1:-1])
        grad_z = torch.abs(fbp[:, :, 1:-1, 1:-1, 1:-1] - fbp[:, :, 2:, 1:-1, 1:-1])
        grad_norm = torch.sqrt(
            F.pad(grad_x, (1, 1, 1, 1, 1, 1)) ** 2
            + F.pad(grad_y, (1, 1, 1, 1, 1, 1)) ** 2
            + F.pad(grad_z, (1, 1, 1, 1, 1, 1)) ** 2
        ).reshape(-1)

        total = int(grad_norm.numel())
        skip = min(max(int(float(start) * total), 0), max(total - 1, 0))
        topk_k = min(skip + int(num_samples), total)
        _, indices = torch.topk(grad_norm, topk_k)
        indices = indices[skip:]
        d = torch.div(indices, H * W, rounding_mode='floor')
        h = torch.div(indices % (H * W), W, rounding_mode='floor')
        w = indices % W
        sampled_coords = torch.stack([d, h, w], dim=1)
        self._create_ablation_gaussians(
            sampled_coords, (D, H, W), ini_intensity, ini_sigma,
            spatial_lr_scale, "gradient_only",
        )
    
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._intensity], 'lr': training_args.intensity_lr, "name": "intensity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('intensity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._intensity = optimizable_tensors["intensity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]



    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_intensities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "intensity": new_intensities,
        "scaling": new_scaling,
        "rotation": new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._intensity = optimizable_tensors["intensity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        new_xyz = samples + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_intensities = self._intensity[selected_pts_mask].repeat(N,1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_intensities, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        
        original_intensity = self.intensity_activation(self._intensity[selected_pts_mask]) / 2
        original_intensity = self.inverse_intensity_activation(original_intensity)
        new_intensity = original_intensity 
        self._intensity[selected_pts_mask] = original_intensity 

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_intensity, sigma_extent):
        grads = torch.norm(self._xyz.grad, dim=-1, keepdim=True)

        self.densify_and_clone(grads, max_grad, sigma_extent)
        self.densify_and_split(grads, max_grad, sigma_extent)
    
        prune_mask = (
                    (self.get_intensity < min_intensity).squeeze() 
                    | (torch.min(self.get_scaling, dim=1).values < 0.003)
                )
        
        print("Pruning {} points".format(prune_mask.sum()))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    
    def grid_sample(self, grid):
        # grid: [batchsize, z, x, y, 3]
        grid_shape = grid.shape
        #intensity_grid = torch.empty(grid_shape, device="cuda")
        # expand dimensions for broadcasting
        grid_expanded = grid.unsqueeze(-2)  # [batchsize, z, x, y, 1, 3]
        intensity_grid = self.compute_intensity(self._xyz, grid_expanded, self.get_intensity, self.get_inv_covariance, self.get_scaling)
  
        return intensity_grid

    def compute_intensity(self, gaussian_centers, grid_point, intensity, inv_covariance, scaling):
        # grid_point: [1, z, x, y, 1, 3]
        z, x, y = grid_point.shape[1:4]
        # Initialize intensity_grid outside the loop
        intensity_grid = torch.zeros(1, z, x, y, 1, device='cuda', requires_grad=True)

        intensity_grid = compute_intensity(
            gaussian_centers.contiguous(),
            grid_point.contiguous(),
            intensity.contiguous(),
            inv_covariance.contiguous(),
            scaling.contiguous(),
            intensity_grid.contiguous(),
        )
        
        return intensity_grid
