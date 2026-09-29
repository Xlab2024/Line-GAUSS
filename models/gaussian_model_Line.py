import torch

from models.gaussian_model import GaussianModel as BaseGaussianModel
from gs_utils.general_utils import build_rotation


# Line-CT Gaussian model with sampling-aware 3D Gaussian scale-space filtering.
class GaussianModelLine(BaseGaussianModel):
    def __init__(self):
        super().__init__()
        self.filter_3D = None
        self.smooth_filter_enabled = True
        self.smooth_filter_s = 0.2
        self.smooth_filter_interval = 100
        self.smooth_filter_margin = 0.15
        self.smooth_filter_min_depth = 1e-4
        self._last_filter_iter = -1

    def _has_valid_filter(self):
        """Check whether the cached filter matches the current Gaussian set."""
        if self.filter_3D is None:
            return False
        return self.filter_3D.shape[0] == self._xyz.shape[0]

    def _invalidate_filter(self):
        self.filter_3D = None
        self._last_filter_iter = -1

    def configure_smooth_filter(
        self,
        enabled=True,
        smooth_filter_s=0.2,
        update_interval=100,
        margin=0.15,
        min_depth=1e-4,
    ):
        self.smooth_filter_enabled = bool(enabled)
        self.smooth_filter_s = float(smooth_filter_s)
        self.smooth_filter_interval = int(update_interval)
        self.smooth_filter_margin = float(margin)
        self.smooth_filter_min_depth = float(min_depth)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        if (not self.smooth_filter_enabled) or (not self._has_valid_filter()):
            return scales
        return torch.sqrt(torch.square(scales) + torch.square(self.filter_3D))

    @property
    def get_intensity_with_3D_filter(self):
        intensity = self.get_intensity
        if (not self.smooth_filter_enabled) or (not self._has_valid_filter()):
            return intensity

        scales = self.get_scaling
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(torch.clamp(det1 / det2, min=1e-8))
        return intensity * coef[..., None]

    @property
    def get_inv_covariance_with_3D_filter(self):
        scaling = self.get_scaling_with_3D_filter
        rotation = self.get_rotation
        scaling_inv_squared = 1.0 / torch.clamp(scaling * scaling, min=1e-12)
        s_inv_squared = torch.diag_embed(scaling_inv_squared)
        r = build_rotation(rotation)
        r_transpose = r.transpose(1, 2)
        covariance_inv = torch.matmul(r, torch.matmul(s_inv_squared, r_transpose))
        return covariance_inv

    def maybe_update_3D_filter(self, iteration, line_projector):
        if not self.smooth_filter_enabled:
            return
        if self._xyz.numel() == 0:
            return
        if self.smooth_filter_interval <= 0:
            return
        need_refresh = (not self._has_valid_filter())
        if need_refresh or (iteration == 0) or (iteration - self._last_filter_iter >= self.smooth_filter_interval):
            self.compute_3D_filter_line(line_projector)
            self._last_filter_iter = int(iteration)

    @torch.no_grad()
    def compute_3D_filter_line(self, line_projector):
        """Compute the sampling-aware 3D Gaussian filter for linear CT geometry."""
        if self._xyz.numel() == 0:
            self.filter_3D = None
            return

        impl = line_projector.forward_projector.impl
        p = impl.param

        device = self._xyz.device
        dtype = self._xyz.dtype

        s_list_source = torch.from_numpy(p.s_list_source).to(device=device, dtype=dtype)
        s_ratio = torch.tensor(float(p.s_ratio), device=device, dtype=dtype)
        source_origin = torch.tensor(float(p.source_origin), device=device, dtype=dtype)
        origin_det = torch.tensor(float(p.origin_det), device=device, dtype=dtype)

        s_det = s_list_source * s_ratio
        zeros = torch.zeros_like(s_list_source)

        # Source and detector move along world Y direction (not X)
        # Corresponding to ASTRA Y axis, which is now H (height) in our volume layout
        sources = torch.stack(
            [
                zeros,
                s_list_source,
                source_origin * torch.ones_like(s_list_source),
            ],
            dim=1,
        )

        det_centers = torch.stack(
            [
                zeros,
                -s_det,
                -origin_det * torch.ones_like(s_list_source),
            ],
            dim=1,
        )

        u_vec = torch.stack(
            [
                torch.tensor(float(p.det_spacing_x), device=device, dtype=dtype) * torch.ones_like(s_list_source),
                torch.zeros_like(s_list_source),
                torch.zeros_like(s_list_source),
            ],
            dim=1,
        )
        v_vec = torch.stack(
            [
                torch.zeros_like(s_list_source),
                torch.tensor(float(p.det_spacing_y), device=device, dtype=dtype) * torch.ones_like(s_list_source),
                torch.zeros_like(s_list_source),
            ],
            dim=1,
        )

        # Detector center / principal point correction
        # Keep smooth-filter geometry consistent with ASTRA cone_vec geometry.
        det_center_u = getattr(p, 'det_center_u', None)
        det_center_v = getattr(p, 'det_center_v', None)

        if det_center_u is not None and det_center_v is not None:
            det_center_u = float(det_center_u)
            det_center_v = float(det_center_v)

            if bool(getattr(p, 'det_center_in_raw_pixels', False)):
                raw_downsample = float(getattr(p, 'det_center_raw_downsample', 1.0))
                if raw_downsample <= 0:
                    raise ValueError("det_center_raw_downsample must be positive.")
                det_center_u = det_center_u / raw_downsample
                det_center_v = det_center_v / raw_downsample

            ideal_u = float(p.det_cols) / 2.0
            ideal_v = float(p.det_rows) / 2.0

            delta_u = det_center_u - ideal_u
            delta_v = det_center_v - ideal_v

            sign = float(getattr(p, 'det_center_postalignment_sign', -1.0))
            shift_u = sign * delta_u
            shift_v = sign * delta_v

            det_centers = det_centers + shift_u * u_vec + shift_v * v_vec

        ray_dir = det_centers - sources
        ray_dir = ray_dir / torch.clamp(torch.norm(ray_dir, dim=1, keepdim=True), min=1e-8)

        vol_nx = float(p.nx)  # W dimension
        vol_ny = float(p.ny)  # H dimension (world Y, source movement direction)
        vol_nz = float(p.nz)  # D dimension
        voxel_spacing = float(p.voxel_spacing)
        sx = vol_nx * voxel_spacing  # W -> world X
        sy = vol_ny * voxel_spacing  # H -> world Y
        sz = vol_nz * voxel_spacing  # D -> world Z

        xyz = self.get_xyz  # [N, 3] in format [D, H, W]
        # Map to world coordinates: W->X, H->Y, D->Z
        points_world = torch.stack(
            [
                (xyz[:, 2] - 0.5) * sx,  # W dimension -> world X
                (xyz[:, 1] - 0.5) * sy,  # H dimension -> world Y
                (xyz[:, 0] - 0.5) * sz,  # D dimension -> world Z
            ],
            dim=1,
        )

        # If a global Z offset is specified in the projector params, apply it
        # so Gaussian-model world Z aligns with ASTRA vol_geom window (which
        # uses p.z_offset). Convert to tensor matching device/dtype.
        delta_z = float(getattr(p, 'z_offset', 0.0))
        if delta_z != 0.0:
            points_world[:, 2] = points_world[:, 2] + delta_z

        sp = points_world.unsqueeze(0) - sources.unsqueeze(1)
        depth = torch.sum(sp * ray_dir.unsqueeze(1), dim=-1)

        plane_dist = torch.sum((det_centers - sources) * ray_dir, dim=-1, keepdim=True)
        denom = torch.clamp(depth, min=self.smooth_filter_min_depth)
        t = plane_dist / denom

        hit = sources.unsqueeze(1) + t.unsqueeze(-1) * sp
        rel = hit - det_centers.unsqueeze(1)

        u_den = torch.sum(u_vec * u_vec, dim=1, keepdim=True)
        v_den = torch.sum(v_vec * v_vec, dim=1, keepdim=True)
        u_pix = torch.sum(rel * u_vec.unsqueeze(1), dim=-1) / torch.clamp(u_den, min=1e-12)
        v_pix = torch.sum(rel * v_vec.unsqueeze(1), dim=-1) / torch.clamp(v_den, min=1e-12)

        u_lim = (float(p.det_cols) * 0.5) * (1.0 + self.smooth_filter_margin)
        v_lim = (float(p.det_rows) * 0.5) * (1.0 + self.smooth_filter_margin)

        valid_depth = depth > self.smooth_filter_min_depth
        in_screen = (u_pix.abs() <= u_lim) & (v_pix.abs() <= v_lim)
        valid = valid_depth & in_screen


        sd_view = plane_dist

        f_eff_u = sd_view / torch.tensor(float(p.det_spacing_x), device=device, dtype=dtype)
        f_eff_v = sd_view / torch.tensor(float(p.det_spacing_y), device=device, dtype=dtype)
        f_eff = torch.maximum(f_eff_u, f_eff_v)


        nu = torch.where(valid, f_eff / torch.clamp(depth, min=self.smooth_filter_min_depth), torch.zeros_like(depth))
        nu_k, _ = torch.max(nu, dim=0)

        has_valid = torch.any(valid, dim=0)
        safe_nu = torch.where(has_valid, nu_k, torch.full_like(nu_k, 1e-6))
        t_hat = 1.0 / torch.clamp(safe_nu, min=1e-6)

        filter_3d = t_hat * (self.smooth_filter_s ** 0.5)
        self.filter_3D = filter_3d[..., None]

    def densification_postfix(self, new_xyz, new_intensities, new_scaling, new_rotation):
        super().densification_postfix(new_xyz, new_intensities, new_scaling, new_rotation)
        self._invalidate_filter()

    def prune_points(self, mask):
        super().prune_points(mask)
        self._invalidate_filter()

    def grid_sample(self, grid):
        grid_expanded = grid.unsqueeze(-2)
        if self.smooth_filter_enabled and self._has_valid_filter():
            intensity_grid = self.compute_intensity(
                self._xyz,
                grid_expanded,
                self.get_intensity_with_3D_filter,
                self.get_inv_covariance_with_3D_filter,
                self.get_scaling_with_3D_filter,
            )
        else:
            intensity_grid = self.compute_intensity(
                self._xyz,
                grid_expanded,
                self.get_intensity,
                self.get_inv_covariance,
                self.get_scaling,
            )
        return intensity_grid
