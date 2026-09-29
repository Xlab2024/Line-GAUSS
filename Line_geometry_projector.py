import numpy as np
import torch
import torch.nn as nn

import astra


class Initialization_LineCT:
    def __init__(self, image_size, num_proj, proj_size,
                 source_det=800, source_origin=100, dis_step=0.59,
                 det_spacing=2.0, voxel_spacing=0.15, z_offset=0.0,
                 det_center_u=None, det_center_v=None,
                 det_center_in_raw_pixels=False,
                 det_center_raw_downsample=1.0,
                 det_center_postalignment_sign=-1.0):
        """
        Linear (line) CT geometry: source moves linearly in Y direction
        
        Args:
            num_proj: number of projections
            proj_size: [det_rows, det_cols] detector resolution
            proj_size[0] is the number of rows (detector pixels in Y direction), proj_size[1] is the number of cols (detector pixels in X direction)
            source_det: source-to-detector distance (SD)
            source_origin: source-to-rotation-center distance (SO)
            dis_step: step size for source movement in Y direction
            det_spacing: detector pixel spacing
            voxel_spacing: voxel size in object domain
        """
        self.image_size = image_size 
        self.num_proj = int(num_proj)
        self.proj_size = proj_size    # [det_rows, det_cols]

        self.source_det = float(source_det)
        self.source_origin = float(source_origin)
        self.origin_det = self.source_det - self.source_origin
        self.dis_step = float(dis_step)

        self.det_spacing_x = float(det_spacing)
        self.det_spacing_y = float(det_spacing)
        self.voxel_spacing = float(voxel_spacing)
        self.z_offset = float(z_offset)

        # Detector center / principal point correction
        self.det_center_u = None if det_center_u is None else float(det_center_u)
        self.det_center_v = None if det_center_v is None else float(det_center_v)
        self.det_center_in_raw_pixels = bool(det_center_in_raw_pixels)
        self.det_center_raw_downsample = float(det_center_raw_downsample)
        self.det_center_postalignment_sign = float(det_center_postalignment_sign)

        self.nz = int(image_size[0])  # D (depth / slice index)
        self.ny = int(image_size[1])  # H (height, ASTRA Y direction of source movement)
        self.nx = int(image_size[2])  # W (width, ASTRA X direction)  

        self.det_rows = int(proj_size[0])
        self.det_cols = int(proj_size[1])

        # Source displacement positions: linear motion in Y direction
        self.s_list_source = np.linspace(
            -(num_proj - 1) / 2 * dis_step,
            (num_proj - 1) / 2 * dis_step,
            num_proj,
            dtype=np.float32
        )

        # Magnification ratio for detector movement
        self.s_ratio = self.origin_det / self.source_origin


class _AstraLineCT:
    def __init__(self, param: Initialization_LineCT):
        self.param = param
        self.proj_geom = self._build_proj_geom()
        self.vol_geom = self._build_vol_geom()

    def _build_proj_geom(self):
        """Build ASTRA cone_vec geometry for linear CT"""
        p = self.param
        vectors = np.zeros((p.num_proj, 12), dtype=np.float32)

        for i, s_S in enumerate(p.s_list_source):
            s_D = s_S * p.s_ratio

            # Source point (only moving in Y direction)
            vectors[i, 0] = 0.0
            vectors[i, 1] = s_S 
            vectors[i, 2] = p.source_origin   

            # Detector center point
            vectors[i, 3] = 0.0
            vectors[i, 4] = -s_D
            vectors[i, 5] = -p.origin_det

            # Detector coordinate system
            # u vector: detector column direction, ASTRA X, corresponds to reconstruction W
            vectors[i, 6] = p.det_spacing_x
            vectors[i, 7] = 0.0
            vectors[i, 8] = 0.0

            # v vector: detector row direction, ASTRA Y, corresponds to reconstruction H
            vectors[i, 9] = 0.0
            vectors[i, 10] = p.det_spacing_y
            vectors[i, 11] = 0.0

        # ------------------------------------------------------------
        # Detector center / principal point correction
        # ------------------------------------------------------------
        # projection data shape = [P, det_rows, det_cols]
        # u direction = detector cols = horizontal direction
        # v direction = detector rows = vertical direction
        #
        # ASTRA cone_vec:
        # vectors[:, 3:6]  = detector center
        # vectors[:, 6:9]  = u vector, one detector-column step
        # vectors[:, 9:12] = v vector, one detector-row step
        # ------------------------------------------------------------
        if p.det_center_u is not None and p.det_center_v is not None:
            det_center_u = float(p.det_center_u)
            det_center_v = float(p.det_center_v)

            if p.det_center_in_raw_pixels:
                if p.det_center_raw_downsample <= 0:
                    raise ValueError("det_center_raw_downsample must be positive.")

                det_center_u = det_center_u / p.det_center_raw_downsample
                det_center_v = det_center_v / p.det_center_raw_downsample

            ideal_u = p.det_cols / 2.0
            ideal_v = p.det_rows / 2.0

            delta_u = det_center_u - ideal_u
            delta_v = det_center_v - ideal_v

            shift_u = p.det_center_postalignment_sign * delta_u
            shift_v = p.det_center_postalignment_sign * delta_v

            # Move detector center in physical vector geometry.
            vectors[:, 3:6] += shift_u * vectors[:, 6:9] + shift_v * vectors[:, 9:12]

            print("[LineCT detector center correction]")
            print(f"  detector size       : rows={p.det_rows}, cols={p.det_cols}")
            print(f"  ideal center        : u={ideal_u:.6f}, v={ideal_v:.6f}")
            print(f"  calibrated center   : u={det_center_u:.6f}, v={det_center_v:.6f}")
            print(f"  delta pixels        : du={delta_u:.6f}, dv={delta_v:.6f}")
            print(f"  sign                : {p.det_center_postalignment_sign:.6f}")
            print(f"  vector shift pixels : shift_u={shift_u:.6f}, shift_v={shift_v:.6f}")
            print(
                f"  physical shift      : "
                f"du={shift_u * p.det_spacing_x:.6f} mm, "
                f"dv={shift_v * p.det_spacing_y:.6f} mm"
            )
        else:
            print("[LineCT detector center correction] disabled, use ideal detector center.")
            print(f"  ideal center        : u={p.det_cols / 2.0:.6f}, v={p.det_rows / 2.0:.6f}")

        return astra.create_proj_geom('cone_vec', p.det_rows, p.det_cols, vectors)

    def _build_vol_geom(self):
        """Build ASTRA volume geometry for linear CT"""
        p = self.param
        vol_geom = astra.create_vol_geom(p.nx, p.ny, p.nz)

        sx = p.nx * p.voxel_spacing # W (ASTRA X)
        sy = p.ny * p.voxel_spacing # H (ASTRA Y, source movement direction)
        sz = p.nz * p.voxel_spacing # D (ASTRA Z)

        vol_geom['option'] = vol_geom.get('option', {})
        vol_geom['option']['WindowMinX'] = -sx / 2.0
        vol_geom['option']['WindowMaxX'] = sx / 2.0
        vol_geom['option']['WindowMinY'] = -sy / 2.0
        vol_geom['option']['WindowMaxY'] = sy / 2.0
        delta_z = getattr(p, 'z_offset', 0.0)
        vol_geom['option']['WindowMinZ'] = -sz / 2.0 + delta_z
        vol_geom['option']['WindowMaxZ'] = sz / 2.0 + delta_z

        return vol_geom

    def forward_project_np(self, vol_hwd):
        """
        vol_hwd: [H, W, D] float32 numpy
        return:  float32 numpy
        """
        
        vol_dhw = np.transpose(vol_hwd, (2,0,1)).astype(np.float32, copy=False)

        vol_id = astra.data3d.create('-vol', self.vol_geom, vol_dhw)
        sino_id = astra.data3d.create('-sino', self.proj_geom)
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id)
        sino = astra.data3d.get(sino_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)

        if sino.ndim != 3:
            raise RuntimeError(f'Unexpected sinogram dim: {sino.shape}')

        if sino.shape[1] == self.param.num_proj:
            sino = np.transpose(sino, (1, 0, 2))
        elif sino.shape[0] == self.param.num_proj:
            pass
        else:
            raise RuntimeError(f'Unexpected sinogram shape from ASTRA: {sino.shape}')

        return sino.astype(np.float32, copy=False)

    def back_project_np(self, sino_puv):
        """
        sino_puv: [num_proj, det_rows, det_cols] float32 numpy
        return: [H,W, D] float32 numpy
        """
        sino_astra = np.transpose(sino_puv, (1, 0, 2)).astype(np.float32, copy=False) # ASTRA projection:  (det_row, angle, det_col)

        sino_id = astra.data3d.create('-sino', self.proj_geom, sino_astra)
        vol_id = astra.data3d.create('-vol', self.vol_geom)

        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id)
        vol_dhw = astra.data3d.get(vol_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(sino_id)

        vol_hwd = np.transpose(vol_dhw, (1, 2, 0)).astype(np.float32, copy=False)  # [H, W, D]
        return vol_hwd


class _LineCTProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, projector_impl, voxel_spacing):
        """
        volume: [B, , Z]
        return: [B, P, U, V]
        """
        device = volume.device
        batch = int(volume.shape[0])

        out_list = []
        for b in range(batch):
            vol_np = volume[b].detach().to('cpu', dtype=torch.float32).numpy()
            sino_np = projector_impl.forward_project_np(vol_np)
            out_list.append(torch.from_numpy(sino_np))

        out = torch.stack(out_list, dim=0).to(device=device, dtype=volume.dtype)
        scale = float(voxel_spacing)
        out = out / scale

        ctx.projector_impl = projector_impl
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        projector_impl = ctx.projector_impl
        scale = ctx.scale
        device = grad_output.device

        grad_scaled = grad_output / scale
        batch = int(grad_scaled.shape[0])

        grad_in_list = []
        for b in range(batch):
            sino_np = grad_scaled[b].detach().to('cpu', dtype=torch.float32).numpy()
            vol_np = projector_impl.back_project_np(sino_np)
            grad_in_list.append(torch.from_numpy(vol_np))

        grad_input = torch.stack(grad_in_list, dim=0).to(device=device, dtype=grad_output.dtype)
        return grad_input, None, None


class Projection_LineCT(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.impl = _AstraLineCT(param)

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(-1)
        if x.ndim != 4:
            raise ValueError(f'Expected 4D volume input [B, H, W, D], got shape {tuple(x.shape)}')
        if x.shape[1] != self.param.ny or x.shape[2] != self.param.nx or x.shape[3] != self.param.nz:
            raise ValueError(
                f'Unexpected volume shape {tuple(x.shape)}; expected [B, H, W, D] '
                f'with image_size={self.param.image_size}'
            )
        return _LineCTProjectFunction.apply(x.contiguous(), self.impl, self.param.voxel_spacing)


class BP_LineCT(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.impl = _AstraLineCT(param)

    def forward(self, x):
        """
        x: [B, num_proj, det_rows, det_cols]
        return: [B, H, W, D]
        """
        device = x.device
        batch = int(x.shape[0])

        out_list = []
        for b in range(batch):
            sino_np = x[b].detach().to('cpu', dtype=torch.float32).numpy()
            vol_np = self.impl.back_project_np(sino_np)
            out_list.append(torch.from_numpy(vol_np))

        out = torch.stack(out_list, dim=0).to(device=device, dtype=x.dtype)
        return out


class LineCT3DProjector:
    def __init__(self, image_size, proj_size, num_proj,
                 source_det=800, source_origin=100, dis_step=0.59,
                 det_spacing=2.0, voxel_spacing=0.15, z_offset=0.0,
                 det_center_u=None, det_center_v=None,
                 det_center_in_raw_pixels=False,
                 det_center_raw_downsample=1.0,
                 det_center_postalignment_sign=-1.0):
        
        geo_param = Initialization_LineCT(
            image_size=image_size,
            num_proj=num_proj,
            proj_size=proj_size,
            source_det=source_det,
            source_origin=source_origin,
            dis_step=dis_step,
            det_spacing=det_spacing,
            voxel_spacing=voxel_spacing,
            z_offset=z_offset,
            det_center_u=det_center_u,
            det_center_v=det_center_v,
            det_center_in_raw_pixels=det_center_in_raw_pixels,
            det_center_raw_downsample=det_center_raw_downsample,
            det_center_postalignment_sign=det_center_postalignment_sign
        )

        self.forward_projector = Projection_LineCT(geo_param)
        self.bp = BP_LineCT(geo_param)

    def forward_project(self, volume):
        return self.forward_projector(volume)

    def backward_project(self, projs):
        return self.bp(projs)
