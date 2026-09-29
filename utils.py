import os
import yaml
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils

from data import ImageDataset, ImageDataset_2D, ImageDataset_3D


def get_config(config):
    try:
        with open(config, 'r', encoding='utf-8') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)
    except UnicodeDecodeError:
        with open(config, 'r', encoding='gbk') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory



def get_data_loader(data, img_path, img_dim, img_slice,
                    train, batch_size, 
                    num_workers=1, 
                    return_data_idx=False):
    # If img_path is empty or data indicates realdata without an image,
    # provide a dummy dataset that yields a zero image and matching grid.
    if (not img_path) and ('real' in data or data == 'realdata'):
        # REAL_DATA: Provide a dummy dataset when no img_path supplied for real-data runs.
        # This avoids FileNotFoundError during training when visual image is optional.
        # The dummy dataset yields a zero volume and matching grid used only for visualization/grid.
        class Dummy3D(Dataset):
            def __init__(self, img_dim):
                self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
                z, h, w = self.img_dim
                img = np.zeros((z, h, w), dtype=np.float32)
                img = torch.tensor(img)[None, ...]  # [B, C, H, W]
                self.img = img.permute(1, 2, 3, 0)  # [C, H, W, 1]

            def __getitem__(self, idx):
                # For real-data dummy dataset we return a zero grid matching image dims.
                # REAL_DATA: placeholder grid to satisfy visualization API.
                # Return a coordinate grid with shape [Z, X, Y, 3] (z,x,y coords)
                # so that DataLoader adds the batch dim and downstream code
                # sees [B, Z, X, Y, 3], matching `GaussianModel.grid_sample` expectation.
                z, x, y = self.img_dim
                # build meshgrid in (z, x, y) order and normalize by size to match
                # how gaussians normalize sampled indices (coords / [D,H,W]).
                zz = torch.arange(z, dtype=torch.float32)
                xx = torch.arange(x, dtype=torch.float32)
                yy = torch.arange(y, dtype=torch.float32)
                gz, gx, gy = torch.meshgrid(zz, xx, yy)
                grid = torch.stack((gz, gx, gy), dim=-1) / torch.tensor([z, x, y], dtype=torch.float32)
                return grid, self.img

            def __len__(self):
                return 1

        dataset = Dummy3D(img_dim)
    else:
        if data == 'phantom':
            dataset = ImageDataset(img_path, img_dim)
        elif '3d' in data:
            dataset = ImageDataset_3D(img_path, img_dim)
        else:
            dataset = ImageDataset_2D(img_path, img_dim, img_slice)

    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=train, 
                        drop_last=train, 
                        num_workers=num_workers)
    return loader


def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor_cpu = tensor.detach().cpu()
    # Normalize slice_idx to a list and clamp to available range to avoid IndexError
    if not isinstance(slice_idx, (list, tuple)):
        slice_idx = [int(slice_idx)]
    else:
        slice_idx = [int(x) for x in slice_idx]
    available = tensor_cpu.shape[1] if tensor_cpu.ndim > 1 else tensor_cpu.shape[0]
    valid_idx = [i for i in slice_idx if 0 <= i < available]
    if len(valid_idx) == 0:
        # fallback: use a simple evenly spaced set of slices across available range
        step = max(1, available // max(1, image_num))
        valid_idx = list(range(0, available, step))[:image_num]
    image_num = len(valid_idx)
    tensor_proc = tensor_cpu[0, valid_idx, ...].permute(0, 3, 1, 2)  # [n, c, h, w]
    # rotate each slice clockwise 90 degrees to match desired orientation
    # try:
    #     tensor_proc = torch.rot90(tensor_proc, k=-1, dims=(2, 3))
    # except Exception:
    #     pass
    # resize slices to 256x256 for consistent viewers
    try:
        tensor_proc = F.interpolate(tensor_proc, size=(256, 256), mode='bilinear', align_corners=False)
    except Exception:
        pass
    image_grid = vutils.make_grid(tensor_proc, nrow=image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)
    # REAL_DATA: stitched PNGs (grid images) should NOT produce raw/.npy exports.
    # Raw/.npy exports are created from projection/bp volumes via save_image_3d_slices.


def save_image_3d_slices(tensor, slice_idx, output_dir, prefix='slice'):
    os.makedirs(output_dir, exist_ok=True)
    tensor_cpu = tensor.detach().cpu()
    # Normalize and clamp slice indices to available range
    if not isinstance(slice_idx, (list, tuple)):
        slice_idx = [int(slice_idx)]
    else:
        slice_idx = [int(x) for x in slice_idx]
    available = tensor_cpu.shape[1] if tensor_cpu.ndim > 1 else tensor_cpu.shape[0]
    valid_idx = [i for i in slice_idx if 0 <= i < available]
    if len(valid_idx) == 0:
        valid_idx = list(range(available))
    slice_tensor = tensor_cpu[0, valid_idx, ...].permute(0, 3, 1, 2)
    # rotate and resize previews so PNGs match raw orientation/size
    # try:
    #     slice_tensor = torch.rot90(slice_tensor, k=-1, dims=(2, 3))
    # except Exception:
    #     pass
    try:
        slice_tensor = F.interpolate(slice_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    except Exception:
        pass
    # REAL_DATA: only save raw/.npy for projection and bp volumes (prefix 'proj' or 'bp').
    if prefix in ('proj', 'bp', 'fbp', 'recon', 'test', 'recon_full_bg_roi'):
        try:
            base = os.path.join(output_dir, prefix + "_volume")
            # For projections we want filename order: det_cols x det_rows x num_proj
            if prefix == 'proj':
                _save_raw_and_npy_from_tensor(tensor_cpu[0], base, reorder_for_name=( 2, 1, 0))
            else:
                # For reconstructed volumes (bp/fbp/recon/test) do NOT rotate the (H,W)
                # slices when saving raw/.npy. ImageJ expects filenames in `W x H x D` order,
                # so transpose to (W, H, Z) when building the filename.
                _save_raw_and_npy_from_tensor(tensor_cpu[0], base, reorder_for_name=(2, 1, 0), rotate90=False, target_shape=None)
        except Exception:
            pass
    # Iterate over the *valid* indices used to build `slice_tensor` to avoid empty slices
    for idx, slice_num in enumerate(valid_idx):
        file_path = os.path.join(output_dir, f"{prefix}_{slice_num}.png")
        vutils.save_image(slice_tensor[idx:idx + 1], file_path, normalize=True, scale_each=True)


def _save_raw_and_npy_from_tensor(tensor_or_array, out_path_base, reorder_for_name=None, rotate90=False, target_shape=None):
    """Save a tensor/ndarray to out_path_base_{dims}.raw and .npy.

    - tensor_or_array: Torch tensor or numpy array. Expected shapes:
      [Z,H,W,1] or [P,H,W,1] or [Z,H,W] etc. If there is a trailing channel dim
      or leading batch dim, caller should pass the sliced item (we assume no batch dim).
    - out_path_base: path without extension; function appends _<dimstr>.raw/.npy
    """
    # Convert to numpy
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().numpy()
    else:
        arr = np.array(tensor_or_array)

    # Remove singleton channel dim at end if present
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # If there is a leading channel dim (C, Z, H) or similar, try to squeeze
    arr = np.squeeze(arr)

    # Ensure 3D array
    if arr.ndim != 3:
        # If it's 2D (single slice), expand to (1, H, W)
        if arr.ndim == 2:
            arr = arr[None, ...]
        else:
            # as fallback, try to flatten to 3D by adding axes
            while arr.ndim < 3:
                arr = arr[None, ...]

    # Optionally rotate each (H,W) slice clockwise 90 degrees to match viewer orientation.
    if rotate90:
        try:
            arr = np.rot90(arr, k=-1, axes=(1, 2))
        except Exception:
            pass

    # If a target_shape is requested, resample the 3D volume to that shape using trilinear interpolation.
    if target_shape is not None:
        try:
            if tuple(arr.shape) != tuple(target_shape):
                t = torch.tensor(arr, dtype=torch.float32)[None, None, ...]  # [1,1,Z,H,W]
                t_resized = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
                arr = t_resized[0, 0].cpu().numpy()
        except Exception:
            pass

    # Build filename with dims. Allow reordering for human-friendly naming.
    arr_for_name = arr
    if reorder_for_name is not None:
        try:
            arr_for_name = np.transpose(arr, reorder_for_name)
        except Exception:
            arr_for_name = arr

    # Build filename with dims
    shape_str = 'x'.join(str(int(x)) for x in arr_for_name.shape)
    raw_path = f"{out_path_base}_{shape_str}.raw"
    npy_path = f"{out_path_base}_{shape_str}.npy"

    # Choose dtype for saving: preserve integer types, otherwise float32
    if np.issubdtype(arr.dtype, np.integer):
        save_arr = arr
    else:
        save_arr = arr.astype(np.float32, copy=False)

    # Save .npy
    try:
        np.save(npy_path, save_arr)
    except Exception:
        pass

    # Save raw binary
    try:
        # ensure contiguous
        save_arr.ravel().tofile(raw_path)
    except Exception:
        pass



def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (B, H, W, C)
    coordinates: (2, ...)
    '''
    bs, h, w, c = input.size()

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)

    f00 = input[:, co_floor[0], co_floor[1], :]
    f10 = input[:, co_floor[0], co_ceil[1], :]
    f01 = input[:, co_ceil[0], co_floor[1], :]
    f11 = input[:, co_ceil[0], co_ceil[1], :]
    d1 = d1[None, :, :, None].expand(bs, -1, -1, c)
    d2 = d2[None, :, :, None].expand(bs, -1, -1, c)

    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    
    return fx1 + d2 * (fx2 - fx1)


def ct_parallel_project_2d(img, theta):
	bs, h, w, c = img.size()

	# (y, x)=(i, j): [0, w] -> [-0.5, 0.5]
	y, x = torch.meshgrid([torch.arange(h, dtype=torch.float32) / h - 0.5,
							torch.arange(w, dtype=torch.float32) / w - 0.5])

	# Rotation transform matrix: simulate parallel projection rays
	x_rot = x * torch.cos(theta) - y * torch.sin(theta)
	y_rot = x * torch.sin(theta) + y * torch.cos(theta)

	# Reverse back to index [0, w]
	x_rot = (x_rot + 0.5) * w
	y_rot = (y_rot + 0.5) * h

	# Resample (x, y) index of the pixel on the projection ray-theta
	sample_coords = torch.stack([y_rot, x_rot], dim=0).cuda()  # [2, h, w]
	img_resampled = map_coordinates(img, sample_coords) # [b, h, w, c]

	# Compute integral projections along rays
	proj = torch.mean(img_resampled, dim=1, keepdim=True) # [b, 1, w, c]

	return proj


def ct_parallel_project_2d_batch(img, thetas):
    '''
    img: input tensor [B, H, W, C]
    thetas: list of projection angles
    '''
    projs = []
    for theta in thetas:
        proj = ct_parallel_project_2d(img, theta)
        projs.append(proj)
    projs = torch.cat(projs, dim=1)  # [b, num, w, c]

    return projs
