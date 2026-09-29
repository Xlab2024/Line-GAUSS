import torch
from torch.autograd import Function
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "_ext_build")
os.makedirs(_BUILD_DIR, exist_ok=True)

_SCRIPTS_DIR = os.path.join(os.path.dirname(sys.executable), "Scripts")
if os.path.isdir(_SCRIPTS_DIR):
    os.environ["PATH"] = _SCRIPTS_DIR + os.pathsep + os.environ.get("PATH", "")


_compute_intensity_cuda = None


def _clear_stale_build_lock(max_age_seconds=None):
    if max_age_seconds is None:
        max_age_seconds = int(os.environ.get("COMPUTE_INTENSITY_LOCK_MAX_AGE", "120"))
    lock_path = os.path.join(_BUILD_DIR, "lock")
    if not os.path.exists(lock_path):
        return

    try:
        lock_age = time.time() - os.path.getmtime(lock_path)
        if lock_age > max_age_seconds:
            print(f"[Compute_intensity] Removing stale build lock (age={lock_age:.1f}s): {lock_path}", flush=True)
            os.remove(lock_path)
        else:
            print(f"[Compute_intensity] Build lock exists (age={lock_age:.1f}s), waiting for other process...", flush=True)
    except Exception as exc:
        print(f"[Compute_intensity] Failed to inspect/remove build lock: {exc}", flush=True)


def _get_compute_intensity_cuda():
    global _compute_intensity_cuda
    if _compute_intensity_cuda is None:
        print("[Compute_intensity] Loading/compiling CUDA extension...", flush=True)
        _clear_stale_build_lock()
        # Suppress specific nvcc diagnostic warnings (e.g. #221) that are benign
        # and can clutter build output on some CUDA/host combos. Keep optimization flag.
        extra_cuda_cflags = ['-O3', '-Xcudafe', '--diag_suppress=221']
        _compute_intensity_cuda = load(
            name='compute_intensity_cuda_v2',
            sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discretize_grid.cu')],
            extra_cflags=['-O3'],
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=_BUILD_DIR,
            verbose=True,
        )
        print("[Compute_intensity] CUDA extension ready.", flush=True)
    return _compute_intensity_cuda


class IntensityComputation(Function):
    @staticmethod
    def forward(ctx, gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
        ctx.save_for_backward(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
        compute_intensity_cuda = _get_compute_intensity_cuda()

        # Call the forward CUDA function
        intensity_grid = compute_intensity_cuda.compute_intensity(
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )
        
        return intensity_grid

    @staticmethod
    def backward(ctx, grad_output):
        gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid = ctx.saved_tensors
        compute_intensity_cuda = _get_compute_intensity_cuda()

        # Call the backward CUDA function
        grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_intensity_grid = compute_intensity_cuda.compute_intensity_backward(
            grad_output,
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scalings,
            intensity_grid
        )

        return grad_gaussian_centers, None, grad_intensities, grad_inv_covariances, None, grad_intensity_grid


def compute_intensity(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid):
    return IntensityComputation.apply(gaussian_centers, grid_points, intensities, inv_covariances, scalings, intensity_grid)
