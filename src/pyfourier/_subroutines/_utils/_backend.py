"""
"""

__all__ = ["detect_gpu_backend"]

import numba as nb

def detect_gpu_backend():
    """
    Detect if a GPU device is present and select GPU backend.
    
    If ``CuPy`` is installed, select it. This can support
    both ``CUDA`` and ``AMD`` devices, depending on the installed version.
    
    If ``CuPy`` is not installed, select ``Numba``. This require less
    dependencies but does not support ``AMD`` devices.

    Returns
    -------
    gpu_available : bool
        Whether a GPU is available or not.
    gpu_backend : str
        Type of GPU backend (``"numba"`` or ``"cupy"``).

    """
    if nb.cuda.is_available():
        gpu_available = True
        try:
            gpu_backend = "cupy"
        except Exception:
            gpu_backend = "numba"
    else:
        gpu_available = False
    
    return gpu_available, gpu_backend

