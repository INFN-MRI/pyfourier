"""
Backend (computational and tensor) subpackage.

Adapted from https://github.com/mikgroup/sigpy/blob/main/sigpy/backend.py#L30

Also contains support routines for backend agnostic tensor creation (``zeros``)
and type checking (``is_complex``).

"""

__all__ = [
    "calc_blocks_per_grid",
    "get_backend",
    "to_backend",
    "get_device",
    "to_device",
    "astype",
    "ascontiguous",
    "zeros",
]

import warnings

import numpy as np
import numba as nb

from . import _config

if _config.cupy_enabled:
    import cupy as cp
if _config.pytorch_enabled:
    import torch


def calc_blocks_per_grid(numel, threadsperblock=128):
    """
    Calculate number of blocks per grid for a given grid size and threads per block.

    Parameters
    ----------
    numel : int
        Number of elements in the grid.
    threadsperblock : int, optional
        Desired number of threads per block.
        The default is 128.

    Returns
    -------
    blockspergrid : int
        Number of block per grid corresponding to input grid size
        and number of threads per block.

    """
    return (numel + (threadsperblock - 1)) // threadsperblock


def get_backend(input):
    """
    Get an appropriate module from :mod:`numpy`, :mod:`cupy` or :mod:`torch`.

    Parameters
    ----------
    input : npt.ArrayLike
        Input array like (np.ndarray, cp.ndarray or torch.Tensor).

    Returns
    -------
    module : ModuleType
        :mod:`numpy`, :mod:`cupy` or :mod:`torch`
        is returned based on input.

    """
    if isinstance(input, nb.cuda.cudadrv.devicearray.DeviceNDArray):
        return nb
    elif _config.pytorch_enabled and isinstance(input, torch.Tensor):
        return torch
    elif _config.cupy_enabled:
        return cp.get_array_module(input)
    else:
        return np


def get_device(input):
    """
    Get Device from input array.

    Parameters
    ----------
    input : int | npt.ArrayLike
        Input array like (np.ndarray, cp.ndarray or torch.Tensor) or Device.

    Returns
    -------
    device_id : int
        Device identifier. CPU is identified by -1; >= 0 marks the corresponding
        GPU.

    """
    if isinstance(input, int):
        device_id = input
    elif isinstance(input, np.ndarray):
        return -1
    elif isinstance(input, nb.cuda.cudadrv.driver.Device):
        device_id = input.id
    elif isinstance(input, nb.cuda.cudadrv.devicearray.DeviceNDArray):
        device_id = nb.cuda.get_current_device().id
    elif _config.cupy_enabled and isinstance(input, cp.cuda.Device):
        device_id = input.id
    elif _config.cupy_enabled and isinstance(input, cp.ndarray):
        device_id = input.device.id
    elif _config.pytorch_enabled and isinstance(input, torch.device):
        device_id = input.index
    elif _config.pytorch_enabled and isinstance(input, torch.Tensor):
        device_id = input.device.index
    else:
        raise ValueError(
            f"Accepts int, ArrayLike, Device, nb.cuda.cudadrv.driver.Device, cupy.cuda.Device or torch.device, got {input}"
        )
    return device_id


def to_backend(backend, *input):
    """
    Switch tensor backend of input array.

    Parameters
    ----------
    backend : ModuleType
        Target tensor backend (numpy, numba, cupy, torch).
    input : list[npt.ArrayLike]
        Input array(s) like (np.ndarray, cp.ndarray or torch.Tensor).


    Returns
    -------
    input : list[npt.ArrayLike]
        Zero-copy representation of input array(s) in target backend.

    """
    return [_to_backend(backend, arr) if arr is not None else None for arr in input]


def _to_backend(backend, input):
    # get initial backend
    ibackend = get_backend(input)
    device_id = get_device(input)

    # if the target backend is the same as source, exit
    if ibackend.__name__ == backend.__name__:
        return input

    # detach
    if ibackend.__name__ == "torch":
        input = input.detach().contiguous()

    # create torch tensor
    if backend.__name__ == "torch":
        if device_id == -1:
            return torch.as_tensor(input)
        return torch.as_tensor(input, device="cuda:" + str(device_id))

    # create cupy array
    if backend.__name__ == "cupy":
        with cp.cuda.Device(device_id):
            return cp.asarray(input)

    # create numba.cuda array
    if backend.__name__ == "numba":
        with nb.cuda.devices.gpus[device_id]:
            return nb.cuda.as_cuda_array(input)

    # create numpy array
    return np.asarray(input)


def to_device(input, device_id, backend=None):
    """
    Switch input array to a given device / backend.

    Parameters
    ----------
    input : npt.ArrayLike
        Input array like (np.ndarray, cp.ndarray or torch.Tensor).
    device_id : int
        Device identifier. CPU is identified by -1; >= 0 marks the corresponding
        GPU.
    backend : ModuleType, optional
        Target tensor backend (numpy, numba, cupy, torch).
        The default is None (use same backend as ``input``).

    Raises
    ------
    ValueError
        If input is a NumPy ndarray, no backend is specified and GPU is requested,
        throws an error (NumPy only supports CPU).

    Returns
    -------
    output : npt.ArrayLike
        Output array like (np.ndarray, cp.ndarray or torch.Tensor)
        on the target device (and backend, if specified).

    """
    # move to backend
    if backend is None:
        backend = get_backend(input)
        if backend.__name__ == "numpy" and device_id != -1:
            raise ValueError(
                f"Numpy backend only support CPU; requested device cuda:{device_id}; please specify a GPU-powered target backend."
            )
    else:
        input = to_backend(input, backend)

    # move torch tensor
    if backend.__name__ == "torch":
        if device_id == -1:
            return torch.as_tensor(input)
        return torch.as_tensor(input, device="cuda:" + str(device_id))

    # move cupy array
    if backend.__name__ == "cupy":
        with cp.cuda.Device(device_id):
            return cp.asarray(input)

    # move numba.cuda array
    if backend.__name__ == "numba":
        with nb.cuda.devices.gpus[device_id]:
            return nb.cuda.as_cuda_array(input)

    return np.asarray(input)


def astype(input, dtype):
    """
    Cast input tensor as different type.

    Parameters
    ----------
    input : npt.ArrayLike
        Input array like (np.ndarray, cp.ndarray or torch.Tensor).
    dtype : ModuleType.dtype
        Target datatype..

    Returns
    -------
    output : npt.ArrayLike
        Output array like (np.ndarray, cp.ndarray or torch.Tensor)
        of the desired dtype.

    """
    backend = get_backend(input)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if backend.__name__ == "torch":
            return input.to(dtype)
        else:
            return input.astype(dtype)


def ascontiguous(input):
    """
    Enforce data contiguity.

    Parameters
    ----------
    input : npt.ArrayLike
        Input array like (np.ndarray, cp.ndarray or torch.Tensor).

    Returns
    -------
    output : npt.ArrayLike
        Output contigous array like (np.ndarray, cp.ndarray or torch.Tensor)

    """
    backend = get_backend(input)
    if backend.__name__ == "torch":
        return input.contiguous()
    else:
        return backend.ascontiguousarray(input)


def zeros(shape, dtype, device, backend):
    """
    Create a zeroes tensor of given shape, data type, device and backend.

    Parameters
    ----------
    shape : int | tuple | list
        Output shape.
    dtype : ModuleType.dtype
        Output tensor dtype.
    device : ModuleType.device
        Computational device of output tensor.
    backend : ModuleType
        Tensor backend (numpy, numba, cupy or torch).

    Returns
    -------
    output : ArrayLike
        Zeroes tensor of given shape, data type, device and backend.

    """
    # create torch tensor
    if backend.__name__ == "torch":
        return torch.zeros(shape, dtype=dtype, device=device)

    # create cupy array
    if backend.__name__ == "cupy":
        with device:
            return cp.zeros(shape, dtype=dtype)

    # create numba.cuda array
    if backend.__name__ == "numba" and device.id != -1:
        with device:
            output = np.zeros(shape, dtype=dtype.name)
            return nb.cuda.as_cuda_array(output)

    return np.zeros(input, dtype=dtype)
