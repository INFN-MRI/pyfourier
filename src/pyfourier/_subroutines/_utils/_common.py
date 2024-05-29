"""Common sub-routines between noncart2cart, cart2noncart, dense2sparse and sparse2dense."""

__all__ = ["_is_complex_grid", "_get_shape"]

import numpy as np

from . import _backend
from . import _config

if _config.cupy_enabled:
    import cupy as cp
if _config.pytorch_enabled:
    import torch


def _get_oversamp_shape(shape, oversamp, ndim):
    """Determine oversampled shape."""
    return [np.ceil(oversamp[n] * shape[n]).astype(np.int16) for n in ndim]


def _is_complex_grid(input, basis, backend):
    """Determine if we need to perform complex gridding."""
    # check if data is complex
    is_complex = _is_complex(input)
    if is_complex:
        asreal = False
    else:
        asreal = True

    # check if basis is complex
    if basis is not None:
        complex_basis = _is_complex(input, basis)
        if complex_basis:
            input = _backend.astype(input, backend.complex64)
        elif is_complex:
            basis = _backend.astype(basis, backend.complex64)

    return input, basis, is_complex, asreal


def _is_complex(input):
    backend = _backend.get_backend(input)

    # create torch tensor
    if backend.__name__ == "torch":
        return torch.is_complex(input)
    elif backend.__name__ == "cupy":
        return cp.iscomplexobj(input)
    else:
        return np.iscomplexobj(input)


def _get_shape(data_in, index, shape, basis):  # noqa
    # get input sizes
    nframes = index[0].shape[0]
    npts = np.prod(shape)

    # reformat data for computation
    if nframes == 1:
        batch_shape = data_in.shape[:-2]
    else:
        batch_shape = data_in.shape[:-3]
    batch_size = int(np.prod(batch_shape))  # ncoils * nslices

    # get number of coefficients
    if basis is not None:
        ncoeff = basis.shape[0]
    else:
        ncoeff = nframes

    return batch_shape, batch_size, ncoeff, nframes, npts
