"""Dense to Sparse masking subroutines."""

__all__ = ["_dense2sparse"]

import gc as _gc
import numba as _nb

from ... import _utils

from . import _1D
from . import _1D_subspace
from . import _2D
from . import _2D_subspace
from . import _3D
from . import _3D_subspace

_mask = [_1D._mask, _2D._mask, _3D._mask]
_mask_subspace = [_1D_subspace._mask, _2D_subspace._mask, _3D_subspace._mask]


_cpu = _nb
if _utils.cupy_enabled():
    import cupy as _cp

    _gpu = _cp
else:
    _gpu = _nb


def _dense2sparse(data_in, mask, basis=None, device=None, threadsperblock=128):  # noqa
    # collect garbage
    _gc.collect()

    # unpack input
    index = mask.index
    dshape = mask.dshape  # data shape
    ishape = mask.ishape  # image shape
    ndim = mask.ndim
    device = mask.device
    device_tag = _utils.get_device_tag(device)

    # get tensor backend
    backend = _utils.get_backend(data_in)

    # check if the data is complex and harmonize data_in and basis if necessary
    data_in, basis, _, asreal = _utils._is_complex_grid(data_in, basis, backend)

    # get input sizes
    batch_shape, batch_size, ncoeff, nframes, npts = _utils._get_shape(
        data_in, index, dshape, basis
    )

    # reformat data for computation
    data_in = data_in.reshape(batch_size, ncoeff, *ishape)
    data_in = _utils.ascontiguous(data_in.swapaxes(0, 1))

    # preallocate output data
    data_out = _utils.zeros(
        (nframes, batch_size, npts), dtype=data_in.dtype, device=device, backend=backend
    )

    # get grid_function
    if basis is None:
        _do_mask = _mask[ndim - 1][device_tag]
    else:
        _do_mask = _mask_subspace[ndim - 1][device_tag]

    # switch to numba / cupy
    if device_tag == "cpu":
        data_out, data_in, basis = _utils.to_backend(_cpu, data_out, data_in, basis)
    else:
        data_out, data_in, basis = _utils.to_backend(_gpu, data_out, data_in, basis)

    # do actual gridding
    if device_tag == "cpu" and basis is None:
        _do_mask(data_out, data_in, index)
    elif device_tag == "cpu" and basis is not None:
        _do_mask(data_out, data_in, index, basis)
    if basis is None:
        blockspergrid = _utils.calc_blocks_per_grid(
            nframes * batch_size * npts, threadsperblock
        )
        _do_mask[blockspergrid, threadsperblock](data_out, data_in, index)
    else:
        blockspergrid = _utils.calc_blocks_per_grid(
            nframes * batch_size * npts, threadsperblock
        )
        _do_mask[blockspergrid, threadsperblock](data_out, data_in, index, basis)

    # switch to original backend
    data_out = _utils.to_backend(backend, data_out)

    # back to real, if required
    if asreal:
        data_out = _utils.astype(data_out, backend.float32)

    # reformat for output
    if nframes == 1:
        data_out = data_out[0].reshape(*batch_shape, *dshape)
    else:
        data_out = data_out.swapaxes(0, 1)
        data_out = data_out.reshape(*batch_shape, nframes, *dshape)

    # collect garbage
    _gc.collect()

    return data_out
