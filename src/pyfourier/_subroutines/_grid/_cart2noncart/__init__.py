"""Cartesian to NonCartesian degridding subroutines."""

__all__ = ["_cart2noncart"]

import gc as _gc
import numba as _nb

from ... import _utils

from . import _1D
from . import _1D_subspace
from . import _2D
from . import _2D_subspace
from . import _3D
from . import _3D_subspace
from . import _3Dstack
from . import _3Dstack_subspace

_degrid = {
    False: [_1D._degrid, _2D._degrid, _3D._degrid],
    True: [_3Dstack._degrid, _3Dstack._degrid, _3Dstack._degrid],
}
_degrid_subspace = {
    False: [_1D_subspace._degrid, _2D_subspace._degrid, _3D_subspace._degrid],
    True: [
        _3Dstack_subspace._degrid,
        _3Dstack_subspace._degrid,
        _3Dstack_subspace._degrid,
    ],
}


def _cart2noncart(
    data_in, interpolator, basis=None, device=None, threadsperblock=128
):  # noqa
    # collect garbage
    _gc.collect()

    # unpack input
    index = interpolator.index
    value = interpolator.value
    dshape = interpolator.dshape  # data shape
    ishape = interpolator.ishape  # image shape
    ndim = interpolator.ndim
    is_stack = interpolator.is_stack
    scale = interpolator.scale
    device = interpolator.device

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
    data_in = _utils.contiguous(data_in.swapaxes(0, 1))

    # preallocate output data
    data_out = _utils.zeros(
        (nframes, batch_size, npts), dtype=data_in.dtype, device=device, backend=backend
    )

    # get grid_function
    if basis is None:
        _do_degridding = _degrid[is_stack][ndim - 1][device]
    else:
        _do_degridding = _degrid_subspace[is_stack][ndim - 1][device]

    # switch to numba
    data_out, data_in, basis = _utils.to_backend(_nb, data_out, data_in, basis)

    # do actual gridding
    if device == "cpu" and basis is None:
        _do_degridding(data_out, data_in, value, index)
    elif device == "cpu" and basis is not None:
        _do_degridding(data_out, data_in, value, index, basis)
    if basis is None:
        blockspergrid = _utils.calc_blocks_per_grid(
            nframes * batch_size * npts, threadsperblock
        )
        _do_degridding[blockspergrid, threadsperblock](data_out, data_in, value, index)
    else:
        blockspergrid = _utils.calc_blocks_per_grid(
            nframes * batch_size * npts, threadsperblock
        )
        _do_degridding[blockspergrid, threadsperblock](
            data_out, data_in, value, index, basis
        )

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

    return data_out / scale
