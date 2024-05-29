"""NonCartesian to Cartesian gridding subroutines."""

__all__ = ["_noncart2cart"]

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

_grid = {
    False: [_1D._grid, _2D._grid, _3D._grid],
    True: [_3Dstack._grid, _3Dstack._grid, _3Dstack._grid],
}
_grid_subspace = {
    False: [_1D_subspace._grid, _2D_subspace._grid, _3D_subspace._grid],
    True: [_3Dstack_subspace._grid, _3Dstack_subspace._grid, _3Dstack_subspace._grid],
}


def _noncart2cart(
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
    device_tag = _utils.get_device_tag(device)

    # get tensor backend
    backend = _utils.get_backend(data_in)

    # check if the data is complex and harmonize data_in and basis if necessary
    data_in, basis, is_complex, asreal = _utils._is_complex_grid(
        data_in, basis, backend
    )

    # get input sizes
    batch_shape, batch_size, ncoeff, nframes, npts = _utils._get_shape(
        data_in, index, dshape, basis
    )

    # transpose basis if required
    if basis is not None:
        basis = _utils.contiguous(basis.conj().T)

    # reformat data for computation
    data_in = data_in.reshape(batch_size, nframes, npts)
    data_in = _utils.ascontiguous(data_in.swapaxes(0, 1))

    # preallocate output data
    data_out = _utils.zeros(
        (ncoeff, batch_size, *ishape),
        dtype=data_in.dtype,
        device=device,
        backend=backend,
    )

    # get grid_function
    if basis is None:
        _do_gridding = _grid[is_stack][ndim - 1][device_tag][is_complex]
    else:
        _do_gridding = _grid_subspace[is_stack][ndim - 1][device_tag][is_complex]

    # switch to numba
    data_out, data_in, basis = _utils.to_backend(_nb, data_out, data_in, basis)

    # do actual gridding
    if device_tag == "cpu" and basis is None:
        _do_gridding(data_out, data_in, value, index)
    elif device_tag == "cpu" and basis is not None:
        _do_gridding(data_out, data_in, value, index, basis)
    if basis is None:
        blockspergrid = _utils.calc_blocks_per_grid(
            nframes * batch_size * npts, threadsperblock
        )
        _do_gridding[blockspergrid, threadsperblock](data_out, data_in, value, index)
    else:
        blockspergrid = _utils.calc_blocks_per_grid(
            nframes * batch_size * npts, threadsperblock
        )
        _do_gridding[blockspergrid, threadsperblock](
            data_out, data_in, value, index, basis
        )

    # switch to original backend
    data_out = _utils.to_backend(backend, data_out)

    # back to real, if required
    if asreal:
        data_out = _utils.astype(data_out, backend.float32)

    # reformat for output
    if nframes == 1:
        data_out = data_out[0].reshape(*batch_shape, *ishape)
    else:
        data_out = data_out.swapaxes(0, 1)
        data_out = data_out.reshape(*batch_shape, ncoeff, *ishape)

    # collect garbage
    _gc.collect()

    return data_out / scale
