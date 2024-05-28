"""2D gridding subroutines with embedded subspace projection."""

__all__ = ["_grid"]

import numba as nb

from ... import _utils

# detect GPU
gpu_available, gpu_backend = _utils.detect_gpu_backend()


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _grid_nb(cart_data, noncart_data, interp_value, interp_index, basis):  # noqa
    # get sizes
    ncoeff, batch_size, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    yindex, xindex = interp_index
    yvalue, xvalue = interp_value

    # get interpolator width
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while spreading data
                        cart_data[coeff, batch, idy, idx] += (
                            val
                            * basis[coeff, frame]
                            * noncart_data[frame, batch, point]
                        )


_grid = {"cpu": {False: _grid_nb, True: _grid_nb}}

# %% GPU
if gpu_available and gpu_backend == "numba":
    from numba import cuda

    def _get_grid_nbcuda(is_complex):
        _update = _utils._update[is_complex]

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _grid_nbcuda(cart_data, noncart_data, interp_value, interp_index, basis):
            # get sizes
            ncoeff, batch_size, _, _ = cart_data.shape
            nframes = noncart_data.shape[0]
            npts = noncart_data.shape[-1]

            # unpack interpolator
            yindex, xindex = interp_index
            yvalue, xvalue = interp_value

            # get interpolator width
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes * batch_size * npts:
                # get current frame and k-space index
                frame = i // (batch_size * npts)
                tmp = i % (batch_size * npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while spreading data
                        for coeff in range(ncoeff):
                            _update(
                                cart_data,
                                (coeff, batch, idy, idx),
                                val
                                * basis[coeff, frame]
                                * noncart_data[frame, batch, point],
                            )

        return _grid_nbcuda

    _grid["gpu"] = {False: _get_grid_nbcuda(False), True: _get_grid_nbcuda(True)}

if gpu_available and gpu_backend == "cupy":
    from cupyx import jit

    def _get_grid_cupy(is_complex):
        _update = _utils._update[is_complex]

        @jit.rawkernel()  # pragma: no cover
        def _grid_cupy(cart_data, noncart_data, interp_value, interp_index, basis):
            # get sizes
            ncoeff, batch_size, _, _ = cart_data.shape
            nframes = noncart_data.shape[0]
            npts = noncart_data.shape[-1]

            # unpack interpolator
            yindex, xindex = interp_index
            yvalue, xvalue = interp_value

            # get interpolator width
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = jit.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes * batch_size * npts:
                # get current frame and k-space index
                frame = i // (batch_size * npts)
                tmp = i % (batch_size * npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        # do adjoint low rank projection (low-rank subspace -> time domain)
                        # while spreading data
                        for coeff in range(ncoeff):
                            _update(
                                cart_data,
                                (coeff, batch, idy, idx),
                                val
                                * basis[coeff, frame]
                                * noncart_data[frame, batch, point],
                            )

        return _grid_cupy

    _grid["gpu"] = {False: _get_grid_cupy(False), True: _get_grid_cupy(True)}
