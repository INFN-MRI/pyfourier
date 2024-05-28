"""3D stack of trajectories de-gridding subroutines with embedded subspace projection."""

__all__ = ["_degrid"]

import numba as nb

from ... import _utils

# detect GPU
gpu_available, gpu_backend = _utils.detect_gpu_backend()


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _degrid_nb(
    noncart_data, cart_data, interp_value, interp_index, basis_adjoint
):  # noqa
    # get sizes
    ncoeff, batch_size, _, _, _ = cart_data.shape
    nframes = noncart_data.shape[0]
    npts = noncart_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = interp_index
    _, yvalue, xvalue = interp_value

    # get interpolator width
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        idz = zindex[frame, point, 0]

        for i_y in range(ywidth):
            idy = yindex[frame, point, i_y]
            valy = yvalue[frame, point, i_y]

            for i_x in range(xwidth):
                idx = xindex[frame, point, i_x]
                val = valy * xvalue[frame, point, i_x]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while gathering data
                for coeff in range(ncoeff):
                    noncart_data[frame, batch, point] += (
                        val
                        * basis_adjoint[frame, coeff]
                        * cart_data[coeff, batch, idz, idy, idx]
                    )


_degrid = {"cpu": _degrid_nb}

# %% GPU
if gpu_available and gpu_backend == "numba":
    from numba import cuda

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _degrid_nbcuda(
        noncart_data, cart_data, interp_value, interp_index, basis_adjoint
    ):
        # get sizes
        ncoeff, batch_size, _, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        _, yvalue, xvalue = interp_value

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

            # gather data within kernel radius
            idz = zindex[frame, point, 0]

            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    # do adjoint low rank projection (low-rank subspace -> time domain)
                    # while gathering data
                    for coeff in range(ncoeff):
                        noncart_data[frame, batch, point] += (
                            val
                            * basis_adjoint[frame, coeff]
                            * cart_data[coeff, batch, idz, idy, idx]
                        )

    _degrid["gpu"] = _degrid_nbcuda

if gpu_available and gpu_backend == "cupy":
    from cupyx import jit

    @jit.rawkernel()  # pragma: no cover
    def _degrid_cupy(
        noncart_data, cart_data, interp_value, interp_index, basis_adjoint
    ):
        # get sizes
        ncoeff, batch_size, _, _, _ = cart_data.shape
        nframes = noncart_data.shape[0]
        npts = noncart_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = interp_index
        _, yvalue, xvalue = interp_value

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

            # gather data within kernel radius
            idz = zindex[frame, point, 0]

            for i_y in range(ywidth):
                idy = yindex[frame, point, i_y]
                valy = yvalue[frame, point, i_y]

                for i_x in range(xwidth):
                    idx = xindex[frame, point, i_x]
                    val = valy * xvalue[frame, point, i_x]

                    # do adjoint low rank projection (low-rank subspace -> time domain)
                    # while gathering data
                    for coeff in range(ncoeff):
                        noncart_data[frame, batch, point] += (
                            val
                            * basis_adjoint[frame, coeff]
                            * cart_data[coeff, batch, idz, idy, idx]
                        )

    _degrid["gpu"] = _degrid_cupy
