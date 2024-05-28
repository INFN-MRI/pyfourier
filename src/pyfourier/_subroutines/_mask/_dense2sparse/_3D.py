"""3D masking subroutines."""

__all__ = ["_mask"]

import numba as nb

from ... import _utils

# detect GPU
gpu_available, gpu_backend = _utils.detect_gpu_backend()


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _mask_nb(sparse_data, dense_data, index):  # noqa
    # get sizes
    nframes, batch_size, _, _, _ = dense_data.shape
    npts = sparse_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = index

    # parallelize over frames, batches and k-space points
    for i in nb.prange(nframes * batch_size * npts):  # pylint: disable=not-an-iterable
        # get current frame and k-space index
        frame = i // (batch_size * npts)
        tmp = i % (batch_size * npts)
        batch = tmp // npts
        point = tmp % npts

        # gather data within kernel radius
        idz = zindex[frame, point]
        idy = yindex[frame, point]
        idx = xindex[frame, point]
        sparse_data[frame, batch, point] += dense_data[frame, batch, idz, idy, idx]


_mask = {"cpu": _mask_nb}

# %% GPU
if gpu_available and gpu_backend == "numba":
    from numba import cuda

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _mask_nbcuda(sparse_data, dense_data, index):
        # get sizes
        nframes, batch_size, _, _, _ = dense_data.shape
        npts = sparse_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = cuda.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            idz = zindex[frame, point]
            idy = yindex[frame, point]
            idx = xindex[frame, point]
            sparse_data[frame, batch, point] += dense_data[frame, batch, idz, idy, idx]

    _mask["gpu"] = _mask_nbcuda

if gpu_available and gpu_backend == "cupy":
    from cupyx import jit

    @jit.rawkernel()  # pragma: no cover
    def _mask_cupy(sparse_data, dense_data, index):
        # get sizes
        nframes, batch_size, _, _, _ = dense_data.shape
        npts = sparse_data.shape[-1]

        # unpack interpolator
        zindex, yindex, xindex = index

        # parallelize over frames, batches and k-space points
        i = jit.grid(1)  # pylint: disable=too-many-function-args
        if i < nframes * batch_size * npts:
            # get current frame and k-space index
            frame = i // (batch_size * npts)
            tmp = i % (batch_size * npts)
            batch = tmp // npts
            point = tmp % npts

            # gather data within kernel radius
            idz = zindex[frame, point]
            idy = yindex[frame, point]
            idx = xindex[frame, point]
            sparse_data[frame, batch, point] += dense_data[frame, batch, idz, idy, idx]

    _mask["gpu"] = _mask_cupy
