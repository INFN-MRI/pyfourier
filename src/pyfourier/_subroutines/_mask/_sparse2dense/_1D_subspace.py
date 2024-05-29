"""1D zero-filling subroutines with embedded subspace projection."""

__all__ = ["_zerofill"]

import numba as nb

from ... import _utils

# detect GPU
gpu_available, gpu_backend = _utils.detect_gpu_backend()


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill_nb(dense_data, sparse_data, index, basis):  # noqa
    # get sizes
    ncoeff, batch_size, _ = dense_data.shape
    nframes = sparse_data.shape[0]
    npts = sparse_data.shape[-1]

    # unpack interpolator
    xindex = index[0]

    # parallelize over low-rank coefficients and batches
    for i in nb.prange(ncoeff * batch_size):  # pylint: disable=not-an-iterable
        # get current low-rank coefficient and batch index
        coeff = i // batch_size
        batch = i % batch_size

        # iterate over frames in current coefficient/batch
        for frame in range(nframes):
            # iterate over non-cartesian point of current frame
            for point in range(npts):
                idx = xindex[frame, point]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while spreading data
                dense_data[coeff, batch, idx] += (
                    basis[coeff, frame] * sparse_data[frame, batch, point]
                )


_zerofill = {"cpu": {False: _zerofill_nb, True: _zerofill_nb}}

# %% GPU
if gpu_available and gpu_backend == "numba":
    from numba import cuda

    def _get_zerofill_nbcuda(is_complex):
        _update = _utils._update[is_complex]

        @cuda.jit(fastmath=True)  # pragma: no cover
        def _zerofill_nbcuda(dense_data, sparse_data, index, basis):
            # get sizes
            ncoeff, batch_size, _ = dense_data.shape
            nframes = sparse_data.shape[0]
            npts = sparse_data.shape[-1]

            # unpack interpolator
            xindex = index[0]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes * batch_size * npts:
                # get current frame and k-space index
                frame = i // (batch_size * npts)
                tmp = i % (batch_size * npts)
                batch = tmp // npts
                point = tmp % npts

                idx = xindex[frame, point]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while spreading data
                for coeff in range(ncoeff):
                    _update(
                        dense_data,
                        (coeff, batch, idx),
                        basis[coeff, frame] * sparse_data[frame, batch, point],
                    )

        return _zerofill_nbcuda

    _zerofill["gpu"] = {
        False: _get_zerofill_nbcuda(False),
        True: _get_zerofill_nbcuda(True),
    }

if gpu_available and gpu_backend == "cupy":
    from cupyx import jit

    def _get_zerofill_cupy(is_complex):
        _update = _utils._update[is_complex]

        @jit.rawkernel()  # pragma: no cover
        def _zerofill_cupy(dense_data, sparse_data, index, basis):
            # get sizes
            ncoeff, batch_size, _ = dense_data.shape
            nframes = sparse_data.shape[0]
            npts = sparse_data.shape[-1]

            # unpack interpolator
            xindex = index[0]

            # parallelize over frames, batches and k-space points
            i = jit.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes * batch_size * npts:
                # get current frame and k-space index
                frame = i // (batch_size * npts)
                tmp = i % (batch_size * npts)
                batch = tmp // npts
                point = tmp % npts

                idx = xindex[frame, point]

                # do adjoint low rank projection (low-rank subspace -> time domain)
                # while spreading data
                for coeff in range(ncoeff):
                    _update(
                        dense_data,
                        (coeff, batch, idx),
                        basis[coeff, frame] * sparse_data[frame, batch, point],
                    )

        return _zerofill_cupy

    _zerofill["gpu"] = {
        False: _get_zerofill_cupy(False),
        True: _get_zerofill_cupy(True),
    }
