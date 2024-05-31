"""Batched matrix vector product subroutines."""

__all__ = ["_bdot"]

import gc

import numpy as np
import numba as nb

from .. import _utils

# detect GPU
gpu_available, gpu_backend = _utils.detect_gpu_backend()


def _bdot(data_in, gram_matrix, threadsperblock=128):  # noqa
    # collect garbage
    gc.collect()

    # unpack input
    value = gram_matrix.value
    ndim = gram_matrix.ndim
    islowrank = gram_matrix.islowrank
    haszmap = gram_matrix.zmap_s_kernel is not None
    device = gram_matrix.device
    device_tag = _utils.get_device_tag(device)

    # get tensor backend
    backend = _utils.get_backend(data_in)

    # check if the data is complex and harmonize data_in and basis if necessary
    data_in, basis, _, asreal = _utils._is_complex_grid(data_in, value, backend)

    if islowrank is True:
        # keep original shape
        shape = data_in.shape

        # reformat data for computation
        if haszmap:
            data_in = data_in.reshape(
                data_in.shape[0],
                -1,
                data_in.shape[-ndim - 1],
                np.prod(data_in.shape[-ndim:]),
            )  # (nseg, nbatches, ncontrasts, nvoxels)

            data_in = _utils.transpose(
                data_in, (3, 1, 0, 2)
            )  # (nvoxels, nbatches, nseg, ncontrasts)
        else:
            data_in = data_in.reshape(
                -1, data_in.shape[-ndim - 1], np.prod(data_in.shape[-ndim:])
            )  # (nbatches, ncontrasts, nvoxels)

            data_in = _utils.transpose(
                data_in, (2, 0, 1)
            )  # (nvoxels, nbatches, ncontrasts)

        # maks sure it is continous
        data_in = _utils.ascontiguous(data_in)

        # preallocate output data
        data_out = _utils.zeros(
            data_in.shape,
            dtype=data_in.dtype,
            device=device,
            backend=backend,
        )

        # get grid_function
        if haszmap:
            _do_matvec_prod = _matvec_prod[device_tag]
        else:
            _do_matvec_prod = _matvec_prod_zmap[device_tag]

        # switch to numba / cupy
        if device_tag == "cpu":
            data_out, data_in, basis = _utils.to_backend(nb, data_out, data_in, basis)
        else:
            data_out, data_in, basis = _utils.to_backend(
                gpu_backend, data_out, data_in, basis
            )

        # do actual gridding
        if device_tag == "cpu":
            _do_matvec_prod(data_out, data_in, value)
        else:
            blockspergrid = _utils.calc_blocks_per_grid(
                data_in.shape[0], threadsperblock
            )
            _do_matvec_prod[blockspergrid, threadsperblock](data_out, data_in, value)

        # switch to original backend
        data_out = _utils.to_backend(backend, data_out)

        # reformat for output
        if haszmap:
            data_out = _utils.transpose(
                data_out, (2, 1, 3, 0)
            )  # (nseg, nbatches, ncontrasts, nvoxels)

        else:
            data_out = _utils.transpose(
                data_out, (1, 2, 0)
            )  # (nbatches, ncontrasts, nvoxels)

        data_out = _utils.ascontiguous(data_out).reshape(*shape)
    else:
        data_out = value * data_in

    # back to real, if required
    if asreal:
        data_out = _utils.astype(data_out, backend.float32)

    return data_out


# %% subroutines
@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_nb(out, in_a, in_b):
    row, col = in_b.shape

    for i in range(row):
        for j in range(col):
            out[j] += in_b[i][j] * in_a[j]

    return out


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _matvec_prod_nb(data_out, data_in, value):
    # get data dimension
    nvoxels, batch_size, _ = data_in.shape

    for i in nb.prange(nvoxels * batch_size):
        voxel = i // batch_size
        batch = i % batch_size

        _dot_nb(data_out[voxel][batch], data_in[voxel][batch], value[voxel])


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _matvec_prod_zmap_nb(data_out, data_in, value):
    # get data dimension
    nvoxels, batch_size, ncoeff, _ = data_in.shape

    for i in nb.prange(nvoxels * batch_size * ncoeff):
        voxel = i // (batch_size * ncoeff)
        j = i % (batch_size * ncoeff)
        batch = j // ncoeff
        coeff = j % ncoeff

        _dot_nb(
            data_out[voxel][batch][coeff],
            data_in[voxel][batch][coeff],
            value[voxel][coeff],
        )


_matvec_prod = {"cpu": _matvec_prod_nb}
_matvec_prod_zmap = {"cpu": _matvec_prod_zmap_nb}

# %% GPU
if gpu_available and gpu_backend == "numba":
    from numba import cuda

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_prod_nbcuda(out, in_a, in_b):
        row, col = in_b.shape

        for i in range(row):
            for j in range(col):
                out[j] += in_b[i][j] * in_a[j]

        return out

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _matvec_prod_nbcuda(data_out, data_in, value):
        # get data dimension
        nvoxels, batch_size, _ = data_in.shape

        i = cuda.grid(1)
        if i < nvoxels * batch_size:
            voxel = i // batch_size
            batch = i % batch_size

            _dot_prod_nbcuda(
                data_out[voxel][batch], data_in[voxel][batch], value[voxel]
            )

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _matvec_prod_zmap_nbcuda(data_out, data_in, value):
        # get data dimension
        nvoxels, batch_size, ncoeff, _ = data_in.shape

        i = cuda.grid(1)
        if i < nvoxels * batch_size * ncoeff:
            voxel = i // (batch_size * ncoeff)
            j = i % (batch_size * ncoeff)
            batch = j // ncoeff
            coeff = j % ncoeff

            _dot_prod_nbcuda(
                data_out[voxel][batch][coeff],
                data_in[voxel][batch][coeff],
                value[voxel][coeff],
            )

    _matvec_prod["gpu"] = _matvec_prod_nbcuda
    _matvec_prod_zmap["gpu"] = _matvec_prod_zmap_nbcuda

if gpu_available and gpu_backend == "cupy":
    from cupyx import jit

    @jit.rawkernel(device=True)  # pragma: no cover
    def _dot_prod_cupy(out, in_a, in_b):
        row, col = in_b.shape

        for i in range(row):
            for j in range(col):
                out[j] += in_b[i][j] * in_a[j]

        return out

    @jit.rawkernel()  # pragma: no cover
    def _matvec_prod_cupy(data_out, data_in, value):
        # get data dimension
        nvoxels, batch_size, _ = data_in.shape

        i = jit.grid(1)
        if i < nvoxels * batch_size:
            voxel = i // batch_size
            batch = i % batch_size

            _dot_prod_nbcuda(
                data_out[voxel][batch], data_in[voxel][batch], value[voxel]
            )

    @jit.rawkernel()  # pragma: no cover
    def _matvec_prod_zmap_cupy(data_out, data_in, value):
        # get data dimension
        nvoxels, batch_size, ncoeff, _ = data_in.shape

        i = jit.grid(1)
        if i < nvoxels * batch_size * ncoeff:
            voxel = i // (batch_size * ncoeff)
            j = i % (batch_size * ncoeff)
            batch = j // ncoeff
            coeff = j % ncoeff

            _dot_prod_nbcuda(
                data_out[voxel][batch][coeff],
                data_in[voxel][batch][coeff],
                value[voxel][coeff],
            )

    _matvec_prod["gpu"] = _matvec_prod_cupy
    _matvec_prod_zmap["gpu"] = _matvec_prod_zmap_cupy
