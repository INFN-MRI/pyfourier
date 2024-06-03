"""
Python implementation of the GRAPPA operator formalism.

Adapted for convenience from PyGRAPPA.

"""

__all__ = ["grog_interp"]

import gc
from types import SimpleNamespace

import numpy as np
import numba as nb

from .. import _subroutines

# detect GPU
gpu_available, gpu_backend = _subroutines.detect_gpu_backend()


def grog_interp(
    input,
    calib,
    coord,
    shape,
    lamda=0.01,
    nsteps=11,
    device=None,
    threadsperblock=128,
):
    """
    GRAPPA Operator Gridding (GROG) interpolation of Non-Cartesian datasets.

    Parameters
    ----------
    input : ArrayLike
        Input Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    calib : ArrayLike
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    coord : ArrayLike
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape, 0.5 * shape)``.
    shape : Sequence[int]
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01``.
    nsteps : int, optional
        K-space interpolation grid discretization. Defaults to ``11``
        steps (i.e., ``dk = -0.5, -0.4, ..., 0.0, ..., 0.4, 0.5``)
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    output : ArrayLike
        Output sparse Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.
    indexes : ArrayLike
        Sampled k-space points indexes of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    weights : ArrayLike
        Number of occurrences of each k-space sample of shape ``(ncontrasts, nviews, nsamples)``.

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.

    """
    # collect garbage
    gc.collect()

    # detect backend and device
    backend = _subroutines.get_backend(input)
    idevice = _subroutines.get_device(input)

    # if not provided, use original device
    if device is None:
        device = idevice
    else:
        if isinstance(device, str):
            if device == "cpu":
                device = -1
            else:
                device = int(device.split(":")[-1])

    # cast to device if necessary
    input = _subroutines.to_device(input, device)
    calib = _subroutines.to_device(calib, device)
    coord = _subroutines.to_device(coord, device)

    # rescale calib
    calib = calib / backend.linalg.norm(calib)

    # default to odds steps to explicitly have 0
    nsteps = 2 * (nsteps // 2) + 1

    # get number of spatial dimes
    ndim = coord.shape[-1]

    # get grappa operator
    kern = _calc_grappaop(calib, ndim, lamda, device)

    # get coord shape
    cshape = coord.shape

    # reshape coordinates
    ishape = input.shape
    input = input.reshape(*ishape[: -(len(cshape) - 1)], int(np.prod(cshape[:-1])))

    # bring coil axes to the front
    input = input[..., None]
    input = input.swapaxes(-3, -1)
    dshape = input.shape
    input = input.reshape(
        -1, *input.shape[-2:]
    )  # (nslices, nsamples, ncoils) or (nsamples, ncoils)

    # perform product
    deltas = (backend.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)

    # get Gx, Gy, Gz
    Gx = _weight_grid(
        kern.Gx, deltas
    )  # 2D: (nsteps, nslices, nc, nc); 3D: (nsteps, nc, nc)
    Gy = _weight_grid(
        kern.Gy, deltas
    )  # 2D: (nsteps, nslices, nc, nc); 3D: (nsteps, nc, nc)

    if ndim == 3:
        Gz = _weight_grid(kern.Gz, deltas)  # (nsteps, nc, nc), 3D only
    else:
        Gz = None

    # build G
    if ndim == 2:
        Gx = Gx[None, ...]
        Gy = Gy[:, None, ...]
        Gx = np.repeat(Gx, nsteps, axis=0)  # (nsteps, nsteps, nslices, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=1)  # (nsteps, nsteps, nslices, nc, nc)
        Gx = Gx.reshape(-1, *Gx.shape[-3:])  # (nsteps**2, nslices, nc, nc)
        Gy = Gy.reshape(-1, *Gy.shape[-3:])  # (nsteps**2, nslices, nc, nc)
        G = Gx @ Gy  # (nsteps**2, nslices, nc, nc)
    elif ndim == 3:
        Gx = Gx[None, None, ...]
        Gy = Gy[None, :, None, ...]
        Gz = Gz[:, None, None, ...]
        Gx = np.repeat(Gx, nsteps, axis=0)  # (nsteps, nsteps, nsteps, nc, nc)
        Gx = np.repeat(Gx, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=0)  # (nsteps, nsteps, nsteps, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Gz = np.repeat(Gz, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Gz = np.repeat(Gz, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Gx = Gx.reshape(-1, *Gx.shape[-2:])  # (nsteps**3, nc, nc)
        Gy = Gy.reshape(-1, *Gy.shape[-2:])  # (nsteps**3, nc, nc)
        Gz = Gz.reshape(-1, *Gz.shape[-2:])  # (nsteps**3, nc, nc)
        G = Gx @ Gy @ Gz  # (nsteps**3, nc, nc)

    # build indexes
    indexes = backend.round(coord)
    lut = indexes - coord
    lut = backend.floor(10 * lut).to(int) + int(nsteps // 2)
    lut = lut.reshape(-1, ndim)  # (nsamples, ndim)
    if backend.__name__ == "torch":
        lut = lut * backend.as_tensor([1, nsteps, nsteps**2])[:ndim]
    else:
        lut = lut * backend.asarray([1, nsteps, nsteps**2])[:ndim]
    lut = lut.sum(axis=-1)

    if ndim == 2:
        input = input.swapaxes(0, 1)  # (nsamples, nslices, ncoils)

    # perform interpolation
    output = do_interpolation(input, G, lut, threadsperblock)

    # finalize indexes
    if np.isscalar(shape):
        shape = [shape] * ndim
    if backend.__name__ == "torch":
        indexes = indexes + backend.as_tensor(list(shape[-ndim:])[::-1]) // 2
        indexes = indexes.to(int)
    else:
        indexes = indexes + backend.asarray(list(shape[-ndim:])[::-1]) // 2
        indexes = indexes.astype(int)

    # flatten indexes
    unfolding = [1] + list(np.cumprod(list(shape)[::-1]))[: ndim - 1]
    if backend.__name__ == "torch":
        flattened_idx = backend.as_tensor(unfolding, dtype=int) * indexes
    else:
        flattened_idx = backend.asarray(unfolding, dtype=int) * indexes
    flattened_idx = flattened_idx.sum(axis=-1).flatten()

    # count elements
    _, idx, counts = backend.unique(
        flattened_idx, return_inverse=True, return_counts=True
    )
    weights = counts[idx]

    # count
    weights = weights.reshape(*indexes.shape[:-1])
    if backend.__name__ == "torch":
        weights = weights.to(backend.float32)
    else:
        weights = weights.astype(backend.float32)
    weights = 1.0 / weights

    # finalize
    if ndim == 2:
        output = output.swapaxes(0, 1)  # (nslices, nsamples, ncoils)
    output = output.reshape(*dshape)
    output = output.swapaxes(-3, -1)
    output = output[..., 0]
    output = output.reshape(ishape)

    # remove out-of-boundaries
    shape = list(shape[-ndim:])[::-1]  # (x, y, z)
    for n in range(ndim):
        outside = indexes[..., n] < 0
        output[..., outside] = 0.0
        indexes[..., n][outside] = 0
        outside = indexes[..., n] >= shape[n]
        indexes[..., n][outside] = shape[n] - 1
        output[..., outside] = 0.0

    # cast back to original device
    output = _subroutines.to_device(output, device)
    indexes = _subroutines.to_device(indexes, device)
    weights = _subroutines.to_device(weights, device)

    # collect garbage
    gc.collect()

    return output, indexes, weights


# %% subroutines
def _calc_grappaop(calib, ndim, lamda, device):
    # expand
    if len(calib.shape) == 3:  # single slice (nc, ny, nx)
        calib = calib[:, None, :, :]

    # compute kernels
    if ndim == 2:
        gy, gx = _grappa_op_2d(calib, lamda)
    elif ndim == 3:
        gz, gy, gx = _grappa_op_3d(calib, lamda)

    # prepare output
    GrappaOp = SimpleNamespace(Gx=gx, Gy=gy)

    if ndim == 3:
        GrappaOp.Gz = gz
    else:
        GrappaOp.Gz = None

    return GrappaOp


def _grappa_op_2d(calib, lamda):
    """Return a batch of 2D GROG operators (one for each z)."""
    # infer backend
    backend = _subroutines.get_backend(input)

    # coil axis in the back
    calib = backend.moveaxis(calib, 0, -1)
    nz, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sy = backend.reshape(calib[:, :-1, :, :], (nz, -1, nc))
    Sx = backend.reshape(calib[:, :, :-1, :], (nz, -1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Ty = backend.reshape(calib[:, 1:, :, :], (nz, -1, nc))
    Tx = backend.reshape(calib[:, :, 1:, :], (nz, -1, nc))

    # train the operators:
    Syh = Sy.conj().swapaxes(2, 1)
    lamda0 = lamda * backend.linalg.norm(Syh, dim=(1, 2)) / Syh.shape[1]
    Gy = backend.linalg.solve(
        _bdot(Syh, Sy) + lamda0[:, None, None] * backend.eye(Syh.shape[1])[None, ...],
        _bdot(Syh, Ty),
    )

    Sxh = Sx.conj().swapaxes(2, 1)
    lamda0 = lamda * backend.linalg.norm(Sxh, dim=(1, 2)) / Sxh.shape[1]
    Gx = backend.linalg.solve(
        _bdot(Sxh, Sx) + lamda0[:, None, None] * backend.eye(Sxh.shape[1])[None, ...],
        _bdot(Sxh, Tx),
    )

    return Gy, Gx


def _grappa_op_3d(calib, lamda):
    """Return 3D GROG operator."""
    # infer backend
    backend = _subroutines.get_backend(input)

    # coil axis in the back
    calib = backend.moveaxis(calib, 0, -1)
    _, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sz = backend.reshape(calib[:-1, :, :, :], (-1, nc))
    Sy = backend.reshape(calib[:, :-1, :, :], (-1, nc))
    Sx = backend.reshape(calib[:, :, :-1, :], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tz = backend.reshape(calib[1:, :, :, :], (-1, nc))
    Ty = backend.reshape(calib[:, 1:, :, :], (-1, nc))
    Tx = backend.reshape(calib[:, :, 1:, :], (-1, nc))

    # train the operators:
    Szh = Sz.conj().swapaxes(1, 0)
    lamda0 = lamda * backend.linalg.norm(Szh) / Szh.shape[0]
    Gz = backend.linalg.solve(Szh @ Sz + lamda0 * backend.eye(Szh.shape[0]), Szh @ Tz)

    Syh = Sy.conj().swapaxes(1, 0)
    lamda0 = lamda * backend.linalg.norm(Syh) / Syh.shape[0]
    Gy = backend.linalg.solve(Syh @ Sy + lamda0 * backend.eye(Syh.shape[0]), Syh @ Ty)

    Sxh = Sx.conj().swapaxes(1, 0)
    lamda0 = lamda * backend.linalg.norm(Sxh) / Sxh.shape[0]
    Gx = backend.linalg.solve(Sxh @ Sx + lamda0 * backend.eye(Sxh.shape[0]), Sxh @ Tx)

    return Gz, Gy, Gx


def _bdot(a, b):
    # infer backend
    backend = _subroutines.get_backend(input)
    return backend.einsum("...ij,...jk->...ik", a, b)


def _weight_grid(A, weight):
    # infer backend
    backend = _subroutines.get_backend(input)

    # decompose
    L, V = backend.linalg.eig(A)

    # raise to power along expanded first dim
    if len(L.shape) == 2:  # 3D case, (nc, nc)
        L = L[None, ...] ** weight[:, None, None]
    else:  # 2D case, (nslices, nc, nc)
        L = L[None, ...] ** weight[:, None, None, None]

    # unsqueeze batch dimension for V
    V = V[None, ...]

    # create batched diagonal
    if backend.__name__ == "torch":
        Ld = backend.diag_embed(L)
    else:
        Ld = np.expand_dims(L, axis=1) @ np.eye(L.shape[-1])

    # put together and return
    return V @ Ld @ backend.linalg.inv(V)


def do_interpolation(noncart, G, lut, threadsperblock):
    # infer backend and devic tag
    backend = _subroutines.get_backend(noncart)
    device = _subroutines.get_device(noncart)
    device_tag = _subroutines.get_device_tag(device)

    # switch to numba / cupy
    if device_tag == "cpu":
        G = _subroutines.to_backend(nb, G)
        lut = _subroutines.to_backend(nb, lut)
        noncart = _subroutines.to_backend(nb, noncart)
    else:
        G = _subroutines.to_backend(gpu_backend, G)
        lut = _subroutines.to_backend(gpu_backend, lut)
        noncart = _subroutines.to_backend(gpu_backend, noncart)

    # initialize data
    cart = backend.zeros(noncart.shape, noncart.dtype, noncart.device, backend)

    # actual interpolation
    if device_tag == "cpu":
        _interp(cart, noncart, G, lut)
    else:
        nsamples, batch_size, ncoils = noncart.shape
        blockspergrid = _subroutines.calc_blocks_per_grid(
            nsamples * batch_size, threadsperblock
        )
        _interp_cuda[blockspergrid, threadsperblock](cart, noncart, G, lut)

    # to original backend
    cart = _subroutines.to_backend(backend, cart)

    return cart


# %% NUMBA
@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_product(out, in_a, in_b):
    row, col = in_b.shape

    for i in range(row):
        for j in range(col):
            out[j] += in_b[i][j] * in_a[j]

    return out


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp(data_out, data_in, interp, lut):  # noqa
    # get data dimension
    nsamples, batch_size, _ = data_in.shape

    for i in nb.prange(nsamples * batch_size):
        sample = i // batch_size
        batch = i % batch_size
        idx = lut[sample]

        _dot_product(
            data_out[sample][batch], data_in[sample][batch], interp[idx][batch]
        )


# %% CUDA
if gpu_available and gpu_backend == "numba":
    from numba import cuda

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_product_nbcuda(out, in_a, in_b):
        row, col = in_b.shape

        for i in range(row):
            for j in range(col):
                out[j] += in_b[i][j] * in_a[j]

        return out

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interp_nbcuda(data_out, data_in, interp, lut):
        # get data dimension
        nvoxels, batch_size, _ = data_in.shape

        i = cuda.grid(1)
        if i < nvoxels * batch_size:
            sample = i // batch_size
            batch = i % batch_size
            idx = lut[sample]

            _dot_product_nbcuda(
                data_out[sample][batch], data_in[sample][batch], interp[idx][batch]
            )

    _interp_cuda = _interp_nbcuda

if gpu_available and gpu_backend == "cupy":
    from cupyx import jit

    @jit.rawkernel(device=True)  # pragma: no cover
    def _dot_product_cupy(out, in_a, in_b):
        row, col = in_b.shape

        for i in range(row):
            for j in range(col):
                out[j] += in_b[i][j] * in_a[j]

        return out

    @jit.rawkernel()  # pragma: no cover
    def _interp_cupy(data_out, data_in, interp, lut):
        # get data dimension
        nvoxels, batch_size, _ = data_in.shape

        i = jit.grid(1)
        if i < nvoxels * batch_size:
            sample = i // batch_size
            batch = i % batch_size
            idx = lut[sample]

            _dot_product_cupy(
                data_out[sample][batch], data_in[sample][batch], interp[idx][batch]
            )

    _interp_cuda = _interp_cupy
