"""NUFFT or sparse FFT self-adjoint planning routines."""

__all__ = ["plan_spgram"]

from dataclasses import dataclass

import numpy as np

from .. import _subroutines
from .. import _nufft

if _subroutines.pytorch_enabled:
    import torch

    USE_TORCH = True
else:
    USE_TORCH = False


def plan_spgram(
    coord,
    shape,
    basis=None,
    zmap=None,
    L=6,
    nbins=(40, 40),
    dt=None,
    T=None,
    L_batch_size=None,
    weight=None,
    oversamp=None,
    width=6,
    device="cpu",
):
    """
    Compute spatio-temporal kernel for fast self-adjoint Sparse FFT / NUFFT operation.

    Parameters
    ----------
    coord : ArrayLike
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Sequence[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    basis : ArrayLike, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeffs)``; can be ``None``. The default is ``None``.
    zmap : ArrayLike, optional
        Field map in [Hz]; can be real (B0 map) or complex (R2* + 1i * B0).
        The default is ``None``.
    L : int, optional
        Number of zmap segmentations. The default is ``6``.
    nbins : int | Sequence[int], optional
        Granularity of exponential approximation.
        For real zmap, it is a scalar (1D histogram).
        For complex zmap, it must be a tuple of ints (2D histogram).
        The default is ``(40, 40)``.
    dt : float, optional
        Dwell time in ms. The default is ``None``.
    T : ArrayLike, optional
        Tensor with shape ``(npts,)``, representing the sampling instant of
        each k-space point along the readout. When T is ``None``, this is
        inferred from ``dt`` (if provided), assuming that readout starts
        immediately after excitation (i.e., TE=0).
    L_batch_size : int, optional
        Number of zmap segments to be processed in parallel. If ``None``,
        process all segments simultaneously. The default is ``None``.
    weight: ArrayLike, optional
        Tensor to be used as a weight for the output k-space data (e.g., dcf**0.5).
        Must be broadcastable with ``kspace`` (i.e., the output).
        The default is a tensor of ``1.0``.
    width : int | Sequence[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``4``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.

    Returns
    -------
    gram_matrix : GramMatrix
        Structure containing Toeplitz kernel (i.e., Fourier transform of system tPSF).

    """
    # switch to torch if possible
    if USE_TORCH:
        coord = _subroutines.to_backend(torch, coord)
        if basis is not None:
            basis = _subroutines.to_backend(torch, basis)
        if zmap is not None:
            zmap = _subroutines.to_backend(torch, zmap)
        if T is not None:
            T = _subroutines.to_backend(torch, T)
        if weight is not None:
            weight = _subroutines.to_backend(torch, weight)

    # detect backend and device
    backend = _subroutines.get_backend(coord)

    # if not provided, use original device
    if isinstance(device, str):
        if device == "cpu":
            device = -1
        else:
            device = int(device.split(":")[-1])

    # expand singleton dimensions
    ndim = coord.shape[-1]

    # kernel oversampling
    oversamp = np.asarray([2.0] * ndim)

    # shape
    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.asarray(shape, dtype=np.int16)[-ndim:]

    # offload to device
    coord = _subroutines.to_device(coord, device)

    # if weight are not provided, assume uniform sampling density
    if weight is None:
        weight = _subroutines.ones(coord.shape[:-1], backend.float32, device, backend)
    else:
        weight = _subroutines.to_device(weight, device)

    # if zmap is provided, offload to device
    if zmap is None:
        zmap = _subroutines.to_device(zmap, device)

    # if spatio-temporal basis is provided, check reality and offload to device
    if basis is not None:
        islowrank = True
        # isreal = _subroutines.isreal(basis)
        _, ncoeff = basis.shape
        basis = _subroutines.to_device(basis, device)
    else:
        islowrank = False
        isreal = False
        ncoeff = coord.shape[0]

    if isreal:
        dtype = backend.float32
    else:
        dtype = backend.complex64

    # create input data for PSF estimation
    if basis is not None:
        # initialize temporary arrays
        delta = _subroutines.ones(
            list(coord.shape[:-1])[::-1] + [ncoeff], dtype, device, backend
        )
        delta = delta * basis
        delta = _subroutines.transpose(delta, list(np.arange(delta.ndim - 1, -1, -1)))
        delta = _subroutines.ascontiguous(delta)
    else:
        # initialize temporary arrays
        delta = _subroutines.ones(list(coord.shape[:-1]), dtype, device, backend)

    # compute zmap approximation
    if zmap is not None:
        # get time
        if T is None:
            assert dt is not None, "Please provide raster time dt if T is not known"
            T = dt * np.arange(coord.shape[-2], dtype=np.float32)

        # compute zmap spatial and temporal basis
        zmap_t_kernel, zmap_s_kernel = _subroutines.mri_exp_approx(zmap, T, L, nbins)

        # defaut z batch size
        if L_batch_size is None:
            L_batch_size = L
    else:
        zmap_t_kernel, zmap_s_kernel = None, None

    # modify delta to account for zmap
    if zmap_t_kernel is not None:
        zmap_t_kernel = zmap_t_kernel.T  # (npts, L)
        delta = delta[..., None] * zmap_t_kernel
        delta = delta[None, ...].swapaxes(0, -1)[..., 0]

    # check for Cartesian axes
    is_cart = [
        np.allclose(shape[ax] * coord[..., ax], np.round(shape[ax] * coord[..., ax]))
        for ax in range(ndim)
    ]
    is_cart = np.asarray(is_cart[::-1])  # (z, y, x)

    # Cartesian axes have osf = 1.0 and kernel width = 1 (no interpolation)
    oversamp[is_cart] = 1.0

    # get oversampled grid shape
    shape = _subroutines._get_oversamp_shape(shape, oversamp, ndim)

    # switch oversamp backend
    oversamp = _subroutines.to_device(oversamp, device, backend)

    # calculate PSF
    st_kernel = _nufft.nufft_adj(
        weight * delta,
        coord * oversamp,
        shape,
        basis,
        device,
        width=width,
    )

    # FFT
    st_kernel = _subroutines.fft(st_kernel, axes=range(-ndim, 0))

    # squeeze
    if zmap is not None:
        if st_kernel.shape[1] == 1:
            st_kernel = st_kernel[:, 0]
    else:
        if st_kernel.shape[0] == 1:
            st_kernel = st_kernel[0]

    # keep only real part if basis is real
    # if isreal:
    #     st_kernel = st_kernel.real

    # fftshift kernel to accelerate computation
    st_kernel = _subroutines.ifftshift(st_kernel, axes=list(range(-ndim, 0)))

    if basis is not None:
        st_kernel = st_kernel.reshape(
            *st_kernel.shape[:ndim], np.prod(st_kernel.shape[-ndim:])
        )
        if zmap is not None:
            st_kernel = _subroutines.ascontiguous(
                _subroutines.transpose(st_kernel, (2, 1, 0))
            )
        else:
            st_kernel = _subroutines.ascontiguous(
                _subroutines.transpose(st_kernel, (0, 3, 2, 1))
            )

    # normalize
    st_kernel /= backend.mean(abs(st_kernel[st_kernel != 0]))

    # remove NaN
    st_kernel = backend.nan_to_num(st_kernel)

    return GramMatrix(
        st_kernel, tuple(shape), ndim, islowrank, zmap_s_kernel, L_batch_size, device
    )


# %% local utils
@dataclass
class GramMatrix:
    value: object
    shape: tuple
    ndim: int
    islowrank: bool
    zmap_s_kernel: object
    zmap_batch_size: int
    device: str

    def to(self, device):  # noqa
        if self.device is None and device != self.device:
            self.value = _subroutines.to_device(self.value, device)
            if self.zmap_s_kernel is not None:
                self.zmap_s_kernel = _subroutines.to_device(self.zmap_s_kernel, device)
            self.device = device

        return self
