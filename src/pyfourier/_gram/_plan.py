"""FFT self-adjoint planning routines."""

__all__ = ["plan_gram"]

from dataclasses import dataclass

import numpy as np

from .. import _subroutines

if _subroutines.pytorch_enabled:
    import torch

    USE_TORCH = True
else:
    USE_TORCH = False


def plan_gram(
    mask,
    shape,
    basis=None,
    zmap=None,
    L=6,
    nbins=(40, 40),
    T=None,
    L_batch_size=None,
    oversamp=None,
    width=6,
    device="cpu",
):
    """
    Compute spatio-temporal kernel for fast self-adjoint dense FFT operation.

    Parameters
    ----------
    mask : ArrayLike
        Binary k-space mask indexes of shape ``(ncontrasts, ny, nx)`` (2D)
        or ``(ncontrasts, nz, ny, nx)`` (3D).
    shape : Sequence[int]
        Grid size of shape ``(ndim,)``.
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
    T : ArrayLike, optional
        Tensor with shape ``(npts,)``, representing the sampling instant of
        each k-space point along the readout. When T is ``None``, this is
        inferred from ``dt`` (if provided), assuming that readout starts
        immediately after excitation (i.e., TE=0).
    L_batch_size : int, optional
        Number of zmap segments to be processed in parallel. If ``None``,
        process all segments simultaneously. The default is ``None``.
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
    assert (
        len(shape) == 2
    ), "we support dense Gram FFT for 2D or separable 3D imaging (i.e., FFT along readout) only"

    # if basis is not None, mask should be (nt, nz, ny) or (nt, ny, nx) / (nt, ny, 1)
    if basis is not None:
        if len(mask.shape) == 1:  # (ny,)
            mask = mask[None, :, None]  # (1, ny, 1)
        elif len(mask.shape) == 2:
            mask = mask[None, ...]  # (1, nz, ny) or (1, ny, nx)

    # switch to torch if possible
    if USE_TORCH:
        mask = _subroutines.to_backend(torch, mask)
        if zmap is not None:
            zmap = _subroutines.to_backend(torch, zmap)
        if T is not None:
            T = _subroutines.to_backend(torch, T)

    # get backend
    # backend = _subroutines.get_backend(mask)
    device = _subroutines.get_device(mask)

    # offload to device
    mask = _subroutines.to_device(mask, device)

    # if zmap is provided, offload to device
    if zmap is None:
        zmap = _subroutines.to_device(zmap, device)

    # if spatio-temporal basis is provided, check reality and offload to device
    if basis is not None:
        basis = _subroutines.to_device(basis, device)

    # compute zmap approximation
    if zmap is not None:

        # compute zmap spatial and temporal basis
        Tshape = T.shape
        zmap_t_kernel, zmap_s_kernel = _subroutines.mri_exp_approx(
            zmap, T.flatten(), L, nbins
        )
        zmap_t_kernel = zmap_t_kernel.reshape(*Tshape)

        # defaut z batch size
        if L_batch_size is None:
            L_batch_size = L

    else:
        zmap_t_kernel, zmap_s_kernel = None, None

    # compute
    if basis is not None and mask is not None:
        islowrank = True
        T, K = basis.shape
        nt, ny, nz = mask.shape
        tmp = (
            _subroutines.transpose(mask, [2, 1, 0])[..., None, None] * basis[:, None, :]
        )  # (nz, ny, nt, 1, k) / # (nx, ny, nt, 1, k) / # (1, ny, nt, 1, k)
        st_kernel = tmp * basis.conj()[:, :, None]  # (nz, ny, nt, k, k)
        st_kernel = st_kernel.sum(axis=-3).swapaxes(0, 1)  # (ny, nz, nt, k, k)
        st_kernel = _subroutines.fftshift(
            st_kernel, axes=(0, 1)
        )  # (ny, nz, nt, k, k) / # (ny, nx, nt, 1, k) / # (ny, 1, nt, 1, k)
    else:
        islowrank = False
        st_kernel = mask

    # apply zmap temporal interpolator
    if zmap_s_kernel is not None and st_kernel is not None:
        if basis is None:
            st_kernel = zmap_s_kernel * st_kernel
        else:
            st_kernel = zmap_s_kernel[..., None, None] * st_kernel
    elif zmap_s_kernel is not None:
        st_kernel = zmap_s_kernel

    return GramMatrix(
        st_kernel, tuple(shape), 2, islowrank, zmap_s_kernel, L_batch_size, device
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
