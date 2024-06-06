"""FFT planning routines."""

__all__ = ["plan_fft"]

import numpy as np

from .. import _subroutines

if _subroutines.pytorch_enabled:
    import torch
    USE_TORCH = True
else:
    USE_TORCH = False


def plan_fft(
    mask, shape, zmap=None, L=6, nbins=(40, 40), dt=None, T=None, L_batch_size=None
):
    """
    Precompute sparse FFT object.

    Parameters
    ----------
    mask : ArrayLike
        Binary k-space mask indexes of shape ``(ncontrasts, ny, nx)`` (2D)
        or ``(ncontrasts, nz, ny, nx)`` (3D).
    shape : Sequence[int]
        Grid size of shape ``(ndim,)``.
    zmap : ArrayLike, optional
        Field map in [Hz]; can be real (B0 map) or complex (R2* + 1i * B0).
        The default is ``None``.
    L : int, optional
        Number of zmap segmentations. The default is ``6``.
    nbins : int | Iterable[int], optional
        Granularity of exponential approximation.
        For real zmap, it is a scalar (1D histogram).
        For complex zmap, it must be a tuple of ints (2D histogram).
        The default is ``(40, 40)``.
    dt : float, optional
        Dwell time in ``[s]``. The default is ``None``.
    T : ArrayLike, optional
        Tensor with shape ``(npts,)``, representing the sampling instant of
        each k-space point along the readout. When T is ``None``, this is
        inferred from ``dt`` (if provided), assuming that readout starts
        immediately after excitation (i.e., TE=0). Units are ``[s]``.
        The default is ``None``.
    L_batch_size : int, optional
        Number of zmap segments to be processed in parallel. If ``None``,
        process all segments simultaneously. The default is ``None``.

    Returns
    -------
    mask : Mask
        Structure containing sparse sampling matrix:

        * mask (``ArrayLike``): binary k-space mask of shape ``(ncontrasts, ny, nx)`` (2D) or ``(ncontrasts, nz, ny, nx)`` (3D).
        * shape (``Sequence[int]``): Grid shape of shape (ndim,). Order of axes is (z, y, x).
        * zmap_s_kernel (``ArrayLike``): zmap spatial basis.
        * zmap_t_kernel (``ArrayLike``): zmap temporal basis.
        * zmap_batch_size (``int``): zmap processing batch size.
        * device (``str``): computational device.

    """
    # switch to torch if possible
    if USE_TORCH:
        mask = _subroutines.to_backend(torch, mask)
        if zmap is not None:
            zmap = _subroutines.to_backend(torch, zmap)
        if T is not None:
            T = _subroutines.to_backend(torch, T)
            
    # compute zmap approximation
    if zmap is not None:
        # get time
        if T is None:
            assert dt is not None, "Please provide raster time dt if T is not known"
            T = dt * np.arange(shape[-1], dtype=np.float32)
            
        # compute zmap spatial and temporal basis
        Tshape = T.shape
        zmap_t_kernel, zmap_s_kernel = _subroutines.mri_exp_approx(zmap, T.flatten(), L, nbins)
        zmap_t_kernel = zmap_t_kernel.reshape(*Tshape)
        
        # defaut z batch size
        if L_batch_size is None:
            L_batch_size = L
    else:
        zmap_t_kernel, zmap_s_kernel = None, None

    return _subroutines.Mask(mask, shape, zmap_t_kernel, zmap_s_kernel, L_batch_size)
