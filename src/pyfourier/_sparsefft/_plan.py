"""Sparse FFT planning subroutines."""

__all__ = ["plan_spfft"]

import numpy as np

from .. import _subroutines


def plan_spfft(
    indexes, shape, zmap=None, L=6, nbins=(40, 40), dt=None, T=None, L_batch_size=None
):
    """
    Precompute sparse FFT object.

    Parameters
    ----------
    indexes : torch.Tensor
        Sampled k-space points indexes of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Iterable[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    device : int, optional
        Computational device (``-1`` for CPU;  ``n=0, 1,...nGPUs`` for GPU).
        The default is ``None`` (use same device as input when NUFFT is applied).
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
    mask : dict
        Structure containing sparse sampling matrix:

        * indexes (``torch.Tensor[int]``): indexes of the non-zero entries of interpolator sparse matrix of shape (ndim, ncoord).
        * dshape (``Iterable[int]``): oversample grid shape of shape (ndim,). Order of axes is (z, y, x).
        * ishape (``Iterable[int]``): interpolator shape (ncontrasts, nview, nsamples)
        * ndim (``int``): number of spatial dimensions.
        * zmap_s_kernel (``ArrayLike``): zmap spatial basis.
        * zmap_t_kernel (``ArrayLike``): zmap temporal basis.
        * zmap_batch_size (``int``): zmap processing batch size.
        * device (``str``): computational device.

    Notes
    -----
    Sampled point indexes axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape is
    assumed to be ``(z, y, x)``.

    Indexes tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

        * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
        * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # get backend
    backend = _subroutines.get_backend(indexes)

    # get parameters
    ndim = indexes.shape[-1]

    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.array(shape, dtype=np.int16)

    # normalize coord between [-mtx / 2 to mtx /2] regardless of input
    # normalization
    indexes = _subroutines.normalize_coordinates(indexes, shape, True)

    # check for Cartesian axes
    is_cart = [
        np.allclose(indexes[..., ax], np.round(indexes[..., ax])) for ax in range(ndim)
    ]
    is_cart = np.asarray(is_cart[::-1])  # (z, y, x)

    # assert all axes are Cartesian
    assert (
        is_cart.all()
    ), "Input coordinates must lie on Cartesian grid, got non-uniform coord! Please use NUFFT instead."
    indexes = _subroutines.astype(indexes, backend.int16)

    # get t
    if zmap is not None:
        # get time
        if T is None:
            assert dt is not None, "Please provide raster time dt if T is not known"
            T = dt * np.arange(indexes.shape[-2], dtype=np.float32)

        # compute zmap spatial and temporal basis
        zmap_t_kernel, zmap_s_kernel = _subroutines.mri_exp_approx(zmap, T, L, nbins)

        # defaut z batch size
        if L_batch_size is None:
            L_batch_size = L
    else:
        zmap_t_kernel, zmap_s_kernel = None, None

    return _subroutines.Mask(indexes, shape, zmap_t_kernel, zmap_s_kernel, L_batch_size)
