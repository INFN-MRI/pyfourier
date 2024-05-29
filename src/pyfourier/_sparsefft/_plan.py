"""Sparse FFT planning subroutines."""

__all__ = ["plan_spfft"]

import numpy as np

from .. import _subroutines


def plan_spfft(indexes, shape, device=None):
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

    Returns
    -------
    interpolator : dict
        Structure containing sparse interpolator matrix:

            * index (``torch.Tensor[int]``): indexes of the non-zero entries of interpolator sparse matrix of shape (ndim, ncoord).
            * dshape (``Iterable[int]``): oversample grid shape of shape (ndim,). Order of axes is (z, y, x).
            * ishape (``Iterable[int]``): interpolator shape (ncontrasts, nview, nsamples)
            * ndim (``int``): number of spatial dimensions.
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

    return _subroutines.Mask(indexes, shape, device)
