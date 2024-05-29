"""NUFFT planning subroutines."""

__all__ = ["plan_nufft"]

from dataclasses import dataclass

import math
import numpy as np

from .. import _subroutines


def plan_nufft(coord, shape, width=4, oversamp=1.25, device=None):
    """
    Precompute NUFFT object.

    Parameters
    ----------
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Iterable[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    width : int | Iterable[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``3``.
    oversamp : float | Iterable[float], optional
        Grid oversampling factor of shape ``(ndim,)``.
        If scalar, isotropic oversampling is assumed.
        The default is ``1.125``.
    device : int, optional
        Computational device (``-1`` for CPU;  ``n=0, 1,...nGPUs`` for GPU).
        The default is ``None`` (use same device as input when NUFFT is applied).

    Returns
    -------
    interpolator : NUFFTPlan
        Structure containing sparse interpolator matrix:

        * ndim (``int``): number of spatial dimensions.
        * oversampling (``Iterable[float]``): grid oversampling factor (z, y, x).
        * width (``Iterable[int]``): kernel width (z, y, x).
        * beta (``Iterable[float]``): Kaiser Bessel parameter (z, y, x).
        * os_shape (``Iterable[int]``): oversampled grid shape (z, y, x).
        * shape (``Iterable[int]``): grid shape (z, y, x).
        * interpolator (``Interpolator``): precomputed interpolator object.
        * device (``str``): computational device.

    Notes
    -----
    Non-uniform coordinates axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, kernel width
    and oversampling factors are assumed to be ``(y, x)`` and ``(z, y, x)``.

    Coordinates tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

    * ``coord.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
    * ``coord.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """
    # get parameters
    ndim = coord.shape[-1]

    if np.isscalar(width):
        width = np.asarray([width] * ndim, dtype=np.int16)
    else:
        width = np.asarray(width, dtype=np.int16)

    if np.isscalar(oversamp):
        oversamp = np.asarray([oversamp] * ndim, dtype=np.float32)
    else:
        oversamp = np.asarray(oversamp, dtype=np.float32)

    # calculate Kaiser-Bessel beta parameter
    beta = math.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    if np.isscalar(shape):
        shape = np.asarray([shape] * ndim, dtype=np.int16)
    else:
        shape = np.asarray(shape, dtype=np.int16)[-ndim:]

    # normalize coord between [-mtx / 2 to mtx /2] regardless of input
    # normalization
    coord = _subroutines.normalize_coordinates(coord, shape)

    # check for Cartesian axes
    is_cart = [
        np.allclose(coord[..., ax], np.round(coord[..., ax])) for ax in range(ndim)
    ]
    is_cart = np.asarray(is_cart[::-1])  # (z, y, x)

    # determine whether trajectory is a stack of trajectories or not
    # (i.e., z axis is cartesian)
    is_stack = is_cart[0]

    # Cartesian axes have osf = 1.0 and kernel width = 1 (no interpolation)
    oversamp[is_cart] = 1.0
    width[is_cart] = 1

    # get oversampled grid shape
    os_shape = _subroutines._get_oversamp_shape(shape, oversamp, ndim)

    # rescale trajectory
    coord = _subroutines._scale_coord(coord, shape[::-1], oversamp[::-1])

    # compute interpolator
    interpolator = _subroutines.Interpolator(coord, os_shape, is_stack, width, beta)

    # transform to tuples
    ndim = int(ndim)
    oversamp = tuple(oversamp)
    width = tuple(width)
    beta = tuple(beta)
    os_shape = tuple(os_shape)
    shape = tuple(shape)

    return NUFFTPlan(ndim, oversamp, width, beta, os_shape, shape, interpolator, device)


# %% local utils
@dataclass
class NUFFTPlan:
    ndim: int
    oversamp: tuple
    width: tuple
    beta: tuple
    os_shape: tuple
    shape: tuple
    interpolator: object
    device: int

    def to(self, device):
        """
        Dispatch internal attributes to selected device.

        Parameters
        ----------
        device : str
            Computational device ("cpu" or "cuda:n", with n=0, 1,...nGPUs).

        """
        if self.device is None or device != self.device:
            self.interpolator = self.interpolator.to(device)
            self.device = device

        return self
