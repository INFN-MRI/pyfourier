"""NUFFT planning routines."""

__all__ = ["plan_nufft"]

from dataclasses import dataclass

import math
import numpy as np

from .. import _subroutines


def plan_nufft(
    coord,
    shape,
    width=4,
    oversamp=1.25,
    zmap=None,
    L=6,
    nbins=(40, 40),
    dt=None,
    T=None,
    L_batch_size=None,
):
    """
    Precompute NUFFT object.

    Parameters
    ----------
    coord : ArrayLike
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Sequence[int]
        Oversampled grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    width : int | Sequence[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``3``.
    oversamp : float | Sequence[float], optional
        Grid oversampling factor of shape ``(ndim,)``.
        If scalar, isotropic oversampling is assumed.
        The default is ``1.125``.
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
    nufft_plan : NUFFTPlan
        Structure containing sparse interpolator matrix:

        * ndim (``int``): number of spatial dimensions.
        * oversampling (``Iterable[float]``): grid oversampling factor (z, y, x).
        * width (``Iterable[int]``): kernel width (z, y, x).
        * beta (``Iterable[float]``): Kaiser Bessel parameter (z, y, x).
        * os_shape (``Iterable[int]``): oversampled grid shape (z, y, x).
        * shape (``Iterable[int]``): grid shape (z, y, x).
        * interpolator (``Interpolator``): precomputed interpolator object.
        * zmap_s_kernel (``ArrayLike``): zmap spatial basis.
        * zmap_t_kernel (``ArrayLike``): zmap temporal basis.
        * zmap_batch_size (``int``): zmap processing batch size.
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

    # Cartesian axes have osf = 1.0 and kernel width = 1 (no interpolation)
    oversamp[is_cart] = 1.0
    width[is_cart] = 1

    # get oversampled grid shape
    os_shape = _subroutines._get_oversamp_shape(shape, oversamp, ndim)

    # rescale trajectory
    coord = _subroutines._scale_coord(coord, shape[::-1], oversamp[::-1])

    # compute interpolator
    interpolator = _subroutines.Interpolator(coord, os_shape, width, beta)

    # transform to tuples
    ndim = int(ndim)
    oversamp = tuple(oversamp)
    width = tuple(width)
    beta = tuple(beta)
    os_shape = tuple(os_shape)
    shape = tuple(shape)

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

    return NUFFTPlan(
        ndim,
        oversamp,
        width,
        beta,
        os_shape,
        shape,
        interpolator,
        zmap_t_kernel,
        zmap_s_kernel,
        L_batch_size,
        None,
    )


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
    zmap_t_kernel: object
    zmap_s_kernel: object
    zmap_batch_size: int
    device: int

    def to(self, device):  # noqa
        if self.device is None or device != self.device:
            self.interpolator = self.interpolator.to(device)
            if self.zmap_s_kernel is not None:
                self.zmap_t_kernel = _subroutines.to_device(self.zmap_t_kernel, device)
                self.zmap_s_kernel = _subroutines.to_device(self.zmap_s_kernel, device)
            self.device = device

        return self
