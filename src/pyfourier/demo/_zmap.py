"""Field map generation."""

__all__ = ["zmap"]

import numpy as np


def zmap(ndim, npix, B0max=500, B0min=-30, mask=None):
    """
    Generate B0 map

    Parameters
    ----------
    ndim : int
        Number of spatial dimension.
    npix : int
        Matrix size.
    B0max : float, optional
        Maximum off resonance frequency (Hz) in the map.
        The default is 250 Hz.
    mask : NDArray, optional
        Object support.
        If provided, rescale zmap so that
        maximum value within mask is B0max,
        otherwise use the whole matrix as a mask.
        The default is None.

    Returns
    -------
    zmap : NDArray
        (Real) field map in [Hz] of shape ``(npix, npix)``
        (2D) or ``(npix, npix, npix)`` (3D).

    """
    # grid boundaries
    start = -npix // 2
    stop = npix // 2

    # generate cartesian grid
    if ndim == 2:
        g = np.mgrid[start:stop, start:stop]
    else:
        g = np.mgrid[start:stop, start:stop, start:stop]

    # radial grid
    r = (g**2).sum(axis=0) ** 0.5
    zmap = r.copy()

    # normalize and rescale
    if mask is None:
        zmax = zmap.max()
    else:
        zmax = zmap[mask].max()

    return B0max * zmap / zmax + B0min
