"""Sampling mask generation."""

__all__ = ["random_mask"]

import numpy as np


def random_mask(ndim, npix, ncontrasts, R=1):
    """
    Initialize 2D / 3D random sampling mask.

    Parameters
    ----------
    ndim : int
        Number of spatial dimension.
    npix : int
        Matrix size.
    ncontrasts : int
        Number of contrast in the image.
    R : int, optional
        Acceleration factor.

    Returns
    -------
    nask : NDArray
        Boolean k-space sampling mask.

    """
    # initialize
    if ndim == 2:
        mask = np.zeros((ncontrasts, 1, npix, 1), dtype=bool)
    else:
        mask = np.zeros((ncontrasts, npix, npix, 1), dtype=bool)

    # flatten
    shape = mask.shape
    mask = mask.reshape(ncontrasts, -1)

    # number of true elements
    nsamples = np.ceil(npix ** (ndim - 1) / R)
    nsamples = int(nsamples)

    # fill mask
    mask[:, :nsamples] = True

    # reshuffle
    tmp = []
    for n in range(mask.shape[0]):
        idx = np.random.permutation(np.arange(mask.shape[-1]))
        tmp.append(mask[n][idx])

    # stack
    mask = np.stack(tmp, axis=0)

    # reshape
    mask = mask.reshape(*shape)

    # add
    # mask = np.repeat(mask, npix, axis=-1)

    return mask
