"""Trajectory normalization and scaling subroutines."""

__all__ = ["normalize_coordinates", "_scale_coord"]

import numpy as np

from . import _utils


def normalize_coordinates(input, shape, center=False):
    """
    Normalize input coordinates to [-0.5 * shape[i], 0.5 * shape[i]], with i = x, y (, z).

    Input can have any normalization (e.g., 1 / m, rad / m, -0.5:0.5, -pi:pi).

    Parameters
    ----------
    input : npt.ArrayLike
        Input coordinates with arbitrary units.
    shape : npt.ArrayLike
        Matrix shape of (x, y) or (x, y, z).
    center : bool, optional
        If True, shift coordinates to have a final normalization between
        [0, shape[i]] with i = x, y (, z). The default is False.

    Returns
    -------
    coord : npt.ArrayLike
        Output normalized coordinates.

    """
    backend = _utils.get_backend(input)
    device = _utils.get_device(input)

    # normalize
    cabs = (input**2).sum(axis=-1) ** 0.5  #
    coord = 0.5 * input / cabs[..., None]  # coord between -0.5 and 0.5
    shape = _utils.to_device(shape, device, backend)

    if center:
        return shape * coord + (shape // 2)
    return shape * coord


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.clone()
    for i in range(-ndim, 0):
        scale = np.ceil(oversamp[i] * shape[i]) / shape[i]
        shift = np.ceil(oversamp[i] * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output
