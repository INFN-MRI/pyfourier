"""
Apodization subroutines.

Adapted from https://github.com/mikgroup/sigpy/blob/main/sigpy/fourier.py#L312

"""

__all__ = ["_apodize"]

import math
import numpy as np

from . import _utils


def _apodize(data_in, ndim, oversamp, width, beta):  # noqa
    # get tensor backend and device
    backend = _utils.get_backend(data_in)
    device = _utils.get_device(data_in)

    # initialize output
    data_out = data_in

    for n in range(1, ndim + 1):
        axis = -n
        if width[axis] != 1:
            i = data_out.shape[axis]
            os_i = np.ceil(oversamp[axis] * i)
            idx = _utils.arange(i, backend.float32, device, backend)

            # Calculate apodization
            apod = (
                beta[axis] ** 2 - (math.pi * width[axis] * (idx - i // 2) / os_i) ** 2
            ) ** 0.5
            apod /= backend.sinh(apod)

            # normalize by DC
            apod = apod / apod[int(i // 2)]

            # avoid NaN
            apod = backend.nan_to_num(apod, nan=1.0)

            # apply to axis
            data_out *= apod.reshape([i] + [1] * (-axis - 1))

    return data_out
