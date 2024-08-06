"""Interpolator planning subroutines."""

__all__ = ["Interpolator"]

import numpy as np
import numba as nb

from .. import _utils

# detect GPU
_, gpu_backend = _utils.detect_gpu_backend()


class Interpolator:  # noqa
    def __init__(self, coord, shape, width, beta):

        # expand singleton dimensions
        ishape = list(coord.shape[:-1])
        ndim = coord.shape[-1]

        while len(ishape) < 3:
            ishape = [1] + ishape

        nframes = ishape[0]
        ishape = ishape[1:]

        # parse input sizes
        npts = np.prod(ishape)

        # expand
        if np.isscalar(shape):
            shape = ndim * [shape]
        else:
            shape = shape[-ndim:]
        if np.isscalar(width):
            width = ndim * [width]
        if np.isscalar(beta):
            beta = ndim * [beta]

        # revert axis (z, y, x) -> (x, y, z)
        shape = shape[::-1]
        width = width[::-1]
        beta = beta[::-1]

        # compute kernel scaling
        scale = _get_kernel_scaling(beta, width)

        # arg reshape
        coord = coord.reshape([nframes * npts, ndim]).T

        # preallocate interpolator
        index = []
        value = []

        for i in range(ndim):  # (x, y, z)
            # kernel value
            value.append(np.zeros((nframes * npts, width[i]), dtype=np.float32))

            # kernel index
            index.append(np.zeros((nframes * npts, width[i]), dtype=np.int32))

        # actual precomputation
        for i in range(ndim):  # (x, y, z)
            _prepare_interpolator(
                value[i], index[i], coord[i], width[i], beta[i], shape[i]
            )

        # fix cartesian axes
        for i in range(ndim):  # (x, y, z)
            if width[i] == 1:
                index[i] = coord[i][..., None].to(np.int32)
                value[i] = 0 * value[i] + 1.0

        # reformat for output
        for i in range(ndim):
            index[i] = index[i].reshape([nframes, npts, width[i]])
            value[i] = value[i].reshape([nframes, npts, width[i]])

        # revert axis (x, y, z) > (z, y, x)
        index = index[::-1]
        value = value[::-1]
        shape = shape[::-1]

        # transform to tuples
        self.index = tuple(index)
        self.value = tuple(value)
        self.dshape = tuple(ishape)
        self.ishape = tuple(shape)
        self.scale = scale
        self.ndim = ndim
        self.device = None

    def to(self, device):  # noqa
        if self.device is None or device != self.device:

            # get device tag
            device_tag = _utils.get_device_tag(device)
            if device_tag == "cpu":
                backend = nb
            else:
                backend = gpu_backend

            self.index = list(self.index)
            self.value = list(self.value)

            # switch to numba / cupy
            self.index = [_utils.to_device(idx, device, backend) for idx in self.index]
            self.value = [_utils.to_device(val, device, backend) for val in self.value]

            self.index = tuple(self.index)
            self.value = tuple(self.value)
            self.device = device

        return self


# %% subroutines
def _get_kernel_scaling(beta, width):
    # init kernel centered on k-space node
    value = []

    # fill the three axes
    for ax in range(len(width)):
        start = np.ceil(-width[ax] / 2)
        value.append(
            np.array(
                [
                    _kaiser_bessel_kernel((start + el) / (width[ax] / 2), beta[ax])
                    for el in range(width[ax])
                ]
            )
        )

    # fix cartesian axes
    for ax in range(len(width)):
        if width[ax] == 1:
            value[ax] = np.array([1.0])

    value = np.stack(np.meshgrid(*value), axis=0).prod(axis=0)

    return value.sum()


def _prepare_interpolator(
    interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape
):  # noqa
    """Preparation routine wrapper."""
    interp_value = _utils.to_backend(nb, interp_value)
    interp_index = _utils.to_backend(nb, interp_index)
    coord = _utils.to_backend(nb, coord.copy())

    _prepare_interpolator_nb(
        interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape
    )


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _kaiser_bessel_kernel(x, beta):
    if abs(x) > 1:
        return 0

    x = beta * (1 - x**2) ** 0.5
    t = x / 3.75
    if x < 3.75:
        return (
            1
            + 3.5156229 * t**2
            + 3.0899424 * t**4
            + 1.2067492 * t**6
            + 0.2659732 * t**8
            + 0.0360768 * t**10
            + 0.0045813 * t**12
        )
    else:
        return (
            x**-0.5
            * np.exp(x)
            * (
                0.39894228
                + 0.01328592 * t**-1
                + 0.00225319 * t**-2
                - 0.00157565 * t**-3
                + 0.00916281 * t**-4
                - 0.02057706 * t**-5
                + 0.02635537 * t**-6
                - 0.01647633 * t**-7
                + 0.00392377 * t**-8
            )
        )


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _prepare_interpolator_nb(
    interp_value, interp_index, coord, kernel_width, kernel_param, grid_shape
):
    # get sizes
    npts = coord.shape[0]
    kernel_width = interp_index.shape[-1]

    for i in nb.prange(npts):  # pylint: disable=not-an-iterable
        x_0 = np.ceil(coord[i] - kernel_width / 2)

        for x_i in range(kernel_width):
            val = _kaiser_bessel_kernel(
                ((x_0 + x_i) - coord[i]) / (kernel_width / 2), kernel_param
            )

            # save interpolator
            interp_value[i, x_i] = val
            interp_index[i, x_i] = (x_0 + x_i) % grid_shape
