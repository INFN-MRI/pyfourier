"""
Resizing (cropping and padding) subroutines.

Adapted from https://github.com/mikgroup/sigpy/blob/main/sigpy/util.py#L120

"""

__all__ = ["_resize"]

import numpy as np

from . import _utils


# this is also in main deep-mr package, I decided to violate DRY for now.
def _resize(input, oshape):
    # get tensor backend and device
    backend = _utils.get_backend(input)

    if isinstance(oshape, int):
        oshape = [oshape]

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input

    # shift not supported for now
    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [
        min(i - si, o - so) for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)
    ]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = _utils.zeros(oshape1, input.dtype, device, backend)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output


# %% subroutines
def _expand_shapes(*shapes):
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)

    shapes_exp = [np.asarray([1] * (max_ndim - len(shape)) + shape) for shape in shapes]
    shapes_exp = np.stack(shapes_exp, axis=0)  # (nshapes, max_ndim)
    shapes_exp = np.max(shapes_exp, axis=0)

    # restore original shape in non-padded portions
    shapes_exp = [list(shapes_exp[: -len(shape)]) + shape for shape in shapes]

    return tuple(shapes_exp)
