"""Sampling pattern planning subroutines."""

__all__ = ["FFTPlan"]

import numpy as np
import numba as nb

from .. import _utils

# detect GPU
_, gpu_backend = _utils.detect_gpu_backend()


class FFTPlan:  # noqa
    def __init__(
        self, sparse, indexes, shape, zmap_t_kernel, zmap_s_kernel, L_batch_size
    ):

        ndim = indexes.shape[-1]

        # expand
        if np.isscalar(shape):
            shape = ndim * [shape]

        if sparse:

            # expand singleton dimensions
            ishape = list(indexes.shape[:-1])

            while len(ishape) < 3:
                ishape = [1] + ishape

            nframes = ishape[0]
            ishape = ishape[1:]

            # parse input sizes
            npts = np.prod(ishape)

            # arg reshape
            indexes = indexes.reshape([nframes, npts, ndim])
            indexes = indexes.permute(2, 0, 1)

            # send to numba
            index = [_utils.to_backend(nb, idx) for idx in indexes]

            # revert axis (x, y, z) > (z, y, x)
            index = index[::-1]

            # transform to tuples
            self.index = tuple(index)
        else:
            self.index = index

        self.dshape = tuple(ishape)
        self.ishape = tuple(shape)
        self.ndim = ndim
        self.zmap_t_kernel = zmap_t_kernel
        self.zmap_s_kernel = zmap_s_kernel
        self.zmap_batch_size = L_batch_size
        self.sparse = sparse
        self.device = None

    def to(self, device):  # noqa
        if self.device is None or device != self.device:

            # get device tag
            if self.sparse:
                device_tag = _utils.get_device_tag(device)
                if device_tag == "cpu":
                    backend = nb
                else:
                    backend = gpu_backend

                # switch to numba / cupy
                self.index = list(self.index)
                self.index = [
                    _utils.to_device(idx, device, backend) for idx in self.index
                ]
                self.index = tuple(self.index)
            else:
                self.index = _utils.to_device(self.index, device)

            if self.zmap_s_kernel is not None:
                self.zmap_t_kernel = _utils.to_device(self.zmap_t_kernel, device)
                self.zmap_s_kernel = _utils.to_device(self.zmap_s_kernel, device)

            self.device = device

        return self
