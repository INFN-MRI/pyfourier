"""Sparse (Cartesian and Non-Cartesian) Gram FFT routines."""

__all__ = ["_sparsegram"]

import gc
import numpy as np

from .. import _subroutines


def _sparsegram(image, gram_matrix, threadsperblock, norm):  # noqa
    # unpack plan
    ndim = gram_matrix.ndim
    os_shape = gram_matrix.shape
    zmap_s_kernel = gram_matrix.zmap_s_kernel
    zmap_batch_size = gram_matrix.zmap_batch_size

    # get image shape
    shape = image.shape[-ndim:]

    # perform nufft
    if zmap_s_kernel is None:
        image = _do_sparsegram(
            image, ndim, shape, os_shape, gram_matrix, threadsperblock, norm
        )
    else:
        # init kspace
        image = 0.0

        # compute number of chunks
        n_zmap_batches = int(np.ceil(zmap_s_kernel.shape[0] / zmap_batch_size))

        # loop over chunks
        for n in range(n_zmap_batches):
            start = n * zmap_batch_size
            stop = min(zmap_s_kernel.shape[0], (n + 1) * zmap_batch_size)

            # current batch spatial coefficients
            C = zmap_s_kernel[start:stop]
            C = C[..., None].swapaxes(0, -1)

            # temporary image
            itmp = C * image[..., None]
            itmp = itmp[None, ...].swapaxes(0, -1)[..., 0]

            # apply gram filter
            itmp = _do_sparsegram(
                itmp, ndim, shape, os_shape, gram_matrix, threadsperblock, norm
            )

            # update image
            itmp = (C.conj() * itmp).sum(axis=0)
            image = image + itmp

    return image


# %% local subroutines
def _do_sparsegram(image, ndim, shape, os_shape, gram_matrix, threadsperblock, norm):
    # collect garbage
    gc.collect()

    # zero-pad
    image = _subroutines._resize(image, list(image.shape[:-ndim]) + list(os_shape))

    # FFT
    kspace = _subroutines.fft(image, axes=range(-ndim, 0), norm=norm, centered=False)

    # interpolate
    kspace = _subroutines._bdot(kspace, gram_matrix, threadsperblock)

    # IFFT
    image = _subroutines.ifft(kspace, axes=range(-ndim, 0), norm=norm, centered=False)

    # crop
    image = _subroutines._resize(image, list(image.shape[:-ndim]) + list(shape))

    # collect garbage
    gc.collect()

    return image
