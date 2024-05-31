"""Toeplitz FFT /NUFFT routines."""

__all__ = ["_toeplitz"]

import gc
import numpy as np

from .. import _subroutines


def _toeplitz(image, gram_matrix, threadsperblock, norm):  # noqa
    # unpack plan
    ndim = gram_matrix.ndim
    os_shape = gram_matrix.shape
    zmap_s_kernel = gram_matrix.zmap_s_kernel
    zmap_batch_size = gram_matrix.zmap_batch_size

    # get image shape
    shape = image.shape[-ndim:]

    # perform nufft
    if zmap_s_kernel is None:
        image = _do_toeplitz(
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

            # temporary image
            if ndim == 1:
                itmp = C * image[..., None, :]
            elif ndim == 2:
                itmp = C * image[..., None, :, :]
            elif ndim == 3:
                itmp = C * image[..., None, :, :, :]

            # apply gram filter
            itmp = _do_toeplitz(
                itmp, ndim, shape, os_shape, gram_matrix, threadsperblock, norm
            )

            # update image
            itmp = (C.conj() * itmp).sum(axis=-ndim - 1)
            image = image + itmp

    return image


# %% local subroutines
def _do_toeplitz(image, ndim, shape, os_shape, gram_matrix, threadsperblock, norm):
    # collect garbage
    gc.collect()

    # zero-pad
    image = _subroutines._resize(image, list(image.shape[:-ndim]) + list(os_shape))

    # FFT
    kspace = _subroutines.fft(image, axes=range(-ndim, 0), norm=norm)

    # interpolate
    kspace = _subroutines._bdot(kspace, gram_matrix, threadsperblock)

    # IFFT
    image = _subroutines.ifft(kspace, axes=range(-ndim, 0), norm=norm)

    # crop
    image = _subroutines._resize(image, list(image.shape[:-ndim]) + list(shape))

    # collect garbage
    gc.collect()

    return image
