"""Dense Gram FFT routines."""

__all__ = ["_gram"]

import gc
import numpy as np

from .. import _subroutines


def _gram(image, gram_matrix, norm):  # noqa
    # unpack plan
    ndim = gram_matrix.ndim
    shape = gram_matrix.shape
    zmap_s_kernel = gram_matrix.zmap_s_kernel
    zmap_batch_size = gram_matrix.zmap_batch_size

    if gram_matrix.st_kernel is None:
        return image

    # perform nufft
    if zmap_s_kernel is None:
        image = _do_gram(image, gram_matrix, norm)
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
            C = C[..., None].swapaxes(0, -1)[0]  # (nz, ny, nseg)

            # temporary image
            itmp = image[..., None] * C
            itmp = itmp[None, ...].swapaxes(0, -1)[..., 0]

            # apply gram filter
            itmp = _do_gram(itmp, gram_matrix, norm)

            # update image
            itmp = itmp[..., None].swapaxes(0, -1)[0]
            itmp = (C.conj() * itmp).sum(axis=-1)
            image = image + itmp

    return image


# %% local subroutines
def _do_gram(image, gram_matrix, norm):

    # collect garbage
    gc.collect()

    # unpack input
    islowrank = gram_matrix.islowrank
    haszmap = gram_matrix.zmap_s_kernel is not None

    # keep original shape
    shape = image.shape

    # reshape for computation
    if islowrank:
        if haszmap:
            image = image.reshape(
                image.shape[0], -1, image.shape[-3], np.prod(image.shape[-2:])
            )  # (nseg, nbatches, ncontrasts, nvoxels)
            image = _subroutines.transpose(
                image, (1, 0, 3, 2)
            )  # (nbatches, nseg, nvoxels, ncontrasts)
            image = image.reshape(
                image.shape[:2], *shape[-2:], image.shape[-1]
            )  # (nbatches, nseg, nz, ny, ncontrasts)
        else:
            image = image.reshape(
                -1, image.shape[-3], np.prod(image.shape[-2:])
            )  # (nbatches, ncontrasts, nvoxels)
            image = image.swapaxes(1, -1)  # (nbatches, nvoxels, ncontrasts)
            image = image.reshape(
                image.shape[0], *shape[-2:], image.shape[-1]
            )  # (nbatches, nz, ny, ncontrasts)
    else:
        if haszmap:
            image = image.reshape(
                image.shape[0], -1, image.shape[-3:]
            )  # (nseg, nbatches, (ncontrasts), nz, ny)
            image = image.swapaxes(0, 1)  # (nbatches, nseg, (ncontrasts), nz, ny)

    # FFT
    kspace = _subroutines.fft(image, axes=range(-2, 0), norm=norm, centered=False)

    # interpolate
    if islowrank:
        kspace = kspace @ gram_matrix.st_kernel
    else:
        kspace = kspace * gram_matrix.st_kernel

    # IFFT
    image = _subroutines.ifft(kspace, axes=range(-2, 0), norm=norm, centered=False)

    # reshape for output
    if islowrank:
        if haszmap:
            image = image.reshape(
                image.shape[:2], -1, image.shape[-1]
            )  # (nbatches, nseg, nvoxels, ncontrasts)
            image = _subroutines.transpose(
                image, (1, 0, 3, 2)
            )  # (nseg, nbatches, ncontrasts, nvoxels)
        else:
            image = image.reshape(
                image.shape[0], -1, image.shape[-1]
            )  # (nbatches, nvoxels, ncontrasts)
            image = _subroutines.transpose(
                image, (0, 2, 1)
            )  # (nbatches, nvoxels, nvoxels)
    else:
        if haszmap:
            image = image.swapaxes(0, 1)  # (nseg, nbatches, (ncontrasts), nz, ny)

    # keep original shape
    image = image.reshape(*shape)

    # collect garbage
    gc.collect()

    return image
