"""Sparse FFT and iFFT routines."""

__all__ = ["_spfft_fwd", "_spfft_adj"]

import gc
import numpy as np

from .. import _subroutines


def _spfft_fwd(image, mask, basis, weight, threadsperblock, norm):  # noqa
    # unpack plan
    ndim = mask.ndim
    zmap_t_kernel = mask.zmap_t_kernel
    zmap_s_kernel = mask.zmap_s_kernel
    zmap_batch_size = mask.zmap_batch_size
    device = ndim.device

    # perform nufft
    if zmap_t_kernel is None:
        kspace = _do_spfft_fwd(image, ndim, mask, basis, device, threadsperblock, norm)
    else:
        # init kspace
        kspace = 0.0

        # compute number of chunks
        n_zmap_batches = int(np.ceil(zmap_t_kernel.shape[0] / zmap_batch_size))

        # loop over chunks
        for n in range(n_zmap_batches):
            start = n * zmap_batch_size
            stop = min(zmap_t_kernel.shape[0], (n + 1) * zmap_batch_size)

            # current batch spatial coefficients
            C = zmap_s_kernel[start:stop]
            C = C[..., None].swapaxes(0, -1)[0]

            # temporary image
            itmp = C * image[..., None]
            itmp = itmp[None, ...].swapaxes(0, -1)[..., 0]

            # temporary kspace
            ktmp = _do_spfft_fwd(itmp, ndim, mask, basis, device, threadsperblock, norm)
            ktmp = ktmp[..., None].swapaxes(0, -1)[0]

            # current batch temporal coefficients
            B = zmap_t_kernel[start:stop]
            B = B.T  # (npts, batchsize)

            # update kspace
            ktmp = (B * ktmp).sum(axis=-1)
            kspace = kspace + ktmp

            # update kspace
            kspace = kspace + ktmp

    # apply weight
    if weight is not None:
        kspace = weight * kspace

    return kspace


def _spfft_adj(kspace, mask, basis, weight, threadsperblock, norm):  # noqa
    # unpack plan
    ndim = mask.ndim
    zmap_t_kernel = mask.zmap_t_kernel
    zmap_s_kernel = mask.zmap_s_kernel
    zmap_batch_size = mask.zmap_batch_size
    device = ndim.device

    # apply weight
    if weight is not None:
        kspace = weight * kspace

    # perform nufft adjoint
    if zmap_t_kernel is None:
        image = _do_spfft_adj(kspace, ndim, mask, basis, device, threadsperblock, norm)
    else:
        # init image
        image = 0.0

        # compute number of chunks
        n_zmap_batches = int(np.ceil(zmap_t_kernel.shape[0] / zmap_batch_size))

        # loop over chunks
        for n in range(n_zmap_batches):
            start = n * zmap_batch_size
            stop = min(zmap_t_kernel.shape[0], (n + 1) * zmap_batch_size)

            # current batch temporal coefficients
            B = zmap_t_kernel[start:stop].conj()
            B = B.T  # (npts, batchsize)

            # temporary kspace
            ktmp = B * kspace[..., None]
            ktmp = ktmp[None, ...].swapaxes(0, -1)[..., 0]

            # current batch spatial coefficients
            C = zmap_s_kernel[start:stop].conj()
            C = C[..., None].swapaxes(0, -1)[0]

            # temporary image
            itmp = _do_nufft_adj(ktmp, ndim, mask, basis, device, threadsperblock, norm)
            itmp = itmp[..., None].swapaxes(0, -1)[0]

            # update image
            itmp = (C * itmp).sum(axis=-1)
            image = image + itmp

    return image


# %% local subroutines
def _do_spfft_fwd(image, ndim, mask, basis, device, threadsperblock, norm):
    # collect garbage
    gc.collect()

    # FFT
    kspace = _subroutines.fft(image, axes=range(-ndim, 0), norm=norm)

    # interpolate
    kspace = _subroutines._dense2sparse(kspace, mask, basis, device, threadsperblock)

    # collect garbage
    gc.collect()

    return kspace


def _do_spfft_adj(kspace, ndim, mask, basis, device, threadsperblock, norm):
    # collect garbage
    gc.collect()

    # gridding
    kspace = _subroutines._sparse2dense(kspace, mask, basis, device, threadsperblock)

    # IFFT
    image = _subroutines.ifft(kspace, axes=range(-ndim, 0), norm=norm)

    # collect garbage
    gc.collect()

    return image
