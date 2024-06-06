"""FFT and iFFT routines."""

__all__ = ["_fft_fwd", "_fft_adj"]

import numpy as np

from .. import _subroutines


def _fft_fwd(image, mask, basis, norm):  # noqa
    # unpack plan
    ndim = len(mask.shape)
    zmap_t_kernel = mask.zmap_t_kernel
    zmap_s_kernel = mask.zmap_s_kernel
    zmap_batch_size = mask.zmap_batch_size

    # perform nufft
    if zmap_s_kernel is None:
        kspace = _do_fft_fwd(image, ndim, mask, basis, norm)
    else:
        # init kspace
        kspace = 0.0

        # compute number of chunks
        n_zmap_batches = int(np.ceil(zmap_s_kernel.shape[0] / zmap_batch_size))

        # loop over chunks
        for n in range(n_zmap_batches):
            start = n * zmap_batch_size
            stop = min(zmap_s_kernel.shape[0], (n + 1) * zmap_batch_size)

            # current batch spatial coefficients
            C = zmap_s_kernel[start:stop]
            C = C[..., None].swapaxes(0, -1)[0]

            # temporary image
            itmp = C * image[..., None]
            itmp = itmp[None, ...].swapaxes(0, -1)[..., 0]

            # temporary kspace
            ktmp = _do_fft_fwd(itmp, ndim, mask, basis, norm)
            ktmp = ktmp[..., None].swapaxes(0, -1)[0]

            # current batch temporal coefficients
            B = zmap_t_kernel[start:stop]
            B = B.T  # (npts, batchsize)

            # update kspace
            ktmp = (B * ktmp).sum(axis=-1)
            kspace = kspace + ktmp

            # update kspace
            kspace = kspace + ktmp

    return kspace


def _fft_adj(kspace, mask, basis, norm):  # noqa
    # unpack plan
    ndim = len(mask.shape)
    zmap_t_kernel = mask.zmap_t_kernel
    zmap_s_kernel = mask.zmap_s_kernel
    zmap_batch_size = mask.zmap_batch_size

    # perform nufft adjoint
    if zmap_s_kernel is None:
        image = _do_fft_adj(kspace, ndim, mask, basis, norm)
    else:
        # init image
        image = 0.0

        # compute number of chunks
        n_zmap_batches = int(np.ceil(zmap_s_kernel.shape[0] / zmap_batch_size))

        # loop over chunks
        for n in range(n_zmap_batches):
            start = n * zmap_batch_size
            stop = min(zmap_s_kernel.shape[0], (n + 1) * zmap_batch_size)

            # current batch temporal coefficients
            B = zmap_t_kernel[start:stop]
            B = B.T  # (npts, batchsize)

            # temporary kspace
            ktmp = B.conj() * kspace[..., None]
            ktmp = ktmp[None, ...].swapaxes(0, -1)[..., 0]

            # current batch spatial coefficients
            C = zmap_s_kernel[start:stop]
            C = C[..., None].swapaxes(0, -1)[0]

            # temporary image
            itmp = _do_fft_adj(ktmp, ndim, mask, basis, norm)
            itmp = itmp[..., None].swapaxes(0, -1)[0]

            # update image
            itmp = (C.conj() * itmp).sum(axis=-1)
            image = image + itmp

    return image


# %% local subroutines
def _do_fft_fwd(image, ndim, mask, basis, norm):

    # FFT
    kspace = _subroutines.fft(image, axes=range(-ndim, 0), norm=norm)
    
    # Backproject on contrast space
    kspace = _basis_adj(kspace, basis, ndim)
    
    # Sample
    if mask.indexes is not None:
        kspace = kspace * mask.indexes

    return kspace


def _do_fft_adj(kspace, ndim, mask, basis, norm):
    
    # Sample
    if mask.indexes is not None:
        kspace = kspace * mask.indexes
        
    # Project on subspace
    kspace = _basis_fwd(kspace, basis, ndim)

    # IFFT
    image = _subroutines.ifft(kspace, axes=range(-ndim, 0), norm=norm)
    
    return image


def _basis_fwd(kspace, basis, ndim):
    if basis is not None:
        b = basis # (T, K)
        kspace = kspace[..., None].swapaxes(-ndim-1, -1)
        kspace = kspace @ b
        kspace = kspace.swapaxes(-ndim-1, -1)[..., 0]
        
    return kspace


def _basis_adj(kspace, basis, ndim):
    if basis is not None:
        b = basis.conj().T  # (K, T)
        kspace = kspace[..., None].swapaxes(-ndim-1, -1)
        kspace = kspace @ b
        kspace = kspace.swapaxes(-ndim-1, -1)[..., 0]
        
    return kspace

