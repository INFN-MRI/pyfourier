"""NUFFT and NUFFT adjoint routines."""

__all__ = ["_nufft_fwd", "_nufft_adj"]

import gc

from .. import _subroutines


def _nufft_fwd(image, nufft_plan, basis, weight, device, threadsperblock, norm):  # noqa
    # collect garbage
    gc.collect()

    # unpack plan
    ndim = nufft_plan.ndim
    oversamp = nufft_plan.oversamp
    width = nufft_plan.width
    beta = nufft_plan.beta
    os_shape = nufft_plan.os_shape
    interpolator = nufft_plan.interpolator
    device = nufft_plan.device

    # apodize
    image = _subroutines._apodize(image, ndim, oversamp, width, beta)

    # zero-pad
    image = _subroutines._resize(image, list(image.shape[:-ndim]) + list(os_shape))

    # FFT
    kspace = _subroutines.fft(image, axes=range(-ndim, 0), norm=norm)

    # interpolate
    kspace = _subroutines._cart2noncart(
        kspace, interpolator, basis, device, threadsperblock
    )

    # apply weight
    if weight is not None:
        kspace = weight * kspace

    # collect garbage
    gc.collect()

    return kspace


def _nufft_adj(
    kspace, nufft_plan, basis, weight, device, threadsperblock, norm
):  # noqa
    # collect garbage
    gc.collect()

    # unpack plan
    ndim = nufft_plan.ndim
    oversamp = nufft_plan.oversamp
    width = nufft_plan.width
    beta = nufft_plan.beta
    shape = nufft_plan.shape
    interpolator = nufft_plan.interpolator
    device = nufft_plan.device

    # apply weight
    if weight is not None:
        kspace = weight * kspace

    # gridding
    kspace = _subroutines._noncart2cart(
        kspace, interpolator, basis, device, threadsperblock
    )

    # IFFT
    image = _subroutines.ifft(kspace, axes=range(-ndim, 0), norm=norm)

    # crop
    image = _subroutines._resize(image, list(image.shape[:-ndim]) + list(shape))

    # apodize
    image = _subroutines._apodize(image, ndim, oversamp, width, beta)

    # collect garbage
    gc.collect()

    return image
