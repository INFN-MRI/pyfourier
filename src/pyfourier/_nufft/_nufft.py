"""NUFFT and NUFFT adjoint routines."""

__all__ = ["_nufft_fwd", "_nufft_adj", "_nufft_backward"]

import gc
import numpy as np

from .. import _subroutines


def _nufft_fwd(image, nufft_plan, basis, weight, threadsperblock, norm):  # noqa
    # unpack plan
    ndim = nufft_plan.ndim
    oversamp = nufft_plan.oversamp
    width = nufft_plan.width
    beta = nufft_plan.beta
    os_shape = nufft_plan.os_shape
    interpolator = nufft_plan.interpolator
    zmap_t_kernel = nufft_plan.zmap_t_kernel
    zmap_s_kernel = nufft_plan.zmap_s_kernel
    zmap_batch_size = nufft_plan.zmap_batch_size
    device = nufft_plan.device

    # perform nufft
    if zmap_t_kernel is None:
        kspace = _do_nufft(
            image,
            ndim,
            oversamp,
            width,
            beta,
            os_shape,
            interpolator,
            basis,
            device,
            threadsperblock,
            norm,
        )
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
            ktmp = _do_nufft(
                itmp,
                ndim,
                oversamp,
                width,
                beta,
                os_shape,
                interpolator,
                basis,
                device,
                threadsperblock,
                norm,
            )
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


def _nufft_adj(kspace, nufft_plan, basis, weight, threadsperblock, norm):  # noqa
    # unpack plan
    ndim = nufft_plan.ndim
    oversamp = nufft_plan.oversamp
    width = nufft_plan.width
    beta = nufft_plan.beta
    shape = nufft_plan.shape
    interpolator = nufft_plan.interpolator
    zmap_t_kernel = nufft_plan.zmap_t_kernel
    zmap_s_kernel = nufft_plan.zmap_s_kernel
    zmap_batch_size = nufft_plan.zmap_batch_size
    device = nufft_plan.device

    # apply weight
    if weight is not None:
        kspace = weight * kspace

    # perform nufft adjoint
    if zmap_t_kernel is None:
        image = _do_nufft_adj(
            kspace,
            ndim,
            oversamp,
            width,
            beta,
            shape,
            interpolator,
            basis,
            device,
            threadsperblock,
            norm,
        )
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
            itmp = _do_nufft_adj(
                ktmp,
                ndim,
                oversamp,
                width,
                beta,
                shape,
                interpolator,
                basis,
                device,
                threadsperblock,
                norm,
            )
            itmp = itmp[..., None].swapaxes(0, -1)[0]

            # update image
            itmp = (C * itmp).sum(axis=-1)
            image = image + itmp

    return image


if _subroutines.pytorch_enabled:
    import torch

    def _nufft_backward(
        kspace, image, coord, nufft_plan, basis, weight, threadsperblock, norm
    ):  # noqa
        # preallocate output
        grad = torch.zeros_like(coord)

        # get shape and ndim
        shape = nufft_plan.shape
        ndim = coord.shape[-1]

        # compute grid
        grid_axes = [
            torch.linspace(
                -shape[ax] / 2, shape[ax] / 2 - 1, shape[ax], dtype=image.dtype
            )
            for ax in range(ndim)
        ]
        grid = torch.stack(torch.meshgrid(*grid_axes, indexing="ij"), axis=0)

        # backpropagate ramped signal
        ramped_image = torch.stack([grid[ax] * image for ax in range(ndim)], axis=0)
        backprop_ramp = _nufft_fwd(
            ramped_image, nufft_plan, basis, weight, threadsperblock, norm
        )

        # actual grad
        grad = (backprop_ramp.conj() * kspace).real

        # reshape from (ndim, ..., ncontrasts, npts) to (..., ncontrasts, npts, ndim)
        grad = grad[..., None].swapaxes(0, -1)[0]

        # sum over batches
        grad = grad.reshape(-1, coord.shape).sum(axis=0)

        return grad

else:

    def _nufft_backward():  # noqa
        pass


# %% local subroutines
def _do_nufft(
    image,
    ndim,
    oversamp,
    width,
    beta,
    os_shape,
    interpolator,
    basis,
    device,
    threadsperblock,
    norm,
):
    # collect garbage
    gc.collect()

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

    # collect garbage
    gc.collect()

    return kspace


def _do_nufft_adj(
    kspace,
    ndim,
    oversamp,
    width,
    beta,
    shape,
    interpolator,
    basis,
    device,
    threadsperblock,
    norm,
):
    # collect garbage
    gc.collect()

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
