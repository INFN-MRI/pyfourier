"""Sparse (Cartesian and Non-Cartesian) Gram FFT main wrapper."""

__all__ = ["sparse_gram"]

from .. import _subroutines

from . import _sparsegram

if _subroutines.pytorch_enabled:
    import torch
    USE_TORCH = True
else:
    USE_TORCH = False


def sparse_gram(
    image,
    gram_matrix,
    norm=None,
    device="cpu",
    threadsperblock=128,
):
    """
    N-dimensional sparse (Cartesian and Non-Cartesian) Gram FFT.

    Parameters
    ----------
    image : ArrayLike
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    gram_matrix : GramMatrix
        Structure containing Toeplitz kernel (i.e., Fourier transform of system tPSF).
    norm : str, optional
        Keyword to specify the normalization mode (``None`` or ``"ortho"``).
    device : str | int, optional
        Computational device.
        Can be either specified as a string (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``),
        or integer (``-1 (="cpu")`` or ``n (="cuda:n")``, with ``n=0, 1,...nGPUs``)
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.

    Returns
    -------
    image : ArrayLike
        Output image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).

    """
    # detect backend and device
    backend = _subroutines.get_backend(image)
    idevice = _subroutines.get_device(image)

    # if not provided, use original device
    if device is None:
        device = idevice
    else:
        if isinstance(device, str):
            if device == "cpu":
                device = -1
            else:
                device = int(device.split(":")[-1])

    # make sure datatype is correct
    dtype = image.dtype
    if dtype in (backend.float16, backend.float32, backend.float64):
        image = _subroutines.astype(image, backend.float32)
    else:
        image = _subroutines.astype(image, backend.complex64)

    # cast to device if necessary
    gram_matrix.to(device)
    image = _subroutines.to_device(image, device)

    # perform operation
    if backend.__name__ == "torch":
        image = SparseGram.apply(image, gram_matrix, threadsperblock, norm)
    else:
        image = _sparsegram._sparsegram(image, gram_matrix, threadsperblock, norm)

    # return
    image = _subroutines.astype(image, dtype)
    return _subroutines.to_device(image, idevice)


# %% local subroutines
if _subroutines.pytorch_enabled:

    class SparseGram(torch.autograd.Function):
        @staticmethod
        def forward(image, gram_matrix, threadsperblock, norm):
            return _sparsegram._sparsegram(image, gram_matrix, threadsperblock, norm)

        @staticmethod
        def setup_context(ctx, inputs, output):
            _, gram_matrix, threadsperblock, norm = inputs
            ctx.set_materialize_grads(False)
            ctx.gram_matrix = gram_matrix
            ctx.threadsperblock = threadsperblock
            ctx.norm = norm

        @staticmethod
        def backward(ctx, image):
            gram_matrix = ctx.gram_matrix
            threadsperblock = ctx.threadsperblock
            norm = ctx.norm

            # gradient with respect to image
            grad_image = _sparsegram._sparsegram(image, gram_matrix, threadsperblock, norm)

            return (
                grad_image,
                None,
                None,
                None,
            )
