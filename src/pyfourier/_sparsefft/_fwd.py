"""Sparse FFT main wrapper."""

__all__ = ["sparse_fft"]

from .. import _subroutines

from . import _sparsefft
from . import _plan

if _subroutines.pytorch_enabled:
    import torch


def sparse_fft(
    image,
    indexes=None,
    shape=None,
    mask=None,
    basis=None,
    zmap=None,
    L=6,
    nbins=(40, 40),
    dt=None,
    T=None,
    L_batch_size=None,
    weight=None,
    norm=None,
    device="cpu",
    threadsperblock=128,
):
    """
    N-dimensional sparse Fast Fourier Transform.

    Parameters
    ----------
    image : ArrayLike
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    indexes : ArrayLike
        Sampled k-space points indexes of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Iterable[int], optional
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
        The default is ``None`` (grid size equals to input data size, i.e. ``osf = 1``).
    basis : ArrayLike, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeff)``; can be ``None``. The default is ``None``.
    zmap : ArrayLike, optional
        Field map in [Hz]; can be real (B0 map) or complex (R2* + 1i * B0).
        The default is ``None``.
    L : int, optional
        Number of zmap segmentations. The default is ``6``.
    nbins : int, optional
        Granularity of exponential approximation.
        For real zmap, it is a scalar (1D histogram).
        For complex zmap, it must be a tuple of ints (2D histogram).
        The default is ``(40, 40)``.
    dt : float, optional
        Dwell time in ms. The default is ``None``.
    T : ArrayLike, optional
        Tensor with shape ``(npts,)``, representing the sampling instant of
        each k-space point along the readout. When T is ``None``, this is
        inferred from ``dt`` (if provided), assuming that readout starts
        immediately after excitation (i.e., TE=0).
    L_batch_size : int, optional
        Number of zmap segments to be processed in parallel. If ``None``,
        process all segments simultaneously. The default is ``None``.
    weight: ArrayLike, optional
        Tensor to be used as a weight for the output k-space data (e.g., dcf**0.5).
        Must be broadcastable with ``kspace`` (i.e., the output). The default is ``None``.
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
    kspace : ArrayLike
        Output sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

    Notes
    -----
    Sampled points indexes axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape is assumed to be ``(z, y, x)``.

    Indexes tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

    * ``indexes.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
    * ``indexes.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

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

    # if not provided, plan interpolator
    if mask is None:
        if shape is None:
            shape = image.shape[-indexes.shape[-1] :]

        indexes = _subroutines.astype(indexes, backend.int16)
        mask = _plan.plan_spfft(indexes, shape, zmap, L, nbins, dt, T, L_batch_size)

    # make sure datatype is correct
    dtype = image.dtype
    if dtype in (backend.float16, backend.float32, backend.float64):
        image = _subroutines.astype(image, backend.float32)
    else:
        image = _subroutines.astype(image, backend.complex64)

    # handle basis
    if basis is not None:
        # make sure datatype is correct
        if basis.dtype in (backend.float16, backend.float32, backend.float64):
            basis = _subroutines.astype(basis, backend.float32)
        else:
            basis = _subroutines.astype(basis, backend.complex64)

    # handle weight
    if weight is not None:
        # make sure datatype is correct
        if weight.dtype in (backend.float16, backend.float32, backend.float64):
            weight = _subroutines.astype(weight, backend.float32)
        else:
            weight = _subroutines.astype(weight, backend.complex64)

    # cast to device if necessar
    mask.to(device)
    image = _subroutines.to_device(image, device)
    if basis is not None:
        basis = _subroutines.to_device(basis, device)
    if weight is not None:
        weight = _subroutines.to_device(weight, device)

    # perform operation
    if backend.__name__ == "torch":
        kspace = SparseFFT.apply(image, mask, basis, weight, threadsperblock, norm)
    else:
        kspace = _sparsefft._spfft_fwd(
            image, mask, basis, weight, threadsperblock, norm
        )

    # return
    kspace = _subroutines.astype(kspace, dtype)
    return _subroutines.to_device(kspace, idevice)


# %% local subroutines
if _subroutines.pytorch_enabled:

    class SparseFFT(torch.autograd.Function):
        @staticmethod
        def forward(image, mask, basis, weight, threadsperblock, norm):
            return _sparsefft._spfft_fwd(
                image, mask, basis, weight, threadsperblock, norm
            )

        @staticmethod
        def setup_context(ctx, inputs, output):
            _, mask, basis, weight, threadsperblock, norm = inputs
            ctx.set_materialize_grads(False)
            ctx.mask = mask
            ctx.basis = basis
            ctx.weight = weight
            ctx.threadsperblock = threadsperblock
            ctx.norm = norm

        @staticmethod
        def backward(ctx, kspace):
            mask = ctx.mask
            basis = ctx.basis
            weight = ctx.weight
            threadsperblock = ctx.threadsperblock
            norm = ctx.norm

            # gradient with respect to samples
            grad_kspace = _sparsefft._spfft_adj(
                kspace, mask, basis, weight, threadsperblock, norm
            )

            return (
                grad_kspace,
                None,
                None,
                None,
                None,
                None,
            )
