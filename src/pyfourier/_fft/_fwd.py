"""FFT main wrapper."""

__all__ = ["fftn"]

from .. import _subroutines

from . import _fft
from . import _plan

if _subroutines.pytorch_enabled:
    import torch

    USE_TORCH = True
else:
    USE_TORCH = False


def fftn(
    ndim,
    image,
    mask=None,
    plan=None,
    basis=None,
    zmap=None,
    L=6,
    nbins=(40, 40),
    dt=None,
    T=None,
    L_batch_size=None,
    norm=None,
    device="cpu",
):
    """
    N-dimensional Fast Fourier Transform.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions for FFT computation.
    image : ArrayLike
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    mask : ArrayLike
        Binary k-space mask indexes of shape ``(ncontrasts, ny, nx)`` (2D)
        or ``(ncontrasts, nz, ny, nx)`` (3D).
    plan : FFTPlan, optional
        Structure containing sparse sampling matrix:

        * indexes (``ArrayLike``): indexes of the non-zero entries of interpolator sparse matrix of shape (ndim, ncoord).
        * shape (``Sequence[int]``): oversampled grid shape of shape (ndim,). Order of axes is (z, y, x).
        * zmap_s_kernel (``ArrayLike``): zmap spatial basis.
        * zmap_t_kernel (``ArrayLike``): zmap temporal basis.
        * zmap_batch_size (``int``): zmap processing batch size.
        * device (``str``): computational device.

    basis : ArrayLike, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeff)``; can be ``None``. The default is ``None``.
    zmap : ArrayLike, optional
        Field map in [Hz]; can be real (B0 map) or complex (R2* + 1i * B0).
        The default is ``None``.
    L : int, optional
        Number of zmap segmentations. The default is ``6``.
    nbins : int | Sequence[int], optional
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
    norm : str, optional
        Keyword to specify the normalization mode (``None`` or ``"ortho"``).
    device : str | int, optional
        Computational device.
        Can be either specified as a string (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``),
        or integer (``-1 (="cpu")`` or ``n (="cuda:n")``, with ``n=0, 1,...nGPUs``)
        The default is ``cpu``.

    Returns
    -------
    kspace : ArrayLike
        Output sparse kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

    """
    # switch to torch if possible
    if USE_TORCH:
        ibackend = _subroutines.get_backend(image)
        image = _subroutines.to_backend(torch, image)
        if mask is not None:
            mask = _subroutines.to_backend(torch, mask)
        if basis is not None:
            basis = _subroutines.to_backend(torch, basis)
        if zmap is not None:
            zmap = _subroutines.to_backend(torch, zmap)
        if T is not None:
            T = _subroutines.to_backend(torch, T)

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

    # infer shape
    shape = image.shape[-ndim:]

    # if not provided, plan interpolator
    if plan is None:
        mask = _subroutines.astype(mask, backend.int16)
        plan = _plan.plan_fft(mask, shape, zmap, L, nbins, dt, T, L_batch_size)

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

    # cast to device if necessary
    mask.to(device)
    image = _subroutines.to_device(image, device)
    if basis is not None:
        basis = _subroutines.to_device(basis, device)

    # perform operation
    if backend.__name__ == "torch":
        kspace = FFT.apply(image, plan, basis, norm)
    else:
        kspace = _fft._fft_fwd(image, plan, basis, norm)

    # return
    kspace = _subroutines.astype(kspace, dtype)
    kspace = _subroutines.to_device(kspace, idevice)

    # original backend
    if USE_TORCH:
        kspace = _subroutines.to_backend(ibackend, kspace)

    return kspace


# %% local subroutines
if _subroutines.pytorch_enabled:

    class FFT(torch.autograd.Function):
        @staticmethod
        def forward(image, plan, basis, norm):
            return _fft._fft_fwd(image, plan, basis, norm)

        @staticmethod
        def setup_context(ctx, inputs, output):
            _, plan, basis, norm = inputs
            ctx.set_materialize_grads(False)
            ctx.plan = plan
            ctx.basis = basis
            ctx.norm = norm

        @staticmethod
        def backward(ctx, kspace):
            plan = ctx.plan
            basis = ctx.basis
            norm = ctx.norm

            # gradient with respect to samples
            grad_kspace = _fft._fft_adj(kspace, plan, basis, norm)

            return (
                grad_kspace,
                None,
                None,
                None,
            )
