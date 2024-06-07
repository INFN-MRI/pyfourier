"""NUFFT main wrapper."""

__all__ = ["nufft"]

from .. import _subroutines

from . import _nufft
from . import _plan

if _subroutines.pytorch_enabled:
    import torch
    USE_TORCH = True
else:
    USE_TORCH = False


def nufft(
    image,
    coord=None,
    shape=None,
    nufft_plan=None,
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
    width=4,
    oversamp=1.25,
):
    """
    N-dimensional Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    image : ArrayLike
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    coord : ArrayLike
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Sequence[int], optional
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
        The default is ``None`` (grid size equals to input data size, i.e. ``osf = 1``).
    nufft_plan : NUFFTPlan, optional
        Structure containing sparse interpolator matrix:

        * ndim (``int``): number of spatial dimensions.
        * oversampling (``Iterable[float]``): grid oversampling factor (z, y, x).
        * width (``Iterable[int]``): kernel width (z, y, x).
        * beta (``Iterable[float]``): Kaiser Bessel parameter (z, y, x).
        * os_shape (``Iterable[int]``): oversampled grid shape (z, y, x).
        * shape (``Iterable[int]``): grid shape (z, y, x).
        * interpolator (``Interpolator``): precomputed interpolator object.
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
    threadsperblock : int, optional
        CUDA blocks size (for GPU only). The default is ``128``.
    width : int | Sequence[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``4``.
    oversamp : float | Sequence[float], optional
        Grid oversampling factor of shape ``(ndim,)``.
        If scalar, isotropic oversampling is assumed.
        The default is ``1.25``.

    Returns
    -------
    kspace : ArrayLike
        Output Non-Cartesian kspace of shape ``(..., ncontrasts, nviews, nsamples)``.

    Notes
    -----
    Non-uniform coordinates axes ordering is assumed to be ``(x, y)`` for 2D signals
    and ``(x, y, z)`` for 3D. Conversely, axes ordering for grid shape, kernel width
    and Kaiser Bessel parameters are assumed to be ``(z, y, x)``.

    Coordinates tensor shape is ``(ncontrasts, nviews, nsamples, ndim)``. If there are less dimensions
    (e.g., single-shot or single contrast trajectory), assume singleton for the missing ones:

    * ``coord.shape = (nsamples, ndim) -> (1, 1, nsamples, ndim)``
    * ``coord.shape = (nviews, nsamples, ndim) -> (1, nviews, nsamples, ndim)``

    """        
    # switch to torch if possible
    if USE_TORCH:
        ibackend = _subroutines.get_backend(image)
        image = _subroutines.to_backend(torch, image)
        if coord is not None:
            coord = _subroutines.to_backend(torch, coord)
        if basis is not None:
            basis = _subroutines.to_backend(torch, basis)
        if zmap is not None:
            zmap = _subroutines.to_backend(torch, zmap)
        if T is not None:
            T = _subroutines.to_backend(torch, T)
        if weight is not None:
            weight = _subroutines.to_backend(torch, weight)
            
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
    if nufft_plan is None:
        if shape is None:
            shape = image.shape[-coord.shape[-1] :]

        coord = _subroutines.astype(coord, backend.float32)
        nufft_plan = _plan.plan_nufft(
            coord, shape, width, oversamp, zmap, L, nbins, dt, T, L_batch_size,
        )

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
    nufft_plan.to(device)
    image = _subroutines.to_device(image, device)
    if basis is not None:
        basis = _subroutines.to_device(basis, device)
    if weight is not None:
        weight = _subroutines.to_device(weight, device)

    # perform operation
    if backend.__name__ == "torch":
        if coord is None:
            kspace = NUFFT.apply(
                image, nufft_plan, basis, weight, threadsperblock, norm
            )
        else:
            kspace = NUFFTTraj.apply(
                image, coord, nufft_plan, basis, weight, threadsperblock, norm
            )
    else:
        kspace = _nufft._nufft_fwd(
            image, nufft_plan, basis, weight, threadsperblock, norm
        )

    # return
    kspace = _subroutines.astype(kspace, dtype)
    kspace = _subroutines.to_device(kspace, idevice)
    
    # original backend
    if USE_TORCH:
        kspace = _subroutines.to_backend(ibackend, kspace)
        
    return kspace



# %% local subroutines
if _subroutines.pytorch_enabled:

    class NUFFT(torch.autograd.Function):
        @staticmethod
        def forward(image, nufft_plan, basis, weight, threadsperblock, norm):
            return _nufft._nufft_fwd(
                image, nufft_plan, basis, weight, threadsperblock, norm
            )

        @staticmethod
        def setup_context(ctx, inputs, output):
            _, nufft_plan, basis, weight, threadsperblock, norm = inputs
            ctx.set_materialize_grads(False)
            ctx.nufft_plan = nufft_plan
            ctx.basis = basis
            ctx.weight = weight
            ctx.threadsperblock = threadsperblock
            ctx.norm = norm

        @staticmethod
        def backward(ctx, kspace):
            nufft_plan = ctx.nufft_plan
            basis = ctx.basis
            weight = ctx.weight
            threadsperblock = ctx.threadsperblock
            norm = ctx.norm

            # gradient with respect to samples
            grad_kspace = _nufft._nufft_adj(
                kspace, nufft_plan, basis, weight, threadsperblock, norm
            )

            return (
                grad_kspace,
                None,
                None,
                None,
                None,
                None,
            )

    class NUFFTTraj(torch.autograd.Function):
        @staticmethod
        def forward(image, coord, nufft_plan, basis, weight, threadsperblock, norm):
            return _nufft._nufft_fwd(
                image, nufft_plan, basis, weight, threadsperblock, norm
            )

        @staticmethod
        def setup_context(ctx, inputs, output):
            image, coord, nufft_plan, basis, weight, threadsperblock, norm = inputs
            ctx.save_for_backward(image, coord)
            ctx.set_materialize_grads(False)
            ctx.nufft_plan = nufft_plan
            ctx.basis = basis
            ctx.weight = weight
            ctx.threadsperblock = threadsperblock
            ctx.norm = norm

        @staticmethod
        def backward(ctx, kspace):
            image, coord = ctx.saved_tensors
            nufft_plan = ctx.nufft_plan
            basis = ctx.basis
            weight = ctx.weight
            threadsperblock = ctx.threadsperblock
            norm = ctx.norm

            # gradient with respect to samples
            grad_kspace = _nufft._nufft_adj(
                kspace, nufft_plan, basis, weight, threadsperblock, norm
            )

            # gradient with respect to trajectory
            grad_coord = _nufft._nufft_backward(
                kspace, image, coord, nufft_plan, basis, weight, threadsperblock, norm
            )

            return (
                grad_kspace,
                grad_coord,
                None,
                None,
                None,
                None,
                None,
            )
