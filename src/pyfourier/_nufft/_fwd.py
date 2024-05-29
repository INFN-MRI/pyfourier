"""NUFFT main wrapper."""

__all__ = ["nufft"]

from . import _nufft

from .. import _subroutines

if _subroutines.pytorch_enabled:
    import torch


def nufft(
    image,
    coord=None,
    shape=None,
    nufft_plan=None,
    basis=None,
    device=-1,
    threadsperblock=128,
    width=4,
    oversamp=1.25,
):
    """
    N-dimensional Non-Uniform Fast Fourier Transform.

    Parameters
    ----------
    image : torch.Tensor
        Input image of shape ``(..., ncontrasts, ny, nx)`` (2D)
        or ``(..., ncontrasts, nz, ny, nx)`` (3D).
    coord : torch.Tensor
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
        Coordinates must be normalized between ``(-0.5 * shape, 0.5 * shape)``.
    shape : int | Iterable[int], optional
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
        The default is ``None`` (grid size equals to input data size, i.e. ``osf = 1``).
    basis : torch.Tensor, optional
        Low rank subspace projection operator
        of shape ``(ncontrasts, ncoeff)``; can be ``None``. The default is ``None``.
    device : str, optional
        Computational device (``cpu`` or ``cuda:n``, with ``n=0, 1,...nGPUs``).
        The default is ``cpu``.
    threadsperblock : int
        CUDA blocks size (for GPU only). The default is ``128``.
    width : int | Iterable[int], optional
        Interpolation kernel full-width of shape ``(ndim,)``.
        If scalar, isotropic kernel is assumed.
        The default is ``4``.
    oversamp : float | Iterable[float], optional
        Grid oversampling factor of shape ``(ndim,)``.
        If scalar, isotropic oversampling is assumed.
        The default is ``1.25``.

    Returns
    -------
    kspace : torch.Tensor
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
    # get number of dimensions
    ndim = coord.shape[-1]

    # get shape if not provided
    if shape is None:
        shape = image.shape[-ndim:]

    # plan interpolator
    nufft_plan = _nufft.plan_nufft(coord, shape, width, oversamp, device)

    # detect backend and device
    ibackend = _subroutines.get_backend(image)
    idevice = _subroutines.get_device(image)

    # make sure datatype is correct
    if image.dtype in (ibackend.float16, ibackend.float32, ibackend.float64):
        image = _subroutines.astype(image, ibackend.float32)
    else:
        image = _subroutines.astype(image, ibackend.complex64)

    # handle basis
    if basis is not None:
        # make sure datatype is correct
        if image.dtype in (ibackend.float16, ibackend.float32, ibackend.float64):
            basis = _subroutines.astype(basis, ibackend.float32)
        else:
            basis = _subroutines.astype(basis, ibackend.complex64)

    # cast to device if necessary
    if device is not None:
        nufft_plan.to(device)

    pass


# %% local subroutines
if _subroutines.pytorch_enabled:
    pass
