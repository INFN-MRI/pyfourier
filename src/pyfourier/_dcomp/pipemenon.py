"""Estimation of the density compensation via iterative NUFFT."""

__all__ = ["pipe_menon_dcf"]

from .. import _subroutines
from .. import _nufft


def pipe_menon_dcf(
    coord, shape, niter=30, threadsperblock=128, device=None, width=6, oversamp=1.5
):
    r"""
    Compute Pipe Menon density compensation factor.

    Perform the following iteration:

    .. math::

        w = \frac{w}{|G^H G w|}

    with :math:`G` as the gridding operator.

    Parameters
    ----------
    coord : ArrayLike
        K-space coordinates of shape ``(ncontrasts, nviews, nsamples, ndims)``.
    shape : int | Sequence[int], optional
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    niter : int, optional
        Number of algorithm iterations. The default is ``30``.
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
    wi : ArrayLike
        DCF array of shape ``(..., nshots, npts)`` .

    References
    ----------
    Pipe, James G., and Padmanabhan Menon.
    Sampling Density Compensation in MRI:
    Rationale and an Iterative Numerical Solution.
    Magnetic Resonance in Medicine 41, no. 1 (1999): 17986.


    """
    # detect backend and device
    backend = _subroutines.get_backend(coord)
    idevice = _subroutines.get_device(coord)

    # if not provided, use original device
    if device is None:
        device = idevice
    else:
        if isinstance(device, str):
            if device == "cpu":
                device = -1
            else:
                device = int(device.split(":")[-1])

    # cast to device if necessary
    coord = _subroutines.to_device(coord, device)

    # initialize weights
    w = _subroutines.ones(coord.shape[:-1], coord.dtype, device, backend)

    # plan NUFFT
    nufft_plan = _nufft.plan_nufft(coord, shape, width, oversamp)
    nufft_plan = nufft_plan.to(device)

    # update weights
    for it in range(niter):
        # compute gridding self-adjoint
        GHGw = _GHG(w, nufft_plan, device, threadsperblock)

        # normalize
        w = w / backend.abs(GHGw)

    # original device
    w = _subroutines.to_device(w, idevice)

    return w


# %% local subroutines
def _GHG(input, nufft_plan, device, threadsperblock):
    Gw = _nufft.nufft_adj(
        input,
        nufft_plan=nufft_plan,
        device=device,
        threadsperblock=threadsperblock,
        norm="ortho",
    )
    return _nufft.nufft(
        Gw,
        nufft_plan=nufft_plan,
        device=device,
        threadsperblock=threadsperblock,
        norm="ortho",
    )
