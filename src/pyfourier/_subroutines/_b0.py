"""Utils for B0 informed reconstruction."""

__all__ = ["mri_exp_approx"]

import math

from . import _utils


def mri_exp_approx(zmap, t, lseg=6, bins=(40, 40), mask=None):  # , toeplitz=False):
    r"""
    Create B [L, nt] and Ct [L, (nz), ny, nx] matrices to approximate exp(-2i*pi*b0*t) [nt, (nz), ny, nx].

    From Sigpy: https://github.com/mikgroup/sigpy and MIRT (mri_exp_approx.m): https://web.eecs.umich.edu/~fessler/code/

    Parameters
    ----------
    zmap : npt.ArrayLike
        Rate map defined as ``zmap = R2*_map + 1j * B0_map``.
        ``*_map`` and ``t`` should have reciprocal units.
        If ``zmap`` is real, assume ``zmap = B0_map``.
        Expected shape is ``(nz, ny, nx)``.
    t : npt.ArrayLike
        Readout time in ``[s]`` of shape ``(npts,)``.
    lseg : int, optional
        Number of time segments. The default is ``6``.
    bins : int || tuple(int), optional
        Number of histogram bins to use for ``(B0, T2)``. The default is ``(40, 40)``
        If it is a scalar, assume ``bins = (bins, 40)``.
    mask : npt.ArrayLike, optional
        Boolean mask to avoid histogram of background values.
        The default is ``None`` (use the whole map).

    Returns
    -------
    b : npt.ArrayLike
        Temporal interpolator of shape ``(lseg, npts)``.
    ct : npt.ArrayLike
        Off-resonance phase map at each time segment center of shape
        ``(lseg, *zmap.shape)``.

    """
    # get backend and device
    backend = _utils.get_backend(input)
    device = _utils.get_device(input)

    # default
    if isinstance(bins, (list, tuple)) is False:
        bins = (bins, 5)

    # set acorr
    # acorr = toeplitz

    # transform to list
    bins = list(bins)

    # get field map
    if _utils.isreal(zmap).all().item():
        r2star = None
        b0 = zmap
        zmap = 0.0 + 1j * b0
    else:
        r2star = zmap.real
        b0 = zmap.imag

    # default mask
    if mask is None:
        mask = backend.ones_like(zmap, dtype=bool)

    # Hz to radians / s
    zmap = 2 * math.pi * zmap

    # create histograms
    if r2star is not None:
        z = zmap[mask].ravel()
        z = backend.stack((z.imag, z.real), axis=1)
        hk, ze = backend.histogramdd(z, bins=bins)
        ze = list(ze)

        # get bin centers
        zc = [e[1:] - e[1] / 2 for e in ze]

        # autocorr of histogram, for Toeplitz
        # if acorr:
        #     hk = _corr2d(hk, hk)
        #     zc[0] = backend.arange(-(bins[0] - 1), bins[0]) * (zc[0][1] - zc[0][0])
        #     zc[1] = backend.linspace(2 * zc[1].min(), 2 * zc[1].max(), 2 * bins[1] - 1)

        zk = _outer_sum(1j * zc[0], zc[1])  # [K1 K2]
    else:
        z = zmap[mask].ravel()
        hk, ze = backend.histogram(z, bins=bins[0])

        # get bin centers
        zc = ze[1:] - ze[1] / 2

        # complexify
        zk = 0 + 1j * zc  # [K 1]

        # autocorr of histogram, for Toeplitz
        # if acorr:
        #     hk = _corr1d(hk, hk)
        #     zk = backend.arange(-(bins[0] - 1), bins[0]) * zk[1] - zk[0]

    # flatten histogram values and centers
    hk = hk.flatten()
    zk = zk.flatten()

    # generate time for each segment
    tl = backend.linspace(0, lseg, lseg) / lseg * t[-1]  # time seg centers in [s]

    # complexify histogram and time
    hk = _utils.to_device(_utils.astype(hk, zk.dtype), device)
    tl = _utils.to_device(_utils.astype(tl, zk.dtype), device)
    t = _utils.to_device(_utils.astype(t, zk.dtype), device)

    # prepare for basis calculation
    ch = backend.exp(-tl[:, None, ...] @ zk[None, ...])
    w = backend.diag(hk**0.5)
    p = backend.linalg.pinv(w @ ch.t()) @ w

    # actual temporal basis calculation
    b = p @ backend.exp(zk[:, None, ...] * t[None, ...])

    # get spatial coeffs
    ct = backend.exp(-tl * zmap[..., None])
    ct = ct[None, ...].swapaxes(0, -1)[..., 0]  # (..., lseg) -> (lseg, ...)

    # clean-up of spatial coeffs
    ct = backend.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)

    return b, ct


# %% utils
# def _corr1d(a, b):
#     if backend.__name__ == "torch":
#         a1 = a.unsqueeze(0).unsqueeze(0)
#         b1 = b.unsqueeze(0).unsqueeze(0)
#         padsize = b1.shape[-1] - 1
#         return torch.nn.functional.conv1d(a1, b1, padding=padsize)[0][0]
#     else:


# def _corr2d(a, b):
#     a1 = a.unsqueeze(0).unsqueeze(0)
#     b1 = b.unsqueeze(0).unsqueeze(0)
#     padsize = (b1.shape[-2] - 1, b1.shape[-1] - 1)
#     return torch.nn.functional.conv2d(a1, b1, padding=padsize)[0][0]


def _outer_sum(xx, yy):
    xx = xx[:, None, ...]  # Add a singleton dimension at axis 1
    yy = yy[None, ...]  # Add a singleton dimension at axis 0
    ss = xx + yy  # Compute the outer sum
    return ss
