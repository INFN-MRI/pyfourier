"""Spiral trajectory generation."""

__all__ = ["spiral"]

from types import SimpleNamespace

import numpy as np

from .. import _dcomp

def spiral(ndim, fov, npix, ncontrasts, ninterleaves, R=None):
    """
    Initialize 2D / 3D spiral trajectory.

    Parameters
    ----------
    ndim : int
        Number of spatial dimension.
        Set to ``2`` for 2D Spiral, ``3``
        for 3D Spiral projection.
    fov : float
        Field of view in ``[mm]``.
    npix : int
        Matrix size.
    ncontrasts : int
        Number of contrast in the image.
    ninterleaves : int
        Number of interleaves to fully sample a k-space plane.
    R : int, optional
        In-plane acceleration. 
        If ``None``, set ``R = ninterleaves`` for
        2D acquisitions (``ndim == 2``) with multiple contrasts
        (``ncontrasts > 1``), i.e., single interleaf per contrast.
        In other cases (single contrast 2D acquisitions and single-/multi-contrast
        3D acquisitions), set ``R = 1`` (fully sampled disk).
        The default is ``None``.

    Returns
    -------
    output : SimpleNamespace
        Structure with the following fields:   
        
        *k: k-space trajectory of shape ``(ncontrasts, ninterleaves, npts, ndim)``
            normalized between ``-npix // 2`` and ``npix // 2``.
        *dcf: density compensation factors of shape ``(ncontrasts, ninterleaves, npts)``.
        *shape: Matrix size ``(npix, npix)`` (2D) or ``(npix, npix, npix)`` (3D).
        
    """
    # rescale inputs
    fov *= 1e-3 # mm -> m
    
    # default undersampling
    if R is None:
        if ndim == 2 and ncontrasts > 1:
            R = ninterleaves
        else:
            R = 1
            
    # generate golden angle angular rotation across frames
    phi = _golden_angle_list(ncontrasts)
    
    # generate spiral interleaf
    karm = _spiral(fov, npix, R)
    
    # rotate
    if ndim == 2: # 2D spiral
        k = _2d_rotation(karm, phi)
    else: # 3D spiral
        k = _3d_rotation(karm, phi)
        
    # compute dcf
    dcf = _dcomp.voronoi(k, nshots=ninterleaves, fullspoke=True, rotation_axis_3D="x")
    
    # normalize trajectory
    kabs = (k**2).sum(axis=-1)**0.5
    k = 0.5 * k / kabs[..., None] * npix
    
    # initialize output
    output = SimpleNamespace(k=k, shape=[npix] * ndim, dcf=dcf)
    
    return output

# %% local utils
def _spiral(fov, N, ninterleaves, R=1.0, f_sampling=1.0, alpha=1.0, gm=22, sm=250, gamma=2.678e8):
    res = fov / N

    lam = 0.5 / res  # in m**(-1)
    n = 1 / (1 - (1 - ninterleaves * R / fov / lam) ** (1 / alpha))
    w = 2 * np.pi * n
    Tea = lam * w / gamma / gm / (alpha + 1)  # in s
    Tes = np.sqrt(lam * w**2 / sm / gamma) / (alpha / 2 + 1)  # in s
    Ts2a = (
        Tes ** ((alpha + 1) / (alpha / 2 + 1))
        * (alpha / 2 + 1)
        / Tea
        / (alpha + 1)
    ) ** (
        1 + 2 / alpha
    )  # in s

    if Ts2a < Tes:
        tautrans = (Ts2a / Tes) ** (1 / (alpha / 2 + 1))

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * (
                t <= Ts2a
            ) + ((t - Ts2a) / Tea + tautrans ** (alpha + 1)) ** (
                1 / (alpha + 1)
            ) * (
                t > Ts2a
            ) * (
                t <= Tea
            ) * (
                Tes >= Ts2a
            )

        Tend = Tea
    else:

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * (t <= Tes)

        Tend = Tes

    def k(t):
        return lam * tau(t) ** alpha * np.exp(w * tau(t) * 1j)

    dt = Tea * 1e-4  # in s

    Dt = dt * f_sampling / fov / abs(k(Tea) - k(Tea - dt))  # in s

    t = np.linspace(0, Tend, int(Tend / Dt))
    kt = k(t)  # in rad

    # generating cloned interleaves
    k = kt
    for i in range(1, ninterleaves):
        k = np.stack((k, kt[0:] * np.exp(2 * np.pi * 1j * i / ninterleaves)), axis=0)

    k = np.stack((np.real(k), np.imag(k)), axis=-1)

    return k

def _golden_angle_list(length):
    golden_ratio = (np.sqrt(5.0) + 1.0) / 2.0
    conj_golden_ratio = 1 / golden_ratio

    m = np.arange(length, dtype=np.float32) * conj_golden_ratio
    phi = (180 * m) % 360

    return phi

def _2d_rotation(input, phi):
    coord_in = input.T
    coord_out = np.zeros(
        (*coord_in.shape, len(phi)), dtype=coord_in.dtype)
    coord_out[0] = coord_in[0] * np.cos(phi) - coord_in[1] * np.sin(phi)
    coord_out[1] = coord_in[0] * np.sin(phi) + coord_in[1] * np.cos(phi)

    return coord_out.T

def _3d_rotation(coord_in, phi):
    coord_in = input.T
    coord_out = np.zeros(
        (3, *coord_in.shape[1:], len(phi)), dtype=coord_in.dtype)
    coord_out[0] = coord_in[0]
    coord_out[1] = coord_in[1] * np.cos(phi) - coord_in[2] * np.sin(phi)
    coord_out[2] = coord_in[1] * np.sin(phi) + coord_in[2] * np.cos(phi)

    return coord_out

