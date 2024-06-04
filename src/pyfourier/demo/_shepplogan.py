"""Multicontrast, multicoil Shepp-Logan generation."""

__all__ = ["shepp_logan"]

import numpy as np

from . import _birdcage
from . import _tse


def shepp_logan(ndim, npix, ncontrasts, ncoils, ncoeff):
    """
    Create low-rank subspace coefficients for a Shepp-Logan phantom.

    Assume Spin-Echo acquisition with infinite TR.

    Parameters
    ----------
    ndim : int
        Number of spatial dimension.
        Set to ``2`` for 2D Spiral, ``3``
        for 3D Spiral projection.
    npix : int
        Matrix size.
    ncontrasts : int
        Number of T2 contrasts (echoes) in the train.
    ncoils : int
        Number of coils for multi-channel acquisition
    ncoeff : int
        Subspace basis size.

    Returns
    -------
        tensor: low-rank subspace coefficient for the Shepp-Logan phantom.
    """
    # shape
    shape = ndim * [npix]
    center = int(npix // 2)

    # get tissue segmentation mask
    discrete_model = np.round(_shepp_logan(shape, dtype=np.float32))

    # generate coil
    if ncoils > 1:
        smap = _birdcage.birdcage_maps([ncoils] + shape)
        caldata = smap * discrete_model
        caldata = np.fft.fftshift(
            np.fft.fftn(
                np.fft.fftshift(caldata, axes=list(range(-ndim, 0))),
                axes=list(range(-ndim, 0)),
            ),
            axes=list(range(-ndim, 0)),
        )

        if ndim == 2:
            caldata = caldata[..., center - 12 : center + 12, center - 12 : center + 12]
        else:
            caldata = caldata[
                ...,
                center - 12 : center + 12,
                center - 12 : center + 12,
                center - 12 : center + 12,
            ]
    else:
        caldata = None

    # single contrast
    if ncontrasts == 1:
        if ncoils == 1:
            return discrete_model, None, None
        else:
            return smap * discrete_model, caldata, None

    # generate multicontrast data
    discrete_model = discrete_model.astype(np.int32)

    # collapse vessels rois, csf rois and re-order indexes
    discrete_model[discrete_model == 1] = 1
    discrete_model[discrete_model == 2] = 1
    discrete_model[discrete_model == 3] = 1
    discrete_model[discrete_model == 4] = 1
    discrete_model[discrete_model == 5] = 2
    discrete_model[discrete_model == 6] = 2
    discrete_model[discrete_model == 7] = 3
    discrete_model[discrete_model == 8] = 3
    discrete_model = np.asarray(discrete_model)  # pylint: disable=no-member

    # assign relaxation values to different regions values
    t2_wm = 70
    t2_gm = 83
    t2_csf = 329

    # collect in a single array
    t2 = np.asarray([t2_wm, t2_gm, t2_csf])

    # simulate
    te = np.linspace(1, 300, ncontrasts)
    sig = np.exp(-te[:, None] / t2[None, :])
    sig = np.asarray(sig, dtype=np.float32)

    # get basis
    basis, _ = _tse.create_subspace_basis(ncontrasts, ncoeff)
    sig = basis @ sig

    # assign to tissue mask to create output image
    output = np.zeros((ncoeff, *shape), dtype=np.float32)

    for n in range(ncoeff):
        output[n, discrete_model == 1] = sig[n, 0]
        output[n, discrete_model == 2] = sig[n, 1]
        output[n, discrete_model == 3] = sig[n, 2]

    if ncoils == 1:
        return output, None, basis
    else:
        return smap[:, None, ...] * output, caldata, basis

    return output


# %% local utils
def _shepp_logan(shape, dtype=np.complex64):
    return _phantom(shape, sl_amps, sl_scales, sl_offsets, sl_angles, dtype)


sl_amps = [8, 7, 6, 5, 4, 3, 2, 1]

sl_scales = [
    [0.6900, 0.920, 0.810],  # white big
    [0.6624, 0.874, 0.780],  # gray big
    [0.1100, 0.310, 0.220],  # right black
    [0.1600, 0.410, 0.280],  # left black
    [0.2100, 0.250, 0.410],  # gray center blob
    [0.0460, 0.046, 0.050],  # left small dot
    [0.0230, 0.023, 0.020],  # mid small dot
    [0.0230, 0.023, 0.020],
]


def _phantom(shape, amps, scales, offsets, angles, dtype):
    if len(shape) == 2:
        ndim = 2
        shape = (1, shape[-2], shape[-1])
    elif len(shape) == 3:
        ndim = 3
    else:
        raise ValueError("Incorrect dimension")

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[
        -(shape[-3] // 2) : ((shape[-3] + 1) // 2),
        -(shape[-2] // 2) : ((shape[-2] + 1) // 2),
        -(shape[-1] // 2) : ((shape[-1] + 1) // 2),
    ]

    coords = np.stack(
        (
            x.ravel() / shape[-1] * 2,
            y.ravel() / shape[-2] * 2,
            z.ravel() / shape[-3] * 2,
        )
    )

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):
        _ellipsoid(amp, scale, offset, angle, coords, out)
    if ndim == 2:
        return out[0, :, :]
    else:
        return out


def _ellipsoid(amp, scale, offset, angle, coords, out):
    R = _rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / np.reshape(
        scale, (3, 1)
    )

    r2 = np.sum(coords**2, axis=0).reshape(out.shape)

    out[r2 <= 1] = amp


def _rotation_matrix(angle):
    cphi = np.cos(np.radians(angle[0]))
    sphi = np.sin(np.radians(angle[0]))
    ctheta = np.cos(np.radians(angle[1]))
    stheta = np.sin(np.radians(angle[1]))
    cpsi = np.cos(np.radians(angle[2]))
    spsi = np.sin(np.radians(angle[2]))
    alpha = [
        [
            cpsi * cphi - ctheta * sphi * spsi,
            cpsi * sphi + ctheta * cphi * spsi,
            spsi * stheta,
        ],
        [
            -spsi * cphi - ctheta * sphi * cpsi,
            -spsi * sphi + ctheta * cphi * cpsi,
            cpsi * stheta,
        ],
        [stheta * sphi, -stheta * cphi, ctheta],
    ]
    return np.array(alpha)


sl_offsets = [
    [0.0, 0.0, 0],
    [0.0, -0.0184, 0],
    [0.22, 0.0, 0],
    [-0.22, 0.0, 0],
    [0.0, 0.35, -0.15],
    [-0.08, -0.605, 0],
    [0.0, -0.606, 0],
    [0.06, -0.605, 0],
]

sl_angles = [
    [0, 0, 0],
    [0, 0, 0],
    [-18, 0, 10],
    [18, 0, 10],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]
