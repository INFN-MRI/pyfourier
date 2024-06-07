"""Centered fft subroutines."""

__all__ = ["fft", "ifft", "fftshift", "ifftshift"]

from . import _utils

if _utils.mklfft_enabled:
    import mkl_fft


def fft(input, axes=None, norm="ortho", centered=True):
    """
    Centered Fast Fourier Transform.

    Adapted from [1].

    Parameters
    ----------
    input : ArrayLike
        Input signal.
    axes : Sequence[int], optional
        Axes over which to compute the FFT.
        If not specified, apply FFT over all the axes.
    norm : str, optional
        FFT normalization. The default is ``ortho``.
    centered : bool, optional
        FFT centering. The default is ``True``.

    Returns
    -------
    output : ArrayLike
        Output signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pyfourier as pyft

    First, create test image:

    >>> image = np.zeros(32, 32, dtype=np.complex64)
    >>> image = image[16, 16] = 1.0

    We now perform a 2D FFT:

    >>> kspace = pyft.fft(image)

    We can visualize the data:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 2)
    >>> im = ax[0].imshow(abs(image))
    >>> ax[0].set_title("Image", color="orangered", fontweight="bold")
    >>> ax[0].axis("off")
    >>> ax[0].set_alpha(0.0)
    >>> fig.colorbar(im, ax=ax[0], shrink=0.5)
    >>> ksp = ax[1].imshow(abs(kspace))
    >>> ax[1].set_title("k-Space", color="orangered", fontweight="bold")
    >>> ax[1].axis("off")
    >>> ax[1].set_alpha(0.0)
    >>> fig.colorbar(ksp, ax=ax[1], shrink=0.5)
    >>> plt.show()

    References
    ----------
    [1] https://github.com/mikgroup/sigpy

    """
    backend = _utils.get_backend(input)
    ax = _normalize_axes(axes, input.ndim)

    if backend.__name__ == "torch":
        if centered:
            output = backend.fft.fftshift(
                backend.fft.fftn(
                    backend.fft.ifftshift(input, dim=ax), dim=ax, norm=norm
                ),
                dim=ax,
            )
        else:
            output = backend.fft.fftn(input, dim=ax, norm=norm)
    elif backend.__name__ == "numpy" and _utils.mklfft_enabled():
        if centered:
            output = backend.fft.fftshift(
                mkl_fft.fftn(
                    backend.fft.ifftshift(input, axes=ax), axes=ax, norm=norm
                ),
                axes=ax,
            )
        else:
            output = backend.fft.fftn(input, axes=ax, norm=norm)
    else:
        if centered:
            output = backend.fft.fftshift(
                backend.fft.fftn(
                    backend.fft.ifftshift(input, axes=ax), axes=ax, norm=norm
                ),
                axes=ax,
            )
        else:
            output = backend.fft.fftn(input, axes=ax, norm=norm)

    return output


def ifft(input, axes=None, norm="ortho", centered=True):
    """
    Centered inverse Fast Fourier Transform.

    Adapted from [1].

    Parameters
    ----------
    input :  ArrayLike
        Input signal.
    axes : Sequence[int]
        Axes over which to compute the iFFT.
        If not specified, apply iFFT over all the axes.
    norm : str, optional
        FFT normalization. The default is ``ortho``.
    centered : bool, optional
        FFT centering. The default is ``True``.

    Returns
    -------
    output :  ArrayLike
        Output signal.

    Examples
    --------
    >>> import numpy as np
    >>> import pyfourier as pyft

    First, create test image:

    >>> kspace = np.ones(32, 32, dtype=np.complex64)

    We now perform a 2D iFFT:

    >>> image = pyft.ifft(kspace)

    We can visualize the data:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 2)
    >>> ksp = ax[1].imshow(abs(kspace))
    >>> ax[0].set_title("k-Space", color="orangered", fontweight="bold")
    >>> ax[0].axis("off")
    >>> ax[0].set_alpha(0.0)
    >>> fig.colorbar(ksp, ax=ax[0], shrink=0.5)
    >>> im = ax[0].imshow(abs(image))
    >>> ax[1].set_title("Image", color="orangered", fontweight="bold")
    >>> ax[1].axis("off")
    >>> ax[1].set_alpha(0.0)
    >>> fig.colorbar(im, ax=ax[1], shrink=0.5)
    >>> plt.show()

    References
    ----------
    [1] https://github.com/mikgroup/sigpy

    """
    backend = _utils.get_backend(input)
    ax = _normalize_axes(axes, input.ndim)
    if backend.__name__ == "torch":
        if centered:
            output = backend.fft.fftshift(
                backend.fft.ifftn(
                    backend.fft.ifftshift(input, dim=ax), dim=ax, norm=norm
                ),
                dim=ax,
            )
        else:
            output = backend.fft.ifftn(input, dim=ax, norm=norm)
    elif backend.__name__ == "numpy" and _utils.mklfft_enabled():
        if centered:
            output = backend.fft.fftshift(
                mkl_fft.ifftn(
                    backend.fft.ifftshift(input, axes=ax), axis=ax, norm=norm
                ),
                axes=ax,
            )
        else:
            output = backend.fft.ifftn(input, axis=ax, norm=norm)
    else:
        if centered:
            output = backend.fft.fftshift(
                backend.fft.ifftn(
                    backend.fft.ifftshift(input, axes=ax), axis=ax, norm=norm
                ),
                axes=ax,
            )
        else:
            output = backend.fft.ifftn(input, axis=ax, norm=norm)

    return output


def fftshift(input, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axes : Sequence[int], optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : ArrayLike
        The shifted array.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2., ..., -3., -2., -1.])
    >>> np.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Shift the zero-frequency component only along the second axis:

    >>> import numpy as np
    >>> import pyfourier as pyft
    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> pyft.fftshift(freqs, axes=(1,))
    array([[ 2.,  0.,  1.],
           [-4.,  3.,  4.],
           [-1., -3., -2.]])

    """
    backend = _utils.get_backend(input)
    ax = _normalize_axes(axes, input.ndim)
    if backend.__name__ == "torch":
        return backend.fft.fftshift(input, dim=ax)
    else:
        return backend.fft.fftshift(input, axes=ax)


def ifftshift(x, axes=None):
    """
    Invert `fftshift`.

    Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    axes : Sequence[int], optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ArrayLike
        The shifted array.

    See Also
    --------
    fftshift : Shift zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> import pyfourier as pyft
    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> pyft.ifftshift(pyft.fftshift(freqs))
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])

    """
    backend = _utils.get_backend(input)
    ax = _normalize_axes(axes, input.ndim)
    if backend.__name__ == "torch":
        return backend.fft.ifftshift(input, dim=ax)
    else:
        return backend.fft.ifftshift(input, axes=ax)


# %% local subroutines
def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))
