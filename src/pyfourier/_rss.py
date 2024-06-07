"""Multi-coil combination sub-routines."""

__all__ = ["rss"]

def rss(input, axis=None, keepdim=False):
    """
    Perform root sum-of-squares combination of a signal.

    Parameters
    ----------
    input : ArrayLike
        Input signal (real- or complex-valued).
    axis : int, optional
        Combination axis.  If ``None``, combine along all dimensions,
        reducing to a scalar. The default is ``None``.
    keepdim : bool, optional
        If ``True``, maintain the combined axis as a singleton dimension.
        The default is ``False`` (squeeze the combination axis).

    Returns
    -------
    output : ArrayLike
        Real-valued output combined signal.

    Examples
    --------
    >>> import numpy as np 
    >>> import pyfourier as pyft

    Generate an example signal:

    >>> signal = np.ones(10, 4, 4)

    We can compute the rss of all signal elements as:

    >>> output = np.rss(signal)
    >>> output
    12.6491

    We can compute rss along the first axis only (i.e., coil combination) as:

    >>> output = pyft.rss(signal, axis=0)
    >>> output.shape
    (4, 4)

    The axis can be explicitly maintained instead of squeezed as

    >>> output = pyft.rss(signal, axis=0, keepdim=True)
    >>> output.shape
    (1, 4, 4)


    """
    if axis is None:
        return (input * input.conj()).sum() ** 0.5

    output = (input * input.conj()).sum(axis=axis, keepdim=keepdim) ** 0.5

    return output