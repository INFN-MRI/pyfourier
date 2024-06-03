"""
Pre-interpolation routines.

These routines are used to pre-interpolate k-space data
on a Cartesian grid before reconstructing, casting the NUFFT problem
to a sparse FFT problem. Useful for iterative reconstruction, 
where pre-interpolation can be performed once and cheap sparse FFT  / iFFT
(potentially using Toeplitz embedding) can be used through iterations.

Currently, only GROG interpolation is provided (for multi-coil MR).


"""

from ._grog import *  # noqa

__all__ = ["grog_interp"]  # noqa
