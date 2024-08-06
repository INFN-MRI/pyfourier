"""PyFourier: Python package for dense, sparse and Non-Cartesian FFT."""

# Read version from installed package
from importlib.metadata import version

__version__ = version("pyfourier")

from . import demo  # noqa
from ._fft import fftn, ifftn, plan_fft  # noqa
from ._sparsefft import sparse_fftn, sparse_ifftn, plan_spfft  # noqa
from ._nufft import nufft, nufft_adj, plan_nufft  # noqa
from ._gram import gram, plan_gram  # noqa
from ._sparsegram import sparse_gram, plan_spgram  # noqa
from ._subroutines import fft, ifft  # noqa
from ._dcomp import pipemenon, voronoi  # noqa
from ._rss import rss  # noqa

__all__ = []

# %% Forward and adjoint Fourier operations for MR imaging
__all__.extend(["fftn", "ifftn"])
__all__.extend(["sparse_fftn", "sparse_ifftn"])
__all__.extend(["nufft", "nufft_adj"])

# %% FFT plan constructor
__all__.extend(["plan_fft"])
__all__.extend(["plan_spfft"])
__all__.extend(["plan_nufft"])

# %% Gram (dense, sparse and Non-Cartesian)
__all__.extend(["plan_gram"])
__all__.extend(["plan_spgram"])
__all__.extend(["sparse_gram"])
__all__.extend(["gram"])

# %% Utilities: centered FFT along arbitrary axis
__all__.extend(["fft", "ifft"])

# %% Utilities: density compensation
__all__.extend(["voronoi"])
__all__.extend(["pipemenon"])

# %% Utilities: multi-coil combination
__all__.extend(["rss"])
