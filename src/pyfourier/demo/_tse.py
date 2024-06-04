"""Low rank subspace basis creation for TSE sequence."""

__all__ = ["create_subspace_basis"]

import numpy as np

def create_subspace_basis(nechoes, ncoeff):
    """ 
    Generate low-rank temporal subspace basis for a Spin-Echo acquisition.

    Parameters
    ----------
    nechoes : int
        Number of echoes in the train.
    ncoeff : int 
        Subspace basis size.

    Returns
    -------
    basis : NDArray
        Low-rank subspace basis.
    sig : NDArray 
        Signal ensemble used to compute basis via SVD.
        
    """
    # assume T2 spin echo
    t2 = np.linspace(1, 329, 300)

    # create echos
    te = np.linspace(1, 300, nechoes)

    # simulate signals (analytically)
    sig = np.exp(-te[None, :] / t2[:, None])

    # get basis
    _, _, basis = np.linalg.svd(sig, full_matrices=False)

    # select subspace
    basis = basis[:ncoeff, :].T

    return basis, sig