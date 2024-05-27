"""2D zero-filling subroutines."""

__all__ = ["_zerofill"]

import numba as nb

from ... import _utils

# detect GPU
gpu_available, gpu_backend = _utils.detect_gpu_backend()

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _zerofill_nb(dense_data, sparse_data, index):  # noqa
    # get sizes
    nframes, batch_size, _, _ = dense_data.shape
    npts = sparse_data.shape[-1]

    # unpack interpolator
    yindex, xindex = index

    # parallelize over frames and batches
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):
            idy = yindex[frame, point]
            idx = xindex[frame, point]
            dense_data[frame, batch, idy, idx] += sparse_data[frame, batch, point]
    
_zerofill = {"cpu": {False: _zerofill_nb, True: _zerofill_nb}}
                    
# %% GPU
if gpu_available and gpu_backend == "numba":
    
    from numba import cuda
    
    def _get_zerofill_nbcuda(is_complex):
        
        _update = _utils._update[is_complex]
    
        @cuda.jit(fastmath=True)  # pragma: no cover
        def _zerofill_nbcuda(
            
        return _zerofill_nbcuda
  
    _zerofill["gpu"] = {False: _get_zerofill_nbcuda(False), True: _get_zerofill_nbcuda(True)}
    
if gpu_available and gpu_backend == "cupy":
    
    from cupyx import jit
    
    def _get_zerofill_cupy(is_complex):
        
        _update = _utils._update[is_complex]
    
        @jit.rawkernel()  # pragma: no cover
        def _zerofill_cupy(
             
        return _zerofill_cupy
  
    _zerofill["gpu"] = {False: _get_zerofill_cupy(False), True: _get_zerofill_cupy(True)}
