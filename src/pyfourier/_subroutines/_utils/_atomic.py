"""Atomic update subroutines."""

__all__ = ["_update"]

from . import _backend

# detect GPU
gpu_available, gpu_backend = _backend.detect_gpu_backend()

_update = {}

if gpu_available and gpu_backend == "numba":
    
    from numba import cuda
    
    @cuda.jit(device=True, inline=True)  # pragma:  no cover
    def _update_nb_real(output, index, value):
        cuda.atomic.add(output, index, value)
    
    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _update_nb_complex(output, index, value):
        cuda.atomic.add(output.real, index, value.real)
        cuda.atomic.add(output.imag, index, value.imag)
        
    _update = {False: _update_nb_real, True: _update_nb_complex}
    
if gpu_available and gpu_backend == "cupy":
    
    from cupyx import jit
    
    @jit.rawkernel(device=True)  # pragma:  no cover
    def _update_cupy_real(output, index, value):
        jit.atomic_add(output, index, value)
    
    @jit.rawkernel(device=True)  # pragma:  no cover
    def _update_cupy_complex(output, index, value):
        jit.atomic_add(output.real, index, value.real)
        jit.atomic_add(output.imag, index, value.imag)
        
    _update = {False: _update_cupy_real, True: _update_cupy_complex}