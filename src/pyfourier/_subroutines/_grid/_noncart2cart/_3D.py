"""3D gridding subroutines."""

__all__ = ["_grid"]

import numba as nb

from ... import _utils

# detect GPU
gpu_available, gpu_backend = _utils.detect_gpu_backend()

@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _grid_nb(cart_data, noncart_data, interp_value, interp_index):  # noqa
    # get sizes
    nframes, batch_size, _, _, _ = cart_data.shape
    npts = noncart_data.shape[-1]

    # unpack interpolator
    zindex, yindex, xindex = interp_index
    zvalue, yvalue, xvalue = interp_value

    # get interpolator width
    zwidth = zindex.shape[-1]
    ywidth = yindex.shape[-1]
    xwidth = xindex.shape[-1]

    # parallelize over frames and batches
    for i in nb.prange(nframes * batch_size):  # pylint: disable=not-an-iterable
        # get current frame and batch index
        frame = i // batch_size
        batch = i % batch_size

        # iterate over non-cartesian point of current frame/batch
        for point in range(npts):
            # spread data within kernel radius
            for i_z in range(zwidth):
                idz = zindex[frame, point, i_z]
                valz = zvalue[frame, point, i_z]

                for i_y in range(ywidth):
                    idy = yindex[frame, point, i_y]
                    valy = valz * yvalue[frame, point, i_y]

                    for i_x in range(xwidth):
                        idx = xindex[frame, point, i_x]
                        val = valy * xvalue[frame, point, i_x]

                        cart_data[frame, batch, idz, idy, idx] += (
                            val * noncart_data[frame, batch, point]
                        )
    
_grid = {"cpu": {False: _grid_nb, True:_grid_nb}}
                    
# %% GPU
if gpu_available and gpu_backend == "numba":
    
    from numba import cuda
    
    def _get_grid_nbcuda(is_complex):
        
        _update = _utils._update[is_complex]
    
        @cuda.jit(fastmath=True)  # pragma: no cover
        def _grid_nbcuda(cart_data, noncart_data, interp_value, interp_index):
            # get sizes
            nframes, batch_size, _, _, _ = cart_data.shape
            npts = noncart_data.shape[-1]

            # unpack interpolator
            zindex, yindex, xindex = interp_index
            zvalue, yvalue, xvalue = interp_value

            # get interpolator width
            zwidth = zindex.shape[-1]
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = cuda.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes * batch_size * npts:
                # get current frame and k-space index
                frame = i // (batch_size * npts)
                tmp = i % (batch_size * npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_z in range(zwidth):
                    idz = zindex[frame, point, i_z]
                    valz = zvalue[frame, point, i_z]

                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = valz * yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            _update(
                                cart_data,
                                (frame, batch, idz, idy, idx),
                                val * noncart_data[frame, batch, point],
                            )
                        
        return _grid_nbcuda
  
    _grid["gpu"] = {False: _get_grid_nbcuda(False), True: _get_grid_nbcuda(True)}
    
if gpu_available and gpu_backend == "cupy":
    
    from cupyx import jit
    
    def _get_grid_cupy(is_complex):
        
        _update = _utils._update[is_complex]
    
        @jit.rawkernel()  # pragma: no cover
        def _grid_cupy(cart_data, noncart_data, interp_value, interp_index):
            # get sizes
            nframes, batch_size, _, _, _ = cart_data.shape
            npts = noncart_data.shape[-1]

            # unpack interpolator
            zindex, yindex, xindex = interp_index
            zvalue, yvalue, xvalue = interp_value

            # get interpolator width
            zwidth = zindex.shape[-1]
            ywidth = yindex.shape[-1]
            xwidth = xindex.shape[-1]

            # parallelize over frames, batches and k-space points
            i = jit.grid(1)  # pylint: disable=too-many-function-args
            if i < nframes * batch_size * npts:
                # get current frame and k-space index
                frame = i // (batch_size * npts)
                tmp = i % (batch_size * npts)
                batch = tmp // npts
                point = tmp % npts

                # spread data within kernel radius
                for i_z in range(zwidth):
                    idz = zindex[frame, point, i_z]
                    valz = zvalue[frame, point, i_z]

                    for i_y in range(ywidth):
                        idy = yindex[frame, point, i_y]
                        valy = valz * yvalue[frame, point, i_y]

                        for i_x in range(xwidth):
                            idx = xindex[frame, point, i_x]
                            val = valy * xvalue[frame, point, i_x]

                            _update(
                                cart_data,
                                (frame, batch, idz, idy, idx),
                                val * noncart_data[frame, batch, point],
                            )
                        
        return _grid_cupy
  
    _grid["gpu"] = {False: _get_grid_cupy(False), True: _get_grid_cupy(True)}
