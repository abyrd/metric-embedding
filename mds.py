#!run me with python

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State
from matplotlib.colors           import LinearSegmentedColormap
import pylab as pl 
import numpy as np
import time 
import geotools
import math
import pyproj
import sys

import pycuda.driver as cuda
import pycuda.autoinit as autoinit
from   pycuda.compiler import SourceModule

# Going from 2000 to 3000 doubles runtime on pdx. pixelwise would solve this. 500m range is fast!
# Maybe increase range over iterations.
WALK_RANGE  = 1500 
OBSTRUCTION = 1.4
WALK_SPEED  = 1.3 # m/sec
DIMENSIONS  = 3   

BIGINT = 100000

print 'Loading matrix...'
#npz = np.load('od_matrix.npz')
npz = np.load('od_matrix_trimet_linked.npz')
station_labels = npz['station_labels']
station_coords = npz['station_coords'] 
grid_latlon = npz['grid_latlon']
grid_dim    = npz['grid_dim']
matrix      = npz['matrix'].astype(np.int32)

# force OD matrix symmetry for test
# THIS was responsible for the coordinate drift!!!
# need to symmetrize it before copy to device
matrix = (matrix + matrix.T) / 2

matrix_gpu = cuda.mem_alloc(matrix.nbytes)
cuda.memcpy_htod(matrix_gpu, matrix)

station_coords_int = station_coords.round().astype(np.int32)
station_coords_gpu = cuda.mem_alloc(station_coords_int.nbytes)
cuda.memcpy_htod(station_coords_gpu, station_coords_int)

print "done."

min_x = 0
min_y = 0
max_x, max_y = grid_dim
n_gridpoints = int(max_x * max_y)
n_stations   = len(station_labels)


# precompute time matrix template
walk_cells = np.int32(WALK_RANGE / 100.0 / OBSTRUCTION)
times = np.arange(walk_cells * 2, dtype=np.float32) - walk_cells
times = np.sqrt((times**2)[:, np.newaxis] + (times ** 2)[np.newaxis, :]) * 100 * OBSTRUCTION
#times[times > WALK_RANGE] = np.inf
times /= WALK_SPEED
times[times > WALK_RANGE / OBSTRUCTION] = BIGINT
time_template = (times.round()).astype(np.int32)
time_template_gpu = cuda.mem_alloc(time_template.nbytes)
cuda.memcpy_htod(time_template_gpu, time_template)

# for non-gpu version
temp_time_template = np.empty_like(times)

# make and register custom color map

cdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.2,  0.0, 1.0),
                   (1.0,  0.1, 0.0)),

         'green': ((0.0,  0.0, 0.1),
                   (0.1,  0.9, 0.0),
                   (0.2,  0.0, 0.8),
                   (0.4,  0.1, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.1,  0.0, 1.0),
                   (0.2,  0.1, 0.0),
                   (1.0,  0.0, 0.0))}

mymap = LinearSegmentedColormap('mymap', cdict)
mymap.set_over( (1.0, 0.0, 1.0) )
#mymap.set_bad ( (0.5, 0.0, 0.5) )
mymap.set_bad ( (0.3, 0.3, 0.3) )
pl.plt.register_cmap(cmap=mymap)

grid   = np.zeros( n_gridpoints, dtype=np.float32 ) # dummy numpy array for sizing memory allocation
coords = np.random.random( (n_gridpoints, DIMENSIONS) ).astype(np.float32) # initialize coordinates to random
forces_gpu   = cuda.mem_alloc( coords.nbytes ) # float32
vectors_gpu  = cuda.mem_alloc( coords.nbytes ) # float32
err_gpu      = cuda.mem_alloc( grid.nbytes ) # for float32 cell error accumulation
tt_gpu       = cuda.mem_alloc( grid.nbytes ) # for int32 travel times
norms_gpu    = cuda.mem_alloc( grid.nbytes ) # float32
adjust_gpu   = cuda.mem_alloc( grid.nbytes ) # float32 
uvectors_gpu = cuda.mem_alloc( coords.nbytes ) # float32
coords_gpu   = cuda.mem_alloc( coords.nbytes ) # float32
cuda.memcpy_htod(coords_gpu, coords)

# times could be merged into forces kernel, if done by pixel not station.
# integrate kernel could be GPUArray operation; also helps clean up code by using GPUArrays.
mod = SourceModule("""
 
    __global__ void ttime (
        int origin_idx,
        int max_x,
        int max_y,
        int n_stations,
        int *station_coords,
        int *station_times,
        int *tt_grid,
        int *time_template,
        int walk_cells)
    {
        // for addressing into a linear array; cannot use z axis in blocks
        int blockSize = (blockDim.x * blockDim.y);
        int idx = (blockIdx.x * blockSize * gridDim.y) + (blockIdx.y * blockSize) + (threadIdx.x * blockDim.y) + threadIdx.y; 
   
        int x_0, y_0;
        int big_offset, lil_offset;
        int x, y;
        if (idx < n_stations) {
            x_0 = int(station_coords[idx * 2 + 0]);
            y_0 = int(station_coords[idx * 2 + 1]);
            for (x = - walk_cells; x < walk_cells; x++) {
                for (y = - walk_cells; y < walk_cells; y++) {
                    big_offset = ((x_0 + x) * max_y + (y_0 + y));
                    lil_offset = ((walk_cells + x) * walk_cells * 2 + (walk_cells + y));
                    // attention, atomic operations only work on ints
                    // and slows it way down (8x) for big maps, take a chance on race conditions?
                    // in that case everything can be floats. it looks to have little effect. there is visible noise in output...
                    // maybe switch to atomic operations at the end? using per pixel ops would solve this problem.
                    // atomicMin( &(tt_grid[big_offset]), time_template[lil_offset] + station_times[origin_idx * n_stations + idx] );
                    tt_grid[big_offset] = min(tt_grid[big_offset], time_template[lil_offset] + station_times[origin_idx * n_stations + idx]);
                    // tt_grid[big_offset] = time_template[lil_offset] + station_times[origin_idx * n_stations + idx];
                }    
            }
       }
    }

    __global__ void force ( 
        int   origin_idx,
        int   max_x,
        int   max_y,
        int   D,
        float *coords,
        float *vectors, 
        float *norms, 
        int   *tt, 
        float *adjust, 
        float *forces, 
        float *err,
        int   normalize)
    {
        int d;
        float tmp = 0;
        // for addressing into a linear array; cannot use z axis in blocks
        int blockSize = (blockDim.x * blockDim.y);
        int idx = (blockIdx.x * blockSize * gridDim.y) + (blockIdx.y * blockSize) + (threadIdx.x * blockDim.y) + threadIdx.y; 

        if (idx < max_x * max_y) 
        {
            for (d = 0; d < D; d++)
                tmp += pow((vectors[idx * D + d] = ( coords[idx * D + d] - coords[origin_idx * D + d]) ), float(2.0) );
            norms[idx]  = sqrt(tmp); 
            adjust[idx] = float(tt[idx]) - norms[idx];
            // check for divide by zero - avoid propagating NANs.
            // if (norms[idx] != 0) 
            if (norms[idx] != 0 && tt[idx] != 100000) 
                for (d = 0; d < D; d++)
                    // be sure to use a float to normalize! otherwise stable configuration drifts. (NO...)
                    // maybe drift is caused by difference between integral station coordinates and real coordinates. NO
                    // maybe caused by lack of nan_to_num? maybe just normal convergence behavior!
                    forces[idx * D + d] += ( vectors[idx * D + d] / norms[idx] ) * adjust[idx] / float(normalize); 
            err[idx] += abs(adjust[idx]);
        }
    }

    __global__ void integrate (
        int   max_x,
        int   max_y,
        int   D,
        float *coords,
        float *vels)
    {
       int d;
        // for addressing into a linear array; cannot use z axis in blocks
        int blockSize = (blockDim.x * blockDim.y);
        int idx = (blockIdx.x * blockSize * gridDim.y) + (blockIdx.y * blockSize) + (threadIdx.x * blockDim.y) + threadIdx.y; 
        if (idx < max_x * max_y) 
            for (d = 0; d < D; d++)
                coords[idx * D + d] += vels[idx * D + d];
    }
    
    """)

ttime_kernel     = mod.get_function("ttime")
force_kernel     = mod.get_function("force")
integrate_kernel = mod.get_function("integrate")

tt_grid = np.empty( grid_dim, dtype=np.int32 ) 
n_pass  = 0        
t_start = time.time()
while (1) :
    t_start_inner = time.time()
    # seems to be an acceptable way to zeo floats on this hardware
    cuda.memset_d32(err_gpu,    0, n_gridpoints)
    cuda.memset_d32(adjust_gpu, 0, n_gridpoints) # not necessary, for testing
    cuda.memset_d32(norms_gpu,  0, n_gridpoints) # not necessary, for testing
    cuda.memset_d32(forces_gpu, 0, n_gridpoints * DIMENSIONS)
            
    n_iter  = 0
    for origin_idx in range( n_stations ) :
        # print 'origin: ', origin_idx
        
        cuda.memset_d32(tt_gpu, BIGINT, n_gridpoints) # 0x7f800000 is bit pattern for positive infinity
        ttime_kernel(np.int32(origin_idx), max_x, max_y, np.int32(n_stations), station_coords_gpu, matrix_gpu, tt_gpu, time_template_gpu, walk_cells, block=(16,16,1), grid=(32,1))
        autoinit.context.synchronize()
        
#        station_times = matrix[origin_idx]
#        tt_grid.fill(BIGINT) # was np.inf
#        for dest_idx in range( n_stations ) :
#            # set up time window
#            dest_coords = station_coords[dest_idx]
#            x_0 = round(dest_coords[0] - walk_cells)
#            x_1 = round(dest_coords[0] + walk_cells)
#            y_0 = round(dest_coords[1] - walk_cells)
#            y_1 = round(dest_coords[1] + walk_cells)
#            np.add(time_template, station_times[dest_idx], temp_time_template)
#            tt_grid_view = tt_grid[x_0:x_1, y_0:y_1]
#            np.minimum(tt_grid_view, temp_time_template, tt_grid_view)
#
#        # cuda.memcpy_htod(tt_gpu, tt_grid)
#        print time_template
#        
#        pl.imshow(  tt_grid, cmap=mymap, origin='bottom') #, vmin=0, vmax=100 )
#        pl.colorbar()
#        pl.show()
#        pl.close()
#        cuda.memcpy_dtoh(tt_grid, tt_gpu)
#        print tt_grid
#        pl.imshow(  tt_grid.reshape(grid_dim).T / 60, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
#        pl.colorbar()
#        pl.show()
#        pl.close()

        # kludge, this indexing should be done differently.
        c = station_coords_int[origin_idx]
        c = c[0] * max_y + c[1]   # seems tested right... C language is row-major
        force_kernel(c, max_x, max_y, np.int32(DIMENSIONS), coords_gpu, vectors_gpu, norms_gpu, tt_gpu, adjust_gpu, forces_gpu, err_gpu, np.int32(n_stations), block=(16,16,1), grid=(32,40))
        autoinit.context.synchronize()
        
        if n_iter % 30  == 5 and n_pass < -1:
            cuda.memcpy_dtoh(grid, adjust_gpu)
            pl.imshow( grid.reshape( (max_x, max_y) ).T / 60.0, cmap=mymap, origin='bottom') #, vmin=0, vmax=100 )
            pl.colorbar()
            pl.show()

        n_iter += 1
        if n_iter % 10 == 0 : 
            sys.stdout.write( "\r%i%% (%i stations averaging %f seconds) " % (n_iter * 100 / n_stations, n_iter, (time.time() - t_start_inner) / n_iter) )
            sys.stdout.flush()
            
    integrate_kernel( max_x, max_y, np.int32(DIMENSIONS), coords_gpu, forces_gpu, err_gpu, block=(16,16,1), grid=(32,40) )
    autoinit.context.synchronize()
    n_pass += 1
    if n_pass % 1 == 0:
        cuda.memcpy_dtoh(coords, forces_gpu)
        velocities = np.sqrt(np.sum(coords**2, axis = 1)).reshape(grid_dim) 
        pl.imshow( velocities.T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
        pl.title( 'Velocity ( sec / timestep) - step %03d' % n_pass )
        pl.colorbar()
        #pl.show()
        pl.savefig( 'img/vel%03d.png' % n_pass )
        pl.close()
        
        cuda.memcpy_dtoh(grid, err_gpu)
        pl.imshow( grid.reshape(grid_dim).T / 60.0 / np.float32(n_stations), cmap=mymap, origin='bottom', vmin=0, vmax=100 )
        pl.title( 'Average absolute error (min) - step %03d' %n_pass )
        pl.colorbar()
        pl.savefig( 'img/err%03d.png' % n_pass )
        pl.close()
        
    print "End of pass number %i." % n_pass
    print "Runtime %i minutes, average pass length %f minutes. " % ( (time.time() - t_start)/60, (time.time() - t_start) / n_pass / 60.0 )
    
    
