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

import pycuda.driver   as cuda
import pycuda.autoinit as autoinit
from   pycuda.compiler import SourceModule

#MATRIX_FILE       = './od_matrix_trimet_linked.npz'
MATRIX_FILE       = './od_matrix_BART.npz'
DIMENSIONS        = 4   
CUDA_BLOCK_SHAPE  = (16, 16, 1) # Best to put a multiple of 32 threads in a block. For square blocks, this means 8x8 or 16x16. 8x8 is actually much faster on BART! 
IMAGES_EVERY      = 1
STATION_BLOCK_LEN = 500
N_NEARBY_STATIONS = 8

print 'Loading matrix...'
npz = np.load(MATRIX_FILE)
station_coords = npz['station_coords'] 
grid_dim       = npz['grid_dim']
matrix         = npz['matrix'].astype(np.int32)

# EVERYTHING SHOULD BE IN FLOAT32 for ease of debugging. even times.
# Matrix and others should be textures, arrays, or in constant memory, to do cacheing.
# As it is, I'm doing explicit cacheing.

# force OD matrix symmetry for test
# THIS was responsible for the coordinate drift!!!
# need to symmetrize it before copy to device
matrix = (matrix + matrix.T) / 2

matrix_gpu = cuda.mem_alloc(matrix.nbytes)
cuda.memcpy_htod(matrix_gpu, matrix)

station_coords_int = station_coords.round().astype(np.int32)
station_coords_gpu = cuda.mem_alloc(station_coords_int.nbytes)
cuda.memcpy_htod(station_coords_gpu, station_coords_int)

min_x = 0
min_y = 0
max_x, max_y = grid_dim
n_gridpoints = int(max_x * max_y)
n_stations   = len(station_coords)

# this could just be done by using ceil()
cuda_grid_shape = (int(max_x) / CUDA_BLOCK_SHAPE[0] + 1, int(max_y) / CUDA_BLOCK_SHAPE[1] + 1)
cuda_threads_per_block = CUDA_BLOCK_SHAPE[0] * CUDA_BLOCK_SHAPE[1]

if N_NEARBY_STATIONS > n_stations :
    N_NEARBY_STATIONS = n_stations

if N_NEARBY_STATIONS > cuda_threads_per_block :
    N_NEARBY_STATIONS = cuda_threads_per_block

print "----PARAMETERS----"
print "Input file:            ", MATRIX_FILE
print "Number of stations:    ", n_stations
print "OD matrix shape:       ", matrix.shape    
print "Station coords shape:  ", station_coords_int.shape 
print "Map dimensions:        ", grid_dim
print "CUDA block dimensions: ", CUDA_BLOCK_SHAPE 
print "CUDA grid dimensions:  ", cuda_grid_shape


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
mymap.set_bad ( (0.5, 0.0, 0.5) )
pl.plt.register_cmap(cmap=mymap)

error  = np.zeros( n_gridpoints, dtype=np.float32 ) 
coords = np.random.random( (n_gridpoints, DIMENSIONS) ).astype(np.float32) # initialize coordinates to random
forces_gpu = cuda.mem_alloc( coords.nbytes ) # 3D float32 accumulate forces over one timestep
error_gpu  = cuda.mem_alloc( error.nbytes  ) # 2D float32 cell error accumulation
coords_gpu = cuda.mem_alloc( coords.nbytes ) # 3D float32 time-space coordinates for each map cell
cuda.memcpy_htod(coords_gpu, coords)

nearby_idx     = np.zeros( (cuda_grid_shape[0], cuda_grid_shape[1], N_NEARBY_STATIONS), dtype=np.int32 ) 
nearby_idx_gpu = cuda.mem_alloc( nearby_idx.nbytes ) 
nearby_x_gpu = cuda.mem_alloc( nearby_idx.nbytes ) 
nearby_y_gpu = cuda.mem_alloc( nearby_idx.nbytes ) 

test      = np.zeros( 1000, dtype = np.int32 ) # for holding test results
test1_gpu = cuda.mem_alloc( test.nbytes ) # for holding test results exported from device
test2_gpu = cuda.mem_alloc( test.nbytes ) # for holding test results exported from device
test3_gpu = cuda.mem_alloc( test.nbytes ) # for holding test results exported from device

# times could be merged into forces kernel, if done by pixel not station.
# integrate kernel could be GPUArray operation; also helps clean up code by using GPUArrays.
# DIM should be replaced by python script, so as not to define twice. 

print 'Compiling kernel source...'
src = open("unified_mds.c").read()
src = src.replace( 'DIM',        str(DIMENSIONS)         )
src = src.replace( 'GRID_X',     str(cuda_grid_shape[0]) )
src = src.replace( 'GRID_Y',     str(cuda_grid_shape[1]) )
src = src.replace( 'N_STATIONS', str(n_stations)         )
src = src.replace( 'STATION_BLOCK_LEN', str(STATION_BLOCK_LEN) )
src = src.replace( 'N_NEARBY_STATIONS', str(N_NEARBY_STATIONS) )
#print src
mod              = SourceModule( src )
stations_kernel  = mod.get_function("stations"  )
unified_kernel   = mod.get_function("unified"  )
integrate_kernel = mod.get_function("integrate")
    
stations_kernel(station_coords_gpu, nearby_stations_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
autoinit.context.synchronize()

t_start = time.time()
n_pass = 0
while (1) :
    n_pass += 1    
    # Pay attention to grid sizes: if you don't run the integrator on the coordinates connected to stations, 
    # they don't move... so the whole thing stabilizes in a couple of cycles.    
    
    
    np.set_printoptions(threshold=np.nan)
#    cuda.memcpy_dtoh(nearby_stations, nearby_stations_gpu)
#    print nearby_stations
    
    for subset_low in range(0, n_stations, STATION_BLOCK_LEN) :
        subset_high = subset_low + STATION_BLOCK_LEN
        if subset_high > n_stations : subset_high = n_stations
        sys.stdout.write( "\rLaunching kernel for station range %03i to %03i of %03i." % (subset_low, subset_high, n_stations) )
        sys.stdout.flush()
        unified_kernel(np.int32(subset_low), np.int32(subset_high), max_x, max_y, station_coords_gpu, nearby_stations_gpu, matrix_gpu, coords_gpu, forces_gpu, error_gpu, test1_gpu, test2_gpu, test3_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
        autoinit.context.synchronize()
        time.sleep(0.5)  # let the user interface catch up.
        
    print "Integrating..."
    integrate_kernel(max_x, max_y, coords_gpu, forces_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
    autoinit.context.synchronize()

    if n_pass % IMAGES_EVERY == 0: # Make images of progress every N passes
    
        cuda.memcpy_dtoh(test, test1_gpu)
        print test[:20]
        cuda.memcpy_dtoh(test, test2_gpu)
        print test[:20]
        cuda.memcpy_dtoh(test, test3_gpu)
        print test[:20]
        
        cuda.memcpy_dtoh(coords, forces_gpu)
        velocities = np.sqrt(np.sum(coords**2, axis = 1)).reshape(grid_dim) 
        pl.imshow( velocities.T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
        pl.title( 'Velocity ( sec / timestep) - step %03d' % n_pass )
        pl.colorbar()
        #pl.show()
        pl.savefig( 'img/vel%03d.png' % n_pass )
        pl.close()
        
        cuda.memcpy_dtoh(error, error_gpu)
        pl.imshow( error.reshape(grid_dim).T / 60.0 / np.float32(n_stations), cmap=mymap, origin='bottom', vmin=0, vmax=100 )
        pl.title( 'Average absolute error (min) - step %03d' %n_pass )
        pl.colorbar()
        pl.savefig( 'img/err%03d.png' % n_pass )
        pl.close()

        cuda.memcpy_dtoh(error, error_gpu)
        pl.imshow( error.reshape(grid_dim).T / 60, cmap=mymap, origin='bottom') #, vmin=0, vmax=100 )
        pl.title( 'Test Output - step %03d' %n_pass )
        pl.colorbar()
        pl.savefig( 'img/err%03d.png' % n_pass )
        pl.close()
    
    print "End of pass number %i." % n_pass
    print "Runtime %i minutes, average pass length %f minutes. " % ( (time.time() - t_start) / 60.0, (time.time() - t_start) / n_pass / 60.0 )
    
    
