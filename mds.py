#!/opt/local/bin/python

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
import pycuda.gpuarray as gpuarray

#MATRIX_FILE       = './od_matrix_trimet_linked.npz'
MATRIX_FILE        = './od_matrix_BART.npz'
DIMENSIONS         = 4   
CUDA_BLOCK_SHAPE   = (8, 8, 1) # Best to put a multiple of 32 threads in a block. For square blocks, this means 8x8 or 16x16. 8x8 is actually much faster on BART! 
IMAGES_EVERY       = 1
STATION_BLOCK_SIZE = 500
N_NEARBY_STATIONS  = 10

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

station_coords_int = station_coords.round().astype(np.int32)

# to be removed when textures are working
station_coords_gpu = gpuarray.to_gpu(station_coords_int)
matrix_gpu = gpuarray.to_gpu(matrix)

max_x, max_y = grid_dim
n_gridpoints = int(max_x * max_y)
n_stations   = len(station_coords)

cuda_grid_shape = ( int( math.ceil( float(max_x)/CUDA_BLOCK_SHAPE[0] ) ), int( math.ceil( float(max_y)/CUDA_BLOCK_SHAPE[1] ) ) )

print "----PARAMETERS----"
print "Input file:            ", MATRIX_FILE
print "Number of stations:    ", n_stations
print "OD matrix shape:       ", matrix.shape    
print "Station coords shape:  ", station_coords_int.shape 
print "Station cache size:    ", N_NEARBY_STATIONS
print "Map dimensions:        ", grid_dim
print "Numner of map points:  ", n_gridpoints
print "CUDA block dimensions: ", CUDA_BLOCK_SHAPE 
print "CUDA grid dimensions:  ", cuda_grid_shape

assert station_coords.shape == (n_stations, 2)
assert N_NEARBY_STATIONS <= n_stations

# Make and register custom color map for pylab graphs

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

# set up arrays for calculations

coords_gpu = gpuarray.to_gpu(np.random.random( (max_x, max_y, DIMENSIONS) ).astype(np.float32))   # initialize coordinates to random values in range 0...1
forces_gpu = gpuarray.zeros( (int(max_x), int(max_y), DIMENSIONS), dtype=np.float32 )             # 3D float32 accumulate forces over one timestep
errors_gpu = gpuarray.zeros( (int(max_x), int(max_y)),             dtype=np.float32 )             # 2D float32 cell error accumulation
near_stations_gpu = gpuarray.zeros( (cuda_grid_shape[0], cuda_grid_shape[1], N_NEARBY_STATIONS), dtype=np.int32)

debug_gpu = gpuarray.zeros( 100, dtype = np.int32 )

# times could be merged into forces kernel, if done by pixel not station.
# integrate kernel could be GPUArray operation; also helps clean up code by using GPUArrays.
# DIM should be replaced by python script, so as not to define twice. 
src = open("unified_mds.c").read()
src = src.replace( 'N_NEARBY_STATIONS_PYTHON', str(N_NEARBY_STATIONS) )
src = src.replace( 'N_STATIONS_PYTHON', str(n_stations) )
mod = SourceModule(src)
stations_kernel  = mod.get_function("stations"  )
unified_kernel   = mod.get_function("unified"  )
integrate_kernel = mod.get_function("integrate")

matrix_texref         = mod.get_texref('tex_matrix')
station_coords_texref = mod.get_texref('tex_station_coords')
near_stations_texref  = mod.get_texref('tex_near_stations')

cuda.matrix_to_texref(matrix, matrix_texref, order="F") # copy directly to device with texref - made for 2D x 1channel textures
cuda.matrix_to_texref(station_coords_int, station_coords_texref, order="F") # fortran ordering, because we will be accessing with texND() instead of C-style indices
near_stations_gpu.bind_to_texref_ext(near_stations_texref)

# note, cuda.In and cuda.Out are from the perspective of the KERNEL not the host app!
stations_kernel(near_stations_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
autoinit.context.synchronize()

print "Near stations cache:"
print near_stations_gpu

t_start = time.time()
n_pass = 0
while (1) :
    n_pass += 1    
    # Pay attention to grid sizes when testing: if you don't run the integrator on the coordinates connected to stations, 
    # they don't move... so the whole thing stabilizes in a couple of cycles.    
    
    for subset_low in range(0, n_stations, STATION_BLOCK_SIZE) :
        subset_high = subset_low + STATION_BLOCK_SIZE
        if subset_high > n_stations : subset_high = n_stations
        sys.stdout.write( "\rLaunching kernel for station range %03i to %03i of %03i." % (subset_low, subset_high, n_stations) )
        sys.stdout.flush()
        # adding texrefs in kernel call seems to change nothing, leaving them out.
        unified_kernel(np.int32(n_stations), np.int32(subset_low), np.int32(subset_high), max_x, max_y, station_coords_gpu, matrix_gpu, coords_gpu, forces_gpu, errors_gpu, debug_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)
        autoinit.context.synchronize()
        time.sleep(0.5)  # let the user interface catch up.
        
    print "Integrating..."
    integrate_kernel(max_x, max_y, coords_gpu, forces_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
    autoinit.context.synchronize()

    if n_pass % IMAGES_EVERY == 0: # Make images of progress every N passes
    
        #print 'Kernel debug output:'
        #print debug_gpu
        
        velocities = np.sqrt(np.sum((forces_gpu**2).get(), axis = 1)) 
        pl.imshow( velocities.T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
        pl.title( 'Velocity ( sec / timestep) - step %03d' % n_pass )
        pl.colorbar()
        #pl.show()
        pl.savefig( 'img/vel%03d.png' % n_pass )
        pl.close()
        
        pl.imshow( (errors_gpu / 60 / float(n_stations)).get().T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
        pl.title( 'Average absolute error (min) - step %03d' %n_pass )
        pl.colorbar()
        pl.savefig( 'img/err%03d.png' % n_pass )
        pl.close()

#        cuda.memcpy_dtoh(grid, err_gpu)
#        pl.imshow( grid.reshape(grid_dim).T / 60, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
#        pl.title( 'Test Output - step %03d' %n_pass )
#        pl.colorbar()
#        pl.savefig( 'img/err%03d.png' % n_pass )
#        pl.close()
    
    print "End of pass number %i." % n_pass
    print "Runtime %i minutes, average pass length %f minutes. " % ( (time.time() - t_start) / 60.0, (time.time() - t_start) / n_pass / 60.0 )
    
    
