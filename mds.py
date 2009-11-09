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

station_coords_int = gpuarray.to_gpu(station_coords.round().astype(np.int32))
station_coords_gpu = station_coords_int.gpudata

min_x = 0
min_y = 0
max_x, max_y = grid_dim
n_gridpoints = int(max_x * max_y)
n_stations   = len(station_coords)

cuda_grid_shape = (int(max_x) / CUDA_BLOCK_SHAPE[0] + 1, int(max_y) / CUDA_BLOCK_SHAPE[1] + 1)


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

# careful, N_NEARBY STATIONS is hardcoded at 10 below.
near_stations = gpuarray.zeros ((cuda_grid_shape[0], cuda_grid_shape[1], 10), dtype=np.int32)
near_stations_gpu = near_stations.gpudata

test      = np.zeros( 100, dtype = np.int32 ) # for holding test results
test1_gpu = cuda.mem_alloc( test.nbytes ) # for holding test results exported from device
test2_gpu = cuda.mem_alloc( test.nbytes ) # for holding test results exported from device
test3_gpu = cuda.mem_alloc( test.nbytes ) # for holding test results exported from device

# times could be merged into forces kernel, if done by pixel not station.
# integrate kernel could be GPUArray operation; also helps clean up code by using GPUArrays.
# DIM should be replaced by python script, so as not to define twice. 

mod              = SourceModule( open("unified_mds.c").read() )
stations_kernel  = mod.get_function("stations"  )
unified_kernel   = mod.get_function("unified"  )
integrate_kernel = mod.get_function("integrate")

stations_kernel(np.int32(n_stations), station_coords_gpu, near_stations_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
autoinit.context.synchronize()

near_stations_texref = mod.get_texref('near_stations')
# borrowed from pycuda slicing code
near_stations_1D = gpuarray.GPUArray(
                    shape=(near_stations.size, ),
                    dtype=near_stations.dtype,
                    allocator=near_stations.allocator,
                    base=near_stations,
                    gpudata=int(near_stations.gpudata))
                    
print '3d rep:'
print near_stations
print '1D rep:'
print near_stations_1D
near_stations.bind_to_texref_ext(near_stations_texref, channels=1)
print near_stations_texref.get_format()
print near_stations_texref.get_filter_mode()
print near_stations_texref.get_address_mode(0)
print near_stations_texref.get_address_mode(1)
print near_stations_texref.get_address_mode(2)
print near_stations_texref.get_flags()
#sys.exit()

station_coords_texref = mod.get_texref('station_coords')
station_coords_int.bind_to_texref_ext(station_coords_texref, channels=1)
# cuda.matrix_to_texref(station_coords, station_coords_texref, order="C")

def bind_to_texref_ext_2D(self, texref, channels=1, allow_offset=False):
    assert len(self.shape) == 2
    read_as_int = np.integer in self.dtype.type.__mro__

    ad = cuda.ArrayDescriptor()
    ad.width, ad.height = self.shape
    ad.format = cuda.dtype_to_array_format(self.dtype)
    ad.num_channels = channels
    if read_as_int:
        texref.set_flags(texref.get_flags() | cuda.TRSF_READ_AS_INTEGER)

    texref.set_address_2d(self.gpudata, ad, 255) #self.dtype.itemsize)

#    if read_as_int:
#        texref.set_flags(texref.get_flags() | cuda.TRSF_READ_AS_INTEGER)

matrix_texref = mod.get_texref('matrix')
matrix = np.zeros((64, 64), dtype=np.int32)
for i in range(len(matrix)) :
    for j in range(len(matrix)) :
        matrix[i, j] = i*100 + j

matrix_gpu = gpuarray.to_gpu(matrix)
bind_to_texref_ext_2D(matrix_gpu, matrix_texref)
matrix_texref.set_filter_mode(cuda.filter_mode.LINEAR)
#cuda.matrix_to_texref(matrix, matrix_texref, order="C")

testtexture_array = gpuarray.to_gpu(np.array(range(0, 10), dtype=np.int32) )
print testtexture_array
testtexture_texref = mod.get_texref('testtexture')
testtexture_texref.set_address_mode(0, cuda.address_mode.CLAMP)
testtexture_texref.set_address_mode(1, cuda.address_mode.CLAMP)
testtexture_texref.set_filter_mode(cuda.filter_mode.POINT)
assert testtexture_texref.get_flags() == 0
print testtexture_texref
testtexture_array.bind_to_texref_ext(testtexture_texref, channels=1)

#while (1):
#    autoinit.context.synchronize()

t_start = time.time()
n_pass = 0
while (1) :
    n_pass += 1    
    # Pay attention to grid sizes: if you don't run the integrator on the coordinates connected to stations, 
    # they don't move... so the whole thing stabilizes in a couple of cycles.    
    
    for subset_low in range(0, n_stations, STATION_BLOCK_SIZE) :
        subset_high = subset_low + STATION_BLOCK_SIZE
        if subset_high > n_stations : subset_high = n_stations
        sys.stdout.write( "\rLaunching kernel for station range %03i to %03i of %03i." % (subset_low, subset_high, n_stations) )
        sys.stdout.flush()
        # texrefs in call seems to change nothing.
        unified_kernel(np.int32(n_stations), np.int32(subset_low), np.int32(subset_high), max_x, max_y, station_coords_gpu, matrix_gpu, coords_gpu, forces_gpu, error_gpu, test1_gpu, test2_gpu, test3_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape, texrefs=[near_stations_texref, station_coords_texref, matrix_texref, testtexture_texref])    
        autoinit.context.synchronize()
        time.sleep(0.5)  # let the user interface catch up.
        
    print "Integrating..."
    integrate_kernel(max_x, max_y, coords_gpu, forces_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
    autoinit.context.synchronize()

    if n_pass % IMAGES_EVERY == 0: # Make images of progress every N passes
    
        cuda.memcpy_dtoh(test, test1_gpu)
        print test
        cuda.memcpy_dtoh(test, test2_gpu)
        print test
        cuda.memcpy_dtoh(test, test3_gpu)
        print test
        
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

#        cuda.memcpy_dtoh(grid, err_gpu)
#        pl.imshow( grid.reshape(grid_dim).T / 60, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
#        pl.title( 'Test Output - step %03d' %n_pass )
#        pl.colorbar()
#        pl.savefig( 'img/err%03d.png' % n_pass )
#        pl.close()
    
    print "End of pass number %i." % n_pass
    print "Runtime %i minutes, average pass length %f minutes. " % ( (time.time() - t_start) / 60.0, (time.time() - t_start) / n_pass / 60.0 )
    
    
