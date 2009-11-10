#!/usb/bin/env python2.6

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
import threading

import pycuda.driver   as cuda
#import pycuda.autoinit as autoinit
from   pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

CUDA_BLOCK_SHAPE = (8, 8, 1) # Best to put a multiple of 32 threads in a block. For square blocks, this means 8x8 or 16x16. 8x8 is actually much faster on BART! 

class MDS(threading.Thread):
    def __init__(self, matrix_file, dimensions, n_iterations, images_every, station_block_size, n_nearby_stations, debug_output, interface): 
        threading.Thread.__init__(self)
        self.MATRIX_FILE = matrix_file
        self.DIMENSIONS  = dimensions
        self.N_ITERATIONS = n_iterations
        self.IMAGES_EVERY = images_every
        self.STATION_BLOCK_SIZE = station_block_size
        self.N_NEARBY_STATIONS = n_nearby_stations 
        self.DEBUG_OUTPUT = debug_output
        self.INTERFACE = interface
        print 'MDS Thread initialized'

    def run(self) :
        cuda_dev = cuda.Device(0)
        cuda_context = cuda_dev.make_context()
        print 'Loading matrix...'
        npz = np.load(self.MATRIX_FILE)
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

        print "\n----PARAMETERS----"
        print "Input file:            ", self.MATRIX_FILE
        print "Number of stations:    ", n_stations
        print "OD matrix shape:       ", matrix.shape    
        print "Station coords shape:  ", station_coords_int.shape 
        print "Station cache size:    ", self.N_NEARBY_STATIONS
        print "Map dimensions:        ", grid_dim
        print "Number of map points:  ", n_gridpoints
        print "Target space dimensionality: ", self.DIMENSIONS
        print "CUDA block dimensions: ", CUDA_BLOCK_SHAPE 
        print "CUDA grid dimensions:  ", cuda_grid_shape

        assert station_coords.shape == (n_stations, 2)
        assert self.N_NEARBY_STATIONS <= n_stations
        
        #sys.exit()

        # Make and register custom color map for pylab graphs

        cdict = {'red':   ((0.0,  0.0, 0.0),
                           (0.2,  0.0, 0.0),
                           (0.4,  0.9, 0.9),
                           (1.0,  0.0, 0.0)),

                 'green': ((0.0,  0.0, 0.1),
                           (0.05, 0.9, 0.9),
                           (0.1,  0.0, 0.0),
                           (0.4,  0.9, 0.9),
                           (0.6,  0.0, 0.0),
                           (1.0,  0.0, 0.0)),

                 'blue':  ((0.0,  0.0, 0.0),
                           (0.05, 0.0, 0.0),
                           (0.2,  0.9, 0.9),
                           (0.3,  0.0, 0.0),
                           (1.0,  0.0, 0.0))}

        mymap = LinearSegmentedColormap('mymap', cdict)
        mymap.set_over( (1.0, 0.0, 1.0) )
        mymap.set_bad ( (0.0, 0.0, 0.0) )
        pl.plt.register_cmap(cmap=mymap)

        # set up arrays for calculations

        coords_gpu = gpuarray.to_gpu(np.random.random( (max_x, max_y, self.DIMENSIONS) ).astype(np.float32))   # initialize coordinates to random values in range 0...1
        forces_gpu = gpuarray.zeros( (int(max_x), int(max_y), self.DIMENSIONS), dtype=np.float32 )             # 3D float32 accumulate forces over one timestep
        weights_gpu = gpuarray.zeros( (int(max_x), int(max_y)),             dtype=np.float32 )             # 2D float32 cell error accumulation
        errors_gpu = gpuarray.zeros( (int(max_x), int(max_y)),             dtype=np.float32 )             # 2D float32 cell error accumulation
        near_stations_gpu = gpuarray.zeros( (cuda_grid_shape[0], cuda_grid_shape[1], self.N_NEARBY_STATIONS), dtype=np.int32)

        debug_gpu     = gpuarray.zeros( n_gridpoints, dtype = np.int32 )
        debug_img_gpu = gpuarray.zeros_like( errors_gpu )

        print "\n----COMPILATION----"
        # times could be merged into forces kernel, if done by pixel not station.
        # integrate kernel could be GPUArray operation; also helps clean up code by using GPUArrays.
        # DIM should be replaced by python script, so as not to define twice. 
        src = open("unified_mds.cu").read()
        src = src.replace( 'N_NEARBY_STATIONS_PYTHON', str(self.N_NEARBY_STATIONS) )
        src = src.replace( 'N_STATIONS_PYTHON', str(n_stations) )
        src = src.replace( 'DIMENSIONS_PYTHON', str(self.DIMENSIONS) )
        mod = SourceModule(src, options=["--ptxas-options=-v"])
        stations_kernel  = mod.get_function("stations"  )
        forces_kernel    = mod.get_function("forces"  )
        integrate_kernel = mod.get_function("integrate")

        matrix_texref         = mod.get_texref('tex_matrix')
        station_coords_texref = mod.get_texref('tex_station_coords')
        near_stations_texref  = mod.get_texref('tex_near_stations')
        #ts_coords_texref  = mod.get_texref('tex_ts_coords') could be a 4-channel 2 dim texture, or 3 dim texture. or just 1D.

        cuda.matrix_to_texref(matrix, matrix_texref, order="F") # copy directly to device with texref - made for 2D x 1channel textures
        cuda.matrix_to_texref(station_coords_int, station_coords_texref, order="F") # fortran ordering, because we will be accessing with texND() instead of C-style indices
        near_stations_gpu.bind_to_texref_ext(near_stations_texref)

        # note, cuda.In and cuda.Out are from the perspective of the KERNEL not the host app!
        stations_kernel(near_stations_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
        cuda_context.synchronize()

        #print "Near stations list:"
        #print near_stations_gpu
        print "\n----CALCULATION----"
        t_start = time.time()
        n_pass = 0
        while (n_pass < self.N_ITERATIONS) :
            n_pass += 1    
            # Pay attention to grid sizes when testing: if you don't run the integrator on the coordinates connected to stations, 
            # they don't move... so the whole thing stabilizes in a couple of cycles.    
            # Stations are worked on in blocks to avoid locking up the GPU with one giant kernel.
            for subset_low in range(0, n_stations, self.STATION_BLOCK_SIZE) :
                subset_high = subset_low + self.STATION_BLOCK_SIZE
                if subset_high > n_stations : subset_high = n_stations
                sys.stdout.write( "\rpass %03i / station %04i of %04i / total runtime %03.1f min " % (n_pass, subset_high, n_stations, (time.time() - t_start) / 60.0) )
                sys.stdout.flush()
                self.INTERFACE.showText("pass %03i / station %04i of %04i / total runtime %03.1f min / avg pass time %02.1f sec" % (n_pass, subset_high, n_stations, (time.time() - t_start) / 60.0, (time.time() - t_start) / n_pass + (subset_low/n_stations)))
                # adding texrefs in kernel call seems to change nothing, leaving them out.
                # max_x and max_y could be #defined in kernel source, along with STATION_BLOCK_SIZE 
                forces_kernel(np.int32(n_stations), np.int32(subset_low), np.int32(subset_high), max_x, max_y, coords_gpu, forces_gpu, weights_gpu, errors_gpu, debug_gpu, debug_img_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)
                cuda_context.synchronize()
                # print coords_gpu, forces_gpu
                time.sleep(0.5)  # let the user interface catch up.
                
            integrate_kernel(max_x, max_y, coords_gpu, forces_gpu, weights_gpu, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)    
            cuda_context.synchronize()

            #print self.IMAGES_EVERY
            if (self.IMAGES_EVERY > 0) and (n_pass % self.IMAGES_EVERY == 0) :
            
                #print 'Kernel debug output:'
                #print debug_gpu
                
                velocities = np.sqrt(np.sum(forces_gpu.get() ** 2, axis = 2)) 
                pl.imshow( velocities.T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
                pl.title( 'Velocity ( sec / timestep) - step %03d' % n_pass )
                pl.colorbar()
                pl.savefig( 'img/vel%03d.png' % n_pass )
                pl.close()
                
                pl.imshow( (errors_gpu.get() / weights_gpu.get() / 60.0 ).T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
                pl.title( 'Average absolute error (min) - step %03d' %n_pass )
                pl.colorbar()
                pl.savefig( 'img/err%03d.png' % n_pass )
                pl.close()

                pl.imshow( (debug_img_gpu.get() / 60.0).T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
                pl.title( 'Debugging Output - step %03d' %n_pass )
                pl.colorbar()
                pl.savefig( 'img/debug%03d.png' % n_pass )
                pl.close()
                
                #self.INTERFACE.showImage('img/err%03d.png' % n_pass)
                self.INTERFACE.update
              
            sys.stdout.write( "/ avg pass time %02.1f sec" % ( (time.time() - t_start) / n_pass, ) )
            sys.stdout.flush()
