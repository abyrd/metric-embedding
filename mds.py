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
DIMENSIONS  = 4   

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
# DIM should be replaced by python script, so as not to define twice. 
mod = SourceModule("""
 
    #define DIM 4
    #define OBSTRUCTION 1.4
    #define WALK_SPEED  1.3
    #define INF         0x7f800000 
    #define N_NEARBY_STATIONS 10
    
    __global__ void unified (
        int   n_stations,
        int   max_x,
        int   max_y,
        int   *station_coords, // load into shared mem as ints
        int   *matrix,
        float *coords,
        float *forces,
        float *errors)
    {
        float *origin_coord, *this_coord;
        int   *matrix_row;
        float *this_error;
        float *this_force;
        float vector[DIM];
        float force[DIM]; 
        float adjust, dist;
        float error = 0;  // must initialize because error is cumulative per pass
        float norm  = 0;  // must initialize because components accumulated
        float tt, ds_tt;
        int   d;
        int   x = blockIdx.x * blockDim.x + threadIdx.x; 
        int   y = blockIdx.y * blockDim.y + threadIdx.y; 
        int   osx, osy, dsx, dsy;
        int   os_idx, ds_idx;
        __shared__ int   near_stations_idx   [N_NEARBY_STATIONS];
        __shared__ float near_stations_dist  [N_NEARBY_STATIONS];  // could later be used to store matrix times. make an alias pointer?
        __shared__ int   near_stations_x     [N_NEARBY_STATIONS];
        __shared__ int   near_stations_y     [N_NEARBY_STATIONS];
        __shared__ int   near_stations_coords[N_NEARBY_STATIONS][DIM];
        
        // this could be done once in another kernel before execution
        // each cell calculates its own distances (saves them if enough space?)
        // at each pass, block loads relevant its time-space coordinates
        if (threadIdx.x == blockDim.x / 2 && threadIdx.y == blockDim.y / 2) {  // the thread at the middle of the block will collect nearby stations for the block
            int   slots_filled = 0;
            float max_dist = 0;
            int   max_slot;
            for (ds_idx = 0; ds_idx < n_stations; ds_idx++) {  // for every station,
                dsx = station_coords[ds_idx * 2 + 0]; // get the destination station's geographic x coordinate 
                dsy = station_coords[ds_idx * 2 + 1]; // and the destination station's geographic y coordinate.
                dist = sqrt( pow(float(dsx - x), 2) + pow(float(dsy - y), 2)) * 100; // Find the geographic distance from the station to our texel
                if (slots_filled < N_NEARBY_STATIONS) { // fill up all the slots first, keeping track of farthest station
                    near_stations_idx [slots_filled] = ds_idx;
                    near_stations_dist[slots_filled] = dist;
                    if (dist > max_dist) {
                        max_dist = dist;
                        max_slot = slots_filled;
                    } 
                    slots_filled++;
                } else { // repeatedly replace max slot with closer stations
                    if (dist < max_dist) {
                        near_stations_idx [max_slot] = ds_idx;
                        near_stations_dist[max_slot] = dist;
                        max_dist = 0;
                        for (int slot = 0; slot < N_NEARBY_STATIONS; slot++) {
                            if (near_stations_dist[slot] > max_dist) {
                                max_dist = near_stations_dist[slot];
                                max_slot = slot;
                            }
                        }
                    }
                }
            } 
            // We now have the N nearest station indices.
            // Get their coords in geographic and time-space
            for (int idx = 0; idx < N_NEARBY_STATIONS; idx++) {
                ds_idx = near_stations_idx[idx];
                near_stations_x[idx] = station_coords[ds_idx * 2 + 0]; 
                near_stations_y[idx] = station_coords[ds_idx * 2 + 1]; 
                for (d = 0; d < DIM; d++) {
                    // BLAH near_stations_coords[idx][d] = coords[]
                }
            }
        }
        
        if ( x < max_x && y < max_y ) {   // check that this thread is inside the map
            this_coord = &( coords[(x * max_y + y) * DIM] );
            this_force = &( forces[(x * max_y + y) * DIM] );
            this_error = &( errors[ x * max_y + y       ] );
            #pragma unroll
            for (d = 0; d < DIM; d++) force[d] = 0;                // initialize force to 0. (Be sure to do this before the first station (outside loops!)
            for (os_idx = 0; os_idx < n_stations; os_idx++) {         // For every origin station,
                osx = station_coords[os_idx * 2 + 0];                  // Get the origin station's geographic x coordinate
                osy = station_coords[os_idx * 2 + 1];                  // and its geographic y coordinate,
                origin_coord = &( coords[(osx * max_y + osy) * DIM] ); // then a pointer to its time-space coordinates.
                // begin time kernel equivalent
                // could grab matrix line here into shared mem, as 8bit int in minutes
                matrix_row = &( matrix[os_idx * n_stations] );         // Get a pointer to the relevant OD matrix row,
                tt = INF;                                              // and set the current best travel time to +infinity.
                // maybe speed up by using  manhattan distance - even correct in american cities - nope, doesn't speed up. global memory is the read slowdown.
                for (ds_idx = 0; ds_idx < n_stations; ds_idx++) {      // For every destination station,
                    dsx = station_coords[ds_idx * 2 + 0];              // get the destination station's geographic x coordinate 
                    dsy = station_coords[ds_idx * 2 + 1];              // and the destination station's geographic y coordinate.
                    dist = sqrt( pow(float(dsx - x), 2) + pow(float(dsy - y), 2)) * 100;    // Find the geographic distance from the station to our texel, then
                    //if (dist > 3000) dist = INF;                       // ---changes neither speed nor convergence
                    ds_tt = (dist * OBSTRUCTION / WALK_SPEED)          // derive a travel time from the origin station to this texel 
                            + matrix_row[ds_idx];                      // through the destination station.
                    tt = min(tt, ds_tt);                               // Save the travel time if it is better than the current best time.
                    // could also use destination station to texel distance to make a force.
                }
                
                #pragma unroll
                for (d = 0; d < DIM; d++) {
                    vector[d] = this_coord[d] - origin_coord[d];  // Find the vector from the origin station to this texel in time-space,
                    norm += pow(vector[d], 2);                    // and accumulate the terms of its length (norm).
                }
                norm   = sqrt(norm);   // Finally, take the square root to find the distance in time-space.
                adjust = tt - norm ;   // How much would the point need to move to match true travel time?
                
                // use __isnan()                                                                                                    
                if (norm != 0) {       // avoid propagating nans - is there a way to do this without diverging? division by zero should give 0, would give right result... reformulate...
                    #pragma unroll
                    for (d = 0; d < DIM; d++) {
                        force[d] += ((vector[d] / norm) * adjust) / n_stations;      // add the shortest travel time force to the cumulative force for this texel
                                                                                     // changed = to += and it still doesn't work!
                    }
                    error += abs(adjust);
                }
            }

            // ATTENTION: the following lines move the coordinates, so if blocks exceed number of processors available,
            // some points will move before the others finish (or even start) calculating. This gives a chunky effect on output.
            // __threadfence(); is not sufficient so I split them off into another kernel.

            #pragma unroll
            for (d = 0; d < DIM; d++) {
                this_force[d]  = force[d]; // Output forces to device global memory.
            }
            *this_error = error;  // Output errors to device global memory
        }
    }
    
    __global__ void integrate (
        int   max_x,
        int   max_y,
        float *coords,
        float *forces)
    {
        int   d;
        float *this_coord;
        float *this_force;
        int   x = blockIdx.x * blockDim.x + threadIdx.x; 
        int   y = blockIdx.y * blockDim.y + threadIdx.y; 
        if ( x < max_x && y < max_y ) {   // check that this thread is inside the map
            this_coord = &( coords[(x * max_y + y) * DIM] );
            this_force = &( forces[(x * max_y + y) * DIM] );
            #pragma unroll
            for (d = 0; d < DIM; d++) {
                this_coord[d] += this_force[d]; // integrate - could be done after each iteration instead of all at once at the end.
            }
        }
    }
                
    """)

unified_kernel   = mod.get_function("unified")
integrate_kernel = mod.get_function("integrate")

n_pass  = 0        
t_start = time.time()
while ( n_pass < 300 ) :
    t_start_inner = time.time()
    # seems to be an acceptable way to zero floats on this hardware
    cuda.memset_d32(err_gpu,    0, n_gridpoints)
    cuda.memset_d32(forces_gpu, 0, n_gridpoints * DIMENSIONS)
            
    # attention to grid sizes: if you don't run the integrator on the coordinates connected to stations, they don't move... so the whole thing stabilizes in a couple of cycles.
    
    unified_kernel(np.int32(n_stations), max_x, max_y, station_coords_gpu, matrix_gpu, coords_gpu, forces_gpu, err_gpu, block=(16,16,1), grid=(38, 38))    
    #unified_kernel(np.int32(10), max_x, max_y, station_coords_gpu, matrix_gpu, coords_gpu, forces_gpu, err_gpu, block=(16,16,1), grid=(38, 38))    
    autoinit.context.synchronize()
    
    integrate_kernel(max_x, max_y, coords_gpu, forces_gpu, block=(16,16,1), grid=(38, 38))    
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

#        cuda.memcpy_dtoh(grid, err_gpu)
#        pl.imshow( grid.reshape(grid_dim).T / 60, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
#        pl.title( 'Test Output - step %03d' %n_pass )
#        pl.colorbar()
#        pl.savefig( 'img/err%03d.png' % n_pass )
#        pl.close()
        
    print "End of pass number %i." % n_pass
    print "Runtime %i minutes, average pass length %f minutes. " % ( (time.time() - t_start)/60, (time.time() - t_start) / n_pass / 60.0 )
    
    
