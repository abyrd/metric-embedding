#!run me with python

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State
from matplotlib.colors import LinearSegmentedColormap
from pylab  import *
from numpy  import *
from random import shuffle
from random import sample
import time 
from socket import *
import geotools
from math import ceil
import struct
import pyproj
import sys

WALK_RANGE  = 7500
OBSTRUCTION = 1.4
WALK_SPEED  = 1.3 # m/sec
DIMENSIONS  = 5   # seems slower with 3 dimensions than 4 ?

print 'Loading matrix...'
npz = load('od_matrix.npz')
#npz = load('od_matrix_trimet_hpm.npz')
station_labels = npz['station_labels']
station_coords = npz['station_coords']
grid_latlon    = npz['grid_latlon']
grid_dim = npz['grid_dim']
matrix   = npz['matrix']
print "done."

min_x = 0
min_y = 0
max_x, max_y = grid_dim
n_gridpoints = max_x * max_y
n_stations   = len(station_labels)

station_coords_int = station_coords.round().astype(int)

# precompute time matrix template
walk_cells = int(WALK_RANGE / 100 / OBSTRUCTION)
times = arange(walk_cells * 2) - walk_cells
times = sqrt((times**2)[:, newaxis] + (times ** 2)[newaxis, :]) * 100 * OBSTRUCTION
times[times > WALK_RANGE] = inf
times /= WALK_SPEED
# time_template = times.astype(int)
time_template = times

# make and register custom color map
cdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.2,  0.0, 1.0),
                   (1.0,  0.2, 0.0)),

         'green': ((0.0,  0.0, 1.0),
                   (0.1,  0.2, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.1,  0.0, 1.0),
                   (0.2,  0.2, 0.0),
                   (1.0,  0.0, 0.0))}
                   
cdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.2,  0.0, 1.0),
                   (1.0,  0.1, 0.0)),

         'green': ((0.0,  0.0, 1.0),
                   (0.1,  0.1, 0.0),
                   (0.2,  0.0, 1.0),
                   (0.4,  0.1, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.1,  0.0, 1.0),
                   (0.2,  0.1, 0.0),
                   (1.0,  0.0, 0.0))}

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
#mymap.set_under('g', 1.0)
mymap.set_bad ( (0.5, 0.0, 0.5) )
plt.register_cmap(cmap=mymap)

# function to display a time grid with contour lines   
# file choice works with png vs. screen; for svg must change matplotlib config file 
def cplot(Z, filename = None, show_contours = False) :
    imshow(Z, cmap=mymap, origin='bottom', vmin=0, vmax=100)
    colorbar()
    if show_contours :
        levels = array([15, 30, 60, 90])
        CS = contour(Z, levels, linewidths=2)
        clabel(CS, inline=1, fmt='%d', fontsize=14)
    if filename is not None : 
        savefig(filename)
        close()
    else : show()


# force OD matrix symmetry for test
matrix = (matrix + matrix.T) / 2

coords      = random.random( (n_gridpoints, DIMENSIONS) )
coords_grid = coords.reshape (max_x, max_y, DIMENSIONS)    # identifier change to coords_2d ?

n_pass  = 0
temp_time_template = empty_like(time_template)
while (1) :
    t_start  = time.time()
    pass_err_max = 0
    pass_err = 0
    n_iter   = 0
    err_to   = zeros( n_gridpoints )
    error_to = zeros( n_gridpoints )
    times_to = zeros( n_gridpoints )    
    forces   = zeros( coords.shape )
    forces_grid = forces.reshape( coords_grid.shape )

    for origin_idx in range( n_stations ) :
        # print 'origin: ', origin_idx
        station_times = matrix[origin_idx]
        tt_grid = ones( grid_dim, dtype=int8 ) * inf 
        # tt_grid *= inf # a bit faster than remaking the array, but doesn't work
        # for every station, make a view of the travel time grid
        # and combine with the template using minimum function
        for dest_idx in range( n_stations ) :
            # set up time window
            dest_coords = station_coords[dest_idx]
            x_0 = round(dest_coords[0] - walk_cells)
            x_1 = round(dest_coords[0] + walk_cells)
            y_0 = round(dest_coords[1] - walk_cells)
            y_1 = round(dest_coords[1] + walk_cells)
            add(time_template, station_times[dest_idx], temp_time_template)
            tt_grid_view = tt_grid[x_0:x_1, y_0:y_1]
            minimum(tt_grid_view, temp_time_template, tt_grid_view)
            
        # cplot(tt_grid.T / 60)
        tt = tt_grid.ravel()  # returns a 1d reference, flatten() would return a copy

        # use a rounded version of the stop's coordinates as _indices_ into the coords table, so distances are relative to that point's current position
        # broadcasting OK - all dimensions to add are indices to the left (tested)
        vectors  = coords - coords_grid[ tuple( station_coords_int[origin_idx] ) ] 
        # square(vectors, vectors) # in-place squaring, not terribly much faster
        # norms = sum(vectors, axis=1)
        # sqrt(norms, norms)
        norms    = sqrt(sum(vectors**2, axis = 1))  # OK for finding norms, i.e. length of vectors (tested). could be precomputed.
        # print coords
        # print norms    
        # put nan to num in following line?
        adjust   = nan_to_num( tt - norms ) # both are same dim, ok for element-wise subtraction
        # adjust   = tt - norms # both are same dim, ok for element-wise subtraction
        uvectors = vectors / norms[:,newaxis]   # divide every 3vector element-wise by norms duplicated into axis 1
        # forces  += (uvectors * adjust[:,newaxis]) / (n_stations) # filter out NaNs (from perfectly reproduced distances, which give null vectors)
        forces  += nan_to_num(uvectors * adjust[:,newaxis]) / (n_stations) # filter out NaNs (from perfectly reproduced distances, which give null vectors)
        # nan to num slows down by about 0,1 sec
        # forces  += uvectors * adjust[:,newaxis] / (n_stations * 2) # filter out NaNs (from perfectly reproduced distances, which give null vectors)
        
        # Accumulate elements of the Kruskal stress to evaluate later
        # about 1/4 of runtime for BART
        err_to   += abs(adjust)
        # error_to += adjust ** 2
        # times_to += tt ** 2
        # next two lines slow down 0,08 sec per station
        # pass_err += sum(abs(adjust))
        # pass_err_max = max(pass_err_max, max(abs(adjust)))
        n_iter += 1
        if n_iter % 10 == 0 : 
            print "%i%% (%i stations averaging %f seconds)" % (n_iter * 100 / n_stations, n_iter, (time.time() - t_start) / n_iter)
    accels  = forces  # / n_stops   # normalize by number of total origins per pass, not total destinations
    vels    = accels  # Euler integration. Should be += but we consider that damping effectively cancels out all previous accelerations.
    coords += vels    # Euler integration - in both steps, timestep is implicitly 1. this worked well in testing.
    n_pass += 1
    # pass_err = pass_err / (n_stations * n_gridpoints)
    # k_energy = sum( sqrt( (vels**2).sum(axis=1) ) )
    # avg_k_energy = k_energy / n_gridpoints
    # stress^2 = summed squares of all errors / summed squares of all distances.
    # print "Average absolute error", pass_err
    # print "Maximum error in this pass", pass_err_max
    # print "Kinetic energy total / average", k_energy, avg_k_energy
    # stress = sqrt(error_to / times_to).reshape(grid_dim)
    err = (err_to / n_stations).reshape(grid_dim)
    # print "Max err:", err.max()
    # print err.max(), err.min()
    force_norms = sqrt(sum(forces**2, axis = 1)).reshape(grid_dim) # actually this is the same thing as velocities!

    #    from mpl_toolkits.mplot3d import Axes3D
    #    import matplotlib.pyplot as plt
    #    # fig = plt.figure()
    #    ax = Axes3D(figure())
    #    samples = coords[abs(coords[:,1]) < 10000]
    #    samples = array( sample(samples, 3000) )
    #    #print samples
    #    ax.scatter(samples[:,0], samples[:,1], samples[:,2])
    #    ax.set_xlabel('X')
    #    ax.set_ylabel('Y')
    #    savefig('particles%03d.png' % n_pass, dpi=100 )
    #    close()
    #    # plt.show()

    #gcf().set_size_inches(15, 8)
    #subplot(1, 2, 1)
    #title( 'Force ( sec / timestep)' )
    #imshow(force_norms.T, cmap=mymap, origin='bottom', vmin=0, vmax=1000)
    #colorbar(shrink=0.7)
    #subplot(1, 2, 2)
    #title( 'Average error (min)' )
    #imshow(err.T / 60, cmap=mymap, origin='bottom', vmin=0, vmax=100)
    #colorbar(shrink=0.7)
    #savefig('force-err%03d.png' % n_pass, dpi=100 )
    #close()
    
    imshow( force_norms.T, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
    title( 'Force ( sec / timestep) - step %03d' % n_pass )
    colorbar()
    savefig( 'img/force%03d.png' % n_pass )
    close()

    imshow( err.T / 60, cmap=mymap, origin='bottom', vmin=0, vmax=100 )
    title( 'Average error (min) - step %03d' %n_pass )
    colorbar()
    savefig( 'img/error%03d.png' % n_pass )
    close()

    # cplot ( force_norms.T, filename=('force%03d.png' % n_pass ) )
    # cplot ( err.T / 60, filename=('err%03d.png' % n_pass ) )
    # print "Total stress for this pass:", sqrt( sum(error_to) / sum(times_to) )
    # if avg_k_energy < 20 :
    #    break
    # visu_pos_b(coords)
    print "End of pass number %i." % n_pass

