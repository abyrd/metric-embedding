#!run me with python

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State
from pylab  import *
from numpy  import *
from random import shuffle
import time 
from socket import *
import geotools
from math import ceil
import struct
import pyproj
import sys

reporter = sys.stdout

def cplot(Z) :
    imshow(Z, cmap=cm.gray, origin='bottom')
    levels = array([15, 30, 60, 90])
    CS = contour(Z, levels, linewidths=2)
    clabel(CS, inline=1, fmt='%d', fontsize=14)
    show()

gtfsdb = GTFSDatabase  ( '../gsdata/bart.gtfsdb' )
gdb    = GraphDatabase ( '../gsdata/bart.linked.gsdb' )
#gtfsdb = GTFSDatabase  ( '../gsdata/trimet_13sep2009.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.hpm.linked.gsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.nolink.gsdb' )

g  = gdb.incarnate()
t0 = 1253800000

proj = pyproj.Proj( proj='sinu', ellps='WGS84' )
geod = pyproj.Geod( ellps='WGS84' )
min_lon, min_lat, max_lon, max_lat = gtfsdb.extent()
min_lon, min_lat, arc_dist = geod.fwd(min_lon, min_lat, 180+45, 5000)
min_x, min_y = proj( min_lon, min_lat )
proj = pyproj.Proj( proj='sinu', ellps='WGS84', lon_0=min_lon, y_0=-min_y )

stop_vertices = [v for v in g.vertices if v.label[:4] == 'sta-']
stop_labels   = [v.label for v in stop_vertices]
stop_coords   = []
# doing it this way gives unicode station names, which give a bus error... who knows why.
# for stop_id, stop_name, lat, lon in gtfsdb.stops() :
for sl in stop_labels :
    stop_id, stop_name, lat, lon = gtfsdb.stop(sl[4:])
    stop_coords.append(proj(lon, lat))

stop_coords  = array(stop_coords) / 100    
n_stops      = len(stop_labels)
grid_extent  = (stop_coords.max(axis = 0) + (20, 20)).astype(int)
n_gridpoints = prod( grid_extent )
matrix = empty( (n_stops, n_stops), dtype=float )

#print stop_labels
#print stop_coords
#print stop_vertices

for origin_idx, origin_label in enumerate(stop_labels) :
    if reporter : 
        reporter.write( "\rProcessing %i / %i ..." % (origin_idx + 1, n_stops) )
        reporter.flush()
    g.spt_in_place(origin_label, None, State(1, t0))
    for dest_idx, dest_vertex in enumerate(stop_vertices) :
        # first board time should be subtracted here
        if dest_vertex.payload is None : delta_t = inf
        else : delta_t = dest_vertex.payload.time - t0
        matrix[origin_idx, dest_idx] = delta_t
print
# force OD matrix symmetry for test
matrix = (matrix + matrix.T) / 2

# precompute mask matrix here, use view later
times_x = arange(grid_extent[0] * 2) - grid_extent[0]
times_y = arange(grid_extent[1] * 2) - grid_extent[1]
times   = sqrt((times_x**2)[:, newaxis] + (times_y ** 2)[newaxis, :]) * 100 / 1.3  # m/sec

DIMENSIONS  = 4
coords      = random.random( (n_gridpoints, DIMENSIONS) )
coords_grid = coords.reshape(grid_extent[0], grid_extent[1], DIMENSIONS)

# initialize first two dimensions to planar coords
# starting 'closer' to the real solution speeds up convergence
v  = 35.0   # km per hour
v *= 1000   # m per hour
v /= 3600   # meters per second
t  = 1 / v  # seconds per meter
t *= 100    # seconds per 100 m
#for i in range(len(point_list)) :
#    coords[i][0] = point_list[i][0] * t
#    coords[i][1] = point_list[i][1] * t

# fix travel times in order to test array copy time
# tt = ones(len(point_list)) * 100

udp = socket( AF_INET, SOCK_DGRAM )
# i think timestep 0.7 just makes up for the fact that the forces were applied 2x, i.e. 0.7 x 0.7 = 0.5
# TIMESTEP = 0.7  
TIMESTEP = 1.0
n_pass  = 0
while (1) :
    t_start  = time.time()
    pass_err_max = 0
    pass_err = 0
    n_iter   = 0
    error_to = zeros( coords.shape[0] )
    times_to = zeros( coords.shape[0] )    
    forces   = zeros( coords.shape )
    forces_grid = forces.reshape( coords_grid.shape )

    temp_ttgrid = empty( grid_extent )
    for origin_idx in range( n_stops ) :
        result = matrix[origin_idx]
        ttgrid = ones( grid_extent ) * inf # actually should be set to times from point of departure 
        for dest_idx in range(n_stops) :
            # set up time window
            dest_coords = stop_coords[dest_idx]
            x_1 = round(2 * grid_extent[0] - dest_coords[0])
            x_0 = x_1 - grid_extent[0]
            y_1 = round(2 * grid_extent[1] - dest_coords[1])
            y_0 = y_1 - grid_extent[1]
            add(times[x_0:x_1, y_0:y_1], result[dest_idx], temp_ttgrid)
            minimum(ttgrid, temp_ttgrid, ttgrid)  # third param is destination variable, i.e. do operation in-place
        # cplot((ttgrid / 60).T)
        # visu_times(tt)
        tt = ttgrid.ravel()  # returns a 1d reference, flatten() would return a copy
        # use a rounded version of the stop's coordinates as _indices_ into the coords table, so distances are relative to that point's current position
        # broadcasting OK - all dimensions to add are indices to the left (tested)
        vectors  = coords - coords_grid[ tuple( stop_coords[origin_idx].round() ) ] 
        norms    = sqrt(sum(vectors**2, axis = 1))  # OK for finding norms, tested
        # print coords
        # print norms    
        adjust   = tt - norms  # both are same dim, ok for element-wise subtraction
        # stress     = sqrt(sum(adjust**2))
        # stress_rms = sqrt(sum(adjust**2) / adjust.size) 
        # print "Stress relative to current point is : %f / %f / %f" % (stress, stress_rms, stress_b)
        # when norms fall to zero, division gives NaN
        # use nan_to_num() or myarr[np.isnan(myarr)] = 0
        uvectors = vectors / norms[:,newaxis]   # divide every 3vector element-wise by norms duplicated into axis 1
        forces  += nan_to_num(uvectors * adjust[:,newaxis]) / (n_stops * 2) # filter out NaNs (from perfectly reproduced distances, which give null vectors)
        # should this also be done after each full pass?
        # vels += accels  # instantaneous acceleration vectors added to existing velocities
        # WARNING following line is now very asymmetric application of forces.
        # forces_grid[ tuple( stop_coords[origin_idx].round() ) ] -= nan_to_num(uvectors * adjust[:,newaxis]).sum(axis=0) / (n_gridpoints)  # they should also push back collectively on the origin point 
        # update pos here or after a full pass? after a pass looks much more stable and predictable.
        # coords  += vels * TIMESTEP  # timestep * velocity applied to move points
        # visu_pos(coords)
        # broadcasting always goes up dimensions
        # so shape 3, 3, 2 can be divided by a shape 3, 2 but not a 3, 3
        # this implies that you must put the most general dimesnsion first, like in math notation:
        # X sub 1, X sub 1 prime, Y sub 1, Y sub 1 prime
        #
        # broadcasting can extend an existing axis, or create a new 'higer order axis'
        # to avoid ambiguity, you have to explicitly create a new axis to show which way to broadcast.
        # broadcasting just means implicit, space-saving copy by reference into the inexistent 
        # indices of another dimension.
        
        # Accumulate elements of the Kruskal stress to evaluate later
        error_to += adjust ** 2
        times_to += tt ** 2
        pass_err += sum(abs(adjust))
        pass_err_max = max(pass_err_max, max(abs(adjust)))
        n_iter += 1
        if n_iter % 10 == 0 : 
            print "%i%% (%i iterations averaging %f seconds)" % (n_iter * 100 / n_stops, n_iter, (time.time() - t_start) / n_iter)
    accels  = forces # / n_stops   # normalize by number of total origins per pass, not total destinations
    vels    = accels * TIMESTEP  # Euler integration. Should be += but we consider that damping effectively cancels out all previous accelerations.
    coords += vels   * TIMESTEP  # Euler integration
    n_pass += 1
    pass_err = pass_err / (n_stops * n_gridpoints)
    k_energy = sum( sqrt( (vels**2).sum(axis=1) ) )
    avg_k_energy = k_energy / n_gridpoints
    # stress^2 = summed squares of all errors / summed squares of all distances.
    print "Average absolute error %i." % pass_err
    print "Maximum error in this pass %i." % pass_err_max
    print "Kinetic energy total %i average %i." % (k_energy, avg_k_energy)
    print "Stress for trips to each cell:"
    print sqrt(error_to / times_to)
    print "Total stress for this pass:", sqrt( sum(error_to) / sum(times_to) )
    # if avg_k_energy < 20 :
    #    break
    # visu_pos_b(coords)
    print "End of pass number %i." % n_pass

