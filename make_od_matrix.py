#!/usr/bin/env python2.3
#
#  make od matrix
#  

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State
from pylab import *
from numpy import *
from random import shuffle
import time 
from socket import *
import geotools
from math import ceil
import struct
import pyproj

def cplot(Z) :
    imshow(Z, cmap=cm.gray, origin='bottom')
    levels = array([15, 30, 60, 90])
    CS = contour(Z, levels, linewidths=2)
    clabel(CS, inline=1, fmt='%d', fontsize=14)
    show()

#gtfsdb = GTFSDatabase  ( '../gsdata/bart.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/bart.linked.gsdb' )
gtfsdb = GTFSDatabase  ( '../gsdata/trimet_13sep2009.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.nolink.gsdb' )
gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.hpm.linked.gsdb' )
g = gdb.incarnate()

t0 = 1253800000

station_labels, matrix = geotools.od_matrix(g, t0)

n_stations = len(station_labels)
min_lon, min_lat, max_lon, max_lat = gtfsdb.extent()
proj = pyproj.Proj( proj='sinu', ellps='WGS84' )
geod = pyproj.Geod( ellps='WGS84' )
min_lon, min_lat, arc_dist = geod.fwd(min_lon, min_lat, 180+45, 1000)
min_x,   min_y   = proj( min_lon, min_lat )
proj = pyproj.Proj( proj='sinu', ellps='WGS84', lon_0=min_lon, y_0=-min_y, m=0.1 )

station_coords = empty( (len(station_labels), 2) )
for i, label in enumerate(station_labels) :
    stop_id, stop_name, lat, lon = gtfsdb.stop(label[4:])
    station_coords[i] = proj( lon, lat )

station_coords /= 100
grid_size = (station_coords.max(axis=0) + (10, 10)).astype(int)

# precompute distance mask matrix here, use view later
print "precomputing distance matrix..."
distances_x = arange(grid_size[0] * 2) - grid_size[0]
distances_y = arange(grid_size[1] * 2) - grid_size[1]
distances   = sqrt((distances_x**2)[:, newaxis] + (distances_y ** 2)[newaxis, :]) * 100 / 1.3  # m/sec, so actually gives times

# station at random
for origin_idx in range( 2 ) :  #n_stations) :
    print origin_idx
    result = matrix[origin_idx]
    grid = ones( grid_size ) * inf # actually should be set to times from point of departure 
    temp_grid = empty( grid_size )
    for dest_idx in range(n_stations) :
        # set up time window
        dest_coords = station_coords[dest_idx]
        x_1 = round(2 * grid_size[0] - dest_coords[0])
        x_0 = x_1 - grid_size[0]
        y_1 = round(2 * grid_size[1] - dest_coords[1])
        y_0 = y_1 - grid_size[1]
        add(distances[x_0:x_1, y_0:y_1], result[dest_idx], temp_grid)
        minimum(grid, temp_grid , grid)  # destination variabe to do operation in-place
    cplot((grid / 60).T)

# save("od_matrix.npy", matrix)
# attention, station labels is not an ndarray. maybe store lat/lon information, and grid information?
savez("od_matrix.npz", station_labels = station_labels, station_coords = station_coords, matrix=matrix)

# 1d arrays for xccord and yccord (ranges); recenter, square, and broadcast them to calculate distances 