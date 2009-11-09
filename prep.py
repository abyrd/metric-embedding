#!/usr/bin/env python2.6
#
#  analyze a GTFSDB / GraphserverDB pair as follows:
#  
#  make a grid of evenly-spaced points over the GTFS's geographic area
#  store the lat/lon coordinates of these points in an array
#  
#  make an OD travel time matrix for a given time of day. using this matrix's station indices:
#  make an array of station labels
#  make an array of station lat/long
#  make an array of the closest gridpoint to each station
#  
#  then find the closest stations to each gridpoint.
#  save (station, distance) pairs for all gridpoints
#
#  store all this information in a file on disk for later use
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

from math import sin, cos, tan, atan, degrees, radians, pi, sqrt, atan2, asin, ceil
from graphserver.core import Graph, Street, State
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from numpy import zeros, inf
import sys

# Define input files.
# The GTFS database is used for lat/lon coordinates and geographic extent
# The Graphserver database is used for the OD matrix computation

#gtfsdb = GTFSDatabase  ( '../gsdata/bart.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/bart.linked.gsdb' )
gtfsdb = GTFSDatabase  ( '../gsdata/trimet_13sep2009.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.nolink.gsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.hpm.linked.gsdb' )
gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.linked.gsdb' )

# Calculate an origin-destination matrix for the graph's stations
print "Loading Graphserver DB..."
g = gdb.incarnate()
print "Making OD matrix..."
t0 = 1253800000
station_vertices = [v for v in g.vertices if v.label[0:4] == 'sta-']
station_labels   = [v.label for v in station_vertices]
n_stations = len(station_vertices)
matrix     = zeros( (n_stations, n_stations), dtype=float ) #dtype could be uint16 except that there are inf's ---- why?
for origin_idx in range(n_stations) :
    sys.stdout.write( "\rProcessing %i / %i ..." % (origin_idx, n_stations) )
    sys.stdout.flush()
    origin_label = station_labels[origin_idx]
    g.spt_in_place(origin_label, None, State(1, t0))
    for dest_idx in range(n_stations) :
        dest_vertex = station_vertices[dest_idx]
        # first board time should be subtracted here
        if dest_vertex.payload is None :
            print "Unreachable vertex. Set to infinity."
            delta_t = inf
        else :
            delta_t = dest_vertex.payload.time - t0
            
        if delta_t < 0:
            print "Negative trip time; set to 0."
            delta_t = 0
            
        matrix[origin_idx, dest_idx] = delta_t

# Set up distance-preserving projection system
# Make a grid over the study area and save its geographic coordinates 
MARGIN = 8000 # meters beyond all stations, diagonally
min_lon, min_lat, max_lon, max_lat = gtfsdb.extent()
geod = pyproj.Geod( ellps='WGS84' )
min_lon, min_lat, arc_dist = geod.fwd(min_lon, min_lat, 180+45, MARGIN)
max_lon, max_lat, arc_dist = geod.fwd(max_lon, max_lat,     45, MARGIN)
proj = pyproj.Proj( proj='sinu', ellps='WGS84' )
min_x, min_y = proj( min_lon, min_lat )
proj = pyproj.Proj( proj='sinu', ellps='WGS84', lon_0=min_lon, y_0=-min_y ) # why doesn't m parameter work for scaling by 100?
grid_dim = array( proj( max_lon, max_lat ), dtype=int32 ) / 100
max_x, max_y = grid_dim
print "\nMaking grid with dimesions: ", max_x, max_y
# later, use reshape/flat to switch between 1d and 2d array representation
grid_latlon = empty( (max_x, max_y, 2), dtype=float32 )
for y in range( 0, max_y ) :
    for x in range( 0, max_x ) :
        # inverse project meters to lat/lon
        grid_latlon[x, y] = proj ( x * 100, y * 100, inverse=True)
             
print x * y, "points, done."

print "Saving station coordinates..."
station_coords = empty( (n_stations, 2), dtype=float32 )
for i, label in enumerate(station_labels) :
    stop_id, stop_name, lat, lon = gtfsdb.stop(label[4:])
    station_coords[i] = proj( lon, lat )

station_coords /= 100

savez("od_matrix.npz", station_labels=station_labels, station_coords=station_coords, grid_dim=grid_dim, grid_latlon=grid_latlon, matrix=matrix )
# cannot save station distance lists because they are not an array.



















