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
from PyQt4 import QtCore, QtGui

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State, WalkOptions
from pylab import *
from numpy import *
from random import shuffle
import time, os
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
#gtfsdb = GTFSDatabase  ( '../gsdata/trimet_13sep2009.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.nolink.gsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.hpm.linked.gsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.linked.gsdb' )

class MakeMatrixThread(QtCore.QThread) :
    def __init__(self, parent = None) :
        QtCore.QThread.__init__(self, parent)
        self.exiting = False
    
    def makeMatrix(self, gtfsdb, gdb) :
        self.gtfsdb = str(gtfsdb) 
        self.gdb    = str(gdb)
        self.start()
        
    def run(self) :
        self.gtfsdb = GTFSDatabase  ( self.gtfsdb ) 
        self.gdb    = GraphDatabase ( self.gdb    )
        # Calculate an origin-destination matrix for the graph's stations
        print "Loading Graphserver DB..."
        self.emit( QtCore.SIGNAL( 'say(QString)' ), QtCore.QString( 'Loading SQLite Graphserver graph...') )
        g = self.gdb.incarnate()
        
        # Set up distance-preserving projection system
        # Make a grid over the study area and save its geographic coordinates 
        MARGIN = 8000 # meters beyond all stations, diagonally
        min_lon, min_lat, max_lon, max_lat = self.gtfsdb.extent()
        geod = pyproj.Geod( ellps='WGS84' )
        min_lon, min_lat, arc_dist = geod.fwd(min_lon, min_lat, 180+45, MARGIN)
        max_lon, max_lat, arc_dist = geod.fwd(max_lon, max_lat,     45, MARGIN)
        proj = pyproj.Proj( proj='sinu', ellps='WGS84' )
        min_x, min_y = proj( min_lon, min_lat )
        proj = pyproj.Proj( proj='sinu', ellps='WGS84', lon_0=min_lon, y_0=-min_y ) # why doesn't m parameter work for scaling by 100?
        grid_dim = array( proj( max_lon, max_lat ), dtype=int32 ) / 100
        max_x, max_y = grid_dim
        print "\nMaking grid with dimesions: ", max_x, max_y
        self.emit( QtCore.SIGNAL( 'say(QString)' ), QtCore.QString( 'Making %i by %i grid...' % ( max_x, max_y ) ) )
        # later, use reshape/flat to switch between 1d and 2d array representation
        grid_latlon = empty( (max_x, max_y, 2), dtype=float32 )
        for y in range( 0, max_y ) :
            self.emit( QtCore.SIGNAL( 'progress(int, int)' ), y, max_y )
            for x in range( 0, max_x ) :
                # inverse project meters to lat/lon
                grid_latlon[x, y] = proj ( x * 100, y * 100, inverse=True)

        station_vertices = [v for v in g.vertices if v.label[0:4] == 'sta-']
        station_labels   = [v.label for v in station_vertices]
        n_stations = len(station_vertices)
        print 'Finding station coordinates...'
        self.emit( QtCore.SIGNAL( 'say(QString)' ), QtCore.QString( 'Projecting station coordinates...'  ) )
        station_coords = empty( (n_stations, 2), dtype=float32 )
        for i, label in enumerate(station_labels) :
            stop_id, stop_name, lat, lon = self.gtfsdb.stop(label[4:])
            station_coords[i] = proj( lon, lat )
            if i % 20 == 0 : self.emit( QtCore.SIGNAL( 'progress(int, int)' ), i, n_stations )
        station_coords /= 100
        
        # ELIMINATE STATIONS WITH SAME INTEGRAL COORDINATES
        #self.emit( QtCore.SIGNAL( 'say(QString)' ), QtCore.QString( 'Eliminating equivalent stations...'  ) )
        #while len(station_coords) > 0 :
        #    coord = 
        #    mask = station_coords != station_coords[i]
        #    station_coords = station_coords[mask]
        # newer version follows
        #self.emit( QtCore.SIGNAL( 'say(QString)' ), QtCore.QString( 'Eliminating equivalent stations...' ) )
        #station_labels = np.array(station_labels)
        #station_coords_new = []
        #station_labels_new = []
        #while len(station_coords) > 0 :
        #    coord = np.round(station_coords[0])
        #    minIdx = np.argmin(np.sum(np.abs(station_coords - coord), axis=1))
        #    station_labels_new.append(station_labels[minIdx])
        #    station_coords_new.append(station_coords[minIdx])
        #    mask = np.any(np.round(station_coords) != coord, axis=1)
        #    #print mask
        #    #print len(station_coords)
        #    #print coord
        #    #print station_coords[np.logical_not(mask)]
        #    station_coords = station_coords[mask][:]
        #    station_labels = station_labels[mask][:]
        #    self.emit( QtCore.SIGNAL( 'progress(int, int)' ), n_stations - len(station_coords_new), n_stations )
        #
        #station_labels = station_labels_new
        #station_coords = station_coords_new
        #station_vertices = [g.get_vertex(slabel) for slabel in station_labels_new]
        #n_stations = len(station_labels)
        #print len(station_labels), len(station_coords), len(station_vertices)
        
        print "Making OD matrix..."
        os.environ['TZ'] = 'US/Pacific'
        time.tzset()
        t0s = "Fri Jan 22 08:00:00 2010"
        t0t = time.strptime(t0s)
        d0s = time.strftime('%a %b %d %Y', t0t)
        t0  = int(time.mktime(t0t))
        print 'search date: ', d0s
        print 'search time: ', time.ctime(t0), t0

        wo = WalkOptions() 
        wo.max_walk = 20000 
        wo.walking_overage = 0.1
        wo.walking_speed = 0.8 # trimet uses 0.03 miles / 1 minute
        wo.transfer_penalty = 60 * 10
        wo.walking_reluctance = 2
        wo.max_transfers = 40
        wo.transfer_slack = 60 * 4

        matrix     = zeros( (n_stations, n_stations), dtype=float ) #dtype could be uint16 except that there are inf's ---- why?
        colortable = [QtGui.QColor(i, i, i).rgb() for i in range(256)]
        colortable[254] = QtGui.QColor(050, 128, 050).rgb()  
        colortable[255] = QtGui.QColor(255, 050, 050).rgb()  
        matrixImage = QtGui.QImage(max_x, max_y, QtGui.QImage.Format_Indexed8)
        matrixImage.fill(0)
        matrixImage.setColorTable(colortable)
        for origin_idx in range(n_stations) :
            sys.stdout.write( "\rProcessing %i / %i ..." % (origin_idx, n_stations) )
            sys.stdout.flush()
            self.emit( QtCore.SIGNAL( 'say(QString)' ), QtCore.QString( 'Making OD matrix (station %i/%i)...' % (origin_idx, n_stations) ) )
            self.emit( QtCore.SIGNAL( 'progress(int, int)' ), origin_idx, n_stations )
                        
            origin_label = station_labels[origin_idx]
            #g.spt_in_place(origin_label, None, State(1, t0), wo)
            spt = g.shortest_path_tree( origin_label, None, State(1, t0), wo )
            for dest_idx in range(n_stations) :
                dest_label = station_labels[dest_idx]
                dest_vertex = spt.get_vertex(dest_label)
                # first board time should be subtracted here
                # if dest_vertex.payload is None :
                if dest_vertex is None :
                    print "Unreachable vertex. Set to infinity.", dest_idx, dest_label
                    delta_t = inf
                else :
                    # delta_t = dest_vertex.best_state.time - t0 
                    bs = dest_vertex.best_state
                    delta_t = bs.time - t0 - bs.initial_wait
                if delta_t < 0:
                    print "Negative trip time; set to 0."
                    delta_t = 0
                
                matrix[origin_idx, dest_idx] = delta_t

                #sys.stdout.write( '%i %i\n' % (delta_t, dest_vertex.payload.initial_wait) )
                #sys.stdout.flush()
                #time.sleep(0.5)
                
                if   dest_idx == origin_idx - 1 : color = 254
                elif dest_idx == origin_idx :     color = 255
                else :
                    color = 253 - delta_t * 3 / 60
                    if color < 0 : color = 0
                coord = station_coords[dest_idx]
                x = coord[0]
                y = coord[1]
                if color >= 254 : 
                    for x2 in range(x-1, x+2) :
                        for y2 in range(y-1, y+2) :
                            matrixImage.setPixel(x2, y2, color)    
                else :
                    matrixImage.setPixel(x, y, color)    

            self.emit( QtCore.SIGNAL( 'display(QImage)' ), matrixImage )
            spt.destroy()
            #time.sleep(1)
                    
        print x * y, "points, done."
        
        self.emit( QtCore.SIGNAL( 'say(QString)' ), QtCore.QString('Saving as gzipped numpy ndarrays...') )
        
        savez("od_matrix.npz", station_labels=station_labels, station_coords=station_coords, grid_dim=grid_dim, grid_latlon=grid_latlon, matrix=matrix )
        # cannot save station distance lists because they are not an array.



















