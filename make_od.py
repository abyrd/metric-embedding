#!/usr/bin/env python
#
# Make OD matrix between all unique station linking points
# With additional information on station to grid distances

import numpy as np
import time
import os

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb   import OSMDB
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State, WalkOptions

from PIL import Image
from multiprocessing import Pool

os.environ['TZ'] = 'US/Pacific'
time.tzset()
t0s = "Mon May 17 08:50:00 2010"
t0t = time.strptime(t0s)
d0s = time.strftime('%a %b %d %Y', t0t)
t0  = time.mktime(t0t)
print 'search date: ', d0s
print 'search time: ', time.ctime(t0), t0

gtfsdb = GTFSDatabase  ('./trimet.gtfsdb')
gdb    = GraphDatabase ('./test.gdb'  )
osmdb  = OSMDB         ('./testgrid.osmdb'  )
g      = gdb.incarnate ()

wo = WalkOptions() 
wo.max_walk = 2000 
wo.walking_overage = 0.0
wo.walking_speed = 1.0 # trimet uses 0.03 miles / 1 minute - but it uses straight line distance as well
wo.transfer_penalty = 99999
wo.walking_reluctance = 1
wo.max_transfers = 0
# make much higher?
wo.transfer_slack = 60 * 5
wo_foot = wo

wo = WalkOptions() 
wo.max_walk = 2000 
wo.walking_overage = 0.0
wo.walking_speed = 1.0 # trimet uses 0.03 miles / 1 minute - but it uses straight line distance as well
wo.transfer_penalty = 60 * 10
wo.walking_reluctance = 1.5
wo.max_transfers = 5
wo.transfer_slack = 60 * 4
wo_transit = wo

print "Fetching grid from OSMDB..."
grid = list(osmdb.execute("SELECT x, y, vertex FROM grid"))
max_x, max_y = osmdb.execute("SELECT max(x), max(y) FROM grid").next()

print "Finding unique GTFS station linking points..."
stop_vertices = [e[0] for e in gtfsdb.execute("SELECT DISTINCT osm_vertex FROM osm_links")]
n_stop_vertices = len(stop_vertices)
matrix_indices = list(enumerate(stop_vertices))

matrix = [[None for _ in range(n_stop_vertices)] for _ in range(n_stop_vertices)]
close_stations = [[[] for _ in range(max_y)] for _ in range(max_x)]

f = open('output_%d' % os.getpid(), 'w')

for oi, ov in matrix_indices : 

def do_one (x):
    oi, ov = x    
    # for testing    
    # if oi > 5 : break
    # do a transit + walk search
    print oi, ov, "transit"    
    f.write('O %s\n' % ov)
    spt = g.shortest_path_tree( 'osm-%s' % ov, None, State(1, t0), wo_transit )
    if spt == None : 
        print oi, ov, "spt was None."
        continue
    for di, dv in matrix_indices : 
        v = spt.get_vertex('osm-%s' % dv)
        #if v is None :
        #    matrix[(oi, di)] = None
        if v is not None :
            #matrix[oi][di] = v.best_state.time - t0
            f.write('T %s %d\n' % (dv, int(v.best_state.time - t0)) )
    spt.destroy()

#    print "saving grid image..."
#    im = Image.new("L", (max_x, max_y))
#    for (x, y, vertex) in grid :
#        v = spt.get_vertex('osm-%s'%vertex)
#        if v != None : 
#            c = ((v.best_state.time - t0) / (120. * 60)) * 255
#            im.putpixel((x, max_y - y - 1), 255 - c)
   
    # be careful that img dir exists
#    im.save('img/%03d_%s.png' % (oi, ov))

    # then do a walk-only search
    print oi, ov, "walk"    
    spt = g.shortest_path_tree( 'osm-%s' % ov, None, State(1, t0), wo_foot )
    for (x, y, vertex) in grid :        
        v = spt.get_vertex('osm-%s' % vertex)
        if v is not None :
            vp = v.best_state
            t = vp.time - t0
            if t < 3600 and vp.num_transfers == 0 :
                f.write('W %s %d\n' % (vertex, int(t)) )
                # (close_stations[x][y]).append((oi, t))
                # vp.narrate()
    spt.destroy()

f.close()
