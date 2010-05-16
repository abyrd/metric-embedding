#!/usr/bin/env python
#
# Test which connects to Tri Met web services to check results
# version for working directly with Graphserver SPTs
#
# Please be nice and do not hit Tri-met's web server too fast!

import numpy as np
import random
import time
import httplib
import os

from BeautifulSoup import BeautifulStoneSoup

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb   import OSMDB
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State, WalkOptions

from PIL import Image

SAMPLE_SIZE = 40
SHOW_GS_ROUTE = True
os.environ['TZ'] = 'US/Pacific'
time.tzset()
t0s = "Mon May 17 08:50:00 2010"
t0t = time.strptime(t0s)
d0s = time.strftime('%a %b %d %Y', t0t)
t0  = time.mktime(t0t)
print 'search date: ', d0s
print 'search time: ', time.ctime(t0), t0

gtfsdb = GTFSDatabase  ('/Users/andrew/devel/data/trimet.gtfsdb')
gdb    = GraphDatabase ('/Users/andrew/devel/data/test.gdb'  )
osmdb  = OSMDB         ('/Users/andrew/devel/data/test.osmdb'  )
g      = gdb.incarnate ()

wo = WalkOptions() 
wo.max_walk = 2000 
wo.walking_overage = 0.0
wo.walking_speed = 1.0 # trimet uses 0.03 miles / 1 minute - but it uses straight line distance as well
wo.transfer_penalty = 99999
wo.walking_reluctance = 1
wo.max_transfers = 0
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
station_vertices = [e[0] for e in gtfsdb.execute("SELECT DISTINCT osm_vertex FROM osm_links")]

origins = ['sta-%s' % s[0] for s in gtfsdb.stops()]
random.shuffle(origins)
close_stations = {}
for e in grid : close_stations[e[2]] = []
for o in origins : 
    print o
    spt = g.shortest_path_tree( o, None, State(1, t0), wo_foot )
    for (x, y, vertex) in grid :
        
    spt = g.shortest_path_tree( o, None, State(1, t0), wo )
    if spt == None : continue
    print "saving image..."
    im = Image.new("L", (max_x, max_y))
    for (x, y, vertex) in grid :
        v = spt.get_vertex('osm-%s'%vertex)
        if v != None : 
            c = ((v.best_state.time - t0) / (120. * 60)) * 255
            im.putpixel((x, max_y - y - 1), 255 - c)
   
    # be careful that img dir exists
    im.save('img/%s.png' % (o))
    spt.destroy()
