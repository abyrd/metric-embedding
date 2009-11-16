#!/usr//bin/env python
#
# Test which connects to Tri Met web services to check results
# version for working directly with OD matrices, not MDS results.
#

import numpy as np
import pylab as pl
import random
import time
import httplib

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State, WalkOptions

t0 = 1259600000
print time.ctime(t0) # Mon Nov 30 17:53:20 2009
TRIP_TIME  = '08:53AM'
TRIP_DATE  = '11-30-2009'
URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromcoord/%s/tocoord/%s/date/%s/time/%s/walk/0.999/appId/6AC697CF5EB8719DB6F3AEF0B'

gtfsdb = GTFSDatabase  ('../gsdata/trimet-29nov2009.gtfsdb')
gdb    = GraphDatabase ('../gsdata/trimet-29nov2009.gsdb'  )
g      = gdb.incarnate ()

station_labels = [s[0] for s in gtfsdb.stops()]

origins      = station_labels[:]
destinations = station_labels[:]
random.shuffle(origins)
random.shuffle(destinations)
pairs = zip(origins, destinations)

wo = WalkOptions() 
wo.max_walk = 1600
wo.walking_overage = 100
wo.walking_speed = 1.3

errors = []
for o, d in pairs : 
    og = gtfsdb.stop(o)
    dg = gtfsdb.stop(d)
    #print 'Origin (mds, gtfs)'
    #print o, og
    #print 'Destination (mds, gtfs)'
    #print d, dg
    print 'from %s to %s (%f, %f) -> (%f, %f)' % (og[1], dg[1], og[2], og[3], dg[2], dg[3])

    #replace
    spt = g.shortest_path_tree( 'sta-' + o, 'sta-' + d, State(1, t0), wo )
    vertices, edges = spt.path( 'sta-' + d )
    if vertices is None:
        print 'No path found in GS.'
        continue
        
    for i in range(len(vertices)) :
        v  = vertices[i]
        vp = v.payload
        print "%s %3i %04i %10s %15s" % (time.ctime(vp.time), vp.weight / 60, vp.initial_wait, vp.trip_id, v.label),
        try: 
            e  = edges[i]
            ep = e.payload
            print type(ep).__name__
        except:
            print "ARRIVAL"

    print "\nwalked %i meters, %i vehicles." % (vp.dist_walked, vp.num_transfers)

    tm = (vp.time - t0 - vp.initial_wait) / 60.
    
    from_str = '%f,%f' % (og[2], og[3])
    to_str   = '%f,%f' % (dg[2], dg[3])
    conn = httplib.HTTPConnection('developer.trimet.org')
    conn.request("GET", URL_FORMAT % (from_str, to_str, TRIP_DATE, TRIP_TIME) )
    r1 = conn.getresponse()
    #print r1.status, r1.reason
    data = r1.read()
    #print data
    idx  = data.find('response success') + 18
    if data[idx] == 't' :
        idx0 = data.find('<duration>') + 10
        idx1 = data.find('</duration>')
        tw = int(data[idx0:idx1])
        conn.close()
        diff = tm - tw
        print 'Travel time %03.2f (gs) %i (web) %f (diff)' % (tm, tw, diff)
        errors.append(diff)
    else :
        print 'Search failed.', data
    time.sleep(5)



