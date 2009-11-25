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
import os

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State, WalkOptions

SAMPLE_SIZE = 20
SHOW_GS_ROUTE = False
os.environ['TZ'] = 'US/Pacific'
time.tzset()
t0s = "Mon Nov 30 08:50:00 2009"
t0t = time.strptime(t0s)
d0s = time.strftime('%a %b %d %Y', t0t)
t0  = time.mktime(t0t)
print d0s
print time.ctime(t0), t0

gtfsdb = GTFSDatabase  ('../data/trimet-29nov2009.gtfsdb')
gdb    = GraphDatabase ('../data/trimet.gsdb'  )
g      = gdb.incarnate ()

station_labels = [s[0] for s in gtfsdb.stops()]

origins      = station_labels[:]
destinations = station_labels[:]
random.shuffle(origins)
random.shuffle(destinations)
pairs = zip(origins, destinations)[:SAMPLE_SIZE]

wo = WalkOptions() 
wo.max_walk = 800 # about 1/2 mile
wo.walking_overage = 0.2
wo.walking_speed = 0.8 # trimet uses 0.03 miles / 1 minute
wo.transfer_penalty = 60 * 15
wo.walking_reluctance = 4

errors = []
normalize = 0
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
        print 'Graphserver search failed.'
        continue
        
    if SHOW_GS_ROUTE :
        print 'Time                      ETime  Weight  IWait     TripID    Vertex label   Outgoing edge class'
        for i in range(len(vertices)) :
            v  = vertices[i]
            vp = v.payload
            # zeros here and below are initial wait
            print "%s  %5.1f  %6.1f  %5i %10s %15s  " % (time.ctime(vp.time), (vp.time - t0) / 60.0, vp.weight, 0, vp.trip_id, v.label),
            try: 
                e  = edges[i]
                ep = e.payload
                print type(ep).__name__
            except:
                print "ARRIVAL"
        print ''
    else : vp = vertices[-1].payload
    
    print "walked %i meters, %i vehicles." % (vp.dist_walked, vp.num_transfers)

    tm = (vp.time - t0 - 0) / 60.0 + 5
    
    
    URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromPlace/%s/toPlace/%s/date/%s/time/%s/walk/0.6/appId/6AC697CF5EB8719DB6F3AEF0B'
    url_time = time.strftime('%I:%M%p',  t0t) # '08:53AM'
    url_date = time.strftime('%m-%d-%Y', t0t) # '11-30-2009'
    url = URL_FORMAT % (o, d, url_date, url_time)
    conn = httplib.HTTPConnection('developer.trimet.org')
    conn.request("GET", url)
    r1 = conn.getresponse()
    #print r1.status, r1.reason
    data = r1.read()
    #print data.replace('>', '>\n')
    idx  = data.find('response success') + 18
    if data[idx] == 't' :
        idx0 = data.find('<duration>') + 10
        idx1 = data.find('</duration>')
        tw = int(data[idx0:idx1])
        conn.close()
        diff = tm - tw
        idx0 = data.find('<endTime>') + 9
        idx1 = data.find('</endTime>')
        print time.ctime(vp.time), data[idx0:idx1]
        # print 'Travel time %03.1f (gs) %i (web) %f (diff)' % (tm, tw, diff)
        errors.append(diff)
    else :
        print 'Web API search failed.' # , data
    # use rawinput to wait for enter
    time.sleep(2)

print 'Errors:', errors
print 'RMS error:', np.sqrt( np.sum(np.array(errors) ** 2.0) / SAMPLE_SIZE )
