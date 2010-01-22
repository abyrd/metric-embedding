#!/usr//bin/env python
#
# Test which connects to Tri Met web services to check results
# version for working directly with Graphserver SPTs, not MDS results.
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

SAMPLE_SIZE = 30
SHOW_GS_ROUTE = False
os.environ['TZ'] = 'US/Pacific'
time.tzset()
t0s = "Mon Jan 20 08:50:00 2010"
t0t = time.strptime(t0s)
d0s = time.strftime('%a %b %d %Y', t0t)
t0  = time.mktime(t0t)
print 'search date: ', d0s
print 'search time: ', time.ctime(t0), t0

gtfsdb = GTFSDatabase  ('../data/trimet-20100117.gtfsdb')
gdb    = GraphDatabase ('../data/trimet-linked-20100117.gsdb'  )
g      = gdb.incarnate ()

station_labels = [s[0] for s in gtfsdb.stops()]

origins      = station_labels[:]
destinations = station_labels[:]
random.shuffle(origins)
random.shuffle(destinations)
pairs = zip(origins, destinations)[:SAMPLE_SIZE]

wo = WalkOptions() 
wo.max_walk = 1000 # about 1/2 mile
wo.walking_overage = 0.2
wo.walking_speed = 0.8 # trimet uses 0.03 miles / 1 minute
wo.transfer_penalty = 0
wo.walking_reluctance = 1

residuals  = []
magnitudes = []
normalize = 0
for o, d in pairs : 
    og = gtfsdb.stop(o)
    dg = gtfsdb.stop(d)
    #print 'Origin (mds, gtfs)'
    #print o, og
    #print 'Destination (mds, gtfs)'
    #print d, dg
    print 'from %s to %s (%f, %f) -> (%f, %f)' % (og[0], dg[0], og[2], og[3], dg[2], dg[3])

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

    tg = vp.time
    
    
    URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromPlace/%s/toPlace/%s/date/%s/time/%s/walk/0.6/appId/6AC697CF5EB8719DB6F3AEF0B'
    url_time = time.strftime('%I:%M%p',  t0t) # '08:53AM'
    url_date = time.strftime('%m-%d-%Y', t0t) # '11-30-2009'
    url = URL_FORMAT % (o, d, url_date, url_time)
    conn = httplib.HTTPConnection('developer.trimet.org')
    conn.request("GET", url)
    r1 = conn.getresponse()
    #print r1.status, r1.reason
    data = r1.read()
    conn.close()
    #print data.replace('>', '>\n')
    idx  = data.find('response success') + 18
    if data[idx] == 't' :
        idx0 = data.find('<endTime>') + 9
        idx1 = data.find('</endTime>')
        tw = time.mktime(time.strptime(d0s + ' ' + data[idx0:idx1], '%a %b %d %Y %I:%M %p'))
        print 'gs:   ', time.ctime(tg) 
        print 'web:  ', time.ctime(tw) 
        diff = (tw - tg) / 60.0
        print 'diff: %04.1f min' % diff
        residuals.append(diff)
        magnitudes.append((vp.time - t0) / 60.0)
    else :
        print 'Web API search failed.' # , data
    # use rawinput to wait for enter
    time.sleep(2)

print 'Residuals, magnitudes: ', zip( residuals, magnitudes )
print 'Normalized errors (percent): ', [ e[0] / e[1] * 100 for e in zip( residuals, magnitudes ) ]
print 
print 'RMS error:', np.sqrt( np.sum(np.array(residuals) ** 2.0) / SAMPLE_SIZE )
