#!/usr//bin/env python
#
# Test which connects to Tri Met web services to check results
#

import numpy as np
import pylab as pl
import random
import time
import httplib

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase

TRIP_TIME  = '07:46AM'
TRIP_DATE  = '11-19-2009'
URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromcoord/%s/tocoord/%s/date/%s/time/%s/appId/6AC697CF5EB8719DB6F3AEF0B'

gtfsdb = GTFSDatabase  ('../gsdata/trimet_13sep2009.gtfsdb')
gdb    = GraphDatabase ('../gsdata/trimet_13sep2009.linked.gsdb')

t0 = 1253800000
#g  = gdb.incarnate()

npz = np.load('data/od_matrix_trimet_linked.npz')
station_labels = npz['station_labels']
station_coords = npz['station_coords']
grid_dim       = npz['grid_dim']
matrix         = npz['matrix'].astype(np.int32)

r = np.load('data/result.PDX-300-iterations.npy')

origins = list(zip(station_labels, np.round(station_coords).astype(np.int32)))
destinations = origins[:] # copy 1 level

random.shuffle(origins)
random.shuffle(destinations)

pairs = zip(origins, destinations)
errors = []

for o, d in pairs : 
    og = gtfsdb.stop(o[0][4:])
    dg = gtfsdb.stop(d[0][4:])
    #print 'Origin (mds, gtfs)'
    #print o, og
    #print 'Destination (mds, gtfs)'
    #print d, dg
    print 'from %s to %s (%f, %f) -> (%f, %f)' % (og[1], dg[1], og[2], og[3], dg[2], dg[3])

    p1 = r[ o[1][0], o[1][1] ]
    p2 = r[ d[1][0], d[1][1] ]
    vec = p1 - p2
    tm = np.sqrt(np.sum(vec ** 2)) / 60
    
    llo = (og[2], og[3])
    lld = (dg[2], dg[3])
    conn = httplib.HTTPConnection('developer.trimet.org')
    from_str = '%f,%f' % llo
    to_str   = '%f,%f' % lld
    conn.request("GET", URL_FORMAT % (from_str, to_str, TRIP_DATE, TRIP_TIME) )
    r1 = conn.getresponse()
    #print r1.status, r1.reason
    data = r1.read()
    print data
    idx  = data.find('response success') + 18
    if data[idx] == 't' :
        idx0 = data.find('<duration>') + 10
        idx1 = data.find('</duration>')
        tw = int(data[idx0:idx1])
        conn.close()
        diff = tm - tw
        print 'Travel time %03.2f (mds) %i (web) %f (diff)' % (tm, tw, diff)
        errors.append(diff)
    else :
        print 'Search failed.', data
    time.sleep(5)



