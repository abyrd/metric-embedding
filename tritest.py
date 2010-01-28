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

TRIP_TIME  = '08:00AM'
TRIP_DATE  = '01-29-2010'
URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromcoord/%s/tocoord/%s/date/%s/time/%s/appId/6AC697CF5EB8719DB6F3AEF0B'

gtfsdb = GTFSDatabase  ('../data/pdx/trimet-20100117.gtfsdb')

npz = np.load('../data/pdx/trimet-20100117.od_matrix.npz')
station_labels = npz['station_labels']
station_coords = npz['station_coords']
grid_dim       = npz['grid_dim']
matrix         = npz['matrix'].astype(np.int32)

matrix = (matrix + matrix.T) / 2

r = np.load('results/pdx-5d-1000i/result.npy')

station_idx = dict( zip(station_labels, range(len(station_labels))) )

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
    tmds = np.sqrt(np.sum(vec ** 2)) / 60
    
    tmat = matrix[ station_idx[ o[0] ], station_idx[ d[0] ] ] / 60.

    llo = (og[2], og[3])
    lld = (dg[2], dg[3])
    conn = httplib.HTTPConnection('developer.trimet.org')
    from_str = '%f,%f' % llo
    to_str   = '%f,%f' % lld
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
        diff = tmds - tmat
        percent = diff / tmat * 100
        print 'Travel time %03.2f (mds) %i (mat) %i (web) %f (diff) %f%%' % (tmds, tmat, tw, diff, percent)
        errors.append(diff)
    else :
        print 'Search failed.', data
    time.sleep(5)



