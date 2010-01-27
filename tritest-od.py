#!/usr//bin/env python
#
# Test which connects to Tri Met web services to check results
# version for working directly with OD matrices, not MDS results.
#

import numpy as np
import random
import time
import httplib

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase

TRIP_TIME  = '08:00AM'
TRIP_DATE  = '01-29-2010'
URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromcoord/%s/tocoord/%s/date/%s/time/%s/walk/0.999/appId/6AC697CF5EB8719DB6F3AEF0B'

print 'search date: ', TRIP_DATE
print 'search time: ', TRIP_TIME

gtfsdb = GTFSDatabase  ('../data/pdx/trimet-20100117.gtfsdb')
npz = np.load('../data/pdx/trimet-20100117.od_matrix.npz')

station_labels = npz['station_labels']
matrix         = npz['matrix'].astype(np.int32)

station_idx = dict( zip(station_labels, range(len(station_labels))) )

origins = list(station_labels)
destinations = origins[:] # copy 1 level

random.shuffle(origins)
random.shuffle(destinations)

pairs = zip(origins, destinations)
errors = []

for o, d in pairs : 
    og = gtfsdb.stop(o[4:])
    dg = gtfsdb.stop(d[4:])
    #print 'Origin (mds, gtfs)'
    #print o, og
    #print 'Destination (mds, gtfs)'
    #print d, dg
    print 'from %s to %s (%f, %f) -> (%f, %f)' % (og[1], dg[1], og[2], og[3], dg[2], dg[3])

    tm = matrix[station_idx[o], station_idx[d]] / 60.
    
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
        print 'Travel time %03.2f (matrix) %i (web) %f (diff)' % (tm, tw, diff)
        errors.append(diff)
    else :
        print 'Search failed.', data
    time.sleep(5)



