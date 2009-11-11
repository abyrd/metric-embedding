#!/usr//bin/env python
#
# Test which connects to Tri Met web services to check results
#

import numpy as np
import pylab as pl

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase

gtfsdb = GTFSDatabase  ('../gsdata/trimet_13sep2009.gtfsdb')
gdb    = GraphDatabase ('../gsdata/trimet_13sep2009.linked.gsdb')

t0 = 1253800000
g  = gdb.incarnate()

station_labels   = [v.label for v in station_vertices]
n_stations = len(station_vertices)
    
for i, label in enumerate(station_labels) :
    stop_id, stop_name, lat, lon = gtfsdb.stop(label[4:])
    print stop_id, stop_name, lat,lon

r = np.open('result.npy')


