#! python
#
# gs_grid
#
# loads a gtfsdb and a corresponding gsdb
# makes a grid of 100x100 points within the extents of the transit system
# links each one to its neighbours and to all stations within 1km radius
# (walk links)
# proceeds to do random full path searches and send the results to a visualization system
# by tcp-ip.
# also sends new cocordinates to morph the graph.

from graphserver.core import Graph, State, Street, WalkOptions
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.compiler.vincenty import vincenty
from graphserver.graphdb import GraphDatabase
import random
import time
from socket import *

if (False) :
    gdb    = GraphDatabase ('gsdata/trimet_13sep2009.linked.gsdb')
    gtfsdb = GTFSDatabase  ('gsdata/trimet_13sep2009.gtfsdb'); 
    g      = gdb.incarnate ()
    #g = Graph
    #g.load ('gsdata/trimet_13sep2009.gsbin', '')
else :
    gdb    = GraphDatabase ('gsdata/bart.linked.gsdb')
    gtfsdb = GTFSDatabase  ('gsdata/bart.gtfsdb'); 
    g      = gdb.incarnate ()

gridcells = []
for i in range(100) :
    for j in range (100) :
        gridcells.append(g.add_vertex("grid-%02i-%02i" % (i, j)).label)

gridcells.sort()
radius = 1000
obstruction = 1.4
min_lon, min_lat, max_lon, max_lat = gtfsdb.extent()

print min_lat, min_lon

lat1 = min_lat
for i in range(100) :
    lon1 = min_lon
    for j in range (100) :
        # make outgoing edges to neighbours
        for i2, j2 in [(i, j-1), (i, j+1), (i-1, j), (i+1, j)] :
            # print i, j, i2, j2
            try :
                # print  "grid-%02d-%02d" % (i, j), "grid-%02d-%02d" % (i2, j2)
                g.add_edge( "grid-%02d-%02d" % (i, j), "grid-%02d-%02d" % (i2, j2), Street("walk", 1000) )
            except :
                # I shall fail on mesh edges
                continue
                
        # make links to nearby transit stops
        for stop_id, stop_name, lat2, lon2 in gtfsdb.nearby_stops(lat1, lon1, radius) :
            # print stop_id
            dd = obstruction * vincenty( lat1, lon1, lat2, lon2 )
            g.add_edge( "grid-%02d-%02d" % (i, j), "sta-%s"%stop_id, Street("walk", dd) )
            g.add_edge( "sta-%s"%stop_id, "grid-%02d-%02d" % (i, j), Street("walk", dd) )
        lon1 += 0.009
        # print i, j, lat1, lon1
    lat1 += 0.004

# coords to download a texture map
print "http://maps.google.com/?ie=UTF8&ll=%f,%f&spn=%f,%f" % (min_lat, min_lon, lat1 - min_lat, lon1 - min_lon)

t0 = 1253730000
udp = socket( AF_INET, SOCK_DGRAM )
# wo = WalkOptions();
# wo.max_walk = 0x2000000000;

def transmit(result) :
    result = ''.join(result)
    udp.sendto ( result, ('localhost', 6000) )
    print "sent %d values" % len(result)

# send only > 0 then add to value in visualizer
# sparkle instead of scan (a bit ugly)
# gridcells = random.sample(gridcells, len(gridcells))

interesting = [v.label for v in g.vertices if v.label[0:5] == 'grid-' and v.degree_out > 4]
#interesting = random.sample(interesting, len(interesting))

# interesting = gridcells

max_disp = 60 * 70 # transmit everything under max_disp seconds (normalized)
for vl1 in interesting :
    result = [] 
    print "Full shortest path tree from %s:" % vl1
    g.spt_in_place( vl1,  None, State(1, t0))
    for vl2 in gridcells :
        row = int(vl2[5:7])
        col = int(vl2[8:10])
        if len(result) >= 8000 : 
            transmit(result)
            result = []
        v = g.get_vertex(vl2)
        if v.payload is not None :
            t = (v.payload.time - t0) * 255 / max_disp
            if t > 255 : t = 255            
            t = 255-t
        else :
            t = 0
        if t != 0 :
            result.append(chr(row))
            result.append(chr(col))
            result.append(chr(t))
    transmit(result)
    
