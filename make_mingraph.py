#!/usr/bin/env python2.6
#
# make_mingraph
# take a normal OSM+GTFS graphserver graph
# remove its transit component
# and link all the transit stops with single edges representing the shortest time found in the GTFSDB
 
from graphserver.core import Graph, ElapseTime
from graphserver.graphdb import GraphDatabase
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
#from geotools import *
import sys, time
import ligeos as lg
from optparse import OptionParser

def copy_relevant_elements( min_graph, full_graph ) :
    seen = {}
    for n, v in enumerate(full_graph.vertices) :
        if n % 10000 == 0 : print 'vertex %d' % (n)
        if v.label[:4] == 'psv-' : continue
        for e in v.outgoing :
            cn = e.payload.__class__.__name__
            seen[cn] = 1
            if cn == 'Street' :
                t = e.payload.length # divided by speed, here implicily 1 m/sec
            elif cn == 'Link':
                t = 0 # sec
            elif cn == 'TripBoard':
                continue
            elif cn == 'TripAlight':
                continue
            elif cn == 'Crossing':
                #print "ignoring crossing:", e.from_v.label, e.to_v.label
                continue
            else :
                continue
            fl = e.from_v.label
            tl = e.to_v.label
            min_graph.add_vertex(fl)
            min_graph.add_vertex(tl)
            min_graph.add_edge(fl, tl, ElapseTime(int(t)))
            
    print seen
            
def add_min_transit(graph, gtfsdb) :
    VPREFIX = 'sta-'
    HALF_TRANSFER_COST = 3 * 60
    print 'ANALYZING'
    best_times = {}
    trips = list(gtfsdb.execute('SELECT DISTINCT trip_id, route_id FROM trips'))
    n_trips = len(trips)
    for n, (trip_id, route_id) in enumerate(trips) :
        if n % 1000 == 0 : print "TRIP %d/%d" % (n, n_trips)
        #if n > 1000 : break
        
        stop_times = []
        r = gtfsdb.execute('SELECT stop_id, arrival_time, departure_time FROM stop_times WHERE trip_id == %s ORDER BY stop_sequence' % trip_id)
        for sid, arv, dep in r :
            rsid = sid + '_r' + route_id
            graph.add_vertex(VPREFIX + rsid)
            best_times[(rsid, sid)] = HALF_TRANSFER_COST
            best_times[(sid, rsid)] = HALF_TRANSFER_COST
            stop_times.append((rsid, arv, dep))
            
        last_sid, last_arv, last_dep = stop_times[0]
        for sid, arv, dep in stop_times[1:] :
            pair1 = (last_sid, sid)
            pair2 = (sid, last_sid)
            t = arv - last_dep
            if t < 1 :
                print "NEGATIVE OR ZERO TRAVEL TIME"
                sys.exit(0)
            try:
                curr_t = best_times[pair1]
                if curr_t > t:
                    #print 'storing new time better than current one:', t, curr_t
                    best_times[pair1] = t
                    best_times[pair2] = t
            except KeyError:
                #print 'insert new time observation:', t, pair1, pair2
                best_times[pair1] = t
                best_times[pair2] = t
            #print last_sid, "->", sid, 
            last_sid, last_dep = sid, dep
    print 'ADDING'
    n_edges = len(best_times)
    print best_times
    for n, ((v1, v2), t) in enumerate(best_times.iteritems()) :
        if n % 1000 == 0 : print "EDGE %d/%d" % (n, n_edges)
        v1 = VPREFIX + v1
        v2 = VPREFIX + v2
        #graph.add_vertex(v1)
        #graph.add_vertex(v2)
        graph.add_edge(v1, v2, ElapseTime(t))
        
def main(graph_db, gtfs_db):     
    graphdb = GraphDatabase( graph_db )
    gtfsdb = GTFSDatabase( gtfs_db )
    print "loading existing graph"
    full_graph = graphdb.incarnate()
    print "copying relevant vertices and edges from full graph"
    min_graph = Graph()
    copy_relevant_elements( min_graph, full_graph )    
    print "adding minimum-time transit trip edges"
    add_min_transit( min_graph, gtfsdb )
    print "writing out new graph to database"
    min_graphdb = GraphDatabase( 'min_graph.gdb', overwrite=True )
    min_graphdb.populate(min_graph)
    print "DONE."
    sys.exit(0)    
            

if __name__=='__main__':
    usage = """usage: python zzzz.py <graph_database> <gtfs_database>"""
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        exit(-1)
        
    graph_db = args[0]
    gtfs_db  = args[1]    
    main(graph_db, gtfs_db)
