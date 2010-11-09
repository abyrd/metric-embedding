#!/usr/bin/env python2.6
#
# make_mingraph
# take a normal OSM+GTFS graphserver graph
# remove its transit component
# and link all the transit stops with single edges representing the shortest time found in the GTFSDB
 
from graphserver.core import Graph, ElapseTime, State, WalkOptions
from graphserver.graphdb import GraphDatabase
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB
import sys, time, os
from optparse import OptionParser         

if __name__=='__main__':
    usage = """usage: python zzzz.py <graph_database> <assist_graph_database> <osm_database>"""
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 3:
        parser.print_help()
        exit(-1)
        
    graph_db = args[0]
    assist_graph_db = args[1]
    osm_db  = args[2]    
    graphdb = GraphDatabase( graph_db )
    assistgraphdb = GraphDatabase( assist_graph_db )
    osmdb = OSMDB( osm_db )
    g = graphdb.incarnate()
    ag = assistgraphdb.incarnate() 

    nodes = {}
    for id, tags, lat, lon, endnode_refs in osmdb.nodes():
        nodes[id] = (lat, lon)

    os.environ['TZ'] = 'US/Pacific'
    time.tzset()
    t0s = "Thu Jul 01 08:50:00 2010"
    t0t = time.strptime(t0s)
    d0s = time.strftime('%a %b %d %Y', t0t)
    t0  = time.mktime(t0t)
    print 'search date: ', d0s
    print 'search time: ', time.ctime(t0), t0

    wo = WalkOptions() 
    wo.max_walk = 2000 
    wo.walking_overage = 0.0
    wo.walking_speed = 1.0 
    wo.transfer_penalty = 60 * 5
    wo.walking_reluctance = 1.0
    wo.max_transfers = 4

    orig = 'osm-40530493' #east side
    dest = 'osm-40542000' #west side
    assist_spt = ag.shortest_path_tree(dest, None, State(1, 0))
    #spt = g.shortest_path_tree_assist(assist_spt, orig, dest, State(1, t0), wo)
    spt = g.shortest_path_tree(orig, dest, State(1, t0), wo)
    print 'number of vertices in final spt:', len(spt.vertices)
    for v in spt.vertices:
        vl = v.label
        if vl[:4] == 'osm-':
            av = assist_spt.get_vertex(vl)
            c = nodes[vl[4:]]
            t = (v.state.time - t0) / 60.0
            at = (av.state.time) / 60.0
            if t > 90 : t = 91
            if at > 90 : at = 91
            print '%s, %f, %f, %f, %f' % (vl, c[0], c[1], t, at)
    sys.exit(0)    
