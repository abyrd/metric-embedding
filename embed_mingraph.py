#!/usr/bin/env python2.6
#
# embed_mingraph
#
# Open a graph whose shortest path metric gives the
# greatest lower bound on travel time possible with a non-time-dependent graph.
# Make an edge-list representation in arrays.
# Iteratively improve an embedding of this metric into an L1-normed real vector space.
 
from graphserver.core import Graph, ElapseTime, State, WalkOptions
from graphserver.graphdb import GraphDatabase
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB
import sys, time, os
from optparse import OptionParser         
import numpy as np

if __name__=='__main__':
    usage = """usage: python zzzz.py <assist_graph_database>"""
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        exit(-1)
        
    assist_graph_db = args[0]
    assistgraphdb = GraphDatabase( assist_graph_db )
    ag = assistgraphdb.incarnate() 

    v = ag.vertices
    e = ag.edges
    nv = len(v)
    ne = len(e)

    vl = {}

    va = np.empty((nv,), dtype=np.int32)
    ea = np.empty((ne,), dtype=np.int32)
    wa = np.empty((ne,), dtype=np.float32)
    
    va.fill(-1)
    ea.fill(-1)
    wa.fill(np.inf)

    i = 0
    for u in v :
        vl[u.label] = i
        i += 1

    i = 0
    for u in v :
        j = vl[u.label]
        va[j] = i        
        for d in v.outgoing :
            ea[i] = vl[d.tov]  
            wa[i] = d.payload.seconds
            j += 1

    print vl
    print va
    print ea
    print wa

