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

usage = """usage: python zzzz.py <assist_graph_database>"""
parser = OptionParser(usage=usage)
(options, args) = parser.parse_args()
if len(args) != 1:
    parser.print_help()
    exit(-1)
    
assist_graph_db = args[0]
assistgraphdb = GraphDatabase( assist_graph_db )
# test on synthetic grid graph
#ag = assistgraphdb.incarnate() 
ag = Graph()
edges = []
fs = '%i__%i'
for x in range(20):
    for y in range(5):
        o  = fs % (x, y)
        d1 = fs % (x, y-1) 
        d2 = fs % (x, y+1) 
        d3 = fs % (x-1, y) 
        d4 = fs % (x+1, y) 
        edges.append((o, d1))
        edges.append((o, d2))
        edges.append((o, d3))
        edges.append((o, d4))

for o, d in edges:
    ag.add_vertex(o)
    ag.add_vertex(d)
    ag.add_edge(o, d, ElapseTime(1))
    
# must remove dead-end vertices since they have no slot in the edge list representation
# this also happens in non-synthetic graphs!
p = 0
n = 1
while n > 0 :
    n = 0
    p += 1
    print 'removing dead-end vertices, pass', p
    for v in ag.vertices:
        if v.degree_out == 0 : 
            print v.label
            ag.remove_vertex(v.label)
            n += 1

v = list(ag.vertices)
e = ag.edges
nv = len(v)
ne = len(e)

vl = {}

va = np.empty((nv + 1,), dtype=np.int32) # edge list offsets, extra element allows calculating edge list length for last vertex
ea = np.empty((ne,), dtype=np.int32)
wa = np.empty((ne,), dtype=np.float32)

va.fill(-1)
ea.fill(-1)
wa.fill(np.inf)

print 'making edgelist representation...'
for i, u in enumerate(v) :
    vl[u.label] = i
    #vi[i] = u.label #exactly the same as v

i = 0
for u in v :
    j = vl[u.label]
    va[j] = i
    for d in u.outgoing :
        ea[i] = vl[d.to_v.label]  
        wa[i] = d.payload.seconds
        i += 1

va[-1] = i # extra element encodes length of final vertex's edge list 
print nv, ne
print va
print ea
print wa

TASKS = nv

# updating cost array needed for parallelisation
# otherwise another thread might unset modified?
cost = np.empty((TASKS, nv), dtype=np.float32)
cost.fill(np.inf)

modify = np.empty((TASKS, nv), dtype=np.int8)
modify.fill(0)

# set cost of starting vertices to 0
# and enqueue them
for t in range(TASKS) : 
    cost[t, t]   = 0
    modify[t, t] = 1
    
print "begin"
while modify.any() :
    print '. modified costs: ', np.sum(modify, axis=1)
    for i in range(nv) :
        for t in range(TASKS) :
            if modify[t, i] == 0 : continue
            #print "task %i vertex %i checking" % (t, i)
            modify[t, i] = 0
            oc = cost[t, i]
            for ei in range(va[i], va[i+1]) :
                to = ea[ei]
                nc = wa[ei] + oc
                #print "vertex %i old %f new %f:" % (to, cost[t, to], nc),
                if cost[t, to] > nc : 
                   #print "updated"
                   cost[t, to] = nc
                   modify[t, to] = 1 
                    
            
print "end"
print cost
print modify

# embed
DIM = 2
coord  = np.random.rand(nv, DIM)
force  = np.empty ((nv, DIM), dtype=np.float32)

#for _ in range(20) :
while (True) :
    force.fill(0)
    for t in range(TASKS) :
        vector = coord - coord[t]
        dist = np.sum(np.abs(vector), axis=1)
        #dist = np.sqrt(np.sum(vector * vector, axis=1))
        adjust = (cost[t] / dist) - 1
        adjust[t] = 0 # avoid NaNs, could use nantonum
        #print 'task', t
        #print coord[t]
        #print coord
        #print vector
        #print dist
        #print cost[t]
        #print adjust
        force += (vector * adjust[:, np.newaxis])
        #print force
    coord += force / TASKS
    stress = np.sum(np.sqrt(np.sum(force * force, axis=1))) / nv
    print stress
    if stress < 0.001 : break

for c in coord :
    for d in range(DIM) :
        print c[d],
    print