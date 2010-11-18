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

import pycuda.driver   as cuda
import pycuda.autoinit 
from   pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import math

def synthetic(X, Y) :
    g = Graph()
    edges = []
    fs = '%i_%i'
    for x in range(X):
        for y in range(Y):
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
        g.add_vertex(o)
        g.add_vertex(d)
        g.add_edge(o, d, ElapseTime(1))
    return g

def net(x, y) :
    g = Graph()
    edges = []
    fs = '%i_%i'
    for x0 in range(x):
        for y0 in range(y):
            for x1 in range(x):
                for y1 in range(y):
                    if x0 == x1 and y0 == y1 : continue
                    dx = 10*(x1-x0)
                    dy = 10*(y1-y0)
                    d = math.sqrt(dx*dx + dy*dy)
                    l0 = fs % (x0, y0)
                    l1 = fs % (x1, y1)
                    g.add_vertex(l0)
                    g.add_vertex(l1)
                    g.add_edge(l0, l1, ElapseTime(int(d)))
    return g

def remove_deadends(g) :
    # must remove dead-end vertices since they have no slot in the edge list representation
    # this also happens in non-synthetic graphs due to OSM dead ends!
    p = 0
    n = 1
    while n > 0 :
        n = 0
        p += 1
        print 'removing dead-end vertices, pass', p
        for v in g.vertices:
            if v.degree_out == 0 : 
                print v.label
                g.remove_vertex(v.label)
                n += 1

def apsp_cpu () :
    # updating cost array needed for parallelisation?
    # otherwise another thread might unset modified?
    cost = np.empty((n_tasks, nv), dtype=np.float32)
    cost.fill(np.inf)
    modify = np.empty((n_tasks, nv), dtype=np.int8)
    modify.fill(0)

    # set cost of starting vertices to 0 and enqueue them
    for t in range(n_tasks) : 
        cost[t, t]   = 0
        modify[t, t] = 1
        
    print "begin"
    while modify.any() :
        print '   modified costs: ', np.sum(modify, axis=1)
        for i in range(nv) :
            for t in range(n_tasks) :
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
    return cost

def apsp_gpu () :
    src = open('mingraph_kernels.cu').read()
    mod = SourceModule(src, options=["--ptxas-options=-v"])
    scatter = mod.get_function("scatter")
    vertex = gpuarray.to_gpu(va)
    edge   = gpuarray.to_gpu(ea)
    weight = gpuarray.to_gpu(wa)
    cost   = np.empty((nv, n_tasks), dtype=np.int32)
    modify = np.zeros_like(cost)
    cost.fill(999)
    for i in range(n_tasks) :
        cost[i, i] = 0
        modify[i, i] = 1   
    for i in range(nv): # need better break condition
        #print cost.get()
        #print modify.get()        
        scatter(np.int32(nv), vertex, edge, weight, cuda.InOut(cost), cuda.InOut(modify), block=(nv,1,1), grid=(nv,1))
        pycuda.autoinit.context.synchronize()    
    return cost

usage = """usage: python zzzz.py <assist_graph_database>"""
parser = OptionParser(usage=usage)
(options, args) = parser.parse_args()
if len(args) != 1:
    parser.print_help()
    exit(-1)  
assist_graph_db = args[0]
assistgraphdb = GraphDatabase( assist_graph_db )
#ag = assistgraphdb.incarnate()
 
#ag = synthetic(20, 20)
ag = net(15, 20)
remove_deadends(ag)

v = list(ag.vertices)
nv = len(v)
ne = len(ag.edges)

print 'making edgelist representation...'
vl = {} # hashtable from vertex labels to indices in va.
va = np.empty((nv + 1,), dtype=np.int32) # edge list offsets, extra element allows calculating edge list length for last vertex.
ea = np.empty((ne,),     dtype=np.int32) # one entry per edge, index into va for edge destination. 
wa = np.empty((ne,),     dtype=np.int32) # one entry per edge, weight of edge.
va.fill(-1)
ea.fill(-1)
wa.fill(999)
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

n_tasks = nv
#cost = apsp_cpu()
cost = apsp_gpu()

# embed
DIM = 2
coord  = np.random.rand(nv, DIM)
force  = np.empty ((nv, DIM), dtype=np.float32)

#for _ in range(20) :
while (True) :
    force.fill(0)
    for t in range(n_tasks) :
        vector = coord - coord[t]
        # L1 metric        
        dist = np.sum(np.abs(vector), axis=1)
        # L2 metric
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
    coord += force / n_tasks
    stress = np.sum(np.sqrt(np.sum(force * force, axis=1))) / nv # actually this is kinetic energy, not stress.
    print stress
    if stress < 0.001 : break

for c in coord :
    for d in range(DIM) :
        print c[d],
    print
