#!/usr/bin/env python2.6
#
# embed_mingraph
#
# Open a graph whose shortest path metric gives the
# greatest lower bound on travel time possible with a non-time-dependent graph.
# Make an edge-list representation in arrays.
# Iteratively improve an embedding of this metric into an L1-normed real vector space.
 
from graphserver.core import Graph, ElapseTime, State, WalkOptions, VertexNotFoundError
from graphserver.graphdb import GraphDatabase
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB
import sys, time, os, random
from optparse import OptionParser         
import numpy as np

import pycuda.driver   as cuda
import pycuda.autoinit 
from   pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import math

def net(X, Y) :
    g = Graph()
    edges = []
    fs = '%i_%i'
    for x in range(X):
        for y in range(Y):
            g.add_vertex(fs % (x,y))
            o = (x, y)
            d = (x, y-1) 
            edges.append((o, d, 3))
            d = (x, y+1) 
            edges.append((o, d, 3))
            d = (x-1, y) 
            edges.append((o, d, 4))
            d = (x+1, y) 
            edges.append((o, d, 4))
            d = (x-1, y-1) 
            edges.append((o, d, 5))
            d = (x-1, y+1) 
            edges.append((o, d, 5))
            d = (x+1, y-1) 
            edges.append((o, d, 5))
            d = (x+1, y+1) 
            edges.append((o, d, 5))            
    for o, d, w in edges:
        try:
            g.add_edge(fs%o, fs%d, ElapseTime(w))
        except VertexNotFoundError:
            pass
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

def apsp_gpu (block_size) :
    cost   = np.empty((block_size, nv), dtype=np.int32)
    modify = np.zeros_like(cost)
    cost.fill(9999)
    # maybe add random numbers to a hashmap until it is the right size, guaranteeing uniqueness
    origins = range(nv)
    random.shuffle(origins)
    origins = origins[:block_size]
    for i in range(block_size) :
        cost  [i, origins[i]] = 0
        modify[i, origins[i]] = 1   
    while modify.any() :
        #be careful to get parameters right, otherwise mysterious segfaults ('kernel launch errors') or worse will happen.        
        scatter(np.int32(nv), vertex, edge, weight, cuda.InOut(cost), cuda.InOut(modify), block=(block_size,1,1), grid=(nv,1))
        pycuda.autoinit.context.synchronize()    
    return origins, cost

usage = """usage: python zzzz.py <assist_graph_database>"""
parser = OptionParser(usage=usage)
(options, args) = parser.parse_args()
if len(args) != 1:
    parser.print_help()
    exit(-1)  
assist_graph_db = args[0]
assistgraphdb = GraphDatabase( assist_graph_db )
#ag = assistgraphdb.incarnate()
 
ag = net(19, 19)

ag.add_edge('0_9', '5_9', ElapseTime(5))
ag.add_edge('5_9', '0_9', ElapseTime(5))
ag.add_edge('5_9', '10_9', ElapseTime(5))
ag.add_edge('10_9', '5_9', ElapseTime(5))
ag.add_edge('10_9', '15_9', ElapseTime(5))
ag.add_edge('15_9', '10_9', ElapseTime(5))
ag.add_edge('15_9', '18_9', ElapseTime(5))
ag.add_edge('18_9', '15_9', ElapseTime(5))

#ag.add_edge('9_5', '9_9', ElapseTime(3))
#ag.add_edge('9_9', '9_5', ElapseTime(3))
#ag.add_edge('9_9', '9_13', ElapseTime(3))
#ag.add_edge('9_13', '9_9', ElapseTime(3))
#ag.add_edge('9_13', '9_17', ElapseTime(3))
#ag.add_edge('9_17', '9_13', ElapseTime(3))

remove_deadends(ag)

v  = list(ag.vertices)
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

# === END MAKE EDGELIST REPRESENTATION ===


# === EMBED ===
DIM = 3
coord = np.random.rand(nv, DIM)
force = np.zeros((nv, DIM))
error = np.zeros((nv,    ))

from enthought.mayavi import mlab
X = np.zeros((19, 19))
Y = np.zeros_like(X)
Z = np.zeros_like(X)
S = np.zeros_like(X)
mesh = mlab.mesh(X, Y, Z, scalars=S)
mesh_source = mesh.mlab_source

src = open('mingraph_kernels.cu').read()
mod = SourceModule(src, options=["--ptxas-options=-v"])
scatter = mod.get_function("scatter")
vertex = gpuarray.to_gpu(va)
edge   = gpuarray.to_gpu(ea)
weight = gpuarray.to_gpu(wa)

def display_coords(coord) :
    for label, index in vl.iteritems():
        x, y = label.split('_')
        X[x, y], Y[x, y], Z[x, y] = coord[index, 0:3]
        S[x, y] = error[index]
    mesh_source.set(x=X, y=Y, z=Z, scalars=S)

def mds_iterate():
    BLOCK_SIZE = 64
    global coord, force, cost, error
    force.fill(0)
    error.fill(0)
    origins, cost = apsp_gpu(BLOCK_SIZE)
    for i in range(BLOCK_SIZE) :
        vector = coord - coord[origins[i]]
        # l2 metric
        dist = np.sqrt(np.sum(vector * vector, axis=1))
        adjust = (cost[i] / dist) - 1
        adjust[origins[i]] = 0 # avoid NaNs, could use nantonum
        error += abs(adjust)
        force += vector * adjust[:, np.newaxis]
    coord += force / BLOCK_SIZE
    error /= BLOCK_SIZE
    # kinetic energy and peak, mean, median, standard deviation of errors
    print np.sum(np.abs(force)), np.max(error), np.mean(error), np.std(error) 

@mlab.animate(delay=10)
def mayavi_mds():
    while(True):
        mds_iterate()
        display_coords(coord)
        yield

mayavi_mds()
mlab.show()


