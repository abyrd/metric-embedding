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
            print o,d,w,'skipped'
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

def apsp_gpu (v_low, block_size) :
    src = open('mingraph_kernels.cu').read()
    mod = SourceModule(src, options=["--ptxas-options=-v"])
    scatter = mod.get_function("scatter")
    vertex = gpuarray.to_gpu(va)
    edge   = gpuarray.to_gpu(ea)
    weight = gpuarray.to_gpu(wa)
    cost   = np.empty((nv, block_size), dtype=np.int32)
    modify = np.zeros_like(cost)
    cost.fill(9999)
    for i in range(block_size) :
        cost  [v_low + i, i] = 0
        modify[v_low + i, i] = 1   
    for i in range(nv): # need better break condition
        #be careful to get parameters right, otherwise mysterious segfaults ('kernel launch errors') will happen.        
        # need v_low parameter
        scatter(vertex, edge, weight, cuda.InOut(cost), cuda.InOut(modify), block=(block_size,1,1), grid=(nv,1))
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
 
ag = net(19, 19)

ag.add_edge('0_9', '5_9', ElapseTime(5))
ag.add_edge('5_9', '0_9', ElapseTime(5))
ag.add_edge('5_9', '10_9', ElapseTime(5))
ag.add_edge('10_9', '5_9', ElapseTime(5))
ag.add_edge('10_9', '15_9', ElapseTime(5))
ag.add_edge('15_9', '10_9', ElapseTime(5))
ag.add_edge('15_9', '18_9', ElapseTime(5))
ag.add_edge('18_9', '15_9', ElapseTime(5))

ag.add_edge('9_5', '9_9', ElapseTime(3))
ag.add_edge('9_9', '9_5', ElapseTime(3))
ag.add_edge('9_9', '9_13', ElapseTime(3))
ag.add_edge('9_13', '9_9', ElapseTime(3))
ag.add_edge('9_13', '9_17', ElapseTime(3))
ag.add_edge('9_17', '9_13', ElapseTime(3))

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
cost = apsp_gpu(0, n_tasks)
print cost

# embed
DIM = 3
coord  = np.random.rand(nv, DIM)
force  = np.empty ((nv, DIM), dtype=np.float32)

from enthought.mayavi import mlab
X = np.zeros((19, 19))
Y = np.zeros_like(X)
Z = np.zeros_like(X)
S = np.zeros_like(X)
mesh = mlab.mesh(X, Y, Z, scalars=S)
mesh_source = mesh.mlab_source

def display_coords(coord) :
    for label, index in vl.iteritems():
        x, y = label.split('_')
        if DIM > 3:
            X[x, y], Y[x, y], Z[x, y], S[x, y] = coord[index, :4]
            S[x, y] = err[index]
        else:
            X[x, y], Y[x, y], Z[x, y] = coord[index]
            #S[x, y] = np.sum(vel[index])
            #S[x, y] = sum(abs(force[index]))
            S[x, y] = err[index]
    mesh_source.set(x=X, y=Y, z=Z, scalars=S)
    #outline.extent = mesh_source.extent
    #mlab.view(focalpoint=coord[vl['10_10']])    

err = np.zeros((nv,))

def mds_iterate(coord, force, cost):
    global err
    force.fill(0)
    err.fill(0)
    for t in range(n_tasks) :
        vector = coord - coord[t]
        # IF YOU USE AN L1 METRIC, OF COURSE YOUR RESULTS WILL LOOK STRANGE
        # L1 metric        
        #dist = np.sum(np.abs(vector), axis=1)
        # L2 metric
        dist = np.sqrt(np.sum(vector * vector, axis=1))
        # L4 metric 
        #dist = np.sqrt(np.sqrt(np.sum(vector * vector * vector * vector, axis=1)))
        adjust = (cost[t] / dist) - 1
        adjust[t] = 0 # avoid NaNs, could use nantonum
        err += np.abs(adjust)
        #print 'task', t
        #print coord[t]
        #print coord
        #print vector
        #print dist
        #print cost[t]
        #print adjust
        force += vector * adjust[:, np.newaxis]
        #force += (vector * (adjust * (adjust > 0))[:, np.newaxis])
        #print force
    #vel /= 2
    coord += force / n_tasks
    err /= n_tasks
    #coord += force / (n_tasks)
    #stress = np.sum(np.sqrt(np.sum(force * force, axis=1))) / nv # actually this is kinetic energy, not stress.
    # kinetic energy, peak to peak, mean, median, standard deviation
    print np.sum(np.abs(force)), np.max(err), np.mean(err), np.std(err) 
    #stress = np.sqrt(np.sum(adjust * adjust) / n_tasks)
    #plot3s(coord, i)    
    #i += 1
    #if ke < 100 : mlab.show() # 1 percent RMS error

@mlab.animate(delay=10)
def mayavi_mds():
    while(True):
        mds_iterate(coord, force, cost)
        display_coords(coord)
        yield

mayavi_mds()
# redo axes since extent start out as all zeros
#axes = mlab.axes(extent=(-100, 100, -100, 100, -100, 100))
#outline = mlab.outline()
mlab.show()

for c in coord :
    for d in range(DIM) :
        print c[d],
    print


