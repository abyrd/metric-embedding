#!/usr/bin/env python2.6
# coding=UTF8
#
# Adaptation of single source shortest path algorithm presented in:
# P. Harish and P. J. Narayanan, "Accelerating large graph algorithms on the GPU using CUDA", 
# in IEEE International Conference on High Performance Computing (HiPC), December 2007. 
#
# CUDA implementation of SSSP: The SSSP problem does not traverse the graph in 
# levels. The cost of a visited vertex may change due to a low cost path being discovered 
# later. The termination is based on the change in cost. 
# In our implementation, we use a vertex array Va an edge array Ea , boolean mask 
# Ma of size |V |, and a weight array Wa of size |E|. In each iteration each vertex checks 
# if it is in the mask Ma . If yes, it fetches its current cost from the cost array Ca and its 
# neighbor’s weights from the weight array Wa. The cost of each neighbor is updated if 
# greater than the cost of current vertex plus the edge weight to that neighbor. The new 
# cost is not reﬂected in the cost array but is updated in an alternate array Ua . At the end 
# of the execution of the kernel, a second kernel compares cost Ca with updating cost 
# Ua . It updates the cost Ca only if it is more than Ua and makes its own entry in the 
# mask Ma . The updating cost array reﬂects the cost array after each kernel execution for 
# consistency. 
# The second stage of kernel execution is required as there is no synchronization 
# between the CUDA multiprocessors. Updating the cost at the time of modiﬁcation itself 
# can result in read after write inconsistencies. The second stage kernel also toggles a ﬂag 
# if any mask is set. If this ﬂag is not set the execution stops. 
# Newer version of CUDA hardware (ver 1.1) supports atomic read/write operations 
# in the global memory which can help resolve inconsistencies. 8800 GTX is CUDA 
# version 1.0 GPU and does not support such operations. Timings for SSSP CUDA im- 
# plementations are given in Figure 4. 
#
# Algorithm 3 CUDA SSSP (Graph G(V, E, W), Source Vertex S) 
# 1:  Create vertex array Va from all vertices in G(V, E , W ), 
# 2:  Create edge array Ea from all edges in G(V, E , W ) 
# 3:  Create weight array Wa from all weights in G(V, E , W ) 
# 4:  Create mask array Ma , cost array Ca and Updating cost array Ua of size V 
# 5:  Initialize mask Ma to false, cost array Ca and Updating cost array Ua to ∞ 
# 6:  Ma [S] ← true 
# 7:  Ca [S] ← 0 
# 8:  Ua [S] ← 0 
# 9:  while Ma not Empty do 
# 10:     for each vertex V in parallel do 
# 11:         Invoke CUDA SSSP KERNEL1(Va , Ea , Wa , Ma , Ca , Ua ) on the grid 
# 12:         Invoke CUDA SSSP KERNEL2(Va , Ea , Wa , Ma , Ca , Ua ) on the grid 
# 13:     end for 
# 14: end while 
#
# Algorithm 4 CUDA SSSP KERNEL1 (Va , Ea , Wa , Ma , Ca , Ua ) 
# 1:  t id ← getThreadID 
# 2:  if Ma [t id ] then 
# 3:     Ma [t id ] ← false 
# 4:     for all neighbors nid of t id do 
# 5:         if Ua [nid ]> Ca [t id ]+Wa [nid ] then 
# 6:            Ua [nid ] ← Ca [t id ]+Wa [nid ] 
# 7:         end if 
# 8:     end for 
# 9: end if 
#
# Algorithm 5 CUDA SSSP KERNEL2 (Va , Ea , Wa , Ma , Ca , Ua ) 
# 1:  t id ← getThreadID 
# 2:  if Ca [t id ] > Ua [t id ] then 
# 3:     Ca [t id ] ← Ua [t id ] 
# 4:     Ma [t id ] ← true 
# 5:  end if 
# 6:  Ua [t id ] ← Ca [t id ]
#
# ...
#
# Another alternative to ﬁnd all pair shortest paths is to run SSSP algorithm from every 
# vertex in graph G(V, E, W) (Algorithm 7). This will require only the ﬁnal output 
# size to be of O(V^2), all intermediate calculations do not require this space. The ﬁnal 
# output could be stored in the CPU memory. Each iteration of SSSP will output a vector 
# of size O(V), which can be copied back to the CPU memory. This approach does not 
# require the graph to be represented as an adjacency matrix, hence the representation 
# given in section 3.1 can be used , which makes this algorithm suitable for large graphs. 
# We implemented this approach and the results are given in Figure 6. This runs faster 
# than the parallel Floyd Warshall algorithm because it is a single O(V) operation looping 
# over O(V) threads. In contrast, the Floyd Warshall algorithm requires a single O(V) op- 
# eration looping over O(V^2) threads which creates extra overhead for context switching 
# the threads on the SIMD processors. Thus, due to the overhead for context switching of 
# threads, the Floyd Warshall algorithm exhibits a slow down.
#
# Algorithm 7 APSP USING SSSP(G(V, E, W)) 
# 1:  Create vertex array Va, edge array Ea, weight array Wa from G(V, E, W) 
# 2:  Create mask array Ma, cost array Ca and updating cost array Ua of size V 
# 3:  for S from 1 to V do 
# 4:      Ma [S] ← true 
# 5:      Ca [S] ← 0 
# 6:      while Ma not Empty do 
# 7:          for each vertex V in parallel do 
# 8:              Invoke CUDA SSSP KERNEL1(Va , Ea , Wa , Ma , Ca , Ua ) on the grid 
# 9:              Invoke CUDA SSSP KERNEL2(Va , Ea , Wa , Ma , Ca , Ua ) on the grid 
# 10:         end for 
# 11:     end while 
# 12: end for 

import time, math, pyproj, sys, cPickle
import numpy as np

from graphserver.graphdb         import GraphDatabase
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase

import pycuda.driver   as cuda
import pycuda.autoinit as autoinit
import pycuda.gpuarray as gpuarray
from   pycuda.compiler import SourceModule

# Best to put a multiple of 32 threads (1 warp) in a block. Multiples of 64 schedule better.
CUDA_BLOCK_SHAPE = (64, 1, 1) 
#GSDB_FILENAME    = '../gsdata/bart.linked.gsdb'
GSDB_FILENAME    = '../gsdata/trimet_13sep2009.linked.gsdb'

def from_gsdb(gsdb) :
    nV = gsdb.num_vertices()
    nE = gsdb.num_edges()
    V  = np.empty( nV, dtype=np.int32   )
    E  = np.empty( nE, dtype=np.int32   )
    W  = np.empty( nE, dtype=np.float32 )

    print 'Indexing edges...'
    gsdb.execute("CREATE INDEX IF NOT EXISTS edges_vertex1 ON edges (vertex1)")
    print 'Building vertex hash...'
    v_idx = dict( zip(gsdb.all_vertex_labels(), range(nV)) )
    v_tbl = [[] for e in range(nV)] 
    print 'Fetching incoming edges... '
    # original was designed for undirected graphs.
    # also : alignment, textures, broadcasting/packing, calendars, and paths should be worked on.
    # Atomic min to global memory instead of 2 kernels. This should improve cache use. Or even load into registers.
    # seems to go very slow when ordering by vertex2 not vertex1
    edges = gsdb.all_edges()
    iE = 0
    for vo_label, vd_label, edge in edges :
        v_tbl[v_idx[vo_label]].append(v_idx[vd_label])
        iE += 1
        if iE % 10000 == 0 :
            sys.stdout.write('\rProcessing edge %i / %i (%02i%%)' % (iE, nE, iE * 100. / nE))
            sys.stdout.flush()
        
    print 'Making ndarray edgelist representation... '
    iE = 0
    for iV in range(nV) :
        V[iV] = iE
        for e in v_tbl[iV] :
            E[iE] = e
            W[iE] = 1 # must be filled in...
            iE += 1            
                
    print V
    print E
    print W
    
    print '\nSaving in numpy gzipped format...'
    np.savez( 'edgelist.npz', V=V, E=E, W=W)
    return (V, E, W)
    
src = """
// should use texture references for read-only parameters, allows caching
// could also arrange vertices for spatial locality

#define INF 0x7f800000 

texture<int, 1, cudaReadModeElementType> tV;
texture<int, 1, cudaReadModeElementType> tE;
texture<int, 1, cudaReadModeElementType> tW;

__global__ void init ( int *M, float *C, float *U, int S ) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == S) {
        M[tid] = 1;
        C[tid] = 0;
        U[tid] = 0;
    } else {
        M[tid] = 0;
        C[tid] = INF;
        U[tid] = INF;
    }
}

__global__ void SSSP1 ( int *V, int *E, float *W, int *M, float *C, float *U ) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int e_off = V[tid];
    int e_len = V[tid + 1] - e_off;
    if (M[tid]) { 
        M[tid] = 0;
        for (int i = 0; i < e_len; i++) { 
             int nid = E[e_off + i];
             if (U[nid] > C[tid] + W[e_off + i]) { 
                 U[nid] = C[tid] + W[e_off + i];
             } 
        }
    } 
}

__global__ void SSSP2 ( int *V, int *E, float *W, int *M, float *C, float *U ) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (C[tid] > U[tid]) { 
        C[tid] = U[tid]; 
        M[tid] = 1; 
    }
    U[tid] = C[tid];
}
"""

gsdb = GraphDatabase(GSDB_FILENAME)
# V, E, W = from_gsdb(gsdb)
npz = np.load( 'edgelist.npz' )
V = npz['V']
E = npz['E']
W = npz['W']
V = gpuarray.to_gpu(V)        
E = gpuarray.to_gpu(E)
W = gpuarray.to_gpu(W)

M = gpuarray.zeros(V.shape, dtype=np.int32) # necessary to fill with 0 for now, since edges of blocks are not handled.
C = gpuarray.empty(V.shape, dtype=np.float32) 
U = gpuarray.empty(V.shape, dtype=np.float32) 

mod = SourceModule(src, options=["--ptxas-options=-v"])
init_kernel   = mod.get_function("init")
SSSP_kernel_1 = mod.get_function("SSSP1")
SSSP_kernel_2 = mod.get_function("SSSP2")
cuda_grid_shape = (len(V) / CUDA_BLOCK_SHAPE[0], 1)
np.set_printoptions(threshold=np.nan)

t0 = time.time()
for c in range(100) :
    print c
    n_iter = 0
    init_kernel (M, C, U, np.int32(15), block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape) 
    autoinit.context.synchronize()
    #while np.any(M.get()) :
    for i in range(40) :
        #print n_iter
        SSSP_kernel_1 (V, E, W, M, C, U, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape) 
        autoinit.context.synchronize()
        SSSP_kernel_2 (V, E, W, M, C, U, block=CUDA_BLOCK_SHAPE, grid=cuda_grid_shape)
        autoinit.context.synchronize()
        #print C
        #n_iter += 1
print (time.time() - t0) / 100.




