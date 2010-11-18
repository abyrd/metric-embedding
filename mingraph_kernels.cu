// CUDA kernels for embedding shortest path metric into normed vector space

// Calculate all pairs shortest path.
// after Okuyama, Ino, and Hagihara 2008.
__global__ void scatter (int *vertex, int *edge, int *weight, int *cost, int *modify) {
    int fromv_tindex = blockIdx.x + blockDim.x * threadIdx.x; // damn fortran ordering
    if ( !modify[fromv_tindex] ) return;  // kill thread if this vertex was not changed in the last pass
    int fromv_cost = cost[fromv_tindex];  // get current cost for this vertex
    modify[fromv_tindex] = 0;
    int edge_index_low  = vertex[blockIdx.x];      // block number is vertex number (one vertex per block)
    int edge_index_high = vertex[blockIdx.x + 1];  // edges out of a vertex are contiguous
    for (int edge_index = edge_index_low; edge_index < edge_index_high; edge_index++) {
        int new_cost = fromv_cost + weight[edge_index];
        int tov_tindex = edge[edge_index] + blockDim.x * threadIdx.x;
        if (new_cost < atomicMin(cost + tov_tindex, new_cost)) { // atomicMin returns old value
            modify[tov_tindex] = 1; // enqueue the modified vertex for the next round
        }
    }
}

// accumulate forces proportional to embedding error
// (each block should work on blockdim.x different origins, randomly)
__global__ void force (float *coord, float *force, int *cost) {
    int tindex = blockIdx.x + blockDim.x * threadIdx.x; // damn fortran ordering
    int tdindex = tindex * D;
    float dist = 0;
    float vector[D];
    for (int d = 0; d < D; d++) {
        vector[d] = (coord[tdindex + d] - ???);
        dist += abs(vector[d]); // l1 norm
    }
    if (dist == 0) return; // avoid division by zero when points are superimposed
    float adjust = cost[tindex] / dist - 1;
    for (int d = 0; d < D; d++) force[tdindex + d] += adjust * vector[d];    
}

// shift embedded points according to forces, then reset forces
__global__ void integrate (float *coord, float *force) {
    int tdindex = D * (blockIdx.x + blockDim.x * threadIdx.x); // damn fortran ordering
    for (int i = tdindex; i < tdindex + D; i++) {
        coord[i] += force[i] / blockDim.x; // push points around
        force[i] = 0; // reset force to zero
    }
}
