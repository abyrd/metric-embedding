// CUDA kernels for calculating all pairs shortest path

__global__ void scatter (int n_vert, int *vertex, int *edge, int *weight, int *cost, int *modify) {
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

