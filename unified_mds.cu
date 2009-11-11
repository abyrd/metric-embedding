
#define OBSTRUCTION       1.4
#define WALK_SPEED        1.3
#define INF               0x7f800000 
#define DIM               DIMENSIONS_PYTHON
#define N_NEARBY_STATIONS N_NEARBY_STATIONS_PYTHON
#define N_STATIONS        N_STATIONS_PYTHON

texture<int, 1, cudaReadModeElementType> tex_near_stations;
texture<int, 2, cudaReadModeElementType> tex_station_coords;
texture<int, 2, cudaReadModeElementType> tex_matrix;



/*
 *  [ stations kernel ]
 *  
 *  Finds nearby stations to each thread block and saves them.
 */

__global__ void stations (int *glb_near_stations)
{
    int   d;
    float dist;
    int   x = blockIdx.x * blockDim.x + threadIdx.x; 
    int   y = blockIdx.y * blockDim.y + threadIdx.y; 
    int   os_x, os_y, ds_x, ds_y;
    int   os_idx, ds_idx;
    
    //replace with local (register) variables?
    __shared__ float blk_origin_coord [DIM];
    __shared__ float blk_near_time[N_NEARBY_STATIONS];   // First used for selecting nearby stations, then for cacheing rows from the OD matrix.
    __shared__ int   blk_near_idx [N_NEARBY_STATIONS];
    __shared__ int   blk_near_x   [N_NEARBY_STATIONS];
    __shared__ int   blk_near_y   [N_NEARBY_STATIONS];
    
    if (threadIdx.x == int(blockDim.x / 2) && threadIdx.y == int(blockDim.y / 2)) {     // The thread in the physical center of the block builds a list of nearby stations.
        int   slots_filled = 0;
        float max_dist = 0;
        int   max_slot;
        for (ds_idx = 0; ds_idx < N_STATIONS; ds_idx++) {                               // For every station:
            ds_x = tex2D(tex_station_coords, ds_idx, 0);                                  // Get the station's geographic x coordinate. 
            ds_y = tex2D(tex_station_coords, ds_idx, 1);                                  // Get the station's geographic y coordinate.
            dist = sqrt( pow(float(ds_x - x), 2) + pow(float(ds_y - y), 2)) * 100;      // Find the geographic distance from the station to this texel.
            if (slots_filled < N_NEARBY_STATIONS) {                                     // First, fill up all the nearby station slots, keeping track of the 'worst' station.
                blk_near_idx [slots_filled] = ds_idx;
                blk_near_time[slots_filled] = dist;
                if (dist > max_dist) {
                    max_dist = dist;
                    max_slot = slots_filled;
                } 
                slots_filled++;
            } else {                                                                    // Then, keep replacing the worst station each time a closer one is found.
                if (dist < max_dist) {
                    blk_near_idx [max_slot] = ds_idx;
                    blk_near_time[max_slot] = dist;
                    max_dist = 0;
                    for (int slot = 0; slot < N_NEARBY_STATIONS; slot++) {              // Scan through the list to find the new worst.
                        if (blk_near_time[slot] > max_dist) {
                            max_dist = blk_near_time[slot];
                            max_slot = slot;
                        }
                    }
                }
            }
        } 
        int *p = glb_near_stations + (blockIdx.x * gridDim.y + blockIdx.y) * N_NEARBY_STATIONS;
        // should index the pointer or increment?
        for (int i = 0; i < N_NEARBY_STATIONS; i++) {                                   // Go through the completed list of nearby stations.
            *(p++) = blk_near_idx[i];                                                   // For each index that was recorded:
        }
    }
    
}    
  

/*
 *  [ forces kernel ]
 *  
 *  Finds both network travel times and forces on a texel-by-texel basis.
 */
 
__global__ void forces (
    int   n_stations,
    int   s_low,
    int   s_high,
    int   max_x,
    int   max_y,
    float *glb_coords, // make them a texture? test performance - this is an optimisation.
    float *glb_forces,
    float *glb_weights,
    float *glb_errors,
    int   *debug,
    float *debug_img)
{
    float coord [DIM];
    float vector[DIM];
    float force [DIM]; 
    float *glb_coord;
    float *glb_error;
    float *glb_force;
    float *glb_weight;
    float adjust, dist;
    float error = 0;  // must initialize because error is cumulative per pass
    float error_here;
    float error_max = 0;
    float norm;
    float tt, ds_tt;
    float weight = 0;
    float weight_here;
    int   d;
    int   x = blockIdx.x * blockDim.x + threadIdx.x; 
    int   y = blockIdx.y * blockDim.y + threadIdx.y; 
    int   os_x, os_y, ds_x, ds_y;
    int   os_idx, ds_idx;
    
    __shared__ float blk_origin_coord [DIM];
    __shared__ float blk_near_time[N_NEARBY_STATIONS];   // First used for selecting nearby stations, then for cacheing rows from the OD matrix.
    __shared__ int   blk_near_idx [N_NEARBY_STATIONS];
    __shared__ int   blk_near_x   [N_NEARBY_STATIONS];
    __shared__ int   blk_near_y   [N_NEARBY_STATIONS];
    
    // ATTENTION this causes errors for some reason for blocks outside the mapped geographic area
    // OR is it when there are stations outside the grid?
    // watch out for grid dimensions

    /* LOAD BY EACH THREAD... not really necessary, and doesn't work
    int thread_num = threadIdx.x * blockDim.y + y;
    if (thread_num < N_NEARBY_STATIONS) {
        int s_idx = tex1Dfetch(tex_near_stations, (blockIdx.x * gridDim.y + blockIdx.y) * N_NEARBY_STATIONS + thread_num);
        blk_near_idx[thread_num] = s_idx;
        blk_near_x  [thread_num] = tex2D(tex_station_coords, s_idx, 0);
        blk_near_y  [thread_num] = tex2D(tex_station_coords, s_idx, 1);
        if (blockIdx.x == 1 && blockIdx.y == 1) {
            debug[thread_num] = blk_near_idx[thread_num];
            debug[thread_num] = 55;
        }
    }
    */
    if (threadIdx.x == 1 && threadIdx.y == 1) {     
        for (int i = 0; i < N_NEARBY_STATIONS; i++) {
            int idx = tex1Dfetch(tex_near_stations, (blockIdx.x * gridDim.y + blockIdx.y) * N_NEARBY_STATIONS + i);
            blk_near_idx [i] = idx;
            blk_near_x   [i] = tex2D(tex_station_coords, idx, 0);
            blk_near_y   [i] = tex2D(tex_station_coords, idx, 1);
            if (blockIdx.x == 10 && blockIdx.y == 30) {
               debug[i] = blk_near_idx[i];
            }
        }
    }
    
    __syncthreads();  // All threads in this block wait here, to avoid reading from the near stations list until it is fully built.

    // if (isnan(glb_coords[(x * max_y + y) * DIM])) return; // should kill off useless threads, but also those that should be loading shared memory. shared mem should be exchanged for registers anyway.
    
    // TO DO: block-level cacheing of global memory accesses, __prefixed fast math functions.
    // when changing code, I see slight differences in the evolution of error images - this is probably due to random initial configuration of points.
    if ( x < max_x && y < max_y ) {          // Check that this thread is inside the map.
        glb_coord = &( glb_coords[(x * max_y + y) * DIM] );                             // Make a pointer to this texel's time-space coordinate in global memory.
        for (d = 0; d < DIM; d++) coord[d] = glb_coord[d];                              // Copy the time-space coordinate for this texel from global to local memory.
        for (d = 0; d < DIM; d++) force[d] = 0;                                         // Initialize this timestep's force to 0. (Do this before calculating forces, outside the loops!)
        for (os_idx = s_low; os_idx < s_high; os_idx++) {                               // For every origin station:
            os_x = tex2D(tex_station_coords, os_idx, 0);                                  // Get the origin station's geographic x coordinate.
            os_y = tex2D(tex_station_coords, os_idx, 1);                                  // Get the origin station's geographic y coordinate.
            if (threadIdx.x == 1 && threadIdx.y == 1) {                                 // The first thread in each block fetches some origin-specific data to block shared memory.
                    float *glb_origin_coord = &(glb_coords[(os_x*max_y + os_y) * DIM]); // Make a pointer to the gobal time-space coordinates for this origin station.
                    for (d = 0; d < DIM; d++)                        
                        blk_origin_coord[d] = glb_origin_coord[d];                      // Copy origin time-space coordinates from global to block-shared memory. Maybe it should be in register.
                    for (int i = 0; i < N_NEARBY_STATIONS; i++) 
                        blk_near_time[i] = tex2D(tex_matrix, os_idx, blk_near_idx[i]);  // Copy relevant OD matrix row entries into block-shared nearby stations table. Texture cache is probably sufficient, locality is very good. Test performance later.
                        // this caching is not really necessary since you have textured it - test this idea later.
            }
            __syncthreads();                                                            // All threads in the block must wait for thread (1, 1) to load data into block-shared memory.
            tt = INF;                                                                   // Set the current best travel time to +infinity.
            // Using manhattan distance does not noticeably speed up computation. 
            // Global memory is the bottleneck, so computation is better than a lookup table.
            // Is it OK to unroll long loops, like the nearby stations code, as below?
            for (int i = 0; i < N_NEARBY_STATIONS; i++) {                               // For every destination station in this block's nearby stations list:
                ds_x  = blk_near_x [i];                                                 // Get the destination station's geographic x coordinate. 
                ds_y  = blk_near_y [i];                                                 // Get the destination station's geographic y coordinate.
                dist  = sqrt( pow(float(ds_x - x), 2) + pow(float(ds_y - y), 2)) * 100; // Find the geographic distance from the destination station to our texel.
                ds_tt = (dist * OBSTRUCTION / WALK_SPEED) + blk_near_time[i];           // Derive a travel time from the origin station to our texel, through the current destination station.
                tt = min(tt, ds_tt);                                                    // Save this travel time if it is better than the current best.
            }                                                                           // We could also use the destination station to texel distance to make additional forces.
            norm = 0; // ADDED RECENTLY... WAS WRONG, initialized at declaration instead of just before accumulation in inner loop
            for (d = 0; d < DIM; d++) {
                vector[d] = coord[d] - blk_origin_coord[d];                             // Find the vector from the origin station to this texel in time-space.
                norm     += pow(vector[d], 2);                                          // Accumulate terms to find the norm.
            }
            norm   = sqrt(norm);                                                        // Take the square root to find the norm, i.e. the distance in time-space.
            adjust = tt - norm ;                                                        // How much would the point need to move to match the best travel time?
            // global influence cutoff above T minutes
            // if (tt > 60 * 100) adjust = 0;
            // global influence scaling like gravity, relative to tt - scale adjustment according to travel time to point
            // turn off weighting
            // weight_here = (tt < 120*60) * (1 - 1 / (120*60 - (tt-1)));
            weight_here = 1;
            weight += weight_here;
                                                                                                                                                                                                                                                                                                                            // use __isnan() ? anyway, this is in the outer loop, and should almost never diverge within a warp.                                                                                                 
            if (norm != 0) {                                                            // Avoid propagating nans through division by zero. Force should be 0, so add skip this step / add nothing.
                for (d = 0; d < DIM; d++) 
                    force[d] += ((vector[d] / norm) * adjust * weight_here);            // Find a unit vector, scale it by the desired 'correct' time-space distance. (weighted)
                // why is this in the loop?
                error_here = abs(adjust) * weight_here;                                    // Accumulate error to each texel, so we can observe progress as the program runs. (weighted)
                error += error_here;
                error_max = max(error_max, error_here);
            }
                        
        }

        // DEBUGGING OUTPUT
        // visualize travel times for last origin station
        // debug_img[ x * max_y + y ] = tt;
        
        // Force computation for all origin stations is now finished for this timestep.
        // We should perform an Euler integration to move the coordinates.
        // However, when the number of executing blocks exceeds number of processors on the device,
        // some points will move before others finish (or even start) calculating.
        // Therefore integration has been split off into another kernel to allow global, device-level thread synchronization.

        glb_force  = glb_forces  + (x * max_y + y) * DIM;     // Make a pointer to a force record in global memory. WAIT you can do this with C array notation
        glb_error  = glb_errors  + (x * max_y + y)      ;     // Make a pointer to an error record in global memory.
        glb_weight = glb_weights + (x * max_y + y)      ;     // Make a pointer to an error record in global memory.
        if (s_low > 0) {
            for (d = 0; d < DIM; d++) force[d] += glb_force[d];     // ADD Output forces to device global memory.
            error  += *glb_error;
            weight += *glb_weight;
            // visualize max error per cell
            error_max = max(error_max, debug_img[ x * max_y + y ]);
        }
        for (d = 0; d < DIM; d++) glb_force[d]  = force[d];     // SET Output forces to device global memory.
        *glb_error  = error;                                     // SET Output this texel's cumulative error to device global memory.
        *glb_weight = weight;                                     // SET Output this texel's cumulative error to device global memory.
        debug_img[ x * max_y + y ] = error_max;
    }
}


/*
 *   "integrate" CUDA kernel.
 *   Applies forces accumulated in the unified kernel to time-space coordinates in global device memory.
 *   It has been split out to allow synchronization of all threads before moving any points.
 */

__global__ void integrate (
    int   max_x,
    int   max_y,
    float *glb_coords,
    float *glb_forces,
    float *glb_weights)
{
    int   d;
    float *glb_coord;
    float *glb_force;
    float *glb_weight;
    int   x = blockIdx.x * blockDim.x + threadIdx.x; 
    int   y = blockIdx.y * blockDim.y + threadIdx.y; 
    if ( x < max_x && y < max_y ) {                              // Check that this thread is inside the map
        glb_coord  = &( glb_coords [(x * max_y + y) * DIM] );    // Make a pointer to time-space coordinates in global memory
        glb_force  = &( glb_forces [(x * max_y + y) * DIM] );    // Make a pointer to forces in global memory
        glb_weight = &( glb_weights[(x * max_y + y)      ] );    // Make a pointer to forces in global memory
        for (d = 0; d < DIM; d++) glb_force[d] /= *glb_weight;    // Scale forces by the sum of all weights acting on this cell
        for (d = 0; d < DIM; d++) glb_coord[d] += glb_force [d]; // Euler integration, 1 timestep
    }
}
