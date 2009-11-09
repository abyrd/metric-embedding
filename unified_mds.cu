
#define OBSTRUCTION       1.4
#define WALK_SPEED        1.3
#define INF               0x7f800000 

// Some all-caps constants will be replaced by the calling host code before compilation

/*
 *  "unified" CUDA kernel.
 *  The new monolithic time-space MDS kernel.
 *  Finds both network travel times and forces on a texel-by-texel basis.
 */
  
__global__ void stations (
    int   *glb_station_coords,
    int   *glb_nearby_stations )

{                      
    int   b_x = blockIdx.x;
    int   b_y = blockIdx.y;  
    int   ds_x, ds_y;
    float dist;
    float nearby_times[N_NEARBY_STATIONS];
    
    int *glb_nearby_idx = glb_nearby_stations + ((b_x * gridDim.y + b_y) * N_NEARBY_STATIONS * 3);
    int *glb_nearby_x   = glb_nearby_idx + N_NEARBY_STATIONS;
    int *glb_nearby_y   = glb_nearby_idx + N_NEARBY_STATIONS * 2;
    
    if (threadIdx.x == 1 && threadIdx.y == 1) {     // The thread in the physical center of the block builds a list of nearby stations.
        int   slots_filled = 0;
        float max_dist     = 0;
        int   max_slot;
        for (int ds_idx = 0; ds_idx < N_STATIONS; ds_idx++) {                               // For every station:
            ds_x = glb_station_coords[ds_idx * 2 + 0];                                  // Get the station's geographic x coordinate. 
            ds_y = glb_station_coords[ds_idx * 2 + 1];                                  // Get the station's geographic y coordinate.
            dist = sqrt( pow(float(ds_x - b_x), 2) + pow(float(ds_y - b_y), 2)) * 100;      // Find the geographic distance from the station to this texel.
            if (slots_filled < N_NEARBY_STATIONS) {                                     // First, fill up all the nearby station slots, keeping track of the 'worst' station.
                glb_nearby_idx [slots_filled] = ds_idx;
                nearby_times   [slots_filled] = dist;
                if (dist > max_dist) {
                    max_dist = dist;
                    max_slot = slots_filled;
                } 
                slots_filled++;
            } else {                                                                    // Then, keep replacing the worst station each time a closer one is found.
                if (dist < max_dist) {
                    glb_nearby_idx [max_slot] = ds_idx;
                    nearby_times   [max_slot] = dist;
                    max_dist = 0;
                    for (int slot = 0; slot < N_NEARBY_STATIONS; slot++) {              // Scan through the list to find the new worst.
                        if (nearby_times[slot] > max_dist) {
                            max_dist = nearby_times[slot];
                            max_slot = slot;
                        }
                    }
                }
            }
        } 
        for (int i = 0; i < N_NEARBY_STATIONS; i++) {                                   // Go through the completed list of nearby stations.
            int ds_idx = glb_nearby_idx[i];
            glb_nearby_x[i] = glb_station_coords[ds_idx * 2 + 0];                         // Copy its x geographic coordinate from global to block shared memory.
            glb_nearby_y[i] = glb_station_coords[ds_idx * 2 + 1];                         // Copy its y geographic coordinate from global to block shared memory.
        }
    }
}    
    

__global__ void unified (
    int   n_stations,
    int   s_low,
    int   s_high,
    int   max_x,
    int   max_y,
    int   *glb_station_coords,
    int   *glb_matrix,
    float *glb_coords,
    float *glb_forces,
    float *glb_errors,
    int   *test1,
    int   *test2,
    int   *test3)
{
    float coord [DIM];
    float vector[DIM];
    float force [DIM]; 
    float *glb_coord;
    float *glb_error;
    float *glb_force;
    float adjust, dist;
    float error = 0;  // must initialize because error is cumulative per pass
    float norm  = 0;  // must initialize because components accumulated
    float tt, ds_tt;
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
    
    if (threadIdx.x == int(blockDim.x / 2) && threadIdx.y == int(blockDim.y / 2)) {     // The thread in the physical center of the block builds a list of nearby stations.
        int   slots_filled = 0;
        float max_dist = 0;
        int   max_slot;
        for (ds_idx = 0; ds_idx < n_stations; ds_idx++) {                               // For every station:
            ds_x = glb_station_coords[ds_idx * 2 + 0];                                  // Get the station's geographic x coordinate. 
            ds_y = glb_station_coords[ds_idx * 2 + 1];                                  // Get the station's geographic y coordinate.
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
        for (int i = 0; i < N_NEARBY_STATIONS; i++) {                                   // Go through the completed list of nearby stations.
            ds_idx = blk_near_idx[i];                                                   // For each index that was recorded:
            blk_near_x[i] = glb_station_coords[ds_idx * 2 + 0];                         // Copy its x geographic coordinate from global to block shared memory.
            blk_near_y[i] = glb_station_coords[ds_idx * 2 + 1];                         // Copy its y geographic coordinate from global to block shared memory.
        }
    }
    
    __syncthreads();  // All threads in this block wait here, to avoid reading from the near stations list until it is fully built.
    
    // TO DO: block-level cacheing of global memory accesses, __prefixed fast math functions.
    
    if ( x < max_x && y < max_y ) {                                                     // Check that this thread is inside the map.
        glb_coord = &( glb_coords[(x * max_y + y) * DIM] );                             // Make a pointer to this texel's time-space coordinate in global memory.
        for (d = 0; d < DIM; d++) coord[d] = glb_coord[d];                              // Copy the time-space coordinate for this texel from global to local memory.
        for (d = 0; d < DIM; d++) force[d] = 0;                                         // Initialize this timestep's force to 0. (Do this before calculating forces, outside the loops!)
        for (os_idx = s_low; os_idx < s_high; os_idx++) {                               // For every origin station:
            os_x = glb_station_coords[os_idx * 2 + 0];                                  // Get the origin station's geographic x coordinate.
            os_y = glb_station_coords[os_idx * 2 + 1];                                  // Get the origin station's geographic y coordinate.
            if (threadIdx.x == 1 && threadIdx.y == 1) {                                 // The first thread in each block fetches some origin-specific data to block shared memory.
                    float *glb_origin_coord = &(glb_coords[(os_x*max_y + os_y) * DIM]); // Make a pointer to the gobal time-space coordinates for this origin station.
                    int   *glb_matrix_row   = &(glb_matrix[os_idx * n_stations] );      // Make a pointer to the relevant global OD matrix row.
                    for (d = 0; d < DIM; d++)                        
                        blk_origin_coord[d] = glb_origin_coord[d];                      // Copy origin time-space coordinates from global to block-shared memory.
                    for (int i = 0; i < N_NEARBY_STATIONS; i++) 
                        blk_near_time[i] = glb_matrix_row[blk_near_idx[i]];             // Copy relevant OD matrix row entries into block-shared nearby stations table.
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
            norm = 0; // ADDED RECENTLY... WAS WRONG
            for (d = 0; d < DIM; d++) {
                vector[d] = coord[d] - blk_origin_coord[d];                             // Find the vector from the origin station to this texel in time-space.
                norm     += pow(vector[d], 2);                                          // Accumulate terms to find the norm.
            }
            norm   = sqrt(norm);                                                        // Take the square root to find the norm, i.e. the distance in time-space.
            adjust = tt - norm ;                                                        // How much would the point need to move to match the best travel time?
            
                                                                                        // use __isnan() ? anyway, this is in the outer loop, and should almost never diverge within a warp.                                                                                                 
            if (norm != 0) {                                                            // Avoid propagating nans through division by zero. Force should be 0, so add skip this step / add nothing.
                for (d = 0; d < DIM; d++) 
                    force[d] += ((vector[d] / norm) * adjust) / n_stations;             // Find a unit vector, scale it by the desired 'correct' time-space distance, and normalize by the number of total forces acting.
                error += abs(adjust);                                                   // Accumulate error to each texel, so we can observe progress as the program runs.
            }
            // TEST TRAVEL TIME MAP COMPUTATION
            // error = tt;
        }

        // Force computation for all origin stations is now finished for this timestep.
        // We should perform an Euler integration to move the coordinates.
        // However, when the number of executing blocks exceeds number of processors on the device,
        // some points will move before others finish (or even start) calculating.
        // Therefore integration has been split off into another kernel to allow global, device-level thread synchronization.

        glb_force = glb_forces + (x * max_y + y) * DIM;     // Make a pointer to a force record in global memory.
        glb_error = glb_errors + (x * max_y + y)      ;     // Make a pointer to an error record in global memory.
        if (s_low > 0) {
            for (d = 0; d < DIM; d++) force[d] += glb_force[d];     // ADD Output forces to device global memory.
            error += *glb_error;
        }
        for (d = 0; d < DIM; d++) glb_force[d]  = force[d];     // SET Output forces to device global memory.
        *glb_error  = error;                                     // SET Output this texel's cumulative error to device global memory.
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
    float *glb_forces)
{
    int   d;
    float *glb_coord;
    float *glb_force;
    int   x = blockIdx.x * blockDim.x + threadIdx.x; 
    int   y = blockIdx.y * blockDim.y + threadIdx.y; 
    if ( x < max_x && y < max_y ) {                             // Check that this thread is inside the map
        glb_coord = &( glb_coords[(x * max_y + y) * DIM] );     // Make a pointer to time-space coordinates in global memory
        glb_force = &( glb_forces[(x * max_y + y) * DIM] );     // Make a pointer to forces in global memory
        for (d = 0; d < DIM; d++) glb_coord[d] += glb_force[d]; // Euler integration, 1 timestep
    }
}
