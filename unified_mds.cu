
#define OBSTRUCTION       1.4
#define WALK_SPEED        1.3
#define INF               0x7f800000 

texture<int, 1, cudaReadModeElementType> tex_near_stations;
texture<int, 2, cudaReadModeElementType> tex_station_coords;
texture<int, 2, cudaReadModeElementType> tex_matrix;

// preprocess to replace blockDim.x/y

/*
 *  [ stations kernel ]
 *  
 *  Finds nearby stations to each thread block and saves them.
 */

__global__ void stations (int *glb_near_stations, int max_x, int max_y)
{
    int   x = blockIdx.x * blockDim.x + threadIdx.x; 
    int   y = blockIdx.y * blockDim.y + threadIdx.y; 
    int   ds_x, ds_y;
    int   ds_idx;
    float dist;
    
    int   blk_near_idx [@N_NEAR_STATIONS];
    float blk_near_dist[@N_NEAR_STATIONS];
    
    if (x <  max_x && y < max_y) {     // Check that this thread is inside the map.
        int   slots_filled = 0;
        float max_dist = 0;
        int   max_slot;
        for (ds_idx = 0; ds_idx < @N_STATIONS; ds_idx++) {                               // For every station:
            ds_x = tex2D(tex_station_coords, ds_idx, 0);                                  // Get the station's geographic x coordinate. 
            ds_y = tex2D(tex_station_coords, ds_idx, 1);                                  // Get the station's geographic y coordinate.
            dist = sqrt( pow(float(ds_x - x), 2) + pow(float(ds_y - y), 2)) * 100;      // Find the geographic distance from the station to this texel.
            if (slots_filled < @N_NEAR_STATIONS) {                                     // First, fill up all the nearby station slots, keeping track of the 'worst' station.
                blk_near_idx [slots_filled] = ds_idx;
                blk_near_dist[slots_filled] = dist;
                if (dist > max_dist) {
                    max_dist = dist;
                    max_slot = slots_filled;
                } 
                slots_filled++;
            } else {                                                                    // Then, keep replacing the worst station each time a closer one is found.
                if (dist < max_dist) {
                    blk_near_idx [max_slot] = ds_idx;
                    blk_near_dist[max_slot] = dist;
                    max_dist = 0;
                    for (int slot = 0; slot < @N_NEAR_STATIONS; slot++) {              // Scan through the list to find the new worst.
                        if (blk_near_dist[slot] > max_dist) {
                            max_dist = blk_near_dist[slot];
                            max_slot = slot;
                        }
                    }
                }
            }
        } 
        
        // mark this cell for uselessness
        float min_dist = INF;
        for (int i = 0; i < @N_NEAR_STATIONS; i++) min_dist = min(min_dist, blk_near_dist[i]);
        if (min_dist > 2000) blk_near_idx[0] = -1;

        int *p = glb_near_stations + (x * max_y + y) * @N_NEAR_STATIONS * 2;
        // should index the pointer or increment? constant array subscripts in unrolled loop would be better.
        for (int i = 0; i < @N_NEAR_STATIONS; i++) {                                   
            *(p++) = blk_near_idx[i];                                                   
            *(p++) = int(blk_near_dist[i]);
        }
    }
}    
  

/*
 *  [ forces kernel ]
 *  
 *  Finds both network travel times and forces on a texel-by-texel basis.
 */
 
__global__ void forces (
    int   n_stations, //replace with constant
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
    float coord [@DIM];
    float vector[@DIM];
    float force [@DIM]; 
    float *glb_coord;
    float *glb_error;
    float *glb_force;
    float *glb_weight;
    float adjust;
    float error = 0;  // must initialize because error is cumulative per pass
    float error_here;
    float error_max = 0;
    float norm;
    float tt;
    float weight = 0;
    float weight_here;
    int   x = blockIdx.x * blockDim.x + threadIdx.x; 
    int   y = blockIdx.y * blockDim.y + threadIdx.y; 
    int   os_x, os_y, os_idx;
    
    float blk_origin_coord [@DIM];
    float blk_near_dist[@N_NEAR_STATIONS];  // Dist from this texel to the destination station  
    int   blk_near_idx [@N_NEAR_STATIONS];
       
    if (isnan(glb_coords[(x * max_y + y) * @DIM])) return; // should kill off useless threads, but also those that should be loading shared memory. shared mem should be exchanged for registers anyway.
                                                             // actually, it doesn't help. memory accesses must be really slow. calculation is faster.
                                                             
    // when changing code, I see slight differences in the evolution of error images - this is probably due to random initial configuration of points.
    if ( x < max_x && y < max_y ) {          // Check that this thread is inside the map.
        @unroll @N_NEAR_STATIONS
        blk_near_idx  [@I] = tex1Dfetch(tex_near_stations, (x * max_y + y) * @N_NEAR_STATIONS * 2 + @I * 2 + 0);
        if (blk_near_idx[0] == -1) return; // this cell was marked useless because it is too far from transit        
        @unroll @N_NEAR_STATIONS
        blk_near_dist [@I] = tex1Dfetch(tex_near_stations, (x * max_y + y) * @N_NEAR_STATIONS * 2 + @I * 2 + 1);
        
        glb_coord = &( glb_coords[(x * max_y + y) * @DIM] );                             // Make a pointer to this texel's time-space coordinate in global memory.
        @unroll @DIM
        coord[@I] = glb_coord[@I];                              // Copy the time-space coordinate for this texel from global to local memory.
        @unroll @DIM
        force[@I] = 0;                                         // Initialize this timestep's force to 0. (Do this before calculating forces, outside the loops!)
        for (os_idx = s_low; os_idx < s_high; os_idx++) {                               // For every origin station:
            os_x = tex2D(tex_station_coords, os_idx, 0);                            // Get the origin station's geographic x coordinate.
            os_y = tex2D(tex_station_coords, os_idx, 1);                            // Get the origin station's geographic y coordinate.
            float *glb_origin_coord = &(glb_coords[(os_x*max_y + os_y) * @DIM]);     // Make a pointer to the gobal time-space coordinates for this origin station.
            @unroll @DIM
            blk_origin_coord[@I] = glb_origin_coord[@I];                          // Copy origin time-space coordinates from global to block-shared memory. Maybe it should be in register.
            tt = INF;                                                                   // Set the current best travel time to +infinity.
            // Using manhattan distance does not noticeably speed up computation. 
            // Global memory is the bottleneck, so computation is better than a lookup table.
            // Is it OK to unroll long loops, like the nearby stations code, as below?
            // might matrix times be flipped, depending on element ordering conventions?
            
            @unroll @N_NEAR_STATIONS
            tt = min(tt, blk_near_dist[@I] * OBSTRUCTION / WALK_SPEED + tex2D(tex_matrix, os_idx, blk_near_idx[@I])); 

            norm = 0; // Init here, just before accumulation in inner loop

            @unroll @DIM
            vector[@I] = coord[@I] - blk_origin_coord[@I]; norm += pow(vector[@I], 2);                             // Find the vector from the origin station to this texel in time-space.
                                                                                                 // Accumulate terms to find the norm.
            norm   = sqrt(norm);                                                        // Take the square root to find the norm, i.e. the distance in time-space.
            adjust = tt - norm ;                                                        // How much would the point need to move to match the best travel time?
            // global influence scaling like gravity, relative to tt - scale adjustment according to travel time to point
            // weight_here = (tt < 120*60) * (1 - 1 / (120*60 - (tt-1)));
            // turn off weighting
            weight_here = 1;
            // global influence cutoff above T minutes
            // if (tt > 60 * 120) weight_here = 0; else weight_here = 1;
            // weight_here = 1 / (tt + 1);
            weight += weight_here;
                                                                                                                                                                                                                                                                                                                            // use __isnan() ? anyway, this is in the outer loop, and should almost never diverge within a warp.                                                                                                 
            if (norm != 0) {       
                @unroll @DIM                                          // Avoid propagating nans through division by zero. Force should be 0, so add skip this step / add nothing.
                force[@I] += ((vector[@I] / norm) * adjust * weight_here);            // Find a unit vector, scale it by the desired 'correct' time-space distance. (weighted)
                // why is this in the if block?
                error_here = pow(abs(adjust) * weight_here, 2);                                    // Accumulate error to each texel, so we can observe progress as the program runs. (weighted)
                error += error_here;                                        // now using rms error - should weight be inside or outside square? for now it's always 1.
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

        glb_force  = glb_forces  + (x * max_y + y) * @DIM;     // Make a pointer to a force record in global memory. You could do this with C array notation
        glb_error  = glb_errors  + (x * max_y + y)      ;     // Make a pointer to an error record in global memory.
        glb_weight = glb_weights + (x * max_y + y)      ;     // Make a pointer to an error record in global memory.
        if (s_low > 0) {
            @unroll @DIM
            force[@I] += glb_force[@I];     // ADD Output forces to device global memory.
            error  += *glb_error;
            weight += *glb_weight;
            // visualize max error per cell
            error_max = max(error_max, debug_img[ x * max_y + y ]);
        }
        @unroll @DIM
        glb_force[@I]  = force[@I];     // SET Output forces to device global memory.
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
        glb_coord  = &( glb_coords [(x * max_y + y) * @DIM] );    // Make a pointer to time-space coordinates in global memory
        glb_force  = &( glb_forces [(x * max_y + y) * @DIM] );    // Make a pointer to forces in global memory
        glb_weight = &( glb_weights[(x * max_y + y)      ] );    // Make a pointer to forces in global memory
        for (d = 0; d < @DIM; d++) glb_force[d] /= *glb_weight;    // Scale forces by the sum of all weights acting on this cell
        for (d = 0; d < @DIM; d++) glb_coord[d] += glb_force [d]; // Euler integration, 1 timestep
    }
}
