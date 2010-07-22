// this version works just like the GPU version, from a per-pixel perspective.

#include <math.h>
#include <stdio.h>

#define ITERATIONS 1
#define DIM 6
#define XMAX 700
#define YMAX 500
#define N_STATIONS 60
#define N_NEARBY   15

// variable declarations
int coords[XMAX][YMAX][DIM];
int forces[XMAX][YMAX][DIM];
int matrix[N_STATIONS][N_STATIONS];
int nearby[XMAX][YMAX][N_NEARBY][2];
int s_coords[N_STATIONS][DIM];

int main () {
    for (int iter=0; iter<ITERATIONS; iter++) {
        printf("iteration %d\n", iter);
        for (int si=0; si < N_STATIONS; si++) {
            printf("station %d\n", si);
            for (int x=0; x<XMAX; x++) {
                for (int y = 0; y<YMAX; y++) {
                    int best_t = 99999999;
                    for (int ni=0; ni<N_NEARBY; ni++) {
                        int si2 = nearby[x][y][ni][0];
                        int t = matrix[si][si2] + nearby[x][y][ni][1];
                        if (t<best_t) best_t = t;
                    }
                    int v[DIM];                                    
                    float l = 0;            
                    for (int d=0; d<DIM; d++) {
                        v[d] = coords[x][y][d] - s_coords[si][d];
                        l += v[d];
                    }
                    l = sqrt(l);
                    int adjust = best_t - l;
                    for (int d=0; d<DIM; d++) v[d] = v[d] * adjust / l;
                    for (int d=0; d<DIM; d++) forces[x][y][d] += v[d];
                } // for y
            } // for x
        } // for si
    } // for iter
} // main
