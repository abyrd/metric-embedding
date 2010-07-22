// this version works differently from the GPU version, from a per-station perspective.

#include <math.h>
#include <stdio.h>

#define ITERATIONS 1
#define DIM 6
#define XMAX 700
#define YMAX 500 / 2   // divide by 2 to simulate unreachable cells
#define N_STATIONS 60
#define N_NEARBY   400 // 400 cells in a 2km block

// variable declarations
int coords[XMAX][YMAX][DIM];
int forces[XMAX][YMAX][DIM];
int matrix[N_STATIONS][N_STATIONS];
int nearby[N_STATIONS][N_NEARBY];
int s_coords[N_STATIONS][DIM];

int main () {
    for (int iter=0; iter<ITERATIONS; iter++) {
        printf("iteration %d\n", iter);
        for (int si=0; si < N_STATIONS; si++) {
            printf("station %d\n", si);
            int best_t = 99999999;
            for (int si2=0; si2<N_STATIONS; si2++) {
                int t = matrix[si][si2];
                for (int ni=0; ni<N_NEARBY; ni++) {
                    int t2 = t + nearby[si2][ni];
                    if (t2<best_t) best_t = t2;
                }
            }
            for (int x=0; x<XMAX; x++) {
                for (int y=0; y<YMAX; y++) {
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
