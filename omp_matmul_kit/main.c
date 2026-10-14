#include <stdio.h>
#include "matrix.h"

// Usage: ./main N threads runs [b]
//   N       : matrix dimension (m = n = p = N, must be a multiple of b)
//   threads : number of OpenMP threads
//   runs    : number of repetitions (each timed with omp_get_wtime)
//   b       : block size (default 16)
// Prints one line per run: N,threads,b,run,time_seconds,status

int main( int argc, char **argv )
{
    if( argc < 4 ) { fprintf(stderr,"usage: %s N threads runs [b]\n", argv[0]); return 1; }
    int N = atoi(argv[1]), numt = atoi(argv[2]), runs = atoi(argv[3]);
    int b = (argc > 4) ? atoi(argv[4]) : 16;
    if( N % b ) { fprintf(stderr,"N must be a multiple of b\n"); return 1; }

    double alpha = 1.3, beta = 2.9;
    size_t sz = (size_t)N*N*sizeof(double);
    double *A = malloc(sz), *B = malloc(sz), *C = malloc(sz), *Corg = malloc(sz);
    if( !A || !B || !C || !Corg ) { printf("Out of Memory\n"); exit(1); }

    srand(12345);
    for( size_t x = 0; x < (size_t)N*N; x++ ) { A[x]=rand(); B[x]=rand(); Corg[x]=rand(); }

    omp_set_num_threads( numt );
    for( int run = 1; run <= runs; run++ )
    {
        memcpy( C, Corg, sz );

        double t1 = omp_get_wtime();
        MatMul( N, N, N, b, alpha, beta, A, B, C );
        double t2 = omp_get_wtime();

        int success = 1;
        for( int r = 0; r < NUM_VALIDATES; r++ )
        {
            int row = rand() % N, col = rand() % N;
            double ans = Corg[(size_t)row*N+col] * beta;
            for( int k = 0; k < N; k++ )
                ans += alpha * A[(size_t)row*N+k] * B[(size_t)k*N+col];
            double c = C[(size_t)row*N+col];
            double mn = ans < c ? ans : c, mx = ans < c ? c : ans;
            if( mn > 0.0001 && (mx-mn)/mx > 0.01 ) success = 0;
        }
        printf( "%d,%d,%d,%d,%.4f,%s\n", N, numt, b, run, t2-t1, success ? "Passed" : "Failed" );
        fflush( stdout );
    }
    free(A); free(B); free(C); free(Corg);
    return 0;
}
