#include <stdio.h>
#include "matrix.h"

// Input contains three matrices A,B,C and alpha and Beta.
// A of dimensions m X n.
// B of dimensions n X p.
// C of dimensions m X p.
// Output is a matrix D of dimensions m X p.
// Task is to write a routine which computes a NEW matrix C such that:
// C = alpha x A x B + beta x C

// This main function measures the execution time (omp_get_wtime) of MatMul
// with 1,2,3,4 threads and with workload = 1x, 2x, 4x, repeating each
// setting NRUNS times, then prints the average-time table and the speedup table.
//
// Workload: matrix multiplication costs O(N^3), so workload kx uses
// N_k = N_1x * cbrt(k) (rounded to a multiple of the block size b).
// N_1x is chosen so that 1 thread takes about 20 s at 1x on the test machine.
//
// Usage: ./main [N_1x] [runs]      (defaults: N_1x = 1600, runs = 10)
//   e.g. quick test: ./main 160 2

#define NTHREADS_MAX 4
#define NWORK        3
#define BLOCK        16            /* block size b (same as original, numt < 4) */
#define N1X_DEFAULT  2000
#define NRUNS_DEFAULT 10

//MatMul( m, n, p, b, alpha, beta, A, B, C ) is the function that solves computes the matrix C.

int main( int argc, char **argv )
{
    double *A, *B, *C, *Corg, alpha, beta, time1, time2;
    int m, n, p, b ;
    int row, col, numt ;
    char status[100] ;

    int N1x   = ( argc > 1 ) ? atoi( argv[1] ) : N1X_DEFAULT ;
    int nruns = ( argc > 2 ) ? atoi( argv[2] ) : NRUNS_DEFAULT ;
    int mult[NWORK] = { 1, 2, 4 } ;          /* workload factors */
    int dim[NWORK] ;
    double cbrt_k[NWORK] = { 1.0, 1.259921049894873, 1.587401051968199 } ; /* cbrt(1,2,4) */
    int w, run, failed = 0 ;
    double tsum[NWORK][NTHREADS_MAX+1] = {{0}} ;

    if ( N1x < BLOCK || nruns < 1 ) { printf( "usage: %s [N_1x>=16] [runs>=1]\n", argv[0] ) ; return 1 ; }
    for( w = 0 ; w < NWORK ; w++ )
        dim[w] = (int)( N1x * cbrt_k[w] / BLOCK + 0.5 ) * BLOCK ;
    int MAXN = dim[NWORK-1] ;                /* largest matrix size (4x) */

    alpha = (double) 1.3 ;
    beta  = (double) 2.9 ;

    A = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    B = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    C = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    Corg = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    if ( ( A == NULL ) || ( B == NULL ) || ( C == NULL ) || ( Corg == NULL ) )
    {
        printf( "Out of Memory\n" ) ;
        exit(1) ;
    }

    for( row = 0 ; row < MAXN ; row++ )
        for( col = 0 ; col < MAXN ; col++ )
            A[row*MAXN+col] = rand() ;

    for( row = 0 ; row < MAXN ; row++ )
        for( col = 0 ; col < MAXN ; col++ )
            B[row*MAXN+col] = rand() ;

    for( row = 0 ; row < MAXN ; row++ )
        for( col = 0 ; col < MAXN ; col++ )
            Corg[row*MAXN+col] = rand() ;

    printf( "Workload N: 1x=%d  2x=%d  4x=%d   block=%d   runs=%d\n", dim[0], dim[1], dim[2], BLOCK, nruns ) ;

    for( w = 0 ; w < NWORK ; w++ )
    {
        m = n = p = dim[w] ;
        for( numt = 1 ; numt <= NTHREADS_MAX ; numt++ )
        {
            for( run = 1 ; run <= nruns ; run++ )
            {
                for( row = 0 ; row < m ; row++ )
                    for( col = 0 ; col < p ; col++ )
                        C[row*p+col] = Corg[row*p+col] ;
                b = BLOCK ;
                if ( b > m ) b = m ;
                if ( b > n ) b = n ;
                if ( b > p ) b = p ;

                omp_set_num_threads( numt ) ;
                time1 = omp_get_wtime() ;
                MatMul( m, n, p, b, alpha, beta, A, B, C ) ;
                time2 = omp_get_wtime() ;

                /* Validate random entries */
                int success = 1, r ;
                for(  r = 0 ; r < NUM_VALIDATES ; r++ )
                {
                    double min, max, err, ans ;

                    row = ( rand() % m ) ;
                    col = ( rand() % p ) ;

                    int k ;
                    ans = Corg[row*p+col] * beta ;
                    for( k = 0 ; k < n ; k++ )
                        ans += alpha * A[row*n+k]*B[k*p+col] ;

                    if (ans > C[row*p+col])
                    {
                        min = C[row*p+col] ;
                        max = ans ;
                    }
                    else
                    {
                        min = ans ;
                        max = C[row*p+col] ;
                    }
                    err = (double) (max-min)/ (double) max ;

                    if ( ( min > (double) 0.0001 ) && ( err > (double) 0.01 ) )
                        success = 0 ;
                }

                if ( success == 0 ) { strcpy( status, "\x1b[31mFailed\x1b[0m" ) ; failed++ ; }
                else
                    strcpy( status, "\x1b[32mPassed\x1b[0m" ) ;

                tsum[w][numt] += time2 - time1 ;
                printf( "{workload=%dx,m=%d,n=%d,p=%d,numt=%d,b=%d,run=%d,time=%.4f,status=%s}\n",
                          mult[w], m, n, p, numt, b, run, time2-time1, status ) ;
                fflush( stdout ) ;
            }
            printf("\n");
        }
    }

    /* ---- Result tables ---- */
    double T[NWORK][NTHREADS_MAX+1] ;
    for( w = 0 ; w < NWORK ; w++ )
        for( numt = 1 ; numt <= NTHREADS_MAX ; numt++ )
            T[w][numt] = tsum[w][numt] / nruns ;

    printf( "\n| Avg Exe Time (seconds) | Workload = 1x | Workload = 2x | Workload = 4x |\n" ) ;
    printf( "| ---------------------- | ------------- | ------------- | ------------- |\n" ) ;
    for( numt = 1 ; numt <= NTHREADS_MAX ; numt++ )
    {
        printf( "| Threads=%d              ", numt ) ;
        for( w = 0 ; w < NWORK ; w++ )
            printf( "| T%d%d = %-8.3f ", numt, mult[w], T[w][numt] ) ;
        printf( "|\n" ) ;
    }

    printf( "\n| Speedup   | Workload = 1x | Workload = 2x | Workload = 4x |\n" ) ;
    printf( "| --------- | ------------- | ------------- | ------------- |\n" ) ;
    for( numt = 1 ; numt <= NTHREADS_MAX ; numt++ )
    {
        printf( "| Threads=%d ", numt ) ;
        for( w = 0 ; w < NWORK ; w++ )
            printf( "| T1%d/T%d%d = %-6.3f ", mult[w], numt, mult[w], T[w][1] / T[w][numt] ) ;
        printf( "|\n" ) ;
    }

    printf( "\nValidation: %s\n", failed ? "SOME RUNS FAILED" : "all runs Passed" ) ;
    return 0;
}