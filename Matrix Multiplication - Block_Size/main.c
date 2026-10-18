#include <stdio.h>
#include "matrix.h"

// Input contains three matrices A,B,C and alpha and Beta.
// A of dimensions m X n.
// B of dimensions n X p.
// C of dimensions m X p.
// Output is a matrix D of dimensions m X p.
// Task is to write a routine which computes a NEW matrix C such that:
// C = alpha x A x B + beta x C

// This main function studies the effect of the BLOCK size b on MatMul.
// Fixed matrix size N x N (default N = 1600), block sizes b = 4, 8, 16, 32
// (doubling), thread counts 1 and 4. Each setting is repeated NRUNS times,
// timed with omp_get_wtime(), and the average time + speedup are printed.
//
// Usage: ./main.exe [N] [runs]      (defaults: N = 1600, runs = 10)
//   N must be a multiple of 32 (the largest block), e.g. quick test: ./main.exe 160 2

#define N_DEFAULT     1600
#define NRUNS_DEFAULT 10
#define NBLOCKS       4
#define NTHR          2

//MatMul( m, n, p, b, alpha, beta, A, B, C ) is the function that solves computes the matrix C.

int main( int argc, char **argv )
{
    double *A, *B, *C, *Corg, alpha, beta, time1, time2;
    int m, n, p, b ;
    int row, col, numt ;
    char status[100] ;

    const int blocks[NBLOCKS] = { 4, 8, 16, 32 } ;   /* BLOCK sizes: 4 .. 32, step x2 */
    const int threads[NTHR]   = { 1, 4 } ;           /* thread counts */

    int N     = ( argc > 1 ) ? atoi( argv[1] ) : N_DEFAULT ;
    int nruns = ( argc > 2 ) ? atoi( argv[2] ) : NRUNS_DEFAULT ;
    int bi, ti, run, failed = 0 ;
    double tsum[NBLOCKS][NTHR] = {{0}} ;

    if ( N < 32 || ( N % 32 ) != 0 || nruns < 1 )
    {
        printf( "usage: %s [N: multiple of 32] [runs>=1]\n", argv[0] ) ;
        return 1 ;
    }

    alpha = (double) 1.3 ;
    beta  = (double) 2.9 ;

    A = (double *) malloc( sizeof(double) * N * N ) ;
    B = (double *) malloc( sizeof(double) * N * N ) ;
    C = (double *) malloc( sizeof(double) * N * N ) ;
    Corg = (double *) malloc( sizeof(double) * N * N ) ;
    if ( ( A == NULL ) || ( B == NULL ) || ( C == NULL ) || ( Corg == NULL ) )
    {
        printf( "Out of Memory\n" ) ;
        exit(1) ;
    }

    for( row = 0 ; row < N ; row++ )
        for( col = 0 ; col < N ; col++ )
            A[row*N+col] = rand() ;

    for( row = 0 ; row < N ; row++ )
        for( col = 0 ; col < N ; col++ )
            B[row*N+col] = rand() ;

    for( row = 0 ; row < N ; row++ )
        for( col = 0 ; col < N ; col++ )
            Corg[row*N+col] = rand() ;

    m = n = p = N ;
    printf( "N=%d   blocks=4,8,16,32   threads=1,4   runs=%d\n", N, nruns ) ;

    for( bi = 0 ; bi < NBLOCKS ; bi++ )
    {
        for( ti = 0 ; ti < NTHR ; ti++ )
        {
            numt = threads[ti] ;
            for( run = 1 ; run <= nruns ; run++ )
            {
                for( row = 0 ; row < m ; row++ )
                    for( col = 0 ; col < p ; col++ )
                        C[row*p+col] = Corg[row*p+col] ;
                b = blocks[bi] ;
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

                tsum[bi][ti] += time2 - time1 ;
                printf( "{m=%d,n=%d,p=%d,numt=%d,b=%d,run=%d,time=%.4f,status=%s}\n",
                          m, n, p, numt, b, run, time2-time1, status ) ;
                fflush( stdout ) ;
            }
            printf("\n");
        }
    }

    /* ---- Result tables ---- */
    double T[NBLOCKS][NTHR] ;
    int best[NTHR] = { 0, 0 } ;
    for( bi = 0 ; bi < NBLOCKS ; bi++ )
        for( ti = 0 ; ti < NTHR ; ti++ )
        {
            T[bi][ti] = tsum[bi][ti] / nruns ;
            if ( T[bi][ti] < T[best[ti]][ti] ) best[ti] = bi ;
        }

    printf( "\nAvg Exe Time (seconds), N=%d, avg of %d runs  (* = fastest block for that thread count)\n", N, nruns ) ;
    printf( "| Block b | Threads=1       | Threads=4       | Speedup T1/T4 |\n" ) ;
    printf( "| ------- | --------------- | --------------- | ------------- |\n" ) ;
    for( bi = 0 ; bi < NBLOCKS ; bi++ )
    {
        printf( "| %-7d ", blocks[bi] ) ;
        for( ti = 0 ; ti < NTHR ; ti++ )
            printf( "| %-8.3f %-6s ", T[bi][ti], ( best[ti] == bi ) ? "*" : "" ) ;
        printf( "| %-13.3f |\n", T[bi][0] / T[bi][1] ) ;
    }

    printf( "\nTime relative to b=16 (value < 1.00 = faster than b=16)\n" ) ;
    printf( "| Block b | Threads=1 | Threads=4 |\n" ) ;
    printf( "| ------- | --------- | --------- |\n" ) ;
    for( bi = 0 ; bi < NBLOCKS ; bi++ )
        printf( "| %-7d | %-9.3f | %-9.3f |\n", blocks[bi], T[bi][0] / T[2][0], T[bi][1] / T[2][1] ) ;

    printf( "\nValidation: %s\n", failed ? "SOME RUNS FAILED" : "all runs Passed" ) ;
    return 0;
}