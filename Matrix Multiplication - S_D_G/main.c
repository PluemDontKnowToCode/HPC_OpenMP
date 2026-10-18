#include <stdio.h>
#include "matrix.h"

// Compare OpenMP loop schedules: static, dynamic, guided
//   threads  : 1 and 4
//   workload : 1x, 2x, 4x   (N_k = N_1x * cbrt(k), rounded to a multiple of block size)
//   each setting is repeated NRUNS times; average time and speedup are printed.
//
// Usage: ./main.exe [N_1x] [runs] [chunk]
//   defaults: N_1x = 1600, runs = 10, chunk = 0 (= OpenMP default chunk)
//   quick test: ./main.exe 160 2

#define NTHREADS     2
#define NWORK        3
#define NSCHED       3
#define BLOCK        16
#define N1X_DEFAULT  1600
#define NRUNS_DEFAULT 10

int main( int argc, char **argv )
{
    double *A, *B, *C, *Corg, alpha, beta, time1, time2 ;
    int m, n, p, b ;
    int row, col ;
    char status[100] ;

    int N1x   = ( argc > 1 ) ? atoi( argv[1] ) : N1X_DEFAULT ;
    int nruns = ( argc > 2 ) ? atoi( argv[2] ) : NRUNS_DEFAULT ;
    int chunk = ( argc > 3 ) ? atoi( argv[3] ) : 0 ;

    int mult[NWORK] = { 1, 2, 4 } ;
    int dim[NWORK] ;
    double cbrt_k[NWORK] = { 1.0, 1.259921049894873, 1.587401051968199 } ;
    int thr[NTHREADS] = { 1, 4 } ;
    omp_sched_t sched_kind[NSCHED] = { omp_sched_static, omp_sched_dynamic, omp_sched_guided } ;
    const char *sched_name[NSCHED] = { "static", "dynamic", "guided" } ;
    int s, t, w, run, failed = 0 ;
    double tsum[NSCHED][NWORK][NTHREADS] = {{{0}}} ;

    if ( N1x < BLOCK || nruns < 1 ) { printf( "usage: %s [N_1x>=16] [runs>=1] [chunk>=0]\n", argv[0] ) ; return 1 ; }
    for( w = 0 ; w < NWORK ; w++ )
        dim[w] = (int)( N1x * cbrt_k[w] / BLOCK + 0.5 ) * BLOCK ;
    int MAXN = dim[NWORK-1] ;

    alpha = 1.3 ;
    beta  = 2.9 ;

    A    = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    B    = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    C    = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    Corg = (double *) malloc( sizeof(double) * MAXN * MAXN ) ;
    if ( !A || !B || !C || !Corg ) { printf( "Out of Memory\n" ) ; exit(1) ; }

    for( row = 0 ; row < MAXN ; row++ )
        for( col = 0 ; col < MAXN ; col++ )
        {
            A[row*MAXN+col]    = rand() ;
            B[row*MAXN+col]    = rand() ;
            Corg[row*MAXN+col] = rand() ;
        }

    printf( "Workload N: 1x=%d  2x=%d  4x=%d   block=%d   runs=%d   chunk=%d\n",
            dim[0], dim[1], dim[2], BLOCK, nruns, chunk ) ;

    for( s = 0 ; s < NSCHED ; s++ )
    {
        omp_set_schedule( sched_kind[s], chunk ) ;   /* used by schedule(runtime) */
        for( w = 0 ; w < NWORK ; w++ )
        {
            m = n = p = dim[w] ;
            for( t = 0 ; t < NTHREADS ; t++ )
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

                    omp_set_num_threads( thr[t] ) ;
                    time1 = omp_get_wtime() ;
                    MatMul( m, n, p, b, alpha, beta, A, B, C ) ;
                    time2 = omp_get_wtime() ;

                    /* Validate random entries */
                    int success = 1, r ;
                    for( r = 0 ; r < NUM_VALIDATES ; r++ )
                    {
                        double min, max, err, ans ;
                        row = rand() % m ;
                        col = rand() % p ;

                        int k ;
                        ans = Corg[row*p+col] * beta ;
                        for( k = 0 ; k < n ; k++ )
                            ans += alpha * A[row*n+k]*B[k*p+col] ;

                        if ( ans > C[row*p+col] ) { min = C[row*p+col] ; max = ans ; }
                        else                      { min = ans ; max = C[row*p+col] ; }
                        err = (max-min) / max ;

                        if ( ( min > 0.0001 ) && ( err > 0.01 ) )
                            success = 0 ;
                    }

                    if ( success == 0 ) { strcpy( status, "\x1b[31mFailed\x1b[0m" ) ; failed++ ; }
                    else                  strcpy( status, "\x1b[32mPassed\x1b[0m" ) ;

                    tsum[s][w][t] += time2 - time1 ;
                    printf( "{schedule=%s,workload=%dx,m=%d,n=%d,p=%d,numt=%d,b=%d,run=%d,time=%.4f,status=%s}\n",
                            sched_name[s], mult[w], m, n, p, thr[t], b, run, time2-time1, status ) ;
                    fflush( stdout ) ;
                }
                printf( "\n" ) ;
            }
        }
    }

    /* ---- Result tables ---- */
    double T[NSCHED][NWORK][NTHREADS] ;
    for( s = 0 ; s < NSCHED ; s++ )
        for( w = 0 ; w < NWORK ; w++ )
            for( t = 0 ; t < NTHREADS ; t++ )
                T[s][w][t] = tsum[s][w][t] / nruns ;

    for( s = 0 ; s < NSCHED ; s++ )
    {
        printf( "\n=== schedule(%s%s) ===\n", sched_name[s], chunk ? ",chunk" : "" ) ;
        printf( "| Avg Exe Time (seconds) | Workload = 1x | Workload = 2x | Workload = 4x |\n" ) ;
        printf( "| ---------------------- | ------------- | ------------- | ------------- |\n" ) ;
        for( t = 0 ; t < NTHREADS ; t++ )
        {
            printf( "| Threads=%d              ", thr[t] ) ;
            for( w = 0 ; w < NWORK ; w++ )
                printf( "| T%d%d = %-8.3f ", thr[t], mult[w], T[s][w][t] ) ;
            printf( "|\n" ) ;
        }
        printf( "\n| Speedup   | Workload = 1x | Workload = 2x | Workload = 4x |\n" ) ;
        printf( "| --------- | ------------- | ------------- | ------------- |\n" ) ;
        for( t = 0 ; t < NTHREADS ; t++ )
        {
            printf( "| Threads=%d ", thr[t] ) ;
            for( w = 0 ; w < NWORK ; w++ )
                printf( "| T1%d/T%d%d = %-6.3f ", mult[w], thr[t], mult[w], T[s][w][0] / T[s][w][t] ) ;
            printf( "|\n" ) ;
        }
    }

    printf( "\n=== Compare schedules (4 threads, avg time in seconds) ===\n" ) ;
    printf( "| Schedule | Workload = 1x | Workload = 2x | Workload = 4x |\n" ) ;
    printf( "| -------- | ------------- | ------------- | ------------- |\n" ) ;
    for( s = 0 ; s < NSCHED ; s++ )
    {
        printf( "| %-8s ", sched_name[s] ) ;
        for( w = 0 ; w < NWORK ; w++ )
            printf( "| %-13.3f ", T[s][w][1] ) ;
        printf( "|\n" ) ;
    }

    printf( "\nValidation: %s\n", failed ? "SOME RUNS FAILED" : "all runs Passed" ) ;
    free( A ) ; free( B ) ; free( C ) ; free( Corg ) ;
    return 0 ;
}