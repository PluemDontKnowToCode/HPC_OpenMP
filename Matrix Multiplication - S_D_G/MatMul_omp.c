#include <stdio.h>
#include "matrix.h"

// OpenMP code to parallelize C = alpha x A x B + beta x C using block multiplication.
// The scheduling policy (static / dynamic / guided) is chosen at run time via
// schedule(runtime); main.c selects it with omp_set_schedule().
//
// Fixes vs. the original version:
//  - Ctemp is now allocated PER THREAD (the original shared one buffer between
//    all threads -> data race -> wrong results when numt > 1).
//  - Ctemp is freed (the original leaked it).

void MatMul( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *C )
{
    #pragma omp parallel
    {
        double *Ctemp = (double *) malloc( b * b * sizeof(double) ) ;
        int ii, jj, kk, k ;

        #pragma omp for collapse(2) schedule(runtime)
        for( int i = 0 ; i < m ; i += b )
        {
            for( int j = 0 ; j < p ; j += b )
            {
                for( ii = i ; ii < i+b ; ii++ )
                    for( jj = j ; jj < j+b ; jj++ )
                    {
                        Ctemp[(ii-i)*b + (jj-j)] = C[ii*p + jj] ;
                        C[ii*p + jj] = 0 ;
                    }

                for( k = 0 ; k < n ; k += b )
                    for( ii = i ; ii < i+b ; ii++ )
                        for( jj = j ; jj < j+b ; jj++ )
                            for( kk = k ; kk < k+b ; kk++ )
                                C[ii*p+jj] += A[ii*n+kk] * B[kk*p+jj] ;

                for( ii = i ; ii < i+b ; ii++ )
                    for( jj = j ; jj < j+b ; jj++ )
                        C[ii*p+jj] = alpha * C[ii*p + jj] + beta * Ctemp[(ii-i)*b + (jj-j)] ;
            }
        }
        free( Ctemp ) ;
    }
}