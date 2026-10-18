#include <stdio.h>
#include "matrix.h"

// C = alpha x A x B + beta x C  (block algorithm, block size b)
// Fix vs original: Ctemp was ONE shared buffer malloc'ed outside the parallel
// region, so threads overwrote each other's b x b temp block (data race).
// Now every thread allocates its own Ctemp inside the parallel region.
// Also: Ctemp is freed (original leaked it).

void MatMul( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *C )
{
    #pragma omp parallel
    {
        int i, j, k, ii, jj, kk;
        double *Ctemp = (double*)malloc( (size_t)b*b*sizeof(double) );

        #pragma omp for collapse(2)
        for( i = 0; i < m; i += b )
        {
            for( j = 0; j < p; j += b )
            {
                for( ii = i; ii < i+b; ii++ )
                    for( jj = j; jj < j+b; jj++ )
                    {
                        Ctemp[(ii-i)*b + (jj-j)] = C[ii*p + jj];
                        C[ii*p + jj] = 0;
                    }

                for( k = 0; k < n; k += b )
                    for( ii = i; ii < i+b; ii++ )
                        for( jj = j; jj < j+b; jj++ )
                        {
                            for( kk = k; kk < k+b; kk++ )
                                C[ii*p+jj] += A[ii*n+kk] * B[kk*p+jj];
                        }

                for( ii = i; ii < i+b; ii++ )
                    for( jj = j; jj < j+b; jj++ )
                        C[ii*p+jj] = alpha * C[ii*p+jj] + beta * Ctemp[(ii-i)*b + (jj-j)];
            }
        }
        free( Ctemp );
    }
}