/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

const unsigned int BLOCK_SIZE = 512; 

__global__ void VecAdd(int n, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}


void basicVecAdd( float *A,  float *B, float *C, int n, cudaStream_t stream)
{

    // Initialize thread block and kernel grid dimensions ---------------------

    //INSERT CODE HERE
    dim3 DimGrid((n-1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    VecAdd <<<DimGrid, DimBlock, 0, stream>>> (n, A, B, C);
}

