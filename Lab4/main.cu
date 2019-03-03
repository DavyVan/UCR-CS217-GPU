/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

const unsigned int numStream = 3;

int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d[numStream], *B_d[numStream], *C_d[numStream];
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize;

    // Initialize streams
    cudaStream_t streams[numStream];
    for (int i = 0; i < numStream; i++)
        cudaStreamCreate(&streams[i]);
   
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 1000000;
    } else if (argc == 2) {
        VecSize = atoi(argv[1]);   
    } else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    const int segmentLen = VecSize / numStream;

    // A_h = (float*) malloc( sizeof(float)*A_sz );
    cudaHostAlloc((void**)&A_h, A_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    // B_h = (float*) malloc( sizeof(float)*B_sz );
    cudaHostAlloc((void**)&B_h, B_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    // C_h = (float*) malloc( sizeof(float)*C_sz );
    cudaHostAlloc((void**)&C_h, C_sz*sizeof(float), cudaHostAllocDefault);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    // cudaMalloc((float**) &A_d, sizeof(float) * VecSize);
    // cudaMalloc((float**) &B_d, sizeof(float) * VecSize);
    // cudaMalloc((float**) &C_d, sizeof(float) * VecSize);
    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cudaMalloc((float**) &A_d[i], sizeof(float) * segmentLen);
            cudaMalloc((float**) &B_d[i], sizeof(float) * segmentLen);
            cudaMalloc((float**) &C_d[i], sizeof(float) * segmentLen);
        }
        else    // remainder
        {
            cudaMalloc((float**) &A_d[i], sizeof(float) * (segmentLen + VecSize % numStream));
            cudaMalloc((float**) &B_d[i], sizeof(float) * (segmentLen + VecSize % numStream));
            cudaMalloc((float**) &C_d[i], sizeof(float) * (segmentLen + VecSize % numStream));
        }
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    // cudaMemcpy(A_d, A_h, sizeof(float) * VecSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(B_d, B_h, sizeof(float) * VecSize, cudaMemcpyHostToDevice);
    
    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cudaMemcpyAsync(A_d[i], A_h + i*segmentLen, sizeof(float)*segmentLen, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(B_d[i], B_h + i*segmentLen, sizeof(float)*segmentLen, cudaMemcpyHostToDevice, streams[i]);
        }
        else
        {
            cudaMemcpyAsync(A_d[i], A_h + i*segmentLen, sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(B_d[i], B_h + i*segmentLen, sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyHostToDevice, streams[i]);
        }
    }

    // cudaDeviceSynchronize();
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    // startTime(&timer);

    // basicVecAdd(A_d, B_d, C_d, VecSize); //In kernel.cu
    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            basicVecAdd(A_d[i], B_d[i], C_d[i], segmentLen, streams[i]);
        }
        else
        {
            basicVecAdd(A_d[i], B_d[i], C_d[i], segmentLen + VecSize % numStream, streams[i]);
        }
    }
        
    // cuda_ret = cudaDeviceSynchronize();
	// if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    // startTime(&timer);

    //INSERT CODE HERE
    // cudaMemcpy(C_h, C_d, sizeof(float) * VecSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream)
        {
            printf("stream-%d, offset=%d, size=%d\n", i, i*segmentLen, segmentLen);
            cudaMemcpyAsync(C_h + i*segmentLen, C_d[i], sizeof(float)*segmentLen, cudaMemcpyDeviceToHost, streams[i]);
        }
        else
        {
            printf("stream-%d, offset=%d, size=%d, VecSize%%numStream=%u%%%u=%d\n", i, i*segmentLen, segmentLen + VecSize % numStream, VecSize, numStream, VecSize%numStream);
            cudaMemcpyAsync(C_h + i*segmentLen, C_d[i], sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyDeviceToHost, streams[i]);
        }
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

    // free(A_h);
    // free(B_h);
    // free(C_h);
    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);

    //INSERT CODE HERE
    // cudaFree(A_d);
    // cudaFree(B_d);
    // cudaFree(C_d);
    for (int i = 0; i < numStream; i++)
    {
        cudaFree(A_d[i]);
        cudaFree(B_d[i]);
        cudaFree(C_d[i]);
        cudaStreamDestroy(streams[i]);
    }
    return 0;

}
