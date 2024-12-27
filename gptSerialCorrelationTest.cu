// gptSerialCorrelationTest.cu

#include "gptRandom.h"
#include <cuda.h>
#include <iostream>

// Kernel for calculating the serial correlation of the array elements
__global__ void gptSerialCorrelationKernel(dfloat* array, dfloat* sumX, dfloat* sumY, dfloat* sumXY, dfloat* sumX2, dfloat* sumY2, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Variables to store sums for correlation calculation
    dfloat x, y;
    dfloat partialSumX = 0.0, partialSumY = 0.0;
    dfloat partialSumXY = 0.0, partialSumX2 = 0.0, partialSumY2 = 0.0;

    if (idx < N - 1) {
        x = array[idx];
        y = array[idx + 1];

        partialSumX += x;
        partialSumY += y;
        partialSumXY += x * y;
        partialSumX2 += x * x;
        partialSumY2 += y * y;
    }

    // Use atomic operations to accumulate the results across threads
    atomicAdd(sumX, partialSumX);
    atomicAdd(sumY, partialSumY);
    atomicAdd(sumXY, partialSumXY);
    atomicAdd(sumX2, partialSumX2);
    atomicAdd(sumY2, partialSumY2);
}

// Launcher function for the serial correlation test
void gptSerialCorrelationLauncher(dfloat* array, int N, dfloat* serialCorrelationResult) {
    dfloat *d_sumX, *d_sumY, *d_sumXY, *d_sumX2, *d_sumY2;
    cudaMalloc((void**)&d_sumX, sizeof(dfloat));
    cudaMalloc((void**)&d_sumY, sizeof(dfloat));
    cudaMalloc((void**)&d_sumXY, sizeof(dfloat));
    cudaMalloc((void**)&d_sumX2, sizeof(dfloat));
    cudaMalloc((void**)&d_sumY2, sizeof(dfloat));
    cudaMemset(d_sumX, 0, sizeof(dfloat));
    cudaMemset(d_sumY, 0, sizeof(dfloat));
    cudaMemset(d_sumXY, 0, sizeof(dfloat));
    cudaMemset(d_sumX2, 0, sizeof(dfloat));
    cudaMemset(d_sumY2, 0, sizeof(dfloat));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing for throughput model
    cudaEvent_t tic, toc;
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    cudaEventRecord(tic);
    // Launch kernel
    gptSerialCorrelationKernel<<<blocksPerGrid, threadsPerBlock>>>(array, d_sumX, d_sumY, d_sumXY, d_sumX2, d_sumY2, N);
    cudaEventRecord(toc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    float elapsed;
    cudaEventElapsedTime(&elapsed, tic, toc);

    size_t bytes = (N + 5) * sizeof(dfloat); // Data array + 5 sums
    int flop = 8 * N; // metric from nvprof
    FILE* out = fopen("gptSerialCorrelationTiming.data", "a");
    fprintf(out, "%d, %7.5e, %7.5e, %7.5e, %%%% N, time (s), memory throughput (bytes/s), flop throughput (GFLOPS/s)\n",
        N, elapsed, (bytes / 1.e6) / elapsed, (flop / 1.e6) / elapsed);
    fclose(out);

    // Copy results back to host
    dfloat sumX, sumY, sumXY, sumX2, sumY2;
    cudaMemcpy(&sumX, d_sumX, sizeof(dfloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumY, d_sumY, sizeof(dfloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumXY, d_sumXY, sizeof(dfloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumX2, d_sumX2, sizeof(dfloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumY2, d_sumY2, sizeof(dfloat), cudaMemcpyDeviceToHost);

    cudaFree(d_sumX);
    cudaFree(d_sumY);
    cudaFree(d_sumXY);
    cudaFree(d_sumX2);
    cudaFree(d_sumY2);

    // Calculate serial correlation using the formula
    dfloat numerator = (N * sumXY - sumX * sumY);
    dfloat denominator =  sqrt((N * sumX2 - sumX * sumX) * (N * sumY2 - sumY * sumY));
    *serialCorrelationResult = (denominator != 0) ? numerator / denominator : 0.0;
}

// Driver function for the serial correlation test
void gptSerialCorrelationTest(dfloat* array, int N) {
    dfloat serialCorrelation;
    gptSerialCorrelationLauncher(array, N, &serialCorrelation);

    std::cout << "Serial Correlation Test:\n";
    std::cout << "This test checks the correlation between successive values in the array. For a random sequence, this should be close to 0.\n";
    std::cout << "Computed serial correlation: " << serialCorrelation << "\n";

    // Calculate quality rating based on closeness to zero
    int quality = 100 - static_cast<int>(fabs(serialCorrelation) * 200); // Scaling to penalize large deviations
    quality = max(0, quality); // Clamp quality to range [0, 100]

    std::cout << "Computed Serial Correlation Quality Rating: " << quality << "/100\n";
    std::cout << "Test completed.\n";
}
