// xaSerialCorrelationTest.cu

#include "gptRandom.h"
#include <cuda.h>
#include <iostream>

#define p_mask 0xffffffff

__forceinline__ __device__ dfloat warpSum(const dfloat v) {
    int t = threadIdx.x; // just lane info
    dfloat sum = v;
    sum += __shfl_sync(p_mask, sum, t ^ 1);
    sum += __shfl_sync(p_mask, sum, t ^ 2);
    sum += __shfl_sync(p_mask, sum, t ^ 4);
    sum += __shfl_sync(p_mask, sum, t ^ 8);
    sum += __shfl_sync(p_mask, sum, t ^ 16);
    return sum;
}

// Kernel for calculating the serial correlation of the array elements
template <int p_Nloads, int p_Nwarp>
__global__ void xaSerialCorrelationKernel(dfloat* array, dfloat* sumX, dfloat* sumY, dfloat* sumXY, dfloat* sumX2, dfloat* sumY2, int N) {

    __shared__ dfloat s_sumX[p_Nwarp];
    __shared__ dfloat s_sumY[p_Nwarp];
    __shared__ dfloat s_sumXY[p_Nwarp];
    __shared__ dfloat s_sumX2[p_Nwarp];
    __shared__ dfloat s_sumY2[p_Nwarp];

    int t = threadIdx.x; // thread within warp
    int w = threadIdx.y; // warp within block
    int b = blockIdx.x; // block within grid

    dfloat r_sumX = 0.0, r_sumY = 0.0;
    dfloat r_sumXY = 0.0, r_sumX2 = 0.0, r_sumY2 = 0.0;

#pragma unroll
    for (int r = 0; r < p_Nloads; r++) {
        const int m = t + 32 * (r + w * p_Nloads + b * p_Nloads * p_Nwarp);
        if (m < N - 1) {
            dfloat x = array[m];
            dfloat y = array[m + 1];
            r_sumX += x;
            r_sumY += y;
            r_sumXY += x * y;
            r_sumX2 += x * x;
            r_sumY2 += y * y;
        }
    }

    r_sumX = warpSum(r_sumX);
    r_sumY = warpSum(r_sumY);
    r_sumXY = warpSum(r_sumXY);
    r_sumX2 = warpSum(r_sumX2);
    r_sumY2 = warpSum(r_sumY2);

    if (t == 0) {
        s_sumX[w] = r_sumX;
        s_sumY[w] = r_sumY;
        s_sumXY[w] = r_sumXY;
        s_sumX2[w] = r_sumX2;
        s_sumY2[w] = r_sumY2;
    }
    __syncthreads();

    if (w == 0) {
        r_sumX = warpSum(s_sumX[t]);
        r_sumY = warpSum(s_sumY[t]);
        r_sumXY = warpSum(s_sumXY[t]);
        r_sumX2 = warpSum(s_sumX2[t]);
        r_sumY2 = warpSum(s_sumY2[t]);
        if (t == 0) {
            atomicAdd(sumX, r_sumX);
            atomicAdd(sumY, r_sumY);
            atomicAdd(sumXY, r_sumXY);
            atomicAdd(sumX2, r_sumX2);
            atomicAdd(sumY2, r_sumY2);
        }
    }
}

// Launcher function for the serial correlation test
void xaSerialCorrelationLauncher(dfloat* array, int N, dfloat* serialCorrelationResult) {
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

    // Timing throughput
    cudaEvent_t tic, toc;
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    cudaEventRecord(tic);
    const int p_Nloads = 4; // the number of samples processed by each thread
    const int p_Nwarp = 32; // Number of warps per block
    dim3 threadsPerBlock(32, p_Nwarp);
    dim3 blocksPerGrid((N + p_Nloads * 32 * p_Nwarp - 1) / (p_Nloads * 32 * p_Nwarp));

    // Launch kernel
    xaSerialCorrelationKernel<p_Nloads, p_Nwarp><<<blocksPerGrid, threadsPerBlock>>>(array, d_sumX, d_sumY, d_sumXY, d_sumX2, d_sumY2, N);
    cudaEventRecord(toc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    float elapsed;
    cudaEventElapsedTime(&elapsed, tic, toc);

    size_t bytes = (N + 5) * sizeof(dfloat); // Data array + 5 sums
    int flop = 21504 * threadsPerBlock.x * threadsPerBlock.y * blocksPerGrid.x; // metric from nvprof N = 1
    FILE* out = fopen("xaSerialCorrelationTiming.data", "a");
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
    dfloat denominator = sqrt((N * sumX2 - sumX * sumX) * (N * sumY2 - sumY * sumY));
    *serialCorrelationResult = (denominator != 0) ? numerator / denominator : 0.0;
}

// Driver function for the serial correlation test
void xaSerialCorrelationTest(dfloat* array, int N) {
    dfloat serialCorrelation;
    xaSerialCorrelationLauncher(array, N, &serialCorrelation);

    std::cout << "XA Serial Correlation Test:\n";
    std::cout << "This test checks the correlation between successive values in the array. For a random sequence, this should be close to 0.\n";
    std::cout << "Computed serial correlation: " << serialCorrelation << "\n";

    // Calculate quality rating based on closeness to zero
    int quality = 100 - static_cast<int>(fabs(serialCorrelation) * 200); // Scaling to penalize large deviations
    quality = max(0, quality); // Clamp quality to range [0, 100]

    std::cout << "Computed Serial Correlation Quality Rating: " << quality << "/100\n";
    std::cout << "Test completed.\n";
}
