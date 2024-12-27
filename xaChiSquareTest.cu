// xaChiSquareTest.cu

#include "gptRandom.h"
#include <cuda.h>
#include <iostream>

#define NUM_BINS 50 // Number of bins for the Chi-Square test
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

// Kernel to count values in each bin
template <int p_Nloads, int p_Nwarp>
__global__ void xaChiSquareKernel(dfloat* array, int* binCounts, int N) {

    __shared__ int s_binCounts[NUM_BINS][p_Nwarp];

    int t = threadIdx.x; // thread within warp
    int w = threadIdx.y; // warp within block
    int b = blockIdx.x; // block within grid

    int r_binCounts[NUM_BINS] = { 0 };

#pragma unroll
    for (int r = 0; r < p_Nloads; r++) {
        const int m = t + 32 * (r + w * p_Nloads + b * p_Nloads * p_Nwarp);
        dfloat xm = (m < N) ? array[m] : 0.f;
        int bin = min(static_cast<int>(xm * NUM_BINS), NUM_BINS - 1);
        r_binCounts[bin] += (m < N) ? 1 : 0;
    }

    // warp reduction
#pragma unroll
    for (int i = 0; i < NUM_BINS; i++) {
        r_binCounts[i] = warpSum(r_binCounts[i]);
    }

    // block reduction
#pragma unroll
    for (int i = 0; i < NUM_BINS; i++) {
        if (t == 0) {
            s_binCounts[i][w] = r_binCounts[i];
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < NUM_BINS; i++) {
        r_binCounts[i] = warpSum(s_binCounts[i][t]);
        if (t == 0 && w == 0) {
            atomicAdd(binCounts + i, r_binCounts[i]);
        }
    }
}

// Launcher function for the Chi-Square test
void xaChiSquareLauncher(dfloat* array, int N, dfloat* chiSquareResult) {
    int* d_binCounts;
    cudaMalloc((void**)&d_binCounts, NUM_BINS * sizeof(int));
    cudaMemset(d_binCounts, 0, NUM_BINS * sizeof(int));

    const int p_Nloads = 19; // the number of samples processed by each thread
    const int p_Nwarp = 32; // Number of warps per block
    dim3 threadsPerBlock(32, p_Nwarp);
    dim3 blocksPerGrid((N + p_Nloads * 32 * p_Nwarp - 1) / (p_Nloads * 32 * p_Nwarp));

    // timing for througput model
    cudaEvent_t tic, toc;
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    cudaEventRecord(tic);
    // Launch kernel to count occurrences in bins
    xaChiSquareKernel<p_Nloads, p_Nwarp><<<blocksPerGrid, threadsPerBlock>>>(array, d_binCounts, N);
    cudaEventRecord(toc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    float elapsed;
    cudaEventElapsedTime(&elapsed, tic, toc);

    size_t bytes = 2 * N * sizeof(dfloat);
    int flop = 1 * N;
    FILE* out = fopen("xaChiSquareTiming.data", "a");
    fprintf(out, "%d, %7.5e, %7.5e, %7.5e, %%%% N, time (s), memory throughput (bytes/s), flop throughput (GFLOPS/s)\n",
        N, elapsed, (bytes / 1.e6) / elapsed, (flop / 1.e6) / elapsed);
    fclose(out);
    // Copy bin counts back to host
    int h_binCounts[NUM_BINS];
    cudaMemcpy(h_binCounts, d_binCounts, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_binCounts);
    // Compute Chi-Square statistic
    dfloat expectedCount = static_cast<dfloat>(N) / NUM_BINS;
    *chiSquareResult = 0.0;
    for (int i = 0; i < NUM_BINS; ++i) {
        dfloat diff = h_binCounts[i] - expectedCount;
        *chiSquareResult += (diff * diff) / expectedCount;
    }
}

// Driver function for the Chi-Square test
void xaChiSquareTest(dfloat* array, int N) {
    dfloat chiSquare;
    xaChiSquareLauncher(array, N, &chiSquare);

    // Degrees of freedom for Chi-Square test: number of bins - 1
    const int degreesOfFreedom = NUM_BINS - 1;

    std::cout << "XA Chi-Square Test:\n";
    std::cout << "This test assesses the uniformity of the array values using a Chi-Square goodness-of-fit test.\n";
    std::cout << "Computed XA Chi-Square value: " << chiSquare << "\n";

    // Quality Rating based on Chi-Square statistic
    dfloat threshold = degreesOfFreedom * 10.0; // A rough threshold for quality rating (TW tweaked)
    int quality = 100 - static_cast<int>((chiSquare / threshold) * 100);
    quality = max(0, min(quality, 100)); // Clamp to range [0, 100]

    std::cout << "Computed XA Chi Square Quality Rating: " << quality << "/100\n";
    std::cout << "Test completed.\n";
}
