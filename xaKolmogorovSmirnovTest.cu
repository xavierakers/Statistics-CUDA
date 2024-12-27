// gptKolmogorovSmirnovTest.cu

#include "gptRandom.h"
#include <cuda.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define p_mask 0xffffffff

__forceinline__ __device__ dfloat warpMax(const dfloat v) {
    // 5 instructions to compute the maximum in a warp
    int t = threadIdx.x; // thread index within the warp
    dfloat maxVal = v;

    // Compare the value with other threads in the warp using shuffle operations
    maxVal = max(maxVal, __shfl_sync(p_mask, maxVal, t ^ 1)); // Compare with thread 1
    maxVal = max(maxVal, __shfl_sync(p_mask, maxVal, t ^ 2)); // Compare with thread 2
    maxVal = max(maxVal, __shfl_sync(p_mask, maxVal, t ^ 4)); // Compare with thread 4
    maxVal = max(maxVal, __shfl_sync(p_mask, maxVal, t ^ 8)); // Compare with thread 8
    maxVal = max(maxVal, __shfl_sync(p_mask, maxVal, t ^ 16)); // Compare with thread 16

    return maxVal;
}

// Kernel to compute the maximum difference (KS statistic)
template <int p_Nloads, int p_Nwarp>
__global__ void xaKolmogorovSmirnovDifferenceKernel(dfloat* sortedArray, dfloat* maxDiff, int N) {

    __shared__ dfloat s_maxDiff[p_Nwarp];

    int t = threadIdx.x; // thread within warp
    int w = threadIdx.y; // warp within block
    int b = blockIdx.x; // block within grid

    dfloat r_maxDiff = 0.0f;
#pragma unroll
    for (int r = 0; r < p_Nloads; r++) {
        const int m = t + 32 * (r + w * p_Nloads + b * p_Nloads * p_Nwarp);
        dfloat theoreticalCDF = (m < N) ? sortedArray[m] : 0.f; // theoreticalCDF
        dfloat empiricalCDF = static_cast<dfloat>(m + 1) / N;
        dfloat diff = fabs(empiricalCDF - theoreticalCDF);
        r_maxDiff = (m < N && r_maxDiff < diff) ? diff : r_maxDiff;
    }

    // now each thread in warp has their maxdiff
    // warp level reduction
    r_maxDiff = warpMax(r_maxDiff);

    if (t == 0) {
        s_maxDiff[w] = r_maxDiff;
    }
    __syncthreads();

    if (w == 0) {
        r_maxDiff = warpMax(s_maxDiff[t]);
        if (t == 0) {
            if (std::is_same<dfloat, double>::value)
                atomicMax(reinterpret_cast<unsigned long long*>(maxDiff), __double_as_longlong(r_maxDiff));
            else
                atomicMax(reinterpret_cast<unsigned int*>(maxDiff), __float_as_int(r_maxDiff));
        }
    }
}

void xaKolmogorovSmirnovLauncher(dfloat* array, int N, dfloat* ksResult) {
    // Step 1: Create a copy of the original array for sorting
    dfloat* sortedArray;
    cudaMalloc((void**)&sortedArray, N * sizeof(dfloat));
    cudaMemcpy(sortedArray, array, N * sizeof(dfloat), cudaMemcpyDeviceToDevice);

    // Step 2: Sort the copy on the device
    thrust::device_ptr<dfloat> d_array_ptr(sortedArray);
    thrust::sort(d_array_ptr, d_array_ptr + N);

    // Step 3: Allocate and initialize maxDiff
    dfloat initialDiff = 0.0;
    dfloat* d_maxDiff;
    cudaMalloc((void**)&d_maxDiff, sizeof(dfloat));
    cudaMemcpy(d_maxDiff, &initialDiff, sizeof(dfloat), cudaMemcpyHostToDevice);

    // timing for throughput model
    cudaEvent_t tic, toc;
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    cudaEventRecord(tic);
    // Step 4: Launch kernel to calculate the KolmogorovSmirnov statistic
    const int p_Nloads = 10; // the count of numbers processed by each thread
    const int p_Nwarp = 32;
    dim3 threadsPerBlock(32, p_Nwarp);
    dim3 blocksPerGrid((N + p_Nloads * 32 * p_Nwarp - 1) / (p_Nloads * 32 * p_Nwarp));
    xaKolmogorovSmirnovDifferenceKernel<p_Nloads, p_Nwarp><<<blocksPerGrid, threadsPerBlock>>>(sortedArray, d_maxDiff, N);
    cudaEventRecord(toc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    float elapsed;
    cudaEventElapsedTime(&elapsed, tic, toc);

    size_t bytes = (N + 1) * sizeof(dfloat);
    int flop = 2 * threadsPerBlock.x * threadsPerBlock.y * blocksPerGrid.x; // metric from nvprof N = 1
    FILE* out = fopen("xaKolmogorovSmirnovTiming.data", "a");
    fprintf(out, "%d, %7.5e, %7.5e, %7.5e, %%%% N, time (s), memory throughput (bytes/s), flop throughput (GFLOPS/s)\n",
        N, elapsed, (bytes / 1.e6) / elapsed, (flop / 1.e6) / elapsed);
    fclose(out);
    // Step 5: Retrieve maxDiff as a double
    dfloat maxDiff;
    cudaMemcpy(&maxDiff, d_maxDiff, sizeof(dfloat), cudaMemcpyDeviceToHost);
    *ksResult = static_cast<double>(maxDiff); // Convert to double if necessary

    // Free temporary memory
    cudaFree(d_maxDiff);
    cudaFree(sortedArray);
}

// Driver function for the KolmogorovSmirnov test
void xaKolmogorovSmirnovTest(dfloat* array, int N) {
    dfloat ksStatistic;
    xaKolmogorovSmirnovLauncher(array, N, &ksStatistic);

    std::cout << "XA Kolmogorov-Smirnov (KolmogorovSmirnov) Test:\n";
    std::cout << "This test assesses the uniformity of the array values by comparing the empirical and theoretical CDFs.\n";
    std::cout << "Computed KolmogorovSmirnov Statistic: " << ksStatistic << "\n";

    // Quality Rating based on KolmogorovSmirnov statistic
    int quality = 100 - static_cast<int>(ksStatistic * 200); // Penalize larger deviations
    quality = max(0, min(quality, 100)); // Clamp quality to range [0, 100]

    std::cout << "Computed Kolomgorov-Smirnov Quality Rating: " << quality << "/100\n";
    std::cout << "Test completed.\n";
}
