// gptKolmogorovSmirnovTest.cu

#include "gptRandom.h"
#include <cuda.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// Reduction kernel to find the maximum deviation
__global__ void gptMaxReductionKernel(dfloat* deviations, int N) {
    extern __shared__ dfloat sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    sharedData[tid] = (idx < N) ? deviations[idx] : 0.0;
    __syncthreads();

    // Perform reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }

    // Write the result for this block to deviations array
    if (tid == 0)
        deviations[blockIdx.x] = sharedData[0];
}

// Kernel to compute the maximum difference (KS statistic)
__global__ void gptKolmogorovSmirnovDifferenceKernel(dfloat* sortedArray, dfloat* maxDiff, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Calculate the empirical CDF for the sorted values
        dfloat empiricalCDF = static_cast<dfloat>(idx + 1) / N;
        dfloat theoreticalCDF = sortedArray[idx];
        dfloat diff = fabs(empiricalCDF - theoreticalCDF);

        if (std::is_same<dfloat, double>::value)
            atomicMax(reinterpret_cast<unsigned long long*>(maxDiff), __double_as_longlong(diff));
        else
            atomicMax(reinterpret_cast<unsigned int*>(maxDiff), __float_as_int(diff));
    }
}

void gptKolmogorovSmirnovLauncher(dfloat* array, int N, dfloat* ksResult) {
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
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaEvent_t tic, toc;
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    cudaEventRecord(tic);
    // Step 4: Launch kernel to calculate the KolmogorovSmirnov statistic
    gptKolmogorovSmirnovDifferenceKernel<<<blocksPerGrid, threadsPerBlock>>>(sortedArray, d_maxDiff, N);
    cudaEventRecord(toc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    float elapsed;
    cudaEventElapsedTime(&elapsed, tic, toc);

    size_t bytes = (N + 1) * sizeof(dfloat);
    int flop = 3 * N; // metric from nvprof N = 1
    FILE* out = fopen("gptKolmogorovSmirnovTiming.data", "a");
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
void gptKolmogorovSmirnovTest(dfloat* array, int N) {
    dfloat ksStatistic;
    gptKolmogorovSmirnovLauncher(array, N, &ksStatistic);

    std::cout << "Kolmogorov-Smirnov (KolmogorovSmirnov) Test:\n";
    std::cout << "This test assesses the uniformity of the array values by comparing the empirical and theoretical CDFs.\n";
    std::cout << "Computed KolmogorovSmirnov Statistic: " << ksStatistic << "\n";

    // Quality Rating based on KolmogorovSmirnov statistic
    int quality = 100 - static_cast<int>(ksStatistic * 200); // Penalize larger deviations
    quality = max(0, min(quality, 100)); // Clamp quality to range [0, 100]

    std::cout << "Computed Kolomgorov-Smirnov Quality Rating: " << quality << "/100\n";
    std::cout << "Test completed.\n";
}
