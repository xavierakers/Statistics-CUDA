// gptChiSquareTest.cu

#include "gptRandom.h"
#include <cuda.h>
#include <iostream>

// #define NUM_BINS 50 // Number of bins for the Chi-Square test
#define NUM_BINS 50
// Kernel for counting values in each bin
__global__ void gptChiSquareKernel(dfloat* array, int* binCounts, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = min(static_cast<int>(array[idx] * NUM_BINS), NUM_BINS - 1); // Map value to bin
        atomicAdd(&binCounts[bin], 1); // Increment bin count atomically
    }
}

// Launcher function for the Chi-Square test
void gptChiSquareLauncher(dfloat* array, int N, dfloat* chiSquareResult) {
    int* d_binCounts;
    cudaMalloc((void**)&d_binCounts, NUM_BINS * sizeof(int));
    cudaMemset(d_binCounts, 0, NUM_BINS * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // timing for througput model
    cudaEvent_t tic, toc;
    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    cudaEventRecord(tic);
    // Launch kernel to count occurrences in bins
    gptChiSquareKernel<<<blocksPerGrid, threadsPerBlock>>>(array, d_binCounts, N);
    cudaEventRecord(toc);
    cudaDeviceSynchronize();

    float elapsed;
    cudaEventElapsedTime(&elapsed, tic, toc);
    size_t bytes = 2 * N * sizeof(dfloat);
    int flop = 1 * N;
    FILE* out = fopen("gptChiSquareTiming.data", "a");
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
void gptChiSquareTest(dfloat* array, int N) {
    dfloat chiSquare;
    gptChiSquareLauncher(array, N, &chiSquare);

    // Degrees of freedom for Chi-Square test: number of bins - 1
    const int degreesOfFreedom = NUM_BINS - 1;

    std::cout << "Chi-Square Test:\n";
    std::cout << "This test assesses the uniformity of the array values using a Chi-Square goodness-of-fit test.\n";
    std::cout << "Computed Chi-Square value: " << chiSquare << "\n";

    // Quality Rating based on Chi-Square statistic
    dfloat threshold = degreesOfFreedom * 10.0; // A rough threshold for quality rating (TW tweaked)
    int quality = 100 - static_cast<int>((chiSquare / threshold) * 100);
    quality = max(0, min(quality, 100)); // Clamp to range [0, 100]

    std::cout << "Computed Chi Square Quality Rating: " << quality << "/100\n";
    std::cout << "Test completed.\n";
}

// UNIT test

#define TEST_SIZE (1000 * NUM_BINS) // Total number of elements, divisible by NUM_BINS

__global__ void initializeTestArray(dfloat* array, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Distribute values evenly across bins: bin 0 contains values in [0, 0.1), bin 1 in [0.1, 0.2), etc.
        array[idx] = (idx % NUM_BINS) / static_cast<dfloat>(NUM_BINS);
    }
}

void testChiSquarePredictableOutput() {
    dfloat* d_array;
    cudaMalloc((void**)&d_array, TEST_SIZE * sizeof(dfloat));

    // Initialize the array with predictable values
    int threadsPerBlock = 256;
    int blocksPerGrid = (TEST_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    initializeTestArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, TEST_SIZE);
    cudaDeviceSynchronize();

    // Run Chi-Square test
    dfloat chiSquare;
    gptChiSquareLauncher(d_array, TEST_SIZE, &chiSquare);

    // Free the array
    cudaFree(d_array);

    // Expected outcome for a perfect distribution is a Chi-Square value close to 0
    std::cout << "Unit Test: Chi-Square Test with Predictable Output\n";
    std::cout << "Expected Chi-Square value: 0 (perfectly uniform distribution)\n";
    std::cout << "Computed Chi-Square Quality value: " << chiSquare << "\n";

    if (chiSquare < 1.0) {
        std::cout << "Test passed.\n";
    } else {
        std::cout << "Test failed.\n";
    }
}
