// gpt generated random number tester
#include "gptRandom.h" // Header for test function declarations and macros
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

template <typename curandStateType>
__global__ void generateRandomNumbers(dfloat* array, int N, int seed, dfloat val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        //    curandState state;
        curandStateType state;
        curand_init(seed, idx, 0, &state);
        array[idx] = curand_uniform(&state);
    }
}

template <>
__global__ void generateRandomNumbers<dfloat>(dfloat* array, int N, int seed, dfloat val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        array[idx] = val;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: N must be a positive integer." << std::endl;
        return 1;
    }

    dfloat* d_array;
    cudaMalloc((void**)&d_array, N * sizeof(dfloat));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dfloat val = 1;

    // reference test using a constant array
    printf("Quality: const array  ---------------------------------------------\n");
    generateRandomNumbers<dfloat><<<blocksPerGrid, threadsPerBlock>>>(d_array, N, 1234, val);
    gptRandomTester(d_array, N);

    // // Initialize random numbers & test
    printf("Quality: curandStateMRG32k3a_t ------------------------------------------------\n");
    generateRandomNumbers<curandStateMRG32k3a_t><<<blocksPerGrid, threadsPerBlock>>>(d_array, N, 1234, val);
    gptRandomTester(d_array, N);

    printf("Quality: curandStatePhilox4_32_10_t --------------------------------------------\n");
    generateRandomNumbers<curandStatePhilox4_32_10_t><<<blocksPerGrid, threadsPerBlock>>>(d_array, N, 1234, val);
    gptRandomTester(d_array, N);

    printf("Quality: curandStateXORWOW_t ---------------------------------------------\n");
    generateRandomNumbers<curandStateXORWOW_t><<<blocksPerGrid, threadsPerBlock>>>(d_array, N, 1234, val);
    gptRandomTester(d_array, N);

    printf("Quality: HOST drand48  ---------------------------------------------\n");
    dfloat* h_array = (dfloat*)calloc(N, sizeof(dfloat));
    for (int n = 0; n < N; ++n)
        h_array[n] = drand48();
    cudaMemcpy(d_array, h_array, N * sizeof(dfloat), cudaMemcpyDefault);

    gptRandomTester(d_array, N);

    cudaFree(d_array);
    return 0;
}
