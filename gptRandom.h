// gptRandom.h

#ifndef GPTRANDOM_H
#define GPTRANDOM_H

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cufft.h>
#include <iostream>

// Define dfloat to allow easy switching between float and double
#define dfloatSize 4

// Type definitions based on dfloat
// Determine precision and set appropriate types and functions
#if dfloatSize == 8
typedef double dfloat;
typedef double cufftReal_t;
typedef cufftDoubleReal cufftRealCufft_t;
typedef cufftDoubleComplex cufftComplex_t;
#define CUFFT_EXEC_FORWARD cufftExecD2Z
#define CUFFT_FORWARD_PLAN CUFFT_D2Z
#define CUFFT_COMPLEX_SIZE sizeof(cufftDoubleComplex)
#define CUFFT_REAL_SIZE sizeof(cufftDoubleReal)
#define CUDA_ABS fabs
#define CUDA_EXP exp
#define CUDA_EXP2 exp2
#else
typedef float dfloat;
typedef float cufftReal_t;
typedef cufftReal cufftRealCufft_t;
typedef cufftComplex cufftComplex_t;
#define CUFFT_EXEC_FORWARD cufftExecR2C
#define CUFFT_FORWARD_PLAN CUFFT_R2C
#define CUFFT_COMPLEX_SIZE sizeof(cufftComplex)
#define CUFFT_REAL_SIZE sizeof(cufftReal)
#define CUDA_ABS fabsf
#define CUDA_EXP expf
#define CUDA_EXP2 exp2f
#endif

// Declaration of the tester function
void gptRandomTester(dfloat* array, int N);

// Declaration of Launchers
void gptMeanLauncher(dfloat* array, int N, dfloat* meanResult);

// Declaration of the statistical tests
// void gptMeanTest(dfloat *array, int N);
// void gptVarianceTest(dfloat *array, int N);
void gptSerialCorrelationTest(dfloat* array, int N);
void xaSerialCorrelationTest(dfloat* array, int N);
void gptChiSquareTest(dfloat* array, int N);
void xaChiSquareTest(dfloat* array, int N);
void gptKolmogorovSmirnovTest(dfloat* array, int N);
void xaKolmogorovSmirnovTest(dfloat* array, int N);
// void gptEntropyTest(dfloat *array, int N);
// void gptAutocorrelationTest(dfloat *array, int N, int lag = 1);
// void gptSpectralTest(dfloat *array, int N);
// void gptBirthdaySpacingTest(dfloat *array, int N);
// void gptMatrixRankTest(dfloat *array, int N);
// void gptLaggedSumTest(dfloat *array, int N);
// void gptRandomExcursionsTest(dfloat *array, int N);
// void gptLinearComplexityTest(dfloat *sequence, int N);
// void gptOverlappingTemplateMatchingTest(dfloat *sequence, int N);
// void gptLongestRunOfOnesTest(dfloat *sequence, int N);
// void gptMaurerUniversalTest(dfloat *sequence, int N);
// void gptApproximateEntropyTest(dfloat *sequence, int N);
// void gptDftTest(dfloat *sequence, int N);

// TW variant(s) of test
// void twVarianceTest(dfloat *x, int N);
// void twMeanTest(dfloat *x, int N);

// Declaration of unit tests
void testGptChiSquarePredictableOutput();
void testAutocorrelationPredictableOutput(int N, int lag);

// Declaration of kernels
__global__ void gptMeanKernel(dfloat* array, dfloat* result, int N);
__global__ void convertToBinary(dfloat* sequence, int* binarySequence, int N);

#define gptCheckCudaError()                                                      \
    {                                                                            \
        cudaError_t err = cudaGetLastError();                                    \
        if (err != cudaSuccess) {                                                \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        }                                                                        \
    }

#endif // GPTRANDOM_H
