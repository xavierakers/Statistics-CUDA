// gptRandomTester.cu

#include "gptRandom.h"
#include <iostream>

void printDivider() {

    printf("\n------------------------------------------\n");
}

void printDeviceArray(int N, dfloat* d_array) {
    dfloat* h_array = (dfloat*)calloc(N, sizeof(dfloat));
    cudaMemcpy(h_array, d_array, N * sizeof(dfloat), cudaMemcpyDefault);
    for (int n = 0; n < N; ++n) {
        printf("%g\n", h_array[n]);
    }
    free(h_array);
}

void gptRandomTester(dfloat* d_array, int N) {
    std::cout << "Running randomness test suite on d_array of size " << N << "...\n";

    // Example test call - each test function prints its own results
    // printDivider();
    // gptMeanTest(d_array, N);

    // printDivider();
    // twMeanTest(d_array, N);

    // printDivider();
    // gptVarianceTest(d_array, N);

    // printDivider();
    // twVarianceTest(d_array, N);

    printDivider();
    gptSerialCorrelationTest(d_array, N);

    printDivider();
    xaSerialCorrelationTest(d_array, N);

    printDivider();
    gptChiSquareTest(d_array, N);
    //  testChiSquarePredictableOutput(); // run unit test

    printDivider();
    xaChiSquareTest(d_array, N);

    printDivider();
    gptKolmogorovSmirnovTest(d_array, N);

    printDivider();
    xaKolmogorovSmirnovTest(d_array, N);

    // printDivider();
    // gptEntropyTest(d_array, N);

    // printDivider();
    // int maxLag = 4;
    // for(int lag=1;lag<=maxLag;++lag){
    //   gptAutocorrelationTest(d_array, N, lag);
    //   //    testAutocorrelationPredictableOutput(N, lag);
    // }
    // // large
    // for(int frac=4;frac<=20;++frac){
    //   gptAutocorrelationTest(d_array, N, N/frac);
    //   //    testAutocorrelationPredictableOutput(N, N/frac);
    // }

    // printDivider();
    // gptSpectralTest(d_array, N);

    // printDivider();
    // gptBirthdaySpacingTest(d_array, N);

    // printDivider();
    // gptLaggedSumTest(d_array, N);

    // printDivider();
    // gptLongestRunOfOnesTest(d_array, N);

    // // Quality metrics are dubious for the following tests

    // printDivider();
    // gptOverlappingTemplateMatchingTest(d_array, N);

    // printDivider();
    // gptRandomExcursionsTest(d_array, N);

    // printDivider();
    // gptMatrixRankTest(d_array, N);

    // printDivider();
    // gptApproximateEntropyTest(d_array, N);

    // printDivider();
    // gptDftTest(d_array, N);

    // SLOW and DUBIOUS
    //  printDivider();
    //  gptMaurerUniversalTest(d_array, N);

    // EXPENSIVE:
    //  printDivider();
    //  gptLinearComplexityTest(d_array, N);

    std::cout << "Randomness test suite completed.\n";
}
