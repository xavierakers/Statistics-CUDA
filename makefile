# Compiler and flags
NVCC := nvcc
NVCC_FLAGS := --use_fast_math -O3 -arch=sm_60 -std=c++20

# Project files
EXECUTABLE := main

SRC_FILES := main.cu\
gptRandomTester.cu\
gptChiSquareTest.cu\
xaChiSquareTest.cu\
gptKolmogorovSmirnovTest.cu\
xaKolmogorovSmirnovTest.cu\
gptSerialCorrelationTest.cu\
xaSerialCorrelationTest.cu\
# gptMeanTest.cu\
gptVarianceTest.cu\
gptChiSquareTest.cu\
gptKolmogorovSmirnovTest.cu\
gptEntropyTest.cu\
gptAutocorrelationTest.cu\
gptSpectralTest.cu\
gptBirthdaySpacingTest.cu\
gptMatrixRankTest.cu\
gptLaggedSumTest.cu\
gptRandomExcursionsTest.cu\
gptLinearComplexityTest.cu\
gptOverlappingTemplateMatchingTest.cu\
gptLongestRunOfOnesTest.cu\
gptMaurerUniversalTest.cu\
gptApproximateEntropyTest.cu\
gptDftTest.cu\
twVarianceTest.cu
# twMeanTest.cu\

OBJ_FILES := $(SRC_FILES:.cu=.o)

# Build the project
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ_FILES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJ_FILES) -lcudart -lcurand -lcublas -lcufft

# Compile each .cu file to .o
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJ_FILES) $(EXECUTABLE)


# Run the executable
run: $(EXECUTABLE)
	./$(EXECUTABLE) 100000000
