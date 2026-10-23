CXX ?= c++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
SM ?= 80

.PHONY: cpu cuda memcheck
cpu: test_codec_utils
	./test_codec_utils

test_codec_utils: test_codec_utils.cpp ../csrc/CodecUtils.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror $< -o $@

test_cuda: test_cuda.cu ../csrc/L_API.cu $(wildcard ../csrc/*.cuh ../csrc/*.h) ../build/L_API.cuh
	$(NVCC) -std=c++17 -O2 -lineinfo -gencode arch=compute_$(SM),code=sm_$(SM) test_cuda.cu ../csrc/L_API.cu -o $@

cuda: test_cuda
	./test_cuda

memcheck: test_cuda
	$(CUDA_PATH)/bin/compute-sanitizer --tool memcheck --error-exitcode 1 ./test_cuda
