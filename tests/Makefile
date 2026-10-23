CXX ?= c++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
SM ?= 80

.PHONY: cpu cuda memcheck sanitize thread-sanitize benchmark-host
cpu: test_codec_utils test_host_packer
	./test_codec_utils
	./test_host_packer

test_codec_utils: test_codec_utils.cpp ../csrc/CodecUtils.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror $< -o $@

test_cuda: test_cuda.cu ../csrc/L_API.cu $(wildcard ../csrc/*.cuh ../csrc/*.h) ../build/L_API.cuh
	$(NVCC) -std=c++17 -O2 -Xcompiler=-pthread -lineinfo -gencode arch=compute_$(SM),code=sm_$(SM) test_cuda.cu ../csrc/L_API.cu -o $@

cuda: test_cuda
	./test_cuda

memcheck: test_cuda
	$(CUDA_PATH)/bin/compute-sanitizer --tool memcheck --error-exitcode 1 ./test_cuda

HOST_PACKER_HEADERS = ../csrc/HostPacker.h ../csrc/CodecUtils.h reference_packer.h

test_host_packer: test_host_packer.cpp $(HOST_PACKER_HEADERS)
	$(CXX) -std=c++17 -O3 -pthread -Wall -Wextra -Werror $< -o $@

sanitize:
	$(CXX) -std=c++17 -O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer test_host_packer.cpp -o test_host_packer_sanitize
	./test_host_packer_sanitize

../tools/bench_host_packer: ../tools/bench_host_packer.cpp $(HOST_PACKER_HEADERS)
	$(CXX) -std=c++17 -O3 -pthread -Wall -Wextra -Werror $< -o $@

benchmark-host: ../tools/bench_host_packer
	../tools/bench_host_packer verify
	../tools/bench_host_packer legacy
	../tools/bench_host_packer serial
	../tools/bench_host_packer parallel

thread-sanitize:
	$(CXX) -std=c++17 -O1 -g -pthread -fsanitize=thread test_host_packer.cpp -o test_host_packer_tsan
	./test_host_packer_tsan
