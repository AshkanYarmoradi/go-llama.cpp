.PHONY: test clean

INCLUDE_PATH := $(abspath ./)
LIBRARY_PATH := $(abspath ./)

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

#
# Compile flags
#

BUILD_TYPE?=
# keep standard at C11 and C++17
CFLAGS   = -I./llama.cpp -I./llama.cpp/include -I./llama.cpp/ggml/include -I. -O3 -DNDEBUG -std=c11 -fPIC
CXXFLAGS = -I./llama.cpp -I./llama.cpp/include -I./llama.cpp/ggml/include -I. -I./llama.cpp/common -I./common -O3 -DNDEBUG -std=c++17 -fPIC
LDFLAGS  =

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function

# OS specific
# TODO: support Windows
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),OpenBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	# Use all CPU extensions that are available:
	CFLAGS += -march=native -mtune=native
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS += -mcpu=power9
		CXXFLAGS += -mcpu=power9
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif
ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif
ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif
ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, 2, 3
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

# Backends. These pass the GGML_* options llama.cpp reads today
# (llama.cpp/ggml/CMakeLists.txt). The LLAMA_* names used here before are
# either fatal (LLAMA_CUBLAS) or ignored with only a CMake warning, which is
# how the GPU and BLAS build types stopped working unnoticed.
#
# GO_TAGS is the Go build tag whose llama_<tag>.go file holds the backend's
# link flags. libbinding.a is repacked from CMake's objects, so CMake's own
# link dependencies do not come with it; the tag file restates them, after
# llama.go's -lbinding. Env CGO_LDFLAGS cannot do that job: cmd/go puts it
# before the package's flags, where an --as-needed linker drops the libraries.
#
# Switching BUILD_TYPE rebuilds llama.cpp from scratch (see build/.build_type).
#
# The backend options are added with `override`. A plain += is ignored when
# CMAKE_ARGS comes from the make command line (make CMAKE_ARGS=...), and the
# build would come out CPU-only under a GPU BUILD_TYPE and Go tag. metal does
# without it, since llama.cpp turns Metal on by default on Apple anyway.

# OpenBLAS and BLIS need pkg-config as well: unless CMAKE_ARGS sets
# -DBLAS_INCLUDE_DIRS=<dir>, ggml-blas finds the BLAS headers through it
# (llama.cpp/ggml/src/ggml-blas/CMakeLists.txt).
ifeq ($(BUILD_TYPE),openblas)
	override CMAKE_ARGS+=-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
	GO_TAGS=openblas
endif

ifeq ($(BUILD_TYPE),blis)
	override CMAKE_ARGS+=-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME
	GO_TAGS=blis
endif

ifeq ($(BUILD_TYPE),cublas)
	# NCCL only speeds up multi-GPU splits. Left to auto-detection, it makes
	# -lnccl a link requirement on some hosts and not on others.
	# Building where there is no GPU (CI, docker build)? Add
	# -DCMAKE_CUDA_ARCHITECTURES=<arch> to CMAKE_ARGS, because GGML_NATIVE
	# otherwise asks nvcc to compile for the GPU it finds.
	override CMAKE_ARGS+=-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF
	GO_TAGS=cublas
endif

ifeq ($(BUILD_TYPE),hipblas)
	ROCM_HOME ?= /opt/rocm
	GPU_TARGETS ?= gfx900;gfx90a;gfx1030;gfx1031;gfx1100
	# ROCm's clang compiles only the HIP sources, as llama.cpp's docs/build.md
	# sets it up. C and C++ stay on the default toolchain, so ggml's OpenMP is
	# the libgomp that llama.go's -fopenmp links. Needs ROCm 6.1 or newer.
	override CMAKE_ENV+=HIPCXX="$(ROCM_HOME)/llvm/bin/clang" HIP_PATH="$(ROCM_HOME)" ROCM_PATH="$(ROCM_HOME)"
	# CMake takes a ;-separated list. A comma-separated one still works.
	comma := ,
	override CMAKE_ARGS+=-DGGML_HIP=ON -DGPU_TARGETS="$(subst $(comma),;,$(GPU_TARGETS))"
	GO_TAGS=hipblas
endif

ifeq ($(BUILD_TYPE),vulkan)
	# Needs the Vulkan headers and loader, glslc and SPIRV-Headers
	# (Debian/Ubuntu: libvulkan-dev glslc spirv-headers).
	override CMAKE_ARGS+=-DGGML_VULKAN=ON
	GO_TAGS=vulkan
endif

# Any other BUILD_TYPE, llama.cpp's own backend names (cuda, hip) included,
# would set no options and build CPU-only, so it stops here instead. `make
# clean` is let through, since a stale BUILD_TYPE may still be exported.
ifneq ($(MAKECMDGOALS),clean)
ifeq ($(BUILD_TYPE),clblas)
$(error BUILD_TYPE=clblas was removed: llama.cpp no longer has a CLBlast backend. Use BUILD_TYPE=vulkan for a vendor-neutral GPU build, or cublas/hipblas)
endif
ifneq ($(filter-out openblas blis cublas hipblas vulkan metal,$(BUILD_TYPE)),)
$(error Unknown BUILD_TYPE=$(BUILD_TYPE). Use cublas, hipblas, vulkan, metal, openblas or blis, or leave it empty for a CPU build)
endif
endif

ifeq ($(BUILD_TYPE),metal)
	EXTRA_LIBS=
	CGO_LDFLAGS+="-framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
	CMAKE_ARGS+=-DGGML_METAL=ON
	EXTRA_TARGETS+=llama.cpp/ggml-metal.o
endif

# TODO: support Windows
ifeq ($(GPU_TESTS),true)
	TEST_LABEL=gpu
else
	TEST_LABEL=!gpu
endif

# Parallel jobs for the llama.cpp build. A CUDA build needs a few GB of RAM
# per nvcc job, so lower it with JOBS=n on a small machine.
ifndef JOBS
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
endif

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I CGO_LDFLAGS:  $(CGO_LDFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I BUILD_TYPE:  $(BUILD_TYPE))
$(info I CMAKE_ARGS:  $(CMAKE_ARGS))
$(info I GO_TAGS:  $(GO_TAGS))
$(info I JOBS:  $(JOBS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

# Use this if you want to set the default behavior

# build/ holds one backend's CMake cache and objects. This stamp records the
# BUILD_TYPE that produced them, and a different one wipes build/ so nothing
# is reused from a cache configured for another backend. The stamp is only
# rewritten when the value changes, so an unchanged BUILD_TYPE rebuilds
# nothing. Changing CMAKE_ARGS alone is not tracked: run `make clean` first.
build/.build_type: FORCE
	@mkdir -p build
	@if ! echo '$(BUILD_TYPE)' | cmp -s - $@ 2>/dev/null; then \
		if [ -e build/build_complete ]; then \
			echo "BUILD_TYPE is now '$(BUILD_TYPE)': rebuilding llama.cpp from scratch"; \
		fi; \
		rm -rf build && mkdir -p build && echo '$(BUILD_TYPE)' > $@; \
	fi

.PHONY: FORCE
FORCE:

# Build llama.cpp via cmake. Only the libraries the binding links are built:
# the tools and the unified `llama` app (both on by default) add minutes of
# compiling, download the server's web UI, and leave main()-bearing objects
# for the sweep below to pack into libbinding.a.
build/build_complete: build/.build_type
	mkdir -p build
	cd build && CC="$(CC)" CXX="$(CXX)" $(CMAKE_ENV) cmake ../llama.cpp $(CMAKE_ARGS) \
		-DLLAMA_BUILD_COMMON=ON \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_TOOLS=OFF \
		-DLLAMA_BUILD_APP=OFF \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_OPENMP=ON
	cd build && VERBOSE=1 cmake --build . --config Release --parallel $(JOBS)
	touch build/build_complete

binding.o: build/build_complete
	$(CXX) $(CXXFLAGS) -I./llama.cpp -I./llama.cpp/include -I./llama.cpp/common binding.cpp -o binding.o -c $(LDFLAGS)

# Find all object files from cmake build and combine into static library
# We use a counter to uniquely name files to avoid conflicts when files have the same basename
# Two kinds of object are not part of llama.cpp and are skipped: CMake's
# compiler-identification leftovers (CMakeFiles/<version>/CompilerId*; the
# nvcc run keeps its objects, one with a main()), and the vulkan-shaders-gen
# host tool, which the Vulkan build compiles to generate its shaders.
# The old archive is removed first, because `ar rcs` would append to it.
libbinding.a: binding.o build/build_complete
	@echo "Creating static library from cmake build objects..."
	@rm -rf obj_temp && mkdir -p obj_temp
	@counter=0; \
	for f in $$(find build -name "*.o" -type f ! -path "*/CompilerId*" ! -path "*/vulkan-shaders-gen-prefix/*" | sort); do \
		counter=$$((counter + 1)); \
		cp "$$f" "obj_temp/$${counter}_$$(basename $$f)"; \
	done
	@cp binding.o obj_temp/
	@rm -f libbinding.a
	ar rcs libbinding.a obj_temp/*.o
	@rm -rf obj_temp

clean:
	rm -rf *.o
	rm -rf *.a
	rm -rf build
	rm -rf prepare
	rm -rf obj_temp
	rm -rf llama.cpp/*.o

ggllm-test-model.bin:
	wget -q https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q2_K.gguf -O ggllm-test-model.bin

# --tags compiles the backend's llama_<tag>.go, which carries its link flags.
test: ggllm-test-model.bin libbinding.a
	C_INCLUDE_PATH=${INCLUDE_PATH} CGO_LDFLAGS=${CGO_LDFLAGS} LIBRARY_PATH=${LIBRARY_PATH} TEST_MODEL=ggllm-test-model.bin go run github.com/onsi/ginkgo/v2/ginkgo $(if $(GO_TAGS),--tags=$(GO_TAGS)) --label-filter="$(TEST_LABEL)" --flake-attempts 5 -v -r ./...