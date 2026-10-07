//go:build cublas

package llama

// Link flags for a libbinding.a built with `make BUILD_TYPE=cublas`.
//
// libbinding.a is repacked from CMake's objects, so ggml-cuda's link
// dependencies are restated here, where they land after llama.go's -lbinding:
//   - cudart and cublas are ggml-cuda's own link line; cublasLt is cublas's.
//   - cuda is the driver API (cuDeviceGet, cuMemCreate, ...), which ggml-cuda
//     calls unless llama.cpp is built with GGML_CUDA_NO_VMM=ON. lib64/stubs
//     lets the link succeed where no driver is installed, as in CI; the real
//     libcuda.so.1 is loaded at run time.
//
// For a toolkit outside /usr/local/cuda, add its -L path through CGO_LDFLAGS.
// Unlike a -l, a -L applies wherever it sits on the link line.

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcublas -lcublasLt -lcudart -lcuda
*/
import "C"
