//go:build hipblas

package llama

// Link flags for a libbinding.a built with `make BUILD_TYPE=hipblas`: the
// hipBLAS, rocBLAS and HIP runtime libraries that ggml-hip links, restated
// here because libbinding.a does not carry CMake's link dependencies.
//
// For ROCm outside /opt/rocm, add its -L path through CGO_LDFLAGS.

/*
#cgo LDFLAGS: -L/opt/rocm/lib -lhipblas -lrocblas -lamdhip64
*/
import "C"
