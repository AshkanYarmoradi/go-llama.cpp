//go:build blis

package llama

// Link flags for a libbinding.a built with `make BUILD_TYPE=blis`, which
// builds llama.cpp's BLAS backend against BLIS (GGML_BLAS_VENDOR=FLAME).

/*
#cgo LDFLAGS: -lblis
*/
import "C"
