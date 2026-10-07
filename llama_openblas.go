//go:build openblas

package llama

// Link flags for a libbinding.a built with `make BUILD_TYPE=openblas`, which
// builds llama.cpp's BLAS backend against OpenBLAS.

/*
#cgo LDFLAGS: -lopenblas
*/
import "C"
