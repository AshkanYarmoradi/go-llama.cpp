//go:build vulkan

package llama

// Link flags for a libbinding.a built with `make BUILD_TYPE=vulkan`: the
// Vulkan loader, which ggml-vulkan links.

/*
#cgo LDFLAGS: -lvulkan
*/
import "C"
