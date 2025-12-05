<div align="center">

<img width="300" src="./images/logo.png">

# 🦙 go-llama.cpp

### *Blazing Fast LLM Inference in Go*

[![Go Reference](https://pkg.go.dev/badge/github.com/AshkanYarmoradi/go-llama.cpp.svg)](https://pkg.go.dev/github.com/AshkanYarmoradi/go-llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go Report Card](https://goreportcard.com/badge/github.com/AshkanYarmoradi/go-llama.cpp)](https://goreportcard.com/report/github.com/AshkanYarmoradi/go-llama.cpp)

**High-performance [llama.cpp](https://github.com/ggerganov/llama.cpp) bindings for Go — run LLMs locally with the power of C++ and the simplicity of Go.**

[Getting Started](#-quick-start) •
[Features](#-features) •
[API Reference](#-api-reference) •
[GPU Acceleration](#-acceleration) •
[Examples](#-examples)

</div>

---

## 🌟 About This Fork

> **Note**: The original `go-skynet/go-llama.cpp` repository was unmaintained for over a year. As the llama.cpp ecosystem evolved rapidly with new features, samplers, and breaking API changes, the Go bindings fell behind.
>
> **I decided to fork and actively maintain this project** to ensure the Go community has access to the latest llama.cpp capabilities. This fork is fully updated to support the modern llama.cpp API including the new sampler chain architecture and GGUF format.

### What's New in This Fork (December 2025)

- ✅ **Updated to latest llama.cpp** — Full compatibility with modern GGUF models
- ✅ **New Sampler Chain API** — Modern sampling architecture with composable samplers
- ✅ **XTC Sampler** — Cross-Token Coherence for improved generation quality
- ✅ **DRY Sampler** — "Don't Repeat Yourself" penalty to reduce repetition
- ✅ **TopNSigma Sampler** — Statistical sampling for better token selection
- ✅ **Model Info API** — Query model metadata (vocab size, layers, parameters, etc.)
- ✅ **Chat Templates** — Native support for model chat templates
- ✅ **Fixed Build System** — Proper static linking with all CPU optimizations

---

## 🚀 Features

```
┌─────────────────────────────────────────────────────────────────┐
│  🎯 Performance First                                           │
│  ────────────────────                                           │
│  • Zero-copy data passing to C++                                │
│  • Minimal CGO overhead                                         │
│  • Native CPU optimizations (AVX, AVX2, AVX-512)               │
├─────────────────────────────────────────────────────────────────┤
│  🔧 Flexible Sampling                                           │
│  ────────────────────                                           │
│  • Temperature, Top-K, Top-P, Min-P                            │
│  • Repetition & Presence Penalties                              │
│  • XTC, DRY, TopNSigma (NEW!)                                  │
│  • Mirostat v1 & v2                                            │
├─────────────────────────────────────────────────────────────────┤
│  ⚡ GPU Acceleration                                            │
│  ────────────────────                                           │
│  • NVIDIA CUDA / cuBLAS                                        │
│  • AMD ROCm / HIPBlas                                          │
│  • Apple Metal (M1/M2/M3)                                      │
│  • OpenCL / CLBlast                                            │
├─────────────────────────────────────────────────────────────────┤
│  📦 Model Support                                               │
│  ────────────────────                                           │
│  • All GGUF quantization formats                               │
│  • LLaMA, Mistral, Qwen, Phi, and 100+ architectures          │
│  • LoRA adapter loading                                        │
│  • Embeddings generation                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Requirements

- Go 1.20+
- C/C++ compiler (GCC, Clang, or MSVC)
- CMake 3.14+
- (Optional) CUDA Toolkit for NVIDIA GPU support
- (Optional) ROCm for AMD GPU support

---

## 🎯 Quick Start

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/AshkanYarmoradi/go-llama.cpp
cd go-llama.cpp

# Build the bindings
make libbinding.a

# Run the example
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/path/to/model.gguf" -t 8
```

### Basic Usage

```go
package main

import (
    "fmt"
    llama "github.com/AshkanYarmoradi/go-llama.cpp"
)

func main() {
    // Load model
    model, err := llama.New("model.gguf",
        llama.SetContext(2048),
        llama.SetGPULayers(35),
    )
    if err != nil {
        panic(err)
    }
    defer model.Free()

    // Generate text
    response, err := model.Predict("Explain quantum computing in simple terms:",
        llama.SetTemperature(0.7),
        llama.SetTopP(0.9),
        llama.SetTokens(256),
    )
    if err != nil {
        panic(err)
    }
    
    fmt.Println(response)
}
```

---

## 🎛️ API Reference

### New Sampler Options

```go
// XTC (Cross-Token Coherence) - Improves coherence between tokens
llama.SetXTC(probability, threshold float64)

// DRY (Don't Repeat Yourself) - Reduces repetitive patterns  
llama.SetDRY(multiplier, base float64, allowedLength, penaltyLastN int)

// TopNSigma - Statistical sampling based on standard deviations
llama.SetTopNSigma(n float64)
```

### Model Information API

```go
// Get comprehensive model metadata
info := model.GetModelInfo()
fmt.Printf("Model: %s\n", info.Description)
fmt.Printf("Vocabulary: %d tokens\n", info.NVocab)
fmt.Printf("Context Length: %d\n", info.NCtxTrain)
fmt.Printf("Embedding Dim: %d\n", info.NEmbd)
fmt.Printf("Layers: %d\n", info.NLayer)
fmt.Printf("Parameters: %d\n", info.NParams)
fmt.Printf("Size: %d bytes\n", info.Size)

// Get chat template for chat models
template := model.GetChatTemplate()
```

### All Sampling Options

| Option | Description | Default |
|--------|-------------|---------|
| `SetTemperature(t)` | Randomness (0.0 = deterministic) | 0.8 |
| `SetTopK(k)` | Limit to top K tokens | 40 |
| `SetTopP(p)` | Nucleus sampling threshold | 0.9 |
| `SetMinP(p)` | Minimum probability threshold | 0.05 |
| `SetRepeatPenalty(p)` | Penalize repeated tokens | 1.1 |
| `SetPresencePenalty(p)` | Penalize token presence | 0.0 |
| `SetFrequencyPenalty(p)` | Penalize token frequency | 0.0 |
| `SetXTC(prob, thresh)` | Cross-token coherence | disabled |
| `SetDRY(...)` | Don't Repeat Yourself | disabled |
| `SetTopNSigma(n)` | Statistical sigma sampling | disabled |
| `SetMirostat(mode)` | Mirostat sampling (1 or 2) | disabled |
| `SetMirostatTAU(tau)` | Mirostat target entropy | 5.0 |
| `SetMirostatETA(eta)` | Mirostat learning rate | 0.1 |

---

## ⚠️ Important Notes

### GGUF Format Only

This library works **exclusively** with the modern `gguf` file format. The legacy `ggml` format is no longer supported.

> Need `ggml` support? Use the legacy tag: [`pre-gguf`](https://github.com/AshkanYarmoradi/go-llama.cpp/releases/tag/pre-gguf)

### Converting Models

```bash
# Convert HuggingFace models to GGUF
python llama.cpp/convert_hf_to_gguf.py /path/to/model --outfile model.gguf

# Quantize for smaller size
./llama.cpp/build/bin/llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

---

## ⚡ Acceleration

### CPU (Default)

The default build uses optimized CPU code with automatic SIMD detection.

```bash
make libbinding.a
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "model.gguf" -t 8
```

### OpenBLAS

```bash
BUILD_TYPE=openblas make libbinding.a
CGO_LDFLAGS="-lopenblas" LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run -tags openblas ./examples -m "model.gguf" -t 8
```

### 🟢 NVIDIA CUDA

```bash
BUILD_TYPE=cublas make libbinding.a
CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64/" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "model.gguf" -ngl 35
```

### 🔴 AMD ROCm

```bash
BUILD_TYPE=hipblas make libbinding.a
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ \
  CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -unwindlib=libgcc -lrocblas -lhipblas" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "model.gguf" -ngl 64
```

### 🔵 Intel OpenCL

```bash
BUILD_TYPE=clblas CLBLAS_DIR=/path/to/clblast make libbinding.a
CGO_LDFLAGS="-lOpenCL -lclblast -L/usr/local/lib64/" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "model.gguf"
```

### 🍎 Apple Metal (M1/M2/M3)

```bash
BUILD_TYPE=metal make libbinding.a
CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build ./examples/main.go
cp build/bin/ggml-metal.metal .
./main -m "model.gguf" -ngl 1
```

---

## 📖 Examples

### Streaming Output

```go
model.SetTokenCallback(func(token string) bool {
    fmt.Print(token)
    return true // continue generation
})

model.Predict("Write a story about a robot:",
    llama.SetTokens(500),
    llama.SetTemperature(0.8),
)
```

### Embeddings

```go
model, _ := llama.New("model.gguf", llama.EnableEmbeddings())

embeddings, _ := model.Embeddings("The quick brown fox")
fmt.Printf("Vector dimension: %d\n", len(embeddings))
```

### With LoRA Adapter

```go
model, _ := llama.New("base-model.gguf",
    llama.SetLoraAdapter("adapter.bin"),
    llama.SetLoraBase("base-model.gguf"),
)
```

---

## 🤝 Contributing

Contributions are welcome! This fork is actively maintained and I'm happy to review PRs for:

- Bug fixes
- Performance improvements  
- New llama.cpp feature bindings
- Documentation improvements
- Test coverage

---

## 📚 Resources

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — The C++ inference engine
- **[GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** — Model file format specification
- **[Hugging Face GGUF Models](https://huggingface.co/models?library=gguf)** — Pre-quantized models

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🦙 by the Go + LLM community**

*If you find this useful, consider giving it a ⭐*

</div>
