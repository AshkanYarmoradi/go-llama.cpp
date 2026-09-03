<div align="center">

<img width="600" src="./images/logo.png">

# 🦙 go-llama.cpp

### *Go bindings for llama.cpp — actively maintained*

[![Go Reference](https://pkg.go.dev/badge/github.com/AshkanYarmoradi/go-llama.cpp.svg)](https://pkg.go.dev/github.com/AshkanYarmoradi/go-llama.cpp)
[![CI](https://github.com/AshkanYarmoradi/go-llama.cpp/actions/workflows/test.yaml/badge.svg)](https://github.com/AshkanYarmoradi/go-llama.cpp/actions/workflows/test.yaml)
[![Go Report Card](https://goreportcard.com/badge/github.com/AshkanYarmoradi/go-llama.cpp)](https://goreportcard.com/report/github.com/AshkanYarmoradi/go-llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Run GGUF models locally from Go, on top of [llama.cpp](https://github.com/ggerganov/llama.cpp).**

[Quick start](#quick-start) •
[Generating text](#generating-text) •
[Driving inference yourself](#driving-inference-yourself) •
[GPU backends](#gpu-backends) •
[API reference](#api-reference)

</div>

---

## About this fork

`go-skynet/go-llama.cpp` went unmaintained for over a year while llama.cpp
changed underneath it — a new sampler architecture, the GGUF format, and
several rounds of breaking API changes. This fork tracks upstream: a Dependabot
job bumps the llama.cpp submodule daily, and the binding is updated to match
when the engine's API moves.

It targets the modern llama.cpp C API. Functions upstream marks deprecated are
deliberately not exposed — the binding wraps the replacement instead.

---

## Requirements

| | |
|---|---|
| **Go** | 1.25 or newer (the `go` directive in `go.mod`; older toolchains will fetch it) |
| **C++ compiler** | GCC, Clang, or Apple Clang, with C++17 support |
| **CMake** | 3.14+ (builds the vendored llama.cpp) |
| **Optional** | CUDA Toolkit, ROCm, or Xcode for GPU backends |

---

## Quick start

This is **not a `go get`-only package.** It links against a static library
built from the vendored llama.cpp submodule, so you clone and build first:

```bash
git clone --recurse-submodules https://github.com/AshkanYarmoradi/go-llama.cpp
cd go-llama.cpp
make libbinding.a
```

Then build or run with the library on the search path:

```bash
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m /path/to/model.gguf -t 8
```

To use it from your own module, add a `replace` directive pointing at your
checkout and set the same two environment variables when you build.

---

## Generating text

```go
package main

import (
    "fmt"

    llama "github.com/AshkanYarmoradi/go-llama.cpp"
)

func main() {
    model, err := llama.New("model.gguf",
        llama.SetContext(4096),
        llama.SetGPULayers(35),
    )
    if err != nil {
        panic(err)
    }
    defer model.Free()

    out, err := model.Predict("Explain quantum computing in simple terms:",
        llama.SetTemperature(0.7),
        llama.SetTopP(0.9),
        llama.SetTokens(256),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(out)
}
```

### Streaming

Return `false` from the callback to stop generation early.

```go
model.SetTokenCallback(func(token string) bool {
    fmt.Print(token)
    return true
})
model.Predict("Write a story about a robot:", llama.SetTokens(500))
```

### Chat models

`ApplyChatTemplate` renders a conversation using the template stored in the
model's own GGUF metadata, so you do not hardcode any one model's prompt
format. Pass a name (from `BuiltinChatTemplates()`) to override it.

```go
prompt, err := model.ApplyChatTemplate("", []llama.ChatMessage{
    {Role: "system", Content: "You are terse."},
    {Role: "user", Content: "How much is 2+2?"},
}, true) // true: end with the assistant turn opener

out, _ := model.Predict(prompt)
```

llama.cpp recognises a fixed set of templates rather than running a full Jinja
engine. A model whose template it cannot place returns `ErrNoChatTemplate` —
format the prompt yourself in that case.

### Embeddings

```go
model, _ := llama.New("model.gguf", llama.EnableEmbeddings)
vec, _ := model.Embeddings("The quick brown fox")
```

> A context configured for embeddings does not produce usable logits, so
> `Predict` on it returns garbage. Load the model twice if you need both, or
> flip the context with `SetEmbeddings`.

---

## Driving inference yourself

For anything `Predict` cannot express — several conversations sharing one
context, speculative decoding, custom stopping rules — build a batch, decode
it, and sample.

```go
tokens := model.Tokenize("The capital of France is", true, false)

batch := llama.NewBatch(len(tokens), 1)
defer batch.Free()
for i, tok := range tokens {
    batch.Add(tok, int32(i), []int32{0}, i == len(tokens)-1)
}
if model.Decode(batch) != 0 {
    panic("decode failed")
}

chain := llama.NewSamplerChain()
defer chain.Free()
chain.Add(llama.SamplerTopK(40))
chain.Add(llama.SamplerTopP(0.95, 1))
chain.Add(model.SamplerPenalties(64, 1.1, 0, 0))
chain.Add(llama.SamplerTemp(0.8))
chain.Add(llama.SamplerDist(1234))

next := chain.Sample(model, -1)
chain.Accept(next)
fmt.Print(model.TokenToPiece(next, false))
```

A chain **owns** every stage added to it — `chain.Free()` frees them all, and a
stage must not be freed separately after `Add`.

> **Stop on `model.IsEOG(tok)`, not on `tok == EOS`.** Many models define
> several end-of-turn tokens; `IsEOG` asks the vocabulary about all of them.

### Sequences and the KV cache

A context holds several independent conversations, each identified by a
sequence id in `[0, ContextParams().NSeqMax)`.

```go
model.MemorySeqCopy(0, 1, 0, -1)  // fork sequence 0 into 1, sharing the prefix
model.MemorySeqRemove(1, -1, -1)  // drop sequence 1 entirely
```

Context shifting — making room by dropping the oldest tokens — is an evict
followed by a slide:

```go
if model.MemoryCanShift() {
    model.MemorySeqRemove(0, 0, nDiscard)
    model.MemorySeqAdd(0, nDiscard, -1, -nDiscard)
}
```

### Checkpointing one conversation

Whole-context state grows with the entire KV cache. Per-sequence state captures
a single slot, and restores into any sequence id:

```go
data, _ := model.SequenceStateData(0)
model.SetSequenceStateData(data, 1)

// Or to disk, with the token list alongside it.
model.SaveSequenceFile("slot0.bin", 0, tokens)
tokens, _ = model.LoadSequenceFile("slot0.bin", 0)
```

---

## Concurrency

**A `*LLama` is not safe for concurrent use.** Each context has one KV cache and
one output buffer, so concurrent `Predict` or `Decode` calls on the same model
corrupt each other. Serialize with a mutex, or give each goroutine its own
context.

Distinct sequence ids isolate *conversations*, not *callers* — they still need
serializing.

---

## GPU backends

Pick a `BUILD_TYPE`, rebuild `libbinding.a`, and pass the matching linker
flags. Use `llama.SupportsGPUOffload()` at runtime to check what the library
was actually compiled with.

<details>
<summary><b>CPU (default)</b></summary>

```bash
make libbinding.a
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m model.gguf -t 8
```
</details>

<details>
<summary><b>NVIDIA CUDA</b></summary>

```bash
BUILD_TYPE=cublas make libbinding.a
CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64/" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m model.gguf -ngl 35
```
</details>

<details>
<summary><b>AMD ROCm</b></summary>

```bash
BUILD_TYPE=hipblas make libbinding.a
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ \
  CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -unwindlib=libgcc -lrocblas -lhipblas" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m model.gguf -ngl 64
```
</details>

<details>
<summary><b>Apple Metal (M1/M2/M3)</b></summary>

```bash
BUILD_TYPE=metal make libbinding.a
CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build ./examples/main.go
cp build/bin/ggml-metal.metal .
./main -m model.gguf -ngl 1
```
</details>

<details>
<summary><b>OpenBLAS</b></summary>

```bash
BUILD_TYPE=openblas make libbinding.a
CGO_LDFLAGS="-lopenblas" LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD \
  go run -tags openblas ./examples -m model.gguf -t 8
```
</details>

<details>
<summary><b>Intel OpenCL / CLBlast</b></summary>

```bash
BUILD_TYPE=clblas CLBLAS_DIR=/path/to/clblast make libbinding.a
CGO_LDFLAGS="-lOpenCL -lclblast -L/usr/local/lib64/" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m model.gguf
```
</details>

---

## API reference

Full documentation is on
[pkg.go.dev](https://pkg.go.dev/github.com/AshkanYarmoradi/go-llama.cpp). This
is a map of what exists.

### Generation

| | |
|---|---|
| `New(path, ...ModelOption)` / `Free()` | load and release a model |
| `Predict(text, ...PredictOption)` | generate in one call |
| `SetTokenCallback(fn)` | stream tokens; return false to stop |
| `Embeddings(text)` / `TokenEmbeddings(tokens)` | embedding vectors |
| `ApplyChatTemplate(tmpl, msgs, addAssistant)` | render a conversation |
| `BuiltinChatTemplates()` / `GetChatTemplate(name)` | template discovery |

### Tokenization

`Tokenize` · `Detokenize` · `TokenToPiece` · `TokenizeString`

### Low-level inference

| | |
|---|---|
| `NewBatch(maxTokens, maxSeq)`, `Add`, `Reset`, `Len`, `Free` | batch assembly |
| `Decode(batch)` / `Encode(batch)` | run the graph |
| `Logits(i)` · `TokenEmbedding(i)` · `SequenceEmbedding(seq)` | outputs |
| `Synchronize()` | wait for queued computation |

### Sampling

Chain: `NewSamplerChain`, `Add`, `Sample`, `Accept`, `Reset`, `Free`, `Perf`.

Stages: `SamplerGreedy` · `SamplerDist` · `SamplerTopK` · `SamplerTopP` ·
`SamplerMinP` · `SamplerTypical` · `SamplerTemp` · `SamplerTempExt` ·
`SamplerXTC` · `SamplerTopNSigma` · `SamplerMirostatV2` ·
`SamplerPenalties`\* · `SamplerGrammar`\* · `SamplerDRY`\*

\* methods on `*LLama` — they need the model's vocabulary.

### KV cache

`MemoryClear` · `MemorySeqRemove` · `MemorySeqCopy` · `MemorySeqKeep` ·
`MemorySeqAdd` · `MemorySeqDiv` · `MemorySeqPosMin` · `MemorySeqPosMax` ·
`MemoryCanShift`

### State persistence

`StateSize` · `StateData` · `SetStateData` · `SaveSessionFile` ·
`LoadSessionFile` · `SequenceStateSize` · `SequenceStateData` ·
`SetSequenceStateData` · `SaveSequenceFile` · `LoadSequenceFile`

### Introspection

| | |
|---|---|
| `GetModelInfo()` | vocab size, layers, heads, parameters, size |
| `Architecture()` | RoPE type, file type, encoder/decoder, recurrent/hybrid |
| `ContextParams()` | the geometry the context *actually* uses |
| `ModelMetadata()` / `ModelMetadataValue(key)` | raw GGUF key-value header |
| `VocabType()` · `TokenText` · `TokenScore` · `TokenAttr` | vocabulary details |
| `IsEOG` · `IsControlToken` · `SuppressTokens` | token classification |
| `GetSpecialTokens()` | BOS, EOS, EOT, PAD, MASK, FIM tokens |
| `Perf()` / `PerfReset()` | prompt-eval and eval timings and counts |

### Runtime control

`Threads` / `SetThreads` · `SetEmbeddings` · `SetCausalAttn` · `ApplyLoRA` /
`ClearLoRA`

### Package level

`Version()` · `TimeUS()` · `SystemInfo()` · `SupportsMmap` · `SupportsMlock` ·
`SupportsGPUOffload` · `SupportsRPC` · `MaxDevices` · `MaxParallelSequences` ·
`FileTypeName` · `FlashAttnTypeName`

### Sampling options for `Predict`

| Option | Description | Default |
|---|---|---|
| `SetTemperature(t)` | randomness; ≤ 0 selects greedy | 0.8 |
| `SetTopK(k)` | keep the k likeliest tokens | 40 |
| `SetTopP(p)` | nucleus sampling threshold | 0.95 |
| `SetMinP(p)` | minimum probability relative to the best token | 0.05 |
| `SetPenalty(p)` | repetition penalty (1.0 disables) | 1.1 |
| `SetPresencePenalty(p)` | flat penalty for any seen token | 0.0 |
| `SetFrequencyPenalty(p)` | penalty scaled by occurrence count | 0.0 |
| `SetTypicalP(p)` | locally typical sampling | 1.0 |
| `SetXTCProbability(p)` / `SetXTCThreshold(t)` | exclude-top-choices sampling | disabled |
| `SetDRYMultiplier(m)` / `SetDRYBase(b)` / `SetDRYAllowedLength(n)` / `SetDRYPenaltyLastN(n)` | DRY repetition penalty | disabled |
| `SetTopNSigma(n)` | keep tokens within n standard deviations | disabled |
| `SetMirostat(mode)` | Mirostat v1 or v2 | disabled |
| `SetMirostatTAU(tau)` | Mirostat target entropy | 5.0 |
| `SetMirostatETA(eta)` | Mirostat learning rate | 0.1 |
| `WithGrammar(gbnf)` | constrain output to a GBNF grammar | none |
| `SetStopWords(...)` | stop generation at any of these strings | none |
| `SetLogitBias(spec)` | bias specific tokens, as `token(+\|-)value` | none |

---

## Models

Only the **GGUF** format is supported; legacy `ggml` files are not. For those,
use the [`pre-gguf`](https://github.com/AshkanYarmoradi/go-llama.cpp/releases/tag/pre-gguf)
tag.

```bash
# Convert a Hugging Face model
python llama.cpp/convert_hf_to_gguf.py /path/to/model --outfile model.gguf

# Quantize it
./llama.cpp/build/bin/llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

Ready-made GGUF models: [Hugging Face](https://huggingface.co/models?library=gguf).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, how to run
the tests against a real model, and what CI checks.

Security issues: see [SECURITY.md](SECURITY.md).

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

*If you find this useful, consider giving it a ⭐*

</div>
