# Contributing

Thanks for helping keep these bindings current with llama.cpp.

## Getting set up

You need Go 1.26 or newer, a C++17 compiler, CMake and make.

```bash
git clone --recurse-submodules https://github.com/AshkanYarmoradi/go-llama.cpp
cd go-llama.cpp
make libbinding.a
```

`make libbinding.a` builds the vendored llama.cpp and links it into a static
library. It takes a while the first time. Everything after that builds against
it:

```bash
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build ./...
```

### GPU and BLAS builds

A backend takes two settings that must match: `BUILD_TYPE` for `make`, and
the Go build tag for every `go` command after it.

```bash
make BUILD_TYPE=cublas libbinding.a
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build -tags cublas ./...
```

| `BUILD_TYPE` | Backend | Go tag | Needs |
|---|---|---|---|
| (empty) | CPU; on Apple also Metal, which llama.cpp builds by default | none | nothing extra |
| `metal` | Apple Metal, the same build as the default there | none | macOS |
| `cublas` | NVIDIA CUDA | `cublas` | the CUDA toolkit under `/usr/local/cuda` |
| `hipblas` | AMD ROCm/HIP | `hipblas` | ROCm 6.1 or newer under `/opt/rocm` (`ROCM_HOME` and `GPU_TARGETS` override the Makefile's defaults) |
| `vulkan` | Vulkan (NVIDIA, AMD, Intel) | `vulkan` | the Vulkan headers and loader, `glslc` and SPIRV-Headers (Ubuntu: `libvulkan-dev glslc spirv-headers`) |
| `openblas` | CPU with OpenBLAS | `openblas` | OpenBLAS, and pkg-config to find its headers unless `CMAKE_ARGS` sets `-DBLAS_INCLUDE_DIRS` |
| `blis` | CPU with BLIS | `blis` | BLIS, and pkg-config to find its headers unless `CMAKE_ARGS` sets `-DBLAS_INCLUDE_DIRS` |

How the Makefile handles them:

- The tag compiles `llama_<tag>.go`, which holds the backend's link flags.
  Keep them there, not in `CGO_LDFLAGS`: cmd/go places `CGO_LDFLAGS` before
  the package's `-lbinding`, where a linker that uses `--as-needed` drops the
  libraries.
- `build/.build_type` records the `BUILD_TYPE` the build was configured with,
  and a different one wipes `build/` and rebuilds llama.cpp from scratch.
  Changing only `CMAKE_ARGS` is not tracked, so run `make clean` first: for
  example before a CPU-only macOS build with `CMAKE_ARGS=-DGGML_METAL=OFF`.
- The `cublas`, `hipblas`, `vulkan`, `openblas` and `blis` options are added
  with `override CMAKE_ARGS+=`, so a `CMAKE_ARGS` given on the make command
  line adds to them. A plain `+=` is ignored in that case, and the build would
  come out CPU-only. `metal` keeps a plain `+=`, because llama.cpp turns Metal
  on by default on Apple anyway.
- A `BUILD_TYPE` the Makefile does not know, llama.cpp's own names `cuda` and
  `hip` included, stops `make` with an error. `clblas` stops with one that
  points to `vulkan`, because llama.cpp removed its CLBlast backend.
- `JOBS=n` limits the parallel compile. A CUDA build needs a few GB of RAM per
  job. On a build machine that lacks the GPU the library will run on, add
  `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to `CMAKE_ARGS`: otherwise llama.cpp
  compiles for the GPU it finds.
- The tag files expect CUDA under `/usr/local/cuda` and ROCm under
  `/opt/rocm`. For another location, pass its library directory as `-L` in
  `CGO_LDFLAGS`. Unlike a `-l`, a `-L` works wherever it sits on the link line.

A new backend needs, in the Makefile, its `GGML_*` option added with
`override`, `GO_TAGS` set to its tag (`make test` passes it to Ginkgo), and
its name in the list of known types. It also needs a `llama_<tag>.go` with its
link flags, and its tag in the `go vet` loop, both in
`.github/workflows/lint.yaml` and under [Before you push](#before-you-push).

## The layers

A change usually touches three files, in this order:

| File | What lives there |
|---|---|
| `binding.cpp` | C++ that calls llama.cpp and hides its types behind `void*` |
| `binding.h` | the `extern "C"` surface cgo sees |
| `llama.go` | the Go API, with the doc comments users actually read |

`llama.go` should read like Go, not like a C header. Return `[]int32` rather
than a pointer and a length, `error` rather than a status code, and a typed
enum with a `String` method rather than a bare int.

`options.go` holds the `ModelOption` and `PredictOption` functions. An option
that stops having an effect stays, so existing code keeps compiling. Give it,
and the `ModelOptions` or `PredictOptions` field behind it, a
`// Deprecated: has no effect.` paragraph. On the option, follow that sentence
with what to use instead, or with why nothing replaces it. Then add the option
to the README's "Accepted but ignored" table and to the migration guide's
table of options that do nothing.

### Buffer conventions

C functions that fill a caller's buffer follow one of two contracts, and the
comment on the function, in `binding.h` or at its definition in
`binding.cpp`, says which:

- **snprintf semantics** for strings: return the length the value *needs*. A
  return `>= buf_size` means it was truncated, and the caller retries at that
  size. `get_model_chat_template` works this way.
- **negative-required-size** for arrays: return the count written, or the
  negative of the count needed. `tokenize_text` works this way.

Pick whichever matches the underlying llama.cpp function and say so in the
comment. Do not silently truncate. That was a real bug in `GetChatTemplate`:
models with a template over 4 KiB got a quietly cut-off result.

### C++ exceptions must not reach cgo

An exception that escapes into cgo calls `std::terminate`: the process dies
with `SIGABRT` and no Go error is ever returned. Wrap anything that can throw.

Things that throw, and have:

- `std::stoi` / `std::stof` throw on malformed input, which is any option
  string that came from a caller.
- Most llama.cpp file operations throw internally, though the `LLAMA_API`
  entry points for state and quantization catch their own. Check before
  relying on it.
- llama.cpp looks token ids up with `.at()`, which throws `std::out_of_range`
  for an id outside the vocabulary. `TokenToPiece` and `Detokenize` ended the
  process that way. Their wrappers now check the id first, with
  `token_in_vocab` or `token_has_text`.
- A grammar sampler stage throws `std::runtime_error` for a token its grammar
  does not allow. It drops its parse state first, so the next time the stage
  is applied, llama.cpp asserts. Catching is not enough there: `sampler_accept`
  asks the grammar stages before it accepts, and `sampler_sample` restarts
  them after a throw.

The pattern is to catch, report on stderr, and fall back to the default or to
the value the Go layer already treats as "not available":

```cpp
try {
    model_params.main_gpu = std::stoi(maingpu);
} catch (const std::exception & e) {
    fprintf(stderr, "%s: ignoring malformed main_gpu %s: %s\n", __func__, maingpu, e.what());
}
```

### Engine aborts cannot be caught

`GGML_ABORT` and `GGML_ASSERT` end the process from inside llama.cpp, and no
`try` stops them. The only defence is never to hand the engine a value it
aborts on:

- The enum-to-name lookups abort on a value outside the enum:
  `llama_flash_attn_type_name`, `llama_load_mode_name`. Range-check first.
- `llama_load_mode_from_str` aborts on a name it does not recognise. Match
  against the engine's own names first, as `load_mode_from_str` does.
- `llama_decode` asserts that a batch fits in `n_batch`. `decode_batch` checks
  against `max_batch_tokens` first, so `Decode` returns -1 instead.
- The KV-cache operations assert on a sequence id at or past `n_seq_max`. The
  `memory_seq_*` wrappers check with `seq_in_range`, and the `state_seq_*`
  ones with `seq_state_id_ok`, which also lets -1 (every sequence) through.
- The per-token vocabulary accessors assert on a model without a vocabulary
  (`LLAMA_VOCAB_TYPE_NONE`). The wrappers check with `token_has_text`.

A few aborts still have no guard. Most are in `Sampler.Sample`: an output that
did not request logits, a chain with no stage that picks a token, and a grammar
stage placed after a truncation or picking stage that is handed an
end-of-generation token its grammar does not allow yet. Outside it, tokenizing
text with a model that has no vocabulary aborts, and so does shifting positions
(`MemorySeqAdd`, `MemorySeqDiv`, or `Predict` once the context is full) on an
M-RoPE model. [Tier 3](docs/production.md#tier-3-process-aborts) of the failure
model lists each one with the guard a caller can write. When an abort stays,
say so in the Go doc comment, as `Sampler.Sample` does.

Upstream moves functions from throwing to aborting, too. llama.cpp `6805ae35d`
did that to `llama_load_mode_from_str`, and the `try`/`catch` the binding had
around it went on compiling while guarding nothing.

Neither `go vet` nor the compile check catches either kind. Only a test that
feeds in a bad value does, so write that test.

### Callbacks into Go

C reaches Go through three `//export` functions: `tokenCallback`,
`goAbortCallback` and `goLogCallback`. Each looks the Go function up under a
lock and releases the lock before calling it. Keep that in any new one.
`tokenCallback` used to hold its lock across the call, so a token callback
that called `SetTokenCallback` deadlocked.

Releasing the lock lets a callback call `SetTokenCallback`, or methods on
other models. It does not make a model reentrant: a token callback runs in the
middle of its model's `Predict`, so it must not call `Predict` or `Free` on
that model.

### Mirrored enums

Enum values copied into `llama.go` as Go constants must be backed by a
`static_assert` in `binding.cpp`. If llama.cpp renumbers an enum, that turns a
silent wrong answer into a build failure. `PoolingType`, `VocabType`,
`TokenAttr` and `RopeType` share the "Guards for the enum values mirrored in
llama.go" block. The later ones (`LogLevel`, `LoadMode`, `SeqStateFlags` and
the `DefaultSeed` constant) sit next to their wrappers, which is where a new
one goes. `grep -n static_assert binding.cpp` lists them all.

## Testing

Tests are [Ginkgo](https://onsi.github.io/ginkgo/) specs in `llama_test.go`.
Most need a real model and skip without one. The specs labelled `gpu` are
meant for a GPU build, so leave them out the way CI does:

```bash
export TEST_MODEL=/path/to/model.gguf
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go test -v -timeout 1h ./... -ginkgo.label-filter='!gpu'
```

`go test` kills a test binary after 10 minutes by default, and a full run on a
CPU can take longer, hence `-timeout 1h`. That is the suite timeout the Ginkgo
CLI uses. With a GPU or BLAS `libbinding.a`, add its tag (`-tags cublas`);
without it the test binary does not link.

`make test` is what CI runs: it downloads CodeLlama-7B-Instruct Q2_K (about
2.8 GB) and runs the Ginkgo CLI with the same filter and the build type's tag,
giving a failing spec up to five attempts.
`make GPU_TESTS=true BUILD_TYPE=cublas test` runs only the `gpu` specs
instead. If you already have a Llama-family GGUF, point `TEST_MODEL` at it and
use the command above. A few specs assume CodeLlama's properties (it adds BOS,
it is decoder-only), so models from other families can fail them.

Put new specs in their own `Context` block rather than adding `It`s inside an
existing one, and place it above the `gpu`-labelled `Context` at the end of
the file. Sibling PRs conflict much less that way.

A context holds one sequence unless the model is loaded with `SetNSeqMax(n)`,
and the binding treats a sequence id outside `[0, NSeqMax)` as absent. A spec
that decodes more than one sequence has to load its model with `SetNSeqMax`.

Cover the edges, not just the happy path: out-of-range tokens and sequence
ids, batches larger than `NBatch`, buffers a byte too small, empty inputs, and
a model that lacks the feature.

### What CI runs

| Workflow | Jobs | Runs on | Checks |
|---|---|---|---|
| CI (`test.yaml`) | `ubuntu-latest`, `macOS-latest`, `macOS-metal-latest`, each with Go 1.26.x and the latest stable Go | every PR, pushes to `main`, tags | `make test`: Linux CPU, macOS with `CMAKE_ARGS="-DGGML_METAL=OFF"`, and macOS with `BUILD_TYPE=metal` |
| Lint (`lint.yaml`) | `Go`, `C++` | every PR, pushes to `main`, tags | the commands under [Before you push](#before-you-push) |
| GPU builds (`build-gpu.yaml`) | `ubuntu-cuda-build`, `ubuntu-vulkan-build` | every PR, pushes to `main`, tags | `make BUILD_TYPE=cublas` (in the `nvidia/cuda` image) and `BUILD_TYPE=vulkan` build; the backend's objects are in `libbinding.a` and none defines `main()`; the test binary and the example link and depend on the backend's libraries |
| GPU tests (`test-gpu.yaml`) | `ubuntu-cuda` | started by hand, a PR labelled `gpu`, or a push to `main` or a tag once the repository variable `GPU_RUNNER` is `true` | `make GPU_TESTS=true BUILD_TYPE=cublas test` on a self-hosted NVIDIA runner, which fails unless CUDA found a device and layers were offloaded to it |

The GPU builds jobs run on hosted runners without a GPU, so they compile and
link but never run a model. CI never builds `hipblas`, `openblas` or `blis`:
Lint only type-checks the package under their tags with `go vet`. If you
change the Makefile's handling of one,
or its `llama_<tag>.go`, build it yourself and say in the PR which backends
you built.

### Adding an Example

`Example` functions in `example_test.go` and `example_loop_test.go` are what
pkg.go.dev lists under Examples, and they are the source the README's code is
copied from.

- Name it after a real identifier (`ExampleLLama_Predict`, `ExampleBatch_Add`),
  or use a lower-case suffix for a package-level recipe (`Example_perplexity`).
  `go vet` rejects an `Example` whose name refers to nothing.
- Add `// Output:` only when the Example runs without a model. `go test` runs
  every Example that has one and compares what it prints, and an Example
  cannot skip, so one that loads a model would fail on any machine without
  that file. One that needs a model has no `// Output:`: it is compiled and
  vetted on every PR but never run, so back its behaviour with a spec in
  `llama_test.go`.
- Keep `// Output:` to values a spec already asserts, so a llama.cpp bump
  breaks both or neither.
- The example files are in package `llama_test`, like `llama_test.go`, which
  dot-imports this package. A helper named after an exported identifier
  (`New`, `Version`) fails to compile there; `generate` is fine.
- `example_loop_test.go` is a whole-file example: pkg.go.dev shows the entire
  file, `generate` included, only while it holds exactly one `Example` and no
  `Test`, `Benchmark` or `Fuzz` functions. Put new Examples in
  `example_test.go`.

## Documentation

- Go code in `README.md` and `docs/` mirrors an `Example` function in
  `example_test.go`, or `example_loop_test.go` for the whole-file generation
  loop, wherever one exists. Change both in the same PR. The Example is the
  copy `go vet` checks. Nothing checks a Go block that has no Example, so
  paste it into a function in a scratch module that `replace`s this one and
  run `go vet` there before you push.
- When the docs cite a spec as evidence, quote its `It` text exactly as it is
  written in `llama_test.go`, so that searching for the quote finds the spec.
  Renaming a spec means updating every page that quotes it.
- Never hard-code a llama.cpp commit SHA, build number, release or star
  count, or coverage number in the README. Dependabot checks llama.cpp for a
  new commit every day, so any such number goes stale fast. Link to the
  Releases page or `docs/engine-coverage.md` instead.

## Before you push

These are the same checks the Lint workflow runs, and they take seconds:

```bash
gofmt -l -e .
for t in "" cublas hipblas vulkan openblas blis; do go vet -tags "$t" ./...; done
go mod tidy && git diff --exit-code go.mod go.sum
./scripts/check-binding-symbols.sh
./scripts/engine-coverage.sh --check   # CI only warns when this is stale
```

Plus the C++ compile check, which reproduces CI's `binding.o` step without
building llama.cpp. This is what catches an upstream API break:

```bash
c++ -I./llama.cpp -I./llama.cpp/include -I./llama.cpp/ggml/include \
    -I. -I./llama.cpp/common \
    -std=c++17 -fPIC -Wall -Wextra -Wpedantic -Wcast-qual \
    -Wno-unused-function -fsyntax-only binding.cpp
```

`check-binding-symbols.sh` exists because cgo compiles `binding.h` into the
package: a function declared there but never defined type-checks fine and only
fails at link time, twenty minutes into a full CI run.

## When llama.cpp breaks the build

Dependabot bumps the submodule daily, so upstream API changes show up as a red
Dependabot PR. To fix one:

1. Read the compiler error. It usually names the changed signature directly.
2. Confirm against the header at that commit:
   `gh api repos/ggml-org/llama.cpp/contents/include/llama.h?ref=<sha>`
3. Check how llama.cpp's own `common/sampling.cpp` calls it. Matching upstream
   is nearly always right.
4. If the change forces a breaking Go API change, take it and explain why in
   the commit message. The engine's contract wins.

A bump can also break a single backend: a renamed `GGML_*` CMake option, or a
library ggml-cuda newly links. The GPU builds workflow catches both for CUDA
and Vulkan. CMake only warns about an option it does not read, but the
backend's objects are then missing from `libbinding.a` and the job fails; a
missing library fails the link step. Fix the option in the Makefile or the
flags in `llama_<tag>.go`.

A bump can build cleanly and still go red. If the test log ends in
`SIGABRT` and `signal arrived during cgo execution`, the engine aborted on
something the binding handed it, and its `file:line: message` is printed just
above the Go traceback. See [Engine aborts cannot be caught](#engine-aborts-cannot-be-caught).

## Commit messages

Conventional prefixes: `feat:`, `fix:`, `ci:`, `docs:`, `build:`, `refactor:`.

Write the body for someone reading `git log` in a year. Say what changed, and
why it had to change that way, especially when a signature moved because
upstream forced it.

## Pull requests

One concern per PR. The history here is one feature per PR (see #180–#184), and
it makes both review and bisection tractable.

Add an entry under `[Unreleased]` in `CHANGELOG.md` for anything a user of
the package would notice: a new API, changed behaviour, a fixed bug, or a
build change.

If your PR closes engine API gaps, list the `llama_*` functions it newly covers
in the description. That is how the coverage story stays legible. Then run
`./scripts/engine-coverage.sh --write` and commit the regenerated
`docs/engine-coverage.md`; Lint warns when the page is out of date.
