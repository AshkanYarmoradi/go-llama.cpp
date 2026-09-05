# Contributing

Thanks for helping keep these bindings current with llama.cpp.

## Getting set up

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

### Buffer conventions

C functions that fill a caller's buffer follow one of two contracts, and the
header says which:

- **snprintf semantics** for strings: return the length the value *needs*. A
  return `>= buf_size` means it was truncated, and the caller retries at that
  size. `get_model_chat_template` works this way.
- **negative-required-size** for arrays: return the count written, or the
  negative of the count needed. `tokenize_text` works this way.

Pick whichever matches the underlying llama.cpp function and say so in the
comment. Do not silently truncate — that was a real bug in `GetChatTemplate`,
where models with a template over 4 KiB got a quietly cut-off result.

### C++ exceptions must not reach cgo

An exception that escapes into cgo calls `std::terminate`: the process dies
with `SIGABRT` and no Go error is ever returned. Wrap anything that can throw.

Things that throw, and have:

- `llama_load_mode_from_str` throws `std::invalid_argument` for a name it does
  not recognise.
- `std::stoi` / `std::stof` throw on malformed input — which is any option
  string that came from a caller.
- Most llama.cpp file operations throw internally, though the `LLAMA_API`
  entry points for state and quantization catch their own. Check before
  relying on it.

The pattern is to catch, report on stderr, and return the value the Go layer
already treats as "not available":

```cpp
try {
    return (int) llama_load_mode_from_str(str);
} catch (const std::exception & e) {
    fprintf(stderr, "%s: %s\n", __func__, e.what());
    return -1;
}
```

Neither `go vet` nor the compile check catches this — only a test that feeds in
a bad value does. Write that test.

### Mirrored enums

Enum values copied into `llama.go` as Go constants must be backed by a
`static_assert` in `binding.cpp`. If llama.cpp renumbers an enum, that turns a
silent wrong answer into a build failure. See the block at the bottom of
`binding.cpp`.

## Testing

Tests are [Ginkgo](https://onsi.github.io/ginkgo/) specs in `llama_test.go`.
Most need a real model and skip without one:

```bash
export TEST_MODEL=/path/to/model.gguf
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go test -v ./...
```

`make test` will download a CodeLlama-7B Q2_K model and run against it, which
is what CI does. That is several gigabytes — point `TEST_MODEL` at a model you
already have if you can.

Put new specs in their own `Context` block rather than adding `It`s inside an
existing one. Sibling PRs conflict much less that way.

Cover the edges, not just the happy path: out-of-range tokens, buffers a byte
too small, empty inputs, and a model that lacks the feature.

## Before you push

These are the same checks CI runs, and they take seconds:

```bash
gofmt -l -e .
go vet ./...
./scripts/check-binding-symbols.sh
```

Plus the C++ compile check, which reproduces CI's `binding.o` step without
building llama.cpp — this is what catches an upstream API break:

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

1. Read the compiler error — it usually names the changed signature directly.
2. Confirm against the header at that commit:
   `gh api repos/ggml-org/llama.cpp/contents/include/llama.h?ref=<sha>`
3. Check how llama.cpp's own `common/sampling.cpp` calls it. Matching upstream
   is nearly always right.
4. If the change forces a breaking Go API change, take it and explain why in
   the commit message. The engine's contract wins.

## Commit messages

Conventional prefixes: `feat:`, `fix:`, `ci:`, `docs:`, `build:`, `refactor:`.

Write the body for someone reading `git log` in a year. Say what changed, and
why it had to change that way — especially when a signature moved because
upstream forced it.

## Pull requests

One concern per PR. The history here is one feature per PR (see #180–#184), and
it makes both review and bisection tractable.

If your PR closes engine API gaps, list the `llama_*` functions it newly covers
in the description. That is how the coverage story stays legible.
