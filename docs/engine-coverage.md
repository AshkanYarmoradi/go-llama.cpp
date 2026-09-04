# Engine coverage

This binding aims to cover llama.cpp's C API. This page records where that
stands and, more usefully, which functions are left out **on purpose** — so an
audit that finds them missing does not have to re-derive the reasoning.

Measured against `llama.cpp/include/llama.h` at **v0.3.0** (`c1d0e7a`).

|  |  |
|---|---|
| `LLAMA_API` declarations | 236 |
| Marked `DEPRECATED` upstream | 36 |
| **Live API surface** | **200** |
| **Covered by the binding** | **186** |
| Deliberately excluded | 14 |

Deprecated functions are not wrapped. llama.cpp names its replacement in every
`DEPRECATED` message, and the binding wraps that instead — for example
`llama_vocab_bos` rather than `llama_token_bos`.

## Deliberately excluded

### Not a real symbol (1)

| Function | Why |
|---|---|
| `llama_decode_with_sampler` | Commented out in `llama.h` (line 1534, `// TODO: extend in the future`). It appears in a text scan of the header but does not exist in the library. |

### Requires binding ggml first (2)

| Function | Why |
|---|---|
| `llama_attach_threadpool` | Takes `ggml_threadpool` objects. Wrapping it means exposing a slice of ggml's threading API, which is a separate project. |
| `llama_detach_threadpool` | As above. |

Thread counts are controllable without it — see `SetThreads`.

### Needs a C function-pointer vtable Go cannot supply (3)

| Function | Why |
|---|---|
| `llama_sampler_init` | Builds a custom sampler from a `llama_sampler_i` struct of C function pointers. Go cannot populate one, and a bridge per method would cost a cgo transition per token per stage. |
| `llama_sampler_apply` | Operates on a `llama_token_data_array` the caller owns — only meaningful when implementing a custom sampler. |
| `llama_sampler_copy` | Copies into a caller-allocated sampler; the same constraint. `Sampler.Clone` covers the useful case. |

Every sampler llama.cpp ships is exposed. See `SamplerTopK` and friends.

### Training API — out of scope (3)

| Function | Why |
|---|---|
| `llama_opt_init` | This is an inference binding. The optimizer API needs `ggml-opt.h` dataset and callback types, and a fundamentally different lifecycle. |
| `llama_opt_epoch` | As above. |
| `llama_opt_param_filter_all` | As above. |

### Superseded by something already exposed (5)

| Function | Why |
|---|---|
| `llama_get_logits` | `llama.h` marks it "TODO: deprecate in favor of `llama_get_logits_ith`", which is wrapped as `Logits(i)`. |
| `llama_get_model` | Recovers the model from a context. The binding holds both pointers in its own state, so it is never needed. |
| `llama_perf_sampler_print` | Prints to stderr. `Sampler.Perf` returns the same numbers as data. |
| `llama_model_init_from_user` | Constructs a model from caller-supplied tensors, for embedding llama.cpp in a larger ggml program rather than loading a GGUF file. |
| `llama_model_load_from_file_ptr` | Loads from an open `FILE*`, which does not cross the cgo boundary usefully. `NewFromSplits` covers the case that motivates it. |

## If you need one of these

Open an issue using the **Missing llama.cpp API** template and say what you are
building. Several of the exclusions above are judgement calls about cost versus
demand, and a concrete use case changes that calculation.

## Keeping this current

The audit is a text scan of the header, so it needs no build:

```bash
# every LLAMA_API symbol
grep -oE 'LLAMA_API [a-zA-Z_0-9 *]*\b(llama_[a-z_0-9]+)\s*\(' llama.cpp/include/llama.h \
  | grep -oE 'llama_[a-z_0-9]+\s*\($' | tr -d ' (' | sort -u > /tmp/engine_api.txt

# which of them binding.cpp references
while read -r fn; do
  grep -q "\b$fn\b" binding.cpp || echo "MISSING $fn"
done < /tmp/engine_api.txt
```

Subtract the `DEPRECATED` ones (`grep -n DEPRECATED llama.cpp/include/llama.h`)
and the table above. Anything left over is a real gap.
