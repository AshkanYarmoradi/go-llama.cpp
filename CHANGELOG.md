# Changelog

Notable changes to the Go API. The vendored llama.cpp submodule is bumped
daily by Dependabot and those bumps are not listed individually — only the ones
that changed this binding's behaviour.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Chat template application.** `ApplyChatTemplate` renders a `[]ChatMessage`
  into a prompt using the model's own GGUF template, or a named one from
  `BuiltinChatTemplates()`. Returns `ErrNoChatTemplate` when llama.cpp cannot
  place the template, rather than emitting a malformed prompt.
- **Per-sequence state persistence.** `SequenceStateData` /
  `SetSequenceStateData` checkpoint a single conversation slot without
  serializing the rest of the KV cache, and restore into any sequence id.
  File variants: `SaveSequenceFile` / `LoadSequenceFile`.
- **Session files.** `SaveSessionFile` / `LoadSessionFile` store the prompt
  tokens alongside the context state. In-memory whole-context state is now
  reachable too, via `StateData` / `SetStateData`.
- **Context introspection and control.** `ContextParams()` reports the geometry
  the context actually runs with — the engine clamps and rounds what it was
  asked for. Plus `Threads` / `SetThreads`, `SetEmbeddings`, `SetCausalAttn`
  and `Synchronize`.
- **KV-cache completion.** `MemorySeqAdd`, `MemorySeqDiv`, `MemorySeqPosMin`,
  `MemorySeqPosMax` and `MemoryCanShift`. Together with `MemorySeqRemove`
  these make context shifting expressible.
- **Performance counters.** `Perf()` / `PerfReset()` on both the context and a
  sampler chain, replacing stderr scraping.
- **Vocabulary introspection.** `VocabType`, `TokenText`, `TokenScore`,
  `TokenAttr`, `IsEOG`, `IsControlToken`, `AddSeparator`, `SuppressTokens`.
  `IsEOG` is the correct stop condition for a generation loop: many models
  define several end-of-turn tokens, and comparing against `EOS` alone misses
  them.
- **Model architecture queries.** `Architecture()` reports RoPE type, file
  type, encoder/decoder presence, recurrent/hybrid/diffusion flags, embedding
  widths and classifier labels.
- **Package-level helpers.** `Version()`, `TimeUS()`, `FileTypeName()`,
  `FlashAttnTypeName()`.
- Package documentation (`doc.go`), `CONTRIBUTING.md`, `SECURITY.md` and this
  changelog.
- `scripts/check-binding-symbols.sh`, which fails the build if `binding.h`
  declares a function `binding.cpp` does not define — previously a link-time
  error that only surfaced twenty minutes into CI.

### Changed

- **Breaking:** `SamplerPenalties` is now a method on `*LLama` rather than a
  package function. llama.cpp v0.3.0 added a required `n_vocab` parameter to
  `llama_sampler_init_penalties`, and the backend asserts it is non-zero, so
  the stage cannot be built without the model. This matches `SamplerGrammar`
  and `SamplerDRY`, which were already methods for the same reason.

      llama.SamplerPenalties(...)   ->   model.SamplerPenalties(...)

- Enum values mirrored into Go (`PoolingType`, `VocabType`, `TokenAttr`,
  `RopeType`) are now guarded by `static_assert`s against the engine headers,
  so an upstream renumbering becomes a build failure here instead of a silent
  misdecode.
- CI: the `push` trigger targeted `master`, which has never matched this
  repo's `main` branch, so no merge has ever been validated. GPU tests are now
  opt-in (`workflow_dispatch` or the `gpu` label) instead of leaving a queued
  check on every PR for 24 hours. A new Lint workflow runs gofmt, vet and a
  `binding.cpp` compile check in about a minute.

### Fixed

- **Build against llama.cpp v0.3.0.** `llama_sampler_init_penalties` gained a
  leading `n_vocab` parameter and `llama_sampler_init_dry` dropped
  `n_ctx_train`. Both are called from the binding, so the submodule bump broke
  the build.
- **`GetChatTemplate` silently truncated.** It used a fixed 4 KiB buffer and
  returned however much fit. Most instruct models have templates larger than
  that, so callers received a quietly cut-off template. Both layers now report
  the length the value needs and the Go side grows to it.
- `apply_chat_template` was a stub that ignored every argument and returned
  `-1`; see Added above.

## Earlier

Before this changelog, changes were tracked only in the commit history. Notable
entries:

- `feat: composable sampler objects` (#184)
- `feat: low-level batching, decode/encode, and KV-cache control` (#183)
- `feat: multi-LoRA adapters via ApplyLoRA / ClearLoRA` (#182)
- `fix: apply the parsed logit_bias in the sampler chain` (#181)
- `feat: bounds-safe tokenize / detokenize / token-to-piece` (#180)
