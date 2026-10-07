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
- **`SetNSeqMax(n)`**, a `ModelOption` for llama.cpp's `n_seq_max`: how many
  distinct sequences a context can hold. Contexts were always created with
  one, so `Decode` rejected any batch using a sequence id above 0, and
  `MemorySeqCopy` or restoring state into another sequence id could not be
  used. The default stays llama.cpp's 1; `ContextParams().NSeqMax` reports the
  value in effect. `New` fails if `n` exceeds `MaxParallelSequences()` or the
  context's batch size (the smaller of `SetNBatch` and `SetContext`), which
  the engine would otherwise abort on. The `MemorySeq` methods and the
  sequence state reads, restores and files treat an id at or above `NSeqMax`
  as a sequence that is not there. With more than one sequence, the engine
  aborted on such an id in the `MemorySeq` methods and the restores; the
  reads reported it as an empty sequence, except on a DeepSeek-V4 cache (see
  Fixed).

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
- **RoPE now defaults to the model's trained values.** `DefaultModelOptions`
  set `FreqRopeBase: 10000` and `FreqRopeScale: 1.0`, and the binding passed
  both to every context, overriding what the GGUF file says. A model trained
  with another base ran with the wrong positional encoding and degraded
  output: the CodeLlama model CI tests with was trained with 1,000,000, and
  Llama 3 uses 500,000. Both now default to 0, which means "from the model".
  `WithRopeFreqBase` and `WithRopeFreqScale` still override; pass
  `WithRopeFreqBase(10000), WithRopeFreqScale(1)` for the old behaviour.
- **The `SetThreads` `PredictOption` works, for one call.** It was stored and
  never read, so `Predict` always ran on the context's thread count (llama.cpp
  starts it at 4). It now sets the thread count for generation and prompt
  processing for that call and restores the context's setting afterwards.
  `DefaultOptions.Threads` is now 0, which keeps the context's setting; change
  that with `(*LLama).SetThreads(n, nBatch)`.
- **`WithGrammar` with a grammar that does not parse makes `Predict` fail**
  instead of generating unconstrained text.
- `Embeddings` tokenizes with the model's special tokens (`add_special`), as
  llama.cpp's embedding example does. It used to add them only when the
  vocabulary asks for BOS, so vectors change for models that add EOS or SEP
  but not BOS.
- `Embeddings`, `TokenEmbeddings` and `TokenizeString` ignore their
  `PredictOption` arguments. The only one they used, `SetTokens`, sized the
  output buffers behind the overruns listed under Fixed.

### Deprecated

- Options that have no effect are now marked `Deprecated: has no effect.`, so
  staticcheck and editors flag them. They were already ignored, so nothing
  changes at run time: `IgnoreEOS`, `EnableF16KV`, `SetPathPromptCache`,
  `EnablePromptCacheAll`, `EnablePromptCacheRO`, `SetMlock`, `SetMemoryMap`,
  `SetPredictionMainGPU`, `SetPredictionTensorSplit`, the predict-side
  `SetRopeFreqBase` and `SetRopeFreqScale` (use `WithRopeFreqBase` and
  `WithRopeFreqScale` at load), `SetNDraft`, `SetTailFreeSamplingZ`,
  `SetPenalizeNL`, `SetModelSeed`, `EnableF16Memory`, `EnabelLowVRAM` and
  `SetLoraBase`, along with the `ModelOptions` and `PredictOptions` fields
  behind them.

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
- **Some enum lookups ended the process.** llama.cpp aborts, rather than
  throws, on a value it does not recognise, and an abort inside a cgo call
  cannot be recovered from. `ParseLoadMode` with an unknown name (since
  llama.cpp `6805ae35d` turned its `throw` into an abort), and
  `LoadMode.String` or `FlashAttnTypeName` with a value outside the enum, all
  killed the process. They now return `LoadModeAuto`, `"LoadMode(n)"` and `""`
  respectively.
- **`Embeddings` and `TokenEmbeddings` overran the Go heap.** The output
  buffer was sized from `SetTokens`, 128 floats by default (with
  `SetTokens(0)`, 99,999,999 for `Embeddings` and 9,999,999 for
  `TokenEmbeddings`), while the engine wrote `n_embd` floats into it: 4096
  for a 7B model. They now return exactly one vector of the model's embedding
  width, and the C side never writes past the buffer it is given.
- **`Embeddings` returned the same vector for every input** of a generative
  model. It read the first output row, which is the hidden state of the BOS
  token. It now returns the pooled sequence embedding when the context pools
  (`ContextParams().Pooling` is not `PoolingNone`), and otherwise the last
  token's. `TokenEmbeddings` embeds the token ids as given; it used to
  detokenize them and tokenize the text again, adding a second BOS. Input
  longer than `ContextParams().NBatch` returns an error instead of aborting
  the process, as does a `TokenEmbeddings` token id outside the vocabulary.
- **`SetEmbeddings` did not change what `Embeddings` accepts.** A model loaded
  without `EnableEmbeddings` kept returning "model loaded without embeddings"
  after `SetEmbeddings(true)`, contrary to its documentation.
- **`WithGrammar` did not constrain `Predict` and could abort the process.**
  The grammar stage ran after the token had already been picked, and every
  token was then accepted twice, so an off-grammar pick threw a C++ exception
  across cgo. The grammar now runs first. Accepting each token twice also
  counted it twice in the repetition-penalty history of every `Predict` call,
  grammar or not; each token is now accepted once. Any C++ exception inside
  `Predict` now returns "inference failed" instead of aborting.
- **`Predict` included the end-of-generation token's text**, such as `</s>`,
  at the end of its result and passed it to the token callback.
- **`SetStopWords` trimmed a character set, not the stop word.** It used
  `strings.TrimRight`, which also removed any trailing characters that occur
  in the stop word: output `hello` with stop word `lo` came back as `he`. It
  now uses `strings.TrimSuffix` and returns `hel`.
- **`TokenizeString` overran the Go heap** on text longer than its token
  limit (128 by default). It now returns every token.
- **`Predict` leaked C memory on every call**: the prompt and every option
  string, plus the whole parameter block when it failed. `TokenizeString`,
  `Embeddings` and `TokenEmbeddings` leaked the same way.
- **`LoadState` failed on a fresh context.** It read only as many bytes as the
  loading context's own state held, so a file saved from a fuller context was
  cut short. It now reads the whole file.
- **The `SetTokenCallback` `PredictOption` deleted the persistent callback**
  set with `(*LLama).SetTokenCallback` after a successful call, and stayed
  registered after a failed one. It now applies to its call only, and the
  persistent callback is back in effect afterwards. A token callback can also
  call `SetTokenCallback` itself, which used to deadlock.
- **`SetSequenceSampler(seq, nil)` did not detach a backend sampler**: it
  returned `false` without reaching the engine. It now detaches and returns
  `true`, as does a chain with no stages. Passing a bare stage instead of a
  chain returns `false`; the engine used to read it as a chain.
- **`MemorySeqRemove` with a negative sequence id other than -1 aborted the
  process**, although a negative id is documented to match every sequence.
  It now does.
- **Calling `Free` twice freed the model twice.** The second call is now a
  no-op, as is `Free` on a nil `*LLama`.
- **Oversized batches aborted the process.** `Decode` with more tokens than
  `ContextParams().NBatch` hit an engine assertion; it now returns -1, as does
  `Encode` beyond `NUbatch`. `Predict` sent its prompt in chunks of `SetBatch`
  (512 by default) whatever the context accepted, so a prompt longer than a
  smaller `NBatch` aborted; the chunks are now capped at `NBatch`.
- **DRY never ran unless `SetDRYPenaltyLastN` was given a positive window.**
  The default window, -1, is documented as the context size, but since
  llama.cpp `a6aa6f545` (the change that also dropped `n_ctx_train`, see the
  v0.3.0 build fix above) `llama_sampler_init_dry` clamps a negative window to
  0, which disables the stage. So `SetDRYMultiplier` alone left `Predict`
  unchanged, and `SamplerDRY(m, b, n, -1)` built an empty stage (`Name()`
  reported `"?dry"`). The binding now resolves a negative window to the
  context size, `ContextParams().NCtx`, itself; 0 still disables DRY. A
  larger window is now cut to the context size as well: the engine allocates
  the whole window up front, so a window such as `math.MaxInt32` asked for
  about 16 GiB and ended the process when that could not be had. Should the
  allocation still fail, `SamplerDRY` returns an empty `Sampler`, which `Add`
  ignores.
- **`SetTensorSplit` values could leak from one `New` into the next.** The
  split was parsed into a function-level static array that was never cleared,
  so a model loaded with a shorter split than an earlier one inherited the
  earlier split's trailing proportions, and concurrent `New` calls wrote the
  same array. Each load now parses into its own zeroed buffer of
  `MaxDevices()` entries, the number the engine reads.
- **`TokenToPiece` and `Detokenize` aborted the process on a token id outside
  the vocabulary.** llama.cpp throws `std::out_of_range` for one, and the
  exception crossed cgo. `TokenToPiece` now returns `""` for such a token and
  `Detokenize` returns `""` when any token is out of range, as `TokenText`
  already did. `TokenToPiece`, `TokenText`, `TokenScore`, `TokenAttr` and
  `IsControlToken` also return their empty value for every token of a model
  without a vocabulary (`VocabNone`, such as an audio codec), on which
  llama.cpp asserts.
- **`Sampler.Accept` and `Sampler.Sample` could abort the process.** A
  grammar stage throws `std::out_of_range` for a token id outside the
  vocabulary and `std::runtime_error` for a token its grammar does not allow,
  and both crossed cgo. An end-of-generation token offered before the grammar
  was complete, or -1 offered to an adaptive-p stage that had not sampled
  yet, aborted inside llama.cpp. `Accept` now ignores a negative id and asks
  the grammar stages first: a token one of them would refuse is recorded by
  no stage, and the reason goes to stderr. `Sample` returns -1 when a stage
  throws, which a grammar stage does when it comes after a truncation or
  picking stage and is handed a token it does not allow. It then restarts
  the chain's grammar stages, because llama.cpp leaves a grammar that refused
  a token unable to continue and asserts the next time it is applied. Such a
  pick that is an end-of-generation token still aborts inside llama.cpp, so
  grammar stages belong first in the chain. `Sample` on an empty `Sampler`
  returns -1 instead of crashing.
- **`SetSequenceStateDataWith` with `SeqStateOnDevice` could abort the
  process.** An on-device capture leaves the data in the context, one
  snapshot per sequence id, and returns only the cell layout. llama.cpp
  asserted when the bytes named a sequence with no snapshot (bytes captured
  on the host or by another `LLama`), aborted when they no longer matched the
  snapshot (an earlier capture of the same sequence, or other flags), and
  where the sizes happened to agree restored the newer data into the older
  layout. The binding now remembers the latest on-device capture of each
  sequence and restores only those bytes with those flags; anything else
  returns an error. On a recurrent or hybrid model holding more than one
  sequence, capturing sequence -1 on the device aborted too; its size is now
  0 and the capture returns an error.
- **Sequence state reads accepted sequence ids the context does not hold.**
  `SequenceStateSize`, `SequenceStateData`, their `With` variants and
  `SaveSequenceFile` now follow the rule the restores use: an id outside
  `[0, NSeqMax)`, other than -1 for every sequence, is not there. The size is
  0, and the others return an error without writing a file. The engine
  reported an id from `NSeqMax` to 255 as an empty sequence, and
  `SaveSequenceFile` wrote a file for it, except on a DeepSeek-V4 cache
  holding more than one sequence, where such an id aborted the process. Past
  255, its per-cell sequence bitset throws `std::out_of_range`, which only a
  `catch` inside llama.cpp kept from crossing cgo.
- `TokenEmbedding` and `SequenceEmbedding` always read `n_embd` floats. For a
  model whose output embedding width is smaller than `n_embd` that read past
  the engine's row, as it did for a reranker, whose row is its scores; where
  the output width is larger it returned a truncated row.
- Wrong doc comments: `SetMMap`, `SetTensorSplit`, `NewModelOptions`, the
  `SetTokenCallback` option, `SetRepeat` (it is the penalty look-back window)
  and `SetNKeep`. `Sampler.Sample` now documents that it already accepts the
  token, so callers must not `Accept` it again.

## Earlier

Before this changelog, changes were tracked only in the commit history. Notable
entries:

- `feat: composable sampler objects` (#184)
- `feat: low-level batching, decode/encode, and KV-cache control` (#183)
- `feat: multi-LoRA adapters via ApplyLoRA / ClearLoRA` (#182)
- `fix: apply the parsed logit_bias in the sampler chain` (#181)
- `feat: bounds-safe tokenize / detokenize / token-to-piece` (#180)
