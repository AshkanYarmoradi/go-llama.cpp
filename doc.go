/*
Package llama runs GGUF language models inside your Go program. It binds
[llama.cpp], which is compiled into your binary through cgo: there is no
server to run, no API key, and no network access at run time.

The API has two levels:

  - [LLama.Predict] runs a whole generation in one call: tokenize the prompt,
    decode, sample, stop. Most programs need only this.
  - The low-level API exposes llama.cpp's own parts: batches ([NewBatch],
    [LLama.Decode]), sampler chains ([NewSamplerChain]), the KV cache (the
    Memory methods) and saved state. Use it when you drive generation
    yourself.

Around them sit chat templates ([LLama.ApplyChatTemplate]), embeddings,
model inspection ([LLama.GetModelInfo]), log routing ([SetLogHandler]),
LoRA adapters ([LLama.ApplyLoRA]) and quantization ([Quantize]).

This is not a go get-only package: you build llama.cpp once with make before
the first go build. See Building and GPUs below. Longer guides live in the
repository: [getting started], [cookbook], [production] and [how it works].

# Quick start

Load a model, ask it something, free it:

	package main

	import (
		"fmt"
		"log"

		llama "github.com/AshkanYarmoradi/go-llama.cpp"
	)

	func main() {
		model, err := llama.New("model.gguf", llama.SetContext(2048))
		if err != nil {
			log.Fatal(err)
		}
		defer model.Free()

		answer, err := model.Predict("[INST] Answer to the following question:\nhow much is 2+2?\n[/INST]",
			llama.SetTokens(32))
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(answer)
	}

model.gguf can be any GGUF model file; the [getting started] guide shows where
to find one. On every pull request, CI's "predicts successfully" spec sends
this prompt to CodeLlama-7B-Instruct, quantized to Q2_K, and checks that the
answer contains 4. The prompt uses CodeLlama's
instruction format. For other models, let the model's own chat template build
the prompt (see Chat).

Options are functions. A [ModelOption] is fixed when [New] loads the model:

  - [SetContext] sets how many tokens of prompt and reply the model sees at
    once. The default is 512. llama.cpp rounds it up to a multiple of 256,
    and 0 means the model's training length.
  - [SetGPULayers] sets how many layers run on the GPU. The default is 0: the
    weights stay in RAM, though a GPU build may still send large prompt
    batches to the GPU.
  - [SetNBatch] sets the most tokens one [LLama.Decode] may take. The default
    is 512, and a generative model caps it at the context size;
    ContextParams().NBatch reports the value in use.
  - [WithRopeFreqBase] and [WithRopeFreqScale] override the RoPE values the
    model was trained with. By default the model's own values are used.

A [PredictOption] applies to one [LLama.Predict] call:

  - [SetTokens] caps the reply. The default is 128 tokens; 0 means generate
    until the model ends its turn. The result is cut off, without an error,
    at 8 bytes per SetTokens token plus the prompt's length and 1 KiB, and
    never holds more than 4 MiB; a token callback (see Streaming) still sees
    every piece.
  - [SetTemperature], [SetTopK], [SetTopP] and [SetMinP] tune sampling
    (defaults 0.8, 40, 0.95 and 0.05). A temperature of 0 or below always
    picks the most likely token. Only [WithGrammar] and [SetLogitBias] still
    apply then; the repetition penalties and DRY are left out.
  - [SetDRYMultiplier] turns on DRY ("don't repeat yourself"), which
    penalises tokens that would extend a repeat of text already generated.
    It is off by default. [SetDRYPenaltyLastN] sets how many recent tokens
    it scans: the default of -1, like any negative value, means the context
    size, a larger value is cut to the context size, and 0 turns DRY off.
  - [SetSeed] fixes the random seed so a call can be repeated. The default
    of -1 picks a new seed for every call.
  - [SetStopWords] stops generating once the reply ends with one of the
    words, and removes that word from the result.
  - [SetThreads] sets the CPU threads for this call only. Without it the
    context's own setting applies: llama.cpp's default of 4, or whatever
    [LLama.SetThreads] chose.

[DefaultModelOptions] and [DefaultOptions] hold every default. They are
shared package variables: read them, but pass options instead of changing
them.

# Chat

Chat models expect a specific prompt format. [LLama.ApplyChatTemplate] renders
a conversation with the template stored in the GGUF file when you pass "" as
the template:

	msgs := []llama.ChatMessage{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "How much is 2+2?"},
	}
	prompt, err := model.ApplyChatTemplate("", msgs, true)
	if errors.Is(err, llama.ErrNoChatTemplate) {
		prompt, err = model.ApplyChatTemplate("chatml", msgs, true)
	}
	if err != nil {
		return err
	}
	reply, err := model.Predict(prompt, llama.SetTokens(256))
	if err != nil {
		return err
	}
	msgs = append(msgs, llama.ChatMessage{Role: "assistant", Content: reply})

The last argument, true, ends the prompt with the opening of an assistant
turn, which is what you want before generating. The reply stops where the
model ends its turn and does not include the end-of-turn token's text.

llama.cpp does not run a full Jinja engine. It recognises a fixed set of
well-known templates, and returns [ErrNoChatTemplate] when the model has no
template or one it does not recognise. Fall back to a name from
[BuiltinChatTemplates], as above, or format the prompt yourself.

Predict clears the KV cache before every call, so it remembers nothing between
turns. For the next turn, append the user's message to msgs and render the
whole history again.

# Streaming

The [SetTokenCallback] option hands you each piece of text as it is
generated. Return false to stop early; Predict then returns the text that came
before that piece:

	_, err := model.Predict(prompt,
		llama.SetTokens(512),
		llama.SetTokenCallback(func(piece string) bool {
			fmt.Print(piece)
			return true
		}),
	)
	if err != nil {
		return err
	}

The callback runs on the goroutine that called Predict, before the next token
is generated, so keep it short. A piece can end in the middle of a multi-byte
UTF-8 character; buffer pieces if you need whole runes. With [SetStopWords],
the stop word's pieces reach the callback before generation stops; only
Predict's result has the word removed.

The binding releases its callback lock before it calls the callback, so the
callback may call [LLama.SetTokenCallback], and may use other models that no
other goroutine is using. It must not call Predict, Decode or Free on the
model that is generating, or change that model's KV cache: Predict is still
using it.

The option covers one call. [LLama.SetTokenCallback] installs a callback for
every later Predict on that model. A call that passes the option uses the
option's callback, and the model's own comes back afterwards.

# Cancelling

A token callback can stop generation between tokens, but not while llama.cpp
is still working through a long prompt. [LLama.SetAbortCallback] can: the
engine polls it while it computes, and returning true stops the work in
progress. Use it to honour a [context.Context]:

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	model.SetAbortCallback(func() bool { return ctx.Err() != nil })
	defer model.SetAbortCallback(nil)

	reply, err := model.Predict(prompt, llama.SetTokens(0))
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err() // stopped by the deadline, not a failure
		}
		return err
	}
	fmt.Println(reply)

An aborted Predict returns an error and no text; collect pieces with a token
callback if you need the partial reply. In your own loop, [LLama.Decode]
returns 2 when aborted. The callback runs on an engine thread and is called
very often, so it must be cheap and safe for concurrent use.

Only the CPU backend polls the abort callback while it computes. Work already
handed to a GPU, Metal included, runs to the end. With every layer offloaded,
little of the graph runs on the CPU, so a decode may finish before the
callback is seen; a token callback still stops generation between tokens.

# Embeddings

Load the model with [EnableEmbeddings], then call [LLama.Embeddings]:

	model, err := llama.New("embedding-model.gguf", llama.EnableEmbeddings)
	if err != nil {
		return err
	}
	defer model.Free()

	vec, err := model.Embeddings("The quick brown fox")
	if err != nil {
		return err
	}
	fmt.Println(len(vec) == model.Architecture().EmbdOut) // true

The result has the model's output embedding width, [Architecture].EmbdOut,
which equals ModelInfo.EmbeddingSize for nearly every model. A model with a
pooling type, as embedding models usually have, returns the pooled embedding
of the whole text. A model without one, such as a causal chat model, returns
the embedding of the last token; for those, [LLama.ContextParams] reports
[PoolingNone]. A reranker ([PoolingRank]) returns its scores instead. The
text is decoded in one batch, so text that tokenizes to more than
ContextParams().NBatch tokens returns an error. Like Predict, each call clears
the KV cache first.

[LLama.TokenEmbeddings] does the same for token ids you already have, without
tokenizing again; an id outside the vocabulary is an error.
[LLama.SetEmbeddings] switches a loaded context between embeddings and
generation; Embeddings and TokenEmbeddings return an error while it is off.
After your own [LLama.Decode], [LLama.TokenEmbedding] and
[LLama.SequenceEmbedding] read the vectors directly.

# Writing your own loop

Predict is a fixed recipe. To choose your own sampling, stopping rule or cache
handling, write the loop yourself. This function decodes a prompt and samples
from a chain you build, without stop words, context shifting or splitting long
prompts:

	// generate decodes prompt, then samples up to maxTokens tokens from chain,
	// stopping at the model's end-of-generation token.
	func generate(model *llama.LLama, chain *llama.Sampler, prompt string, maxTokens int) (string, error) {
		model.MemoryClear(true) // start from an empty cache, as Predict does

		tokens := model.Tokenize(prompt, true, true) // add BOS, parse chat-template markup
		if len(tokens) == 0 {
			return "", errors.New("empty prompt")
		}
		if n := model.ContextParams().NBatch; len(tokens) > n {
			return "", fmt.Errorf("prompt is %d tokens; one Decode takes at most %d", len(tokens), n)
		}

		batch := llama.NewBatch(len(tokens), 1)
		defer batch.Free()
		for i, t := range tokens {
			// Only the last prompt token needs logits: it is the one sampled from.
			if err := batch.Add(t, int32(i), []int32{0}, i == len(tokens)-1); err != nil {
				return "", err
			}
		}

		var out strings.Builder
		pos := int32(len(tokens))
		for range maxTokens {
			switch rc := model.Decode(batch); rc {
			case 0:
			case 1:
				return out.String(), errors.New("the context is full")
			case 2:
				return out.String(), errors.New("stopped by the abort callback")
			default:
				return out.String(), fmt.Errorf("decode failed with status %d", rc)
			}

			next := chain.Sample(model, -1) // Sample also accepts the token: no Accept call
			if next < 0 {
				return out.String(), errors.New("the sampler gave no token; the reason is on stderr")
			}
			if model.IsEOG(next) {
				break
			}
			out.WriteString(model.TokenToPiece(next, false))

			batch.Reset()
			if err := batch.Add(next, pos, []int32{0}, true); err != nil {
				return out.String(), err
			}
			pos++
		}
		return out.String(), nil
	}

Build the chain from Predict's default stages, in Predict's order, and call it:

	chain := llama.NewSamplerChain()
	defer chain.Free() // frees every stage too
	chain.Add(model.SamplerPenalties(64, 1.1, 0, 0))
	chain.Add(llama.SamplerTopK(40))
	chain.Add(llama.SamplerTopP(0.95, 1))
	chain.Add(llama.SamplerMinP(0.05, 1))
	chain.Add(llama.SamplerTemp(0.8))
	chain.Add(llama.SamplerDist(llama.DefaultSeed)) // DefaultSeed asks for a random seed

	text, err := generate(model, chain, "[INST] Name three Go proverbs. [/INST]", 128)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)

The package's generationLoop example runs the same loop as a complete program.
Points that are easy to get wrong:

  - [Sampler.Sample] already accepts the token into the chain. Do not call
    [Sampler.Accept] for it as well; Accept is for tokens you chose some
    other way. Accept ignores a negative id, and a chain with a grammar stage
    refuses a token its grammar does not allow next: no stage records it,
    and the reason goes to stderr.
  - Put grammar stages ([LLama.SamplerGrammar], [LLama.SamplerGrammarLazy])
    first in the chain, ahead of truncation stages such as top-k and of the
    stage that picks the token. A grammar placed later can be handed a token
    it does not allow. Sample then returns -1 and restarts the chain's
    grammar stages from the beginning of their grammar; if that token ends
    generation, llama.cpp ends the process instead.
  - Check Sample's result for -1. Sample returns -1 when a stage fails, as
    in the case above, and for an empty [Sampler], which is what
    SamplerGrammar returns for a grammar that does not parse. The reason
    goes to stderr.
  - Sample only from an output that asked for logits, with the last argument
    of [Batch.Add]. Index -1 means the last output.
  - Stop on [LLama.IsEOG], not on a single end-of-sequence id. Many models
    have several end-of-turn tokens.
  - [LLama.SamplerDRY] reads its window as [SetDRYPenaltyLastN] does: a
    negative penaltyLastN is the context size, a larger one is cut to the
    context size, and 0 disables the stage.
  - One batch holds at most ContextParams().NBatch tokens. Decode returns -1
    for a larger one, so feed a long prompt in several batches.

[LLama.Decode] returns 0 on success, 1 when the KV cache has no room left, 2
when the abort callback stopped it, and a negative value on error. When the
cache is full and [LLama.MemoryCanShift] reports true, drop old tokens and
slide the rest down. Here keep is how many tokens to preserve at the start,
and n how many to drop after them:

	if model.MemoryCanShift() {
		model.MemorySeqRemove(0, keep, keep+n)
		model.MemorySeqAdd(0, keep+n, -1, -n)
	}

Generation then continues at position model.MemorySeqPosMax(0)+1.

A context holds one sequence, id 0, by default. The SetNSeqMax load option
raises that limit (llama.cpp's n_seq_max). [New] fails if the value is above
[MaxParallelSequences] or above the context's batch size, the smaller of
[SetNBatch] and [SetContext]. Each sequence is then a separate conversation in
the same KV cache: tag tokens with their sequence id in [Batch.Add], share a
common prompt prefix with [LLama.MemorySeqCopy] and drop a finished one with
[LLama.MemorySeqRemove]. ContextParams().NSeqMax reports the limit, and
ContextParams().NCtxSeq each sequence's share of the context. Decode rejects a
batch that uses an id outside [0, NSeqMax). The Memory and sequence state
methods treat such an id as a sequence that is not there, apart from the
negative ids some of them take to mean every sequence (see Errors and process
aborts).

To pause and resume, [LLama.SaveSessionFile] writes the KV cache together with
its tokens, and [LLama.LoadSessionFile] restores both, in this process or
another. [LLama.StateData] and [LLama.SetStateData] save and restore the same
cache in memory without its tokens, and [LLama.SaveState] and [LLama.LoadState]
do the same with a file; keep the tokens yourself. All of them restore into a
fresh context loaded from the same model with the same options.
Logits are not part of the saved state, so remove the last token from the
cache and decode it again before you sample. Because Predict clears the cache,
continue a restored session with your own loop.

# Memory and lifetime

A [LLama] holds the weights, one context and its KV cache outside the Go heap,
where the garbage collector does not see them. Nothing frees them, so call
[LLama.Free] when you are done. A second Free is a no-op, but no other method
may be called after Free.

Memory use is roughly the model file plus a KV cache that grows with
[SetContext]. Every [New] creates a separate model and context, with its own
KV cache and its own copy of any GPU layers.

The other C-backed values follow the same pattern:

  - Free every [Batch] from [NewBatch].
  - A sampler stage belongs to you until [Sampler.Add] puts it in a chain.
    From then on the chain owns it, and [Sampler.Free] on the chain frees
    every stage. Do not free a stage after adding it.
  - [Sampler.At] lends you a stage of a chain; never free it. [Sampler.Remove]
    and [Sampler.Clone] return samplers you own and must free.
  - Free a chain that holds a grammar or infill stage ([LLama.SamplerGrammar],
    [LLama.SamplerGrammarLazy], [LLama.SamplerInfill]) before you free the
    model it came from: those stages keep a pointer to its vocabulary.
  - A chain attached with [LLama.SetSequenceSampler] must stay alive while it
    is attached. Passing a nil chain detaches it.

Values the package returns, such as logits, token slices, strings and state
bytes, are Go copies that stay valid after Free. Slices you pass in are only
read during the call.

# Concurrency

A [LLama] is not safe for concurrent use. Its context has one KV cache and one
output buffer, so every call that touches it, from Predict and Decode to the
Memory and State methods, must be serialized. Either guard each model with a
mutex:

	type lockedModel struct {
		mu    sync.Mutex
		model *llama.LLama
	}

	func (m *lockedModel) Predict(prompt string, opts ...llama.PredictOption) (string, error) {
		m.mu.Lock()
		defer m.mu.Unlock()
		return m.model.Predict(prompt, opts...)
	}

or give each worker its own [New], which multiplies the KV cache and any GPU
memory by the number of workers. Several sequences in one context let a
single caller interleave conversations in one batch; they do not make
concurrent calls safe.

[Sampler] and [Batch] values are not safe for concurrent use either. Separate
models can run on separate goroutines, but the log handler set with
[SetLogHandler] is process-wide and is called from engine threads, possibly at
the same time. It must be safe for concurrent use and must not call back into
this package.

# Errors and process aborts

Most mistakes come back as Go errors: a model that fails to load (the error
quotes the path), a full [Batch], an unknown chat template
([ErrNoChatTemplate]), a state or session file that cannot be restored, and
input to [LLama.Embeddings] or [LLama.TokenEmbeddings] that is too long or
holds an id outside the vocabulary. [New] also fails when SetNSeqMax asks for
more sequences than [MaxParallelSequences] or the context's batch size
allows. In the sequence state methods, an id outside [0, NSeqMax) other than
-1, which means every sequence, is an error: [LLama.SequenceStateData], the
restores and [LLama.SaveSequenceFile] return one, SaveSequenceFile writes no
file, and [LLama.SequenceStateSize] reports 0.

[LLama.Predict] returns one generic "inference failed" error. The usual
causes are a prompt longer than the context minus 4 tokens, a [WithGrammar]
grammar that does not parse, a decode failure, the abort callback, a context
too small to shift, and a C++ exception inside the engine. The binding prints
its own reasons to stderr; the engine's go to llama.cpp's log, which is also
stderr unless you route it elsewhere:

	llama.SetLogHandler(func(level llama.LogLevel, text string) {
		if level == llama.LogLevelWarn || level == llama.LogLevelError {
			log.Print(level, ": ", text)
		}
	})

A handler that discards everything silences the engine; SetLogHandler(nil)
restores stderr.

Other out-of-range input is ignored or gets an empty or sentinel value:

  - [LLama.TokenToPiece] returns "" for a token id outside the vocabulary,
    and [LLama.Detokenize] returns "" if any id is outside it.
  - [Sampler.Sample] returns -1 when it has no token to give, and
    [Sampler.Accept] ignores a token it cannot take (see Writing your own
    loop).
  - [LLama.Decode] returns -1 for a batch larger than ContextParams().NBatch
    or one that uses a sequence id outside [0, NSeqMax).
  - The Memory methods ignore a sequence id the context does not hold: they
    do nothing, or report false or -1. [LLama.MemorySeqRemove] takes any
    negative id to mean every sequence.

[LLama.SamplerGrammar] with a grammar that does not parse returns an empty
stage, which [Sampler.Add] ignores, so generation runs unconstrained. Add it
first in the chain, and check [Sampler.Len] after adding it.
[LLama.SamplerDRY] also returns an empty stage if llama.cpp cannot allocate
it.

A few mistakes still end the process from inside llama.cpp, and recover
cannot catch them:

  - [Sampler.Sample] on an output that did not ask for logits.
  - [Sampler.Sample] on a chain with no stage that picks a token, such as
    [SamplerDist] or [SamplerGreedy].
  - [Sampler.Sample] when a grammar stage placed after a truncation or
    picking stage refuses an end-of-generation token.

Some models' caches cannot shift positions: [LLama.MemoryCanShift] reports
false for M-RoPE models such as Qwen2-VL, Qwen3-VL and Qwen3.5. On those,
[LLama.MemorySeqAdd] and [LLama.MemorySeqDiv] do nothing, and a
[LLama.Predict] whose prompt and reply outgrow ContextParams().NCtxSeq
stops at the full context and returns what it generated.

Using a [LLama], [Batch] or [Sampler] after its Free is undefined and can
crash the process; only a second [LLama.Free] is safe. The [production]
guide describes the full failure model.

# Building and GPUs

Go's module download leaves out git submodules and never runs make, so build
the library from a clone first. You need Go 1.26 or newer, a C++17 compiler,
CMake and git, on Linux or macOS. Windows is not supported.

	git clone --recurse-submodules https://github.com/AshkanYarmoradi/go-llama.cpp
	cd go-llama.cpp
	make libbinding.a
	LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build ./...

The first build compiles all of llama.cpp and takes several minutes. To use
the checkout from your own module, point the import path at it:

	go mod edit -replace github.com/AshkanYarmoradi/go-llama.cpp=../go-llama.cpp

If you set LIBRARY_PATH and C_INCLUDE_PATH, point them at the checkout, not at
your module.

BUILD_TYPE picks the backend. For most backends, a build tag of the same name
links it into your program:

	BUILD_TYPE   backend                         go build tag
	(empty)      CPU; on macOS also Metal        none
	metal        Apple Metal                     none
	cublas       NVIDIA CUDA                     cublas
	hipblas      AMD ROCm/HIP                    hipblas
	vulkan       Vulkan (NVIDIA, AMD, Intel)     vulkan
	openblas     CPU with OpenBLAS               openblas
	blis         CPU with BLIS                   blis

For example, for an NVIDIA GPU:

	make BUILD_TYPE=cublas libbinding.a
	LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build -tags cublas ./...

Each backend needs its own toolkit on the build machine:

  - cublas: the CUDA toolkit. The link flags expect it in /usr/local/cuda.
    On a machine without a GPU, add -DCMAKE_CUDA_ARCHITECTURES=<arch> to
    CMAKE_ARGS, because llama.cpp otherwise compiles for the GPU it finds.
  - hipblas: ROCm 6.1 or newer, in /opt/rocm. ROCM_HOME points the build at
    another install, and GPU_TARGETS picks the AMD architectures to compile
    for.
  - vulkan: the Vulkan headers and loader, glslc and SPIRV-Headers
    (libvulkan-dev, glslc and spirv-headers on Debian and Ubuntu).
  - openblas and blis: the library and pkg-config, which the build uses to
    find the BLAS headers unless CMAKE_ARGS sets -DBLAS_INCLUDE_DIRS.

The tag's file carries the link flags, so you do not set CGO_LDFLAGS, except
to add a -L path for a CUDA or ROCm install outside those directories.
CMAKE_ARGS passes extra CMake options, from the environment or the make
command line; cublas, hipblas, vulkan, openblas and blis add their own options
either way. JOBS=n limits the parallel compile jobs, which helps a CUDA build
on a machine with little RAM.

Changing BUILD_TYPE rebuilds llama.cpp from scratch on its own, but changing
CMAKE_ARGS does not: run make clean first. For a CPU-only build on macOS, run
make clean, then CMAKE_ARGS=-DGGML_METAL=OFF make libbinding.a. An unknown
BUILD_TYPE stops make with an error, and so does clblas: CLBlast was removed
from llama.cpp, so use vulkan instead.

CI runs the test suite on the CPU on Linux and macOS, and with
BUILD_TYPE=metal on macOS. Its "GPU builds" workflow (jobs ubuntu-cuda-build
and ubuntu-vulkan-build) compiles and links the cublas and vulkan builds on
hosted runners that have no GPU, so nothing runs on a GPU there. The specs
labelled gpu run on an NVIDIA GPU only in the separate "GPU tests" workflow,
which needs a self-hosted runner. It starts when run by hand, when a pull
request carries the gpu label, and on pushes to main or a tag only once the
repository variable GPU_RUNNER is true. hipblas, openblas and blis pass
go vet under their build tags but are not built in CI.

Offloading layers is opt-in for each model, because [SetGPULayers] defaults
to 0. [SupportsGPUOffload] reports whether llama.cpp found a GPU it can use:

	opts := []llama.ModelOption{llama.SetContext(4096)}
	if llama.SupportsGPUOffload() {
		opts = append(opts, llama.SetGPULayers(99)) // 99 covers every layer of most models
	}
	model, err := llama.New("model.gguf", opts...)

While loading, llama.cpp logs how many layers it offloaded to the GPU. The
check can only run once the binary has started, and a GPU build needs its
backend's shared libraries for that: the CUDA libraries and the NVIDIA
driver's libcuda.so.1 for cublas, the ROCm libraries for hipblas, and the
Vulkan loader for vulkan. Without them the program does not start, so it
cannot fall back to the CPU.

With several GPUs, [SetTensorSplit] divides the model between them: a
comma-separated list of proportions, one per device, such as "3,1". A device
the list does not name gets nothing. Entries past [MaxDevices] are ignored,
and a list with an entry that is not a number is ignored as a whole, leaving
llama.cpp's default split; both are reported on stderr.

# Migrating from go-skynet

This package is a maintained fork of github.com/go-skynet/go-llama.cpp. [New]
and [LLama.Predict] keep their signatures and the package is still named
llama, so most programs change one import line:

	import llama "github.com/go-skynet/go-llama.cpp"      // before
	import llama "github.com/AshkanYarmoradi/go-llama.cpp" // after

Then rebuild libbinding.a from a fresh clone. Six functions are gone, together
with the option fields MulMatQ, Perplexity, NegativePrompt and
NegativePromptScale:

  - Eval: use [LLama.Tokenize], [NewBatch] and [LLama.Decode].
  - SpeculativeSampling: no built-in replacement.
  - SetMulMatQ: delete the call.
  - SetPerplexity: request logits for every token with [Batch.Add] and read
    them with [LLama.Logits], as the perplexity example does.
  - SetNegativePrompt and SetNegativePromptScale: no replacement.

Some code still compiles but behaves differently:

  - Predict starts from an empty KV cache on every call, and the prompt-cache
    options ([SetPathPromptCache], [EnablePromptCacheAll] and
    [EnablePromptCacheRO]) do nothing. Save and resume with
    [LLama.SaveSessionFile] and your own loop.
  - Options that no longer do anything are marked Deprecated, so staticcheck
    (check SA1019) flags every use: the three prompt-cache options above;
    the load options [SetModelSeed], [EnableF16Memory], [EnabelLowVRAM] and
    [SetLoraBase]; the Predict options [IgnoreEOS], [EnableF16KV],
    [SetMlock], [SetMemoryMap], [SetPredictionMainGPU],
    [SetPredictionTensorSplit], [SetNDraft], [SetTailFreeSamplingZ] and
    [SetPenalizeNL]; and the per-call [SetRopeFreqBase] and
    [SetRopeFreqScale], whose load-time counterparts are [WithRopeFreqBase]
    and [WithRopeFreqScale]. The option fields behind them are marked too.
  - The default sampler chain gained a min-p stage of 0.05. SetMinP(0)
    removes it.
  - RoPE base and scale now come from the model instead of 10000 and 1.0.
    WithRopeFreqBase(10000) and WithRopeFreqScale(1) restore go-skynet's
    values.

The [migration guide] covers every change.

[llama.cpp]: https://github.com/ggml-org/llama.cpp
[getting started]: https://github.com/AshkanYarmoradi/go-llama.cpp/blob/main/docs/getting-started.md
[cookbook]: https://github.com/AshkanYarmoradi/go-llama.cpp/blob/main/docs/cookbook.md
[production]: https://github.com/AshkanYarmoradi/go-llama.cpp/blob/main/docs/production.md
[how it works]: https://github.com/AshkanYarmoradi/go-llama.cpp/blob/main/docs/how-it-works.md
[migration guide]: https://github.com/AshkanYarmoradi/go-llama.cpp/blob/main/docs/migrating-from-go-skynet.md
*/
package llama
