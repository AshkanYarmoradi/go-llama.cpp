/*
Package llama provides Go bindings for [llama.cpp], the C++ inference engine
for GGUF models.

It exposes two layers. The high-level layer runs a whole generation in one
call and is what most programs want. The low-level layer maps onto llama.cpp's
own primitives — batches, decoding, the KV cache, sampler chains — for programs
that need to drive inference themselves, such as servers juggling several
conversations in one context.

# Building

The binding links against a static library built from the vendored llama.cpp
submodule, so it is not a `go get`-only package:

	git clone --recurse-submodules https://github.com/AshkanYarmoradi/go-llama.cpp
	cd go-llama.cpp
	make libbinding.a

Then build your program with the library on the search path:

	LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build ./...

See the README for GPU backends (CUDA, ROCm, Metal, OpenCL, OpenBLAS).

# Generating text

Load a model, generate, free it. [LLama.Free] releases the model and context;
without it the memory lives until the process exits.

	model, err := llama.New("model.gguf",
		llama.SetContext(4096),
		llama.SetGPULayers(35),
	)
	if err != nil {
		return err
	}
	defer model.Free()

	out, err := model.Predict("Explain quantum computing:",
		llama.SetTemperature(0.7),
		llama.SetTokens(256),
	)

[LLama.SetTokenCallback] streams tokens as they are produced; returning false
from the callback stops generation early.

# Chat models

[LLama.ApplyChatTemplate] renders a conversation with the template stored in
the model's own GGUF metadata, so you do not have to hardcode one model's
prompt format:

	prompt, err := model.ApplyChatTemplate("", []llama.ChatMessage{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "How much is 2+2?"},
	}, true)

llama.cpp recognises a fixed set of templates rather than running a full Jinja
engine, so a model whose template it cannot place returns [ErrNoChatTemplate].

# Driving inference directly

For anything Predict cannot express — several conversations sharing one
context, speculative decoding, custom stopping rules — build a batch, decode
it, and sample:

	chain := llama.NewSamplerChain()
	defer chain.Free()
	chain.Add(llama.SamplerTopK(40))
	chain.Add(llama.SamplerTopP(0.95, 1))
	chain.Add(model.SamplerPenalties(64, 1.1, 0, 0))
	chain.Add(llama.SamplerTemp(0.8))
	chain.Add(llama.SamplerDist(seed))

	batch := llama.NewBatch(len(tokens), 1)
	defer batch.Free()
	for i, tok := range tokens {
		batch.Add(tok, int32(i), []int32{0}, i == len(tokens)-1)
	}
	if model.Decode(batch) != 0 {
		return errors.New("decode failed")
	}

	next := chain.Sample(model, -1)
	chain.Accept(next)

A sampler chain owns every stage added to it: [Sampler.Free] on the chain frees
them all, and a stage must not be freed separately after [Sampler.Add].

Stop on [LLama.IsEOG] rather than comparing against EOS. Many models define
several end-of-turn tokens, and IsEOG asks the vocabulary about all of them.

# Sequences and the KV cache

A context holds several sequences, each an independent conversation identified
by a sequence id in [0, ContextParams.NSeqMax). The Memory methods manage them:
[LLama.MemorySeqRemove] evicts a range, [LLama.MemorySeqCopy] shares a prefix
between sequences, and [LLama.MemorySeqAdd] shifts positions.

Context shifting — making room by dropping the oldest tokens — is those two
together:

	if model.MemoryCanShift() {
		model.MemorySeqRemove(0, 0, nDiscard)
		model.MemorySeqAdd(0, nDiscard, -1, -nDiscard)
	}

[LLama.SequenceStateData] checkpoints one sequence without serializing the
others, and [LLama.SetSequenceStateData] restores it into any sequence id.

# Embeddings

Load with [EnableEmbeddings] and call [LLama.Embeddings]. A context configured
for embeddings does not produce usable logits, so Predict on it returns
garbage; load the model twice if you need both, or flip the context with
[LLama.SetEmbeddings].

# Concurrency

A [LLama] value is not safe for concurrent use. Each context has one KV cache
and one output buffer, so concurrent Predict or Decode calls on the same model
corrupt each other. Serialize access with a mutex, or give each goroutine its
own context. Distinct sequence ids within one context isolate conversations,
not callers — they still need serializing.

# Engine coverage

The binding tracks llama.cpp's C API. Functions llama.cpp marks deprecated are
deliberately not exposed; use the replacement the engine points at.

[llama.cpp]: https://github.com/ggerganov/llama.cpp
*/
package llama
