package llama

// #cgo CXXFLAGS: -I${SRCDIR}/llama.cpp/common -I${SRCDIR}/llama.cpp/include -I${SRCDIR}/llama.cpp/ggml/include -I${SRCDIR}/llama.cpp -std=c++17
// #cgo LDFLAGS: -L${SRCDIR}/ -lbinding -lm -lstdc++
// #cgo linux LDFLAGS: -fopenmp
// #cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit
// #cgo darwin CXXFLAGS: -std=c++17
// #include "binding.h"
// #include <stdlib.h>
import "C"
import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

const (
	// bytesPerTokenEstimate is a generous upper bound on the UTF-8 length of a
	// single decoded token, used to size the Predict output buffer.
	bytesPerTokenEstimate = 8

	// maxPredictBytes caps that buffer so an unbounded token limit does not ask
	// for a huge allocation. Output beyond it is truncated; use
	// SetTokenCallback to stream longer generations.
	maxPredictBytes = 4 * 1024 * 1024
)

type LLama struct {
	state       unsafe.Pointer
	embeddings  bool
	contextSize int
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)
	modelPath := C.CString(model)
	defer C.free(unsafe.Pointer(modelPath))
	loraBase := C.CString(mo.LoraBase)
	defer C.free(unsafe.Pointer(loraBase))
	loraAdapter := C.CString(mo.LoraAdapter)
	defer C.free(unsafe.Pointer(loraAdapter))

	result := C.load_model(modelPath,
		C.int(mo.ContextSize), C.int(mo.Seed),
		C.bool(mo.F16Memory), C.bool(mo.MLock), C.bool(mo.Embeddings), C.bool(mo.MMap), C.bool(mo.LowVRAM),
		C.int(mo.NGPULayers), C.int(mo.NBatch), C.CString(mo.MainGPU), C.CString(mo.TensorSplit), C.bool(mo.NUMA),
		C.float(mo.FreqRopeBase), C.float(mo.FreqRopeScale),
		loraAdapter, loraBase,
	)

	if result == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	ll := &LLama{state: result, contextSize: mo.ContextSize, embeddings: mo.Embeddings}
	return ll, nil
}

func (l *LLama) Free() {
	C.llama_binding_free_model(l.state)
}

// ApplyLoRA loads a LoRA adapter from path and applies it to the context with
// the given scale. Adapters stack: each call adds to the active set. Adapters
// applied this way are released by ClearLoRA or when the model is freed.
func (l *LLama) ApplyLoRA(path string, scale float32) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if C.apply_lora_adapter(l.state, cPath, C.float(scale)) != 0 {
		return fmt.Errorf("failed to apply LoRA adapter %q", path)
	}
	return nil
}

// ClearLoRA detaches and frees every LoRA adapter previously applied with
// ApplyLoRA, returning the context to the base model weights.
func (l *LLama) ClearLoRA() {
	C.clear_lora_adapters(l.state)
}

// Batch accumulates tokens — each with a position, sequence IDs, and a flag for
// whether its output is wanted — for a Decode or Encode call. It wraps the
// engine's llama_batch; call Free when done.
type Batch struct {
	ptr      unsafe.Pointer
	capacity int
}

// NewBatch allocates a batch holding up to maxTokens tokens, each assignable to
// up to maxSeq sequences.
func NewBatch(maxTokens, maxSeq int) *Batch {
	return &Batch{
		ptr:      C.batch_init(C.int(maxTokens), C.int(maxSeq)),
		capacity: maxTokens,
	}
}

// Free releases the batch. It must not be used afterwards.
func (b *Batch) Free() { C.batch_free(b.ptr) }

// Reset empties the batch so it can be refilled without reallocating.
func (b *Batch) Reset() { C.batch_clear(b.ptr) }

// Len returns the number of tokens currently in the batch.
func (b *Batch) Len() int { return int(C.batch_n_tokens(b.ptr)) }

// Add appends token at position pos for the given sequence IDs (defaulting to
// sequence 0 when none are given). Set logits to request the model's output for
// this token. It returns an error if the batch is full or too many sequence IDs
// are supplied.
func (b *Batch) Add(token int32, pos int32, seqIDs []int32, logits bool) error {
	if len(seqIDs) == 0 {
		seqIDs = []int32{0}
	}
	ret := int(C.batch_add(b.ptr, C.int(token), C.int(pos),
		(*C.int)(unsafe.Pointer(&seqIDs[0])), C.int(len(seqIDs)), C.bool(logits)))
	switch ret {
	case -1:
		return fmt.Errorf("batch is full (capacity %d)", b.capacity)
	case -2:
		return fmt.Errorf("too many sequence IDs for one token (%d)", len(seqIDs))
	}
	return nil
}

// Decode runs the batch through the model using the KV cache. The return value
// is llama.cpp's decode status: 0 success, 1 = no KV slot (reduce the batch or
// grow the context), 2 = aborted, negative = error.
func (l *LLama) Decode(b *Batch) int {
	return int(C.decode_batch(l.state, b.ptr))
}

// Encode runs the batch through the encoder of an encoder-decoder model. It
// returns 0 on success, negative on error.
func (l *LLama) Encode(b *Batch) int {
	return int(C.encode_batch(l.state, b.ptr))
}

// Logits returns the vocabulary logits for the i-th output token of the last
// decoded batch; i = -1 selects the last token. It returns nil when no logits
// are available for that index (the token was added without requesting output).
func (l *LLama) Logits(i int) []float32 {
	n := int(C.get_model_n_vocab(l.state))
	if n <= 0 {
		return nil
	}
	out := make([]float32, n)
	got := int(C.get_logits_ith(l.state, C.int(i),
		(*C.float)(unsafe.Pointer(&out[0])), C.int(n)))
	if got <= 0 {
		return nil
	}
	return out[:got]
}

// TokenEmbedding returns the embedding vector for the i-th output token of the
// last decoded batch; i = -1 selects the last token. Returns nil if unavailable.
func (l *LLama) TokenEmbedding(i int) []float32 {
	return l.embeddingBuf(func(out []float32) int {
		return int(C.get_embeddings_ith(l.state, C.int(i),
			(*C.float)(unsafe.Pointer(&out[0])), C.int(len(out))))
	})
}

// SequenceEmbedding returns the pooled embedding for an entire sequence (for a
// context configured with pooled embeddings). Returns nil if unavailable.
func (l *LLama) SequenceEmbedding(seqID int32) []float32 {
	return l.embeddingBuf(func(out []float32) int {
		return int(C.get_embeddings_seq(l.state, C.int(seqID),
			(*C.float)(unsafe.Pointer(&out[0])), C.int(len(out))))
	})
}

func (l *LLama) embeddingBuf(fn func(out []float32) int) []float32 {
	n := int(C.get_model_n_embd(l.state))
	if n <= 0 {
		return nil
	}
	out := make([]float32, n)
	got := fn(out)
	if got <= 0 {
		return nil
	}
	return out[:got]
}

// MemoryClear drops the context's KV cache. When clearData is true the
// underlying data buffers are cleared too, not just the cell metadata.
func (l *LLama) MemoryClear(clearData bool) {
	C.memory_clear(l.state, C.bool(clearData))
}

// MemorySeqRemove removes tokens in [p0, p1) for sequence seqID from the KV
// cache. Pass p0 < 0 to start at 0 and p1 < 0 to run to the end; seqID < 0
// matches every sequence. It reports whether the removal succeeded.
func (l *LLama) MemorySeqRemove(seqID, p0, p1 int32) bool {
	return bool(C.memory_seq_rm(l.state, C.int(seqID), C.int(p0), C.int(p1)))
}

// MemorySeqCopy copies tokens in [p0, p1) from sequence src to dst in the KV
// cache — the shared-prefix trick for parallel sequences.
func (l *LLama) MemorySeqCopy(src, dst, p0, p1 int32) {
	C.memory_seq_cp(l.state, C.int(src), C.int(dst), C.int(p0), C.int(p1))
}

// MemorySeqKeep removes every sequence from the KV cache except seqID.
func (l *LLama) MemorySeqKeep(seqID int32) {
	C.memory_seq_keep(l.state, C.int(seqID))
}

// MemorySeqAdd shifts the positions of tokens in [p0, p1) of sequence seqID by
// delta. Pass p0 < 0 to start at 0 and p1 < 0 to run to the end. This is how a
// context is "slid" forward after evicting a prefix; check MemoryCanShift first,
// since not every cache type supports it.
func (l *LLama) MemorySeqAdd(seqID, p0, p1, delta int32) {
	C.memory_seq_add(l.state, C.int(seqID), C.int(p0), C.int(p1), C.int(delta))
}

// MemorySeqDiv integer-divides the positions of tokens in [p0, p1) of sequence
// seqID by d (which must be > 1) — the position-interpolation trick for
// stretching a context beyond its trained length.
func (l *LLama) MemorySeqDiv(seqID, p0, p1 int32, d int) {
	C.memory_seq_div(l.state, C.int(seqID), C.int(p0), C.int(p1), C.int(d))
}

// MemorySeqPosMin returns the smallest position held in the KV cache for
// seqID, or -1 if the sequence is empty. It is non-zero only for caches that
// evict from the front, such as sliding-window attention.
func (l *LLama) MemorySeqPosMin(seqID int32) int32 {
	return int32(C.memory_seq_pos_min(l.state, C.int(seqID)))
}

// MemorySeqPosMax returns the largest position held in the KV cache for seqID,
// or -1 if the sequence is empty. Every position in
// [MemorySeqPosMin, MemorySeqPosMax] is guaranteed to be present.
func (l *LLama) MemorySeqPosMax(seqID int32) int32 {
	return int32(C.memory_seq_pos_max(l.state, C.int(seqID)))
}

// MemoryCanShift reports whether the context's KV cache supports position
// shifting via MemorySeqAdd.
func (l *LLama) MemoryCanShift() bool {
	return bool(C.memory_can_shift(l.state))
}

// PoolingType identifies how a context pools token embeddings into a single
// sequence embedding.
type PoolingType int

// Pooling strategies, mirroring llama_pooling_type.
const (
	PoolingUnspecified PoolingType = -1
	PoolingNone        PoolingType = 0
	PoolingMean        PoolingType = 1
	PoolingCLS         PoolingType = 2
	PoolingLast        PoolingType = 3
	PoolingRank        PoolingType = 4
)

// String returns the llama.cpp name of the pooling strategy.
func (p PoolingType) String() string {
	switch p {
	case PoolingUnspecified:
		return "unspecified"
	case PoolingNone:
		return "none"
	case PoolingMean:
		return "mean"
	case PoolingCLS:
		return "cls"
	case PoolingLast:
		return "last"
	case PoolingRank:
		return "rank"
	default:
		return fmt.Sprintf("PoolingType(%d)", int(p))
	}
}

// ContextParams reports the geometry the context actually runs with. The engine
// may clamp or round what New was asked for, so prefer these over the values in
// ModelOptions when sizing batches or picking sequence ids.
type ContextParams struct {
	// NCtx is the total context size across all sequences.
	NCtx int
	// NCtxSeq is the per-sequence context size.
	NCtxSeq int
	// NBatch is the maximum number of tokens a single Decode may submit.
	NBatch int
	// NUbatch is the physical micro-batch size the graph is built for.
	NUbatch int
	// NSeqMax is the number of sequences the KV cache can hold, so valid
	// sequence ids are [0, NSeqMax).
	NSeqMax int
	// NRSSeq is the number of recurrent-state sequences, for models with a
	// recurrent (Mamba-style) memory.
	NRSSeq int
	// Pooling is the embedding pooling strategy.
	Pooling PoolingType
}

// ContextParams returns the geometry of the loaded context.
func (l *LLama) ContextParams() ContextParams {
	return ContextParams{
		NCtx:    int(C.context_n_ctx(l.state)),
		NCtxSeq: int(C.context_n_ctx_seq(l.state)),
		NBatch:  int(C.context_n_batch(l.state)),
		NUbatch: int(C.context_n_ubatch(l.state)),
		NSeqMax: int(C.context_n_seq_max(l.state)),
		NRSSeq:  int(C.context_n_rs_seq(l.state)),
		Pooling: PoolingType(C.context_pooling_type(l.state)),
	}
}

// Threads returns the thread counts the context currently uses: nThreads for
// single-token generation and nThreadsBatch for prompt and batch processing.
func (l *LLama) Threads() (nThreads, nThreadsBatch int) {
	return int(C.context_n_threads(l.state)), int(C.context_n_threads_batch(l.state))
}

// SetThreads changes the thread counts used for generation (nThreads) and for
// prompt/batch processing (nThreadsBatch). It takes effect on the next decode.
func (l *LLama) SetThreads(nThreads, nThreadsBatch int) {
	C.context_set_n_threads(l.state, C.int(nThreads), C.int(nThreadsBatch))
}

// SetEmbeddings switches the context between producing logits and producing
// embeddings. It also updates what Embeddings reports, so a model loaded with
// EnableEmbeddings can be flipped back to generation and vice versa.
func (l *LLama) SetEmbeddings(enabled bool) {
	C.context_set_embeddings(l.state, C.bool(enabled))
}

// SetCausalAttn selects causal (each token attends only to the past) or
// non-causal attention. Encoder-style embedding models want non-causal.
func (l *LLama) SetCausalAttn(causal bool) {
	C.context_set_causal_attn(l.state, C.bool(causal))
}

// Synchronize blocks until all queued computation on the context has finished.
// The output accessors do this for you; it is only needed when timing manually.
func (l *LLama) Synchronize() {
	C.context_synchronize(l.state)
}

// Perf holds llama.cpp's context-level performance counters.
//
// The engine skips its own timing calls unless asked; the binding enables
// them when it creates the context, so these are populated rather than zero.
//
// PromptTokens and EvalTokens have a floor of 1: the engine clamps them so its
// own reporting can divide by them without guarding against zero. A context
// that has decoded nothing, or one just reset by PerfReset, therefore reports 1
// rather than 0, and the two cases cannot be told apart. Use the timings to
// decide whether any work has actually happened.
type Perf struct {
	// StartMS is the absolute start time in milliseconds.
	StartMS float64
	// LoadMS is the time spent loading the model.
	LoadMS float64
	// PromptEvalMS is the time spent processing prompt tokens.
	PromptEvalMS float64
	// EvalMS is the time spent generating tokens.
	EvalMS float64
	// PromptTokens is the number of prompt tokens processed, floored at 1.
	PromptTokens int
	// EvalTokens is the number of tokens generated, floored at 1.
	EvalTokens int
	// GraphsReused is the number of times a compute graph was reused.
	GraphsReused int
}

// Perf returns the context's performance counters, accumulated since load or
// since the last PerfReset.
func (l *LLama) Perf() Perf {
	var tStart, tLoad, tPEval, tEval C.double
	var nPEval, nEval, nReused C.int
	C.perf_context(l.state, &tStart, &tLoad, &tPEval, &tEval, &nPEval, &nEval, &nReused)
	return Perf{
		StartMS:      float64(tStart),
		LoadMS:       float64(tLoad),
		PromptEvalMS: float64(tPEval),
		EvalMS:       float64(tEval),
		PromptTokens: int(nPEval),
		EvalTokens:   int(nEval),
		GraphsReused: int(nReused),
	}
}

// PerfReset restarts the context's performance counters. Timings go back to
// zero; the token counts return to their floor of 1, not 0 (see Perf).
func (l *LLama) PerfReset() { C.perf_context_reset(l.state) }

// SamplerPerf holds a sampler chain's performance counters.
type SamplerPerf struct {
	// SampleMS is the time spent sampling, in milliseconds.
	SampleMS float64
	// Samples is the number of tokens sampled.
	Samples int
}

// Perf returns the chain's sampling counters. It reports zeroes for a stage
// that was not created by NewSamplerChain.
func (s *Sampler) Perf() SamplerPerf {
	var tSample C.double
	var nSample C.int
	C.perf_sampler(s.ptr, &tSample, &nSample)
	return SamplerPerf{SampleMS: float64(tSample), Samples: int(nSample)}
}

// PerfReset zeroes the chain's sampling counters.
func (s *Sampler) PerfReset() { C.perf_sampler_reset(s.ptr) }

// Version returns the version string of the linked llama.cpp library.
func Version() string { return C.GoString(C.llama_version_str()) }

// TimeUS returns llama.cpp's monotonic clock in microseconds. It shares a time
// base with the values in Perf, so it is the right clock for measuring spans
// that are compared against them.
func TimeUS() int64 { return int64(C.llama_time_us_val()) }

// Sampler is a single sampling stage or a chain of stages, wrapping llama.cpp's
// sampler API. Construct stages (SamplerTopK, SamplerTemp, ...), add them to a
// chain from NewSamplerChain, then Sample from a decoded context.
type Sampler struct {
	ptr unsafe.Pointer
}

// NewSamplerChain creates an empty sampler chain. The chain takes ownership of
// every stage added to it and frees them all when the chain's Free is called.
func NewSamplerChain() *Sampler { return &Sampler{ptr: C.sampler_chain_init()} }

// Add appends a stage to the chain, transferring ownership of the stage to the
// chain. Do not call Free on a stage after adding it. A nil or empty stage is
// ignored (e.g. an invalid grammar).
func (s *Sampler) Add(stage *Sampler) {
	if stage == nil || stage.ptr == nil {
		return
	}
	C.sampler_chain_add(s.ptr, stage.ptr)
}

// Free releases the sampler (and, for a chain, every stage added to it).
func (s *Sampler) Free() { C.sampler_free(s.ptr) }

// Reset clears any internal sampler state (penalty history, grammar position…).
func (s *Sampler) Reset() { C.sampler_reset(s.ptr) }

// Accept informs stateful stages that token was chosen.
func (s *Sampler) Accept(token int32) { C.sampler_accept(s.ptr, C.int(token)) }

// Sample selects a token from the logits of the idx-th output of the last
// Decode/Predict on model (idx = -1 selects the last token).
func (s *Sampler) Sample(model *LLama, idx int) int32 {
	return int32(C.sampler_sample(model.state, s.ptr, C.int(idx)))
}

// Sampler stages. min_keep floors how many candidates a truncation stage keeps.
func SamplerGreedy() *Sampler          { return &Sampler{ptr: C.sampler_init_greedy()} }
func SamplerDist(seed uint32) *Sampler { return &Sampler{ptr: C.sampler_init_dist(C.uint(seed))} }
func SamplerTopK(k int) *Sampler       { return &Sampler{ptr: C.sampler_init_top_k(C.int(k))} }
func SamplerTopP(p float32, minKeep int) *Sampler {
	return &Sampler{ptr: C.sampler_init_top_p(C.float(p), C.int(minKeep))}
}
func SamplerMinP(p float32, minKeep int) *Sampler {
	return &Sampler{ptr: C.sampler_init_min_p(C.float(p), C.int(minKeep))}
}
func SamplerTypical(p float32, minKeep int) *Sampler {
	return &Sampler{ptr: C.sampler_init_typical(C.float(p), C.int(minKeep))}
}
func SamplerTemp(t float32) *Sampler { return &Sampler{ptr: C.sampler_init_temp(C.float(t))} }
func SamplerTempExt(t, delta, exponent float32) *Sampler {
	return &Sampler{ptr: C.sampler_init_temp_ext(C.float(t), C.float(delta), C.float(exponent))}
}
func SamplerXTC(p, t float32, minKeep int, seed uint32) *Sampler {
	return &Sampler{ptr: C.sampler_init_xtc(C.float(p), C.float(t), C.int(minKeep), C.uint(seed))}
}
func SamplerTopNSigma(n float32) *Sampler {
	return &Sampler{ptr: C.sampler_init_top_n_sigma(C.float(n))}
}
func SamplerMirostatV2(seed uint32, tau, eta float32) *Sampler {
	return &Sampler{ptr: C.sampler_init_mirostat_v2(C.uint(seed), C.float(tau), C.float(eta))}
}

// SamplerPenalties builds a repetition/frequency/presence penalty stage. lastN
// is how many recent tokens to consider (0 disables the stage); repeat, freq
// and present are the repetition, frequency and presence penalties, where 1.0,
// 0.0 and 0.0 respectively mean "disabled". It is a method on LLama because the
// engine sizes the stage from the model's vocabulary.
func (l *LLama) SamplerPenalties(lastN int, repeat, freq, present float32) *Sampler {
	return &Sampler{ptr: C.sampler_init_penalties(l.state, C.int(lastN), C.float(repeat), C.float(freq), C.float(present))}
}

// SamplerGrammar builds a GBNF grammar-constrained stage from the model's vocab.
// The returned stage is empty (ignored by Add) if the grammar fails to parse.
func (l *LLama) SamplerGrammar(grammar, root string) *Sampler {
	cG := C.CString(grammar)
	defer C.free(unsafe.Pointer(cG))
	cR := C.CString(root)
	defer C.free(unsafe.Pointer(cR))
	return &Sampler{ptr: C.sampler_init_grammar(l.state, cG, cR)}
}

// SamplerDRY builds a DRY ("Don't Repeat Yourself") stage from the model's vocab.
func (l *LLama) SamplerDRY(multiplier, base float32, allowedLength, penaltyLastN int) *Sampler {
	return &Sampler{ptr: C.sampler_init_dry(l.state, C.float(multiplier), C.float(base),
		C.int(allowedLength), C.int(penaltyLastN))}
}

// ModelInfo contains information about the loaded model
type ModelInfo struct {
	VocabSize          int
	ContextLength      int
	EmbeddingSize      int
	LayerCount         int
	HeadCount          int
	HeadCountKV        int
	SlidingWindow      int // n_swa; 0 when the model does not use sliding-window attention
	RopeFreqScaleTrain float32
	ModelSize          int64
	ParamCount         int64
	Description        string
}

// GetModelInfo returns information about the loaded model
func (l *LLama) GetModelInfo() ModelInfo {
	descBuf := make([]byte, 256)
	C.get_model_description(l.state, (*C.char)(unsafe.Pointer(&descBuf[0])), C.int(len(descBuf)))

	return ModelInfo{
		VocabSize:          int(C.get_model_n_vocab(l.state)),
		ContextLength:      int(C.get_model_n_ctx_train(l.state)),
		EmbeddingSize:      int(C.get_model_n_embd(l.state)),
		LayerCount:         int(C.get_model_n_layer(l.state)),
		HeadCount:          int(C.get_model_n_head(l.state)),
		HeadCountKV:        int(C.get_model_n_head_kv(l.state)),
		SlidingWindow:      int(C.get_model_n_swa(l.state)),
		RopeFreqScaleTrain: float32(C.get_model_rope_freq_scale_train(l.state)),
		ModelSize:          int64(C.get_model_size(l.state)),
		ParamCount:         int64(C.get_model_n_params(l.state)),
		Description:        string(descBuf[:cStrLen(descBuf)]),
	}
}

// ModelMetadata returns every key-value entry stored in the GGUF model header
// (architecture, tokenizer settings, quantization, and so on).
func (l *LLama) ModelMetadata() map[string]string {
	n := int(C.get_model_meta_count(l.state))
	meta := make(map[string]string, n)
	for i := 0; i < n; i++ {
		key, ok := grow(func(buf []byte) int {
			return int(C.get_model_meta_key_by_index(l.state, C.int(i),
				(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		})
		if !ok {
			continue
		}
		val, ok := grow(func(buf []byte) int {
			return int(C.get_model_meta_val_str_by_index(l.state, C.int(i),
				(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		})
		if !ok {
			continue
		}
		meta[key] = val
	}
	return meta
}

// ModelMetadataValue returns the metadata value for a single key. The boolean
// is false when the key is not present in the model header.
func (l *LLama) ModelMetadataValue(key string) (string, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	return grow(func(buf []byte) int {
		return int(C.get_model_meta_val_str(l.state, cKey,
			(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	})
}

// grow calls fn with an increasingly large buffer until the written value fits.
// It follows the snprintf contract used by the llama.cpp string accessors: fn
// returns the length that would be written (a value >= len(buf) means the
// output was truncated), or a negative value when the entry is absent.
func grow(fn func(buf []byte) int) (string, bool) {
	buf := make([]byte, 512)
	n := fn(buf)
	if n < 0 {
		return "", false
	}
	if n >= len(buf) {
		buf = make([]byte, n+1)
		if n = fn(buf); n < 0 {
			return "", false
		}
	}
	return string(buf[:n]), true
}

// ChatMessage is a single turn in a chat, as consumed by ApplyChatTemplate.
// Role is conventionally "system", "user" or "assistant"; which roles a given
// template understands is up to that template.
type ChatMessage struct {
	Role    string
	Content string
}

// ErrNoChatTemplate is returned when a chat template cannot be applied: either
// the model carries no template of its own and none was supplied, or the
// template is not one llama.cpp knows how to render.
var ErrNoChatTemplate = errors.New("llama: no usable chat template")

// ApplyChatTemplate renders messages into a single prompt string using tmpl.
// Pass an empty tmpl to use the template stored in the model's GGUF metadata.
//
// When addAssistant is true the result ends with the token(s) that open an
// assistant turn, which is what you want before generating a reply.
//
// Note that llama.cpp does not run a full Jinja engine here: it recognises a
// fixed set of well-known templates. A model whose template is not on that list
// returns ErrNoChatTemplate, and the caller should format the prompt itself.
func (l *LLama) ApplyChatTemplate(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	var cTmpl *C.char
	if tmpl != "" {
		cTmpl = C.CString(tmpl)
		defer C.free(unsafe.Pointer(cTmpl))
	}

	// Two parallel C arrays of char*, one allocation per string, all freed on
	// the way out. Keeping them parallel avoids needing a JSON round-trip.
	n := len(messages)
	var roles, contents **C.char
	if n > 0 {
		roleSlice := make([]*C.char, n)
		contentSlice := make([]*C.char, n)
		defer func() {
			for i := 0; i < n; i++ {
				C.free(unsafe.Pointer(roleSlice[i]))
				C.free(unsafe.Pointer(contentSlice[i]))
			}
		}()
		for i, m := range messages {
			roleSlice[i] = C.CString(m.Role)
			contentSlice[i] = C.CString(m.Content)
		}
		roles = (**C.char)(unsafe.Pointer(&roleSlice[0]))
		contents = (**C.char)(unsafe.Pointer(&contentSlice[0]))
	}

	// Size the first attempt from the input, then retry once at the exact
	// length the engine reports if the guess was short.
	size := chatBufferHint(messages)
	for attempt := 0; attempt < 2; attempt++ {
		buf := make([]byte, size)
		ret := int(C.apply_chat_template(l.state, cTmpl, roles, contents, C.int(n),
			C.bool(addAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int(size)))
		switch {
		case ret < 0:
			return "", ErrNoChatTemplate
		case ret <= size:
			return string(buf[:ret]), nil
		}
		size = ret + 1
	}
	return "", fmt.Errorf("llama: chat template output kept growing past %d bytes", size)
}

// chatBufferHint is llama.cpp's own recommendation: twice the total message
// length, with a floor that comfortably covers the template's own boilerplate.
func chatBufferHint(messages []ChatMessage) int {
	total := 0
	for _, m := range messages {
		total += len(m.Role) + len(m.Content)
	}
	if size := 2 * total; size > 1024 {
		return size
	}
	return 1024
}

// BuiltinChatTemplates returns the names of the chat templates built into
// llama.cpp. Any of these can be passed as the tmpl argument to
// ApplyChatTemplate.
func BuiltinChatTemplates() []string {
	n := int(C.chat_builtin_template_count())
	if n <= 0 {
		return nil
	}
	names := make([]string, 0, n)
	buf := make([]byte, 256)
	for i := 0; i < n; i++ {
		ret := int(C.chat_builtin_template_name(C.int(i),
			(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		if ret <= 0 {
			continue
		}
		if ret > len(buf) {
			buf = make([]byte, ret+1)
			ret = int(C.chat_builtin_template_name(C.int(i),
				(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
			if ret <= 0 {
				continue
			}
		}
		names = append(names, string(buf[:ret]))
	}
	return names
}

// GetChatTemplate returns the model's chat template as stored in its GGUF
// metadata. Pass an empty name for the default template, or a name to select
// one of several (e.g. "rag", "tool_use"). It returns "" when the model has no
// such template.
func (l *LLama) GetChatTemplate(name string) string {
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}

	// Templates regularly exceed a naive fixed buffer, so grow to the length
	// the binding reports rather than returning a silently truncated template.
	buf := make([]byte, 4096)
	ret := int(C.get_model_chat_template(l.state, cName, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	if ret <= 0 {
		return ""
	}
	if ret > len(buf) {
		buf = make([]byte, ret+1)
		ret = int(C.get_model_chat_template(l.state, cName, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		if ret <= 0 || ret > len(buf) {
			return ""
		}
	}
	return string(buf[:ret])
}

// cStrLen returns the length of a null-terminated C string in a byte slice
func cStrLen(b []byte) int {
	for i, v := range b {
		if v == 0 {
			return i
		}
	}
	return len(b)
}

// SpecialTokens contains the special token IDs for the model's vocabulary.
// A field is -1 (LLAMA_TOKEN_NULL) when the vocabulary does not define it.
type SpecialTokens struct {
	BOS    int32 // Beginning of sentence
	EOS    int32 // End of sentence
	EOT    int32 // End of turn
	NL     int32 // Newline
	SEP    int32 // Separator
	PAD    int32 // Padding
	MASK   int32 // Mask
	FIMPre int32 // Fill-in-the-middle prefix
	FIMSuf int32 // Fill-in-the-middle suffix
	FIMMid int32 // Fill-in-the-middle middle
	FIMPad int32 // Fill-in-the-middle padding
	FIMRep int32 // Fill-in-the-middle repository separator
	FIMSep int32 // Fill-in-the-middle separator
}

// GetSpecialTokens returns the special token IDs for the model
func (l *LLama) GetSpecialTokens() SpecialTokens {
	return SpecialTokens{
		BOS:    int32(C.get_vocab_bos(l.state)),
		EOS:    int32(C.get_vocab_eos(l.state)),
		EOT:    int32(C.get_vocab_eot(l.state)),
		NL:     int32(C.get_vocab_nl(l.state)),
		SEP:    int32(C.get_vocab_sep(l.state)),
		PAD:    int32(C.get_vocab_pad(l.state)),
		MASK:   int32(C.get_vocab_mask(l.state)),
		FIMPre: int32(C.get_vocab_fim_pre(l.state)),
		FIMSuf: int32(C.get_vocab_fim_suf(l.state)),
		FIMMid: int32(C.get_vocab_fim_mid(l.state)),
		FIMPad: int32(C.get_vocab_fim_pad(l.state)),
		FIMRep: int32(C.get_vocab_fim_rep(l.state)),
		FIMSep: int32(C.get_vocab_fim_sep(l.state)),
	}
}

// GetVocabAddBOS returns whether the model automatically adds BOS token
func (l *LLama) GetVocabAddBOS() bool {
	return bool(C.get_vocab_add_bos(l.state))
}

// GetVocabAddEOS returns whether the model automatically adds EOS token
func (l *LLama) GetVocabAddEOS() bool {
	return bool(C.get_vocab_add_eos(l.state))
}

// Backend capability queries. They reflect how the underlying llama.cpp library
// was compiled and can be called without a loaded model.

// SupportsMmap reports whether memory-mapped model loading is available.
func SupportsMmap() bool { return bool(C.backend_supports_mmap()) }

// SupportsMlock reports whether locking the model into RAM is available.
func SupportsMlock() bool { return bool(C.backend_supports_mlock()) }

// SupportsGPUOffload reports whether offloading layers to a GPU is available.
func SupportsGPUOffload() bool { return bool(C.backend_supports_gpu_offload()) }

// SupportsRPC reports whether the RPC backend is available.
func SupportsRPC() bool { return bool(C.backend_supports_rpc()) }

// MaxDevices returns the maximum number of devices the backend can address.
func MaxDevices() int { return int(C.backend_max_devices()) }

// MaxParallelSequences returns the maximum number of sequences that can be
// decoded in parallel.
func MaxParallelSequences() int { return int(C.backend_max_parallel_sequences()) }

// ModelHasEncoder returns whether the model has an encoder component
func (l *LLama) ModelHasEncoder() bool {
	return bool(C.model_has_encoder(l.state))
}

// ModelHasDecoder returns whether the model has a decoder component
func (l *LLama) ModelHasDecoder() bool {
	return bool(C.model_has_decoder(l.state))
}

// ModelIsRecurrent returns whether the model uses a recurrent architecture (e.g., Mamba, RWKV)
func (l *LLama) ModelIsRecurrent() bool {
	return bool(C.model_is_recurrent(l.state))
}

// SystemInfo returns a string with system information relevant to llama.cpp
func SystemInfo() string {
	buf := make([]byte, 4096)
	ret := C.get_system_info((*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	if ret <= 0 {
		return ""
	}
	return string(buf[:ret])
}

func (l *LLama) LoadState(state string) error {
	d := C.CString(state)
	w := C.CString("rb")
	result := C.load_state(l.state, d, w)

	defer C.free(unsafe.Pointer(d)) // free allocated C string
	defer C.free(unsafe.Pointer(w)) // free allocated C string

	if result != 0 {
		return fmt.Errorf("error while loading state")
	}

	return nil
}

func (l *LLama) SaveState(dst string) error {
	d := C.CString(dst)
	w := C.CString("wb")

	C.save_state(l.state, d, w)

	defer C.free(unsafe.Pointer(d)) // free allocated C string
	defer C.free(unsafe.Pointer(w)) // free allocated C string

	_, err := os.Stat(dst)
	return err
}

// Token Embeddings
func (l *LLama) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	outSize := po.Tokens
	if po.Tokens == 0 {
		outSize = 9999999
	}

	floats := make([]float32, outSize)

	myArray := (*C.int)(C.malloc(C.size_t(len(tokens)) * C.sizeof_int))

	// Copy the values from the Go slice to the C array
	for i, v := range tokens {
		(*[1<<31 - 1]int32)(unsafe.Pointer(myArray))[i] = int32(v)
	}

	params := C.llama_allocate_params(C.CString(""), C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.MinP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), nil, C.int(0),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale),
		C.int(po.NDraft),
		C.float(po.XTCProbability), C.float(po.XTCThreshold),
		C.float(po.DRYMultiplier), C.float(po.DRYBase), C.int(po.DRYAllowedLength), C.int(po.DRYPenaltyLastN),
		C.float(po.TopNSigma),
	)
	ret := C.get_token_embeddings(params, l.state, myArray, C.int(len(tokens)), (*C.float)(&floats[0]))
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}
	return floats, nil
}

// Embeddings
func (l *LLama) Embeddings(text string, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	floats := make([]float32, po.Tokens)
	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.MinP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale),
		C.int(po.NDraft),
		C.float(po.XTCProbability), C.float(po.XTCThreshold),
		C.float(po.DRYMultiplier), C.float(po.DRYBase), C.int(po.DRYAllowedLength), C.int(po.DRYPenaltyLastN),
		C.float(po.TopNSigma),
	)

	ret := C.get_embeddings(params, l.state, (*C.float)(&floats[0]))
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}

	return floats, nil
}

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(l.state, po.TokenCallback)
	}

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	// A token decodes to several bytes, so the output buffer has to be larger
	// than the token count. The C side truncates to the size we pass, so this
	// is a hard bound rather than a guess that could overrun.
	outSize := po.Tokens*bytesPerTokenEstimate + len(text) + 1024
	if outSize > maxPredictBytes || outSize < 0 {
		outSize = maxPredictBytes
	}
	out := make([]byte, outSize)

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.MinP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale),
		C.int(po.NDraft),
		C.float(po.XTCProbability), C.float(po.XTCThreshold),
		C.float(po.DRYMultiplier), C.float(po.DRYBase), C.int(po.DRYAllowedLength), C.int(po.DRYPenaltyLastN),
		C.float(po.TopNSigma),
	)
	ret := C.llama_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])), C.int(len(out)), C.bool(po.DebugMode))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")

	for _, s := range po.StopPrompts {
		res = strings.TrimRight(res, s)
	}

	C.llama_free_params(params)

	if po.TokenCallback != nil {
		setCallback(l.state, nil)
	}

	return res, nil
}

// tokenize has an interesting return property: negative lengths (potentially) have meaning.
// Therefore, return the length seperate from the slice and error - all three can be used together
func (l *LLama) TokenizeString(text string, opts ...PredictOption) (int32, []int32, error) {
	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 4096 // ???
	}
	out := make([]C.int, po.Tokens)

	var fakeDblPtr **C.char

	// copy pasted and modified minimally. Should I simplify down / do we need an "allocate defaults"
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.MinP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), fakeDblPtr, C.int(0),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale),
		C.int(po.NDraft),
		C.float(po.XTCProbability), C.float(po.XTCThreshold),
		C.float(po.DRYMultiplier), C.float(po.DRYBase), C.int(po.DRYAllowedLength), C.int(po.DRYPenaltyLastN),
		C.float(po.TopNSigma),
	)

	tokRet := C.llama_tokenize_string(params, l.state, (*C.int)(unsafe.Pointer(&out[0]))) //, C.int(po.Tokens), true)

	if tokRet < 0 {
		return int32(tokRet), []int32{}, fmt.Errorf("llama_tokenize_string returned negative count %d", tokRet)
	}

	// TODO: Is this loop still required to unbox cgo to go?
	gTokRet := int32(tokRet)

	gLenOut := min(len(out), int(gTokRet))

	goSlice := make([]int32, gLenOut)
	for i := 0; i < gLenOut; i++ {
		goSlice[i] = int32(out[i])
	}

	return gTokRet, goSlice, nil
}

// Tokenize converts text into token IDs. addSpecial controls whether the
// model's configured special tokens (such as BOS) are prepended/appended;
// parseSpecial controls whether special-token markup in the text is parsed into
// single tokens rather than treated as literal characters. Unlike
// TokenizeString it is bounds-safe and does not allocate a sampling params
// struct.
func (l *LLama) Tokenize(text string, addSpecial, parseSpecial bool) []int32 {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// A token never spans fewer than one byte, so len(text) plus a small margin
	// for added special tokens is a safe upper bound on the count. The negative
	// return path below still handles any underestimate.
	tokens := make([]int32, len(text)+8)
	call := func() int {
		return int(C.tokenize_text(l.state, cText, C.int(len(text)),
			(*C.int)(unsafe.Pointer(&tokens[0])), C.int(len(tokens)),
			C.bool(addSpecial), C.bool(parseSpecial)))
	}
	n := call()
	if n < 0 {
		tokens = make([]int32, -n)
		if n = call(); n < 0 {
			return nil
		}
	}
	return tokens[:n]
}

// Detokenize converts a sequence of token IDs back into text. removeSpecial
// drops leading BOS-style tokens; unparseSpecial renders special tokens as their
// text form instead of an empty string.
func (l *LLama) Detokenize(tokens []int32, removeSpecial, unparseSpecial bool) string {
	if len(tokens) == 0 {
		return ""
	}
	return growNeg(func(buf []byte) int {
		return int(C.detokenize_text(l.state,
			(*C.int)(unsafe.Pointer(&tokens[0])), C.int(len(tokens)),
			(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)),
			C.bool(removeSpecial), C.bool(unparseSpecial)))
	}, len(tokens)*4+16)
}

// TokenToPiece returns the text fragment a single token decodes to. When
// special is true, control and special tokens render to their text form.
func (l *LLama) TokenToPiece(token int32, special bool) string {
	return growNeg(func(buf []byte) int {
		return int(C.token_to_piece_str(l.state, C.int(token),
			(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)), C.bool(special)))
	}, 64)
}

// growNeg calls fn with a growing byte buffer following the llama.cpp tokenizer
// contract: fn returns the number of bytes written, or the negative of the
// required size when the buffer is too small. initial sizes the first attempt.
func growNeg(fn func(buf []byte) int, initial int) string {
	if initial < 1 {
		initial = 1
	}
	buf := make([]byte, initial)
	n := fn(buf)
	if n < 0 {
		buf = make([]byte, -n)
		n = fn(buf)
	}
	if n < 0 {
		return ""
	}
	return string(buf[:n])
}

// CGo only allows us to use static calls from C to Go, we can't just dynamically pass in func's.
// This is the next best thing, we register the callbacks in this map and call tokenCallback from
// the C code. We also attach a finalizer to LLama, so it will unregister the callback when the
// garbage collection frees it.

// SetTokenCallback registers a callback for the individual tokens created when running Predict. It
// will be called once for each token. The callback shall return true as long as the model should
// continue predicting the next token. When the callback returns false the predictor will return.
// The tokens are just converted into Go strings, they are not trimmed or otherwise changed. Also
// the tokens may not be valid UTF-8.
// Pass in nil to remove a callback.
//
// It is save to call this method while a prediction is running.
func (l *LLama) SetTokenCallback(callback func(token string) bool) {
	setCallback(l.state, callback)
}

var (
	m         sync.RWMutex
	callbacks = map[uintptr]func(string) bool{}
)

//export tokenCallback
func tokenCallback(statePtr unsafe.Pointer, token *C.char) bool {
	m.RLock()
	defer m.RUnlock()

	if callback, ok := callbacks[uintptr(statePtr)]; ok {
		return callback(C.GoString(token))
	}

	return true
}

// setCallback can be used to register a token callback for LLama. Pass in a nil callback to
// remove the callback.
func setCallback(statePtr unsafe.Pointer, callback func(string) bool) {
	m.Lock()
	defer m.Unlock()

	if callback == nil {
		delete(callbacks, uintptr(statePtr))
	} else {
		callbacks[uintptr(statePtr)] = callback
	}
}
