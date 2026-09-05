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

// LoRACount returns how many adapters are currently applied to the context.
// Adapters are indexed by application order, which is the order ApplyLoRA was
// called in since the last ClearLoRA.
func (l *LLama) LoRACount() int {
	return int(C.lora_adapter_count(l.state))
}

// LoRAMetadata returns the GGUF key-value header of the i-th applied adapter —
// the rank, alpha, and whatever the training tool recorded. It returns nil for
// an index outside the applied set.
func (l *LLama) LoRAMetadata(i int) map[string]string {
	n := int(C.lora_adapter_meta_count(l.state, C.int(i)))
	if n <= 0 {
		return nil
	}
	out := make(map[string]string, n)
	for j := 0; j < n; j++ {
		key := adapterString(func(buf []byte) int {
			return int(C.lora_adapter_meta_key_by_index(l.state, C.int(i), C.int(j),
				(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		})
		if key == "" {
			continue
		}
		out[key] = adapterString(func(buf []byte) int {
			return int(C.lora_adapter_meta_val_str_by_index(l.state, C.int(i), C.int(j),
				(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		})
	}
	return out
}

// LoRAMetadataValue returns one metadata value from the i-th applied adapter.
// The second result reports whether the key was present.
func (l *LLama) LoRAMetadataValue(i int, key string) (string, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var found bool
	v := adapterString(func(buf []byte) int {
		ret := int(C.lora_adapter_meta_val_str(l.state, C.int(i), cKey,
			(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		found = ret >= 0
		return ret
	})
	return v, found
}

// LoRAInvocationTokens returns the invocation tokens of the i-th applied
// adapter when it is an activated LoRA (aLoRA) — one that stays dormant until
// the model emits that exact token sequence, and applies from then on.
//
// It returns nil for a plain LoRA, which has no invocation sequence, and for an
// index outside the applied set.
func (l *LLama) LoRAInvocationTokens(i int) []int32 {
	n := int(C.lora_adapter_alora_tokens(l.state, C.int(i), nil, 0))
	if n == 0 || n == -1 {
		return nil
	}
	if n < 0 {
		n = -n
	}
	out := make([]int32, n)
	got := int(C.lora_adapter_alora_tokens(l.state, C.int(i),
		(*C.int)(unsafe.Pointer(&out[0])), C.int(n)))
	if got <= 0 {
		return nil
	}
	return out[:got]
}

// SetControlVector applies a control (steering) vector to the context: a
// direction added to the residual stream of layers ilStart through ilEnd
// inclusive, which nudges generation toward or away from some behaviour
// without retraining anything.
//
// data is nEmbd x nLayers floats laid out starting from layer 1, so len(data)
// must be a multiple of nEmbd. Pass nil to clear the active vector.
func (l *LLama) SetControlVector(data []float32, nEmbd, ilStart, ilEnd int) error {
	if len(data) == 0 {
		if ret := int(C.set_control_vector(l.state, nil, 0, 0, 0, 0)); ret != 0 {
			return fmt.Errorf("llama: failed to clear the control vector (%d)", ret)
		}
		return nil
	}
	if nEmbd <= 0 || len(data)%nEmbd != 0 {
		return fmt.Errorf("llama: control vector length %d is not a multiple of n_embd %d", len(data), nEmbd)
	}
	ret := int(C.set_control_vector(l.state,
		(*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)),
		C.int(nEmbd), C.int(ilStart), C.int(ilEnd)))
	if ret != 0 {
		return fmt.Errorf("llama: failed to apply the control vector (%d)", ret)
	}
	return nil
}

// ClearControlVector removes any control vector applied to the context.
func (l *LLama) ClearControlVector() error {
	return l.SetControlVector(nil, 0, 0, 0)
}

// adapterString runs fn against a buffer, growing once if fn reports it needs
// more room (snprintf semantics). It returns "" when fn reports an error.
func adapterString(fn func(buf []byte) int) string {
	buf := make([]byte, 256)
	ret := fn(buf)
	if ret < 0 {
		return ""
	}
	if ret > len(buf) {
		buf = make([]byte, ret+1)
		ret = fn(buf)
		if ret < 0 || ret > len(buf) {
			return ""
		}
	}
	return string(buf[:ret])
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

// SetSequenceSampler attaches a sampler chain to a sequence so the backend
// samples inside the compute graph, instead of copying a full vocabulary of
// logits back to host memory for the CPU to sample from. After a Decode, read
// the result with SampledToken rather than Logits plus Sampler.Sample.
//
// chain must be a chain from NewSamplerChain, not a bare stage. The caller
// keeps ownership: the chain must stay alive, and unfreed, for as long as it
// is attached to the context.
//
// This is marked experimental upstream. It reports whether the engine accepted
// the chain; a context that was not built for backend sampling returns false,
// and CPU sampling through Sampler.Sample keeps working.
func (l *LLama) SetSequenceSampler(seqID int32, chain *Sampler) bool {
	if chain == nil || chain.ptr == nil {
		return false
	}
	return bool(C.set_sequence_sampler(l.state, C.int(seqID), chain.ptr))
}

// SampledToken returns the token the backend sampled for the i-th output of
// the last Decode; i = -1 selects the last. It returns -1 when the backend
// sampled nothing for that index, which is the normal result if no sampler is
// attached to the sequence.
//
// Reading a token does not advance the sampler's state — that happens when the
// token is accepted. With multiple outputs, accept a contiguous prefix in
// output order.
func (l *LLama) SampledToken(i int) int32 {
	return int32(C.get_sampled_token(l.state, C.int(i)))
}

// SampledCandidates returns the token ids the backend sampler kept for the
// i-th output. These are what map an index in SampledProbs or SampledLogits
// back to a vocabulary token: candidates[k] is the token whose probability is
// probs[k]. It returns nil when the backend sampled nothing.
func (l *LLama) SampledCandidates(i int) []int32 {
	n := int(C.get_sampled_candidates(l.state, C.int(i), nil, 0))
	if n <= 0 {
		return nil
	}
	out := make([]int32, n)
	got := int(C.get_sampled_candidates(l.state, C.int(i),
		(*C.int)(unsafe.Pointer(&out[0])), C.int(n)))
	if got <= 0 {
		return nil
	}
	return out[:got]
}

// SampledProbs returns the probabilities the backend sampler produced for the
// i-th output, aligned with SampledCandidates. It returns nil when the backend
// produced none — a chain without a distribution stage yields no probabilities.
func (l *LLama) SampledProbs(i int) []float32 {
	return sampledFloats(func(out []float32) int {
		var p *C.float
		if len(out) > 0 {
			p = (*C.float)(unsafe.Pointer(&out[0]))
		}
		return int(C.get_sampled_probs(l.state, C.int(i), p, C.int(len(out))))
	})
}

// SampledLogits returns the logits the backend sampler kept for the i-th
// output, aligned with SampledCandidates. Unlike Logits, which returns the
// whole vocabulary, this is only the candidates that survived the chain. It
// returns nil when the backend kept none.
func (l *LLama) SampledLogits(i int) []float32 {
	return sampledFloats(func(out []float32) int {
		var p *C.float
		if len(out) > 0 {
			p = (*C.float)(unsafe.Pointer(&out[0]))
		}
		return int(C.get_sampled_logits(l.state, C.int(i), p, C.int(len(out))))
	})
}

// sampledFloats probes fn for the available count, then fills a buffer of
// exactly that size.
func sampledFloats(fn func(out []float32) int) []float32 {
	n := fn(nil)
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

// LogLevel is the severity of a llama.cpp log record.
type LogLevel int

// Log severities, mirroring ggml_log_level.
const (
	LogLevelNone  LogLevel = 0
	LogLevelDebug LogLevel = 1
	LogLevelInfo  LogLevel = 2
	LogLevelWarn  LogLevel = 3
	LogLevelError LogLevel = 4
	// LogLevelCont continues the previous record rather than starting a new
	// one. llama.cpp uses it to build a line in pieces, so a handler that
	// prefixes each record with a timestamp or level should skip the prefix
	// for these.
	LogLevelCont LogLevel = 5
)

// String returns a short uppercase name for the level.
func (l LogLevel) String() string {
	switch l {
	case LogLevelNone:
		return "NONE"
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	case LogLevelCont:
		return "CONT"
	default:
		return fmt.Sprintf("LogLevel(%d)", int(l))
	}
}

var (
	logMu      sync.RWMutex
	logHandler func(LogLevel, string)
)

// SetLogHandler routes llama.cpp's log output to fn instead of stderr. The
// engine is chatty — model loading alone produces dozens of lines — so this is
// how that output gets into a real logger, filtered by level, or silenced.
//
// Pass nil to restore llama.cpp's own stderr output.
//
// The text arrives exactly as the engine emits it, including its trailing
// newline, and a record may be a fragment: see LogLevelCont.
//
// The handler is called from whatever thread the engine is running on,
// including during New and Predict, so it must be safe for concurrent use and
// should not call back into this package.
//
// llama.cpp's logger state is global, so this affects every model in the
// process, not just one.
func SetLogHandler(fn func(level LogLevel, text string)) {
	logMu.Lock()
	defer logMu.Unlock()

	logHandler = fn
	C.set_log_callback(C.bool(fn != nil))
}

// LogHandlerInstalled reports whether llama.cpp is currently routing its log
// output through this package.
//
// It asks the engine rather than trusting local state, so it also returns
// false when other code in the process has taken the logger over — llama.cpp's
// logger is global, and the last caller to set it wins.
//
// Note that "not installed" does not mean "no logging": SetLogHandler(nil)
// restores llama.cpp's own stderr output rather than silencing it. To silence
// the engine, install a handler that discards.
func LogHandlerInstalled() bool {
	return bool(C.has_log_callback())
}

//export goLogCallback
func goLogCallback(level C.int, text *C.char) {
	logMu.RLock()
	fn := logHandler
	logMu.RUnlock()

	if fn == nil {
		return
	}
	fn(LogLevel(level), C.GoString(text))
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

// SamplerInfill builds a fill-in-the-middle stage. It belongs after the
// truncation stages (top-k, top-p): it merges candidates that share a prefix
// and prefers an end-of-generation token once the hole is filled, which is what
// keeps FIM generation from running past the insertion point.
func (l *LLama) SamplerInfill() *Sampler {
	return &Sampler{ptr: C.sampler_init_infill(l.state)}
}

// SamplerAdaptiveP builds an adaptive-p stage, which steers sampling toward
// tokens near the target probability instead of using a fixed cutoff. target is
// in [0, 1] (negative disables); decay is the EMA factor in [0, 0.99], giving a
// history of roughly 1/(1-decay) tokens.
//
// llama.cpp recommends running it with min-p as the only other truncation stage
// in the chain — stacking it behind top-k and top-p defeats the adaptation.
func SamplerAdaptiveP(target, decay float32, seed uint32) *Sampler {
	return &Sampler{ptr: C.sampler_init_adaptive_p(C.float(target), C.float(decay), C.uint(seed))}
}

// LogitBias shifts the logit of one token before sampling. A large negative
// bias effectively bans a token; a large positive one forces it.
type LogitBias struct {
	Token int32
	Bias  float32
}

// SamplerLogitBias builds a stage that applies the given biases. Put it first
// in a chain so the adjustment is visible to every stage after it, including
// greedy selection.
func (l *LLama) SamplerLogitBias(biases []LogitBias) *Sampler {
	if len(biases) == 0 {
		return &Sampler{}
	}
	tokens := make([]int32, len(biases))
	values := make([]float32, len(biases))
	for i, b := range biases {
		tokens[i] = b.Token
		values[i] = b.Bias
	}
	return &Sampler{ptr: C.sampler_init_logit_bias(l.state, C.int(len(biases)),
		(*C.int)(unsafe.Pointer(&tokens[0])),
		(*C.float)(unsafe.Pointer(&values[0])))}
}

// SamplerGrammarLazy builds a grammar stage that stays inactive until the
// output matches one of triggerPatterns (regular expressions) or produces one
// of triggerTokens, and constrains everything from that point on.
//
// This is how a tool-call grammar is applied only once the model has actually
// started emitting a call, leaving ordinary prose unconstrained.
func (l *LLama) SamplerGrammarLazy(grammar, root string, triggerPatterns []string, triggerTokens []int32) *Sampler {
	cG := C.CString(grammar)
	defer C.free(unsafe.Pointer(cG))
	cR := C.CString(root)
	defer C.free(unsafe.Pointer(cR))

	var patterns **C.char
	if len(triggerPatterns) > 0 {
		cPatterns := make([]*C.char, len(triggerPatterns))
		defer func() {
			for _, p := range cPatterns {
				C.free(unsafe.Pointer(p))
			}
		}()
		for i, p := range triggerPatterns {
			cPatterns[i] = C.CString(p)
		}
		patterns = (**C.char)(unsafe.Pointer(&cPatterns[0]))
	}

	var tokens *C.int
	if len(triggerTokens) > 0 {
		tokens = (*C.int)(unsafe.Pointer(&triggerTokens[0]))
	}

	return &Sampler{ptr: C.sampler_init_grammar_lazy(l.state, cG, cR,
		patterns, C.int(len(triggerPatterns)),
		tokens, C.int(len(triggerTokens)))}
}

// Len returns the number of stages in a chain, or 0 for a single stage.
func (s *Sampler) Len() int { return int(C.sampler_chain_n(s.ptr)) }

// At returns the i-th stage of a chain without transferring ownership: the
// returned Sampler must not be freed, and becomes invalid when the chain is.
// It returns nil if i is out of range.
func (s *Sampler) At(i int) *Sampler {
	if i < 0 || i >= s.Len() {
		return nil
	}
	p := C.sampler_chain_get(s.ptr, C.int(i))
	if p == nil {
		return nil
	}
	return &Sampler{ptr: p}
}

// Remove detaches the i-th stage from a chain and transfers ownership of it to
// the caller, who must Free it. It returns nil if i is out of range.
func (s *Sampler) Remove(i int) *Sampler {
	if i < 0 || i >= s.Len() {
		return nil
	}
	p := C.sampler_chain_remove(s.ptr, C.int(i))
	if p == nil {
		return nil
	}
	return &Sampler{ptr: p}
}

// Name returns llama.cpp's name for the stage, such as "top-k" or "dist"; a
// chain is named "chain".
//
// The engine decorates the name to describe the stage's state, so match on a
// substring rather than equality:
//
//	"?top-k"   the stage was built with parameters that disable it
//	"+top-k"   it has been initialised and runs on the backend
//	"-top-k"   it has been initialised but the backend cannot run it
//
// The bare name is what you see before the chain has sampled anything.
func (s *Sampler) Name() string {
	buf := make([]byte, 64)
	ret := int(C.sampler_name(s.ptr, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	if ret <= 0 || ret > len(buf) {
		return ""
	}
	return string(buf[:ret])
}

// Clone returns an independent copy of the sampler, including any accumulated
// state, so a speculative branch can be explored without disturbing the
// original. Cloning a chain clones every stage in it. The copy is owned by the
// caller and must be Freed.
//
// Every stage this package can build implements cloning, so this cannot fail
// for them; it returns nil only for an empty Sampler.
func (s *Sampler) Clone() *Sampler {
	p := C.sampler_clone(s.ptr)
	if p == nil {
		return nil
	}
	return &Sampler{ptr: p}
}

// Seed returns the RNG seed a stage was built with, or DefaultSeed for a stage
// that does not use randomness.
func (s *Sampler) Seed() uint32 { return uint32(C.sampler_get_seed(s.ptr)) }

// DefaultSeed is llama.cpp's LLAMA_DEFAULT_SEED: the value Sampler.Seed reports
// for a stage with no seed of its own, and the value that asks the engine to
// pick a random one.
const DefaultSeed uint32 = 0xFFFFFFFF

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

// VocabType identifies the tokenizer family a model's vocabulary uses.
type VocabType int

// Tokenizer families, mirroring llama_vocab_type.
const (
	VocabNone   VocabType = 0 // model carries no vocabulary
	VocabSPM    VocabType = 1 // SentencePiece: byte-level BPE with byte fallback
	VocabBPE    VocabType = 2 // GPT-2 style byte-level BPE
	VocabWPM    VocabType = 3 // BERT WordPiece
	VocabUGM    VocabType = 4 // T5 Unigram
	VocabRWKV   VocabType = 5 // RWKV greedy tokenization
	VocabPLaMo2 VocabType = 6 // PLaMo-2 Aho-Corasick
)

// String returns the llama.cpp name of the tokenizer family.
func (v VocabType) String() string {
	switch v {
	case VocabNone:
		return "none"
	case VocabSPM:
		return "spm"
	case VocabBPE:
		return "bpe"
	case VocabWPM:
		return "wpm"
	case VocabUGM:
		return "ugm"
	case VocabRWKV:
		return "rwkv"
	case VocabPLaMo2:
		return "plamo2"
	default:
		return fmt.Sprintf("VocabType(%d)", int(v))
	}
}

// TokenAttr is a bitmask describing how a token behaves during tokenization.
type TokenAttr int

// Token attributes, mirroring llama_token_attr. Test them with Has.
const (
	TokenAttrUndefined   TokenAttr = 0
	TokenAttrUnknown     TokenAttr = 1 << 0
	TokenAttrUnused      TokenAttr = 1 << 1
	TokenAttrNormal      TokenAttr = 1 << 2
	TokenAttrControl     TokenAttr = 1 << 3
	TokenAttrUserDefined TokenAttr = 1 << 4
	TokenAttrByte        TokenAttr = 1 << 5
	TokenAttrNormalized  TokenAttr = 1 << 6
	TokenAttrLStrip      TokenAttr = 1 << 7
	TokenAttrRStrip      TokenAttr = 1 << 8
	TokenAttrSingleWord  TokenAttr = 1 << 9
)

// Has reports whether every bit of attr is set.
func (a TokenAttr) Has(attr TokenAttr) bool { return a&attr == attr }

// String renders the set bits as a pipe-joined list.
func (a TokenAttr) String() string {
	if a == TokenAttrUndefined {
		return "undefined"
	}
	names := []struct {
		bit  TokenAttr
		name string
	}{
		{TokenAttrUnknown, "unknown"},
		{TokenAttrUnused, "unused"},
		{TokenAttrNormal, "normal"},
		{TokenAttrControl, "control"},
		{TokenAttrUserDefined, "user_defined"},
		{TokenAttrByte, "byte"},
		{TokenAttrNormalized, "normalized"},
		{TokenAttrLStrip, "lstrip"},
		{TokenAttrRStrip, "rstrip"},
		{TokenAttrSingleWord, "single_word"},
	}
	var set []string
	for _, n := range names {
		if a&n.bit != 0 {
			set = append(set, n.name)
		}
	}
	if len(set) == 0 {
		return fmt.Sprintf("TokenAttr(%d)", int(a))
	}
	return strings.Join(set, "|")
}

// VocabType returns the tokenizer family of the model's vocabulary.
func (l *LLama) VocabType() VocabType {
	return VocabType(C.get_vocab_type(l.state))
}

// TokenText returns the raw vocabulary entry for token: the piece exactly as
// the tokenizer stores it, which for SPM models still carries the U+2581 word
// boundary marker and for BPE models the byte-level escapes. It is the right
// thing for inspecting a vocabulary and the wrong thing for building output:
// use TokenToPiece for text you intend to concatenate.
//
// It returns "" for a token outside the vocabulary.
func (l *LLama) TokenText(token int32) string {
	buf := make([]byte, 256)
	ret := int(C.get_vocab_token_text(l.state, C.int(token),
		(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	if ret <= 0 {
		return ""
	}
	if ret > len(buf) {
		buf = make([]byte, ret+1)
		ret = int(C.get_vocab_token_text(l.state, C.int(token),
			(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
		if ret <= 0 || ret > len(buf) {
			return ""
		}
	}
	return string(buf[:ret])
}

// TokenScore returns the vocabulary score for token, used by SPM and UGM
// tokenizers to choose between competing merges. It is 0 for tokenizers that
// do not use scores and for a token outside the vocabulary.
func (l *LLama) TokenScore(token int32) float32 {
	return float32(C.get_vocab_token_score(l.state, C.int(token)))
}

// TokenAttr returns the attribute bitmask for token, or TokenAttrUndefined for
// a token outside the vocabulary.
func (l *LLama) TokenAttr(token int32) TokenAttr {
	return TokenAttr(C.get_vocab_token_attr(l.state, C.int(token)))
}

// IsEOG reports whether token ends generation. This covers every end-of-turn
// and end-of-sequence token the model defines, not only EOS, and is what a
// generation loop should actually test against.
func (l *LLama) IsEOG(token int32) bool {
	return bool(C.vocab_token_is_eog(l.state, C.int(token)))
}

// IsControlToken reports whether token is a control token rather than text.
func (l *LLama) IsControlToken(token int32) bool {
	return bool(C.vocab_token_is_control(l.state, C.int(token)))
}

// AddSeparator reports whether the tokenizer inserts a separator token: the
// SEP counterpart to the BOS and EOS flags in SpecialTokens.
func (l *LLama) AddSeparator() bool {
	return bool(C.get_vocab_add_sep(l.state))
}

// SuppressTokens returns the tokens the model's vocabulary marks as never to
// be generated. It returns nil when the vocabulary suppresses nothing.
func (l *LLama) SuppressTokens() []int32 {
	n := int(C.get_vocab_suppress_tokens(l.state, nil, 0))
	if n == 0 {
		return nil
	}
	if n < 0 {
		n = -n
	}
	out := make([]int32, n)
	got := int(C.get_vocab_suppress_tokens(l.state,
		(*C.int)(unsafe.Pointer(&out[0])), C.int(n)))
	if got <= 0 {
		return nil
	}
	return out[:got]
}

// RopeType identifies the rotary position embedding variant a model uses.
type RopeType int

// RoPE variants, mirroring llama_rope_type. The values past NORM come from
// ggml and are deliberately not a contiguous range.
const (
	RopeNone   RopeType = -1
	RopeNorm   RopeType = 0
	RopeNeox   RopeType = 2
	RopeMrope  RopeType = 8
	RopeVision RopeType = 24
	RopeImrope RopeType = 40
)

// String returns a short name for the RoPE variant.
func (r RopeType) String() string {
	switch r {
	case RopeNone:
		return "none"
	case RopeNorm:
		return "norm"
	case RopeNeox:
		return "neox"
	case RopeMrope:
		return "mrope"
	case RopeVision:
		return "vision"
	case RopeImrope:
		return "imrope"
	default:
		return fmt.Sprintf("RopeType(%d)", int(r))
	}
}

// Architecture describes structural properties of the loaded model that decide
// how it has to be driven: which graph halves exist, how positions are encoded,
// and how it was quantized.
type Architecture struct {
	// RopeType is the rotary position embedding variant.
	RopeType RopeType
	// FileType is the llama_ftype the model was quantized to.
	FileType int
	// FileTypeName is the engine's name for FileType, e.g. "Q4_K - Medium".
	FileTypeName string
	// HasEncoder and HasDecoder report which halves of an encoder-decoder
	// model are present.
	HasEncoder bool
	HasDecoder bool
	// DecoderStartToken opens decoding on an encoder-decoder model, or is -1
	// when the model defines none.
	DecoderStartToken int32
	// IsRecurrent is true for recurrent (Mamba-style) models, which keep a
	// rolling state instead of a growing KV cache.
	IsRecurrent bool
	// IsHybrid is true for models that mix attention and recurrent layers.
	IsHybrid bool
	// IsDiffusion is true for diffusion language models, which do not
	// generate strictly left to right.
	IsDiffusion bool
	// EmbdInp and EmbdOut are the input and output embedding widths. They
	// differ from ModelInfo.EmbeddingSize on models with a projection.
	EmbdInp int
	EmbdOut int
	// LayersNextN is the number of multi-token-prediction layers, 0 when the
	// model has none.
	LayersNextN int
	// ClassifierLabels names the outputs of a classifier head, nil for a
	// model without one.
	ClassifierLabels []string
}

// Architecture returns the structural properties of the loaded model.
func (l *LLama) Architecture() Architecture {
	a := Architecture{
		RopeType:          RopeType(C.get_model_rope_type(l.state)),
		FileType:          int(C.get_model_ftype(l.state)),
		HasEncoder:        bool(C.model_has_encoder(l.state)),
		HasDecoder:        bool(C.model_has_decoder(l.state)),
		DecoderStartToken: int32(C.get_model_decoder_start_token(l.state)),
		IsRecurrent:       bool(C.model_is_recurrent(l.state)),
		IsHybrid:          bool(C.model_is_hybrid(l.state)),
		IsDiffusion:       bool(C.model_is_diffusion(l.state)),
		EmbdInp:           int(C.get_model_n_embd_inp(l.state)),
		EmbdOut:           int(C.get_model_n_embd_out(l.state)),
		LayersNextN:       int(C.get_model_n_layer_nextn(l.state)),
	}
	a.FileTypeName = FileTypeName(a.FileType)

	if n := int(C.get_model_n_cls_out(l.state)); n > 0 {
		buf := make([]byte, 128)
		for i := 0; i < n; i++ {
			ret := int(C.get_model_cls_label(l.state, C.int(i),
				(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
			if ret <= 0 || ret > len(buf) {
				continue
			}
			a.ClassifierLabels = append(a.ClassifierLabels, string(buf[:ret]))
		}
	}
	return a
}

// FileTypeName returns llama.cpp's name for a llama_ftype value, for example
// "Q4_K - Medium". It returns "" for an unrecognised value.
func FileTypeName(ftype int) string {
	buf := make([]byte, 128)
	ret := int(C.ftype_name(C.int(ftype), (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	if ret <= 0 || ret > len(buf) {
		return ""
	}
	return string(buf[:ret])
}

// FlashAttnTypeName returns llama.cpp's name for a llama_flash_attn_type value.
func FlashAttnTypeName(t int) string {
	buf := make([]byte, 128)
	ret := int(C.flash_attn_type_name(C.int(t), (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	if ret <= 0 || ret > len(buf) {
		return ""
	}
	return string(buf[:ret])
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

// StateSize returns the number of bytes StateData will produce for the whole
// context, including every sequence in the KV cache.
func (l *LLama) StateSize() int64 {
	return int64(C.state_get_size(l.state))
}

// StateData serializes the entire context — KV cache, RNG, logits — into a
// byte slice that SetStateData can restore into a context created with the
// same model and the same geometry.
//
// It is the in-memory counterpart of SaveState. Prefer SequenceStateData when
// you only need one conversation, since whole-context state grows with the
// full KV cache.
func (l *LLama) StateData() ([]byte, error) {
	return stateBuf(func(buf []byte) int64 {
		var p *C.uchar
		if len(buf) > 0 {
			p = (*C.uchar)(unsafe.Pointer(&buf[0]))
		}
		return int64(C.state_get_data(l.state, p, C.longlong(len(buf))))
	})
}

// SetStateData restores a context previously serialized by StateData. The
// context must have been created from the same model with the same geometry;
// restoring mismatched state is not detected and will misbehave.
func (l *LLama) SetStateData(data []byte) error {
	if len(data) == 0 {
		return errors.New("llama: empty state data")
	}
	n := int64(C.state_set_data(l.state, (*C.uchar)(unsafe.Pointer(&data[0])), C.longlong(len(data))))
	if n == 0 {
		return errors.New("llama: failed to restore context state")
	}
	return nil
}

// SaveSessionFile writes the context state plus tokens to path, in llama.cpp's
// session file format. Unlike SaveState the token list travels with the state,
// so another process can resume a conversation it did not start.
func (l *LLama) SaveSessionFile(path string, tokens []int32) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var p *C.int
	if len(tokens) > 0 {
		p = (*C.int)(unsafe.Pointer(&tokens[0]))
	}
	if !bool(C.state_save_file(l.state, cPath, p, C.int(len(tokens)))) {
		return fmt.Errorf("llama: failed to write session file %q", path)
	}
	return nil
}

// LoadSessionFile restores a session written by SaveSessionFile and returns
// the tokens stored alongside it.
//
// The engine rejects a session whose token count exceeds the buffer rather
// than truncating, so the buffer is sized from the context: a session can
// never hold more tokens than the context it was captured from.
func (l *LLama) LoadSessionFile(path string) ([]int32, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	capacity := int(C.context_n_ctx(l.state))
	if capacity <= 0 {
		capacity = l.contextSize
	}
	tokens := make([]int32, capacity)
	n := int(C.state_load_file(l.state, cPath, (*C.int)(unsafe.Pointer(&tokens[0])), C.int(capacity)))
	if n < 0 {
		return nil, fmt.Errorf("llama: failed to read session file %q", path)
	}
	return tokens[:n], nil
}

// SequenceStateSize returns the number of bytes SequenceStateData will produce
// for a single sequence.
func (l *LLama) SequenceStateSize(seqID int32) int64 {
	return int64(C.state_seq_get_size(l.state, C.int(seqID)))
}

// SequenceStateData serializes just the KV-cache state of one sequence. This
// is the checkpoint a server wants: it captures a single conversation slot
// without dragging along every other sequence in the context.
func (l *LLama) SequenceStateData(seqID int32) ([]byte, error) {
	return stateBuf(func(buf []byte) int64 {
		var p *C.uchar
		if len(buf) > 0 {
			p = (*C.uchar)(unsafe.Pointer(&buf[0]))
		}
		return int64(C.state_seq_get_data(l.state, p, C.longlong(len(buf)), C.int(seqID)))
	})
}

// SetSequenceStateData restores sequence state produced by SequenceStateData
// into destSeqID, which need not be the sequence it was captured from — that
// is what makes it usable for moving a conversation between context slots.
func (l *LLama) SetSequenceStateData(data []byte, destSeqID int32) error {
	if len(data) == 0 {
		return errors.New("llama: empty sequence state data")
	}
	n := int64(C.state_seq_set_data(l.state, (*C.uchar)(unsafe.Pointer(&data[0])),
		C.longlong(len(data)), C.int(destSeqID)))
	if n == 0 {
		return fmt.Errorf("llama: failed to restore state into sequence %d", destSeqID)
	}
	return nil
}

// SaveSequenceFile writes one sequence's state and its tokens to path.
func (l *LLama) SaveSequenceFile(path string, seqID int32, tokens []int32) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var p *C.int
	if len(tokens) > 0 {
		p = (*C.int)(unsafe.Pointer(&tokens[0]))
	}
	if !bool(C.state_seq_save_file(l.state, cPath, C.int(seqID), p, C.int(len(tokens)))) {
		return fmt.Errorf("llama: failed to write sequence file %q", path)
	}
	return nil
}

// LoadSequenceFile restores a file written by SaveSequenceFile into destSeqID
// and returns the tokens stored with it. The file's token count is probed
// first, so the buffer is always exactly the right size.
func (l *LLama) LoadSequenceFile(path string, destSeqID int32) ([]int32, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	n := int(C.state_seq_file_token_count(l.state, cPath))
	if n < 0 {
		return nil, fmt.Errorf("llama: failed to read sequence file %q", path)
	}

	// A zero-token file still carries state, so the load must happen either
	// way; give the engine a non-nil pointer to write into regardless.
	tokens := make([]int32, n+1)
	got := int(C.state_seq_load_file(l.state, cPath, C.int(destSeqID),
		(*C.int)(unsafe.Pointer(&tokens[0])), C.int(n)))
	if got < 0 {
		return nil, fmt.Errorf("llama: failed to load sequence file %q", path)
	}
	return tokens[:got], nil
}

// stateBuf runs fn against a buffer sized by fn's own report. fn returns the
// bytes written, or the negative of the size it needs; the first call probes
// with an empty buffer, so exactly one allocation of the right size happens.
func stateBuf(fn func(buf []byte) int64) ([]byte, error) {
	need := fn(nil)
	if need == 0 {
		return nil, errors.New("llama: state is empty")
	}
	if need < 0 {
		need = -need
	}
	buf := make([]byte, need)
	got := fn(buf)
	if got <= 0 {
		return nil, errors.New("llama: failed to serialize state")
	}
	return buf[:got], nil
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
