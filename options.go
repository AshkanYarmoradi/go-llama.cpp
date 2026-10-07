package llama

// ModelOptions is the load-time configuration New and NewFromSplits apply.
// Build it with ModelOption values rather than by hand.
type ModelOptions struct {
	// ContextSize is the context size in tokens; 0 means the model's training
	// context. The engine rounds it up to a multiple of 256.
	ContextSize int
	// Deprecated: has no effect. Seed a sampler instead (SetSeed,
	// SamplerDist).
	Seed int
	// NBatch is the most tokens one decode may submit. A generative model
	// caps it at the context size.
	NBatch int
	// Deprecated: has no effect.
	F16Memory bool
	MLock     bool
	MMap      bool
	// Deprecated: has no effect.
	LowVRAM     bool
	Embeddings  bool
	NUMA        bool
	NGPULayers  int
	MainGPU     string
	TensorSplit string
	// FreqRopeBase and FreqRopeScale override the RoPE base frequency and
	// scaling factor. 0, the default, uses the values the model was trained
	// with.
	FreqRopeBase  float32
	FreqRopeScale float32
	// Deprecated: has no effect.
	LoraBase    string
	LoraAdapter string
	// NSeqMax is the number of distinct sequences the context can hold; 0
	// keeps llama.cpp's default of 1.
	NSeqMax int
}

// PredictOptions is the per-call configuration Predict applies. Build it with
// PredictOption values rather than by hand.
type PredictOptions struct {
	Seed int
	// Threads is the thread count for this call only; 0 keeps the context's
	// setting (see (*LLama).SetThreads).
	Threads                            int
	Tokens, TopK, Repeat, Batch, NKeep int
	TopP, MinP, Temperature, Penalty   float32
	// Deprecated: has no effect.
	NDraft int
	// Deprecated: has no effect.
	F16KV       bool
	DebugMode   bool
	StopPrompts []string
	// Deprecated: has no effect. Predict always stops at an end-of-generation
	// token.
	IgnoreEOS bool

	// Deprecated: has no effect. llama.cpp removed tail-free sampling.
	TailFreeSamplingZ float32
	TypicalP          float32
	FrequencyPenalty  float32
	PresencePenalty   float32
	Mirostat          int
	MirostatETA       float32
	MirostatTAU       float32
	// Deprecated: has no effect.
	PenalizeNL    bool
	LogitBias     string
	TokenCallback func(string) bool

	// Deprecated: has no effect.
	PathPromptCache string
	// Deprecated: has no effect. Memory locking is a load-time setting
	// (EnableMLock).
	MLock bool
	// Deprecated: has no effect. Memory mapping is a load-time setting
	// (SetMMap).
	MMap bool
	// Deprecated: has no effect.
	PromptCacheAll bool
	// Deprecated: has no effect.
	PromptCacheRO bool
	Grammar       string
	// Deprecated: has no effect. Use the load-time SetMainGPU.
	MainGPU string
	// Deprecated: has no effect. Use the load-time SetTensorSplit.
	TensorSplit string

	// Rope parameters

	// Deprecated: has no effect. Use the load-time WithRopeFreqBase.
	RopeFreqBase float32
	// Deprecated: has no effect. Use the load-time WithRopeFreqScale.
	RopeFreqScale float32

	// XTC sampling parameters
	XTCProbability float32
	XTCThreshold   float32

	// DRY sampling parameters (Don't Repeat Yourself)
	DRYMultiplier    float32
	DRYBase          float32
	DRYAllowedLength int
	DRYPenaltyLastN  int

	// Top-N Sigma sampling
	TopNSigma float32
}

type PredictOption func(p *PredictOptions)

type ModelOption func(p *ModelOptions)

var DefaultModelOptions ModelOptions = ModelOptions{
	ContextSize:   512,
	Seed:          0,
	F16Memory:     false,
	MLock:         false,
	Embeddings:    false,
	MMap:          true,
	LowVRAM:       false,
	NBatch:        512,
	FreqRopeBase:  0, // the model's trained value
	FreqRopeScale: 0, // the model's trained value
}

var DefaultOptions PredictOptions = PredictOptions{
	Seed:              -1,
	Threads:           0, // keep the context's setting
	Tokens:            128,
	Penalty:           1.1,
	Repeat:            64,
	Batch:             512,
	NKeep:             64,
	TopK:              40,
	TopP:              0.95,
	MinP:              0.05,
	TailFreeSamplingZ: 1.0,
	TypicalP:          1.0,
	Temperature:       0.8,
	FrequencyPenalty:  0.0,
	PresencePenalty:   0.0,
	Mirostat:          0,
	MirostatTAU:       5.0,
	MirostatETA:       0.1,
	MMap:              true,
	RopeFreqBase:      10000,
	RopeFreqScale:     1.0,
	// XTC defaults (disabled)
	XTCProbability: 0.0,
	XTCThreshold:   0.5,
	// DRY defaults (disabled)
	DRYMultiplier:    0.0,
	DRYBase:          1.75,
	DRYAllowedLength: 2,
	DRYPenaltyLastN:  -1, // the context size
	// Top-N Sigma default (disabled)
	TopNSigma: 0.0,
}

// SetLoraBase sets the base model for a LoRA adapter.
//
// Deprecated: has no effect. llama.cpp no longer takes a separate base model
// for an adapter.
func SetLoraBase(s string) ModelOption {
	return func(p *ModelOptions) {
		p.LoraBase = s
	}
}

func SetLoraAdapter(s string) ModelOption {
	return func(p *ModelOptions) {
		p.LoraAdapter = s
	}
}

// SetContext sets the context size.
func SetContext(c int) ModelOption {
	return func(p *ModelOptions) {
		p.ContextSize = c
	}
}

// WithRopeFreqBase overrides the RoPE base frequency the model was trained
// with. 0, the default, keeps the trained value.
func WithRopeFreqBase(f float32) ModelOption {
	return func(p *ModelOptions) {
		p.FreqRopeBase = f
	}
}

// WithRopeFreqScale overrides the RoPE frequency scaling factor the model was
// trained with. 0, the default, keeps the trained value.
func WithRopeFreqScale(f float32) ModelOption {
	return func(p *ModelOptions) {
		p.FreqRopeScale = f
	}
}

// SetModelSeed sets a load-time seed.
//
// Deprecated: has no effect. Seed a sampler instead, with SetSeed or
// SamplerDist.
func SetModelSeed(c int) ModelOption {
	return func(p *ModelOptions) {
		p.Seed = c
	}
}

// SetMMap selects whether the model file is memory-mapped rather than read
// into memory. It is on by default.
func SetMMap(b bool) ModelOption {
	return func(p *ModelOptions) {
		p.MMap = b
	}
}

// SetNBatch sets the  n_Batch
func SetNBatch(n_batch int) ModelOption {
	return func(p *ModelOptions) {
		p.NBatch = n_batch
	}
}

// SetNSeqMax sets how many distinct sequences the context can hold (llama.cpp's
// n_seq_max). The default, 1, allows only sequence id 0; decoding a batch with
// a higher sequence id needs SetNSeqMax(n) with n greater than that id.
// ContextParams().NSeqMax reports the value in effect.
//
// Each sequence gets its own slice of the KV cache, ContextParams().NCtxSeq
// tokens: the context size divided between the sequences, which the engine
// rounds up to a multiple of 256, growing the total to fit.
//
// n may not exceed MaxParallelSequences, nor the context's batch size: the
// smaller of SetNBatch and SetContext (the model's training context when
// SetContext is 0), before any rounding. A larger value makes New fail.
func SetNSeqMax(n int) ModelOption {
	return func(p *ModelOptions) {
		p.NSeqMax = n
	}
}

// SetTensorSplit sets how the model is split across GPUs, as a comma-separated
// list of proportions, one per device. Devices past the end of the list get
// none, and entries past MaxDevices() are ignored. A list with an entry that is
// not a number is ignored as a whole.
func SetTensorSplit(maingpu string) ModelOption {
	return func(p *ModelOptions) {
		p.TensorSplit = maingpu
	}
}

// SetMainGPU sets the main_gpu
func SetMainGPU(maingpu string) ModelOption {
	return func(p *ModelOptions) {
		p.MainGPU = maingpu
	}
}

// SetPredictionTensorSplit sets the tensor split for the GPU
//
// Deprecated: has no effect. Use the load-time SetTensorSplit.
func SetPredictionTensorSplit(maingpu string) PredictOption {
	return func(p *PredictOptions) {
		p.TensorSplit = maingpu
	}
}

// SetPredictionMainGPU sets the main_gpu
//
// Deprecated: has no effect. Use the load-time SetMainGPU.
func SetPredictionMainGPU(maingpu string) PredictOption {
	return func(p *PredictOptions) {
		p.MainGPU = maingpu
	}
}

// SetRopeFreqBase sets a per-call RoPE base frequency.
//
// Deprecated: has no effect. RoPE is fixed when the context is created; use
// the load-time WithRopeFreqBase.
func SetRopeFreqBase(rfb float32) PredictOption {
	return func(p *PredictOptions) {
		p.RopeFreqBase = rfb
	}
}

// SetRopeFreqScale sets a per-call RoPE frequency scaling factor.
//
// Deprecated: has no effect. RoPE is fixed when the context is created; use
// the load-time WithRopeFreqScale.
func SetRopeFreqScale(rfs float32) PredictOption {
	return func(p *PredictOptions) {
		p.RopeFreqScale = rfs
	}
}

// SetNDraft sets the number of tokens to draft for speculative decoding.
//
// Deprecated: has no effect. Predict does not do speculative decoding.
func SetNDraft(nd int) PredictOption {
	return func(p *PredictOptions) {
		p.NDraft = nd
	}
}

// SetMinP sets the min_p sampling parameter
func SetMinP(minp float32) PredictOption {
	return func(p *PredictOptions) {
		p.MinP = minp
	}
}

// SetXTCProbability sets the XTC sampling probability (0.0 = disabled)
func SetXTCProbability(prob float32) PredictOption {
	return func(p *PredictOptions) {
		p.XTCProbability = prob
	}
}

// SetXTCThreshold sets the XTC sampling threshold
func SetXTCThreshold(threshold float32) PredictOption {
	return func(p *PredictOptions) {
		p.XTCThreshold = threshold
	}
}

// SetDRYMultiplier sets the DRY (Don't Repeat Yourself) multiplier (0.0 = disabled)
func SetDRYMultiplier(multiplier float32) PredictOption {
	return func(p *PredictOptions) {
		p.DRYMultiplier = multiplier
	}
}

// SetDRYBase sets the DRY base value
func SetDRYBase(base float32) PredictOption {
	return func(p *PredictOptions) {
		p.DRYBase = base
	}
}

// SetDRYAllowedLength sets the DRY allowed length
func SetDRYAllowedLength(length int) PredictOption {
	return func(p *PredictOptions) {
		p.DRYAllowedLength = length
	}
}

// SetDRYPenaltyLastN sets how many of the most recently generated tokens DRY
// scans for repeats. A negative value, the default -1, means the context size,
// ContextParams().NCtx, and a larger value is cut to the context size; 0
// disables DRY.
func SetDRYPenaltyLastN(n int) PredictOption {
	return func(p *PredictOptions) {
		p.DRYPenaltyLastN = n
	}
}

// SetTopNSigma sets the top-n sigma sampling parameter (0.0 = disabled)
func SetTopNSigma(n float32) PredictOption {
	return func(p *PredictOptions) {
		p.TopNSigma = n
	}
}

// EnabelLowVRAM asks for a low-VRAM mode.
//
// Deprecated: has no effect. llama.cpp removed the low-VRAM mode.
var EnabelLowVRAM ModelOption = func(p *ModelOptions) {
	p.LowVRAM = true
}

var EnableNUMA ModelOption = func(p *ModelOptions) {
	p.NUMA = true
}

var EnableEmbeddings ModelOption = func(p *ModelOptions) {
	p.Embeddings = true
}

// EnableF16Memory asks for a 16-bit KV cache.
//
// Deprecated: has no effect. The KV cache is already 16-bit by default.
var EnableF16Memory ModelOption = func(p *ModelOptions) {
	p.F16Memory = true
}

// EnableF16KV asks for a 16-bit KV cache.
//
// Deprecated: has no effect. The KV cache is chosen at load, and is already
// 16-bit by default.
var EnableF16KV PredictOption = func(p *PredictOptions) {
	p.F16KV = true
}

var Debug PredictOption = func(p *PredictOptions) {
	p.DebugMode = true
}

// EnablePromptCacheAll asks for the whole prompt to be cached.
//
// Deprecated: has no effect. Predict has no prompt cache; use
// SaveSessionFile and LoadSessionFile with Decode.
var EnablePromptCacheAll PredictOption = func(p *PredictOptions) {
	p.PromptCacheAll = true
}

// EnablePromptCacheRO asks for the prompt cache to be read-only.
//
// Deprecated: has no effect. Predict has no prompt cache.
var EnablePromptCacheRO PredictOption = func(p *PredictOptions) {
	p.PromptCacheRO = true
}

var EnableMLock ModelOption = func(p *ModelOptions) {
	p.MLock = true
}

// NewModelOptions returns DefaultModelOptions with opts applied in order.
func NewModelOptions(opts ...ModelOption) ModelOptions {
	p := DefaultModelOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

// IgnoreEOS asks Predict to keep generating past an end-of-generation token.
//
// Deprecated: has no effect. Predict always stops at an end-of-generation
// token; drive generation with Decode to go past one.
var IgnoreEOS PredictOption = func(p *PredictOptions) {
	p.IgnoreEOS = true
}

// WithGrammar constrains Predict's output to a GBNF grammar whose start rule is
// named root. The grammar is applied before any other sampling stage, so every
// token Predict picks is one the grammar allows. A grammar that does not parse
// makes Predict return an error.
func WithGrammar(s string) PredictOption {
	return func(p *PredictOptions) {
		p.Grammar = s
	}
}

// SetMlock sets the memory lock.
//
// Deprecated: has no effect. Memory locking is a load-time setting; use
// EnableMLock.
func SetMlock(b bool) PredictOption {
	return func(p *PredictOptions) {
		p.MLock = b
	}
}

// SetMemoryMap sets memory mapping.
//
// Deprecated: has no effect. Memory mapping is a load-time setting; use
// SetMMap.
func SetMemoryMap(b bool) PredictOption {
	return func(p *PredictOptions) {
		p.MMap = b
	}
}

// SetGPULayers sets the number of GPU layers to use to offload computation
func SetGPULayers(n int) ModelOption {
	return func(p *ModelOptions) {
		p.NGPULayers = n
	}
}

// SetTokenCallback streams this Predict call's output: fn receives each
// token's text as it is generated, and returning false stops generation. It
// applies to this call only, taking precedence over a callback set with
// (*LLama).SetTokenCallback, which is back in effect once the call returns.
func SetTokenCallback(fn func(string) bool) PredictOption {
	return func(p *PredictOptions) {
		p.TokenCallback = fn
	}
}

// SetStopWords sets the prompts that will stop predictions.
func SetStopWords(stop ...string) PredictOption {
	return func(p *PredictOptions) {
		p.StopPrompts = stop
	}
}

// SetSeed sets the random seed for sampling text generation.
func SetSeed(seed int) PredictOption {
	return func(p *PredictOptions) {
		p.Seed = seed
	}
}

// SetThreads sets the number of threads this Predict call uses, for both
// generation and prompt processing. The context's own setting, which
// (*LLama).SetThreads changes and llama.cpp starts at 4, is restored when the
// call returns. 0, the default, leaves the context's setting in effect.
func SetThreads(threads int) PredictOption {
	return func(p *PredictOptions) {
		p.Threads = threads
	}
}

// SetTokens sets the number of tokens to generate.
func SetTokens(tokens int) PredictOption {
	return func(p *PredictOptions) {
		p.Tokens = tokens
	}
}

// SetTopK sets the value for top-K sampling.
func SetTopK(topk int) PredictOption {
	return func(p *PredictOptions) {
		p.TopK = topk
	}
}

// SetTopP sets the value for nucleus sampling.
func SetTopP(topp float32) PredictOption {
	return func(p *PredictOptions) {
		p.TopP = topp
	}
}

// SetTemperature sets the temperature value for text generation.
func SetTemperature(temp float32) PredictOption {
	return func(p *PredictOptions) {
		p.Temperature = temp
	}
}

// SetPathPromptCache sets the session file to store the prompt cache.
//
// Deprecated: has no effect. Predict has no prompt cache; use
// SaveSessionFile and LoadSessionFile with Decode.
func SetPathPromptCache(f string) PredictOption {
	return func(p *PredictOptions) {
		p.PathPromptCache = f
	}
}

// SetPenalty sets the repetition penalty for text generation.
func SetPenalty(penalty float32) PredictOption {
	return func(p *PredictOptions) {
		p.Penalty = penalty
	}
}

// SetRepeat sets how many of the most recent tokens the repetition, frequency
// and presence penalties look back over (llama.cpp's repeat_last_n).
func SetRepeat(repeat int) PredictOption {
	return func(p *PredictOptions) {
		p.Repeat = repeat
	}
}

// SetBatch sets the batch size: the most prompt tokens Predict decodes at
// once. It is capped at the context's own batch size, ContextParams().NBatch.
func SetBatch(size int) PredictOption {
	return func(p *PredictOptions) {
		p.Batch = size
	}
}

// SetNKeep sets how many tokens from the start of the prompt are kept when the
// context fills up and older tokens have to be discarded.
func SetNKeep(n int) PredictOption {
	return func(p *PredictOptions) {
		p.NKeep = n
	}
}

// Create a new PredictOptions object with the given options.
func NewPredictOptions(opts ...PredictOption) PredictOptions {
	p := DefaultOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

// SetTailFreeSamplingZ sets the tail free sampling, parameter z.
//
// Deprecated: has no effect. llama.cpp removed tail-free sampling.
func SetTailFreeSamplingZ(tfz float32) PredictOption {
	return func(p *PredictOptions) {
		p.TailFreeSamplingZ = tfz
	}
}

// SetTypicalP sets the typicality parameter, p_typical.
func SetTypicalP(tp float32) PredictOption {
	return func(p *PredictOptions) {
		p.TypicalP = tp
	}
}

// SetFrequencyPenalty sets the frequency penalty parameter, freq_penalty.
func SetFrequencyPenalty(fp float32) PredictOption {
	return func(p *PredictOptions) {
		p.FrequencyPenalty = fp
	}
}

// SetPresencePenalty sets the presence penalty parameter, presence_penalty.
func SetPresencePenalty(pp float32) PredictOption {
	return func(p *PredictOptions) {
		p.PresencePenalty = pp
	}
}

// SetMirostat sets the mirostat parameter.
func SetMirostat(m int) PredictOption {
	return func(p *PredictOptions) {
		p.Mirostat = m
	}
}

// SetMirostatETA sets the mirostat ETA parameter.
func SetMirostatETA(me float32) PredictOption {
	return func(p *PredictOptions) {
		p.MirostatETA = me
	}
}

// SetMirostatTAU sets the mirostat TAU parameter.
func SetMirostatTAU(mt float32) PredictOption {
	return func(p *PredictOptions) {
		p.MirostatTAU = mt
	}
}

// SetPenalizeNL sets whether to penalize newlines or not.
//
// Deprecated: has no effect. llama.cpp's penalty stage no longer singles out
// newlines.
func SetPenalizeNL(pnl bool) PredictOption {
	return func(p *PredictOptions) {
		p.PenalizeNL = pnl
	}
}

// SetLogitBias sets the logit bias parameter.
func SetLogitBias(lb string) PredictOption {
	return func(p *PredictOptions) {
		p.LogitBias = lb
	}
}
