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
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
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

// ModelInfo contains information about the loaded model
type ModelInfo struct {
	VocabSize     int
	ContextLength int
	EmbeddingSize int
	LayerCount    int
	ModelSize     int64
	ParamCount    int64
	Description   string
}

// GetModelInfo returns information about the loaded model
func (l *LLama) GetModelInfo() ModelInfo {
	descBuf := make([]byte, 256)
	C.get_model_description(l.state, (*C.char)(unsafe.Pointer(&descBuf[0])), C.int(len(descBuf)))

	return ModelInfo{
		VocabSize:     int(C.get_model_n_vocab(l.state)),
		ContextLength: int(C.get_model_n_ctx_train(l.state)),
		EmbeddingSize: int(C.get_model_n_embd(l.state)),
		LayerCount:    int(C.get_model_n_layer(l.state)),
		ModelSize:     int64(C.get_model_size(l.state)),
		ParamCount:    int64(C.get_model_n_params(l.state)),
		Description:   string(descBuf[:cStrLen(descBuf)]),
	}
}

// GetChatTemplate returns the chat template for the model
// Pass empty string for default template, or template name for specific template
func (l *LLama) GetChatTemplate(name string) string {
	buf := make([]byte, 4096)
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}
	ret := C.get_model_chat_template(l.state, cName, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	if ret <= 0 {
		return ""
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

// SpecialTokens contains the special token IDs for the model's vocabulary
type SpecialTokens struct {
	BOS int32 // Beginning of sentence
	EOS int32 // End of sentence
	EOT int32 // End of turn
	NL  int32 // Newline
	SEP int32 // Separator
}

// GetSpecialTokens returns the special token IDs for the model
func (l *LLama) GetSpecialTokens() SpecialTokens {
	return SpecialTokens{
		BOS: int32(C.get_vocab_bos(l.state)),
		EOS: int32(C.get_vocab_eos(l.state)),
		EOT: int32(C.get_vocab_eot(l.state)),
		NL:  int32(C.get_vocab_nl(l.state)),
		SEP: int32(C.get_vocab_sep(l.state)),
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
	out := make([]byte, po.Tokens)

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
	ret := C.llama_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])), C.bool(po.DebugMode))
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
