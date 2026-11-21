package llama

// #include "binding.h"
// #include <stdlib.h>
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// NewFromFileOffset loads a GGUF embedded at a byte offset in a file. With
// mmap enabled, the GGUF data must be aligned to the engine's tensor alignment.
func NewFromFileOffset(path string, offset int64, opts ...ModelOption) (*LLama, error) {
	file, err := openFileAt(path, offset)
	if err != nil {
		return nil, err
	}
	defer C.close_native_file(file)
	return NewFromNativeFile(file, opts...)
}

func openFileAt(path string, offset int64) (unsafe.Pointer, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	file := C.open_file_at(cpath, C.longlong(offset))
	if file == nil {
		return nil, fmt.Errorf("llama: cannot open %q at offset %d", path, offset)
	}
	return file, nil
}

// NewFromNativeFile loads the model from a native C FILE* at its current
// position. The caller owns the file; the function does not close it. This
// is for C interop, not an os.File pointer.
func NewFromNativeFile(file unsafe.Pointer, opts ...ModelOption) (*LLama, error) {
	if file == nil {
		return nil, errors.New("llama: nil native model file")
	}
	return newModelWithInput(nil, file, nil, nil, nil, NewModelOptions(opts...))
}

// NewFromNativeGGUF builds a model from a native gguf_context and a C
// llama_model_set_tensor_data_t callback. All arguments must be native C
// pointers (not Go pointers); the metadata, callback, and its userdata must
// remain valid throughout loading. The callback supplies the model's weights.
func NewFromNativeGGUF(metadata, setTensorData, userData unsafe.Pointer, opts ...ModelOption) (*LLama, error) {
	if metadata == nil || setTensorData == nil {
		return nil, errors.New("llama: native GGUF metadata and tensor callback are required")
	}
	return newModelWithInput(nil, nil, metadata, setTensorData, userData, NewModelOptions(opts...))
}

// ApplyLoRAFromFileOffset loads a LoRA adapter embedded in a file.
func (l *LLama) ApplyLoRAFromFileOffset(path string, offset int64, scale float32) error {
	file, err := openFileAt(path, offset)
	if err != nil {
		return err
	}
	defer C.close_native_file(file)
	return l.ApplyLoRAFromNativeFile(file, scale)
}

// ApplyLoRAFromNativeFile loads and applies an adapter from a native C FILE*
// at its current position. The caller retains ownership of the file.
func (l *LLama) ApplyLoRAFromNativeFile(file unsafe.Pointer, scale float32) error {
	if file == nil || C.apply_lora_adapter_file(l.state, file, C.float(scale)) != 0 {
		return errors.New("llama: failed to apply native LoRA adapter")
	}
	return nil
}

// AttachNativeThreadpool installs native ggml_threadpool_t handles. They
// must remain alive until detached or the context is freed. The caller owns
// both pools; batchThreadpool may be nil.
func (l *LLama) AttachNativeThreadpool(threadpool, batchThreadpool unsafe.Pointer) {
	C.attach_threadpool(l.state, threadpool, batchThreadpool)
}

// DetachThreadpool switches back to the engine's automatic threadpool.
func (l *LLama) DetachThreadpool() { C.detach_threadpool(l.state) }

// NewNativeSampler constructs a sampler from a native llama_sampler_i vtable
// with name/apply callbacks and native sampler context. This is a C interop
// entry point: the vtable must outlive the sampler, and the free callback
// owns its context. No pointer into Go memory may be passed here.
func NewNativeSampler(iface, ctx unsafe.Pointer) *Sampler {
	ptr := C.native_sampler_init(iface, ctx)
	if ptr == nil {
		return nil
	}
	return &Sampler{ptr: ptr}
}

// SamplerMirostatV1 creates a Mirostat 1.0 stage. nVocab is the model's
// vocabulary size, and m is the number of tokens used to estimate the slope.
func SamplerMirostatV1(nVocab int32, seed uint32, tau, eta float32, m int32) *Sampler {
	if nVocab <= 0 || m <= 0 {
		return nil
	}
	return &Sampler{ptr: C.sampler_init_mirostat_v1(C.int(nVocab), C.uint(seed), C.float(tau), C.float(eta), C.int(m))}
}

// CopyFrom copies another sampler's state into this existing sampler. The
// two must have compatible types and chains must have identical stage types.
// Returns false instead of invoking the engine on an incompatible pair.
func (s *Sampler) CopyFrom(source *Sampler) bool {
	return s != nil && source != nil && bool(C.sampler_copy(s.ptr, source.ptr))
}

// PerfPrint emits a sampler chain's performance counters through the engine
// log. It is a no-op on a single stage.
func (s *Sampler) PerfPrint() {
	if s != nil && s.chain {
		C.perf_sampler_print(s.ptr)
	}
}

// OptimizerType matches ggml_opt_optimizer_type.
type OptimizerType int

const (
	OptimizerAdamW OptimizerType = iota
	OptimizerSGD
)

// NativeOptimizerOptions maps llama_opt_params. The optional callbacks are
// native C function pointers and the userdata is C-owned. Nil selects the
// engine's all-parameter filter and default optimizer settings.
type NativeOptimizerOptions struct {
	ContextSize   uint32
	Optimizer     OptimizerType
	Filter        unsafe.Pointer
	FilterData    unsafe.Pointer
	GetParams     unsafe.Pointer
	GetParamsData unsafe.Pointer
}

// OptParamFilterAll exposes the engine's always-true parameter filter.
func OptParamFilterAll(tensor, userData unsafe.Pointer) bool {
	return bool(C.opt_param_filter_all(tensor, userData))
}

// InitNativeOptimizer initializes the engine's training graph. This is a
// low-level training interface for a training-compatible model (not ordinary
// quantized inference weights). The native callbacks and their userdata must
// remain valid until Free. Like upstream, initialization is once per context.
func (l *LLama) InitNativeOptimizer(opts NativeOptimizerOptions) bool {
	return bool(C.opt_init(l.state, C.uint(opts.ContextSize), C.int(opts.Optimizer),
		opts.Filter, opts.FilterData, opts.GetParams, opts.GetParamsData))
}

// NativeOptimizerEpoch runs a training/evaluation epoch over a native ggml
// optimization dataset and result handles. split is the training datapoint
// count; the remaining points are for evaluation. The dataset must have the
// token and sparse-label geometry required by the trained model, and the
// optional callbacks must have the native ggml_opt_epoch_callback signature.
func (l *LLama) NativeOptimizerEpoch(dataset, trainResult, evalResult unsafe.Pointer, split int64,
	trainCallback, evalCallback unsafe.Pointer) bool {
	return bool(C.opt_epoch(l.state, dataset, trainResult, evalResult, C.longlong(split),
		trainCallback, evalCallback))
}
