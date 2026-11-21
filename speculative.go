package llama

// #include "binding.h"
import "C"

import (
	"math"
	"unsafe"
)

// ContextType identifies the upstream context graph type.
type ContextType int

const (
	ContextDefault ContextType = iota
	ContextMTP
)

// DFlashVariant identifies a DFlash draft model's head architecture.
type DFlashVariant int

const (
	DFlashNone DFlashVariant = iota - 1
	DFlash
	DFlash2
	DSpark
	LiliCorr
)

func (l *LLama) DFlashVariant() DFlashVariant {
	return DFlashVariant(C.get_model_dflash_variant(l.state))
}

// DFlashSelectorTopK returns the DFlash2 selector width, or 0 without a selector.
func (l *LLama) DFlashSelectorTopK() int {
	return int(C.get_model_dflash_selector_top_k(l.state))
}

// TargetLayerIDs returns a copy of the target-model layer indices extracted
// by this draft model.
func (l *LLama) TargetLayerIDs() []int32 {
	n := int(C.get_model_target_layer_ids_n(l.state))
	p := C.get_model_target_layer_ids(l.state)
	if p == nil || n <= 0 {
		return nil
	}
	return append([]int32(nil), unsafe.Slice((*int32)(unsafe.Pointer(p)), n)...)
}

// NativeContext returns a borrowed llama_context pointer for C interop. Do
// not free it or use it after LLama.Free.
func (l *LLama) NativeContext() unsafe.Pointer {
	if l == nil || l.state == nil {
		return nil
	}
	return C.native_context(l.state)
}

// NativeModel returns a borrowed const llama_model pointer for C interop.
// Do not modify or free it; it is valid until LLama.Free.
func (l *LLama) NativeModel() unsafe.Pointer {
	if l == nil || l.state == nil {
		return nil
	}
	return C.native_model(l.state)
}

// NativePairedContext returns the borrowed context paired by the engine,
// or nil if there is none.
func (l *LLama) NativePairedContext() unsafe.Pointer {
	return C.native_context_other(l.state)
}

// SetContextStateSource makes this context read state rows from source.
// Pass nil to detach. Source must live until it is detached or this context
// is freed. Returns whether the context supports state-row batches.
func (l *LLama) SetContextStateSource(source *LLama) bool {
	var state unsafe.Pointer
	if source != nil {
		state = source.state
		if state == nil {
			return false
		}
	}
	ok := bool(C.set_ctx_state_src(l.state, state))
	l.stateSource = source
	return ok
}

// SetNextNEmbeddings enables next-token prediction embeddings. Masked mode
// outputs only entries marked as outputs in the batch.
func (l *LLama) SetNextNEmbeddings(enabled, masked bool) {
	C.set_embeddings_nextn(l.state, C.bool(enabled), C.bool(masked))
}

// SetNextNLayerOffset selects the appended NextN block for the MTP graph.
func (l *LLama) SetNextNLayerOffset(offset int32) bool {
	return bool(C.set_nextn_layer_offset(l.state, C.int(offset)))
}

// NativeLogits returns a borrowed contiguous float buffer of output logits.
// It is invalidated by later processing or freeing the context. Use Logits
// for a Go-owned row; callers of this native interface must know the output
// count and vocabulary width.
func (l *LLama) NativeLogits() unsafe.Pointer {
	return unsafe.Pointer(C.native_logits(l.state))
}

// NativeEmbeddings returns a borrowed contiguous float buffer of output
// embeddings, or nil when unavailable. Use TokenEmbedding for a Go-owned
// row. The buffer is invalidated by processing or freeing the context.
func (l *LLama) NativeEmbeddings() unsafe.Pointer {
	return unsafe.Pointer(C.native_embeddings(l.state))
}

// NativeNextNEmbeddings returns the borrowed contiguous NextN embedding
// buffer; its row count depends on the last batch and masking mode.
// Do not retain it across context operations or Free.
func (l *LLama) NativeNextNEmbeddings() unsafe.Pointer {
	return unsafe.Pointer(C.native_embeddings_nextn(l.state))
}

// NextNEmbedding copies one NextN embedding row from the last batch.
// Enable NextN embeddings and process a batch first.
func (l *LLama) NextNEmbedding(index int32) []float32 {
	n := int(C.get_model_n_embd_out(l.state))
	batch := int64(l.ContextParams().NBatch)
	if n <= 0 || int64(index) < -batch || int64(index) >= batch || l.NativeNextNEmbeddings() == nil {
		return nil
	}
	out := make([]float32, n)
	got := int(C.get_embeddings_nextn_ith(l.state, C.int(index),
		(*C.float)(unsafe.Pointer(&out[0])), C.int(n)))
	if got <= 0 {
		return nil
	}
	return out[:got]
}

// SetLayerInputEmbeddings enables extraction of input embeddings for a layer.
// Returns false for a layer outside the model's range.
func (l *LLama) SetLayerInputEmbeddings(layer uint32, enabled bool) bool {
	return bool(C.set_embeddings_layer_inp(l.state, C.uint(layer), C.bool(enabled)))
}

// UnsafeLayerInputEmbeddings returns a borrowed pointer to a layer's
// contiguous input embedding buffer. The caller must have enabled extraction
// for the layer and run a batch in host mode; the engine asserts if the
// buffer is unavailable. Do not retain the pointer across processing or Free.
func (l *LLama) UnsafeLayerInputEmbeddings(layer uint32) unsafe.Pointer {
	return unsafe.Pointer(C.native_embeddings_layer_inp(l.state, C.uint(layer)))
}

// SetLayerInputEmbeddingsDevice selects device-buffer delivery instead of
// host extraction for enabled layers.
func (l *LLama) SetLayerInputEmbeddingsDevice(enabled bool) {
	C.set_embeddings_layer_inp_dev(l.state, C.bool(enabled))
}

// SupportsEmbeddingRows reports whether this context can process embedding
// rows from a paired context's device buffer.
func (l *LLama) SupportsEmbeddingRows() bool {
	return bool(C.supports_embd_row(l.state))
}

// SetNextNEmbeddingsDevice requests device-buffer NextN output. Returns false
// if the engine must fall back to host memory.
func (l *LLama) SetNextNEmbeddingsDevice(enabled bool) bool {
	return bool(C.set_embeddings_nextn_dev(l.state, C.bool(enabled)))
}

func (l *LLama) nextNRowSize(row0, rows int64) int {
	p := l.ContextParams()
	maxRows := int64(p.NBatch) + int64(p.NSeqMax)
	width := int64(C.get_model_n_embd_out(l.state))
	if row0 < 0 || rows <= 0 || row0 > maxRows || rows > maxRows-row0 ||
		width <= 0 || rows > int64(math.MaxInt)/width {
		return 0
	}
	return int(rows * width)
}

// NextNDeviceRows copies rows from the device's NextN buffer to Go memory.
// Returns nil when unavailable or out of range.
func (l *LLama) NextNDeviceRows(row0, rows int64) []float32 {
	n := l.nextNRowSize(row0, rows)
	if n == 0 {
		return nil
	}
	out := make([]float32, n)
	if !bool(C.get_embeddings_nextn_rows(l.state, C.longlong(row0), C.longlong(rows),
		(*C.float)(unsafe.Pointer(&out[0])))) {
		return nil
	}
	return out
}

// SetNextNDeviceRows writes rows of the device's NextN buffer from Go memory.
// Data must contain rows * the model's output embedding width floats.
func (l *LLama) SetNextNDeviceRows(row0, rows int64, data []float32) bool {
	n := l.nextNRowSize(row0, rows)
	return n > 0 && len(data) == n && bool(C.set_embeddings_nextn_rows(l.state,
		C.longlong(row0), C.longlong(rows), (*C.float)(unsafe.Pointer(&data[0]))))
}
