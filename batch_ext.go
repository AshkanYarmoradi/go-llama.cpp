package llama

// #include "binding.h"
import "C"

import "unsafe"

// ProcessType identifies the operation performed by llama_process.
type ProcessType int

const (
	ProcessEncode ProcessType = iota
	ProcessDecode
)

// BatchEmbedding is a row-major embedding passed to the extended batch API.
// Data must contain Rows * Width floats. The engine copies it into the batch.
type BatchEmbedding struct {
	Data  []float32
	Rows  int
	Width int
}

func (e BatchEmbedding) valid() bool {
	return e.Rows > 0 && e.Width > 0 && len(e.Data)%e.Rows == 0 && len(e.Data)/e.Rows == e.Width
}

// ExtendedBatch wraps llama_batch_ext, which is allocated for a specific
// context. It can contain token IDs, embeddings, and hidden state. Free it
// before freeing the owning LLama. After each add, set the position explicitly.
type ExtendedBatch struct {
	ptr     unsafe.Pointer
	owner   *LLama
	embdPos int
	entries []extBatchEntry
	invalid bool
}

type extBatchEntry struct {
	positions  int
	ready      bool
	positioned bool
}

// NewExtendedBatch allocates a context-bound batch. Returns nil if allocation
// fails or the context has already been freed.
func (l *LLama) NewExtendedBatch() *ExtendedBatch {
	if l == nil || l.state == nil {
		return nil
	}
	ptr := C.ext_batch_init(l.state)
	if ptr == nil {
		return nil
	}
	nPos := 1
	rope := RopeType(C.get_model_rope_type(l.state))
	if rope == RopeMrope || rope == RopeImrope {
		nPos = 4
	}
	return &ExtendedBatch{ptr: ptr, owner: l, embdPos: nPos}
}

func (b *ExtendedBatch) active() bool {
	return b != nil && b.ptr != nil && b.owner.state != nil && !b.invalid
}

// Free releases the engine batch. Calling Free again is safe.
func (b *ExtendedBatch) Free() {
	if b != nil && b.ptr != nil {
		C.ext_batch_free(b.ptr)
		b.ptr = nil
		b.entries = nil
	}
}

// Reset clears the batch, including any incomplete entry from a failed add.
func (b *ExtendedBatch) Reset() {
	if b != nil && b.ptr != nil && b.owner.state != nil {
		C.ext_batch_clear(b.ptr)
		b.entries = b.entries[:0]
		b.invalid = false
	}
}

// Len returns the number of successfully added entries.
func (b *ExtendedBatch) Len() int { return len(b.entries) }

func (b *ExtendedBatch) record(idx int, positions int, ready bool) int {
	if idx >= 0 {
		b.entries = append(b.entries, extBatchEntry{positions: positions, ready: ready})
	} else if idx == -2 {
		// Upstream may append a placeholder before rejecting an embedding/token.
		// Never process or modify that incomplete batch until the caller resets it.
		b.invalid = true
	}
	return idx
}

// Add adds an empty entry to populate with SetTokenEmbedding. Add methods
// return the entry index, or -1 (full), -2 (invalid), -3 (invalid sequence).
func (b *ExtendedBatch) Add(seqID int32) int {
	if !b.active() {
		return -2
	}
	return b.record(int(C.ext_batch_add(b.ptr, C.int(seqID))), b.embdPos, false)
}

// AddToken adds a token ID to the given sequence.
func (b *ExtendedBatch) AddToken(seqID, token int32) int {
	if !b.active() || token < 0 || token >= int32(C.get_model_n_vocab(b.owner.state)) {
		return -2
	}
	return b.record(int(C.ext_batch_add_token(b.ptr, C.int(seqID), C.int(token))), 1, true)
}

// AddEmbedding adds an embedding instead of a token ID. If the engine returns
// -2 for an unsupported embedding shape, Reset must precede any further use.
func (b *ExtendedBatch) AddEmbedding(seqID int32, e BatchEmbedding) int {
	if !b.active() || !e.valid() {
		return -2
	}
	return b.record(int(C.ext_batch_add_embd(b.ptr, C.int(seqID),
		(*C.float)(unsafe.Pointer(&e.Data[0])), C.size_t(e.Rows), C.size_t(e.Width))), b.embdPos, true)
}

// AddEmbeddingRow adds a row of the paired context's device layer input buffer.
func (b *ExtendedBatch) AddEmbeddingRow(seqID, row int32) int {
	if !b.active() || row < 0 || b.owner.paired == nil || b.owner.paired.state == nil ||
		int(row) >= b.owner.paired.ContextParams().NBatch || !b.owner.SupportsEmbeddingRows() {
		return -2
	}
	return b.record(int(C.ext_batch_add_embd_row(b.ptr, C.int(seqID), C.int(row))), b.embdPos, true)
}

func (b *ExtendedBatch) has(idx int) bool {
	return b.active() && idx >= 0 && idx < len(b.entries)
}

// AddSequence assigns an entry to an additional sequence. Call it before the
// Set methods, as required by llama.cpp.
func (b *ExtendedBatch) AddSequence(idx int, seqID int32) bool {
	return b.has(idx) && bool(C.ext_batch_add_seq(b.ptr, C.int(idx), C.int(seqID)))
}

// SetPosition sets an entry's position: one for a token ID or a normal
// embedding, four (t, y, x, z) for an M-RoPE embedding.
func (b *ExtendedBatch) SetPosition(idx int, pos []int32) bool {
	if !b.has(idx) || len(pos) != b.entries[idx].positions {
		return false
	}
	ok := bool(C.ext_batch_set_pos(b.ptr, C.int(idx), (*C.int)(unsafe.Pointer(&pos[0]))))
	if ok {
		b.entries[idx].positioned = true
	}
	return ok
}

// SetTokenEmbedding adds an embedding to a token entry, or populates an entry
// created with Add.
func (b *ExtendedBatch) SetTokenEmbedding(idx int, e BatchEmbedding) bool {
	if !b.has(idx) || !e.valid() {
		return false
	}
	ok := bool(C.ext_batch_set_embd_token(b.ptr, C.int(idx),
		(*C.float)(unsafe.Pointer(&e.Data[0])), C.size_t(e.Rows), C.size_t(e.Width)))
	if ok {
		b.entries[idx].ready = true
	}
	return ok
}

// SetStateEmbedding supplies hidden state from an earlier stage.
func (b *ExtendedBatch) SetStateEmbedding(idx int, e BatchEmbedding) bool {
	return b.has(idx) && e.valid() && bool(C.ext_batch_set_embd_state(b.ptr, C.int(idx),
		(*C.float)(unsafe.Pointer(&e.Data[0])), C.size_t(e.Rows), C.size_t(e.Width)))
}

// SetStateEmbeddingRow supplies hidden state from a device buffer row.
func (b *ExtendedBatch) SetStateEmbeddingRow(idx int, row int32) bool {
	if !b.has(idx) || row < 0 || b.owner.stateSource == nil || b.owner.stateSource.state == nil {
		return false
	}
	p := b.owner.stateSource.ContextParams()
	return int64(row) < int64(p.NBatch)+int64(p.NSeqMax) &&
		bool(C.ext_batch_set_embd_state_row(b.ptr, C.int(idx), C.int(row)))
}

// SetOutputEmbedding requests output embeddings for this entry.
func (b *ExtendedBatch) SetOutputEmbedding(idx int, enabled bool) bool {
	return b.has(idx) && bool(C.ext_batch_set_output_embd(b.ptr, C.int(idx), C.bool(enabled)))
}

// SetOutputLogits requests output logits for this entry.
func (b *ExtendedBatch) SetOutputLogits(idx int, enabled bool) bool {
	return b.has(idx) && bool(C.ext_batch_set_output_logits(b.ptr, C.int(idx), C.bool(enabled)))
}

// Process runs a context-bound extended batch through the encoder or decoder.
// It returns the engine's status, or -1 for an invalid batch or operation.
func (l *LLama) Process(t ProcessType, b *ExtendedBatch) int {
	if !b.active() || b.owner != l || (t != ProcessEncode && t != ProcessDecode) {
		return -1
	}
	for _, entry := range b.entries {
		if !entry.ready || !entry.positioned {
			return -1
		}
	}
	return int(C.ext_batch_process(l.state, b.ptr, C.int(t), C.int(len(b.entries))))
}
