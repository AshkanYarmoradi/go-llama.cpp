package llama_test

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AshkanYarmoradi/go-llama.cpp"
	. "github.com/AshkanYarmoradi/go-llama.cpp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LLama binding", func() {
	testModelPath := os.Getenv("TEST_MODEL")

	Context("Declaration", func() {
		It("fails with no model", func() {
			model, err := New("not-existing")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})
	Context("Inferencing tests (using "+testModelPath+") ", func() {
		getModel := func() (*LLama, error) {
			model, err := New(
				testModelPath,
				EnableF16Memory,
				SetContext(128),
				SetMMap(true),
				SetNBatch(512),
			)
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model, err
		}

		It("predicts successfully", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := getModel()
			text, err := model.Predict(`[INST] Answer to the following question:
how much is 2+2?
[/INST]`)
			Expect(err).ToNot(HaveOccurred(), text)
			Expect(text).To(ContainSubstring("4"), text)
		})

		It("predicts with min_p sampling", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(
				testModelPath,
				EnableF16Memory,
				SetContext(128),
				SetMMap(true),
				SetNBatch(512),
			)
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			text, err := model.Predict(`[INST] Answer to the following question:
how much is 2+2?
[/INST]`, llama.SetMinP(0.05),
			)
			Expect(err).ToNot(HaveOccurred(), text)
			Expect(text).To(ContainSubstring("4"), text)
		})

		It("applies logit bias during generation", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := getModel()
			Expect(err).ToNot(HaveOccurred())

			// A bias string is "token(+|-)value". This exercises the logit-bias
			// path, which was previously parsed but never added to the sampler
			// chain; generation must still succeed and produce output.
			text, err := model.Predict(`[INST] Answer to the following question:
how much is 2+2?
[/INST]`, llama.SetLogitBias("5+1.0"))
			Expect(err).ToNot(HaveOccurred(), text)
			Expect(text).ToNot(BeEmpty())
		})

		It("tokenizes strings successfully", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := getModel()
			l, tokens, err := model.TokenizeString("A STRANGE GAME.\nTHE ONLY WINNING MOVE IS NOT TO PLAY.\n\nHOW ABOUT A NICE GAME OF CHESS?",
				SetRopeFreqBase(10000.0), SetRopeFreqScale(1))

			Expect(err).ToNot(HaveOccurred())
			Expect(l).To(BeNumerically(">", 0))
			Expect(int(l)).To(Equal(len(tokens)))
		})

		It("tokenizes and detokenizes round-trip", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			const text = "The quick brown fox jumps over the lazy dog."

			tokens := model.Tokenize(text, false, false)
			Expect(tokens).ToNot(BeEmpty())
			// A token never spans fewer than one byte.
			Expect(len(tokens)).To(BeNumerically("<=", len(text)+8))

			// Detokenizing reconstructs the text (allowing for tokenizer
			// normalization at the edges, so assert on a stable inner substring).
			round := model.Detokenize(tokens, false, false)
			Expect(round).To(ContainSubstring("quick brown fox"))

			// Each token decodes to a non-empty piece.
			Expect(model.TokenToPiece(tokens[0], false)).ToNot(BeEmpty())
		})

		It("returns special tokens", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			tokens := model.GetSpecialTokens()
			Expect(tokens.BOS).To(BeNumerically(">=", 0))
			Expect(tokens.EOS).To(BeNumerically(">=", 0))
			// BOS and EOS should be different tokens
			Expect(tokens.BOS).ToNot(Equal(tokens.EOS))
		})

		It("returns vocab add BOS/EOS flags", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			// CodeLlama adds BOS automatically
			addBos := model.GetVocabAddBOS()
			Expect(addBos).To(BeTrue())
		})

		It("returns model architecture info", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			// CodeLlama is a decoder-only transformer
			Expect(model.ModelHasEncoder()).To(BeFalse())
			Expect(model.ModelHasDecoder()).To(BeTrue())
			Expect(model.ModelIsRecurrent()).To(BeFalse())
		})

		It("returns model info", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			info := model.GetModelInfo()
			Expect(info.VocabSize).To(BeNumerically(">", 0))
			Expect(info.ContextLength).To(BeNumerically(">", 0))
			Expect(info.EmbeddingSize).To(BeNumerically(">", 0))
			Expect(info.LayerCount).To(BeNumerically(">", 0))
			Expect(info.ModelSize).To(BeNumerically(">", 0))
			Expect(info.ParamCount).To(BeNumerically(">", 0))
			Expect(info.Description).ToNot(BeEmpty())
		})

		It("exposes extended model geometry", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			info := model.GetModelInfo()
			Expect(info.HeadCount).To(BeNumerically(">", 0))
			Expect(info.HeadCountKV).To(BeNumerically(">", 0))
			// Attention heads are shared (MHA/GQA/MQA): KV heads never exceed query heads.
			Expect(info.HeadCountKV).To(BeNumerically("<=", info.HeadCount))
			Expect(info.RopeFreqScaleTrain).To(BeNumerically(">", 0))
			Expect(info.SlidingWindow).To(BeNumerically(">=", 0))
		})

		It("exposes model metadata", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			meta := model.ModelMetadata()
			Expect(meta).ToNot(BeEmpty())
			Expect(meta).To(HaveKey("general.architecture"))

			arch, ok := model.ModelMetadataValue("general.architecture")
			Expect(ok).To(BeTrue())
			Expect(arch).ToNot(BeEmpty())
			Expect(arch).To(Equal(meta["general.architecture"]))

			_, ok = model.ModelMetadataValue("this.key.does.not.exist")
			Expect(ok).To(BeFalse())
		})

		It("reports an error for a missing LoRA adapter", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, _ := getModel()
			// Applying a nonexistent adapter must fail cleanly: the loader
			// returns null rather than throwing across the C boundary.
			err := model.ApplyLoRA("does-not-exist.gguf", 1.0)
			Expect(err).To(HaveOccurred())

			// Clearing when nothing is applied must be a safe no-op.
			model.ClearLoRA()
		})
	})

	Context("System info", func() {
		It("returns system info string", func() {
			info := llama.SystemInfo()
			Expect(info).ToNot(BeEmpty())
		})
	})

	Context("Backend capabilities", func() {
		It("reports capability flags without a loaded model", func() {
			// These reflect compiled-in backend features and must not require a model.
			Expect(llama.MaxParallelSequences()).To(BeNumerically(">=", 1))
			Expect(llama.MaxDevices()).To(BeNumerically(">=", 0))
			// mmap is available on every platform in the CI matrix.
			Expect(llama.SupportsMmap()).To(BeTrue())
			// The remaining queries must at least execute without panicking.
			_ = llama.SupportsMlock()
			_ = llama.SupportsGPUOffload()
			_ = llama.SupportsRPC()
		})
	})

	Context("Low-level batching", func() {
		It("bounds-checks batch capacity", func() {
			// Batch allocation and the capacity guard need no model.
			batch := NewBatch(1, 1)
			defer batch.Free()
			Expect(batch.Add(1, 0, []int32{0}, false)).To(Succeed())
			// The second token exceeds capacity 1 and must be rejected.
			Expect(batch.Add(2, 1, []int32{0}, false)).To(HaveOccurred())
			Expect(batch.Len()).To(Equal(1))
		})

		It("decodes a batch and returns next-token logits", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			model.MemoryClear(true)

			tokens := model.Tokenize("The capital of France is", true, false)
			Expect(tokens).ToNot(BeEmpty())

			batch := NewBatch(len(tokens), 1)
			defer batch.Free()
			for i, tok := range tokens {
				// Request output only for the final token of the sequence.
				Expect(batch.Add(tok, int32(i), []int32{0}, i == len(tokens)-1)).To(Succeed())
			}
			Expect(batch.Len()).To(Equal(len(tokens)))
			Expect(model.Decode(batch)).To(Equal(0))

			logits := model.Logits(-1)
			Expect(logits).To(HaveLen(model.GetModelInfo().VocabSize))

			// A real forward pass yields a well-defined, in-range argmax.
			best, bestIdx := logits[0], 0
			for i, v := range logits {
				if v > best {
					best, bestIdx = v, i
				}
			}
			Expect(bestIdx).To(BeNumerically(">=", 0))
			Expect(bestIdx).To(BeNumerically("<", model.GetModelInfo().VocabSize))
		})
	})

	Context("Composable samplers", func() {
		It("builds a chain and samples a valid token", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())

			// Predict once so the context holds real logits to sample from.
			_, err = model.Predict("The capital of France is", SetTokens(4))
			Expect(err).ToNot(HaveOccurred())

			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(SamplerTopK(40))
			chain.Add(SamplerTopP(0.95, 1))
			chain.Add(SamplerTemp(0.8))
			chain.Add(SamplerDist(1234))

			// Sample also accepts the token, so there is no Accept here.
			tok := chain.Sample(model, -1)
			Expect(tok).To(BeNumerically(">=", 0))
			Expect(tok).To(BeNumerically("<", int32(model.GetModelInfo().VocabSize)))
		})
	})

	Context("Library information", func() {
		It("reports a llama.cpp version", func() {
			Expect(Version()).ToNot(BeEmpty())
		})

		It("exposes a monotonic clock", func() {
			t0 := TimeUS()
			Expect(t0).To(BeNumerically(">", int64(0)))
			Expect(TimeUS()).To(BeNumerically(">=", t0))
		})
	})

	Context("Context introspection and control", func() {
		It("reports the geometry the context actually uses", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			p := model.ContextParams()
			Expect(p.NCtx).To(BeNumerically(">", 0))
			Expect(p.NCtxSeq).To(BeNumerically(">", 0))
			Expect(p.NCtxSeq).To(BeNumerically("<=", p.NCtx))
			Expect(p.NBatch).To(BeNumerically(">", 0))
			Expect(p.NUbatch).To(BeNumerically(">", 0))
			Expect(p.NUbatch).To(BeNumerically("<=", p.NBatch))
			Expect(p.NSeqMax).To(BeNumerically(">=", 1))
			Expect(p.NRSSeq).To(BeNumerically(">=", 0))
			Expect(p.Pooling.String()).ToNot(BeEmpty())
		})

		It("round-trips the thread counts", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			model.SetThreads(2, 3)
			gen, batch := model.Threads()
			Expect(gen).To(Equal(2))
			Expect(batch).To(Equal(3))
		})

		It("tracks KV-cache sequence positions across a decode", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			model.MemoryClear(true)
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(-1)), "cache should be empty after a clear")

			tokens := model.Tokenize("The capital of France is", true, false)
			Expect(tokens).ToNot(BeEmpty())

			batch := NewBatch(len(tokens), 1)
			defer batch.Free()
			for i, tok := range tokens {
				Expect(batch.Add(tok, int32(i), []int32{0}, i == len(tokens)-1)).To(Succeed())
			}
			Expect(model.Decode(batch)).To(Equal(0))

			Expect(model.MemorySeqPosMin(0)).To(BeNumerically(">=", int32(0)))
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(tokens) - 1)))

			// The standard context-shift idiom: evict the oldest token, then
			// slide the remainder back so positions stay contiguous from 0.
			// Not every cache type supports it.
			if model.MemoryCanShift() {
				Expect(model.MemorySeqRemove(0, 0, 1)).To(BeTrue())
				model.MemorySeqAdd(0, 1, -1, -1)
				Expect(model.MemorySeqPosMin(0)).To(Equal(int32(0)))
				Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(tokens) - 2)))
			}

			model.MemorySeqKeep(0)
			Expect(model.MemorySeqPosMax(1)).To(Equal(int32(-1)))
		})

		It("accumulates and resets performance counters", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			_, err = model.Predict("The capital of France is", SetTokens(8))
			Expect(err).ToNot(HaveOccurred())

			perf := model.Perf()
			Expect(perf.LoadMS).To(BeNumerically(">", 0))
			// The binding turns off no_perf, so the engine actually records
			// timings; a multi-token prompt decode lands in the prompt-eval
			// bucket and a generated token in the eval bucket.
			Expect(perf.PromptEvalMS).To(BeNumerically(">", 0))
			Expect(perf.PromptTokens).To(BeNumerically(">", 1))
			Expect(perf.EvalMS).To(BeNumerically(">", 0))

			model.PerfReset()
			reset := model.Perf()
			Expect(reset.PromptEvalMS).To(BeNumerically("==", 0))
			Expect(reset.EvalMS).To(BeNumerically("==", 0))
			// llama.cpp floors these counters at 1 so its own reporting can
			// divide by them, so a reset context reports 1 rather than 0.
			Expect(reset.PromptTokens).To(Equal(1))
			Expect(reset.EvalTokens).To(Equal(1))
			// Load time survives a reset; only the eval counters restart.
			Expect(reset.LoadMS).To(Equal(perf.LoadMS))
		})
	})

	Context("Chat templates", func() {
		It("lists the built-in templates", func() {
			names := BuiltinChatTemplates()
			Expect(names).ToNot(BeEmpty())
			Expect(names).To(ContainElement("chatml"))
			for _, n := range names {
				Expect(n).ToNot(BeEmpty())
			}
		})

		It("renders a chat with an explicit template", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			msgs := []ChatMessage{
				{Role: "system", Content: "You are terse."},
				{Role: "user", Content: "How much is 2+2?"},
			}

			out, err := model.ApplyChatTemplate("chatml", msgs, true)
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(ContainSubstring("You are terse."))
			Expect(out).To(ContainSubstring("How much is 2+2?"))
			Expect(out).To(ContainSubstring("<|im_start|>"))
			// addAssistant opens the reply turn.
			Expect(out).To(HaveSuffix("<|im_start|>assistant\n"))
		})

		It("omits the assistant turn when not requested", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			msgs := []ChatMessage{{Role: "user", Content: "hi"}}
			out, err := model.ApplyChatTemplate("chatml", msgs, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(out).ToNot(HaveSuffix("<|im_start|>assistant\n"))
		})

		It("grows the buffer for a message larger than the initial guess", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			long := strings.Repeat("word ", 4000)
			out, err := model.ApplyChatTemplate("chatml", []ChatMessage{{Role: "user", Content: long}}, true)
			Expect(err).ToNot(HaveOccurred())
			Expect(len(out)).To(BeNumerically(">=", len(long)))
			Expect(out).To(ContainSubstring(long))
		})

		It("reports an unusable template rather than truncating", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			_, err = model.ApplyChatTemplate("not-a-real-template", []ChatMessage{{Role: "user", Content: "hi"}}, true)
			Expect(err).To(MatchError(ErrNoChatTemplate))
		})

		It("handles an empty message list", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			out, err := model.ApplyChatTemplate("chatml", nil, true)
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(Equal("<|im_start|>assistant\n"))
		})
	})

	Context("Vocabulary and architecture introspection", func() {
		newModel := func() *LLama {
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model
		}

		It("names enum values without a model", func() {
			// LLAMA_FTYPE_ALL_F32 == 0 is stable across every llama.cpp release.
			Expect(FileTypeName(0)).ToNot(BeEmpty())
			Expect(FlashAttnTypeName(-1)).To(Equal("auto"))
			// The engine aborts on a value outside the enum, so the binding
			// must answer for it.
			Expect(FlashAttnTypeName(99)).To(BeEmpty())
			Expect(VocabSPM.String()).To(Equal("spm"))
			Expect(RopeNeox.String()).To(Equal("neox"))
			Expect(PoolingMean.String()).To(Equal("mean"))
		})

		It("renders token attribute bitmasks", func() {
			a := TokenAttrControl | TokenAttrByte
			Expect(a.Has(TokenAttrControl)).To(BeTrue())
			Expect(a.Has(TokenAttrByte)).To(BeTrue())
			Expect(a.Has(TokenAttrNormal)).To(BeFalse())
			Expect(a.Has(TokenAttrControl | TokenAttrByte)).To(BeTrue())
			Expect(a.String()).To(Equal("control|byte"))
			Expect(TokenAttrUndefined.String()).To(Equal("undefined"))
		})

		It("reports the tokenizer family", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			Expect(model.VocabType()).ToNot(Equal(VocabNone))
			Expect(model.VocabType().String()).ToNot(ContainSubstring("VocabType("))
		})

		It("describes individual tokens", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			special := model.GetSpecialTokens()
			Expect(special.EOS).To(BeNumerically(">=", int32(0)))

			// EOS must both be a control token and end generation.
			Expect(model.TokenText(special.EOS)).ToNot(BeEmpty())
			Expect(model.IsEOG(special.EOS)).To(BeTrue())
			Expect(model.IsControlToken(special.EOS)).To(BeTrue())
			Expect(model.TokenAttr(special.EOS).Has(TokenAttrControl)).To(BeTrue())

			// A plain word token must not.
			toks := model.Tokenize("hello", false, false)
			Expect(toks).ToNot(BeEmpty())
			Expect(model.IsEOG(toks[0])).To(BeFalse())
			Expect(model.IsControlToken(toks[0])).To(BeFalse())
			Expect(model.TokenText(toks[0])).ToNot(BeEmpty())
		})

		It("rejects out-of-range tokens instead of reading past the vocabulary", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			for _, bad := range []int32{-1, int32(model.GetModelInfo().VocabSize), 1 << 30} {
				Expect(model.TokenText(bad)).To(BeEmpty())
				Expect(model.TokenScore(bad)).To(Equal(float32(0)))
				Expect(model.TokenAttr(bad)).To(Equal(TokenAttrUndefined))
				Expect(model.IsEOG(bad)).To(BeFalse())
				Expect(model.IsControlToken(bad)).To(BeFalse())
			}
		})

		It("reports the model architecture", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			arch := model.Architecture()
			// A decoder-only causal LM: decoder present, encoder absent.
			Expect(arch.HasDecoder).To(BeTrue())
			Expect(arch.HasEncoder).To(BeFalse())
			Expect(arch.RopeType).ToNot(Equal(RopeNone))
			Expect(arch.FileTypeName).ToNot(BeEmpty())
			Expect(arch.EmbdInp).To(BeNumerically(">", 0))
			Expect(arch.EmbdOut).To(BeNumerically(">", 0))
			Expect(arch.FileTypeName).To(Equal(FileTypeName(arch.FileType)))
		})
	})

	Context("State and session persistence", func() {
		newModel := func() *LLama {
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model
		}

		// decodeInto pushes tokens through the model on seqID and returns them.
		decodeInto := func(model *LLama, text string, seqID int32) []int32 {
			tokens := model.Tokenize(text, true, false)
			Expect(tokens).ToNot(BeEmpty())
			batch := NewBatch(len(tokens), 1)
			defer batch.Free()
			for i, tok := range tokens {
				Expect(batch.Add(tok, int32(i), []int32{seqID}, i == len(tokens)-1)).To(Succeed())
			}
			Expect(model.Decode(batch)).To(Equal(0))
			return tokens
		}

		// stepFrom decodes one more token at pos on seqID and returns the
		// resulting logits. Restored state is checked with this rather than by
		// comparing logits buffers directly: llama_context::state_write_data
		// serializes the memory module and the architecture string, and
		// nothing else, so the logits buffer is deliberately not part of a
		// saved state. What must round-trip is the cache — and the way to show
		// that is that the next token predicts identically.
		stepFrom := func(model *LLama, tok int32, pos int32, seqID int32) []float32 {
			batch := NewBatch(1, 1)
			defer batch.Free()
			Expect(batch.Add(tok, pos, []int32{seqID}, true)).To(Succeed())
			Expect(model.Decode(batch)).To(Equal(0))
			return append([]float32(nil), model.Logits(-1)...)
		}

		It("round-trips whole-context state in memory", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			tokens := decodeInto(model, "The capital of France is", 0)

			data, err := model.StateData()
			Expect(err).ToNot(HaveOccurred())
			Expect(int64(len(data))).To(BeNumerically("<=", model.StateSize()))

			// What the model predicts one step past the captured state.
			expected := stepFrom(model, tokens[0], int32(len(tokens)), 0)
			Expect(expected).ToNot(BeEmpty())

			// Disturb the context. The cache is cleared first because
			// decodeInto always starts at position 0, and decoding into a
			// sequence that still holds tokens collides on position.
			model.MemoryClear(true)
			decodeInto(model, "Something else entirely", 0)

			Expect(model.SetStateData(data)).To(Succeed())
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(tokens)-1)), "cache did not come back")

			// Same step from the restored cache must predict the same thing.
			Expect(stepFrom(model, tokens[0], int32(len(tokens)), 0)).To(Equal(expected))
		})

		It("rejects empty state data", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			Expect(model.SetStateData(nil)).To(HaveOccurred())
			Expect(model.SetSequenceStateData(nil, 0)).To(HaveOccurred())
		})

		It("round-trips a session file with its token list", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			tokens := decodeInto(model, "The capital of France is", 0)
			expected := stepFrom(model, tokens[0], int32(len(tokens)), 0)

			// Capture after the extra step so the file holds that cache state.
			path := filepath.Join(GinkgoT().TempDir(), "session.bin")
			Expect(model.SaveSessionFile(path, tokens)).To(Succeed())

			model.MemoryClear(true)
			got, err := model.LoadSessionFile(path)
			Expect(err).ToNot(HaveOccurred())
			Expect(got).To(Equal(tokens))
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(tokens))))

			// The saved cache already includes the extra step, so redoing it
			// would collide; drop it first, then repeat and compare.
			Expect(model.MemorySeqRemove(0, int32(len(tokens)), -1)).To(BeTrue())
			Expect(stepFrom(model, tokens[0], int32(len(tokens)), 0)).To(Equal(expected))
		})

		It("round-trips one sequence without disturbing the others", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(testModelPath, EnableF16Memory, SetContext(256), SetMMap(true), SetNBatch(512), SetNSeqMax(2))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			Expect(model.ContextParams().NSeqMax).To(Equal(2))

			a := decodeInto(model, "The capital of France is", 0)
			decodeInto(model, "The largest ocean is", 1)

			Expect(model.SequenceStateSize(0)).To(BeNumerically(">", int64(0)))
			data, err := model.SequenceStateData(0)
			Expect(err).ToNot(HaveOccurred())
			Expect(data).ToNot(BeEmpty())

			posMax1 := model.MemorySeqPosMax(1)

			// Drop sequence 0 only, then restore it from the checkpoint.
			Expect(model.MemorySeqRemove(0, -1, -1)).To(BeTrue())
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(-1)))

			Expect(model.SetSequenceStateData(data, 0)).To(Succeed())
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(a) - 1)))

			// Sequence 1 was never touched.
			Expect(model.MemorySeqPosMax(1)).To(Equal(posMax1))
		})

		It("restores a sequence file into a different sequence id", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(testModelPath, EnableF16Memory, SetContext(256), SetMMap(true), SetNBatch(512), SetNSeqMax(2))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			Expect(model.ContextParams().NSeqMax).To(Equal(2))

			tokens := decodeInto(model, "The capital of France is", 0)

			path := filepath.Join(GinkgoT().TempDir(), "seq.bin")
			Expect(model.SaveSequenceFile(path, 0, tokens)).To(Succeed())

			model.MemoryClear(true)

			// Saved from sequence 0, restored into sequence 1.
			got, err := model.LoadSequenceFile(path, 1)
			Expect(err).ToNot(HaveOccurred())
			Expect(got).To(Equal(tokens))
			Expect(model.MemorySeqPosMax(1)).To(Equal(int32(len(tokens) - 1)))
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(-1)))
		})

		It("reports an error for a missing or malformed file", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			dir := GinkgoT().TempDir()
			_, err := model.LoadSessionFile(filepath.Join(dir, "does-not-exist.bin"))
			Expect(err).To(HaveOccurred())

			_, err = model.LoadSequenceFile(filepath.Join(dir, "does-not-exist.bin"), 0)
			Expect(err).To(HaveOccurred())

			junk := filepath.Join(dir, "junk.bin")
			Expect(os.WriteFile(junk, []byte("not a state file at all"), 0o600)).To(Succeed())
			_, err = model.LoadSequenceFile(junk, 0)
			Expect(err).To(HaveOccurred())
		})
	})

	Context("Sampler chain introspection", func() {
		newModel := func() *LLama {
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model
		}

		It("reports stage names, count and seeds", func() {
			chain := NewSamplerChain()
			defer chain.Free()

			Expect(chain.Len()).To(Equal(0))
			Expect(chain.Name()).To(Equal("chain"))

			chain.Add(SamplerTopK(40))
			chain.Add(SamplerTemp(0.8))
			chain.Add(SamplerDist(1234))
			Expect(chain.Len()).To(Equal(3))

			Expect(chain.At(0).Name()).To(ContainSubstring("top-k"))
			Expect(chain.At(2).Seed()).To(Equal(uint32(1234)))

			// Stages without randomness report the default seed.
			Expect(chain.At(0).Seed()).To(Equal(DefaultSeed))

			Expect(chain.At(-1)).To(BeNil())
			Expect(chain.At(3)).To(BeNil())
		})

		It("removes a stage and hands over ownership", func() {
			chain := NewSamplerChain()
			defer chain.Free()

			chain.Add(SamplerTopK(40))
			chain.Add(SamplerTemp(0.8))
			Expect(chain.Len()).To(Equal(2))

			removed := chain.Remove(0)
			Expect(removed).ToNot(BeNil())
			Expect(removed.Name()).To(ContainSubstring("top-k"))
			Expect(chain.Len()).To(Equal(1))
			Expect(chain.At(0).Name()).To(ContainSubstring("temp"))

			// Ownership transferred, so this is the caller's to free.
			removed.Free()

			Expect(chain.Remove(5)).To(BeNil())
		})

		It("clones a chain independently", func() {
			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(SamplerTopK(40))
			chain.Add(SamplerDist(99))

			clone := chain.Clone()
			Expect(clone).ToNot(BeNil())
			defer clone.Free()

			Expect(clone.Len()).To(Equal(2))
			Expect(clone.At(1).Seed()).To(Equal(uint32(99)))

			// Mutating the clone leaves the original alone.
			clone.Remove(0).Free()
			Expect(clone.Len()).To(Equal(1))
			Expect(chain.Len()).To(Equal(2))
		})

		It("tolerates an empty sampler", func() {
			var empty Sampler
			Expect(empty.Len()).To(Equal(0))
			Expect(empty.Name()).To(BeEmpty())
			Expect(empty.Seed()).To(Equal(DefaultSeed))
			Expect(empty.Clone()).To(BeNil())
			empty.Free()
		})

		It("builds the model-bound stages", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			_, err := model.Predict("The capital of France is", SetTokens(4))
			Expect(err).ToNot(HaveOccurred())

			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(model.SamplerInfill())
			chain.Add(SamplerAdaptiveP(0.5, 0.9, 7))
			chain.Add(SamplerDist(1))
			Expect(chain.Len()).To(Equal(3))

			tok := chain.Sample(model, -1)
			Expect(tok).To(BeNumerically(">=", int32(0)))
			Expect(tok).To(BeNumerically("<", int32(model.GetModelInfo().VocabSize)))
		})

		It("bans a token with a large negative logit bias", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			_, err := model.Predict("The capital of France is", SetTokens(4))
			Expect(err).ToNot(HaveOccurred())

			// Find what greedy would pick with no bias applied.
			plain := NewSamplerChain()
			plain.Add(SamplerGreedy())
			favourite := plain.Sample(model, -1)
			plain.Free()

			// Ban it, and greedy must choose something else.
			biased := NewSamplerChain()
			defer biased.Free()
			biased.Add(model.SamplerLogitBias([]LogitBias{{Token: favourite, Bias: -1e9}}))
			biased.Add(SamplerGreedy())

			Expect(biased.Sample(model, -1)).ToNot(Equal(favourite))
		})

		It("ignores an empty logit bias list", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(model.SamplerLogitBias(nil))
			Expect(chain.Len()).To(Equal(0))
		})

		It("builds a lazy grammar stage", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			s := model.SamplerGrammarLazy(
				`root ::= "yes" | "no"`, "root",
				[]string{"ANSWER:"}, nil,
			)
			Expect(s).ToNot(BeNil())
			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(s)
			Expect(chain.Len()).To(Equal(1))
		})
	})

	Context("Backend sampling", func() {
		newModel := func() *LLama {
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model
		}

		It("rejects a bare stage and detaches with a nil or empty chain", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			// The engine would read a single stage as if it were a chain.
			stage := SamplerTopK(40)
			defer stage.Free()
			Expect(model.SetSequenceSampler(0, stage)).To(BeFalse())

			// Nothing is attached, so detaching succeeds trivially.
			Expect(model.SetSequenceSampler(0, nil)).To(BeTrue())
			Expect(model.SetSequenceSampler(0, &Sampler{})).To(BeTrue())

			// The engine also detaches on a chain with no stages, but on its
			// own reports false for it.
			empty := NewSamplerChain()
			defer empty.Free()
			Expect(model.SetSequenceSampler(0, empty)).To(BeTrue())
		})

		It("reports no sampled output when no sampler is attached", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			_, err := model.Predict("The capital of France is", SetTokens(4))
			Expect(err).ToNot(HaveOccurred())

			// Without a backend sampler these are all empty, and must say so
			// rather than reading past a null pointer.
			Expect(model.SampledToken(-1)).To(Equal(int32(-1)))
			Expect(model.SampledCandidates(-1)).To(BeNil())
			Expect(model.SampledProbs(-1)).To(BeNil())
			Expect(model.SampledLogits(-1)).To(BeNil())
		})

		It("samples on the backend when a chain is attached", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			// The chain must outlive the attachment, so it is freed last.
			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(SamplerTopK(40))
			chain.Add(SamplerTemp(0.8))
			chain.Add(SamplerDist(1234))

			if !model.SetSequenceSampler(0, chain) {
				Skip("this context was not built for backend sampling")
			}

			tokens := model.Tokenize("The capital of France is", true, false)
			Expect(tokens).ToNot(BeEmpty())
			batch := NewBatch(len(tokens), 1)
			defer batch.Free()
			for i, tok := range tokens {
				Expect(batch.Add(tok, int32(i), []int32{0}, i == len(tokens)-1)).To(Succeed())
			}
			Expect(model.Decode(batch)).To(Equal(0))

			tok := model.SampledToken(-1)
			Expect(tok).To(BeNumerically(">=", int32(0)))
			Expect(tok).To(BeNumerically("<", int32(model.GetModelInfo().VocabSize)))

			// Candidates index the probability array, so when both are present
			// they must be the same length and the token must be among them.
			cands := model.SampledCandidates(-1)
			probs := model.SampledProbs(-1)
			if len(probs) > 0 {
				Expect(cands).To(HaveLen(len(probs)))
				Expect(cands).To(ContainElement(tok))
				for _, p := range probs {
					Expect(p).To(BeNumerically(">=", float32(0)))
					Expect(p).To(BeNumerically("<=", float32(1)))
				}
			}
		})
	})

	Context("Adapters and control vectors", func() {
		newModel := func() *LLama {
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model
		}

		It("reports an empty adapter set on a fresh model", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			Expect(model.LoRACount()).To(Equal(0))

			// Every accessor must reject an index outside the applied set
			// rather than indexing into an empty vector.
			for _, bad := range []int{-1, 0, 99} {
				Expect(model.LoRAMetadata(bad)).To(BeNil())
				Expect(model.LoRAInvocationTokens(bad)).To(BeNil())
				_, ok := model.LoRAMetadataValue(bad, "adapter.lora.alpha")
				Expect(ok).To(BeFalse())
			}
		})

		It("rejects a control vector whose length does not match n_embd", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			nEmbd := model.GetModelInfo().EmbeddingSize
			Expect(nEmbd).To(BeNumerically(">", 0))

			// One element short of a whole layer.
			bad := make([]float32, nEmbd+1)
			Expect(model.SetControlVector(bad, nEmbd, 1, 2)).To(HaveOccurred())
			Expect(model.SetControlVector(bad, 0, 1, 2)).To(HaveOccurred())
		})

		It("applies and clears a control vector", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := newModel()
			defer model.Free()

			info := model.GetModelInfo()
			nEmbd := info.EmbeddingSize
			nLayers := 2

			// A zero vector is a no-op direction, so this exercises the plumbing
			// without asserting anything about what steering does to output.
			vec := make([]float32, nEmbd*nLayers)
			Expect(model.SetControlVector(vec, nEmbd, 1, nLayers)).To(Succeed())

			// Generation still works with a vector applied.
			out, err := model.Predict("The capital of France is", SetTokens(4))
			Expect(err).ToNot(HaveOccurred())
			Expect(out).ToNot(BeEmpty())

			Expect(model.ClearControlVector()).To(Succeed())
		})
	})

	Context("Log routing", func() {
		AfterEach(func() {
			// The engine's logger is global, so always hand it back.
			SetLogHandler(nil)
		})

		It("names the log levels", func() {
			Expect(LogLevelInfo.String()).To(Equal("INFO"))
			Expect(LogLevelError.String()).To(Equal("ERROR"))
			Expect(LogLevelCont.String()).To(Equal("CONT"))
			Expect(LogLevel(42).String()).To(ContainSubstring("42"))
		})

		It("installs and removes the handler", func() {
			SetLogHandler(func(LogLevel, string) {})
			Expect(LogHandlerInstalled()).To(BeTrue())

			// SetLogHandler(nil) hands the logger back to llama.cpp: the engine
			// installs its own stderr default rather than clearing the callback,
			// so "not installed" here means "not ours", not "no logging".
			SetLogHandler(nil)
			Expect(LogHandlerInstalled()).To(BeFalse())
		})

		It("captures the engine's output during a model load", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			var mu sync.Mutex
			var records []string
			var levels []LogLevel

			SetLogHandler(func(l LogLevel, text string) {
				mu.Lock()
				defer mu.Unlock()
				records = append(records, text)
				levels = append(levels, l)
			})

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			model.Free()

			mu.Lock()
			defer mu.Unlock()
			// Loading a model is chatty; if nothing arrived the bridge is not wired.
			Expect(records).ToNot(BeEmpty())
			Expect(strings.Join(records, "")).To(ContainSubstring("llama"))
			for _, l := range levels {
				Expect(l).To(BeNumerically(">=", LogLevelNone))
				Expect(l).To(BeNumerically("<=", LogLevelCont))
			}
		})

		It("stops delivering after the handler is removed", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			var mu sync.Mutex
			count := 0
			SetLogHandler(func(LogLevel, string) {
				mu.Lock()
				defer mu.Unlock()
				count++
			})

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			model.Free()

			mu.Lock()
			afterFirst := count
			mu.Unlock()
			Expect(afterFirst).To(BeNumerically(">", 0))

			SetLogHandler(nil)

			model2, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			model2.Free()

			mu.Lock()
			defer mu.Unlock()
			Expect(count).To(Equal(afterFirst), "handler was called after being removed")
		})
	})

	Context("Model file utilities", func() {
		It("round-trips load mode names", func() {
			for _, m := range []LoadMode{LoadModeNone, LoadModeMmap, LoadModeMlock, LoadModeMmapMlock, LoadModeDirectIO} {
				name := m.String()
				Expect(name).ToNot(BeEmpty())
				Expect(ParseLoadMode(name)).To(Equal(m), "round trip failed for %s", name)
			}
			// An unrecognised name falls back to auto-detection.
			Expect(ParseLoadMode("definitely-not-a-load-mode")).To(Equal(LoadModeAuto))
			// A mode the engine does not define renders numerically rather
			// than reaching llama_load_mode_name, which aborts on it.
			Expect(LoadMode(99).String()).To(Equal("LoadMode(99)"))
		})

		It("builds and parses sharded model paths", func() {
			path := SplitPath("/models/llama", 2, 5)
			Expect(path).ToNot(BeEmpty())
			Expect(path).To(ContainSubstring("llama"))

			// SplitPrefix is the inverse.
			Expect(SplitPrefix(path, 2, 5)).To(Equal("/models/llama"))

			// A path that does not follow the scheme yields nothing.
			Expect(SplitPrefix("/models/plain.gguf", 2, 5)).To(BeEmpty())
		})

		It("reports the tensor override limit", func() {
			Expect(MaxTensorBuftOverrides()).To(BeNumerically(">", 0))
		})

		It("reports an error for an unreadable quantization input", func() {
			err := Quantize(filepath.Join(GinkgoT().TempDir(), "missing.gguf"),
				filepath.Join(GinkgoT().TempDir(), "out.gguf"),
				QuantizeOptions{FileType: 2})
			Expect(err).To(HaveOccurred())
		})

		It("aborts a decode from the callback", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			tokens := model.Tokenize("The capital of France is", true, false)
			Expect(tokens).ToNot(BeEmpty())

			var calls int32
			model.SetAbortCallback(func() bool {
				atomic.AddInt32(&calls, 1)
				return true // abort immediately
			})

			batch := NewBatch(len(tokens), 1)
			defer batch.Free()
			for i, tok := range tokens {
				Expect(batch.Add(tok, int32(i), []int32{0}, i == len(tokens)-1)).To(Succeed())
			}

			// llama.cpp reports 2 for an aborted decode.
			Expect(model.Decode(batch)).To(Equal(2))
			Expect(atomic.LoadInt32(&calls)).To(BeNumerically(">", 0))

			// With the callback removed the same batch decodes normally.
			model.SetAbortCallback(nil)
			model.MemoryClear(true)
			Expect(model.Decode(batch)).To(Equal(0))
		})
	})

	// These inputs used to abort the process: llama.cpp and the binding's own
	// parsing both throw C++ exceptions on malformed values, and an exception
	// crossing into cgo calls std::terminate. Each must now surface as an
	// ordinary Go error or a documented fallback.
	Context("Malformed input does not crash the process", func() {
		It("falls back to auto for an unknown load mode name", func() {
			Expect(ParseLoadMode("definitely-not-a-load-mode")).To(Equal(LoadModeAuto))
			Expect(ParseLoadMode("")).To(Equal(LoadModeAuto))
		})

		It("ignores a non-numeric main GPU", func() {
			model, err := New("not-existing", SetMainGPU("not-a-number"))
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})

		It("ignores a non-numeric tensor split", func() {
			model, err := New("not-existing", SetTensorSplit("a,b,c"))
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})

		It("ignores a malformed logit bias", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			for _, bad := range []string{"5+notanumber", "notatoken+1", "+", "garbage"} {
				out, err := model.Predict("The capital of France is", SetTokens(4), SetLogitBias(bad))
				Expect(err).ToNot(HaveOccurred(), "logit bias %q", bad)
				Expect(out).ToNot(BeEmpty())
			}
		})
	})

	Context("Sharded loading and partial sequence state", func() {
		It("rejects an empty shard list", func() {
			model, err := NewFromSplits(nil)
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})

		It("reports an error for shards that do not exist", func() {
			model, err := NewFromSplits([]string{"no-such-shard-1.gguf", "no-such-shard-2.gguf"})
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})

		It("loads a single-file model through the splits path", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			// One shard is the degenerate case and must behave like New.
			model, err := NewFromSplits([]string{testModelPath}, EnableF16Memory, SetContext(128), SetMMap(true))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()
			Expect(model.GetModelInfo().VocabSize).To(BeNumerically(">", 0))
		})

		It("captures a smaller checkpoint with partial-only state", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			tokens := model.Tokenize("The capital of France is", true, false)
			batch := NewBatch(len(tokens), 1)
			defer batch.Free()
			for i, tok := range tokens {
				Expect(batch.Add(tok, int32(i), []int32{0}, i == len(tokens)-1)).To(Succeed())
			}
			Expect(model.Decode(batch)).To(Equal(0))

			full := model.SequenceStateSizeWith(0, SeqStateAll)
			Expect(full).To(Equal(model.SequenceStateSize(0)), "SeqStateAll must match the default")
			Expect(full).To(BeNumerically(">", int64(0)))

			partial := model.SequenceStateSizeWith(0, SeqStatePartialOnly)
			Expect(partial).To(BeNumerically(">", int64(0)))
			Expect(partial).To(BeNumerically("<=", full))

			data, err := model.SequenceStateDataWith(0, SeqStateAll)
			Expect(err).ToNot(HaveOccurred())
			model.MemoryClear(true)
			Expect(model.SetSequenceStateDataWith(data, 0, SeqStateAll)).To(Succeed())
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(tokens) - 1)))

			Expect(model.SetSequenceStateDataWith(nil, 0, SeqStateAll)).To(HaveOccurred())
		})
	})
	Context("Sampler chain safety", func() {
		// Regression: chain-only operations on a single stage used to reinterpret
		// the stage's context as a chain — Len returned garbage and Remove
		// corrupted the heap — while Perf/PerfReset called llama_perf_sampler,
		// which aborts the whole process on a non-chain. They must now be safe
		// no-ops. None of this needs a loaded model.
		It("treats a single stage's chain operations as safe no-ops", func() {
			s := SamplerTopK(40)
			defer s.Free()

			Expect(s.Len()).To(Equal(0))
			Expect(s.At(0)).To(BeNil())
			Expect(s.Remove(0)).To(BeNil())
			Expect(s.Perf()).To(Equal(SamplerPerf{}))
			Expect(func() { s.PerfReset() }).ToNot(Panic())

			// Add to a non-chain is ignored: the stage is not consumed, so it is
			// still ours to Free (a double-free here would signal a botched guard).
			orphan := SamplerTemp(0.8)
			defer orphan.Free()
			Expect(func() { s.Add(orphan) }).ToNot(Panic())

			Expect(s.Name()).ToNot(BeEmpty())
		})

		It("reports and indexes stages on a real chain", func() {
			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(SamplerTopK(40))
			chain.Add(SamplerTopP(0.95, 1))
			chain.Add(SamplerTemp(0.8))

			Expect(chain.Len()).To(Equal(3))
			Expect(chain.At(0)).ToNot(BeNil())
			Expect(chain.At(2)).ToNot(BeNil())
			Expect(chain.At(3)).To(BeNil())
			Expect(func() { chain.Perf() }).ToNot(Panic())
			Expect(func() { chain.PerfReset() }).ToNot(Panic())
		})

		It("preserves chain semantics across Clone", func() {
			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(SamplerTopK(40))
			chain.Add(SamplerTemp(0.8))

			clone := chain.Clone()
			Expect(clone).ToNot(BeNil())
			defer clone.Free()
			Expect(clone.Len()).To(Equal(chain.Len()))
		})
	})
	Context("Defensive guards (regression)", func() {
		It("does not divide sequence positions by zero", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			_, err = model.Predict("The capital of France is", SetTokens(4))
			Expect(err).ToNot(HaveOccurred())

			// d <= 1 must be a no-op: d == 0 used to reach an integer division by
			// zero in the engine and kill the process (an uncatchable SIGFPE, so
			// the value here is that the test binary survives to the next line).
			Expect(func() { model.MemorySeqDiv(0, 0, -1, 0) }).ToNot(Panic())
			Expect(func() { model.MemorySeqDiv(0, 0, -1, 1) }).ToNot(Panic())
			Expect(model.MemorySeqPosMax(0)).To(BeNumerically(">=", 0))
		})

		It("reports an error when a model cannot be written", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			// Parent directory does not exist, so the write fails; SaveModel must
			// surface that rather than inferring success from a stale file.
			bad := filepath.Join(GinkgoT().TempDir(), "nonexistent-dir", "model.gguf")
			Expect(model.SaveModel(bad)).ToNot(Succeed())
		})

		It("reports an error when state cannot be written", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model, err := New(testModelPath, EnableF16Memory, SetContext(128), SetMMap(true), SetNBatch(512))
			Expect(err).ToNot(HaveOccurred())
			defer model.Free()

			bad := filepath.Join(GinkgoT().TempDir(), "nonexistent-dir", "state.bin")
			Expect(model.SaveState(bad)).ToNot(Succeed())
		})
	})
	// Each spec pins one fix from an audit of the binding. Several of these
	// used to overrun the Go heap, abort the process from inside llama.cpp or
	// throw a C++ exception across cgo, so for those the binary surviving the
	// spec is part of what it checks.
	Context("Binding defect fixes (regression)", func() {
		const prompt = "The capital of France is"

		load := func(opts ...ModelOption) *LLama {
			model, err := New(testModelPath, append([]ModelOption{SetContext(128), SetMMap(true)}, opts...)...)
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model
		}

		// decode pushes text through model on seqID in one batch, requesting
		// output for the last token, and returns the tokens and the status.
		decode := func(model *LLama, text string, seqID int32) ([]int32, int) {
			tokens := model.Tokenize(text, true, false)
			Expect(tokens).ToNot(BeEmpty())
			batch := NewBatch(len(tokens), 1)
			defer batch.Free()
			for i, tok := range tokens {
				Expect(batch.Add(tok, int32(i), []int32{seqID}, i == len(tokens)-1)).To(Succeed())
			}
			return tokens, model.Decode(batch)
		}

		cosine := func(a, b []float32) float64 {
			var dot, na, nb float64
			for i := range a {
				dot += float64(a[i]) * float64(b[i])
				na += float64(a[i]) * float64(a[i])
				nb += float64(b[i]) * float64(b[i])
			}
			return dot / (math.Sqrt(na) * math.Sqrt(nb))
		}

		It("leaves RoPE and thread counts to the model and context by default", func() {
			Expect(DefaultModelOptions.FreqRopeBase).To(BeZero())
			Expect(DefaultModelOptions.FreqRopeScale).To(BeZero())
			Expect(DefaultOptions.Threads).To(BeZero())
			Expect(NewModelOptions().NSeqMax).To(BeZero())
		})

		It("runs the model with its trained RoPE base unless overridden", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			byDefault := load()
			defer byDefault.Free()

			arch, ok := byDefault.ModelMetadataValue("general.architecture")
			Expect(ok).To(BeTrue())
			raw, ok := byDefault.ModelMetadataValue(arch + ".rope.freq_base")
			if !ok {
				Skip("the model does not record its RoPE base")
			}
			trained, err := strconv.ParseFloat(raw, 32)
			Expect(err).ToNot(HaveOccurred())
			if trained == 10000 {
				Skip("the model was trained with the old default base, so the two cannot be told apart")
			}

			// The old default of 10000 replaced the trained base (1e6 for the
			// CodeLlama CI model), so these predicted differently.
			explicit := load(WithRopeFreqBase(float32(trained)))
			defer explicit.Free()

			text := "The capital of France is Paris. The capital of Germany is"
			_, status := decode(byDefault, text, 0)
			Expect(status).To(Equal(0))
			got := append([]float32(nil), byDefault.Logits(-1)...)
			_, status = decode(explicit, text, 0)
			Expect(status).To(Equal(0))
			want := explicit.Logits(-1)

			Expect(got).To(HaveLen(len(want)))
			maxDiff := 0.0
			for i := range want {
				maxDiff = math.Max(maxDiff, math.Abs(float64(got[i]-want[i])))
			}
			Expect(maxDiff).To(BeNumerically("<", 1e-2))
		})

		It("returns one n_embd vector per text from Embeddings", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(EnableEmbeddings)
			defer model.Free()
			nEmbd := model.GetModelInfo().EmbeddingSize

			// The buffer used to be sized from SetTokens (128 by default)
			// while the engine wrote n_embd floats into it.
			a, err := model.Embeddings("The quick brown fox")
			Expect(err).ToNot(HaveOccurred())
			Expect(a).To(HaveLen(nEmbd))
			finite := true
			for _, v := range a {
				finite = finite && !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
			}
			Expect(finite).To(BeTrue())

			// It also returned the first token's row, which for a causal model
			// is the BOS state and so the same for every input.
			b, err := model.Embeddings("An entirely different sentence", SetTokens(0))
			Expect(err).ToNot(HaveOccurred())
			Expect(b).To(HaveLen(nEmbd))
			Expect(b).ToNot(Equal(a))
		})

		It("embeds token ids directly with TokenEmbeddings", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(EnableEmbeddings)
			defer model.Free()

			const text = "The quick brown fox"
			fromText, err := model.Embeddings(text)
			Expect(err).ToNot(HaveOccurred())

			ids := model.Tokenize(text, true, true)
			tokens := make([]int, len(ids))
			for i, id := range ids {
				tokens[i] = int(id)
			}
			// The ids used to be detokenized and tokenized again, BOS and all.
			fromTokens, err := model.TokenEmbeddings(tokens)
			Expect(err).ToNot(HaveOccurred())
			Expect(fromTokens).To(HaveLen(len(fromText)))
			Expect(cosine(fromTokens, fromText)).To(BeNumerically(">", 0.9999))

			for _, bad := range [][]int{nil, {-1}, {model.GetModelInfo().VocabSize}} {
				_, err = model.TokenEmbeddings(bad)
				Expect(err).To(HaveOccurred(), "tokens %v", bad)
			}
		})

		It("rejects input longer than NBatch instead of aborting", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(EnableEmbeddings)
			defer model.Free()

			nBatch := model.ContextParams().NBatch
			long := strings.Repeat("hello ", nBatch+16)
			Expect(len(model.Tokenize(long, true, true))).To(BeNumerically(">", nBatch))
			_, err := model.Embeddings(long)
			Expect(err).To(HaveOccurred())

			// The context is still usable afterwards.
			v, err := model.Embeddings("hello")
			Expect(err).ToNot(HaveOccurred())
			Expect(v).To(HaveLen(model.GetModelInfo().EmbeddingSize))
		})

		It("lets SetEmbeddings change what Embeddings accepts", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			_, err := model.Embeddings("hello")
			Expect(err).To(HaveOccurred())

			model.SetEmbeddings(true)
			v, err := model.Embeddings("hello")
			Expect(err).ToNot(HaveOccurred())
			Expect(v).To(HaveLen(model.GetModelInfo().EmbeddingSize))

			model.SetEmbeddings(false)
			_, err = model.Embeddings("hello")
			Expect(err).To(HaveOccurred())

			// Generation works again once embeddings are off.
			_, err = model.Predict(prompt, SetTokens(2), SetTemperature(0))
			Expect(err).ToNot(HaveOccurred())
		})

		It("constrains Predict to a WithGrammar grammar", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			// The grammar used to run after the token was already picked, and
			// the token was then accepted twice, so an off-grammar pick threw
			// across cgo and aborted the process.
			out, err := model.Predict("[INST] Is the sky blue? Answer yes or no. [/INST]",
				WithGrammar(`root ::= "yes" | "no"`), SetTokens(8))
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(BeElementOf("yes", "no"))

			// A grammar that does not parse is an error, not unconstrained text.
			_, err = model.Predict(prompt, WithGrammar(`root ::= (`), SetTokens(2))
			Expect(err).To(HaveOccurred())
		})

		It("leaves the end-of-generation token out of the result and the callback", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			// Once "yes" or "no" is complete the grammar allows nothing but an
			// end-of-generation token, so generation always ends on one.
			var pieces []string
			out, err := model.Predict("[INST] Is the sky blue? Answer yes or no. [/INST]",
				WithGrammar(`root ::= "yes" | "no"`), SetTokens(8),
				SetTokenCallback(func(piece string) bool {
					pieces = append(pieces, piece)
					return true
				}))
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(BeElementOf("yes", "no"))
			Expect(strings.Join(pieces, "")).To(Equal(out))

			eos := model.TokenToPiece(model.GetSpecialTokens().EOS, true)
			Expect(eos).ToNot(BeEmpty())
			for _, p := range pieces {
				Expect(p).ToNot(ContainSubstring(eos))
			}
		})

		It("trims a stop word from the end of the result as a suffix", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			// The grammar forces "hello", so generation stops on "lo" at its
			// end. Trimmed as a set of characters, the "l" before it went too.
			out, err := model.Predict(prompt, WithGrammar(`root ::= "hello"`), SetStopWords("lo"), SetTokens(8))
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(Equal("hel"))
		})

		It("applies the SetThreads option to one call and restores the context's setting", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()
			model.SetThreads(2, 3)

			var during [2]int
			record := SetTokenCallback(func(string) bool {
				during[0], during[1] = model.Threads()
				return true
			})

			_, err := model.Predict(prompt, SetThreads(1), SetTokens(2), SetTemperature(0), record)
			Expect(err).ToNot(HaveOccurred())
			Expect(during).To(Equal([2]int{1, 1}))
			gen, batch := model.Threads()
			Expect([]int{gen, batch}).To(Equal([]int{2, 3}))

			// Without the option the context's own setting is used.
			during = [2]int{}
			_, err = model.Predict(prompt, SetTokens(2), SetTemperature(0), record)
			Expect(err).ToNot(HaveOccurred())
			Expect(during).To(Equal([2]int{2, 3}))
		})

		It("restores the persistent token callback after a per-call one", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			var persistent, perCall int
			model.SetTokenCallback(func(string) bool {
				persistent++
				return true
			})

			_, err := model.Predict(prompt, SetTokens(2), SetTemperature(0),
				SetTokenCallback(func(string) bool {
					perCall++
					return true
				}))
			Expect(err).ToNot(HaveOccurred())
			Expect(perCall).To(BeNumerically(">", 0))
			Expect(persistent).To(Equal(0), "the per-call callback should take precedence")

			// The option used to delete the persistent callback on its way out.
			_, err = model.Predict(prompt, SetTokens(2), SetTemperature(0))
			Expect(err).ToNot(HaveOccurred())
			Expect(persistent).To(BeNumerically(">", 0))
		})

		It("lets a token callback call SetTokenCallback", func(_ SpecContext) {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			// The callback used to run under the registry's read lock, so this
			// call waited on itself forever. The timeout turns a hang into a
			// failure.
			calls := 0
			_, err := model.Predict(prompt, SetTokens(2), SetTemperature(0),
				SetTokenCallback(func(string) bool {
					calls++
					model.SetTokenCallback(nil)
					return true
				}))
			Expect(err).ToNot(HaveOccurred())
			Expect(calls).To(BeNumerically(">", 0))
		}, SpecTimeout(2*time.Minute))

		It("returns every token from TokenizeString", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			// Far more tokens than the SetTokens default of 128, which used to
			// size the buffer the engine wrote every token into.
			text := strings.Repeat("hello world ", 100)
			n, tokens, err := model.TokenizeString(text)
			Expect(err).ToNot(HaveOccurred())
			Expect(int(n)).To(Equal(len(tokens)))
			Expect(len(tokens)).To(BeNumerically(">", DefaultOptions.Tokens))
			Expect(tokens).To(Equal(model.Tokenize(text, model.GetVocabAddBOS(), true)))
		})

		It("opens sequence ids above 0 with SetNSeqMax", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			single := load()
			defer single.Free()
			Expect(single.ContextParams().NSeqMax).To(Equal(1))
			_, status := decode(single, prompt, 1)
			Expect(status).ToNot(Equal(0), "sequence 1 should not exist in a default context")

			multi := load(SetNSeqMax(2))
			defer multi.Free()
			Expect(multi.ContextParams().NSeqMax).To(Equal(2))
			tokens, status := decode(multi, prompt, 1)
			Expect(status).To(Equal(0))
			Expect(multi.MemorySeqPosMax(1)).To(Equal(int32(len(tokens) - 1)))

			// More sequences than the engine supports fails the load cleanly.
			model, err := New(testModelPath, SetContext(128), SetNSeqMax(MaxParallelSequences()+1))
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())

			// So do more sequences than the batch holds, which is SetNBatch or
			// the context size if that is smaller. The engine reserves an
			// output row per sequence and used to abort when they did not fit.
			model, err = New(testModelPath, SetContext(128), SetNBatch(16), SetNSeqMax(32))
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
			model, err = New(testModelPath, SetContext(128), SetNSeqMax(129))
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})

		It("ignores sequence ids the context does not hold", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			// With more than one sequence the KV cache keeps a stream per
			// sequence and aborts on an id past them, where a single-sequence
			// cache tolerated it.
			model := load(SetNSeqMax(2))
			defer model.Free()
			tokens, status := decode(model, prompt, 0)
			Expect(status).To(Equal(0))

			Expect(model.MemorySeqPosMax(2)).To(Equal(int32(-1)))
			Expect(model.MemorySeqPosMin(2)).To(Equal(int32(-1)))
			Expect(model.MemorySeqRemove(2, -1, -1)).To(BeFalse())
			model.MemorySeqCopy(0, 2, -1, -1)
			model.MemorySeqKeep(2)
			model.MemorySeqAdd(2, -1, -1, 1)
			model.MemorySeqDiv(2, -1, -1, 2)
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(tokens)-1)), "sequence 0 should be untouched")

			data, err := model.SequenceStateData(0)
			Expect(err).ToNot(HaveOccurred())
			Expect(model.SetSequenceStateData(data, 2)).ToNot(Succeed())
			Expect(model.SetSequenceStateDataWith(data, 2, SeqStateAll)).ToNot(Succeed())
			path := filepath.Join(GinkgoT().TempDir(), "seq.bin")
			Expect(model.SaveSequenceFile(path, 0, tokens)).To(Succeed())
			_, err = model.LoadSequenceFile(path, 2)
			Expect(err).To(HaveOccurred())

			// A negative id means every sequence; the engine accepts only -1.
			Expect(model.MemorySeqRemove(-2, -1, -1)).To(BeTrue())
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(-1)))
		})

		It("restores a SaveState file into a fresh context", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()
			tokens, status := decode(model, prompt, 0)
			Expect(status).To(Equal(0))

			path := filepath.Join(GinkgoT().TempDir(), "state.bin")
			Expect(model.SaveState(path)).To(Succeed())

			// An empty context holds less state than the file. LoadState used
			// to read only as much as the loading context held.
			fresh := load()
			defer fresh.Free()
			Expect(fresh.StateSize()).To(BeNumerically("<", model.StateSize()))
			Expect(fresh.LoadState(path)).To(Succeed())
			Expect(fresh.MemorySeqPosMax(0)).To(Equal(int32(len(tokens) - 1)))

			Expect(fresh.LoadState(filepath.Join(GinkgoT().TempDir(), "missing.bin"))).ToNot(Succeed())
		})

		It("detaches a backend sampler with a nil chain", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			// Deferred first so it is freed last: a chain must outlive its
			// attachment.
			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(SamplerTopK(40))
			chain.Add(SamplerDist(1234))

			model := load()
			defer model.Free()

			if !model.SetSequenceSampler(0, chain) {
				Skip("this context was not built for backend sampling")
			}
			_, status := decode(model, prompt, 0)
			Expect(status).To(Equal(0))
			Expect(model.SampledToken(-1)).To(BeNumerically(">=", int32(0)))

			// nil used to return false without reaching the engine, leaving the
			// chain attached.
			Expect(model.SetSequenceSampler(0, nil)).To(BeTrue())
			model.MemoryClear(true)
			_, status = decode(model, prompt, 0)
			Expect(status).To(Equal(0))
			Expect(model.SampledToken(-1)).To(Equal(int32(-1)))
		})

		It("makes a second Free a no-op", func() {
			// A nil model needs no TEST_MODEL.
			var none *LLama
			none.Free()

			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			model.Free()
			// This used to free the context and model a second time; reaching
			// the end of the spec is the check.
			model.Free()
		})

		It("rejects a Decode batch larger than NBatch instead of aborting", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(SetNBatch(16))
			defer model.Free()
			nBatch := model.ContextParams().NBatch
			Expect(nBatch).To(Equal(16))

			tok := model.Tokenize("hello", false, false)[0]
			batch := NewBatch(nBatch+1, 1)
			defer batch.Free()
			for i := 0; i <= nBatch; i++ {
				Expect(batch.Add(tok, int32(i), nil, i == nBatch)).To(Succeed())
			}
			Expect(model.Decode(batch)).To(Equal(-1))
		})

		It("rejects an Encode batch larger than NUbatch instead of aborting", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(SetNBatch(16))
			defer model.Free()
			nUbatch := model.ContextParams().NUbatch
			Expect(nUbatch).To(Equal(16))

			// An encoder pass cannot be split into micro-batches, and the
			// engine asserts on one larger than NUbatch before it looks at
			// whether the model has an encoder at all.
			tok := model.Tokenize("hello", false, false)[0]
			batch := NewBatch(nUbatch+1, 1)
			defer batch.Free()
			for i := 0; i <= nUbatch; i++ {
				Expect(batch.Add(tok, int32(i), nil, true)).To(Succeed())
			}
			Expect(model.Encode(batch)).To(Equal(-1))
		})

		It("feeds Predict a prompt longer than NBatch in chunks", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(SetContext(256), SetNBatch(16))
			defer model.Free()

			long := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 4)
			Expect(len(model.Tokenize(long, true, false))).To(BeNumerically(">", model.ContextParams().NBatch))

			// The default SetBatch(512) used to go to the engine as one batch
			// of the whole prompt, which it aborts on.
			_, err := model.Predict(long, SetTokens(2), SetTemperature(0))
			Expect(err).ToNot(HaveOccurred())
		})

		It("applies DRY with the default look-back window of -1", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()

			// llama.cpp now clamps a negative window to 0, which disables the
			// stage; the binding resolves -1 to the context size first.
			byDefault := model.SamplerDRY(0.8, 1.75, 2, -1)
			defer byDefault.Free()
			Expect(byDefault.Name()).To(Equal("dry"))
			disabled := model.SamplerDRY(0.8, 1.75, 2, 0)
			defer disabled.Free()
			Expect(disabled.Name()).To(Equal("?dry"))
			// The engine allocates the whole window up front, so this one
			// asked for about 16 GiB and ended the process. It is cut to the
			// context size.
			huge := model.SamplerDRY(0.8, 1.75, 2, math.MaxInt32)
			defer huge.Free()
			Expect(huge.Name()).To(Equal("dry"))

			// Predict builds its DRY stage the same way. Top-k 1 makes the
			// output deterministic, and with only 8 tokens generated a window
			// of 64 behaves exactly like one the size of the context.
			repetitive := strings.TrimSpace(strings.Repeat("dog ", 16))
			predict := func(lastN int) string {
				out, err := model.Predict(repetitive, SetTokens(8), SetTemperature(0.5), SetTopK(1),
					SetDRYMultiplier(50), SetDRYAllowedLength(1), SetDRYPenaltyLastN(lastN))
				Expect(err).ToNot(HaveOccurred())
				return out
			}
			window := predict(64)
			if window == predict(0) {
				Skip("DRY does not change this model's output for the prompt")
			}
			Expect(predict(-1)).To(Equal(window))
		})

		It("renders token ids outside the vocabulary as nothing instead of aborting", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()
			nVocab := int32(model.GetModelInfo().VocabSize)
			valid := model.Tokenize("hello", false, false)
			Expect(valid).ToNot(BeEmpty())
			Expect(model.TokenToPiece(valid[0], true)).ToNot(BeEmpty())
			Expect(model.Detokenize(valid, false, false)).ToNot(BeEmpty())

			// llama.cpp throws std::out_of_range for each of these, which
			// aborted the process from inside cgo.
			for _, tok := range []int32{-1, nVocab, nVocab + 1000} {
				Expect(model.TokenToPiece(tok, true)).To(BeEmpty())
				Expect(model.TokenToPiece(tok, false)).To(BeEmpty())
				Expect(model.Detokenize([]int32{tok}, false, true)).To(BeEmpty())
				withBad := append(append([]int32(nil), valid...), tok)
				Expect(model.Detokenize(withBad, false, false)).To(BeEmpty())
				Expect(model.IsEOG(tok)).To(BeFalse())
				Expect(model.IsControlToken(tok)).To(BeFalse())
			}
		})

		It("reports no sequence state for ids the context does not hold", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(SetNSeqMax(2))
			defer model.Free()
			tokens, status := decode(model, prompt, 0)
			Expect(status).To(Equal(0))
			Expect(model.SequenceStateSize(0)).To(BeNumerically(">", 0))
			Expect(model.SequenceStateSize(-1)).To(BeNumerically(">", 0), "-1 means every sequence")

			// The engine described 2, past NSeqMax, as an empty sequence and
			// wrote it out. Past LLAMA_MAX_SEQ (256), the KV cache's sequence
			// bitset throws std::out_of_range. -2 is neither a sequence nor -1.
			dir := GinkgoT().TempDir()
			for _, seq := range []int32{2, 300, -2} {
				Expect(model.SequenceStateSize(seq)).To(BeZero())
				Expect(model.SequenceStateSizeWith(seq, SeqStatePartialOnly)).To(BeZero())
				_, err := model.SequenceStateData(seq)
				Expect(err).To(HaveOccurred())
				_, err = model.SequenceStateDataWith(seq, SeqStateAll)
				Expect(err).To(HaveOccurred())
				path := filepath.Join(dir, "seq"+strconv.Itoa(int(seq))+".bin")
				Expect(model.SaveSequenceFile(path, seq, tokens)).ToNot(Succeed())
				Expect(path).ToNot(BeAnExistingFile())
			}
			Expect(model.MemorySeqPosMax(0)).To(Equal(int32(len(tokens)-1)), "sequence 0 should be untouched")
		})

		It("refuses sampler tokens a stage cannot take instead of aborting", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load()
			defer model.Free()
			_, status := decode(model, prompt, 0)
			Expect(status).To(Equal(0))
			nVocab := int32(model.GetModelInfo().VocabSize)
			offGrammar := model.Tokenize(" maybe", false, false)
			Expect(offGrammar).ToNot(BeEmpty())
			const yesNo = `root ::= "yes" | "no"`

			// A grammar stage threw std::out_of_range for an id outside the
			// vocabulary and std::runtime_error for a token its grammar does
			// not allow, both across cgo, and aborted on end-of-generation
			// before its grammar was complete. Accept refuses them all, so
			// the grammar has not moved and still constrains Sample.
			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(model.SamplerGrammar(yesNo, "root"))
			chain.Add(SamplerGreedy())
			Expect(chain.Len()).To(Equal(2))
			for _, tok := range []int32{-1, nVocab, nVocab + 1000, offGrammar[0], model.GetSpecialTokens().EOS} {
				chain.Accept(tok)
			}
			tok := chain.Sample(model, -1)
			Expect([]string{"y", "ye", "yes", "n", "no"}).To(ContainElement(model.TokenToPiece(tok, true)))

			// With the grammar after the stage that picks, a pick it does not
			// allow threw from inside Sample, and left the grammar in a state
			// llama.cpp asserts on the next time it is applied. The bias
			// forces such a pick each time; the second Sample shows the
			// grammar was restarted.
			misordered := NewSamplerChain()
			defer misordered.Free()
			misordered.Add(model.SamplerLogitBias([]LogitBias{{Token: offGrammar[0], Bias: 1000}}))
			misordered.Add(SamplerGreedy())
			misordered.Add(model.SamplerGrammar(yesNo, "root"))
			Expect(misordered.Len()).To(Equal(3))
			Expect(misordered.Sample(model, -1)).To(Equal(int32(-1)))
			Expect(misordered.Sample(model, -1)).To(Equal(int32(-1)))

			// An adaptive-p stage that has not sampled yet asserted on -1,
			// and Sample on an empty Sampler (an unparsable grammar)
			// dereferenced a null pointer.
			adaptive := SamplerAdaptiveP(0.5, 0.9, 42)
			defer adaptive.Free()
			adaptive.Accept(-1)
			empty := model.SamplerGrammar("not a grammar", "root")
			Expect(empty.Sample(model, -1)).To(Equal(int32(-1)))
		})

		It("restores on-device sequence state only from its latest capture", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}
			model := load(SetNSeqMax(2))
			defer model.Free()
			tokens, status := decode(model, prompt, 0)
			Expect(status).To(Equal(0))
			last := int32(len(tokens) - 1)

			// No on-device capture of sequence 0 exists yet; llama.cpp
			// asserted on these host bytes restored with SeqStateOnDevice.
			host, err := model.SequenceStateDataWith(0, SeqStateAll)
			Expect(err).ToNot(HaveOccurred())
			Expect(model.SetSequenceStateDataWith(host, 1, SeqStateOnDevice)).ToNot(Succeed())

			first, err := model.SequenceStateDataWith(0, SeqStateOnDevice)
			Expect(err).ToNot(HaveOccurred())
			Expect(model.SetSequenceStateDataWith(first, 1, SeqStateOnDevice)).To(Succeed())
			Expect(model.MemorySeqPosMax(1)).To(Equal(last))

			// Other flags, or bytes that are not the capture's, are refused.
			Expect(model.SetSequenceStateDataWith(first, 1, SeqStateOnDevice|SeqStatePartialOnly)).ToNot(Succeed())
			tampered := append([]byte(nil), first...)
			tampered[len(tampered)-1] ^= 0xff
			Expect(model.SetSequenceStateDataWith(tampered, 1, SeqStateOnDevice)).ToNot(Succeed())
			Expect(model.SetSequenceStateDataWith(first[:4], 1, SeqStateOnDevice)).ToNot(Succeed())

			// A new capture of sequence 0 replaces the context's snapshot. The
			// first bytes describe one cell fewer than it now holds, which
			// llama.cpp aborted on.
			batch := NewBatch(1, 1)
			defer batch.Free()
			Expect(batch.Add(tokens[0], last+1, []int32{0}, true)).To(Succeed())
			Expect(model.Decode(batch)).To(Equal(0))
			second, err := model.SequenceStateDataWith(0, SeqStateOnDevice)
			Expect(err).ToNot(HaveOccurred())
			Expect(model.SetSequenceStateDataWith(first, 1, SeqStateOnDevice)).ToNot(Succeed())
			Expect(model.SetSequenceStateDataWith(second, 1, SeqStateOnDevice)).To(Succeed())
			Expect(model.MemorySeqPosMax(1)).To(Equal(last + 1))
		})
	})
	Context("Inferencing tests with GPU (using "+testModelPath+") ", Label("gpu"), func() {
		getModel := func() (*LLama, error) {
			model, err := New(
				testModelPath,
				llama.EnableF16Memory, llama.SetContext(128), llama.EnableEmbeddings, llama.SetGPULayers(10),
			)
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			return model, err
		}

		It("predicts successfully", func() {
			if testModelPath == "" {
				Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
			}

			model, err := getModel()
			text, err := model.Predict(`[INST] Answer to the following question:
how much is 2+2?
[/INST]`)
			Expect(err).ToNot(HaveOccurred(), text)
			Expect(text).To(ContainSubstring("4"), text)
		})
	})
})
