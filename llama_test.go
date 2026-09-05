package llama_test

import (
	"os"

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
			_, err = model.Predict("The capital of France is")
			Expect(err).ToNot(HaveOccurred())

			chain := NewSamplerChain()
			defer chain.Free()
			chain.Add(SamplerTopK(40))
			chain.Add(SamplerTopP(0.95, 1))
			chain.Add(SamplerTemp(0.8))
			chain.Add(SamplerDist(1234))

			tok := chain.Sample(model, -1)
			Expect(tok).To(BeNumerically(">=", 0))
			Expect(tok).To(BeNumerically("<", int32(model.GetModelInfo().VocabSize)))
			chain.Accept(tok)
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
