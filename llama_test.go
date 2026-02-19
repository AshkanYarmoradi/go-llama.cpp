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
	})

	Context("System info", func() {
		It("returns system info string", func() {
			info := llama.SystemInfo()
			Expect(info).ToNot(BeEmpty())
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
