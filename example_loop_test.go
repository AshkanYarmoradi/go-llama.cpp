package llama_test

import (
	"errors"
	"fmt"
	"log"
	"strings"

	llama "github.com/AshkanYarmoradi/go-llama.cpp"
)

// generate decodes prompt, then samples up to maxTokens tokens from chain,
// stopping at the model's end-of-generation token.
func generate(model *llama.LLama, chain *llama.Sampler, prompt string, maxTokens int) (string, error) {
	model.MemoryClear(true) // start from an empty cache, as Predict does

	tokens := model.Tokenize(prompt, true, true) // add BOS, parse chat-template markup
	if len(tokens) == 0 {
		return "", errors.New("empty prompt")
	}
	if n := model.ContextParams().NBatch; len(tokens) > n {
		return "", fmt.Errorf("prompt is %d tokens; one Decode takes at most %d", len(tokens), n)
	}

	batch := llama.NewBatch(len(tokens), 1)
	defer batch.Free()
	for i, t := range tokens {
		// Only the last prompt token needs logits: it is the one sampled from.
		if err := batch.Add(t, int32(i), []int32{0}, i == len(tokens)-1); err != nil {
			return "", err
		}
	}

	var out strings.Builder
	pos := int32(len(tokens))
	for range maxTokens {
		switch rc := model.Decode(batch); rc {
		case 0:
		case 1:
			return out.String(), errors.New("the context is full")
		case 2:
			return out.String(), errors.New("stopped by the abort callback")
		default:
			return out.String(), fmt.Errorf("decode failed with status %d", rc)
		}

		next := chain.Sample(model, -1) // Sample also accepts the token: no Accept call
		if next < 0 {
			return out.String(), errors.New("the sampler gave no token; the reason is on stderr")
		}
		if model.IsEOG(next) {
			break
		}
		out.WriteString(model.TokenToPiece(next, false))

		batch.Reset()
		if err := batch.Add(next, pos, []int32{0}, true); err != nil {
			return out.String(), err
		}
		pos++
	}
	return out.String(), nil
}

// The low-level API is batches, Decode and a sampler chain. This file builds a
// complete generation loop from them, with Predict's default stages in
// Predict's order.
func Example_generationLoop() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	chain := llama.NewSamplerChain()
	defer chain.Free() // frees every stage too
	chain.Add(model.SamplerPenalties(64, 1.1, 0, 0))
	chain.Add(llama.SamplerTopK(40))
	chain.Add(llama.SamplerTopP(0.95, 1))
	chain.Add(llama.SamplerMinP(0.05, 1))
	chain.Add(llama.SamplerTemp(0.8))
	chain.Add(llama.SamplerDist(llama.DefaultSeed)) // DefaultSeed asks for a random seed

	text, err := generate(model, chain, "[INST] Name three Go proverbs. [/INST]", 128)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)
}
