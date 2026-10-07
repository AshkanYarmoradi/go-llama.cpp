package llama_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	llama "github.com/AshkanYarmoradi/go-llama.cpp"
)

// The smallest useful program: load a GGUF model, ask it something, free it.
// CI's "predicts successfully" spec sends this prompt to CodeLlama-7B-Instruct
// (quantized to Q2_K) and checks the answer has a 4.
func Example() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	answer, err := model.Predict("[INST] Answer to the following question:\nhow much is 2+2?\n[/INST]",
		llama.SetTokens(32))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}

// New loads a model and creates a context for it. GPU offload is opt-in, so
// ask for it only when llama.cpp finds a GPU it can offload to.
func ExampleNew() {
	opts := []llama.ModelOption{
		llama.SetContext(4096), // prompt plus reply, in tokens; rounded up to a multiple of 256
	}
	if llama.SupportsGPUOffload() {
		opts = append(opts, llama.SetGPULayers(99)) // more layers than the model has offloads all of them
	}

	// For a sharded model, pass the first shard (name-00001-of-00003.gguf)
	// and llama.cpp finds the rest.
	model, err := llama.New("model.gguf", opts...)
	if err != nil {
		log.Fatal(err) // failed loading model "model.gguf"
	}
	defer model.Free() // the garbage collector never frees a model

	model.SetThreads(runtime.NumCPU(), runtime.NumCPU()) // the default is 4
	fmt.Println(model.GetModelInfo().Description)
}

// NewFromSplits takes the shards in order, for files that do not follow
// llama.cpp's "-00001-of-0000N.gguf" naming. An empty list is an error.
func ExampleNewFromSplits() {
	model, err := llama.NewFromSplits([]string{"part-a.gguf", "part-b.gguf"}, llama.SetContext(4096))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()
	fmt.Println(model.GetModelInfo().Description)
}

// Predict runs a whole generation in one call. Options tune that call only.
func ExampleLLama_Predict() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	// Each call starts from an empty KV cache: send the whole conversation
	// every time.
	out, err := model.Predict("Write a haiku about Go:",
		llama.SetTokens(64),       // at most 64 new tokens
		llama.SetTemperature(0.7), // lower is more focused; 0 or below always picks the likeliest token
		llama.SetTopP(0.9),
		llama.SetSeed(42), // repeatable on the same model, build and machine; the default -1 is random
	)
	if err != nil {
		// The error only says "inference failed". The binding prints the reason
		// to stderr; llama.cpp's own log lines go to the handler set with
		// SetLogHandler, if there is one.
		log.Fatal(err)
	}
	fmt.Println(out) // the generated text, without the end-of-generation token
}

// WithGrammar constrains Predict's output to a GBNF grammar whose start rule
// is root. A grammar that does not parse makes Predict return an error.
func ExampleWithGrammar() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	answer, err := model.Predict("[INST] Is the sky blue? Answer yes or no. [/INST]",
		llama.WithGrammar(`root ::= "yes" | "no"`),
		llama.SetTokens(8),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer) // yes or no, and nothing else
}

// DRY ("don't repeat yourself") penalises a token that would extend a repeat
// of text already generated. It is off by default.
func ExampleSetDRYMultiplier() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	out, err := model.Predict("List twenty different fruits, one per line:",
		llama.SetTokens(256),
		llama.SetTemperature(0.7),    // Predict leaves DRY out at a temperature of 0 or below
		llama.SetDRYMultiplier(0.8),  // 0, the default, turns DRY off
		llama.SetDRYPenaltyLastN(-1), // the default window, -1, is the context size; 0 turns DRY off
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(out)
}

// Stream the reply as it is generated. The callback applies to this call only.
func ExampleSetTokenCallback() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	_, err = model.Predict("Tell me a short story about a robot:",
		llama.SetTokens(256),
		// The callback runs inside Predict, with no lock held. It may use
		// other models, but must not call Predict, Decode or Free on this one.
		llama.SetTokenCallback(func(piece string) bool {
			fmt.Print(piece) // a piece can end in the middle of a multi-byte character
			return true      // false stops generation
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
}

// An abort callback lets a context.Context deadline stop the engine in the
// middle of a token, which a token callback cannot do.
func ExampleLLama_SetAbortCallback() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Called often, from the engine's compute thread: keep it cheap.
	model.SetAbortCallback(func() bool { return ctx.Err() != nil })
	defer model.SetAbortCallback(nil)

	_, err = model.Predict("Count to one million:", llama.SetTokens(0)) // 0 means no token limit
	if err != nil {
		if ctx.Err() != nil {
			fmt.Println("stopped:", ctx.Err()) // a timeout, not a failure
			return
		}
		log.Fatal(err)
	}
}

// Render a conversation with the model's own chat template, falling back to a
// built-in one when the GGUF file has none llama.cpp recognises.
func ExampleLLama_ApplyChatTemplate() {
	model, err := llama.New("model.gguf", llama.SetContext(4096))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	msgs := []llama.ChatMessage{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "How much is 2+2?"},
	}
	prompt, err := model.ApplyChatTemplate("", msgs, true) // "" uses the template in the GGUF file
	if errors.Is(err, llama.ErrNoChatTemplate) {
		prompt, err = model.ApplyChatTemplate("chatml", msgs, true)
	}
	if err != nil {
		log.Fatal(err)
	}

	reply, err := model.Predict(prompt, llama.SetTokens(64))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(reply)
}

// Any of these names can be passed to ApplyChatTemplate. No model is needed.
func ExampleBuiltinChatTemplates() {
	names := llama.BuiltinChatTemplates()
	fmt.Println(slices.Contains(names, "chatml"))
	// Output: true
}

// Route llama.cpp's log output into log/slog. The handler is global: it sees
// every model in the process.
func ExampleSetLogHandler() {
	levels := map[llama.LogLevel]slog.Level{
		llama.LogLevelDebug: slog.LevelDebug,
		llama.LogLevelInfo:  slog.LevelInfo,
		llama.LogLevelWarn:  slog.LevelWarn,
		llama.LogLevelError: slog.LevelError,
	}

	// llama.cpp builds some lines in pieces, sending the rest of a line as
	// LogLevelCont records. Collect them and log whole lines.
	var (
		mu    sync.Mutex // the engine logs from its own threads
		level llama.LogLevel
		line  strings.Builder
	)
	llama.SetLogHandler(func(l llama.LogLevel, text string) {
		mu.Lock()
		defer mu.Unlock()
		if l != llama.LogLevelCont {
			level = l
		}
		line.WriteString(text)
		if strings.HasSuffix(text, "\n") {
			slog.Log(context.Background(), levels[level], strings.TrimSuffix(line.String(), "\n"))
			line.Reset()
		}
	})
	defer llama.SetLogHandler(nil) // nil hands the output back to stderr

	// To silence llama.cpp instead, install a handler that does nothing. The
	// binding's own few lines, such as "loading model from", still go to stderr.
	// llama.SetLogHandler(func(llama.LogLevel, string) {})

	model, err := llama.New("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	fmt.Println(llama.LogHandlerInstalled()) // true while a handler is set
}

// Tokenize and Detokenize convert between text and token ids. TokenToPiece
// shows the text of one token.
func ExampleLLama_Tokenize() {
	model, err := llama.New("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	tokens := model.Tokenize("The quick brown fox", true, false) // add BOS; treat markup as plain text
	for _, t := range tokens {
		fmt.Printf("%d %q\n", t, model.TokenToPiece(t, false))
	}
	fmt.Println(model.Detokenize(tokens, true, false)) // drop BOS again

	// Ids run from 0 to VocabSize-1. One outside that range has no text:
	// TokenToPiece returns "", and so does Detokenize for a list holding one.
	outside := int32(model.GetModelInfo().VocabSize)
	fmt.Printf("%q\n", model.TokenToPiece(outside, false)) // ""
}

// Embeddings returns one vector for a whole text. Load the model with
// EnableEmbeddings, or switch a loaded one with SetEmbeddings(true).
func ExampleLLama_Embeddings() {
	model, err := llama.New("embedding-model.gguf", llama.EnableEmbeddings)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	// Each text is decoded in one batch, so it may be at most
	// ContextParams().NBatch tokens long; longer text is an error.
	a, err := model.Embeddings("The quick brown fox")
	if err != nil {
		log.Fatal(err)
	}
	b, err := model.Embeddings("A fast auburn fox")
	if err != nil {
		log.Fatal(err)
	}

	// The vector is as wide as the model's output embeddings. An embedding
	// model pools every token into it (mean, cls or last); a model that does
	// not pool, such as a chat model, returns its last token's embedding.
	fmt.Println(len(a) == model.Architecture().EmbdOut, model.ContextParams().Pooling)

	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	fmt.Printf("cosine similarity: %.2f\n", dot/math.Sqrt(na*nb))
}

// What the GGUF file says about the model, next to what the context was
// actually given.
func ExampleLLama_GetModelInfo() {
	model, err := llama.New("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	info := model.GetModelInfo()
	fmt.Println(info.Description)
	fmt.Println("parameters:", info.ParamCount, "layers:", info.LayerCount)
	fmt.Println("trained context:", info.ContextLength, "loaded context:", model.ContextParams().NCtx)
	if arch, ok := model.ModelMetadataValue("general.architecture"); ok {
		fmt.Println("architecture:", arch)
	}
}

// A context runs on 4 threads unless told otherwise. Generation (one token at
// a time) and prompt processing (whole batches) take separate counts.
func ExampleLLama_SetThreads() {
	model, err := llama.New("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	model.SetThreads(runtime.NumCPU(), runtime.NumCPU())
	gen, batch := model.Threads()
	fmt.Println("generation:", gen, "batch:", batch)

	// The SetThreads PredictOption changes the count for one call. The
	// context's own setting is back in place afterwards.
	if _, err := model.Predict("Hello", llama.SetThreads(2), llama.SetTokens(8)); err != nil {
		log.Fatal(err)
	}
}

// Check at startup what the binary was built with, rather than finding out
// from a slow first reply.
func ExampleSupportsGPUOffload() {
	log.Printf("llama.cpp %s", llama.Version())
	log.Print(llama.SystemInfo()) // CPU features and compiled-in backends

	layers := 99 // more than most models have, so all of them
	if !llama.SupportsGPUOffload() {
		// A CPU-only build, or a GPU build that found no usable device. To
		// offload, rebuild libbinding.a with BUILD_TYPE=cublas, hipblas or
		// vulkan and build Go with the tag of the same name (-tags cublas).
		// Apple builds include Metal by default. A GPU build never gets here
		// without its backend's libraries, such as the NVIDIA driver's
		// libcuda.so.1 for cublas: it does not start.
		log.Print("no usable GPU (CPU-only build, or no GPU device found); running on the CPU")
		layers = 0
	}

	model, err := llama.New("model.gguf", llama.SetGPULayers(layers))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()
}

// Perplexity measures how well a model predicts a text: lower is better.
// Requesting logits for every token gets all the predictions from one Decode.
func Example_perplexity() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	tokens := model.Tokenize("The quick brown fox jumps over the lazy dog.", true, false)
	if n := model.ContextParams().NBatch; len(tokens) < 2 || len(tokens) > n {
		log.Fatalf("need between 2 and %d tokens, got %d", n, len(tokens)) // one Decode takes at most NBatch
	}

	batch := llama.NewBatch(len(tokens), 1)
	defer batch.Free()
	for i, t := range tokens {
		if err := batch.Add(t, int32(i), []int32{0}, true); err != nil {
			log.Fatal(err)
		}
	}
	if rc := model.Decode(batch); rc != 0 {
		log.Fatalf("decode: status %d", rc)
	}

	// Logits(i) is the prediction for token i+1. Average its negative
	// log-softmax probability over the text.
	var nll float64
	for i := range len(tokens) - 1 {
		logits := model.Logits(i)
		peak := slices.Max(logits)
		var sum float64
		for _, l := range logits {
			sum += math.Exp(float64(l - peak))
		}
		nll -= float64(logits[tokens[i+1]]-peak) - math.Log(sum)
	}
	fmt.Printf("perplexity: %.2f\n", math.Exp(nll/float64(len(tokens)-1)))
}

// A chain owns the stages added to it, and Free on the chain frees them all.
// No model is needed to build one.
func ExampleNewSamplerChain() {
	chain := llama.NewSamplerChain()
	defer chain.Free()

	chain.Add(llama.SamplerTopK(40))
	chain.Add(llama.SamplerTemp(0.8))
	chain.Add(llama.SamplerDist(1234)) // the last stage picks the token

	fmt.Println(chain.Len(), chain.At(2).Seed())
	// Output: 3 1234
}

// Clone copies a chain, every stage and its state included. Clone and Remove
// hand over a sampler you must Free; At only lends one.
func ExampleSampler_Clone() {
	chain := llama.NewSamplerChain()
	defer chain.Free()
	chain.Add(llama.SamplerTopK(40))
	chain.Add(llama.SamplerDist(99))

	clone := chain.Clone()
	defer clone.Free()
	fmt.Println(clone.Len(), clone.At(1).Seed())

	clone.Remove(0).Free() // the original keeps both stages
	fmt.Println(clone.Len(), chain.Len())
	// Output:
	// 2 99
	// 1 2
}

// A batch has a fixed capacity, and Add reports a full batch instead of
// writing past it.
func ExampleBatch_Add() {
	batch := llama.NewBatch(1, 1) // one token, one sequence
	defer batch.Free()

	fmt.Println(batch.Add(1, 0, []int32{0}, false))
	fmt.Println(batch.Add(2, 1, []int32{0}, false))
	fmt.Println(batch.Len())
	// Output:
	// <nil>
	// batch is full (capacity 1)
	// 1
}

// TokenAttr is a bitmask. Has tests bits; String names the ones set.
func ExampleTokenAttr_Has() {
	a := llama.TokenAttrControl | llama.TokenAttrByte
	fmt.Println(a.Has(llama.TokenAttrControl), a.Has(llama.TokenAttrNormal), a)
	// Output: true false control|byte
}

// SplitPrefix is the inverse of SplitPath. It returns "" for a file name that
// does not follow the shard naming scheme.
func ExampleSplitPrefix() {
	shard := llama.SplitPath("/models/llama", 2, 5) // shard numbers count from 0
	fmt.Printf("%q\n", llama.SplitPrefix(shard, 2, 5))
	fmt.Printf("%q\n", llama.SplitPrefix("/models/plain.gguf", 2, 5))
	// Output:
	// "/models/llama"
	// ""
}

// Ban a token: a large negative bias, placed first in the chain, hides it from
// every later stage.
func ExampleLLama_SamplerLogitBias() {
	model, err := llama.New("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	// Predict leaves the logits of its last step in the context.
	if _, err := model.Predict("The capital of France is", llama.SetTokens(4)); err != nil {
		log.Fatal(err)
	}

	greedy := llama.NewSamplerChain()
	greedy.Add(llama.SamplerGreedy())
	favourite := greedy.Sample(model, -1)
	greedy.Free()

	banned := llama.NewSamplerChain()
	defer banned.Free()
	banned.Add(model.SamplerLogitBias([]llama.LogitBias{{Token: favourite, Bias: -1e9}}))
	banned.Add(llama.SamplerGreedy())

	fmt.Println(banned.Sample(model, -1) != favourite) // true
}

// A grammar stage goes first in a chain, ahead of truncation stages such as
// top-k and of the stage that picks the token, so every pick is one the
// grammar allows. Placed later, it can be handed a token it refuses: Sample
// then returns -1 and restarts the grammar, or ends the process if that token
// is an end-of-generation token.
func ExampleLLama_SamplerGrammar() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	chain := llama.NewSamplerChain()
	defer chain.Free() // before model.Free: the grammar stage was built from the model
	chain.Add(model.SamplerGrammar(`root ::= "yes" | "no"`, "root"))
	if chain.Len() == 0 {
		// A grammar that does not parse gives an empty stage, which Add ignores.
		log.Fatal("the grammar does not parse")
	}
	chain.Add(llama.SamplerTopK(40))
	chain.Add(llama.SamplerTemp(0.8))
	chain.Add(llama.SamplerDist(llama.DefaultSeed))

	// generate, from the generationLoop example, samples until the model ends
	// its turn, which the grammar allows only after "yes" or "no".
	answer, err := generate(model, chain, "[INST] Is the sky blue? Answer yes or no. [/INST]", 8)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer)
}

// When the KV cache is full, drop the oldest tokens after a kept prefix and
// slide the rest back. Predict does this itself; a hand-written loop has to.
func ExampleLLama_MemorySeqAdd() {
	model, err := llama.New("model.gguf", llama.SetContext(2048))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	tokens := model.Tokenize("Once upon a time, in a land far away, there lived a dragon.", true, false)
	batch := llama.NewBatch(len(tokens), 1)
	defer batch.Free()
	for i, t := range tokens {
		if err := batch.Add(t, int32(i), []int32{0}, i == len(tokens)-1); err != nil {
			log.Fatal(err)
		}
	}
	if rc := model.Decode(batch); rc != 0 {
		log.Fatalf("decode: status %d", rc)
	}

	if !model.MemoryCanShift() {
		log.Fatal("this model's cache cannot shift positions")
	}
	const keep = 1 // the BOS token, or a whole system prompt
	past := model.MemorySeqPosMax(0) + 1
	discard := (past - keep) / 2
	model.MemorySeqRemove(0, keep, keep+discard)
	model.MemorySeqAdd(0, keep+discard, -1, -discard)

	// The next token goes straight after the shifted ones.
	fmt.Println("next position:", model.MemorySeqPosMax(0)+1, "was:", past)
}

// A session file holds the KV cache and the tokens behind it, so another
// process with the same model and options can carry on from there.
func ExampleLLama_SaveSessionFile() {
	model, err := llama.New("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Free()

	tokens := model.Tokenize("You are a helpful assistant.", true, false)
	batch := llama.NewBatch(len(tokens), 1)
	defer batch.Free()
	for i, t := range tokens {
		if err := batch.Add(t, int32(i), []int32{0}, i == len(tokens)-1); err != nil {
			log.Fatal(err)
		}
	}
	if rc := model.Decode(batch); rc != 0 {
		log.Fatalf("decode: status %d", rc)
	}

	path := filepath.Join(os.TempDir(), "session.bin")
	if err := model.SaveSessionFile(path, tokens); err != nil {
		log.Fatal(err)
	}

	// Later, starting from an empty cache as a new process would:
	model.MemoryClear(true)
	restored, err := model.LoadSessionFile(path)
	if err != nil {
		log.Fatal(err)
	}

	// The file holds the cache but not the logits. To sample, drop the last
	// cached token and decode it again.
	last := int32(len(restored) - 1)
	if !model.MemorySeqRemove(0, last, -1) {
		log.Fatal("cannot remove the last token")
	}
	batch.Reset()
	if err := batch.Add(restored[last], last, []int32{0}, true); err != nil {
		log.Fatal(err)
	}
	if rc := model.Decode(batch); rc != 0 {
		log.Fatalf("decode: status %d", rc)
	}

	chain := llama.NewSamplerChain()
	defer chain.Free()
	chain.Add(llama.SamplerGreedy())
	fmt.Println(model.TokenToPiece(chain.Sample(model, -1), false))
}

// Quantize writes a smaller copy of a GGUF file. It needs no loaded model.
func ExampleQuantize() {
	opts := llama.QuantizeOptions{
		FileType:             15, // LLAMA_FTYPE_MOSTLY_Q4_K_M in llama.h
		Threads:              runtime.NumCPU(),
		QuantizeOutputTensor: true, // llama.cpp's default; the zero value, false, leaves output.weight in its source type
	}
	if err := llama.Quantize("model-f16.gguf", "model-q4_k_m.gguf", opts); err != nil {
		log.Fatal(err)
	}
	fmt.Println("wrote", llama.FileTypeName(15)) // wrote Q4_K - Medium
}
