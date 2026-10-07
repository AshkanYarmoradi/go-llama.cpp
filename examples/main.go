package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	llama "github.com/AshkanYarmoradi/go-llama.cpp"
)

var (
	threads     = 4
	tokens      = 512
	gpulayers   = 0
	seed        = -1
	contextSize = 4096
)

func main() {
	var model string

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&model, "m", "./models/model.gguf", "path to GGUF model file to load")
	flags.IntVar(&gpulayers, "ngl", 0, "Number of GPU layers to use")
	flags.IntVar(&threads, "t", runtime.NumCPU(), "number of threads to use during computation")
	flags.IntVar(&tokens, "n", 512, "number of tokens to predict")
	flags.IntVar(&contextSize, "c", 4096, "context size in tokens")
	flags.IntVar(&seed, "s", -1, "predict RNG seed, -1 for random seed")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Printf("Parsing program arguments failed: %s", err)
		os.Exit(1)
	}

	// The context has to hold the prompt plus everything we generate. Leave
	// embeddings off: an embeddings context computes an output for every
	// prompt token rather than only the last, which costs memory and time
	// that generation has no use for.
	l, err := llama.New(model,
		llama.SetContext(contextSize),
		llama.SetGPULayers(gpulayers),
	)
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		os.Exit(1)
	}
	defer l.Free()

	// Apply -t to the context, for generation and prompt processing alike.
	// Without this the context runs on llama.cpp's default of 4 threads.
	l.SetThreads(threads, threads)

	fmt.Printf("Model loaded successfully.\n")

	reader := bufio.NewReader(os.Stdin)

	for {
		text := readMultiLineInput(reader)

		prompt, err := chatPrompt(l, text)
		if err != nil {
			fmt.Printf("Applying the chat template failed: %s\n", err)
			os.Exit(1)
		}

		_, err = l.Predict(prompt, llama.Debug, llama.SetTokenCallback(func(token string) bool {
			fmt.Print(token)
			return true
		}), llama.SetTokens(tokens), llama.SetTopK(40), llama.SetTopP(0.9), llama.SetTemperature(0.7), llama.SetSeed(seed))
		if err != nil {
			fmt.Printf("Predicting failed: %s\n", err)
			os.Exit(1)
		}
		fmt.Printf("\n\n")
	}
}

// chatPrompt wraps one user message in the model's own chat template, so an
// instruct model sees the turn markers it was trained on. A model without a
// template llama.cpp can apply gets the text as typed.
//
// Each message is answered on its own: Predict clears the KV cache before it
// runs, so nothing carries over from the previous turn.
func chatPrompt(l *llama.LLama, text string) (string, error) {
	prompt, err := l.ApplyChatTemplate("", []llama.ChatMessage{{Role: "user", Content: text}}, true)
	if errors.Is(err, llama.ErrNoChatTemplate) {
		return text, nil
	}
	return prompt, err
}

// readMultiLineInput reads input until an empty line is entered.
func readMultiLineInput(reader *bufio.Reader) string {
	var lines []string
	fmt.Print(">>> ")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				os.Exit(0)
			}
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}

		if len(strings.TrimSpace(line)) == 0 {
			break
		}

		lines = append(lines, line)
	}

	return strings.TrimRight(strings.Join(lines, ""), "\r\n")
}
