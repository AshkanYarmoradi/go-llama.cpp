package main

import (
	"bufio"
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

	// The context has to hold the prompt plus everything we generate. Do not
	// enable embeddings here: an embeddings context returns pooled vectors
	// instead of token logits, which turns generation into garbage. Load the
	// model a second time with llama.EnableEmbeddings if you need both.
	l, err := llama.New(model,
		llama.SetContext(contextSize),
		llama.SetGPULayers(gpulayers),
	)
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		os.Exit(1)
	}
	defer l.Free()

	fmt.Printf("Model loaded successfully.\n")

	reader := bufio.NewReader(os.Stdin)

	for {
		text := readMultiLineInput(reader)

		_, err := l.Predict(text, llama.Debug, llama.SetTokenCallback(func(token string) bool {
			fmt.Print(token)
			return true
		}), llama.SetTokens(tokens), llama.SetThreads(threads), llama.SetTopK(40), llama.SetTopP(0.9), llama.SetTemperature(0.7), llama.SetSeed(seed))
		if err != nil {
			fmt.Printf("Predicting failed: %s\n", err)
			os.Exit(1)
		}
		fmt.Printf("\n\n")
	}
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

	text := strings.Join(lines, "")
	fmt.Println("Sending", text)
	return text
}
