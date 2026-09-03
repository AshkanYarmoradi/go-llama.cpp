# Security Policy

## Reporting a vulnerability

Report security issues through
[GitHub's private vulnerability reporting](https://github.com/AshkanYarmoradi/go-llama.cpp/security/advisories/new)
rather than a public issue.

Please include what you have: the affected version or commit, what an attacker
gains, and a reproducer if you have one. I will acknowledge within a few days
and keep you updated as a fix comes together.

## Supported versions

This project tracks llama.cpp's `master`. Fixes land on `main`; there are no
maintained release branches. If you are pinned to an older commit, the fix is
to move forward.

## Scope

This repository is a binding layer. Roughly:

**In scope** — bugs in this repo's own code:

- Memory safety in `binding.cpp`: unchecked buffer sizes, out-of-bounds
  indexing, use-after-free, or missing bounds checks on caller-supplied
  indices and token ids.
- Unsafe pointer handling in `llama.go`, including slices passed to C that Go
  may move or free.
- The build system fetching or executing something it should not.

**Out of scope** — report these to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/security) instead:

- Vulnerabilities in the llama.cpp engine itself or in ggml.
- Malicious GGUF model files. Model parsing happens entirely inside llama.cpp.

## Things that are not vulnerabilities

Worth stating, because they come up:

- **Model output.** What a language model generates is not a security boundary.
  Prompt injection, jailbreaks, and harmful completions are properties of the
  model, not of this binding.
- **Loading an untrusted model.** A GGUF file is executable-adjacent input to a
  large C++ parser. Treat model files like any other untrusted binary: only
  load ones you trust the source of.
- **Concurrent misuse.** A `*LLama` is documented as not safe for concurrent
  use. Data races from calling it concurrently are a usage error, not a
  vulnerability.
