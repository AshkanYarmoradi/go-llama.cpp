<!--
Thanks for contributing. Keep one concern per PR — it makes review and
bisection much easier.
-->

## What this changes

<!-- What the change does, and why it has to work this way. -->

## Engine APIs newly covered

<!--
If this wraps llama.cpp functions that had no Go surface, list them:

  llama_sampler_init_infill, llama_sampler_name

Then run ./scripts/engine-coverage.sh --write and commit the regenerated
docs/engine-coverage.md.

Delete this section if it does not apply.
-->

## Breaking changes

<!--
Any change to an exported Go signature. Show before and after, and say what
forced it — an upstream API change is a good reason, tidiness is not.

Delete this section if there are none.
-->

## Checks

- [ ] `gofmt -l -e .` is clean
- [ ] `go vet` passes under every build tag (the loop in CONTRIBUTING.md)
- [ ] `go mod tidy && git diff --exit-code go.mod go.sum` shows no changes
- [ ] `./scripts/check-binding-symbols.sh` passes
- [ ] `binding.cpp` compiles with the CI warning flags (see CONTRIBUTING.md)
- [ ] Tested against a real model with `TEST_MODEL` set, or explained below why not
- [ ] New tests live in their own `Context` block, above the `gpu`-labelled one
- [ ] Exported Go identifiers have doc comments
- [ ] User-visible changes have an entry under `[Unreleased]` in `CHANGELOG.md`
- [ ] Go code changed in `README.md` or `docs/` has its `Example` changed too, or was vetted in a scratch module
- [ ] If this touches the Makefile or a `llama_<tag>.go`: each affected `BUILD_TYPE` was built and linked with its tag, or is named below as untested (CI never builds `hipblas`, `openblas` or `blis`)
