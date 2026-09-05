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
- [ ] `go vet ./...` passes
- [ ] `./scripts/check-binding-symbols.sh` passes
- [ ] `binding.cpp` compiles with the CI warning flags (see CONTRIBUTING.md)
- [ ] Tested against a real model with `TEST_MODEL` set, or explained below why not
- [ ] New tests live in their own `Context` block
- [ ] Exported Go identifiers have doc comments
