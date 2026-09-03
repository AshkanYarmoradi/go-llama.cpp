#!/usr/bin/env bash
#
# Verify that every function declared in binding.h is actually defined in
# binding.cpp.
#
# cgo compiles binding.h into the Go package, so a declaration with no
# definition type-checks fine and only fails much later, at link time, inside a
# full `make test` that first builds all of llama.cpp. This check reproduces the
# same failure in a couple of seconds using only the llama.cpp headers.
#
# It compiles binding.cpp to an object file and asks the toolchain which symbols
# it defines, rather than pattern-matching definitions -- so it agrees with the
# linker by construction.
#
# Usage: scripts/check-binding-symbols.sh
# Exits non-zero and lists the offending names if any declaration is unbacked.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f llama.cpp/include/llama.h ]; then
    echo "error: llama.cpp submodule is not checked out." >&2
    echo "       run: git submodule update --init --recursive" >&2
    exit 1
fi

CXX="${CXX:-c++}"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

obj="$workdir/binding.o"

"$CXX" \
    -I./llama.cpp -I./llama.cpp/include -I./llama.cpp/ggml/include \
    -I. -I./llama.cpp/common \
    -std=c++17 -fPIC -O0 \
    -c binding.cpp -o "$obj"

# Symbols binding.o defines. The binding is extern "C", so these are plain
# unmangled names.
"$(command -v nm)" -g --defined-only "$obj" \
    | awk '$2 ~ /^[TtWwDd]$/ { print $3 }' \
    | sed 's/^_//' \
    | sort -u > "$workdir/defined.txt"

# Function names declared in binding.h.
#
# Only the extern "C" block counts: that is the ABI surface cgo links against.
# The C++-linkage helpers below it (create_vector, delete_vector) are name
# mangled and internal to binding.cpp, so they are out of scope here.
#
# Declarations marked `extern` are also skipped -- those are cgo exports such as
# tokenCallback, which Go defines and C only calls.
#
# Strip comments, then split on ';' so each declaration is one record however it
# wraps across lines, and take the identifier before the parameter list.
awk '/extern "C" \{/ { inblock = 1; next } inblock && /^\}$/ { exit } inblock' binding.h \
    | sed -e 's://.*::' \
    | tr '\n' ' ' \
    | tr ';' '\n' \
    | grep -v '\bextern\b' \
    | grep -oE '\b[a-zA-Z_][a-zA-Z_0-9]*[[:space:]]*\(' \
    | tr -d ' (' \
    | grep -vxE 'if|for|while|switch|return|sizeof' \
    | sort -u > "$workdir/declared.txt"

if [ ! -s "$workdir/declared.txt" ]; then
    echo "error: found no declarations in binding.h -- has its structure changed?" >&2
    exit 1
fi

missing="$(comm -23 "$workdir/declared.txt" "$workdir/defined.txt" || true)"

if [ -n "$missing" ]; then
    echo "error: declared in binding.h but not defined in binding.cpp:" >&2
    echo "$missing" | sed 's/^/  /' >&2
    echo >&2
    echo "This would fail at link time when cgo builds the package." >&2
    exit 1
fi

echo "ok: all $(wc -l < "$workdir/declared.txt" | tr -d ' ') declarations in binding.h are defined in binding.cpp"
