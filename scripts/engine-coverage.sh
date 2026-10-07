#!/usr/bin/env bash
#
# Measure how much of llama.cpp's C API the binding uses, and generate
# docs/engine-coverage.md from the result.
#
# The API is every function llama.h declares with LLAMA_API, minus the ones
# wrapped in DEPRECATED(...). A function counts as covered when binding.cpp
# names it outside comments and string literals. It is a text scan of two
# files, so it needs no build. It sticks to POSIX awk, sed, sort and comm, and
# gives the same answer under gawk or mawk, on Linux CI or Git Bash.
#
# Usage:
#   scripts/engine-coverage.sh             print the counts and every gap
#   scripts/engine-coverage.sh --markdown  print docs/engine-coverage.md
#   scripts/engine-coverage.sh --write     regenerate docs/engine-coverage.md
#   scripts/engine-coverage.sh --check     exit 1 if docs/engine-coverage.md
#                                          is out of date
#
# Why a function is left out lives in the notes below, and the page is built
# from them. Edit the notes, not the page.
#
# Exit status: 0 on success, 1 when --check finds the page out of date, 2 when
# the scan itself fails (a missing submodule, or a header it cannot parse).

set -euo pipefail

# Byte-wise sorting and character classes, so sort and comm agree everywhere.
export LC_ALL=C

usage() {
    sed -n 's/^# \{0,1\}//; /^Usage:/,/^$/p' "$0"
}

mode=report
case "${1:-}" in
    "") ;;
    --markdown | --write | --check) mode="${1#--}" ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

# After the arguments, so usage can still read "$0" by its relative path.
cd "$(dirname "$0")/.."

header=llama.cpp/include/llama.h
binding=binding.cpp
page=docs/engine-coverage.md

# The one line of the page that names the llama.cpp commit. --check ignores
# it, so a llama.cpp bump that leaves the API alone does not flag the page.
stamp_prefix='Last regenerated against llama.cpp'

if [ ! -f "$header" ]; then
    echo "error: llama.cpp submodule is not checked out." >&2
    echo "       run: git submodule update --init --recursive" >&2
    exit 2
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

# ---------------------------------------------------------------------------
# Notes: why each uncovered function is uncovered.
#
#   @excluded <title>   starts a group of deliberate exclusions
#   @pending <title>    starts a group of functions not wrapped yet
#   llama_<name> <why>  one function; <why> may be empty
#   anything else       prose for the current group, shown before its table
#                       if it comes before the first function, after it if not;
#                       a blank line starts a new paragraph
#
# A group whose functions all have an empty <why> is a bullet list, not a table.
# A live function with no note is listed under "Not reviewed yet". A note for a
# function the binding now uses, or that llama.h no longer declares, is
# reported so it can be deleted.
# ---------------------------------------------------------------------------
cat > "$workdir/notes.txt" <<'NOTES'
@excluded Requires binding ggml first
llama_attach_threadpool Takes `ggml_threadpool` objects. Wrapping it means exposing a slice of ggml's threading API, which is a separate project.
llama_detach_threadpool As above.
llama_model_init_from_user Builds a model from a ggml `gguf_context` and a C callback that fills in each tensor's data. It serves programs that create weights in memory rather than load a GGUF file.
Thread counts can be set without a threadpool: see `(*LLama).SetThreads`.

@excluded Needs a C function-pointer vtable Go cannot supply
llama_sampler_init Builds a custom sampler from a `llama_sampler_i` struct of C function pointers. Go cannot populate one, and a bridge per method would cost a cgo transition per token per stage.
llama_sampler_copy Copies one sampler into another the caller already allocated; the same constraint. `(*Sampler).Clone` covers the useful case.
Every built-in sampler llama.cpp ships has a Go constructor, such as `SamplerTopK` or `(*LLama).SamplerGrammar`, except Mirostat v1 (`llama_sampler_init_mirostat`). `Predict` uses it when you pass `SetMirostat(1)`, but there is no stage for building your own chain with it. Mirostat v2 has one: `SamplerMirostatV2`.

@excluded Training API, out of scope
llama_opt_init This is an inference binding. The optimizer API needs `ggml-opt.h` dataset and callback types, and a different lifecycle.
llama_opt_epoch As above.
llama_opt_param_filter_all As above.

@excluded Takes a C `FILE*`
llama_model_load_from_file_ptr Loads a GGUF from an open `FILE*`, starting at its current offset, so a model can sit inside a larger file. Go has no `FILE*` to pass; supporting it means a new path-and-offset entry point in C. `New` loads by path and `NewFromSplits` loads split files.
llama_adapter_lora_init_from_file_ptr The same, for a LoRA adapter. `(*LLama).ApplyLoRA` and `SetLoraAdapter` load adapters by path.

@excluded Superseded by something already exposed
llama_get_logits `llama.h` plans to deprecate it in favour of `llama_get_logits_ith`, which the binding wraps as `(*LLama).Logits(i)`.
llama_get_embeddings Returns every output token's row in one array. `llama.h` plans to deprecate it in favour of `llama_get_embeddings_ith`, which the binding wraps as `(*LLama).TokenEmbedding(i)`. `(*LLama).SequenceEmbedding` reads pooled rows with `llama_get_embeddings_seq`.
llama_get_model Recovers the model from a context. The binding keeps both pointers in its own state, so it never needs it.
llama_perf_sampler_print Writes the numbers to llama.cpp's log, which goes to stderr unless a handler is installed with `SetLogHandler`. `(*Sampler).Perf` returns them as data.
llama_load_mode_from_str Aborts the process on a name it does not recognise. `ParseLoadMode` matches names against `llama_load_mode_name` instead and returns `LoadModeAuto` for an unknown one.

@pending Extended batch API
`llama_batch_ext` is a second batch type beside `llama_batch`, which `Batch`, `(*LLama).Decode` and `(*LLama).Encode` already wrap. `llama_batch` takes either token ids or embedding rows for the whole batch; this one can mix them token by token, gives a token several positions (M-RoPE), and carries hidden state over from an earlier stage, which multimodal and multi-token-prediction models use. `llama_process` runs an encode or a decode on one.
`llama.h` still lists reading logits and embeddings back from it as a TODO. Wrapping it needs a decision on whether it replaces `Batch` or sits beside it.
llama_batch_ext_init
llama_batch_ext_free
llama_batch_ext_clear
llama_batch_ext_add
llama_batch_ext_add_token
llama_batch_ext_add_embd
llama_batch_ext_add_seq
llama_batch_ext_set_embd_token
llama_batch_ext_set_embd_state
llama_batch_ext_set_output_embd
llama_batch_ext_set_output_logits
llama_batch_ext_set_pos
llama_process
NOTES

# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

# Remove comments, blank out string and character literals, and drop
# preprocessor lines unless keep_pp is set. Line structure is preserved, and
# CRLF input reads the same as LF.
strip_awk='
function skip_literal(s, i, q,    n, c) {
    n = length(s)
    for (i++; i <= n; i++) {
        c = substr(s, i, 1)
        if (c == "\\") i++
        else if (c == q) return i + 1
    }
    return n + 1
}
{
    sub(/\r$/, "")
    line = $0
    if (in_directive) {
        in_directive = (line ~ /\\$/)
        print ""
        next
    }
    if (!keep_pp && !in_comment && !in_raw && line ~ /^[ \t]*#/) {
        in_directive = (line ~ /\\$/)
        print ""
        next
    }
    out = ""
    n = length(line)
    i = 1
    while (i <= n) {
        if (in_comment) {
            j = index(substr(line, i), "*/")
            if (j == 0) break
            i += j + 1
            in_comment = 0
            out = out " "
            continue
        }
        if (in_raw) {
            j = index(substr(line, i), raw_end)
            if (j == 0) break
            i += j - 1 + length(raw_end)
            in_raw = 0
            out = out "\"\""
            continue
        }
        c = substr(line, i, 1)
        if (c == "/" && substr(line, i + 1, 1) == "/") break
        if (c == "/" && substr(line, i + 1, 1) == "*") {
            in_comment = 1
            i += 2
            continue
        }
        # C++ raw string: R"delim( ... )delim"
        if (c == "\"" && out ~ /(^|[^A-Za-z0-9_])(u8|u|U|L)?R$/) {
            j = index(substr(line, i + 1), "(")
            raw_end = ")" substr(line, i + 1, j - 1) "\""
            in_raw = 1
            i += j + 1
            continue
        }
        # String or character literal. A single quote straight after a digit
        # or letter is a C++14 digit separator, not a literal.
        if (c == "\"" || (c == "\047" && out !~ /[A-Za-z0-9_]$/)) {
            i = skip_literal(line, i, c)
            out = out c c
            continue
        }
        out = out c
        i++
    }
    print out
}'

# Split the stripped header into statements and print "name live" or
# "name deprecated" for each LLAMA_API declaration, however it wraps across
# lines and on whichever side of LLAMA_API the DEPRECATED( macro sits.
decl_awk='
{ text = text " " $0 }
END {
    n = split(text, part, /[;{}]/)
    for (k = 1; k <= n; k++) {
        s = part[k]
        if (s !~ /(^|[^A-Za-z0-9_])LLAMA_API([^A-Za-z0-9_]|$)/) continue
        deprecated = (s ~ /DEPRECATED[ \t]*\(/)
        gsub(/[A-Za-z0-9_]*DEPRECATED[ \t]*\(/, " ", s)
        if (!match(s, /[A-Za-z_][A-Za-z0-9_]*[ \t]*\(/)) {
            print "error: no function name in LLAMA_API declaration:" s | "cat 1>&2"
            failed = 1
            continue
        }
        name = substr(s, RSTART, RLENGTH)
        sub(/[ \t]*\($/, "", name)
        print name, (deprecated ? "deprecated" : "live")
    }
    exit (failed ? 1 : 0)
}'

# Every identifier in the stripped source, one per line.
ident_awk='
{
    s = $0
    while (match(s, /[A-Za-z_][A-Za-z0-9_]*/)) {
        print substr(s, RSTART, RLENGTH)
        s = substr(s, RSTART + RLENGTH)
    }
}'

if ! awk -v keep_pp=0 "$strip_awk" "$header" | awk "$decl_awk" > "$workdir/decls.txt"; then
    echo "error: could not parse $header -- has its structure changed?" >&2
    exit 2
fi
awk '{ print $1 }' "$workdir/decls.txt" | sort -u > "$workdir/all.txt"
awk '$2 == "deprecated" { print $1 }' "$workdir/decls.txt" | sort -u > "$workdir/deprecated.txt"
comm -23 "$workdir/all.txt" "$workdir/deprecated.txt" > "$workdir/live.txt"

# Preprocessor lines stay in for binding.cpp: a macro that calls a function
# still counts.
awk -v keep_pp=1 "$strip_awk" "$binding" | awk "$ident_awk" | sort -u > "$workdir/used.txt"

comm -12 "$workdir/live.txt" "$workdir/used.txt" > "$workdir/covered.txt"
comm -23 "$workdir/live.txt" "$workdir/used.txt" > "$workdir/gaps.txt"
comm -12 "$workdir/deprecated.txt" "$workdir/used.txt" > "$workdir/deprecated_used.txt"

count() { wc -l < "$1" | tr -d ' '; }

n_all=$(count "$workdir/all.txt")
n_deprecated=$(count "$workdir/deprecated.txt")
n_live=$(count "$workdir/live.txt")
n_covered=$(count "$workdir/covered.txt")
n_deprecated_used=$(count "$workdir/deprecated_used.txt")

# llama.h has declared well over a hundred functions for years. Far fewer means
# the scan broke, not that the API shrank.
if [ "$n_all" -lt 100 ] || [ "$n_covered" -eq 0 ]; then
    echo "error: found $n_all LLAMA_API functions in $header and $n_covered in $binding" >&2
    echo "       -- has the structure of either file changed?" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Classify each gap against the notes. Prints "name kind" per gap, where kind
# is excluded, pending or untriaged, and renders the groups as Markdown into
# excluded.md and pending.md.
# ---------------------------------------------------------------------------
classify_awk='
function err(msg) {
    print "error: notes in scripts/engine-coverage.sh: " msg | "cat 1>&2"
    failed = 1
}
function para(acc, line) {
    sub(/^[ \t]+/, "", line)
    if (acc == "") return line
    return acc (brk ? "\n\n" : " ") line
}
function render(grp, out,    i, f) {
    if (!cnt[grp]) return
    printf "### %s (%d)\n\n", title[grp], cnt[grp] > out
    if (pre[grp] != "") printf "%s\n\n", pre[grp] > out
    if (has_reason[grp]) {
        printf "| Function | %s |\n|---|---|\n", (kind[grp] == "excluded" ? "Why" : "Notes") > out
        for (i = 1; i <= nfn[grp]; i++) {
            f = fn[grp, i]
            if (f in gap) printf "| `%s` | %s |\n", f, why[f] > out
        }
    } else {
        for (i = 1; i <= nfn[grp]; i++) {
            f = fn[grp, i]
            if (f in gap) printf "- `%s`\n", f > out
        }
    }
    printf "\n" > out
    if (post[grp] != "") printf "%s\n\n", post[grp] > out
}
FILENAME == gaps_file { gap[$1] = 1; order[++ngaps] = $1; next }
/^[ \t]*$/ { brk = 1; next }
/^@(excluded|pending) / {
    g++
    kind[g] = substr($1, 2)
    title[g] = $0
    sub(/^@[a-z]+ +/, "", title[g])
    brk = 0
    next
}
/^llama_[A-Za-z0-9_]*( |$)/ {
    if (!g) { err($1 " comes before any @excluded or @pending line"); next }
    name = $1
    reason = $0
    sub(/^[^ ]+ */, "", reason)
    if (reason ~ /\|/) err(name ": a | in the text would split its table cell")
    if (name in noted) err(name " is listed twice")
    noted[name] = g
    fn[g, ++nfn[g]] = name
    why[name] = reason
    if (name in gap) {
        cnt[g]++
        if (reason != "") has_reason[g] = 1
    } else {
        stale = stale " " name
    }
    brk = 0
    next
}
{
    if (!g) { err("text before any @excluded or @pending line"); next }
    if (nfn[g] == 0) pre[g] = para(pre[g], $0)
    else post[g] = para(post[g], $0)
    brk = 0
}
END {
    if (failed) exit 1
    nun = 0
    for (k = 1; k <= ngaps; k++) {
        name = order[k]
        if (name in noted) {
            print name, kind[noted[name]]
        } else {
            print name, "untriaged"
            untriaged[++nun] = name
        }
    }
    printf "" > excluded_md
    printf "" > pending_md
    for (k = 1; k <= g; k++)
        render(k, kind[k] == "excluded" ? excluded_md : pending_md)
    if (nun) {
        printf "### Not reviewed yet (%d)\n\n", nun > pending_md
        printf "No decision is recorded for these yet. Usually llama.cpp added them since\nthe last review, or the binding stopped using them. Wrap them, or record why\nnot as a note in `scripts/engine-coverage.sh`.\n\n" > pending_md
        for (k = 1; k <= nun; k++) printf "- `%s`\n", untriaged[k] > pending_md
        printf "\n" > pending_md
    }
    close(excluded_md)
    close(pending_md)
    if (stale != "") {
        print "note: scripts/engine-coverage.sh has notes for functions binding.cpp now uses" | "cat 1>&2"
        print "      or llama.h no longer declares; delete them:" stale | "cat 1>&2"
    }
}'

if ! awk -v gaps_file="$workdir/gaps.txt" \
        -v excluded_md="$workdir/excluded.md" \
        -v pending_md="$workdir/pending.md" \
        "$classify_awk" "$workdir/gaps.txt" "$workdir/notes.txt" > "$workdir/classes.txt"; then
    exit 2
fi

n_excluded=$(awk '$2 == "excluded"' "$workdir/classes.txt" | wc -l | tr -d ' ')
n_pending=$((n_live - n_covered - n_excluded))

# The commit that was scanned. Guarded on llama.cpp/.git so a tarball without
# git metadata does not report the superproject's HEAD instead.
sha=""
commit_date=""
if [ -e llama.cpp/.git ] && sha=$(git -C llama.cpp rev-parse HEAD 2>/dev/null); then
    commit_date=$(git -C llama.cpp log -1 --format=%cs HEAD 2>/dev/null || true)
    pinned=$(git rev-parse -q --verify HEAD:llama.cpp 2>/dev/null || true)
    if [ -n "$pinned" ] && [ "$pinned" != "$sha" ]; then
        echo "warning: llama.cpp is checked out at ${sha:0:7} but this commit pins ${pinned:0:7};" >&2
        echo "         run: git submodule update --init --recursive" >&2
    fi
else
    sha=""
fi

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
report() {
    if [ -n "$sha" ]; then
        echo "llama.cpp ${sha:0:7} (${commit_date:-date unknown})"
    fi
    printf '%-22s %4d\n' \
        "LLAMA_API functions" "$n_all" \
        "deprecated" "$n_deprecated" \
        "live" "$n_live"
    printf '%-22s %4d/%d\n' "covered" "$n_covered" "$n_live"
    printf '%-22s %4d\n' \
        "excluded on purpose" "$n_excluded" \
        "not wrapped yet" "$n_pending"
    echo
    echo "not covered ($((n_live - n_covered))):"
    for kind in excluded pending untriaged; do
        awk -v kind="$kind" '$2 == kind { printf "  %-10s %s\n", $2, $1 }' "$workdir/classes.txt"
    done
    if [ "$n_deprecated_used" -gt 0 ]; then
        echo
        echo "deprecated but still used by binding.cpp ($n_deprecated_used):"
        sed 's/^/  /' "$workdir/deprecated_used.txt"
    fi
}

render_page() {
    cat <<'EOF'
# Engine coverage

<!-- Generated by scripts/engine-coverage.sh. Edit the notes in that script and run it with --write; do not edit this file by hand. -->

This binding aims to cover llama.cpp's C API. This page records where that
stands and, more usefully, which functions are left out **on purpose**, so an
audit that finds them missing does not have to re-derive the reasoning.

The page is generated by [`scripts/engine-coverage.sh`](../scripts/engine-coverage.sh)
from `llama.cpp/include/llama.h` and `binding.cpp`, and CI re-checks it on
every pull request.
EOF
    if [ -n "$sha" ]; then
        printf '%s [`%s`](https://github.com/ggml-org/llama.cpp/commit/%s), committed %s.\n' \
            "$stamp_prefix" "${sha:0:7}" "$sha" "${commit_date:-on an unknown date}"
    else
        printf '%s an unknown commit.\n' "$stamp_prefix"
    fi
    cat <<EOF

|  |  |
|---|---|
| Functions declared with \`LLAMA_API\` | $n_all |
| Marked \`DEPRECATED\` upstream | $n_deprecated |
| **Live API surface** | **$n_live** |
| **Used by the binding** | **$n_covered** |
| Excluded on purpose | $n_excluded |
| Not wrapped yet | $n_pending |

EOF
    if [ "$n_deprecated_used" -eq 0 ]; then
        cat <<'EOF'
Deprecated functions are not wrapped. Where the `DEPRECATED` message names a
replacement, the binding wraps that instead: `llama_vocab_bos` rather than
`llama_token_bos`, for example.

EOF
    else
        cat <<'EOF'
Deprecated functions are not counted. Where the `DEPRECATED` message names a
replacement, the binding should use that instead: `llama_vocab_bos` rather than
`llama_token_bos`, for example. These deprecated ones are still in use and
will break when llama.cpp removes them:

EOF
        sed 's/.*/- `&`/' "$workdir/deprecated_used.txt"
        echo
    fi
    if [ "$n_excluded" -gt 0 ]; then
        echo "## Excluded on purpose ($n_excluded)"
        echo
        cat "$workdir/excluded.md"
    fi
    if [ "$n_pending" -gt 0 ]; then
        echo "## Not wrapped yet ($n_pending)"
        echo
        cat <<'EOF'
Live functions the binding does not use yet, with what wrapping them would
take.

EOF
        cat "$workdir/pending.md"
    fi
    cat <<'EOF'
## If you need one of these

Open an issue with the [Missing llama.cpp API](https://github.com/AshkanYarmoradi/go-llama.cpp/issues/new?template=engine_api_gap.yml)
template and say what you are building. Several of the exclusions above are
judgement calls about cost versus demand, and a concrete use case changes that
calculation.

## How it is measured

- Every function declared with `LLAMA_API` in `llama.h` counts once. Comments
  are removed first, so commented-out prototypes do not count, and
  declarations that wrap across lines are joined.
- A declaration inside `DEPRECATED(...)` is deprecated, whichever side of
  `LLAMA_API` the macro sits on. The live API surface is everything else.
- A live function is used when its name appears as an identifier in
  `binding.cpp`, outside comments and string literals. That shows the binding
  uses it, not that every parameter reaches Go; the
  [API reference](https://pkg.go.dev/github.com/AshkanYarmoradi/go-llama.cpp)
  shows what does.

## Regenerating

```bash
git submodule update --init --recursive    # scan the pinned llama.h
./scripts/engine-coverage.sh               # print the counts and every gap
./scripts/engine-coverage.sh --write       # rewrite this page
./scripts/engine-coverage.sh --check       # exit 1 if this page is out of date
```

It needs only bash and POSIX tools, and gives the same result on Linux and in
Git Bash on Windows. The Lint workflow runs `--check` and turns a stale page
into a warning, not a failure, so a llama.cpp bump is never blocked by this
page. The check skips the "Last regenerated" line, so it fires when the counts
or lists change, not on every bump.

The reasons on this page live as notes in `scripts/engine-coverage.sh`. To
record why a function is left out, add a note there and run `--write`. After
wrapping a function that has a note, the script names the note so you can
delete it.

EOF
    echo "<details>"
    echo "<summary>The $n_covered live functions the binding uses</summary>"
    echo
    echo '```text'
    cat "$workdir/covered.txt"
    echo '```'
    echo
    echo "</details>"
    echo
    echo "<details>"
    echo "<summary>The $n_deprecated deprecated functions</summary>"
    echo
    echo '```text'
    cat "$workdir/deprecated.txt"
    echo '```'
    echo
    echo "</details>"
}

case "$mode" in
    report)
        report
        ;;
    markdown)
        render_page
        ;;
    write)
        render_page > "$workdir/page.md"
        cp "$workdir/page.md" "$page"
        echo "wrote $page: $n_covered/$n_live live functions covered"
        ;;
    check)
        echo "$n_covered/$n_live live llama.h functions covered"
        if [ ! -f "$page" ]; then
            echo "$page does not exist; run: ./scripts/engine-coverage.sh --write" >&2
            exit 1
        fi
        render_page | grep -v "^$stamp_prefix" > "$workdir/expected.md" || true
        tr -d '\r' < "$page" | grep -v "^$stamp_prefix" > "$workdir/committed.md" || true
        if ! diff -u "$workdir/committed.md" "$workdir/expected.md"; then
            echo
            echo "$page is out of date; run: ./scripts/engine-coverage.sh --write" >&2
            exit 1
        fi
        echo "$page is up to date"
        ;;
esac
