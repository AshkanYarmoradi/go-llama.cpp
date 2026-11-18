package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testHeader = `
#define LLAMA_API __declspec(dllexport)
#define MULTILINE(x) \
    ignored(x)
extern "C" {
struct llama_context;
typedef int32_t llama_token;
typedef bool (*llama_callback)(float progress, void * data);
typedef struct llama_params {
    int count;
    union { int number; char text[16]; };
    bool (*callback)(void * data);
} llama_params;
enum llama_mode { FIRST, SECOND = FIRST + 2, THIRD };
LLAMA_API int llama_run(struct llama_params * params);
DEPRECATED(LLAMA_API void llama_old(void), "use llama_run; instead");
}
`

func fixture(t *testing.T) (string, string, []string) {
	t.Helper()
	dir := t.TempDir()
	header, path := filepath.Join(dir, "llama.h"), filepath.Join(dir, "reviewed.json")
	writeFile(t, header, testHeader)
	api, err := declarations(testHeader)
	if err != nil {
		t.Fatal(err)
	}
	if len(api) != 7 {
		t.Fatalf("got %d declarations, want 7: %v", len(api), api)
	}
	m := manifest{Baseline: "Existing API for the test.", API: map[string]review{}}
	for name, decl := range api {
		m.API[name] = review{Declaration: decl, Decision: "baseline"}
	}
	var data bytes.Buffer
	if err := writeJSON(&data, m); err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, data.String())
	return header, path, []string{"-header", header, "-manifest", path}
}

func writeFile(t *testing.T, path, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestReviewGate(t *testing.T) {
	cases := []struct{ name, before, after, item string }{
		{"function", "LLAMA_API int llama_run", "LLAMA_API void llama_new(void); LLAMA_API int llama_run", "function llama_new"},
		{"function signature", "int llama_run", "int64_t llama_run", "function llama_run"},
		{"struct field", "int count;", "int count; bool new_option;", "struct llama_params"},
		{"field type", "int count;", "int64_t count;", "struct llama_params"},
		{"nested union field", "int number;", "int number; double extra;", "struct llama_params"},
		{"callback field", "bool (*callback)", "void (*callback)", "struct llama_params"},
		{"enum member", "FIRST, SECOND", "FIRST, INSERTED, SECOND", "enum llama_mode"},
		{"enum value", "FIRST + 2", "FIRST + 3", "enum llama_mode"},
		{"enum implicit order", "FIRST, SECOND = FIRST + 2, THIRD", "FIRST, THIRD, SECOND = FIRST + 2", "enum llama_mode"},
		{"new enum", "enum llama_mode", "enum llama_extra { EXTRA }; enum llama_mode", "enum llama_extra"},
		{"new struct", "struct llama_context;", "struct llama_context; struct llama_extra { int value; };", "struct llama_extra"},
		{"typedef", "typedef int32_t llama_token;", "typedef int64_t llama_token;", "typedef llama_token"},
		{"removed function", "LLAMA_API int llama_run(struct llama_params * params);", "", "function llama_run"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			header, _, args := fixture(t)
			writeFile(t, header, strings.Replace(testHeader, tc.before, tc.after, 1))
			var out bytes.Buffer
			if err := run(args, &out); err == nil || !strings.Contains(out.String(), tc.item) {
				t.Fatalf("change passed or wrong diagnostic: %v\n%s", err, &out)
			}
		})
	}
}

func TestFormattingAndCommentsDoNotNeedReview(t *testing.T) {
	header, _, args := fixture(t)
	writeFile(t, header, strings.ReplaceAll(strings.ReplaceAll(testHeader, "int count;", "int /* option */ count ; // explanation"), "\n", "\r\n"))
	if err := run(args, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
}

func TestExclusionIsSpecificToTheReviewedDeclaration(t *testing.T) {
	header, path, args := fixture(t)
	writeFile(t, header, strings.Replace(testHeader, "int count;", "int count; bool new_option;", 1))
	before, _ := os.ReadFile(path)
	reviewArgs := append(append([]string{}, args...), "-review", "struct llama_params", "-decision", "excluded")
	if err := run(reviewArgs, &bytes.Buffer{}); err == nil {
		t.Fatal("accepted an exclusion without a reason")
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(before, after) {
		t.Fatal("failed review modified the manifest")
	}
	if err := run(append(reviewArgs, "-reason", "This configuration is intentionally left at the engine default."), &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if err := run(args, &bytes.Buffer{}); err != nil {
		t.Fatalf("reviewed exclusion failed: %v", err)
	}
	writeFile(t, header, strings.Replace(testHeader, "int count;", "int count; bool new_option; int another;", 1))
	if err := run(args, &bytes.Buffer{}); err == nil {
		t.Fatal("an exclusion silently accepted a later field")
	}
}

func TestRemovedAPIRequiresReviewAndReintroductionRequiresAnother(t *testing.T) {
	header, _, args := fixture(t)
	writeFile(t, header, strings.Replace(testHeader, "LLAMA_API int llama_run(struct llama_params * params);", "", 1))
	if err := run(append(append([]string{}, args...), "-review", "function llama_run", "-decision", "removed", "-reason", "The engine removed this entry point."), &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if err := run(args, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	writeFile(t, header, testHeader)
	if err := run(args, &bytes.Buffer{}); err == nil {
		t.Fatal("reintroduced API passed without review")
	}
}

func TestReviewDecisionsOnlyApproveOneChange(t *testing.T) {
	for _, decision := range []string{"supported", "deferred", "excluded"} {
		t.Run(decision, func(t *testing.T) {
			header, _, args := fixture(t)
			writeFile(t, header, testHeader+"\nLLAMA_API void llama_new(void);\nenum llama_extra { EXTRA };\n")
			record := func(item string) {
				t.Helper()
				if err := run(append(append([]string{}, args...), "-review", item, "-decision", decision, "-reason", "Reviewed this change independently."), &bytes.Buffer{}); err != nil {
					t.Fatal(err)
				}
			}
			record("function llama_new")
			var out bytes.Buffer
			if err := run(args, &out); err == nil || !strings.Contains(out.String(), "enum llama_extra") || strings.Contains(out.String(), "function llama_new") {
				t.Fatalf("review did not leave the other change pending: %v\n%s", err, &out)
			}
			record("enum llama_extra")
			if err := run(args, &bytes.Buffer{}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestMalformedInputFailsClosed(t *testing.T) {
	for _, header := range []string{"", "/* truncated", "LLAMA_API void llama_broken(", "struct llama_bad { int x;", "extern \"C\" {", "LLAMA_API unexpected;", "static inline void llama_new(void) {}", "struct llama_bad { int x]; };", "LLAMA_API int llama_run(void); LLAMA_API float llama_run(void);"} {
		if _, err := declarations(header); err == nil {
			t.Errorf("accepted malformed or unsupported header %q", header)
		}
	}
	_, path, args := fixture(t)
	for _, data := range []string{"{}", "not json", `{"baseline":"test","api":{}}`, `{"baseline":"test","api":{"function llama_run":{"declaration":"x","decision":"excluded"}}}`} {
		writeFile(t, path, data)
		if err := run(args, &bytes.Buffer{}); err == nil {
			t.Errorf("accepted invalid manifest %s", data)
		}
	}
}
