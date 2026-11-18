// Command api-review checks that changes to llama.h have a recorded review.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"text/scanner"
)

type review struct {
	Declaration string `json:"declaration"`
	Decision    string `json:"decision"`
	Reason      string `json:"reason,omitempty"`
}

type manifest struct {
	Baseline string            `json:"baseline"`
	API      map[string]review `json:"api"`
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, out io.Writer) error {
	flags := flag.NewFlagSet("api-review", flag.ContinueOnError)
	header := flags.String("header", "llama.cpp/include/llama.h", "public header to check")
	path := flags.String("manifest", "scripts/api-review/reviewed.json", "recorded API reviews")
	item := flags.String("review", "", "one API item to review, e.g. 'struct llama_model_params'")
	decision := flags.String("decision", "", "supported, deferred, excluded, or removed")
	reason := flags.String("reason", "", "review rationale and Go API mapping, or reason for exclusion")
	list := flags.Bool("list", false, "print current declarations as JSON without approving them")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || (*item == "" && (*decision != "" || *reason != "")) || (*list && *item != "") {
		return fmt.Errorf("use -review with -decision and -reason, or run without flags to check")
	}
	source, err := os.ReadFile(*header)
	if err != nil {
		return err
	}
	api, err := declarations(string(source))
	if err != nil {
		return fmt.Errorf("parse %s: %w", *header, err)
	}
	if *list {
		return writeJSON(out, api)
	}
	data, err := os.ReadFile(*path)
	if err != nil {
		return err
	}
	var recorded manifest
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&recorded); err != nil {
		return fmt.Errorf("read %s: %w", *path, err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("%s must contain one JSON object", *path)
	}
	if err := validate(recorded); err != nil {
		return fmt.Errorf("%s: %w", *path, err)
	}
	if *item != "" {
		decl, exists := api[*item]
		if strings.TrimSpace(*reason) == "" {
			return fmt.Errorf("-reason is required; record how the change is supported or why it is deferred/excluded")
		}
		switch *decision {
		case "supported", "deferred", "excluded":
			if !exists {
				return fmt.Errorf("%q is not in the header; use -list to find its exact name", *item)
			}
		case "removed":
			if exists || recorded.API[*item].Declaration == "" {
				return fmt.Errorf("%q must be a previously recorded item absent from the header", *item)
			}
		default:
			return fmt.Errorf("-decision must be supported, deferred, excluded, or removed")
		}
		recorded.API[*item] = review{Declaration: decl, Decision: *decision, Reason: *reason}
		var buf strings.Builder
		if err := writeJSON(&buf, recorded); err != nil {
			return err
		}
		if err := os.WriteFile(*path, []byte(buf.String()), 0644); err != nil {
			return err
		}
		fmt.Fprintf(out, "Recorded %s: %s. Run the check again before committing.\n", *decision, *item)
		return nil
	}
	changes := differences(api, recorded)
	if len(changes) != 0 {
		fmt.Fprintln(out, "Unreviewed llama.cpp API changes:")
		for _, change := range changes {
			fmt.Fprintln(out, change)
		}
		return fmt.Errorf("review each change and record it with -review, -decision and -reason; see CONTRIBUTING.md#reviewing-llamacpp-api-changes")
	}
	fmt.Fprintf(out, "ok: all %d public declarations match their recorded reviews\n", len(api))
	return nil
}

func writeJSON(out io.Writer, value any) error {
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	return enc.Encode(value)
}

func validate(m manifest) error {
	if strings.TrimSpace(m.Baseline) == "" || len(m.API) == 0 {
		return fmt.Errorf("missing baseline explanation or API reviews")
	}
	for name, r := range m.API {
		switch r.Decision {
		case "baseline":
		case "supported", "deferred", "excluded", "removed":
			if strings.TrimSpace(r.Reason) == "" {
				return fmt.Errorf("%s: review reason is required", name)
			}
		default:
			return fmt.Errorf("%s: invalid review decision %q", name, r.Decision)
		}
		if (r.Decision == "removed") != (r.Declaration == "") {
			return fmt.Errorf("%s: only removed items may have an empty declaration", name)
		}
	}
	return nil
}

func differences(api map[string]string, m manifest) []string {
	var changes []string
	for name, decl := range api {
		r, exists := m.API[name]
		if !exists || r.Decision == "removed" {
			changes = append(changes, "added "+name+"\n  + "+decl)
		} else if r.Declaration != decl {
			changes = append(changes, "changed "+name+"\n  - "+r.Declaration+"\n  + "+decl)
		}
	}
	for name, r := range m.API {
		if _, exists := api[name]; !exists && r.Decision != "removed" {
			changes = append(changes, "removed "+name+"\n  - "+r.Declaration)
		}
	}
	slices.Sort(changes)
	return changes
}

// Scan declarations, not C semantics. Keep complete aggregates so field order,
// anonymous unions, callback signatures and implicit enum values need review too.
func declarations(source string) (map[string]string, error) {
	var lexer scanner.Scanner
	lexer.Init(strings.NewReader(source))
	lexer.Mode = scanner.ScanIdents | scanner.ScanInts | scanner.ScanFloats | scanner.ScanStrings | scanner.ScanChars | scanner.ScanComments | scanner.SkipComments
	lexer.Whitespace &^= 1 << '\n'
	var scanErr error
	lexer.Error = func(s *scanner.Scanner, message string) {
		scanErr = fmt.Errorf("%s: %s", s.Position, message)
	}
	var tokens []string
	directive, continued := false, false
	for tok := lexer.Scan(); tok != scanner.EOF; tok = lexer.Scan() {
		if tok == '\n' {
			directive = directive && continued
			continued = false
			continue
		}
		if tok == '#' {
			directive = true
		}
		if directive {
			continued = tok == '\\'
			continue
		}
		tokens = append(tokens, lexer.TokenText())
	}
	if scanErr != nil {
		return nil, scanErr
	}
	api := make(map[string]string)
	var stmt, stack []string
	linkage := false
	for i := 0; i < len(tokens); i++ {
		token := tokens[i]
		if len(stmt) == 0 && token == "extern" && i+2 < len(tokens) && tokens[i+1] == `"C"` && tokens[i+2] == "{" {
			if linkage {
				return nil, fmt.Errorf("nested extern C block")
			}
			linkage = true
			i += 2
			continue
		}
		if len(stmt) == 0 && token == "}" && linkage {
			linkage = false
			continue
		}
		stmt = append(stmt, token)
		switch token {
		case "{", "(", "[":
			stack = append(stack, token)
		case "}", ")", "]":
			if len(stack) == 0 || stack[len(stack)-1] != map[string]string{"}": "{", ")": "(", "]": "["}[token] {
				return nil, fmt.Errorf("unbalanced declaration: %s", strings.Join(stmt, " "))
			}
			stack = stack[:len(stack)-1]
		}
		if token != ";" || len(stack) != 0 {
			continue
		}
		name, err := declarationName(stmt)
		if err != nil {
			return nil, err
		}
		decl := strings.Join(stmt, " ")
		if old, exists := api[name]; exists && old != decl {
			if strings.HasPrefix(name, "struct ") || strings.HasPrefix(name, "enum ") || strings.HasPrefix(name, "union ") {
				if !slices.Contains(stmt, "{") {
					stmt = nil
					continue
				}
				if strings.Contains(old, "{") {
					return nil, fmt.Errorf("conflicting declarations for %s", name)
				}
			} else {
				return nil, fmt.Errorf("conflicting declarations for %s", name)
			}
		}
		api[name] = decl
		stmt = nil
	}
	if len(stmt) != 0 || len(stack) != 0 || linkage {
		return nil, fmt.Errorf("incomplete declaration or extern C block")
	}
	if len(api) == 0 {
		return nil, fmt.Errorf("no public declarations found")
	}
	return api, nil
}

func declarationName(tokens []string) (string, error) {
	if slices.Contains(tokens, "LLAMA_API") {
		for i, token := range tokens[:len(tokens)-1] {
			if strings.HasPrefix(token, "llama_") && tokens[i+1] == "(" {
				return "function " + token, nil
			}
		}
	} else {
		start := 0
		if tokens[0] == "typedef" {
			start = 1
		}
		kind := tokens[start]
		if (kind == "struct" || kind == "enum" || kind == "union") && len(tokens) > start+2 && (tokens[start+2] == "{" || tokens[start+2] == ";") {
			return kind + " " + tokens[start+1], nil
		}
		if tokens[0] == "typedef" {
			for i := 0; i+3 < len(tokens); i++ {
				if tokens[i] == "(" && tokens[i+1] == "*" && tokens[i+3] == ")" {
					return "typedef " + tokens[i+2], nil
				}
			}
			return "typedef " + tokens[len(tokens)-2], nil
		}
	}
	return "", fmt.Errorf("unsupported public declaration; update the scanner: %s", strings.Join(tokens, " "))
}
