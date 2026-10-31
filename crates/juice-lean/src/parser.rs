use anyhow::{Context, bail};
use juice_core::{ProofContext, ProofTarget, Span};
use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    pub text: String,
    pub span: Span,
    pub line_indent_col: usize,
}

/// Lexical scanner for Lean 4 that accurately tracks byte offsets, 0-based line numbers,
/// and UTF-16 code-unit columns while skipping `--` line comments, nested `/- ... -/` block
/// comments, and `"..."` string literals.
pub fn scan_code_identifiers(source: &str) -> Vec<TokenSpan> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let mut idx = 0usize;
    let mut line = 0u32;
    let mut col_utf16 = 0u32;
    let mut line_indent_col = 0usize;
    let mut seen_non_ws_on_line = false;
    let mut block_comment_depth = 0usize;
    let mut in_line_comment = false;
    let mut in_string = false;

    while idx < bytes.len() {
        let rest = &source[idx..];
        let ch = rest.chars().next().expect("non-empty slice");
        let ch_bytes = ch.len_utf8();
        let ch_utf16 = ch.len_utf16() as u32;

        if ch == '\n' {
            idx += 1;
            line += 1;
            col_utf16 = 0;
            line_indent_col = 0;
            seen_non_ws_on_line = false;
            in_line_comment = false;
            continue;
        }

        if in_line_comment {
            idx += ch_bytes;
            col_utf16 += ch_utf16;
            continue;
        }

        if block_comment_depth > 0 {
            if rest.starts_with("/-") {
                block_comment_depth += 1;
                idx += 2;
                col_utf16 += 2;
            } else if rest.starts_with("-/") {
                block_comment_depth -= 1;
                idx += 2;
                col_utf16 += 2;
            } else {
                idx += ch_bytes;
                col_utf16 += ch_utf16;
            }
            continue;
        }

        if in_string {
            if ch == '\\' {
                idx += ch_bytes;
                col_utf16 += ch_utf16;
                if idx < bytes.len() {
                    let escaped = source[idx..].chars().next().expect("escaped char");
                    if escaped == '\n' {
                        idx += 1;
                        line += 1;
                        col_utf16 = 0;
                        line_indent_col = 0;
                        seen_non_ws_on_line = false;
                    } else {
                        idx += escaped.len_utf8();
                        col_utf16 += escaped.len_utf16() as u32;
                    }
                }
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            idx += ch_bytes;
            col_utf16 += ch_utf16;
            continue;
        }

        // Not inside comment or string:
        if !seen_non_ws_on_line {
            if ch == ' ' {
                line_indent_col += 1;
            } else if ch == '\t' {
                line_indent_col += 2;
            } else {
                seen_non_ws_on_line = true;
            }
        }

        if rest.starts_with("--") {
            in_line_comment = true;
            idx += 2;
            col_utf16 += 2;
            continue;
        }

        if rest.starts_with("/-") {
            block_comment_depth = 1;
            idx += 2;
            col_utf16 += 2;
            continue;
        }

        if ch == '"' {
            in_string = true;
            idx += 1;
            col_utf16 += 1;
            continue;
        }

        if is_lean_ident_start(ch) {
            let start_byte = idx;
            let start_line = line;
            let start_col_utf16 = col_utf16;
            let token_line_indent = line_indent_col;

            idx += ch_bytes;
            col_utf16 += ch_utf16;
            while idx < bytes.len() {
                let next_ch = source[idx..].chars().next().expect("ident continuation");
                if is_lean_ident_continue(next_ch) {
                    idx += next_ch.len_utf8();
                    col_utf16 += next_ch.len_utf16() as u32;
                } else {
                    break;
                }
            }
            tokens.push(TokenSpan {
                text: source[start_byte..idx].to_string(),
                span: Span {
                    start_byte,
                    end_byte: idx,
                    start_line,
                    start_col_utf16,
                    end_line: line,
                    end_col_utf16: col_utf16,
                },
                line_indent_col: token_line_indent,
            });
            continue;
        }

        idx += ch_bytes;
        col_utf16 += ch_utf16;
    }

    tokens
}

fn is_lean_ident_start(ch: char) -> bool {
    ch.is_alphabetic() || ch == '_'
}

fn is_lean_ident_continue(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_' || ch == '\'' || ch == '?' || ch == '!'
}

#[derive(Debug, Clone)]
struct DeclHeader {
    kind: String,
    name: String,
    start_byte: usize,
    signature_end_byte: usize,
}

/// Extracts `import ...` lines at the top of a Lean 4 file.
pub fn split_imports_header(source: &str) -> (String, usize) {
    let mut end_offset = 0usize;
    let mut cursor = 0usize;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim();
        let next_cursor = cursor + line.len();
        if trimmed.starts_with("import ") {
            end_offset = next_cursor;
        } else if !trimmed.is_empty() && !trimmed.starts_with("--") && end_offset > 0 {
            break;
        }
        cursor = next_cursor;
    }
    (source[..end_offset].to_string(), end_offset)
}

/// Discovers all unfilled `sorry` proof targets in a Lean 4 source string.
pub fn extract_proof_targets(
    source: &str,
    workspace_root: &Path,
    source_file: Option<&Path>,
) -> Vec<ProofTarget> {
    let tokens = scan_code_identifiers(source);
    let (imports_header, imports_end_byte) = split_imports_header(source);

    let decl_keywords = ["theorem", "lemma", "def", "abbrev", "example"];
    let mut decls: Vec<DeclHeader> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        let tok = &tokens[i];
        if decl_keywords.contains(&tok.text.as_str()) {
            let kind = tok.text.clone();
            let name = if kind == "example" {
                format!("example_L{}", tok.span.start_line + 1)
            } else if let Some(next_tok) = tokens.get(i + 1) {
                next_tok.text.clone()
            } else {
                "anonymous".to_string()
            };
            let start_byte = tok.span.start_byte;
            let sig_end = find_decl_assign_offset(source, start_byte).unwrap_or(source.len());
            decls.push(DeclHeader {
                kind,
                name,
                start_byte,
                signature_end_byte: sig_end,
            });
        }
        i += 1;
    }

    let file_label = source_file
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("Scratch.lean");

    let mut targets = Vec::new();
    let mut sorry_index = 0usize;

    for (tok_idx, tok) in tokens.iter().enumerate() {
        if tok.text != "sorry" {
            continue;
        }

        let enclosing = decls
            .iter()
            .rfind(|d| d.start_byte <= tok.span.start_byte);

        let (decl_kind, decl_name, decl_start, sig_end) = match enclosing {
            Some(d) => (
                d.kind.clone(),
                d.name.clone(),
                d.start_byte,
                d.signature_end_byte.min(tok.span.start_byte),
            ),
            None => (
                "theorem".to_string(),
                format!("goal_{sorry_index}"),
                imports_end_byte,
                tok.span.start_byte,
            ),
        };

        let decl_signature = source[decl_start..sig_end]
            .trim()
            .trim_end_matches(":=")
            .trim()
            .to_string();

        // Determine if `sorry` is directly after `:=` (term-mode) or inside a `by` block.
        let prev_ident = tok_idx.checked_sub(1).and_then(|pi| tokens.get(pi));
        let between_assign_and_sorry = &source[sig_end..tok.span.start_byte];
        let has_by_in_decl = tokens
            .iter()
            .take(tok_idx)
            .rev()
            .take_while(|t| t.span.start_byte >= sig_end)
            .any(|t| t.text == "by" || t.text == "=>");
        let is_term_mode_sorry = !has_by_in_decl
            && prev_ident.map(|t| t.text.as_str()) != Some("by")
            && between_assign_and_sorry.contains(":=");

        let indent_col = if tok.span.start_col_utf16 as usize == tok.line_indent_col {
            tok.line_indent_col
        } else {
            tok.line_indent_col + 2
        };

        let preamble = if decl_start > imports_end_byte {
            source[imports_end_byte..decl_start].trim().to_string()
        } else {
            String::new()
        };

        let id = format!("{file_label}::{decl_name}#{sorry_index}");
        sorry_index += 1;

        targets.push(ProofTarget {
            id,
            decl_name,
            decl_kind,
            decl_signature,
            full_source: source.to_string(),
            sorry_span: tok.span,
            indent_col,
            is_term_mode_sorry,
            context: ProofContext {
                workspace_root: workspace_root.to_path_buf(),
                source_file: source_file.map(Path::to_path_buf),
                imports_header: imports_header.clone(),
                preamble,
            },
        });
    }

    targets
}

fn find_decl_assign_offset(source: &str, decl_start: usize) -> Option<usize> {
    let slice = &source[decl_start..];
    let mut paren_depth = 0i32;
    let mut brace_depth = 0i32;
    let mut bracket_depth = 0i32;
    let bytes = slice.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren_depth += 1,
            b')' => paren_depth = (paren_depth - 1).max(0),
            b'{' => brace_depth += 1,
            b'}' => brace_depth = (brace_depth - 1).max(0),
            b'[' => bracket_depth += 1,
            b']' => bracket_depth = (bracket_depth - 1).max(0),
            b':' if i + 1 < bytes.len()
                && bytes[i + 1] == b'='
                && paren_depth == 0
                && brace_depth == 0
                && bracket_depth == 0 =>
            {
                return Some(decl_start + i);
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Formats a sequence of tactics (plus an optional trailing `sorry` for partial exploration)
/// into the target's `sorry` span, returning `(modified_source, cursor_line, cursor_col_utf16)`.
pub fn render_target_with_tactics(
    target: &ProofTarget,
    tactics: &[String],
    keep_trailing_sorry: bool,
) -> (String, u32, u32) {
    let indent = " ".repeat(target.indent_col.max(2));
    let mut lines: Vec<String> = Vec::new();

    for t in tactics {
        for sub_line in t.lines() {
            lines.push(sub_line.to_string());
        }
    }
    if keep_trailing_sorry || lines.is_empty() {
        lines.push("sorry".to_string());
    }

    let replacement = if target.is_term_mode_sorry {
        let body = lines
            .iter()
            .map(|l| format!("{indent}{l}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("by\n{body}")
    } else if lines.len() == 1 {
        lines[0].clone()
    } else {
        let mut out = lines[0].clone();
        for l in &lines[1..] {
            out.push('\n');
            out.push_str(&indent);
            out.push_str(l);
        }
        out
    };

    let prefix = &target.full_source[..target.sorry_span.start_byte];
    let suffix = &target.full_source[target.sorry_span.end_byte..];
    let combined = format!("{prefix}{replacement}{suffix}");

    // Compute exact UTF-16 line/col at the start of the trailing `sorry` (if present)
    // or at the end of the replacement block so `$/lean/plainGoal` queries the exact open state.
    let probe_byte_offset = if keep_trailing_sorry {
        prefix.len()
            + replacement
                .rfind("sorry")
                .unwrap_or(replacement.len())
    } else {
        prefix.len() + replacement.len()
    };

    let (cursor_line, cursor_col_utf16) = byte_offset_to_utf16_pos(&combined, probe_byte_offset);
    (combined, cursor_line, cursor_col_utf16)
}

pub fn byte_offset_to_utf16_pos(source: &str, byte_offset: usize) -> (u32, u32) {
    let clamped = byte_offset.min(source.len());
    let mut line = 0u32;
    let mut col_utf16 = 0u32;
    for ch in source[..clamped].chars() {
        if ch == '\n' {
            line += 1;
            col_utf16 = 0;
        } else {
            col_utf16 += ch.len_utf16() as u32;
        }
    }
    (line, col_utf16)
}

/// Safely splices verified tactic scripts into one or more non-overlapping `sorry` spans
/// and guarantees byte-for-byte preservation of all text outside the target `sorry` spans.
pub fn splice_verified_proofs(
    original: &str,
    patches: &[(&ProofTarget, String)],
) -> anyhow::Result<String> {
    if patches.is_empty() {
        return Ok(original.to_string());
    }

    let mut sorted: Vec<(&ProofTarget, &str)> =
        patches.iter().map(|(t, p)| (*t, p.as_str())).collect();
    sorted.sort_by_key(|(t, _)| t.sorry_span.start_byte);

    for pair in sorted.windows(2) {
        if pair[0].0.sorry_span.end_byte > pair[1].0.sorry_span.start_byte {
            bail!("Overlapping sorry spans cannot be spliced simultaneously");
        }
    }

    let mut output = String::with_capacity(original.len() + 256);
    let mut cursor = 0usize;

    for (target, proof_script) in &sorted {
        let span = target.sorry_span;
        if span.end_byte > original.len() || &original[span.start_byte..span.end_byte] != "sorry" {
            bail!(
                "Target span {}..{} no longer matches literal `sorry`",
                span.start_byte,
                span.end_byte
            );
        }

        let forbidden = scan_forbidden_constructs(proof_script);
        if !forbidden.is_empty() {
            bail!(
                "Rejected candidate proof for `{}` due to forbidden constructs: {}",
                target.decl_name,
                forbidden.join(", ")
            );
        }

        output.push_str(&original[cursor..span.start_byte]);

        let indent = " ".repeat(target.indent_col.max(2));
        let proof_lines: Vec<&str> = proof_script.lines().collect();
        if target.is_term_mode_sorry {
            output.push_str("by\n");
            for (idx, pl) in proof_lines.iter().enumerate() {
                if idx > 0 {
                    output.push('\n');
                }
                output.push_str(&indent);
                output.push_str(pl);
            }
        } else if proof_lines.len() <= 1 {
            output.push_str(proof_script.trim());
        } else {
            output.push_str(proof_lines[0]);
            for pl in &proof_lines[1..] {
                output.push('\n');
                output.push_str(&indent);
                output.push_str(pl);
            }
        }

        cursor = span.end_byte;
    }

    output.push_str(&original[cursor..]);

    // Anti-tamper check: verify all non-sorry segments appear verbatim in order.
    verify_non_sorry_preservation(original, &output, &sorted)
        .context("Statement anti-tamper integrity check failed")?;

    Ok(output)
}

fn verify_non_sorry_preservation(
    original: &str,
    patched: &str,
    sorted_patches: &[(&ProofTarget, &str)],
) -> anyhow::Result<()> {
    let mut orig_cursor = 0usize;
    let mut search_from = 0usize;
    for (target, _) in sorted_patches {
        let preserved = &original[orig_cursor..target.sorry_span.start_byte];
        if !patched[search_from..].starts_with(preserved) {
            bail!(
                "Bytes preceding `{}` were unexpectedly altered",
                target.decl_name
            );
        }
        search_from += preserved.len();
        orig_cursor = target.sorry_span.end_byte;
        // Advance `search_from` to the next preserved anchor after the spliced proof.
        let next_anchor_end = sorted_patches
            .iter()
            .find(|(nt, _)| nt.sorry_span.start_byte >= orig_cursor)
            .map(|(nt, _)| nt.sorry_span.start_byte)
            .unwrap_or(original.len());
        let trailing_anchor = &original[orig_cursor..next_anchor_end];
        if !trailing_anchor.is_empty() {
            let rel = patched[search_from..]
                .find(trailing_anchor)
                .ok_or_else(|| anyhow::anyhow!("Trailing declaration text altered"))?;
            search_from += rel;
        }
    }
    Ok(())
}

/// Scans Lean proof snippets for soundness-bypassing metaprogramming or axiom constructs.
pub fn scan_forbidden_constructs(code: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let tokens = scan_code_identifiers(code);
    for tok in &tokens {
        match tok.text.as_str() {
            "axiom" => violations.push("axiom declaration".to_string()),
            "unsafe" => violations.push("unsafe keyword".to_string()),
            "sorry" | "admit" | "sorryAx" => {
                violations.push(format!("unclosed placeholder `{}`", tok.text))
            }
            "implemented_by" | "extern" => {
                violations.push(format!("FFI/compiler override `{}`", tok.text))
            }
            "elab_rules" | "macro_rules" => {
                violations.push(format!("metaprogramming rule `{}`", tok.text))
            }
            _ => {}
        }
    }
    violations.sort();
    violations.dedup();
    violations
}

/// Extracts `"Try this: <tactic>"` suggestions emitted by Lean's `exact?`, `apply?`, `rw?`, `simp?`.
pub fn extract_try_this_suggestion(diagnostic_msg: &str) -> Option<String> {
    static TRY_THIS_RE: OnceLock<Regex> = OnceLock::new();
    let re = TRY_THIS_RE.get_or_init(|| {
        Regex::new(r"(?m)Try this:\s*(?:\n\s*)?(.+)$").expect("valid Try this regex")
    });
    let caps = re.captures(diagnostic_msg)?;
    let suggestion = caps.get(1)?.as_str().trim().to_string();
    if suggestion.is_empty() {
        None
    } else {
        Some(suggestion)
    }
}

/// Extracts top-level declared theorem/lemma/def names in a Lean 4 file for `#print axioms` auditing.
pub fn list_auditable_declarations(source: &str) -> Vec<String> {
    let tokens = scan_code_identifiers(source);
    let mut names = Vec::new();
    let mut current_namespaces: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        match tokens[i].text.as_str() {
            "namespace" => {
                if let Some(ns) = tokens.get(i + 1) {
                    current_namespaces.push(ns.text.clone());
                }
            }
            "end" => {
                if let Some(ns) = tokens.get(i + 1)
                    && current_namespaces.last() == Some(&ns.text)
                {
                    current_namespaces.pop();
                }
            }
            "theorem" | "lemma" | "def" | "abbrev" => {
                if let Some(name_tok) = tokens.get(i + 1) {
                    let full_name = if current_namespaces.is_empty() {
                        name_tok.text.clone()
                    } else {
                        format!("{}.{}", current_namespaces.join("."), name_tok.text)
                    };
                    if !names.contains(&full_name) {
                        names.push(full_name);
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn ignores_sorry_in_nested_comments_and_strings_and_tracks_utf16() {
        let src = r#"import Mathlib
-- sorry in a line comment
/- outer comment
   /- nested sorry inside block comment -/
   still commented sorry
-/
def msg : String := "this sorry is inside a string literal"

theorem unicode_test (n : ℕ) (x : 𝔽) : n + 0 = n := by
  sorry
"#;
        let targets = extract_proof_targets(src, Path::new("/tmp"), Some(Path::new("Test.lean")));
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].decl_name, "unicode_test");
        assert!(!targets[0].is_term_mode_sorry);
        assert_eq!(targets[0].sorry_span.start_line, 9);
        assert_eq!(targets[0].sorry_span.start_col_utf16, 2);
        // Note: `𝔽` (\u{1D53D}) is 4 UTF-8 bytes and 2 UTF-16 code units; verify UTF-16 helper!
        let f_pos = src.find('𝔽').unwrap() + '𝔽'.len_utf8();
        let (line, col16) = byte_offset_to_utf16_pos(src, f_pos);
        assert_eq!(line, 8);
        // "theorem unicode_test (n : ℕ) (x : " is 34 chars (all 1 UTF-16 unit) + '𝔽' (2 UTF-16 units) = 36
        assert_eq!(col16, 36);
    }

    #[test]
    fn splices_term_mode_and_tactic_mode_proofs_and_blocks_cheats() {
        let src = "theorem t1 (n : Nat) : n = n := sorry\n\ntheorem t2 (n : Nat) : n + 0 = n := by\n  sorry\n";
        let targets = extract_proof_targets(src, &PathBuf::from("/tmp"), None);
        assert_eq!(targets.len(), 2);
        assert!(targets[0].is_term_mode_sorry);
        assert!(!targets[1].is_term_mode_sorry);

        let patched = splice_verified_proofs(
            src,
            &[(&targets[0], "rfl".to_string()), (&targets[1], "simp\n".to_string())],
        )
        .expect("clean splice");
        assert!(patched.contains("theorem t1 (n : Nat) : n = n := by\n  rfl"));
        assert!(patched.contains("theorem t2 (n : Nat) : n + 0 = n := by\n  simp"));

        let cheat_err =
            splice_verified_proofs(src, &[(&targets[0], "axiom bad : False\nexact False.elim bad".to_string())]);
        assert!(cheat_err.is_err());
    }

    #[test]
    fn extracts_try_this_from_lean_search_tactics() {
        let msg = "Try this: exact Nat.add_comm a b";
        assert_eq!(
            extract_try_this_suggestion(msg).as_deref(),
            Some("exact Nat.add_comm a b")
        );
        let multiline = "info: Try this:\n  simp only [Nat.add_zero]";
        assert_eq!(
            extract_try_this_suggestion(multiline).as_deref(),
            Some("simp only [Nat.add_zero]")
        );
    }
}
