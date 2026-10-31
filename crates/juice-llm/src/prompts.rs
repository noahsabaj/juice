use juice_core::{NormalizedGoal, PremiseHit, SketchSubLemma};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoformalizedClaim {
    pub decl_name: String,
    pub imports: Vec<String>,
    pub definitions: String,
    pub theorem_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackTranslationVerdict {
    pub back_translation: String,
    pub faithful: bool,
    pub critique: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedSketch {
    pub informal_plan: String,
    pub sub_lemmas: Vec<SketchSubLemma>,
    /// Complete tactic block for the outer theorem using `have h_i : ... := by sorry`.
    pub lean_sketch_tactics: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposedTactic {
    pub tactic: String,
    pub prior: f64,
}

/// Strips `<think>...</think>` blocks emitted by reasoning models (DeepSeek-R1, Goedel-Prover-V2, Qwen).
pub fn strip_think_blocks(raw: &str) -> String {
    static THINK_RE: OnceLock<Regex> = OnceLock::new();
    let re = THINK_RE
        .get_or_init(|| Regex::new(r"(?s)<think>.*?</think>").expect("valid think block regex"));
    re.replace_all(raw, "").trim().to_string()
}

pub fn extract_fenced_block(raw: &str, lang_hint: Option<&str>) -> Option<String> {
    let cleaned = strip_think_blocks(raw);
    if let Some(lang) = lang_hint {
        let marker = format!("```{lang}");
        if let Some(start) = cleaned.find(&marker) {
            let after = &cleaned[start + marker.len()..];
            let after = after.strip_prefix('\n').unwrap_or(after);
            if let Some(end) = after.find("```") {
                return Some(after[..end].trim().to_string());
            }
        }
    }
    if let Some(start) = cleaned.find("```") {
        let after = &cleaned[start + 3..];
        let after_first_line = after
            .find('\n')
            .map(|idx| &after[idx + 1..])
            .unwrap_or(after);
        if let Some(end) = after_first_line.find("```") {
            return Some(after_first_line[..end].trim().to_string());
        }
    }
    None
}

pub fn extract_json_payload<T: for<'de> Deserialize<'de>>(raw: &str) -> anyhow::Result<T> {
    let cleaned = strip_think_blocks(raw);
    if let Some(json_block) = extract_fenced_block(&cleaned, Some("json"))
        && let Ok(parsed) = serde_json::from_str::<T>(&json_block)
    {
        return Ok(parsed);
    }
    if let (Some(first_brace), Some(last_brace)) = (cleaned.find('{'), cleaned.rfind('}'))
        && first_brace < last_brace
    {
        let slice = &cleaned[first_brace..=last_brace];
        if let Ok(parsed) = serde_json::from_str::<T>(slice) {
            return Ok(parsed);
        }
    }
    Ok(serde_json::from_str::<T>(&cleaned)?)
}

pub fn build_autoformalize_prompt(
    claim_text: &str,
    has_mathlib: bool,
    premises: &[PremiseHit],
    prior_feedback: Option<&str>,
) -> (String, String) {
    let system = "You are an expert Lean 4 mathematician and formal verification engineer. \
Translate mathematical statements accurately into idiomatic Lean 4 declarations without weakening hypotheses, \
dropping domain constraints, or introducing vacuous assumptions. Never declare `axiom` or `unsafe`."
        .to_string();

    let import_note = if has_mathlib {
        "Mathlib is available (`import Mathlib`). Use standard Mathlib types and predicates."
    } else {
        "Only Lean 4 Core/Std is available (do NOT `import Mathlib`). Use standard Lean 4 types (`Nat`, `Int`, `List`, `Prop`)."
    };

    let mut premise_section = String::new();
    if !premises.is_empty() {
        premise_section.push_str("\nVerified workspace/library declarations you may reference:\n");
        for p in premises.iter().take(12) {
            premise_section.push_str(&format!("- `{}` : `{}`\n", p.name, p.signature));
        }
    }

    let feedback_section = prior_feedback
        .map(|f| format!("\nPrevious attempt failed verification/critique — fix this issue:\n{f}\n"))
        .unwrap_or_default();

    let user = format!(
        r#"Formalize the following mathematical claim into Lean 4.
{import_note}
{premise_section}{feedback_section}
Input claim:
\"\"\"
{claim_text}
\"\"\"

Respond ONLY with a JSON object matching this exact schema:
```json
{{
  "decl_name": " descriptive_snake_case_name ",
  "imports": ["Mathlib"],
  "definitions": "-- optional helper `def`s if needed, or empty string",
  "theorem_signature": "theorem descriptive_snake_case_name (...) : ..."
}}
```
Do NOT include `:= by sorry` at the end of `theorem_signature`; provide only the `theorem <name> <binders> : <type>` signature."#
    );

    (system, user)
}

pub fn build_back_translation_prompt(
    original_claim: &str,
    definitions: &str,
    theorem_signature: &str,
) -> (String, String) {
    let system = "You are a strict formal-methods semantic auditor. \
First translate the Lean 4 theorem signature blindly back into precise mathematical English, \
then check whether it faithfully captures EVERY hypothesis, edge case, quantifier, and conclusion of the original claim (neither weaker, stronger, nor vacuously trivial)."
        .to_string();

    let user = format!(
        r#"Audit this Lean 4 formalization against the original mathematical statement.

Original statement:
\"\"\"
{original_claim}
\"\"\"

Lean 4 auxiliary definitions:
```lean
{definitions}
```

Lean 4 theorem signature:
```lean
{theorem_signature}
```

Respond ONLY with JSON:
```json
{{
  "back_translation": "Plain mathematical English translation of the Lean 4 signature",
  "faithful": true,
  "critique": "If faithful is false, explain the exact semantic mismatch or missing constraint; otherwise 'Exact match.'"
}}
```"#
    );

    (system, user)
}

pub fn build_planner_sketch_prompt(
    decl_signature: &str,
    open_goals: &[NormalizedGoal],
    premises: &[PremiseHit],
    refuted_lemmas: &[(String, String)],
    prior_errors: &[String],
) -> (String, String) {
    let system = "You are Tier-1 Planner in a formal Lean 4 reasoning engine (Draft-Sketch-Prove architecture). \
Decompose hard theorems into 1–4 crisp intermediate `have` sub-lemmas whose proofs are left as `by sorry`, \
followed by tactics that close the main goal using those `have` hypotheses. \
Never declare `axiom` or `unsafe`, and never repeat a sub-lemma already refuted by a counterexample."
        .to_string();

    let goals_pp = open_goals
        .iter()
        .map(|g| g.canonical_pp.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut premise_text = String::new();
    if !premises.is_empty() {
        premise_text.push_str("\nVerified local/Mathlib theorems available in scope:\n");
        for p in premises.iter().take(15) {
            premise_text.push_str(&format!("- `{}` : `{}`\n", p.name, p.signature));
        }
    }

    let mut refuted_text = String::new();
    if !refuted_lemmas.is_empty() {
        refuted_text.push_str("\nKNOWN FALSE SUB-LEMMAS (refuted by counterexample — DO NOT PROPOSE THESE):\n");
        for (stmt, witness) in refuted_lemmas {
            refuted_text.push_str(&format!("- FALSE: `{stmt}` (counterexample: {witness})\n"));
        }
    }

    let mut err_text = String::new();
    if !prior_errors.is_empty() {
        err_text.push_str("\nPrevious sketch compiler errors to avoid:\n");
        for e in prior_errors.iter().take(4) {
            err_text.push_str(&format!("- {e}\n"));
        }
    }

    let user = format!(
        r#"Decompose this Lean 4 theorem into an executable `have`-lemma proof sketch.

Declaration:
```lean
{decl_signature}
```

Current goal state:
```lean
{goals_pp}
```
{premise_text}{refuted_text}{err_text}
Respond ONLY with JSON matching this schema:
```json
{{
  "informal_plan": "Concise mathematical proof strategy",
  "sub_lemmas": [
    {{
      "name": "h1",
      "statement": "exact Lean 4 proposition type of h1",
      "informal_reason": "why h1 holds"
    }}
  ],
  "lean_sketch_tactics": "have h1 : <type> := by\n  sorry\nexact ..."
}}
```
Important: `lean_sketch_tactics` is placed directly inside a `by` block (do NOT wrap it in `theorem ... := by`), and MUST close the outer goal assuming each `have h_i : ... := by sorry`."#
    );

    (system, user)
}

pub fn build_tactic_sampling_prompt(
    decl_signature: &str,
    tactic_prefix: &[String],
    open_goals: &[NormalizedGoal],
    premises: &[PremiseHit],
    k: usize,
) -> (String, String) {
    let system = "You are a Lean 4 tactic prediction engine. Given the current open tactic goal state, \
propose distinct, high-probability next Lean 4 tactic steps (each either a single tactic line like `induction n with | zero => simp | succ n ih => sorry` or atomic step like `intro h`, `rw [...]`, `apply ...`, `have h : ... := ...`). \
Never emit `sorry`, `admit`, or `axiom`."
        .to_string();

    let goals_pp = open_goals
        .iter()
        .map(|g| g.canonical_pp.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let prefix_str = if tactic_prefix.is_empty() {
        "(at root of `by` block)".to_string()
    } else {
        tactic_prefix.join("\n")
    };

    let mut premise_text = String::new();
    if !premises.is_empty() {
        premise_text.push_str("\nVerified premises in scope:\n");
        for p in premises.iter().take(10) {
            premise_text.push_str(&format!("- `{}` : `{}`\n", p.name, p.signature));
        }
    }

    let user = format!(
        r#"Theorem: `{decl_signature}`
Tactics applied so far:
```lean
{prefix_str}
```

Remaining Lean 4 goal state:
```lean
{goals_pp}
```
{premise_text}
Return up to {k} distinct candidate next Lean 4 tactics ordered from most promising to least promising as JSON:
```json
{{
  "tactics": [
    "tactic_candidate_1",
    "tactic_candidate_2"
  ]
}}
```"#
    );

    (system, user)
}

#[derive(Deserialize)]
struct TacticListEnvelope {
    tactics: Vec<String>,
}

/// Parses tactic candidates from either structured JSON (`{"tactics": [...]}`)
/// or raw Lean code outputs emitted by specialized fine-tuned models (e.g. DeepSeek-Prover / Goedel-Prover).
pub fn parse_tactic_candidates(raw: &str) -> Vec<String> {
    if let Ok(env) = extract_json_payload::<TacticListEnvelope>(raw) {
        return sanitize_tactics(env.tactics);
    }

    let cleaned = strip_think_blocks(raw);
    if let Some(block) = extract_fenced_block(&cleaned, Some("lean"))
        .or_else(|| extract_fenced_block(&cleaned, None))
    {
        return sanitize_tactics(
            block
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with("--"))
                .map(ToString::to_string)
                .collect(),
        );
    }

    sanitize_tactics(
        cleaned
            .lines()
            .map(|l| l.trim().trim_start_matches("- ").trim_matches('`'))
            .filter(|l| !l.is_empty() && !l.starts_with("--") && !l.starts_with('#'))
            .map(ToString::to_string)
            .collect(),
    )
}

fn sanitize_tactics(items: Vec<String>) -> Vec<String> {
    let mut seen = Vec::new();
    for item in items {
        let trimmed = item.trim().to_string();
        if trimmed.is_empty()
            || trimmed == "sorry"
            || trimmed == "admit"
            || trimmed.contains("axiom ")
            || trimmed.contains("sorryAx")
        {
            continue;
        }
        if !seen.contains(&trimmed) {
            seen.push(trimmed);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_reasoning_think_tags_and_parses_json() {
        let raw = r#"<think>
Let's carefully analyze the goal `⊢ n + 0 = n`.
We can use `Nat.add_zero` or `rfl`.
</think>
```json
{
  "tactics": ["rfl", "simp", "sorry", "exact Nat.add_zero n"]
}
```"#;
        let tactics = parse_tactic_candidates(raw);
        assert_eq!(tactics, vec!["rfl", "simp", "exact Nat.add_zero n"]);
    }

    #[test]
    fn parses_raw_lean_block_from_prover_model() {
        let raw = "<think>induction</think>\n```lean\nintro n\ninduction n\n```";
        let tactics = parse_tactic_candidates(raw);
        assert_eq!(tactics, vec!["intro n", "induction n"]);
    }
}
