use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticMsg {
    pub line: u32,
    pub column: u32,
    pub severity: DiagnosticSeverity,
    pub message: String,
}

/// Canonical pretty-printed goal state stripped of ephemeral macro hygiene suffixes
/// so equivalent logical proof states merge cleanly in the MCGS DAG.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NormalizedGoal {
    pub canonical_pp: String,
}

impl NormalizedGoal {
    pub fn new(raw: impl AsRef<str>) -> Self {
        Self {
            canonical_pp: Self::canonicalize(raw.as_ref()),
        }
    }

    /// Parses Lean's `$/lean/plainGoal` response into zero or more normalized goals.
    pub fn parse_multi(raw: &str) -> Vec<Self> {
        let trimmed = raw.trim();
        let lower = trimmed.to_lowercase();
        if trimmed.is_empty()
            || lower == "no goals"
            || lower.contains("goals accomplished")
        {
            return Vec::new();
        }

        // Strip optional fenced markdown blocks returned by newer Lean LSP versions.
        let stripped = trimmed
            .strip_prefix("```lean")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed);
        let stripped = stripped.strip_suffix("```").unwrap_or(stripped).trim();
        let stripped_lower = stripped.to_lowercase();
        if stripped.is_empty()
            || stripped_lower == "no goals"
            || stripped_lower.contains("goals accomplished")
        {
            return Vec::new();
        }

        // Lean separates multiple open goals by blank lines (\n\n) where each block contains `⊢`.
        let mut goals = Vec::new();
        for chunk in stripped.split("\n\n") {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }
            goals.push(Self::new(chunk));
        }
        goals
    }

    pub fn canonicalize(input: &str) -> String {
        static HYGIENE_RE: OnceLock<Regex> = OnceLock::new();
        let re = HYGIENE_RE.get_or_init(|| {
            Regex::new(r"\._@[A-Za-z0-9_.]*_hyg\.[0-9]+").expect("valid hygiene regex")
        });
        let without_hygiene = re.replace_all(input, "");
        without_hygiene
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: u32,
    pub start_col_utf16: u32,
    pub end_line: u32,
    pub end_col_utf16: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofContext {
    pub workspace_root: PathBuf,
    pub source_file: Option<PathBuf>,
    pub imports_header: String,
    pub preamble: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofTarget {
    pub id: String,
    pub decl_name: String,
    pub decl_kind: String,
    pub decl_signature: String,
    pub full_source: String,
    pub sorry_span: Span,
    pub indent_col: usize,
    /// True when the `sorry` is directly in term position (`:= sorry`) rather than inside a `by` block.
    pub is_term_mode_sorry: bool,
    pub context: ProofContext,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofStateSnapshot {
    pub target: ProofTarget,
    /// Tactic lines applied sequentially at the target `sorry` hole.
    pub tactic_prefix: Vec<String>,
    pub open_goals: Vec<NormalizedGoal>,
}

impl ProofStateSnapshot {
    pub fn is_solved(&self) -> bool {
        self.open_goals.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TacticStepResult {
    pub accepted: bool,
    pub next_state: Option<ProofStateSnapshot>,
    pub remaining_goals: Vec<NormalizedGoal>,
    pub diagnostics: Vec<DiagnosticMsg>,
    /// Populated when a search tactic (`exact?`, `apply?`, `rw?`, `simp?`) emits `"Try this: ..."`.
    pub suggested_tactic: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SketchSubLemma {
    pub name: String,
    pub statement: String,
    pub informal_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchValidation {
    pub valid: bool,
    pub compiled_sketch: String,
    pub sub_targets: Vec<ProofTarget>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CounterexampleResult {
    Found { witness: String },
    NoCounterexampleFound,
    Inconclusive(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VacuityCheck {
    pub is_vacuous: bool,
    pub witness_proof: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PremiseHit {
    pub name: String,
    pub kind: String,
    pub signature: String,
    pub module: String,
    pub doc: Option<String>,
    pub score: f64,
    pub verified_by_kernel: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxiomPolicy {
    pub allowed_axioms: Vec<String>,
    pub allow_native_decide: bool,
}

impl Default for AxiomPolicy {
    fn default() -> Self {
        Self::strict()
    }
}

impl AxiomPolicy {
    pub fn strict() -> Self {
        Self {
            allowed_axioms: vec![
                "propext".to_string(),
                "Classical.choice".to_string(),
                "Quot.sound".to_string(),
            ],
            allow_native_decide: false,
        }
    }

    pub fn with_native_decide(mut self, allow: bool) -> Self {
        self.allow_native_decide = allow;
        if allow {
            for ax in ["Lean.ofReduceBool", "Lean.trustCompiler"] {
                if !self.allowed_axioms.iter().any(|existing| existing == ax) {
                    self.allowed_axioms.push(ax.to_string());
                }
            }
        }
        self
    }

    pub fn is_axiom_allowed(&self, axiom_name: &str) -> bool {
        self.allowed_axioms.iter().any(|a| a == axiom_name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditReport {
    pub verified: bool,
    pub declarations_checked: Vec<String>,
    pub axioms_used: Vec<String>,
    pub forbidden_axioms: Vec<String>,
    pub has_sorry: bool,
    pub forbidden_constructs: Vec<String>,
    pub raw_output: String,
}

#[async_trait]
pub trait VerifierBackend: Send + Sync {
    fn backend_name(&self) -> &'static str;

    /// Discover unfilled proof holes (`sorry` targets) in a file or project directory.
    async fn discover_targets(&self, path: &Path) -> anyhow::Result<Vec<ProofTarget>>;

    /// Obtain the initial goal state snapshot at a target `sorry`.
    async fn initial_state(&self, target: &ProofTarget) -> anyhow::Result<ProofStateSnapshot>;

    /// Validate that a high-level sketch (containing `have h_i : ... := by sorry` sub-lemmas)
    /// typechecks and closes the outer theorem modulo those inner `sorry` holes.
    async fn validate_sketch(
        &self,
        target: &ProofTarget,
        sketch_body: &str,
    ) -> anyhow::Result<SketchValidation>;

    /// Probe a candidate lemma statement for concrete counterexamples (e.g. via `plausible` / `decide`).
    async fn probe_counterexample(
        &self,
        context: &ProofContext,
        lemma_stmt: &str,
    ) -> anyhow::Result<CounterexampleResult>;

    /// Check if a formalized theorem's hypotheses are contradictory (`hypotheses ⊢ False`).
    async fn check_vacuity(
        &self,
        context: &ProofContext,
        theorem_decl: &str,
    ) -> anyhow::Result<VacuityCheck>;

    /// Typecheck a full declaration snippet and return compiler diagnostics.
    async fn check_declaration(
        &self,
        context: &ProofContext,
        code: &str,
    ) -> anyhow::Result<Vec<DiagnosticMsg>>;

    /// Evaluate a single tactic or tactic block against an open proof state snapshot.
    async fn apply_tactic(
        &self,
        state: &ProofStateSnapshot,
        tactic: &str,
    ) -> anyhow::Result<TacticStepResult>;

    /// Search verified workspace / library premises matching a goal or query string.
    async fn search_premises(
        &self,
        context: &ProofContext,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<PremiseHit>>;

    /// Perform cold, out-of-process kernel certification and `#print axioms` audit.
    async fn cold_audit(
        &self,
        file_path: &Path,
        decl_names: &[String],
        policy: &AxiomPolicy,
    ) -> anyhow::Result<AuditReport>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_lean_hygiene_and_splits_multigoals() {
        let raw = "```lean\ncase succ\nn._@.Mathlib.Init._hyg.412 : Nat\n⊢ n._@.Mathlib.Init._hyg.412 + 0 = n._@.Mathlib.Init._hyg.412\n\ncase zero\n⊢ 0 + 0 = 0\n```";
        let goals = NormalizedGoal::parse_multi(raw);
        assert_eq!(goals.len(), 2);
        assert_eq!(goals[0].canonical_pp, "case succ\nn : Nat\n⊢ n + 0 = n");
        assert_eq!(goals[1].canonical_pp, "case zero\n⊢ 0 + 0 = 0");
    }

    #[test]
    fn empty_and_accomplished_goals_parse_as_solved() {
        assert!(NormalizedGoal::parse_multi("").is_empty());
        assert!(NormalizedGoal::parse_multi("no goals").is_empty());
        assert!(NormalizedGoal::parse_multi("```lean\nGoals accomplished! 🎉\n```").is_empty());
    }

    #[test]
    fn strict_axiom_policy_rejects_sorry_and_native_decide_by_default() {
        let policy = AxiomPolicy::strict();
        assert!(policy.is_axiom_allowed("propext"));
        assert!(policy.is_axiom_allowed("Classical.choice"));
        assert!(policy.is_axiom_allowed("Quot.sound"));
        assert!(!policy.is_axiom_allowed("sorryAx"));
        assert!(!policy.is_axiom_allowed("Lean.trustCompiler"));

        let relaxed = AxiomPolicy::strict().with_native_decide(true);
        assert!(relaxed.is_axiom_allowed("Lean.trustCompiler"));
        assert!(relaxed.is_axiom_allowed("Lean.ofReduceBool"));
        assert!(!relaxed.is_axiom_allowed("sorryAx"));
    }
}
