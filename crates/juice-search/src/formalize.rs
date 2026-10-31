use crate::planner::TwoTierProver;
use anyhow::bail;
use juice_core::{
    AuditReport, AxiomPolicy, DiagnosticSeverity, EventSink, LeanConfig, ProofContext,
    RunCheckpoint, SearchConfig, SearchEvent, VerifierBackend,
};
use juice_lean::{
    parser::extract_proof_targets,
    workspace::resolve_workspace,
};
use juice_llm::ModelRouter;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone)]
pub struct AutoformalizeOptions {
    pub prove_theorems: bool,
    pub max_formalize_attempts: usize,
    pub policy: AxiomPolicy,
}

impl Default for AutoformalizeOptions {
    fn default() -> Self {
        Self {
            prove_theorems: true,
            max_formalize_attempts: 4,
            policy: AxiomPolicy::strict(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalizedItem {
    pub original_claim: String,
    pub decl_name: String,
    pub theorem_signature: String,
    pub back_translation: String,
    pub back_translation_faithful: bool,
    pub is_vacuous: bool,
    pub proved: bool,
    pub proof_script: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoformalizeOutcome {
    pub lean_source: String,
    pub items: Vec<FormalizedItem>,
    pub audit_report: Option<AuditReport>,
}

/// Translates natural-language math claims, Markdown notes, or LaTeX (`.tex`) papers
/// into non-vacuous, back-translation-audited, kernel-certified Lean 4 files.
pub struct Autoformalizer {
    verifier: Arc<dyn VerifierBackend>,
    router: ModelRouter,
    search_cfg: SearchConfig,
    lean_cfg: LeanConfig,
    events: EventSink,
}

impl Autoformalizer {
    pub fn new(
        verifier: Arc<dyn VerifierBackend>,
        router: ModelRouter,
        search_cfg: SearchConfig,
        lean_cfg: LeanConfig,
        events: EventSink,
    ) -> Self {
        Self {
            verifier,
            router,
            search_cfg,
            lean_cfg,
            events,
        }
    }

    pub async fn formalize_input(
        &self,
        raw_input: &str,
        target_output_path: Option<&Path>,
        opts: &AutoformalizeOptions,
    ) -> anyhow::Result<AutoformalizeOutcome> {
        let resolved_ws = resolve_workspace(target_output_path, &self.lean_cfg);
        let claims = extract_claims_from_document(raw_input);
        if claims.is_empty() {
            bail!("No mathematical claims or theorem blocks found in input");
        }

        let mut all_imports: BTreeSet<String> = BTreeSet::new();
        if resolved_ws.has_mathlib {
            all_imports.insert("Mathlib".to_string());
        }

        let mut accumulated_defs = String::new();
        let mut rendered_decls: Vec<String> = Vec::new();
        let mut items: Vec<FormalizedItem> = Vec::new();

        let prover = TwoTierProver::new(
            self.verifier.clone(),
            self.router.clone(),
            self.search_cfg.clone(),
            self.events.clone(),
        );
        let mut checkpoint = RunCheckpoint::new("formalize", "formalize");

        for claim in claims {
            let preview: String = claim.chars().take(72).collect();
            (self.events)(SearchEvent::Autoformalizing {
                claim_preview: preview,
            });

            let imports_header = render_imports(&all_imports);
            let base_context = ProofContext {
                workspace_root: resolved_ws.root.clone(),
                source_file: target_output_path.map(Path::to_path_buf),
                imports_header: imports_header.clone(),
                preamble: accumulated_defs.clone(),
            };

            let premises = self
                .verifier
                .search_premises(&base_context, &claim, 12)
                .await
                .unwrap_or_default();

            let mut feedback: Option<String> = None;
            let mut accepted_claim = None;
            let mut last_bt = String::new();
            let mut last_bt_ok = false;

            for _attempt in 1..=opts.max_formalize_attempts {
                let candidate = self
                    .router
                    .autoformalize_claim(&claim, resolved_ws.has_mathlib, &premises, feedback.as_deref())
                    .await?;

                let mut trial_imports = all_imports.clone();
                if resolved_ws.has_mathlib {
                    for imp in &candidate.imports {
                        let trimmed = imp.trim();
                        if !trimmed.is_empty() {
                            trial_imports.insert(trimmed.to_string());
                        }
                    }
                }

                let trial_ctx = ProofContext {
                    workspace_root: resolved_ws.root.clone(),
                    source_file: target_output_path.map(Path::to_path_buf),
                    imports_header: render_imports(&trial_imports),
                    preamble: format!("{}\n{}", accumulated_defs, candidate.definitions)
                        .trim()
                        .to_string(),
                };

                let stub_code = format!(
                    "{} := by\n  sorry",
                    candidate.theorem_signature.trim().trim_end_matches(":=")
                );

                // Gate 1: Lean 4 typecheck of the formalized signature
                let diags = self
                    .verifier
                    .check_declaration(&trial_ctx, &stub_code)
                    .await?;
                let errors: Vec<String> = diags
                    .iter()
                    .filter(|d| d.severity == DiagnosticSeverity::Error)
                    .map(|d| d.message.clone())
                    .collect();
                if !errors.is_empty() {
                    feedback = Some(format!(
                        "Lean 4 compiler rejected the formalized signature:\n{}",
                        errors.join("\n")
                    ));
                    continue;
                }

                // Gate 2: Vacuity check (contradictory hypotheses `hypotheses ⊢ False`)
                let vacuity = self
                    .verifier
                    .check_vacuity(&trial_ctx, &candidate.theorem_signature)
                    .await?;
                (self.events)(SearchEvent::VacuityChecked {
                    decl_name: candidate.decl_name.clone(),
                    is_vacuous: vacuity.is_vacuous,
                    reason: vacuity.reason.clone(),
                });
                if vacuity.is_vacuous {
                    feedback = Some(
                        vacuity
                            .reason
                            .unwrap_or_else(|| "Hypotheses are contradictory (vacuously true)".into()),
                    );
                    continue;
                }

                // Gate 3: Independent blind back-translation critique
                let bt = self
                    .router
                    .verify_back_translation(
                        &claim,
                        &candidate.definitions,
                        &candidate.theorem_signature,
                    )
                    .await?;
                last_bt = bt.back_translation.clone();
                last_bt_ok = bt.faithful;
                (self.events)(SearchEvent::BackTranslationChecked {
                    decl_name: candidate.decl_name.clone(),
                    faithful: bt.faithful,
                    critique: bt.critique.clone(),
                });

                if !bt.faithful {
                    feedback = Some(format!(
                        "Back-translation semantic mismatch: {} (Back-translated as: \"{}\")",
                        bt.critique, bt.back_translation
                    ));
                    continue;
                }

                all_imports = trial_imports;
                accepted_claim = Some(candidate);
                break;
            }

            let finalized = accepted_claim.ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to produce a type-valid, non-vacuous, semantically faithful Lean 4 formalization after {} attempts. Last feedback: {}",
                    opts.max_formalize_attempts,
                    feedback.unwrap_or_default()
                )
            })?;

            if !finalized.definitions.trim().is_empty()
                && !finalized.definitions.trim().starts_with("--")
            {
                accumulated_defs.push_str(finalized.definitions.trim());
                accumulated_defs.push_str("\n\n");
            }

            let sig_clean = finalized
                .theorem_signature
                .trim()
                .trim_end_matches(":=")
                .trim()
                .to_string();

            let mut proved = false;
            let mut proof_script = None;

            if opts.prove_theorems {
                let single_doc = format!(
                    "{}\n{}\n\n{} := by\n  sorry\n",
                    render_imports(&all_imports),
                    accumulated_defs.trim(),
                    sig_clean
                );
                let targets = extract_proof_targets(
                    &single_doc,
                    &resolved_ws.root,
                    target_output_path,
                );
                if let Some(target) = targets.last() {
                    let solve_rep = prover.prove_target(target, &mut checkpoint).await?;
                    proved = solve_rep.solved;
                    proof_script = solve_rep.proof_script;
                }
            }

            let rendered_body = if let Some(pf) = &proof_script {
                let indented = pf
                    .lines()
                    .map(|l| format!("  {l}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{sig_clean} := by\n{indented}")
            } else {
                format!("{sig_clean} := by\n  sorry")
            };
            rendered_decls.push(rendered_body);

            items.push(FormalizedItem {
                original_claim: claim,
                decl_name: finalized.decl_name,
                theorem_signature: sig_clean,
                back_translation: last_bt,
                back_translation_faithful: last_bt_ok,
                is_vacuous: false,
                proved,
                proof_script,
            });
        }

        let mut full_lean = String::new();
        let imports_block = render_imports(&all_imports);
        if !imports_block.is_empty() {
            full_lean.push_str(&imports_block);
            full_lean.push_str("\n\n");
        }
        if !accumulated_defs.trim().is_empty() {
            full_lean.push_str(accumulated_defs.trim());
            full_lean.push_str("\n\n");
        }
        full_lean.push_str(&rendered_decls.join("\n\n"));
        full_lean.push('\n');

        let mut audit_report = None;
        if let Some(out_path) = target_output_path {
            if let Some(parent) = out_path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(out_path, &full_lean)?;
            if opts.prove_theorems && items.iter().all(|it| it.proved) {
                let decl_names: Vec<String> = items.iter().map(|i| i.decl_name.clone()).collect();
                if let Ok(rep) = self
                    .verifier
                    .cold_audit(out_path, &decl_names, &opts.policy)
                    .await
                {
                    (self.events)(SearchEvent::ColdAuditCompleted {
                        decl_name: decl_names.join(", "),
                        verified: rep.verified,
                        axioms: rep.axioms_used.clone(),
                    });
                    audit_report = Some(rep);
                }
            }
        }

        Ok(AutoformalizeOutcome {
            lean_source: full_lean,
            items,
            audit_report,
        })
    }
}

fn render_imports(imports: &BTreeSet<String>) -> String {
    imports
        .iter()
        .map(|m| format!("import {m}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extracts mathematical claims from LaTeX environments (`\begin{theorem}...\end{theorem}`, etc.)
/// or returns the trimmed prose input as a standalone claim.
pub fn extract_claims_from_document(raw: &str) -> Vec<String> {
    static LATEX_THM_RE: OnceLock<Regex> = OnceLock::new();
    let thm_re = LATEX_THM_RE.get_or_init(|| {
        Regex::new(
            r"(?s)\\begin\{(theorem|lemma|proposition|corollary)\}(?:\[[^\]]*\])?(.*?)\\end\{(?:theorem|lemma|proposition|corollary)\}",
        )
        .expect("valid latex theorem regex")
    });
    static LATEX_DEF_RE: OnceLock<Regex> = OnceLock::new();
    let def_re = LATEX_DEF_RE.get_or_init(|| {
        Regex::new(r"(?s)\\begin\{definition\}(?:\[[^\]]*\])?(.*?)\\end\{definition\}")
            .expect("valid latex def regex")
    });

    if thm_re.is_match(raw) {
        let defs_context: Vec<String> = def_re
            .captures_iter(raw)
            .filter_map(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
            .collect();
        let def_prefix = if defs_context.is_empty() {
            String::new()
        } else {
            format!(
                "Context definitions from paper:\n{}\n\nTheorem to formalize:\n",
                defs_context.join("\n")
            )
        };

        return thm_re
            .captures_iter(raw)
            .filter_map(|c| {
                let body = c.get(2)?.as_str().trim();
                if body.is_empty() {
                    None
                } else {
                    Some(format!("{def_prefix}{body}"))
                }
            })
            .collect();
    }

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        Vec::new()
    } else {
        vec![trimmed.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use juice_core::{
        CounterexampleResult, DiagnosticMsg, NormalizedGoal, PremiseHit, ProofStateSnapshot,
        ProofTarget, SketchValidation, TacticStepResult, VacuityCheck, events::noop_event_sink,
    };
    use juice_llm::{CandidateOutput, CompletionRequest, LlmProvider};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn extracts_latex_definitions_and_theorems() {
        let tex = r#"
\documentclass{article}
\begin{document}
\begin{definition}[Even]
An integer $n$ is even if $n = 2k$ for some integer $k$.
\end{definition}
\begin{theorem}[Sum of Evens]
The sum of two even integers is even.
\end{theorem}
\begin{corollary}
$2n + 2m$ is even.
\end{corollary}
\end{document}
"#;
        let claims = extract_claims_from_document(tex);
        assert_eq!(claims.len(), 2);
        assert!(claims[0].contains("An integer $n$ is even"));
        assert!(claims[0].contains("The sum of two even integers is even."));
        assert!(claims[1].contains("$2n + 2m$ is even."));
    }

    struct VacuityMockVerifier;

    #[async_trait]
    impl VerifierBackend for VacuityMockVerifier {
        fn backend_name(&self) -> &'static str {
            "mock-vacuity"
        }
        async fn discover_targets(&self, _path: &Path) -> anyhow::Result<Vec<ProofTarget>> {
            Ok(vec![])
        }
        async fn initial_state(&self, target: &ProofTarget) -> anyhow::Result<ProofStateSnapshot> {
            Ok(ProofStateSnapshot {
                target: target.clone(),
                tactic_prefix: vec![],
                open_goals: vec![NormalizedGoal::new("n : Nat\n⊢ n + 0 = n")],
            })
        }
        async fn validate_sketch(
            &self,
            _target: &ProofTarget,
            _sketch_body: &str,
        ) -> anyhow::Result<SketchValidation> {
            unreachable!()
        }
        async fn probe_counterexample(
            &self,
            _context: &ProofContext,
            _lemma_stmt: &str,
        ) -> anyhow::Result<CounterexampleResult> {
            Ok(CounterexampleResult::NoCounterexampleFound)
        }
        async fn check_vacuity(
            &self,
            _context: &ProofContext,
            theorem_decl: &str,
        ) -> anyhow::Result<VacuityCheck> {
            if theorem_decl.contains("h_contra : False") {
                Ok(VacuityCheck {
                    is_vacuous: true,
                    witness_proof: Some("contradiction".into()),
                    reason: Some("Contradictory hypothesis `h_contra : False`".into()),
                })
            } else {
                Ok(VacuityCheck {
                    is_vacuous: false,
                    witness_proof: None,
                    reason: None,
                })
            }
        }
        async fn check_declaration(
            &self,
            _context: &ProofContext,
            _code: &str,
        ) -> anyhow::Result<Vec<DiagnosticMsg>> {
            Ok(vec![])
        }
        async fn apply_tactic(
            &self,
            state: &ProofStateSnapshot,
            tactic: &str,
        ) -> anyhow::Result<TacticStepResult> {
            if tactic == "rfl" {
                Ok(TacticStepResult {
                    accepted: true,
                    next_state: Some(ProofStateSnapshot {
                        target: state.target.clone(),
                        tactic_prefix: vec!["rfl".into()],
                        open_goals: vec![],
                    }),
                    remaining_goals: vec![],
                    diagnostics: vec![],
                    suggested_tactic: None,
                })
            } else {
                Ok(TacticStepResult {
                    accepted: false,
                    next_state: None,
                    remaining_goals: state.open_goals.clone(),
                    diagnostics: vec![],
                    suggested_tactic: None,
                })
            }
        }
        async fn search_premises(
            &self,
            _context: &ProofContext,
            _query: &str,
            _limit: usize,
        ) -> anyhow::Result<Vec<PremiseHit>> {
            Ok(vec![])
        }
        async fn cold_audit(
            &self,
            _file_path: &Path,
            decl_names: &[String],
            _policy: &AxiomPolicy,
        ) -> anyhow::Result<AuditReport> {
            Ok(AuditReport {
                verified: true,
                declarations_checked: decl_names.to_vec(),
                axioms_used: vec![],
                forbidden_axioms: vec![],
                has_sorry: false,
                forbidden_constructs: vec![],
                raw_output: String::new(),
            })
        }
    }

    struct VacuityRepairLlm {
        attempts: AtomicUsize,
    }

    #[async_trait]
    impl LlmProvider for VacuityRepairLlm {
        fn provider_name(&self) -> &str {
            "mock"
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>> {
            let prompt = &req.messages[0].content;
            if prompt.contains("Formalize the following mathematical claim") {
                let n = self.attempts.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    // Attempt 1: vacuously true theorem!
                    Ok(vec![CandidateOutput {
                        text: r#"{
                            "decl_name": "add_zero_nat",
                            "imports": [],
                            "definitions": "",
                            "theorem_signature": "theorem add_zero_nat (n : Nat) (h_contra : False) : n + 0 = n"
                        }"#
                        .into(),
                        log_prob: None,
                    }])
                } else {
                    assert!(prompt.contains("Contradictory hypothesis"));
                    Ok(vec![CandidateOutput {
                        text: r#"{
                            "decl_name": "add_zero_nat",
                            "imports": [],
                            "definitions": "",
                            "theorem_signature": "theorem add_zero_nat (n : Nat) : n + 0 = n"
                        }"#
                        .into(),
                        log_prob: None,
                    }])
                }
            } else {
                // Back-translation check
                Ok(vec![CandidateOutput {
                    text: r#"{
                        "back_translation": "For every natural number n, n + 0 = n.",
                        "faithful": true,
                        "critique": "Exact match."
                    }"#
                    .into(),
                    log_prob: None,
                }])
            }
        }
    }

    #[tokio::test]
    async fn autoformalizer_rejects_vacuous_hypothesis_and_proves_repaired_claim() {
        let verifier = Arc::new(VacuityMockVerifier);
        let llm = Arc::new(VacuityRepairLlm {
            attempts: AtomicUsize::new(0),
        });
        let router = ModelRouter::from_providers(llm.clone(), llm);
        let formalizer = Autoformalizer::new(
            verifier,
            router,
            SearchConfig::default(),
            LeanConfig::default(),
            noop_event_sink(),
        );

        let res = formalizer
            .formalize_input(
                "For every natural number n, n + 0 = n.",
                None,
                &AutoformalizeOptions::default(),
            )
            .await
            .expect("autoformalization succeeds");

        assert_eq!(res.items.len(), 1);
        assert!(res.items[0].proved);
        assert!(res.items[0].back_translation_faithful);
        assert!(!res.lean_source.contains("h_contra"));
        assert!(res.lean_source.contains("theorem add_zero_nat (n : Nat) : n + 0 = n := by\n  rfl"));
    }
}
