use crate::hammers::try_hammer_cascade;
use crate::mcgs::McgsEngine;
use anyhow::Context;
use juice_core::{
    AuditReport, AxiomPolicy, CheckpointStore, CounterexampleResult, EventSink, ProofTarget,
    RunCheckpoint, SearchConfig, SearchEvent, VerifierBackend,
};
use juice_lean::parser::splice_verified_proofs;
use juice_llm::ModelRouter;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSolveReport {
    pub target_id: String,
    pub decl_name: String,
    pub solved: bool,
    pub proof_script: Option<String>,
    pub nodes_expanded: usize,
    pub elapsed_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillBatchOutcome {
    pub run_id: String,
    pub checkpoint_path: PathBuf,
    pub reports: Vec<TargetSolveReport>,
    /// File path -> original Lean 4 source before patching
    pub original_files: BTreeMap<PathBuf, String>,
    /// File path -> patched Lean 4 source code containing verified proofs
    pub patched_files: BTreeMap<PathBuf, String>,
    pub audit_reports: BTreeMap<PathBuf, AuditReport>,
}

/// Aristotle-style two-tier reasoning engine combining:
/// - Tier 0: Zero-cost deterministic Lean 4 hammer cascade
/// - Tier 1: Informal Planner + `have` sub-lemma Draft-Sketch-Prove decomposition + `plausible` counterexample probe
/// - Tier 2: Parallel Monte Carlo Graph Search (MCGS) over deduplicated goal states
pub struct TwoTierProver {
    verifier: Arc<dyn VerifierBackend>,
    router: ModelRouter,
    cfg: SearchConfig,
    events: EventSink,
}

impl TwoTierProver {
    pub fn new(
        verifier: Arc<dyn VerifierBackend>,
        router: ModelRouter,
        cfg: SearchConfig,
        events: EventSink,
    ) -> Self {
        Self {
            verifier,
            router,
            cfg,
            events,
        }
    }

    pub async fn prove_target(
        &self,
        target: &ProofTarget,
        checkpoint: &mut RunCheckpoint,
    ) -> anyhow::Result<TargetSolveReport> {
        let start = Instant::now();

        if let Some(cached_proof) = checkpoint.solved_targets.get(&target.id).cloned() {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            (self.events)(SearchEvent::TargetCompleted {
                target_id: target.id.clone(),
                decl_name: target.decl_name.clone(),
                success: true,
                proof: Some(cached_proof.clone()),
                elapsed_ms,
            });
            return Ok(TargetSolveReport {
                target_id: target.id.clone(),
                decl_name: target.decl_name.clone(),
                solved: true,
                proof_script: Some(cached_proof),
                nodes_expanded: 0,
                elapsed_ms,
                error: None,
            });
        }

        let init_snapshot = self.verifier.initial_state(target).await?;
        (self.events)(SearchEvent::TargetStarted {
            target_id: target.id.clone(),
            decl_name: target.decl_name.clone(),
            total_goals: init_snapshot.open_goals.len(),
        });

        // Tier 0: Instant deterministic hammer check (works 100% offline with zero LLM tokens!).
        if self.cfg.enable_hammers
            && let Some(win_tactic) = try_hammer_cascade(&*self.verifier, &init_snapshot).await
        {
            (self.events)(SearchEvent::HammerHit {
                target_id: target.id.clone(),
                tactic: win_tactic.clone(),
            });
            checkpoint
                .solved_targets
                .insert(target.id.clone(), win_tactic.clone());
            checkpoint.nodes_expanded += 1;
            let elapsed_ms = start.elapsed().as_millis() as u64;
            (self.events)(SearchEvent::TargetCompleted {
                target_id: target.id.clone(),
                decl_name: target.decl_name.clone(),
                success: true,
                proof: Some(win_tactic.clone()),
                elapsed_ms,
            });
            return Ok(TargetSolveReport {
                target_id: target.id.clone(),
                decl_name: target.decl_name.clone(),
                solved: true,
                proof_script: Some(win_tactic),
                nodes_expanded: 1,
                elapsed_ms,
                error: None,
            });
        }

        // Retrieve verified workspace/Mathlib premises to prevent hallucinated theorem names.
        let premises = self
            .verifier
            .search_premises(&target.context, &target.decl_signature, 15)
            .await
            .unwrap_or_default();

        let mcgs = McgsEngine::new(
            self.verifier.clone(),
            self.router.clone(),
            self.cfg.clone(),
            self.events.clone(),
        );

        let mut total_expanded = 0usize;

        // Quick shallow direct MCGS pass before paying Tier-1 multi-lemma decomposition overhead.
        let direct_budget = (self.cfg.max_nodes / 3).max(4);
        let direct_outcome = mcgs
            .search(init_snapshot.clone(), &premises, direct_budget)
            .await?;
        total_expanded += direct_outcome.nodes_expanded;
        checkpoint.nodes_expanded += direct_outcome.nodes_expanded;

        if let Some(proof) = direct_outcome.proof_script {
            checkpoint
                .solved_targets
                .insert(target.id.clone(), proof.clone());
            let elapsed_ms = start.elapsed().as_millis() as u64;
            (self.events)(SearchEvent::TargetCompleted {
                target_id: target.id.clone(),
                decl_name: target.decl_name.clone(),
                success: true,
                proof: Some(proof.clone()),
                elapsed_ms,
            });
            return Ok(TargetSolveReport {
                target_id: target.id.clone(),
                decl_name: target.decl_name.clone(),
                solved: true,
                proof_script: Some(proof),
                nodes_expanded: total_expanded,
                elapsed_ms,
                error: None,
            });
        }

        // Tier 1: Draft-Sketch-Prove loop with `plausible` counterexample probing.
        let mut prior_errors: Vec<String> = Vec::new();
        for round in 1..=self.cfg.max_planner_rounds {
            (self.events)(SearchEvent::PlannerSketching {
                target_id: target.id.clone(),
                round,
            });

            let refuted_vec: Vec<(String, String)> = checkpoint
                .refuted_lemmas
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            let sketch_plan = match self
                .router
                .plan_lemma_sketch(
                    &target.decl_signature,
                    &init_snapshot.open_goals,
                    &premises,
                    &refuted_vec,
                    &prior_errors,
                )
                .await
            {
                Ok(p) => p,
                Err(err) => {
                    prior_errors.push(format!("Planner formatting error: {err}"));
                    continue;
                }
            };

            // Step 1: Verify the outer sketch compiles and closes the main goal assuming its `have` sub-lemmas!
            let validation = self
                .verifier
                .validate_sketch(target, &sketch_plan.lean_sketch_tactics)
                .await?;
            if !validation.valid {
                prior_errors = validation.errors;
                continue;
            }

            // Step 2: Probe every sketched `have` sub-lemma for quick counterexamples (`plausible` / `decide`).
            let mut counterexample_hit = false;
            if self.cfg.enable_counterexample_probe {
                for sub_lemma in &sketch_plan.sub_lemmas {
                    if checkpoint.refuted_lemmas.contains_key(&sub_lemma.statement) {
                        counterexample_hit = true;
                        break;
                    }
                    if let Ok(CounterexampleResult::Found { witness }) = self
                        .verifier
                        .probe_counterexample(&target.context, &sub_lemma.statement)
                        .await
                    {
                        (self.events)(SearchEvent::CounterexampleFound {
                            target_id: target.id.clone(),
                            lemma: sub_lemma.statement.clone(),
                            witness: witness.clone(),
                        });
                        checkpoint
                            .refuted_lemmas
                            .insert(sub_lemma.statement.clone(), witness.clone());
                        prior_errors.push(format!(
                            "Sub-lemma `{}` (`{}`) is FALSE: {}",
                            sub_lemma.name, sub_lemma.statement, witness
                        ));
                        counterexample_hit = true;
                        break;
                    }
                }
            }
            if counterexample_hit {
                continue;
            }

            (self.events)(SearchEvent::SketchAccepted {
                target_id: target.id.clone(),
                sub_lemmas_count: validation.sub_targets.len(),
            });

            // If the sketch closed the theorem with ZERO remaining `sorry` holes, we're done immediately!
            if validation.sub_targets.is_empty() {
                let clean_sketch = sketch_plan.lean_sketch_tactics.trim().to_string();
                checkpoint
                    .solved_targets
                    .insert(target.id.clone(), clean_sketch.clone());
                let elapsed_ms = start.elapsed().as_millis() as u64;
                (self.events)(SearchEvent::TargetCompleted {
                    target_id: target.id.clone(),
                    decl_name: target.decl_name.clone(),
                    success: true,
                    proof: Some(clean_sketch.clone()),
                    elapsed_ms,
                });
                return Ok(TargetSolveReport {
                    target_id: target.id.clone(),
                    decl_name: target.decl_name.clone(),
                    solved: true,
                    proof_script: Some(clean_sketch),
                    nodes_expanded: total_expanded,
                    elapsed_ms,
                    error: None,
                });
            }

            // Step 3: Prove each inner sub-target in the validated sketch via Tier-2 MCGS!
            let per_sub_budget =
                (self.cfg.max_nodes / validation.sub_targets.len().max(1)).max(8);
            let mut sub_patches: Vec<(&ProofTarget, String)> = Vec::new();
            let mut all_subs_solved = true;

            for sub_target in &validation.sub_targets {
                if let Some(cached_sub) =
                    checkpoint.solved_sublemmas.get(&sub_target.decl_signature).cloned()
                {
                    sub_patches.push((sub_target, cached_sub));
                    continue;
                }

                let sub_init = self.verifier.initial_state(sub_target).await?;
                let sub_premises = self
                    .verifier
                    .search_premises(&sub_target.context, &sub_target.decl_signature, 10)
                    .await
                    .unwrap_or_else(|_| premises.clone());

                let sub_res = mcgs
                    .search(sub_init, &sub_premises, per_sub_budget)
                    .await?;
                total_expanded += sub_res.nodes_expanded;
                checkpoint.nodes_expanded += sub_res.nodes_expanded;

                if let Some(sub_proof) = sub_res.proof_script {
                    (self.events)(SearchEvent::SubLemmaSolved {
                        target_id: target.id.clone(),
                        lemma_name: sub_target.decl_name.clone(),
                        proof: sub_proof.clone(),
                    });
                    checkpoint
                        .solved_sublemmas
                        .insert(sub_target.decl_signature.clone(), sub_proof.clone());
                    sub_patches.push((sub_target, sub_proof));
                } else {
                    all_subs_solved = false;
                    let stuck_summary = sub_res
                        .stuck_goals
                        .iter()
                        .map(|g| g.canonical_pp.as_str())
                        .collect::<Vec<_>>()
                        .join("; ");
                    prior_errors.push(format!(
                        "Could not close sub-goal within search budget. Stuck goals: {stuck_summary}"
                    ));
                    break;
                }
            }

            if all_subs_solved {
                // Splice all proven sub-lemmas into `validation.compiled_sketch`,
                // then extract the exact substituted region corresponding to the outer `target.sorry_span`!
                let fully_spliced =
                    splice_verified_proofs(&validation.compiled_sketch, &sub_patches)?;
                let suffix_len = target.full_source.len() - target.sorry_span.end_byte;
                let inner_end = fully_spliced.len().saturating_sub(suffix_len);
                let mut extracted =
                    fully_spliced[target.sorry_span.start_byte..inner_end].to_string();
                if target.is_term_mode_sorry
                    && let Some(stripped) = extracted.strip_prefix("by\n")
                {
                    extracted = stripped.to_string();
                }
                let final_proof = dedent_block(&extracted);

                checkpoint
                    .solved_targets
                    .insert(target.id.clone(), final_proof.clone());
                let elapsed_ms = start.elapsed().as_millis() as u64;
                (self.events)(SearchEvent::TargetCompleted {
                    target_id: target.id.clone(),
                    decl_name: target.decl_name.clone(),
                    success: true,
                    proof: Some(final_proof.clone()),
                    elapsed_ms,
                });
                return Ok(TargetSolveReport {
                    target_id: target.id.clone(),
                    decl_name: target.decl_name.clone(),
                    solved: true,
                    proof_script: Some(final_proof),
                    nodes_expanded: total_expanded,
                    elapsed_ms,
                    error: None,
                });
            }
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;
        (self.events)(SearchEvent::TargetCompleted {
            target_id: target.id.clone(),
            decl_name: target.decl_name.clone(),
            success: false,
            proof: None,
            elapsed_ms,
        });

        Ok(TargetSolveReport {
            target_id: target.id.clone(),
            decl_name: target.decl_name.clone(),
            solved: false,
            proof_script: None,
            nodes_expanded: total_expanded,
            elapsed_ms,
            error: Some(if prior_errors.is_empty() {
                "Exhausted search budget without finding a kernel-valid proof".to_string()
            } else {
                prior_errors.join(" | ")
            }),
        })
    }

    /// Discovers and proves `sorry` targets across a file or Lake directory, checkpoints progress,
    /// splices verified proofs, and runs cold `#print axioms` certification on completely filled files.
    pub async fn fill_path(
        &self,
        path: &Path,
        target_filter: Option<&str>,
        resume_run_id: Option<&str>,
        policy: &AxiomPolicy,
        write_in_place: bool,
    ) -> anyhow::Result<FillBatchOutcome> {
        let mut targets = self.verifier.discover_targets(path).await?;
        if let Some(filter) = target_filter {
            targets.retain(|t| t.decl_name.contains(filter) || t.id.contains(filter));
        }

        let ws_root = targets
            .first()
            .map(|t| t.context.workspace_root.clone())
            .unwrap_or_else(|| {
                if path.is_dir() {
                    path.to_path_buf()
                } else {
                    path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
                }
            });

        let seed = targets
            .iter()
            .map(|t| format!("{}:{}", t.id, t.decl_signature))
            .collect::<Vec<_>>()
            .join("\n");
        let store = CheckpointStore::new(&ws_root);
        let mut checkpoint = store.load_or_create(&seed, resume_run_id)?;
        let mut checkpoint_path = store.save(&mut checkpoint)?;

        let mut reports = Vec::with_capacity(targets.len());
        let mut by_file: BTreeMap<PathBuf, Vec<(ProofTarget, String)>> = BTreeMap::new();
        let mut total_targets_per_file: BTreeMap<PathBuf, usize> = BTreeMap::new();

        for target in &targets {
            if let Some(sf) = &target.context.source_file {
                *total_targets_per_file.entry(sf.clone()).or_insert(0) += 1;
            }
            let report = self.prove_target(target, &mut checkpoint).await?;
            checkpoint_path = store.save(&mut checkpoint)?;

            if let (true, Some(proof), Some(sf)) = (
                report.solved,
                report.proof_script.clone(),
                target.context.source_file.clone(),
            ) {
                by_file
                    .entry(sf)
                    .or_default()
                    .push((target.clone(), proof));
            }
            reports.push(report);
        }

        let mut original_files = BTreeMap::new();
        let mut patched_files = BTreeMap::new();
        let mut audit_reports = BTreeMap::new();

        for (file_path, solved_pairs) in &by_file {
            let original = std::fs::read_to_string(file_path)
                .with_context(|| format!("Failed to read {}", file_path.display()))?;
            let patch_refs: Vec<(&ProofTarget, String)> = solved_pairs
                .iter()
                .map(|(t, p)| (t, p.clone()))
                .collect();
            let patched = splice_verified_proofs(&original, &patch_refs)?;
            original_files.insert(file_path.clone(), original);

            let all_file_sorries_closed =
                total_targets_per_file.get(file_path).copied().unwrap_or(0) == solved_pairs.len();

            if all_file_sorries_closed {
                // Run cold `#print axioms` audit on a temp file in the same directory BEFORE overwriting user source!
                let parent = file_path.parent().unwrap_or_else(|| Path::new("."));
                let tmp_candidate = tempfile::Builder::new()
                    .prefix(".juice_verify_")
                    .suffix(".lean")
                    .tempfile_in(parent)?;
                std::fs::write(tmp_candidate.path(), &patched)?;

                let decl_names: Vec<String> = solved_pairs
                    .iter()
                    .map(|(t, _)| t.decl_name.clone())
                    .collect();
                let audit = self
                    .verifier
                    .cold_audit(tmp_candidate.path(), &decl_names, policy)
                    .await?;

                (self.events)(SearchEvent::ColdAuditCompleted {
                    decl_name: decl_names.join(", "),
                    verified: audit.verified,
                    axioms: audit.axioms_used.clone(),
                });

                if !audit.verified {
                    anyhow::bail!(
                        "Cold kernel axiom audit rejected candidate proof for {}: forbidden axioms={:?}, has_sorry={}",
                        file_path.display(),
                        audit.forbidden_axioms,
                        audit.has_sorry
                    );
                }
                audit_reports.insert(file_path.clone(), audit);
            }

            if write_in_place {
                std::fs::write(file_path, &patched)?;
            }
            patched_files.insert(file_path.clone(), patched);
        }

        Ok(FillBatchOutcome {
            run_id: checkpoint.run_id,
            checkpoint_path,
            reports,
            original_files,
            patched_files,
            audit_reports,
        })
    }
}

fn dedent_block(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= 1 {
        return s.trim().to_string();
    }
    let min_indent = lines[1..]
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| *c == ' ').count())
        .min()
        .unwrap_or(0);

    let mut out = Vec::with_capacity(lines.len());
    out.push(lines[0].trim().to_string());
    for l in &lines[1..] {
        if l.len() >= min_indent {
            out.push(l[min_indent..].trim_end().to_string());
        } else {
            out.push(l.trim().to_string());
        }
    }
    out.join("\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use juice_core::{
        DiagnosticMsg, NormalizedGoal, PremiseHit, ProofContext, ProofStateSnapshot,
        SketchValidation, Span, TacticStepResult, VacuityCheck, events::noop_event_sink,
    };
    use juice_lean::parser::extract_proof_targets;
    use juice_llm::{CandidateOutput, CompletionRequest, LlmProvider};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Tests that when Tier 1 proposes a FALSE intermediate lemma (`have h1 : ∀ n : Nat, n > 0`),
    /// `probe_counterexample` catches `n = 0`, feeds it back to Tier 1, and Tier 1 revises the sketch
    /// to a valid lemma that Tier 2 closes!
    struct CounterexampleLoopVerifier;

    #[async_trait]
    impl VerifierBackend for CounterexampleLoopVerifier {
        fn backend_name(&self) -> &'static str {
            "mock-cx"
        }
        async fn discover_targets(&self, _path: &Path) -> anyhow::Result<Vec<ProofTarget>> {
            Ok(vec![])
        }
        async fn initial_state(&self, target: &ProofTarget) -> anyhow::Result<ProofStateSnapshot> {
            let goal = if target.decl_signature.contains("h_good") || target.full_source.contains("h_good") {
                "⊢ GoodLemma"
            } else {
                "⊢ HardTheorem"
            };
            Ok(ProofStateSnapshot {
                target: target.clone(),
                tactic_prefix: vec![],
                open_goals: vec![NormalizedGoal::new(goal)],
            })
        }
        async fn validate_sketch(
            &self,
            target: &ProofTarget,
            sketch_body: &str,
        ) -> anyhow::Result<SketchValidation> {
            let lines: Vec<String> = sketch_body.lines().map(ToString::to_string).collect();
            let (compiled, _, _) =
                juice_lean::parser::render_target_with_tactics(target, &lines, false);
            let subs = extract_proof_targets(&compiled, Path::new("/tmp"), None);
            Ok(SketchValidation {
                valid: true,
                compiled_sketch: compiled,
                sub_targets: subs,
                errors: vec![],
            })
        }
        async fn probe_counterexample(
            &self,
            _context: &ProofContext,
            lemma_stmt: &str,
        ) -> anyhow::Result<CounterexampleResult> {
            if lemma_stmt.contains("n > 0") {
                Ok(CounterexampleResult::Found {
                    witness: "Found a counter-example! n := 0".to_string(),
                })
            } else {
                Ok(CounterexampleResult::NoCounterexampleFound)
            }
        }
        async fn check_vacuity(
            &self,
            _context: &ProofContext,
            _theorem_decl: &str,
        ) -> anyhow::Result<VacuityCheck> {
            Ok(VacuityCheck {
                is_vacuous: false,
                witness_proof: None,
                reason: None,
            })
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
            let is_good_sub = state
                .open_goals
                .first()
                .map(|g| g.canonical_pp.contains("GoodLemma"))
                .unwrap_or(false);
            if is_good_sub && tactic == "omega" {
                Ok(TacticStepResult {
                    accepted: true,
                    next_state: Some(ProofStateSnapshot {
                        target: state.target.clone(),
                        tactic_prefix: vec!["omega".into()],
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
                axioms_used: vec!["propext".into()],
                forbidden_axioms: vec![],
                has_sorry: false,
                forbidden_constructs: vec![],
                raw_output: String::new(),
            })
        }
    }

    struct AdaptivePlannerLlm {
        planner_calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmProvider for AdaptivePlannerLlm {
        fn provider_name(&self) -> &str {
            "mock"
        }
        fn model_name(&self) -> &str {
            "mock-planner"
        }
        async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>> {
            let prompt = &req.messages[0].content;
            if prompt.contains("Decompose this Lean 4 theorem") {
                let n = self.planner_calls.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    // Round 1: propose a FALSE lemma `∀ n : Nat, n > 0`!
                    Ok(vec![CandidateOutput {
                        text: r#"{
                            "informal_plan": "Try naive positivity",
                            "sub_lemmas": [{"name": "h_bad", "statement": "∀ n : Nat, n > 0", "informal_reason": "guess"}],
                            "lean_sketch_tactics": "have h_bad : ∀ n : Nat, n > 0 := by\n  sorry\nexact h_bad"
                        }"#
                        .to_string(),
                        log_prob: None,
                    }])
                } else {
                    // Verify the counterexample was fed back into the prompt!
                    assert!(prompt.contains("counterexample"));
                    Ok(vec![CandidateOutput {
                        text: r#"{
                            "informal_plan": "Use corrected non-negativity",
                            "sub_lemmas": [{"name": "h_good", "statement": "∀ n : Nat, n >= 0", "informal_reason": "Nat bound"}],
                            "lean_sketch_tactics": "have h_good : ∀ n : Nat, n >= 0 := by\n  sorry\nexact h_good"
                        }"#
                        .to_string(),
                        log_prob: None,
                    }])
                }
            } else {
                // Direct tactic proposals fail on HardTheorem so Tier-1 planner is exercised.
                Ok(vec![CandidateOutput {
                    text: r#"{"tactics": ["noop"]}"#.to_string(),
                    log_prob: None,
                }])
            }
        }
    }

    #[tokio::test]
    async fn planner_rejects_false_sublemma_via_counterexample_and_replans() {
        let verifier = Arc::new(CounterexampleLoopVerifier);
        let llm = Arc::new(AdaptivePlannerLlm {
            planner_calls: AtomicUsize::new(0),
        });
        let router = ModelRouter::from_providers(llm.clone(), llm.clone());
        let cfg = SearchConfig {
            max_nodes: 6,
            concurrency: 1,
            ..Default::default()
        };

        let prover = TwoTierProver::new(verifier, router, cfg, noop_event_sink());
        let src = "theorem hard_thm : HardTheorem := by\n  sorry\n";
        let target = ProofTarget {
            id: "Test.lean::hard_thm#0".into(),
            decl_name: "hard_thm".into(),
            decl_kind: "theorem".into(),
            decl_signature: "theorem hard_thm : HardTheorem".into(),
            full_source: src.into(),
            sorry_span: Span {
                start_byte: 39,
                end_byte: 44,
                start_line: 1,
                start_col_utf16: 2,
                end_line: 1,
                end_col_utf16: 7,
            },
            indent_col: 2,
            is_term_mode_sorry: false,
            context: ProofContext {
                workspace_root: PathBuf::from("/tmp"),
                source_file: None,
                imports_header: String::new(),
                preamble: String::new(),
            },
        };

        let mut cp = RunCheckpoint::new("run1", "hash1");
        let report = prover.prove_target(&target, &mut cp).await.expect("proves");
        assert!(report.solved);
        assert!(cp.refuted_lemmas.contains_key("∀ n : Nat, n > 0"));
        let final_proof = report.proof_script.unwrap();
        assert!(final_proof.contains("have h_good"));
        assert!(final_proof.contains("omega"));
    }
}
