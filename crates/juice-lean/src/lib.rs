pub mod audit;
pub mod lsp_pool;
pub mod parser;
pub mod premises;
pub mod workspace;

use crate::audit::run_cold_kernel_audit;
use crate::lsp_pool::LspWorkerPool;
use crate::parser::{
    extract_proof_targets, extract_try_this_suggestion, render_target_with_tactics,
    scan_forbidden_constructs,
};
use crate::premises::{PremiseIndex, query_loogle_fallback};
use crate::workspace::{
    DoctorReport, detect_toolchain, find_enclosing_lake_project, project_has_mathlib,
    resolve_workspace,
};
use anyhow::{Context, bail};
use async_trait::async_trait;
use juice_core::{
    AuditReport, AxiomPolicy, CounterexampleResult, DiagnosticMsg, DiagnosticSeverity, JuiceConfig,
    LeanConfig, PremiseHit, ProofContext, ProofStateSnapshot, ProofTarget, SearchConfig,
    SketchValidation, TacticStepResult, VacuityCheck, VerifierBackend,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use walkdir::WalkDir;

pub struct LeanBackend {
    lean_cfg: LeanConfig,
    search_cfg: SearchConfig,
    pools: Mutex<HashMap<PathBuf, Arc<LspWorkerPool>>>,
    indices: Mutex<HashMap<PathBuf, Arc<PremiseIndex>>>,
}

impl LeanBackend {
    pub fn new(lean_cfg: LeanConfig, search_cfg: SearchConfig) -> Self {
        Self {
            lean_cfg,
            search_cfg,
            pools: Mutex::new(HashMap::new()),
            indices: Mutex::new(HashMap::new()),
        }
    }

    async fn pool_for(&self, workspace_root: &Path) -> Arc<LspWorkerPool> {
        let mut guard = self.pools.lock().await;
        if let Some(existing) = guard.get(workspace_root) {
            return existing.clone();
        }
        let pool = Arc::new(LspWorkerPool::new(
            workspace_root.to_path_buf(),
            self.lean_cfg.worker_pool_size,
        ));
        guard.insert(workspace_root.to_path_buf(), pool.clone());
        pool
    }

    async fn index_for(&self, workspace_root: &Path) -> Arc<PremiseIndex> {
        let mut guard = self.indices.lock().await;
        if let Some(existing) = guard.get(workspace_root) {
            return existing.clone();
        }
        let root_owned = workspace_root.to_path_buf();
        let built = tokio::task::spawn_blocking(move || PremiseIndex::load_or_build(&root_owned))
            .await
            .unwrap_or_default();
        let arc = Arc::new(built);
        guard.insert(workspace_root.to_path_buf(), arc.clone());
        arc
    }

    fn step_timeout(&self) -> Duration {
        Duration::from_millis(self.search_cfg.step_timeout_ms.max(500))
    }

    /// Runs `juice doctor` health check across toolchain, Mathlib cache, and LLM config.
    pub async fn doctor(cfg: &JuiceConfig, target_dir: Option<&Path>) -> DoctorReport {
        let tc = detect_toolchain().await;
        let cwd = target_dir
            .map(Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        let detected_lake_project = find_enclosing_lake_project(&cwd);
        let mathlib_cache_dir = cfg.lean.resolved_mathlib_cache_dir();
        let mathlib_cache_ready = mathlib_cache_dir.join("lakefile.toml").is_file()
            && project_has_mathlib(&mathlib_cache_dir);

        let planner_model = format!(
            "{:?}:{}",
            cfg.models.planner.provider, cfg.models.planner.model
        );
        let prover_ep = cfg.models.effective_prover();
        let prover_model = format!("{:?}:{}", prover_ep.provider, prover_ep.model);
        let planner_key_present = cfg.models.planner.resolve_api_key().is_some()
            || cfg.models.planner.provider == juice_core::ProviderKind::OpenaiCompat;
        let ready_for_local_lean = tc.lean.is_some();

        DoctorReport {
            toolchain: tc,
            detected_lake_project,
            mathlib_cache_dir,
            mathlib_cache_ready,
            planner_model,
            prover_model,
            planner_key_present,
            ready_for_local_lean,
        }
    }
}

#[async_trait]
impl VerifierBackend for LeanBackend {
    fn backend_name(&self) -> &'static str {
        "lean4"
    }

    async fn discover_targets(&self, path: &Path) -> anyhow::Result<Vec<ProofTarget>> {
        if !path.exists() {
            bail!("Target path does not exist: {}", path.display());
        }

        let resolved_ws = resolve_workspace(Some(path), &self.lean_cfg);
        let mut files = Vec::new();
        if path.is_file() {
            files.push(path.to_path_buf());
        } else {
            for entry in WalkDir::new(path)
                .follow_links(false)
                .into_iter()
                .filter_entry(|e| {
                    let name = e.file_name().to_string_lossy();
                    name != ".lake" && name != ".git" && name != ".juice" && name != "build"
                })
                .flatten()
            {
                if entry.file_type().is_file()
                    && entry.path().extension().and_then(|s| s.to_str()) == Some("lean")
                {
                    files.push(entry.path().to_path_buf());
                }
            }
            files.sort();
        }

        let mut all_targets = Vec::new();
        for f in files {
            let source = std::fs::read_to_string(&f)
                .with_context(|| format!("Failed to read {}", f.display()))?;
            let targets = extract_proof_targets(&source, &resolved_ws.root, Some(&f));
            all_targets.extend(targets);
        }
        Ok(all_targets)
    }

    async fn initial_state(&self, target: &ProofTarget) -> anyhow::Result<ProofStateSnapshot> {
        let pool = self.pool_for(&target.context.workspace_root).await;
        let eval = pool
            .evaluate(
                "init.lean",
                &target.full_source,
                target.sorry_span.start_line,
                target.sorry_span.start_col_utf16,
                self.step_timeout() * 3, // Allow extra warmup time on initial import load
            )
            .await?;

        let fatal_errors: Vec<_> = eval
            .diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();
        if !fatal_errors.is_empty() {
            bail!(
                "Target file contains Lean type errors before proving `{}`: {}",
                target.decl_name,
                fatal_errors[0].message
            );
        }

        Ok(ProofStateSnapshot {
            target: target.clone(),
            tactic_prefix: Vec::new(),
            open_goals: eval.goals,
        })
    }

    async fn validate_sketch(
        &self,
        target: &ProofTarget,
        sketch_body: &str,
    ) -> anyhow::Result<SketchValidation> {
        // Check for illegal metaprogramming/axioms while permitting inner `sorry` sub-goals.
        let non_sorry_violations: Vec<String> = scan_forbidden_constructs(sketch_body)
            .into_iter()
            .filter(|v| !v.contains("`sorry`"))
            .collect();
        if !non_sorry_violations.is_empty() {
            return Ok(SketchValidation {
                valid: false,
                compiled_sketch: String::new(),
                sub_targets: Vec::new(),
                errors: non_sorry_violations,
            });
        }

        let sketch_lines: Vec<String> = sketch_body.lines().map(ToString::to_string).collect();
        let (compiled_sketch, end_line, end_col) =
            render_target_with_tactics(target, &sketch_lines, false);

        let pool = self.pool_for(&target.context.workspace_root).await;
        let eval = pool
            .evaluate(
                "sketch.lean",
                &compiled_sketch,
                end_line,
                end_col,
                self.step_timeout() * 2,
            )
            .await?;

        let errors: Vec<String> = eval
            .diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .map(|d| format!("L{}:{}: {}", d.line + 1, d.column + 1, d.message))
            .collect();

        if !errors.is_empty() {
            return Ok(SketchValidation {
                valid: false,
                compiled_sketch,
                sub_targets: Vec::new(),
                errors,
            });
        }

        // Extract only the newly introduced inner `sorry` targets inside this sketch span.
        let sketch_end_byte = compiled_sketch
            .len()
            .saturating_sub(target.full_source.len() - target.sorry_span.end_byte);
        let sub_targets: Vec<ProofTarget> = extract_proof_targets(
            &compiled_sketch,
            &target.context.workspace_root,
            target.context.source_file.as_deref(),
        )
        .into_iter()
        .filter(|t| {
            t.sorry_span.start_byte >= target.sorry_span.start_byte
                && t.sorry_span.end_byte <= sketch_end_byte
        })
        .collect();

        Ok(SketchValidation {
            valid: true,
            compiled_sketch,
            sub_targets,
            errors: Vec::new(),
        })
    }

    async fn probe_counterexample(
        &self,
        context: &ProofContext,
        lemma_stmt: &str,
    ) -> anyhow::Result<CounterexampleResult> {
        let has_mathlib = context.imports_header.contains("Mathlib")
            || project_has_mathlib(&context.workspace_root);

        let tactic = if has_mathlib {
            "plausible"
        } else {
            "decide"
        };

        let imports = if has_mathlib && !context.imports_header.contains("Plausible") {
            format!("{}\nimport Plausible\n", context.imports_header.trim())
        } else {
            context.imports_header.clone()
        };

        let probe_doc = format!(
            "{}\n{}\n\nexample : {} := by\n  {}\n",
            imports.trim(),
            context.preamble.trim(),
            lemma_stmt.trim(),
            tactic
        );

        let pool = self.pool_for(&context.workspace_root).await;
        let Ok(eval) = pool
            .evaluate("counterexample_probe.lean", &probe_doc, 0, 0, self.step_timeout())
            .await
        else {
            return Ok(CounterexampleResult::Inconclusive(
                "Counterexample probe timed out".to_string(),
            ));
        };

        for d in &eval.diagnostics {
            let msg_lower = d.message.to_lowercase();
            if msg_lower.contains("found a counter-example")
                || msg_lower.contains("found a counterexample")
            {
                return Ok(CounterexampleResult::Found {
                    witness: d.message.clone(),
                });
            }
            if tactic == "decide" && msg_lower.contains("proved that the proposition is false") {
                return Ok(CounterexampleResult::Found {
                    witness: d.message.clone(),
                });
            }
        }

        Ok(CounterexampleResult::NoCounterexampleFound)
    }

    async fn check_vacuity(
        &self,
        context: &ProofContext,
        theorem_decl: &str,
    ) -> anyhow::Result<VacuityCheck> {
        // Replace final `: <Conclusion>` with `: False` when hypotheses exist to check if hypotheses
        // immediately contradict one another (`hypotheses ⊢ False`).
        let Some(colon_idx) = find_top_level_conclusion_colon(theorem_decl) else {
            return Ok(VacuityCheck {
                is_vacuous: false,
                witness_proof: None,
                reason: None,
            });
        };

        let header_before_colon = theorem_decl[..colon_idx].trim();
        // Only check vacuity when explicit binder hypotheses `(h : ...)` exist!
        if !header_before_colon.contains('(') && !header_before_colon.contains('[') {
            return Ok(VacuityCheck {
                is_vacuous: false,
                witness_proof: None,
                reason: None,
            });
        }

        let pool = self.pool_for(&context.workspace_root).await;
        for probe_tactic in ["contradiction", "omega", "simp_all", "aesop"] {
            let doc = format!(
                "{}\n{}\n\n{} : False := by\n  {}\n",
                context.imports_header.trim(),
                context.preamble.trim(),
                header_before_colon,
                probe_tactic
            );
            if let Ok(eval) = pool
                .evaluate("vacuity.lean", &doc, 0, 0, self.step_timeout())
                .await
            {
                let has_err = eval
                    .diagnostics
                    .iter()
                    .any(|d| d.severity == DiagnosticSeverity::Error);
                let has_sorry_warn = eval
                    .diagnostics
                    .iter()
                    .any(|d| d.message.contains("uses 'sorry'"));
                if !has_err && !has_sorry_warn {
                    return Ok(VacuityCheck {
                        is_vacuous: true,
                        witness_proof: Some(probe_tactic.to_string()),
                        reason: Some(format!(
                            "Theorem hypotheses are contradictory (`⊢ False` closed immediately by `{probe_tactic}`)"
                        )),
                    });
                }
            }
        }

        Ok(VacuityCheck {
            is_vacuous: false,
            witness_proof: None,
            reason: None,
        })
    }

    async fn check_declaration(
        &self,
        context: &ProofContext,
        code: &str,
    ) -> anyhow::Result<Vec<DiagnosticMsg>> {
        let doc = format!(
            "{}\n{}\n\n{}\n",
            context.imports_header.trim(),
            context.preamble.trim(),
            code.trim()
        );
        let pool = self.pool_for(&context.workspace_root).await;
        let eval = pool
            .evaluate("check_decl.lean", &doc, 0, 0, self.step_timeout() * 2)
            .await?;
        Ok(eval.diagnostics)
    }

    async fn apply_tactic(
        &self,
        state: &ProofStateSnapshot,
        tactic: &str,
    ) -> anyhow::Result<TacticStepResult> {
        let forbidden = scan_forbidden_constructs(tactic);
        if !forbidden.is_empty() {
            return Ok(TacticStepResult {
                accepted: false,
                next_state: None,
                remaining_goals: state.open_goals.clone(),
                diagnostics: vec![DiagnosticMsg {
                    line: 0,
                    column: 0,
                    severity: DiagnosticSeverity::Error,
                    message: format!("Forbidden construct in tactic: {}", forbidden.join(", ")),
                }],
                suggested_tactic: None,
            });
        }

        let mut candidate_prefix = state.tactic_prefix.clone();
        candidate_prefix.push(tactic.trim().to_string());

        let pool = self.pool_for(&state.target.context.workspace_root).await;

        // First try WITHOUT a trailing `sorry` to see if `tactic` completely closes all remaining goals!
        let (closed_doc, closed_line, closed_col) =
            render_target_with_tactics(&state.target, &candidate_prefix, false);
        let closed_eval = pool
            .evaluate(
                "step_closed.lean",
                &closed_doc,
                closed_line,
                closed_col,
                self.step_timeout(),
            )
            .await?;

        let suggested_tactic = closed_eval
            .diagnostics
            .iter()
            .find_map(|d| extract_try_this_suggestion(&d.message));

        let closed_errors: Vec<&DiagnosticMsg> = closed_eval
            .diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();

        if closed_errors.is_empty() {
            // If `exact?` / `apply?` / `simp?` suggested a concrete tactic, record the clean replacement!
            if let Some(clean) = &suggested_tactic
                && let Some(last) = candidate_prefix.last_mut()
            {
                *last = clean.clone();
            }
            let next_snapshot = ProofStateSnapshot {
                target: state.target.clone(),
                tactic_prefix: candidate_prefix,
                open_goals: Vec::new(),
            };
            return Ok(TacticStepResult {
                accepted: true,
                next_state: Some(next_snapshot),
                remaining_goals: Vec::new(),
                diagnostics: closed_eval.diagnostics,
                suggested_tactic,
            });
        }

        // Otherwise, evaluate WITH a trailing `sorry` to inspect intermediate subgoals opened by the tactic.
        let (partial_doc, sorry_line, sorry_col) =
            render_target_with_tactics(&state.target, &candidate_prefix, true);
        let partial_eval = pool
            .evaluate(
                "step_partial.lean",
                &partial_doc,
                sorry_line,
                sorry_col,
                self.step_timeout(),
            )
            .await?;

        let partial_errors: Vec<&DiagnosticMsg> = partial_eval
            .diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();

        if !partial_errors.is_empty() || partial_eval.goals.is_empty() {
            return Ok(TacticStepResult {
                accepted: false,
                next_state: None,
                remaining_goals: state.open_goals.clone(),
                diagnostics: partial_eval.diagnostics,
                suggested_tactic: None,
            });
        }

        let partial_suggestion = partial_eval
            .diagnostics
            .iter()
            .find_map(|d| extract_try_this_suggestion(&d.message));
        if let Some(clean) = &partial_suggestion
            && let Some(last) = candidate_prefix.last_mut()
        {
            *last = clean.clone();
        }

        let next_snapshot = ProofStateSnapshot {
            target: state.target.clone(),
            tactic_prefix: candidate_prefix,
            open_goals: partial_eval.goals.clone(),
        };

        Ok(TacticStepResult {
            accepted: true,
            next_state: Some(next_snapshot),
            remaining_goals: partial_eval.goals,
            diagnostics: partial_eval.diagnostics,
            suggested_tactic: partial_suggestion,
        })
    }

    async fn search_premises(
        &self,
        context: &ProofContext,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<PremiseHit>> {
        let mut hits = Vec::new();

        // If `source_file` points to a local file/folder outside `workspace_root` (e.g. standalone `.lean` folder),
        // index and search that local directory first before searching the global Mathlib workspace.
        if let Some(sf) = &context.source_file {
            let local_dir = if sf.is_dir() {
                Some(sf.as_path())
            } else {
                sf.parent()
            };
            if let Some(dir) = local_dir
                && dir != context.workspace_root
                && dir.exists()
            {
                let local_idx = self.index_for(dir).await;
                hits.extend(local_idx.search(query, limit));
            }
        }

        let index = self.index_for(&context.workspace_root).await;
        for h in index.search(query, limit * 2) {
            if !hits.iter().any(|existing| existing.name == h.name) {
                hits.push(h);
            }
        }

        if hits.len() < limit && self.search_cfg.online_premise_search {
            let online = query_loogle_fallback(query, limit).await;
            for h in online {
                if !hits.iter().any(|existing| existing.name == h.name) {
                    hits.push(h);
                }
            }
        }

        // Batch-verify candidate premise identifiers against `#check` in the warm LSP pool
        // whenever an active `imports_header` / `preamble` scope is present and `lean` is installed.
        let has_active_scope =
            !context.imports_header.trim().is_empty() || !context.preamble.trim().is_empty();
        let tc = detect_toolchain().await;
        if has_active_scope && tc.lean.is_some() && !hits.is_empty() {
            let mut check_doc = format!("{}\n{}\n", context.imports_header, context.preamble);
            let base_lines = check_doc.lines().count() as u32;
            let mut line_to_idx: HashMap<u32, usize> = HashMap::new();
            for (idx, h) in hits.iter().enumerate() {
                let line_no = base_lines + (idx as u32);
                line_to_idx.insert(line_no, idx);
                check_doc.push_str(&format!("#check {}\n", h.name));
            }

            let pool = self.pool_for(&context.workspace_root).await;
            if let Ok(eval) = pool
                .evaluate("premise_check.lean", &check_doc, 0, 0, self.step_timeout())
                .await
            {
                let mut errored_indices = std::collections::HashSet::new();
                for d in &eval.diagnostics {
                    if d.severity == DiagnosticSeverity::Error
                        && let Some(&idx) = line_to_idx.get(&d.line)
                    {
                        errored_indices.insert(idx);
                    }
                }
                for (idx, h) in hits.iter_mut().enumerate() {
                    if !errored_indices.contains(&idx) {
                        h.verified_by_kernel = true;
                    }
                }
                // Rank `#check`-verified in-scope premises ahead of unimported sibling declarations,
                // and prune online-only unverified hallucinations.
                hits.retain(|h| h.verified_by_kernel || h.doc.is_none());
                hits.sort_by(|a, b| {
                    b.verified_by_kernel
                        .cmp(&a.verified_by_kernel)
                        .then_with(|| {
                            b.score
                                .partial_cmp(&a.score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                });
            }
        }

        hits.truncate(limit);
        Ok(hits)
    }

    async fn cold_audit(
        &self,
        file_path: &Path,
        decl_names: &[String],
        policy: &AxiomPolicy,
    ) -> anyhow::Result<AuditReport> {
        run_cold_kernel_audit(file_path, decl_names, policy, &self.lean_cfg).await
    }
}

/// Finds the top-level `:` separating a theorem's binders from its return proposition type.
pub fn find_top_level_conclusion_colon(signature: &str) -> Option<usize> {
    let bytes = signature.as_bytes();
    let mut paren = 0i32;
    let mut brace = 0i32;
    let mut bracket = 0i32;
    let mut last_colon = None;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren += 1,
            b')' => paren = (paren - 1).max(0),
            b'{' => brace += 1,
            b'}' => brace = (brace - 1).max(0),
            b'[' => bracket += 1,
            b']' => bracket = (bracket - 1).max(0),
            b':' if paren == 0 && brace == 0 && bracket == 0 => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    break;
                }
                last_colon = Some(i);
            }
            _ => {}
        }
        i += 1;
    }
    last_colon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_top_level_conclusion_colon_ignoring_binders() {
        let sig = "theorem demo {α : Type} (n : Nat) (h : n > 0) : n + 1 > 1";
        let idx = find_top_level_conclusion_colon(sig).expect("colon");
        assert_eq!(&sig[idx..], ": n + 1 > 1");
    }
}
