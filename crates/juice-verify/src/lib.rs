pub mod ast;
pub mod engine;
pub mod frontend;
pub mod lang;
pub mod proposer;
pub mod repair;
pub mod runtime;
pub mod sarif;

use anyhow::{Context, bail};
pub use ast::VerifyFunction;
pub use engine::{
    CounterexampleReport, FunctionVerificationReport, VerificationBadge, VerifyOptions,
    verify_equivalence, verify_single_function,
};
pub use frontend::{parse_functions_from_source, parse_or_lift_functions};
use juice_core::JuiceConfig;
use juice_llm::ModelRouter;
use juice_smt::Z3Solver;
pub use lang::{SourceLanguage, VerifyType};
pub use proposer::{ObligationKind, SafetyObligation};
pub use repair::{
    VerifiedRepairOutcome, emit_regression_test, splice_function_in_file, synthesize_verified_fix,
};
pub use runtime::{RuntimeDoctorReport, RuntimeReplayVerdict, runtime_doctor};
pub use sarif::render_sarif_report;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct VerifyCommandOptions {
    pub target_fn: Option<String>,
    pub cli_spec: Option<String>,
    pub equiv: Option<(String, String)>,
    pub fix: bool,
    pub in_place: bool,
    pub emit_test: bool,
    pub unwind_depth: usize,
    pub replay_runtime: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileVerificationReport {
    pub path: PathBuf,
    pub language: SourceLanguage,
    pub functions: Vec<FunctionVerificationReport>,
    pub original_source: String,
    pub patched_source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceVerificationSummary {
    pub files: Vec<FileVerificationReport>,
    pub total_functions: usize,
    pub unbounded_proven: usize,
    pub bounded_verified: usize,
    pub bugs_found: usize,
    pub fixed_functions: usize,
}

impl WorkspaceVerificationSummary {
    pub fn has_unfixed_bugs(&self) -> bool {
        self.bugs_found > 0
    }
}

pub fn collect_verifiable_files(path: &Path) -> anyhow::Result<Vec<(PathBuf, SourceLanguage)>> {
    if path.is_file() {
        let lang = SourceLanguage::from_path(path).ok_or_else(|| {
            anyhow::anyhow!(
                "Unsupported source extension for `{}` (expected `.rs`, `.c`, `.h`, `.py`, `.ts`, `.js`)",
                path.display()
            )
        })?;
        return Ok(vec![(path.to_path_buf(), lang)]);
    }

    let mut out = Vec::new();
    for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let s = p.to_string_lossy();
        if s.contains("/target/")
            || s.contains("/node_modules/")
            || s.contains("/.git/")
            || s.contains("/.venv/")
            || s.contains("/__pycache__/")
        {
            continue;
        }
        if let Some(lang) = SourceLanguage::from_path(p) {
            out.push((p.to_path_buf(), lang));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    if out.is_empty() {
        bail!(
            "No Rust, C, Python, or TypeScript files found under `{}`",
            path.display()
        );
    }
    Ok(out)
}

pub async fn verify_path(
    path: &Path,
    cfg: &JuiceConfig,
    cmd_opts: &VerifyCommandOptions,
    router: Option<&ModelRouter>,
) -> anyhow::Result<WorkspaceVerificationSummary> {
    let solver = Z3Solver::discover(cfg.verify.z3_path.as_deref(), cfg.verify.z3_timeout_secs)?;
    let files = collect_verifiable_files(path)?;

    let mut file_reports = Vec::new();
    let mut total_functions = 0usize;
    let mut unbounded_proven = 0usize;
    let mut bounded_verified = 0usize;
    let mut bugs_found = 0usize;
    let mut fixed_functions = 0usize;

    for (file_path, lang) in files {
        let source = std::fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read `{}`", file_path.display()))?;
        let rep = verify_source_buffer(
            &file_path,
            &source,
            lang,
            &solver,
            cfg,
            cmd_opts,
            router,
        )
        .await?;

        if cmd_opts.fix
            && cmd_opts.in_place
            && let Some(patched) = &rep.patched_source
        {
            std::fs::write(&file_path, patched)
                .with_context(|| format!("Failed to write verified fix to `{}`", file_path.display()))?;
        }

        for fn_rep in &rep.functions {
            total_functions += 1;
            if fn_rep.verified_fix_source.is_some() {
                fixed_functions += 1;
            }
            match fn_rep.badge {
                VerificationBadge::UnboundedProven => unbounded_proven += 1,
                VerificationBadge::BoundedVerified { .. } => bounded_verified += 1,
                VerificationBadge::BugFound => bugs_found += 1,
                VerificationBadge::Unknown => {}
            }
        }
        file_reports.push(rep);
    }

    Ok(WorkspaceVerificationSummary {
        files: file_reports,
        total_functions,
        unbounded_proven,
        bounded_verified,
        bugs_found,
        fixed_functions,
    })
}

pub async fn verify_inline_code(
    code: &str,
    lang: SourceLanguage,
    cfg: &JuiceConfig,
    cmd_opts: &VerifyCommandOptions,
    router: Option<&ModelRouter>,
) -> anyhow::Result<FileVerificationReport> {
    let solver = Z3Solver::discover(cfg.verify.z3_path.as_deref(), cfg.verify.z3_timeout_secs)?;
    let virtual_path = PathBuf::from(format!("inline.{}", lang.extension()));
    verify_source_buffer(&virtual_path, code, lang, &solver, cfg, cmd_opts, router).await
}

async fn verify_source_buffer(
    file_path: &Path,
    source: &str,
    lang: SourceLanguage,
    solver: &Z3Solver,
    cfg: &JuiceConfig,
    cmd_opts: &VerifyCommandOptions,
    router: Option<&ModelRouter>,
) -> anyhow::Result<FileVerificationReport> {
    let parsed_fns = parse_or_lift_functions(source, lang, router).await?;
    let engine_opts = VerifyOptions {
        target_fn: cmd_opts.target_fn.clone(),
        cli_spec: cmd_opts.cli_spec.clone(),
        equiv_pair: cmd_opts.equiv.clone(),
        unwind_depth: cmd_opts.unwind_depth,
        replay_runtime: cmd_opts.replay_runtime,
        escalate_to_lean: cfg.verify.escalate_to_lean,
    };

    let mut fn_reports = Vec::new();
    let mut working_source = source.to_string();
    let mut any_patched = false;

    if let Some((fast_name, naive_name)) = &cmd_opts.equiv {
        let fast_fn = parsed_fns
            .iter()
            .find(|f| &f.name == fast_name)
            .ok_or_else(|| anyhow::anyhow!("Function `{fast_name}` not found in `{}`", file_path.display()))?;
        let naive_fn = parsed_fns
            .iter()
            .find(|f| &f.name == naive_name)
            .ok_or_else(|| anyhow::anyhow!("Function `{naive_name}` not found in `{}`", file_path.display()))?;
        let mut rep = verify_equivalence(
            fast_fn,
            naive_fn,
            source,
            &parsed_fns,
            solver,
            &engine_opts,
        )
        .await?;
        if cmd_opts.emit_test
            && let Some(first_cex) = rep.counterexamples.first()
        {
            rep.generated_test = Some(emit_regression_test(fast_fn, first_cex));
        }
        fn_reports.push(rep);
        return Ok(FileVerificationReport {
            path: file_path.to_path_buf(),
            language: lang,
            functions: fn_reports,
            original_source: source.to_string(),
            patched_source: None,
        });
    }

    for func in &parsed_fns {
        if let Some(filter) = &cmd_opts.target_fn
            && !func.name.contains(filter)
        {
            continue;
        }

        let mut rep = verify_single_function(
            func,
            &working_source,
            &parsed_fns,
            solver,
            cfg,
            &engine_opts,
            router,
        )
        .await?;

        if cmd_opts.emit_test
            && let Some(first_cex) = rep.counterexamples.first()
        {
            rep.generated_test = Some(emit_regression_test(func, first_cex));
        }

        if cmd_opts.fix && rep.badge == VerificationBadge::BugFound {
            // Re-parse against `working_source` in case an earlier function in the same file was patched
            let current_fns = parse_functions_from_source(&working_source, lang)?;
            if let Some(cur_func) = current_fns.iter().find(|f| f.name == func.name)
                && let Some(repair) = synthesize_verified_fix(
                    cur_func,
                    &working_source,
                    &rep,
                    solver,
                    cfg,
                    &engine_opts,
                    router,
                )
                .await?
            {
                working_source = repair.patched_full_file;
                any_patched = true;
                let mut fixed_rep = repair.certified_report;
                fixed_rep.generated_test = rep.generated_test;
                fixed_rep.verified_fix_source = Some(repair.patched_function_source);
                rep = fixed_rep;
            }
        }

        fn_reports.push(rep);
    }

    Ok(FileVerificationReport {
        path: file_path.to_path_buf(),
        language: lang,
        functions: fn_reports,
        original_source: source.to_string(),
        patched_source: any_patched.then_some(working_source),
    })
}
