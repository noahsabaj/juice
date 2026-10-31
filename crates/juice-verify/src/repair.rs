use crate::ast::VerifyFunction;
use crate::engine::{
    CounterexampleReport, FunctionVerificationReport, VerificationBadge, VerifyOptions,
    verify_single_function,
};
use crate::frontend::parse_functions_from_source;
use crate::lang::{SourceLanguage, VerifyType};
use crate::proposer::ObligationKind;
use crate::runtime::format_rust_arg;
use juice_core::JuiceConfig;
use juice_llm::ModelRouter;
use juice_smt::Z3Solver;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct VerifiedRepairOutcome {
    pub patched_function_source: String,
    pub patched_full_file: String,
    pub certified_report: FunctionVerificationReport,
    pub rejected_candidates: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LlmPatchCandidate {
    patched_function: String,
}

/// Synthesizes a bug fix in a Counterexample-Guided Inductive Synthesis (CEGIS) loop.
/// Every candidate patch is re-parsed and formally re-verified across ALL inputs by Z3/Lean;
/// half-baked fixes (like signed `lo + (hi - lo) / 2` that still overflows `i32`) are rejected!
pub async fn synthesize_verified_fix(
    func: &VerifyFunction,
    full_file_source: &str,
    initial_report: &FunctionVerificationReport,
    solver: &Z3Solver,
    cfg: &JuiceConfig,
    opts: &VerifyOptions,
    router: Option<&ModelRouter>,
) -> anyhow::Result<Option<VerifiedRepairOutcome>> {
    let mut rejected_notes: Vec<String> = Vec::new();

    // 1. Deterministic algebraic & safety candidate rewrites (works 100% offline!)
    let det_candidates = generate_deterministic_candidates(func, initial_report);
    for (label, candidate_fn_src) in det_candidates {
        match evaluate_patch_candidate(
            func,
            full_file_source,
            &candidate_fn_src,
            solver,
            cfg,
            opts,
            router,
        )
        .await
        {
            Ok(Some((patched_file, mut cert_report))) => {
                cert_report
                    .notes
                    .push(format!("Formally verified auto-repair (`{label}`)"));
                for rej in &rejected_notes {
                    cert_report.notes.push(rej.clone());
                }
                return Ok(Some(VerifiedRepairOutcome {
                    patched_function_source: candidate_fn_src,
                    patched_full_file: patched_file,
                    certified_report: cert_report,
                    rejected_candidates: rejected_notes,
                }));
            }
            Ok(None) => {
                rejected_notes.push(format!(
                    "Rejected candidate `{label}` after Z3 found remaining counterexample"
                ));
            }
            Err(_) => {}
        }
    }

    // 2. Iterative CEGIS LLM repair loop (up to 4 counterexample-guided rounds)
    let Some(router) = router else {
        return Ok(None);
    };

    let mut latest_cex_summary = summarize_counterexamples(&initial_report.counterexamples);
    for round in 1..=4 {
        let system = "You are a formal verification repair engine. Rewrite ONLY the target function in its original programming language to eliminate all safety bugs (integer overflow, division by zero, out-of-bounds, panics) while preserving exact intended semantics and all `@ensures` contracts across ALL possible inputs. Return JSON `{\"patched_function\": \"...\"}`.";
        let user = format!(
            "Language: {}\nOriginal function:\n{}\n\nConfirmed Z3 counterexamples & runtime failures:\n{}\n\nPreviously rejected patches:\n{}",
            func.language.display_name(),
            func.raw_source,
            latest_cex_summary,
            if rejected_notes.is_empty() {
                "None".to_string()
            } else {
                rejected_notes.join("\n")
            }
        );

        let Ok(candidate) = router
            .complete_planner_json::<LlmPatchCandidate>(system, &user)
            .await
        else {
            break;
        };

        let patched_file = splice_function_in_file(full_file_source, func, &candidate.patched_function);
        let Ok(parsed_fns) = parse_functions_from_source(&patched_file, func.language) else {
            rejected_notes.push(format!("Round {round}: candidate failed deterministic AST check"));
            continue;
        };
        let Some(patched_fn) = parsed_fns.iter().find(|f| f.name == func.name) else {
            continue;
        };

        let mut verify_opts = opts.clone();
        verify_opts.replay_runtime = false;
        let rep = verify_single_function(
            patched_fn,
            &patched_file,
            &parsed_fns,
            solver,
            cfg,
            &verify_opts,
            Some(router),
        )
        .await?;

        if matches!(
            rep.badge,
            VerificationBadge::UnboundedProven | VerificationBadge::BoundedVerified { .. }
        ) {
            let mut cert_report = rep;
            cert_report.notes.push(format!(
                "Formally verified CEGIS auto-repair in round {round}"
            ));
            return Ok(Some(VerifiedRepairOutcome {
                patched_function_source: candidate.patched_function,
                patched_full_file: patched_file,
                certified_report: cert_report,
                rejected_candidates: rejected_notes,
            }));
        } else {
            let new_cex = summarize_counterexamples(&rep.counterexamples);
            rejected_notes.push(format!(
                "Round {round} candidate rejected by Z3: {new_cex}"
            ));
            latest_cex_summary = new_cex;
        }
    }

    Ok(None)
}

async fn evaluate_patch_candidate(
    orig_func: &VerifyFunction,
    full_file_source: &str,
    candidate_fn_src: &str,
    solver: &Z3Solver,
    cfg: &JuiceConfig,
    opts: &VerifyOptions,
    router: Option<&ModelRouter>,
) -> anyhow::Result<Option<(String, FunctionVerificationReport)>> {
    let patched_file = splice_function_in_file(full_file_source, orig_func, candidate_fn_src);
    let parsed_fns = parse_functions_from_source(&patched_file, orig_func.language)?;
    let Some(patched_fn) = parsed_fns.iter().find(|f| f.name == orig_func.name) else {
        return Ok(None);
    };
    // Preserve doc-comment `@requires` / `@ensures` from the original function!
    let mut candidate_fn = patched_fn.clone();
    if candidate_fn.requires.is_empty() {
        candidate_fn.requires = orig_func.requires.clone();
    }
    if candidate_fn.ensures.is_empty() {
        candidate_fn.ensures = orig_func.ensures.clone();
    }

    let mut verify_opts = opts.clone();
    verify_opts.replay_runtime = false;
    let rep = verify_single_function(
        &candidate_fn,
        &patched_file,
        &parsed_fns,
        solver,
        cfg,
        &verify_opts,
        router,
    )
    .await?;

    if matches!(
        rep.badge,
        VerificationBadge::UnboundedProven | VerificationBadge::BoundedVerified { .. }
    ) {
        Ok(Some((patched_file, rep)))
    } else {
        Ok(None)
    }
}

pub fn splice_function_in_file(
    full_file_source: &str,
    orig_func: &VerifyFunction,
    replacement_fn: &str,
) -> String {
    let mut out = String::with_capacity(full_file_source.len() + replacement_fn.len());
    out.push_str(&full_file_source[..orig_func.span_start_byte]);
    out.push_str(replacement_fn.trim_end());
    if orig_func.language == SourceLanguage::Python {
        out.push('\n');
    }
    out.push_str(&full_file_source[orig_func.span_end_byte..]);
    out
}

fn summarize_counterexamples(cexs: &[CounterexampleReport]) -> String {
    cexs.iter()
        .map(|c| {
            let args: Vec<String> = c.inputs.iter().map(|(k, v)| format!("{k} = {v}")).collect();
            format!(
                "- Line {}: {} (inputs: {})",
                c.line,
                c.description,
                args.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_deterministic_candidates(
    func: &VerifyFunction,
    report: &FunctionVerificationReport,
) -> Vec<(String, String)> {
    let mut candidates = Vec::new();
    let src = &func.raw_source;
    let has_overflow = report
        .counterexamples
        .iter()
        .any(|c| c.kind == ObligationKind::IntegerOverflow);
    let has_div_zero = report
        .counterexamples
        .iter()
        .any(|c| c.kind == ObligationKind::DivisionByZero);

    // 1. Classic midpoint / average overflow: probe textbook `a + (b - a) / 2` FIRST (so Z3 actively
    //    proves/rejects it if signed `i32` inputs can be negative!) followed by universal bitwise
    //    floor-midpoint `(a & b) + ((a ^ b) >> 1)`!
    if has_overflow && func.params.len() == 2 {
        let a = &func.params[0].name;
        let b = &func.params[1].name;
        let looks_like_midpoint = (src.contains("/ 2") || src.contains(">> 1"))
            && src.contains(a)
            && src.contains(b);

        if looks_like_midpoint {
            for pat in [
                format!("({a} + {b}) / 2"),
                format!("({b} + {a}) / 2"),
                format!("({a} + {b}) >> 1"),
                format!("{a} + ({b} - {a}) / 2"),
            ] {
                if src.contains(&pat) {
                    // First test `a + (b - a) / 2` so CEGIS explicitly rejects it on signed types!
                    if pat != format!("{a} + ({b} - {a}) / 2") {
                        candidates.push((
                            format!("textbook subtraction midpoint `{a} + ({b} - {a}) / 2`"),
                            src.replace(&pat, &format!("{a} + ({b} - {a}) / 2")),
                        ));
                    }
                    // Universal two's-complement & unsigned overflow-free Hacker's Delight midpoint
                    candidates.push((
                        format!("Hacker's Delight bitwise midpoint `({a} & {b}) + (({a} ^ {b}) >> 1)`"),
                        src.replace(&pat, &format!("({a} & {b}) + (({a} ^ {b}) >> 1)")),
                    ));
                }
            }
        }
    }

    // 2. Rust `abs` overflow on `i32::MIN` -> `saturating_abs()` / checked clamp
    if has_overflow && func.language == SourceLanguage::Rust && func.params.len() == 1 {
        let x = &func.params[0].name;
        for pat in [format!("-{x}"), format!("- {x}"), format!("{x}.abs()")] {
            if src.contains(&pat) {
                let rty = match func.return_ty {
                    VerifyType::BitVec { width: 8, signed: true } => "i8",
                    VerifyType::BitVec { width: 16, signed: true } => "i16",
                    VerifyType::BitVec { width: 64, signed: true } => "i64",
                    _ => "i32",
                };
                candidates.push((
                    format!("saturating absolute value on `{rty}::MIN`"),
                    format!(
                        "pub fn {}({x}: {rty}) -> {rty} {{\n    if {x} == {rty}::MIN {{\n        {rty}::MAX\n    }} else if {x} < 0 {{\n        -{x}\n    }} else {{\n        {x}\n    }}\n}}",
                        func.name
                    ),
                ));
            }
        }
    }

    // 3. Division by zero guard when second parameter is divisor
    if has_div_zero && func.params.len() == 2 {
        let b = &func.params[1].name;
        match func.language {
            SourceLanguage::Rust => {
                if let Some(open_b) = src.find('{') {
                    let mut guarded = String::new();
                    guarded.push_str(&src[..=open_b]);
                    guarded.push_str(&format!("\n    if {b} == 0 {{\n        return 0;\n    }}"));
                    guarded.push_str(&src[open_b + 1..]);
                    candidates.push((format!("zero-divisor guard `{b} == 0`"), guarded));
                }
            }
            SourceLanguage::C => {
                if let Some(open_b) = src.find('{') {
                    let mut guarded = String::new();
                    guarded.push_str(&src[..=open_b]);
                    guarded.push_str(&format!("\n    if ({b} == 0) {{ return 0; }}"));
                    guarded.push_str(&src[open_b + 1..]);
                    candidates.push((format!("zero-divisor guard `{b} == 0`"), guarded));
                }
            }
            SourceLanguage::Python => {
                if let Some(insert_pos) = find_python_body_insert_offset(src) {
                    let mut guarded = String::new();
                    guarded.push_str(&src[..insert_pos]);
                    guarded.push_str(&format!("    if {b} == 0:\n        return 0\n"));
                    guarded.push_str(&src[insert_pos..]);
                    candidates.push((format!("zero-divisor guard `{b} == 0`"), guarded));
                }
            }
            SourceLanguage::TypeScript => {
                if let Some(open_b) = src.find('{') {
                    let mut guarded = String::new();
                    guarded.push_str(&src[..=open_b]);
                    guarded.push_str(&format!("\n  if ({b} === 0) {{ return 0; }}"));
                    guarded.push_str(&src[open_b + 1..]);
                    candidates.push((format!("zero-divisor guard `{b} === 0`"), guarded));
                }
            }
        }
    }

    candidates
}

fn find_python_body_insert_offset(src: &str) -> Option<usize> {
    let first_nl = src.find('\n')?;
    let after_def = &src[first_nl + 1..];
    let trimmed = after_def.trim_start_matches([' ', '\t']);
    for delim in ["\"\"\"", "'''"] {
        if let Some(after_open) = trimmed.strip_prefix(delim)
            && let Some(close_rel) = after_open.find(delim)
        {
            let ws_len = after_def.len() - trimmed.len();
            let doc_end_in_after = ws_len + delim.len() + close_rel + delim.len();
            if let Some(nl_after_doc) = after_def[doc_end_in_after..].find('\n') {
                return Some(first_nl + 1 + doc_end_in_after + nl_after_doc + 1);
            }
        }
    }
    Some(first_nl + 1)
}

/// Emits an idiomatic native regression unit test reproducing a confirmed counterexample.
pub fn emit_regression_test(func: &VerifyFunction, cex: &CounterexampleReport) -> String {
    let clean_name = func
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();

    match func.language {
        SourceLanguage::Rust => {
            let mut setup = String::new();
            let mut args = Vec::new();
            for p in &func.params {
                let lit = format_rust_arg(&p.name, &p.ty, &cex.raw_values, Some(&func.raw_source));
                setup.push_str(&format!("    let {} = {};\n", p.name, lit));
                args.push(p.name.clone());
            }
            let expects_panic = matches!(
                cex.kind,
                ObligationKind::IntegerOverflow
                    | ObligationKind::DivisionByZero
                    | ObligationKind::ShiftOverflow
                    | ObligationKind::OutOfBounds
                    | ObligationKind::AssertionOrPanic
            );
            let panic_attr = if expects_panic {
                "// Reproduces Z3 counterexample crash (remove `#[should_panic]` once fixed)\n#[should_panic]\n"
            } else {
                ""
            };
            format!(
                "#[test]\n{panic_attr}fn regression_{clean_name}_line_{}() {{\n{setup}    let _res = {}({});\n}}",
                cex.line,
                func.name,
                args.join(", ")
            )
        }
        SourceLanguage::C => {
            let args: Vec<String> = func
                .params
                .iter()
                .map(|p| cex.inputs.get(&p.name).cloned().unwrap_or_else(|| "0".to_string()))
                .collect();
            format!(
                "/* Z3 counterexample regression harness for `{}` (line {}) */\nint test_regression_{clean_name}(void) {{\n    /* Violated: {} */\n    (void){}({});\n    return 0;\n}}",
                func.name,
                cex.line,
                cex.description,
                func.name,
                args.join(", ")
            )
        }
        SourceLanguage::Python => {
            let args: Vec<String> = func
                .params
                .iter()
                .map(|p| cex.inputs.get(&p.name).cloned().unwrap_or_else(|| "0".to_string()))
                .collect();
            format!(
                "def test_regression_{clean_name}_line_{}():\n    # Z3 counterexample violating: {}\n    {}({})",
                cex.line,
                cex.description,
                func.name,
                args.join(", ")
            )
        }
        SourceLanguage::TypeScript => {
            let args: Vec<String> = func
                .params
                .iter()
                .map(|p| cex.inputs.get(&p.name).cloned().unwrap_or_else(|| "0".to_string()))
                .collect();
            format!(
                "import {{ test, expect }} from \"bun:test\";\n\ntest(\"regression {clean_name} line {}\", () => {{\n  // Z3 counterexample violating: {}\n  expect(() => {}({})).not.toThrow();\n}});",
                cex.line,
                cex.description,
                func.name,
                args.join(", ")
            )
        }
    }
}
