use crate::parser::{list_auditable_declarations, scan_forbidden_constructs};
use crate::workspace::{ToolchainPaths, detect_toolchain, resolve_workspace};
use anyhow::Context;
use juice_core::{AuditReport, AxiomPolicy, LeanConfig};
use regex::Regex;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::OnceLock;
use tokio::process::Command;

/// Runs a cold, isolated out-of-process kernel compilation and `#print axioms` audit on `file_path`.
pub async fn run_cold_kernel_audit(
    file_path: &Path,
    explicit_decls: &[String],
    policy: &AxiomPolicy,
    lean_cfg: &LeanConfig,
) -> anyhow::Result<AuditReport> {
    let tc = detect_toolchain().await;
    run_cold_kernel_audit_with_toolchain(file_path, explicit_decls, policy, lean_cfg, &tc).await
}

pub async fn run_cold_kernel_audit_with_toolchain(
    file_path: &Path,
    explicit_decls: &[String],
    policy: &AxiomPolicy,
    lean_cfg: &LeanConfig,
    tc: &ToolchainPaths,
) -> anyhow::Result<AuditReport> {
    let lean_bin = tc.lean.as_ref().context(
        "Lean 4 binary (`lean`) not found on PATH or ~/.elan/bin. Run `juice doctor` or `juice setup`.",
    )?;

    let canonical_file = std::fs::canonicalize(file_path)
        .with_context(|| format!("Target file does not exist: {}", file_path.display()))?;
    let source = std::fs::read_to_string(&canonical_file)?;

    let decls = if explicit_decls.is_empty() {
        list_auditable_declarations(&source)
    } else {
        explicit_decls.to_vec()
    };

    let forbidden_constructs = scan_forbidden_constructs(&source);
    let has_lexical_sorry = forbidden_constructs
        .iter()
        .any(|c| c.contains("`sorry`") || c.contains("`admit`") || c.contains("`sorryAx`"));

    // Build an isolated temporary harness file appending `#print axioms <decl>` after the source.
    // Appending inside an isolated temp copy in the same directory ensures local relative imports
    // and Lake package paths resolve identically while preventing the user file itself from being mutated.
    let parent_dir = canonical_file
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let temp_harness = tempfile::Builder::new()
        .prefix(".juice_audit_")
        .suffix(".lean")
        .tempfile_in(parent_dir)
        .context("Failed to create isolated audit harness file")?;

    let mut harness_code = source.clone();
    harness_code.push_str("\n\n-- juice cold kernel axiom audit harness\n");
    for decl in &decls {
        harness_code.push_str(&format!("#print axioms {decl}\n"));
    }
    std::fs::write(temp_harness.path(), &harness_code)?;

    let resolved_ws = resolve_workspace(Some(&canonical_file), lean_cfg);
    let output = if resolved_ws.has_lakefile
        && let Some(lake_bin) = tc.lake.as_ref()
    {
        Command::new(lake_bin)
            .arg("env")
            .arg(lean_bin)
            .arg(temp_harness.path())
            .current_dir(&resolved_ws.root)
            .output()
            .await
            .context("Failed to spawn `lake env lean` for cold kernel audit")?
    } else {
        Command::new(lean_bin)
            .arg(temp_harness.path())
            .current_dir(parent_dir)
            .output()
            .await
            .context("Failed to spawn `lean` for cold kernel audit")?
    };

    let combined_out = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
    .trim()
    .to_string();

    Ok(evaluate_audit_output(
        output.status.success(),
        &combined_out,
        &decls,
        forbidden_constructs,
        has_lexical_sorry,
        policy,
    ))
}

/// Pure evaluator for `#print axioms` compiler output and anti-cheat lexical scan results.
pub fn evaluate_audit_output(
    exit_success: bool,
    compiler_output: &str,
    decls: &[String],
    forbidden_constructs: Vec<String>,
    has_lexical_sorry: bool,
    policy: &AxiomPolicy,
) -> AuditReport {
    static DEPENDS_RE: OnceLock<Regex> = OnceLock::new();
    let re = DEPENDS_RE.get_or_init(|| {
        Regex::new(r"'[^']+'\s+depends on axioms:\s*\[([^\]]*)\]").expect("valid axioms regex")
    });

    let mut axioms_set: BTreeSet<String> = BTreeSet::new();
    for caps in re.captures_iter(compiler_output) {
        if let Some(list_match) = caps.get(1) {
            for item in list_match.as_str().split(',') {
                let ax = item.trim();
                if !ax.is_empty() {
                    axioms_set.insert(ax.to_string());
                }
            }
        }
    }

    let has_sorry = has_lexical_sorry
        || axioms_set.contains("sorryAx")
        || compiler_output.contains("declaration uses 'sorry'");

    let axioms_used: Vec<String> = axioms_set.into_iter().collect();
    let forbidden_axioms: Vec<String> = axioms_used
        .iter()
        .filter(|ax| !policy.is_axiom_allowed(ax))
        .cloned()
        .collect();

    let verified = exit_success
        && !has_sorry
        && forbidden_axioms.is_empty()
        && forbidden_constructs.is_empty();

    AuditReport {
        verified,
        declarations_checked: decls.to_vec(),
        axioms_used,
        forbidden_axioms,
        has_sorry,
        forbidden_constructs,
        raw_output: compiler_output.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certifies_clean_standard_axioms() {
        let out = "'MyThm.even_add' depends on axioms: [propext, Classical.choice, Quot.sound]\n'MyThm.refl_nat' does not depend on any axioms";
        let report = evaluate_audit_output(
            true,
            out,
            &["MyThm.even_add".into(), "MyThm.refl_nat".into()],
            vec![],
            false,
            &AxiomPolicy::strict(),
        );
        assert!(report.verified);
        assert!(!report.has_sorry);
        assert!(report.forbidden_axioms.is_empty());
        assert_eq!(
            report.axioms_used,
            vec!["Classical.choice", "Quot.sound", "propext"]
        );
    }

    #[test]
    fn rejects_sorry_ax_and_custom_axioms_and_native_decide() {
        let sorry_out = "'MyThm.unfinished' depends on axioms: [propext, sorryAx]";
        let r1 = evaluate_audit_output(
            true,
            sorry_out,
            &["MyThm.unfinished".into()],
            vec![],
            false,
            &AxiomPolicy::strict(),
        );
        assert!(!r1.verified);
        assert!(r1.has_sorry);
        assert_eq!(r1.forbidden_axioms, vec!["sorryAx"]);

        let native_out = "'MyThm.fast' depends on axioms: [Lean.ofReduceBool, Lean.trustCompiler]";
        let r2 = evaluate_audit_output(
            true,
            native_out,
            &["MyThm.fast".into()],
            vec![],
            false,
            &AxiomPolicy::strict(),
        );
        assert!(!r2.verified);
        assert_eq!(r2.forbidden_axioms.len(), 2);

        let r3 = evaluate_audit_output(
            true,
            native_out,
            &["MyThm.fast".into()],
            vec![],
            false,
            &AxiomPolicy::strict().with_native_decide(true),
        );
        assert!(r3.verified);
    }
}
