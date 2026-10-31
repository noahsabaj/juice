use juice_core::JuiceConfig;
use juice_smt::locate_z3;
use juice_verify::{
    SourceLanguage, VerificationBadge, VerifyCommandOptions, render_sarif_report,
    verify_inline_code, verify_path,
};

fn default_opts() -> VerifyCommandOptions {
    VerifyCommandOptions {
        target_fn: None,
        cli_spec: None,
        equiv: None,
        fix: false,
        in_place: false,
        emit_test: true,
        unwind_depth: 16,
        replay_runtime: true,
    }
}

#[tokio::test]
async fn rust_midpoint_overflow_caught_replayed_and_formally_repaired() {
    if locate_z3(None).is_none() {
        return;
    }
    let cfg = JuiceConfig::default();
    let code = r#"
pub fn midpoint(lo: i32, hi: i32) -> i32 {
    (lo + hi) / 2
}
"#;
    // 1. Detect overflow + auto-proposed midpoint postcondition violation
    let buggy = verify_inline_code(code, SourceLanguage::Rust, &cfg, &default_opts(), None)
        .await
        .expect("verifies rust");
    assert_eq!(buggy.functions.len(), 1);
    assert_eq!(buggy.functions[0].badge, VerificationBadge::BugFound);
    assert!(!buggy.functions[0].counterexamples.is_empty());
    assert!(
        buggy.functions[0]
            .generated_test
            .as_deref()
            .unwrap_or_default()
            .contains("#[test]")
    );

    // 2. Auto-repair rejects `lo + (hi - lo) / 2` and proves `(lo & hi) + ((lo ^ hi) >> 1)` across all 2^64 pairs
    let mut fix_opts = default_opts();
    fix_opts.fix = true;
    let repaired = verify_inline_code(code, SourceLanguage::Rust, &cfg, &fix_opts, None)
        .await
        .expect("repairs rust");
    assert_eq!(
        repaired.functions[0].badge,
        VerificationBadge::UnboundedProven
    );
    assert!(
        repaired
            .patched_source
            .as_deref()
            .unwrap_or_default()
            .contains("(lo & hi) + ((lo ^ hi) >> 1)")
    );
    assert!(
        repaired.functions[0]
            .notes
            .iter()
            .any(|n| n.contains("Rejected candidate"))
    );
}

#[tokio::test]
async fn c_pointer_bounds_and_sarif_generation() {
    if locate_z3(None).is_none() {
        return;
    }
    let cfg = JuiceConfig::default();
    let dir = tempfile::tempdir().unwrap();
    let c_path = dir.path().join("buf.c");
    std::fs::write(
        &c_path,
        r#"
#include <stdint.h>

// @requires len > 0
int32_t bad_tail(const int32_t* buf, int32_t len) {
    return buf[len];
}

// @requires len > 0
int32_t good_tail(const int32_t* buf, int32_t len) {
    return buf[len - 1];
}
"#,
    )
    .unwrap();

    let summary = verify_path(&c_path, &cfg, &default_opts(), None)
        .await
        .expect("verifies C");
    assert_eq!(summary.total_functions, 2);
    assert_eq!(summary.unbounded_proven, 1);
    assert_eq!(summary.bugs_found, 1);

    let sarif = render_sarif_report(&summary.files);
    assert_eq!(sarif["version"], "2.1.0");
    let results = sarif["runs"][0]["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["ruleId"], "juice/out-of-bounds");
}

#[tokio::test]
async fn python_unbounded_loop_induction_and_bounded_fallback() {
    if locate_z3(None).is_none() {
        return;
    }
    let cfg = JuiceConfig::default();
    // 1. Standard counted loop proves via unbounded inductive invariant
    let counted = r#"
def prefix_sum(n: int) -> int:
    """
    @requires n >= 0
    """
    total = 0
    for i in range(n):
        total += i
    return total
"#;
    let rep_unbounded =
        verify_inline_code(counted, SourceLanguage::Python, &cfg, &default_opts(), None)
            .await
            .unwrap();
    assert_eq!(
        rep_unbounded.functions[0].badge,
        VerificationBadge::UnboundedProven
    );

    // 2. Collatz-like loop without synthesized invariant falls back cleanly to BoundedVerified
    let collatz_step = r#"
def hailstone_steps(x: int) -> int:
    """
    @requires x > 1
    """
    steps = 0
    cur = x
    while cur > 1:
        if cur % 2 == 0:
            cur = cur // 2
        else:
            cur = 3 * cur + 1
        steps += 1
    return steps
"#;
    let mut bmc_opts = default_opts();
    bmc_opts.unwind_depth = 8;
    let rep_bounded =
        verify_inline_code(collatz_step, SourceLanguage::Python, &cfg, &bmc_opts, None)
            .await
            .unwrap();
    assert_eq!(
        rep_bounded.functions[0].badge,
        VerificationBadge::BoundedVerified { depth: 8 }
    );
}

#[tokio::test]
async fn typescript_equivalence_detects_divergence_and_proves_match() {
    if locate_z3(None).is_none() {
        return;
    }
    let cfg = JuiceConfig::default();
    let ts_buggy = r#"
export function fast_clamp(x: number): number {
  if (x < 0) return 0;
  if (x > 100) return 100;
  if (x === 100) return 99;
  return x;
}

export function ref_clamp(x: number): number {
  return x < 0 ? 0 : (x > 100 ? 100 : x);
}
"#;
    let mut eq_opts = default_opts();
    eq_opts.equiv = Some(("fast_clamp".to_string(), "ref_clamp".to_string()));
    let rep_div = verify_inline_code(ts_buggy, SourceLanguage::TypeScript, &cfg, &eq_opts, None)
        .await
        .unwrap();
    assert_eq!(rep_div.functions[0].badge, VerificationBadge::BugFound);
    assert_eq!(
        rep_div.functions[0].counterexamples[0]
            .inputs
            .get("x")
            .map(String::as_str),
        Some("100")
    );

    let ts_clean = ts_buggy.replace("if (x === 100) return 99;", "");
    let rep_ok = verify_inline_code(&ts_clean, SourceLanguage::TypeScript, &cfg, &eq_opts, None)
        .await
        .unwrap();
    assert_eq!(
        rep_ok.functions[0].badge,
        VerificationBadge::UnboundedProven
    );
}
