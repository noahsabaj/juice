use juice_core::{ProofStateSnapshot, VerifierBackend};

/// Ordered zero-LLM deterministic Lean 4 automation tactics tried on every new goal state.
/// Cheapest core kernel tactics run first, followed by arithmetic/algebraic decision procedures
/// and Mathlib library-search tactics (`exact?`, `apply?`).
pub const DEFAULT_HAMMERS: &[&str] = &[
    "rfl",
    "trivial",
    "decide",
    " simp",
    "norm_num",
    "ring",
    "linarith",
    "omega",
    "aesop",
    "exact?",
    "apply?",
];

/// Runs the deterministic hammer cascade against `state`, returning the winning clean tactic
/// (substituting `"Try this: ..."` suggestions from `exact?` / `apply?`) as soon as all open goals close.
pub async fn try_hammer_cascade(
    verifier: &dyn VerifierBackend,
    state: &ProofStateSnapshot,
) -> Option<String> {
    for &raw_hammer in DEFAULT_HAMMERS {
        let hammer = raw_hammer.trim();
        if let Ok(step) = verifier.apply_tactic(state, hammer).await
            && step.accepted
            && step.remaining_goals.is_empty()
        {
            let clean = step
                .suggested_tactic
                .unwrap_or_else(|| hammer.to_string());
            return Some(clean);
        }
    }
    None
}
