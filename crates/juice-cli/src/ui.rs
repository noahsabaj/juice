use indicatif::{ProgressBar, ProgressStyle};
use juice_core::{EventSink, SearchEvent};
use similar::{ChangeTag, TextDiff};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct TerminalProgress {
    spinner: Option<ProgressBar>,
}

impl TerminalProgress {
    pub fn new(json_mode: bool) -> (Self, EventSink) {
        if json_mode {
            return (
                Self { spinner: None },
                Arc::new(|_| {}),
            );
        }

        let pb = ProgressBar::new_spinner();
        let style = ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner());
        pb.set_style(style);
        pb.enable_steady_tick(Duration::from_millis(90));

        let pb_shared = Arc::new(Mutex::new(pb.clone()));
        let sink: EventSink = Arc::new(move |ev: SearchEvent| {
            let Ok(bar) = pb_shared.lock() else {
                return;
            };
            match ev {
                SearchEvent::TargetStarted {
                    decl_name,
                    total_goals,
                    ..
                } => {
                    bar.set_message(format!(
                        "Proving `{decl_name}` ({total_goals} initial goal(s))..."
                    ));
                }
                SearchEvent::Autoformalizing { claim_preview } => {
                    bar.set_message(format!("Autoformalizing: \"{claim_preview}\"..."));
                }
                SearchEvent::VacuityChecked {
                    decl_name,
                    is_vacuous,
                    reason,
                } => {
                    if is_vacuous {
                        bar.println(format!(
                            "  [VACUOUS ✗] `{decl_name}` rejected: {}",
                            reason.unwrap_or_default()
                        ));
                    }
                }
                SearchEvent::BackTranslationChecked {
                    decl_name,
                    faithful,
                    critique,
                } => {
                    if faithful {
                        bar.println(format!(
                            "  [SEMANTIC ✓] `{decl_name}` passed blind back-translation check"
                        ));
                    } else {
                        bar.println(format!(
                            "  [SEMANTIC ↻] `{decl_name}` refining formalization: {critique}"
                        ));
                    }
                }
                SearchEvent::HammerHit { tactic, .. } => {
                    bar.println(format!("  [HAMMER ⚡] Closed subgoal via `{tactic}`"));
                }
                SearchEvent::PlannerSketching { target_id, round } => {
                    bar.set_message(format!(
                        "Tier-1 Planner sketching sub-lemmas for `{target_id}` (round {round})..."
                    ));
                }
                SearchEvent::CounterexampleFound { lemma, witness, .. } => {
                    bar.println(format!(
                        "  [COUNTEREXAMPLE ✗] Refuted false sub-lemma `{lemma}` -> {witness}"
                    ));
                }
                SearchEvent::SketchAccepted {
                    sub_lemmas_count, ..
                } => {
                    bar.println(format!(
                        "  [SKETCH ✓] Outer theorem compiles with {sub_lemmas_count} sub-goal(s)"
                    ));
                }
                SearchEvent::NodeExpanded {
                    target_id,
                    nodes_expanded,
                    dag_unique_states,
                    best_open_goals,
                } => {
                    bar.set_message(format!(
                        "MCGS `{target_id}` | expanded: {nodes_expanded} | unique DAG states: {dag_unique_states} | min open goals: {best_open_goals}"
                    ));
                }
                SearchEvent::SubLemmaSolved { lemma_name, .. } => {
                    bar.println(format!("  [SUB-LEMMA ✓] Closed `{lemma_name}`"));
                }
                SearchEvent::ColdAuditCompleted {
                    decl_name,
                    verified,
                    axioms,
                } => {
                    if verified {
                        let ax = if axioms.is_empty() {
                            "none (pure kernel reduction)".to_string()
                        } else {
                            axioms.join(", ")
                        };
                        bar.println(format!(
                            "  [KERNEL CERTIFIED ✓] `{decl_name}` (axioms: [{ax}])"
                        ));
                    } else {
                        bar.println(format!("  [KERNEL REJECTED ✗] `{decl_name}`"));
                    }
                }
                SearchEvent::TargetCompleted {
                    decl_name,
                    success,
                    elapsed_ms,
                    ..
                } => {
                    if success {
                        bar.println(format!(
                            "  [SOLVED ✓] `{decl_name}` in {:.2}s",
                            (elapsed_ms as f64) / 1000.0
                        ));
                    } else {
                        bar.println(format!(
                            "  [UNSOLVED ✗] `{decl_name}` after {:.2}s",
                            (elapsed_ms as f64) / 1000.0
                        ));
                    }
                }
            }
        });

        (Self { spinner: Some(pb) }, sink)
    }

    pub fn finish(&self) {
        if let Some(pb) = &self.spinner {
            pb.finish_and_clear();
        }
    }
}

pub fn render_unified_diff(path: &Path, original: &str, patched: &str) -> String {
    let diff = TextDiff::from_lines(original, patched);
    let label = path.display();
    let mut out = format!("--- a/{label}\n+++ b/{label}\n");
    for hunk in diff.unified_diff().context_radius(3).iter_hunks() {
        out.push_str(&format!("{}", hunk.header()));
        for change in hunk.iter_changes() {
            let sign = match change.tag() {
                ChangeTag::Delete => "-",
                ChangeTag::Insert => "+",
                ChangeTag::Equal => " ",
            };
            out.push_str(sign);
            out.push_str(change.value());
            if change.missing_newline() {
                out.push('\n');
            }
        }
    }
    out
}
