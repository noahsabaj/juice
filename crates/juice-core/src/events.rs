use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SearchEvent {
    TargetStarted {
        target_id: String,
        decl_name: String,
        total_goals: usize,
    },
    Autoformalizing {
        claim_preview: String,
    },
    VacuityChecked {
        decl_name: String,
        is_vacuous: bool,
        reason: Option<String>,
    },
    BackTranslationChecked {
        decl_name: String,
        faithful: bool,
        critique: String,
    },
    HammerHit {
        target_id: String,
        tactic: String,
    },
    PlannerSketching {
        target_id: String,
        round: usize,
    },
    CounterexampleFound {
        target_id: String,
        lemma: String,
        witness: String,
    },
    SketchAccepted {
        target_id: String,
        sub_lemmas_count: usize,
    },
    NodeExpanded {
        target_id: String,
        nodes_expanded: usize,
        dag_unique_states: usize,
        best_open_goals: usize,
    },
    SubLemmaSolved {
        target_id: String,
        lemma_name: String,
        proof: String,
    },
    ColdAuditCompleted {
        decl_name: String,
        verified: bool,
        axioms: Vec<String>,
    },
    TargetCompleted {
        target_id: String,
        decl_name: String,
        success: bool,
        proof: Option<String>,
        elapsed_ms: u64,
    },
}

pub type EventSink = Arc<dyn Fn(SearchEvent) + Send + Sync>;

pub fn noop_event_sink() -> EventSink {
    Arc::new(|_| {})
}
