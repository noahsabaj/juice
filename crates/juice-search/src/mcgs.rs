use crate::hammers::try_hammer_cascade;
use juice_core::{
    EventSink, NormalizedGoal, PremiseHit, ProofStateSnapshot, SearchConfig, SearchEvent,
    VerifierBackend,
};
use juice_llm::ModelRouter;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;

pub type StateId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GoalStackKey(Vec<NormalizedGoal>);

impl GoalStackKey {
    fn from_goals(goals: &[NormalizedGoal]) -> Self {
        // Preserve goal-stack order because Lean 4 sequential tactics operate on the top goal first.
        Self(goals.to_vec())
    }
}

#[derive(Debug, Clone)]
struct TacticEdge {
    tactic: String,
    prior: f64,
    visits: usize,
    total_value: f64,
    in_flight: usize,
    target_state: Option<StateId>,
    rejected: bool,
}

#[derive(Debug, Clone)]
struct GraphNode {
    snapshot: ProofStateSnapshot,
    visits: usize,
    expanded: bool,
    expanding_in_flight: bool,
    /// `Some(tail_tactics)` once this node (or any downstream OR-branch) closes all goals.
    solved_tail: Option<Vec<String>>,
    edges: Vec<TacticEdge>,
    /// Incoming `(parent_state_id, edge_idx)` pairs for upward solution propagation across the DAG.
    parents: Vec<(StateId, usize)>,
}

struct ProofDag {
    nodes: Vec<GraphNode>,
    state_table: HashMap<GoalStackKey, StateId>,
    nodes_expanded: usize,
    best_open_goals: usize,
}

impl ProofDag {
    fn new(root_snapshot: ProofStateSnapshot) -> Self {
        let best_open_goals = root_snapshot.open_goals.len();
        let key = GoalStackKey::from_goals(&root_snapshot.open_goals);
        let solved_tail = if root_snapshot.open_goals.is_empty() {
            Some(Vec::new())
        } else {
            None
        };
        let root = GraphNode {
            snapshot: root_snapshot,
            visits: 1,
            expanded: false,
            expanding_in_flight: false,
            solved_tail,
            edges: Vec::new(),
            parents: Vec::new(),
        };
        let mut state_table = HashMap::new();
        state_table.insert(key, 0);
        Self {
            nodes: vec![root],
            state_table,
            nodes_expanded: 0,
            best_open_goals,
        }
    }

    fn root_solution(&self) -> Option<String> {
        self.nodes
            .first()
            .and_then(|n| n.solved_tail.as_ref())
            .map(|lines| lines.join("\n"))
    }

    fn intern_state(
        &mut self,
        snapshot: ProofStateSnapshot,
        parent: (StateId, usize),
    ) -> (StateId, bool) {
        let key = GoalStackKey::from_goals(&snapshot.open_goals);
        if let Some(&existing_id) = self.state_table.get(&key) {
            if !self.nodes[existing_id].parents.contains(&parent) {
                self.nodes[existing_id].parents.push(parent);
            }
            // If the merged node was already solved via another path, propagate immediately!
            if let Some(tail) = self.nodes[existing_id].solved_tail.clone() {
                let edge_tactic = self.nodes[parent.0].edges[parent.1].tactic.clone();
                let mut parent_tail = vec![edge_tactic];
                parent_tail.extend(tail);
                self.mark_solved_and_propagate(parent.0, parent_tail);
            }
            return (existing_id, false);
        }

        let new_id = self.nodes.len();
        self.best_open_goals = self.best_open_goals.min(snapshot.open_goals.len());
        let is_solved = snapshot.open_goals.is_empty();
        let node = GraphNode {
            snapshot,
            visits: 0,
            expanded: is_solved,
            expanding_in_flight: false,
            solved_tail: if is_solved { Some(Vec::new()) } else { None },
            edges: Vec::new(),
            parents: vec![parent],
        };
        self.state_table.insert(key, new_id);
        self.nodes.push(node);

        if is_solved {
            let edge_tactic = self.nodes[parent.0].edges[parent.1].tactic.clone();
            self.mark_solved_and_propagate(parent.0, vec![edge_tactic]);
        }

        (new_id, true)
    }

    fn mark_solved_and_propagate(&mut self, start_id: StateId, tail: Vec<String>) {
        let mut stack = vec![(start_id, tail)];
        while let Some((node_id, candidate_tail)) = stack.pop() {
            if self.nodes[node_id].solved_tail.is_some() {
                continue;
            }
            self.nodes[node_id].solved_tail = Some(candidate_tail.clone());
            let parents = self.nodes[node_id].parents.clone();
            for (parent_id, edge_idx) in parents {
                if self.nodes[parent_id].solved_tail.is_none() {
                    let tactic = self.nodes[parent_id].edges[edge_idx].tactic.clone();
                    let mut p_tail = vec![tactic];
                    p_tail.extend(candidate_tail.clone());
                    stack.push((parent_id, p_tail));
                }
            }
        }
    }

    fn backup_value(&mut self, path: &[(StateId, Option<usize>)], value: f64) {
        for &(node_id, edge_opt) in path {
            self.nodes[node_id].visits += 1;
            if let Some(edge_idx) = edge_opt {
                let edge = &mut self.nodes[node_id].edges[edge_idx];
                edge.visits += 1;
                edge.total_value += value;
                edge.in_flight = edge.in_flight.saturating_sub(1);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct McgsOutcome {
    pub proof_script: Option<String>,
    pub nodes_expanded: usize,
    pub unique_states: usize,
    pub stuck_goals: Vec<NormalizedGoal>,
}

enum NextWorkItem {
    ExpandNode {
        node_id: StateId,
        snapshot: ProofStateSnapshot,
        path: Vec<(StateId, Option<usize>)>,
    },
    EvaluateEdge {
        parent_id: StateId,
        edge_idx: usize,
        snapshot: ProofStateSnapshot,
        tactic: String,
        path: Vec<(StateId, Option<usize>)>,
    },
    Wait,
    Exhausted,
}

/// Parallel Monte Carlo Graph Search engine with PUCT selection, state deduplication,
/// virtual loss for async concurrency, and automatic deterministic hammer cascades.
pub struct McgsEngine {
    verifier: Arc<dyn VerifierBackend>,
    router: ModelRouter,
    cfg: SearchConfig,
    events: EventSink,
}

impl McgsEngine {
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

    pub async fn search(
        &self,
        root_snapshot: ProofStateSnapshot,
        premises: &[PremiseHit],
        node_budget: usize,
    ) -> anyhow::Result<McgsOutcome> {
        let target_id = root_snapshot.target.id.clone();

        if root_snapshot.open_goals.is_empty() {
            return Ok(McgsOutcome {
                proof_script: Some("rfl".to_string()),
                nodes_expanded: 0,
                unique_states: 1,
                stuck_goals: Vec::new(),
            });
        }

        if self.cfg.enable_hammers
            && let Some(winning_hammer) = try_hammer_cascade(&*self.verifier, &root_snapshot).await
        {
            (self.events)(SearchEvent::HammerHit {
                target_id: target_id.clone(),
                tactic: winning_hammer.clone(),
            });
            return Ok(McgsOutcome {
                proof_script: Some(winning_hammer),
                nodes_expanded: 1,
                unique_states: 1,
                stuck_goals: Vec::new(),
            });
        }

        let dag = Arc::new(Mutex::new(ProofDag::new(root_snapshot)));
        let worker_count = self.cfg.concurrency.max(1);
        let premises_arc = Arc::new(premises.to_vec());

        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let dag_ref = dag.clone();
            let verifier = self.verifier.clone();
            let router = self.router.clone();
            let cfg = self.cfg.clone();
            let events = self.events.clone();
            let premises_ref = premises_arc.clone();
            let tid = target_id.clone();

            handles.push(tokio::spawn(async move {
                loop {
                    let work = select_next_work(&dag_ref, cfg.c_puct, node_budget).await;
                    match work {
                        NextWorkItem::Exhausted => break,
                        NextWorkItem::Wait => {
                            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                        }
                        NextWorkItem::ExpandNode {
                            node_id,
                            snapshot,
                            path,
                        } => {
                            let proposals = router
                                .propose_tactics(
                                    &snapshot.target.decl_signature,
                                    &snapshot.tactic_prefix,
                                    &snapshot.open_goals,
                                    &premises_ref,
                                    cfg.tactic_samples,
                                )
                                .await
                                .unwrap_or_default();

                            let mut guard = dag_ref.lock().await;
                            guard.nodes[node_id].expanding_in_flight = false;
                            guard.nodes[node_id].expanded = true;
                            guard.nodes_expanded += 1;

                            if proposals.is_empty() {
                                guard.backup_value(&path, 0.0);
                            } else {
                                guard.nodes[node_id].edges = proposals
                                    .into_iter()
                                    .map(|p| TacticEdge {
                                        tactic: p.tactic,
                                        prior: p.prior,
                                        visits: 0,
                                        total_value: 0.0,
                                        in_flight: 0,
                                        target_state: None,
                                        rejected: false,
                                    })
                                    .collect();
                                let val = heuristic_state_value(&snapshot.open_goals);
                                guard.backup_value(&path, val);
                            }

                            let expanded = guard.nodes_expanded;
                            let unique = guard.nodes.len();
                            let best_open = guard.best_open_goals;
                            drop(guard);

                            (events)(SearchEvent::NodeExpanded {
                                target_id: tid.clone(),
                                nodes_expanded: expanded,
                                dag_unique_states: unique,
                                best_open_goals: best_open,
                            });
                        }
                        NextWorkItem::EvaluateEdge {
                            parent_id,
                            edge_idx,
                            snapshot,
                            tactic,
                            path,
                        } => {
                            let step_res = verifier.apply_tactic(&snapshot, &tactic).await;
                            match step_res {
                                Ok(step) if step.accepted => {
                                    let next_snap = step.next_state.unwrap_or_else(|| {
                                        let mut pfx = snapshot.tactic_prefix.clone();
                                        pfx.push(tactic.clone());
                                        ProofStateSnapshot {
                                            target: snapshot.target.clone(),
                                            tactic_prefix: pfx,
                                            open_goals: step.remaining_goals.clone(),
                                        }
                                    });
                                    // Record clean suggested replacement if `exact?` / `simp?` emitted one.
                                    let clean_tactic = step
                                        .suggested_tactic
                                        .clone()
                                        .unwrap_or_else(|| tactic.clone());

                                    let (child_id, is_brand_new) = {
                                        let mut guard = dag_ref.lock().await;
                                        guard.nodes[parent_id].edges[edge_idx].tactic =
                                            clean_tactic;
                                        let (cid, is_new) =
                                            guard.intern_state(next_snap.clone(), (parent_id, edge_idx));
                                        guard.nodes[parent_id].edges[edge_idx].target_state =
                                            Some(cid);
                                        (cid, is_new)
                                    };

                                    // Run zero-cost hammers immediately on any newly discovered non-empty goal state!
                                    if is_brand_new
                                        && !next_snap.open_goals.is_empty()
                                        && cfg.enable_hammers
                                        && let Some(win_hammer) =
                                            try_hammer_cascade(&*verifier, &next_snap).await
                                    {
                                        (events)(SearchEvent::HammerHit {
                                            target_id: tid.clone(),
                                            tactic: win_hammer.clone(),
                                        });
                                        let mut guard = dag_ref.lock().await;
                                        guard.mark_solved_and_propagate(
                                            child_id,
                                            vec![win_hammer],
                                        );
                                        guard.backup_value(&path, 1.0);
                                        continue;
                                    }

                                    let val = if next_snap.open_goals.is_empty() {
                                        1.0
                                    } else {
                                        heuristic_state_value(&next_snap.open_goals)
                                    };
                                    let mut guard = dag_ref.lock().await;
                                    guard.backup_value(&path, val);
                                }
                                _ => {
                                    let mut guard = dag_ref.lock().await;
                                    guard.nodes[parent_id].edges[edge_idx].rejected = true;
                                    guard.backup_value(&path, 0.0);
                                }
                            }
                        }
                    }
                }
            }));
        }

        for h in handles {
            let _ = h.await;
        }

        let guard = dag.lock().await;
        let proof_script = guard.root_solution();
        let stuck_goals = guard
            .nodes
            .iter()
            .filter(|n| n.solved_tail.is_none() && !n.snapshot.open_goals.is_empty())
            .min_by_key(|n| n.snapshot.open_goals.len())
            .map(|n| n.snapshot.open_goals.clone())
            .unwrap_or_default();

        Ok(McgsOutcome {
            proof_script,
            nodes_expanded: guard.nodes_expanded,
            unique_states: guard.nodes.len(),
            stuck_goals,
        })
    }
}

fn heuristic_state_value(open_goals: &[NormalizedGoal]) -> f64 {
    if open_goals.is_empty() {
        1.0
    } else {
        1.0 / (1.0 + 0.45 * (open_goals.len() as f64))
    }
}

async fn select_next_work(
    dag: &Arc<Mutex<ProofDag>>,
    c_puct: f64,
    node_budget: usize,
) -> NextWorkItem {
    let mut guard = dag.lock().await;
    if guard.root_solution().is_some() || guard.nodes_expanded >= node_budget {
        return NextWorkItem::Exhausted;
    }

    let mut current_id: StateId = 0;
    let mut visited_on_path: HashSet<StateId> = HashSet::new();
    let mut path: Vec<(StateId, Option<usize>)> = Vec::new();
    let mut saw_in_flight = false;

    loop {
        visited_on_path.insert(current_id);
        let node = &guard.nodes[current_id];

        if node.solved_tail.is_some() {
            return NextWorkItem::Exhausted;
        }

        if !node.expanded {
            if node.expanding_in_flight {
                return NextWorkItem::Wait;
            }
            let snap = node.snapshot.clone();
            guard.nodes[current_id].expanding_in_flight = true;
            path.push((current_id, None));
            return NextWorkItem::ExpandNode {
                node_id: current_id,
                snapshot: snap,
                path,
            };
        }

        // Select best non-rejected, cycle-free edge via PUCT.
        let parent_visits = (node.visits.max(1)) as f64;
        let mut best_edge: Option<(usize, f64)> = None;

        for (idx, edge) in node.edges.iter().enumerate() {
            if edge.rejected {
                continue;
            }
            if let Some(cid) = edge.target_state {
                if visited_on_path.contains(&cid) {
                    continue;
                }
                if guard.nodes[cid].expanding_in_flight {
                    saw_in_flight = true;
                }
            }
            if edge.target_state.is_none() && edge.in_flight > 0 {
                saw_in_flight = true;
                continue;
            }

            let denom = (edge.visits + edge.in_flight) as f64;
            let q = if denom > 0.0 {
                edge.total_value / denom
            } else {
                0.5
            };
            let u = c_puct * edge.prior * parent_visits.sqrt() / (1.0 + denom);
            let score = q + u;
            if best_edge.map(|(_, bs)| score > bs).unwrap_or(true) {
                best_edge = Some((idx, score));
            }
        }

        let Some((chosen_idx, _)) = best_edge else {
            if saw_in_flight {
                return NextWorkItem::Wait;
            }
            // Mark current node dead-end if not root; if root, we are completely exhausted.
            if let Some(&(prev_node, Some(prev_edge))) = path.last() {
                guard.nodes[prev_node].edges[prev_edge].rejected = true;
                return NextWorkItem::Wait;
            }
            return NextWorkItem::Exhausted;
        };

        guard.nodes[current_id].edges[chosen_idx].in_flight += 1;
        path.push((current_id, Some(chosen_idx)));

        if let Some(child_id) = guard.nodes[current_id].edges[chosen_idx].target_state {
            current_id = child_id;
        } else {
            let snap = guard.nodes[current_id].snapshot.clone();
            let tactic = guard.nodes[current_id].edges[chosen_idx].tactic.clone();
            return NextWorkItem::EvaluateEdge {
                parent_id: current_id,
                edge_idx: chosen_idx,
                snapshot: snap,
                tactic,
                path,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use juice_core::{
        AuditReport, AxiomPolicy, CounterexampleResult, DiagnosticMsg, ProofContext, ProofTarget,
        SketchValidation, Span, TacticStepResult, VacuityCheck, events::noop_event_sink,
    };
    use juice_llm::{CandidateOutput, CompletionRequest, LlmProvider};
    use std::path::{Path, PathBuf};

    struct MockDiamondVerifier;

    #[async_trait]
    impl VerifierBackend for MockDiamondVerifier {
        fn backend_name(&self) -> &'static str {
            "mock"
        }
        async fn discover_targets(&self, _path: &Path) -> anyhow::Result<Vec<ProofTarget>> {
            Ok(vec![])
        }
        async fn initial_state(&self, target: &ProofTarget) -> anyhow::Result<ProofStateSnapshot> {
            Ok(ProofStateSnapshot {
                target: target.clone(),
                tactic_prefix: vec![],
                open_goals: vec![NormalizedGoal::new("⊢ A ∧ B")],
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
            let cur = state
                .open_goals
                .first()
                .map(|g| g.canonical_pp.as_str())
                .unwrap_or("");
            // Two distinct tactics (`split_left` and `split_alt`) converge to the SAME goal `⊢ SubGoal`,
            // testing MCGS DAG state deduplication; then `omega` hammer closes `⊢ SubGoal`!
            let next_goals = match (cur, tactic) {
                ("⊢ A ∧ B", "split_left") | ("⊢ A ∧ B", "split_alt") => {
                    Some(vec![NormalizedGoal::new("⊢ SubGoal")])
                }
                ("⊢ SubGoal", "omega") => Some(vec![]),
                _ => None,
            };

            if let Some(rem) = next_goals {
                let mut pfx = state.tactic_prefix.clone();
                pfx.push(tactic.to_string());
                Ok(TacticStepResult {
                    accepted: true,
                    next_state: Some(ProofStateSnapshot {
                        target: state.target.clone(),
                        tactic_prefix: pfx,
                        open_goals: rem.clone(),
                    }),
                    remaining_goals: rem,
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
            _decl_names: &[String],
            _policy: &AxiomPolicy,
        ) -> anyhow::Result<AuditReport> {
            unreachable!()
        }
    }

    struct MockTacticLlm;

    #[async_trait]
    impl LlmProvider for MockTacticLlm {
        fn provider_name(&self) -> &str {
            "mock"
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        async fn complete(&self, _req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>> {
            Ok(vec![CandidateOutput {
                text: r#"{"tactics": ["bad_tactic", "split_left", "split_alt"]}"#.to_string(),
                log_prob: None,
            }])
        }
    }

    #[tokio::test]
    async fn mcgs_expands_merges_states_and_closes_via_hammer() {
        let verifier = Arc::new(MockDiamondVerifier);
        let llm = Arc::new(MockTacticLlm);
        let router = ModelRouter::from_providers(llm.clone(), llm);
        let cfg = SearchConfig {
            concurrency: 2,
            ..Default::default()
        };
        let engine = McgsEngine::new(verifier.clone(), router, cfg, noop_event_sink());

        let dummy_target = ProofTarget {
            id: "test#0".to_string(),
            decl_name: "demo".to_string(),
            decl_kind: "theorem".to_string(),
            decl_signature: "theorem demo : A ∧ B".to_string(),
            full_source: "theorem demo : A ∧ B := by sorry".to_string(),
            sorry_span: Span {
                start_byte: 26,
                end_byte: 31,
                start_line: 0,
                start_col_utf16: 26,
                end_line: 0,
                end_col_utf16: 31,
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
        let root_snap = verifier.initial_state(&dummy_target).await.unwrap();
        let outcome = engine.search(root_snap, &[], 16).await.expect("mcgs runs");
        let proof = outcome.proof_script.expect("solved!");
        assert!(proof.contains("omega"));
        assert!(proof.contains("split_left") || proof.contains("split_alt"));
    }
}
