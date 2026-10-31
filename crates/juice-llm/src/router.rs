use crate::prompts::{
    AutoformalizedClaim, BackTranslationVerdict, PlannedSketch, ProposedTactic,
    build_autoformalize_prompt, build_back_translation_prompt, build_planner_sketch_prompt,
    build_tactic_sampling_prompt, extract_json_payload, parse_tactic_candidates,
};
use crate::provider::{CompletionRequest, LlmMessage, LlmProvider, build_provider};
use juice_core::{ModelsConfig, NormalizedGoal, PremiseHit};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct ModelRouter {
    planner: Arc<dyn LlmProvider>,
    prover: Arc<dyn LlmProvider>,
    planner_temp: f32,
    prover_temp: f32,
}

impl ModelRouter {
    pub fn from_config(cfg: &ModelsConfig) -> Self {
        let planner = build_provider(&cfg.planner);
        let prover_cfg = cfg.effective_prover();
        let prover = build_provider(prover_cfg);
        Self {
            planner,
            prover,
            planner_temp: cfg.planner.temperature,
            prover_temp: prover_cfg.temperature,
        }
    }

    pub fn from_providers(
        planner: Arc<dyn LlmProvider>,
        prover: Arc<dyn LlmProvider>,
    ) -> Self {
        Self {
            planner,
            prover,
            planner_temp: 0.3,
            prover_temp: 0.6,
        }
    }

    pub fn planner_summary(&self) -> String {
        format!("{}:{}", self.planner.provider_name(), self.planner.model_name())
    }

    pub fn prover_summary(&self) -> String {
        format!("{}:{}", self.prover.provider_name(), self.prover.model_name())
    }

    pub async fn complete_planner_text(
        &self,
        system: &str,
        user: &str,
    ) -> anyhow::Result<String> {
        let req = CompletionRequest {
            system: Some(system.to_string()),
            messages: vec![LlmMessage::user(user.to_string())],
            temperature: self.planner_temp,
            max_tokens: 4096,
            n: 1,
        };
        let outputs = self.planner.complete(&req).await?;
        let first = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("Planner returned empty completion"))?;
        Ok(first.text.clone())
    }

    pub async fn complete_planner_json<T: serde::de::DeserializeOwned>(
        &self,
        system: &str,
        user: &str,
    ) -> anyhow::Result<T> {
        let text = self.complete_planner_text(system, user).await?;
        extract_json_payload::<T>(&text)
    }

    pub async fn autoformalize_claim(
        &self,
        claim_text: &str,
        has_mathlib: bool,
        premises: &[PremiseHit],
        prior_feedback: Option<&str>,
    ) -> anyhow::Result<AutoformalizedClaim> {
        let (system, user) =
            build_autoformalize_prompt(claim_text, has_mathlib, premises, prior_feedback);
        let req = CompletionRequest {
            system: Some(system),
            messages: vec![LlmMessage::user(user)],
            temperature: self.planner_temp,
            max_tokens: 4096,
            n: 1,
        };
        let outputs = self.planner.complete(&req).await?;
        let first = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("Planner returned empty autoformalization output"))?;
        extract_json_payload::<AutoformalizedClaim>(&first.text)
    }

    pub async fn verify_back_translation(
        &self,
        original_claim: &str,
        definitions: &str,
        theorem_signature: &str,
    ) -> anyhow::Result<BackTranslationVerdict> {
        let (system, user) =
            build_back_translation_prompt(original_claim, definitions, theorem_signature);
        let req = CompletionRequest {
            system: Some(system),
            messages: vec![LlmMessage::user(user)],
            temperature: 0.1,
            max_tokens: 2048,
            n: 1,
        };
        let outputs = self.planner.complete(&req).await?;
        let first = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("Planner returned empty back-translation critique"))?;
        extract_json_payload::<BackTranslationVerdict>(&first.text)
    }

    pub async fn plan_lemma_sketch(
        &self,
        decl_signature: &str,
        open_goals: &[NormalizedGoal],
        premises: &[PremiseHit],
        refuted_lemmas: &[(String, String)],
        prior_errors: &[String],
    ) -> anyhow::Result<PlannedSketch> {
        let (system, user) = build_planner_sketch_prompt(
            decl_signature,
            open_goals,
            premises,
            refuted_lemmas,
            prior_errors,
        );
        let req = CompletionRequest {
            system: Some(system),
            messages: vec![LlmMessage::user(user)],
            temperature: self.planner_temp,
            max_tokens: 4096,
            n: 1,
        };
        let outputs = self.planner.complete(&req).await?;
        let first = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("Planner returned empty sketch response"))?;
        extract_json_payload::<PlannedSketch>(&first.text)
    }

    pub async fn propose_tactics(
        &self,
        decl_signature: &str,
        tactic_prefix: &[String],
        open_goals: &[NormalizedGoal],
        premises: &[PremiseHit],
        k: usize,
    ) -> anyhow::Result<Vec<ProposedTactic>> {
        let (system, user) =
            build_tactic_sampling_prompt(decl_signature, tactic_prefix, open_goals, premises, k);
        let req = CompletionRequest {
            system: Some(system),
            messages: vec![LlmMessage::user(user)],
            temperature: self.prover_temp,
            max_tokens: 1024,
            n: 1,
        };
        let outputs = self.prover.complete(&req).await?;

        let mut raw_scores: HashMap<String, f64> = HashMap::new();
        let mut order: Vec<String> = Vec::new();

        for cand in outputs {
            let parsed = parse_tactic_candidates(&cand.text);
            let base_lp = cand.log_prob.unwrap_or(0.0);
            for (rank, tactic) in parsed.into_iter().enumerate() {
                // Combine logprob (if provided by vLLM/OpenAI) with rank decay.
                let score = base_lp - (rank as f64) * 0.35;
                raw_scores
                    .entry(tactic.clone())
                    .and_modify(|existing| {
                        if score > *existing {
                            *existing = score;
                        }
                    })
                    .or_insert_with(|| {
                        order.push(tactic);
                        score
                    });
            }
        }

        if order.is_empty() {
            return Ok(Vec::new());
        }

        order.truncate(k);
        let max_score = order
            .iter()
            .filter_map(|t| raw_scores.get(t).copied())
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = order
            .iter()
            .map(|t| (raw_scores[t] - max_score).exp())
            .sum();

        let mut proposals: Vec<ProposedTactic> = order
            .into_iter()
            .map(|tactic| {
                let prior = (raw_scores[&tactic] - max_score).exp() / exp_sum;
                ProposedTactic { tactic, prior }
            })
            .collect();

        proposals.sort_by(|a, b| {
            b.prior
                .partial_cmp(&a.prior)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(proposals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::CandidateOutput;
    use async_trait::async_trait;

    struct StaticProvider {
        reply: String,
    }

    #[async_trait]
    impl LlmProvider for StaticProvider {
        fn provider_name(&self) -> &str {
            "mock"
        }
        fn model_name(&self) -> &str {
            "mock-prover"
        }
        async fn complete(&self, _req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>> {
            Ok(vec![CandidateOutput {
                text: self.reply.clone(),
                log_prob: Some(-0.2),
            }])
        }
    }

    #[tokio::test]
    async fn router_normalizes_tactic_priors_to_one() {
        let p = Arc::new(StaticProvider {
            reply: r#"{"tactics": ["intro h", "cases h", "simp"]}"#.to_string(),
        });
        let router = ModelRouter::from_providers(p.clone(), p);
        let res = router
            .propose_tactics(
                "theorem demo : P → P",
                &[],
                &[NormalizedGoal::new("⊢ P → P")],
                &[],
                4,
            )
            .await
            .expect("tactics");
        assert_eq!(res.len(), 3);
        let sum: f64 = res.iter().map(|t| t.prior).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(res[0].prior > res[1].prior);
        assert!(res[1].prior > res[2].prior);
    }
}
