use anyhow::{Context, bail};
use async_trait::async_trait;
use juice_core::{ModelEndpointConfig, ProviderKind};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: MessageRole,
    pub content: String,
}

impl LlmMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub system: Option<String>,
    pub messages: Vec<LlmMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateOutput {
    pub text: String,
    /// Unnormalized log-likelihood or heuristic confidence score (higher is more likely).
    pub log_prob: Option<f64>,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
    async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>>;
}

pub fn build_provider(cfg: &ModelEndpointConfig) -> Arc<dyn LlmProvider> {
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| Client::new());

    match cfg.provider {
        ProviderKind::Anthropic => Arc::new(AnthropicClient {
            client,
            cfg: cfg.clone(),
        }),
        ProviderKind::Gemini => Arc::new(GeminiClient {
            client,
            cfg: cfg.clone(),
        }),
        ProviderKind::Openai | ProviderKind::OpenaiCompat => Arc::new(OpenAiCompatClient {
            client,
            cfg: cfg.clone(),
        }),
    }
}

pub struct AnthropicClient {
    client: Client,
    cfg: ModelEndpointConfig,
}

#[async_trait]
impl LlmProvider for AnthropicClient {
    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn model_name(&self) -> &str {
        &self.cfg.model
    }

    async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>> {
        let api_key = self.cfg.resolve_api_key().context(
            "Missing Anthropic API key. Set ANTHROPIC_API_KEY or configure api_key_env in juice.toml.",
        )?;
        let base = self
            .cfg
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com")
            .trim_end_matches('/');
        let url = format!("{base}/v1/messages");

        let messages: Vec<Value> = req
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                };
                json!({ "role": role, "content": m.content })
            })
            .collect();

        let mut body = json!({
            "model": self.cfg.model,
            "max_tokens": req.max_tokens.max(self.cfg.max_tokens),
            "temperature": req.temperature,
            "messages": messages,
        });

        if let Some(sys) = &req.system {
            body["system"] = json!(sys);
        }
        if let Some(budget) = self.cfg.thinking_budget {
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": budget,
            });
            // Anthropic requires temperature = 1 when extended thinking is enabled.
            body["temperature"] = json!(1.0);
        }

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .context("Failed to connect to Anthropic API")?;

        let status = resp.status();
        let payload: Value = resp.json().await.context("Invalid Anthropic JSON response")?;
        if !status.is_success() {
            bail!("Anthropic API error ({status}): {payload}");
        }

        let mut text_parts = Vec::new();
        if let Some(arr) = payload.get("content").and_then(Value::as_array) {
            for block in arr {
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(t) = block.get("text").and_then(Value::as_str)
                {
                    text_parts.push(t.to_string());
                }
            }
        }

        Ok(vec![CandidateOutput {
            text: text_parts.join("\n"),
            log_prob: None,
        }])
    }
}

pub struct OpenAiCompatClient {
    client: Client,
    cfg: ModelEndpointConfig,
}

#[async_trait]
impl LlmProvider for OpenAiCompatClient {
    fn provider_name(&self) -> &str {
        match self.cfg.provider {
            ProviderKind::Openai => "openai",
            _ => "openai-compat",
        }
    }

    fn model_name(&self) -> &str {
        &self.cfg.model
    }

    async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>> {
        let base = self
            .cfg
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1")
            .trim_end_matches('/');
        let url = format!("{base}/chat/completions");

        let api_key = self.cfg.resolve_api_key();
        if self.cfg.provider == ProviderKind::Openai && api_key.is_none() {
            bail!(
                "Missing OpenAI API key. Set OPENAI_API_KEY or configure [models.planner] in juice.toml."
            );
        }

        let mut messages = Vec::new();
        if let Some(sys) = &req.system {
            messages.push(json!({ "role": "system", "content": sys }));
        }
        for m in &req.messages {
            let role = match m.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            };
            messages.push(json!({ "role": role, "content": m.content }));
        }

        let body = json!({
            "model": self.cfg.model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "n": req.n.max(1),
        });

        let mut builder = self.client.post(&url).json(&body);
        if let Some(key) = &api_key {
            builder = builder.bearer_auth(key);
        }

        let resp = builder
            .send()
            .await
            .with_context(|| format!("Failed to reach OpenAI-compatible endpoint at {url}"))?;

        let status = resp.status();
        let payload: Value = resp
            .json()
            .await
            .context("Invalid JSON from OpenAI-compatible endpoint")?;

        if !status.is_success() {
            bail!("OpenAI-compatible API error ({status}) at {url}: {payload}");
        }

        let mut outputs = Vec::new();
        if let Some(choices) = payload.get("choices").and_then(Value::as_array) {
            for choice in choices {
                let text = choice
                    .pointer("/message/content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let log_prob = extract_openai_avg_logprob(choice);
                if !text.trim().is_empty() {
                    outputs.push(CandidateOutput { text, log_prob });
                }
            }
        }

        Ok(outputs)
    }
}

fn extract_openai_avg_logprob(choice: &Value) -> Option<f64> {
    let tokens = choice.pointer("/logprobs/content")?.as_array()?;
    if tokens.is_empty() {
        return None;
    }
    let sum: f64 = tokens
        .iter()
        .filter_map(|t| t.get("logprob").and_then(Value::as_f64))
        .sum();
    Some(sum / (tokens.len() as f64))
}

pub struct GeminiClient {
    client: Client,
    cfg: ModelEndpointConfig,
}

#[async_trait]
impl LlmProvider for GeminiClient {
    fn provider_name(&self) -> &str {
        "gemini"
    }

    fn model_name(&self) -> &str {
        &self.cfg.model
    }

    async fn complete(&self, req: &CompletionRequest) -> anyhow::Result<Vec<CandidateOutput>> {
        let api_key = self.cfg.resolve_api_key().context(
            "Missing Gemini API key. Set GEMINI_API_KEY or configure api_key_env in juice.toml.",
        )?;
        let base = self
            .cfg
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta")
            .trim_end_matches('/');
        let url = format!("{base}/models/{}:generateContent", self.cfg.model);

        let contents: Vec<Value> = req
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "model",
                };
                json!({
                    "role": role,
                    "parts": [{ "text": m.content }]
                })
            })
            .collect();

        let mut body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": req.temperature,
                "maxOutputTokens": req.max_tokens.max(self.cfg.max_tokens),
            }
        });

        if let Some(sys) = &req.system {
            body["systemInstruction"] = json!({
                "parts": [{ "text": sys }]
            });
        }

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to connect to Gemini API")?;

        let status = resp.status();
        let payload: Value = resp.json().await.context("Invalid Gemini JSON response")?;
        if !status.is_success() {
            bail!("Gemini API error ({status}): {payload}");
        }

        let mut outputs = Vec::new();
        if let Some(candidates) = payload.get("candidates").and_then(Value::as_array) {
            for cand in candidates {
                let mut parts_text = Vec::new();
                if let Some(parts) = cand.pointer("/content/parts").and_then(Value::as_array) {
                    for part in parts {
                        // Skip internal thought parts if flagged
                        if part.get("thought").and_then(Value::as_bool) == Some(true) {
                            continue;
                        }
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            parts_text.push(t.to_string());
                        }
                    }
                }
                let joined = parts_text.join("\n");
                if !joined.trim().is_empty() {
                    outputs.push(CandidateOutput {
                        text: joined,
                        log_prob: cand.get("avgLogprobs").and_then(Value::as_f64),
                    });
                }
            }
        }

        Ok(outputs)
    }
}
