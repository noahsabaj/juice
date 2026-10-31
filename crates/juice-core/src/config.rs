use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    Anthropic,
    Openai,
    Gemini,
    OpenaiCompat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEndpointConfig {
    pub provider: ProviderKind,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
}

fn default_temperature() -> f32 {
    0.4
}

fn default_max_tokens() -> u32 {
    4096
}

impl ModelEndpointConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(key) = &self.api_key
            && !key.trim().is_empty()
        {
            return Some(key.clone());
        }
        if let Some(env_name) = &self.api_key_env
            && let Ok(val) = std::env::var(env_name)
            && !val.trim().is_empty()
        {
            return Some(val);
        }
        let default_env = match self.provider {
            ProviderKind::Anthropic => Some("ANTHROPIC_API_KEY"),
            ProviderKind::Openai => Some("OPENAI_API_KEY"),
            ProviderKind::Gemini => Some("GEMINI_API_KEY"),
            ProviderKind::OpenaiCompat => Some("OPENAI_API_KEY"),
        };
        default_env
            .and_then(|name| std::env::var(name).ok())
            .filter(|v| !v.trim().is_empty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub planner: ModelEndpointConfig,
    #[serde(default)]
    pub prover: Option<ModelEndpointConfig>,
}

impl ModelsConfig {
    pub fn effective_prover(&self) -> &ModelEndpointConfig {
        self.prover.as_ref().unwrap_or(&self.planner)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_max_nodes")]
    pub max_nodes: usize,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_tactic_samples")]
    pub tactic_samples: usize,
    #[serde(default = "default_c_puct")]
    pub c_puct: f64,
    #[serde(default = "default_max_planner_rounds")]
    pub max_planner_rounds: usize,
    #[serde(default = "default_step_timeout_ms")]
    pub step_timeout_ms: u64,
    #[serde(default = "default_true")]
    pub enable_hammers: bool,
    #[serde(default = "default_true")]
    pub enable_counterexample_probe: bool,
    #[serde(default)]
    pub online_premise_search: bool,
}

fn default_max_nodes() -> usize {
    64
}
fn default_concurrency() -> usize {
    4
}
fn default_tactic_samples() -> usize {
    6
}
fn default_c_puct() -> f64 {
    1.4
}
fn default_max_planner_rounds() -> usize {
    3
}
fn default_step_timeout_ms() -> u64 {
    8000
}
fn default_true() -> bool {
    true
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_nodes: default_max_nodes(),
            concurrency: default_concurrency(),
            tactic_samples: default_tactic_samples(),
            c_puct: default_c_puct(),
            max_planner_rounds: default_max_planner_rounds(),
            step_timeout_ms: default_step_timeout_ms(),
            enable_hammers: true,
            enable_counterexample_probe: true,
            online_premise_search: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeanConfig {
    #[serde(default = "default_worker_pool_size")]
    pub worker_pool_size: usize,
    #[serde(default)]
    pub allow_native_decide: bool,
    #[serde(default)]
    pub mathlib_cache_dir: Option<PathBuf>,
}

fn default_worker_pool_size() -> usize {
    2
}

impl Default for LeanConfig {
    fn default() -> Self {
        Self {
            worker_pool_size: default_worker_pool_size(),
            allow_native_decide: false,
            mathlib_cache_dir: None,
        }
    }
}

impl LeanConfig {
    pub fn resolved_mathlib_cache_dir(&self) -> PathBuf {
        if let Some(dir) = &self.mathlib_cache_dir {
            return dir.clone();
        }
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".juice/cache"))
            .join("juice")
            .join("mathlib")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JuiceConfig {
    pub models: ModelsConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub lean: LeanConfig,
}

impl Default for JuiceConfig {
    fn default() -> Self {
        Self::detect_from_env()
    }
}

impl JuiceConfig {
    pub fn global_config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from(".config"))
            .join("juice")
            .join("config.toml")
    }

    pub fn detect_from_env() -> Self {
        let planner = if std::env::var("ANTHROPIC_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            ModelEndpointConfig {
                provider: ProviderKind::Anthropic,
                model: std::env::var("JUICE_PLANNER_MODEL")
                    .unwrap_or_else(|_| "claude-sonnet-4-5".to_string()),
                api_key: None,
                api_key_env: Some("ANTHROPIC_API_KEY".to_string()),
                base_url: None,
                temperature: 0.3,
                max_tokens: 4096,
                thinking_budget: None,
            }
        } else if std::env::var("GEMINI_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            ModelEndpointConfig {
                provider: ProviderKind::Gemini,
                model: std::env::var("JUICE_PLANNER_MODEL")
                    .unwrap_or_else(|_| "gemini-2.5-pro".to_string()),
                api_key: None,
                api_key_env: Some("GEMINI_API_KEY".to_string()),
                base_url: None,
                temperature: 0.3,
                max_tokens: 4096,
                thinking_budget: None,
            }
        } else if std::env::var("OPENAI_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            ModelEndpointConfig {
                provider: ProviderKind::Openai,
                model: std::env::var("JUICE_PLANNER_MODEL")
                    .unwrap_or_else(|_| "gpt-4.1".to_string()),
                api_key: None,
                api_key_env: Some("OPENAI_API_KEY".to_string()),
                base_url: None,
                temperature: 0.3,
                max_tokens: 4096,
                thinking_budget: None,
            }
        } else {
            let base_url = std::env::var("OLLAMA_HOST")
                .map(|host| format!("{}/v1", host.trim_end_matches('/')))
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
            ModelEndpointConfig {
                provider: ProviderKind::OpenaiCompat,
                model: std::env::var("JUICE_PLANNER_MODEL")
                    .unwrap_or_else(|_| "Goedel-Prover-V2-8B".to_string()),
                api_key: None,
                api_key_env: None,
                base_url: Some(base_url),
                temperature: 0.4,
                max_tokens: 4096,
                thinking_budget: None,
            }
        };

        let prover = std::env::var("JUICE_PROVER_MODEL").ok().map(|prover_model| {
            let base_url = std::env::var("JUICE_PROVER_BASE_URL").ok();
            let provider = if base_url.is_some() {
                ProviderKind::OpenaiCompat
            } else {
                planner.provider
            };
            ModelEndpointConfig {
                provider,
                model: prover_model,
                api_key: None,
                api_key_env: planner.api_key_env.clone(),
                base_url,
                temperature: 0.6,
                max_tokens: 1024,
                thinking_budget: None,
            }
        });

        Self {
            models: ModelsConfig { planner, prover },
            search: SearchConfig::default(),
            lean: LeanConfig::default(),
        }
    }

    pub fn load(workspace_dir: Option<&Path>) -> anyhow::Result<Self> {
        if let Some(dir) = workspace_dir {
            let local_toml = dir.join("juice.toml");
            if local_toml.exists() {
                let raw = std::fs::read_to_string(&local_toml)?;
                return Ok(toml::from_str(&raw)?);
            }
        }

        let global_toml = Self::global_config_path();
        if global_toml.exists() {
            let raw = std::fs::read_to_string(&global_toml)?;
            return Ok(toml::from_str(&raw)?);
        }

        Ok(Self::detect_from_env())
    }

    pub fn sample_toml() -> &'static str {
        r#"# juice configuration file (MIT OR Apache-2.0)

[models.planner]
provider = "anthropic" # Options: "anthropic", "openai", "gemini", "openai-compat"
model = "claude-sonnet-4-5"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.3
max_tokens = 4096

# Optional dedicated Lean 4 tactic prover (falls back to [models.planner] if omitted)
# [models.prover]
# provider = "openai-compat"
# model = "Goedel-LM/Goedel-Prover-V2-8B"
# base_url = "http://localhost:11434/v1"
# temperature = 0.6
# max_tokens = 1024

[search]
max_nodes = 64
concurrency = 4
tactic_samples = 6
c_puct = 1.4
max_planner_rounds = 3
step_timeout_ms = 8000
enable_hammers = true
enable_counterexample_probe = true
online_premise_search = false

[lean]
worker_pool_size = 2
allow_native_decide = false
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_toml_round_trips() {
        let cfg: JuiceConfig = toml::from_str(JuiceConfig::sample_toml()).expect("valid sample");
        assert_eq!(cfg.models.planner.provider, ProviderKind::Anthropic);
        assert!(cfg.models.prover.is_none());
        assert_eq!(cfg.models.effective_prover().model, "claude-sonnet-4-5");
        assert_eq!(cfg.search.max_nodes, 64);
        assert!(!cfg.lean.allow_native_decide);
    }

    #[test]
    fn dual_role_routing_parses_prover_override() {
        let raw = r#"
[models.planner]
provider = "gemini"
model = "gemini-2.5-pro"

[models.prover]
provider = "openai-compat"
model = "deepseek-prover-v2"
base_url = "https://api.fireworks.ai/inference/v1"
"#;
        let cfg: JuiceConfig = toml::from_str(raw).expect("parses dual role");
        assert_eq!(cfg.models.planner.provider, ProviderKind::Gemini);
        assert_eq!(
            cfg.models.effective_prover().provider,
            ProviderKind::OpenaiCompat
        );
        assert_eq!(cfg.models.effective_prover().model, "deepseek-prover-v2");
    }
}
