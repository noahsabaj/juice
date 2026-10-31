mod ui;

use crate::ui::{TerminalProgress, render_unified_diff};
use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use juice_core::{AxiomPolicy, JuiceConfig, ProofContext, VerifierBackend};
use juice_lean::{
    LeanBackend,
    workspace::{resolve_workspace, setup_standalone_workspace},
};
use juice_llm::ModelRouter;
use juice_search::{AutoformalizeOptions, Autoformalizer, TwoTierProver};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

#[derive(Debug, Parser)]
#[command(
    name = "juice",
    version,
    about = "Open-source formal reasoning, autoformalization, and Lean 4 theorem-proving CLI (MIT OR Apache-2.0)"
)]
pub struct Cli {
    /// Emit machine-readable JSON output
    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Discover and prove `sorry` placeholders in a Lean 4 file or Lake project
    Fill {
        /// Target `.lean` file or project directory (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Write verified proofs back to source files in-place
        #[arg(long)]
        in_place: bool,

        /// Print unified git-style diff of verified proof replacements
        #[arg(long)]
        diff: bool,

        /// Filter to declarations matching this substring
        #[arg(long)]
        target: Option<String>,

        /// Maximum MCGS node expansions per goal
        #[arg(long)]
        budget: Option<usize>,

        /// Number of concurrent search workers
        #[arg(long)]
        concurrency: Option<usize>,

        /// Resume a specific checkpointed run ID from `.juice/runs/`
        #[arg(long)]
        resume: Option<String>,

        /// Permit `Lean.trustCompiler` / `native_decide` in cold axiom audit
        #[arg(long)]
        allow_native_decide: bool,
    },

    /// Autoformalize a natural-language math claim, Markdown note, or LaTeX (`.tex`) paper into Lean 4
    Formalize {
        /// Path to `.tex` / `.md` / `.txt` file, or inline mathematical statement
        input: String,

        /// Write generated Lean 4 code to this `.lean` file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Only formalize and typecheck statements (leave proofs as `sorry`)
        #[arg(long)]
        statements_only: bool,

        /// Maximum MCGS node expansions per theorem
        #[arg(long)]
        budget: Option<usize>,

        /// Permit `Lean.trustCompiler` / `native_decide` in cold axiom audit
        #[arg(long)]
        allow_native_decide: bool,
    },

    /// Shorthand to autoformalize and prove a standalone mathematical claim
    Prove {
        /// Mathematical claim in natural language or LaTeX
        claim: String,

        /// Optional `.lean` output file to write and cold-audit
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Maximum MCGS node expansions
        #[arg(long)]
        budget: Option<usize>,

        /// Permit `Lean.trustCompiler` / `native_decide` in cold axiom audit
        #[arg(long)]
        allow_native_decide: bool,
    },

    /// Run a cold out-of-process `lake env lean` + `#print axioms` anti-cheat verification audit
    Audit {
        /// Path to `.lean` file to certify
        path: PathBuf,

        /// Specific declaration names to check (checks all top-level theorems/defs if omitted)
        #[arg(long = "decl")]
        decls: Vec<String>,

        /// Permit `Lean.trustCompiler` / `native_decide` axioms
        #[arg(long)]
        allow_native_decide: bool,
    },

    /// Search verified local project and `Mathlib` theorem declarations
    Search {
        /// Keywords or mathematical signature tokens
        query: String,

        /// Optional workspace path (defaults to current directory or cached Mathlib)
        #[arg(long)]
        path: Option<PathBuf>,

        /// Maximum number of premise hits to return
        #[arg(short, long, default_value_t = 10)]
        limit: usize,

        /// Enable online Loogle (`loogle.lean-lang.org`) fallback
        #[arg(long)]
        online_search: bool,
    },

    /// Bootstrap `elan`, Lean 4, and the shared standalone `~/.cache/juice/mathlib` workspace
    Setup {
        /// Skip `Mathlib` download and create a minimal Lean 4 core/Std workspace
        #[arg(long)]
        no_mathlib: bool,

        /// Automatically install `elan` without prompting if Lean 4 is missing
        #[arg(short, long)]
        yes: bool,
    },

    /// Check Lean 4 toolchain, `Mathlib` cache, and configured LLM providers
    Doctor,

    /// Inspect or initialize `juice` configuration (`~/.config/juice/config.toml`)
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Run built-in Model Context Protocol (`stdio` JSON-RPC 2.0) server for AI coding agents
    Mcp,
}

#[derive(Debug, Subcommand)]
pub enum ConfigAction {
    /// Write a starter `config.toml` file
    Init {
        /// Write `juice.toml` in the current directory instead of `~/.config/juice/config.toml`
        #[arg(long)]
        local: bool,
        /// Overwrite existing config file
        #[arg(long)]
        force: bool,
    },
    /// Show resolved active configuration
    Show,
    /// Print global config file path
    Path,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let json_mode = cli.json;

    match run_cli(cli).await {
        Ok(code) => code,
        Err(err) => {
            if json_mode {
                let payload = serde_json::json!({
                    "ok": false,
                    "error": format!("{err:#}"),
                });
                println!("{}", serde_json::to_string_pretty(&payload).unwrap_or_default());
            } else {
                eprintln!("error: {err:#}");
            }
            ExitCode::FAILURE
        }
    }
}

async fn run_cli(cli: Cli) -> anyhow::Result<ExitCode> {
    let cwd = std::env::current_dir().ok();
    let mut cfg = JuiceConfig::load(cwd.as_deref())?;

    match cli.command {
        Commands::Fill {
            path,
            in_place,
            diff,
            target,
            budget,
            concurrency,
            resume,
            allow_native_decide,
        } => {
            if let Some(b) = budget {
                cfg.search.max_nodes = b;
            }
            if let Some(c) = concurrency {
                cfg.search.concurrency = c;
            }
            let policy = AxiomPolicy::strict()
                .with_native_decide(allow_native_decide || cfg.lean.allow_native_decide);

            let (progress, sink) = TerminalProgress::new(cli.json);
            let verifier: Arc<dyn VerifierBackend> =
                Arc::new(LeanBackend::new(cfg.lean.clone(), cfg.search.clone()));
            let router = ModelRouter::from_config(&cfg.models);
            let prover = TwoTierProver::new(verifier, router, cfg.search.clone(), sink);

            let outcome = prover
                .fill_path(
                    &path,
                    target.as_deref(),
                    resume.as_deref(),
                    &policy,
                    in_place,
                )
                .await?;
            progress.finish();

            let total = outcome.reports.len();
            let solved = outcome.reports.iter().filter(|r| r.solved).count();

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&outcome)?);
            } else if total == 0 {
                println!("No `sorry` placeholders found in {}.", path.display());
            } else {
                println!(
                    "\nSummary: solved {solved}/{total} `sorry` target(s) (run_id: {}, checkpoint: {})",
                    outcome.run_id,
                    outcome.checkpoint_path.display()
                );
                if diff || !in_place {
                    for (file_path, patched) in &outcome.patched_files {
                        if let Some(orig) = outcome.original_files.get(file_path) {
                            println!("\n{}", render_unified_diff(file_path, orig, patched));
                        }
                    }
                }
                if in_place && !outcome.patched_files.is_empty() {
                    println!(
                        "Patched {} file(s) in-place.",
                        outcome.patched_files.len()
                    );
                }
            }

            Ok(if solved == total {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }

        Commands::Formalize {
            input,
            output,
            statements_only,
            budget,
            allow_native_decide,
        } => {
            if let Some(b) = budget {
                cfg.search.max_nodes = b;
            }
            let raw_text = if Path::new(&input).is_file() {
                std::fs::read_to_string(&input)
                    .with_context(|| format!("Failed to read input file `{input}`"))?
            } else {
                input
            };

            let (progress, sink) = TerminalProgress::new(cli.json);
            let verifier: Arc<dyn VerifierBackend> =
                Arc::new(LeanBackend::new(cfg.lean.clone(), cfg.search.clone()));
            let router = ModelRouter::from_config(&cfg.models);
            let formalizer = Autoformalizer::new(
                verifier,
                router,
                cfg.search.clone(),
                cfg.lean.clone(),
                sink,
            );
            let opts = AutoformalizeOptions {
                prove_theorems: !statements_only,
                max_formalize_attempts: 4,
                policy: AxiomPolicy::strict()
                    .with_native_decide(allow_native_decide || cfg.lean.allow_native_decide),
            };

            let outcome = formalizer
                .formalize_input(&raw_text, output.as_deref(), &opts)
                .await?;
            progress.finish();

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&outcome)?);
            } else {
                println!("\n-- Generated & Verified Lean 4 --\n{}", outcome.lean_source);
                if let Some(out_path) = &output {
                    println!("Wrote output to {}", out_path.display());
                }
            }

            let all_ok = statements_only || outcome.items.iter().all(|i| i.proved);
            Ok(if all_ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }

        Commands::Prove {
            claim,
            output,
            budget,
            allow_native_decide,
        } => {
            if let Some(b) = budget {
                cfg.search.max_nodes = b;
            }
            let (progress, sink) = TerminalProgress::new(cli.json);
            let verifier: Arc<dyn VerifierBackend> =
                Arc::new(LeanBackend::new(cfg.lean.clone(), cfg.search.clone()));
            let router = ModelRouter::from_config(&cfg.models);
            let formalizer = Autoformalizer::new(
                verifier,
                router,
                cfg.search.clone(),
                cfg.lean.clone(),
                sink,
            );
            let opts = AutoformalizeOptions {
                prove_theorems: true,
                max_formalize_attempts: 4,
                policy: AxiomPolicy::strict()
                    .with_native_decide(allow_native_decide || cfg.lean.allow_native_decide),
            };

            let outcome = formalizer
                .formalize_input(&claim, output.as_deref(), &opts)
                .await?;
            progress.finish();

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&outcome)?);
            } else {
                println!("\n{}", outcome.lean_source);
            }

            let all_proved = outcome.items.iter().all(|i| i.proved);
            Ok(if all_proved {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }

        Commands::Audit {
            path,
            decls,
            allow_native_decide,
        } => {
            let policy = AxiomPolicy::strict()
                .with_native_decide(allow_native_decide || cfg.lean.allow_native_decide);
            let verifier = LeanBackend::new(cfg.lean.clone(), cfg.search.clone());
            let report = verifier.cold_audit(&path, &decls, &policy).await?;

            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else if report.verified {
                println!(
                    "[VERIFIED ✓] {} passed cold Lean 4 kernel & axiom audit.",
                    path.display()
                );
                println!(
                    "  Declarations checked : {}",
                    if report.declarations_checked.is_empty() {
                        "(module compilation)".to_string()
                    } else {
                        report.declarations_checked.join(", ")
                    }
                );
                println!(
                    "  Axioms used          : {}",
                    if report.axioms_used.is_empty() {
                        "none".to_string()
                    } else {
                        report.axioms_used.join(", ")
                    }
                );
            } else {
                eprintln!(
                    "[REJECTED ✗] {} failed cold kernel audit!",
                    path.display()
                );
                if report.has_sorry {
                    eprintln!("  - File still depends on `sorry` / `sorryAx`");
                }
                if !report.forbidden_axioms.is_empty() {
                    eprintln!(
                        "  - Forbidden axioms used: {}",
                        report.forbidden_axioms.join(", ")
                    );
                }
                if !report.forbidden_constructs.is_empty() {
                    eprintln!(
                        "  - Forbidden constructs: {}",
                        report.forbidden_constructs.join(", ")
                    );
                }
                if !report.raw_output.is_empty() {
                    eprintln!("\nCompiler output:\n{}", report.raw_output);
                }
            }

            Ok(if report.verified {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }

        Commands::Search {
            query,
            path,
            limit,
            online_search,
        } => {
            if online_search {
                cfg.search.online_premise_search = true;
            }
            let ws = resolve_workspace(path.as_deref(), &cfg.lean);
            let verifier = LeanBackend::new(cfg.lean.clone(), cfg.search.clone());
            let ctx = ProofContext {
                workspace_root: ws.root.clone(),
                source_file: path,
                imports_header: if ws.has_mathlib {
                    "import Mathlib".to_string()
                } else {
                    String::new()
                },
                preamble: String::new(),
            };
            let hits = verifier.search_premises(&ctx, &query, limit).await?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else if hits.is_empty() {
                println!(
                    "No matching declarations found in {} (tip: run `juice setup` for Mathlib or pass `--online-search`).",
                    ws.root.display()
                );
            } else {
                for h in &hits {
                    let badge = if h.verified_by_kernel { "[#check ✓]" } else { "[indexed]" };
                    println!("{badge} {} ({}) :: {}", h.name, h.module, h.signature);
                }
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Setup { no_mathlib, yes } => {
            let dir = setup_standalone_workspace(&cfg.lean, !no_mathlib, yes).await?;
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "ok": true,
                        "workspace_dir": dir,
                        "mathlib": !no_mathlib
                    }))?
                );
            } else {
                println!(
                    "[READY ✓] Standalone Lean 4 workspace initialized at {}",
                    dir.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Doctor => {
            let report = LeanBackend::doctor(&cfg, cwd.as_deref()).await;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("juice doctor");
                println!("------------");
                println!(
                    "  elan binary         : {}",
                    report
                        .toolchain
                        .elan
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "not found".into())
                );
                println!(
                    "  lean binary         : {}",
                    report
                        .toolchain
                        .lean
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "not found (run `juice setup --yes`)".into())
                );
                if let Some(ver) = &report.toolchain.lean_version {
                    println!("  lean version        : {ver}");
                }
                println!(
                    "  lake binary         : {}",
                    report
                        .toolchain
                        .lake
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "not found".into())
                );
                println!(
                    "  active Lake project : {}",
                    report
                        .detected_lake_project
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "none in current directory (will use standalone cache)".into())
                );
                println!(
                    "  mathlib cache       : {} ({})",
                    report.mathlib_cache_dir.display(),
                    if report.mathlib_cache_ready {
                        "ready"
                    } else {
                        "not initialized — run `juice setup`"
                    }
                );
                println!("  planner model       : {}", report.planner_model);
                println!("  prover model        : {}", report.prover_model);
                println!(
                    "  model credentials   : {}",
                    if report.planner_key_present {
                        "configured / local endpoint"
                    } else {
                        "missing API key (set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY)"
                    }
                );
            }
            Ok(ExitCode::SUCCESS)
        }

        Commands::Config { action } => match action {
            ConfigAction::Init { local, force } => {
                let target_path = if local {
                    PathBuf::from("juice.toml")
                } else {
                    JuiceConfig::global_config_path()
                };
                if target_path.exists() && !force {
                    bail!(
                        "Config file already exists at {} (pass `--force` to overwrite)",
                        target_path.display()
                    );
                }
                if let Some(parent) = target_path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&target_path, JuiceConfig::sample_toml())?;
                println!("Wrote starter configuration to {}", target_path.display());
                Ok(ExitCode::SUCCESS)
            }
            ConfigAction::Show => {
                if cli.json {
                    println!("{}", serde_json::to_string_pretty(&cfg)?);
                } else {
                    println!("{}", toml::to_string_pretty(&cfg)?);
                }
                Ok(ExitCode::SUCCESS)
            }
            ConfigAction::Path => {
                println!("{}", JuiceConfig::global_config_path().display());
                Ok(ExitCode::SUCCESS)
            }
        },

        Commands::Mcp => {
            juice_mcp::run_mcp_stdio(cfg).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn clap_cli_schema_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn unified_diff_renders_clean_hunk() {
        let before = "theorem demo (n : Nat) : n = n := by\n  sorry\n";
        let after = "theorem demo (n : Nat) : n = n := by\n  rfl\n";
        let diff = render_unified_diff(Path::new("Demo.lean"), before, after);
        assert!(diff.contains("--- a/Demo.lean"));
        assert!(diff.contains("-  sorry"));
        assert!(diff.contains("+  rfl"));
    }
}
