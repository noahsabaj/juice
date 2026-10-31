use anyhow::{Context, bail};
use juice_core::LeanConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainPaths {
    pub elan: Option<PathBuf>,
    pub lean: Option<PathBuf>,
    pub lake: Option<PathBuf>,
    pub lean_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub toolchain: ToolchainPaths,
    pub detected_lake_project: Option<PathBuf>,
    pub mathlib_cache_dir: PathBuf,
    pub mathlib_cache_ready: bool,
    pub planner_model: String,
    pub prover_model: String,
    pub planner_key_present: bool,
    pub ready_for_local_lean: bool,
}

#[derive(Debug, Clone)]
pub struct ResolvedWorkspace {
    pub root: PathBuf,
    pub has_lakefile: bool,
    pub has_mathlib: bool,
}

pub fn locate_binary(name: &str) -> Option<PathBuf> {
    if let Ok(path_var) = std::env::var("PATH") {
        for entry in std::env::split_paths(&path_var) {
            let candidate = entry.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    if let Some(home) = dirs::home_dir() {
        let elan_candidate = home.join(".elan").join("bin").join(name);
        if elan_candidate.is_file() {
            return Some(elan_candidate);
        }
    }

    None
}

pub async fn detect_toolchain() -> ToolchainPaths {
    let elan = locate_binary("elan");
    let lean = locate_binary("lean");
    let lake = locate_binary("lake");

    let lean_version = if let Some(lean_bin) = &lean {
        Command::new(lean_bin)
            .arg("--version")
            .output()
            .await
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    } else {
        None
    };

    ToolchainPaths {
        elan,
        lean,
        lake,
        lean_version,
    }
}

pub fn find_enclosing_lake_project(start: &Path) -> Option<PathBuf> {
    let mut current = if start.is_file() {
        start.parent()?.to_path_buf()
    } else {
        start.to_path_buf()
    };

    loop {
        if current.join("lakefile.toml").is_file() || current.join("lakefile.lean").is_file() {
            return Some(current);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

pub fn project_has_mathlib(project_root: &Path) -> bool {
    if project_root
        .join(".lake")
        .join("packages")
        .join("mathlib")
        .is_dir()
    {
        return true;
    }
    for name in ["lakefile.toml", "lakefile.lean"] {
        let p = project_root.join(name);
        if let Ok(contents) = std::fs::read_to_string(p)
            && contents.to_lowercase().contains("mathlib")
        {
            return true;
        }
    }
    false
}

/// Resolves the active Lake workspace for `target_path` (if inside a Lake project),
/// or falls back to the managed global `~/.cache/juice/mathlib` workspace (or a lightweight temporary directory).
pub fn resolve_workspace(target_path: Option<&Path>, lean_cfg: &LeanConfig) -> ResolvedWorkspace {
    if let Some(p) = target_path
        && let Some(root) = find_enclosing_lake_project(p)
    {
        let has_mathlib = project_has_mathlib(&root);
        return ResolvedWorkspace {
            root,
            has_lakefile: true,
            has_mathlib,
        };
    }

    if let Ok(cwd) = std::env::current_dir()
        && let Some(root) = find_enclosing_lake_project(&cwd)
    {
        let has_mathlib = project_has_mathlib(&root);
        return ResolvedWorkspace {
            root,
            has_lakefile: true,
            has_mathlib,
        };
    }

    let cache_dir = lean_cfg.resolved_mathlib_cache_dir();
    if cache_dir.join("lakefile.toml").is_file() || cache_dir.join("lakefile.lean").is_file() {
        let has_mathlib = project_has_mathlib(&cache_dir);
        return ResolvedWorkspace {
            root: cache_dir,
            has_lakefile: true,
            has_mathlib,
        };
    }

    let fallback = target_path
        .and_then(|p| if p.is_dir() { Some(p.to_path_buf()) } else { p.parent().map(Path::to_path_buf) })
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    ResolvedWorkspace {
        root: fallback,
        has_lakefile: false,
        has_mathlib: false,
    }
}

/// Bootstraps the shared standalone `~/.cache/juice/mathlib` Lake workspace and downloads precompiled `.olean` caches.
pub async fn setup_standalone_workspace(
    lean_cfg: &LeanConfig,
    with_mathlib: bool,
    auto_install_elan: bool,
) -> anyhow::Result<PathBuf> {
    let mut tc = detect_toolchain().await;

    if tc.lean.is_none() || tc.lake.is_none() {
        if !auto_install_elan {
            bail!(
                "Lean 4 (`elan` / `lake`) was not found on PATH or ~/.elan/bin.\n\
                 Re-run `juice setup --yes` to install `elan` automatically, or install manually via:\n\
                 curl https://elan.lean-lang.org/elan-init.sh -sSf | sh"
            );
        }
        install_elan_default().await?;
        tc = detect_toolchain().await;
    }

    let lake_bin = tc
        .lake
        .context("`lake` binary still missing after elan bootstrap")?;

    let cache_dir = lean_cfg.resolved_mathlib_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let lakefile_path = cache_dir.join("lakefile.toml");
    if !lakefile_path.exists() {
        let lakefile_body = if with_mathlib {
            r#"name = "juice_scratch"
version = "0.1.0"
defaultTargets = ["JuiceScratch"]

[[lean_lib]]
name = "JuiceScratch"

[[require]]
name = "mathlib"
scope = "leanprover-community"
"#
        } else {
            r#"name = "juice_scratch"
version = "0.1.0"
defaultTargets = ["JuiceScratch"]

[[lean_lib]]
name = "JuiceScratch"
"#
        };
        std::fs::write(&lakefile_path, lakefile_body)?;
        std::fs::write(
            cache_dir.join("JuiceScratch.lean"),
            "-- Managed scratch root for juice standalone formalizations\n",
        )?;
    }

    if with_mathlib {
        let update_status = Command::new(&lake_bin)
            .arg("update")
            .current_dir(&cache_dir)
            .status()
            .await
            .context("Failed to execute `lake update` in managed mathlib workspace")?;
        if !update_status.success() {
            bail!("`lake update` exited with non-zero status in {}", cache_dir.display());
        }

        let cache_status = Command::new(&lake_bin)
            .args(["exe", "cache", "get"])
            .current_dir(&cache_dir)
            .status()
            .await
            .context("Failed to execute `lake exe cache get`")?;
        if !cache_status.success() {
            tracing::warn!("`lake exe cache get` returned non-zero; continuing with workspace");
        }
    }

    Ok(cache_dir)
}

async fn install_elan_default() -> anyhow::Result<()> {
    let status = Command::new("sh")
        .arg("-c")
        .arg("curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:stable")
        .status()
        .await
        .context("Failed to download and run elan-init.sh")?;
    if !status.success() {
        bail!("elan installer exited with status {status}");
    }
    Ok(())
}
