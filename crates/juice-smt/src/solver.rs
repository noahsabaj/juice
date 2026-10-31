use crate::ir::SmtScript;
use crate::locator::locate_z3;
use crate::sexpr::{SExpr, SmtValue, extract_bindings_from_sexpr, format_sexpr, parse_sexprs};
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtOutcome {
    /// Property holds (negated hazard/postcondition is unsatisfiable).
    Unsat,
    /// Counterexample discovered satisfying the negated property.
    Sat(BTreeMap<String, SmtValue>),
    /// Solver timed out or returned `unknown`.
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct Z3Solver {
    z3_bin: PathBuf,
    timeout: Duration,
}

impl Z3Solver {
    pub fn discover(configured: Option<&Path>, timeout_secs: u64) -> anyhow::Result<Self> {
        let z3_bin = locate_z3(configured).ok_or_else(|| {
            anyhow::anyhow!(
                "Could not locate `z3` binary on PATH or `/opt/homebrew/bin/z3`. Install Z3 (`brew install z3` or `apt install z3`)."
            )
        })?;
        Ok(Self {
            z3_bin,
            timeout: Duration::from_secs(timeout_secs.max(1)),
        })
    }

    pub fn new(z3_bin: PathBuf, timeout: Duration) -> Self {
        Self { z3_bin, timeout }
    }

    pub fn z3_path(&self) -> &Path {
        &self.z3_bin
    }

    pub async fn check(&self, script: &SmtScript) -> anyhow::Result<SmtOutcome> {
        let rendered = script.render();
        self.check_raw(&rendered).await
    }

    pub async fn check_raw(&self, smtlib: &str) -> anyhow::Result<SmtOutcome> {
        let timeout_secs = self.timeout.as_secs().max(1);
        let mut child = Command::new(&self.z3_bin)
            .arg("-in")
            .arg("-smt2")
            .arg(format!("-T:{timeout_secs}"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("Failed to spawn Z3 at {}", self.z3_bin.display()))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(smtlib.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        let output = match tokio::time::timeout(
            self.timeout + Duration::from_secs(2),
            child.wait_with_output(),
        )
        .await
        {
            Ok(res) => res?,
            Err(_) => {
                return Ok(SmtOutcome::Unknown(format!(
                    "Z3 exceeded wall-clock timeout ({timeout_secs}s)"
                )));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let exprs = parse_sexprs(&stdout).with_context(|| {
            format!(
                "Failed to parse Z3 output.\nstdout:\n{stdout}\nstderr:\n{stderr}\nscript:\n{smtlib}"
            )
        })?;

        for (idx, expr) in exprs.iter().enumerate() {
            match expr {
                SExpr::Atom(a) if a == "unsat" => return Ok(SmtOutcome::Unsat),
                SExpr::Atom(a) if a == "unknown" => {
                    return Ok(SmtOutcome::Unknown("Z3 returned `unknown`".to_string()));
                }
                SExpr::Atom(a) if a == "timeout" => {
                    return Ok(SmtOutcome::Unknown("Z3 timed out".to_string()));
                }
                SExpr::Atom(a) if a == "sat" => {
                    let bindings = exprs
                        .get(idx + 1)
                        .map(extract_bindings_from_sexpr)
                        .unwrap_or_default();
                    return Ok(SmtOutcome::Sat(bindings));
                }
                SExpr::List(items)
                    if items.first() == Some(&SExpr::Atom("error".to_string())) =>
                {
                    bail!(
                        "Z3 reported SMT-LIB2 error: {}\nScript:\n{}",
                        format_sexpr(expr),
                        smtlib
                    );
                }
                _ => {}
            }
        }

        bail!(
            "Z3 produced no sat/unsat verdict.\nstdout: {stdout}\nstderr: {stderr}"
        )
    }
}
