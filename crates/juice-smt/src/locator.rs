use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Z3DoctorReport {
    pub z3_bin: Option<PathBuf>,
    pub version: Option<String>,
}

pub fn locate_z3(configured: Option<&Path>) -> Option<PathBuf> {
    if let Some(custom) = configured
        && custom.is_file()
    {
        return Some(custom.to_path_buf());
    }

    if let Ok(env_bin) = std::env::var("Z3_BIN") {
        let p = PathBuf::from(env_bin);
        if p.is_file() {
            return Some(p);
        }
    }

    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join("z3");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    for fallback in [
        "/opt/homebrew/bin/z3",
        "/usr/local/bin/z3",
        "/usr/bin/z3",
    ] {
        let p = PathBuf::from(fallback);
        if p.is_file() {
            return Some(p);
        }
    }

    None
}

pub fn z3_doctor(configured: Option<&Path>) -> Z3DoctorReport {
    let z3_bin = locate_z3(configured);
    let version = z3_bin.as_ref().and_then(|bin| {
        Command::new(bin)
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    });
    Z3DoctorReport { z3_bin, version }
}
