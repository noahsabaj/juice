use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCheckpoint {
    pub run_id: String,
    pub content_hash: String,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    /// `target_id` -> verified tactic script
    pub solved_targets: BTreeMap<String, String>,
    /// `lemma_signature` -> verified tactic script
    pub solved_sublemmas: BTreeMap<String, String>,
    /// `lemma_signature` -> counterexample witness discovered by `plausible`/`decide`
    pub refuted_lemmas: BTreeMap<String, String>,
    pub nodes_expanded: usize,
}

impl RunCheckpoint {
    pub fn new(run_id: impl Into<String>, content_hash: impl Into<String>) -> Self {
        let now = now_unix();
        Self {
            run_id: run_id.into(),
            content_hash: content_hash.into(),
            created_at_unix: now,
            updated_at_unix: now,
            solved_targets: BTreeMap::new(),
            solved_sublemmas: BTreeMap::new(),
            refuted_lemmas: BTreeMap::new(),
            nodes_expanded: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointStore {
    runs_dir: PathBuf,
}

impl CheckpointStore {
    pub fn new(workspace_root: &Path) -> Self {
        Self {
            runs_dir: workspace_root.join(".juice").join("runs"),
        }
    }

    pub fn compute_run_id(seed_content: &str) -> (String, String) {
        let mut hasher = Sha256::new();
        hasher.update(seed_content.as_bytes());
        let full_hash = format!("{:x}", hasher.finalize());
        let short_id = full_hash[..16].to_string();
        (short_id, full_hash)
    }

    pub fn checkpoint_path(&self, run_id: &str) -> PathBuf {
        self.runs_dir.join(format!("{run_id}.json"))
    }

    pub fn load_or_create(&self, seed_content: &str, explicit_run_id: Option<&str>) -> anyhow::Result<RunCheckpoint> {
        let (computed_id, content_hash) = Self::compute_run_id(seed_content);
        let run_id = explicit_run_id.unwrap_or(&computed_id);
        let path = self.checkpoint_path(run_id);
        if path.exists() {
            let raw = std::fs::read_to_string(&path)?;
            let cp: RunCheckpoint = serde_json::from_str(&raw)?;
            return Ok(cp);
        }
        Ok(RunCheckpoint::new(run_id, content_hash))
    }

    pub fn save(&self, checkpoint: &mut RunCheckpoint) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(&self.runs_dir)?;
        checkpoint.updated_at_unix = now_unix();
        let final_path = self.checkpoint_path(&checkpoint.run_id);
        let tmp_path = self.runs_dir.join(format!("{}.json.tmp", checkpoint.run_id));
        let json = serde_json::to_string_pretty(checkpoint)?;
        std::fs::write(&tmp_path, json)?;
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(final_path)
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_and_resumes_checkpoint_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CheckpointStore::new(dir.path());
        let seed = "theorem add_zero (n : Nat) : n + 0 = n := by sorry";
        let mut cp = store.load_or_create(seed, None).expect("create");
        cp.solved_targets
            .insert("Demo.lean::add_zero#0".to_string(), "rfl".to_string());
        cp.refuted_lemmas
            .insert("∀ n : Nat, n > 0".to_string(), "n = 0".to_string());
        cp.nodes_expanded = 7;

        let path = store.save(&mut cp).expect("save");
        assert!(path.exists());

        let resumed = store.load_or_create(seed, None).expect("resume");
        assert_eq!(resumed.run_id, cp.run_id);
        assert_eq!(
            resumed.solved_targets.get("Demo.lean::add_zero#0").map(String::as_str),
            Some("rfl")
        );
        assert_eq!(
            resumed.refuted_lemmas.get("∀ n : Nat, n > 0").map(String::as_str),
            Some("n = 0")
        );
        assert_eq!(resumed.nodes_expanded, 7);
    }
}
