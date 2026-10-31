use juice_core::PremiseHit;
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedEntry {
    pub name: String,
    pub kind: String,
    pub signature: String,
    pub module: String,
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PremiseIndex {
    pub entries: Vec<IndexedEntry>,
}

impl PremiseIndex {
    pub fn cache_path(workspace_root: &Path) -> PathBuf {
        workspace_root.join(".juice").join("premise_index.json")
    }

    pub fn load_or_build(workspace_root: &Path) -> Self {
        let cache = Self::cache_path(workspace_root);
        if cache.is_file()
            && let Ok(raw) = std::fs::read_to_string(&cache)
            && let Ok(idx) = serde_json::from_str::<Self>(&raw)
            && !idx.entries.is_empty()
        {
            return idx;
        }

        let built = Self::build_from_workspace(workspace_root);
        if !built.entries.is_empty()
            && let Some(parent) = cache.parent()
            && std::fs::create_dir_all(parent).is_ok()
            && let Ok(json) = serde_json::to_string(&built)
        {
            let _ = std::fs::write(&cache, json);
        }
        built
    }

    pub fn build_from_workspace(workspace_root: &Path) -> Self {
        let mut entries = Vec::new();
        if !workspace_root.exists() {
            return Self { entries };
        }

        // Index workspace `.lean` files plus `.lake/packages` declarations (capped for fast startup).
        let max_files = 4000usize;
        let mut scanned = 0usize;

        for dir_entry in WalkDir::new(workspace_root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                name != ".git" && name != "target" && name != ".juice" && name != "build"
            })
            .flatten()
        {
            if !dir_entry.file_type().is_file() {
                continue;
            }
            let path = dir_entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("lean") {
                continue;
            }
            scanned += 1;
            if scanned > max_files {
                break;
            }
            if let Ok(content) = std::fs::read_to_string(path) {
                let module = derive_module_name(workspace_root, path);
                entries.extend(index_lean_source(&content, &module));
            }
        }

        Self { entries }
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<PremiseHit> {
        if self.entries.is_empty() || limit == 0 {
            return Vec::new();
        }
        let q_tokens = tokenize_math_query(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        let q_lower = query.to_lowercase();

        let mut scored: Vec<(f64, &IndexedEntry)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let mut score = 0.0f64;
                let name_lower = entry.name.to_lowercase();
                if name_lower == q_lower {
                    score += 12.0;
                } else if name_lower.contains(&q_lower) && q_lower.len() >= 3 {
                    score += 5.0;
                }

                for qt in &q_tokens {
                    if name_lower.split(['.', '_']).any(|seg| seg == qt) {
                        score += 3.0;
                    } else if name_lower.contains(qt.as_str()) {
                        score += 1.5;
                    }
                    if entry.tokens.iter().any(|t| t == qt) {
                        score += 1.0;
                    }
                }

                if score > 0.0 {
                    // Normalize gently by signature verbosity so concise fundamental lemmas rank higher.
                    let norm = 1.0 + (entry.tokens.len() as f64).ln_1p() * 0.25;
                    Some((score / norm, entry))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(limit)
            .map(|(score, e)| PremiseHit {
                name: e.name.clone(),
                kind: e.kind.clone(),
                signature: e.signature.clone(),
                module: e.module.clone(),
                doc: None,
                score,
                verified_by_kernel: false,
            })
            .collect()
    }
}

pub fn index_lean_source(source: &str, module: &str) -> Vec<IndexedEntry> {
    static DECL_LINE_RE: OnceLock<Regex> = OnceLock::new();
    let re = DECL_LINE_RE.get_or_init(|| {
        Regex::new(
            r"(?m)^(?:@\[[^\]]+\]\s*)*(?:protected\s+|noncomputable\s+)*(theorem|lemma|def|abbrev)\s+([A-Za-z_][A-Za-z0-9_'.?!]*)\s*([^\n]*)",
        )
        .expect("valid decl regex")
    });

    let mut namespace_stack: Vec<String> = Vec::new();
    let mut entries = Vec::new();

    // Fast line-oriented scan tracking active namespaces and declaration headers.
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(ns) = trimmed.strip_prefix("namespace ") {
            let ns_name = ns.split_whitespace().next().unwrap_or("");
            if !ns_name.is_empty() {
                namespace_stack.push(ns_name.to_string());
            }
            continue;
        }
        if let Some(end_ns) = trimmed.strip_prefix("end ") {
            let ns_name = end_ns.split_whitespace().next().unwrap_or("");
            if namespace_stack.last().map(String::as_str) == Some(ns_name) {
                namespace_stack.pop();
            }
            continue;
        }
        if let Some(caps) = re.captures(trimmed) {
            let kind = caps[1].to_string();
            let raw_name = &caps[2];
            let rest = caps[3].split(":=").next().unwrap_or("").trim();
            let full_name = if namespace_stack.is_empty() || raw_name.contains('.') {
                raw_name.to_string()
            } else {
                format!("{}.{}", namespace_stack.join("."), raw_name)
            };
            let signature = format!("{kind} {full_name} {rest}").trim().to_string();
            let tokens = tokenize_math_query(&signature);
            entries.push(IndexedEntry {
                name: full_name,
                kind,
                signature,
                module: module.to_string(),
                tokens,
            });
        }
    }

    entries
}

fn derive_module_name(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).unwrap_or(file);
    let mut parts: Vec<String> = rel
        .iter()
        .map(|os| os.to_string_lossy().to_string())
        .collect();
    if let Some(last) = parts.last_mut()
        && let Some(stripped) = last.strip_suffix(".lean")
    {
        *last = stripped.to_string();
    }
    // Trim `.lake/packages/<pkg>/` prefix if present.
    if parts.len() > 3 && parts[0] == ".lake" && parts[1] == "packages" {
        parts.drain(0..3);
    }
    parts.join(".")
}

pub fn tokenize_math_query(text: &str) -> Vec<String> {
    let stop_words: HashSet<&'static str> = [
        "theorem", "lemma", "def", "by", "sorry", "the", "for", "all", "that", "prove", "show",
        "with", "and",
    ]
    .into_iter()
    .collect();

    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '\'' {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else {
            if cur.len() >= 2 && !stop_words.contains(cur.as_str()) && !out.contains(&cur) {
                out.push(cur.clone());
            }
            cur.clear();
            // Capture salient math symbols too!
            if matches!(ch, 'ℕ' | 'ℤ' | 'ℚ' | 'ℝ' | 'ℂ' | '≤' | '≥' | '∣' | '∑' | '∏') {
                let s = ch.to_string();
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
    }
    if cur.len() >= 2 && !stop_words.contains(cur.as_str()) && !out.contains(&cur) {
        out.push(cur);
    }
    out
}

/// Optional online fallback querying Loogle (`https://loogle.lean-lang.org/json`) when `--online-search` is enabled.
pub async fn query_loogle_fallback(query: &str, limit: usize) -> Vec<PremiseHit> {
    let Ok(client) = Client::builder().timeout(Duration::from_secs(5)).build() else {
        return Vec::new();
    };
    let Ok(resp) = client
        .get("https://loogle.lean-lang.org/json")
        .query(&[("q", query)])
        .send()
        .await
    else {
        return Vec::new();
    };
    let Ok(json) = resp.json::<Value>().await else {
        return Vec::new();
    };
    let Some(hits) = json.get("hits").and_then(Value::as_array) else {
        return Vec::new();
    };

    hits.iter()
        .take(limit)
        .filter_map(|h| {
            let name = h.get("name")?.as_str()?.to_string();
            let module = h
                .get("module")
                .and_then(Value::as_str)
                .unwrap_or("Mathlib")
                .to_string();
            let sig = h
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let doc = h.get("doc").and_then(Value::as_str).map(ToString::to_string);
            Some(PremiseHit {
                name,
                kind: "theorem".to_string(),
                signature: sig,
                module,
                doc,
                score: 5.0,
                verified_by_kernel: false,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_namespaced_lean_declarations_and_ranks_hits() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("Arithmetic.lean");
        std::fs::write(
            &file_path,
            r#"namespace MyNat

theorem add_comm_custom (a b : Nat) : a + b = b + a := by sorry
lemma mul_zero_custom (n : Nat) : n * 0 = 0 := by sorry
def double (n : Nat) : Nat := n + n

end MyNat
"#,
        )
        .unwrap();

        let index = PremiseIndex::build_from_workspace(dir.path());
        assert_eq!(index.entries.len(), 3);

        let hits = index.search("add_comm Nat", 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].name, "MyNat.add_comm_custom");
        assert_eq!(hits[0].module, "Arithmetic");
    }
}
