use crate::workspace::{detect_toolchain, find_enclosing_lake_project};
use anyhow::{Context, bail};
use juice_core::{DiagnosticMsg, DiagnosticSeverity, NormalizedGoal};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Notify, oneshot};

#[derive(Debug, Clone)]
pub struct DocumentEval {
    pub goals: Vec<NormalizedGoal>,
    pub diagnostics: Vec<DiagnosticMsg>,
}

#[derive(Debug, Default, Clone)]
struct UriTrackState {
    diagnostics_version: i64,
    progress_done_version: i64,
    diagnostics: Vec<DiagnosticMsg>,
}

struct SharedReaderState {
    pending: HashMap<u64, oneshot::Sender<Value>>,
    docs: HashMap<String, UriTrackState>,
}

/// Asynchronous JSON-RPC 2.0 LSP client session speaking `Content-Length` framed packets
/// to a `lean --server` worker process (or an in-memory duplex stream in unit tests).
pub struct LspSession {
    writer: Mutex<Box<dyn AsyncWrite + Unpin + Send>>,
    state: Arc<Mutex<SharedReaderState>>,
    notify: Arc<Notify>,
    next_id: AtomicU64,
    opened_versions: Mutex<HashMap<String, i64>>,
    child: Mutex<Option<Child>>,
}

impl LspSession {
    pub async fn connect_streams<R, W>(
        reader: R,
        writer: W,
        root_dir: &Path,
        child: Option<Child>,
    ) -> anyhow::Result<Self>
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let state = Arc::new(Mutex::new(SharedReaderState {
            pending: HashMap::new(),
            docs: HashMap::new(),
        }));
        let notify = Arc::new(Notify::new());

        let state_clone = state.clone();
        let notify_clone = notify.clone();
        tokio::spawn(async move {
            let mut buf_reader = BufReader::new(reader);
            while let Ok(Some(msg)) = read_lsp_frame(&mut buf_reader).await {
                if let Some(id) = msg.get("id").and_then(Value::as_u64)
                    && (msg.get("result").is_some() || msg.get("error").is_some())
                {
                    let mut guard = state_clone.lock().await;
                    if let Some(tx) = guard.pending.remove(&id) {
                        let _ = tx.send(msg);
                    }
                    continue;
                }

                if let Some(method) = msg.get("method").and_then(Value::as_str) {
                    match method {
                        "textDocument/publishDiagnostics" => {
                            if let Some(params) = msg.get("params")
                                && let Some(uri) = params.get("uri").and_then(Value::as_str)
                            {
                                let ver = params.get("version").and_then(Value::as_i64).unwrap_or(1);
                                let diags = parse_lsp_diagnostics(params.get("diagnostics"));
                                let mut guard = state_clone.lock().await;
                                let entry = guard.docs.entry(uri.to_string()).or_default();
                                entry.diagnostics_version = ver;
                                entry.diagnostics = diags;
                                drop(guard);
                                notify_clone.notify_waiters();
                            }
                        }
                        "$/lean/fileProgress" => {
                            if let Some(params) = msg.get("params")
                                && let Some(uri) =
                                    params.pointer("/textDocument/uri").and_then(Value::as_str)
                            {
                                let ver = params
                                    .pointer("/textDocument/version")
                                    .and_then(Value::as_i64)
                                    .unwrap_or(1);
                                let is_done = params
                                    .get("processing")
                                    .and_then(Value::as_array)
                                    .map(Vec::is_empty)
                                    .unwrap_or(false);
                                if is_done {
                                    let mut guard = state_clone.lock().await;
                                    let entry = guard.docs.entry(uri.to_string()).or_default();
                                    entry.progress_done_version = ver;
                                    drop(guard);
                                    notify_clone.notify_waiters();
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        });

        let session = Self {
            writer: Mutex::new(Box::new(writer)),
            state,
            notify,
            next_id: AtomicU64::new(1),
            opened_versions: Mutex::new(HashMap::new()),
            child: Mutex::new(child),
        };

        let root_uri = format!("file://{}", root_dir.display());
        let _init_resp = session
            .request(
                "initialize",
                json!({
                    "processId": std::process::id(),
                    "rootUri": root_uri,
                    "capabilities": {}
                }),
            )
            .await
            .context("Lean LSP `initialize` handshake failed")?;

        session.notify("initialized", json!({})).await?;
        Ok(session)
    }

    pub async fn spawn_lean_worker(workspace_root: &Path) -> anyhow::Result<Self> {
        let tc = detect_toolchain().await;
        let lean_bin = tc.lean.context(
            "Lean 4 (`lean`) binary not found on PATH or ~/.elan/bin. Run `juice setup` or `juice doctor`.",
        )?;

        let lake_root = find_enclosing_lake_project(workspace_root);
        let mut cmd = if let (Some(root), Some(lake_bin)) = (&lake_root, tc.lake.as_ref()) {
            let mut c = Command::new(lake_bin);
            c.arg("env").arg(&lean_bin).arg("--server").current_dir(root);
            c
        } else {
            let mut c = Command::new(&lean_bin);
            c.arg("--server").current_dir(workspace_root);
            c
        };

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = cmd.spawn().context("Failed to spawn `lean --server`")?;
        let stdout = child.stdout.take().context("Missing child stdout")?;
        let stdin = child.stdin.take().context("Missing child stdin")?;
        let effective_root = lake_root.as_deref().unwrap_or(workspace_root);

        Self::connect_streams(stdout, stdin, effective_root, Some(child)).await
    }

    pub async fn request(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        {
            let mut guard = self.state.lock().await;
            guard.pending.insert(id, tx);
        }

        let packet = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_packet(&packet).await?;

        let response = rx
            .await
            .with_context(|| format!("LSP connection closed while awaiting `{method}`"))?;
        if let Some(err) = response.get("error") {
            bail!("Lean LSP `{method}` error: {err}");
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    pub async fn notify(&self, method: &str, params: Value) -> anyhow::Result<()> {
        let packet = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_packet(&packet).await
    }

    async fn write_packet(&self, payload: &Value) -> anyhow::Result<()> {
        let serialized = serde_json::to_vec(payload)?;
        let header = format!("Content-Length: {}\r\n\r\n", serialized.len());
        let mut writer = self.writer.lock().await;
        writer.write_all(header.as_bytes()).await?;
        writer.write_all(&serialized).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Synchronizes `text` into the warm Lean server (preserving cached header imports!)
    /// and queries diagnostics plus open tactic goals at `(line, col_utf16)`.
    pub async fn evaluate_document(
        &self,
        uri: &str,
        text: &str,
        cursor_line: u32,
        cursor_col_utf16: u32,
        timeout: Duration,
    ) -> anyhow::Result<DocumentEval> {
        let is_first_open = !self.opened_versions.lock().await.contains_key(uri);
        // Full umbrella `import Mathlib` memory-maps ~9,100 `.olean` files on initial `didOpen`,
        // which can take 30-60s cold before dropping to <50ms on subsequent `didChange` updates.
        let effective_timeout = if is_first_open && text.contains("import Mathlib") {
            timeout.max(Duration::from_secs(90))
        } else {
            timeout
        };
        tokio::time::timeout(effective_timeout, async {
            let version = {
                let mut versions = self.opened_versions.lock().await;
                if let Some(v) = versions.get_mut(uri) {
                    *v += 1;
                    let ver = *v;
                    drop(versions);
                    self.notify(
                        "textDocument/didChange",
                        json!({
                            "textDocument": { "uri": uri, "version": ver },
                            "contentChanges": [{ "text": text }]
                        }),
                    )
                    .await?;
                    ver
                } else {
                    versions.insert(uri.to_string(), 1);
                    drop(versions);
                    self.notify(
                        "textDocument/didOpen",
                        json!({
                            "textDocument": {
                                "uri": uri,
                                "languageId": "lean4",
                                "version": 1,
                                "text": text,
                            }
                        }),
                    )
                    .await?;
                    1
                }
            };

            // Wait until Lean finishes elaborating `version` via `$/lean/fileProgress`.
            loop {
                let notified = self.notify.notified();
                {
                    let guard = self.state.lock().await;
                    if let Some(doc) = guard.docs.get(uri)
                        && doc.progress_done_version >= version
                    {
                        break;
                    }
                }
                notified.await;
            }

            let goal_res = self
                .request(
                    "$/lean/plainGoal",
                    json!({
                        "textDocument": { "uri": uri },
                        "position": { "line": cursor_line, "character": cursor_col_utf16 }
                    }),
                )
                .await
                .unwrap_or(Value::Null);

            let goals = extract_goals_from_plain_goal_result(&goal_res);
            let diagnostics = {
                let guard = self.state.lock().await;
                guard
                    .docs
                    .get(uri)
                    .map(|d| d.diagnostics.clone())
                    .unwrap_or_default()
            };

            Ok(DocumentEval { goals, diagnostics })
        })
        .await
        .context("Timed out waiting for Lean LSP tactic evaluation")?
    }

    pub async fn shutdown(&self) {
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
        }
    }
}

fn extract_goals_from_plain_goal_result(res: &Value) -> Vec<NormalizedGoal> {
    if res.is_null() {
        return Vec::new();
    }
    if let Some(arr) = res.get("goals").and_then(Value::as_array) {
        return arr
            .iter()
            .filter_map(Value::as_str)
            .map(NormalizedGoal::new)
            .filter(|g| !g.canonical_pp.is_empty())
            .collect();
    }
    if let Some(rendered) = res.get("rendered").and_then(Value::as_str) {
        return NormalizedGoal::parse_multi(rendered);
    }
    Vec::new()
}

fn parse_lsp_diagnostics(raw_diags: Option<&Value>) -> Vec<DiagnosticMsg> {
    let Some(arr) = raw_diags.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for d in arr {
        let line = d
            .pointer("/range/start/line")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let column = d
            .pointer("/range/start/character")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let sev_code = d.get("severity").and_then(Value::as_u64).unwrap_or(1);
        let severity = match sev_code {
            1 => DiagnosticSeverity::Error,
            2 => DiagnosticSeverity::Warning,
            3 => DiagnosticSeverity::Information,
            _ => DiagnosticSeverity::Hint,
        };
        let message = d
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        out.push(DiagnosticMsg {
            line,
            column,
            severity,
            message,
        });
    }
    out
}

pub async fn read_lsp_frame<R: AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> anyhow::Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if content_length.is_some() {
                break;
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:")
            .or_else(|| trimmed.strip_prefix("content-length:"))
        {
            content_length = Some(rest.trim().parse::<usize>()?);
        }
    }

    let len = content_length.context("Missing Content-Length in LSP frame")?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).await?;
    let val: Value = serde_json::from_slice(&body)?;
    Ok(Some(val))
}

/// Pool of warm persistent `lean --server` workers supporting concurrent MCTS expansions
/// and automatic crash/timeout recycling.
pub struct LspWorkerPool {
    workspace_root: PathBuf,
    max_workers: usize,
    idle_workers: Mutex<Vec<(usize, LspSession)>>,
    spawn_counter: AtomicU64,
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl LspWorkerPool {
    pub fn new(workspace_root: PathBuf, max_workers: usize) -> Self {
        let cap = max_workers.max(1);
        Self {
            workspace_root,
            max_workers: cap,
            idle_workers: Mutex::new(Vec::with_capacity(cap)),
            spawn_counter: AtomicU64::new(0),
            semaphore: Arc::new(tokio::sync::Semaphore::new(cap)),
        }
    }

    pub fn max_workers(&self) -> usize {
        self.max_workers
    }

    pub async fn evaluate(
        &self,
        virtual_filename: &str,
        code: &str,
        cursor_line: u32,
        cursor_col_utf16: u32,
        timeout: Duration,
    ) -> anyhow::Result<DocumentEval> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .context("LSP worker pool closed")?;

        let (worker_id, session) = match self.idle_workers.lock().await.pop() {
            Some(pair) => pair,
            None => {
                let id = self.spawn_counter.fetch_add(1, Ordering::SeqCst) as usize;
                let s = LspSession::spawn_lean_worker(&self.workspace_root).await?;
                (id, s)
            }
        };

        let uri = format!(
            "file://{}/.juice_worker_{}_{}",
            self.workspace_root.display(),
            worker_id,
            virtual_filename
        );

        match session
            .evaluate_document(&uri, code, cursor_line, cursor_col_utf16, timeout)
            .await
        {
            Ok(eval) => {
                self.idle_workers.lock().await.push((worker_id, session));
                Ok(eval)
            }
            Err(err) => {
                // Recycle timed-out or broken worker instead of returning it to the pool.
                session.shutdown().await;
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, val: &Value) {
        let bytes = serde_json::to_vec(val).unwrap();
        let hdr = format!("Content-Length: {}\r\n\r\n", bytes.len());
        w.write_all(hdr.as_bytes()).await.unwrap();
        w.write_all(&bytes).await.unwrap();
        w.flush().await.unwrap();
    }

    #[tokio::test]
    async fn lsp_session_handles_interleaved_diagnostics_progress_and_plain_goal() {
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        let (client_r, client_w) = tokio::io::split(client_io);
        let (server_r, mut server_w) = tokio::io::split(server_io);

        // Spawn mock `lean --server` task.
        tokio::spawn(async move {
            let mut reader = BufReader::new(server_r);
            while let Ok(Some(req)) = read_lsp_frame(&mut reader).await {
                let method = req.get("method").and_then(Value::as_str).unwrap_or("");
                match method {
                    "initialize" => {
                        let id = req["id"].as_u64().unwrap();
                        write_frame(
                            &mut server_w,
                            &json!({ "jsonrpc": "2.0", "id": id, "result": { "capabilities": {} } }),
                        )
                        .await;
                    }
                    "textDocument/didOpen" | "textDocument/didChange" => {
                        let uri = req
                            .pointer("/params/textDocument/uri")
                            .and_then(Value::as_str)
                            .unwrap()
                            .to_string();
                        let version = req
                            .pointer("/params/textDocument/version")
                            .and_then(Value::as_i64)
                            .unwrap();
                        let text = req
                            .pointer("/params/textDocument/text")
                            .or_else(|| req.pointer("/params/contentChanges/0/text"))
                            .and_then(Value::as_str)
                            .unwrap_or("");

                        let diags = if text.contains("exact?") {
                            json!([{
                                "range": { "start": { "line": 1, "character": 2 }, "end": { "line": 1, "character": 8 } },
                                "severity": 3,
                                "message": "Try this: exact Nat.add_zero n"
                            }])
                        } else {
                            json!([])
                        };

                        write_frame(
                            &mut server_w,
                            &json!({
                                "jsonrpc": "2.0",
                                "method": "textDocument/publishDiagnostics",
                                "params": { "uri": uri, "version": version, "diagnostics": diags }
                            }),
                        )
                        .await;

                        write_frame(
                            &mut server_w,
                            &json!({
                                "jsonrpc": "2.0",
                                "method": "$/lean/fileProgress",
                                "params": {
                                    "textDocument": { "uri": uri, "version": version },
                                    "processing": []
                                }
                            }),
                        )
                        .await;
                    }
                    "$/lean/plainGoal" => {
                        let id = req["id"].as_u64().unwrap();
                        let line = req
                            .pointer("/params/position/line")
                            .and_then(Value::as_u64)
                            .unwrap_or(0);
                        let goals = if line == 99 {
                            json!({ "goals": [] })
                        } else {
                            json!({ "goals": ["n : Nat\n⊢ n + 0 = n"] })
                        };
                        write_frame(
                            &mut server_w,
                            &json!({ "jsonrpc": "2.0", "id": id, "result": goals }),
                        )
                        .await;
                    }
                    _ => {}
                }
            }
        });

        let session = LspSession::connect_streams(client_r, client_w, Path::new("/tmp"), None)
            .await
            .expect("handshake succeeds");

        let eval1 = session
            .evaluate_document(
                "file:///tmp/Demo.lean",
                "theorem t (n : Nat) : n + 0 = n := by\n  sorry",
                1,
                2,
                Duration::from_secs(2),
            )
            .await
            .expect("didOpen eval");
        assert_eq!(eval1.goals.len(), 1);
        assert_eq!(eval1.goals[0].canonical_pp, "n : Nat\n⊢ n + 0 = n");

        let eval2 = session
            .evaluate_document(
                "file:///tmp/Demo.lean",
                "theorem t (n : Nat) : n + 0 = n := by\n  exact?",
                99,
                2,
                Duration::from_secs(2),
            )
            .await
            .expect("didChange eval");
        assert!(eval2.goals.is_empty());
        assert_eq!(eval2.diagnostics.len(), 1);
        assert!(eval2.diagnostics[0].message.contains("Try this: exact Nat.add_zero n"));
    }
}
