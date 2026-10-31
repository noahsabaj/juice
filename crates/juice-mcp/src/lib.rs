use juice_core::{
    AxiomPolicy, JuiceConfig, ProofContext, VerifierBackend, events::noop_event_sink,
};
use juice_lean::{LeanBackend, parser::extract_proof_targets, workspace::resolve_workspace};
use juice_llm::ModelRouter;
use juice_search::{AutoformalizeOptions, Autoformalizer, TwoTierProver};
use juice_verify::{SourceLanguage, VerifyCommandOptions, verify_inline_code, verify_path};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FramingMode {
    NewlineDelimited,
    ContentLength,
}

/// Runs the `juice` MCP server over standard input and standard output.
pub async fn run_mcp_stdio(cfg: JuiceConfig) -> anyhow::Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    run_mcp_on_streams(stdin, stdout, cfg).await
}

pub async fn run_mcp_on_streams<R, W>(
    reader: R,
    mut writer: W,
    cfg: JuiceConfig,
) -> anyhow::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let verifier: Arc<dyn VerifierBackend> =
        Arc::new(LeanBackend::new(cfg.lean.clone(), cfg.search.clone()));
    let router = ModelRouter::from_config(&cfg.models);
    let mut buf_reader = BufReader::new(reader);

    while let Some((msg, framing)) = read_mcp_message(&mut buf_reader).await? {
        let Some(method) = msg.get("method").and_then(Value::as_str) else {
            continue;
        };
        let id_opt = msg.get("id").cloned();

        // Notifications have no `id` and expect no response.
        let Some(id) = id_opt else {
            continue;
        };

        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
        let response = handle_rpc_request(
            id,
            method,
            params,
            &cfg,
            &verifier,
            &router,
        )
        .await;

        write_mcp_message(&mut writer, &response, framing).await?;
    }

    Ok(())
}

async fn handle_rpc_request(
    id: Value,
    method: &str,
    params: Value,
    cfg: &JuiceConfig,
    verifier: &Arc<dyn VerifierBackend>,
    router: &ModelRouter,
) -> Value {
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "juice",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        }),
        "ping" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {}
        }),
        "tools/list" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "tools": tool_definitions()
            }
        }),
        "tools/call" => {
            let tool_name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            match dispatch_tool_call(tool_name, args, cfg, verifier, router).await {
                Ok(text) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    }
                }),
                Err(err) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": format!("Error: {err:#}") }],
                        "isError": true
                    }
                }),
            }
        }
        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("Method not found: {method}")
            }
        }),
    }
}

async fn dispatch_tool_call(
    tool_name: &str,
    args: Value,
    cfg: &JuiceConfig,
    verifier: &Arc<dyn VerifierBackend>,
    router: &ModelRouter,
) -> anyhow::Result<String> {
    match tool_name {
        "juice_fill_sorry" => {
            let path_str = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("Missing required string argument `path`"))?;
            let in_place = args
                .get("in_place")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let target_filter = args.get("target").and_then(Value::as_str);

            let prover = TwoTierProver::new(
                verifier.clone(),
                router.clone(),
                cfg.search.clone(),
                noop_event_sink(),
            );
            let policy = AxiomPolicy::strict().with_native_decide(cfg.lean.allow_native_decide);
            let outcome = prover
                .fill_path(
                    &PathBuf::from(path_str),
                    target_filter,
                    None,
                    &policy,
                    in_place,
                )
                .await?;
            Ok(serde_json::to_string_pretty(&outcome)?)
        }
        "juice_formalize" => {
            let claim = args
                .get("claim")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("Missing required string argument `claim`"))?;
            let prove = args.get("prove").and_then(Value::as_bool).unwrap_or(true);
            let output_path = args
                .get("output_path")
                .and_then(Value::as_str)
                .map(PathBuf::from);

            let formalizer = Autoformalizer::new(
                verifier.clone(),
                router.clone(),
                cfg.search.clone(),
                cfg.lean.clone(),
                noop_event_sink(),
            );
            let opts = AutoformalizeOptions {
                prove_theorems: prove,
                max_formalize_attempts: 4,
                policy: AxiomPolicy::strict().with_native_decide(cfg.lean.allow_native_decide),
            };
            let outcome = formalizer
                .formalize_input(claim, output_path.as_deref(), &opts)
                .await?;
            Ok(serde_json::to_string_pretty(&outcome)?)
        }
        "juice_audit" => {
            let path_str = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("Missing required string argument `path`"))?;
            let allow_native = args
                .get("allow_native_decide")
                .and_then(Value::as_bool)
                .unwrap_or(cfg.lean.allow_native_decide);
            let policy = AxiomPolicy::strict().with_native_decide(allow_native);
            let report = verifier
                .cold_audit(&PathBuf::from(path_str), &[], &policy)
                .await?;
            Ok(serde_json::to_string_pretty(&report)?)
        }
        "juice_premise_search" => {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("Missing required string argument `query`"))?;
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
            let ws_hint = args
                .get("workspace_path")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let ctx = if let Some(explicit_dir) = &ws_hint
                && explicit_dir.exists()
                && juice_lean::workspace::find_enclosing_lake_project(explicit_dir).is_none()
            {
                ProofContext {
                    workspace_root: explicit_dir.clone(),
                    source_file: Some(explicit_dir.clone()),
                    imports_header: String::new(),
                    preamble: String::new(),
                }
            } else {
                let ws = resolve_workspace(ws_hint.as_deref(), &cfg.lean);
                ProofContext {
                    workspace_root: ws.root,
                    source_file: ws_hint,
                    imports_header: if ws.has_mathlib {
                        "import Mathlib".to_string()
                    } else {
                        String::new()
                    },
                    preamble: String::new(),
                }
            };
            let hits = verifier.search_premises(&ctx, query, limit).await?;
            Ok(serde_json::to_string_pretty(&hits)?)
        }
        "juice_check_goal" => {
            let code = args
                .get("lean_code")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("Missing required string argument `lean_code`"))?;
            let ws_hint = args
                .get("workspace_path")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            let ws = resolve_workspace(ws_hint.as_deref(), &cfg.lean);
            let targets = extract_proof_targets(code, &ws.root, None);
            if let Some(first_target) = targets.first() {
                let snap = verifier.initial_state(first_target).await?;
                Ok(serde_json::to_string_pretty(&snap.open_goals)?)
            } else {
                let ctx = ProofContext {
                    workspace_root: ws.root,
                    source_file: None,
                    imports_header: String::new(),
                    preamble: String::new(),
                };
                let diags = verifier.check_declaration(&ctx, code).await?;
                Ok(serde_json::to_string_pretty(&diags)?)
            }
        }
        "juice_verify_code" => {
            let target_fn = args
                .get("function")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let cli_spec = args
                .get("spec")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let equiv = args
                .get("equiv")
                .and_then(Value::as_str)
                .and_then(|s| s.split_once('='))
                .map(|(a, b)| (a.trim().to_string(), b.trim().to_string()));
            let fix = args.get("fix").and_then(Value::as_bool).unwrap_or(false);
            let in_place = args
                .get("in_place")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let emit_test = args
                .get("emit_test")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let unwind_depth = args
                .get("unwind")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(cfg.verify.default_unwind);
            let replay_runtime = args
                .get("replay_runtime")
                .and_then(Value::as_bool)
                .unwrap_or(cfg.verify.replay_runtime);

            let cmd_opts = VerifyCommandOptions {
                target_fn,
                cli_spec,
                equiv,
                fix,
                in_place,
                emit_test,
                unwind_depth,
                replay_runtime,
            };
            let active_router = cfg
                .models
                .planner
                .resolve_api_key()
                .is_some()
                .then_some(router);

            if let Some(path_str) = args.get("path").and_then(Value::as_str) {
                let summary =
                    verify_path(&PathBuf::from(path_str), cfg, &cmd_opts, active_router).await?;
                Ok(serde_json::to_string_pretty(&summary)?)
            } else if let Some(code) = args.get("code").and_then(Value::as_str) {
                let lang_str = args
                    .get("language")
                    .and_then(Value::as_str)
                    .unwrap_or("rust");
                let lang = SourceLanguage::from_name(lang_str).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Unsupported language `{lang_str}` (expected `rust`, `c`, `python`, or `typescript`)"
                    )
                })?;
                let report =
                    verify_inline_code(code, lang, cfg, &cmd_opts, active_router).await?;
                Ok(serde_json::to_string_pretty(&report)?)
            } else {
                anyhow::bail!("Provide either `path` or inline `code` (+ `language`) to `juice_verify_code`")
            }
        }
        other => anyhow::bail!("Unknown MCP tool `{other}`"),
    }
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "juice_fill_sorry",
            "description": "Discover and formally prove `sorry` placeholders in a Lean 4 file or Lake project using hierarchical planning, MCGS, and cold kernel axiom certification.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to `.lean` file or Lake project directory" },
                    "target": { "type": "string", "description": "Optional declaration name substring filter" },
                    "in_place": { "type": "boolean", "description": "Write verified proofs directly back to source files" }
                },
                "required": ["path"]
            }
        },
        {
            "name": "juice_formalize",
            "description": "Autoformalize natural-language or LaTeX mathematical statements into non-vacuous, back-translation-checked, kernel-verified Lean 4 proofs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "claim": { "type": "string", "description": "Natural-language math problem or LaTeX theorem snippet" },
                    "prove": { "type": "boolean", "description": "Whether to prove the formalized statement (default true)" },
                    "output_path": { "type": "string", "description": "Optional output `.lean` file path" }
                },
                "required": ["claim"]
            }
        },
        {
            "name": "juice_audit",
            "description": "Run a cold out-of-process `lake env lean` + `#print axioms` anti-cheat audit guaranteeing zero `sorryAx` and no unauthorized axioms.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to `.lean` file to audit" },
                    "allow_native_decide": { "type": "boolean", "description": "Allow Lean.trustCompiler / native_decide axioms" }
                },
                "required": ["path"]
            }
        },
        {
            "name": "juice_premise_search",
            "description": "Search local workspace and Mathlib theorem/lemma declarations so Lean 4 proofs use real, kernel-verified identifier names.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Theorem name keywords or mathematical type signature tokens" },
                    "workspace_path": { "type": "string", "description": "Optional path to Lean project" },
                    "limit": { "type": "integer", "description": "Max results (default 10)" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "juice_check_goal",
            "description": "Typecheck a Lean 4 snippet in the warm LSP worker pool and return open tactic goal states (at `sorry`) or compiler diagnostics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "lean_code": { "type": "string", "description": "Lean 4 source snippet" },
                    "workspace_path": { "type": "string", "description": "Optional path to Lean project" }
                },
                "required": ["lean_code"]
            }
        },
        {
            "name": "juice_verify_code",
            "description": "Formally verify Rust, C, Python, or TypeScript functions using Z3 SMT-LIB2 + Lean 4 + live runtime crash replay. Automatically checks integer overflow, division by zero, array bounds, panics, `@requires`/`@ensures` contracts, functional equivalence, and optionally synthesizes formally verified bug fixes (`fix: true`).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to `.rs`, `.c`, `.py`, `.ts` file or directory" },
                    "code": { "type": "string", "description": "Inline source code snippet (when `path` is omitted)" },
                    "language": { "type": "string", "description": "Language for inline `code`: `rust`, `c`, `python`, or `typescript`" },
                    "function": { "type": "string", "description": "Optional function name substring filter" },
                    "spec": { "type": "string", "description": "Optional postcondition rule (e.g. `result >= lo && result <= hi`)" },
                    "equiv": { "type": "string", "description": "Optional equivalence pair `fast_fn=naive_fn`" },
                    "fix": { "type": "boolean", "description": "Synthesize a formally verified patch if a bug is found" },
                    "in_place": { "type": "boolean", "description": "Write verified patch directly to `path`" },
                    "emit_test": { "type": "boolean", "description": "Emit a native regression unit test for counterexamples" },
                    "unwind": { "type": "integer", "description": "Loop/recursion unwinding bound (default 32)" },
                    "replay_runtime": { "type": "boolean", "description": "Replay Z3 counterexamples on real compiler/runtime (default true)" }
                }
            }
        }
    ])
}

async fn read_mcp_message<R: AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> anyhow::Result<Option<(Value, FramingMode)>> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed
            .strip_prefix("Content-Length:")
            .or_else(|| trimmed.strip_prefix("content-length:"))
        {
            let len: usize = rest.trim().parse()?;
            // Consume remaining headers until blank line.
            loop {
                line.clear();
                let hn = reader.read_line(&mut line).await?;
                if hn == 0 || line.trim().is_empty() {
                    break;
                }
            }
            let mut body = vec![0u8; len];
            reader.read_exact(&mut body).await?;
            let val: Value = serde_json::from_slice(&body)?;
            return Ok(Some((val, FramingMode::ContentLength)));
        }
        if trimmed.starts_with('{') {
            let val: Value = serde_json::from_str(trimmed)?;
            return Ok(Some((val, FramingMode::NewlineDelimited)));
        }
    }
}

async fn write_mcp_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    val: &Value,
    framing: FramingMode,
) -> anyhow::Result<()> {
    match framing {
        FramingMode::NewlineDelimited => {
            let s = serde_json::to_string(val)?;
            writer.write_all(s.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }
        FramingMode::ContentLength => {
            let bytes = serde_json::to_vec(val)?;
            let hdr = format!("Content-Length: {}\r\n\r\n", bytes.len());
            writer.write_all(hdr.as_bytes()).await?;
            writer.write_all(&bytes).await?;
        }
    }
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mcp_server_handles_initialize_tools_list_and_premise_search() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Sample.lean"),
            "theorem custom_comm (a b : Nat) : a + b = b + a := by sorry\n",
        )
        .unwrap();

        let (mut client_w, server_r) = tokio::io::duplex(64 * 1024);
        let (server_w, client_r) = tokio::io::duplex(64 * 1024);

        let cfg = JuiceConfig::default();
        let server_task = tokio::spawn(async move {
            run_mcp_on_streams(server_r, server_w, cfg).await.unwrap();
        });

        let mut client_reader = BufReader::new(client_r);

        // 1. initialize
        client_w
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n")
            .await
            .unwrap();
        let mut resp_line = String::new();
        client_reader.read_line(&mut resp_line).await.unwrap();
        let init_json: Value = serde_json::from_str(&resp_line).unwrap();
        assert_eq!(init_json["result"]["serverInfo"]["name"], "juice");

        // 2. tools/list
        client_w
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n")
            .await
            .unwrap();
        resp_line.clear();
        client_reader.read_line(&mut resp_line).await.unwrap();
        let tools_json: Value = serde_json::from_str(&resp_line).unwrap();
        let tools_arr = tools_json["result"]["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 6);

        // 3. tools/call -> juice_premise_search
        let call_req = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "juice_premise_search",
                "arguments": {
                    "query": "custom_comm",
                    "workspace_path": dir.path().to_string_lossy()
                }
            }
        });
        client_w
            .write_all(format!("{}\n", serde_json::to_string(&call_req).unwrap()).as_bytes())
            .await
            .unwrap();
        resp_line.clear();
        client_reader.read_line(&mut resp_line).await.unwrap();
        let call_resp: Value = serde_json::from_str(&resp_line).unwrap();
        assert_eq!(call_resp["result"]["isError"], false);
        let text = call_resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("custom_comm"));

        // 4. tools/call -> juice_verify_code (inline Rust verification + auto-fix)
        if juice_smt::locate_z3(None).is_some() {
            let verify_req = json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "juice_verify_code",
                    "arguments": {
                        "language": "rust",
                        "code": "pub fn midpoint(lo: i32, hi: i32) -> i32 { (lo + hi) / 2 }",
                        "fix": true,
                        "replay_runtime": false
                    }
                }
            });
            client_w
                .write_all(format!("{}\n", serde_json::to_string(&verify_req).unwrap()).as_bytes())
                .await
                .unwrap();
            resp_line.clear();
            client_reader.read_line(&mut resp_line).await.unwrap();
            let verify_resp: Value = serde_json::from_str(&resp_line).unwrap();
            assert_eq!(verify_resp["result"]["isError"], false);
            let vtext = verify_resp["result"]["content"][0]["text"].as_str().unwrap();
            assert!(vtext.contains("unbounded-proven"));
        }

        drop(client_w);
        let _ = server_task.await;
    }
}
