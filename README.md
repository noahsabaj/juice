# `juice`

**Open-source formal reasoning, autoformalization, and Lean 4 theorem-proving CLI.**

`juice` pairs frontier/specialized language models with **Lean 4**'s deterministic kernel so every proof or autoformalized statement it outputs is machine-checked down to foundational axioms—with zero hallucinations.

Dual-licensed under **[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)**.

---

## Key Capabilities

- **`sorry`-Filling (`juice fill`)**: Finds unfinished `sorry` placeholders across `.lean` files or Lake repositories, proves them in dependency order, and outputs unified diffs or patches files in-place.
- **Autoformalization (`juice formalize` & `juice prove`)**: Translates natural-language math problems, Markdown, or LaTeX (`.tex`) papers into non-vacuous, back-translation-verified, kernel-certified Lean 4 proofs.
- **Two-Tier Hierarchical Search**:
  1. **Tier 1 (Draft-Sketch-Prove + `plausible` Counterexample Guard)**: Decomposes hard theorems into `have` sub-lemmas, verifies the outer sketch compiles first, and probes sub-lemmas for quick counterexamples before spending search tokens.
  2. **Tier 2 (Monte Carlo Graph Search + Zero-Cost Hammers)**: Explores deduplicated AND/OR proof-state graphs over a persistent `lean --server` LSP pool, trying built-in tactics (`rfl`, `decide`, `norm_num`, `ring`, `linarith`, `omega`, `simp`, `aesop`, `exact?`, `apply?`) automatically at every node.
- **Offline Premise Discovery (`juice search`)**: Indexes local project + `Mathlib` declarations and verifies candidate theorems with `#check` so models never hallucinate outdated `Mathlib` names.
- **Cold Anti-Cheat Kernel Audit (`juice audit`)**: Runs an isolated `lake env lean` + `#print axioms` check rejecting `sorryAx`, unauthorized `axiom` declarations, `unsafe` blocks, and statement tampering.
- **Built-in MCP Server (`juice mcp`)**: Exposes formal verification and `sorry`-filling tools over `stdio` JSON-RPC 2.0 for AI coding agents (Antigravity, Claude Code, Cursor, VS Code).

---

## Quickstart

### 1. Build & Install

```bash
cargo install --path crates/juice-cli
```

### 2. Check Toolchain & Bootstrap Standalone `Mathlib` Cache

```bash
juice doctor
juice setup --mathlib
```

### 3. Configure Models (Single Model or Dual-Role `planner` + `prover`)

Set an environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`), or create `~/.config/juice/config.toml` via `juice config init`:

```toml
[models.planner]
provider = "anthropic"
model = "claude-sonnet-4-5"
api_key_env = "ANTHROPIC_API_KEY"

# Optional specialized Lean 4 tactic prover (e.g. Ollama, vLLM, or Fireworks):
[models.prover]
provider = "openai-compat"
model = "Goedel-LM/Goedel-Prover-V2-8B"
base_url = "http://localhost:11434/v1"
```

### 4. Fill `sorry` Proofs, Autoformalize Papers, or Run CI Audits

```bash
# Preview verified proofs for all `sorry` placeholders in a Lean file
juice fill MyTheorems.lean --diff

# Patch verified proofs in-place
juice fill ./my-lean-project --in-place

# Autoformalize and prove a standalone natural-language claim
juice prove "Prove that for all natural numbers n, n + 0 = n and 2 * n = n + n" --output EvenSum.lean

# Autoformalize a LaTeX paper into verified Lean 4
juice formalize paper.tex --output FormalizedPaper.lean

# Strict CI anti-cheat axiom audit (exits non-zero on sorryAx or custom axioms)
juice audit MyTheorems.lean

# Launch MCP stdio server for AI coding assistants
juice mcp
```

---

## Workspace Architecture

| Crate | Purpose |
| :--- | :--- |
| `crates/juice-core` | Backend-agnostic `VerifierBackend` trait, config layering, search events, and resumable `.juice/runs` checkpoints |
| `crates/juice-llm` | Native Anthropic, OpenAI, Gemini, and OpenAI-compatible clients + `planner`/`prover` dual-role router |
| `crates/juice-lean` | Persistent `lean --server` LSP pool, Unicode/UTF-16 `sorry` parser, offline `Mathlib` premise index, and cold `#print axioms` auditor |
| `crates/juice-search` | Tier-1 sketch planner, `plausible` counterexample loop, vacuity/back-translation guards, and parallel AND/OR MCGS |
| `crates/juice-mcp` | JSON-RPC 2.0 Model Context Protocol (`stdio`) server |
| `crates/juice-cli` | `juice` binary with `indicatif` live progress UI, unified diff output, and `--json` CI mode |

---

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
