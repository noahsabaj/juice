<div align="center">
  <img src="assets/logo.svg" alt="juice logo" width="128" height="128" />
  <h1><code>juice</code></h1>
  <p><strong>Open-source formal software verification, autoformalization, and Z3 + Lean 4 theorem-proving CLI.</strong></p>
  <p>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-orange.svg" alt="License: MIT OR Apache-2.0" /></a>
    <img src="https://img.shields.io/badge/z3-SMT--LIB2-purple.svg" alt="Z3 SMT-LIB2" />
    <img src="https://img.shields.io/badge/lean-4-blue.svg" alt="Lean 4" />
    <img src="https://img.shields.io/badge/rust-2024-dea584.svg" alt="Rust" />
  </p>
</div>

`juice` is an open-source truth compiler for code and mathematical reasoning. Point it at **Rust, C, Python, or TypeScript** source code (`juice verify`) or **Lean 4 / LaTeX / natural-language math** (`juice fill`, `juice prove`, `juice formalize`): every bug report is grounded by an exact **Z3** bitvector/integer counterexample replayed against your real compiler/runtime, and every proof or auto-repaired patch (`--fix`) is machine-certified across **100% of inputs**.

Dual-licensed under **[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)**.

---

## Key Capabilities

- **Multi-Language Software Formal Verification (`juice verify`)**:
  - Supports **Rust (`.rs`)**, **C (`.c`/`.h`)**, **Python (`.py`)**, and **TypeScript (`.ts`/`.js`)** out of the box—working **100% offline** via deterministic core AST lifting (with optional LLM fallback for complex syntax).
  - **Zero-Config Safety & Spec Proposing**: Automatically checks exact fixed-width integer overflow/underflow (`i8..i128`, `u8..u128`, C types), division/modulo by zero, bit-shift bounds, slice/pointer out-of-bounds indexing, `assert!` / `.unwrap()` panics, `@requires` / `@ensures` doc contracts, `--spec "..."` rules, and `--equiv fast_fn=naive_fn` equivalence proofs.
  - **Zero-False-Positive Live Runtime Crash Replay**: Replays every Z3 counterexample against `rustc -C debug-assertions=yes`, `clang -fsanitize=undefined,address`, `python3`, or `bun`/`node` to confirm real panics, UBSan/ASan traps, or spec divergences.
  - **3-Tier Loop & Recursion Engine**: Attempts unbounded Hoare-logic inductive loop invariants first, escalates recursive/inductive claims to **Lean 4**, and falls back cleanly to Bounded Model Checking (`--unwind K`, default `32`), distinguishing `[UNBOUNDED PROVEN ✓]` from `[BOUNDED VERIFIED ✓ depth=32]`.
  - **Formally Verified Auto-Repair (`--fix`) & Test/SARIF Generation (`--emit-test`, `--sarif`)**: Synthesizes bug fixes inside a Z3/Lean CEGIS loop—rejecting half-baked fixes (like signed `lo + (hi - lo) / 2` that still overflows `i32`) until a patch (like `(lo & hi) + ((lo ^ hi) >> 1)`) is proven across all inputs.
- **Lean 4 `sorry`-Filling (`juice fill`)**: Finds unfinished `sorry` placeholders across `.lean` files or Lake repositories, proves them via hierarchical Draft-Sketch-Prove + parallel Monte Carlo Graph Search (MCGS), and outputs unified diffs or patches files in-place.
- **Autoformalization (`juice formalize` & `juice prove`)**: Translates natural-language math problems, Markdown, or LaTeX (`.tex`) papers into non-vacuous, back-translation-verified, kernel-certified Lean 4 proofs.
- **Offline Premise Discovery (`juice search`) & Cold Anti-Cheat Audit (`juice audit`)**: Indexes local project + `Mathlib` declarations with `#check` filtering and certifies proofs out-of-process via `lake env lean` + `#print axioms`.
- **Built-in MCP Server (`juice mcp`)**: Exposes `juice_verify_code`, `juice_fill_sorry`, `juice_formalize`, `juice_audit`, `juice_premise_search`, and `juice_check_goal` over `stdio` JSON-RPC 2.0 for AI coding agents (Antigravity, Claude Code, Cursor, VS Code).

---

## Quickstart

### 1. Build & Install

```bash
cargo install --path crates/juice-cli
juice doctor
```

### 2. Verify & Auto-Repair Real Code (`juice verify`)

```bash
# Zero-config formal verification of Rust, C, Python, or TypeScript files/directories
juice verify src/binary_search.rs

# Synthesize a formally verified patch (proven across 100% of inputs) and preview unified diff
juice verify src/binary_search.rs --fix --diff

# Prove an optimized implementation equivalent to a reference implementation across all inputs
juice verify src/pricing.ts --equiv fast_discount=naive_discount --emit-test

# Check an explicit postcondition rule and emit OASIS SARIF v2.1.0 for GitHub Code Scanning / CI
juice verify src/allocator.c --spec "result >= 0" --sarif juice-results.sarif
```

### 3. Prove Math Claims, Fill Lean 4 `sorry`s, or Run CI Audits

```bash
# Bootstrap Lean 4 + precompiled Mathlib cache (~9,100 .olean files)
juice setup --yes

# Preview verified proofs for all `sorry` placeholders in a Lean file
juice fill MyTheorems.lean --diff

# Autoformalize and prove a standalone natural-language claim
juice prove "For all natural numbers n, n + 0 = n and 2 * n = n + n" --output EvenSum.lean

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
| `crates/juice-smt` | Pure-Rust Z3 `SMT-LIB2` bitvector/integer/array solver driver, portable widening overflow predicates, and S-expression counterexample model parser |
| `crates/juice-verify` | Multi-language (Rust, C, Python, TypeScript) parser/lifter, zero-config safety & spec proposer, 3-tier Z3/Lean/BMC engine, live runtime crash replay, verified `--fix` CEGIS loop, `--emit-test`, and SARIF v2.1.0 |
| `crates/juice-llm` | Native Anthropic, OpenAI, Gemini, and OpenAI-compatible clients + `planner`/`prover` dual-role router |
| `crates/juice-lean` | Persistent `lean --server` LSP pool, Unicode/UTF-16 `sorry` parser, offline `Mathlib` premise index, and cold `#print axioms` auditor |
| `crates/juice-search` | Tier-1 sketch planner, `plausible` counterexample loop, vacuity/back-translation guards, and parallel AND/OR MCGS |
| `crates/juice-mcp` | JSON-RPC 2.0 Model Context Protocol (`stdio`) server (`juice_verify_code`, `juice_fill_sorry`, `juice_formalize`, `juice_audit`, `juice_premise_search`, `juice_check_goal`) |
| `crates/juice-cli` | `juice` binary (`verify`, `fill`, `formalize`, `prove`, `audit`, `search`, `setup`, `doctor`, `config`, `mcp`) |

---

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
