use crate::ast::{BinOp, Expr, UnaryOp, VerifyFunction};
use crate::engine::extract_array_elem;
use crate::lang::{SourceLanguage, VerifyType};
use juice_smt::SmtValue;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeDoctorReport {
    pub rustc: Option<String>,
    pub clang: Option<String>,
    pub python3: Option<String>,
    pub bun_or_node: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum RuntimeReplayVerdict {
    ConfirmedFailure { toolchain: String, trace: String },
    NoFailureObserved { output: String },
    Skipped,
    ToolchainUnavailable { reason: String },
}

fn find_binary(candidates: &[&str], fallbacks: &[&str]) -> Option<PathBuf> {
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            for name in candidates {
                let p = dir.join(name);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    for fb in fallbacks {
        let p = PathBuf::from(fb);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn query_version(bin: &PathBuf, arg: &str) -> Option<String> {
    std::process::Command::new(bin)
        .arg(arg)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or_default()
                .trim()
                .to_string()
        })
}

pub fn runtime_doctor() -> RuntimeDoctorReport {
    let rustc = find_binary(&["rustc"], &["/Users/nsabaj/.cargo/bin/rustc", "/usr/local/bin/rustc"])
        .and_then(|b| query_version(&b, "--version"));
    let clang = find_binary(&["clang", "gcc"], &["/usr/bin/clang", "/usr/bin/gcc"])
        .and_then(|b| query_version(&b, "--version"));
    let python3 = find_binary(
        &["python3"],
        &["/opt/homebrew/bin/python3", "/usr/bin/python3"],
    )
    .and_then(|b| query_version(&b, "--version"));
    let bun_or_node = find_binary(
        &["bun", "node"],
        &["/opt/homebrew/bin/bun", "/opt/homebrew/bin/node", "/usr/local/bin/node"],
    )
    .and_then(|b| query_version(&b, "--version"));

    RuntimeDoctorReport {
        rustc,
        clang,
        python3,
        bun_or_node,
    }
}

pub async fn replay_counterexample_on_runtime(
    func: &VerifyFunction,
    full_file_source: &str,
    bindings: &BTreeMap<String, SmtValue>,
    postconditions: &[Expr],
    equiv_other: Option<&VerifyFunction>,
) -> RuntimeReplayVerdict {
    match func.language {
        SourceLanguage::Rust => {
            replay_rust(func, full_file_source, bindings, postconditions, equiv_other).await
        }
        SourceLanguage::C => {
            replay_c(func, full_file_source, bindings, postconditions, equiv_other).await
        }
        SourceLanguage::Python => {
            replay_python(func, full_file_source, bindings, postconditions, equiv_other).await
        }
        SourceLanguage::TypeScript => {
            replay_ts(func, full_file_source, bindings, postconditions, equiv_other).await
        }
    }
}

fn render_expr_for_lang(expr: &Expr, lang: SourceLanguage) -> String {
    match expr {
        Expr::Var(v) => v.clone(),
        Expr::IntLit(n) => n.to_string(),
        Expr::BoolLit(b) => match lang {
            SourceLanguage::Python => {
                if *b {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            _ => b.to_string(),
        },
        Expr::NoneLit => match lang {
            SourceLanguage::Rust | SourceLanguage::Python => "None".to_string(),
            SourceLanguage::C => "NULL".to_string(),
            SourceLanguage::TypeScript => "null".to_string(),
        },
        Expr::SomeExpr(inner) => format!("Some({})", render_expr_for_lang(inner, lang)),
        Expr::Len { array } => match lang {
            SourceLanguage::Rust => format!("({}.len() as i128)", render_expr_for_lang(array, lang)),
            SourceLanguage::Python => format!("len({})", render_expr_for_lang(array, lang)),
            SourceLanguage::TypeScript => format!("{}.length", render_expr_for_lang(array, lang)),
            SourceLanguage::C => "0".to_string(),
        },
        Expr::Unary { op, expr: e, .. } => {
            let inner = render_expr_for_lang(e, lang);
            match (op, lang) {
                (UnaryOp::Neg, _) => format!("(-({inner}))"),
                (UnaryOp::Not, SourceLanguage::Python) => format!("(not ({inner}))"),
                (UnaryOp::Not, _) => format!("(!({inner}))"),
                (UnaryOp::BitNot, SourceLanguage::Rust) => format!("(!({inner}))"),
                (UnaryOp::BitNot, _) => format!("(~({inner}))"),
            }
        }
        Expr::Binary {
            op, left, right, ..
        } => {
            let l = render_expr_for_lang(left, lang);
            let r = render_expr_for_lang(right, lang);
            let op_s = match (op, lang) {
                (BinOp::Add | BinOp::WrappingAdd | BinOp::SaturatingAdd, _) => "+",
                (BinOp::Sub | BinOp::WrappingSub | BinOp::SaturatingSub, _) => "-",
                (BinOp::Mul | BinOp::WrappingMul, _) => "*",
                (BinOp::Div, SourceLanguage::Python) => "//",
                (BinOp::Div, _) => "/",
                (BinOp::Rem, _) => "%",
                (BinOp::BitAnd, _) => "&",
                (BinOp::BitOr, _) => "|",
                (BinOp::BitXor, _) => "^",
                (BinOp::Shl, _) => "<<",
                (BinOp::Shr, _) => ">>",
                (BinOp::Eq, SourceLanguage::TypeScript) => "===",
                (BinOp::Eq, _) => "==",
                (BinOp::Ne, SourceLanguage::TypeScript) => "!==",
                (BinOp::Ne, _) => "!=",
                (BinOp::Lt, _) => "<",
                (BinOp::Le, _) => "<=",
                (BinOp::Gt, _) => ">",
                (BinOp::Ge, _) => ">=",
                (BinOp::And, SourceLanguage::Python) => "and",
                (BinOp::And, _) => "&&",
                (BinOp::Or, SourceLanguage::Python) => "or",
                (BinOp::Or, _) => "||",
            };
            format!("(({l}) {op_s} ({r}))")
        }
        Expr::Index { array, index, .. } => format!(
            "{}[{}]",
            render_expr_for_lang(array, lang),
            render_expr_for_lang(index, lang)
        ),
        Expr::Cond {
            cond,
            then_expr,
            else_expr,
        } => {
            let c = render_expr_for_lang(cond, lang);
            let t = render_expr_for_lang(then_expr, lang);
            let e = render_expr_for_lang(else_expr, lang);
            match lang {
                SourceLanguage::Rust => format!("(if {c} {{ {t} }} else {{ {e} }})"),
                SourceLanguage::Python => format!("({t} if {c} else {e})"),
                _ => format!("({c} ? {t} : {e})"),
            }
        }
        _ => "true".to_string(),
    }
}

fn rust_type_name(ty: &VerifyType) -> &'static str {
    match ty {
        VerifyType::Bool => "bool",
        VerifyType::BitVec { width: 8, signed: true } => "i8",
        VerifyType::BitVec { width: 16, signed: true } => "i16",
        VerifyType::BitVec { width: 32, signed: true } => "i32",
        VerifyType::BitVec { width: 64, signed: true } => "i64",
        VerifyType::BitVec { width: 128, signed: true } => "i128",
        VerifyType::BitVec { width: 8, signed: false } => "u8",
        VerifyType::BitVec { width: 16, signed: false } => "u16",
        VerifyType::BitVec { width: 32, signed: false } => "u32",
        VerifyType::BitVec { width: 64, signed: false } => "u64",
        VerifyType::BitVec { width: 128, signed: false } => "u128",
        _ => "i32",
    }
}

pub(crate) fn format_rust_arg(
    param_name: &str,
    ty: &VerifyType,
    bindings: &BTreeMap<String, SmtValue>,
    raw_param_decl: Option<&str>,
) -> String {
    match ty {
        VerifyType::Bool => bindings
            .get(param_name)
            .and_then(|v| v.as_i128(false))
            .map(|n| (n != 0).to_string())
            .unwrap_or_else(|| "false".to_string()),
        VerifyType::BitVec { signed, .. } => {
            let rty = if raw_param_decl.map(|s| s.contains("usize")).unwrap_or(false) {
                "usize"
            } else if raw_param_decl.map(|s| s.contains("isize")).unwrap_or(false) {
                "isize"
            } else {
                rust_type_name(ty)
            };
            if *signed {
                let n = bindings
                    .get(param_name)
                    .and_then(|v| v.as_i128(true))
                    .unwrap_or(0);
                format!("({n}i128 as {rty})")
            } else {
                let u = match bindings.get(param_name) {
                    Some(SmtValue::BitVec { unsigned, .. }) => *unsigned,
                    Some(other) => other.as_i128(false).unwrap_or(0) as u128,
                    None => 0,
                };
                format!("({u}u128 as {rty})")
            }
        }
        VerifyType::OptionType(inner) => {
            let is_some = matches!(
                bindings.get(&format!("{param_name}__is_some")),
                Some(SmtValue::Bool(true))
            );
            if is_some {
                let val_key = format!("{param_name}__val");
                let inner_lit = format_rust_arg(&val_key, inner, bindings, None);
                format!("Some({inner_lit})")
            } else {
                "None".to_string()
            }
        }
        VerifyType::Slice { elem, fixed_len } => {
            let len = fixed_len
                .map(|n| n as i128)
                .or_else(|| {
                    bindings
                        .get(&format!("{param_name}__len"))
                        .and_then(|v| v.as_i128(true))
                })
                .unwrap_or(0)
                .clamp(0, 32) as usize;
            let elem_rty = rust_type_name(elem);
            let mut items = Vec::with_capacity(len);
            for idx in 0..len {
                let val_s = extract_array_elem(bindings.get(param_name), idx as i128, elem.is_signed());
                if elem.is_signed() {
                    items.push(format!("({val_s}i128 as {elem_rty})"));
                } else {
                    items.push(format!("({val_s}u128 as {elem_rty})"));
                }
            }
            if fixed_len.is_some() {
                format!("[{}]", items.join(", "))
            } else if raw_param_decl.map(|s| s.contains("Vec<")).unwrap_or(false) {
                format!("vec![{}]", items.join(", "))
            } else {
                format!("&[{}]", items.join(", "))
            }
        }
        _ => "0".to_string(),
    }
}

async fn run_cmd_timeout(mut cmd: Command, secs: u64) -> anyhow::Result<std::process::Output> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = cmd.spawn()?;
    let out = tokio::time::timeout(Duration::from_secs(secs), child.wait_with_output()).await??;
    Ok(out)
}

async fn replay_rust(
    func: &VerifyFunction,
    full_file_source: &str,
    bindings: &BTreeMap<String, SmtValue>,
    postconditions: &[Expr],
    equiv_other: Option<&VerifyFunction>,
) -> RuntimeReplayVerdict {
    let Some(rustc) = find_binary(
        &["rustc"],
        &["/Users/nsabaj/.cargo/bin/rustc", "/usr/local/bin/rustc"],
    ) else {
        return RuntimeReplayVerdict::ToolchainUnavailable {
            reason: "`rustc` not found".to_string(),
        };
    };

    let Ok(tmp) = tempfile::tempdir() else {
        return RuntimeReplayVerdict::Skipped;
    };

    // Ensure target functions are `pub` inside `mod target_mod`
    let exposed_src = full_file_source.replace("\nfn ", "\npub fn ");
    let exposed_src = if exposed_src.trim_start().starts_with("fn ") {
        format!("pub {}", exposed_src.trim_start())
    } else {
        exposed_src
    };

    let mut let_bindings = String::new();
    let mut call_args = Vec::new();
    for p in &func.params {
        let lit = format_rust_arg(&p.name, &p.ty, bindings, Some(&func.raw_source));
        let_bindings.push_str(&format!("    let {} = {};\n", p.name, lit));
        call_args.push(p.name.clone());
    }

    let mut checks = String::new();
    if let Some(other) = equiv_other {
        checks.push_str(&format!(
            "    let __other = target_mod::{}({});\n    assert_eq!(result, __other, \"equivalence violation\");\n",
            other.name,
            call_args.join(", ")
        ));
    }
    for post in postconditions {
        let rendered = render_expr_for_lang(post, SourceLanguage::Rust);
        checks.push_str(&format!(
            "    assert!({rendered}, \"postcondition violated: {rendered}\");\n"
        ));
    }

    let harness = format!(
        "#[allow(dead_code, unused)]\nmod target_mod {{\n{exposed_src}\n}}\n\nfn main() {{\n{let_bindings}    let result = target_mod::{}({});\n{checks}    println!(\"{{:?}}\", result);\n}}\n",
        func.name,
        call_args.join(", ")
    );

    let src_path = tmp.path().join("replay.rs");
    let bin_path = tmp.path().join("replay_bin");
    if std::fs::write(&src_path, harness).is_err() {
        return RuntimeReplayVerdict::Skipped;
    }

    let mut compile_cmd = Command::new(&rustc);
    compile_cmd
        .args(["--edition", "2021", "-C", "debug-assertions=yes", "-A", "warnings"])
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path);
    match run_cmd_timeout(compile_cmd, 10).await {
        Ok(out) if !out.status.success() => {
            return RuntimeReplayVerdict::ToolchainUnavailable {
                reason: format!(
                    "rustc harness compile skipped: {}",
                    String::from_utf8_lossy(&out.stderr).lines().next().unwrap_or_default()
                ),
            };
        }
        Err(e) => {
            return RuntimeReplayVerdict::ToolchainUnavailable {
                reason: format!("rustc invocation failed: {e}"),
            };
        }
        _ => {}
    }

    match run_cmd_timeout(Command::new(&bin_path), 5).await {
        Ok(run_out) if !run_out.status.success() => {
            let stderr = String::from_utf8_lossy(&run_out.stderr).trim().to_string();
            let lines: Vec<&str> = stderr.lines().collect();
            let trace = if let Some(idx) = lines.iter().position(|l| l.contains("panicked at")) {
                if let Some(next_line) = lines.get(idx + 1)
                    && !next_line.trim().starts_with("note:")
                {
                    format!("{} {}", lines[idx].trim(), next_line.trim())
                } else {
                    lines[idx].trim().to_string()
                }
            } else {
                stderr
            };
            RuntimeReplayVerdict::ConfirmedFailure {
                toolchain: "rustc (-C debug-assertions=yes)".to_string(),
                trace,
            }
        }
        Ok(run_out) => RuntimeReplayVerdict::NoFailureObserved {
            output: String::from_utf8_lossy(&run_out.stdout).trim().to_string(),
        },
        Err(e) => RuntimeReplayVerdict::ConfirmedFailure {
            toolchain: "rustc (-C debug-assertions=yes)".to_string(),
            trace: format!("execution timed out / aborted: {e}"),
        },
    }
}

async fn replay_c(
    func: &VerifyFunction,
    full_file_source: &str,
    bindings: &BTreeMap<String, SmtValue>,
    postconditions: &[Expr],
    equiv_other: Option<&VerifyFunction>,
) -> RuntimeReplayVerdict {
    let Some(clang) = find_binary(&["clang", "gcc"], &["/usr/bin/clang", "/usr/bin/gcc"]) else {
        return RuntimeReplayVerdict::ToolchainUnavailable {
            reason: "`clang` / `gcc` not found".to_string(),
        };
    };

    let Ok(tmp) = tempfile::tempdir() else {
        return RuntimeReplayVerdict::Skipped;
    };

    let sanitized_c = full_file_source.replace("int main(", "int __juice_orig_main(");
    let mut setup_lines = String::new();
    let mut call_args = Vec::new();

    for p in &func.params {
        match &p.ty {
            VerifyType::Bool => {
                let b = bindings
                    .get(&p.name)
                    .and_then(|v| v.as_i128(false))
                    .unwrap_or(0)
                    != 0;
                setup_lines.push_str(&format!("    bool {} = {};\n", p.name, b));
                call_args.push(p.name.clone());
            }
            VerifyType::BitVec { width, signed } => {
                let cty = match (*width, *signed) {
                    (8, true) => "int8_t",
                    (8, false) => "uint8_t",
                    (16, true) => "int16_t",
                    (16, false) => "uint16_t",
                    (32, true) => "int32_t",
                    (32, false) => "uint32_t",
                    (64, true) => "int64_t",
                    _ => "uint64_t",
                };
                if *signed {
                    let n = bindings
                        .get(&p.name)
                        .and_then(|v| v.as_i128(true))
                        .unwrap_or(0);
                    if n == i128::from(i64::MIN) {
                        setup_lines.push_str(&format!(
                            "    {cty} {} = ({cty})(-9223372036854775807LL - 1LL);\n",
                            p.name
                        ));
                    } else {
                        setup_lines
                            .push_str(&format!("    {cty} {} = ({cty})({n}LL);\n", p.name));
                    }
                } else {
                    let u = match bindings.get(&p.name) {
                        Some(SmtValue::BitVec { unsigned, .. }) => *unsigned,
                        Some(other) => other.as_i128(false).unwrap_or(0) as u128,
                        None => 0,
                    };
                    setup_lines
                        .push_str(&format!("    {cty} {} = ({cty})({u}ULL);\n", p.name));
                }
                call_args.push(p.name.clone());
            }
            VerifyType::Slice { elem, fixed_len } => {
                let len = fixed_len
                    .map(|n| n as i128)
                    .or_else(|| {
                        p.linked_len_param
                            .as_ref()
                            .and_then(|ln| bindings.get(ln))
                            .and_then(|v| v.as_i128(true))
                    })
                    .or_else(|| {
                        bindings
                            .get(&format!("{}__len", p.name))
                            .and_then(|v| v.as_i128(true))
                    })
                    .unwrap_or(0);
                let alloc_len = len.clamp(0, 32) as usize;
                let elem_cty = match elem.as_ref() {
                    VerifyType::BitVec { width: 8, signed: false } => "uint8_t",
                    VerifyType::BitVec { width: 8, signed: true } => "int8_t",
                    VerifyType::BitVec { width: 64, signed: true } => "int64_t",
                    VerifyType::BitVec { width: 64, signed: false } => "uint64_t",
                    VerifyType::BitVec { width: 32, signed: false } => "uint32_t",
                    _ => "int32_t",
                };
                setup_lines.push_str(&format!(
                    "    {elem_cty}* {} = ({elem_cty}*)malloc({alloc_len} * sizeof({elem_cty}));\n",
                    p.name
                ));
                for idx in 0..alloc_len {
                    let v = extract_array_elem(bindings.get(&p.name), idx as i128, elem.is_signed());
                    setup_lines.push_str(&format!(
                        "    {}[{idx}] = ({elem_cty})({v}LL);\n",
                        p.name
                    ));
                }
                call_args.push(p.name.clone());
            }
            _ => {
                setup_lines.push_str(&format!("    int {} = 0;\n", p.name));
                call_args.push(p.name.clone());
            }
        }
    }

    let mut checks = String::new();
    if let Some(other) = equiv_other {
        checks.push_str(&format!(
            "    if (result != {}({})) {{ fprintf(stderr, \"equivalence mismatch\\n\"); return 1; }}\n",
            other.name,
            call_args.join(", ")
        ));
    }
    for post in postconditions {
        let c_expr = render_expr_for_lang(post, SourceLanguage::C);
        checks.push_str(&format!(
            "    if (!({c_expr})) {{ fprintf(stderr, \"postcondition violated: {c_expr}\\n\"); return 1; }}\n"
        ));
    }

    let call_stmt = if func.return_ty == VerifyType::Unit {
        format!("    {}({});\n", func.name, call_args.join(", "))
    } else {
        format!(
            "    long long result = (long long){}({});\n{checks}    printf(\"%lld\\n\", result);\n",
            func.name,
            call_args.join(", ")
        )
    };

    let c_harness = format!(
        "#include <stdint.h>\n#include <stdbool.h>\n#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <assert.h>\n#include <limits.h>\n\n{sanitized_c}\n\nint main(void) {{\n{setup_lines}{call_stmt}    return 0;\n}}\n"
    );

    let src_path = tmp.path().join("replay.c");
    let bin_path = tmp.path().join("replay_c_bin");
    if std::fs::write(&src_path, c_harness).is_err() {
        return RuntimeReplayVerdict::Skipped;
    }

    // Try compiling with UBSan + ASan first, falling back to UBSan-only if ASan runtime is absent
    let mut compiled_with = "clang (-fsanitize=undefined,address)".to_string();
    let mut cc = Command::new(&clang);
    cc.args([
        "-O1",
        "-g",
        "-fsanitize=undefined,address",
        "-fno-sanitize-recover=all",
    ])
    .arg(&src_path)
    .arg("-o")
    .arg(&bin_path);

    let ok_asan = run_cmd_timeout(cc, 10)
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !ok_asan {
        compiled_with = "clang (-fsanitize=undefined)".to_string();
        let mut cc_ub = Command::new(&clang);
        cc_ub
            .args(["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"])
            .arg(&src_path)
            .arg("-o")
            .arg(&bin_path);
        match run_cmd_timeout(cc_ub, 10).await {
            Ok(o) if o.status.success() => {}
            Ok(o) => {
                return RuntimeReplayVerdict::ToolchainUnavailable {
                    reason: format!(
                        "clang harness compile skipped: {}",
                        String::from_utf8_lossy(&o.stderr).lines().next().unwrap_or_default()
                    ),
                };
            }
            Err(e) => {
                return RuntimeReplayVerdict::ToolchainUnavailable {
                    reason: format!("clang invocation failed: {e}"),
                };
            }
        }
    }

    match run_cmd_timeout(Command::new(&bin_path), 5).await {
        Ok(run_out) if !run_out.status.success() => {
            let stderr = String::from_utf8_lossy(&run_out.stderr).trim().to_string();
            let trace = stderr
                .lines()
                .find(|l| {
                    l.contains("runtime error:")
                        || l.contains("AddressSanitizer")
                        || l.contains("Assertion")
                        || l.contains("violated")
                        || l.contains("mismatch")
                })
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    if stderr.is_empty() {
                        format!("process terminated with {}", run_out.status)
                    } else {
                        stderr
                    }
                });
            RuntimeReplayVerdict::ConfirmedFailure {
                toolchain: compiled_with,
                trace,
            }
        }
        Ok(run_out) => RuntimeReplayVerdict::NoFailureObserved {
            output: String::from_utf8_lossy(&run_out.stdout).trim().to_string(),
        },
        Err(e) => RuntimeReplayVerdict::ConfirmedFailure {
            toolchain: compiled_with,
            trace: format!("execution aborted: {e}"),
        },
    }
}

async fn replay_python(
    func: &VerifyFunction,
    full_file_source: &str,
    bindings: &BTreeMap<String, SmtValue>,
    postconditions: &[Expr],
    equiv_other: Option<&VerifyFunction>,
) -> RuntimeReplayVerdict {
    let Some(py) = find_binary(
        &["python3"],
        &["/opt/homebrew/bin/python3", "/usr/bin/python3"],
    ) else {
        return RuntimeReplayVerdict::ToolchainUnavailable {
            reason: "`python3` not found".to_string(),
        };
    };

    let Ok(tmp) = tempfile::tempdir() else {
        return RuntimeReplayVerdict::Skipped;
    };

    let target_py = tmp.path().join("target_mod.py");
    if std::fs::write(&target_py, full_file_source).is_err() {
        return RuntimeReplayVerdict::Skipped;
    }

    let mut arg_lits = Vec::new();
    let mut assign_lines = String::new();
    for p in &func.params {
        let lit = match &p.ty {
            VerifyType::Bool => {
                if bindings
                    .get(&p.name)
                    .and_then(|v| v.as_i128(false))
                    .unwrap_or(0)
                    != 0
                {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            VerifyType::OptionType(inner) => {
                let is_some = matches!(
                    bindings.get(&format!("{}__is_some", p.name)),
                    Some(SmtValue::Bool(true))
                );
                if is_some {
                    bindings
                        .get(&format!("{}__val", p.name))
                        .map(|v| v.format_typed(inner.is_signed()))
                        .unwrap_or_else(|| "0".to_string())
                } else {
                    "None".to_string()
                }
            }
            VerifyType::Slice { elem, fixed_len } => {
                let len = fixed_len
                    .map(|n| n as i128)
                    .or_else(|| {
                        bindings
                            .get(&format!("{}__len", p.name))
                            .and_then(|v| v.as_i128(true))
                    })
                    .unwrap_or(0)
                    .clamp(0, 32) as usize;
                let items: Vec<String> = (0..len)
                    .map(|idx| extract_array_elem(bindings.get(&p.name), idx as i128, elem.is_signed()))
                    .collect();
                // Disallow Python negative wraparound `arr[-1]` when formal model flagged `idx < 0` OOB
                format!("_StrictList([{}])", items.join(", "))
            }
            other => bindings
                .get(&p.name)
                .map(|v| v.format_typed(other.is_signed()))
                .unwrap_or_else(|| "0".to_string()),
        };
        assign_lines.push_str(&format!("{} = {}\n", p.name, lit));
        arg_lits.push(p.name.clone());
    }

    let mut checks = String::new();
    if let Some(other) = equiv_other {
        checks.push_str(&format!(
            "__other = target_mod.{}({})\nassert result == __other, f'equivalence mismatch: {{result}} != {{__other}}'\n",
            other.name,
            arg_lits.join(", ")
        ));
    }
    for post in postconditions {
        let py_e = render_expr_for_lang(post, SourceLanguage::Python);
        checks.push_str(&format!(
            "assert ({py_e}), 'postcondition violated: {py_e}'\n"
        ));
    }

    let runner = format!(
        r#"import sys
class _StrictList(list):
    def __getitem__(self, idx):
        if isinstance(idx, int) and (idx < 0 or idx >= len(self)):
            raise IndexError(f"list index {{idx}} out of bounds for length {{len(self)}}")
        return super().__getitem__(idx)

import target_mod
{assign_lines}result = target_mod.{}({})
{checks}print(result)
"#,
        func.name,
        arg_lits.join(", ")
    );

    let runner_py = tmp.path().join("replay.py");
    if std::fs::write(&runner_py, runner).is_err() {
        return RuntimeReplayVerdict::Skipped;
    }

    let mut cmd = Command::new(&py);
    cmd.current_dir(tmp.path()).arg(&runner_py);
    match run_cmd_timeout(cmd, 5).await {
        Ok(out) if !out.status.success() => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let last_line = stderr.lines().last().unwrap_or(&stderr).to_string();
            RuntimeReplayVerdict::ConfirmedFailure {
                toolchain: "python3".to_string(),
                trace: last_line,
            }
        }
        Ok(out) => RuntimeReplayVerdict::NoFailureObserved {
            output: String::from_utf8_lossy(&out.stdout).trim().to_string(),
        },
        Err(e) => RuntimeReplayVerdict::ConfirmedFailure {
            toolchain: "python3".to_string(),
            trace: format!("python3 execution aborted: {e}"),
        },
    }
}

async fn replay_ts(
    func: &VerifyFunction,
    full_file_source: &str,
    bindings: &BTreeMap<String, SmtValue>,
    postconditions: &[Expr],
    equiv_other: Option<&VerifyFunction>,
) -> RuntimeReplayVerdict {
    let bun = find_binary(&["bun"], &["/opt/homebrew/bin/bun"]);
    let node = find_binary(&["node"], &["/opt/homebrew/bin/node", "/usr/local/bin/node"]);
    if bun.is_none() && node.is_none() {
        return RuntimeReplayVerdict::ToolchainUnavailable {
            reason: "Neither `bun` nor `node` found".to_string(),
        };
    }

    let Ok(tmp) = tempfile::tempdir() else {
        return RuntimeReplayVerdict::Skipped;
    };

    let mut assign_lines = String::new();
    let mut arg_names = Vec::new();
    for p in &func.params {
        let lit = match &p.ty {
            VerifyType::Bool => {
                if bindings
                    .get(&p.name)
                    .and_then(|v| v.as_i128(false))
                    .unwrap_or(0)
                    != 0
                {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            VerifyType::OptionType(inner) => {
                let is_some = matches!(
                    bindings.get(&format!("{}__is_some", p.name)),
                    Some(SmtValue::Bool(true))
                );
                if is_some {
                    bindings
                        .get(&format!("{}__val", p.name))
                        .map(|v| v.format_typed(inner.is_signed()))
                        .unwrap_or_else(|| "0".to_string())
                } else {
                    "null".to_string()
                }
            }
            VerifyType::Slice { elem, fixed_len } => {
                let len = fixed_len
                    .map(|n| n as i128)
                    .or_else(|| {
                        bindings
                            .get(&format!("{}__len", p.name))
                            .and_then(|v| v.as_i128(true))
                    })
                    .unwrap_or(0)
                    .clamp(0, 32) as usize;
                let items: Vec<String> = (0..len)
                    .map(|idx| extract_array_elem(bindings.get(&p.name), idx as i128, elem.is_signed()))
                    .collect();
                format!("__strictArray([{}])", items.join(", "))
            }
            other => bindings
                .get(&p.name)
                .map(|v| v.format_typed(other.is_signed()))
                .unwrap_or_else(|| "0".to_string()),
        };
        assign_lines.push_str(&format!("  const {} = {};\n", p.name, lit));
        arg_names.push(p.name.clone());
    }

    let mut checks = String::new();
    if func.return_ty == VerifyType::Int {
        checks.push_str(
            "  if (typeof result === 'number' && !Number.isFinite(result)) { throw new RangeError(`Non-finite numeric result (division by zero / NaN): ${result}`); }\n",
        );
    }
    if let Some(other) = equiv_other {
        checks.push_str(&format!(
            "  const __other = {}({});\n  if (result !== __other) {{ throw new Error(`Equivalence mismatch: ${{result}} !== ${{__other}}`); }}\n",
            other.name,
            arg_names.join(", ")
        ));
    }
    for post in postconditions {
        let ts_e = render_expr_for_lang(post, SourceLanguage::TypeScript);
        checks.push_str(&format!(
            "  if (!({ts_e})) {{ throw new Error('Postcondition violated: {ts_e}'); }}\n"
        ));
    }

    let harness = format!(
        r#"{full_file_source}

function __strictArray<T>(arr: T[]): T[] {{
  return new Proxy(arr, {{
    get(target, prop, receiver) {{
      if (typeof prop === "string" && /^-?\d+$/.test(prop)) {{
        const idx = Number(prop);
        if (idx < 0 || idx >= target.length) {{
          throw new RangeError(`Array index ${{idx}} out of bounds for length ${{target.length}}`);
        }}
      }}
      return Reflect.get(target, prop, receiver);
    }}
  }});
}}

(() => {{
{assign_lines}  const result = {}({});
{checks}  console.log(result);
}})();
"#,
        func.name,
        arg_names.join(", ")
    );

    let replay_ts = tmp.path().join("replay.ts");
    if std::fs::write(&replay_ts, harness).is_err() {
        return RuntimeReplayVerdict::Skipped;
    }

    let (toolchain_name, cmd) = if let Some(bun_bin) = bun {
        let mut c = Command::new(bun_bin);
        c.arg("run").arg(&replay_ts);
        ("bun", c)
    } else {
        let mut c = Command::new(node.unwrap());
        c.arg("--experimental-strip-types").arg(&replay_ts);
        ("node", c)
    };

    match run_cmd_timeout(cmd, 5).await {
        Ok(out) if !out.status.success() => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            let trace = stderr
                .lines()
                .map(str::trim)
                .find(|l| {
                    l.starts_with("error:")
                        || l.starts_with("Error:")
                        || l.starts_with("RangeError:")
                        || l.starts_with("TypeError:")
                })
                .map(ToString::to_string)
                .unwrap_or(stderr);
            RuntimeReplayVerdict::ConfirmedFailure {
                toolchain: toolchain_name.to_string(),
                trace,
            }
        }
        Ok(out) => RuntimeReplayVerdict::NoFailureObserved {
            output: String::from_utf8_lossy(&out.stdout).trim().to_string(),
        },
        Err(e) => RuntimeReplayVerdict::ConfirmedFailure {
            toolchain: toolchain_name.to_string(),
            trace: format!("{toolchain_name} execution aborted: {e}"),
        },
    }
}
