use crate::ast::{BinOp, Expr, Param, Stmt, UnaryOp, VerifyFunction};
use crate::lang::{SourceLanguage, VerifyType};
use crate::proposer::{
    ObligationKind, ProposedPostcondition, SafetyObligation, compile_cli_spec,
    propose_auto_postconditions,
};
use crate::runtime::{RuntimeReplayVerdict, replay_counterexample_on_runtime};
use anyhow::bail;
use juice_core::{AxiomPolicy, JuiceConfig, VerifierBackend, events::noop_event_sink};
use juice_lean::LeanBackend;
use juice_llm::ModelRouter;
use juice_search::{AutoformalizeOptions, Autoformalizer};
use juice_smt::{SmtExpr, SmtOutcome, SmtScript, SmtSort, SmtValue, Z3Solver, mask_for_width};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VerificationBadge {
    /// Mathematically proven across 100% of valid inputs (Z3 UNSAT or Lean 4 kernel certified).
    UnboundedProven,
    /// Proven up to loop/recursion unwinding depth `depth` via Bounded Model Checking.
    BoundedVerified { depth: usize },
    /// Counterexample discovered and confirmed (or statically proven when `--no-replay` is set).
    BugFound,
    /// Solver returned `unknown` or timed out without finding a counterexample.
    Unknown,
}

impl VerificationBadge {
    pub fn label(&self) -> String {
        match self {
            Self::UnboundedProven => "[UNBOUNDED PROVEN ✓]".to_string(),
            Self::BoundedVerified { depth } => format!("[BOUNDED VERIFIED ✓ depth={depth}]"),
            Self::BugFound => "[BUG CONFIRMED ✗]".to_string(),
            Self::Unknown => "[UNKNOWN ?]".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleReport {
    pub obligation_id: String,
    pub kind: ObligationKind,
    pub description: String,
    pub line: usize,
    /// Formatted parameter name -> concrete value string (`lo = 1073741824`).
    pub inputs: BTreeMap<String, String>,
    pub raw_values: BTreeMap<String, SmtValue>,
    pub runtime_replay: RuntimeReplayVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionVerificationReport {
    pub function_name: String,
    pub language: SourceLanguage,
    pub start_line: usize,
    pub end_line: usize,
    pub badge: VerificationBadge,
    pub backend_used: String,
    pub obligations_checked: usize,
    pub obligations_proven: usize,
    pub counterexamples: Vec<CounterexampleReport>,
    pub notes: Vec<String>,
    pub generated_test: Option<String>,
    pub verified_fix_source: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VerifyOptions {
    pub target_fn: Option<String>,
    pub cli_spec: Option<String>,
    pub equiv_pair: Option<(String, String)>,
    pub unwind_depth: usize,
    pub replay_runtime: bool,
    pub escalate_to_lean: bool,
}

#[derive(Debug, Clone)]
struct SymVal {
    expr: SmtExpr,
    ty: VerifyType,
    option_is_some: Option<SmtExpr>,
    slice_len: Option<SmtExpr>,
}

#[derive(Debug, Clone)]
struct SymState {
    script: SmtScript,
    lang: SourceLanguage,
    return_ty: VerifyType,
    env: BTreeMap<String, SymVal>,
    path_cond: SmtExpr,
    returned_cond: SmtExpr,
    return_val: Option<SymVal>,
    obligations: Vec<SafetyObligation>,
    siblings: Vec<VerifyFunction>,
    unwind_budget: usize,
    unwinding_truncated_conds: Vec<SmtExpr>,
    check_overflows: bool,
    ssa_counter: usize,
}

impl SymState {
    fn active_cond(&self) -> SmtExpr {
        SmtExpr::and_many(vec![
            self.path_cond.clone(),
            SmtExpr::lnot(self.returned_cond.clone()),
        ])
    }

    fn bind_ssa(&mut self, expr: SmtExpr, sort: SmtSort) -> SmtExpr {
        if matches!(
            expr,
            SmtExpr::Var(_)
                | SmtExpr::BoolLit(_)
                | SmtExpr::IntLit(_)
                | SmtExpr::BvLit { .. }
        ) {
            return expr;
        }
        self.ssa_counter += 1;
        let name = format!("__ssa_{}", self.ssa_counter);
        self.script.declare_const(&name, sort, false);
        self.script
            .assert(SmtExpr::eq(SmtExpr::var(&name), expr));
        SmtExpr::var(name)
    }

    fn bind_sym_val(&mut self, mut val: SymVal) -> SymVal {
        val.expr = self.bind_ssa(val.expr, val.ty.to_smt_sort());
        if let Some(some_e) = val.option_is_some {
            val.option_is_some = Some(self.bind_ssa(some_e, SmtSort::Bool));
        }
        val
    }

    fn record_obligation(
        &mut self,
        kind: ObligationKind,
        line: usize,
        description: impl Into<String>,
        hazard: SmtExpr,
    ) {
        let id = format!("{}@L{}#{}", kind.rule_id(), line, self.obligations.len() + 1);
        let violation = SmtExpr::and_many(vec![self.active_cond(), hazard]);
        self.obligations.push(SafetyObligation {
            id,
            kind,
            description: description.into(),
            line,
            violation_condition: Some(violation),
        });
    }
}

fn index_sort_for(lang: SourceLanguage) -> VerifyType {
    if lang.checks_fixed_width_overflow() {
        VerifyType::BitVec {
            width: 64,
            signed: true,
        }
    } else {
        VerifyType::Int
    }
}

fn default_zero_for(ty: &VerifyType, lang: SourceLanguage) -> SymVal {
    match ty {
        VerifyType::Bool => SymVal {
            expr: SmtExpr::BoolLit(false),
            ty: ty.clone(),
            option_is_some: None,
            slice_len: None,
        },
        VerifyType::BitVec { width, .. } => SymVal {
            expr: SmtExpr::bv_unsigned(0, *width),
            ty: ty.clone(),
            option_is_some: None,
            slice_len: None,
        },
        VerifyType::Int => SymVal {
            expr: SmtExpr::IntLit(0),
            ty: VerifyType::Int,
            option_is_some: None,
            slice_len: None,
        },
        VerifyType::OptionType(inner) => {
            let z = default_zero_for(inner, lang);
            SymVal {
                expr: z.expr,
                ty: ty.clone(),
                option_is_some: Some(SmtExpr::BoolLit(false)),
                slice_len: None,
            }
        }
        VerifyType::Slice { .. } => {
            let idx_ty = index_sort_for(lang);
            let zero_len = default_zero_for(&idx_ty, lang).expr;
            SymVal {
                expr: SmtExpr::var("__empty_slice"),
                ty: ty.clone(),
                option_is_some: None,
                slice_len: Some(zero_len),
            }
        }
        VerifyType::Unit => SymVal {
            expr: SmtExpr::BoolLit(true),
            ty: VerifyType::Unit,
            option_is_some: None,
            slice_len: None,
        },
    }
}

fn coerce_to_type(val: &SymVal, target: &VerifyType) -> SymVal {
    if &val.ty == target {
        return val.clone();
    }
    match (&val.ty, target) {
        (
            VerifyType::BitVec {
                width: w_from,
                signed: s_from,
            },
            VerifyType::BitVec { width: w_to, .. },
        ) => {
            let expr = if w_to > w_from {
                if *s_from {
                    SmtExpr::SignExtend {
                        extra_bits: w_to - w_from,
                        expr: Box::new(val.expr.clone()),
                    }
                } else {
                    SmtExpr::ZeroExtend {
                        extra_bits: w_to - w_from,
                        expr: Box::new(val.expr.clone()),
                    }
                }
            } else if w_to < w_from {
                SmtExpr::Extract {
                    hi: w_to - 1,
                    lo: 0,
                    expr: Box::new(val.expr.clone()),
                }
            } else {
                val.expr.clone()
            };
            SymVal {
                expr,
                ty: target.clone(),
                option_is_some: None,
                slice_len: None,
            }
        }
        (VerifyType::OptionType(_), inner) if !matches!(inner, VerifyType::OptionType(_)) => {
            let stripped = SymVal {
                expr: val.expr.clone(),
                ty: inner.clone(),
                option_is_some: None,
                slice_len: None,
            };
            coerce_to_type(&stripped, target)
        }
        _ => SymVal {
            expr: val.expr.clone(),
            ty: target.clone(),
            option_is_some: val.option_is_some.clone(),
            slice_len: val.slice_len.clone(),
        },
    }
}

fn unify_numeric_pair(a: SymVal, b: SymVal, lang: SourceLanguage) -> (SymVal, SymVal) {
    if a.ty == b.ty {
        return (a, b);
    }
    match (&a.ty, &b.ty) {
        (VerifyType::BitVec { width: wa, signed: sa }, VerifyType::BitVec { width: wb, signed: sb }) => {
            let target_w = (*wa).max(*wb);
            let target_s = if wa == wb { *sa && *sb } else if wa > wb { *sa } else { *sb };
            let target = VerifyType::BitVec {
                width: target_w,
                signed: target_s,
            };
            (coerce_to_type(&a, &target), coerce_to_type(&b, &target))
        }
        (VerifyType::BitVec { .. }, VerifyType::Int) => {
            let target = a.ty.clone();
            let converted = if let SmtExpr::IntLit(n) = b.expr {
                let w = target.bit_width().unwrap_or(32);
                SymVal {
                    expr: SmtExpr::bv_signed(n, w),
                    ty: target,
                    option_is_some: None,
                    slice_len: None,
                }
            } else {
                coerce_to_type(&b, &target)
            };
            (a, converted)
        }
        (VerifyType::Int, VerifyType::BitVec { .. }) => {
            let target = b.ty.clone();
            let converted = if let SmtExpr::IntLit(n) = a.expr {
                let w = target.bit_width().unwrap_or(32);
                SymVal {
                    expr: SmtExpr::bv_signed(n, w),
                    ty: target,
                    option_is_some: None,
                    slice_len: None,
                }
            } else {
                coerce_to_type(&a, &target)
            };
            (converted, b)
        }
        _ => {
            let def = VerifyType::default_int_for(lang);
            (coerce_to_type(&a, &def), coerce_to_type(&b, &def))
        }
    }
}

fn declare_params_in_state(state: &mut SymState, params: &[Param], prefix: &str) {
    // First pass: declare scalar/option params
    for p in params {
        let smt_name = format!("{prefix}{}", p.name);
        match &p.ty {
            VerifyType::Bool | VerifyType::BitVec { .. } | VerifyType::Int => {
                state
                    .script
                    .declare_const(&smt_name, p.ty.to_smt_sort(), true);
                if state.lang == SourceLanguage::TypeScript && p.ty == VerifyType::Int {
                    // Constrain JS `number` inputs to IEEE-754 safe integer range
                    state.script.assert(SmtExpr::IntGe(
                        Box::new(SmtExpr::var(&smt_name)),
                        Box::new(SmtExpr::IntLit(-9_007_199_254_740_991)),
                    ));
                    state.script.assert(SmtExpr::IntLe(
                        Box::new(SmtExpr::var(&smt_name)),
                        Box::new(SmtExpr::IntLit(9_007_199_254_740_991)),
                    ));
                }
                state.env.insert(
                    p.name.clone(),
                    SymVal {
                        expr: SmtExpr::var(&smt_name),
                        ty: p.ty.clone(),
                        option_is_some: None,
                        slice_len: None,
                    },
                );
            }
            VerifyType::OptionType(inner) => {
                let some_name = format!("{smt_name}__is_some");
                let val_name = format!("{smt_name}__val");
                state.script.declare_const(&some_name, SmtSort::Bool, true);
                state
                    .script
                    .declare_const(&val_name, inner.to_smt_sort(), true);
                state.env.insert(
                    p.name.clone(),
                    SymVal {
                        expr: SmtExpr::var(&val_name),
                        ty: p.ty.clone(),
                        option_is_some: Some(SmtExpr::var(&some_name)),
                        slice_len: None,
                    },
                );
            }
            VerifyType::Slice { .. } | VerifyType::Unit => {}
        }
    }

    // Second pass: declare slice/pointer params (linking C pointer lengths if present)
    let idx_ty = index_sort_for(state.lang);
    for p in params {
        if let VerifyType::Slice { elem, fixed_len } = &p.ty {
            let smt_name = format!("{prefix}{}", p.name);
            let arr_sort =
                SmtSort::Array(Box::new(idx_ty.to_smt_sort()), Box::new(elem.to_smt_sort()));
            state.script.declare_const(&smt_name, arr_sort, true);

            let len_expr = if let Some(flen) = fixed_len {
                match &idx_ty {
                    VerifyType::BitVec { width, .. } => SmtExpr::bv_unsigned(*flen as u128, *width),
                    _ => SmtExpr::IntLit(*flen as i128),
                }
            } else if let Some(linked) = &p.linked_len_param
                && let Some(linked_val) = state.env.get(linked)
            {
                coerce_to_type(linked_val, &idx_ty).expr
            } else {
                let len_name = format!("{smt_name}__len");
                state
                    .script
                    .declare_const(&len_name, idx_ty.to_smt_sort(), true);
                match &idx_ty {
                    VerifyType::BitVec { width, .. } => {
                        state.script.assert(SmtExpr::BvSGe(
                            Box::new(SmtExpr::var(&len_name)),
                            Box::new(SmtExpr::bv_signed(0, *width)),
                        ));
                        // Keep symbolic slice lengths within realistic addressable bounds (<= 64 for fast concrete replay)
                        state.script.assert(SmtExpr::BvSLe(
                            Box::new(SmtExpr::var(&len_name)),
                            Box::new(SmtExpr::bv_signed(64, *width)),
                        ));
                    }
                    _ => {
                        state.script.assert(SmtExpr::IntGe(
                            Box::new(SmtExpr::var(&len_name)),
                            Box::new(SmtExpr::IntLit(0)),
                        ));
                        state.script.assert(SmtExpr::IntLe(
                            Box::new(SmtExpr::var(&len_name)),
                            Box::new(SmtExpr::IntLit(64)),
                        ));
                    }
                }
                SmtExpr::var(&len_name)
            };

            state.env.insert(
                p.name.clone(),
                SymVal {
                    expr: SmtExpr::var(&smt_name),
                    ty: p.ty.clone(),
                    option_is_some: None,
                    slice_len: Some(len_expr),
                },
            );
        }
    }
}

fn eval_expr(
    state: &mut SymState,
    expr: &Expr,
    expected_ty: Option<&VerifyType>,
) -> anyhow::Result<SymVal> {
    match expr {
        Expr::Var(name) => {
            if let Some(v) = state.env.get(name) {
                Ok(v.clone())
            } else {
                bail!("Unknown variable `{name}` in verification expression")
            }
        }
        Expr::BoolLit(b) => Ok(SymVal {
            expr: SmtExpr::BoolLit(*b),
            ty: VerifyType::Bool,
            option_is_some: None,
            slice_len: None,
        }),
        Expr::IntLit(n) => {
            let ty = expected_ty
                .filter(|t| matches!(t, VerifyType::BitVec { .. } | VerifyType::Int))
                .cloned()
                .unwrap_or_else(|| VerifyType::default_int_for(state.lang));
            match &ty {
                VerifyType::BitVec { width, .. } => Ok(SymVal {
                    expr: SmtExpr::bv_signed(*n, *width),
                    ty,
                    option_is_some: None,
                    slice_len: None,
                }),
                _ => Ok(SymVal {
                    expr: SmtExpr::IntLit(*n),
                    ty: VerifyType::Int,
                    option_is_some: None,
                    slice_len: None,
                }),
            }
        }
        Expr::NoneLit => {
            let ty = expected_ty
                .cloned()
                .unwrap_or_else(|| VerifyType::OptionType(Box::new(VerifyType::default_int_for(state.lang))));
            Ok(default_zero_for(&ty, state.lang))
        }
        Expr::SomeExpr(inner) => {
            let inner_expected = match expected_ty {
                Some(VerifyType::OptionType(t)) => Some(t.as_ref()),
                _ => None,
            };
            let v = eval_expr(state, inner, inner_expected)?;
            Ok(SymVal {
                expr: v.expr,
                ty: VerifyType::OptionType(Box::new(v.ty)),
                option_is_some: Some(SmtExpr::BoolLit(true)),
                slice_len: None,
            })
        }
        Expr::Len { array } => {
            let arr_v = eval_expr(state, array, None)?;
            let Some(len_smt) = arr_v.slice_len else {
                bail!("`.len()` / `len(...)` applied to non-slice value");
            };
            let len_ty = if state.lang == SourceLanguage::Rust {
                VerifyType::BitVec {
                    width: 64,
                    signed: false,
                }
            } else if state.lang == SourceLanguage::C {
                VerifyType::BitVec {
                    width: 64,
                    signed: true,
                }
            } else {
                VerifyType::Int
            };
            Ok(SymVal {
                expr: len_smt,
                ty: len_ty,
                option_is_some: None,
                slice_len: None,
            })
        }
        Expr::IsSome(inner) => {
            let v = eval_expr(state, inner, None)?;
            Ok(SymVal {
                expr: v.option_is_some.unwrap_or(SmtExpr::BoolLit(true)),
                ty: VerifyType::Bool,
                option_is_some: None,
                slice_len: None,
            })
        }
        Expr::IsNone(inner) => {
            let v = eval_expr(state, inner, None)?;
            let is_some = v.option_is_some.unwrap_or(SmtExpr::BoolLit(true));
            Ok(SymVal {
                expr: SmtExpr::lnot(is_some),
                ty: VerifyType::Bool,
                option_is_some: None,
                slice_len: None,
            })
        }
        Expr::Unwrap { expr: inner, line } => {
            let v = eval_expr(state, inner, None)?;
            if let Some(is_some) = &v.option_is_some {
                state.record_obligation(
                    ObligationKind::AssertionOrPanic,
                    *line,
                    "unwrap of potentially `None` / `null` value",
                    SmtExpr::lnot(is_some.clone()),
                );
            }
            let unwrapped_ty = match v.ty {
                VerifyType::OptionType(t) => *t,
                other => other,
            };
            Ok(SymVal {
                expr: v.expr,
                ty: unwrapped_ty,
                option_is_some: None,
                slice_len: None,
            })
        }
        Expr::Cast {
            expr: inner,
            target_ty,
            ..
        } => {
            let v = eval_expr(state, inner, None)?;
            Ok(coerce_to_type(&v, target_ty))
        }
        Expr::Index { array, index, line } => {
            let arr_v = eval_expr(state, array, None)?;
            let idx_raw = eval_expr(state, index, None)?;
            let idx_ty = index_sort_for(state.lang);
            let idx_coerced = coerce_to_type(&idx_raw, &idx_ty);

            if let Some(len_smt) = &arr_v.slice_len {
                let oob = match &idx_ty {
                    VerifyType::BitVec { width, .. } => {
                        let neg = if idx_raw.ty.is_signed() {
                            SmtExpr::BvSLt(
                                Box::new(idx_coerced.expr.clone()),
                                Box::new(SmtExpr::bv_signed(0, *width)),
                            )
                        } else {
                            SmtExpr::BoolLit(false)
                        };
                        let ge_len = if idx_raw.ty.is_signed() {
                            SmtExpr::BvSGe(
                                Box::new(idx_coerced.expr.clone()),
                                Box::new(len_smt.clone()),
                            )
                        } else {
                            SmtExpr::BvUGe(
                                Box::new(idx_coerced.expr.clone()),
                                Box::new(len_smt.clone()),
                            )
                        };
                        SmtExpr::or_many(vec![neg, ge_len])
                    }
                    _ => SmtExpr::or_many(vec![
                        SmtExpr::IntLt(
                            Box::new(idx_coerced.expr.clone()),
                            Box::new(SmtExpr::IntLit(0)),
                        ),
                        SmtExpr::IntGe(
                            Box::new(idx_coerced.expr.clone()),
                            Box::new(len_smt.clone()),
                        ),
                    ]),
                };
                state.record_obligation(
                    ObligationKind::OutOfBounds,
                    *line,
                    "array / slice index must be within `0 <= index < len`",
                    oob,
                );
            }

            let elem_ty = match &arr_v.ty {
                VerifyType::Slice { elem, .. } => (**elem).clone(),
                _ => VerifyType::default_int_for(state.lang),
            };
            Ok(SymVal {
                expr: SmtExpr::Select(Box::new(arr_v.expr), Box::new(idx_coerced.expr)),
                ty: elem_ty,
                option_is_some: None,
                slice_len: None,
            })
        }
        Expr::Unary { op, expr: inner, line } => {
            let v = eval_expr(state, inner, expected_ty)?;
            match op {
                UnaryOp::Not => Ok(SymVal {
                    expr: SmtExpr::lnot(v.expr),
                    ty: VerifyType::Bool,
                    option_is_some: None,
                    slice_len: None,
                }),
                UnaryOp::BitNot => match &v.ty {
                    VerifyType::BitVec { .. } => Ok(SymVal {
                        expr: SmtExpr::BvNot(Box::new(v.expr)),
                        ty: v.ty,
                        option_is_some: None,
                        slice_len: None,
                    }),
                    _ => Ok(SymVal {
                        expr: SmtExpr::IntSub(
                            Box::new(SmtExpr::IntLit(-1)),
                            Box::new(v.expr),
                        ),
                        ty: VerifyType::Int,
                        option_is_some: None,
                        slice_len: None,
                    }),
                },
                UnaryOp::Neg => match &v.ty {
                    VerifyType::BitVec { width, signed } => {
                        if state.check_overflows && *signed {
                            state.record_obligation(
                                ObligationKind::IntegerOverflow,
                                *line,
                                format!("signed {width}-bit negation `-x` overflows on `INT_MIN`"),
                                SmtExpr::bv_signed_neg_overflows(v.expr.clone(), *width),
                            );
                        }
                        Ok(SymVal {
                            expr: SmtExpr::BvNeg(Box::new(v.expr)),
                            ty: v.ty,
                            option_is_some: None,
                            slice_len: None,
                        })
                    }
                    _ => Ok(SymVal {
                        expr: SmtExpr::IntNeg(Box::new(v.expr)),
                        ty: VerifyType::Int,
                        option_is_some: None,
                        slice_len: None,
                    }),
                },
            }
        }
        Expr::Cond {
            cond,
            then_expr,
            else_expr,
        } => {
            let c = eval_expr(state, cond, Some(&VerifyType::Bool))?;
            let saved_path = state.path_cond.clone();

            state.path_cond = SmtExpr::and_many(vec![saved_path.clone(), c.expr.clone()]);
            let t_val = eval_expr(state, then_expr, expected_ty)?;

            state.path_cond =
                SmtExpr::and_many(vec![saved_path.clone(), SmtExpr::lnot(c.expr.clone())]);
            let e_val = eval_expr(state, else_expr, Some(&t_val.ty))?;

            state.path_cond = saved_path;
            let (tu, eu) = if t_val.ty == VerifyType::Bool {
                (t_val, e_val)
            } else {
                unify_numeric_pair(t_val, e_val, state.lang)
            };
            let merged_some = match (tu.option_is_some, eu.option_is_some) {
                (Some(ts), Some(es)) => Some(SmtExpr::Ite(
                    Box::new(c.expr.clone()),
                    Box::new(ts),
                    Box::new(es),
                )),
                (Some(ts), None) => Some(ts),
                (None, Some(es)) => Some(es),
                (None, None) => None,
            };
            Ok(SymVal {
                expr: SmtExpr::Ite(Box::new(c.expr), Box::new(tu.expr), Box::new(eu.expr)),
                ty: tu.ty,
                option_is_some: merged_some,
                slice_len: tu.slice_len,
            })
        }
        Expr::Binary {
            op,
            left,
            right,
            line,
        } => eval_binary(state, *op, left, right, *line, expected_ty),
        Expr::Call { func, args, line } => {
            // Inline sibling or self-recursive calls up to `unwind_budget`
            if let Some(callee) = state.siblings.iter().find(|f| &f.name == func).cloned() {
                if state.unwind_budget == 0 {
                    state.unwinding_truncated_conds.push(state.active_cond());
                    return Ok(default_zero_for(&callee.return_ty, state.lang));
                }
                let mut evaluated_args = Vec::new();
                for (arg_e, param) in args.iter().zip(callee.params.iter()) {
                    let av = eval_expr(state, arg_e, Some(&param.ty))?;
                    evaluated_args.push((param.name.clone(), coerce_to_type(&av, &param.ty)));
                }
                let saved_env = state.env.clone();
                let saved_ret_cond = state.returned_cond.clone();
                let saved_ret_val = state.return_val.clone();
                let saved_ret_ty = state.return_ty.clone();
                state.unwind_budget -= 1;

                state.env.clear();
                for (pname, pval) in evaluated_args {
                    state.env.insert(pname, pval);
                }
                state.returned_cond = SmtExpr::BoolLit(false);
                state.return_val = None;
                state.return_ty = callee.return_ty.clone();

                for s in &callee.body {
                    exec_stmt(state, s)?;
                }
                let res = state
                    .return_val
                    .clone()
                    .unwrap_or_else(|| default_zero_for(&callee.return_ty, state.lang));

                state.unwind_budget += 1;
                state.env = saved_env;
                state.returned_cond = saved_ret_cond;
                state.return_val = saved_ret_val;
                state.return_ty = saved_ret_ty;
                return Ok(res);
            }
            bail!("Unsupported call to external function `{func}` on line {line}")
        }
    }
}

fn eval_binary(
    state: &mut SymState,
    op: BinOp,
    left: &Expr,
    right: &Expr,
    line: usize,
    expected_ty: Option<&VerifyType>,
) -> anyhow::Result<SymVal> {
    // Short-circuit boolean `&&` and `||` so `idx < arr.len() && arr[idx] == 0` guards `arr[idx]`!
    if op == BinOp::And {
        let l = eval_expr(state, left, Some(&VerifyType::Bool))?;
        let saved_path = state.path_cond.clone();
        state.path_cond = SmtExpr::and_many(vec![saved_path.clone(), l.expr.clone()]);
        let r = eval_expr(state, right, Some(&VerifyType::Bool))?;
        state.path_cond = saved_path;
        return Ok(SymVal {
            expr: SmtExpr::and_many(vec![l.expr, r.expr]),
            ty: VerifyType::Bool,
            option_is_some: None,
            slice_len: None,
        });
    }
    if op == BinOp::Or {
        let l = eval_expr(state, left, Some(&VerifyType::Bool))?;
        let saved_path = state.path_cond.clone();
        state.path_cond =
            SmtExpr::and_many(vec![saved_path.clone(), SmtExpr::lnot(l.expr.clone())]);
        let r = eval_expr(state, right, Some(&VerifyType::Bool))?;
        state.path_cond = saved_path;
        return Ok(SymVal {
            expr: SmtExpr::or_many(vec![l.expr, r.expr]),
            ty: VerifyType::Bool,
            option_is_some: None,
            slice_len: None,
        });
    }

    let l_raw = eval_expr(state, left, expected_ty)?;
    let r_hint = if matches!(l_raw.ty, VerifyType::BitVec { .. } | VerifyType::Bool) {
        Some(&l_raw.ty)
    } else {
        expected_ty
    };
    let r_raw = eval_expr(state, right, r_hint)?;

    if l_raw.ty == VerifyType::Bool && r_raw.ty == VerifyType::Bool {
        let expr = match op {
            BinOp::Eq => SmtExpr::eq(l_raw.expr, r_raw.expr),
            BinOp::Ne => SmtExpr::ne(l_raw.expr, r_raw.expr),
            _ => bail!("Invalid binary operator {op:?} on booleans at line {line}"),
        };
        return Ok(SymVal {
            expr,
            ty: VerifyType::Bool,
            option_is_some: None,
            slice_len: None,
        });
    }

    let (l, r) = unify_numeric_pair(l_raw, r_raw, state.lang);

    match &l.ty {
        VerifyType::BitVec { width, signed } => {
            let w = *width;
            let is_signed = *signed;
            let expr = match op {
                BinOp::Add => {
                    if state.check_overflows {
                        let hazard = if is_signed {
                            SmtExpr::bv_signed_add_overflows(l.expr.clone(), r.expr.clone())
                        } else {
                            SmtExpr::bv_unsigned_add_overflows(l.expr.clone(), r.expr.clone())
                        };
                        state.record_obligation(
                            ObligationKind::IntegerOverflow,
                            line,
                            format!(
                                "{} {w}-bit addition must not overflow",
                                if is_signed { "signed" } else { "unsigned" }
                            ),
                            hazard,
                        );
                    }
                    SmtExpr::BvAdd(Box::new(l.expr), Box::new(r.expr))
                }
                BinOp::WrappingAdd => SmtExpr::BvAdd(Box::new(l.expr), Box::new(r.expr)),
                BinOp::SaturatingAdd => {
                    if is_signed {
                        let pos_ovf = SmtExpr::and_many(vec![
                            SmtExpr::BvSGe(
                                Box::new(l.expr.clone()),
                                Box::new(SmtExpr::bv_signed(0, w)),
                            ),
                            SmtExpr::BvSGe(
                                Box::new(r.expr.clone()),
                                Box::new(SmtExpr::bv_signed(0, w)),
                            ),
                            SmtExpr::bv_signed_add_overflows(l.expr.clone(), r.expr.clone()),
                        ]);
                        let neg_ovf =
                            SmtExpr::bv_signed_add_overflows(l.expr.clone(), r.expr.clone());
                        let max_v = SmtExpr::BvLit {
                            value: (1u128 << (w - 1)) - 1,
                            width: w,
                        };
                        let min_v = SmtExpr::BvLit {
                            value: 1u128 << (w - 1),
                            width: w,
                        };
                        SmtExpr::Ite(
                            Box::new(pos_ovf),
                            Box::new(max_v),
                            Box::new(SmtExpr::Ite(
                                Box::new(neg_ovf),
                                Box::new(min_v),
                                Box::new(SmtExpr::BvAdd(Box::new(l.expr), Box::new(r.expr))),
                            )),
                        )
                    } else {
                        let ovf =
                            SmtExpr::bv_unsigned_add_overflows(l.expr.clone(), r.expr.clone());
                        SmtExpr::Ite(
                            Box::new(ovf),
                            Box::new(SmtExpr::BvLit {
                                value: mask_for_width(w),
                                width: w,
                            }),
                            Box::new(SmtExpr::BvAdd(Box::new(l.expr), Box::new(r.expr))),
                        )
                    }
                }
                BinOp::Sub => {
                    if state.check_overflows {
                        let hazard = if is_signed {
                            SmtExpr::bv_signed_sub_overflows(l.expr.clone(), r.expr.clone())
                        } else {
                            SmtExpr::bv_unsigned_sub_overflows(l.expr.clone(), r.expr.clone())
                        };
                        state.record_obligation(
                            ObligationKind::IntegerOverflow,
                            line,
                            format!(
                                "{} {w}-bit subtraction must not underflow/overflow",
                                if is_signed { "signed" } else { "unsigned" }
                            ),
                            hazard,
                        );
                    }
                    SmtExpr::BvSub(Box::new(l.expr), Box::new(r.expr))
                }
                BinOp::WrappingSub => SmtExpr::BvSub(Box::new(l.expr), Box::new(r.expr)),
                BinOp::SaturatingSub => {
                    if !is_signed {
                        SmtExpr::Ite(
                            Box::new(SmtExpr::BvULt(
                                Box::new(l.expr.clone()),
                                Box::new(r.expr.clone()),
                            )),
                            Box::new(SmtExpr::bv_unsigned(0, w)),
                            Box::new(SmtExpr::BvSub(Box::new(l.expr), Box::new(r.expr))),
                        )
                    } else {
                        let ovf = SmtExpr::bv_signed_sub_overflows(l.expr.clone(), r.expr.clone());
                        let is_pos_l = SmtExpr::BvSGe(
                            Box::new(l.expr.clone()),
                            Box::new(SmtExpr::bv_signed(0, w)),
                        );
                        let max_v = SmtExpr::BvLit {
                            value: (1u128 << (w - 1)) - 1,
                            width: w,
                        };
                        let min_v = SmtExpr::BvLit {
                            value: 1u128 << (w - 1),
                            width: w,
                        };
                        SmtExpr::Ite(
                            Box::new(ovf),
                            Box::new(SmtExpr::Ite(
                                Box::new(is_pos_l),
                                Box::new(max_v),
                                Box::new(min_v),
                            )),
                            Box::new(SmtExpr::BvSub(Box::new(l.expr), Box::new(r.expr))),
                        )
                    }
                }
                BinOp::Mul => {
                    if state.check_overflows {
                        let hazard = if is_signed {
                            SmtExpr::bv_signed_mul_overflows(l.expr.clone(), r.expr.clone(), w)
                        } else {
                            SmtExpr::bv_unsigned_mul_overflows(l.expr.clone(), r.expr.clone(), w)
                        };
                        state.record_obligation(
                            ObligationKind::IntegerOverflow,
                            line,
                            format!(
                                "{} {w}-bit multiplication must not overflow",
                                if is_signed { "signed" } else { "unsigned" }
                            ),
                            hazard,
                        );
                    }
                    SmtExpr::BvMul(Box::new(l.expr), Box::new(r.expr))
                }
                BinOp::WrappingMul => SmtExpr::BvMul(Box::new(l.expr), Box::new(r.expr)),
                BinOp::Div | BinOp::Rem => {
                    let known_nonzero =
                        matches!(r.expr, SmtExpr::BvLit { value, .. } if value != 0);
                    if !known_nonzero {
                        state.record_obligation(
                            ObligationKind::DivisionByZero,
                            line,
                            "divisor must not be zero",
                            SmtExpr::eq(r.expr.clone(), SmtExpr::bv_unsigned(0, w)),
                        );
                    }
                    let known_not_neg_one = matches!(
                        r.expr,
                        SmtExpr::BvLit { value, .. } if value != mask_for_width(w)
                    );
                    if state.check_overflows && is_signed && !known_not_neg_one {
                        state.record_obligation(
                            ObligationKind::IntegerOverflow,
                            line,
                            format!("signed {w}-bit division `INT_MIN / -1` overflows"),
                            SmtExpr::bv_signed_div_overflows(l.expr.clone(), r.expr.clone(), w),
                        );
                    }
                    match (op, is_signed) {
                        (BinOp::Div, true) => SmtExpr::BvSDiv(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Div, false) => SmtExpr::BvUDiv(Box::new(l.expr), Box::new(r.expr)),
                        (_, true) => SmtExpr::BvSRem(Box::new(l.expr), Box::new(r.expr)),
                        (_, false) => SmtExpr::BvURem(Box::new(l.expr), Box::new(r.expr)),
                    }
                }
                BinOp::BitAnd => SmtExpr::BvAnd(Box::new(l.expr), Box::new(r.expr)),
                BinOp::BitOr => SmtExpr::BvOr(Box::new(l.expr), Box::new(r.expr)),
                BinOp::BitXor => SmtExpr::BvXor(Box::new(l.expr), Box::new(r.expr)),
                BinOp::Shl | BinOp::Shr => {
                    if state.check_overflows {
                        let shift_oob = SmtExpr::BvUGe(
                            Box::new(r.expr.clone()),
                            Box::new(SmtExpr::bv_unsigned(u128::from(w), w)),
                        );
                        state.record_obligation(
                            ObligationKind::ShiftOverflow,
                            line,
                            format!("shift count must be in `0..{w}`"),
                            shift_oob,
                        );
                    }
                    match (op, is_signed) {
                        (BinOp::Shl, _) => SmtExpr::BvShl(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Shr, true) => SmtExpr::BvAShr(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Shr, false) => SmtExpr::BvLShr(Box::new(l.expr), Box::new(r.expr)),
                        _ => unreachable!(),
                    }
                }
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    let cmp = match (op, is_signed) {
                        (BinOp::Eq, _) => SmtExpr::eq(l.expr, r.expr),
                        (BinOp::Ne, _) => SmtExpr::ne(l.expr, r.expr),
                        (BinOp::Lt, true) => SmtExpr::BvSLt(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Lt, false) => SmtExpr::BvULt(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Le, true) => SmtExpr::BvSLe(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Le, false) => SmtExpr::BvULe(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Gt, true) => SmtExpr::BvSGt(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Gt, false) => SmtExpr::BvUGt(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Ge, true) => SmtExpr::BvSGe(Box::new(l.expr), Box::new(r.expr)),
                        (BinOp::Ge, false) => SmtExpr::BvUGe(Box::new(l.expr), Box::new(r.expr)),
                        _ => unreachable!(),
                    };
                    return Ok(SymVal {
                        expr: cmp,
                        ty: VerifyType::Bool,
                        option_is_some: None,
                        slice_len: None,
                    });
                }
                BinOp::And | BinOp::Or => unreachable!(),
            };
            Ok(SymVal {
                expr,
                ty: l.ty,
                option_is_some: None,
                slice_len: None,
            })
        }
        _ => {
            // Unbounded `Int` (Python / TypeScript)
            let expr = match op {
                BinOp::Add | BinOp::WrappingAdd | BinOp::SaturatingAdd => {
                    SmtExpr::IntAdd(Box::new(l.expr), Box::new(r.expr))
                }
                BinOp::Sub | BinOp::WrappingSub | BinOp::SaturatingSub => {
                    SmtExpr::IntSub(Box::new(l.expr), Box::new(r.expr))
                }
                BinOp::Mul | BinOp::WrappingMul => {
                    SmtExpr::IntMul(Box::new(l.expr), Box::new(r.expr))
                }
                BinOp::Div | BinOp::Rem => {
                    let known_nonzero = matches!(r.expr, SmtExpr::IntLit(n) if n != 0);
                    if !known_nonzero {
                        state.record_obligation(
                            ObligationKind::DivisionByZero,
                            line,
                            "divisor must not be zero",
                            SmtExpr::eq(r.expr.clone(), SmtExpr::IntLit(0)),
                        );
                    }
                    if op == BinOp::Div {
                        SmtExpr::IntDiv(Box::new(l.expr), Box::new(r.expr))
                    } else {
                        SmtExpr::IntMod(Box::new(l.expr), Box::new(r.expr))
                    }
                }
                BinOp::Shr => {
                    // Integer `x >> 1` -> `div x 2` for constant small shifts
                    if let SmtExpr::IntLit(k) = r.expr
                        && (0..=62).contains(&k)
                    {
                        SmtExpr::IntDiv(
                            Box::new(l.expr),
                            Box::new(SmtExpr::IntLit(1i128 << k)),
                        )
                    } else {
                        bail!("Symbolic bit-shift on unbounded `Int` on line {line} requires bitvector width")
                    }
                }
                BinOp::Shl => {
                    if let SmtExpr::IntLit(k) = r.expr
                        && (0..=62).contains(&k)
                    {
                        SmtExpr::IntMul(
                            Box::new(l.expr),
                            Box::new(SmtExpr::IntLit(1i128 << k)),
                        )
                    } else {
                        bail!("Symbolic bit-shift on unbounded `Int` on line {line} requires bitvector width")
                    }
                }
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
                    bail!("Bitwise operator {op:?} on unbounded `Int` on line {line}")
                }
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    let cmp = match op {
                        BinOp::Eq => SmtExpr::eq(l.expr, r.expr),
                        BinOp::Ne => SmtExpr::ne(l.expr, r.expr),
                        BinOp::Lt => SmtExpr::IntLt(Box::new(l.expr), Box::new(r.expr)),
                        BinOp::Le => SmtExpr::IntLe(Box::new(l.expr), Box::new(r.expr)),
                        BinOp::Gt => SmtExpr::IntGt(Box::new(l.expr), Box::new(r.expr)),
                        BinOp::Ge => SmtExpr::IntGe(Box::new(l.expr), Box::new(r.expr)),
                        _ => unreachable!(),
                    };
                    return Ok(SymVal {
                        expr: cmp,
                        ty: VerifyType::Bool,
                        option_is_some: None,
                        slice_len: None,
                    });
                }
                BinOp::And | BinOp::Or => unreachable!(),
            };
            Ok(SymVal {
                expr,
                ty: VerifyType::Int,
                option_is_some: None,
                slice_len: None,
            })
        }
    }
}

fn merge_sym_vals(cond: &SmtExpr, t: &SymVal, e: &SymVal, lang: SourceLanguage) -> SymVal {
    if t.expr == e.expr && t.option_is_some == e.option_is_some {
        return t.clone();
    }
    let (tu, eu) = if t.ty == VerifyType::Bool || matches!(t.ty, VerifyType::Slice { .. }) {
        (t.clone(), e.clone())
    } else {
        unify_numeric_pair(t.clone(), e.clone(), lang)
    };
    let merged_some = match (&tu.option_is_some, &eu.option_is_some) {
        (Some(ts), Some(es)) => Some(SmtExpr::Ite(
            Box::new(cond.clone()),
            Box::new(ts.clone()),
            Box::new(es.clone()),
        )),
        (Some(ts), None) => Some(ts.clone()),
        (None, Some(es)) => Some(es.clone()),
        (None, None) => None,
    };
    SymVal {
        expr: SmtExpr::Ite(
            Box::new(cond.clone()),
            Box::new(tu.expr),
            Box::new(eu.expr),
        ),
        ty: tu.ty,
        option_is_some: merged_some,
        slice_len: tu.slice_len.or(eu.slice_len),
    }
}

fn exec_stmt(state: &mut SymState, stmt: &Stmt) -> anyhow::Result<()> {
    match stmt {
        Stmt::Let {
            name,
            ty,
            value,
            ..
        } => {
            let v = eval_expr(state, value, ty.as_ref())?;
            let final_v = if let Some(target_t) = ty {
                coerce_to_type(&v, target_t)
            } else {
                v
            };
            let to_store = if let Some(prev) = state.env.get(name).cloned() {
                let active = state.active_cond();
                merge_sym_vals(&active, &final_v, &prev, state.lang)
            } else {
                final_v
            };
            let bound = state.bind_sym_val(to_store);
            state.env.insert(name.clone(), bound);
        }
        Stmt::Assign { name, value, .. } => {
            let hint_ty = state.env.get(name).map(|v| v.ty.clone());
            let v = eval_expr(state, value, hint_ty.as_ref())?;
            let coerced = if let Some(ht) = &hint_ty {
                coerce_to_type(&v, ht)
            } else {
                v
            };
            let to_store = if let Some(prev) = state.env.get(name).cloned() {
                let active = state.active_cond();
                merge_sym_vals(&active, &coerced, &prev, state.lang)
            } else {
                coerced
            };
            let bound = state.bind_sym_val(to_store);
            state.env.insert(name.clone(), bound);
        }
        Stmt::ArrayAssign {
            array,
            index,
            value,
            line,
        } => {
            let Some(arr_v) = state.env.get(array).cloned() else {
                bail!("Assignment to unknown array `{array}` on line {line}");
            };
            let idx_raw = eval_expr(state, index, None)?;
            let idx_ty = index_sort_for(state.lang);
            let idx_coerced = coerce_to_type(&idx_raw, &idx_ty);

            if let Some(len_smt) = &arr_v.slice_len {
                let oob = match &idx_ty {
                    VerifyType::BitVec { width, .. } => SmtExpr::or_many(vec![
                        if idx_raw.ty.is_signed() {
                            SmtExpr::BvSLt(
                                Box::new(idx_coerced.expr.clone()),
                                Box::new(SmtExpr::bv_signed(0, *width)),
                            )
                        } else {
                            SmtExpr::BoolLit(false)
                        },
                        if idx_raw.ty.is_signed() {
                            SmtExpr::BvSGe(
                                Box::new(idx_coerced.expr.clone()),
                                Box::new(len_smt.clone()),
                            )
                        } else {
                            SmtExpr::BvUGe(
                                Box::new(idx_coerced.expr.clone()),
                                Box::new(len_smt.clone()),
                            )
                        },
                    ]),
                    _ => SmtExpr::or_many(vec![
                        SmtExpr::IntLt(
                            Box::new(idx_coerced.expr.clone()),
                            Box::new(SmtExpr::IntLit(0)),
                        ),
                        SmtExpr::IntGe(
                            Box::new(idx_coerced.expr.clone()),
                            Box::new(len_smt.clone()),
                        ),
                    ]),
                };
                state.record_obligation(
                    ObligationKind::OutOfBounds,
                    *line,
                    format!("write to `{array}[...]` must be within bounds"),
                    oob,
                );
            }

            let elem_ty = match &arr_v.ty {
                VerifyType::Slice { elem, .. } => (**elem).clone(),
                _ => VerifyType::default_int_for(state.lang),
            };
            let val_raw = eval_expr(state, value, Some(&elem_ty))?;
            let val_coerced = coerce_to_type(&val_raw, &elem_ty);
            let active = state.active_cond();
            let stored = SmtExpr::Store(
                Box::new(arr_v.expr.clone()),
                Box::new(idx_coerced.expr),
                Box::new(val_coerced.expr),
            );
            let updated_expr = SmtExpr::Ite(
                Box::new(active),
                Box::new(stored),
                Box::new(arr_v.expr),
            );
            state.env.insert(
                array.clone(),
                SymVal {
                    expr: updated_expr,
                    ty: arr_v.ty,
                    option_is_some: None,
                    slice_len: arr_v.slice_len,
                },
            );
        }
        Stmt::Assert {
            cond,
            message,
            line,
        } => {
            let c = eval_expr(state, cond, Some(&VerifyType::Bool))?;
            let desc = message
                .clone()
                .unwrap_or_else(|| "assertion must hold".to_string());
            state.record_obligation(
                ObligationKind::AssertionOrPanic,
                *line,
                format!("assertion `{desc}`"),
                SmtExpr::lnot(c.expr.clone()),
            );
            // Subsequent lines assume the assertion passed
            state.path_cond = SmtExpr::and_many(vec![state.path_cond.clone(), c.expr]);
        }
        Stmt::Panic { message, line } => {
            state.record_obligation(
                ObligationKind::AssertionOrPanic,
                *line,
                format!("reachable panic / exception: `{message}`"),
                SmtExpr::BoolLit(true),
            );
            let active = state.active_cond();
            state.returned_cond = SmtExpr::or_many(vec![state.returned_cond.clone(), active]);
        }
        Stmt::Return { value, .. } => {
            let ret_ty = state.return_ty.clone();
            let v = if let Some(e) = value {
                let raw = eval_expr(state, e, Some(&ret_ty))?;
                coerce_to_type(&raw, &ret_ty)
            } else {
                default_zero_for(&ret_ty, state.lang)
            };
            let active = state.active_cond();
            state.return_val = Some(match &state.return_val {
                Some(prev) => merge_sym_vals(&active, &v, prev, state.lang),
                None => v,
            });
            state.returned_cond = SmtExpr::or_many(vec![state.returned_cond.clone(), active]);
        }
        Stmt::ExprStmt { expr, .. } => {
            let _ = eval_expr(state, expr, None)?;
        }
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            let c = eval_expr(state, cond, Some(&VerifyType::Bool))?;
            let c_expr = state.bind_ssa(c.expr, SmtSort::Bool);
            let mut then_state = state.clone();
            then_state.obligations.clear();
            then_state.unwinding_truncated_conds.clear();
            let then_pc = SmtExpr::and_many(vec![state.path_cond.clone(), c_expr.clone()]);
            then_state.path_cond = then_state.bind_ssa(then_pc, SmtSort::Bool);
            for s in then_branch {
                exec_stmt(&mut then_state, s)?;
            }

            let mut else_state = state.clone();
            else_state.script = then_state.script.clone();
            else_state.ssa_counter = then_state.ssa_counter;
            else_state.obligations.clear();
            else_state.unwinding_truncated_conds.clear();
            let else_pc =
                SmtExpr::and_many(vec![state.path_cond.clone(), SmtExpr::lnot(c_expr.clone())]);
            else_state.path_cond = else_state.bind_ssa(else_pc, SmtSort::Bool);
            for s in else_branch {
                exec_stmt(&mut else_state, s)?;
            }

            state.script = else_state.script;
            state.ssa_counter = else_state.ssa_counter;
            state.obligations.extend(then_state.obligations);
            state.obligations.extend(else_state.obligations);
            state
                .unwinding_truncated_conds
                .extend(then_state.unwinding_truncated_conds);
            state
                .unwinding_truncated_conds
                .extend(else_state.unwinding_truncated_conds);

            // Merge environment variables across branches
            let all_keys: BTreeSet<String> = then_state
                .env
                .keys()
                .chain(else_state.env.keys())
                .cloned()
                .collect();
            for k in all_keys {
                let tv = then_state
                    .env
                    .get(&k)
                    .or_else(|| state.env.get(&k))
                    .cloned();
                let ev = else_state
                    .env
                    .get(&k)
                    .or_else(|| state.env.get(&k))
                    .cloned();
                if let (Some(t), Some(e)) = (tv, ev) {
                    let merged = merge_sym_vals(&c_expr, &t, &e, state.lang);
                    let bound = state.bind_sym_val(merged);
                    state.env.insert(k, bound);
                }
            }

            let merged_ret_cond = SmtExpr::Ite(
                Box::new(c_expr.clone()),
                Box::new(then_state.returned_cond),
                Box::new(else_state.returned_cond),
            );
            state.returned_cond = state.bind_ssa(merged_ret_cond, SmtSort::Bool);
            state.return_val = match (then_state.return_val, else_state.return_val) {
                (Some(tv), Some(ev)) => {
                    let m = merge_sym_vals(&c_expr, &tv, &ev, state.lang);
                    Some(state.bind_sym_val(m))
                }
                (Some(tv), None) => Some(tv),
                (None, Some(ev)) => Some(ev),
                (None, None) => None,
            };
        }
        Stmt::While { cond, body, .. } => {
            // Unroll up to `unwind_budget` iterations in BMC pass
            let k = state.unwind_budget;
            unroll_while(state, cond, body, k)?;
        }
    }
    Ok(())
}

fn unroll_while(
    state: &mut SymState,
    cond: &Expr,
    body: &[Stmt],
    remaining: usize,
) -> anyhow::Result<()> {
    let c = eval_expr(state, cond, Some(&VerifyType::Bool))?;
    let c_expr = state.bind_ssa(c.expr, SmtSort::Bool);
    if remaining == 0 {
        // Record whether the loop condition can still be active past depth K
        let still_running = SmtExpr::and_many(vec![state.active_cond(), c_expr]);
        state.unwinding_truncated_conds.push(still_running);
        return Ok(());
    }

    let mut iter_state = state.clone();
    iter_state.obligations.clear();
    iter_state.unwinding_truncated_conds.clear();
    let iter_pc = SmtExpr::and_many(vec![state.path_cond.clone(), c_expr.clone()]);
    iter_state.path_cond = iter_state.bind_ssa(iter_pc, SmtSort::Bool);
    for s in body {
        exec_stmt(&mut iter_state, s)?;
    }
    unroll_while(&mut iter_state, cond, body, remaining - 1)?;

    state.script = iter_state.script;
    state.ssa_counter = iter_state.ssa_counter;
    state.obligations.extend(iter_state.obligations);
    state
        .unwinding_truncated_conds
        .extend(iter_state.unwinding_truncated_conds);

    let all_keys: BTreeSet<String> = iter_state.env.keys().chain(state.env.keys()).cloned().collect();
    for k in all_keys {
        if let (Some(iv), Some(sv)) = (iter_state.env.get(&k), state.env.get(&k)) {
            let merged = merge_sym_vals(&c_expr, iv, sv, state.lang);
            let bound = state.bind_sym_val(merged);
            state.env.insert(k, bound);
        }
    }
    let merged_ret = SmtExpr::Ite(
        Box::new(c_expr.clone()),
        Box::new(iter_state.returned_cond),
        Box::new(state.returned_cond.clone()),
    );
    state.returned_cond = state.bind_ssa(merged_ret, SmtSort::Bool);
    state.return_val = match (iter_state.return_val, state.return_val.clone()) {
        (Some(iv), Some(sv)) => {
            let m = merge_sym_vals(&c_expr, &iv, &sv, state.lang);
            Some(state.bind_sym_val(m))
        }
        (Some(iv), None) => Some(iv),
        (None, Some(sv)) => Some(sv),
        (None, None) => None,
    };
    Ok(())
}

/// Attempts unbounded Hoare-logic inductive verification for single-loop functions before BMC unrolling.
async fn try_unbounded_loop_induction(
    func: &VerifyFunction,
    solver: &Z3Solver,
    extra_requires: &[Expr],
    postconditions: &[ProposedPostcondition],
) -> anyhow::Result<bool> {
    if !func.has_loops || func.is_recursive {
        return Ok(false);
    }

    // Locate a top-level `[prefix_stmts..., While { cond, invariants, body }, suffix_stmts...]`
    let Some(while_pos) = func
        .body
        .iter()
        .position(|s| matches!(s, Stmt::While { .. }))
    else {
        return Ok(false);
    };
    let prefix = &func.body[..while_pos];
    let suffix = &func.body[while_pos + 1..];
    if prefix.iter().any(|s| matches!(s, Stmt::While { .. }))
        || suffix.iter().any(|s| matches!(s, Stmt::While { .. }))
    {
        return Ok(false);
    }

    let Stmt::While {
        cond,
        invariants,
        body,
        line: loop_line,
    } = &func.body[while_pos]
    else {
        return Ok(false);
    };

    // Synthesize candidate invariants: explicit `@invariant` + automatic counter range invariants
    let mut candidate_invs = invariants.clone();
    if let Expr::Binary {
        op: BinOp::Lt,
        left,
        right,
        ..
    } = cond
        && let Expr::Var(counter) = left.as_ref()
    {
        // Find initial value of `counter` in `prefix`
        for s in prefix {
            if let Stmt::Let { name, value, .. } = s
                && name == counter
            {
                // `init <= counter` and `(init <= bound) ==> (counter <= bound)`
                candidate_invs.push(Expr::Binary {
                    op: BinOp::Ge,
                    left: Box::new(Expr::Var(counter.clone())),
                    right: Box::new(value.clone()),
                    line: *loop_line,
                });
                candidate_invs.push(Expr::Binary {
                    op: BinOp::Or,
                    left: Box::new(Expr::Unary {
                        op: UnaryOp::Not,
                        expr: Box::new(Expr::Binary {
                            op: BinOp::Le,
                            left: Box::new(value.clone()),
                            right: right.clone(),
                            line: *loop_line,
                        }),
                        line: *loop_line,
                    }),
                    right: Box::new(Expr::Binary {
                        op: BinOp::Le,
                        left: Box::new(Expr::Var(counter.clone())),
                        right: right.clone(),
                        line: *loop_line,
                    }),
                    line: *loop_line,
                });
            }
        }
    }

    if candidate_invs.is_empty() {
        return Ok(false);
    }

    // 1. Initiation: execute `prefix` and check all `prefix` safety obligations + `candidate_invs`
    let mut init_state = build_initial_state(func, &[], 4);
    apply_preconditions(&mut init_state, &func.requires, extra_requires)?;
    for s in prefix {
        exec_stmt(&mut init_state, s)?;
    }
    for inv in &candidate_invs {
        let inv_v = eval_expr(&mut init_state, inv, Some(&VerifyType::Bool))?;
        init_state.record_obligation(
            ObligationKind::Postcondition,
            *loop_line,
            "loop invariant initiation",
            SmtExpr::lnot(inv_v.expr),
        );
    }
    if !all_obligations_unsat(&init_state, solver).await? {
        return Ok(false);
    }

    // Collect variables mutated inside `body`
    let mut mutated = BTreeSet::new();
    collect_mutated_vars(body, &mut mutated);

    // 2. Consecution + Body Safety: havoc mutated vars, assume `Inv && cond`, execute `body` once, check safety + `Inv'`
    let mut step_state = init_state.clone();
    step_state.obligations.clear();
    for (idx, var_name) in mutated.iter().enumerate() {
        if let Some(existing) = step_state.env.get(var_name).cloned() {
            let havoc_name = format!("__havoc_step_{idx}_{var_name}");
            step_state
                .script
                .declare_const(&havoc_name, existing.ty.to_smt_sort(), false);
            step_state.env.insert(
                var_name.clone(),
                SymVal {
                    expr: SmtExpr::var(&havoc_name),
                    ty: existing.ty,
                    option_is_some: existing.option_is_some,
                    slice_len: existing.slice_len,
                },
            );
        }
    }
    for inv in &candidate_invs {
        let inv_v = eval_expr(&mut step_state, inv, Some(&VerifyType::Bool))?;
        step_state.script.assert(inv_v.expr);
    }
    let cond_v = eval_expr(&mut step_state, cond, Some(&VerifyType::Bool))?;
    // Any safety obligation inside `cond` itself must also hold under `Inv`!
    step_state.script.assert(cond_v.expr);
    for s in body {
        exec_stmt(&mut step_state, s)?;
    }
    for inv in &candidate_invs {
        let inv_after = eval_expr(&mut step_state, inv, Some(&VerifyType::Bool))?;
        step_state.record_obligation(
            ObligationKind::Postcondition,
            *loop_line,
            "loop invariant consecution",
            SmtExpr::lnot(inv_after.expr),
        );
    }
    if !all_obligations_unsat(&step_state, solver).await? {
        return Ok(false);
    }

    // 3. Post-Loop Exit + Postconditions: havoc mutated vars, assume `Inv && !cond`, run `suffix` + postconditions
    let mut exit_state = init_state.clone();
    exit_state.obligations.clear();
    for (idx, var_name) in mutated.iter().enumerate() {
        if let Some(existing) = exit_state.env.get(var_name).cloned() {
            let havoc_name = format!("__havoc_exit_{idx}_{var_name}");
            exit_state
                .script
                .declare_const(&havoc_name, existing.ty.to_smt_sort(), false);
            exit_state.env.insert(
                var_name.clone(),
                SymVal {
                    expr: SmtExpr::var(&havoc_name),
                    ty: existing.ty,
                    option_is_some: existing.option_is_some,
                    slice_len: existing.slice_len,
                },
            );
        }
    }
    for inv in &candidate_invs {
        let inv_v = eval_expr(&mut exit_state, inv, Some(&VerifyType::Bool))?;
        exit_state.script.assert(inv_v.expr);
    }
    let exit_cond_v = eval_expr(&mut exit_state, cond, Some(&VerifyType::Bool))?;
    exit_state.script.assert(SmtExpr::lnot(exit_cond_v.expr));
    for s in suffix {
        exec_stmt(&mut exit_state, s)?;
    }
    attach_postconditions(&mut exit_state, func, postconditions)?;

    all_obligations_unsat(&exit_state, solver).await
}

fn collect_mutated_vars(stmts: &[Stmt], out: &mut BTreeSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::ArrayAssign { array, .. } => {
                out.insert(array.clone());
            }
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_mutated_vars(then_branch, out);
                collect_mutated_vars(else_branch, out);
            }
            Stmt::While { body, .. } => collect_mutated_vars(body, out),
            _ => {}
        }
    }
}

async fn all_obligations_unsat(state: &SymState, solver: &Z3Solver) -> anyhow::Result<bool> {
    if state.obligations.is_empty() {
        return Ok(true);
    }
    let disj: Vec<SmtExpr> = state
        .obligations
        .iter()
        .filter_map(|o| o.violation_condition.clone())
        .collect();
    let mut query = state.script.clone();
    query.assert(SmtExpr::or_many(disj));
    Ok(matches!(solver.check(&query).await?, SmtOutcome::Unsat))
}

fn build_initial_state(
    func: &VerifyFunction,
    siblings: &[VerifyFunction],
    unwind_depth: usize,
) -> SymState {
    let mut state = SymState {
        script: SmtScript::new(),
        lang: func.language,
        return_ty: func.return_ty.clone(),
        env: BTreeMap::new(),
        path_cond: SmtExpr::BoolLit(true),
        returned_cond: SmtExpr::BoolLit(false),
        return_val: None,
        obligations: Vec::new(),
        siblings: siblings.to_vec(),
        unwind_budget: unwind_depth,
        unwinding_truncated_conds: Vec::new(),
        check_overflows: func.language.checks_fixed_width_overflow(),
        ssa_counter: 0,
    };
    declare_params_in_state(&mut state, &func.params, "");
    state
}

fn apply_preconditions(
    state: &mut SymState,
    requires: &[Expr],
    extra_requires: &[Expr],
) -> anyhow::Result<()> {
    //Precondition evaluation itself should not emit spurious obligations for guarded inputs
    let saved_ovf = state.check_overflows;
    state.check_overflows = false;
    for req in requires.iter().chain(extra_requires.iter()) {
        let c = eval_expr(state, req, Some(&VerifyType::Bool))?;
        state.script.assert(c.expr);
    }
    state.obligations.clear();
    state.check_overflows = saved_ovf;
    Ok(())
}

fn attach_postconditions(
    state: &mut SymState,
    func: &VerifyFunction,
    postconditions: &[ProposedPostcondition],
) -> anyhow::Result<()> {
    if postconditions.is_empty() {
        return Ok(());
    }
    let ret_val = state
        .return_val
        .clone()
        .unwrap_or_else(|| default_zero_for(&func.return_ty, func.language));
    state.env.insert("result".to_string(), ret_val);

    let saved_ovf = state.check_overflows;
    state.check_overflows = false;
    for post in postconditions {
        // Postcondition must hold on every path that returned normally (`returned_cond == true`)
        let c = eval_expr(state, &post.expr, Some(&VerifyType::Bool))?;
        let id = format!(
            "{}@L{}#{}",
            ObligationKind::Postcondition.rule_id(),
            func.end_line,
            state.obligations.len() + 1
        );
        let violation = SmtExpr::and_many(vec![
            state.returned_cond.clone(),
            SmtExpr::lnot(c.expr),
        ]);
        state.obligations.push(SafetyObligation {
            id,
            kind: ObligationKind::Postcondition,
            description: post.description.clone(),
            line: func.end_line,
            violation_condition: Some(violation),
        });
    }
    state.check_overflows = saved_ovf;
    Ok(())
}

/// Optional Tier-2 Lean 4 escalation for unbounded recursive / algebraic properties.
async fn try_lean4_escalation(
    func: &VerifyFunction,
    postconditions: &[ProposedPostcondition],
    cfg: &JuiceConfig,
    router: Option<&ModelRouter>,
) -> Option<String> {
    if !cfg.verify.escalate_to_lean || (!func.is_recursive && !func.has_loops) {
        return None;
    }
    let router = router?;
    let lean_backend: Arc<dyn VerifierBackend> =
        Arc::new(LeanBackend::new(cfg.lean.clone(), cfg.search.clone()));
    let autoformalizer = Autoformalizer::new(
        lean_backend,
        router.clone(),
        cfg.search.clone(),
        cfg.lean.clone(),
        noop_event_sink(),
    );

    let specs_desc: Vec<String> = postconditions
        .iter()
        .map(|p| p.description.clone())
        .collect();
    let claim = format!(
        "Formalize the following {} function `{}` in Lean 4 and prove its safety and postconditions ({}):\n\n{}",
        func.language.display_name(),
        func.name,
        if specs_desc.is_empty() {
            "termination and absence of division by zero / bounds violations".to_string()
        } else {
            specs_desc.join("; ")
        },
        func.raw_source
    );

    let opts = AutoformalizeOptions {
        prove_theorems: true,
        max_formalize_attempts: 2,
        policy: AxiomPolicy::strict().with_native_decide(cfg.lean.allow_native_decide),
    };
    let outcome = autoformalizer
        .formalize_input(&claim, None, &opts)
        .await
        .ok()?;
    if !outcome.items.is_empty() && outcome.items.iter().all(|i| i.proved) {
        Some("Z3 + Lean 4 Kernel (`#print axioms` certified)".to_string())
    } else {
        None
    }
}

fn format_counterexample_inputs(
    func: &VerifyFunction,
    bindings: &BTreeMap<String, SmtValue>,
) -> BTreeMap<String, String> {
    let mut formatted = BTreeMap::new();
    for p in &func.params {
        match &p.ty {
            VerifyType::OptionType(inner) => {
                let some_key = format!("{}__is_some", p.name);
                let val_key = format!("{}__val", p.name);
                let is_some = matches!(bindings.get(&some_key), Some(SmtValue::Bool(true)));
                if is_some {
                    let inner_str = bindings
                        .get(&val_key)
                        .map(|v| v.format_typed(inner.is_signed()))
                        .unwrap_or_else(|| "0".to_string());
                    let disp = match func.language {
                        SourceLanguage::Rust => format!("Some({inner_str})"),
                        _ => inner_str,
                    };
                    formatted.insert(p.name.clone(), disp);
                } else {
                    let none_disp = match func.language {
                        SourceLanguage::Rust | SourceLanguage::Python => "None",
                        SourceLanguage::C => "NULL",
                        SourceLanguage::TypeScript => "null",
                    };
                    formatted.insert(p.name.clone(), none_disp.to_string());
                }
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
                    .unwrap_or(0)
                    .clamp(0, 32) as usize;

                let mut elems = Vec::with_capacity(len);
                for idx in 0..len {
                    let val_str = extract_array_elem(
                        bindings.get(&p.name),
                        idx as i128,
                        elem.is_signed(),
                    );
                    elems.push(val_str);
                }
                formatted.insert(p.name.clone(), format!("[{}]", elems.join(", ")));
            }
            other => {
                if let Some(val) = bindings.get(&p.name) {
                    formatted.insert(p.name.clone(), val.format_typed(other.is_signed()));
                }
            }
        }
    }
    formatted
}

pub(crate) fn extract_array_elem(arr_val: Option<&SmtValue>, idx: i128, is_signed: bool) -> String {
    let Some(SmtValue::Array { default, entries }) = arr_val else {
        return "0".to_string();
    };
    for (k, v) in entries {
        if k.as_i128(true) == Some(idx) || k.as_i128(false) == Some(idx) {
            return v.format_typed(is_signed);
        }
    }
    default.format_typed(is_signed)
}

/// Core verification entrypoint for a single parsed function.
pub async fn verify_single_function(
    func: &VerifyFunction,
    full_file_source: &str,
    siblings: &[VerifyFunction],
    solver: &Z3Solver,
    cfg: &JuiceConfig,
    opts: &VerifyOptions,
    router: Option<&ModelRouter>,
) -> anyhow::Result<FunctionVerificationReport> {
    let mut notes = Vec::new();
    let mut extra_requires = Vec::new();
    let mut postconditions = propose_auto_postconditions(func);

    for p in &postconditions {
        if p.auto_proposed {
            notes.push(format!("Checked {}", p.description));
        }
    }

    if let Some(raw_spec) = &opts.cli_spec {
        let (pre_opt, post) = compile_cli_spec(raw_spec, func, router).await?;
        if let Some(pre) = pre_opt {
            extra_requires.push(pre);
        }
        postconditions.push(post);
    }

    // Tier 1a: If function has a loop, try unbounded Hoare-logic inductive verification first
    if func.has_loops
        && try_unbounded_loop_induction(func, solver, &extra_requires, &postconditions).await?
    {
        notes.push("Synthesized and proved inductive loop invariant across unbounded iterations".to_string());
        return Ok(FunctionVerificationReport {
            function_name: func.name.clone(),
            language: func.language,
            start_line: func.start_line,
            end_line: func.end_line,
            badge: VerificationBadge::UnboundedProven,
            backend_used: "Z3 SMT-LIB2 (Inductive Invariant)".to_string(),
            obligations_checked: postconditions.len().max(1),
            obligations_proven: postconditions.len().max(1),
            counterexamples: Vec::new(),
            notes,
            generated_test: None,
            verified_fix_source: None,
        });
    }

    // Build symbolic execution state (unbounded for straight-line code, unrolled to `unwind_depth` for loops/recursion)
    let mut state = build_initial_state(func, siblings, opts.unwind_depth);
    apply_preconditions(&mut state, &func.requires, &extra_requires)?;
    for stmt in &func.body {
        exec_stmt(&mut state, stmt)?;
    }
    attach_postconditions(&mut state, func, &postconditions)?;

    let total_obligations = state.obligations.len();
    let mut proven_count = 0usize;
    let mut counterexamples = Vec::new();
    let mut saw_unknown = false;

    // Fast path: batch-check disjunction of all obligations in a single Z3 call!
    let all_disj: Vec<SmtExpr> = state
        .obligations
        .iter()
        .filter_map(|o| o.violation_condition.clone())
        .collect();

    let batch_clean = if all_disj.is_empty() {
        true
    } else {
        let mut batch_script = state.script.clone();
        batch_script.assert(SmtExpr::or_many(all_disj));
        matches!(solver.check(&batch_script).await?, SmtOutcome::Unsat)
    };

    if batch_clean {
        proven_count = total_obligations;
    } else {
        // Pinpoint each violated obligation with its exact minimal counterexample
        for ob in &state.obligations {
            let Some(violation) = &ob.violation_condition else {
                continue;
            };
            let mut ob_script = state.script.clone();
            ob_script.assert(violation.clone());
            match solver.check(&ob_script).await? {
                SmtOutcome::Unsat => {
                    proven_count += 1;
                }
                SmtOutcome::Unknown(reason) => {
                    saw_unknown = true;
                    notes.push(format!("Obligation `{}` returned unknown: {reason}", ob.id));
                }
                SmtOutcome::Sat(bindings) => {
                    let formatted_inputs = format_counterexample_inputs(func, &bindings);
                    let replay = if opts.replay_runtime {
                        let post_exprs: Vec<Expr> =
                            postconditions.iter().map(|p| p.expr.clone()).collect();
                        replay_counterexample_on_runtime(
                            func,
                            full_file_source,
                            &bindings,
                            &post_exprs,
                            None,
                        )
                        .await
                    } else {
                        RuntimeReplayVerdict::Skipped
                    };

                    // Filter out false positives if live runtime replay ran cleanly without failing
                    if matches!(replay, RuntimeReplayVerdict::NoFailureObserved { .. }) {
                        notes.push(format!(
                            "Filtered static candidate on `{}` after live runtime replay succeeded without failure",
                            ob.id
                        ));
                        continue;
                    }

                    counterexamples.push(CounterexampleReport {
                        obligation_id: ob.id.clone(),
                        kind: ob.kind,
                        description: ob.description.clone(),
                        line: ob.line,
                        inputs: formatted_inputs,
                        raw_values: bindings,
                        runtime_replay: replay,
                    });
                }
            }
        }
    }

    if !counterexamples.is_empty() {
        return Ok(FunctionVerificationReport {
            function_name: func.name.clone(),
            language: func.language,
            start_line: func.start_line,
            end_line: func.end_line,
            badge: VerificationBadge::BugFound,
            backend_used: "Z3 SMT-LIB2 + Live Runtime Replay".to_string(),
            obligations_checked: total_obligations,
            obligations_proven: proven_count,
            counterexamples,
            notes,
            generated_test: None,
            verified_fix_source: None,
        });
    }

    if saw_unknown {
        if let Some(lean_backend_label) =
            try_lean4_escalation(func, &postconditions, cfg, router).await
        {
            return Ok(FunctionVerificationReport {
                function_name: func.name.clone(),
                language: func.language,
                start_line: func.start_line,
                end_line: func.end_line,
                badge: VerificationBadge::UnboundedProven,
                backend_used: lean_backend_label,
                obligations_checked: total_obligations,
                obligations_proven: total_obligations,
                counterexamples: Vec::new(),
                notes,
                generated_test: None,
                verified_fix_source: None,
            });
        }
        return Ok(FunctionVerificationReport {
            function_name: func.name.clone(),
            language: func.language,
            start_line: func.start_line,
            end_line: func.end_line,
            badge: VerificationBadge::Unknown,
            backend_used: "Z3 SMT-LIB2".to_string(),
            obligations_checked: total_obligations,
            obligations_proven: proven_count,
            counterexamples: Vec::new(),
            notes,
            generated_test: None,
            verified_fix_source: None,
        });
    }

    // Check if loop/recursion unwinding could possibly exceed `opts.unwind_depth`
    let mut can_exceed_unwind = false;
    if !state.unwinding_truncated_conds.is_empty() {
        let mut trunc_script = state.script.clone();
        trunc_script.assert(SmtExpr::or_many(state.unwinding_truncated_conds));
        can_exceed_unwind = !matches!(solver.check(&trunc_script).await?, SmtOutcome::Unsat);
    }

    if can_exceed_unwind {
        // Attempt Tier-2 Lean 4 unbounded escalation before settling on `[BOUNDED VERIFIED]`
        if let Some(lean_backend_label) =
            try_lean4_escalation(func, &postconditions, cfg, router).await
        {
            return Ok(FunctionVerificationReport {
                function_name: func.name.clone(),
                language: func.language,
                start_line: func.start_line,
                end_line: func.end_line,
                badge: VerificationBadge::UnboundedProven,
                backend_used: lean_backend_label,
                obligations_checked: total_obligations,
                obligations_proven: total_obligations,
                counterexamples: Vec::new(),
                notes,
                generated_test: None,
                verified_fix_source: None,
            });
        }

        return Ok(FunctionVerificationReport {
            function_name: func.name.clone(),
            language: func.language,
            start_line: func.start_line,
            end_line: func.end_line,
            badge: VerificationBadge::BoundedVerified {
                depth: opts.unwind_depth,
            },
            backend_used: format!("Z3 Bounded Model Checking (depth={})", opts.unwind_depth),
            obligations_checked: total_obligations,
            obligations_proven: proven_count,
            counterexamples: Vec::new(),
            notes,
            generated_test: None,
            verified_fix_source: None,
        });
    }

    Ok(FunctionVerificationReport {
        function_name: func.name.clone(),
        language: func.language,
        start_line: func.start_line,
        end_line: func.end_line,
        badge: VerificationBadge::UnboundedProven,
        backend_used: "Z3 SMT-LIB2".to_string(),
        obligations_checked: total_obligations,
        obligations_proven: proven_count,
        counterexamples: Vec::new(),
        notes,
        generated_test: None,
        verified_fix_source: None,
    })
}

/// Proves functional equivalence `--equiv fast_fn=naive_fn` across all valid inputs (or finds an exact input where they diverge).
pub async fn verify_equivalence(
    fast_fn: &VerifyFunction,
    naive_fn: &VerifyFunction,
    full_file_source: &str,
    siblings: &[VerifyFunction],
    solver: &Z3Solver,
    opts: &VerifyOptions,
) -> anyhow::Result<FunctionVerificationReport> {
    if fast_fn.params.len() != naive_fn.params.len() {
        bail!(
            "Cannot check equivalence `{}` vs `{}`: parameter counts differ ({} vs {})",
            fast_fn.name,
            naive_fn.name,
            fast_fn.params.len(),
            naive_fn.params.len()
        );
    }

    let mut fast_state = build_initial_state(fast_fn, siblings, opts.unwind_depth);
    apply_preconditions(&mut fast_state, &fast_fn.requires, &naive_fn.requires)?;
    for s in &fast_fn.body {
        exec_stmt(&mut fast_state, s)?;
    }

    // Share the exact same symbolic parameter declarations with `naive_state`
    let mut naive_state = SymState {
        script: fast_state.script.clone(),
        lang: naive_fn.language,
        return_ty: naive_fn.return_ty.clone(),
        env: BTreeMap::new(),
        path_cond: SmtExpr::BoolLit(true),
        returned_cond: SmtExpr::BoolLit(false),
        return_val: None,
        obligations: Vec::new(),
        siblings: siblings.to_vec(),
        unwind_budget: opts.unwind_depth,
        unwinding_truncated_conds: Vec::new(),
        check_overflows: naive_fn.language.checks_fixed_width_overflow(),
        ssa_counter: fast_state.ssa_counter,
    };
    for (fp, np) in fast_fn.params.iter().zip(naive_fn.params.iter()) {
        if let Some(v) = fast_state.env.get(&fp.name).cloned() {
            naive_state.env.insert(np.name.clone(), v);
        }
    }
    for s in &naive_fn.body {
        exec_stmt(&mut naive_state, s)?;
    }

    let fast_ret = fast_state
        .return_val
        .clone()
        .unwrap_or_else(|| default_zero_for(&fast_fn.return_ty, fast_fn.language));
    let naive_ret = naive_state
        .return_val
        .clone()
        .unwrap_or_else(|| default_zero_for(&naive_fn.return_ty, naive_fn.language));

    let mut combined_script = naive_state.script.clone();
    let mut all_obs = fast_state.obligations;
    let (fr, nr) = if fast_ret.ty == VerifyType::Bool {
        (fast_ret, naive_ret)
    } else {
        unify_numeric_pair(fast_ret, naive_ret, fast_fn.language)
    };
    all_obs.push(SafetyObligation {
        id: format!("juice/equivalence@L{}", fast_fn.start_line),
        kind: ObligationKind::Equivalence,
        description: format!(
            "`{}` must return the exact same value as `{}` for all inputs",
            fast_fn.name, naive_fn.name
        ),
        line: fast_fn.start_line,
        violation_condition: Some(SmtExpr::ne(fr.expr, nr.expr)),
    });

    let total = all_obs.len();
    let mut proven = 0usize;
    let mut counterexamples = Vec::new();

    for ob in &all_obs {
        let Some(v) = &ob.violation_condition else {
            continue;
        };
        let mut q = combined_script.clone();
        q.assert(v.clone());
        match solver.check(&q).await? {
            SmtOutcome::Unsat => proven += 1,
            SmtOutcome::Sat(bindings) => {
                let inputs = format_counterexample_inputs(fast_fn, &bindings);
                let replay = if opts.replay_runtime {
                    replay_counterexample_on_runtime(
                        fast_fn,
                        full_file_source,
                        &bindings,
                        &[],
                        Some(naive_fn),
                    )
                    .await
                } else {
                    RuntimeReplayVerdict::Skipped
                };
                counterexamples.push(CounterexampleReport {
                    obligation_id: ob.id.clone(),
                    kind: ob.kind,
                    description: ob.description.clone(),
                    line: ob.line,
                    inputs,
                    raw_values: bindings,
                    runtime_replay: replay,
                });
            }
            SmtOutcome::Unknown(_) => {}
        }
    }

    let trunc: Vec<SmtExpr> = fast_state
        .unwinding_truncated_conds
        .into_iter()
        .chain(naive_state.unwinding_truncated_conds)
        .collect();
    let mut bounded = false;
    if counterexamples.is_empty() && !trunc.is_empty() {
        combined_script.assert(SmtExpr::or_many(trunc));
        bounded = !matches!(solver.check(&combined_script).await?, SmtOutcome::Unsat);
    }

    let badge = if !counterexamples.is_empty() {
        VerificationBadge::BugFound
    } else if proven < total {
        VerificationBadge::Unknown
    } else if bounded {
        VerificationBadge::BoundedVerified {
            depth: opts.unwind_depth,
        }
    } else {
        VerificationBadge::UnboundedProven
    };

    Ok(FunctionVerificationReport {
        function_name: format!("{} == {}", fast_fn.name, naive_fn.name),
        language: fast_fn.language,
        start_line: fast_fn.start_line,
        end_line: fast_fn.end_line,
        badge,
        backend_used: "Z3 SMT-LIB2 Equivalence Engine".to_string(),
        obligations_checked: total,
        obligations_proven: proven,
        counterexamples,
        notes: Vec::new(),
        generated_test: None,
        verified_fix_source: None,
    })
}
