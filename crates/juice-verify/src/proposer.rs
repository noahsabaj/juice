use crate::ast::{BinOp, Expr, VerifyFunction};
use crate::frontend::parse_expr_str;
use crate::lang::VerifyType;
use juice_llm::ModelRouter;
use juice_smt::SmtExpr;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObligationKind {
    IntegerOverflow,
    DivisionByZero,
    ShiftOverflow,
    OutOfBounds,
    AssertionOrPanic,
    Postcondition,
    Equivalence,
}

impl ObligationKind {
    pub fn rule_id(&self) -> &'static str {
        match self {
            Self::IntegerOverflow => "juice/integer-overflow",
            Self::DivisionByZero => "juice/division-by-zero",
            Self::ShiftOverflow => "juice/shift-overflow",
            Self::OutOfBounds => "juice/out-of-bounds",
            Self::AssertionOrPanic => "juice/panic-or-assert",
            Self::Postcondition => "juice/postcondition",
            Self::Equivalence => "juice/equivalence",
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Self::IntegerOverflow => "Arithmetic overflow / underflow freedom",
            Self::DivisionByZero => "Division / remainder by zero freedom",
            Self::ShiftOverflow => "Bit-shift width bounds",
            Self::OutOfBounds => "Array / slice index bounds",
            Self::AssertionOrPanic => "Assertion / unwrap / panic freedom",
            Self::Postcondition => "Functional postcondition",
            Self::Equivalence => "Functional equivalence",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyObligation {
    pub id: String,
    pub kind: ObligationKind,
    pub description: String,
    pub line: usize,
    #[serde(skip)]
    pub violation_condition: Option<SmtExpr>,
}

#[derive(Debug, Clone)]
pub struct ProposedPostcondition {
    pub description: String,
    pub expr: Expr,
    pub auto_proposed: bool,
}

/// Automatically proposes domain functional postconditions from function signatures and names
/// when the user did not supply explicit `@ensures` or `--spec` rules.
pub fn propose_auto_postconditions(func: &VerifyFunction) -> Vec<ProposedPostcondition> {
    if !func.ensures.is_empty() {
        return func
            .ensures
            .iter()
            .enumerate()
            .map(|(idx, e)| ProposedPostcondition {
                description: format!("@ensures contract #{}", idx + 1),
                expr: e.clone(),
                auto_proposed: false,
            })
            .collect();
    }

    let mut out = Vec::new();
    let lower_name = func.name.to_ascii_lowercase();
    let line = func.start_line;
    let is_numeric_ret = matches!(func.return_ty, VerifyType::BitVec { .. } | VerifyType::Int);

    if !is_numeric_ret {
        return out;
    }

    // `midpoint(a, b)` / `average(a, b)` / `avg(a, b)` with 2 numeric params
    if (lower_name.contains("midpoint") || lower_name == "avg" || lower_name == "average")
        && func.params.len() == 2
        && matches!(
            func.params[0].ty,
            VerifyType::BitVec { .. } | VerifyType::Int
        )
        && func.params[0].ty == func.params[1].ty
    {
        let a = &func.params[0].name;
        let b = &func.params[1].name;
        let expr_str = format!(
            "(!({a} <= {b}) || (result >= {a} && result <= {b})) && (!({b} <= {a}) || (result >= {b} && result <= {a}))"
        );
        if let Ok(expr) = parse_expr_str(&expr_str, func.language, line) {
            out.push(ProposedPostcondition {
                description: format!(
                    "auto-proposed midpoint bound: `min({a}, {b}) <= result <= max({a}, {b})`"
                ),
                expr,
                auto_proposed: true,
            });
        }
    }

    // `abs*` with 1 numeric param
    if (lower_name == "abs" || lower_name.starts_with("abs_")) && func.params.len() == 1 {
        out.push(ProposedPostcondition {
            description: "auto-proposed non-negativity: `result >= 0`".to_string(),
            expr: Expr::Binary {
                op: BinOp::Ge,
                left: Box::new(Expr::Var("result".to_string())),
                right: Box::new(Expr::IntLit(0)),
                line,
            },
            auto_proposed: true,
        });
    }

    // `clamp(val, lo, hi)` with 3 numeric params
    if lower_name.contains("clamp") && func.params.len() == 3 {
        let lo = &func.params[1].name;
        let hi = &func.params[2].name;
        let expr_str = format!("!({lo} <= {hi}) || (result >= {lo} && result <= {hi})");
        if let Ok(expr) = parse_expr_str(&expr_str, func.language, line) {
            out.push(ProposedPostcondition {
                description: format!(
                    "auto-proposed clamp bound: `{lo} <= {hi} ==> {lo} <= result <= {hi}`"
                ),
                expr,
                auto_proposed: true,
            });
        }
    }

    // `min(a, b)` / `max(a, b)`
    if lower_name == "min" && func.params.len() == 2 {
        let a = &func.params[0].name;
        let b = &func.params[1].name;
        if let Ok(expr) = parse_expr_str(
            &format!("result <= {a} && result <= {b} && (result == {a} || result == {b})"),
            func.language,
            line,
        ) {
            out.push(ProposedPostcondition {
                description: format!("auto-proposed minimum spec: `result == min({a}, {b})`"),
                expr,
                auto_proposed: true,
            });
        }
    }
    if lower_name == "max" && func.params.len() == 2 {
        let a = &func.params[0].name;
        let b = &func.params[1].name;
        if let Ok(expr) = parse_expr_str(
            &format!("result >= {a} && result >= {b} && (result == {a} || result == {b})"),
            func.language,
            line,
        ) {
            out.push(ProposedPostcondition {
                description: format!("auto-proposed maximum spec: `result == max({a}, {b})`"),
                expr,
                auto_proposed: true,
            });
        }
    }

    out
}

#[derive(Debug, Deserialize)]
struct FormalizedSpecPayload {
    requires: Option<String>,
    ensures: String,
}

/// Compiles a CLI `--spec "..."` string into optional precondition + postcondition expressions.
/// Works offline for formal expressions (`result >= 0`, `lo <= hi ==> result >= lo`) and uses
/// `ModelRouter` when given plain-English prose.
pub async fn compile_cli_spec(
    raw_spec: &str,
    func: &VerifyFunction,
    router: Option<&ModelRouter>,
) -> anyhow::Result<(Option<Expr>, ProposedPostcondition)> {
    // Support `pre ==> post` shorthand deterministically
    if let Some((pre_s, post_s)) = raw_spec.split_once("==>")
        && let (Ok(pre_e), Ok(post_e)) = (
            parse_expr_str(pre_s.trim(), func.language, func.start_line),
            parse_expr_str(post_s.trim(), func.language, func.start_line),
        )
    {
        return Ok((
            Some(pre_e),
            ProposedPostcondition {
                description: format!("spec: `{raw_spec}`"),
                expr: post_e,
                auto_proposed: false,
            },
        ));
    }

    if let Ok(expr) = parse_expr_str(raw_spec, func.language, func.start_line) {
        return Ok((
            None,
            ProposedPostcondition {
                description: format!("spec: `{raw_spec}`"),
                expr,
                auto_proposed: false,
            },
        ));
    }

    let Some(router) = router else {
        anyhow::bail!(
            "Could not deterministically parse `--spec \"{raw_spec}\"` as a formal expression (use `result`, parameter names, `&&`, `||`, `==>`, `<=`, etc., or configure an LLM API key for natural-language spec translation)."
        );
    };

    let param_list: Vec<String> = func
        .params
        .iter()
        .map(|p| format!("{}: {:?}", p.name, p.ty))
        .collect();
    let system = "Translate a plain-English software specification into boolean C/Rust expression syntax over the function parameters and `result` (using `&&`, `||`, `!`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, `/`, `%`). Return JSON `{\"requires\": null_or_expr_string, \"ensures\": expr_string}`.";
    let user = format!(
        "Function `{}`({})\nSource:\n{}\n\nNatural-language spec: {}",
        func.name,
        param_list.join(", "),
        func.raw_source,
        raw_spec
    );
    let payload: FormalizedSpecPayload = router.complete_planner_json(system, &user).await?;
    let pre = payload
        .requires
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| parse_expr_str(s, func.language, func.start_line))
        .transpose()?;
    let post = parse_expr_str(&payload.ensures, func.language, func.start_line)?;

    Ok((
        pre,
        ProposedPostcondition {
            description: format!("spec `{raw_spec}` (`{}`)", payload.ensures),
            expr: post,
            auto_proposed: false,
        },
    ))
}
