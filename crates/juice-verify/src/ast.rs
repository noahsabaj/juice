use crate::lang::{SourceLanguage, VerifyType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: VerifyType,
    /// For C pointer+length pairs (e.g. `const int* arr, int len`), links `arr` to `len`.
    pub linked_len_param: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    WrappingAdd,
    WrappingSub,
    WrappingMul,
    SaturatingAdd,
    SaturatingSub,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    Var(String),
    IntLit(i128),
    BoolLit(bool),
    NoneLit,
    SomeExpr(Box<Expr>),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
        line: usize,
    },
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
        line: usize,
    },
    Cast {
        expr: Box<Expr>,
        target_ty: VerifyType,
        line: usize,
    },
    Index {
        array: Box<Expr>,
        index: Box<Expr>,
        line: usize,
    },
    Len {
        array: Box<Expr>,
    },
    Unwrap {
        expr: Box<Expr>,
        line: usize,
    },
    IsSome(Box<Expr>),
    IsNone(Box<Expr>),
    Cond {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Call {
        func: String,
        args: Vec<Expr>,
        line: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<VerifyType>,
        value: Expr,
        mutable: bool,
        line: usize,
    },
    Assign {
        name: String,
        value: Expr,
        line: usize,
    },
    ArrayAssign {
        array: String,
        index: Expr,
        value: Expr,
        line: usize,
    },
    If {
        cond: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Vec<Stmt>,
        line: usize,
    },
    While {
        cond: Expr,
        invariants: Vec<Expr>,
        body: Vec<Stmt>,
        line: usize,
    },
    Assert {
        cond: Expr,
        message: Option<String>,
        line: usize,
    },
    Panic {
        message: String,
        line: usize,
    },
    Return {
        value: Option<Expr>,
        line: usize,
    },
    ExprStmt {
        expr: Expr,
        line: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyFunction {
    pub name: String,
    pub language: SourceLanguage,
    pub params: Vec<Param>,
    pub return_ty: VerifyType,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    pub loop_invariants: Vec<Expr>,
    pub body: Vec<Stmt>,
    pub raw_source: String,
    pub span_start_byte: usize,
    pub span_end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub is_recursive: bool,
    pub has_loops: bool,
}

impl VerifyFunction {
    pub fn refresh_flags(&mut self) {
        self.has_loops = stmts_have_loops(&self.body);
        self.is_recursive = stmts_call_fn(&self.body, &self.name);
    }
}

fn stmts_have_loops(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::While { .. } => true,
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => stmts_have_loops(then_branch) || stmts_have_loops(else_branch),
        _ => false,
    })
}

fn stmts_call_fn(stmts: &[Stmt], target: &str) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Let { value, .. }
        | Stmt::Assign { value, .. }
        | Stmt::Assert { cond: value, .. }
        | Stmt::ExprStmt { expr: value, .. } => expr_calls_fn(value, target),
        Stmt::ArrayAssign { index, value, .. } => {
            expr_calls_fn(index, target) || expr_calls_fn(value, target)
        }
        Stmt::Return { value: Some(v), .. } => expr_calls_fn(v, target),
        Stmt::Return { value: None, .. } | Stmt::Panic { .. } => false,
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            expr_calls_fn(cond, target)
                || stmts_call_fn(then_branch, target)
                || stmts_call_fn(else_branch, target)
        }
        Stmt::While { cond, body, .. } => {
            expr_calls_fn(cond, target) || stmts_call_fn(body, target)
        }
    })
}

fn expr_calls_fn(expr: &Expr, target: &str) -> bool {
    match expr {
        Expr::Call { func, args, .. } => {
            func == target || args.iter().any(|a| expr_calls_fn(a, target))
        }
        Expr::Unary { expr: e, .. }
        | Expr::Cast { expr: e, .. }
        | Expr::Len { array: e }
        | Expr::Unwrap { expr: e, .. }
        | Expr::IsSome(e)
        | Expr::IsNone(e)
        | Expr::SomeExpr(e) => expr_calls_fn(e, target),
        Expr::Binary { left, right, .. }
        | Expr::Index {
            array: left,
            index: right,
            ..
        } => expr_calls_fn(left, target) || expr_calls_fn(right, target),
        Expr::Cond {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_calls_fn(cond, target)
                || expr_calls_fn(then_expr, target)
                || expr_calls_fn(else_expr, target)
        }
        Expr::Var(_) | Expr::IntLit(_) | Expr::BoolLit(_) | Expr::NoneLit => false,
    }
}
