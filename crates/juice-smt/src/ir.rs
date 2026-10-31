use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Int,
    BitVec(u32),
    Array(Box<SmtSort>, Box<SmtSort>),
}

impl SmtSort {
    pub fn to_smtlib(&self) -> String {
        match self {
            SmtSort::Bool => "Bool".to_string(),
            SmtSort::Int => "Int".to_string(),
            SmtSort::BitVec(w) => format!("(_ BitVec {w})"),
            SmtSort::Array(idx, elem) => format!("(Array {} {})", idx.to_smtlib(), elem.to_smtlib()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtExpr {
    Var(String),
    BoolLit(bool),
    IntLit(i128),
    BvLit { value: u128, width: u32 },
    Not(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),

    // Unbounded integer arithmetic
    IntNeg(Box<SmtExpr>),
    IntAdd(Box<SmtExpr>, Box<SmtExpr>),
    IntSub(Box<SmtExpr>, Box<SmtExpr>),
    IntMul(Box<SmtExpr>, Box<SmtExpr>),
    IntDiv(Box<SmtExpr>, Box<SmtExpr>),
    IntMod(Box<SmtExpr>, Box<SmtExpr>),
    IntLt(Box<SmtExpr>, Box<SmtExpr>),
    IntLe(Box<SmtExpr>, Box<SmtExpr>),
    IntGt(Box<SmtExpr>, Box<SmtExpr>),
    IntGe(Box<SmtExpr>, Box<SmtExpr>),

    // Fixed-width bitvector operations
    BvNeg(Box<SmtExpr>),
    BvNot(Box<SmtExpr>),
    SignExtend { extra_bits: u32, expr: Box<SmtExpr> },
    ZeroExtend { extra_bits: u32, expr: Box<SmtExpr> },
    Extract { hi: u32, lo: u32, expr: Box<SmtExpr> },
    BvAdd(Box<SmtExpr>, Box<SmtExpr>),
    BvSub(Box<SmtExpr>, Box<SmtExpr>),
    BvMul(Box<SmtExpr>, Box<SmtExpr>),
    BvSDiv(Box<SmtExpr>, Box<SmtExpr>),
    BvUDiv(Box<SmtExpr>, Box<SmtExpr>),
    BvSRem(Box<SmtExpr>, Box<SmtExpr>),
    BvURem(Box<SmtExpr>, Box<SmtExpr>),
    BvAnd(Box<SmtExpr>, Box<SmtExpr>),
    BvOr(Box<SmtExpr>, Box<SmtExpr>),
    BvXor(Box<SmtExpr>, Box<SmtExpr>),
    BvShl(Box<SmtExpr>, Box<SmtExpr>),
    BvLShr(Box<SmtExpr>, Box<SmtExpr>),
    BvAShr(Box<SmtExpr>, Box<SmtExpr>),
    BvSLt(Box<SmtExpr>, Box<SmtExpr>),
    BvSLe(Box<SmtExpr>, Box<SmtExpr>),
    BvSGt(Box<SmtExpr>, Box<SmtExpr>),
    BvSGe(Box<SmtExpr>, Box<SmtExpr>),
    BvULt(Box<SmtExpr>, Box<SmtExpr>),
    BvULe(Box<SmtExpr>, Box<SmtExpr>),
    BvUGt(Box<SmtExpr>, Box<SmtExpr>),
    BvUGe(Box<SmtExpr>, Box<SmtExpr>),

    // Arrays
    Select(Box<SmtExpr>, Box<SmtExpr>),
    Store(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),

    Raw(String),
}

pub fn mask_for_width(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else if width == 0 {
        0
    } else {
        (1u128 << width) - 1
    }
}

pub fn signed_min_bits(width: u32) -> u128 {
    if width == 0 || width > 128 {
        0
    } else {
        1u128 << (width - 1)
    }
}

impl SmtExpr {
    pub fn var(name: impl Into<String>) -> Self {
        SmtExpr::Var(name.into())
    }

    pub fn bv_signed(val: i128, width: u32) -> Self {
        let bits = (val as u128) & mask_for_width(width);
        SmtExpr::BvLit { value: bits, width }
    }

    pub fn bv_unsigned(val: u128, width: u32) -> Self {
        SmtExpr::BvLit {
            value: val & mask_for_width(width),
            width,
        }
    }

    pub fn lnot(e: SmtExpr) -> Self {
        SmtExpr::Not(Box::new(e))
    }

    pub fn eq(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Eq(Box::new(a), Box::new(b))
    }

    pub fn ne(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::lnot(SmtExpr::eq(a, b))
    }

    pub fn and_many(mut items: Vec<SmtExpr>) -> Self {
        items.retain(|e| *e != SmtExpr::BoolLit(true));
        if items.is_empty() {
            SmtExpr::BoolLit(true)
        } else if items.len() == 1 {
            items.pop().unwrap()
        } else {
            SmtExpr::And(items)
        }
    }

    pub fn or_many(mut items: Vec<SmtExpr>) -> Self {
        items.retain(|e| *e != SmtExpr::BoolLit(false));
        if items.is_empty() {
            SmtExpr::BoolLit(false)
        } else if items.len() == 1 {
            items.pop().unwrap()
        } else {
            SmtExpr::Or(items)
        }
    }

    /// Exact portable SMT-LIB2 signed addition overflow predicate via 1-bit sign extension.
    pub fn bv_signed_add_overflows(a: SmtExpr, b: SmtExpr) -> Self {
        let narrow = SmtExpr::SignExtend {
            extra_bits: 1,
            expr: Box::new(SmtExpr::BvAdd(Box::new(a.clone()), Box::new(b.clone()))),
        };
        let wide = SmtExpr::BvAdd(
            Box::new(SmtExpr::SignExtend {
                extra_bits: 1,
                expr: Box::new(a),
            }),
            Box::new(SmtExpr::SignExtend {
                extra_bits: 1,
                expr: Box::new(b),
            }),
        );
        SmtExpr::ne(narrow, wide)
    }

    /// Exact portable SMT-LIB2 unsigned addition overflow predicate via 1-bit zero extension.
    pub fn bv_unsigned_add_overflows(a: SmtExpr, b: SmtExpr) -> Self {
        let narrow = SmtExpr::ZeroExtend {
            extra_bits: 1,
            expr: Box::new(SmtExpr::BvAdd(Box::new(a.clone()), Box::new(b.clone()))),
        };
        let wide = SmtExpr::BvAdd(
            Box::new(SmtExpr::ZeroExtend {
                extra_bits: 1,
                expr: Box::new(a),
            }),
            Box::new(SmtExpr::ZeroExtend {
                extra_bits: 1,
                expr: Box::new(b),
            }),
        );
        SmtExpr::ne(narrow, wide)
    }

    pub fn bv_signed_sub_overflows(a: SmtExpr, b: SmtExpr) -> Self {
        let narrow = SmtExpr::SignExtend {
            extra_bits: 1,
            expr: Box::new(SmtExpr::BvSub(Box::new(a.clone()), Box::new(b.clone()))),
        };
        let wide = SmtExpr::BvSub(
            Box::new(SmtExpr::SignExtend {
                extra_bits: 1,
                expr: Box::new(a),
            }),
            Box::new(SmtExpr::SignExtend {
                extra_bits: 1,
                expr: Box::new(b),
            }),
        );
        SmtExpr::ne(narrow, wide)
    }

    pub fn bv_unsigned_sub_overflows(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::BvULt(Box::new(a), Box::new(b))
    }

    pub fn bv_signed_mul_overflows(a: SmtExpr, b: SmtExpr, width: u32) -> Self {
        let narrow = SmtExpr::SignExtend {
            extra_bits: width,
            expr: Box::new(SmtExpr::BvMul(Box::new(a.clone()), Box::new(b.clone()))),
        };
        let wide = SmtExpr::BvMul(
            Box::new(SmtExpr::SignExtend {
                extra_bits: width,
                expr: Box::new(a),
            }),
            Box::new(SmtExpr::SignExtend {
                extra_bits: width,
                expr: Box::new(b),
            }),
        );
        SmtExpr::ne(narrow, wide)
    }

    pub fn bv_unsigned_mul_overflows(a: SmtExpr, b: SmtExpr, width: u32) -> Self {
        let narrow = SmtExpr::ZeroExtend {
            extra_bits: width,
            expr: Box::new(SmtExpr::BvMul(Box::new(a.clone()), Box::new(b.clone()))),
        };
        let wide = SmtExpr::BvMul(
            Box::new(SmtExpr::ZeroExtend {
                extra_bits: width,
                expr: Box::new(a),
            }),
            Box::new(SmtExpr::ZeroExtend {
                extra_bits: width,
                expr: Box::new(b),
            }),
        );
        SmtExpr::ne(narrow, wide)
    }

    pub fn bv_signed_neg_overflows(a: SmtExpr, width: u32) -> Self {
        SmtExpr::eq(
            a,
            SmtExpr::BvLit {
                value: signed_min_bits(width),
                width,
            },
        )
    }

    /// Two's complement signed division/remainder overflow (`INT_MIN / -1`).
    pub fn bv_signed_div_overflows(a: SmtExpr, b: SmtExpr, width: u32) -> Self {
        SmtExpr::and_many(vec![
            SmtExpr::eq(
                a,
                SmtExpr::BvLit {
                    value: signed_min_bits(width),
                    width,
                },
            ),
            SmtExpr::eq(
                b,
                SmtExpr::BvLit {
                    value: mask_for_width(width),
                    width,
                },
            ),
        ])
    }

    pub fn to_smtlib(&self) -> String {
        match self {
            SmtExpr::Var(v) => sanitize_symbol(v),
            SmtExpr::BoolLit(true) => "true".to_string(),
            SmtExpr::BoolLit(false) => "false".to_string(),
            SmtExpr::IntLit(n) => {
                if *n < 0 {
                    format!("(- {})", n.unsigned_abs())
                } else {
                    n.to_string()
                }
            }
            SmtExpr::BvLit { value, width } => {
                let masked = value & mask_for_width(*width);
                format!("(_ bv{masked} {width})")
            }
            SmtExpr::Not(e) => format!("(not {})", e.to_smtlib()),
            SmtExpr::And(items) => {
                if items.is_empty() {
                    "true".to_string()
                } else if items.len() == 1 {
                    items[0].to_smtlib()
                } else {
                    let inner: Vec<String> = items.iter().map(SmtExpr::to_smtlib).collect();
                    format!("(and {})", inner.join(" "))
                }
            }
            SmtExpr::Or(items) => {
                if items.is_empty() {
                    "false".to_string()
                } else if items.len() == 1 {
                    items[0].to_smtlib()
                } else {
                    let inner: Vec<String> = items.iter().map(SmtExpr::to_smtlib).collect();
                    format!("(or {})", inner.join(" "))
                }
            }
            SmtExpr::Implies(a, b) => format!("(=> {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Eq(a, b) => format!("(= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Ite(c, t, e) => {
                format!("(ite {} {} {})", c.to_smtlib(), t.to_smtlib(), e.to_smtlib())
            }
            SmtExpr::IntNeg(e) => format!("(- {})", e.to_smtlib()),
            SmtExpr::IntAdd(a, b) => format!("(+ {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntSub(a, b) => format!("(- {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntMul(a, b) => format!("(* {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntDiv(a, b) => format!("(div {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntMod(a, b) => format!("(mod {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntLt(a, b) => format!("(< {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntLe(a, b) => format!("(<= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntGt(a, b) => format!("(> {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::IntGe(a, b) => format!("(>= {} {})", a.to_smtlib(), b.to_smtlib()),

            SmtExpr::BvNeg(e) => format!("(bvneg {})", e.to_smtlib()),
            SmtExpr::BvNot(e) => format!("(bvnot {})", e.to_smtlib()),
            SmtExpr::SignExtend { extra_bits, expr } => {
                format!("((_ sign_extend {extra_bits}) {})", expr.to_smtlib())
            }
            SmtExpr::ZeroExtend { extra_bits, expr } => {
                format!("((_ zero_extend {extra_bits}) {})", expr.to_smtlib())
            }
            SmtExpr::Extract { hi, lo, expr } => {
                format!("((_ extract {hi} {lo}) {})", expr.to_smtlib())
            }
            SmtExpr::BvAdd(a, b) => format!("(bvadd {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSub(a, b) => format!("(bvsub {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvMul(a, b) => format!("(bvmul {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSDiv(a, b) => format!("(bvsdiv {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUDiv(a, b) => format!("(bvudiv {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSRem(a, b) => format!("(bvsrem {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvURem(a, b) => format!("(bvurem {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvAnd(a, b) => format!("(bvand {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvOr(a, b) => format!("(bvor {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvXor(a, b) => format!("(bvxor {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvShl(a, b) => format!("(bvshl {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvLShr(a, b) => format!("(bvlshr {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvAShr(a, b) => format!("(bvashr {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSLt(a, b) => format!("(bvslt {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSLe(a, b) => format!("(bvsle {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSGt(a, b) => format!("(bvsgt {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSGe(a, b) => format!("(bvsge {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvULt(a, b) => format!("(bvult {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvULe(a, b) => format!("(bvule {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUGt(a, b) => format!("(bvugt {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUGe(a, b) => format!("(bvuge {} {})", a.to_smtlib(), b.to_smtlib()),

            SmtExpr::Select(arr, idx) => format!("(select {} {})", arr.to_smtlib(), idx.to_smtlib()),
            SmtExpr::Store(arr, idx, val) => {
                format!("(store {} {} {})", arr.to_smtlib(), idx.to_smtlib(), val.to_smtlib())
            }
            SmtExpr::Raw(s) => s.clone(),
        }
    }
}

pub fn sanitize_symbol(raw: &str) -> String {
    let valid_simple = !raw.is_empty()
        && !raw.chars().next().unwrap().is_ascii_digit()
        && raw
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.' | '@' | '!'));
    if valid_simple {
        raw.to_string()
    } else {
        format!("|{}|", raw.replace('|', "_"))
    }
}

#[derive(Debug, Clone, Default)]
pub struct SmtScript {
    pub declarations: Vec<(String, SmtSort)>,
    pub query_vars: Vec<String>,
    pub assertions: Vec<SmtExpr>,
}

impl SmtScript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declare_const(&mut self, name: impl Into<String>, sort: SmtSort, query: bool) {
        let n = name.into();
        if !self.declarations.iter().any(|(existing, _)| existing == &n) {
            self.declarations.push((n.clone(), sort));
        }
        if query && !self.query_vars.contains(&n) {
            self.query_vars.push(n);
        }
    }

    pub fn assert(&mut self, expr: SmtExpr) {
        self.assertions.push(expr);
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("(set-option :produce-models true)\n");
        for (name, sort) in &self.declarations {
            out.push_str(&format!(
                "(declare-const {} {})\n",
                sanitize_symbol(name),
                sort.to_smtlib()
            ));
        }
        for assertion in &self.assertions {
            out.push_str(&format!("(assert {})\n", assertion.to_smtlib()));
        }
        out.push_str("(check-sat)\n");
        if !self.query_vars.is_empty() {
            let vars: Vec<String> = self.query_vars.iter().map(|v| sanitize_symbol(v)).collect();
            out.push_str(&format!("(get-value ({}))\n", vars.join(" ")));
        } else {
            out.push_str("(get-model)\n");
        }
        out
    }
}
