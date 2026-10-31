use crate::ir::mask_for_width;
use anyhow::bail;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtValue {
    Bool(bool),
    Int(i128),
    BitVec {
        width: u32,
        unsigned: u128,
        signed: i128,
    },
    Array {
        default: Box<SmtValue>,
        entries: Vec<(SmtValue, SmtValue)>,
    },
    Raw(String),
}

impl SmtValue {
    pub fn as_i128(&self, prefer_signed: bool) -> Option<i128> {
        match self {
            SmtValue::Int(v) => Some(*v),
            SmtValue::BitVec {
                unsigned, signed, ..
            } => {
                if prefer_signed {
                    Some(*signed)
                } else {
                    i128::try_from(*unsigned).ok()
                }
            }
            SmtValue::Bool(b) => Some(i128::from(*b)),
            _ => None,
        }
    }

    pub fn format_typed(&self, is_signed: bool) -> String {
        match self {
            SmtValue::Bool(b) => b.to_string(),
            SmtValue::Int(n) => n.to_string(),
            SmtValue::BitVec {
                unsigned, signed, ..
            } => {
                if is_signed {
                    signed.to_string()
                } else {
                    unsigned.to_string()
                }
            }
            SmtValue::Array { default, entries } => {
                if entries.is_empty() {
                    format!("[{}; ..]", default.format_typed(is_signed))
                } else {
                    let parts: Vec<String> = entries
                        .iter()
                        .map(|(k, v)| {
                            format!(
                                "{} => {}",
                                k.format_typed(false),
                                v.format_typed(is_signed)
                            )
                        })
                        .collect();
                    format!("{{{}, default: {}}}", parts.join(", "), default.format_typed(is_signed))
                }
            }
            SmtValue::Raw(s) => s.clone(),
        }
    }
}

pub fn twos_complement_to_signed(bits: u128, width: u32) -> i128 {
    if width == 0 {
        return 0;
    }
    if width >= 128 {
        return bits as i128;
    }
    let masked = bits & mask_for_width(width);
    let sign_bit = 1u128 << (width - 1);
    if (masked & sign_bit) != 0 {
        let magnitude = ((1u128 << width) - masked) as i128;
        -magnitude
    } else {
        masked as i128
    }
}

pub fn parse_sexprs(input: &str) -> anyhow::Result<Vec<SExpr>> {
    let tokens = tokenize(input);
    let mut pos = 0;
    let mut exprs = Vec::new();
    while pos < tokens.len() {
        exprs.push(parse_one(&tokens, &mut pos)?);
    }
    Ok(exprs)
}

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == ';' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '(' || c == ')' {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }
        if c == '"' {
            let mut s = String::from('"');
            i += 1;
            while i < chars.len() {
                let ch = chars[i];
                s.push(ch);
                i += 1;
                if ch == '"' {
                    break;
                }
            }
            tokens.push(s);
            continue;
        }
        if c == '|' {
            let mut s = String::new();
            i += 1;
            while i < chars.len() && chars[i] != '|' {
                s.push(chars[i]);
                i += 1;
            }
            if i < chars.len() && chars[i] == '|' {
                i += 1;
            }
            tokens.push(s);
            continue;
        }
        let mut atom = String::new();
        while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '(' && chars[i] != ')' {
            atom.push(chars[i]);
            i += 1;
        }
        tokens.push(atom);
    }
    tokens
}

fn parse_one(tokens: &[String], pos: &mut usize) -> anyhow::Result<SExpr> {
    if *pos >= tokens.len() {
        bail!("Unexpected EOF while parsing S-expression");
    }
    let tok = &tokens[*pos];
    *pos += 1;
    if tok == "(" {
        let mut items = Vec::new();
        while *pos < tokens.len() && tokens[*pos] != ")" {
            items.push(parse_one(tokens, pos)?);
        }
        if *pos >= tokens.len() || tokens[*pos] != ")" {
            bail!("Unclosed '(' in S-expression");
        }
        *pos += 1;
        Ok(SExpr::List(items))
    } else if tok == ")" {
        bail!("Unexpected ')' in S-expression");
    } else {
        Ok(SExpr::Atom(tok.clone()))
    }
}

pub fn decode_smt_value(expr: &SExpr) -> SmtValue {
    match expr {
        SExpr::Atom(a) => {
            if a == "true" {
                return SmtValue::Bool(true);
            }
            if a == "false" {
                return SmtValue::Bool(false);
            }
            if let Some(hex) = a.strip_prefix("#x")
                && let Ok(unsigned) = u128::from_str_radix(hex, 16)
            {
                let width = (hex.len() as u32) * 4;
                return SmtValue::BitVec {
                    width,
                    unsigned,
                    signed: twos_complement_to_signed(unsigned, width),
                };
            }
            if let Some(bin) = a.strip_prefix("#b")
                && let Ok(unsigned) = u128::from_str_radix(bin, 2)
            {
                let width = bin.len() as u32;
                return SmtValue::BitVec {
                    width,
                    unsigned,
                    signed: twos_complement_to_signed(unsigned, width),
                };
            }
            if let Ok(n) = a.parse::<i128>() {
                return SmtValue::Int(n);
            }
            SmtValue::Raw(a.clone())
        }
        SExpr::List(items) => {
            // (- 42)
            if items.len() == 2
                && items[0] == SExpr::Atom("-".to_string())
                && let SmtValueOrInt::Int(n) = decode_int_helper(&items[1])
            {
                return SmtValue::Int(-n);
            }
            // (_ bv123 32)
            if items.len() == 3
                && items[0] == SExpr::Atom("_".to_string())
                && let SExpr::Atom(bv_tok) = &items[1]
                && let SExpr::Atom(w_tok) = &items[2]
                && let Some(dec) = bv_tok.strip_prefix("bv")
                && let (Ok(unsigned), Ok(width)) = (dec.parse::<u128>(), w_tok.parse::<u32>())
            {
                return SmtValue::BitVec {
                    width,
                    unsigned,
                    signed: twos_complement_to_signed(unsigned, width),
                };
            }
            // ((as const (Array ...)) default_val)
            if items.len() == 2
                && let SExpr::List(head) = &items[0]
                && head.len() >= 2
                && head[0] == SExpr::Atom("as".to_string())
                && head[1] == SExpr::Atom("const".to_string())
            {
                let default = decode_smt_value(&items[1]);
                return SmtValue::Array {
                    default: Box::new(default),
                    entries: Vec::new(),
                };
            }
            // (store base idx val)
            if items.len() == 4 && items[0] == SExpr::Atom("store".to_string()) {
                let base = decode_smt_value(&items[1]);
                let idx = decode_smt_value(&items[2]);
                let val = decode_smt_value(&items[3]);
                return match base {
                    SmtValue::Array {
                        default,
                        mut entries,
                    } => {
                        entries.retain(|(k, _)| k != &idx);
                        entries.push((idx, val));
                        SmtValue::Array { default, entries }
                    }
                    other => SmtValue::Array {
                        default: Box::new(other),
                        entries: vec![(idx, val)],
                    },
                };
            }
            SmtValue::Raw(format_sexpr(expr))
        }
    }
}

enum SmtValueOrInt {
    Int(i128),
    Other,
}

fn decode_int_helper(expr: &SExpr) -> SmtValueOrInt {
    if let SExpr::Atom(a) = expr
        && let Ok(n) = a.parse::<i128>()
    {
        return SmtValueOrInt::Int(n);
    }
    SmtValueOrInt::Other
}

pub fn format_sexpr(expr: &SExpr) -> String {
    match expr {
        SExpr::Atom(a) => a.clone(),
        SExpr::List(items) => {
            let parts: Vec<String> = items.iter().map(format_sexpr).collect();
            format!("({})", parts.join(" "))
        }
    }
}

pub fn extract_bindings_from_sexpr(expr: &SExpr) -> BTreeMap<String, SmtValue> {
    let mut map = BTreeMap::new();
    let SExpr::List(items) = expr else {
        return map;
    };

    for item in items {
        let SExpr::List(sub) = item else {
            continue;
        };
        // (get-value (...)) pair: (var_name value)
        if sub.len() == 2
            && let SExpr::Atom(name) = &sub[0]
            && name != "define-fun"
            && name != "model"
        {
            map.insert(name.clone(), decode_smt_value(&sub[1]));
            continue;
        }
        // (model (define-fun name () Sort body) ...)
        if sub.len() >= 5
            && sub[0] == SExpr::Atom("define-fun".to_string())
            && let SExpr::Atom(name) = &sub[1]
            && sub[2] == SExpr::List(Vec::new())
        {
            map.insert(name.clone(), decode_smt_value(&sub[4]));
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_z3_get_value_bitvectors_and_signed_ints() {
        let raw = r#"
sat
((lo #x40000000)
 (hi #x40000000)
 (neg_offset (- 17))
 (flag true))
"#;
        let exprs = parse_sexprs(raw).expect("parses sexprs");
        assert_eq!(exprs[0], SExpr::Atom("sat".to_string()));
        let bindings = extract_bindings_from_sexpr(&exprs[1]);
        assert_eq!(
            bindings.get("lo"),
            Some(&SmtValue::BitVec {
                width: 32,
                unsigned: 0x4000_0000,
                signed: 1_073_741_824,
            })
        );
        assert_eq!(bindings.get("neg_offset"), Some(&SmtValue::Int(-17)));
        assert_eq!(bindings.get("flag"), Some(&SmtValue::Bool(true)));
    }

    #[test]
    fn twos_complement_decodes_negative_32bit_hex() {
        let exprs = parse_sexprs("((x #x80000000) (y #xffffffff))").unwrap();
        let bindings = extract_bindings_from_sexpr(&exprs[0]);
        assert_eq!(
            bindings["x"].as_i128(true),
            Some(i128::from(i32::MIN))
        );
        assert_eq!(bindings["y"].as_i128(true), Some(-1));
        assert_eq!(bindings["y"].as_i128(false), Some(i128::from(u32::MAX)));
    }
}
