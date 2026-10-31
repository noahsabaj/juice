use crate::ast::{BinOp, Expr, Param, Stmt, UnaryOp, VerifyFunction};
use crate::lang::{DocContracts, SourceLanguage, VerifyType, extract_doc_contracts};
use anyhow::{Context, bail};
use juice_llm::ModelRouter;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokKind {
    Ident(String),
    Int(i128),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semi,
    Question,
    Dot,
    DotDot,
    DotDotEq,
    Plus,
    Minus,
    Star,
    Slash,
    FloorDiv,
    Percent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Bang,
    Shl,
    Shr,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Assign,
}

#[derive(Debug, Clone)]
struct Tok {
    kind: TokKind,
    line: usize,
}

fn lex_expr(src: &str, lang: SourceLanguage, base_line: usize) -> anyhow::Result<Vec<Tok>> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = base_line;
    let mut out = Vec::new();

    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // Comments
        if lang == SourceLanguage::Python && c == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if lang != SourceLanguage::Python && c == '/' && i + 1 < chars.len() && chars[i + 1] == '/'
        {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                if chars[i] == '\n' {
                    line += 1;
                }
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }
        // String literals inside assert/panic/expect messages -> skip as dummy ident
        if c == '"' || (c == '\'' && matches!(lang, SourceLanguage::Python | SourceLanguage::TypeScript)) {
            let quote = c;
            i += 1;
            let mut s = String::new();
            while i < chars.len() && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 2;
                    continue;
                }
                if chars[i] == '\n' {
                    line += 1;
                }
                s.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }
            out.push(Tok {
                kind: TokKind::Ident(format!("\"{s}\"")),
                line,
            });
            continue;
        }

        // Numbers
        if c.is_ascii_digit() {
            let mut num_str = String::new();
            let mut radix = 10u32;
            if c == '0' && i + 1 < chars.len() {
                match chars[i + 1] {
                    'x' | 'X' => {
                        radix = 16;
                        i += 2;
                    }
                    'b' | 'B' => {
                        radix = 2;
                        i += 2;
                    }
                    _ => {}
                }
            }
            while i < chars.len()
                && (chars[i].is_ascii_hexdigit() || chars[i] == '_')
                && (radix == 16 || chars[i].is_ascii_digit() || chars[i] == '_')
            {
                if chars[i] != '_' {
                    num_str.push(chars[i]);
                }
                i += 1;
            }
            // Strip optional type suffixes like `i32`, `u64`, `usize`, `ULL`, `L`, `n`
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let val = i128::from_str_radix(&num_str, radix)
                .with_context(|| format!("Invalid integer literal `{num_str}` on line {line}"))?;
            out.push(Tok {
                kind: TokKind::Int(val),
                line,
            });
            continue;
        }

        // Identifiers & paths (`i32::MAX`, `INT_MIN`)
        if c.is_ascii_alphabetic() || c == '_' {
            let mut ident = String::new();
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                ident.push(chars[i]);
                i += 1;
            }
            // Rust path `i32::MAX`
            if i + 1 < chars.len() && chars[i] == ':' && chars[i + 1] == ':' {
                i += 2;
                ident.push_str("::");
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    ident.push(chars[i]);
                    i += 1;
                }
            }
            let mapped = match ident.as_str() {
                "and" if lang == SourceLanguage::Python => TokKind::AndAnd,
                "or" if lang == SourceLanguage::Python => TokKind::OrOr,
                "not" if lang == SourceLanguage::Python => TokKind::Bang,
                "INT_MAX" | "INT32_MAX" | "i32::MAX" => TokKind::Int(i128::from(i32::MAX)),
                "INT_MIN" | "INT32_MIN" | "i32::MIN" => TokKind::Int(i128::from(i32::MIN)),
                "UINT_MAX" | "UINT32_MAX" | "u32::MAX" => TokKind::Int(i128::from(u32::MAX)),
                "INT64_MAX" | "i64::MAX" => TokKind::Int(i128::from(i64::MAX)),
                "INT64_MIN" | "i64::MIN" => TokKind::Int(i128::from(i64::MIN)),
                "UINT64_MAX" | "u64::MAX" => TokKind::Int(i128::from(u64::MAX)),
                _ => TokKind::Ident(ident),
            };
            out.push(Tok { kind: mapped, line });
            continue;
        }

        // Multi-char operators
        let rem = &chars[i..];
        if rem.starts_with(&['.', '.', '=']) {
            out.push(Tok { kind: TokKind::DotDotEq, line });
            i += 3;
            continue;
        }
        if rem.starts_with(&['=', '=', '=']) {
            out.push(Tok { kind: TokKind::EqEq, line });
            i += 3;
            continue;
        }
        if rem.starts_with(&['!', '=', '=']) {
            out.push(Tok { kind: TokKind::NotEq, line });
            i += 3;
            continue;
        }
        if rem.starts_with(&['>', '>', '>']) {
            out.push(Tok { kind: TokKind::Shr, line });
            i += 3;
            continue;
        }
        if rem.len() >= 2 {
            let pair = (rem[0], rem[1]);
            let two = match pair {
                ('.', '.') => Some(TokKind::DotDot),
                ('=', '=') => Some(TokKind::EqEq),
                ('!', '=') => Some(TokKind::NotEq),
                ('<', '=') => Some(TokKind::Le),
                ('>', '=') => Some(TokKind::Ge),
                ('<', '<') => Some(TokKind::Shl),
                ('>', '>') => Some(TokKind::Shr),
                ('&', '&') => Some(TokKind::AndAnd),
                ('|', '|') => Some(TokKind::OrOr),
                ('/', '/') if lang == SourceLanguage::Python => Some(TokKind::FloorDiv),
                _ => None,
            };
            if let Some(kind) = two {
                out.push(Tok { kind, line });
                i += 2;
                continue;
            }
        }

        let single = match c {
            '(' => TokKind::LParen,
            ')' => TokKind::RParen,
            '[' => TokKind::LBracket,
            ']' => TokKind::RBracket,
            '{' => TokKind::LBrace,
            '}' => TokKind::RBrace,
            ',' => TokKind::Comma,
            ':' => TokKind::Colon,
            ';' => TokKind::Semi,
            '?' => TokKind::Question,
            '.' => TokKind::Dot,
            '+' => TokKind::Plus,
            '-' => TokKind::Minus,
            '*' => TokKind::Star,
            '/' => TokKind::Slash,
            '%' => TokKind::Percent,
            '&' => TokKind::Amp,
            '|' => TokKind::Pipe,
            '^' => TokKind::Caret,
            '~' => TokKind::Tilde,
            '!' => TokKind::Bang,
            '<' => TokKind::Lt,
            '>' => TokKind::Gt,
            '=' => TokKind::Assign,
            _ => bail!("Unsupported character `{c}` in expression on line {line}"),
        };
        out.push(Tok { kind: single, line });
        i += 1;
    }

    Ok(out)
}

pub fn parse_expr_str(src: &str, lang: SourceLanguage, base_line: usize) -> anyhow::Result<Expr> {
    let tokens = lex_expr(src, lang, base_line)?;
    if tokens.is_empty() {
        bail!("Empty expression on line {base_line}");
    }
    let mut parser = ExprParser {
        tokens: &tokens,
        pos: 0,
        lang,
    };
    let expr = parser.parse_expr()?;
    Ok(expr)
}

struct ExprParser<'a> {
    tokens: &'a [Tok],
    pos: usize,
    lang: SourceLanguage,
}

impl<'a> ExprParser<'a> {
    fn peek(&self) -> Option<&'a Tok> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<&'a Tok> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn cur_line(&self) -> usize {
        self.peek()
            .or_else(|| self.tokens.last())
            .map(|t| t.line)
            .unwrap_or(1)
    }

    fn eat(&mut self, expected: &TokKind) -> bool {
        if let Some(t) = self.peek()
            && &t.kind == expected
        {
            self.pos += 1;
            return true;
        }
        false
    }

    fn expect(&mut self, expected: &TokKind) -> anyhow::Result<()> {
        if self.eat(expected) {
            Ok(())
        } else {
            bail!(
                "Expected {:?} near {:?} on line {}",
                expected,
                self.peek().map(|t| &t.kind),
                self.cur_line()
            )
        }
    }

    fn parse_expr(&mut self) -> anyhow::Result<Expr> {
        // Rust `if cond { a } else { b }` expression
        if let Some(Tok {
            kind: TokKind::Ident(id),
            ..
        }) = self.peek()
            && id == "if"
        {
            self.next();
            let cond = self.parse_or()?;
            self.expect(&TokKind::LBrace)?;
            let then_expr = self.parse_expr()?;
            self.expect(&TokKind::RBrace)?;
            if let Some(Tok {
                kind: TokKind::Ident(el),
                ..
            }) = self.peek()
                && el == "else"
            {
                self.next();
                let else_expr = if let Some(Tok {
                    kind: TokKind::Ident(next_kw),
                    ..
                }) = self.peek()
                    && next_kw == "if"
                {
                    self.parse_expr()?
                } else {
                    self.expect(&TokKind::LBrace)?;
                    let e = self.parse_expr()?;
                    self.expect(&TokKind::RBrace)?;
                    e
                };
                return Ok(Expr::Cond {
                    cond: Box::new(cond),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                });
            }
        }

        let base = self.parse_or()?;

        // C / TS ternary `cond ? a : b`
        if self.eat(&TokKind::Question) {
            let then_expr = self.parse_expr()?;
            self.expect(&TokKind::Colon)?;
            let else_expr = self.parse_expr()?;
            return Ok(Expr::Cond {
                cond: Box::new(base),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            });
        }

        // Python inline conditional `a if cond else b`
        if self.lang == SourceLanguage::Python
            && let Some(Tok {
                kind: TokKind::Ident(id),
                ..
            }) = self.peek()
            && id == "if"
        {
            self.next();
            let cond = self.parse_or()?;
            if let Some(Tok {
                kind: TokKind::Ident(el),
                ..
            }) = self.peek()
                && el == "else"
            {
                self.next();
                let else_expr = self.parse_expr()?;
                return Ok(Expr::Cond {
                    cond: Box::new(cond),
                    then_expr: Box::new(base),
                    else_expr: Box::new(else_expr),
                });
            }
        }

        Ok(base)
    }

    fn parse_or(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_and()?;
        while self.eat(&TokKind::OrOr) {
            let line = self.cur_line();
            let right = self.parse_and()?;
            left = Expr::Binary {
                op: BinOp::Or,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_cmp()?;
        while self.eat(&TokKind::AndAnd) {
            let line = self.cur_line();
            let right = self.parse_cmp()?;
            left = Expr::Binary {
                op: BinOp::And,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_bitor()?;
        loop {
            // Python `x is None` / `x is not None`
            if let Some(Tok {
                kind: TokKind::Ident(id),
                ..
            }) = self.peek()
                && id == "is"
            {
                self.next();
                let negated = self.eat(&TokKind::Bang);
                if let Some(Tok {
                    kind: TokKind::Ident(n),
                    ..
                }) = self.peek()
                    && n == "None"
                {
                    self.next();
                    left = if negated {
                        Expr::IsSome(Box::new(left))
                    } else {
                        Expr::IsNone(Box::new(left))
                    };
                    continue;
                }
            }

            let op = match self.peek().map(|t| &t.kind) {
                Some(TokKind::EqEq) => BinOp::Eq,
                Some(TokKind::NotEq) => BinOp::Ne,
                Some(TokKind::Lt) => BinOp::Lt,
                Some(TokKind::Le) => BinOp::Le,
                Some(TokKind::Gt) => BinOp::Gt,
                Some(TokKind::Ge) => BinOp::Ge,
                _ => break,
            };
            let line = self.cur_line();
            self.next();
            let right = self.parse_bitor()?;
            if right == Expr::NoneLit && matches!(op, BinOp::Eq | BinOp::Ne) {
                left = if op == BinOp::Eq {
                    Expr::IsNone(Box::new(left))
                } else {
                    Expr::IsSome(Box::new(left))
                };
            } else {
                left = Expr::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                    line,
                };
            }
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_bitxor()?;
        while self.eat(&TokKind::Pipe) {
            let line = self.cur_line();
            let right = self.parse_bitxor()?;
            left = Expr::Binary {
                op: BinOp::BitOr,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_bitand()?;
        while self.eat(&TokKind::Caret) {
            let line = self.cur_line();
            let right = self.parse_bitand()?;
            left = Expr::Binary {
                op: BinOp::BitXor,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_shift()?;
        while self.eat(&TokKind::Amp) {
            let line = self.cur_line();
            let right = self.parse_shift()?;
            left = Expr::Binary {
                op: BinOp::BitAnd,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_add()?;
        loop {
            let op = match self.peek().map(|t| &t.kind) {
                Some(TokKind::Shl) => BinOp::Shl,
                Some(TokKind::Shr) => BinOp::Shr,
                _ => break,
            };
            let line = self.cur_line();
            self.next();
            let right = self.parse_add()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_mul()?;
        loop {
            let op = match self.peek().map(|t| &t.kind) {
                Some(TokKind::Plus) => BinOp::Add,
                Some(TokKind::Minus) => BinOp::Sub,
                _ => break,
            };
            let line = self.cur_line();
            self.next();
            let right = self.parse_mul()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek().map(|t| &t.kind) {
                Some(TokKind::Star) => BinOp::Mul,
                Some(TokKind::Slash) | Some(TokKind::FloorDiv) => BinOp::Div,
                Some(TokKind::Percent) => BinOp::Rem,
                _ => break,
            };
            let line = self.cur_line();
            self.next();
            let right = self.parse_unary()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> anyhow::Result<Expr> {
        let line = self.cur_line();
        if self.eat(&TokKind::Minus) {
            let inner = self.parse_unary()?;
            // Fold `-2147483648` literal directly so `-i32::MIN` vs `i32::MIN` doesn't spuriously overflow
            if let Expr::IntLit(n) = inner {
                return Ok(Expr::IntLit(-n));
            }
            return Ok(Expr::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(inner),
                line,
            });
        }
        if self.eat(&TokKind::Bang) {
            let inner = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(inner),
                line,
            });
        }
        if self.eat(&TokKind::Tilde) {
            let inner = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::BitNot,
                expr: Box::new(inner),
                line,
            });
        }

        // C-style prefix cast `(int64_t)x` or `(unsigned int)x`
        if self.lang == SourceLanguage::C
            && let Some(c_ty) = self.try_parse_c_cast_prefix()
        {
            let inner = self.parse_unary()?;
            return Ok(Expr::Cast {
                expr: Box::new(inner),
                target_ty: c_ty,
                line,
            });
        }

        self.parse_postfix()
    }

    fn try_parse_c_cast_prefix(&mut self) -> Option<VerifyType> {
        let saved = self.pos;
        if !self.eat(&TokKind::LParen) {
            return None;
        }
        let mut words = Vec::new();
        while let Some(Tok {
            kind: TokKind::Ident(w),
            ..
        }) = self.peek()
        {
            words.push(w.clone());
            self.next();
        }
        if words.is_empty() || !self.eat(&TokKind::RParen) {
            self.pos = saved;
            return None;
        }
        let joined = words.join(" ");
        let is_c_type = matches!(
            joined.as_str(),
            "int"
                | "unsigned"
                | "unsigned int"
                | "long"
                | "long long"
                | "unsigned long"
                | "unsigned long long"
                | "short"
                | "unsigned short"
                | "char"
                | "unsigned char"
                | "int8_t"
                | "uint8_t"
                | "int16_t"
                | "uint16_t"
                | "int32_t"
                | "uint32_t"
                | "int64_t"
                | "uint64_t"
                | "size_t"
                | "ssize_t"
                | "bool"
        );
        if is_c_type {
            Some(VerifyType::parse(&joined, SourceLanguage::C))
        } else {
            self.pos = saved;
            None
        }
    }

    fn parse_postfix(&mut self) -> anyhow::Result<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            let line = self.cur_line();
            if self.eat(&TokKind::LBracket) {
                let idx = self.parse_expr()?;
                self.expect(&TokKind::RBracket)?;
                expr = Expr::Index {
                    array: Box::new(expr),
                    index: Box::new(idx),
                    line,
                };
                continue;
            }
            // Rust `as <Type>`
            if let Some(Tok {
                kind: TokKind::Ident(kw),
                ..
            }) = self.peek()
                && kw == "as"
            {
                self.next();
                if let Some(Tok {
                    kind: TokKind::Ident(ty_name),
                    ..
                }) = self.next()
                {
                    let target_ty = VerifyType::parse(ty_name, self.lang);
                    expr = Expr::Cast {
                        expr: Box::new(expr),
                        target_ty,
                        line,
                    };
                    continue;
                }
            }
            // TS non-null assertion `x!` (when not `!=` or `!==`, which lex as single tokens)
            if self.lang == SourceLanguage::TypeScript && self.eat(&TokKind::Bang) {
                expr = Expr::Unwrap {
                    expr: Box::new(expr),
                    line,
                };
                continue;
            }
            if self.eat(&TokKind::Dot) {
                let Some(Tok {
                    kind: TokKind::Ident(member),
                    ..
                }) = self.next()
                else {
                    bail!("Expected member name after `.` on line {line}");
                };
                let member = member.clone();

                // `Math.floor(...)`, `Math.abs(...)`, `Math.min(...)`, `Math.max(...)`, `Number.MAX_SAFE_INTEGER`
                if let Expr::Var(ns) = &expr
                    && (ns == "Math" || ns == "Number")
                {
                    if ns == "Number" && member == "MAX_SAFE_INTEGER" {
                        expr = Expr::IntLit(9_007_199_254_740_991);
                        continue;
                    }
                    if ns == "Number" && member == "MIN_SAFE_INTEGER" {
                        expr = Expr::IntLit(-9_007_199_254_740_991);
                        continue;
                    }
                    let args = self.parse_call_args()?;
                    expr = lower_builtin_call(&format!("{ns}.{member}"), args, line)?;
                    continue;
                }

                if self.peek().map(|t| &t.kind) == Some(&TokKind::LParen) {
                    let mut args = self.parse_call_args()?;
                    expr = match member.as_str() {
                        "len" => Expr::Len {
                            array: Box::new(expr),
                        },
                        "unwrap" | "expect" => Expr::Unwrap {
                            expr: Box::new(expr),
                            line,
                        },
                        "is_some" => Expr::IsSome(Box::new(expr)),
                        "is_none" => Expr::IsNone(Box::new(expr)),
                        "wrapping_add" if args.len() == 1 => Expr::Binary {
                            op: BinOp::WrappingAdd,
                            left: Box::new(expr),
                            right: Box::new(args.remove(0)),
                            line,
                        },
                        "wrapping_sub" if args.len() == 1 => Expr::Binary {
                            op: BinOp::WrappingSub,
                            left: Box::new(expr),
                            right: Box::new(args.remove(0)),
                            line,
                        },
                        "wrapping_mul" if args.len() == 1 => Expr::Binary {
                            op: BinOp::WrappingMul,
                            left: Box::new(expr),
                            right: Box::new(args.remove(0)),
                            line,
                        },
                        "saturating_add" if args.len() == 1 => Expr::Binary {
                            op: BinOp::SaturatingAdd,
                            left: Box::new(expr),
                            right: Box::new(args.remove(0)),
                            line,
                        },
                        "saturating_sub" if args.len() == 1 => Expr::Binary {
                            op: BinOp::SaturatingSub,
                            left: Box::new(expr),
                            right: Box::new(args.remove(0)),
                            line,
                        },
                        "midpoint" if args.len() == 1 => {
                            let rhs = args.remove(0);
                            // `(a & b) + ((a ^ b) >> 1)`
                            let band = Expr::Binary {
                                op: BinOp::BitAnd,
                                left: Box::new(expr.clone()),
                                right: Box::new(rhs.clone()),
                                line,
                            };
                            let bxor = Expr::Binary {
                                op: BinOp::BitXor,
                                left: Box::new(expr),
                                right: Box::new(rhs),
                                line,
                            };
                            let shifted = Expr::Binary {
                                op: BinOp::Shr,
                                left: Box::new(bxor),
                                right: Box::new(Expr::IntLit(1)),
                                line,
                            };
                            Expr::Binary {
                                op: BinOp::Add,
                                left: Box::new(band),
                                right: Box::new(shifted),
                                line,
                            }
                        }
                        "abs" if args.is_empty() => build_abs_expr(expr, line),
                        _ => {
                            let mut full_args = vec![expr];
                            full_args.extend(args);
                            Expr::Call {
                                func: member,
                                args: full_args,
                                line,
                            }
                        }
                    };
                } else if member == "length" {
                    expr = Expr::Len {
                        array: Box::new(expr),
                    };
                } else {
                    expr = Expr::Var(member);
                }
                continue;
            }
            break;
        }
        Ok(expr)
    }

    fn parse_call_args(&mut self) -> anyhow::Result<Vec<Expr>> {
        self.expect(&TokKind::LParen)?;
        let mut args = Vec::new();
        while !self.eat(&TokKind::RParen) {
            if self.peek().is_none() {
                bail!("Unclosed `(` in function call on line {}", self.cur_line());
            }
            args.push(self.parse_expr()?);
            let _ = self.eat(&TokKind::Comma);
        }
        Ok(args)
    }

    fn parse_primary(&mut self) -> anyhow::Result<Expr> {
        let line = self.cur_line();
        if self.eat(&TokKind::LParen) {
            let e = self.parse_expr()?;
            self.expect(&TokKind::RParen)?;
            return Ok(e);
        }
        let Some(tok) = self.next() else {
            bail!("Unexpected end of expression on line {line}");
        };
        match &tok.kind {
            TokKind::Int(v) => Ok(Expr::IntLit(*v)),
            TokKind::Ident(id) => match id.as_str() {
                "true" | "True" => Ok(Expr::BoolLit(true)),
                "false" | "False" => Ok(Expr::BoolLit(false)),
                "None" | "null" | "undefined" | "NULL" => Ok(Expr::NoneLit),
                _ => {
                    if self.peek().map(|t| &t.kind) == Some(&TokKind::LParen) {
                        let args = self.parse_call_args()?;
                        lower_builtin_call(id, args, line)
                    } else {
                        Ok(Expr::Var(id.clone()))
                    }
                }
            },
            other => bail!("Unexpected token {:?} on line {}", other, tok.line),
        }
    }
}

fn build_abs_expr(x: Expr, line: usize) -> Expr {
    Expr::Cond {
        cond: Box::new(Expr::Binary {
            op: BinOp::Lt,
            left: Box::new(x.clone()),
            right: Box::new(Expr::IntLit(0)),
            line,
        }),
        then_expr: Box::new(Expr::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(x.clone()),
            line,
        }),
        else_expr: Box::new(x),
    }
}

fn lower_builtin_call(name: &str, mut args: Vec<Expr>, line: usize) -> anyhow::Result<Expr> {
    match name {
        "len" if args.len() == 1 => Ok(Expr::Len {
            array: Box::new(args.remove(0)),
        }),
        "Some" if args.len() == 1 => Ok(Expr::SomeExpr(Box::new(args.remove(0)))),
        "abs" | "labs" | "llabs" | "Math.abs" if args.len() == 1 => {
            Ok(build_abs_expr(args.remove(0), line))
        }
        "min" | "Math.min" if args.len() == 2 => {
            let b = args.pop().unwrap();
            let a = args.pop().unwrap();
            Ok(Expr::Cond {
                cond: Box::new(Expr::Binary {
                    op: BinOp::Lt,
                    left: Box::new(a.clone()),
                    right: Box::new(b.clone()),
                    line,
                }),
                then_expr: Box::new(a),
                else_expr: Box::new(b),
            })
        }
        "max" | "Math.max" if args.len() == 2 => {
            let b = args.pop().unwrap();
            let a = args.pop().unwrap();
            Ok(Expr::Cond {
                cond: Box::new(Expr::Binary {
                    op: BinOp::Gt,
                    left: Box::new(a.clone()),
                    right: Box::new(b.clone()),
                    line,
                }),
                then_expr: Box::new(a),
                else_expr: Box::new(b),
            })
        }
        "Math.floor" | "Math.trunc" if args.len() == 1 => Ok(args.remove(0)),
        _ => Ok(Expr::Call {
            func: name.to_string(),
            args,
            line,
        }),
    }
}

/// Parses all verifiable top-level functions from a Rust, C, Python, or TypeScript source file.
pub fn parse_functions_from_source(
    source: &str,
    lang: SourceLanguage,
) -> anyhow::Result<Vec<VerifyFunction>> {
    match lang {
        SourceLanguage::Python => parse_python_functions(source),
        SourceLanguage::Rust => parse_rust_functions(source),
        SourceLanguage::C => parse_c_functions(source),
        SourceLanguage::TypeScript => parse_ts_functions(source),
    }
}

#[derive(Debug, Deserialize)]
struct LlmLiftedSource {
    normalized_rust_fn: String,
}

/// Parses functions deterministically first, falling back to `ModelRouter` only if deterministic
/// parsing fails and an LLM endpoint is available.
pub async fn parse_or_lift_functions(
    source: &str,
    lang: SourceLanguage,
    router: Option<&ModelRouter>,
) -> anyhow::Result<Vec<VerifyFunction>> {
    match parse_functions_from_source(source, lang) {
        Ok(fns) if !fns.is_empty() => Ok(fns),
        det_res => {
            let Some(router) = router else {
                return det_res;
            };
            let system = "You lift functions into clean, self-contained Rust core syntax (`pub fn name(params) -> Ret { ... }`) preserving exact bit-widths, overflow hazards, array bounds, and control flow. Return JSON `{\"normalized_rust_fn\": \"...\"}`.";
            let user = format!("Source language: {}\n\nSource code:\n{}", lang.display_name(), source);
            if let Ok(lifted) = router.complete_planner_json::<LlmLiftedSource>(system, &user).await
                && let Ok(mut fns) = parse_rust_functions(&lifted.normalized_rust_fn)
            {
                for f in &mut fns {
                    f.language = lang;
                }
                return Ok(fns);
            }
            det_res
        }
    }
}

fn byte_to_line(source: &str, byte_idx: usize) -> usize {
    source[..byte_idx.min(source.len())]
        .bytes()
        .filter(|b| *b == b'\n')
        .count()
        + 1
}

fn find_matching_brace(source: &str, open_idx: usize) -> anyhow::Result<usize> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    let mut i = open_idx;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }
        if b == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Ok(i);
            }
        }
        i += 1;
    }
    bail!("Unmatched `{{` starting at byte {open_idx}")
}

fn collect_preceding_comments(source: &str, sig_start_byte: usize) -> String {
    let prefix = &source[..sig_start_byte];
    let mut lines: Vec<&str> = Vec::new();
    let mut in_block = false;
    for line in prefix.lines().rev() {
        let t = line.trim();
        if t.is_empty() {
            if !lines.is_empty() && !in_block {
                break;
            }
            continue;
        }
        if t.starts_with("#[") {
            continue;
        }
        if t.ends_with("*/") {
            in_block = true;
            lines.push(line);
            if t.starts_with("/*") {
                in_block = false;
            }
            continue;
        }
        if in_block {
            lines.push(line);
            if t.starts_with("/*") {
                in_block = false;
            }
            continue;
        }
        if t.starts_with("//") || t.starts_with('#') {
            lines.push(line);
        } else {
            break;
        }
    }
    lines.reverse();
    lines.join("\n")
}

fn compile_contracts(
    contracts: &DocContracts,
    lang: SourceLanguage,
    line: usize,
) -> (Vec<Expr>, Vec<Expr>, Vec<Expr>) {
    let requires = contracts
        .requires
        .iter()
        .filter_map(|s| parse_expr_str(s, lang, line).ok())
        .collect();
    let ensures = contracts
        .ensures
        .iter()
        .filter_map(|s| parse_expr_str(s, lang, line).ok())
        .collect();
    let invariants = contracts
        .invariants
        .iter()
        .filter_map(|s| parse_expr_str(s, lang, line).ok())
        .collect();
    (requires, ensures, invariants)
}

fn link_c_pointer_lengths(params: &mut [Param]) {
    let len_candidate = params
        .iter()
        .find(|p| {
            matches!(p.ty, VerifyType::BitVec { .. } | VerifyType::Int)
                && matches!(
                    p.name.as_str(),
                    "len" | "length" | "n" | "size" | "count" | "cap"
                )
        })
        .map(|p| p.name.clone());

    if let Some(len_name) = len_candidate {
        for p in params.iter_mut() {
            if matches!(p.ty, VerifyType::Slice { fixed_len: None, .. }) {
                p.linked_len_param = Some(len_name.clone());
            }
        }
    }
}

fn parse_rust_functions(source: &str) -> anyhow::Result<Vec<VerifyFunction>> {
    let re = regex::Regex::new(
        r"(?m)^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:const[ \t]+)?fn[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(([^)]*)\)[ \t]*(?:->[ \t]*([^{]+?))?[ \t]*\{",
    )?;
    let mut functions = Vec::new();
    let mut cursor = 0usize;

    while let Some(caps) = re.captures(&source[cursor..]) {
        let m = caps.get(0).unwrap();
        let abs_start = cursor + m.start();
        let open_brace = cursor + m.end() - 1;
        let close_brace = find_matching_brace(source, open_brace)?;
        cursor = close_brace + 1;

        let name = caps[1].trim().to_string();
        if name == "main" {
            continue;
        }
        // Skip `#[test]` functions
        let preceding = &source[..abs_start];
        let recent_attr = preceding
            .lines()
            .rev()
            .take(4)
            .any(|l| l.trim().starts_with("#[test]") || l.trim().starts_with("#[cfg(test)]"));
        if recent_attr {
            continue;
        }

        let params_raw = caps[2].trim();
        let mut params = Vec::new();
        for part in split_top_level(params_raw, ',') {
            let p = part.trim();
            if p.is_empty() || p.contains("self") {
                continue;
            }
            if let Some((pname, pty)) = p.split_once(':') {
                let clean_name = pname.trim().trim_start_matches("mut ").trim().to_string();
                params.push(Param {
                    name: clean_name,
                    ty: VerifyType::parse(pty, SourceLanguage::Rust),
                    linked_len_param: None,
                });
            }
        }

        let return_ty = caps
            .get(3)
            .map(|m| VerifyType::parse(m.as_str(), SourceLanguage::Rust))
            .unwrap_or(VerifyType::Unit);

        let start_line = byte_to_line(source, abs_start);
        let end_line = byte_to_line(source, close_brace);
        let body_start_line = byte_to_line(source, open_brace);
        let doc_text = collect_preceding_comments(source, abs_start);
        let contracts = extract_doc_contracts(&doc_text);
        let (requires, ensures, loop_invariants) =
            compile_contracts(&contracts, SourceLanguage::Rust, start_line);

        let body_src = &source[open_brace + 1..close_brace];
        let body = parse_brace_block(
            body_src,
            SourceLanguage::Rust,
            body_start_line,
            return_ty != VerifyType::Unit,
            &loop_invariants,
        )?;

        let mut vf = VerifyFunction {
            name,
            language: SourceLanguage::Rust,
            params,
            return_ty,
            requires,
            ensures,
            loop_invariants,
            body,
            raw_source: source[abs_start..=close_brace].to_string(),
            span_start_byte: abs_start,
            span_end_byte: close_brace + 1,
            start_line,
            end_line,
            is_recursive: false,
            has_loops: false,
        };
        vf.refresh_flags();
        functions.push(vf);
    }

    Ok(functions)
}

fn parse_c_functions(source: &str) -> anyhow::Result<Vec<VerifyFunction>> {
    let re = regex::Regex::new(
        r"(?m)^[ \t]*(?:static[ \t]+)?(?:inline[ \t]+)?([A-Za-z_][A-Za-z0-9_\s\*]*?)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(([^;{}]*)\)[ \t]*\{",
    )?;
    let mut functions = Vec::new();
    let mut cursor = 0usize;

    while let Some(caps) = re.captures(&source[cursor..]) {
        let m = caps.get(0).unwrap();
        let abs_start = cursor + m.start();
        let open_brace = cursor + m.end() - 1;
        let ret_raw = caps[1].trim();
        let name = caps[2].trim().to_string();

        if matches!(
            name.as_str(),
            "if" | "while" | "for" | "switch" | "main"
        ) || matches!(ret_raw, "else" | "return")
        {
            cursor = open_brace + 1;
            continue;
        }

        let close_brace = find_matching_brace(source, open_brace)?;
        cursor = close_brace + 1;

        let params_raw = caps[3].trim();
        let mut params = Vec::new();
        if !params_raw.is_empty() && params_raw != "void" {
            for part in split_top_level(params_raw, ',') {
                let p = part.trim();
                if p.is_empty() {
                    continue;
                }
                if let Some((pty, pname)) = split_c_param(p) {
                    params.push(Param {
                        name: pname,
                        ty: VerifyType::parse(&pty, SourceLanguage::C),
                        linked_len_param: None,
                    });
                }
            }
        }
        link_c_pointer_lengths(&mut params);

        let return_ty = VerifyType::parse(ret_raw, SourceLanguage::C);
        let start_line = byte_to_line(source, abs_start);
        let end_line = byte_to_line(source, close_brace);
        let body_start_line = byte_to_line(source, open_brace);
        let doc_text = collect_preceding_comments(source, abs_start);
        let contracts = extract_doc_contracts(&doc_text);
        let (requires, ensures, loop_invariants) =
            compile_contracts(&contracts, SourceLanguage::C, start_line);

        let body_src = &source[open_brace + 1..close_brace];
        let body = parse_brace_block(
            body_src,
            SourceLanguage::C,
            body_start_line,
            false,
            &loop_invariants,
        )?;

        let mut vf = VerifyFunction {
            name,
            language: SourceLanguage::C,
            params,
            return_ty,
            requires,
            ensures,
            loop_invariants,
            body,
            raw_source: source[abs_start..=close_brace].to_string(),
            span_start_byte: abs_start,
            span_end_byte: close_brace + 1,
            start_line,
            end_line,
            is_recursive: false,
            has_loops: false,
        };
        vf.refresh_flags();
        functions.push(vf);
    }

    Ok(functions)
}

fn split_c_param(raw: &str) -> Option<(String, String)> {
    let trimmed = raw.trim();
    // Handle `int arr[]` or `int arr[10]`
    let (without_arr, arr_suffix) = if let Some(b_idx) = trimmed.find('[') {
        (&trimmed[..b_idx], &trimmed[b_idx..])
    } else {
        (trimmed, "")
    };
    let name_end = without_arr.len();
    let bytes = without_arr.as_bytes();
    let mut name_start = name_end;
    while name_start > 0
        && (bytes[name_start - 1].is_ascii_alphanumeric() || bytes[name_start - 1] == b'_')
    {
        name_start -= 1;
    }
    if name_start == name_end || name_start == 0 {
        return None;
    }
    let name = without_arr[name_start..name_end].trim().to_string();
    let ty = format!("{}{}", without_arr[..name_start].trim(), arr_suffix);
    Some((ty, name))
}

fn parse_ts_functions(source: &str) -> anyhow::Result<Vec<VerifyFunction>> {
    let re = regex::Regex::new(
        r"(?m)^[ \t]*(?:export[ \t]+)?function[ \t]+([A-Za-z_$][A-Za-z0-9_$]*)[ \t]*\(([^)]*)\)[ \t]*(?::[ \t]*([^{]+?))?[ \t]*\{",
    )?;
    let mut functions = Vec::new();
    let mut cursor = 0usize;

    while let Some(caps) = re.captures(&source[cursor..]) {
        let m = caps.get(0).unwrap();
        let abs_start = cursor + m.start();
        let open_brace = cursor + m.end() - 1;
        let close_brace = find_matching_brace(source, open_brace)?;
        cursor = close_brace + 1;

        let name = caps[1].trim().to_string();
        let params_raw = caps[2].trim();
        let mut params = Vec::new();
        for part in split_top_level(params_raw, ',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            if let Some((pname, pty)) = p.split_once(':') {
                params.push(Param {
                    name: pname.trim().trim_end_matches('?').to_string(),
                    ty: VerifyType::parse(pty, SourceLanguage::TypeScript),
                    linked_len_param: None,
                });
            } else {
                params.push(Param {
                    name: p.to_string(),
                    ty: VerifyType::Int,
                    linked_len_param: None,
                });
            }
        }

        let return_ty = caps
            .get(3)
            .map(|m| VerifyType::parse(m.as_str(), SourceLanguage::TypeScript))
            .unwrap_or(VerifyType::Int);

        let start_line = byte_to_line(source, abs_start);
        let end_line = byte_to_line(source, close_brace);
        let body_start_line = byte_to_line(source, open_brace);
        let doc_text = collect_preceding_comments(source, abs_start);
        let contracts = extract_doc_contracts(&doc_text);
        let (requires, ensures, loop_invariants) =
            compile_contracts(&contracts, SourceLanguage::TypeScript, start_line);

        let body_src = &source[open_brace + 1..close_brace];
        let body = parse_brace_block(
            body_src,
            SourceLanguage::TypeScript,
            body_start_line,
            false,
            &loop_invariants,
        )?;

        let mut vf = VerifyFunction {
            name,
            language: SourceLanguage::TypeScript,
            params,
            return_ty,
            requires,
            ensures,
            loop_invariants,
            body,
            raw_source: source[abs_start..=close_brace].to_string(),
            span_start_byte: abs_start,
            span_end_byte: close_brace + 1,
            start_line,
            end_line,
            is_recursive: false,
            has_loops: false,
        };
        vf.refresh_flags();
        functions.push(vf);
    }

    Ok(functions)
}

fn split_top_level(s: &str, delim: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth = depth.saturating_sub(1),
            c if c == delim && depth == 0 => {
                parts.push(&s[start..idx]);
                start = idx + c.len_utf8();
            }
            _ => {}
        }
    }
    if start <= s.len() {
        parts.push(&s[start..]);
    }
    parts
}

fn parse_brace_block(
    src: &str,
    lang: SourceLanguage,
    base_line: usize,
    implicit_tail_return: bool,
    loop_invariants: &[Expr],
) -> anyhow::Result<Vec<Stmt>> {
    let mut stmts = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip whitespace & comments
        while i < bytes.len() {
            if bytes[i].is_ascii_whitespace() {
                i += 1;
            } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            } else {
                break;
            }
        }
        if i >= bytes.len() {
            break;
        }

        let stmt_line = base_line + src[..i].bytes().filter(|b| *b == b'\n').count();
        let rest = &src[i..];

        // `if`
        if starts_with_kw(rest, "if") {
            let (if_stmt, consumed) =
                parse_brace_if(rest, lang, stmt_line, implicit_tail_return, loop_invariants)?;
            let next_i = i + consumed;
            let is_last = src[next_i..].trim().is_empty();
            // If not last in Rust, don't treat `if` branches as implicit returns unless they had explicit `return`
            let adjusted = if implicit_tail_return && !is_last {
                parse_brace_if(rest, lang, stmt_line, false, loop_invariants)?.0
            } else {
                if_stmt
            };
            stmts.push(adjusted);
            i = next_i;
            continue;
        }

        // `while`
        if starts_with_kw(rest, "while") {
            let (w_stmt, consumed) = parse_brace_while(rest, lang, stmt_line, loop_invariants)?;
            stmts.push(w_stmt);
            i += consumed;
            continue;
        }

        // `for`
        if starts_with_kw(rest, "for") {
            let (mut f_stmts, consumed) = parse_brace_for(rest, lang, stmt_line, loop_invariants)?;
            stmts.append(&mut f_stmts);
            i += consumed;
            continue;
        }

        // Read until top-level `;` or end of block
        let (stmt_raw, ended_with_semi, consumed) = scan_until_semicolon(rest);
        i += consumed;
        let trimmed = stmt_raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !ended_with_semi && implicit_tail_return && src[i..].trim().is_empty() {
            let expr = parse_expr_str(trimmed, lang, stmt_line)?;
            stmts.push(Stmt::Return {
                value: Some(expr),
                line: stmt_line,
            });
            continue;
        }

        if let Some(mut parsed) = parse_simple_stmt(trimmed, lang, stmt_line)? {
            stmts.append(&mut parsed);
        }
    }

    Ok(stmts)
}

fn starts_with_kw(s: &str, kw: &str) -> bool {
    let Some(after) = s.strip_prefix(kw) else {
        return false;
    };
    after
        .chars()
        .next()
        .map(|c| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(true)
}

fn find_matching_paren(s: &str, open_idx: usize) -> anyhow::Result<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut i = open_idx;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    bail!("Unmatched `(`")
}

fn find_header_brace(s: &str) -> anyhow::Result<usize> {
    let bytes = s.as_bytes();
    let mut paren_depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren_depth += 1,
            b')' => paren_depth = paren_depth.saturating_sub(1),
            b'{' if paren_depth == 0 => return Ok(i),
            _ => {}
        }
        i += 1;
    }
    bail!("Expected `{{` after condition header")
}

fn parse_brace_if(
    src: &str,
    lang: SourceLanguage,
    line: usize,
    implicit_tail_return: bool,
    loop_invariants: &[Expr],
) -> anyhow::Result<(Stmt, usize)> {
    let after_if = &src[2..];
    let trimmed_offset = after_if.len() - after_if.trim_start().len();
    let after_ws = &after_if[trimmed_offset..];

    let (cond_str, body_and_rest_offset) = if after_ws.starts_with('(')
        && matches!(lang, SourceLanguage::C | SourceLanguage::TypeScript)
    {
        let close_p = find_matching_paren(after_ws, 0)?;
        (
            &after_ws[1..close_p],
            2 + trimmed_offset + close_p + 1,
        )
    } else {
        let brace_rel = find_header_brace(after_ws)?;
        (&after_ws[..brace_rel], 2 + trimmed_offset + brace_rel)
    };

    let cond = parse_expr_str(cond_str.trim(), lang, line)?;
    let rest_from_body = &src[body_and_rest_offset..];
    let ws = rest_from_body.len() - rest_from_body.trim_start().len();
    let body_lead = &rest_from_body[ws..];

    let (then_branch, mut total_consumed) = if body_lead.starts_with('{') {
        let close_b = find_matching_brace(body_lead, 0)?;
        let inner = &body_lead[1..close_b];
        let b_line = line + src[..body_and_rest_offset + ws]
            .bytes()
            .filter(|b| *b == b'\n')
            .count();
        let stmts = parse_brace_block(inner, lang, b_line, implicit_tail_return, loop_invariants)?;
        (stmts, body_and_rest_offset + ws + close_b + 1)
    } else {
        let (single, _, c) = scan_until_semicolon(body_lead);
        let stmts = parse_simple_stmt(single.trim(), lang, line)?.unwrap_or_default();
        (stmts, body_and_rest_offset + ws + c)
    };

    // Optional `else`
    let tail = &src[total_consumed..];
    let tail_ws = tail.len() - tail.trim_start().len();
    let after_then = &tail[tail_ws..];
    let mut else_branch = Vec::new();

    if starts_with_kw(after_then, "else") {
        let after_else = &after_then[4..];
        let e_ws = after_else.len() - after_else.trim_start().len();
        let else_lead = &after_else[e_ws..];
        let else_offset = total_consumed + tail_ws + 4 + e_ws;
        let else_line = line + src[..else_offset].bytes().filter(|b| *b == b'\n').count();

        if starts_with_kw(else_lead, "if") {
            let (nested_if, nested_c) = parse_brace_if(
                else_lead,
                lang,
                else_line,
                implicit_tail_return,
                loop_invariants,
            )?;
            else_branch.push(nested_if);
            total_consumed = else_offset + nested_c;
        } else if else_lead.starts_with('{') {
            let close_b = find_matching_brace(else_lead, 0)?;
            let inner = &else_lead[1..close_b];
            else_branch = parse_brace_block(
                inner,
                lang,
                else_line,
                implicit_tail_return,
                loop_invariants,
            )?;
            total_consumed = else_offset + close_b + 1;
        } else {
            let (single, _, c) = scan_until_semicolon(else_lead);
            else_branch = parse_simple_stmt(single.trim(), lang, else_line)?.unwrap_or_default();
            total_consumed = else_offset + c;
        }
    }

    Ok((
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            line,
        },
        total_consumed,
    ))
}

fn parse_brace_while(
    src: &str,
    lang: SourceLanguage,
    line: usize,
    loop_invariants: &[Expr],
) -> anyhow::Result<(Stmt, usize)> {
    let after_kw = &src[5..];
    let brace_rel = find_header_brace(after_kw)?;
    let cond_raw = after_kw[..brace_rel].trim();
    let cond_unwrapped = cond_raw
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(cond_raw);
    let cond = parse_expr_str(cond_unwrapped, lang, line)?;

    let open_brace = 5 + brace_rel;
    let close_brace = find_matching_brace(src, open_brace)?;
    let b_line = line + src[..open_brace].bytes().filter(|b| *b == b'\n').count();
    let body = parse_brace_block(
        &src[open_brace + 1..close_brace],
        lang,
        b_line,
        false,
        loop_invariants,
    )?;

    Ok((
        Stmt::While {
            cond,
            invariants: loop_invariants.to_vec(),
            body,
            line,
        },
        close_brace + 1,
    ))
}

fn parse_brace_for(
    src: &str,
    lang: SourceLanguage,
    line: usize,
    loop_invariants: &[Expr],
) -> anyhow::Result<(Vec<Stmt>, usize)> {
    let after_kw = &src[3..];
    let brace_rel = find_header_brace(after_kw)?;
    let header = after_kw[..brace_rel].trim();
    let open_brace = 3 + brace_rel;
    let close_brace = find_matching_brace(src, open_brace)?;
    let b_line = line + src[..open_brace].bytes().filter(|b| *b == b'\n').count();
    let mut body = parse_brace_block(
        &src[open_brace + 1..close_brace],
        lang,
        b_line,
        false,
        loop_invariants,
    )?;

    if lang == SourceLanguage::Rust {
        // `for i in start..end` or `for i in start..=end`
        if let Some((var_part, range_part)) = header.split_once(" in ") {
            let var_name = var_part.trim().trim_start_matches("mut ").trim().to_string();
            let (start_s, end_s, inclusive) = if let Some((a, b)) = range_part.split_once("..=") {
                (a.trim(), b.trim(), true)
            } else if let Some((a, b)) = range_part.split_once("..") {
                (a.trim(), b.trim(), false)
            } else {
                bail!("Unsupported Rust `for` iterator `{range_part}` on line {line}");
            };
            let start_expr = parse_expr_str(start_s, lang, line)?;
            let end_expr = parse_expr_str(end_s, lang, line)?;
            let init = Stmt::Let {
                name: var_name.clone(),
                ty: None,
                value: start_expr,
                mutable: true,
                line,
            };
            let cond = Expr::Binary {
                op: if inclusive { BinOp::Le } else { BinOp::Lt },
                left: Box::new(Expr::Var(var_name.clone())),
                right: Box::new(end_expr),
                line,
            };
            body.push(Stmt::Assign {
                name: var_name.clone(),
                value: Expr::Binary {
                    op: BinOp::WrappingAdd,
                    left: Box::new(Expr::Var(var_name)),
                    right: Box::new(Expr::IntLit(1)),
                    line: b_line,
                },
                line: b_line,
            });
            return Ok((
                vec![
                    init,
                    Stmt::While {
                        cond,
                        invariants: loop_invariants.to_vec(),
                        body,
                        line,
                    },
                ],
                close_brace + 1,
            ));
        }
    }

    // C / TS `for (init; cond; step)`
    let inner_hdr = header
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(header);
    let clauses: Vec<&str> = split_top_level(inner_hdr, ';');
    if clauses.len() == 3 {
        let mut out = Vec::new();
        if !clauses[0].trim().is_empty()
            && let Some(mut init_s) = parse_simple_stmt(clauses[0].trim(), lang, line)?
        {
            out.append(&mut init_s);
        }
        let cond = if clauses[1].trim().is_empty() {
            Expr::BoolLit(true)
        } else {
            parse_expr_str(clauses[1].trim(), lang, line)?
        };
        if !clauses[2].trim().is_empty()
            && let Some(mut step_s) = parse_simple_stmt(clauses[2].trim(), lang, b_line)?
        {
            body.append(&mut step_s);
        }
        out.push(Stmt::While {
            cond,
            invariants: loop_invariants.to_vec(),
            body,
            line,
        });
        return Ok((out, close_brace + 1));
    }

    bail!("Unsupported `for` loop header `{header}` on line {line}")
}

fn scan_until_semicolon(src: &str) -> (&str, bool, usize) {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            continue;
        }
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            b';' if depth == 0 => return (&src[..i], true, i + 1),
            _ => {}
        }
        i += 1;
    }
    (src, false, bytes.len())
}

fn parse_simple_stmt(
    stmt: &str,
    lang: SourceLanguage,
    line: usize,
) -> anyhow::Result<Option<Vec<Stmt>>> {
    let s = stmt.trim();
    if s.is_empty() {
        return Ok(None);
    }

    // `return ...`
    if starts_with_kw(s, "return") {
        let val_str = s[6..].trim();
        let value = if val_str.is_empty() {
            None
        } else {
            Some(parse_expr_str(val_str, lang, line)?)
        };
        return Ok(Some(vec![Stmt::Return { value, line }]));
    }

    // `panic!(...)` / `unreachable!(...)` / `throw ...` / `raise ...`
    for panic_kw in ["panic!", "unreachable!", "throw", "raise"] {
        if s.starts_with(panic_kw) {
            return Ok(Some(vec![Stmt::Panic {
                message: s.to_string(),
                line,
            }]));
        }
    }

    // `assert_eq!(a, b)`
    if let Some(inner) = s
        .strip_prefix("assert_eq!(")
        .and_then(|r| r.strip_suffix(')'))
    {
        let parts = split_top_level(inner, ',');
        if parts.len() >= 2 {
            let a = parse_expr_str(parts[0].trim(), lang, line)?;
            let b = parse_expr_str(parts[1].trim(), lang, line)?;
            return Ok(Some(vec![Stmt::Assert {
                cond: Expr::Binary {
                    op: BinOp::Eq,
                    left: Box::new(a),
                    right: Box::new(b),
                    line,
                },
                message: Some(s.to_string()),
                line,
            }]));
        }
    }

    // `assert!(cond, ...)` or `assert(cond)` or Python `assert cond`
    if let Some(inner) = s
        .strip_prefix("assert!(")
        .or_else(|| s.strip_prefix("debug_assert!("))
        .or_else(|| s.strip_prefix("assert("))
        .and_then(|r| r.strip_suffix(')'))
    {
        let parts = split_top_level(inner, ',');
        let cond = parse_expr_str(parts[0].trim(), lang, line)?;
        return Ok(Some(vec![Stmt::Assert {
            cond,
            message: Some(s.to_string()),
            line,
        }]));
    }
    if starts_with_kw(s, "assert") {
        let inner = s[6..].trim();
        let parts = split_top_level(inner, ',');
        let cond = parse_expr_str(parts[0].trim(), lang, line)?;
        return Ok(Some(vec![Stmt::Assert {
            cond,
            message: Some(s.to_string()),
            line,
        }]));
    }

    // `i++` / `++i` / `i--` / `--i`
    if let Some(var) = s.strip_suffix("++").or_else(|| s.strip_prefix("++")) {
        let name = var.trim().to_string();
        return Ok(Some(vec![Stmt::Assign {
            name: name.clone(),
            value: Expr::Binary {
                op: BinOp::Add,
                left: Box::new(Expr::Var(name)),
                right: Box::new(Expr::IntLit(1)),
                line,
            },
            line,
        }]));
    }
    if let Some(var) = s.strip_suffix("--").or_else(|| s.strip_prefix("--")) {
        let name = var.trim().to_string();
        return Ok(Some(vec![Stmt::Assign {
            name: name.clone(),
            value: Expr::Binary {
                op: BinOp::Sub,
                left: Box::new(Expr::Var(name)),
                right: Box::new(Expr::IntLit(1)),
                line,
            },
            line,
        }]));
    }

    // Rust / TS `let` / `const` / `var`
    for decl_kw in ["let ", "const ", "var "] {
        if let Some(rest) = s.strip_prefix(decl_kw) {
            let mutable = decl_kw != "const " && (lang != SourceLanguage::Rust || rest.trim_start().starts_with("mut "));
            let after_mut = rest.trim_start().strip_prefix("mut ").unwrap_or(rest.trim_start());
            if let Some((lhs, rhs)) = split_assignment_once(after_mut) {
                let (name, ty) = if let Some((n, t)) = lhs.split_once(':') {
                    (n.trim().to_string(), Some(VerifyType::parse(t, lang)))
                } else {
                    (lhs.trim().to_string(), None)
                };
                let value = parse_expr_str(rhs.trim(), lang, line)?;
                return Ok(Some(vec![Stmt::Let {
                    name,
                    ty,
                    value,
                    mutable,
                    line,
                }]));
            }
        }
    }

    // Compound assignment `x += y`, `x -= y`, `x *= y`, `x /= y`, `x &= y`, `x |= y`, `x ^= y`
    for (sym, op) in [
        ("+=", BinOp::Add),
        ("-=", BinOp::Sub),
        ("*=", BinOp::Mul),
        ("/=", BinOp::Div),
        ("//=", BinOp::Div),
        ("%=", BinOp::Rem),
        ("&=", BinOp::BitAnd),
        ("|=", BinOp::BitOr),
        ("^=", BinOp::BitXor),
    ] {
        if let Some((lhs, rhs)) = s.split_once(sym) {
            let name = lhs.trim().to_string();
            let rhs_expr = parse_expr_str(rhs.trim(), lang, line)?;
            return Ok(Some(vec![Stmt::Assign {
                name: name.clone(),
                value: Expr::Binary {
                    op,
                    left: Box::new(Expr::Var(name)),
                    right: Box::new(rhs_expr),
                    line,
                },
                line,
            }]));
        }
    }

    // Plain `=` assignment or C typed variable declaration (`int x = ...` or `arr[i] = ...`)
    if let Some((lhs, rhs)) = split_assignment_once(s) {
        let lhs_t = lhs.trim();
        let value = parse_expr_str(rhs.trim(), lang, line)?;

        // Array assignment `arr[idx] = val`
        if let Some(b_idx) = lhs_t.find('[')
            && lhs_t.ends_with(']')
        {
            let arr_name = lhs_t[..b_idx].trim().to_string();
            let idx_str = &lhs_t[b_idx + 1..lhs_t.len() - 1];
            let index = parse_expr_str(idx_str, lang, line)?;
            return Ok(Some(vec![Stmt::ArrayAssign {
                array: arr_name,
                index,
                value,
                line,
            }]));
        }

        // Python typed assignment `x: int = ...`
        if lang == SourceLanguage::Python
            && let Some((n, t)) = lhs_t.split_once(':')
        {
            return Ok(Some(vec![Stmt::Let {
                name: n.trim().to_string(),
                ty: Some(VerifyType::parse(t, lang)),
                value,
                mutable: true,
                line,
            }]));
        }

        // C typed variable declaration `int32_t mid = ...`
        if lang == SourceLanguage::C
            && lhs_t.contains(char::is_whitespace)
            && let Some((cty, cname)) = split_c_param(lhs_t)
        {
            return Ok(Some(vec![Stmt::Let {
                name: cname,
                ty: Some(VerifyType::parse(&cty, lang)),
                value,
                mutable: true,
                line,
            }]));
        }

        return Ok(Some(vec![Stmt::Assign {
            name: lhs_t.to_string(),
            value,
            line,
        }]));
    }

    // C uninitialized variable declaration `int acc;`
    if lang == SourceLanguage::C
        && !s.contains('(')
        && s.contains(char::is_whitespace)
        && let Some((cty, cname)) = split_c_param(s)
    {
        return Ok(Some(vec![Stmt::Let {
            name: cname,
            ty: Some(VerifyType::parse(&cty, lang)),
            value: Expr::IntLit(0),
            mutable: true,
            line,
        }]));
    }

    let expr = parse_expr_str(s, lang, line)?;
    Ok(Some(vec![Stmt::ExprStmt { expr, line }]))
}

/// Splits on single `=` that is NOT `==`, `!=`, `<=`, `>=`, `===`, `!==`.
fn split_assignment_once(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            b'=' if depth == 0 => {
                let prev = if i > 0 { bytes[i - 1] } else { b' ' };
                let next = if i + 1 < bytes.len() {
                    bytes[i + 1]
                } else {
                    b' '
                };
                if !matches!(prev, b'=' | b'!' | b'<' | b'>') && next != b'=' {
                    return Some((&s[..i], &s[i + 1..]));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

// ---------------- Python Indentation-Aware Function Parser ----------------

#[derive(Debug, Clone)]
struct PyLine<'a> {
    line_no: usize,
    indent: usize,
    text: &'a str,
    byte_start: usize,
    byte_end: usize,
}

fn parse_python_functions(source: &str) -> anyhow::Result<Vec<VerifyFunction>> {
    let mut py_lines = Vec::new();
    let mut offset = 0usize;
    let mut in_docstring = false;
    let mut doc_lines: Vec<(usize, String)> = Vec::new();

    for (idx, raw_line) in source.split_inclusive('\n').enumerate() {
        let line_no = idx + 1;
        let line_end = offset + raw_line.len();
        let trimmed_cr = raw_line.trim_end_matches(['\r', '\n']);
        let indent = trimmed_cr.chars().take_while(|c| *c == ' ').count()
            + trimmed_cr.chars().take_while(|c| *c == '\t').count() * 4;
        let stripped = strip_py_comment(trimmed_cr).trim();

        // Capture triple-quoted docstrings for `@requires` / `@ensures`
        if trimmed_cr.trim().contains("\"\"\"") || trimmed_cr.trim().contains("'''") {
            doc_lines.push((line_no, trimmed_cr.to_string()));
            let count = trimmed_cr.matches("\"\"\"").count() + trimmed_cr.matches("'''").count();
            if count == 1 {
                in_docstring = !in_docstring;
            }
            offset = line_end;
            continue;
        }
        if in_docstring {
            doc_lines.push((line_no, trimmed_cr.to_string()));
            offset = line_end;
            continue;
        }

        if !stripped.is_empty() && !stripped.starts_with('@') {
            py_lines.push(PyLine {
                line_no,
                indent,
                text: stripped,
                byte_start: offset,
                byte_end: line_end,
            });
        }
        offset = line_end;
    }

    let def_re = regex::Regex::new(
        r"^def[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(([^)]*)\)[ \t]*(?:->[ \t]*([^:]+))?:$",
    )?;
    let mut functions = Vec::new();
    let mut idx = 0usize;

    while idx < py_lines.len() {
        let pl = &py_lines[idx];
        if pl.indent == 0
            && let Some(caps) = def_re.captures(pl.text)
        {
            let name = caps[1].trim().to_string();
            if name.starts_with("test_") {
                idx += 1;
                continue;
            }

            let params_raw = caps[2].trim();
            let mut params = Vec::new();
            for part in split_top_level(params_raw, ',') {
                let p = part.trim();
                if p.is_empty() || p == "self" {
                    continue;
                }
                let p_no_default = p.split('=').next().unwrap_or(p).trim();
                if let Some((pname, pty)) = p_no_default.split_once(':') {
                    params.push(Param {
                        name: pname.trim().to_string(),
                        ty: VerifyType::parse(pty, SourceLanguage::Python),
                        linked_len_param: None,
                    });
                } else {
                    params.push(Param {
                        name: p_no_default.to_string(),
                        ty: VerifyType::Int,
                        linked_len_param: None,
                    });
                }
            }

            let return_ty = caps
                .get(3)
                .map(|m| VerifyType::parse(m.as_str(), SourceLanguage::Python))
                .unwrap_or(VerifyType::Int);

            let start_byte = pl.byte_start;
            let start_line = pl.line_no;
            idx += 1;
            let body_start = idx;
            while idx < py_lines.len() && py_lines[idx].indent > 0 {
                idx += 1;
            }
            let body_slice = &py_lines[body_start..idx];
            let end_byte = body_slice
                .last()
                .map(|l| l.byte_end)
                .unwrap_or(pl.byte_end);
            let end_line = body_slice
                .last()
                .map(|l| l.line_no)
                .unwrap_or(start_line);

            let mut doc_text = collect_preceding_comments(source, start_byte);
            for (dline, dtext) in &doc_lines {
                if *dline > start_line && *dline <= end_line + 3 {
                    doc_text.push('\n');
                    doc_text.push_str(dtext);
                }
            }
            let contracts = extract_doc_contracts(&doc_text);
            let (requires, ensures, loop_invariants) =
                compile_contracts(&contracts, SourceLanguage::Python, start_line);

            let mut sub_pos = 0usize;
            let body = parse_py_suite(body_slice, &mut sub_pos, &loop_invariants)?;

            // Upgrade untyped Python params that are indexed (`arr[i]` or `len(arr)`) to `Slice`
            let mut vf = VerifyFunction {
                name,
                language: SourceLanguage::Python,
                params,
                return_ty,
                requires,
                ensures,
                loop_invariants,
                body,
                raw_source: source[start_byte..end_byte].trim_end().to_string(),
                span_start_byte: start_byte,
                span_end_byte: end_byte,
                start_line,
                end_line,
                is_recursive: false,
                has_loops: false,
            };
            infer_python_slice_params(&mut vf);
            vf.refresh_flags();
            functions.push(vf);
            continue;
        }
        idx += 1;
    }

    Ok(functions)
}

fn strip_py_comment(line: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    for (idx, c) in line.char_indices() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double => return &line[..idx],
            _ => {}
        }
    }
    line
}

fn parse_py_suite(
    lines: &[PyLine<'_>],
    pos: &mut usize,
    loop_invariants: &[Expr],
) -> anyhow::Result<Vec<Stmt>> {
    let mut stmts = Vec::new();
    let Some(first) = lines.get(*pos) else {
        return Ok(stmts);
    };
    let suite_indent = first.indent;

    while *pos < lines.len() && lines[*pos].indent >= suite_indent {
        let cur = &lines[*pos];
        if cur.indent > suite_indent {
            *pos += 1;
            continue;
        }
        let line_no = cur.line_no;
        let text = cur.text;

        if text == "pass" {
            *pos += 1;
            continue;
        }

        if let Some(cond_s) = text.strip_prefix("if ").and_then(|s| s.strip_suffix(':')) {
            let cond = parse_expr_str(cond_s.trim(), SourceLanguage::Python, line_no)?;
            *pos += 1;
            let then_branch = parse_py_suite(lines, pos, loop_invariants)?;
            let else_branch = parse_py_elif_else(lines, pos, suite_indent, loop_invariants)?;
            stmts.push(Stmt::If {
                cond,
                then_branch,
                else_branch,
                line: line_no,
            });
            continue;
        }

        if let Some(cond_s) = text
            .strip_prefix("while ")
            .and_then(|s| s.strip_suffix(':'))
        {
            let cond = parse_expr_str(cond_s.trim(), SourceLanguage::Python, line_no)?;
            *pos += 1;
            let body = parse_py_suite(lines, pos, loop_invariants)?;
            stmts.push(Stmt::While {
                cond,
                invariants: loop_invariants.to_vec(),
                body,
                line: line_no,
            });
            continue;
        }

        if let Some(hdr) = text.strip_prefix("for ").and_then(|s| s.strip_suffix(':'))
            && let Some((var_s, iter_s)) = hdr.split_once(" in ")
        {
            let var_name = var_s.trim().to_string();
            let iter_t = iter_s.trim();
            if let Some(range_inner) = iter_t
                .strip_prefix("range(")
                .and_then(|r| r.strip_suffix(')'))
            {
                let args = split_top_level(range_inner, ',');
                let (start_e, end_e) = if args.len() == 1 {
                    (
                        Expr::IntLit(0),
                        parse_expr_str(args[0].trim(), SourceLanguage::Python, line_no)?,
                    )
                } else {
                    (
                        parse_expr_str(args[0].trim(), SourceLanguage::Python, line_no)?,
                        parse_expr_str(args[1].trim(), SourceLanguage::Python, line_no)?,
                    )
                };
                *pos += 1;
                let mut body = parse_py_suite(lines, pos, loop_invariants)?;
                body.push(Stmt::Assign {
                    name: var_name.clone(),
                    value: Expr::Binary {
                        op: BinOp::Add,
                        left: Box::new(Expr::Var(var_name.clone())),
                        right: Box::new(Expr::IntLit(1)),
                        line: line_no,
                    },
                    line: line_no,
                });
                stmts.push(Stmt::Let {
                    name: var_name.clone(),
                    ty: Some(VerifyType::Int),
                    value: start_e,
                    mutable: true,
                    line: line_no,
                });
                stmts.push(Stmt::While {
                    cond: Expr::Binary {
                        op: BinOp::Lt,
                        left: Box::new(Expr::Var(var_name)),
                        right: Box::new(end_e),
                        line: line_no,
                    },
                    invariants: loop_invariants.to_vec(),
                    body,
                    line: line_no,
                });
                continue;
            }
        }

        *pos += 1;
        if let Some(mut simple) = parse_simple_stmt(text, SourceLanguage::Python, line_no)? {
            stmts.append(&mut simple);
        }
    }

    Ok(stmts)
}

fn parse_py_elif_else(
    lines: &[PyLine<'_>],
    pos: &mut usize,
    suite_indent: usize,
    loop_invariants: &[Expr],
) -> anyhow::Result<Vec<Stmt>> {
    if *pos >= lines.len() || lines[*pos].indent != suite_indent {
        return Ok(Vec::new());
    }
    let cur = &lines[*pos];
    if let Some(elif_cond) = cur
        .text
        .strip_prefix("elif ")
        .and_then(|s| s.strip_suffix(':'))
    {
        let line_no = cur.line_no;
        let cond = parse_expr_str(elif_cond.trim(), SourceLanguage::Python, line_no)?;
        *pos += 1;
        let then_branch = parse_py_suite(lines, pos, loop_invariants)?;
        let else_branch = parse_py_elif_else(lines, pos, suite_indent, loop_invariants)?;
        return Ok(vec![Stmt::If {
            cond,
            then_branch,
            else_branch,
            line: line_no,
        }]);
    }
    if cur.text == "else:" {
        *pos += 1;
        return parse_py_suite(lines, pos, loop_invariants);
    }
    Ok(Vec::new())
}

fn infer_python_slice_params(vf: &mut VerifyFunction) {
    for p in &mut vf.params {
        if p.ty == VerifyType::Int && stmts_use_as_slice(&vf.body, &p.name) {
            p.ty = VerifyType::Slice {
                elem: Box::new(VerifyType::Int),
                fixed_len: None,
            };
        }
    }
}

fn stmts_use_as_slice(stmts: &[Stmt], name: &str) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Let { value, .. }
        | Stmt::Assign { value, .. }
        | Stmt::Assert { cond: value, .. }
        | Stmt::ExprStmt { expr: value, .. } => expr_uses_as_slice(value, name),
        Stmt::ArrayAssign { array, index, value, .. } => {
            array == name || expr_uses_as_slice(index, name) || expr_uses_as_slice(value, name)
        }
        Stmt::Return { value: Some(v), .. } => expr_uses_as_slice(v, name),
        Stmt::Return { value: None, .. } | Stmt::Panic { .. } => false,
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            expr_uses_as_slice(cond, name)
                || stmts_use_as_slice(then_branch, name)
                || stmts_use_as_slice(else_branch, name)
        }
        Stmt::While { cond, body, .. } => {
            expr_uses_as_slice(cond, name) || stmts_use_as_slice(body, name)
        }
    })
}

fn expr_uses_as_slice(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Index { array, index, .. } => {
            matches!(array.as_ref(), Expr::Var(v) if v == name)
                || expr_uses_as_slice(array, name)
                || expr_uses_as_slice(index, name)
        }
        Expr::Len { array } => {
            matches!(array.as_ref(), Expr::Var(v) if v == name) || expr_uses_as_slice(array, name)
        }
        Expr::Unary { expr: e, .. }
        | Expr::Cast { expr: e, .. }
        | Expr::Unwrap { expr: e, .. }
        | Expr::IsSome(e)
        | Expr::IsNone(e)
        | Expr::SomeExpr(e) => expr_uses_as_slice(e, name),
        Expr::Binary { left, right, .. } => {
            expr_uses_as_slice(left, name) || expr_uses_as_slice(right, name)
        }
        Expr::Cond {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_uses_as_slice(cond, name)
                || expr_uses_as_slice(then_expr, name)
                || expr_uses_as_slice(else_expr, name)
        }
        Expr::Call { args, .. } => args.iter().any(|a| expr_uses_as_slice(a, name)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rust_midpoint_and_doc_contracts() {
        let src = r#"
/// @requires lo <= hi
/// @ensures result >= lo && result <= hi
pub fn midpoint(lo: i32, hi: i32) -> i32 {
    (lo + hi) / 2
}
"#;
        let fns = parse_functions_from_source(src, SourceLanguage::Rust).unwrap();
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].name, "midpoint");
        assert_eq!(fns[0].params.len(), 2);
        assert_eq!(fns[0].requires.len(), 1);
        assert_eq!(fns[0].ensures.len(), 1);
        assert!(matches!(fns[0].body[0], Stmt::Return { value: Some(_), .. }));
    }

    #[test]
    fn parses_python_indented_function() {
        let src = r#"
def safe_avg(arr: list[int]) -> int:
    """
    @requires len(arr) > 0
    """
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total // len(arr)
"#;
        let fns = parse_functions_from_source(src, SourceLanguage::Python).unwrap();
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].name, "safe_avg");
        assert!(fns[0].has_loops);
        assert_eq!(fns[0].requires.len(), 1);
    }
}
