use juice_smt::SmtSort;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceLanguage {
    Rust,
    C,
    Python,
    TypeScript,
}

impl SourceLanguage {
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "rs" => Some(Self::Rust),
            "c" | "h" => Some(Self::C),
            "py" => Some(Self::Python),
            "ts" | "tsx" | "js" | "mjs" => Some(Self::TypeScript),
            _ => None,
        }
    }

    pub fn from_name(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "rust" | "rs" => Some(Self::Rust),
            "c" | "h" => Some(Self::C),
            "python" | "py" => Some(Self::Python),
            "typescript" | "ts" | "javascript" | "js" => Some(Self::TypeScript),
            _ => None,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Rust => "rs",
            Self::C => "c",
            Self::Python => "py",
            Self::TypeScript => "ts",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::C => "C",
            Self::Python => "Python",
            Self::TypeScript => "TypeScript",
        }
    }

    pub fn checks_fixed_width_overflow(&self) -> bool {
        matches!(self, Self::Rust | Self::C)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifyType {
    Bool,
    BitVec { width: u32, signed: bool },
    Int,
    Slice {
        elem: Box<VerifyType>,
        fixed_len: Option<usize>,
    },
    OptionType(Box<VerifyType>),
    Unit,
}

impl VerifyType {
    pub fn default_int_for(lang: SourceLanguage) -> Self {
        match lang {
            SourceLanguage::Rust | SourceLanguage::C => VerifyType::BitVec {
                width: 32,
                signed: true,
            },
            SourceLanguage::Python | SourceLanguage::TypeScript => VerifyType::Int,
        }
    }

    pub fn parse(raw: &str, lang: SourceLanguage) -> Self {
        let s = raw.trim();
        if s.is_empty() {
            return match lang {
                SourceLanguage::Python | SourceLanguage::TypeScript => VerifyType::Int,
                SourceLanguage::Rust | SourceLanguage::C => VerifyType::Unit,
            };
        }

        match lang {
            SourceLanguage::Rust => parse_rust_type(s),
            SourceLanguage::C => parse_c_type(s),
            SourceLanguage::Python => parse_python_type(s),
            SourceLanguage::TypeScript => parse_ts_type(s),
        }
    }

    pub fn to_smt_sort(&self) -> SmtSort {
        match self {
            VerifyType::Bool => SmtSort::Bool,
            VerifyType::BitVec { width, .. } => SmtSort::BitVec(*width),
            VerifyType::Int => SmtSort::Int,
            VerifyType::Slice { elem, .. } => {
                SmtSort::Array(Box::new(SmtSort::Int), Box::new(elem.to_smt_sort()))
            }
            VerifyType::OptionType(inner) => inner.to_smt_sort(),
            VerifyType::Unit => SmtSort::Bool,
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            VerifyType::BitVec { signed, .. } => *signed,
            VerifyType::Int => true,
            VerifyType::OptionType(inner) => inner.is_signed(),
            VerifyType::Slice { elem, .. } => elem.is_signed(),
            VerifyType::Bool | VerifyType::Unit => false,
        }
    }

    pub fn bit_width(&self) -> Option<u32> {
        match self {
            VerifyType::BitVec { width, .. } => Some(*width),
            _ => None,
        }
    }

    pub fn lean_type(&self) -> &'static str {
        match self {
            VerifyType::Bool => "Bool",
            VerifyType::BitVec { signed: false, .. } => "Nat",
            VerifyType::BitVec { signed: true, .. } | VerifyType::Int => "Int",
            VerifyType::Slice { .. } => "List Int",
            VerifyType::OptionType(_) => "Option Int",
            VerifyType::Unit => "Unit",
        }
    }
}

fn parse_rust_type(s: &str) -> VerifyType {
    let trimmed = s.trim();
    if trimmed == "()" {
        return VerifyType::Unit;
    }
    if trimmed == "bool" {
        return VerifyType::Bool;
    }
    if let Some(inner) = trimmed
        .strip_prefix("Option<")
        .and_then(|r| r.strip_suffix('>'))
    {
        return VerifyType::OptionType(Box::new(parse_rust_type(inner)));
    }
    if let Some(rest) = trimmed.strip_prefix('&') {
        let rest = rest.trim().strip_prefix("mut ").unwrap_or(rest.trim());
        if let Some(inner) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
            return parse_rust_slice_inner(inner);
        }
    }
    if let Some(inner) = trimmed.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        return parse_rust_slice_inner(inner);
    }
    if let Some(inner) = trimmed
        .strip_prefix("Vec<")
        .and_then(|r| r.strip_suffix('>'))
    {
        return VerifyType::Slice {
            elem: Box::new(parse_rust_type(inner)),
            fixed_len: None,
        };
    }

    match trimmed {
        "i8" => VerifyType::BitVec { width: 8, signed: true },
        "i16" => VerifyType::BitVec { width: 16, signed: true },
        "i32" => VerifyType::BitVec { width: 32, signed: true },
        "i64" | "isize" => VerifyType::BitVec { width: 64, signed: true },
        "i128" => VerifyType::BitVec { width: 128, signed: true },
        "u8" => VerifyType::BitVec { width: 8, signed: false },
        "u16" => VerifyType::BitVec { width: 16, signed: false },
        "u32" => VerifyType::BitVec { width: 32, signed: false },
        "u64" | "usize" => VerifyType::BitVec { width: 64, signed: false },
        "u128" => VerifyType::BitVec { width: 128, signed: false },
        _ => VerifyType::BitVec { width: 32, signed: true },
    }
}

fn parse_rust_slice_inner(inner: &str) -> VerifyType {
    if let Some((elem_str, len_str)) = inner.split_once(';') {
        let fixed_len = len_str.trim().parse::<usize>().ok();
        VerifyType::Slice {
            elem: Box::new(parse_rust_type(elem_str)),
            fixed_len,
        }
    } else {
        VerifyType::Slice {
            elem: Box::new(parse_rust_type(inner)),
            fixed_len: None,
        }
    }
}

fn parse_c_type(s: &str) -> VerifyType {
    let cleaned = s
        .replace("const", "")
        .replace("volatile", "")
        .replace("restrict", "");
    let trimmed = cleaned.trim();
    if trimmed == "void" {
        return VerifyType::Unit;
    }
    if trimmed == "bool" || trimmed == "_Bool" {
        return VerifyType::Bool;
    }
    if let Some(base) = trimmed.strip_suffix('*') {
        return VerifyType::Slice {
            elem: Box::new(parse_c_type(base)),
            fixed_len: None,
        };
    }
    if let Some(bracket_idx) = trimmed.find('[') {
        let base = &trimmed[..bracket_idx];
        let rest = &trimmed[bracket_idx + 1..];
        let len_opt = rest
            .strip_suffix(']')
            .and_then(|n| n.trim().parse::<usize>().ok());
        return VerifyType::Slice {
            elem: Box::new(parse_c_type(base)),
            fixed_len: len_opt,
        };
    }

    let norm = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    match norm.as_str() {
        "int8_t" | "signed char" | "char" => VerifyType::BitVec { width: 8, signed: true },
        "uint8_t" | "unsigned char" => VerifyType::BitVec { width: 8, signed: false },
        "int16_t" | "short" | "short int" | "signed short" => {
            VerifyType::BitVec { width: 16, signed: true }
        }
        "uint16_t" | "unsigned short" | "unsigned short int" => {
            VerifyType::BitVec { width: 16, signed: false }
        }
        "int32_t" | "int" | "signed" | "signed int" => {
            VerifyType::BitVec { width: 32, signed: true }
        }
        "uint32_t" | "unsigned" | "unsigned int" => {
            VerifyType::BitVec { width: 32, signed: false }
        }
        "int64_t" | "long" | "long int" | "long long" | "long long int" | "ssize_t"
        | "ptrdiff_t" => VerifyType::BitVec { width: 64, signed: true },
        "uint64_t" | "unsigned long" | "unsigned long int" | "unsigned long long" | "size_t" => {
            VerifyType::BitVec { width: 64, signed: false }
        }
        _ => VerifyType::BitVec { width: 32, signed: true },
    }
}

fn parse_python_type(s: &str) -> VerifyType {
    let trimmed = s.trim();
    if trimmed == "None" {
        return VerifyType::Unit;
    }
    if trimmed == "bool" {
        return VerifyType::Bool;
    }
    if let Some(inner) = trimmed
        .strip_prefix("Optional[")
        .and_then(|r| r.strip_suffix(']'))
    {
        return VerifyType::OptionType(Box::new(parse_python_type(inner)));
    }
    if let Some((left, right)) = trimmed.split_once('|') {
        if right.trim() == "None" {
            return VerifyType::OptionType(Box::new(parse_python_type(left)));
        }
        if left.trim() == "None" {
            return VerifyType::OptionType(Box::new(parse_python_type(right)));
        }
    }
    if let Some(inner) = trimmed
        .strip_prefix("list[")
        .or_else(|| trimmed.strip_prefix("List["))
        .or_else(|| trimmed.strip_prefix("Sequence["))
        .and_then(|r| r.strip_suffix(']'))
    {
        return VerifyType::Slice {
            elem: Box::new(parse_python_type(inner)),
            fixed_len: None,
        };
    }
    if trimmed == "list" || trimmed == "List" {
        return VerifyType::Slice {
            elem: Box::new(VerifyType::Int),
            fixed_len: None,
        };
    }
    VerifyType::Int
}

fn parse_ts_type(s: &str) -> VerifyType {
    let trimmed = s.trim();
    if trimmed == "void" {
        return VerifyType::Unit;
    }
    if trimmed == "boolean" {
        return VerifyType::Bool;
    }
    if let Some((left, right)) = trimmed.split_once('|') {
        let r = right.trim();
        let l = left.trim();
        if r == "null" || r == "undefined" {
            return VerifyType::OptionType(Box::new(parse_ts_type(l)));
        }
        if l == "null" || l == "undefined" {
            return VerifyType::OptionType(Box::new(parse_ts_type(r)));
        }
    }
    if let Some(elem) = trimmed.strip_suffix("[]") {
        return VerifyType::Slice {
            elem: Box::new(parse_ts_type(elem)),
            fixed_len: None,
        };
    }
    if let Some(inner) = trimmed
        .strip_prefix("Array<")
        .or_else(|| trimmed.strip_prefix("ReadonlyArray<"))
        .and_then(|r| r.strip_suffix('>'))
    {
        return VerifyType::Slice {
            elem: Box::new(parse_ts_type(inner)),
            fixed_len: None,
        };
    }
    VerifyType::Int
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocContracts {
    pub requires: Vec<String>,
    pub ensures: Vec<String>,
    pub invariants: Vec<String>,
}

pub fn extract_doc_contracts(comment_block: &str) -> DocContracts {
    let mut contracts = DocContracts::default();
    for raw_line in comment_block.lines() {
        let cleaned = raw_line
            .trim()
            .trim_start_matches("///")
            .trim_start_matches("//!")
            .trim_start_matches("//")
            .trim_start_matches("/**")
            .trim_start_matches("/*")
            .trim_end_matches("*/")
            .trim_start_matches('*')
            .trim_start_matches('#')
            .trim_matches('"')
            .trim();

        for prefix in ["@requires ", "requires:", "@pre "] {
            if let Some(rest) = cleaned.strip_prefix(prefix) {
                let expr = rest.trim().trim_end_matches(';').trim();
                if !expr.is_empty() {
                    contracts.requires.push(expr.to_string());
                }
            }
        }
        for prefix in ["@ensures ", "ensures:", "@post "] {
            if let Some(rest) = cleaned.strip_prefix(prefix) {
                let expr = rest.trim().trim_end_matches(';').trim();
                if !expr.is_empty() {
                    contracts.ensures.push(expr.to_string());
                }
            }
        }
        for prefix in ["@invariant ", "invariant:"] {
            if let Some(rest) = cleaned.strip_prefix(prefix) {
                let expr = rest.trim().trim_end_matches(';').trim();
                if !expr.is_empty() {
                    contracts.invariants.push(expr.to_string());
                }
            }
        }
    }
    contracts
}
