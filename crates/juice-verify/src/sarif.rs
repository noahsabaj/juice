use crate::FileVerificationReport;
use crate::proposer::ObligationKind;
use crate::runtime::RuntimeReplayVerdict;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Renders OASIS SARIF v2.1.0 output for GitHub Code Scanning and CI gates.
pub fn render_sarif_report(files: &[FileVerificationReport]) -> Value {
    let mut rules_map: BTreeMap<&'static str, ObligationKind> = BTreeMap::new();
    let mut results: Vec<Value> = Vec::new();

    for file_rep in files {
        let uri = file_rep.path.to_string_lossy().replace('\\', "/");
        for fn_rep in &file_rep.functions {
            for cex in &fn_rep.counterexamples {
                rules_map.insert(cex.kind.rule_id(), cex.kind);
                let inputs_str = cex
                    .inputs
                    .iter()
                    .map(|(k, v)| format!("{k} = {v}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let replay_note = match &cex.runtime_replay {
                    RuntimeReplayVerdict::ConfirmedFailure { toolchain, trace } => {
                        format!(" Confirmed on {toolchain}: {trace}")
                    }
                    _ => String::new(),
                };
                let message = format!(
                    "Formal verification counterexample in `{}`: {} (inputs: `{}`).{}",
                    fn_rep.function_name, cex.description, inputs_str, replay_note
                );

                results.push(json!({
                    "ruleId": cex.kind.rule_id(),
                    "level": "error",
                    "message": {
                        "text": message
                    },
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": uri
                            },
                            "region": {
                                "startLine": cex.line.max(1)
                            }
                        }
                    }],
                    "properties": {
                        "function": fn_rep.function_name,
                        "counterexample": cex.inputs
                    }
                }));
            }
        }
    }

    let rules: Vec<Value> = rules_map
        .into_iter()
        .map(|(id, kind)| {
            json!({
                "id": id,
                "shortDescription": {
                    "text": kind.title()
                }
            })
        })
        .collect();

    json!({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "juice",
                    "version": env!("CARGO_PKG_VERSION"),
                    "informationUri": "https://github.com/noahsabaj/juice",
                    "rules": rules
                }
            },
            "results": results
        }]
    })
}
