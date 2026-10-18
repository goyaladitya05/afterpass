//! Bundle format v0 reader (spec: notes/bundle-format.md). Input is the decompressed JSON.

use serde_json::Value;

use crate::ir::{Grade, Graph, Link, Model, Node, Via, NONE};

fn s(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

fn graph(v: &Value) -> Graph {
    let mut g = Graph::default();
    if let Some(tensors) = v["tensors"].as_object() {
        for (name, t) in tensors {
            let id = g.tensor(name);
            let tensor = &mut g.tensors[id as usize];
            tensor.dtype = s(&t["dtype"]);
            tensor.layout = s(&t["layout"]);
            tensor.constant = t["kind"] == "initializer";
            tensor.shape = t["shape"].as_array().map(|d| d.iter().map(|x| x.as_i64().unwrap_or(-1)).collect());
            if let Some(n) = t["names"].as_array().and_then(|n| n.first()).and_then(|n| n.as_str()) {
                tensor.name = n.to_string(); // OpenVINO runtime tensor ids are "<node>:<port>"; prefer the real name
            }
        }
    }
    for n in v["nodes"].as_array().into_iter().flatten() {
        let mut ids = |key: &str| -> Vec<u32> {
            n[key].as_array().into_iter().flatten()
                .map(|t| t.as_str().filter(|t| !t.is_empty()).map_or(NONE, |t| g.tensor(t)))
                .collect()
        };
        let inputs = ids("inputs");
        let outputs = ids("outputs");
        let attrs = n["attrs"].as_object().map_or(Vec::new(), |a| a.iter().map(|(k, v)| (k.clone(), s(v))).collect());
        g.nodes.push(Node {
            name: s(&n["name"]),
            op: s(&n["op"]),
            inputs,
            outputs,
            attrs,
            kernel: s(&n["kernel"]),
            precision: s(&n["precision"]),
            claims: n["claims"].as_array().into_iter().flatten().map(s).collect(),
            time_us: None,
        });
    }
    g.finish();
    g
}

pub fn parse(json: &[u8]) -> Result<Model, String> {
    let b: Value = serde_json::from_slice(json).map_err(|e| format!("bundle JSON: {e}"))?;
    if b["format"] != "afterpass-bundle" {
        return Err("not an afterpass bundle".into());
    }
    if b["schema_version"] != 0 {
        return Err(format!("unsupported bundle schema version {}", b["schema_version"]));
    }
    let source = graph(&b["source"]);
    if source.nodes.is_empty() {
        return Err("bundle has no source graph".into());
    }
    let mut compiled = graph(&b["compiled"]);
    if let Some(times) = b["profile"]["nodes"].as_object() {
        for (id, t) in times {
            if let (Ok(i), Some(t)) = (id.parse::<usize>(), t.as_f64()) {
                if let Some(n) = compiled.nodes.get_mut(i) {
                    n.time_us = Some(t as f32);
                }
            }
        }
    }
    let links = b["provenance"].as_array().into_iter().flatten().map(|l| Link {
        compiled: l["compiled"].as_u64().map_or(NONE, |v| v as u32),
        source: l["source"].as_u64().map_or(NONE, |v| v as u32),
        grade: match l["grade"].as_str() {
            Some("explicit") => Grade::Explicit,
            Some("inferred") => Grade::Inferred,
            _ => Grade::Unknown,
        },
        via: Via::parse(l["via"].as_str().unwrap_or("")),
    }).collect();
    let c = &b["compiled"];
    let version = s(&c["version"]).split('-').next().unwrap_or("").to_string();
    Ok(Model {
        source,
        compiled: Some(compiled),
        links,
        label: format!("{} {} {} {}", s(&c["runtime"]), version, s(&c["device"]), s(&c["device_name"])).split_whitespace().collect::<Vec<_>>().join(" "),
        runtime: s(&c["runtime"]),
        device: s(&c["device"]),
        version,
        latency_us: b["profile"]["latency"]["median"].as_f64().map(|v| v as f32),
        latency_range: b["profile"]["latency"]["min"].as_f64().zip(b["profile"]["latency"]["max"].as_f64()).map(|(a, b)| (a as f32, b as f32)),
    })
}
