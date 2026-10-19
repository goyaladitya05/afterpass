//! OpenVINO execution graph import: the compiled graph `benchmark_app -exec_graph_path exec.xml` writes (the
//! runtime model serialized as IR, its rt_info as layer data), plus the latency from benchmark_app's statistics
//! report. Builds the same `Model` a capture bundle gives (capture/src/afterpass/ov.py), without afterpass.

use serde_json::Value;

use crate::ir::{Graph, Model, Node, NONE};
use crate::ovir;
use crate::provenance::Linker;

/// rt_info keys mapped to node fields; ov.py's `RT_KEYS`. Other keys stay attributes.
const RT_KEYS: [&str; 8] = ["execTimeMcs", "layerType", "primitiveType", "runtimePrecision", "outputPrecisions", "outputLayouts",
    "execOrder", "originalLayersNames"];

/// An IR whose layers carry the runtime's rt_info: no converter writes `primitiveType`.
pub fn is_exec_graph(g: &Graph) -> bool {
    g.nodes.iter().any(|n| n.attrs.iter().any(|(k, _)| k == "primitiveType"))
}

fn rt(n: &Node, key: &str) -> String {
    n.attrs.iter().find(|(k, _)| k == key).map_or(String::new(), |(_, v)| v.clone())
}

/// The compiled graph: kernels, precisions, claims and per-kernel times from the layers' rt_info.
pub fn compiled(xml: &[u8]) -> Result<Graph, String> {
    let mut g = ovir::parse(xml)?;
    if !is_exec_graph(&g) {
        return Err("not an OpenVINO execution graph: its layers carry no runtime information (primitiveType)".into());
    }
    g.meta.clear();
    for n in &mut g.nodes {
        n.kernel = rt(n, "primitiveType");
        n.precision = rt(n, "runtimePrecision");
        n.claims = rt(n, "originalLayersNames").split(',').filter(|c| !c.is_empty()).map(String::from).collect();
        // whole microseconds, or "not_executed"
        n.time_us = rt(n, "execTimeMcs").parse::<u32>().ok().map(|t| t as f32);
        let layer_type = rt(n, "layerType");
        if !layer_type.is_empty() {
            n.op = layer_type;
        }
        let (precisions, layouts) = (rt(n, "outputPrecisions"), rt(n, "outputLayouts"));
        for (k, &t) in n.outputs.iter().enumerate() {
            let t = &mut g.tensors[t as usize];
            if let Some(p) = precisions.split(',').nth(k).filter(|p| !p.is_empty()) {
                t.dtype = p.to_string();
            }
            t.layout = layouts.split(',').nth(k).unwrap_or("").to_string();
        }
        n.attrs.retain(|(k, _)| !RT_KEYS.contains(&k.as_str()));
    }
    // A kernel that only computes constants runs once, when the first inference prepares them; the runtime's
    // profiling entries (what a capture times) leave such kernels out, so their time is not per inference.
    let constant = g.constant_nodes();
    for (n, constant) in g.nodes.iter_mut().zip(constant) {
        if constant {
            n.time_us = None;
        }
    }
    Ok(g)
}

/// What benchmark_app's statistics report (`-report_folder`: benchmark_report.csv, or .json with -json_stats)
/// says about the run: the target device, and the median, fastest and slowest latency in microseconds.
#[derive(Default, Debug)]
pub struct Report {
    pub device: String,
    pub latency_us: Option<f32>,
    pub latency_range: Option<(f32, f32)>,
}

/// True when the bytes start like a benchmark_app statistics report.
pub fn is_report(b: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&b[..b.len().min(64)]);
    head.starts_with("Command line parameters") || (head.trim_start().starts_with('{') && head.contains("\"cmd_options\""))
}

pub fn report(b: &[u8]) -> Result<Report, String> {
    if !is_report(b) {
        return Err("not a benchmark_app statistics report".into());
    }
    let text = String::from_utf8_lossy(b);
    // key: value pairs of every section; the JSON form has the same keys with string values
    let pairs: Vec<(String, String)> = if text.trim_start().starts_with('{') {
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("benchmark_app report JSON: {e}"))?;
        v.as_object().into_iter().flat_map(|o| o.values()).filter_map(Value::as_object).flatten()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()
    } else {
        text.lines().filter_map(|l| l.split_once(';')).map(|(k, v)| (k.to_string(), v.to_string())).collect()
    };
    let get = |k: &str| pairs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
    let ms = |k: &str| get(k).and_then(|v| v.trim().parse::<f32>().ok()).map(|v| v * 1000.0);
    Ok(Report {
        device: get("target device").or(get("-d")).unwrap_or("").to_string(),
        // "latency (ms)" is the median; with -latency_percentile other than 50 the key names that percentile
        latency_us: ms("latency (ms)"),
        latency_range: ms("min latency").zip(ms("max latency")),
    })
}

/// The compiled graph of `xml` linked to `source`, with the report's device and latency when given. The model's
/// source graph is left empty for the caller to fill.
pub fn import(source: &Graph, xml: &[u8], report_bytes: Option<&[u8]>) -> Result<Model, String> {
    let compiled = compiled(xml)?;
    let report = report_bytes.map(report).transpose()?.unwrap_or_default();
    let format = source.meta.iter().find(|(k, _)| k == "format").map_or("", |(_, v)| v.as_str());
    if format.starts_with("TFLite") {
        // seen on EfficientNet-Lite0 with OpenVINO 2026.4.1: claims such as "Convolution_185"
        return Err("OpenVINO names the ops of a TFLite model itself (Convolution_185), so its execution graph links to none of this \
            model's ops: convert the model with ovc, run benchmark_app on the .xml it writes and open that .xml with exec.xml".into());
    }
    let onnx = format.starts_with("ONNX");
    let linker = Linker::new(source, onnx);
    let links: Vec<_> = compiled.nodes.iter().enumerate().flat_map(|(i, n)| linker.link(i as u32, &n.claims)).collect();
    if links.iter().all(|l| l.source == NONE) {
        return Err("the execution graph names no op of this model: was it compiled from another one?".into());
    }
    Ok(Model {
        compiled: Some(compiled),
        links,
        label: format!("openvino {} benchmark_app", report.device).split_whitespace().collect::<Vec<_>>().join(" "),
        runtime: "openvino".into(),
        device: report.device,
        imported: "benchmark_app".into(),
        latency_us: report.latency_us,
        latency_range: report.latency_range,
        ..Default::default()
    })
}
