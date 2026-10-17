//! Analysis over a loaded session: structural provenance, overlay categories, Findings, search, inspection.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::ir::{Grade, Graph, Node, Via, NONE};
use crate::view::{Item, Pane};

#[derive(Clone, Copy, Debug)]
pub struct LinkRef {
    pub other: u32,
    pub grade: Grade,
    pub via: Via,
}

/// Per pane, per node: links to nodes of the other pane (index 0 is the source pane).
pub type Links = [Vec<Vec<LinkRef>>; 2];

const IO: [&str; 4] = ["Parameter", "Result", "Input", "Output"];

/// Nodes that do work at inference time: not inputs or outputs, not computed from constants only.
pub fn data_path(p: &Pane, v: usize) -> bool {
    !p.hier.folded[v] && !IO.contains(&p.graph.nodes[v].op.as_str())
}

/// Recover links the runtime's names lost, in two passes, all graded inferred via structure.
///
/// 1. Compiled kernels whose claimed names match nothing (a fused MoE kernel named `ReduceSum_123/MOECompressed`)
///    are grouped into connected clusters. The source ops lying between the sources of the cluster's nearest
///    linked upstream kernels and those of its nearest linked downstream kernels are linked to the cluster.
/// 2. A remaining source op nobody claims, whose outputs (through other unclaimed ops) reach exactly one kernel,
///    was computed inside that kernel or eliminated by it: pattern fusions that keep only the last op's name
///    (OpenVINO RMS, RoPE, GELU, SiLU) and layout ops absorbed by TensorRT Myelin.
///
/// Returns how many source ops gained a link.
pub fn infer_structure(src: &Pane, cmp: &Pane, links: &mut Links) -> usize {
    let (ns, nc) = (src.graph.nodes.len(), cmp.graph.nodes.len());
    let named: Vec<bool> = (0..ns).map(|v| !links[0][v].is_empty()).collect();
    let mut linked_src = vec![false; ns];
    let mut add = |links: &mut Links, v: usize, k: u32, linked_src: &mut Vec<bool>| {
        links[0][v].push(LinkRef { other: k, grade: Grade::Inferred, via: Via::Structure });
        links[1][k as usize].push(LinkRef { other: v as u32, grade: Grade::Inferred, via: Via::Structure });
        linked_src[v] = true;
    };

    // pass 1: clusters of compiled kernels without a resolvable claim
    let orphan = |links: &Links, k: usize| data_path(cmp, k) && links[1][k].is_empty();
    let mut seen = vec![false; nc];
    for k0 in 0..nc {
        if seen[k0] || !orphan(links, k0) {
            continue;
        }
        let (mut cluster, mut stack) = (Vec::new(), vec![k0]);
        let (mut up, mut down) = (Vec::new(), Vec::new());
        seen[k0] = true;
        while let Some(k) = stack.pop() {
            cluster.push(k as u32);
            for (nb, dir) in [(&cmp.preds[k], &mut up), (&cmp.succs[k], &mut down)] {
                for &w in nb {
                    let w = w as usize;
                    if orphan(links, w) {
                        if !seen[w] {
                            seen[w] = true;
                            stack.push(w);
                        }
                    } else if !links[1][w].is_empty() {
                        dir.push(w);
                    }
                }
            }
        }
        let sources = |ks: &[usize]| -> Vec<usize> {
            ks.iter().flat_map(|&k| links[1][k].iter().map(|l| l.other as usize)).collect()
        };
        let (sa, sd) = (sources(&up), sources(&down));
        let (Some(&lo), Some(&hi)) = (sa.iter().min(), sd.iter().max()) else { continue };
        // forward from upstream sources and backward from downstream sources, within [lo, hi]
        let reach = |starts: &[usize], adj: &[Vec<u32>]| -> Vec<bool> {
            let mut mark = vec![false; ns];
            let mut st: Vec<usize> = starts.to_vec();
            while let Some(v) = st.pop() {
                for &w in &adj[v] {
                    let w = w as usize;
                    if w >= lo && w <= hi && !mark[w] {
                        mark[w] = true;
                        st.push(w);
                    }
                }
            }
            mark
        };
        let (fwd, bwd) = (reach(&sa, &src.succs), reach(&sd, &src.preds));
        let region: Vec<usize> = (lo..=hi).filter(|&v| fwd[v] && bwd[v] && !named[v] && !linked_src[v] && data_path(src, v)).collect();
        if region.len() > 50_000 {
            continue;
        }
        for v in region {
            for &k in &cluster {
                add(links, v, k, &mut linked_src);
            }
        }
    }

    // pass 2: unclaimed source ops whose outputs reach exactly one kernel
    // reach[v]: None = reaches no kernel yet, Some(k) = exactly kernel k, Some(NONE) = several
    let mut reach: Vec<Option<u32>> = vec![None; ns];
    for v in (0..ns).rev() {
        if !data_path(src, v) {
            continue;
        }
        if !links[0][v].is_empty() {
            let first = links[0][v][0].other;
            reach[v] = Some(if links[0][v].iter().all(|l| l.other == first) { first } else { NONE });
            continue;
        }
        let mut r = None;
        for &s in &src.succs[v] {
            if let Some(k) = reach[s as usize] {
                r = match r {
                    None => Some(k),
                    Some(x) if x == k => Some(x),
                    _ => Some(NONE),
                };
            }
        }
        reach[v] = r;
        if let Some(k) = r.filter(|&k| k != NONE) {
            add(links, v, k, &mut linked_src);
        }
    }
    linked_src.iter().filter(|&&x| x).count()
}

/// Compute precision class of a dtype or runtime precision string, or None for index and bool types.
pub fn precision_class(s: &str) -> Option<&'static str> {
    Some(match s.to_ascii_lowercase().as_str() {
        "f32" | "fp32" | "float" | "float32" => "fp32",
        "f16" | "fp16" | "half" | "float16" => "fp16",
        "bf16" | "bfloat16" => "bf16",
        "i8" | "u8" | "int8" | "uint8" => "int8",
        "i4" | "u4" | "int4" | "uint4" | "nf4" => "int4",
        "f8e4m3" | "f8e5m2" | "fp8" | "float8e4m3fn" | "float8e5m2" => "fp8",
        _ => return None,
    })
}

pub const PRECISIONS: [&str; 6] = ["fp32", "fp16", "bf16", "int8", "int4", "fp8"];

fn out_tensor<'a>(g: &'a Graph, n: &Node) -> Option<&'a crate::ir::Tensor> {
    n.outputs.first().map(|&t| &g.tensors[t as usize])
}

pub fn node_precision(g: &Graph, n: &Node) -> Option<&'static str> {
    precision_class(&n.precision).or_else(|| out_tensor(g, n).and_then(|t| precision_class(&t.dtype)))
}

/// Kernel family: the kernel name without its precision suffix (`brgconv_avx2_f32` gives `brgconv_avx2`,
/// `jit:ir__f16` gives `jit:ir`). TensorRT tactic names are unique per layer, so its layer type is used.
pub fn kernel_family(runtime: &str, n: &Node) -> String {
    if runtime == "tensorrt" || n.kernel.is_empty() {
        return n.op.clone();
    }
    let k = n.kernel.as_str();
    match k.rfind('_') {
        Some(i) if precision_class(&k[i + 1..]).is_some() || matches!(&k[i + 1..], "i32" | "i64" | "dynamic" | "boolean") => {
            k[..i].trim_end_matches('_').to_string()
        }
        _ => k.to_string(),
    }
}

/// Short layout name: `acdb`, `bfyx`, or the bracketed part of a TensorRT format (`(Linear) Row major ...`).
pub fn layout_name(g: &Graph, n: &Node) -> String {
    let l = out_tensor(g, n).map_or("", |t| t.layout.as_str());
    match (l.find('('), l.find(')')) {
        (Some(0), Some(e)) => l[1..e].to_string(),
        _ => l.to_string(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Overlay {
    Precision,
    Layout,
    Device,
    Kernel,
}

impl Overlay {
    pub fn parse(s: &str) -> Option<Overlay> {
        Some(match s {
            "precision" => Overlay::Precision,
            "layout" => Overlay::Layout,
            "device" => Overlay::Device,
            "kernel" => Overlay::Kernel,
            _ => return None,
        })
    }
}

/// Category per box for an overlay, as an index into the returned legend (u16::MAX for none). Collapsed boxes
/// take the most common category among their nodes. The precision legend has a fixed order so colours mean
/// the same thing everywhere.
pub fn categories(p: &Pane, mode: Overlay, runtime: &str, device: &str) -> (Vec<u16>, Vec<String>) {
    let g = &p.graph;
    let cat: Vec<String> = g.nodes.iter().enumerate().map(|(v, n)| {
        if !data_path(p, v) {
            return String::new();
        }
        match mode {
            Overlay::Precision => node_precision(g, n).unwrap_or("").to_string(),
            Overlay::Layout => layout_name(g, n),
            Overlay::Device => {
                if matches!(n.kernel.as_str(), "DPU" | "Shave") { n.kernel.clone() } else if p.compiled { device.to_string() } else { String::new() }
            }
            Overlay::Kernel => if p.compiled { kernel_family(runtime, n) } else { String::new() },
        }
    }).collect();
    let mut legend: Vec<String> = Vec::new();
    if mode == Overlay::Precision {
        legend = PRECISIONS.iter().map(|s| s.to_string()).collect();
    } else {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for c in cat.iter().filter(|c| !c.is_empty()) {
            *counts.entry(c).or_default() += 1;
        }
        let mut v: Vec<(&str, usize)> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        legend.extend(v.into_iter().map(|(c, _)| c.to_string()));
    }
    let index: HashMap<&str, u16> = legend.iter().enumerate().map(|(i, c)| (c.as_str(), i as u16)).collect();
    let boxes = (0..p.scene.kinds.len() as u32).map(|b| {
        let mut counts: HashMap<u16, usize> = HashMap::new();
        for v in p.box_nodes(b) {
            if let Some(&i) = index.get(cat[v as usize].as_str()) {
                *counts.entry(i).or_default() += 1;
            }
        }
        counts.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map_or(u16::MAX, |(i, _)| i)
    }).collect();
    (boxes, legend)
}

fn fmt_us(us: f32) -> String {
    if us >= 1000.0 { format!("{:.2} ms", us / 1000.0) } else { format!("{us:.0} us") }
}

fn item(section: &str, severity: &str, title: String, detail: String, targets: Vec<Value>) -> Value {
    json!({ "section": section, "severity": severity, "title": title, "detail": detail, "targets": targets })
}

fn node_target(pane: usize, v: u32) -> Value {
    json!([pane, v, 0])
}

/// The Findings list: top time sinks and unattributed time, unaccounted source ops, kernels the runtime
/// inserted, precision fallbacks, and outlier blocks. Each item names the nodes it jumps to.
pub fn findings(panes: &[Pane], links: &Links, latency_us: Option<f32>, inferred: usize) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(c) = panes.get(1) {
        let g = &c.graph;
        let kernels: Vec<u32> = (0..g.nodes.len()).filter(|&v| data_path(c, v)).map(|v| v as u32).collect();
        // time
        let mut timed: Vec<(f32, u32)> = kernels.iter().filter_map(|&v| g.nodes[v as usize].time_us.map(|t| (t, v))).collect();
        timed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let total: f32 = timed.iter().map(|x| x.0).sum();
        let base = latency_us.filter(|&l| l > 0.0).unwrap_or(total);
        for (t, v) in timed.iter().take(10) {
            let n = &g.nodes[*v as usize];
            out.push(item("time", "info", format!("{}  {}", n.op, fmt_us(*t)),
                format!("{:.1}% of latency  {}", 100.0 * t / base, n.name), vec![node_target(1, *v)]));
        }
        if let Some(l) = latency_us.filter(|&l| l > 0.0 && total > 0.0 && total < l * 0.9) {
            out.push(item("time", "warn", format!("unattributed  {}", fmt_us(l - total)),
                format!("{:.0}% of latency is in no kernel's profile (dispatch, sync, host work)", 100.0 * (l - total) / l), vec![]));
        }
        // kernels the runtime inserted: data-path compiled nodes with no source link
        let mut inserted: HashMap<&str, (Vec<u32>, f32)> = HashMap::new();
        for &v in &kernels {
            if links[1][v as usize].is_empty() {
                let e = inserted.entry(g.nodes[v as usize].op.as_str()).or_default();
                e.0.push(v);
                e.1 += g.nodes[v as usize].time_us.unwrap_or(0.0);
            }
        }
        let mut ins: Vec<_> = inserted.into_iter().collect();
        ins.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap().then(b.1.0.len().cmp(&a.1.0.len())));
        for (op, (vs, t)) in ins {
            let time = if t > 0.0 { format!(", {}", fmt_us(t)) } else { String::new() };
            out.push(item("inserted", "info", format!("{op}  x{}", vs.len()),
                format!("no source op matches the names they claim: inserted by the runtime or renamed{time}"), vs.iter().map(|&v| node_target(1, v)).collect()));
        }
        // precision fallbacks against the dominant compute precision (by time when profiled)
        let mut weight: HashMap<&str, f32> = HashMap::new();
        for &v in &kernels {
            if let Some(pc) = node_precision(g, &g.nodes[v as usize]) {
                *weight.entry(pc).or_default() += g.nodes[v as usize].time_us.unwrap_or(1.0).max(1e-3);
            }
        }
        if let Some((&dominant, _)) = weight.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            let mut other: HashMap<&str, (Vec<u32>, f32)> = HashMap::new();
            for &v in &kernels {
                let n = &g.nodes[v as usize];
                if let Some(pc) = node_precision(g, n).filter(|&pc| pc != dominant) {
                    let e = other.entry(pc).or_default();
                    e.0.push(v);
                    e.1 += n.time_us.unwrap_or(0.0);
                }
            }
            let mut other: Vec<_> = other.into_iter().collect();
            other.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
            for (pc, (vs, t)) in other {
                let time = if t > 0.0 { format!(", {}", fmt_us(t)) } else { String::new() };
                out.push(item("precision", "warn", format!("{pc}  x{}", vs.len()),
                    format!("kernels outside the dominant {dominant}{time}"), vs.iter().map(|&v| node_target(1, v)).collect()));
            }
        }
        // source ops no compiled node accounts for, even through structure
        let s = &panes[0];
        let mut unaccounted: HashMap<&str, Vec<u32>> = HashMap::new();
        for v in (0..s.graph.nodes.len()).filter(|&v| data_path(s, v) && links[0][v].is_empty()) {
            unaccounted.entry(s.graph.nodes[v].op.as_str()).or_default().push(v as u32);
        }
        let mut un: Vec<_> = unaccounted.into_iter().collect();
        un.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
        for (op, vs) in un {
            out.push(item("unaccounted", "warn", format!("{op}  x{}", vs.len()),
                "no kernel claims these ops and their outputs reach several kernels: unfused, eliminated, or names lost".into(),
                vs.iter().map(|&v| node_target(0, v)).collect()));
        }
        if inferred > 0 {
            out.push(item("unaccounted", "info", format!("{inferred} ops linked by structure"),
                "unclaimed ops whose outputs reach a single kernel; links graded inferred".into(), vec![]));
        }
    }
    for (pi, p) in panes.iter().enumerate() {
        for (gid, gr) in p.hier.groups.iter().enumerate() {
            if gr.outlier && gr.parent != NONE {
                let siblings = p.hier.groups[gr.parent as usize].children.iter()
                    .filter(|&&c| p.hier.groups[c as usize].index.is_some()).count();
                out.push(item("outliers", "info", format!("{}.{}", p.hier.groups[gr.parent as usize].label.rsplit('/').next().unwrap_or(""), gr.label),
                    format!("{} ops, structure differs from its {} siblings", gr.size, siblings - 1), vec![json!([pi, gid, 1])]));
            }
        }
    }
    out
}

/// Nodes whose name, op or kernel contains the query (case-insensitive).
pub fn search(p: &Pane, query: &str, limit: usize) -> Vec<Value> {
    let q = query.to_lowercase();
    p.graph.nodes.iter().enumerate()
        .filter(|(v, n)| !p.hier.folded[*v] && [&n.name, &n.op, &n.kernel].iter().any(|s| s.to_lowercase().contains(&q)))
        .take(limit)
        .map(|(v, n)| json!({ "node": v, "name": n.name, "op": n.op }))
        .collect()
}

fn tensor_json(g: &Graph, t: u32) -> Value {
    if t == NONE {
        return json!(null);
    }
    let x = &g.tensors[t as usize];
    let shape = x.shape.as_ref().map(|s| s.iter().map(|d| if *d < 0 { "?".to_string() } else { d.to_string() }).collect::<Vec<_>>().join("x"));
    let from = (x.producer != NONE).then(|| g.nodes[x.producer as usize].name.clone());
    json!({ "name": x.name, "dtype": x.dtype, "shape": shape, "layout": x.layout, "from": from, "constant": x.constant })
}

/// Everything the Inspector shows for one box.
pub fn inspect(panes: &[Pane], links: &Links, latency_us: Option<f32>, pane: usize, b: u32) -> Value {
    let p = &panes[pane];
    let g = &p.graph;
    let nodes = p.box_nodes(b);
    let kind = match p.box_item(b) {
        Some(Item::Node(_)) => "node",
        Some(Item::Group(_)) => "group",
        Some(Item::Stack(_)) => "stack",
        _ => "other",
    };
    let time: f32 = nodes.iter().filter_map(|&v| g.nodes[v as usize].time_us).sum();
    let mut linked: Vec<(u32, Grade, Via)> = Vec::new();
    if panes.len() > 1 {
        for &v in &nodes {
            linked.extend(links[pane][v as usize].iter().map(|l| (l.other, l.grade, l.via)));
        }
    }
    linked.sort_by_key(|x| (x.0, x.1 as u8));
    linked.dedup_by_key(|x| x.0);
    let other = &panes[(pane + 1) % panes.len()].graph;
    let link_json: Vec<Value> = linked.iter().take(200).map(|&(w, grade, via)| {
        let n = &other.nodes[w as usize];
        json!({ "node": w, "name": n.name, "op": n.op, "grade": format!("{grade:?}").to_lowercase(), "via": via.as_str(), "time_us": n.time_us })
    }).collect();
    let mut out = json!({
        "pane": pane, "kind": kind, "count": nodes.len(),
        "time_us": (time > 0.0).then_some(time),
        "share": latency_us.filter(|_| time > 0.0).map(|l| time / l),
        "links": link_json, "link_count": linked.len(),
    });
    if nodes.len() == 1 {
        let n = &g.nodes[nodes[0] as usize];
        out["node"] = json!({
            "id": nodes[0], "name": n.name, "op": n.op, "kernel": n.kernel, "precision": n.precision, "time_us": n.time_us,
            "attrs": n.attrs.iter().map(|(k, v)| json!([k, if v.len() > 240 { format!("{}...", &v[..v.char_indices().nth(240).map_or(v.len(), |c| c.0)]) } else { v.clone() }])).collect::<Vec<_>>(),
            "inputs": n.inputs.iter().map(|&t| tensor_json(g, t)).collect::<Vec<_>>(),
            "outputs": n.outputs.iter().map(|&t| tensor_json(g, t)).collect::<Vec<_>>(),
            "claims": n.claims,
        });
    }
    out
}
