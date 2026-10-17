//! Analysis over a loaded session: structural provenance, overlay categories, Findings, search, inspection.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::hier::{majority_scope, mask_digits, scope_tokens};
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
    let add = |links: &mut Links, v: usize, k: u32, linked_src: &mut Vec<bool>| {
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
    let us = us + 0.0; // an empty float sum is -0.0
    if us >= 1000.0 { format!("{:.2} ms", us / 1000.0) } else { format!("{us:.0} us") }
}

fn item(section: &str, severity: &str, title: String, detail: String, targets: Vec<Value>) -> Value {
    json!({ "section": section, "severity": severity, "title": title, "detail": detail, "targets": targets })
}

fn node_target(pane: usize, v: u32) -> Value {
    json!([pane, v, 0])
}

/// One compiled graph of a session: what the runtime was and how it was timed.
#[derive(Clone, Default, Debug)]
pub struct Side {
    pub label: String,
    pub runtime: String,
    pub device: String,
    pub version: String,
    pub latency_us: Option<f32>,
    pub latency_range: Option<(f32, f32)>,
    /// Source ops linked by structure rather than by name.
    pub inferred: usize,
}

/// `op x3, Mul, Softmax x2`: counts of a list of names, most common first.
fn counted<'a>(names: impl Iterator<Item = &'a str>, limit: usize) -> String {
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for n in names {
        match counts.iter_mut().find(|(m, _)| *m == n) {
            Some((_, c)) => *c += 1,
            None => counts.push((n, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let mut parts: Vec<String> = counts.iter().take(limit).map(|(n, c)| if *c > 1 { format!("{n} x{c}") } else { n.to_string() }).collect();
    if counts.len() > limit {
        parts.push(format!("+{} more", counts.len() - limit));
    }
    parts.join(", ")
}

/// The Findings list. For each compiled graph: top time sinks and unattributed time, unaccounted source ops,
/// kernels the runtime inserted, precision fallbacks. With two compiled graphs, the comparison comes first.
/// Then outlier blocks of every pane. Each item names the nodes it jumps to.
pub fn findings(panes: &[Pane], links: &[Links], sides: &[Side]) -> Vec<Value> {
    let mut out = Vec::new();
    if panes.len() == 3 {
        out.extend(compare(panes, links, sides));
    }
    for c in 1..panes.len() {
        let tag = if panes.len() > 2 { ["A  ", "B  "][c - 1] } else { "" };
        out.extend(side_findings(&panes[0], &panes[c], c, &links[c - 1], &sides[c - 1], tag));
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

fn side_findings(s: &Pane, c: &Pane, pane: usize, links: &Links, side: &Side, tag: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let g = &c.graph;
    let kernels: Vec<u32> = (0..g.nodes.len()).filter(|&v| data_path(c, v)).map(|v| v as u32).collect();
    // time sinks; kernels repeated per layer (same op and kernel, names equal up to digits) are summed
    let mut sinks: HashMap<(&str, &str, String), (f32, Vec<u32>)> = HashMap::new();
    let mut total = 0.0;
    for &v in &kernels {
        let n = &g.nodes[v as usize];
        if let Some(t) = n.time_us {
            total += t;
            let e = sinks.entry((n.op.as_str(), n.kernel.as_str(), mask_digits(&n.name))).or_default();
            e.0 += t;
            e.1.push(v);
        }
    }
    let mut sinks: Vec<_> = sinks.into_iter().collect();
    sinks.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap().then(a.0.cmp(&b.0)));
    let base = side.latency_us.filter(|&l| l > 0.0).unwrap_or(total);
    for ((op, _, _), (t, mut vs)) in sinks.into_iter().take(10) {
        vs.sort_by(|&x, &y| g.nodes[y as usize].time_us.partial_cmp(&g.nodes[x as usize].time_us).unwrap());
        let first = &g.nodes[vs[0] as usize];
        let (title, detail) = if vs.len() > 1 {
            (format!("{tag}{op} x{}  {}", vs.len(), fmt_us(t)), format!("{:.1}% of latency, {} to {} each  {}", 100.0 * t / base,
                fmt_us(g.nodes[*vs.last().unwrap() as usize].time_us.unwrap_or(0.0)), fmt_us(first.time_us.unwrap_or(0.0)), first.name))
        } else {
            (format!("{tag}{op}  {}", fmt_us(t)), format!("{:.1}% of latency  {}", 100.0 * t / base, first.name))
        };
        out.push(item("time", "info", title, detail, vs.iter().map(|&v| node_target(pane, v)).collect()));
    }
    if let Some(l) = side.latency_us.filter(|&l| l > 0.0 && total > 0.0 && total < l * 0.9) {
        out.push(item("time", "warn", format!("{tag}unattributed  {}", fmt_us(l - total)),
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
    ins.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap().then(b.1.0.len().cmp(&a.1.0.len())).then(a.0.cmp(b.0)));
    for (op, (vs, t)) in ins {
        let time = if t > 0.0 { format!(", {}", fmt_us(t)) } else { String::new() };
        out.push(item("inserted", "info", format!("{tag}{op}  x{}", vs.len()),
            format!("no source op matches the names they claim: inserted by the runtime or renamed{time}"), vs.iter().map(|&v| node_target(pane, v)).collect()));
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
        other.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap().then(a.0.cmp(b.0)));
        for (pc, (vs, t)) in other {
            let time = if t > 0.0 { format!(", {}", fmt_us(t)) } else { String::new() };
            out.push(item("precision", "warn", format!("{tag}{pc}  x{}", vs.len()),
                format!("kernels outside the dominant {dominant}{time}"), vs.iter().map(|&v| node_target(pane, v)).collect()));
        }
    }
    // source ops no compiled node accounts for, even through structure
    let mut unaccounted: HashMap<&str, Vec<u32>> = HashMap::new();
    for v in (0..s.graph.nodes.len()).filter(|&v| data_path(s, v) && links[0][v].is_empty()) {
        unaccounted.entry(s.graph.nodes[v].op.as_str()).or_default().push(v as u32);
    }
    let mut un: Vec<_> = unaccounted.into_iter().collect();
    un.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
    for (op, vs) in un {
        out.push(item("unaccounted", "warn", format!("{tag}{op}  x{}", vs.len()),
            "no kernel claims these ops and their outputs reach several kernels: unfused, eliminated, or names lost".into(),
            vs.iter().map(|&v| node_target(0, v)).collect()));
    }
    if side.inferred > 0 {
        out.push(item("unaccounted", "info", format!("{tag}{} ops linked by structure", side.inferred),
            "unclaimed ops whose outputs reach a single kernel; links graded inferred".into(), vec![]));
    }
    out
}

/// A scope with layer indices masked: [model, layers, 3, mlp] gives `model.layers.#.mlp`.
fn mask_index(scope: &[String]) -> String {
    scope.iter().map(|t| if t.bytes().all(|b| b.is_ascii_digit()) { "#" } else { t.as_str() }).collect::<Vec<_>>().join(".")
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Source ops and the kernels of A and B that compute them: one connected component of the links.
#[derive(Default)]
struct Region {
    src: Vec<u32>,
    k: [Vec<u32>; 2],
    inferred: bool,
}

/// Regions with the same scope (up to layer numbers), source ops and kernel op types: one row per repeated layer.
struct Agg {
    label: String,
    regions: Vec<usize>,
    t: [f32; 2],
}

/// Compare two compiled graphs of one source (panes 1 and 2) through it. The links partition source ops and
/// kernels into regions: connected components where every kernel of A and B in a region computes only source
/// ops of that region, so the region's time on each side is well defined even when the two runtimes fuse
/// differently. Repeated layers are aggregated. Sections: summary, where the time differs, ops fused
/// differently, precision differences, and (same runtime) kernel implementation changes.
pub fn compare(panes: &[Pane], links: &[Links], sides: &[Side]) -> Vec<Value> {
    let s = &panes[0];
    let c = [&panes[1], &panes[2]];
    let ns = s.graph.nodes.len();
    let off = [ns, ns + c[0].graph.nodes.len()];
    let mut parent: Vec<u32> = (0..(off[1] + c[1].graph.nodes.len()) as u32).collect();
    for v in (0..ns).filter(|&v| data_path(s, v)) {
        for side in 0..2 {
            for l in &links[side][0][v] {
                if data_path(c[side], l.other as usize) {
                    let (x, y) = (find(&mut parent, v as u32), find(&mut parent, (off[side] + l.other as usize) as u32));
                    parent[x as usize] = y;
                }
            }
        }
    }
    let mut index: HashMap<u32, usize> = HashMap::new();
    let mut regions: Vec<Region> = Vec::new();
    let mut orphan: [(usize, f32); 2] = [(0, 0.0); 2];
    for v in (0..ns).filter(|&v| data_path(s, v)) {
        let r = find(&mut parent, v as u32);
        let i = *index.entry(r).or_insert_with(|| { regions.push(Region::default()); regions.len() - 1 });
        regions[i].src.push(v as u32);
        regions[i].inferred |= (0..2).any(|side| links[side][0][v].iter().any(|l| l.grade != Grade::Explicit));
    }
    for side in 0..2 {
        for k in (0..c[side].graph.nodes.len()).filter(|&k| data_path(c[side], k)) {
            let r = find(&mut parent, (off[side] + k) as u32);
            match index.get(&r) {
                Some(&i) => regions[i].k[side].push(k as u32),
                None => {
                    orphan[side].0 += 1;
                    orphan[side].1 += c[side].graph.nodes[k].time_us.unwrap_or(0.0);
                }
            }
        }
    }
    let time = |side: usize, ks: &[u32]| -> f32 { ks.iter().filter_map(|&k| c[side].graph.nodes[k as usize].time_us).sum() };
    let ops = |side: usize, ks: &[u32]| -> Vec<&str> {
        let mut v: Vec<&str> = ks.iter().map(|&k| c[side].graph.nodes[k as usize].op.as_str()).collect();
        v.sort_unstable();
        v
    };

    // aggregate repeated regions
    let mut aggs: Vec<Agg> = Vec::new();
    let mut by_key: HashMap<(String, Vec<&str>, [Vec<&str>; 2]), usize> = HashMap::new();
    for (i, r) in regions.iter().enumerate() {
        if r.k[0].is_empty() && r.k[1].is_empty() {
            continue;
        }
        let scope = majority_scope(r.src.iter().map(|&v| scope_tokens(&s.graph.nodes[v as usize].name)).collect());
        let mut label = scope.join(".");
        if label.is_empty() {
            label = counted(r.src.iter().map(|&v| s.graph.nodes[v as usize].op.as_str()), 2);
        }
        let masked = mask_index(&scope);
        let mut src_ops: Vec<&str> = r.src.iter().map(|&v| s.graph.nodes[v as usize].op.as_str()).collect();
        src_ops.sort_unstable();
        let key = (masked.clone(), src_ops, [ops(0, &r.k[0]), ops(1, &r.k[1])]);
        let t = [time(0, &r.k[0]), time(1, &r.k[1])];
        match by_key.get(&key) {
            Some(&a) => {
                let a = &mut aggs[a];
                a.regions.push(i);
                a.t[0] += t[0];
                a.t[1] += t[1];
                if !key.0.is_empty() {
                    a.label = key.0;
                }
            }
            None => {
                by_key.insert(key, aggs.len());
                aggs.push(Agg { label, regions: vec![i], t });
            }
        }
    }

    let mut out = Vec::new();
    let name = ["A", "B"];
    // summary
    let lat = [sides[0].latency_us, sides[1].latency_us];
    if let (Some(la), Some(lb)) = (lat[0], lat[1]) {
        let (slow, fast, ratio) = if la >= lb { ("A", "B", la / lb) } else { ("B", "A", lb / la) };
        let verdict = if ratio < 1.02 { "Within 2% of each other".to_string() } else { format!("{slow} takes {ratio:.2}x as long as {fast}") };
        let range = |side: usize| sides[side].latency_range.map_or(String::new(), |(lo, hi)| format!(", runs {} to {}", fmt_us(lo), fmt_us(hi)));
        out.push(item("summary", "info", format!("latency  A {}, B {}", fmt_us(la), fmt_us(lb)),
            format!("{verdict} (medians). A: {}{}. B: {}{}", sides[0].label, range(0), sides[1].label, range(1)), vec![]));
    }
    let kernels = [0, 1].map(|side| (0..c[side].graph.nodes.len()).filter(|&k| data_path(c[side], k)).count());
    let profiled = [0, 1].map(|side| (0..c[side].graph.nodes.len()).filter(|&k| data_path(c[side], k)).filter_map(|k| c[side].graph.nodes[k].time_us).sum::<f32>());
    let coverage: Vec<String> = (0..2).filter(|&side| profiled[side] > 0.0).map(|side| match lat[side] {
        Some(l) if profiled[side] > l * 1.05 => format!("{} kernel times sum to {:.0}% of latency (the profiler serialises kernels)", name[side], 100.0 * profiled[side] / l),
        Some(l) => format!("{} kernel times cover {:.0}% of latency", name[side], 100.0 * profiled[side] / l),
        None => format!("{} kernel times sum to {}", name[side], fmt_us(profiled[side])),
    }).collect();
    out.push(item("summary", "info", format!("kernels  A {}, B {}", kernels[0], kernels[1]),
        format!("{} regions of source ops compared ({} distinct after folding repeated layers). {}", aggs.iter().map(|a| a.regions.len()).sum::<usize>(), aggs.len(), coverage.join("; ")), vec![]));
    if orphan[0].0 + orphan[1].0 > 0 {
        out.push(item("summary", "info", format!("unlinked kernels  A {}, B {}", orphan[0].0, orphan[1].0),
            format!("kernels with no source op are left out of the comparison: A {}, B {}", fmt_us(orphan[0].1), fmt_us(orphan[1].1)), vec![]));
    }

    // targets: per region, the slowest kernel of each side (a source op when a side has none)
    let targets = |a: &Agg| -> Vec<Value> {
        let mut t = Vec::new();
        for &i in a.regions.iter().take(64) {
            let r = &regions[i];
            for side in 0..2 {
                let best = r.k[side].iter().max_by(|&&x, &&y| c[side].graph.nodes[x as usize].time_us.unwrap_or(0.0)
                    .partial_cmp(&c[side].graph.nodes[y as usize].time_us.unwrap_or(0.0)).unwrap());
                match best {
                    Some(&k) => t.push(node_target(side + 1, k)),
                    None if side == 0 || !r.k[0].is_empty() => t.push(node_target(0, r.src[0])),
                    None => {}
                }
            }
        }
        t
    };
    let times_known = profiled[0] > 0.0 && profiled[1] > 0.0;
    let x = |a: &Agg| if a.regions.len() > 1 { format!(" x{}", a.regions.len()) } else { String::new() };
    let src_ops = |a: &Agg| counted(regions[a.regions[0]].src.iter().map(|&v| s.graph.nodes[v as usize].op.as_str()), 4);
    let hedge = |a: &Agg| if a.regions.iter().any(|&i| regions[i].inferred) { "; links partly inferred" } else { "" };
    let kc = |a: &Agg, side: usize| regions[a.regions[0]].k[side].len();
    let plural = |n: usize| if n == 1 { "kernel" } else { "kernels" };

    // where the time differs
    if times_known {
        let total = lat[0].unwrap_or(profiled[0]).max(lat[1].unwrap_or(profiled[1]));
        let mut order: Vec<&Agg> = aggs.iter().filter(|a| (a.t[0] - a.t[1]).abs() > 0.0).collect();
        order.sort_by(|a, b| (b.t[0] - b.t[1]).abs().partial_cmp(&(a.t[0] - a.t[1]).abs()).unwrap());
        for a in order.into_iter().take(12) {
            let d = a.t[0] - a.t[1];
            let slow = if d > 0.0 { "A" } else { "B" };
            out.push(item("delta", if d.abs() > 0.05 * total { "warn" } else { "info" }, format!("{}{}  {slow} +{}", a.label, x(a), fmt_us(d.abs())),
                format!("A {} in {} {}, B {} in {} {}{}; {}{}", fmt_us(a.t[0]), kc(a, 0), plural(kc(a, 0)), fmt_us(a.t[1]), kc(a, 1), plural(kc(a, 1)),
                    if a.regions.len() > 1 { " each" } else { "" }, src_ops(a), hedge(a)), targets(a)));
        }
    }
    // fused differently
    let mut fused: Vec<&Agg> = aggs.iter().filter(|a| kc(a, 0) != kc(a, 1)).collect();
    fused.sort_by(|a, b| (b.regions.len() * kc(b, 0).abs_diff(kc(b, 1))).cmp(&(a.regions.len() * kc(a, 0).abs_diff(kc(a, 1)))).then(a.label.cmp(&b.label)));
    for a in fused.into_iter().take(12) {
        let r = &regions[a.regions[0]];
        let kinds = |side: usize| if r.k[side].is_empty() { "none".to_string() } else { counted(r.k[side].iter().map(|&k| c[side].graph.nodes[k as usize].op.as_str()), 3) };
        out.push(item("fusion", "info", format!("{}{}  A {}, B {} {}", a.label, x(a), kc(a, 0), kc(a, 1), plural(kc(a, 1))),
            format!("A: {}. B: {}. Source: {}{}", kinds(0), kinds(1), src_ops(a), hedge(a)), targets(a)));
    }
    // precision and kernel implementation differences
    let classes = |side: usize, r: &Region| -> Vec<&'static str> {
        let mut v: Vec<&str> = r.k[side].iter().filter_map(|&k| node_precision(&c[side].graph, &c[side].graph.nodes[k as usize])).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let families = |side: usize, r: &Region| -> Vec<String> {
        let mut v: Vec<String> = r.k[side].iter().map(|&k| kernel_family(&sides[side].runtime, &c[side].graph.nodes[k as usize])).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let weight = |a: &Agg| a.t[0] + a.t[1] + a.regions.len() as f32 * 1e-3;
    let same_target = sides[0].runtime == sides[1].runtime && sides[0].device == sides[1].device;
    let mut prec: Vec<(&Agg, String, String)> = Vec::new();
    let mut impls: Vec<(&Agg, String, String)> = Vec::new();
    for a in &aggs {
        let r = &regions[a.regions[0]];
        if r.k[0].is_empty() || r.k[1].is_empty() || (times_known && a.t[0] + a.t[1] <= 0.0) {
            continue;
        }
        let (pa, pb) = (classes(0, r), classes(1, r));
        if pa != pb && !pa.is_empty() && !pb.is_empty() {
            prec.push((a, pa.join("+"), pb.join("+")));
        } else if same_target && kc(a, 0) == kc(a, 1) {
            let (fa, fb) = (families(0, r), families(1, r));
            if fa != fb {
                impls.push((a, fa.join(", "), fb.join(", ")));
            }
        }
    }
    // a precision change that holds almost everywhere (fp32 on CPU, fp16 on GPU) is one finding, not one per region
    let mut patterns: HashMap<(String, String), (usize, f32)> = HashMap::new();
    for (a, pa, pb) in &prec {
        let e = patterns.entry((pa.clone(), pb.clone())).or_default();
        e.0 += a.regions.len();
        e.1 += weight(a);
    }
    let compared: usize = aggs.iter().filter(|a| !regions[a.regions[0]].k[0].is_empty() && !regions[a.regions[0]].k[1].is_empty()).map(|a| a.regions.len()).sum();
    if let Some(((pa, pb), (n, _))) = patterns.iter().max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap()).filter(|p| p.1.0 * 2 > compared) {
        let (pa, pb, n) = (pa.clone(), pb.clone(), *n);
        let mut top: Vec<&(&Agg, String, String)> = prec.iter().filter(|x| x.1 == pa && x.2 == pb).collect();
        top.sort_by(|a, b| weight(b.0).partial_cmp(&weight(a.0)).unwrap());
        let targets: Vec<Value> = top.iter().flat_map(|x| targets(x.0)).take(128).collect();
        out.push(item("precision-diff", "info", format!("A {pa}, B {pb}  in {n} of {compared} regions"),
            "the two runtimes compute at different precisions throughout; regions that deviate are listed below".into(), targets));
        prec.retain(|x| !(x.1 == pa && x.2 == pb));
    }
    if same_target && prec.is_empty() && impls.is_empty() && aggs.iter().all(|a| kc(a, 0) == kc(a, 1)) {
        out.push(item("summary", "info", "same kernels".into(),
            "every region of source ops runs the same number of kernels, of the same families and precisions, on both sides".into(), vec![]));
    }
    for (list, section) in [(&mut prec, "precision-diff"), (&mut impls, "kernel-diff")] {
        list.sort_by(|a, b| weight(b.0).partial_cmp(&weight(a.0)).unwrap());
        for (a, pa, pb) in list.iter().take(12) {
            let time = if times_known { format!("A {}, B {}; ", fmt_us(a.t[0]), fmt_us(a.t[1])) } else { String::new() };
            let title = if section == "precision-diff" { format!("{}{}  A {pa}, B {pb}", a.label, x(a)) } else { format!("{}{}", a.label, x(a)) };
            let detail = if section == "precision-diff" { format!("{time}{}{}", src_ops(a), hedge(a)) } else { format!("A: {pa}. B: {pb}. {time}{}{}", src_ops(a), hedge(a)) };
            out.push(item(section, "info", title, detail, targets(a)));
        }
    }
    out
}

/// Nodes of pane `to` linked to `nodes` of pane `from`, each with its strongest grade and how it was linked.
/// Pane 0 is the source; two compiled panes are linked through the source ops they share, graded by the
/// weaker of the two links.
pub fn linked_nodes(links: &[Links], from: usize, nodes: &[u32], to: usize) -> Vec<(u32, Grade, Via)> {
    let mut out: Vec<(u32, Grade, Via)> = Vec::new();
    if from == to || from.max(to) > links.len() {
        return out;
    }
    if from == 0 || to == 0 {
        let (c, dir) = if from == 0 { (to, 0) } else { (from, 1) };
        for &v in nodes {
            out.extend(links[c - 1][dir][v as usize].iter().map(|l| (l.other, l.grade, l.via)));
        }
    } else {
        for (v, g1, via1) in linked_nodes(links, from, nodes, 0) {
            out.extend(links[to - 1][0][v as usize].iter().map(|l| {
                let via = if via1 == Via::Structure { via1 } else { l.via };
                (l.other, g1.max(l.grade), via)
            }));
        }
    }
    out.sort_by_key(|x| (x.0, x.1));
    out.dedup_by_key(|x| x.0);
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

/// Everything the Inspector shows for one box: the node, and per other pane the nodes linked to it.
pub fn inspect(panes: &[Pane], links: &[Links], sides: &[Side], pane: usize, b: u32) -> Value {
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
    let latency = if pane > 0 { sides[pane - 1].latency_us } else { None };
    let groups: Vec<Value> = (0..panes.len()).filter(|&q| q != pane).map(|q| {
        let linked = linked_nodes(links, pane, &nodes, q);
        let other = &panes[q].graph;
        let items: Vec<Value> = linked.iter().take(200).map(|&(w, grade, via)| {
            let n = &other.nodes[w as usize];
            json!({ "node": w, "name": n.name, "op": n.op, "grade": format!("{grade:?}").to_lowercase(), "via": via.as_str(), "time_us": n.time_us })
        }).collect();
        json!({ "pane": q, "links": items, "count": linked.len() })
    }).collect();
    let mut out = json!({
        "pane": pane, "kind": kind, "count": nodes.len(),
        "time_us": (time > 0.0).then_some(time),
        "share": latency.filter(|_| time > 0.0).map(|l| time / l),
        "groups": groups,
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
