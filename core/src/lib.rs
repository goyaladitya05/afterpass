//! afterpass core: parsers, unified graph IR, collapsing and layout, compiled to wasm and run in a worker.

pub mod analysis;
pub mod bundle;
pub mod hier;
pub mod ir;
pub mod layout;
pub mod onnx;
pub mod ovir;
pub mod view;

use js_sys::{Float32Array, Function, Object, Reflect, Uint16Array, Uint32Array, Uint8Array};
use wasm_bindgen::prelude::*;

use analysis::{LinkRef, Links, Overlay, Side};
use ir::{Grade, Model, NONE};
use view::Pane;

/// Reads byte ranges through a JS callback `(offset, length) => Uint8Array` (FileReaderSync in a worker).
struct JsSource {
    read: Function,
    len: u64,
}

impl onnx::ByteSource for JsSource {
    fn len(&self) -> u64 {
        self.len
    }
    fn read(&mut self, offset: u64, buf: &mut [u8]) {
        let r = self.read.call2(&JsValue::NULL, &JsValue::from_f64(offset as f64), &JsValue::from_f64(buf.len() as f64));
        if let Ok(a) = r {
            Uint8Array::new(&a).copy_to(buf);
        }
    }
}

fn set(o: &Object, k: &str, v: impl Into<JsValue>) {
    let _ = Reflect::set(o, &JsValue::from_str(k), &v.into());
}

#[wasm_bindgen]
#[derive(Default)]
pub struct Session {
    /// Pane 0 is the source; panes 1 and 2 are compiled graphs of it.
    panes: Vec<Pane>,
    /// Per compiled pane (index pane - 1): links between it and the source.
    links: Vec<Links>,
    sides: Vec<Side>,
}

impl Session {
    pub fn from_model(m: Model) -> Session {
        Session::build(m, false)
    }

    fn build(m: Model, flat: bool) -> Session {
        let mut s = Session::default();
        let ns = m.source.nodes.len();
        s.panes.push(Pane::new(m.source, false, flat));
        if let Some(c) = m.compiled {
            let nc = c.nodes.len();
            s.panes.push(Pane::new(c, true, flat));
            let mut links: Links = [vec![Vec::new(); ns], vec![Vec::new(); nc]];
            for l in m.links {
                if l.source != NONE && (l.source as usize) < ns && (l.compiled as usize) < nc {
                    links[0][l.source as usize].push(LinkRef { other: l.compiled, grade: l.grade, via: l.via });
                    links[1][l.compiled as usize].push(LinkRef { other: l.source, grade: l.grade, via: l.via });
                }
            }
            let inferred = analysis::infer_structure(&s.panes[0], &s.panes[1], &mut links);
            s.links.push(links);
            s.sides.push(Side { label: m.label, runtime: m.runtime, device: m.device, version: m.version, latency_us: m.latency_us, latency_range: m.latency_range, inferred });
        }
        s
    }

    pub fn links(&self) -> &[Links] {
        &self.links
    }

    pub fn findings_json(&self) -> Vec<serde_json::Value> {
        analysis::findings(&self.panes, &self.links, &self.sides)
    }

    pub fn pane(&self, i: usize) -> &Pane {
        &self.panes[i]
    }

    /// Add `other`'s compiled graph as a third pane. Both must be captures of the same source graph.
    pub fn add_compiled(&mut self, mut other: Session) -> Result<(), String> {
        if self.panes.len() != 2 || other.panes.len() != 2 {
            return Err("comparing needs two bundles, each with a compiled graph".into());
        }
        let (a, b) = (&self.panes[0].graph, &other.panes[0].graph);
        let same = a.nodes.len() == b.nodes.len() && a.nodes.iter().zip(&b.nodes).all(|(x, y)| x.op == y.op && x.name == y.name && x.inputs.len() == y.inputs.len());
        if !same {
            return Err(format!("the two bundles were captured from different source graphs ({} and {} nodes)", a.nodes.len(), b.nodes.len()));
        }
        self.panes.push(other.panes.swap_remove(1));
        self.links.append(&mut other.links);
        self.sides.append(&mut other.sides);
        Ok(())
    }
}

#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Session {
        Session::default()
    }

    pub fn load_bundle(json: &[u8]) -> Result<Session, JsError> {
        Ok(Session::from_model(bundle::parse(json).map_err(|e| JsError::new(&e))?))
    }

    pub fn load_ir(xml: &[u8]) -> Result<Session, JsError> {
        let g = ovir::parse(xml).map_err(|e| JsError::new(&e))?;
        Ok(Session::from_model(Model { source: g, ..Default::default() }))
    }

    pub fn load_onnx(read: Function, len: f64) -> Result<Session, JsError> {
        let mut src = JsSource { read, len: len as u64 };
        let g = onnx::parse(&mut src).map_err(|e| JsError::new(&e))?;
        Ok(Session::from_model(Model { source: g, ..Default::default() }))
    }

    /// Rebuild every pane without collapsing (stress testing, graphs without hierarchical names).
    pub fn flatten(&mut self) {
        for p in &mut self.panes {
            let graph = std::mem::take(&mut p.graph);
            let compiled = p.compiled;
            *p = Pane::new(graph, compiled, true);
        }
    }

    pub fn panes(&self) -> usize {
        self.panes.len()
    }

    /// Runtime, version and device of a compiled pane; empty for the source pane.
    pub fn label(&self, pane: usize) -> String {
        pane.checked_sub(1).and_then(|c| self.sides.get(c)).map_or(String::new(), |s| s.label.clone())
    }

    pub fn latency_us(&self, pane: usize) -> f32 {
        pane.checked_sub(1).and_then(|c| self.sides.get(c)).and_then(|s| s.latency_us).unwrap_or(f32::NAN)
    }

    pub fn node_count(&self, pane: usize) -> u32 {
        self.panes[pane].graph.nodes.len() as u32
    }

    /// Render data: boxes [x, y, w, h], kinds, flags, times, labels ("title\tsubtitle" per line), segs [x0, y0, x1, y1].
    pub fn scene(&self, pane: usize) -> Object {
        let sc = &self.panes[pane].scene;
        let o = Object::new();
        set(&o, "boxes", Float32Array::from(&sc.boxes[..]));
        set(&o, "kinds", Uint8Array::from(&sc.kinds[..]));
        set(&o, "flags", Uint8Array::from(&sc.flags[..]));
        set(&o, "times", Float32Array::from(&sc.times[..]));
        set(&o, "segs", Float32Array::from(&sc.segs[..]));
        let labels: Vec<String> = sc.titles.iter().zip(&sc.subtitles).map(|(t, s)| format!("{t}\t{s}")).collect();
        set(&o, "labels", labels.join("\n"));
        set(&o, "width", sc.width);
        set(&o, "height", sc.height);
        o
    }

    pub fn toggle(&mut self, pane: usize, b: u32) -> i32 {
        self.panes[pane].toggle(b)
    }

    pub fn step(&mut self, pane: usize, b: u32, delta: i32) -> i32 {
        self.panes[pane].step(b, delta)
    }

    /// Show another session's source graph as this session's right pane, without provenance.
    pub fn attach(&mut self, mut other: Session) {
        self.panes.truncate(1);
        self.links.clear();
        self.sides.clear();
        if !other.panes.is_empty() {
            self.panes.push(other.panes.swap_remove(0));
        }
    }

    /// Compare: `other` is a capture of the same source by another runtime, device or version.
    pub fn compare(&mut self, other: Session) -> Result<(), JsError> {
        self.add_compiled(other).map_err(|e| JsError::new(&e))
    }

    /// Boxes of pane `to` linked to box `b` of `pane`, with the best grade per box (0 explicit, 1 inferred).
    pub fn linked(&self, pane: usize, b: u32, to: usize) -> Object {
        let o = Object::new();
        if to >= self.panes.len() {
            return o;
        }
        let nodes = self.panes[pane].box_nodes(b);
        let mut best: Vec<(u32, u8)> = Vec::new();
        for (w, grade, _) in analysis::linked_nodes(&self.links, pane, &nodes, to) {
            for ob in self.panes[to].node_boxes([w]) {
                best.push((ob, if grade == Grade::Explicit { 0 } else { 1 }));
            }
        }
        best.sort_unstable();
        best.dedup_by_key(|x| x.0);
        set(&o, "boxes", Uint32Array::from(&best.iter().map(|x| x.0).collect::<Vec<_>>()[..]));
        set(&o, "grades", Uint8Array::from(&best.iter().map(|x| x.1).collect::<Vec<_>>()[..]));
        o
    }

    /// Overlay categories per box: `cats` indexes `legend` (one name per line); 65535 is none.
    pub fn overlay(&self, pane: usize, mode: &str) -> Object {
        let o = Object::new();
        if let Some(m) = Overlay::parse(mode) {
            let side = pane.checked_sub(1).and_then(|c| self.sides.get(c)).cloned().unwrap_or_default();
            let (cats, legend) = analysis::categories(&self.panes[pane], m, &side.runtime, &side.device);
            set(&o, "cats", Uint16Array::from(&cats[..]));
            set(&o, "legend", legend.join("\n"));
        }
        o
    }

    /// Findings as JSON: [{section, severity, title, detail, targets: [[pane, id, is_group]]}].
    pub fn findings(&self) -> String {
        serde_json::to_string(&self.findings_json()).unwrap_or_default()
    }

    pub fn search(&self, pane: usize, query: &str, limit: usize) -> String {
        serde_json::to_string(&analysis::search(&self.panes[pane], query, limit)).unwrap_or_default()
    }

    /// Inspector data for one box, as JSON.
    pub fn inspect(&self, pane: usize, b: u32) -> String {
        analysis::inspect(&self.panes, &self.links, &self.sides, pane, b).to_string()
    }

    /// Expand what is needed to show a node (or, with `group`, a group) and return its box, or -1.
    pub fn reveal(&mut self, pane: usize, id: u32, group: bool) -> i32 {
        let p = &mut self.panes[pane];
        if group { p.reveal_group(id) } else { p.reveal_node(id) }
    }

    /// Details of one box for the status line: node name, op, kernel, precision, output shapes, attributes.
    pub fn describe(&self, pane: usize, b: u32) -> String {
        let p = &self.panes[pane];
        let nodes = p.box_nodes(b);
        if nodes.len() != 1 {
            return format!("{} ops", nodes.len());
        }
        let n = &p.graph.nodes[nodes[0] as usize];
        let shapes: Vec<String> = n.outputs.iter().map(|&t| {
            let t = &p.graph.tensors[t as usize];
            let dims = t.shape.as_ref().map_or("?".into(), |s| s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"));
            format!("{} {}", t.dtype, dims)
        }).collect();
        let mut parts = vec![n.name.clone(), n.op.clone()];
        if !n.kernel.is_empty() {
            parts.push(n.kernel.clone());
        }
        if let Some(t) = n.time_us {
            parts.push(format!("{t:.1} us"));
        }
        parts.push(shapes.join(", "));
        parts.join("  |  ")
    }
}
