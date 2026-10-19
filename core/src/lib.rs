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
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), String> {
        let r = self.read.call2(&JsValue::NULL, &JsValue::from_f64(offset as f64), &JsValue::from_f64(buf.len() as f64));
        let a = Uint8Array::new(&r.map_err(|e| format!("reading the file failed: {e:?}"))?);
        if a.length() as usize != buf.len() {
            return Err(format!("short read at byte {offset}"));
        }
        a.copy_to(buf);
        Ok(())
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
        let mut s = Session::default();
        let ns = m.source.nodes.len();
        s.panes.push(Pane::new(m.source, false, false));
        if let Some(c) = m.compiled {
            let nc = c.nodes.len();
            s.panes.push(Pane::new(c, true, false));
            let mut links: Links = [vec![Vec::new(); ns], vec![Vec::new(); nc]];
            for l in m.links {
                if l.source != NONE && (l.source as usize) < ns && (l.compiled as usize) < nc {
                    links[0][l.source as usize].push(LinkRef { other: l.compiled, grade: l.grade, via: l.via });
                    links[1][l.compiled as usize].push(LinkRef { other: l.source, grade: l.grade, via: l.via });
                }
            }
            let inferred = analysis::infer_structure(&s.panes[0], &s.panes[1], &mut links);
            s.links.push(links);
            s.sides.push(Side { label: m.label, runtime: m.runtime, device: m.device, version: m.version, device_name: m.device_name,
                captured: m.captured, model_file: m.model_file, latency_us: m.latency_us, latency_range: m.latency_range, inferred });
        }
        s
    }

    /// The capture behind a compiled pane; None for the source pane.
    fn side(&self, pane: usize) -> Option<&Side> {
        pane.checked_sub(1).and_then(|c| self.sides.get(c))
    }

    /// Requests can outlive the session they were made for (a click on the old Findings during a load).
    fn has_box(&self, pane: usize, b: u32) -> bool {
        self.panes.get(pane).is_some_and(|p| (b as usize) < p.scene.kinds.len())
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
        // same ops, names and edges (input tensor names), node by node
        fn tensor(g: &ir::Graph, t: u32) -> Option<&str> {
            g.tensors.get(t as usize).map(|t| t.name.as_str())
        }
        let same = a.nodes.len() == b.nodes.len() && a.nodes.iter().zip(&b.nodes).all(|(x, y)| x.op == y.op && x.name == y.name
            && x.inputs.len() == y.inputs.len() && x.inputs.iter().zip(&y.inputs).all(|(&i, &j)| tensor(a, i) == tensor(b, j)));
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
        self.side(pane).map_or(String::new(), |s| s.label.clone())
    }

    /// Who compiled a compiled pane, for its header: JSON with runtime, version, device, device_name, captured and
    /// model; `{}` for a source pane.
    pub fn about(&self, pane: usize) -> String {
        self.side(pane).map_or("{}".into(), |s| serde_json::json!({
            "runtime": s.runtime, "version": s.version, "device": s.device, "device_name": s.device_name,
            "captured": s.captured, "model": s.model_file,
        }).to_string())
    }

    pub fn latency_us(&self, pane: usize) -> f32 {
        self.side(pane).and_then(|s| s.latency_us).unwrap_or(f32::NAN)
    }

    pub fn node_count(&self, pane: usize) -> u32 {
        self.panes.get(pane).map_or(0, |p| p.graph.nodes.len() as u32)
    }

    /// Render data: boxes [x, y, w, h], kinds, flags, times, labels ("title\tsubtitle" per line), segs [x0, y0, x1, y1].
    pub fn scene(&self, pane: usize) -> Object {
        let o = Object::new();
        let Some(p) = self.panes.get(pane) else { return o };
        let sc = &p.scene;
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
        if !self.has_box(pane, b) {
            return -1;
        }
        self.panes[pane].toggle(b)
    }

    pub fn step(&mut self, pane: usize, b: u32, delta: i32) -> i32 {
        if !self.has_box(pane, b) {
            return -1;
        }
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
        if to >= self.panes.len() || !self.has_box(pane, b) {
            return o;
        }
        let nodes = self.panes[pane].box_nodes(b);
        let mut best: Vec<(u32, u8)> = Vec::new();
        for (w, grade, _) in analysis::linked_nodes(&self.links, pane, &nodes, to) {
            if let Some(ob) = self.panes[to].node_box(w) {
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
        if let Some(m) = Overlay::parse(mode).filter(|_| pane < self.panes.len()) {
            let side = self.side(pane).cloned().unwrap_or_default();
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
        let Some(p) = self.panes.get(pane) else { return "[]".into() };
        serde_json::to_string(&analysis::search(p, query, limit)).unwrap_or_default()
    }

    /// Inspector data for one box, as JSON.
    pub fn inspect(&self, pane: usize, b: u32) -> String {
        if !self.has_box(pane, b) {
            return "null".into();
        }
        analysis::inspect(&self.panes, &self.links, &self.sides, pane, b).to_string()
    }

    /// Expand what is needed to show a node (or, with `group`, a group) and return its box, or -1.
    pub fn reveal(&mut self, pane: usize, id: u32, group: bool) -> i32 {
        let Some(p) = self.panes.get_mut(pane) else { return -1 };
        let n = if group { p.hier.groups.len() } else { p.graph.nodes.len() };
        if id as usize >= n {
            return -1;
        }
        if group { p.reveal_group(id) } else { p.reveal_node(id) }
    }

    /// Details of one box for the status line: node name, op, kernel, precision, output shapes, attributes.
    pub fn describe(&self, pane: usize, b: u32) -> String {
        if !self.has_box(pane, b) {
            return String::new();
        }
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
