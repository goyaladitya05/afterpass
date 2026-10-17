//! afterpass core: parsers, unified graph IR, collapsing and layout, compiled to wasm and run in a worker.

pub mod bundle;
pub mod hier;
pub mod ir;
pub mod layout;
pub mod onnx;
pub mod ovir;
pub mod view;

use js_sys::{Float32Array, Function, Object, Reflect, Uint32Array, Uint8Array};
use wasm_bindgen::prelude::*;

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
    panes: Vec<Pane>,
    /// Per node of pane 0 (source): linked compiled nodes with their grade, and the reverse.
    links: [Vec<Vec<(u32, Grade)>>; 2],
    label: String,
    latency_us: Option<f32>,
}

impl Session {
    pub fn from_model(m: Model) -> Session {
        let mut s = Session { label: m.label, latency_us: m.latency_us, ..Default::default() };
        let ns = m.source.nodes.len();
        s.panes.push(Pane::new(m.source, false));
        if let Some(c) = m.compiled {
            let nc = c.nodes.len();
            s.panes.push(Pane::new(c, true));
            s.links = [vec![Vec::new(); ns], vec![Vec::new(); nc]];
            for l in m.links {
                if l.source != NONE && (l.source as usize) < ns && (l.compiled as usize) < nc {
                    s.links[0][l.source as usize].push((l.compiled, l.grade));
                    s.links[1][l.compiled as usize].push((l.source, l.grade));
                }
            }
        }
        s
    }

    pub fn pane(&self, i: usize) -> &Pane {
        &self.panes[i]
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

    pub fn panes(&self) -> usize {
        self.panes.len()
    }

    pub fn label(&self) -> String {
        self.label.clone()
    }

    pub fn latency_us(&self) -> f32 {
        self.latency_us.unwrap_or(f32::NAN)
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

    pub fn toggle(&mut self, pane: usize, b: u32) -> bool {
        self.panes[pane].toggle(b)
    }

    pub fn step(&mut self, pane: usize, b: u32, delta: i32) -> bool {
        self.panes[pane].step(b, delta)
    }

    /// Boxes linked to box `b` of `pane` in the other pane, with the best grade per box (0 explicit, 1 inferred).
    pub fn linked(&self, pane: usize, b: u32) -> Object {
        let o = Object::new();
        if self.panes.len() < 2 {
            return o;
        }
        let other = 1 - pane;
        let mut best: Vec<(u32, u8)> = Vec::new();
        for v in self.panes[pane].box_nodes(b) {
            for &(w, g) in &self.links[pane][v as usize] {
                for ob in self.panes[other].node_boxes([w]) {
                    best.push((ob, if g == Grade::Explicit { 0 } else { 1 }));
                }
            }
        }
        best.sort_unstable();
        best.dedup_by_key(|x| x.0);
        set(&o, "boxes", Uint32Array::from(&best.iter().map(|x| x.0).collect::<Vec<_>>()[..]));
        set(&o, "grades", Uint8Array::from(&best.iter().map(|x| x.1).collect::<Vec<_>>()[..]));
        o
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
