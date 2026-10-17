//! Unified graph IR shared by every parser: nodes refer to tensors by index; provenance is a separate link set.

use std::collections::HashMap;

pub const NONE: u32 = u32::MAX;

#[derive(Default, Clone, Debug)]
pub struct Node {
    pub name: String,
    pub op: String,
    /// Tensor ids in port order; `NONE` is an omitted optional input.
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub attrs: Vec<(String, String)>,
    /// Compiled graphs only: kernel or primitive name, runtime precision, claimed source names, median time.
    pub kernel: String,
    pub precision: String,
    pub claims: Vec<String>,
    pub time_us: Option<f32>,
}

#[derive(Default, Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    /// `None` when the rank is unknown; `-1` marks an unknown or symbolic dim.
    pub shape: Option<Vec<i64>>,
    pub layout: String,
    pub producer: u32,
    /// Weight or constant data (ONNX initializer); such tensors have no producer node.
    pub constant: bool,
}

#[derive(Default, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub tensors: Vec<Tensor>,
    index: HashMap<String, u32>,
}

impl Graph {
    /// Intern a tensor by name, creating it on first use.
    pub fn tensor(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.index.get(name) {
            return id;
        }
        let id = self.tensors.len() as u32;
        self.tensors.push(Tensor { name: name.to_string(), producer: NONE, ..Default::default() });
        self.index.insert(name.to_string(), id);
        id
    }

    pub fn tensor_id(&self, name: &str) -> Option<u32> {
        self.index.get(name).copied()
    }

    /// Set tensor producers once all nodes are in, and drop the name index.
    pub fn finish(&mut self) {
        for (i, n) in self.nodes.iter().enumerate() {
            for &t in &n.outputs {
                self.tensors[t as usize].producer = i as u32;
            }
        }
        self.index = HashMap::new();
    }

    /// Predecessor and successor node lists (deduplicated, in port order).
    pub fn adjacency(&self) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut preds = vec![Vec::new(); self.nodes.len()];
        let mut succs = vec![Vec::new(); self.nodes.len()];
        for (v, n) in self.nodes.iter().enumerate() {
            for &t in &n.inputs {
                if t == NONE {
                    continue;
                }
                let u = self.tensors[t as usize].producer;
                if u != NONE && !preds[v].contains(&u) {
                    preds[v].push(u);
                    succs[u as usize].push(v as u32);
                }
            }
        }
        (preds, succs)
    }

    /// Nodes that only compute constants: constant ops, and ops whose every input is constant.
    /// Nodes are visited in index order, which is topological for every supported format.
    pub fn constant_nodes(&self) -> Vec<bool> {
        let mut c = vec![false; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            c[i] = matches!(n.op.as_str(), "Const" | "Constant")
                || (!n.inputs.is_empty()
                    && !matches!(n.op.as_str(), "Parameter" | "Input")
                    && n.inputs.iter().all(|&t| {
                        t == NONE || {
                            let t = &self.tensors[t as usize];
                            t.constant || (t.producer != NONE && c[t.producer as usize])
                        }
                    }));
        }
        c
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grade {
    Explicit,
    Inferred,
    Unknown,
}

/// How a link was established: matched on a claimed name, a tensor name, a name prefix, an output port, or
/// inferred from graph structure (an unclaimed op whose outputs reach a single kernel).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Via {
    Name,
    Tensor,
    Prefix,
    Port,
    Structure,
    Unknown,
}

impl Via {
    pub fn parse(s: &str) -> Via {
        match s {
            "name" => Via::Name,
            "tensor" => Via::Tensor,
            "prefix" => Via::Prefix,
            "port" => Via::Port,
            _ => Via::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Via::Name => "name",
            Via::Tensor => "tensor",
            Via::Prefix => "prefix",
            Via::Port => "port",
            Via::Structure => "structure",
            Via::Unknown => "",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Link {
    pub compiled: u32,
    pub source: u32,
    pub grade: Grade,
    pub via: Via,
}

/// What one load produced: a source graph, optionally a compiled graph with provenance.
#[derive(Default, Debug)]
pub struct Model {
    pub source: Graph,
    pub compiled: Option<Graph>,
    pub links: Vec<Link>,
    /// Short description for the UI, for example "openvino 2026.4.1 GPU (Arc 140V)".
    pub label: String,
    pub runtime: String,
    pub device: String,
    /// Median whole-inference latency, when profiled.
    pub latency_us: Option<f32>,
}
