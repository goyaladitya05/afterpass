//! OpenVINO IR (.xml) topology parser. The .bin weights file is never needed.
//! Nested Loop / TensorIterator bodies are not expanded in v0: their layer count is kept as an attribute.

use std::collections::HashMap;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

use crate::ir::{Graph, Node, NONE};

fn value(a: &quick_xml::events::attributes::Attribute) -> String {
    a.normalized_value(quick_xml::XmlVersion::Implicit1_0).map(|v| v.into_owned()).unwrap_or_default()
}

fn attr(e: &BytesStart, key: &str) -> Option<String> {
    e.attributes().flatten().find(|a| a.key.as_ref() == key).map(|a| value(&a))
}

#[derive(Default)]
struct Port {
    id: u32,
    precision: String,
    names: String,
    dims: Vec<i64>,
}

struct Layer {
    xml_id: u32,
    node: Node,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

pub fn parse(xml: &[u8]) -> Result<Graph, String> {
    let mut r = Reader::from_reader(xml);
    r.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut path: Vec<String> = Vec::new();
    let mut layers: Vec<Layer> = Vec::new();
    let mut edges: Vec<(u32, u32, u32, u32)> = Vec::new();
    let mut port: Option<(bool, Port)> = None;
    let mut in_dim = false;
    let mut nested = 0usize;

    loop {
        let ev = r.read_event_into(&mut buf).map_err(|e| format!("IR XML error at {}: {e}", r.buffer_position()))?;
        let empty = matches!(ev, Event::Empty(_));
        match ev {
            Event::Start(ref e) | Event::Empty(ref e) => {
                let name = e.name().as_ref().to_string();
                // depth of the element being opened: net=0, layers=1, layer=2, data/input/output=3, port=4, dim=5
                let depth = path.len();
                let in_body = path.iter().any(|p| p == "body");
                match (depth, name.as_str()) {
                    _ if in_body => {
                        if name == "layer" {
                            nested += 1;
                        }
                    }
                    (2, "layer") if path[1] == "layers" => {
                        nested = 0;
                        layers.push(Layer {
                            xml_id: attr(e, "id").and_then(|v| v.parse().ok()).ok_or("layer without id")?,
                            node: Node {
                                name: attr(e, "name").unwrap_or_default(),
                                op: attr(e, "type").unwrap_or_default(),
                                ..Default::default()
                            },
                            inputs: Vec::new(),
                            outputs: Vec::new(),
                        });
                    }
                    (3, "data") if path[2] == "layer" => {
                        if let Some(l) = layers.last_mut() {
                            for a in e.attributes().flatten() {
                                l.node.attrs.push((a.key.as_ref().to_string(), value(&a)));
                            }
                        }
                    }
                    (4, "port") if path[2] == "layer" && (path[3] == "input" || path[3] == "output") => {
                        let p = Port {
                            id: attr(e, "id").and_then(|v| v.parse().ok()).unwrap_or(0),
                            precision: attr(e, "precision").unwrap_or_default().to_lowercase(),
                            names: attr(e, "names").unwrap_or_default(),
                            dims: Vec::new(),
                        };
                        let is_input = path[3] == "input";
                        if empty {
                            finish_port(layers.last_mut(), is_input, p);
                        } else {
                            port = Some((is_input, p));
                        }
                    }
                    (5, "dim") => in_dim = !empty,
                    (2, "edge") if path[1] == "edges" => {
                        let n = |k: &str| attr(e, k).and_then(|v| v.parse::<u32>().ok()).unwrap_or(NONE);
                        edges.push((n("from-layer"), n("from-port"), n("to-layer"), n("to-port")));
                    }
                    _ => {}
                }
                if !empty {
                    path.push(name);
                }
            }
            Event::Text(t) if in_dim => {
                if let Some((_, p)) = port.as_mut() {
                    let s: &str = t.as_ref();
                    p.dims.push(s.trim().parse().unwrap_or(-1));
                }
            }
            Event::End(e) => {
                let name = e.name().as_ref().to_string();
                path.pop();
                match name.as_str() {
                    "dim" => in_dim = false,
                    "port" if path.len() == 4 => {
                        if let Some((is_input, p)) = port.take() {
                            finish_port(layers.last_mut(), is_input, p);
                        }
                    }
                    "layer" if path.len() == 2 && nested > 0 => {
                        if let Some(l) = layers.last_mut() {
                            l.node.attrs.push(("body_layers".into(), nested.to_string()));
                        }
                    }
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    let mut g = Graph::default();
    let by_xml: HashMap<u32, usize> = layers.iter().enumerate().map(|(i, l)| (l.xml_id, i)).collect();
    let mut out_tensor: HashMap<(u32, u32), u32> = HashMap::new();
    for l in &layers {
        for p in &l.outputs {
            let key = format!("{}:{}", l.xml_id, p.id);
            let t = g.tensor(&key);
            let tensor = &mut g.tensors[t as usize];
            tensor.dtype = p.precision.clone();
            tensor.shape = Some(p.dims.clone());
            if let Some(first) = p.names.split(',').next().filter(|s| !s.is_empty()) {
                tensor.name = first.to_string();
            }
            out_tensor.insert((l.xml_id, p.id), t);
        }
    }
    // inputs[layer][k] follows the order of the layer's input ports
    let mut inputs: Vec<Vec<u32>> = layers.iter().map(|l| vec![NONE; l.inputs.len()]).collect();
    for (fl, fp, tl, tp) in edges {
        let (Some(&li), Some(&t)) = (by_xml.get(&tl), out_tensor.get(&(fl, fp))) else { continue };
        if let Some(k) = layers[li].inputs.iter().position(|p| p.id == tp) {
            inputs[li][k] = t;
        }
    }
    for (l, ins) in layers.into_iter().zip(inputs) {
        let mut node = l.node;
        node.inputs = ins;
        node.outputs = l.outputs.iter().map(|p| out_tensor[&(l.xml_id, p.id)]).collect();
        g.nodes.push(node);
    }
    // IR layers are not guaranteed to be stored in topological order
    g.finish();
    topo_sort(&mut g);
    Ok(g)
}

fn finish_port(layer: Option<&mut Layer>, is_input: bool, p: Port) {
    if let Some(l) = layer {
        if is_input {
            l.inputs.push(p);
        } else {
            l.outputs.push(p);
        }
    }
}

/// Reorder nodes topologically (stable: keeps file order among ready nodes).
pub fn topo_sort(g: &mut Graph) {
    let (preds, succs) = g.adjacency();
    let n = g.nodes.len();
    let mut indeg: Vec<usize> = preds.iter().map(|p| p.len()).collect();
    let mut ready: std::collections::BTreeSet<u32> = (0..n as u32).filter(|&i| indeg[i as usize] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(v) = ready.pop_first() {
        order.push(v);
        for &w in &succs[v as usize] {
            indeg[w as usize] -= 1;
            if indeg[w as usize] == 0 {
                ready.insert(w);
            }
        }
    }
    if order.len() != n || order.iter().enumerate().all(|(i, &v)| i as u32 == v) {
        return; // cyclic (not expected) or already sorted
    }
    let nodes = std::mem::take(&mut g.nodes);
    let mut slots: Vec<Option<Node>> = nodes.into_iter().map(Some).collect();
    g.nodes = order.iter().map(|&v| slots[v as usize].take().unwrap()).collect();
    g.finish();
}
