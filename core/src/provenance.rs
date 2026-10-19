//! Link compiled nodes to source nodes by the names they claim, grading every link. A port of the capture's
//! Linker (capture/src/afterpass/provenance.py): grades and `via` must match, so an imported compiled graph
//! reads the same as a captured one.

use std::collections::HashMap;

use crate::ir::{Grade, Graph, Link, Via, NONE};

pub struct Linker<'a> {
    by_name: HashMap<&'a str, Vec<u32>>,
    producer: HashMap<&'a str, u32>,
}

impl<'a> Linker<'a> {
    /// `tensors`: match claims against tensor names too. The capture does for ONNX sources only: an IR
    /// source's tensors are keyed `<node>:<port>` there, which no claim ever is.
    pub fn new(source: &'a Graph, tensors: bool) -> Linker<'a> {
        let mut by_name: HashMap<&str, Vec<u32>> = HashMap::new();
        for (i, n) in source.nodes.iter().enumerate().filter(|(_, n)| !n.name.is_empty()) {
            by_name.entry(&n.name).or_default().push(i as u32);
        }
        let producer = if tensors {
            source.tensors.iter().filter(|t| t.producer != NONE).map(|t| (t.name.as_str(), t.producer)).collect()
        } else {
            HashMap::new()
        };
        Linker { by_name, producer }
    }

    /// (source node, grade, via) for one claimed name.
    fn resolve(&self, name: &str) -> Vec<(u32, Grade, Via)> {
        if let Some(ids) = self.by_name.get(name) {
            let grade = if ids.len() == 1 { Grade::Explicit } else { Grade::Inferred };
            return ids.iter().map(|&i| (i, grade, Via::Name)).collect();
        }
        if let Some(&p) = self.producer.get(name) {
            return vec![(p, Grade::Inferred, Via::Tensor)];
        }
        // OpenVINO GPU names an output port as "<op>.<port>"
        if let Some((base, port)) = name.rsplit_once('.')
            && !port.is_empty() && port.bytes().all(|c| c.is_ascii_digit())
        {
            let found = self.resolve(base);
            if !found.is_empty() {
                return found.into_iter().map(|(i, _, _)| (i, Grade::Inferred, Via::Port)).collect();
            }
        }
        let mut prefix = name;
        while prefix.trim_matches('/').contains('/') {
            prefix = prefix.trim_end_matches('/').rsplit_once('/').map_or("", |(p, _)| p);
            if let Some(ids) = self.by_name.get(prefix) {
                return ids.iter().map(|&i| (i, Grade::Inferred, Via::Prefix)).collect();
            }
            if let Some(&p) = self.producer.get(prefix) {
                return vec![(p, Grade::Inferred, Via::Prefix)];
            }
        }
        Vec::new()
    }

    /// Links of compiled node `compiled`: the best grade per source node, or one unknown link.
    pub fn link(&self, compiled: u32, claims: &[String]) -> Vec<Link> {
        let mut best: Vec<Link> = Vec::new();
        for name in claims {
            for (source, grade, via) in self.resolve(name) {
                match best.iter_mut().find(|l| l.source == source) {
                    Some(l) if grade < l.grade => *l = Link { compiled, source, grade, via },
                    Some(_) => {}
                    None => best.push(Link { compiled, source, grade, via }),
                }
            }
        }
        if best.is_empty() {
            best.push(Link { compiled, source: NONE, grade: Grade::Unknown, via: Via::Unknown });
        }
        best
    }
}
