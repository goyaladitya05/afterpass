//! Collapsing: a scope tree built from node names, structural hashes for repeated blocks, and stacks.
//!
//! 1. Every node gets a scope (token path) from its name, or for compiled nodes from the source names it
//!    claims. Nodes without one inherit the common scope of their neighbours.
//! 2. Scopes form a group tree. Constant-only subgraphs (weights, int4 decompression chains) are folded away.
//! 3. Children with integer tokens (`layers.0`, `layers.1`, ...) are repeat instances. Each gets a structural
//!    hash (Weisfeiler-Lehman over its internal subgraph, ignoring names). Identical runs, or a periodic
//!    pattern such as 3 linear + 1 full attention layers, collapse into one stack drawn once with a xN badge;
//!    instances whose hash differs from the majority are outliers.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::ir::{Graph, NONE};

#[derive(Debug, Default, Clone)]
pub struct Group {
    pub label: String,
    pub parent: u32,
    pub children: Vec<u32>,
    /// Visible nodes directly in this group.
    pub nodes: Vec<u32>,
    /// Visible nodes in the whole subtree.
    pub size: u32,
    /// Smallest node index in the subtree, used to order items.
    pub first: u32,
    pub index: Option<u64>,
    pub hash: u64,
    pub stack: u32,
    pub outlier: bool,
}

#[derive(Debug, Clone)]
pub struct Stack {
    pub container: u32,
    /// Instance groups covered by the stack, in index order.
    pub instances: Vec<u32>,
    /// Instances per repeating unit (1 for plain repeats, 4 for a 3:1 hybrid pattern).
    pub period: u32,
    pub count: u32,
}

#[derive(Debug, Default)]
pub struct Hier {
    pub groups: Vec<Group>,
    pub stacks: Vec<Stack>,
    pub node_group: Vec<u32>,
    pub folded: Vec<bool>,
}

const IO_OPS: [&str; 4] = ["Parameter", "Result", "Input", "Output"];

/// Scope tokens of a node name: `/model/layers.3/attn/MatMul` gives [model, layers, 3, attn].
pub fn scope_tokens(name: &str) -> Vec<String> {
    let name = name.split(" + ").next().unwrap_or(name);
    let name = name.trim_start_matches('/');
    let name = name.strip_prefix("__module.").or_else(|| name.strip_prefix("self.")).unwrap_or(name);
    let pieces: Vec<&str> = name.split('/').filter(|p| !p.is_empty()).collect();
    if pieces.len() < 2 && !name.contains('.') {
        return Vec::new();
    }
    let mut tokens: Vec<String> = Vec::new();
    for (i, p) in pieces.iter().enumerate() {
        if p.contains("::") {
            continue; // op descriptors such as aten::view or ov_ext::linear
        }
        if i + 1 == pieces.len() && pieces.len() > 1 {
            break; // the last piece names the op itself
        }
        tokens.extend(p.split('.').filter(|t| !t.is_empty()).map(str::to_string));
    }
    if pieces.len() == 1 {
        tokens.pop(); // a dotted name such as model.layers.3.mlp: its last token is the op
    }
    tokens
}

fn common_prefix(a: &[String], b: &[String]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn h<T: Hash>(v: T) -> u64 {
    let mut s = DefaultHasher::new();
    v.hash(&mut s);
    s.finish()
}

impl Hier {
    pub fn build(g: &Graph, compiled: bool) -> Hier {
        let n = g.nodes.len();
        let folded = g.constant_nodes();
        let (preds, succs) = g.adjacency();

        // 1. scopes
        let mut scope: Vec<Vec<String>> = g.nodes.iter().map(|node| {
            if compiled {
                let mut claims = node.claims.iter().map(|c| scope_tokens(c));
                match claims.next() {
                    Some(first) => claims.fold(first, |acc, t| acc[..common_prefix(&acc, &t)].to_vec()),
                    None => Vec::new(),
                }
            } else {
                scope_tokens(&node.name)
            }
        }).collect();
        let fixed: Vec<bool> = (0..n).map(|i| !scope[i].is_empty() || IO_OPS.contains(&g.nodes[i].op.as_str())).collect();
        for pass in 0..2 {
            let order: Box<dyn Iterator<Item = usize>> = if pass == 0 { Box::new((0..n).rev()) } else { Box::new(0..n) };
            for v in order {
                if fixed[v] || !scope[v].is_empty() {
                    continue;
                }
                let around = if pass == 0 { &succs[v] } else { &preds[v] };
                let mut it = around.iter().map(|&u| &scope[u as usize]).filter(|s| !s.is_empty());
                if let Some(first) = it.next() {
                    let mut s = first.clone();
                    for t in it {
                        s.truncate(common_prefix(&s, t));
                    }
                    scope[v] = s;
                }
            }
        }

        // 2. group tree
        let mut groups = vec![Group { parent: NONE, first: u32::MAX, stack: NONE, ..Default::default() }];
        let mut child_of: HashMap<(u32, String), u32> = HashMap::new();
        let mut node_group = vec![NONE; n];
        for v in 0..n {
            if folded[v] {
                continue;
            }
            let mut cur = 0u32;
            for t in &scope[v] {
                cur = *child_of.entry((cur, t.clone())).or_insert_with(|| {
                    let id = groups.len() as u32;
                    groups.push(Group {
                        label: t.clone(),
                        parent: cur,
                        index: t.parse().ok(),
                        first: u32::MAX,
                        stack: NONE,
                        ..Default::default()
                    });
                    groups[cur as usize].children.push(id);
                    id
                });
            }
            node_group[v] = cur;
            groups[cur as usize].nodes.push(v as u32);
        }
        let mut hier = Hier { groups, stacks: Vec::new(), node_group, folded };
        hier.compress();
        hier.measure();
        hier.hash(g, &preds, &succs, compiled);
        hier.find_stacks();
        hier
    }

    /// Dissolve one-node groups (except repeat instances) and merge single-child chains.
    fn compress(&mut self) {
        self.measure();
        for gid in (1..self.groups.len()).rev() {
            let gr = &self.groups[gid];
            if gr.size == 1 && gr.index.is_none() && gr.children.is_empty() {
                let (parent, nodes) = (gr.parent, gr.nodes.clone());
                for &v in &nodes {
                    self.node_group[v as usize] = parent;
                }
                self.groups[parent as usize].nodes.extend(nodes);
                self.groups[parent as usize].children.retain(|&c| c != gid as u32);
                self.groups[gid].nodes.clear();
                self.groups[gid].parent = NONE;
            }
        }
        for gid in 0..self.groups.len() {
            loop {
                let gr = &self.groups[gid];
                if gr.parent == NONE && gid != 0 || !gr.nodes.is_empty() || gr.children.len() != 1 {
                    break;
                }
                let c = gr.children[0] as usize;
                if self.groups[c].index.is_some() {
                    break;
                }
                let child = std::mem::take(&mut self.groups[c]);
                for &v in &child.nodes {
                    self.node_group[v as usize] = gid as u32;
                }
                for &cc in &child.children {
                    self.groups[cc as usize].parent = gid as u32;
                }
                let gr = &mut self.groups[gid];
                gr.label = if gr.label.is_empty() { child.label } else { format!("{}/{}", gr.label, child.label) };
                gr.nodes = child.nodes;
                gr.children = child.children;
                self.groups[c].parent = NONE;
            }
        }
    }

    fn measure(&mut self) {
        for gid in (0..self.groups.len()).rev() {
            let gr = &self.groups[gid];
            let mut size = gr.nodes.len() as u32;
            let mut first = gr.nodes.iter().copied().min().unwrap_or(u32::MAX);
            for &c in &gr.children {
                size += self.groups[c as usize].size;
                first = first.min(self.groups[c as usize].first);
            }
            self.groups[gid].size = size;
            self.groups[gid].first = first;
        }
    }

    /// All visible nodes in a group's subtree.
    pub fn members(&self, gid: u32, out: &mut Vec<u32>) {
        let gr = &self.groups[gid as usize];
        out.extend_from_slice(&gr.nodes);
        for &c in &gr.children {
            self.members(c, out);
        }
    }

    /// Structural hash of every repeat instance: WL refinement restricted to the instance's own subgraph.
    /// Node labels use op type, weight shapes (always known, unlike activation shapes) and, for source graphs,
    /// attributes with digits masked inside identifiers (`past_key_values.5.key`). Compiled graphs hash op
    /// types only: their attributes and tactics vary per layer by timing noise.
    fn hash(&mut self, g: &Graph, preds: &[Vec<u32>], succs: &[Vec<u32>], compiled: bool) {
        let mask = |v: &str| -> String {
            if !v.chars().any(|c| c.is_alphabetic()) {
                return v.to_string();
            }
            let mut out = String::with_capacity(v.len());
            for c in v.chars() {
                if !c.is_ascii_digit() {
                    out.push(c);
                } else if !out.ends_with('#') {
                    out.push('#');
                }
            }
            out
        };
        let base: Vec<u64> = g.nodes.iter().map(|node| {
            let weights: Vec<&Option<Vec<i64>>> = node.inputs.iter().filter(|&&t| t != NONE).map(|&t| &g.tensors[t as usize])
                .filter(|t| t.constant || (t.producer != NONE && self.folded[t.producer as usize]))
                .map(|t| &t.shape).collect();
            let attrs: Vec<(&String, String)> = if compiled { Vec::new() } else {
                node.attrs.iter().filter(|(k, _)| k != "offset" && k != "size").map(|(k, v)| (k, mask(v))).collect()
            };
            h((&node.op, weights, attrs, node.inputs.len()))
        }).collect();
        let mut local: HashMap<u32, u64> = HashMap::new();
        let mut members = Vec::new();
        for gid in 0..self.groups.len() {
            if self.groups[gid].index.is_none() || self.groups[gid].parent == NONE {
                continue;
            }
            members.clear();
            self.members(gid as u32, &mut members);
            local.clear();
            for &v in &members {
                local.insert(v, base[v as usize]);
            }
            for _ in 0..3 {
                let next: Vec<(u32, u64)> = members.iter().map(|&v| {
                    let mut p: Vec<u64> = preds[v as usize].iter().filter_map(|u| local.get(u).copied()).collect();
                    let mut s: Vec<u64> = succs[v as usize].iter().filter_map(|u| local.get(u).copied()).collect();
                    p.sort_unstable();
                    s.sort_unstable();
                    (v, h((local[&v], p, s)))
                }).collect();
                local.extend(next);
            }
            let mut labels: Vec<u64> = local.values().copied().collect();
            labels.sort_unstable();
            self.groups[gid].hash = h(labels);
        }
    }

    fn find_stacks(&mut self) {
        for container in 0..self.groups.len() as u32 {
            let mut inst: Vec<u32> = self.groups[container as usize].children.iter().copied()
                .filter(|&c| self.groups[c as usize].index.is_some()).collect();
            if inst.len() < 2 {
                continue;
            }
            inst.sort_by_key(|&c| self.groups[c as usize].index);
            let hashes: Vec<u64> = inst.iter().map(|&c| self.groups[c as usize].hash).collect();
            let n = hashes.len();
            // longest periodic segment (period 1 to 8) covering at least two units
            let mut best: Option<(usize, usize, usize)> = None; // (start, units, period)
            for p in 1..=(n / 2).min(8) {
                let mut a = 0;
                while a + p < n {
                    let mut b = a;
                    while b + p < n && hashes[b] == hashes[b + p] {
                        b += 1;
                    }
                    let units = (b - a + p) / p;
                    if units >= 2 && best.is_none_or(|(_, u, q)| units * p > u * q) {
                        best = Some((a, units, p));
                    }
                    a = b.max(a) + 1;
                }
            }
            let mut stacked = vec![false; n];
            if let Some((a, units, p)) = best {
                self.add_stack(container, &inst[a..a + units * p], p as u32);
                stacked[a..a + units * p].iter_mut().for_each(|x| *x = true);
            }
            // identical consecutive runs among the rest
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && !stacked[i] && !stacked[j] && hashes[j] == hashes[i] {
                    j += 1;
                }
                if j - i >= 2 {
                    self.add_stack(container, &inst[i..j], 1);
                    stacked[i..j].iter_mut().for_each(|x| *x = true);
                }
                i = j;
            }
            // outliers: unstacked instances with a unique hash, when stacks cover most of the container
            if stacked.iter().filter(|&&x| x).count() * 2 >= n {
                let mut counts: HashMap<u64, usize> = HashMap::new();
                for &x in &hashes {
                    *counts.entry(x).or_default() += 1;
                }
                for (k, &c) in inst.iter().enumerate() {
                    self.groups[c as usize].outlier = !stacked[k] && counts[&hashes[k]] == 1;
                }
            }
        }
    }

    fn add_stack(&mut self, container: u32, instances: &[u32], period: u32) {
        let id = self.stacks.len() as u32;
        for &c in instances {
            self.groups[c as usize].stack = id;
        }
        self.stacks.push(Stack { container, instances: instances.to_vec(), period, count: instances.len() as u32 / period });
    }

    /// Label for a stack: `layers.0..21`.
    pub fn stack_label(&self, s: u32) -> String {
        let st = &self.stacks[s as usize];
        let first = &self.groups[st.instances[0] as usize];
        let last = &self.groups[*st.instances.last().unwrap() as usize];
        let base = &self.groups[st.container as usize].label;
        let base = base.rsplit('/').next().unwrap_or(base);
        format!("{base}.{}..{}", first.label, last.label)
    }
}
