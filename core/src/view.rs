//! Collapsible view of one graph. A frame (a group, or one unit of a stack) is laid out on its own; collapsed
//! items are boxes, expanded items nest their frame. Frame layouts are cached, so a toggle only re-lays out the
//! frames on the path to the root, and a stack lays out its template unit once however many instances it has.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::hier::Hier;
use crate::ir::{Graph, NONE};
use crate::layout::layout;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Frame {
    Group(u32),
    /// Unit `k` of stack `s`.
    Unit(u32, u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Item {
    Node(u32),
    Group(u32),
    Stack(u32),
    /// Where data enters (true) or leaves a frame.
    Port(bool),
}

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Kind {
    Node = 0,
    Group = 1,
    Stack = 2,
    Frame = 3,
    Port = 4,
}

pub const CHAR_W: f32 = 7.2;
pub const NODE_H: f32 = 30.0;
pub const GROUP_H: f32 = 44.0;
pub const HEADER: f32 = 24.0;
pub const PAD: f32 = 14.0;
const DEFAULT_VISIBLE: usize = 150;

struct Placed {
    w: f32,
    h: f32,
    items: Vec<(Item, [f32; 4], Option<(Frame, Rc<Placed>)>)>,
    edges: Vec<Vec<(f32, f32)>>,
}

#[derive(Default)]
pub struct Scene {
    pub boxes: Vec<f32>,
    pub kinds: Vec<u8>,
    /// Bit 0: outlier block.
    pub flags: Vec<u8>,
    pub times: Vec<f32>,
    pub titles: Vec<String>,
    pub subtitles: Vec<String>,
    pub segs: Vec<f32>,
    pub width: f32,
    pub height: f32,
    /// What each box shows, and the frame it sits in.
    refs: Vec<(Item, Frame)>,
}

pub struct Pane {
    pub graph: Graph,
    pub compiled: bool,
    pub hier: Hier,
    preds: Vec<Vec<u32>>,
    succs: Vec<Vec<u32>>,
    expanded: HashSet<Item>,
    unit: HashMap<u32, u32>,
    cache: HashMap<Frame, Rc<Placed>>,
    parent: HashMap<Frame, Frame>,
    owner: Vec<u32>,
    pub scene: Scene,
    /// Innermost box showing each node.
    node_box: Vec<u32>,
}

fn clamp_w(chars: usize, min: f32, max: f32) -> f32 {
    (chars as f32 * CHAR_W + 24.0).clamp(min, max)
}

impl Pane {
    pub fn new(graph: Graph, compiled: bool, flat: bool) -> Pane {
        let hier = if flat { Hier::flat(&graph) } else { Hier::build(&graph, compiled) };
        let (preds, succs) = graph.adjacency();
        let n = graph.nodes.len();
        let mut p = Pane {
            graph,
            compiled,
            hier,
            preds,
            succs,
            expanded: HashSet::new(),
            unit: HashMap::new(),
            cache: HashMap::new(),
            parent: HashMap::new(),
            owner: vec![NONE; n],
            scene: Scene::default(),
            node_box: vec![NONE; n],
        };
        p.default_expansion();
        p.rebuild();
        p
    }

    fn items(&self, f: Frame) -> Vec<Item> {
        let h = &self.hier;
        let gid = match f {
            Frame::Group(g) => g,
            Frame::Unit(s, k) => {
                let st = &h.stacks[s as usize];
                let unit = &st.instances[(k * st.period) as usize..((k + 1) * st.period) as usize];
                if st.period > 1 {
                    return unit.iter().map(|&g| Item::Group(g)).collect();
                }
                unit[0]
            }
        };
        let gr = &h.groups[gid as usize];
        let mut items: Vec<(u32, Item)> = gr.nodes.iter().map(|&v| (v, Item::Node(v))).collect();
        for &c in &gr.children {
            let cg = &h.groups[c as usize];
            match cg.stack {
                NONE => items.push((cg.first, Item::Group(c))),
                s if h.stacks[s as usize].instances[0] == c => items.push((cg.first, Item::Stack(s))),
                _ => {}
            }
        }
        items.sort_by_key(|(first, _)| *first);
        items.into_iter().map(|(_, i)| i).collect()
    }

    fn members(&self, item: Item, out: &mut Vec<u32>) {
        match item {
            Item::Node(v) => out.push(v),
            Item::Group(g) => self.hier.members(g, out),
            Item::Stack(s) => {
                for &g in &self.hier.stacks[s as usize].instances {
                    self.hier.members(g, out);
                }
            }
            Item::Port(_) => {}
        }
    }

    fn child_frame(&self, item: Item) -> Option<Frame> {
        if !self.expanded.contains(&item) {
            return None;
        }
        match item {
            Item::Group(g) => Some(Frame::Group(g)),
            Item::Stack(s) => Some(Frame::Unit(s, self.unit.get(&s).copied().unwrap_or(0))),
            _ => None,
        }
    }

    fn default_expansion(&mut self) {
        let mut visible = self.items(Frame::Group(0)).len();
        loop {
            // largest collapsed group among the visible items
            let mut best: Option<(u32, usize)> = None;
            let mut frames = vec![Frame::Group(0)];
            while let Some(f) = frames.pop() {
                for it in self.items(f) {
                    if let Some(cf) = self.child_frame(it) {
                        frames.push(cf);
                    } else if let Item::Group(g) = it {
                        if self.hier.groups[g as usize].index.is_some() {
                            continue; // repeat instances (often outliers) stay collapsed in the overview
                        }
                        let size = self.hier.groups[g as usize].size;
                        if best.is_none_or(|(_, s)| size as usize > s) {
                            best = Some((g, size as usize));
                        }
                    }
                }
            }
            let Some((g, _)) = best else { break };
            let add = self.items(Frame::Group(g)).len();
            if visible + add - 1 > DEFAULT_VISIBLE {
                break;
            }
            visible += add - 1;
            self.expanded.insert(Item::Group(g));
        }
    }

    fn label(&self, item: Item) -> (String, String, f32, f32) {
        let h = &self.hier;
        match item {
            Item::Node(v) => {
                let n = &self.graph.nodes[v as usize];
                let short = n.name.rsplit('/').next().unwrap_or(&n.name).to_string();
                (n.op.clone(), short, clamp_w(n.op.len(), 60.0, 220.0), NODE_H)
            }
            Item::Group(g) => {
                let gr = &h.groups[g as usize];
                let sub = format!("{} ops", gr.size);
                let title = match gr.index {
                    // a repeat instance reads "layers.3", not "3"
                    Some(_) => {
                        let parent = &h.groups[gr.parent as usize].label;
                        format!("{}.{}", parent.rsplit('/').next().unwrap_or(parent), gr.label)
                    }
                    None => gr.label.clone(),
                };
                let w = clamp_w(title.len().max(8), 90.0, 300.0);
                (title, sub, w, GROUP_H)
            }
            Item::Stack(s) => {
                let st = &h.stacks[s as usize];
                let size = h.groups[st.instances[0] as usize].size;
                let title = format!("{}  \u{d7}{}", h.stack_label(s), st.count);
                let sub = if st.period > 1 { format!("{} blocks per unit", st.period) } else { format!("{size} ops each") };
                (title.clone(), sub, clamp_w(title.len() + 5, 120.0, 320.0), GROUP_H)
            }
            Item::Port(_) => (String::new(), String::new(), 8.0, 8.0),
        }
    }

    fn place(&mut self, f: Frame) -> Rc<Placed> {
        if let Some(p) = self.cache.get(&f) {
            return p.clone();
        }
        let mut items = self.items(f);
        // aggregate node edges into item edges, plus boundary ports
        let mut members: Vec<Vec<u32>> = Vec::with_capacity(items.len());
        for (i, &it) in items.iter().enumerate() {
            let mut m = Vec::new();
            self.members(it, &mut m);
            for &v in &m {
                self.owner[v as usize] = i as u32;
            }
            members.push(m);
        }
        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        let (mut ins, mut outs) = (HashSet::new(), HashSet::new());
        for (i, m) in members.iter().enumerate() {
            for &v in m {
                for &u in &self.preds[v as usize] {
                    if self.hier.folded[u as usize] {
                        continue;
                    }
                    match self.owner[u as usize] {
                        NONE => {
                            ins.insert(i as u32);
                        }
                        o if o != i as u32 => {
                            edges.insert((o, i as u32));
                        }
                        _ => {}
                    }
                }
                if self.succs[v as usize].iter().any(|&w| !self.hier.folded[w as usize] && self.owner[w as usize] == NONE) {
                    outs.insert(i as u32);
                }
            }
        }
        for m in &members {
            for &v in m {
                self.owner[v as usize] = NONE;
            }
        }
        let root = f == Frame::Group(0);
        let mut edges: Vec<(u32, u32)> = edges.into_iter().collect();
        edges.sort_unstable();
        if !root && !ins.is_empty() {
            let p = items.len() as u32;
            items.push(Item::Port(true));
            let mut ins: Vec<u32> = ins.into_iter().collect();
            ins.sort_unstable();
            edges.extend(ins.into_iter().map(|i| (p, i)));
        }
        if !root && !outs.is_empty() {
            let p = items.len() as u32;
            items.push(Item::Port(false));
            let mut outs: Vec<u32> = outs.into_iter().collect();
            outs.sort_unstable();
            edges.extend(outs.into_iter().map(|i| (i, p)));
        }
        // sizes, recursing into expanded children
        let mut children = Vec::with_capacity(items.len());
        let mut sizes = Vec::with_capacity(items.len());
        for &it in &items {
            match self.child_frame(it) {
                Some(cf) => {
                    self.parent.insert(cf, f);
                    let c = self.place(cf);
                    let w = (c.w + 2.0 * PAD).max(clamp_w(self.label(it).0.len() + 6, 120.0, 400.0));
                    sizes.push((w, c.h + HEADER + PAD));
                    children.push(Some((cf, c)));
                }
                None => {
                    let (_, _, w, h) = self.label(it);
                    sizes.push((w, h));
                    children.push(None);
                }
            }
        }
        let l = layout(&sizes, &edges);
        let placed = Rc::new(Placed {
            w: l.width,
            h: l.height,
            items: items.into_iter().zip(children).enumerate()
                .map(|(i, (it, c))| (it, [l.pos[i].0, l.pos[i].1, sizes[i].0, sizes[i].1], c))
                .collect(),
            edges: l.edges,
        });
        self.cache.insert(f, placed.clone());
        placed
    }

    fn emit(&mut self, p: &Placed, frame: Frame, ox: f32, oy: f32) {
        for (it, r, child) in &p.items {
            let (x, y) = (ox + r[0], oy + r[1]);
            let id = self.scene.kinds.len() as u32;
            let (title, sub, _, _) = self.label(*it);
            let kind = match (it, child) {
                (_, Some(_)) => Kind::Frame,
                (Item::Node(_), _) => Kind::Node,
                (Item::Group(_), _) => Kind::Group,
                (Item::Stack(_), _) => Kind::Stack,
                (Item::Port(_), _) => Kind::Port,
            };
            let mut m = Vec::new();
            self.members(*it, &mut m);
            let times: Vec<f32> = m.iter().filter_map(|&v| self.graph.nodes[v as usize].time_us).collect();
            for &v in &m {
                self.node_box[v as usize] = id;
            }
            let outlier = match it {
                Item::Group(g) => self.hier.groups[*g as usize].outlier,
                _ => false,
            };
            let title = match (it, child) {
                (Item::Stack(s), Some((Frame::Unit(_, k), _))) => {
                    let st = &self.hier.stacks[*s as usize];
                    format!("{}  unit {}/{}", title, k + 1, st.count)
                }
                _ => title,
            };
            let sc = &mut self.scene;
            sc.boxes.extend_from_slice(&[x, y, r[2], r[3]]);
            sc.kinds.push(kind as u8);
            sc.flags.push(outlier as u8);
            sc.times.push(if times.is_empty() { f32::NAN } else { times.iter().sum() });
            sc.titles.push(title);
            sc.subtitles.push(sub);
            sc.refs.push((*it, frame));
            if let Some((cf, c)) = child {
                self.emit(&c.clone(), *cf, x + PAD, y + HEADER);
            }
        }
        for e in &p.edges {
            for w in e.windows(2) {
                self.scene.segs.extend_from_slice(&[ox + w[0].0, oy + w[0].1, ox + w[1].0, oy + w[1].1]);
            }
        }
    }

    pub fn rebuild(&mut self) {
        let root = self.place(Frame::Group(0));
        self.scene = Scene { width: root.w, height: root.h, ..Default::default() };
        self.node_box.iter_mut().for_each(|b| *b = NONE);
        self.emit(&root, Frame::Group(0), 0.0, 0.0);
    }

    fn invalidate(&mut self, mut f: Frame) {
        loop {
            self.cache.remove(&f);
            match self.parent.get(&f) {
                Some(&p) => f = p,
                None => break,
            }
        }
    }

    fn find(&self, item: Item) -> i32 {
        self.scene.refs.iter().position(|&(i, _)| i == item).map_or(-1, |b| b as i32)
    }

    /// Expand a collapsed group or stack, or collapse an expanded one.
    /// Returns the item's box after the re-layout, or -1 if nothing changed.
    pub fn toggle(&mut self, b: u32) -> i32 {
        let Some(&(item, frame)) = self.scene.refs.get(b as usize) else { return -1 };
        if !matches!(item, Item::Group(_) | Item::Stack(_)) {
            return -1;
        }
        if !self.expanded.remove(&item) {
            self.expanded.insert(item);
        }
        self.invalidate(frame);
        self.rebuild();
        self.find(item)
    }

    /// Show another unit of an expanded stack. Returns the stack's box, or -1.
    pub fn step(&mut self, b: u32, delta: i32) -> i32 {
        let Some(&(item @ Item::Stack(s), frame)) = self.scene.refs.get(b as usize) else { return -1 };
        if !self.expanded.contains(&item) {
            return -1;
        }
        let st = self.hier.stacks[s as usize].clone();
        let k = self.unit.get(&s).copied().unwrap_or(0);
        let next = (k as i32 + delta).rem_euclid(st.count as i32) as u32;
        let p = st.period as usize;
        // carry expansion over: the same relative path in the new unit's instances
        for j in 0..p {
            let (from, to) = (st.instances[k as usize * p + j], st.instances[next as usize * p + j]);
            let carried: Vec<Item> = self.expanded.iter().filter_map(|&it| self.carry(it, from, to)).collect();
            self.expanded.extend(carried);
        }
        self.unit.insert(s, next);
        self.invalidate(frame);
        self.rebuild();
        self.find(item)
    }

    /// Labels from `root` (exclusive) down to group `g`, or None when `g` is not under `root`.
    fn rel_path(&self, mut g: u32, root: u32) -> Option<Vec<&str>> {
        let mut path = Vec::new();
        while g != root {
            if g == NONE || g == 0 {
                return None;
            }
            path.push(self.hier.groups[g as usize].label.as_str());
            g = self.hier.groups[g as usize].parent;
        }
        path.reverse();
        Some(path)
    }

    fn resolve(&self, root: u32, path: &[&str]) -> Option<u32> {
        path.iter().try_fold(root, |g, label| {
            self.hier.groups[g as usize].children.iter().copied().find(|&c| self.hier.groups[c as usize].label == *label)
        })
    }

    /// The item in instance `to` that corresponds to expanded item `it` in instance `from`.
    fn carry(&self, it: Item, from: u32, to: u32) -> Option<Item> {
        match it {
            Item::Group(g) => {
                let path = self.rel_path(g, from)?;
                self.resolve(to, &path).map(Item::Group)
            }
            Item::Stack(s) => {
                let stacks = &self.hier.stacks;
                let c = stacks[s as usize].container;
                let nc = self.resolve(to, &self.rel_path(c, from)?)?;
                let i = (0..stacks.len() as u32).filter(|&t| stacks[t as usize].container == c).position(|t| t == s)?;
                (0..stacks.len() as u32).filter(|&t| stacks[t as usize].container == nc).nth(i).map(Item::Stack)
            }
            _ => None,
        }
    }

    /// Graph nodes shown by a box.
    pub fn box_nodes(&self, b: u32) -> Vec<u32> {
        let mut m = Vec::new();
        if let Some(&(item, _)) = self.scene.refs.get(b as usize) {
            self.members(item, &mut m);
        }
        m
    }

    /// Boxes currently showing the given nodes.
    pub fn node_boxes(&self, nodes: impl IntoIterator<Item = u32>) -> Vec<u32> {
        let mut b: Vec<u32> = nodes.into_iter().map(|v| self.node_box[v as usize]).filter(|&b| b != NONE).collect();
        b.sort_unstable();
        b.dedup();
        b
    }
}
