//! Layered (Sugiyama style) layout, top to bottom: break cycles, longest-path ranks, dummy nodes for long
//! edges, barycenter ordering sweeps, then x placement by alternating order-preserving centring passes.

pub const RANK_GAP: f32 = 36.0;
pub const NODE_GAP: f32 = 18.0;
const DUMMY_W: f32 = 6.0;
const ORDER_SWEEPS: usize = 8;
const PLACE_SWEEPS: usize = 8;
/// Edges spanning more ranks than this are drawn straight instead of getting dummy nodes.
const MAX_SPAN: i32 = 48;

pub struct Layout {
    /// Top-left corner per input node.
    pub pos: Vec<(f32, f32)>,
    /// Polyline per input edge, from the source's bottom to the target's top.
    pub edges: Vec<Vec<(f32, f32)>>,
    pub width: f32,
    pub height: f32,
}

pub fn layout(sizes: &[(f32, f32)], edges: &[(u32, u32)]) -> Layout {
    let n = sizes.len();
    if n == 0 {
        return Layout { pos: Vec::new(), edges: Vec::new(), width: 0.0, height: 0.0 };
    }
    // 1. break cycles: edges to a node on the DFS stack are reversed
    let mut out: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
    for (i, &(u, v)) in edges.iter().enumerate() {
        if u != v {
            out[u as usize].push((v, i));
        }
    }
    let mut reversed = vec![false; edges.len()];
    let mut state = vec![0u8; n]; // 0 new, 1 on stack, 2 done
    for root in 0..n {
        if state[root] != 0 {
            continue;
        }
        let mut stack = vec![(root, 0usize)];
        state[root] = 1;
        while let Some(&mut (v, ref mut k)) = stack.last_mut() {
            if *k < out[v].len() {
                let (w, e) = out[v][*k];
                *k += 1;
                match state[w as usize] {
                    0 => {
                        state[w as usize] = 1;
                        stack.push((w as usize, 0));
                    }
                    1 => reversed[e] = true,
                    _ => {}
                }
            } else {
                state[v] = 2;
                stack.pop();
            }
        }
    }
    let dag: Vec<(usize, usize, usize)> = edges.iter().enumerate()
        .filter(|&(_, &(u, v))| u != v)
        .map(|(i, &(u, v))| if reversed[i] { (v as usize, u as usize, i) } else { (u as usize, v as usize, i) })
        .collect();

    // 2. ranks by longest path, then sources pulled down next to their first consumer
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v, _) in &dag {
        preds[v].push(u);
        succs[u].push(v);
    }
    let mut indeg: Vec<usize> = preds.iter().map(Vec::len).collect();
    let mut queue: std::collections::VecDeque<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    let mut topo = Vec::with_capacity(n);
    let mut rank = vec![0i32; n];
    while let Some(v) = queue.pop_front() {
        topo.push(v);
        for &w in &succs[v] {
            rank[w] = rank[w].max(rank[v] + 1);
            indeg[w] -= 1;
            if indeg[w] == 0 {
                queue.push_back(w);
            }
        }
    }
    for &v in topo.iter().rev() {
        if preds[v].is_empty() && !succs[v].is_empty() {
            rank[v] = succs[v].iter().map(|&w| rank[w]).min().unwrap() - 1;
        }
    }

    // 3. layered graph with dummy chains
    let mut width: Vec<f32> = sizes.iter().map(|s| s.0).collect();
    let mut lrank: Vec<i32> = rank.clone();
    let mut lpred: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut lsucc: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut chains: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
    for &(u, v, e) in &dag {
        let span = rank[v] - rank[u];
        let mut prev = u;
        if span > 1 && span <= MAX_SPAN {
            for r in rank[u] + 1..rank[v] {
                let d = width.len();
                width.push(DUMMY_W);
                lrank.push(r);
                lpred.push(vec![prev]);
                lsucc.push(Vec::new());
                lsucc[prev].push(d);
                chains[e].push(d);
                prev = d;
            }
        }
        lsucc[prev].push(v);
        lpred[v].push(prev);
    }
    let total = width.len();
    let rmin = *lrank.iter().min().unwrap();
    let nranks = (lrank.iter().max().unwrap() - rmin + 1) as usize;
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); nranks];

    // 4. ordering: initial DFS order, then barycenter sweeps
    let mut seen = vec![false; total];
    let mut dfs: Vec<usize> = (0..n).rev().filter(|&v| lpred[v].is_empty()).collect();
    while let Some(v) = dfs.pop() {
        if std::mem::replace(&mut seen[v], true) {
            continue;
        }
        layers[(lrank[v] - rmin) as usize].push(v);
        dfs.extend(lsucc[v].iter().rev().filter(|&&w| !seen[w]));
    }
    for v in 0..total {
        if !seen[v] {
            layers[(lrank[v] - rmin) as usize].push(v);
        }
    }
    let mut order = vec![0f32; total];
    let index = |layers: &Vec<Vec<usize>>, order: &mut Vec<f32>| {
        for l in layers {
            for (i, &v) in l.iter().enumerate() {
                order[v] = i as f32;
            }
        }
    };
    index(&layers, &mut order);
    for sweep in 0..ORDER_SWEEPS {
        let down = sweep % 2 == 0;
        let ranks: Vec<usize> = if down { (1..nranks).collect() } else { (0..nranks.saturating_sub(1)).rev().collect() };
        for r in ranks {
            let key: Vec<(f32, usize)> = layers[r].iter().map(|&v| {
                let nb = if down { &lpred[v] } else { &lsucc[v] };
                let k = if nb.is_empty() { order[v] } else { nb.iter().map(|&w| order[w]).sum::<f32>() / nb.len() as f32 };
                (k, v)
            }).collect();
            let mut key = key;
            key.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(order[a.1].partial_cmp(&order[b.1]).unwrap()));
            layers[r] = key.into_iter().map(|(_, v)| v).collect();
            for (i, &v) in layers[r].iter().enumerate() {
                order[v] = i as f32;
            }
        }
    }

    // 5. x placement
    let mut cx = vec![0f32; total];
    for l in &layers {
        let mut x = 0.0;
        for &v in l {
            cx[v] = x + width[v] / 2.0;
            x += width[v] + NODE_GAP;
        }
    }
    for sweep in 0..PLACE_SWEEPS {
        let down = sweep % 2 == 0;
        let ranks: Vec<usize> = if down { (0..nranks).collect() } else { (0..nranks).rev().collect() };
        for r in ranks {
            let l = &layers[r];
            let want: Vec<f32> = l.iter().map(|&v| {
                let nb = if down { &lpred[v] } else { &lsucc[v] };
                if nb.is_empty() { cx[v] } else { nb.iter().map(|&w| cx[w]).sum::<f32>() / nb.len() as f32 }
            }).collect();
            let gap = |a: usize, b: usize| (width[l[a]] + width[l[b]]) / 2.0 + NODE_GAP;
            let mut left = want.clone();
            for i in 1..l.len() {
                left[i] = left[i].max(left[i - 1] + gap(i - 1, i));
            }
            let mut right = want;
            for i in (0..l.len().saturating_sub(1)).rev() {
                right[i] = right[i].min(right[i + 1] - gap(i, i + 1));
            }
            for (i, &v) in l.iter().enumerate() {
                cx[v] = (left[i] + right[i]) / 2.0;
            }
        }
    }

    // y placement: rank rows sized by their tallest node
    let mut row_h = vec![0f32; nranks];
    for v in 0..n {
        let r = (lrank[v] - rmin) as usize;
        row_h[r] = row_h[r].max(sizes[v].1);
    }
    let mut row_y = vec![0f32; nranks];
    for r in 1..nranks {
        row_y[r] = row_y[r - 1] + row_h[r - 1] + RANK_GAP;
    }
    let xmin = (0..total).map(|v| cx[v] - width[v] / 2.0).fold(f32::INFINITY, f32::min);
    let xmax = (0..total).map(|v| cx[v] + width[v] / 2.0).fold(f32::NEG_INFINITY, f32::max);
    let cy = |v: usize| {
        let r = (lrank[v] - rmin) as usize;
        row_y[r] + row_h[r] / 2.0
    };
    let pos: Vec<(f32, f32)> = (0..n).map(|v| (cx[v] - sizes[v].0 / 2.0 - xmin, cy(v) - sizes[v].1 / 2.0)).collect();
    let polylines = edges.iter().enumerate().map(|(i, &(u, v))| {
        let (u, v) = (u as usize, v as usize);
        if u == v {
            return Vec::new();
        }
        let (a, b) = if reversed[i] { (v, u) } else { (u, v) };
        let mut pts = vec![(cx[a] - xmin, pos[a].1 + sizes[a].1)];
        pts.extend(chains[i].iter().map(|&d| (cx[d] - xmin, cy(d))));
        pts.push((cx[b] - xmin, pos[b].1));
        if reversed[i] {
            pts.reverse();
        }
        pts
    }).collect();
    Layout {
        pos,
        edges: polylines,
        width: xmax - xmin,
        height: row_y[nranks - 1] + row_h[nranks - 1],
    }
}
