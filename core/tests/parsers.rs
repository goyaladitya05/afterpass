use std::io::Read;

use afterpass_core::hier::{scope_tokens, Hier};
use afterpass_core::ir::Grade;
use afterpass_core::{bundle, onnx, ovir, Session};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn gunzip(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(b).read_to_end(&mut out).unwrap();
    out
}

#[test]
fn onnx_topology_without_weights() {
    let bytes = fixture("toy.onnx");
    let g = onnx::parse(&mut &bytes[..]).unwrap();
    assert_eq!(g.nodes.len(), 15);
    assert_eq!(g.nodes[1].name, "/model/layers.0/fc/MatMul");
    let w = g.tensors.iter().find(|t| t.name == "layers.0.w").unwrap();
    assert!(w.constant);
    assert_eq!(w.shape.as_deref(), Some(&[16, 16][..]));
    assert_eq!(w.dtype, "float");
    let logits = g.tensors.iter().find(|t| t.name == "logits").unwrap();
    assert_eq!(logits.shape.as_deref(), Some(&[1, 8, 4][..]));
}

#[test]
fn ir_topology() {
    let g = ovir::parse(&fixture("toy.xml")).unwrap();
    assert_eq!(g.nodes.len(), 26);
    let mm = g.nodes.iter().find(|n| n.name == "/model/layers.0/fc/MatMul").unwrap();
    assert_eq!(mm.op, "MatMul");
    let input = &g.tensors[mm.inputs[0] as usize];
    assert_eq!(g.nodes[input.producer as usize].name, "/embed/Gather");
    assert_eq!(input.shape.as_deref(), Some(&[1, 8, 16][..]));
    assert_eq!(g.constant_nodes().iter().filter(|&&c| c).count(), 9);
}

#[test]
fn bundles() {
    for f in ["toy.openvino-cpu.afterpass.json.gz", "toy.tensorrt-gpu.afterpass.json.gz"] {
        let m = bundle::parse(&gunzip(&fixture(f))).unwrap();
        assert_eq!(m.source.nodes.len(), 15);
        let c = m.compiled.as_ref().unwrap();
        assert!(c.nodes.iter().any(|n| n.time_us.is_some()), "{f}: no times");
        assert!(m.links.iter().any(|l| l.grade == Grade::Explicit), "{f}: no explicit links");
        assert!(m.latency_us.unwrap() > 0.0);
    }
}

#[test]
fn scopes() {
    assert_eq!(scope_tokens("/model/layers.3/attn/MatMul"), ["model", "layers", "3", "attn"]);
    assert_eq!(scope_tokens("__module.model.layers.17.mlp/aten::view/Reshape"), ["model", "layers", "17", "mlp"]);
    assert_eq!(scope_tokens("/model.0/conv/Conv + /model.0/act/Mul"), ["model", "0", "conv"]);
    assert!(scope_tokens("Constant_12").is_empty());
    assert!(scope_tokens("ReduceSum_7265191_Reshape/MOECompressed").is_empty());
    assert_eq!(scope_tokens("/model/layers.0/Add_1"), ["model", "layers", "0"]);
    assert_eq!(scope_tokens("/layer3/layer3.1/conv1/Conv"), ["layer3", "1", "conv1"]);
    assert_eq!(scope_tokens("/model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv"), ["model", "23", "cv3", "0", "0", "0", "conv"]);
}

#[test]
fn stacks_and_outliers() {
    for g in [onnx::parse(&mut &fixture("toy.onnx")[..]).unwrap(), ovir::parse(&fixture("toy.xml")).unwrap()] {
        let h = Hier::build(&g, false);
        assert_eq!(h.stacks.len(), 1);
        assert_eq!(h.stacks[0].count, 2);
        let outliers: Vec<&str> = h.groups.iter().filter(|g| g.outlier).map(|g| g.label.as_str()).collect();
        assert_eq!(outliers, ["2"]);
    }
}

#[test]
fn session_links_both_ways() {
    let m = bundle::parse(&gunzip(&fixture("toy.tensorrt-gpu.afterpass.json.gz"))).unwrap();
    let s = Session::from_model(m);
    assert_eq!(s.panes(), 2);
    for pane in 0..2 {
        let sc = &s.pane(pane).scene;
        assert!(!sc.kinds.is_empty());
        assert_eq!(sc.boxes.len(), sc.kinds.len() * 4);
        assert!(sc.width > 0.0 && sc.height > 0.0);
    }
}

#[test]
fn fused_kernels_land_in_their_majority_scope() {
    use afterpass_core::hier::majority_scope;
    let scopes = |v: &[&str]| v.iter().map(|s| scope_tokens(s)).collect::<Vec<_>>();
    let cross = scopes(&["/m/layers.5/Add", "/m/layers.6/norm/Mul", "/m/layers.6/norm/Pow", "/m/layers.6/norm/Sqrt"]);
    assert_eq!(majority_scope(cross), ["m", "layers", "6", "norm"]);
    assert_eq!(majority_scope(scopes(&["/m/layers.5/Add", "/m/layers.6/Add"])), ["m", "layers"]);
    assert_eq!(majority_scope(scopes(&["/m/layers.5/a/X", "/m/layers.5/b/Y"])), ["m", "layers", "5"]);
    // internal transformation names carry no scope and do not dilute the vote
    assert_eq!(majority_scope(scopes(&["Convert_1", "Reshape_2", "Add_3", "/m/layers.7/mlp/Mul"])), ["m", "layers", "7", "mlp"]);
}

#[test]
fn stepping_a_stack_keeps_inner_expansion() {
    use afterpass_core::ir::Model;
    let g = onnx::parse(&mut &fixture("toy.onnx")[..]).unwrap();
    let mut s = Session::from_model(Model { source: g, ..Default::default() });
    let titles = |s: &Session| s.pane(0).scene.titles.clone();
    let stack = titles(&s).iter().position(|t| t.starts_with("layers.0..1")).unwrap() as u32;
    let b = s.toggle(0, stack);
    assert!(b >= 0);
    let fc = titles(&s).iter().position(|t| t == "fc").unwrap() as u32;
    assert!(s.toggle(0, fc) >= 0);
    let frame = titles(&s).iter().position(|t| t.starts_with("layers.0..1")).unwrap() as u32;
    assert!(s.step(0, frame, 1) >= 0);
    let t = titles(&s);
    assert!(t.iter().any(|t| t.contains("unit 2/2")));
    assert!(t.iter().any(|t| t == "MatMul"), "fc group of the next unit should stay expanded: {t:?}");
}

#[test]
fn structure_recovers_ops_whose_names_were_lost() {
    use afterpass_core::ir::{Graph, Link, Model, Node, Via};
    // source a -> b -> c, compiled kernel k claims a and c (b's name was dropped by a pattern fusion)
    let mut src = Graph::default();
    let (x, ta, tb, tc) = (src.tensor("x"), src.tensor("ta"), src.tensor("tb"), src.tensor("tc"));
    for (name, i, o) in [("/m/a", x, ta), ("/m/b", ta, tb), ("/m/c", tb, tc)] {
        src.nodes.push(Node { name: name.into(), op: "Op".into(), inputs: vec![i], outputs: vec![o], ..Default::default() });
    }
    src.finish();
    let mut cmp = Graph::default();
    let (cx, cy) = (cmp.tensor("x"), cmp.tensor("y"));
    cmp.nodes.push(Node { name: "k".into(), op: "Fused".into(), inputs: vec![cx], outputs: vec![cy],
        claims: vec!["/m/a".into(), "/m/c".into()], ..Default::default() });
    cmp.finish();
    let links = vec![
        Link { compiled: 0, source: 0, grade: Grade::Explicit, via: Via::Name },
        Link { compiled: 0, source: 2, grade: Grade::Explicit, via: Via::Name },
    ];
    let s = Session::from_model(Model { source: src, compiled: Some(cmp), links, ..Default::default() });
    let l = &s.links()[0][0][1];
    assert_eq!(l.len(), 1);
    assert_eq!((l[0].other, l[0].grade, l[0].via), (0, Grade::Inferred, Via::Structure));
}

#[test]
fn reveal_opens_a_collapsed_stack() {
    use afterpass_core::ir::Model;
    let g = onnx::parse(&mut &fixture("toy.onnx")[..]).unwrap();
    let v = g.nodes.iter().position(|n| n.name == "/model/layers.1/fc/Add").unwrap() as u32;
    let mut s = Session::from_model(Model { source: g, ..Default::default() });
    let b = s.reveal(0, v, false);
    assert!(b >= 0);
    let p = s.pane(0);
    assert_eq!(p.box_nodes(b as u32), [v]);
    assert!(p.scene.titles.iter().any(|t| t.contains("unit 2/2")), "{:?}", p.scene.titles);
}

#[test]
fn findings_cover_time_and_provenance() {
    let s = Session::from_model(bundle::parse(&gunzip(&fixture("toy.tensorrt-gpu.afterpass.json.gz"))).unwrap());
    let f = s.findings_json();
    assert!(f.iter().any(|x| x["section"] == "time"));
    for x in &f {
        for t in x["targets"].as_array().unwrap() {
            assert!(t[0].as_u64().unwrap() < 2);
        }
    }
}

#[test]
fn two_runtimes_compare_through_the_source() {
    let load = |f: &str| Session::from_model(bundle::parse(&gunzip(&fixture(f))).unwrap());
    let mut s = load("toy.openvino-cpu.afterpass.json.gz");
    s.add_compiled(load("toy.tensorrt-gpu.afterpass.json.gz")).unwrap();
    assert_eq!(s.links().len(), 2);
    // every linked kernel of A reaches kernels of B through the source ops they share
    let a = s.pane(1);
    let k = (0..a.graph.nodes.len() as u32).find(|&k| !s.links()[0][1][k as usize].is_empty() && afterpass_core::analysis::data_path(a, k as usize)).unwrap();
    assert!(!afterpass_core::analysis::linked_nodes(s.links(), 1, &[k], 2).is_empty());
    let f = s.findings_json();
    let sections: Vec<&str> = f.iter().map(|x| x["section"].as_str().unwrap()).collect();
    assert_eq!(sections[0], "summary");
    assert!(sections.contains(&"delta"));
    assert!(f.iter().any(|x| x["title"].as_str().unwrap().starts_with("B  ")));
    // a capture of another source is refused
    let mut other = load("toy.openvino-cpu.afterpass.json.gz");
    let onnx_only = Session::from_model(afterpass_core::ir::Model { source: onnx::parse(&mut &fixture("toy.onnx")[..]).unwrap(), ..Default::default() });
    assert!(other.add_compiled(onnx_only).is_err());
}

/// Damaged or foreign files must come back as errors: a panic would abort the wasm worker.
#[test]
fn bad_inputs_are_errors_not_panics() {
    let onnx_bytes = fixture("toy.onnx");
    let xml = fixture("toy.xml");
    let json = gunzip(&fixture("toy.openvino-cpu.afterpass.json.gz"));
    let open = |kind: &str, b: &[u8]| -> Result<Session, String> {
        let s = match kind {
            "onnx" => Session::from_model(afterpass_core::ir::Model { source: onnx::parse(&mut &b[..])?, ..Default::default() }),
            "xml" => Session::from_model(afterpass_core::ir::Model { source: ovir::parse(b)?, ..Default::default() }),
            _ => Session::from_model(bundle::parse(b)?),
        };
        let _ = s.findings_json();
        Ok(s)
    };
    let mut cases: Vec<(&str, Vec<u8>)> = vec![("onnx", vec![]), ("xml", vec![]), ("bundle", vec![]),
        ("onnx", b"<html>not a model</html>".to_vec()), ("xml", b"<html><body>hi</body></html>".to_vec()),
        ("bundle", b"{\"format\": \"afterpass-bundle\", \"schema_version\": 0}".to_vec()),
        ("bundle", b"{\"format\": \"afterpass-bundle\", \"schema_version\": 0, \"provenance\": [{\"compiled\": 99, \"source\": 99}]}".to_vec())];
    for (kind, bytes) in [("onnx", &onnx_bytes), ("xml", &xml), ("bundle", &json)] {
        for cut in (0..bytes.len()).step_by(bytes.len() / 97 + 1) {
            cases.push((kind, bytes[..cut].to_vec()));
        }
    }
    // a bundle whose links point outside its graphs
    let mut v: serde_json::Value = serde_json::from_slice(&json).unwrap();
    v["provenance"][0]["compiled"] = 100000.into();
    v["provenance"][1]["source"] = 100000.into();
    cases.push(("bundle", serde_json::to_vec(&v).unwrap()));
    let (mut panics, mut empty) = (Vec::new(), Vec::new());
    for (i, (kind, bytes)) in cases.iter().enumerate() {
        match std::panic::catch_unwind(|| open(kind, bytes).map(|s| s.node_count(0))) {
            Err(_) => panics.push((i, *kind, bytes.len())),
            Ok(Ok(0)) => empty.push((i, *kind, bytes.len())),
            _ => {}
        }
    }
    assert!(panics.is_empty(), "panicked on {panics:?}");
    assert!(empty.is_empty(), "loaded without nodes: {empty:?}");
}

/// Requests for panes, boxes or nodes that do not exist (stale clicks during a load) are refused, not panics.
#[test]
fn stale_ids_are_refused() {
    let mut s = Session::from_model(bundle::parse(&gunzip(&fixture("toy.openvino-cpu.afterpass.json.gz"))).unwrap());
    for pane in [1, 2, 7] {
        assert_eq!(s.reveal(pane, 100_000, false), -1);
        assert_eq!(s.reveal(pane, 100_000, true), -1);
        assert_eq!(s.toggle(pane, 100_000), -1);
        assert_eq!(s.step(pane, 100_000, 1), -1);
        assert_eq!(s.describe(pane, 100_000), "");
        assert_eq!(s.inspect(pane, 100_000), "null");
    }
}
