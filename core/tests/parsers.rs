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
