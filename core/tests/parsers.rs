use std::io::Read;

use afterpass_core::analysis::{data_path, linked_nodes};
use afterpass_core::hier::{majority_scope, scope_tokens, Hier};
use afterpass_core::ir::{Graph, Grade, Link, Model, Node, Via};
use afterpass_core::{bundle, onnx, ovir, Session};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn toy_onnx() -> Graph {
    onnx::parse(&mut &fixture("toy.onnx")[..]).unwrap()
}

fn toy_session(bundle_file: &str) -> Session {
    Session::from_model(bundle::parse(&gunzip(&fixture(bundle_file))).unwrap())
}

fn gunzip(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(b).read_to_end(&mut out).unwrap();
    out
}

#[test]
fn onnx_topology_without_weights() {
    let g = toy_onnx();
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
        // opens as source and compiled panes, both drawn, with findings that point into them
        let s = Session::from_model(m);
        assert_eq!(s.panes(), 2);
        for pane in 0..2 {
            let sc = &s.pane(pane).scene;
            assert!(!sc.kinds.is_empty());
            assert_eq!(sc.boxes.len(), sc.kinds.len() * 4);
            assert!(sc.width > 0.0 && sc.height > 0.0);
        }
        // the compiled pane says who compiled what, and when; the source pane was compiled by no one
        let about: serde_json::Value = serde_json::from_str(&s.about(1)).unwrap();
        assert!(["openvino", "tensorrt"].contains(&about["runtime"].as_str().unwrap()), "{about}");
        assert_eq!((about["model"].as_str(), about["captured"].as_str().map(str::len)), (Some("toy.onnx"), Some(10)), "{about}");
        assert_eq!(s.about(0), "{}");
        let f = s.findings_json();
        assert!(f.iter().any(|x| x["section"] == "time"), "no time findings: {f:?}");
        assert!(f.iter().flat_map(|x| x["targets"].as_array().unwrap()).all(|t| t[0].as_u64().unwrap() < 2));
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
    for g in [toy_onnx(), ovir::parse(&fixture("toy.xml")).unwrap()] {
        let (preds, succs) = g.adjacency();
        let h = Hier::build(&g, &preds, &succs, false);
        assert_eq!(h.stacks.len(), 1);
        assert_eq!(h.stacks[0].count, 2);
        let outliers: Vec<&str> = h.groups.iter().filter(|g| g.outlier).map(|g| g.label.as_str()).collect();
        assert_eq!(outliers, ["2"]);
    }
}

#[test]
fn fused_kernels_land_in_their_majority_scope() {
    let scopes = |v: &[&str]| v.iter().map(|s| scope_tokens(s)).collect::<Vec<_>>();
    let cross = scopes(&["/m/layers.5/Add", "/m/layers.6/norm/Mul", "/m/layers.6/norm/Pow", "/m/layers.6/norm/Sqrt"]);
    assert_eq!(majority_scope(cross), ["m", "layers", "6", "norm"]);
    assert_eq!(majority_scope(scopes(&["/m/layers.5/Add", "/m/layers.6/Add"])), ["m", "layers"]);
    assert_eq!(majority_scope(scopes(&["/m/layers.5/a/X", "/m/layers.5/b/Y"])), ["m", "layers", "5"]);
    // internal transformation names carry no scope and do not dilute the vote
    assert_eq!(majority_scope(scopes(&["Convert_1", "Reshape_2", "Add_3", "/m/layers.7/mlp/Mul"])), ["m", "layers", "7", "mlp"]);
}

#[test]
fn structure_recovers_ops_whose_names_were_lost() {
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

/// A graph from (op, inputs) pairs: node i writes tensor `t{i}` of shape `[8]` and reads the tensors of its
/// inputs, or a graph input when it has none.
fn dag(nodes: &[(&str, &[usize])]) -> Graph {
    let mut g = Graph::default();
    for (i, (op, inputs)) in nodes.iter().enumerate() {
        let ins = if inputs.is_empty() { vec![g.tensor("x")] } else { inputs.iter().map(|j| g.tensor(&format!("t{j}"))).collect() };
        let out = g.tensor(&format!("t{i}"));
        g.tensors[out as usize].shape = Some(vec![8]);
        g.nodes.push(Node { name: format!("n{i}"), op: op.to_string(), inputs: ins, outputs: vec![out], ..Default::default() });
    }
    g.finish();
    g
}

/// The structure links of each source op, given the runtime's explicit (kernel, op) links. Compiled tensors are
/// renamed so that no kernel boundary carries a source tensor name: only structure can place ops.
fn inferred(src: Graph, mut cmp: Graph, named: &[(u32, u32)]) -> Vec<Vec<u32>> {
    for t in &mut cmp.tensors {
        t.name = format!("kernel {}", t.name);
    }
    let links = named.iter().map(|&(compiled, source)| Link { compiled, source, grade: Grade::Explicit, via: Via::Name }).collect();
    let s = Session::from_model(Model { source: src, compiled: Some(cmp), links, ..Default::default() });
    s.links()[0][0].iter().map(|l| l.iter().filter(|l| l.via == Via::Structure).map(|l| l.other).collect()).collect()
}

/// A kernel whose input and output tensors kept their source names computed exactly the ops between them.
#[test]
fn named_kernel_boundaries_place_ops() {
    // source x -> a -> b -> c -> d; one nameless kernel reads x and writes c's output, a second one writes d's
    let src = dag(&[("Op", &[]), ("Op", &[0]), ("Op", &[1]), ("Op", &[2])]);
    let mut cmp = dag(&[("K", &[]), ("K", &[0])]);
    let out = cmp.nodes[0].outputs[0] as usize;
    cmp.tensors[out].name = "t2".into();
    let s = Session::from_model(Model { source: src, compiled: Some(cmp), ..Default::default() });
    let l = &s.links()[0][0];
    for v in 0..3 {
        assert_eq!(l[v].iter().map(|l| (l.other, l.via)).collect::<Vec<_>>(), [(0, Via::Boundary)], "op {v}: {l:?}");
    }
    // the second kernel's output kept no source name: nothing is derived for it
    assert!(l[3].iter().all(|l| l.via != Via::Boundary), "{l:?}");
    // two kernels writing one tensor (TensorRT's in-place concat): neither owns it, nothing is derived
    let src = dag(&[("Op", &[]), ("Op", &[0]), ("Op", &[1]), ("Op", &[2])]);
    let mut cmp = dag(&[("K", &[]), ("K", &[0])]);
    let shared = cmp.nodes[0].outputs[0];
    cmp.tensors[shared as usize].name = "t2".into();
    cmp.nodes[1].outputs = vec![shared];
    let s = Session::from_model(Model { source: src, compiled: Some(cmp), ..Default::default() });
    assert!(s.links()[0][0].iter().flatten().all(|l| l.via != Via::Boundary), "{:?}", s.links()[0][0]);
}

/// A kernel that only moves data is not credited with a compute op.
#[test]
fn inserted_kernels_get_no_compute_ops() {
    let src = dag(&[("Conv", &[]), ("Relu", &[0]), ("Conv", &[1])]);
    // the Relu's kernel lost its name; the runtime inserted a Reorder in front of it
    let cmp = dag(&[("Conv", &[]), ("Reorder", &[0]), ("Eltwise", &[1]), ("Conv", &[2])]);
    assert_eq!(inferred(src, cmp, &[(0, 0), (3, 2)])[1], [2]);
}

/// In a cluster of nameless kernels, dataflow decides which of them can have computed each op.
#[test]
fn cluster_ops_are_narrowed_by_dataflow() {
    // source a -> b, b -> c -> e and b -> d -> f; kernels: 0 = a, 1 -> 2 -> 4 = e, 1 -> 3 -> 5 = f
    let src = dag(&[("Op", &[]), ("Op", &[0]), ("Op", &[1]), ("Op", &[1]), ("Op", &[2]), ("Op", &[3])]);
    let cmp = dag(&[("K", &[]), ("K", &[0]), ("K", &[1]), ("K", &[1]), ("K", &[2]), ("K", &[3])]);
    let l = inferred(src, cmp, &[(0, 0), (4, 4), (5, 5)]);
    // b feeds both branches: only kernel 1 reaches both; c cannot be on d's branch, nor d on c's
    assert_eq!(l[1], [1]);
    assert!(!l[2].contains(&3) && !l[3].contains(&2), "{l:?}");
}

/// A cluster whose region spans a smaller cluster's region does not take that cluster's ops.
#[test]
fn tightest_regions_are_assigned_first() {
    // source a -> b -> c -> d -> e; kernels 0 = a, 4 = c, 5 = e; nameless 1 -> 2 runs from 0 to 5 and nameless
    // 3 sits between 0 and 4, so b, between a and c, belongs to 3
    let src = dag(&[("Op", &[]), ("Op", &[0]), ("Op", &[1]), ("Op", &[2]), ("Op", &[3])]);
    let cmp = dag(&[("K", &[]), ("K", &[0]), ("K", &[1]), ("K", &[0]), ("K", &[3]), ("K", &[2, 4])]);
    let l = inferred(src, cmp, &[(0, 0), (4, 2), (5, 4)]);
    assert_eq!(l[1], [3]);
    assert!(!l[3].is_empty() && !l[3].contains(&3), "{l:?}");
}

#[test]
fn reveal_opens_a_collapsed_stack() {
    let g = toy_onnx();
    let v = g.nodes.iter().position(|n| n.name == "/model/layers.1/fc/Add").unwrap() as u32;
    let mut s = Session::from_model(Model { source: g, ..Default::default() });
    let b = s.reveal(0, v, false);
    assert!(b >= 0);
    let p = s.pane(0);
    assert_eq!(p.box_nodes(b as u32), [v]);
    assert!(p.scene.titles.iter().any(|t| t.contains("unit 2/2")), "{:?}", p.scene.titles);
}

#[test]
fn two_runtimes_compare_through_the_source() {
    let mut s = toy_session("toy.openvino-cpu.afterpass.json.gz");
    s.add_compiled(toy_session("toy.tensorrt-gpu.afterpass.json.gz")).unwrap();
    assert_eq!(s.links().len(), 2);
    // every linked kernel of A reaches kernels of B through the source ops they share
    let a = s.pane(1);
    let k = (0..a.graph.nodes.len() as u32).find(|&k| !s.links()[0][1][k as usize].is_empty() && data_path(a, k as usize)).unwrap();
    assert!(!linked_nodes(s.links(), 1, &[k], 2).is_empty());
    let f = s.findings_json();
    let sections: Vec<&str> = f.iter().map(|x| x["section"].as_str().unwrap()).collect();
    assert_eq!(sections[0], "summary");
    assert!(sections.contains(&"delta"));
    assert!(f.iter().any(|x| x["title"].as_str().unwrap().starts_with("B  ")));
    // a capture of another source is refused
    let mut other = toy_session("toy.openvino-cpu.afterpass.json.gz");
    let onnx_only = Session::from_model(Model { source: toy_onnx(), ..Default::default() });
    assert!(other.add_compiled(onnx_only).is_err());
}

/// A model opened on its own is summarised: format, weights counted from shapes, inputs, outputs, ops.
#[test]
fn plain_models_are_summarised() {
    for g in [toy_onnx(), ovir::parse(&fixture("toy.xml")).unwrap()] {
        let format = g.meta[0].1.clone();
        let s = Session::from_model(Model { source: g, ..Default::default() });
        let f = s.findings_json();
        let titles = |section: &str| f.iter().filter(|x| x["section"] == section).map(|x| x["title"].as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert!(format.starts_with("ONNX") || format.starts_with("OpenVINO IR"), "{format}");
        assert!(titles("model").iter().any(|t| t.contains(" weights, ")), "{f:?}");
        let io = titles("io");
        assert!(io.iter().any(|t| t.starts_with("in ")) && io.iter().any(|t| t.starts_with("out ")), "{io:?}");
        assert!(titles("ops").iter().any(|t| t.starts_with("MatMul x")), "{f:?}");
    }
}

/// Damaged or foreign files must come back as errors: a panic would abort the wasm worker.
#[test]
fn bad_inputs_are_errors_not_panics() {
    let onnx_bytes = fixture("toy.onnx");
    let xml = fixture("toy.xml");
    let json = gunzip(&fixture("toy.openvino-cpu.afterpass.json.gz"));
    let open = |kind: &str, b: &[u8]| -> Result<Session, String> {
        let s = match kind {
            "onnx" => Session::from_model(Model { source: onnx::parse(&mut &b[..])?, ..Default::default() }),
            "xml" => Session::from_model(Model { source: ovir::parse(b)?, ..Default::default() }),
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
    // the JSON files of a Hugging Face repo are named for what they are
    let config = bundle::parse(br#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}"#).err().unwrap();
    assert!(config.starts_with("a Hugging Face model config"), "{config}");
    let pipeline = bundle::parse(br#"{"_class_name": "StableDiffusionPipeline", "_diffusers_version": "0.30.0"}"#).err().unwrap();
    assert!(pipeline.starts_with("a diffusers pipeline index"), "{pipeline}");
    assert!(empty.is_empty(), "loaded without nodes: {empty:?}");
}

/// Requests for panes, boxes or nodes that do not exist (stale clicks during a load) are refused, not panics.
#[test]
fn stale_ids_are_refused() {
    let mut s = toy_session("toy.openvino-cpu.afterpass.json.gz");
    for pane in [1, 2, 7] {
        assert_eq!(s.reveal(pane, 100_000, false), -1);
        assert_eq!(s.reveal(pane, 100_000, true), -1);
        assert_eq!(s.toggle(pane, 100_000), -1);
        assert_eq!(s.step(pane, 100_000, 1), -1);
        assert_eq!(s.describe(pane, 100_000), "");
        assert_eq!(s.inspect(pane, 100_000), "null");
    }
}

/// A node too big to buffer (a Constant holding a 65 MiB tensor) keeps its name, op and edges.
#[test]
fn oversized_nodes_keep_their_edges() {
    fn field(out: &mut Vec<u8>, num: u64, payload: &[u8]) {
        for mut v in [num << 3 | 2, payload.len() as u64] {
            while v >= 0x80 {
                out.push(v as u8 | 0x80);
                v >>= 7;
            }
            out.push(v as u8);
        }
        out.extend_from_slice(payload);
    }
    let (mut big, mut relu, mut graph, mut model) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    field(&mut big, 2, b"c");
    field(&mut big, 3, b"big");
    field(&mut big, 4, b"Constant");
    field(&mut big, 5, &vec![0u8; 65 << 20]);
    field(&mut relu, 1, b"c");
    field(&mut relu, 2, b"y");
    field(&mut relu, 4, b"Relu");
    field(&mut graph, 1, &big);
    field(&mut graph, 1, &relu);
    field(&mut model, 7, &graph);
    let g = onnx::parse(&mut &model[..]).unwrap();
    assert_eq!(g.nodes.len(), 2);
    assert_eq!((g.nodes[0].name.as_str(), g.nodes[0].op.as_str()), ("big", "Constant"));
    assert_eq!(g.tensors[g.nodes[1].inputs[0] as usize].producer, 0);
}

/// Stepping to a unit shown before uses its new expansion, not the cached layout from last time.
#[test]
fn stepping_to_a_visited_unit_shows_the_carried_expansion() {
    let mut s = Session::from_model(Model { source: toy_onnx(), ..Default::default() });
    let find = |s: &Session, f: &dyn Fn(&String) -> bool| s.pane(0).scene.titles.iter().position(f).unwrap() as u32;
    let stack = |t: &String| t.starts_with("layers.0..1");
    s.toggle(0, find(&s, &stack));
    s.step(0, find(&s, &stack), 1); // visit unit 2, so its layout is cached
    s.step(0, find(&s, &stack), -1);
    assert!(s.toggle(0, find(&s, &|t| t == "fc")) >= 0, "expand fc in unit 1");
    let open = s.pane(0).scene.titles.len();
    s.step(0, find(&s, &stack), 1);
    assert!(s.pane(0).scene.titles.iter().any(|t| t.contains("unit 2/2")));
    assert_eq!(s.pane(0).scene.titles.len(), open, "fc should be expanded in unit 2 too");
}
