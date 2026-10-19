//! Structure and timing on the real corpus (skipped when corpus/ is absent).
//! cargo test --release --test corpus -- --nocapture

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::time::Instant;

use afterpass_core::analysis::data_path;
use afterpass_core::ir::{Grade, Link, Model, Via};
use afterpass_core::provenance::Linker;
use afterpass_core::{bundle, onnx, ovir, tflite, Session};

fn corpus(rel: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../corpus").join(rel);
    p.exists().then_some(p)
}

struct FileSource(std::fs::File, u64);

impl onnx::ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.1
    }
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), String> {
        use std::os::unix::fs::FileExt;
        self.0.read_exact_at(buf, offset).map_err(|e| e.to_string())
    }
}

/// Bundles of the given corpus directories, sorted, as (label, decompressed JSON).
fn bundles_in(dirs: &[&str]) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for dir in dirs {
        let Some(d) = corpus(dir) else { continue };
        let mut files: Vec<_> = std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        out.extend(files.into_iter().map(|f| (format!("{dir}/{}", f.file_name().unwrap().to_str().unwrap()), gunzip(&f))));
    }
    out
}

fn gunzip(path: &std::path::Path) -> Vec<u8> {
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap()).read_to_end(&mut json).unwrap();
    json
}

fn report(name: &str, t0: Instant, s: &Session) {
    print!("{name}: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    for p in 0..s.panes() {
        let pane = s.pane(p);
        let h = &pane.hier;
        let stacks: Vec<String> = h.stacks.iter().enumerate()
            .map(|(i, st)| format!("{} x{}{}", h.stack_label(i as u32), st.count, if st.period > 1 { format!(" (period {})", st.period) } else { String::new() }))
            .collect();
        let outliers = h.groups.iter().filter(|g| g.outlier && g.parent != u32::MAX).count();
        print!("\n  pane {p}: {} nodes, {} folded, {} boxes, stacks [{}], {outliers} outliers",
            pane.graph.nodes.len(), h.folded.iter().filter(|&&f| f).count(), pane.scene.kinds.len(), stacks.join(", "));
    }
    println!();
}

#[test]
fn ir_30b() {
    for rel in ["30b/EmbeddedLLM__Qwen2.5-32B-Instruct-int4-sym-ov/openvino_model.xml",
                "30b/OpenVINO__Qwen3.8-27B-int4-ov/openvino_language_model.xml",
                "30b/OpenVINO__Qwen3-30B-A3B-int4-ov/openvino_model.xml"] {
        let Some(p) = corpus(rel) else { continue };
        let t0 = Instant::now();
        let g = ovir::parse(&std::fs::read(p).unwrap()).unwrap();
        let s = Session::from_model(Model { source: g, ..Default::default() });
        report(rel, t0, &s);
    }
}

#[test]
fn onnx_files() {
    for rel in ["bert-base/bert-base.onnx", "tinyllama-1.1b/tinyllama-1.1b.onnx", "yolo11n/yolo11n.onnx", "resnet50/resnet50.onnx",
                "synthetic/moe-48x128.onnx"] {
        let Some(p) = corpus(rel) else { continue };
        let t0 = Instant::now();
        let f = std::fs::File::open(p).unwrap();
        let len = f.metadata().unwrap().len();
        let g = onnx::parse(&mut FileSource(f, len)).unwrap();
        let s = Session::from_model(Model { source: g, ..Default::default() });
        report(rel, t0, &s);
    }
}

#[test]
fn tflite_files() {
    for rel in ["tflite/efficientnet_lite0/efficientnet_lite0.tflite", "tflite/mobilenet_v1_quant/mobilenet_v1_1.0_224_quant.tflite",
                "tflite/bert_classifier/bert_classifier.tflite"] {
        let Some(p) = corpus(rel) else { continue };
        let t0 = Instant::now();
        let f = std::fs::File::open(p).unwrap();
        let len = f.metadata().unwrap().len();
        let g = tflite::parse(&mut FileSource(f, len)).unwrap();
        println!("{:?}", g.meta);
        let s = Session::from_model(Model { source: g, ..Default::default() });
        report(rel, t0, &s);
    }
}

/// Load time, collapsing, provenance coverage and findings per bundle.
#[test]
fn bundles() {
    for (name, json) in bundles_in(&["bundles", "bundles-lnl", "bundles-30b"]) {
        let t0 = Instant::now();
        let s = Session::from_model(bundle::parse(&json).unwrap());
        report(&name, t0, &s);
        let src = s.pane(0);
        let path: Vec<usize> = (0..src.graph.nodes.len()).filter(|&v| data_path(src, v)).collect();
        let by = |structure: bool| path.iter().filter(|&&v| s.links()[0][0][v].iter().any(|l| structure || l.via != Via::Structure)).count();
        let findings = s.findings_json();
        let mut sections = BTreeMap::new();
        for x in &findings {
            *sections.entry(x["section"].as_str().unwrap().to_string()).or_insert(0) += 1;
        }
        println!("  source covered {:.1}% by name, {:.1}% with structure; findings {:?}",
            100.0 * by(false) as f64 / path.len() as f64,
            100.0 * by(true) as f64 / path.len() as f64, sections);
        if name.starts_with("bundles-30b") {
            for x in findings.iter().filter(|x| x["section"] == "time") {
                println!("  time: {} | {}", x["title"].as_str().unwrap(), x["detail"].as_str().unwrap());
            }
        }
    }
}

#[test]
fn compare_pairs() {
    let pairs = [
        ("bundles-lnl/tinyllama-1.1b.openvino-cpu", "bundles-lnl/tinyllama-1.1b.openvino-gpu"),
        ("bundles/tinyllama-1.1b.tensorrt-gpu", "bundles-lnl/tinyllama-1.1b.openvino-gpu"),
        ("bundles-ov2025/tinyllama-1.1b.openvino-cpu", "bundles-ov2026/tinyllama-1.1b.openvino-cpu"),
        ("bundles/yolo11n.tensorrt-gpu", "bundles-lnl/yolo11n.openvino-gpu"),
        ("bundles/resnet50-int8.tensorrt-gpu", "bundles/resnet50-int8.openvino-cpu"),
        ("bundles-trt11/resnet50-int8.tensorrt-gpu", "bundles-lnl/resnet50-int8.openvino-gpu"),
        ("bundles-trt10/tinyllama-1.1b.tensorrt-gpu", "bundles-trt11/tinyllama-1.1b.tensorrt-gpu"),
        ("bundles-trt10/yolo11n.tensorrt-gpu", "bundles-trt11/yolo11n.tensorrt-gpu"),
        ("bundles-trt10/resnet50-int8.tensorrt-gpu", "bundles-trt11/resnet50-int8.tensorrt-gpu"),
    ];
    let load = |f: &str| corpus(&format!("{f}.afterpass.json.gz")).map(|p| Session::from_model(bundle::parse(&gunzip(&p)).unwrap()));
    for (a, b) in pairs {
        let (Some(mut s), Some(other)) = (load(a), load(b)) else { continue };
        let t0 = Instant::now();
        s.add_compiled(other).unwrap();
        let f = s.findings_json();
        println!("== {a} vs {b}: findings in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        for x in f.iter().filter(|x| !["time", "inserted", "precision", "unaccounted", "outliers"].contains(&x["section"].as_str().unwrap())) {
            println!("  [{}] {} | {} | {} targets", x["section"].as_str().unwrap(), x["title"].as_str().unwrap(), x["detail"].as_str().unwrap(), x["targets"].as_array().unwrap().len());
        }
    }
}

/// Accuracy of structural inference against a known answer: hide the explicit links of an eighth of the kernels,
/// let inference recover them, and score the source ops that lost every explicit link.
/// Recall: such ops linked to a kernel that really claimed them. Precision: inferred links that are right.
#[test]
fn structure_accuracy() {
    for (name, json) in bundles_in(&["bundles", "bundles-lnl", "bundles-30b"]) {
        let (mut lost, mut found, mut right, mut predicted, mut among_hidden) = (0, 0, 0, 0, 0);
        for seed in 1..=5u64 {
            let mut m = bundle::parse(&json).unwrap();
            let truth = std::mem::take(&mut m.links);
            let mut kernels: Vec<u32> = truth.iter().filter(|l| l.grade == Grade::Explicit).map(|l| l.compiled).collect();
            kernels.sort_unstable();
            kernels.dedup();
            // a fixed hash picks the hidden kernels, so runs are repeatable
            let hidden: HashSet<u32> = kernels.into_iter()
                .filter(|&k| (k as u64 ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 61 == 0).collect();
            m.links = truth.iter().filter(|l| !hidden.contains(&l.compiled)).cloned().collect();
            let kept: HashSet<u32> = m.links.iter().map(|l| l.source).collect();
            let mut want: HashMap<u32, HashSet<u32>> = HashMap::new();
            for l in truth.iter().filter(|l| hidden.contains(&l.compiled) && !kept.contains(&l.source)) {
                want.entry(l.source).or_default().insert(l.compiled);
            }
            let s = Session::from_model(m);
            let src = s.pane(0);
            for (&v, t) in want.iter().filter(|(v, _)| data_path(src, **v as usize)) {
                let got: Vec<u32> = s.links()[0][0][v as usize].iter().map(|l| l.other).collect();
                lost += 1;
                found += got.iter().any(|k| t.contains(k)) as usize;
                right += got.iter().filter(|k| t.contains(k)).count();
                predicted += got.len();
                among_hidden += got.iter().filter(|k| !t.contains(k) && hidden.contains(k)).count();
            }
        }
        if lost > 0 {
            println!("{name}: {lost} ops lost their links over 5 runs; recall {:.1}%, precision {:.1}% ({right} of {predicted}); wrong links to other hidden kernels {among_hidden}, to other kernels (named or inserted) {}",
                100.0 * found as f64 / lost as f64,
                100.0 * right as f64 / predicted.max(1) as f64, predicted - right - among_hidden);
        }
    }
}

/// The Rust Linker against the capture's: relink every bundle's compiled claims and compare with its provenance.
#[test]
fn linker_matches_capture() {
    let dirs = ["bundles", "bundles-lnl", "bundles-30b", "bundles-m7", "bundles-ort", "bundles-ov2025", "bundles-ov2026",
        "bundles-trt10", "bundles-trt11"];
    for (name, json) in bundles_in(&dirs) {
        let b: serde_json::Value = serde_json::from_slice(&json).unwrap();
        let m = bundle::parse(&json).unwrap();
        let linker = Linker::new(&m.source, b["source"]["format"] == "onnx");
        let key = |l: &Link| (l.compiled, l.source, l.grade, l.via.as_str());
        let mut ours: Vec<_> = m.compiled.as_ref().unwrap().nodes.iter().enumerate()
            .flat_map(|(i, n)| linker.link(i as u32, &n.claims)).map(|l| key(&l)).collect();
        let mut theirs: Vec<_> = m.links.iter().map(key).collect();
        ours.sort_unstable();
        theirs.sort_unstable();
        let differ = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count() + ours.len().abs_diff(theirs.len());
        println!("{name}: {} links, {differ} differ", theirs.len());
        assert_eq!(differ, 0, "{name}");
    }
}

/// benchmark_app's execution graph and report against a capture of the same model on the same machine
/// (corpus/import, written by the commands in docs/notes/18-import.md).
#[test]
fn exec_graph_import_vs_capture() {
    for (model, dir) in [("resnet50/resnet50.onnx", "import/resnet50-cpu"), ("resnet50/resnet50.onnx", "import/resnet50-cpu-f32"),
                         ("tinyllama-1.1b/tinyllama-1.1b.onnx", "import/tinyllama-cpu")] {
        let (Some(src), Some(d)) = (corpus(model), corpus(dir)) else { continue };
        let f = std::fs::File::open(&src).unwrap();
        let len = f.metadata().unwrap().len();
        let g = onnx::parse(&mut FileSource(f, len)).unwrap();
        let mut imported = Session::from_model(Model { source: g, ..Default::default() });
        let report = std::fs::read(d.join("benchmark_report.csv")).unwrap();
        imported.add_exec_graph(&std::fs::read(d.join("exec.xml")).unwrap(), Some(&report), "model").unwrap();
        let bundle_file = std::fs::read_dir(&d).unwrap().flatten().map(|e| e.path())
            .find(|p| p.to_string_lossy().ends_with(".afterpass.json.gz")).unwrap();
        let captured = Session::from_model(bundle::parse(&gunzip(&bundle_file)).unwrap());
        println!("== {model}");
        for (what, s) in [("imported", &imported), ("captured", &captured)] {
            let c = &s.pane(1).graph;
            let mut grades = BTreeMap::new();
            for l in s.links()[0][1].iter().flatten() {
                *grades.entry(format!("{:?}/{}", l.grade, l.via.as_str())).or_insert(0) += 1;
            }
            let unknown = (0..c.nodes.len()).filter(|&k| s.links()[0][1][k].is_empty()).count();
            let timed: Vec<f32> = c.nodes.iter().filter_map(|n| n.time_us).collect();
            let side = s.side(1).unwrap();
            println!("  {what}: {} source, {} compiled nodes, {} timed (sum {:.0} us), latency {:?} us, range {:?}; links {grades:?}, {unknown} kernels unlinked; about {}",
                s.pane(0).graph.nodes.len(), c.nodes.len(), timed.len(), timed.iter().sum::<f32>(), side.latency_us, side.latency_range, s.about(1));
        }
        let titles = |s: &Session, timed: bool| s.findings_json().iter().map(|x| {
            let title = x["title"].as_str().unwrap();
            // a time finding's title ends with the measured time, after two spaces
            format!("[{}] {}", x["section"].as_str().unwrap(), if timed { title } else { title.split("  ").next().unwrap() })
        }).collect::<BTreeSet<_>>();
        let same = titles(&imported, true).intersection(&titles(&captured, true)).count();
        let (a, b) = (titles(&imported, false), titles(&captured, false));
        println!("  findings: {} imported, {} captured, {same} the same; without times {} and {}, {} the same",
            imported.findings_json().len(), captured.findings_json().len(), a.len(), b.len(), a.intersection(&b).count());
        for x in a.difference(&b) {
            println!("  only imported: {x}");
        }
        for x in b.difference(&a) {
            println!("  only captured: {x}");
        }
    }
}
