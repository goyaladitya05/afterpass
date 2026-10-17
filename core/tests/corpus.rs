//! Structure and timing on the real corpus (skipped when corpus/ is absent).
//! cargo test --release --test corpus -- --nocapture

use std::io::Read;
use std::time::Instant;

use afterpass_core::ir::Model;
use afterpass_core::{bundle, onnx, ovir, Session};

fn corpus(rel: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../corpus").join(rel);
    p.exists().then_some(p)
}

struct FileSource(std::fs::File, u64);

impl onnx::ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.1
    }
    fn read(&mut self, offset: u64, buf: &mut [u8]) {
        use std::os::unix::fs::FileExt;
        self.0.read_exact_at(buf, offset).unwrap();
    }
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
    for rel in ["bert-base/bert-base.onnx", "tinyllama-1.1b/tinyllama-1.1b.onnx", "yolo11n/yolo11n.onnx", "resnet50/resnet50.onnx"] {
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
fn bundles() {
    for dir in ["bundles", "bundles-lnl"] {
        let Some(d) = corpus(dir) else { continue };
        let mut files: Vec<_> = std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        for f in files {
            let t0 = Instant::now();
            let mut json = Vec::new();
            flate2::read::GzDecoder::new(std::fs::File::open(&f).unwrap()).read_to_end(&mut json).unwrap();
            let s = Session::from_model(bundle::parse(&json).unwrap());
            report(f.file_name().unwrap().to_str().unwrap(), t0, &s);
        }
    }
}

#[test]
fn synthetic_moe() {
    let Some(p) = corpus("synthetic/moe-48x128.onnx") else { return };
    let t0 = Instant::now();
    let f = std::fs::File::open(p).unwrap();
    let len = f.metadata().unwrap().len();
    let g = onnx::parse(&mut FileSource(f, len)).unwrap();
    let s = Session::from_model(Model { source: g, ..Default::default() });
    report("synthetic/moe-48x128.onnx", t0, &s);
}

#[test]
fn bundles_30b() {
    let Some(d) = corpus("bundles-30b") else { return };
    for f in std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()) {
        let t0 = Instant::now();
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(std::fs::File::open(&f).unwrap()).read_to_end(&mut json).unwrap();
        let s = Session::from_model(bundle::parse(&json).unwrap());
        report(f.file_name().unwrap().to_str().unwrap(), t0, &s);
        for x in s.findings_json().iter().filter(|x| x["section"] == "time") {
            println!("  time: {} | {}", x["title"].as_str().unwrap(), x["detail"].as_str().unwrap());
        }
    }
}

#[test]
fn structural_coverage() {
    for dir in ["bundles", "bundles-lnl", "bundles-30b"] {
        let Some(d) = corpus(dir) else { continue };
        let mut files: Vec<_> = std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        for f in files {
            let mut json = Vec::new();
            flate2::read::GzDecoder::new(std::fs::File::open(&f).unwrap()).read_to_end(&mut json).unwrap();
            let s = Session::from_model(bundle::parse(&json).unwrap());
            let src = s.pane(0);
            let path: Vec<usize> = (0..src.graph.nodes.len()).filter(|&v| afterpass_core::analysis::data_path(src, v)).collect();
            let by = |structure: bool| path.iter().filter(|&&v| s.links()[0][0][v].iter().any(|l| structure || l.via != afterpass_core::ir::Via::Structure)).count();
            let findings = s.findings_json();
            let mut sections = std::collections::BTreeMap::new();
            for x in &findings {
                *sections.entry(x["section"].as_str().unwrap().to_string()).or_insert(0) += 1;
            }
            println!("{}: source covered {:.1}% by name, {:.1}% with structure; findings {:?}",
                f.file_name().unwrap().to_str().unwrap(), 100.0 * by(false) as f64 / path.len() as f64,
                100.0 * by(true) as f64 / path.len() as f64, sections);
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
    ];
    let load = |f: &str| {
        let mut json = Vec::new();
        let path = format!("{}/../corpus/{f}.afterpass.json.gz", env!("CARGO_MANIFEST_DIR"));
        flate2::read::GzDecoder::new(std::fs::File::open(path).ok()?).read_to_end(&mut json).unwrap();
        Some(Session::from_model(bundle::parse(&json).unwrap()))
    };
    for (a, b) in pairs {
        let (Some(mut s), Some(other)) = (load(a), load(b)) else { continue };
        let t0 = std::time::Instant::now();
        s.add_compiled(other).unwrap();
        let f = s.findings_json();
        println!("== {a} vs {b}: findings in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        for x in f.iter().filter(|x| !["time", "inserted", "precision", "unaccounted", "outliers"].contains(&x["section"].as_str().unwrap())) {
            println!("  [{}] {} | {} | {} targets", x["section"].as_str().unwrap(), x["title"].as_str().unwrap(), x["detail"].as_str().unwrap(), x["targets"].as_array().unwrap().len());
        }
    }
}
