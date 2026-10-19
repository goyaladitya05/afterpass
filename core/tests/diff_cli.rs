//! Runs the built afterpass-diff binary on the toy fixtures and on small edits of them.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::{json, Value};

const OV: &str = "toy.openvino-cpu.afterpass.json.gz";
const TRT: &str = "toy.tensorrt-gpu.afterpass.json.gz";

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn diff(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_afterpass-diff")).args(args).output().unwrap()
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).unwrap()
}

/// The openvino fixture as JSON, edited by `f`, written as plain JSON to a temp file.
fn edited(name: &str, f: impl FnOnce(&mut Value)) -> String {
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(fixture(OV)).unwrap()).read_to_end(&mut json).unwrap();
    let mut v: Value = serde_json::from_slice(&json).unwrap();
    f(&mut v);
    let dir: PathBuf = std::env::temp_dir().join(format!("afterpass-diff-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, v.to_string()).unwrap();
    path.to_str().unwrap().to_string()
}

fn latency(median: f64, min: f64) -> impl FnOnce(&mut Value) {
    move |v| {
        v["profile"]["latency"]["median"] = json!(median);
        v["profile"]["latency"]["min"] = json!(min);
    }
}

#[test]
fn text_report_has_the_comparison_sections() {
    let o = diff(&[&fixture(OV), &fixture(TRT)]);
    assert_eq!(code(&o), 0);
    let out = stdout(&o);
    for want in ["\nsummary\n", "latency  A 107 us, B 23 us", "\nwhere the time differs\n", "model.layers.#.fc x3"] {
        assert!(out.contains(want), "missing {want:?} in:\n{out}");
    }
}

#[test]
fn json_output_parses() {
    let o = diff(&["--json", &fixture(OV), &fixture(TRT)]);
    assert_eq!(code(&o), 0);
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(v["findings"].as_array().unwrap().iter().any(|f| f["section"] == "summary"));
    assert_eq!(v["lost_fusions"], 0);
    assert_eq!(v["failed"], json!([]));
    assert!(v["latency"]["b_slower_pct"].as_f64().unwrap() < 0.0);
}

/// The fixture median is 107 us (runs 92 to 285). 300 us with a fastest run of 250 is a real slowdown.
#[test]
fn latency_gate_fails_on_a_real_slowdown() {
    let slow = edited("slow.afterpass.json", latency(300.0, 250.0));
    let o = diff(&["--max-slower", "10", &fixture(OV), &slow]);
    assert_eq!(code(&o), 1, "{}", stdout(&o));
    assert!(stdout(&o).contains("latency: FAIL"));
    // no gate requested, no failure
    assert_eq!(code(&diff(&[&fixture(OV), &slow])), 0);
    // inside the limit
    assert_eq!(code(&diff(&["--max-slower", "300", &fixture(OV), &slow])), 0);
}

#[test]
fn latency_gate_ignores_run_to_run_noise() {
    // 30% slower median, but B's fastest run (90 us) is faster than A's median (107 us)
    let noisy = edited("noisy.afterpass.json", latency(140.0, 90.0));
    let o = diff(&["--max-slower", "10", &fixture(OV), &noisy]);
    assert_eq!(code(&o), 0, "{}", stdout(&o));
    assert!(stdout(&o).contains("within run-to-run noise"));
}

/// A had fused the first layer's Add into its FullyConnected kernel; B (the original) runs it as a separate kernel.
fn fused(v: &mut Value) {
    for p in v["provenance"].as_array_mut().unwrap() {
        if p["compiled"] == 10 {
            p["compiled"] = json!(9);
        }
    }
}

#[test]
fn fusion_gate_fails_when_b_lost_a_fusion() {
    let a = edited("fused.afterpass.json", fused);
    let o = diff(&["--fail-on-lost-fusion", &a, &fixture(OV)]);
    assert_eq!(code(&o), 1, "{}", stdout(&o));
    assert!(stdout(&o).contains("fusion: FAIL"));
    // the other way round B gained a fusion, which is not a regression
    assert_eq!(code(&diff(&["--fail-on-lost-fusion", &fixture(OV), &a])), 0);
    assert_eq!(code(&diff(&["--fail-on-lost-fusion", &fixture(OV), &fixture(OV)])), 0);
}

#[test]
fn bad_input_exits_2_naming_the_file() {
    let missing = diff(&[&fixture(OV), "no-such-file.json.gz"]);
    assert_eq!(code(&missing), 2);
    assert!(stderr(&missing).contains("no-such-file.json.gz"));
    assert_eq!(stderr(&missing).lines().count(), 1);

    let junk = edited("junk.afterpass.json", |v| *v = json!({ "format": "not-a-bundle" }));
    let o = diff(&[&junk, &fixture(OV)]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("junk.afterpass.json"));

    assert_eq!(code(&diff(&[&fixture(OV)])), 2);
    assert_eq!(code(&diff(&["--bogus", &fixture(OV), &fixture(TRT)])), 2);
    assert_eq!(code(&diff(&["--max-slower", "fast", &fixture(OV), &fixture(TRT)])), 2);
}

#[test]
fn help_lists_the_options() {
    let o = diff(&["--help"]);
    assert_eq!(code(&o), 0);
    for opt in ["--json", "--max-slower", "--fail-on-lost-fusion"] {
        assert!(stdout(&o).contains(opt));
    }
}
