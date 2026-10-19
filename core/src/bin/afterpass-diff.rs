//! afterpass-diff: compare two bundles of the same model (the viewer's compare mode) and gate CI on the result.

use std::io::Read;
use std::process::ExitCode;

use afterpass_core::analysis::within_noise;
use afterpass_core::{bundle, Session};
use serde_json::{json, Value};

const HELP: &str = "\
afterpass-diff: compare two bundles of the same model

usage: afterpass-diff [options] A.afterpass.json.gz B.afterpass.json.gz

options:
  --json                 print JSON (findings, latency, gate results) instead of text
  --max-slower PCT       fail if B's median latency is more than PCT% above A's, unless the gap is
                         within run-to-run noise (B's fastest run is no slower than A's median)
  --fail-on-lost-fusion  fail if B runs more kernels than A for some region of source ops
  -h, --help             print this help

Files may be gzip or plain JSON. Exit codes: 0 ok, 1 a gate failed, 2 usage or input error.";

/// Section names as the core emits them, with their headings, in report order.
const SECTIONS: [(&str, &str); 5] = [("summary", "summary"), ("delta", "where the time differs"),
    ("fusion", "fused differently"), ("precision-diff", "precision differs"), ("kernel-diff", "kernels changed")];

struct Args {
    files: Vec<String>,
    json: bool,
    max_slower: Option<f64>,
    lost_fusion: bool,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Option<Args>, String> {
    let mut a = Args { files: Vec::new(), json: false, max_slower: None, lost_fusion: false };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--json" => a.json = true,
            "--fail-on-lost-fusion" => a.lost_fusion = true,
            "--max-slower" => {
                let v = args.next().ok_or("--max-slower needs a percentage")?;
                a.max_slower = Some(v.parse::<f64>().ok().filter(|p| *p >= 0.0).ok_or(format!("--max-slower: {v:?} is not a percentage"))?);
            }
            f if f.starts_with('-') && f.len() > 1 => return Err(format!("unknown option {f} (see --help)")),
            _ => a.files.push(arg),
        }
    }
    if a.files.len() != 2 {
        return Err("expected two bundle files, A and B (see --help)".into());
    }
    Ok(Some(a))
}

fn load(path: &str) -> Result<Session, String> {
    let mut bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&bytes[..]).read_to_end(&mut out).map_err(|e| format!("{path}: bad gzip: {e}"))?;
        bytes = out;
    }
    Ok(Session::from_model(bundle::parse(&bytes).map_err(|e| format!("{path}: {e}"))?))
}

fn run() -> Result<ExitCode, String> {
    let Some(args) = parse_args(std::env::args().skip(1))? else {
        println!("{HELP}");
        return Ok(ExitCode::SUCCESS);
    };
    let mut s = load(&args.files[0])?;
    s.add_compiled(load(&args.files[1])?).map_err(|e| format!("{} and {}: {e}", args.files[0], args.files[1]))?;
    let findings = s.findings_json();
    let (sa, sb) = (s.side(1).unwrap(), s.side(2).unwrap());

    let mut failed: Vec<&str> = Vec::new();
    let mut gates: Vec<String> = Vec::new();
    let latency = match (sa.latency_us, sb.latency_us) {
        (Some(la), Some(lb)) => Some((la, lb, f64::from(lb / la - 1.0) * 100.0, within_noise(sb, sa))),
        _ => None,
    };
    if let Some(limit) = args.max_slower {
        let (_, _, pct, noise) = latency.ok_or_else(|| format!("--max-slower needs a median latency in both bundles, {} or {} has none", args.files[0], args.files[1]))?;
        if pct <= limit {
            gates.push(format!("latency: ok B median is {pct:+.1}% vs A (limit {limit}%)"));
        } else if noise {
            gates.push(format!("latency: ok B median is {pct:+.1}% vs A, within run-to-run noise (limit {limit}%)"));
        } else {
            failed.push("latency");
            gates.push(format!("latency: FAIL B median is {pct:+.1}% vs A (limit {limit}%)"));
        }
    }
    let lost = findings.iter().find_map(|f| f["lost_fusions"].as_u64()).unwrap_or(0);
    if args.lost_fusion {
        if lost > 0 {
            failed.push("fusion");
            gates.push(format!("fusion: FAIL B runs more kernels than A in {lost} regions of source ops"));
        } else {
            gates.push("fusion: ok no region runs more kernels in B than in A".into());
        }
    }

    if args.json {
        let lat = latency.map(|(a, b, pct, noise)| json!({ "a_us": a, "b_us": b, "b_slower_pct": pct, "within_noise": noise }));
        let comparison: Vec<&Value> = findings.iter().filter(|f| SECTIONS.iter().any(|(k, _)| f["section"] == *k)).collect();
        println!("{}", json!({ "a": args.files[0], "b": args.files[1], "findings": comparison, "latency": lat, "lost_fusions": lost, "failed": failed }));
    } else {
        println!("A {}\nB {}", args.files[0], args.files[1]);
        for (key, heading) in SECTIONS {
            let items: Vec<&Value> = findings.iter().filter(|f| f["section"] == key).collect();
            if items.is_empty() {
                continue;
            }
            println!("\n{heading}");
            for f in items {
                println!("  {}\n      {}", f["title"].as_str().unwrap_or(""), f["detail"].as_str().unwrap_or(""));
            }
        }
        if !gates.is_empty() {
            println!("\ngates");
            for g in &gates {
                println!("  {g}");
            }
        }
    }
    Ok(if failed.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

fn main() -> ExitCode {
    run().unwrap_or_else(|e| {
        eprintln!("afterpass-diff: {e}");
        ExitCode::from(2)
    })
}
