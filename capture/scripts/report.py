"""Summarise bundles as the M0 markdown table. Every number is read from the bundle files.

usage: uv run python scripts/report.py <bundle> [<bundle> ...]
"""

import sys
from pathlib import Path

from afterpass import bundle

IO = ("Input", "Output", "Parameter", "Result")  # OpenVINO CPU says Input/Output; GPU and NPU say Parameter/Result


def kernels(nodes: list, topology: bool) -> set:
    """Compiled nodes on the data path: not inputs or outputs, and not computed from constants only.
    Without topology (OpenVINO NPU) every non-IO node counts."""
    producer = {t: n["id"] for n in nodes for t in n["outputs"]}
    const = set()
    for n in sorted(nodes, key=lambda n: n["id"]):  # ids are in topological order for both runtimes
        srcs = [producer.get(t) for t in n["inputs"] if t]
        if n["op"] in ("Const", "Constant") or (topology and n["op"] not in IO and all(s in const for s in srcs)):
            const.add(n["id"])
    return {n["id"] for n in nodes if n["op"] not in IO and n["id"] not in const}


def short(name: str) -> str:
    for junk in ("(R)", "(TM)", "12th Gen ", "Intel ", "NVIDIA GeForce ", " Laptop GPU", " (16GB) (iGPU)"):
        name = name.replace(junk, "")
    return " ".join(name.split())


def row(path: Path) -> list:
    b = bundle.read(path)
    kernels_ = kernels(b["compiled"]["nodes"], b["compiled"].get("topology", True))
    best = {}
    for link in b["provenance"]:
        if link["compiled"] in kernels_:
            rank = {"explicit": 0, "inferred": 1, "unknown": 2}[link["grade"]]
            best[link["compiled"]] = min(best.get(link["compiled"], 2), rank)
    share = lambda r: 100.0 * sum(1 for v in best.values() if v == r) / max(len(kernels_), 1)
    linked = {l["source"] for l in b["provenance"] if l["source"] is not None}
    src_path = kernels(b["source"]["nodes"], True)  # source ops on the data path (constant-only subgraphs excluded)
    src = [n for n in b["source"]["nodes"] if n["id"] in src_path]
    cap = b["capture"]
    return [
        cap["model"], f"{cap['runtime']} {cap['device']}", short(b["compiled"]["device_name"]), len(b["source"]["nodes"]), len(b["compiled"]["nodes"]),
        len(kernels_), f"{share(0):.1f}%", f"{share(1):.1f}%", f"{share(2):.1f}%",
        f"{100.0 * sum(1 for n in src if n['id'] in linked) / max(len(src), 1):.1f}%",
        f"{b['profile']['latency']['median'] / 1000:.2f}", f"{cap['compile_s']:.1f}", f"{cap['capture_s']:.1f}",
        f"{path.stat().st_size / 1024:.0f}",
    ]


HEADER = ["model", "runtime", "device name", "source nodes", "compiled nodes", "kernels", "explicit", "inferred", "unknown",
          "source covered", "latency ms", "compile s", "capture s", "bundle KiB"]

if __name__ == "__main__":
    print("| " + " | ".join(HEADER) + " |")
    print("|" + "---|" * len(HEADER))
    for p in sys.argv[1:]:
        print("| " + " | ".join(str(c) for c in row(Path(p))) + " |")
