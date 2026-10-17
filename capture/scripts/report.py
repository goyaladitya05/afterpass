"""Summarise bundles as the M0 markdown table. Every number is read from the bundle files.

usage: uv run python scripts/report.py <bundle> [<bundle> ...]
"""

import sys
from pathlib import Path

from afterpass import bundle

NON_KERNEL = {"Input", "Output", "Const"}  # OpenVINO parameter, result and constant nodes


def row(path: Path) -> list:
    b = bundle.read(path)
    kernels = {n["id"] for n in b["compiled"]["nodes"] if n["op"] not in NON_KERNEL}
    best = {}
    for link in b["provenance"]:
        if link["compiled"] in kernels:
            rank = {"explicit": 0, "inferred": 1, "unknown": 2}[link["grade"]]
            best[link["compiled"]] = min(best.get(link["compiled"], 2), rank)
    share = lambda r: 100.0 * sum(1 for v in best.values() if v == r) / max(len(kernels), 1)
    linked = {l["source"] for l in b["provenance"] if l["source"] is not None}
    src = [n for n in b["source"]["nodes"] if n["op"] != "Constant"]
    cap = b["capture"]
    return [
        cap["model"], f"{cap['runtime']} {cap['device']}", len(b["source"]["nodes"]), len(b["compiled"]["nodes"]),
        len(kernels), f"{share(0):.1f}%", f"{share(1):.1f}%", f"{share(2):.1f}%",
        f"{100.0 * sum(1 for n in src if n['id'] in linked) / max(len(src), 1):.1f}%",
        f"{b['profile']['latency']['median'] / 1000:.2f}", f"{cap['compile_s']:.1f}", f"{cap['capture_s']:.1f}",
        f"{path.stat().st_size / 1024:.0f}",
    ]


HEADER = ["model", "runtime", "source nodes", "compiled nodes", "kernels", "explicit", "inferred", "unknown",
          "source covered", "latency ms", "compile s", "capture s", "bundle KiB"]

if __name__ == "__main__":
    print("| " + " | ".join(HEADER) + " |")
    print("|" + "---|" * len(HEADER))
    for p in sys.argv[1:]:
        print("| " + " | ".join(str(c) for c in row(Path(p))) + " |")
