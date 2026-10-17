"""Download only the .xml topology of a Hugging Face OpenVINO IR and count its nodes. The .bin is never fetched.

usage: python scripts/ir_topology.py <corpus_dir> <repo> <xml_file> [<repo> <xml_file> ...]
"""

import collections
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def fetch(repo: str, name: str, dst: Path) -> Path:
    path = dst / repo.replace("/", "__") / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{name}") as r:
            path.write_bytes(r.read())
    return path


def count(path: Path) -> dict:
    net = ET.parse(path).getroot()
    top = net.find("layers").findall("layer")
    ops = collections.Counter(layer.get("type") for layer in top)
    nested = sum(1 for layer in top for _ in layer.iter("layer")) - len(top)
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "ir_version": net.get("version"),
        "layers": len(top),
        "layers_without_constants": len(top) - ops["Const"],
        "edges": len(net.find("edges").findall("edge")),
        "nested_body_layers": nested,
        "top_ops": ops.most_common(12),
    }


if __name__ == "__main__":
    corpus, rest = Path(sys.argv[1]), sys.argv[2:]
    for repo, name in zip(rest[::2], rest[1::2]):
        stats = {"repo": repo, **count(fetch(repo, name, corpus / "30b"))}
        print(json.dumps(stats))
