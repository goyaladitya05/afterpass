"""Summarise a bundle: nodes, link coverage by grade, shape coverage, latency against kernel times, top time sinks.

python scripts/bundle_summary.py BUNDLE [--unknown]   (--unknown lists every compiled node with no link)
"""

import collections
import sys

from afterpass import bundle


def main():
    b = bundle.read(sys.argv[1])
    src, comp, prof = b["source"], b["compiled"], b["profile"]
    nodes = comp["nodes"]
    print(f"{comp['runtime']} {comp['version']} {comp['device']} ({comp['device_name']}), {b['capture']['model']}, compile {b['capture']['compile_s']:.2f} s")
    print(f"source nodes {len(src['nodes'])}, compiled nodes {len(nodes)}")
    print("compiled ops:", collections.Counter(n["op"] for n in nodes).most_common(12))
    print("kernels:", collections.Counter(n["kernel"] for n in nodes).most_common(12))

    best = {}
    for l in b["provenance"]:
        rank = {"explicit": 0, "inferred": 1, "unknown": 2}[l["grade"]]
        best[l["compiled"]] = min(best.get(l["compiled"], 3), rank)
    grades = collections.Counter(("explicit", "inferred", "unknown")[r] for r in best.values())
    print(f"compiled nodes by best link: {dict(grades)}")
    vias = collections.Counter(f"{l['grade']}/{l.get('via', '')}" for l in b["provenance"] if l["source"] is not None)
    print(f"links by grade/via: {dict(vias)}")
    linked = {l["source"] for l in b["provenance"] if l["source"] is not None}
    explicit = {l["source"] for l in b["provenance"] if l["grade"] == "explicit"}
    print(f"source ops with a link: {len(linked)} of {len(src['nodes'])} ({len(explicit)} explicit)")
    unlinked = collections.Counter(n["op"] for n in src["nodes"] if n["id"] not in linked)
    print("source ops without a link, by op:", unlinked.most_common(12))

    tensors = comp["tensors"]
    outs = [t for n in nodes for t in n["outputs"]]
    full = sum(1 for t in outs if tensors[t]["shape"] is not None and all(isinstance(d, int) for d in tensors[t]["shape"]))
    print(f"compiled output tensors with a full shape: {full} of {len(outs)}; with a layout: {sum(1 for t in outs if tensors[t].get('layout'))}")

    lat = prof["latency"]
    times = {int(k): v for k, v in prof["nodes"].items()}
    print(f"latency median {lat['median'] / 1e3:.2f} ms (min {lat['min'] / 1e3:.2f}, max {lat['max'] / 1e3:.2f}), {prof['iterations']} runs after {prof['warmup']} warmup")
    print(f"kernel times: {len(times)} nodes, sum {sum(times.values()) / 1e3:.2f} ms = {100 * sum(times.values()) / lat['median']:.0f}% of median latency; unmatched {len(prof['unmatched'])}")
    by_kernel = collections.defaultdict(float)
    for i, t in times.items():
        by_kernel[nodes[i]["kernel"]] += t
    print("time by kernel:", ", ".join(f"{k} {t / 1e3:.2f} ms" for k, t in sorted(by_kernel.items(), key=lambda x: -x[1])[:8]))
    print("top nodes:")
    for i, t in sorted(times.items(), key=lambda x: -x[1])[:5]:
        print(f"  {t / 1e3:8.3f} ms  {nodes[i]['kernel']:32s} {nodes[i]['name']}")
    if "--unknown" in sys.argv:
        for n in nodes:
            if best.get(n["id"]) == 2:
                print("  unknown:", n["op"], n["name"], n["outputs"][:2])


if __name__ == "__main__":
    main()
