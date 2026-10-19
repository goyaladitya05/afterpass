"""Does ONNX Runtime's profiler slow a CPU run? Times alternating sessions with and without enable_profiling.

uv run --group dev python scripts/ort_overhead.py MODEL.onnx [ROUNDS] [ITERATIONS] [WARMUP]

Each round builds one unprofiled and one profiled session (one at a time, so a large model fits in memory),
warms each up and times ITERATIONS runs. Prints the median latency of every session, and for profiled ones
the median model_run duration and the median per-run sum of node durations from the trace.
"""

import gc
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from afterpass.util import random_inputs


def session(model, profile_dir):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(profile_dir) / "prof")
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


def measure(model, profiled, iterations, warmup, tmp):
    s = session(model, tmp if profiled else None)
    dtype = lambda t: np.dtype({"float": "float32", "double": "float64"}.get(t[7:-1], t[7:-1]))
    feeds = random_inputs([(i.name, tuple(i.shape), dtype(i.type)) for i in s.get_inputs()])
    for _ in range(warmup):
        s.run(None, feeds)
    lat = []
    for _ in range(iterations):
        t = time.perf_counter()
        s.run(None, feeds)
        lat.append((time.perf_counter() - t) * 1e6)
    line = f"load {Path('/proc/loadavg').read_text().split()[0]:>5}  {'profiled  ' if profiled else 'unprofiled'} median {statistics.median(lat) / 1e3:9.2f} ms  min {min(lat) / 1e3:9.2f}  max {max(lat) / 1e3:9.2f}"
    if profiled:
        path = s.end_profiling()
        ev = json.loads(Path(path).read_text())
        Path(path).unlink()
        runs = sorted((e for e in ev if e.get("cat") == "Session" and e["name"] == "model_run"), key=lambda e: e["ts"])[warmup:]
        nodes = [e for e in ev if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
        sums = [sum(e["dur"] for e in nodes if r["ts"] <= e["ts"] <= r["ts"] + r["dur"]) for r in runs]
        line += f"  model_run median {statistics.median(r['dur'] for r in runs) / 1e3:9.2f} ms  node sum median {statistics.median(sums) / 1e3:9.2f} ms"
    del s
    gc.collect()
    return line, statistics.median(lat)


def main():
    model = sys.argv[1]
    rounds, iterations, warmup = (int(a) for a in (sys.argv[2:] + ["3", "20", "5"][len(sys.argv) - 2:])[:3])
    print(f"onnxruntime {ort.__version__}, {Path(model).name}, {rounds} rounds of {iterations} timed runs after {warmup} warmup")
    ratios = []
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        for r in range(rounds):
            med = {}
            for profiled in (False, True) if r % 2 == 0 else (True, False):  # alternate the order against drift
                line, med[profiled] = measure(model, profiled, iterations, warmup, tmp)
                print(f"round {r}: {line}", flush=True)
            ratios.append(med[True] / med[False])
    print(f"profiled / unprofiled median latency per round: {', '.join(f'{x:.3f}' for x in ratios)}; median {statistics.median(ratios):.3f}")


if __name__ == "__main__":
    main()
