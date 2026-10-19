"""How much of an ONNX Runtime session's creation time is saving the optimized model? Alternates both kinds.

uv run --group dev python scripts/ort_compile.py MODEL.onnx [ROUNDS]
"""

import gc
import sys
import tempfile
import time
from pathlib import Path

import onnxruntime as ort


def create(model, tmp):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if tmp:
        so.optimized_model_filepath = str(Path(tmp) / "optimized.onnx")
        so.add_session_config_entry("session.optimized_model_external_initializers_file_name", "optimized.bin")
        so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    t = time.perf_counter()
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    dt = time.perf_counter() - t
    size = sum(f.stat().st_size for f in Path(tmp).iterdir()) if tmp else 0
    del s
    gc.collect()
    return dt, size


def main():
    model, rounds = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"onnxruntime {ort.__version__}, {Path(model).name}")
    for r in range(rounds):
        for save in (False, True) if r % 2 == 0 else (True, False):
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                dt, size = create(model, tmp if save else None)
            load = Path("/proc/loadavg").read_text().split()[0]
            print(f"round {r}: load {load:>5}  {'saving   ' if save else 'not saving'} session created in {dt:7.2f} s" + (f", optimized model {size / 1e6:.0f} MB" if save else ""), flush=True)


if __name__ == "__main__":
    main()
