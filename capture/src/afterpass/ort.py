"""ONNX Runtime capture: save the optimized model (the compiled graph), profile every run, read node times from the trace."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi import onnxruntime_pybind11_state as C

from .bundle import _cpu
from .source import _attr, _dims, dtype_name
from .util import random_inputs, stats

# --device values to execution providers. CUDA and TensorRT are planned; CPU is the only one implemented and tested.
PROVIDERS = {"CPU": "CPUExecutionProvider"}
NCHWC = "com.microsoft.nchwc"
CAST = "InsertedPrecisionFreeCast_"  # casts put around a node whose provider has no kernel for its type
# ONNX Runtime's exceptions are not RuntimeErrors; the CLI turns RuntimeError into a one-line message
ERRORS = (C.Fail, C.InvalidArgument, C.InvalidGraph, C.InvalidProtobuf, C.NoSuchFile, C.NotImplemented, C.RuntimeException, C.EPFail)


def _numpy(ort_type: str) -> np.dtype:
    t = ort_type.removeprefix("tensor(").removesuffix(")")
    return np.dtype({"float": "float32", "double": "float64"}.get(t, t))


def _kernel(provider: str, n) -> str:
    """Provider plus the op as ONNX Runtime registers its kernel: "CPU com.microsoft.nchwc::Conv", "CPU MatMul"."""
    op = n.op_type if n.domain in ("", "ai.onnx") else f"{n.domain}::{n.op_type}"
    return f"{provider.removesuffix('ExecutionProvider')} {op}" if provider else op


def _runs(events, warmup: int) -> list[list[dict]]:
    """Node events per timed run, in execution order. A run's nodes lie inside its model_run event."""
    runs = sorted((e for e in events if e.get("cat") == "Session" and e["name"] == "model_run"), key=lambda e: e["ts"])
    nodes = sorted((e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")), key=lambda e: e["ts"])
    out, i = [], 0
    for r in runs:
        while i < len(nodes) and nodes[i]["ts"] < r["ts"]:
            i += 1
        run = []
        while i < len(nodes) and nodes[i]["ts"] <= r["ts"] + r["dur"]:
            run.append(nodes[i])
            i += 1
        out.append(run)
    return out[warmup:]


def _event_ids(g, run: list[dict]) -> dict[str, int]:
    """Profile name to node index. The profiler calls a node without a name "<op type>_<internal index>"; those are
    matched in execution order, which is the order of the saved model, and only if their op types line up."""
    ids = {n.name: i for i, n in enumerate(g.node) if n.name}
    unnamed = [i for i, n in enumerate(g.node) if not n.name]
    events = [e for e in run if e["name"].removesuffix("_kernel_time") not in ids]
    if [e["args"].get("op_name") for e in events] == [g.node[i].op_type for i in unnamed]:
        ids.update((e["name"].removesuffix("_kernel_time"), i) for e, i in zip(events, unnamed))
    return ids


def _session(model_path, provider: str, tmp: Path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # errors only: every NCHWc model otherwise warns that the saved model is hardware specific
    so.optimized_model_filepath = str(tmp / "optimized.onnx")
    # weights go to a side file: an optimized model over 2 GB cannot be saved as one protobuf
    so.add_session_config_entry("session.optimized_model_external_initializers_file_name", "optimized.bin")
    so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    so.enable_profiling = True
    so.profile_file_prefix = str(tmp / "profile")
    return ort.InferenceSession(str(model_path), so, providers=[provider])


def _run(model_path, provider: str, shapes: dict, iterations: int, warmup: int, tmp: Path):
    t0 = time.perf_counter()
    sess = _session(model_path, provider, tmp)
    compile_s = time.perf_counter() - t0

    inputs = sess.get_inputs()
    unknown = [n for n in shapes if n not in {i.name for i in inputs}]
    if unknown:
        raise SystemExit(f"--shape names no input: {unknown} (inputs: {[i.name for i in inputs]})")
    specs = []
    for i in inputs:
        dims = shapes.get(i.name) or i.shape
        if not all(isinstance(d, int) for d in dims):
            raise SystemExit(f"input {i.name} has dynamic dims, pass --shape {i.name}=d0,d1,...")
        specs.append((i.name, tuple(dims), _numpy(i.type)))
    feeds = random_inputs(specs)

    latencies = []
    for k in range(warmup + iterations):
        t = time.perf_counter()
        sess.run(None, feeds)
        if k >= warmup:
            latencies.append((time.perf_counter() - t) * 1e6)
    runs = _runs(json.loads(Path(sess.end_profiling()).read_text()), warmup)
    return compile_s, latencies, runs, onnx.load(str(tmp / "optimized.onnx"), load_external_data=False)


def capture(model_path, device: str, shapes: dict, iterations: int, warmup: int, linker, work_dir=None) -> dict:
    provider = PROVIDERS.get(device)
    if provider not in ort.get_available_providers():
        raise SystemExit(f"ONNX Runtime provider for --device {device} not available (have {ort.get_available_providers()})")
    # the optimized model holds a copy of every weight: keep it on disk next to the bundle, not in a RAM-backed /tmp
    with tempfile.TemporaryDirectory(prefix=".afterpass-ort-", dir=work_dir or Path.cwd()) as tmp:
        try:
            compile_s, latencies, runs, model = _run(model_path, provider, shapes, iterations, warmup, Path(tmp))
        except ERRORS as e:
            raise RuntimeError(str(e)) from None

    g = model.graph
    ids = _event_ids(g, runs[0])
    totals: dict[str, float] = {}
    first: dict[int, dict] = {}
    for run in runs:
        for e in run:
            name = e["name"].removesuffix("_kernel_time")
            totals[name] = totals.get(name, 0.0) + e["dur"]
            if name in ids:
                first.setdefault(ids[name], e)
    order = {i: k for k, i in enumerate(first)}

    tensors: dict[str, dict] = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        tt = vi.type.tensor_type
        tensors[vi.name] = {"dtype": dtype_name(tt.elem_type), "shape": _dims(tt.shape) if tt.HasField("shape") else None}
    for init in g.initializer:
        tensors[init.name] = {"dtype": dtype_name(init.data_type), "shape": list(init.dims), "kind": "initializer"}
    for vi in g.input:
        tensors[vi.name].setdefault("kind", "input")
    for vi in g.output:
        tensors[vi.name]["kind"] = "output"

    nodes, links, blocked = [], [], set()
    for i, n in enumerate(g.node):
        e = first.get(i)
        provider_used = e["args"].get("provider", "") if e else ""
        # the profile has the shapes this run produced; the saved model often has no value_info for them
        traced = e["args"].get("output_type_shape", []) if e else []
        outputs = [t for t in n.output if t]
        for t, ts in zip(outputs, traced if len(traced) == len(outputs) else []):
            (dtype, shape), = ts.items()
            cur = tensors.setdefault(t, {"dtype": dtype, "shape": shape})
            if cur["shape"] is None or None in cur["shape"] or any(isinstance(d, str) for d in cur["shape"]):
                cur["shape"] = shape
        for t in list(n.input) + outputs:
            if t:
                tensors.setdefault(t, {"dtype": "unknown", "shape": None})
        # NCHWc ops and the elementwise ops between them keep data in channel blocks until a ReorderOutput
        if (n.domain == NCHWC and n.op_type != "ReorderOutput") or (n.domain != NCHWC and any(t in blocked for t in n.input)):
            blocked.update(outputs)
            for t in outputs:
                tensors[t]["layout"] = "NCHWc"
        claims = [n.name]
        if n.op_type in ("ReorderInput", "ReorderOutput") or n.name.startswith(CAST):
            pass  # inserted to convert data: takes over the tensor name of the op it serves, computes none
        elif n.domain == NCHWC and n.name.endswith("_nchwc"):
            # an NCHWc node is named after the output tensor of the source op it replaced, its own outputs are new:
            # record that name on the output so the core can bound the node by tensor names (infer_boundaries).
            # Not for a Conv with a fused Sum input (a 4th input): it also computes the residual Add and its
            # activation, but keeps the Conv's name, so the name would be the wrong tensor.
            claims.append(n.name.removesuffix("_nchwc"))
            if len(outputs) == 1 and not (n.op_type == "Conv" and len(n.input) > 3 and n.input[3]):
                tensors[outputs[0]]["names"] = [n.name.removesuffix("_nchwc")]
        else:
            # a fused node keeps the output tensor of the last op it computed, renamed if a cast was put after it
            claims += [t.removeprefix(CAST) for t in outputs]
        attrs = {a.name: _attr(a) for a in n.attribute}
        if n.domain:
            attrs["domain"] = n.domain
        if provider_used:
            attrs["provider"] = provider_used
        nodes.append({
            "id": i,
            "name": n.name,
            "op": n.op_type,
            "kernel": _kernel(provider_used, n),
            "order": order.get(i),
            "inputs": list(n.input),
            "outputs": outputs,
            "attrs": attrs,
            "claims": claims,
        })
        links += linker.link(i, claims)

    per_node = {name: t / len(runs) for name, t in totals.items()}
    return {
        "compiled": {"runtime": "onnxruntime", "version": ort.__version__, "device": device, "device_name": _cpu()["name"],
                     "properties": {"providers": [provider], "graph_optimization_level": "ORT_ENABLE_ALL"},
                     "topology": True, "nodes": nodes, "tensors": tensors},
        "profile": {
            "unit": "us", "iterations": iterations, "warmup": warmup, "source": "enable_profiling trace, mean per node over the timed runs; latency from the same profiled runs",
            "latency": stats(latencies),
            "nodes": {str(ids[name]): t for name, t in per_node.items() if name in ids},
            "unmatched": {name: t for name, t in per_node.items() if name not in ids},
        },
        "links": links,
        "compile_s": compile_s,
    }
