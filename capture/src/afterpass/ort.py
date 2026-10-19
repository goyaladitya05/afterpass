"""ONNX Runtime capture: save the optimized model (the compiled graph), profile every run, read node times from the trace."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi import build_and_package_info
from onnxruntime.capi import onnxruntime_pybind11_state as C

from .bundle import _cpu, _gpus
from .source import _attr, _dims, dtype_name
from .util import random_inputs, stats

# --device values to execution providers in priority order: each node goes to the first one with a kernel for it
PROVIDERS = {"CPU": ["CPUExecutionProvider"],
             "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "TensorRT": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]}
NCHWC = "com.microsoft.nchwc"
CAST = "InsertedPrecisionFreeCast_"  # casts put around a node whose provider has no kernel for its type
COPIED = re.compile(r"_\w+ExecutionProvider$")  # a tensor's name on one side of a copy between devices
MOVES = ("ReorderInput", "ReorderOutput", "MemcpyFromHost", "MemcpyToHost")  # inserted to convert or copy data
# one TensorRT engine: "TensorrtExecutionProvider_TRTKernel_graph_<graph>_<hash>_<subgraph>_<node index>"
ENGINE = "TensorrtExecutionProvider_TRTKernel_"
# ONNX Runtime's exceptions are not RuntimeErrors; the CLI turns RuntimeError into a one-line message
ERRORS = (C.Fail, C.InvalidArgument, C.InvalidGraph, C.InvalidProtobuf, C.NoSuchFile, C.NotImplemented, C.RuntimeException, C.EPFail)


def _numpy(ort_type: str) -> np.dtype:
    t = ort_type.removeprefix("tensor(").removesuffix(")")
    return np.dtype({"float": "float32", "double": "float64"}.get(t, t))


def _kernel(provider: str, domain: str, op: str) -> str:
    """Provider plus the op as ONNX Runtime registers its kernel: "CPU com.microsoft.nchwc::Conv", "CUDA MatMul"."""
    op = op if domain in ("", "ai.onnx") else f"{domain}::{op}"
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


def _load(device: str) -> None:
    """Load the CUDA and cuDNN libraries of their pip packages and, for TensorRT, TensorRT's (libnvinfer.so.10, which
    the provider links to but does not load itself)."""
    if device == "CPU":
        return
    ort.preload_dlls()
    if device == "TensorRT":
        try:
            import tensorrt_libs  # noqa: F401
        except ImportError:
            pass  # a TensorRT on the library path works too; _check says what is missing if there is none


def _check(sess, device: str) -> None:
    """ONNX Runtime does not fail when a provider cannot load: it logs an error and runs the session on the next one."""
    want, have = PROVIDERS[device][0], sess.get_providers()
    if have[0] != want:
        cuda = getattr(build_and_package_info, "cuda_version", "") or "the CUDA version it was built for"
        trt = " and TensorRT 10 (pip install tensorrt-cu13, or tensorrt-cu12 for a CUDA 12 build)" if device == "TensorRT" else ""
        raise SystemExit(f"ONNX Runtime could not load {want} and would run on {have[0]} instead (its error is above). "
                         f"onnxruntime-gpu {ort.__version__} needs a driver for CUDA {cuda}, the CUDA {cuda} and cuDNN 9 "
                         f"libraries (pip install 'onnxruntime-gpu[cuda,cudnn]'){trt}")


def _providers(device: str, tmp: Path) -> list:
    """TensorRT keeps its engines in `tmp`, so the profiled process and the compile load them instead of building them again."""
    if device == "TensorRT":
        return [("TensorrtExecutionProvider", {"trt_engine_cache_enable": "True", "trt_engine_cache_path": str(tmp / "engines")}),
                *PROVIDERS[device][1:]]
    return PROVIDERS[device]


def _session(model_path, device: str, tmp: Path, profile: bool):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # errors only: every NCHWc model otherwise warns that the saved model is hardware specific
    if profile:
        if device != "TensorRT":  # TensorRT engines are compiled nodes, which a session cannot save (_compile)
            so.optimized_model_filepath = str(tmp / "optimized.onnx")
            # weights go to a side file: an optimized model over 2 GB cannot be saved as one protobuf
            so.add_session_config_entry("session.optimized_model_external_initializers_file_name", "optimized.bin")
            so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
        so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
        so.enable_profiling = True
        so.profile_file_prefix = str(tmp / "profile")
    sess = ort.InferenceSession(str(model_path), so, providers=_providers(device, tmp))
    _check(sess, device)
    return sess


def _compile(model_path, tmp: Path) -> None:
    """Save the graph with each TensorRT engine as one node through the compile API (ModelCompiler), which a session
    refuses ("Unable to serialize model as it contains compiled nodes"). Its node names are the profile's."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    for p in _providers("TensorRT", tmp):
        name, options = p if isinstance(p, tuple) else (p, {})
        so.add_provider(name, options)
    ort.ModelCompiler(so, str(model_path), graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                      external_initializers_file_path="optimized.bin").compile_to_file(str(tmp / "optimized.onnx"))


def _feeds(sess, shapes: dict) -> dict:
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
    return random_inputs(specs)


def _timed(sess, feeds: dict, device: str, iterations: int, warmup: int) -> list[float]:
    """Latencies of the timed runs. On a GPU the inputs are copied to it once and the outputs stay there, as in the
    TensorRT capture, so a run is the model's work and not the PCIe copies around it."""
    if device == "CPU":
        def run():
            sess.run(None, feeds)
    else:
        binding = sess.io_binding()
        values = [ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0) for a in feeds.values()]
        for name, v in zip(feeds, values):
            binding.bind_ortvalue_input(name, v)
        for o in sess.get_outputs():
            binding.bind_output(o.name, "cuda")

        def run():
            sess.run_with_iobinding(binding)
            binding.synchronize_outputs()
    latencies = []
    for k in range(warmup + iterations):
        t = time.perf_counter()
        run()
        if k >= warmup:
            latencies.append((time.perf_counter() - t) * 1e6)
    return latencies


def _profiled(model_path, device: str, shapes: dict, iterations: int, warmup: int, tmp: Path) -> dict:
    """Profile `warmup + iterations` runs and save the compiled graph to `tmp`. On a GPU this runs in its own process
    (_measure); on the CPU these runs also give the latency (D73)."""
    _load(device)
    t0 = time.perf_counter()
    sess = _session(model_path, device, tmp, profile=True)
    compile_s = time.perf_counter() - t0
    latencies = _timed(sess, _feeds(sess, shapes), device, iterations, warmup)
    out = {"compile_s": compile_s, "latencies": latencies, "trace": sess.end_profiling(), "providers": sess.get_providers(),
           # which nodes each provider took; for TensorRT the source of a fused node's claims
           "assignment": [[sg.ep_name, [n.name for n in sg.get_nodes()]] for sg in sess.get_provider_graph_assignment_info()],
           # local paths (the engine cache) stay out of a shared bundle
           "options": {k: v for k, v in sess.get_provider_options().get(PROVIDERS[device][0], {}).items() if not k.endswith("path")}}
    if device == "TensorRT":
        del sess
        _compile(model_path, tmp)
    return out


def _measure(model_path, device: str, shapes: dict, iterations: int, warmup: int, tmp: Path) -> tuple[float, list, dict]:
    """compile_s, latencies and the profiled result. On a GPU a node's time in the trace is only its launch, unless
    every launch blocks (CUDA_LAUNCH_BLOCKING=1), which in turn inflates the run (D74). So latency comes from a session
    without profiling, and the trace from a second process: the variable only acts when CUDA starts."""
    if device == "CPU":
        r = _profiled(model_path, device, shapes, iterations, warmup, tmp)
        return r["compile_s"], r["latencies"], r
    _load(device)
    t0 = time.perf_counter()
    sess = _session(model_path, device, tmp, profile=False)
    compile_s = time.perf_counter() - t0
    latencies = _timed(sess, _feeds(sess, shapes), device, iterations, warmup)
    del sess  # free the GPU for the profiled process
    args = {"model": str(model_path), "device": device, "shapes": shapes, "iterations": iterations, "warmup": warmup, "tmp": str(tmp)}
    done = subprocess.run([sys.executable, "-m", "afterpass.ort", json.dumps(args)], env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"})
    if done.returncode:
        raise RuntimeError(f"the profiled run (CUDA_LAUNCH_BLOCKING=1) exited with code {done.returncode}, its error is above")
    return compile_s, latencies, json.loads((tmp / "profiled.json").read_text())


def _fused_sum_output(n, tensors, linker):
    """The source tensor an NCHWc Conv with a Sum input really writes: the Add that reads the Conv's output and the
    sum tensor, then the activation op named in the node, if any. None when the source graph does not show exactly
    that chain."""
    conv_out = n.name.removesuffix("_nchwc")
    summed = (tensors.get(n.input[3], {}).get("names") or [n.input[3]])[0]
    adds = [r for r in linker.readers.get(conv_out, []) if r["op"] == "Add" and summed in r["inputs"]]
    if len(adds) != 1:
        return None
    out = adds[0]["outputs"][0]
    act = next((a.s.decode() for a in n.attribute if a.name == "activation"), "")
    if act:
        acts = [r for r in linker.readers.get(out, []) if r["op"] == act]
        if len(acts) != 1:
            return None
        out = acts[0]["outputs"][0]
    return out


def _engine_nodes(name: str, engines: list[list[str]]) -> list[str]:
    """The nodes a TensorRT engine computes: the provider's subgraphs in the assignment info come in the order of the
    subgraph number in the engine's name ("..._<subgraph>_<node index>")."""
    k = name.rsplit("_", 2)[1]
    return engines[int(k)] if k.isdigit() and int(k) < len(engines) else []


def capture(model_path, device: str, shapes: dict, iterations: int, warmup: int, linker, work_dir=None) -> dict:
    if PROVIDERS[device][0] not in ort.get_available_providers():
        raise SystemExit(f"ONNX Runtime provider for --device {device} not available (have {ort.get_available_providers()}): "
                         "CUDA and TensorRT need onnxruntime-gpu instead of onnxruntime")
    # the optimized model holds a copy of every weight: keep it on disk next to the bundle, not in a RAM-backed /tmp
    with tempfile.TemporaryDirectory(prefix=".afterpass-ort-", dir=work_dir or Path.cwd()) as tmp:
        try:
            compile_s, latencies, prof = _measure(model_path, device, shapes, iterations, warmup, Path(tmp))
        except ERRORS as e:
            raise RuntimeError(str(e)) from None
        events = json.loads(Path(prof["trace"]).read_text())
        model = onnx.load(str(Path(tmp) / "optimized.onnx"), load_external_data=False)
    runs = _runs(events, warmup)
    profiled = [e["dur"] for e in sorted((e for e in events if e.get("cat") == "Session" and e["name"] == "model_run"), key=lambda e: e["ts"])][warmup:]
    assigned = {name: ep for ep, names in prof["assignment"] for name in names}
    engines = [names for ep, names in prof["assignment"] if ep == "TensorrtExecutionProvider"]

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
        provider_used = e["args"].get("provider", "") if e else assigned.get(n.name, "")
        op = n.op_type
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
        if n.op_type in MOVES or n.name.startswith(CAST):
            pass  # inserted to convert or copy data: takes over the tensor name of the op it serves, computes none
        elif n.name.startswith(ENGINE):
            # a TensorRT engine computes the nodes the provider took in its subgraph, numbered as in the assignment
            op, provider_used = "TRTKernel", provider_used or "TensorrtExecutionProvider"
            claims += _engine_nodes(n.name, engines)
        elif n.domain == NCHWC and n.name.endswith("_nchwc"):
            # an NCHWc node is named after the output tensor of the source op it replaced, its own outputs are new:
            # record that name on the output so the core can bound the node by tensor names (infer_boundaries).
            # A Conv with a fused Sum input (a 4th input) computes activation(conv(x) + sum) but keeps the Conv's
            # name, so its output is found in the source graph instead (_fused_sum_output).
            claims.append(n.name.removesuffix("_nchwc"))
            if len(outputs) == 1:
                summed = n.op_type == "Conv" and len(n.input) > 3 and n.input[3]
                out = _fused_sum_output(n, tensors, linker) if summed else n.name.removesuffix("_nchwc")
                if out:
                    tensors[outputs[0]]["names"] = [out]
                    claims.append(out)
        else:
            # a fused node keeps the output tensor of the last op it computed, renamed if a cast or a copy to another
            # device was put after it ("<tensor>_CUDAExecutionProvider")
            claims += [COPIED.sub("", t.removeprefix(CAST)) for t in outputs]
        attrs = {a.name: _attr(a) for a in n.attribute}
        if n.domain and op != "TRTKernel":
            attrs["domain"] = n.domain
        if provider_used:
            attrs["provider"] = provider_used
        nodes.append({
            "id": i,
            "name": n.name,
            "op": op,
            "kernel": _kernel(provider_used, "" if op == "TRTKernel" else n.domain, op),
            "order": order.get(i),
            "inputs": list(n.input),
            "outputs": outputs,
            "attrs": attrs,
            "claims": claims,
        })
        links += linker.link(i, claims)

    per_node = {name: t / len(runs) for name, t in totals.items()}
    gpu = device != "CPU"
    return {
        "compiled": {"runtime": "onnxruntime", "version": ort.__version__, "device": device,
                     "device_name": (_gpus() or [{"name": ""}])[0]["name"] if gpu else _cpu()["name"],
                     "properties": {"providers": prof["providers"], "graph_optimization_level": "ORT_ENABLE_ALL",
                                    **({"provider_options": prof["options"]} if gpu else {})},
                     "topology": True, "nodes": nodes, "tensors": tensors},
        "profile": {
            "unit": "us", "iterations": iterations, "warmup": warmup,
            "source": "enable_profiling trace with CUDA_LAUNCH_BLOCKING=1 in a separate process, mean per node over the timed "
            "runs: blocking makes a node's time its GPU work but slows the run, so compare kernels by these times, not with "
            "latency; latency from a session without profiling, inputs and outputs on the GPU" if gpu else
            "enable_profiling trace, mean per node over the timed runs; latency from the same profiled runs",
            "latency": stats(latencies),
            **({"profiled_latency": stats(profiled)} if gpu else {}),
            "nodes": {str(ids[name]): t for name, t in per_node.items() if name in ids},
            "unmatched": {name: t for name, t in per_node.items() if name not in ids},
        },
        "links": links,
        "compile_s": compile_s,
    }


if __name__ == "__main__":
    # the profiled process of a GPU capture (_measure), started with CUDA_LAUNCH_BLOCKING=1
    a = json.loads(sys.argv[1])
    try:
        r = _profiled(a["model"], a["device"], a["shapes"], a["iterations"], a["warmup"], Path(a["tmp"]))
    except ERRORS as e:
        sys.exit(str(e))
    (Path(a["tmp"]) / "profiled.json").write_text(json.dumps(r))
