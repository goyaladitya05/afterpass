"""OpenVINO capture: compile with PERF_COUNT, run timed inferences, read the runtime model."""

import time

import openvino as ov

from .util import random_inputs, stats

RT_KEYS = ("execTimeMcs", "layerType", "primitiveType", "runtimePrecision", "outputPrecisions", "outputLayouts", "execOrder", "originalLayersNames")


def _rt(op) -> dict:
    info = op.get_rt_info()
    return {k: str(info[k].astype(str)) for k in info.keys()}


def _graph(rt_model, entries, linker):
    """Walk the runtime model. Kept in its own function so every reference into the runtime model is
    released before the compiled model; releasing them after it crashes OpenVINO on Windows."""
    ops = rt_model.get_ordered_ops()
    nodes, tensors, links = [], {}, []
    if any(len(op.get_rt_info()) for op in ops):
        ids = {op.get_friendly_name(): i for i, op in enumerate(ops)}
    else:
        # NPU returns a placeholder runtime model without rt_info: the only per layer record is profiling info.
        ids = {p.node_name: i for i, p in enumerate(entries)}
        for i, p in enumerate(entries):
            nodes.append({"id": i, "name": p.node_name, "op": p.node_type, "kernel": p.exec_type, "order": i,
                          "inputs": [], "outputs": [], "attrs": {}, "claims": [p.node_name]})
            links += linker.link(i, [p.node_name])
        ops = []
    for i, op in enumerate(ops):
        rt = _rt(op)
        precisions = rt.get("outputPrecisions", "").split(",")
        layouts = rt.get("outputLayouts", "").split(",")
        outputs = []
        for port, out in enumerate(op.outputs()):
            tid = f"{i}:{port}"
            outputs.append(tid)
            pshape = out.get_partial_shape()
            tensors[tid] = {
                "dtype": precisions[port] if port < len(precisions) else out.get_element_type().get_type_name(),
                "shape": [d.get_length() if d.is_static else None for d in pshape] if pshape.rank.is_static else None,
                "layout": layouts[port] if port < len(layouts) else None,
                "names": sorted(out.get_names()),
            }
        inputs = [f"{ids[inp.get_source_output().get_node().get_friendly_name()]}:{inp.get_source_output().get_index()}" for inp in op.inputs()]
        claimed = [n for n in rt.get("originalLayersNames", "").split(",") if n]
        nodes.append({
            "id": i,
            "name": op.get_friendly_name(),
            "op": rt.get("layerType", op.get_type_name()),
            "kernel": rt.get("primitiveType"),
            "precision": rt.get("runtimePrecision"),
            "order": int(rt["execOrder"]) if rt.get("execOrder", "").lstrip("-").isdigit() else None,
            "inputs": inputs,
            "outputs": outputs,
            "attrs": {k: v for k, v in rt.items() if k not in RT_KEYS},
            "claims": claimed,
        })
        links += linker.link(i, claimed)
    return nodes, tensors, links, ids, bool(ops)


def capture(model_path, device: str, shapes: dict, iterations: int, warmup: int, linker) -> dict:
    core = ov.Core()
    if device not in core.available_devices:
        raise SystemExit(f"OpenVINO device {device} not available (have {core.available_devices})")
    model = core.read_model(str(model_path))
    unknown = [n for n in shapes if not any(n in p.get_names() for p in model.inputs)]
    if unknown:
        raise SystemExit(f"--shape names no input: {unknown} (inputs: {[p.get_any_name() for p in model.inputs]})")
    if shapes:
        try:
            model.reshape(shapes)
        except RuntimeError as e:  # stateful LLMs keep dynamic KV cache dims; compile dynamic, feed static shapes
            print(f"note: static reshape failed, compiling with dynamic shapes ({str(e).splitlines()[0]})")
    specs = []
    for p in model.inputs:
        name = p.get_any_name()
        if name not in shapes and p.get_partial_shape().is_dynamic:
            raise SystemExit(f"input {name} has dynamic dims, pass --shape {name}=d0,d1,...")
        specs.append((name, tuple(shapes.get(name) or p.get_shape()), p.get_element_type().to_dtype()))

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERF_COUNT": True, "PERFORMANCE_HINT": "LATENCY"})
    compile_s = time.perf_counter() - t0

    feeds = random_inputs(specs)
    req = compiled.create_infer_request()
    # a stateful LLM keeps its KV cache between calls: reset it so every run is the same fresh prefill
    stateful = bool(req.query_state())

    def run() -> float:
        if stateful:
            req.reset_state()
        t = time.perf_counter()
        req.infer(feeds)
        return (time.perf_counter() - t) * 1e6

    # get_profiling_info() gives each node's mean over every inference since compile (all requests), not the
    # last run's time (measured on CPU and GPU, 2026.4.1). The mean over the timed runs alone is recovered from
    # the means before and after them.
    def means() -> dict:
        return {p.node_name: p.real_time.total_seconds() * 1e6 for p in req.get_profiling_info() if p.status == p.Status.EXECUTED}

    for _ in range(warmup):
        run()
    before = means() if warmup else {}
    latencies = [run() for _ in range(iterations)]
    entries = req.get_profiling_info()
    per_node = {name: max(0.0, ((warmup + iterations) * m - warmup * before.get(name, m)) / iterations)
                for name, m in means().items()}

    # The runtime model is a snapshot: execTimeMcs is only filled if taken after an inference.
    nodes, tensors, links, ids, topology = _graph(compiled.get_runtime_model(), entries, linker)
    # GPU sometimes names a profiling entry after the node's claim instead of the node; map it when unambiguous.
    by_claim: dict[str, list[int]] = {}
    for n in nodes:
        if n["op"] in ("Input", "Output", "Parameter", "Result"):  # GPU Results repeat their producer's claims
            continue
        for c in n["claims"]:
            by_claim.setdefault(c, []).append(n["id"])
    for name in list(per_node):
        if name not in ids and len(by_claim.get(name, [])) == 1 and nodes[by_claim[name][0]]["name"] not in per_node:
            ids[name] = by_claim[name][0]

    props = {}
    for key in ("INFERENCE_PRECISION_HINT", "EXECUTION_MODE_HINT", "NUM_STREAMS", "INFERENCE_NUM_THREADS", "PERFORMANCE_HINT"):
        try:
            value = compiled.get_property(key)
            props[key] = value.get_type_name() if isinstance(value, ov.Type) else str(value)
        except Exception:
            pass

    return {
        "compiled": {"runtime": "openvino", "version": ov.get_version(), "device": device,
                     "device_name": core.get_property(device, "FULL_DEVICE_NAME"),
                     "properties": props, "topology": topology, "nodes": nodes, "tensors": tensors},
        "profile": {
            "unit": "us", "iterations": iterations, "warmup": warmup, "source": "get_profiling_info running means, mean per node over the timed runs",
            "latency": stats(latencies),
            "nodes": {str(ids[name]): t for name, t in per_node.items() if name in ids},
            "unmatched": {name: t for name, t in per_node.items() if name not in ids},
        },
        "links": links,
        "compile_s": compile_s,
    }

