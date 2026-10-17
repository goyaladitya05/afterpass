"""OpenVINO capture: compile with PERF_COUNT, run timed inferences, read the runtime model."""

import statistics
import time

import openvino as ov

from .util import random_inputs, stats

RT_KEYS = ("execTimeMcs", "layerType", "primitiveType", "runtimePrecision", "outputPrecisions", "outputLayouts", "execOrder", "originalLayersNames")


def _rt(op) -> dict:
    info = op.get_rt_info()
    return {k: str(info[k].astype(str)) for k in info.keys()}


def capture(model_path, device: str, shapes: dict, iterations: int, warmup: int, linker) -> dict:
    core = ov.Core()
    if device not in core.available_devices:
        raise SystemExit(f"OpenVINO device {device} not available (have {core.available_devices})")
    model = core.read_model(str(model_path))
    if shapes:
        model.reshape(shapes)
    if any(p.get_partial_shape().is_dynamic for p in model.inputs):
        raise SystemExit("model has dynamic input dims, pass --shape name=d0,d1,...")

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERF_COUNT": True, "PERFORMANCE_HINT": "LATENCY"})
    compile_s = time.perf_counter() - t0

    feeds = random_inputs([(p.get_any_name(), tuple(p.get_shape()), p.get_element_type().to_dtype()) for p in model.inputs])
    req = compiled.create_infer_request()
    for _ in range(warmup):
        req.infer(feeds)
    latencies, per_node = [], {}
    for _ in range(iterations):
        t = time.perf_counter()
        req.infer(feeds)
        latencies.append((time.perf_counter() - t) * 1e6)
        for p in req.get_profiling_info():
            if p.status == p.Status.EXECUTED:
                per_node.setdefault(p.node_name, []).append(p.real_time.total_seconds() * 1e6)

    # The runtime model is a snapshot: execTimeMcs is only filled if taken after an inference.
    rt_model = compiled.get_runtime_model()
    ops = rt_model.get_ordered_ops()
    ids = {op.get_friendly_name(): i for i, op in enumerate(ops)}
    nodes, tensors, links = [], {}, []
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
                "dtype": precisions[port] if port < len(precisions) else str(out.get_element_type()),
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
                     "properties": props, "nodes": nodes, "tensors": tensors},
        "profile": {
            "unit": "us", "iterations": iterations, "warmup": warmup, "source": "get_profiling_info, median per node",
            "latency": stats(latencies),
            "nodes": {str(ids[name]): statistics.median(v) for name, v in per_node.items() if name in ids},
        },
        "links": links,
        "compile_s": compile_s,
    }

