"""OpenVINO capture: compile with PERF_COUNT, run timed inferences, read the runtime model."""

import time

import numpy as np
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


def capture(model_path, device: str, shapes: dict, iterations: int, warmup: int, linker, decode: int = 0, static: bool = False) -> dict:
    """Profile one prefill per run, or with `decode` > 0 the `decode` single-token steps that follow the prefill."""
    core = ov.Core()
    if device not in core.available_devices:
        raise SystemExit(f"OpenVINO device {device} not available (have {core.available_devices})")
    model = core.read_model(str(model_path))
    unknown = [n for n in shapes if not any(n in p.get_names() for p in model.inputs)]
    if unknown:
        raise SystemExit(f"--shape names no input: {unknown} (inputs: {[p.get_any_name() for p in model.inputs]})")
    # a stateful LLM stays dynamic, as OpenVINO GenAI compiles it: one compiled graph serves prefill and decode,
    # so the two phases can be compared kernel by kernel; `static` fixes the --shape dims anyway, to see what
    # dynamic shapes cost
    if shapes and (static or not model.get_sinks()):
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

    feeds = random_inputs(specs)
    names = {s[0] for s in specs}
    not_llm = "--decode needs a stateful LLM IR with input_ids and attention_mask inputs (an optimum-intel export)"
    if decode and not {"input_ids", "attention_mask"} <= names:
        raise SystemExit(not_llm)
    prompt = feeds["input_ids"].shape[1] if decode else 0

    def step_feeds(i: int) -> dict:
        """Decode step i: one new token at position prompt + i, attending to every earlier one."""
        f = {"input_ids": np.zeros((1, 1), feeds["input_ids"].dtype),
             "attention_mask": np.ones((1, prompt + i + 1), feeds["attention_mask"].dtype)}
        if "position_ids" in names:
            f["position_ids"] = np.full((1, 1), prompt + i, feeds["position_ids"].dtype)
        if "beam_idx" in names:
            f["beam_idx"] = feeds["beam_idx"]
        return f

    def measure(perf: bool):
        """Compile, then run `warmup` and `iterations` runs (each a prefill, then `decode` steps when decoding).
        Returns the compiled model, its request, the compile time, the timed latencies and, when `perf`, the mean
        time per node over the timed runs or steps."""
        t0 = time.perf_counter()
        compiled = core.compile_model(model, device, {"PERF_COUNT": perf, "PERFORMANCE_HINT": "LATENCY"})
        compile_s = time.perf_counter() - t0
        req = compiled.create_infer_request()
        # a stateful LLM keeps its KV cache between calls: reset it so every run is the same fresh prefill
        stateful = bool(req.query_state())
        if decode and not stateful:
            raise SystemExit(not_llm)

        def run() -> float:
            if stateful:
                req.reset_state()
            t = time.perf_counter()
            req.infer(feeds)
            return (time.perf_counter() - t) * 1e6

        # get_profiling_info() gives each node's mean over every inference since compile (all requests), not the
        # last run's time (measured on CPU and GPU, 2026.4.1). The mean over the timed runs alone is recovered
        # from the means before and after them.
        def means() -> dict:
            if not perf:
                return {}
            return {p.node_name: p.real_time.total_seconds() * 1e6 for p in req.get_profiling_info() if p.status == p.Status.EXECUTED}

        if decode:
            # Each run is a fresh prefill, then `decode` timed steps. The running means also count the prefills,
            # so every decode window is bracketed: the window's total per node is n_after * mean_after - n_before
            # * mean_before, where n counts every inference since compile.
            done, totals, latencies = 0, {}, []
            for it in range(warmup + iterations):
                run()
                done += 1
                before = means() if it >= warmup else {}
                for i in range(decode):
                    t = time.perf_counter()
                    req.infer(step_feeds(i))
                    if it >= warmup:
                        latencies.append((time.perf_counter() - t) * 1e6)
                done += decode
                if it >= warmup:
                    for name, m in means().items():
                        totals[name] = totals.get(name, 0.0) + done * m - (done - decode) * before.get(name, 0.0)
            per_node = {name: max(0.0, total / (iterations * decode)) for name, total in totals.items()}
        else:
            for _ in range(warmup):
                run()
            before = means() if warmup else {}
            latencies = [run() for _ in range(iterations)]
            per_node = {name: max(0.0, ((warmup + iterations) * m - warmup * before.get(name, m)) / iterations)
                        for name, m in means().items()}
        return compiled, req, compile_s, latencies, per_node

    # Latency comes from a compile without PERF_COUNT: profiling slows inference (fastest runs on the Lunar Lake VM,
    # idle: ResNet-50 GPU 1.65 against 3.1 ms, TinyLlama GPU 33.8 against 41.4 ms). The kernel times come from a
    # second compile with it. Slicing drops the first compiled model before the second is built.
    compile_s, latencies = measure(False)[2:4]
    compiled, req, _, _, per_node = measure(True)
    entries = req.get_profiling_info()

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
                     "properties": props, "topology": topology, "nodes": nodes, "tensors": tensors,
                     # as compiled: "?" marks a dynamic dimension (the runtime model only shows the last run's shapes)
                     "input_shapes": {p.get_any_name(): str(p.get_partial_shape()) for p in compiled.inputs}},
        "profile": {
            "unit": "us", "iterations": iterations, "warmup": warmup,
            "source": "get_profiling_info running means, mean per node over the timed " + ("decode steps" if decode else "runs")
            + "; latency from a separate compile without PERF_COUNT",
            **({"phase": "decode", "prompt_tokens": prompt, "decode_steps": decode} if decode else {}),
            "latency": stats(latencies),
            "nodes": {str(ids[name]): t for name, t in per_node.items() if name in ids},
            "unmatched": {name: t for name, t in per_node.items() if name not in ids},
        },
        "links": links,
        "compile_s": compile_s,
    }

