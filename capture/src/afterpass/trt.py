"""TensorRT capture: build with DETAILED profiling verbosity, time layers with IProfiler, export the engine inspector JSON."""

import json
import re
import statistics
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from .util import random_inputs, stats

# Entries look like "[ONNX Layer: X]"; TensorRT 11 separates them with \x1e inside one record and \x1f between records.
ONNX_LAYER = re.compile(r"\[ONNX Layer: (.*?)\](?=[\x1e\x1f]|$)")
LAYER_KEYS = ("Name", "LayerType", "Inputs", "Outputs", "Metadata", "TacticName")


def _cuda(result):
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error {err}")
    return values[0] if len(values) == 1 else values


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times: dict[str, list[float]] = {}

    def report_layer_time(self, name, ms):
        self.times.setdefault(name, []).append(ms * 1000.0)


def _claims(metadata: str) -> list[str]:
    return ONNX_LAYER.findall(metadata or "")


def capture(model_path, device: str, shapes: dict, iterations: int, warmup: int, linker) -> dict:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # strongly typed: precision comes from the model, as TensorRT 11 always does, so versions compile the same graph
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(model_path)):
        raise SystemExit("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    if any(-1 in tuple(t.shape) for t in inputs):
        profile = builder.create_optimization_profile()
        for t in inputs:
            if t.name not in shapes and -1 in tuple(t.shape):
                raise SystemExit(f"input {t.name} has dynamic dims, pass --shape {t.name}=d0,d1,...")
            s = shapes.get(t.name, tuple(t.shape))
            profile.set_shape(t.name, s, s, s)
        config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    compile_s = time.perf_counter() - t0
    if serialized is None:
        raise SystemExit("TensorRT engine build failed")
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    for n in in_names:
        if n in shapes:
            context.set_input_shape(n, shapes[n])
    feeds = random_inputs([(n, tuple(context.get_tensor_shape(n)), trt.nptype(engine.get_tensor_dtype(n))) for n in in_names])
    buffers = []
    for n in names:
        shape = tuple(context.get_tensor_shape(n))
        nbytes = max(int(np.prod(shape)) * np.dtype(trt.nptype(engine.get_tensor_dtype(n))).itemsize, 1)
        ptr = _cuda(cudart.cudaMalloc(nbytes))
        buffers.append(ptr)
        context.set_tensor_address(n, ptr)
        if n in feeds:
            host = np.ascontiguousarray(feeds[n])
            _cuda(cudart.cudaMemcpy(ptr, host.ctypes.data, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    stream = _cuda(cudart.cudaStreamCreate())

    def run():
        if not context.execute_async_v3(int(stream)):
            raise RuntimeError("TensorRT execution failed")
        _cuda(cudart.cudaStreamSynchronize(stream))

    try:
        for _ in range(warmup):
            run()
        latencies = []
        for _ in range(iterations):
            t = time.perf_counter()
            run()
            latencies.append((time.perf_counter() - t) * 1e6)
        profiler = Profiler()
        context.profiler = profiler
        for _ in range(iterations):
            run()

        inspector = engine.create_engine_inspector()
        inspector.execution_context = context
        info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    finally:
        for ptr in buffers:
            cudart.cudaFree(ptr)
        cudart.cudaStreamDestroy(stream)

    nodes, tensors, links = [], {}, []
    for i, layer in enumerate(info["Layers"]):
        for io in layer.get("Inputs", []) + layer.get("Outputs", []):
            tensors.setdefault(io["Name"], {
                "dtype": io.get("Datatype"), "shape": io.get("Dimensions"), "layout": io.get("Format"),
                **{k: v for k, v in io.items() if k not in ("Name", "Datatype", "Dimensions", "Format")},
            })
        claims = _claims(layer.get("Metadata", ""))
        outputs = [o["Name"] for o in layer.get("Outputs", [])]
        nodes.append({
            "id": i,
            "name": layer["Name"],
            "op": layer.get("LayerType"),
            "kernel": layer.get("TacticName"),
            "order": i,
            "inputs": [t["Name"] for t in layer.get("Inputs", [])],
            "outputs": outputs,
            "attrs": {k: v for k, v in layer.items() if k not in LAYER_KEYS},
            "claims": claims,
        })
        links += linker.link(i, claims)

    ids = {n["name"]: n["id"] for n in nodes}
    medians = {name: statistics.median(v) for name, v in profiler.times.items()}
    props = cudart.cudaGetDeviceProperties(0)[1]
    return {
        "compiled": {"runtime": "tensorrt", "version": trt.__version__, "device": device,
                     "device_name": props.name.decode().rstrip("\x00"),
                     "properties": {k: v for k, v in info.items() if k != "Layers"},
                     "topology": True,
                     "nodes": nodes, "tensors": tensors},
        "profile": {
            "unit": "us", "iterations": iterations, "warmup": warmup, "source": "IProfiler, median per layer",
            "latency": stats(latencies),
            "nodes": {str(ids[n]): t for n, t in medians.items() if n in ids},
            "unmatched": {n: t for n, t in medians.items() if n not in ids},
        },
        "links": links,
        "compile_s": compile_s,
    }
