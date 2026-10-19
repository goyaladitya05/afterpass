import json
import subprocess
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

from afterpass import bundle


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Conv + BN + Relu + residual Add + Reshape + MatMul, exercising fusion, reorders and graph-output naming."""
    r = np.random.default_rng(0).random
    # 16 channels: a multiple of every NCHWc block size ONNX Runtime uses on x86 (8 with AVX2, 16 with AVX-512)
    inits = [nh.from_array(r(s).astype("float32"), n) for n, s in
             [("w", (16, 3, 3, 3)), ("b", (16,)), ("s", (16,)), ("bb", (16,)), ("m", (16,)), ("fcw", (4096, 10))]]
    inits += [nh.from_array(r(16).astype("float32") + 1, "v"), nh.from_array(np.array([1, -1], "int64"), "shp")]
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["c"], name="conv1", pads=[1, 1, 1, 1]),
             h.make_node("BatchNormalization", ["c", "s", "bb", "m", "v"], ["bn"], name="bn1"),
             h.make_node("Relu", ["bn"], ["r"], name="relu1"),
             h.make_node("Add", ["r", "c"], ["a"], name="add1"),
             h.make_node("Reshape", ["a", "shp"], ["f"], name="flat"),
             h.make_node("MatMul", ["f", "fcw"], ["y"], name="fc")]
    g = h.make_graph(nodes, "tiny", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 16, 16])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 10])], inits)
    d = tmp_path_factory.mktemp("tiny")
    # IR 10: ONNX Runtime 1.31 refuses the IR 14 that onnx 1.23 writes by default
    onnx.save(h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=10), d / "tiny.onnx")
    return d


def run_capture(d, runtime, device):
    out = d / f"{runtime}.json.gz"
    subprocess.run([sys.executable, "-m", "afterpass", "capture", str(d / "tiny.onnx"), "--runtime", runtime,
                    "--device", device, "--iterations", "3", "--warmup", "1", "-o", str(out)], check=True)
    return bundle.read(out)


def check_invariants(b):
    assert b["schema_version"] == 0
    assert len(b["source"]["nodes"]) == 6
    assert not any("raw_data" in str(n) for n in b["source"]["nodes"])
    compiled_ids = {n["id"] for n in b["compiled"]["nodes"]}
    assert {l["compiled"] for l in b["provenance"]} == compiled_ids
    for l in b["provenance"]:
        assert l["grade"] in ("explicit", "inferred", "unknown")
        assert (l["source"] is None) == (l["grade"] == "unknown")
    assert set(b["profile"]["nodes"]) <= {str(i) for i in compiled_ids}
    assert b["profile"]["latency"]["median"] > 0
    assert b["capture"]["command"] == f"capture tiny.onnx --runtime {b['capture']['runtime']} --device {b['capture']['device']} --iterations 3 --warmup 1"


def explicit_sources(b):
    return {b["source"]["nodes"][l["source"]]["name"] for l in b["provenance"] if l["grade"] == "explicit"}


def test_openvino_cpu(tiny):
    b = run_capture(tiny, "openvino", "CPU")
    check_invariants(b)
    assert {"conv1", "bn1", "relu1", "add1"} <= explicit_sources(b)
    reorders = [n["id"] for n in b["compiled"]["nodes"] if n["op"] == "Reorder"]
    assert reorders and all(l["grade"] == "unknown" for l in b["provenance"] if l["compiled"] in reorders)
    # the MatMul is renamed after its graph output tensor, so its link can only be inferred
    assert any(l["source"] == 5 and l["grade"] == "inferred" and l["via"] == "tensor" for l in b["provenance"])


def test_onnxruntime_cpu(tiny):
    b = run_capture(tiny, "onnxruntime", "CPU")
    check_invariants(b)
    nodes = b["compiled"]["nodes"]
    # every node is timed, carries its provider, and has the shape the run produced
    assert set(b["profile"]["nodes"]) == {str(n["id"]) for n in nodes} and not b["profile"]["unmatched"]
    assert all(n["attrs"]["provider"] == "CPUExecutionProvider" and n["kernel"].startswith("CPU ") for n in nodes)
    assert all(b["compiled"]["tensors"][t]["shape"] for n in nodes for t in n["outputs"])
    nchwc = [n for n in nodes if n["attrs"].get("domain") == "com.microsoft.nchwc"]
    if not nchwc:  # no NCHWc kernels on this CPU (not x86): nothing is fused
        assert explicit_sources(b) == {"conv1", "bn1", "relu1", "add1", "flat", "fc"}
        return
    assert explicit_sources(b) == {"add1", "flat", "fc"}
    # the NCHWc Conv is named after the source Conv's output tensor ("c_nchwc"), so its link is inferred
    assert any(l["source"] == 0 and l["grade"] == "inferred" and l["via"] == "tensor" for l in b["provenance"])
    reorders = [n["id"] for n in nchwc if n["op"] == "ReorderOutput"]
    assert reorders and all(l["grade"] == "unknown" for l in b["provenance"] if l["compiled"] in reorders)
    assert all(b["compiled"]["tensors"][n["outputs"][0]].get("layout") == "NCHWc" for n in nchwc if n["op"] == "Conv")


def test_onnxruntime_profile_events():
    from afterpass import ort
    g = h.make_graph([h.make_node("Relu", ["x"], ["a"], name="r"), h.make_node("Exp", ["a"], ["b"]),
                      h.make_node("Neg", ["b"], ["y"])], "g", [], [])

    def run(ts):  # as traced: the profiler names an unnamed node "<op type>_<internal index>"
        ev = [("r", "Relu"), ("Exp_7", "Exp"), ("Neg_9", "Neg")]
        return [{"cat": "Session", "name": "model_run", "ts": ts, "dur": 10}] + [
            {"cat": "Node", "name": f"{n}_kernel_time", "ts": ts + k + 1, "dur": 1, "args": {"op_name": op}} for k, (n, op) in enumerate(ev)]

    runs = ort._runs(run(200) + run(0) + run(100), warmup=1)
    assert [[e["ts"] for e in r] for r in runs] == [[101, 102, 103], [201, 202, 203]]
    assert ort._event_ids(g, runs[0]) == {"r": 0, "Exp_7": 1, "Neg_9": 2}


def test_tensorrt(tiny):
    pytest.importorskip("tensorrt")
    b = run_capture(tiny, "tensorrt", "GPU")
    check_invariants(b)
    assert explicit_sources(b) == {"conv1", "bn1", "relu1", "add1", "fc"}


def test_tensorrt_metadata_claims():
    trt = pytest.importorskip("afterpass.trt")
    # as emitted by TensorRT 11 for a Conv fused with its Relu, and for two separate records
    assert trt._claims("[ONNX Layer: /conv1/Conv]\x1e[ONNX Layer: /relu/Relu]") == ["/conv1/Conv", "/relu/Relu"]
    assert trt._claims("[ONNX Layer: a]\x1f[ONNX Layer: b]") == ["a", "b"]
    assert trt._claims("") == []


@pytest.fixture(scope="module")
def stateful(tmp_path_factory):
    """A KV-cache-like state that grows by the new tokens on every call, saved as an IR like an optimum-intel LLM."""
    import openvino as ov
    import openvino.opset13 as ops
    ids = ops.parameter([1, -1], ov.Type.i64, name="input_ids")
    mask = ops.parameter([1, -1], ov.Type.i64, name="attention_mask")
    info = ov.op.util.VariableInfo()
    info.data_shape, info.data_type, info.variable_id = ov.PartialShape([1, -1]), ov.Type.f32, "cache"
    var = ov.op.util.Variable(info)
    past = ops.read_value(ops.broadcast(ops.constant(np.float32(0)), ops.constant(np.array([1, 0], "int64"))), var)
    cache = ops.concat([past, ops.convert(ids, ov.Type.f32)], 1)
    axis = ops.constant(np.array([1], "int64"))
    out = ops.add(ops.reduce_sum(cache, axis, True), ops.convert(ops.reduce_sum(mask, axis, True), ov.Type.f32))
    path = tmp_path_factory.mktemp("stateful") / "llm.xml"
    ov.save_model(ov.Model([ops.result(out)], [ops.assign(cache, var)], [ids, mask], "llm"), path)
    return path


def test_openvino_decode(stateful, tiny):
    out = stateful.parent / "decode.json.gz"
    subprocess.run([sys.executable, "-m", "afterpass", "capture", str(stateful), "--runtime", "openvino", "--device", "CPU",
                    "--shape", "input_ids=1,8", "--shape", "attention_mask=1,8", "--iterations", "2", "--warmup", "1",
                    "--decode", "4", "-o", str(out)], check=True)
    p = bundle.read(out)["profile"]
    assert (p["phase"], p["prompt_tokens"], p["decode_steps"]) == ("decode", 8, 4)
    assert p["latency"]["median"] > 0 and p["nodes"]
    assert "?" in bundle.read(out)["compiled"]["input_shapes"]["input_ids"]  # compiled dynamic, as GenAI does
    static = stateful.parent / "static.json.gz"
    subprocess.run([sys.executable, "-m", "afterpass", "capture", str(stateful), "--runtime", "openvino", "--device", "CPU",
                    "--shape", "input_ids=1,8", "--shape", "attention_mask=1,8", "--iterations", "2", "--warmup", "1",
                    "--static", "-o", str(static)], check=True)
    assert bundle.read(static)["compiled"]["input_shapes"]["input_ids"] == "[1,8]"
    # a model without a KV cache has no decode phase
    r = subprocess.run([sys.executable, "-m", "afterpass", "capture", str(tiny / "tiny.onnx"), "--runtime", "openvino",
                        "--decode", "4", "-o", str(tiny / "x.json.gz")], capture_output=True, text=True)
    assert r.returncode != 0 and "--decode needs a stateful LLM IR" in r.stderr


def test_onnxruntime_fused_sum_output():
    ort = pytest.importorskip("afterpass.ort")
    from afterpass.provenance import Linker
    # source: c = Conv(x); a = Add(c, skip); r = Relu(a). ORT fuses all three into one NCHWc Conv named after c.
    src = {"nodes": [{"id": 0, "name": "conv", "op": "Conv", "inputs": ["x"], "outputs": ["c"]},
                     {"id": 1, "name": "add", "op": "Add", "inputs": ["c", "skip"], "outputs": ["a"]},
                     {"id": 2, "name": "relu", "op": "Relu", "inputs": ["a"], "outputs": ["r"]}]}
    node = h.make_node("Conv", ["t0", "w", "b", "t1"], ["t2"], name="c_nchwc", domain="com.microsoft.nchwc", activation="Relu")
    tensors = {"t1": {"names": ["skip"]}}
    assert ort._fused_sum_output(node, tensors, Linker(src)) == "r"
    # the activation the node names is not in the source graph: no name rather than a wrong one
    src["nodes"][2]["op"] = "Sigmoid"
    assert ort._fused_sum_output(node, tensors, Linker(src)) is None


def gpu_providers():
    return pytest.importorskip("onnxruntime").get_available_providers()


@pytest.mark.parametrize("device", ["CUDA", "TensorRT"])
def test_onnxruntime_gpu(tiny, device):
    if "CUDAExecutionProvider" not in gpu_providers() or not bundle._gpus():
        pytest.skip("needs onnxruntime-gpu and an NVIDIA GPU")
    b = run_capture(tiny, "onnxruntime", device)
    check_invariants(b)
    nodes = b["compiled"]["nodes"]
    assert set(b["profile"]["nodes"]) == {str(n["id"]) for n in nodes} and not b["profile"]["unmatched"]
    assert b["profile"]["profiled_latency"]["median"] > 0 and "CUDA_LAUNCH_BLOCKING=1" in b["profile"]["source"]
    assert explicit_sources(b) == {"conv1", "bn1", "relu1", "add1", "flat", "fc"}
    if device == "TensorRT":  # one engine computes the whole model
        assert [(n["op"], n["attrs"]["provider"]) for n in nodes] == [("TRTKernel", "TensorrtExecutionProvider")]


def test_onnxruntime_device_errors(tiny):
    def capture(device):
        return subprocess.run([sys.executable, "-m", "afterpass", "capture", str(tiny / "tiny.onnx"), "--runtime", "onnxruntime",
                               "--device", device, "-o", str(tiny / "x.json.gz")], capture_output=True, text=True)
    r = capture("GPU")
    assert r.returncode != 0 and "support --device CPU, CUDA, TensorRT (got GPU)" in r.stderr
    if "CUDAExecutionProvider" not in gpu_providers():  # the CPU package: say which one is needed
        r = capture("CUDA")
        assert r.returncode != 0 and "need onnxruntime-gpu" in r.stderr


def test_onnxruntime_provider_not_loaded():
    ort = pytest.importorskip("afterpass.ort")

    class Session:  # ONNX Runtime fell back silently: TensorRT's libraries were missing
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort._check(Session(), "CUDA")
    with pytest.raises(SystemExit, match="could not load TensorrtExecutionProvider and would run on CUDAExecutionProvider.*TensorRT 10"):
        ort._check(Session(), "TensorRT")


def test_onnxruntime_tensorrt_engines(tmp_path, monkeypatch):
    """A TensorRT capture assembled from a recorded run: the provider took conv1 and the NonZero and Shape after it, and
    trt_op_types_to_exclude=Relu left relu1 to CUDA (RTX 4050, onnxruntime-gpu 1.31.0, TensorRT 10.16)."""
    ort = pytest.importorskip("afterpass.ort")
    from afterpass import source
    from afterpass.provenance import Linker
    w = nh.from_array(np.zeros((16, 3, 3, 3), "float32"), "w")
    src = h.make_graph([h.make_node("Conv", ["x", "w"], ["c"], name="conv1", pads=[1, 1, 1, 1]), h.make_node("Relu", ["c"], ["r"], name="relu1"),
                        h.make_node("NonZero", ["r"], ["nz"], name="nonzero1"), h.make_node("Shape", ["nz"], ["y"], name="shape1")],
                       "t", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 16, 16])], [h.make_tensor_value_info("y", T.INT64, [2])], [w])
    onnx.save(h.make_model(src, opset_imports=[h.make_opsetid("", 17)], ir_version=10), tmp_path / "t.onnx")
    engine = "TensorrtExecutionProvider_TRTKernel_graph_t_5127982078059117420_{0}_{0}"
    # as ModelCompiler saves it: each engine one node, named as in the trace
    compiled = h.make_graph([h.make_node("TRTKernel_graph_t_5127982078059117420_0", ["x"], ["c"], name=engine.format(0), domain="com.microsoft"),
                             h.make_node("Relu", ["c"], ["r"], name="relu1"),
                             h.make_node("TRTKernel_graph_t_5127982078059117420_1", ["r"], ["y"], name=engine.format(1), domain="com.microsoft")],
                            "t", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 16, 16])], [h.make_tensor_value_info("y", T.INT64, [2])])
    ran = [(engine.format(0), "TensorrtExecutionProvider", {"float": [1, 16, 16, 16]}), ("relu1", "CUDAExecutionProvider", {"float": [1, 16, 16, 16]}),
           (engine.format(1), "TensorrtExecutionProvider", {"int64": [2]})]
    trace = [e for ts in (0, 100) for e in [{"cat": "Session", "name": "model_run", "ts": ts, "dur": 50}] + [
        {"cat": "Node", "name": f"{n}_kernel_time", "ts": ts + k + 1, "dur": 10, "args": {"provider": p, "output_type_shape": [s]}}
        for k, (n, p, s) in enumerate(ran)]]

    def measure(model_path, device, shapes, iterations, warmup, tmp):
        onnx.save(h.make_model(compiled, opset_imports=[h.make_opsetid("", 17)], ir_version=10), tmp / "optimized.onnx")
        (tmp / "trace.json").write_text(json.dumps(trace))
        return 1.0, [30.0], {"trace": str(tmp / "trace.json"), "providers": ort.PROVIDERS["TensorRT"], "options": {},
                             "assignment": [["TensorrtExecutionProvider", ["conv1"]], ["TensorrtExecutionProvider", ["nonzero1", "shape1"]],
                                            ["CUDAExecutionProvider", ["relu1"]]]}
    monkeypatch.setattr(ort, "_measure", measure)
    monkeypatch.setattr(ort.ort, "get_available_providers", lambda: ort.PROVIDERS["TensorRT"])
    r = ort.capture(tmp_path / "t.onnx", "TensorRT", {}, 1, 1, Linker(source.load(tmp_path / "t.onnx")), work_dir=tmp_path)
    nodes = r["compiled"]["nodes"]
    assert [(n["op"], n["kernel"], n["claims"][1:]) for n in nodes] == [
        ("TRTKernel", "Tensorrt TRTKernel", ["conv1"]), ("Relu", "CUDA Relu", ["r"]), ("TRTKernel", "Tensorrt TRTKernel", ["nonzero1", "shape1"])]
    assert {(l["compiled"], l["source"], l["grade"]) for l in r["links"]} == {(0, 0, "explicit"), (1, 1, "explicit"), (2, 2, "explicit"), (2, 3, "explicit")}
    assert r["profile"]["nodes"] == {"0": 10.0, "1": 10.0, "2": 10.0} and r["profile"]["profiled_latency"]["median"] == 50
    assert r["compiled"]["tensors"]["y"]["dtype"] == "int64"
    # an engine number past the assignment claims nothing rather than another engine's nodes
    assert ort._engine_nodes(engine.format(2), [["a"], ["b"]]) == []
