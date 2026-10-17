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
    inits = [nh.from_array(r(s).astype("float32"), n) for n, s in
             [("w", (8, 3, 3, 3)), ("b", (8,)), ("s", (8,)), ("bb", (8,)), ("m", (8,)), ("fcw", (2048, 10))]]
    inits += [nh.from_array(r(8).astype("float32") + 1, "v"), nh.from_array(np.array([1, -1], "int64"), "shp")]
    nodes = [h.make_node("Conv", ["x", "w", "b"], ["c"], name="conv1", pads=[1, 1, 1, 1]),
             h.make_node("BatchNormalization", ["c", "s", "bb", "m", "v"], ["bn"], name="bn1"),
             h.make_node("Relu", ["bn"], ["r"], name="relu1"),
             h.make_node("Add", ["r", "c"], ["a"], name="add1"),
             h.make_node("Reshape", ["a", "shp"], ["f"], name="flat"),
             h.make_node("MatMul", ["f", "fcw"], ["y"], name="fc")]
    g = h.make_graph(nodes, "tiny", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 16, 16])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 10])], inits)
    d = tmp_path_factory.mktemp("tiny")
    onnx.save(h.make_model(g, opset_imports=[h.make_opsetid("", 17)]), d / "tiny.onnx")
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
