"""Regenerate the parser fixtures: a 3-layer toy model whose layers.2 is an outlier.

usage (from capture/): uv run --extra tensorrt python ../core/tests/fixtures/make_fixtures.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

HERE = Path(__file__).parent
D = 16


def model() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    nodes, inits = [], [nh.from_array(rng.random((32, D), dtype=np.float32), "embed.weight")]
    nodes.append(h.make_node("Gather", ["embed.weight", "ids"], ["/embed/Gather_output_0"], name="/embed/Gather"))
    x = "/embed/Gather_output_0"
    for i in range(3):
        p = f"/model/layers.{i}"
        inits += [nh.from_array(rng.random((D, D), dtype=np.float32), f"layers.{i}.w"),
                  nh.from_array(rng.random(D, dtype=np.float32), f"layers.{i}.b")]
        nodes += [h.make_node("MatMul", [x, f"layers.{i}.w"], [f"{p}/mm"], name=f"{p}/fc/MatMul"),
                  h.make_node("Add", [f"{p}/mm", f"layers.{i}.b"], [f"{p}/bias"], name=f"{p}/fc/Add"),
                  h.make_node("Relu", [f"{p}/bias"], [f"{p}/act"], name=f"{p}/act/Relu")]
        y = f"{p}/act"
        if i == 2:  # the outlier: one extra op
            nodes.append(h.make_node("Mul", [y, y], [f"{p}/sq"], name=f"{p}/act/Mul"))
            y = f"{p}/sq"
        nodes.append(h.make_node("Add", [x, y], [f"{p}/out"], name=f"{p}/Add"))
        x = f"{p}/out"
    inits.append(nh.from_array(rng.random((D, 4), dtype=np.float32), "head.weight"))
    nodes.append(h.make_node("MatMul", [x, "head.weight"], ["logits"], name="/head/MatMul"))
    g = h.make_graph(nodes, "toy", [h.make_tensor_value_info("ids", T.INT64, [1, 8])],
                     [h.make_tensor_value_info("logits", T.FLOAT, [1, 8, 4])], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)], producer_name="afterpass-fixture")
    return onnx.shape_inference.infer_shapes(m)


if __name__ == "__main__":
    onnx.save(model(), HERE / "toy.onnx")
    ov.save_model(ov.Core().read_model(HERE / "toy.onnx"), HERE / "toy.xml", compress_to_fp16=False)
    (HERE / "toy.bin").unlink()
    for runtime, device in (("openvino", "CPU"), ("tensorrt", "GPU")):
        out = HERE / f"toy.{runtime}-{device.lower()}.afterpass.json.gz"
        subprocess.run([sys.executable, "-m", "afterpass", "capture", str(HERE / "toy.onnx"), "--runtime", runtime,
                        "--device", device, "--iterations", "5", "--warmup", "1", "-o", str(out)], check=True)
