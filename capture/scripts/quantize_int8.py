"""Quantize an ONNX model to symmetric signed INT8 QDQ, which both OpenVINO and TensorRT accept.

TensorRT takes only signed, zero-centred QDQ outside DLA and no INT32 bias, so this uses ONNX Runtime's quantizer
rather than NNCF and keeps biases in float. TensorRT 11 has no INT8 kernel for ResNet's 3-channel first conv, so
the first conv stays float, as is common.
Calibrates on random normal inputs: the graph the compilers see is right, the accuracy is not.

usage: uv run --with onnxruntime python scripts/quantize_int8.py <in.onnx> <out.onnx> [node to keep float ...]
"""

import sys

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class Random(CalibrationDataReader):
    def __init__(self, name, shape, count=32):
        rng = np.random.default_rng(0)
        self.data = iter([{name: rng.standard_normal(shape).astype(np.float32)} for _ in range(count)])

    def get_next(self):
        return next(self.data, None)


src, dst = sys.argv[1], sys.argv[2]
inp = onnx.load(src, load_external_data=False).graph.input[0]
shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
quantize_static(src, dst, Random(inp.name, shape), quant_format=QuantFormat.QDQ, per_channel=True, nodes_to_exclude=sys.argv[3:],
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False})
print(f"wrote {dst}")
