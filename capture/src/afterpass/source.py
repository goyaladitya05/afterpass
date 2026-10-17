"""ONNX source topology: nodes, tensors, shapes and attributes. Weights are never read."""

from pathlib import Path

import onnx
from onnx import AttributeProto as A
from onnx import TensorProto

SMALL_TENSOR = 16  # constant attributes up to this many elements keep their values


def dtype_name(elem_type: int) -> str:
    return TensorProto.DataType.Name(elem_type).lower() if elem_type else "unknown"


def _dims(shape) -> list:
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or None) for d in shape.dim]


def _attr(a: onnx.AttributeProto):
    if a.type == A.TENSOR:
        t = a.t
        out = {"dtype": dtype_name(t.data_type), "shape": list(t.dims)}
        numel = 1
        for d in t.dims:
            numel *= d
        if numel <= SMALL_TENSOR and t.data_location != TensorProto.EXTERNAL:
            out["value"] = onnx.numpy_helper.to_array(t).tolist()
        return out
    if a.type in (A.GRAPH, A.GRAPHS):
        graphs = [a.g] if a.type == A.GRAPH else list(a.graphs)
        return {"subgraphs": [len(g.node) for g in graphs]}
    value = onnx.helper.get_attribute_value(a)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, list):
        return [v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in value]
    return value


def load(path: Path) -> dict:
    """Return the source section of a bundle for an ONNX model."""
    model = onnx.load(str(path), load_external_data=False)
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception as e:  # shape inference is best effort
        print(f"warning: ONNX shape inference failed: {e}")
    g = model.graph

    tensors: dict[str, dict] = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        tt = vi.type.tensor_type
        tensors[vi.name] = {"dtype": dtype_name(tt.elem_type), "shape": _dims(tt.shape) if tt.HasField("shape") else None}
    for init in g.initializer:
        tensors[init.name] = {"dtype": dtype_name(init.data_type), "shape": list(init.dims), "kind": "initializer"}
    init_names = {i.name for i in g.initializer}
    for vi in g.input:
        if vi.name not in init_names:
            tensors[vi.name]["kind"] = "input"
    for vi in g.output:
        tensors[vi.name]["kind"] = "output"

    nodes = []
    for i, n in enumerate(g.node):
        for t in list(n.input) + list(n.output):
            if t:
                tensors.setdefault(t, {"dtype": "unknown", "shape": None})
        nodes.append({
            "id": i,
            "name": n.name,
            "op": n.op_type if n.domain in ("", "ai.onnx") else f"{n.domain}::{n.op_type}",
            "inputs": list(n.input),
            "outputs": list(n.output),
            "attrs": {a.name: _attr(a) for a in n.attribute},
        })

    return {
        "format": "onnx",
        "file": path.name,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "opset": {(o.domain or "ai.onnx"): o.version for o in model.opset_import},
        "inputs": [vi.name for vi in g.input if vi.name not in init_names],
        "outputs": [vi.name for vi in g.output],
        "nodes": nodes,
        "tensors": tensors,
    }
