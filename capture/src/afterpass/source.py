"""Source topology from ONNX or OpenVINO IR: nodes, tensors, shapes and attributes. Weights never enter a bundle."""

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
    """Return the source section of a bundle for an ONNX model or an OpenVINO IR (.xml)."""
    return load_ir(path) if path.suffix == ".xml" else load_onnx(path)


def _json(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    return str(v)


def load_ir(path: Path) -> dict:
    """OpenVINO IR: the .bin is memory mapped by OpenVINO and never read here; constants keep type and shape only."""
    import openvino as ov

    model = ov.Core().read_model(str(path))
    ops = model.get_ordered_ops()
    index = {op.get_friendly_name(): i for i, op in enumerate(ops)}
    tensors, nodes = {}, []
    for i, op in enumerate(ops):
        outputs = []
        for port, out in enumerate(op.outputs()):
            tid = f"{i}:{port}"
            pshape = out.get_partial_shape()
            tensors[tid] = {
                "dtype": out.get_element_type().get_type_name(),
                "shape": [d.get_length() if d.is_static else None for d in pshape] if pshape.rank.is_static else None,
                "names": sorted(out.get_names()),
            }
            outputs.append(tid)
        kind = op.get_type_name()
        if kind == "Parameter":
            tensors[outputs[0]]["kind"] = "input"
        if kind == "Constant":
            attrs = {"element_type": op.get_element_type().get_type_name(), "shape": list(op.get_shape())}
        else:
            attrs = {k: _json(v) for k, v in op.get_attributes().items()}
        nodes.append({
            "id": i,
            "name": op.get_friendly_name(),
            "op": kind,
            "inputs": [f"{index[x.get_source_output().get_node().get_friendly_name()]}:{x.get_source_output().get_index()}" for x in op.inputs()],
            "outputs": outputs,
            "attrs": attrs,
        })
    return {
        "format": "openvino-ir",
        "file": path.name,
        "producer": str(model.get_rt_info()["Runtime_version"]) if model.has_rt_info("Runtime_version") else "",
        "opset": {},
        "inputs": [nodes[index[p.get_friendly_name()]]["outputs"][0] for p in model.get_parameters()],
        "outputs": [nodes[index[r.get_friendly_name()]]["inputs"][0] for r in model.get_results()],
        "nodes": nodes,
        "tensors": tensors,
    }


def load_onnx(path: Path) -> dict:
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
