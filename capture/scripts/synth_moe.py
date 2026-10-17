"""Synthetic MoE source graph with unrolled experts, for the "30B MoE with unrolled experts" load budget.

Topology follows Qwen3-30B-A3B (48 layers, 128 experts, top-8 routing) exported naively, one subgraph per
expert, with legacy-exporter node names. Weights are tiny (hidden size 8): only the topology matters.

usage: uv run python scripts/synth_moe.py <out.onnx> [layers] [experts]
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

D, TOKENS, TOPK = 8, 4, 8


def build(layers: int, experts: int) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    nodes, inits = [], []

    def w(name, *shape):
        inits.append(nh.from_array(rng.random(shape, dtype=np.float32), name))
        return name

    def op(kind, name, inputs, outputs=None, **attrs):
        outputs = outputs or [f"{name}_output_0"]
        nodes.append(h.make_node(kind, inputs, outputs, name=name, **attrs))
        return outputs[0]

    inits.append(nh.from_array(np.array([2.0], np.float32), "two"))
    inits.append(nh.from_array(np.array([1e-6], np.float32), "eps"))
    inits.append(nh.from_array(np.array([TOPK], np.int64), "k"))
    x = op("Gather", "/model/embed_tokens/Gather", [w("embed.weight", 64, D), "ids"])

    def norm(p, x):
        y = op("Pow", f"{p}/Pow", [x, "two"])
        y = op("ReduceMean", f"{p}/ReduceMean", [y], axes=[-1], keepdims=1)
        y = op("Add", f"{p}/Add", [y, "eps"])
        y = op("Sqrt", f"{p}/Sqrt", [y])
        y = op("Div", f"{p}/Div", [x, y])
        return op("Mul", f"{p}/Mul", [y, w(f"{p}.weight", D)])

    for l in range(layers):
        p = f"/model/layers.{l}"
        hdn = norm(f"{p}/input_layernorm", x)
        q = op("MatMul", f"{p}/self_attn/q_proj/MatMul", [hdn, w(f"{p}.q", D, D)])
        k = op("MatMul", f"{p}/self_attn/k_proj/MatMul", [hdn, w(f"{p}.k", D, D)])
        v = op("MatMul", f"{p}/self_attn/v_proj/MatMul", [hdn, w(f"{p}.v", D, D)])
        kt = op("Transpose", f"{p}/self_attn/Transpose", [k], perm=[1, 0])
        s = op("MatMul", f"{p}/self_attn/MatMul", [q, kt])
        s = op("Softmax", f"{p}/self_attn/Softmax", [s], axis=-1)
        a = op("MatMul", f"{p}/self_attn/MatMul_1", [s, v])
        a = op("MatMul", f"{p}/self_attn/o_proj/MatMul", [a, w(f"{p}.o", D, D)])
        x = op("Add", f"{p}/Add", [x, a])
        hdn = norm(f"{p}/post_attention_layernorm", x)
        logits = op("MatMul", f"{p}/mlp/gate/MatMul", [hdn, w(f"{p}.router", D, experts)])
        probs = op("Softmax", f"{p}/mlp/Softmax", [logits], axis=-1)
        topv, topi = f"{p}/mlp/TopK_output_0", f"{p}/mlp/TopK_output_1"
        op("TopK", f"{p}/mlp/TopK", [probs, "k"], [topv, topi], axis=-1)
        acc = None
        for e in range(experts):
            q = f"{p}/mlp/experts.{e}"
            inits.append(nh.from_array(np.array([e], np.int64), f"{q}.id"))
            m = op("Equal", f"{q}/Equal", [topi, f"{q}.id"])
            m = op("Cast", f"{q}/Cast", [m], to=T.FLOAT)
            m = op("Mul", f"{q}/Mul", [m, topv])
            m = op("ReduceSum", f"{q}/ReduceSum", [m], axes=[-1], keepdims=1)
            g = op("MatMul", f"{q}/gate_proj/MatMul", [hdn, w(f"{q}.gate", D, D)])
            g = op("Mul", f"{q}/act_fn/Mul", [g, op("Sigmoid", f"{q}/act_fn/Sigmoid", [g])])
            u = op("MatMul", f"{q}/up_proj/MatMul", [hdn, w(f"{q}.up", D, D)])
            y = op("MatMul", f"{q}/down_proj/MatMul", [op("Mul", f"{q}/Mul_1", [g, u]), w(f"{q}.down", D, D)])
            y = op("Mul", f"{q}/Mul_2", [y, m])
            acc = y if acc is None else op("Add", f"{q}/Add", [acc, y])
        x = op("Add", f"{p}/Add_1", [x, acc])
    x = norm("/model/norm", x)
    op("MatMul", "/lm_head/MatMul", [x, w("lm_head.weight", D, 64)], ["logits"])
    g = h.make_graph(nodes, "synthetic_moe", [h.make_tensor_value_info("ids", T.INT64, [TOKENS])],
                     [h.make_tensor_value_info("logits", T.FLOAT, [TOKENS, 64])], inits)
    return h.make_model(g, opset_imports=[h.make_opsetid("", 13)], producer_name="afterpass-synth")


if __name__ == "__main__":
    out = sys.argv[1]
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    experts = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    m = build(layers, experts)
    onnx.save(m, out)
    print(f"wrote {out}: {len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers")
