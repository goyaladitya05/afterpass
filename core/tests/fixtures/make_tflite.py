"""Regenerate toy.tflite: two residual conv blocks and a head, written field by field from the official TFLite schema
(tensorflow/compiler/mlir/lite/schema/schema.fbs, schema version 3) with the standard library only.

It covers what the reader must handle: op codes old style (deprecated_builtin_code only) and new (builtin_code above
127), a custom op, an omitted optional input, a dynamic batch in shape_signature, an int8 weight stored after the
flatbuffer (the offset/size layout of files over 2 GB), an empty data buffer, a second subgraph and metadata.

usage: python3 core/tests/fixtures/make_tflite.py
"""

import struct
from pathlib import Path

HERE = Path(__file__).parent


class Table:
    """Fields by id: (struct format, value) for scalars, or Str, Vec, Data, Table or a list of tables."""

    def __init__(self, fields):
        self.fields = fields


class Str(str):
    pass


class Vec:
    def __init__(self, fmt, items):
        self.fmt, self.items = fmt, items


class Data(bytes):
    pass


class Writer:
    """Lays a flatbuffer out front to back, breadth first: a parent is written before its children, so every offset
    to a child is positive, as the format requires."""

    def __init__(self, ident):
        self.buf = bytearray(8)
        self.buf[4:8] = ident
        self.todo = []
        self.marks = {}

    def pad(self, n, skew=0):
        while (len(self.buf) + skew) % n:
            self.buf.append(0)

    def put(self, fmt, *v):
        self.buf += struct.pack("<" + fmt, *v)

    def place(self, obj):
        if isinstance(obj, Table):
            return self.table(obj)
        if isinstance(obj, list):  # a vector of tables: offsets to each
            self.pad(4)
            pos = len(self.buf)
            self.put("I", len(obj))
            for t in obj:
                self.todo.append((len(self.buf), t))
                self.put("I", 0)
            return pos
        if isinstance(obj, Vec):
            self.pad(max(4, struct.calcsize(obj.fmt)), 4)
            pos = len(self.buf)
            self.put("I", len(obj.items))
            for v in obj.items:
                self.put(obj.fmt, v)
            return pos
        # strings end with a NUL; buffer data is aligned to 16 (force_align)
        self.pad(16, 4)
        pos = len(self.buf)
        self.put("I", len(obj))
        self.buf += obj.encode() + b"\0" if isinstance(obj, Str) else obj
        return pos

    def table(self, t):
        ids = sorted(t.fields)
        # inline layout: the vtable offset, then each field aligned to its size
        layout, size = {}, 4
        for i in ids:
            v = t.fields[i]
            n = struct.calcsize(v[0]) if isinstance(v, tuple) else 4
            size += -size % n
            layout[i] = size
            size += n
        size += -size % 4
        self.pad(2)
        vt = len(self.buf)
        n_slots = ids[-1] + 1 if ids else 0
        self.put("HH", 4 + 2 * n_slots, size)
        for i in range(n_slots):
            self.put("H", layout.get(i, 0))
        self.pad(8)
        pos = len(self.buf)
        self.buf += bytes(size)
        struct.pack_into("<i", self.buf, pos, pos - vt)
        for i in ids:
            v, at = t.fields[i], pos + layout[i]
            if isinstance(v, tuple):
                struct.pack_into("<" + v[0], self.buf, at, v[1])
                if len(v) > 2:  # a value to fill in later, named
                    self.marks[v[2]] = at
            else:
                self.todo.append((at, v))
        return pos

    def finish(self, root):
        self.todo.append((0, root))
        while self.todo:
            at, obj = self.todo.pop(0)
            pos = self.place(obj)
            struct.pack_into("<I", self.buf, at, pos - at)
        return self.buf


def tensor(name, shape, dtype=0, buffer=0, signature=None):
    # Tensor: shape 0, type 1, buffer 2, name 3, shape_signature 7
    f = {0: Vec("i", shape), 1: ("b", dtype), 2: ("I", buffer), 3: Str(name)}
    if signature:
        f[7] = Vec("i", signature)
    return Table(f)


def op(code, inputs, outputs, options=None):
    # Operator: opcode_index 0, inputs 1, outputs 2, builtin_options_type 3, builtin_options 4
    f = {0: ("I", code), 1: Vec("i", inputs), 2: Vec("i", outputs)}
    if options:
        f[3], f[4] = ("B", options[0]), Table(options[1])
    return Table(f)


def model():
    FLOAT32, INT32, INT8 = 0, 2, 9
    RELU = ("b", 1)
    b = 16  # channels: each conv kernel is 9 KiB, more than two 4 KiB pages
    tensors = [tensor("serving_default_input:0", [1, 8, 8, b], signature=[-1, 8, 8, b])]
    ops = []
    x = 0
    for i, (w, bias, empty) in enumerate([(1, 2, 7), (3, 4, 0)]):
        p = f"model/block_{i}"
        k = len(tensors)
        tensors += [tensor(f"{p}/conv/Conv2D", [b, 3, 3, b], buffer=w),
                    tensor(f"{p}/conv/BiasAdd/ReadVariableOp", [b], buffer=bias),
                    tensor(f"{p}/conv/Relu;{p}/conv/BiasAdd;{p}/conv/Conv2D", [1, 8, 8, b], signature=[-1, 8, 8, b]),
                    tensor(f"{p}/add/Add", [1, 8, 8, b], buffer=empty, signature=[-1, 8, 8, b])]
        # Conv2DOptions (union type 1): padding (0) left at SAME, stride_w (1), stride_h (2), activation (3)
        ops += [op(0, [x, k, k + 1], [k + 2], (1, {1: ("i", 1), 2: ("i", 1), 3: RELU})),
                op(1, [k + 2, x], [k + 3], (11, {}))]  # AddOptions, activation left at NONE
        x = k + 3
    k = len(tensors)
    tensors += [tensor("model/head/Mean/reduction_indices", [2], INT32, buffer=5),
                tensor("model/head/Mean", [1, b]),
                tensor("model/head/dense/MatMul", [4, b], INT8, buffer=6),
                tensor("model/head/dense/MatMul;model/head/dense/BiasAdd", [1, 4]),
                tensor("model/head/gelu/Gelu", [1, 4]),
                tensor("StatefulPartitionedCall:0", [1, 4])]
    ops += [op(2, [x, k], [k + 1], (27, {})),  # ReducerOptions, keep_dims false
            op(3, [k + 1, k + 2, -1], [k + 3], (8, {})),  # FullyConnectedOptions; no bias
            op(4, [k + 3], [k + 4], (116, {0: ("?", True)})),  # GeluOptions approximate
            op(5, [k + 4], [k + 5])]
    main = Table({0: tensors, 1: Vec("i", [0]), 2: Vec("i", [k + 5]), 3: ops, 4: Str("main")})
    other = Table({0: [tensor("x", [4]), tensor("y", [4])], 1: Vec("i", [0]), 2: Vec("i", [1]),
                   3: [op(1, [0, 0], [1])], 4: Str("decode")})
    # OperatorCode: deprecated_builtin_code 0, custom_code 1, version 2, builtin_code 3
    codes = [Table({0: ("b", 3)}),  # CONV_2D, written before builtin_code existed
             Table({}),  # ADD: code 0, the default
             Table({0: ("b", 40), 3: ("i", 40)}),  # MEAN
             Table({0: ("b", 9), 3: ("i", 9)}),  # FULLY_CONNECTED
             Table({0: ("b", 127), 3: ("i", 150)}),  # GELU, above PLACEHOLDER_FOR_GREATER_OP_CODES
             Table({0: ("b", 32), 1: Str("AfterpassToyOp"), 3: ("i", 32)})]  # CUSTOM
    # TFLite Support metadata (metadata_schema.fbs): ModelMetadata name 0, version 2, author 4, license 5
    meta = Writer(b"M001").finish(Table({0: Str("afterpass toy"), 2: Str("1"), 4: Str("afterpass"), 5: Str("Apache-2.0")}))
    zeros = lambda n: Data(bytes(n))
    # Buffer: data 0, offset 1, size 2; buffer 0 is the empty sentinel
    buffers = [Table({}), Table({0: zeros(b * 9 * b * 4)}), Table({0: zeros(b * 4)}), Table({0: zeros(b * 9 * b * 4)}),
               Table({0: zeros(b * 4)}), Table({0: Data(struct.pack("<2i", 1, 2))}),
               Table({1: ("Q", 0, "offset"), 2: ("Q", 4 * b)}),  # the int8 dense weight, after the flatbuffer
               Table({0: Data(b"")}), Table({0: Data(b"1.14.0".ljust(16, b"\0"))}), Table({0: Data(bytes(meta))})]
    # Model: version 0, operator_codes 1, subgraphs 2, description 3, buffers 4, metadata 6 (Metadata: name 0, buffer 1)
    w = Writer(b"TFL3")
    out = w.finish(Table({0: ("I", 3), 1: codes, 2: [main, other], 3: Str("afterpass fixture"), 4: buffers,
                          6: [Table({0: Str("min_runtime_version"), 1: ("I", 8)}),
                              Table({0: Str("TFLITE_METADATA"), 1: ("I", 9)})]}))
    w.pad(16)
    struct.pack_into("<Q", out, w.marks["offset"], len(out))
    return out + bytes(4 * b)


if __name__ == "__main__":
    (HERE / "toy.tflite").write_bytes(model())
