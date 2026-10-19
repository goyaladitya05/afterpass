//! TFLite reader. A .tflite file is one FlatBuffer: tables that point at their fields and children by offsets, so the
//! reader follows offsets through ranged reads and only fetches the pages it lands on. Weight data is never read: a
//! buffer's table and the length word of its data vector are enough to know that a tensor holds a constant.
//!
//! Field ids follow the official schema, tensorflow/compiler/mlir/lite/schema/schema.fbs (schema version 3, file
//! identifier "TFL3"; checked against commit 5442ef5 of 2026-09-25). A field's id is its position in its table, a
//! union counting twice (its type, then its value).

use std::collections::HashMap;

use crate::ir::{Graph, Node, Tensor, NONE};
use crate::onnx::ByteSource;

const PAGE: u64 = 4096;
/// Larger strings or vectors, or vectors of more tables, are refused: only a damaged file has them, and allocating
/// them could exhaust wasm memory.
const MAX_READ: u64 = 64 << 20;
const MAX_TABLES: u64 = 1 << 24;

/// The file through a page cache: flatbuffers are read in small scattered pieces (a table, its vtable, a string).
struct Reader<'a, S: ByteSource> {
    src: &'a mut S,
    len: u64,
    pages: HashMap<u64, Vec<u8>>,
}

/// A table: where it starts, and where its vtable (the field offsets) starts and how many bytes it has.
#[derive(Clone, Copy)]
struct Table {
    pos: u64,
    vtable: u64,
    vlen: u64,
}

impl<S: ByteSource> Reader<'_, S> {
    fn bytes(&mut self, pos: u64, n: u64) -> Result<Vec<u8>, String> {
        if pos.checked_add(n).is_none_or(|end| end > self.len) {
            return Err(format!("offset {pos} runs past the end of the file"));
        }
        if n > MAX_READ {
            return Err(format!("a {n}-byte field at {pos}"));
        }
        let mut out = Vec::with_capacity(n as usize);
        let mut p = pos;
        while p < pos + n {
            let page = p / PAGE;
            if !self.pages.contains_key(&page) {
                let mut buf = vec![0; PAGE.min(self.len - page * PAGE) as usize];
                self.src.read(page * PAGE, &mut buf)?;
                self.pages.insert(page, buf);
            }
            let data = &self.pages[&page];
            let o = (p - page * PAGE) as usize;
            let take = (data.len() - o).min((pos + n - p) as usize);
            out.extend_from_slice(&data[o..o + take]);
            p += take as u64;
        }
        Ok(out)
    }

    /// A little-endian unsigned integer of `size` bytes.
    fn uint(&mut self, pos: u64, size: u64) -> Result<u64, String> {
        Ok(self.bytes(pos, size)?.iter().rev().fold(0, |v, &b| v << 8 | b as u64))
    }

    /// Follow the unsigned 32-bit offset stored at `at`, which counts from `at` itself.
    fn offset(&mut self, at: u64) -> Result<u64, String> {
        Ok(at + self.uint(at, 4)?)
    }

    /// A table starts with a signed offset back (or forward) to its vtable: u16 vtable size, u16 table size, then one
    /// u16 per field, the field's offset in the table or 0 when the field is absent.
    fn table(&mut self, pos: u64) -> Result<Table, String> {
        let vtable = pos as i64 - self.uint(pos, 4)? as u32 as i32 as i64;
        let vtable = u64::try_from(vtable).map_err(|_| format!("table at {pos} has its vtable before the file"))?;
        Ok(Table { pos, vtable, vlen: self.uint(vtable, 2)? })
    }

    /// Where field `id` is stored, or None when it is absent (left at its default).
    fn field(&mut self, t: Table, id: u64) -> Result<Option<u64>, String> {
        let slot = 4 + 2 * id;
        if slot + 2 > t.vlen {
            return Ok(None);
        }
        let off = self.uint(t.vtable + slot, 2)?;
        Ok((off != 0).then_some(t.pos + off))
    }

    /// A scalar field's raw bits, zero-extended; `default` when absent.
    fn scalar(&mut self, t: Table, id: u64, size: u64, default: u64) -> Result<u64, String> {
        match self.field(t, id)? {
            Some(p) => self.uint(p, size),
            None => Ok(default),
        }
    }

    /// A vector field: the position of its first element and its length, bounds-checked without reading it.
    fn vector(&mut self, t: Table, id: u64, elem: u64) -> Result<Option<(u64, u64)>, String> {
        let Some(p) = self.field(t, id)? else { return Ok(None) };
        let v = self.offset(p)?;
        let n = self.uint(v, 4)?;
        if v + 4 + n * elem > self.len {
            return Err(format!("vector at {v} runs past the end of the file"));
        }
        Ok(Some((v + 4, n)))
    }

    fn string(&mut self, t: Table, id: u64) -> Result<String, String> {
        Ok(match self.vector(t, id, 1)? {
            Some((p, n)) => String::from_utf8_lossy(&self.bytes(p, n)?).into_owned(),
            None => String::new(),
        })
    }

    fn ints(&mut self, t: Table, id: u64) -> Result<Option<Vec<i32>>, String> {
        let Some((p, n)) = self.vector(t, id, 4)? else { return Ok(None) };
        Ok(Some(self.bytes(p, n * 4)?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
    }

    /// Element `i` of a vector of tables, each element an offset to its table.
    fn table_at(&mut self, (p, n): (u64, u64), i: u64) -> Result<Table, String> {
        if i >= n {
            return Err(format!("index {i} out of range ({n} entries)"));
        }
        let pos = self.offset(p + 4 * i)?;
        self.table(pos)
    }

    fn tables(&mut self, t: Table, id: u64) -> Result<Vec<Table>, String> {
        let v = self.vector(t, id, 4)?.unwrap_or((0, 0));
        if v.1 > MAX_TABLES {
            return Err(format!("{} tables in one vector", v.1));
        }
        (0..v.1).map(|i| self.table_at(v, i)).collect()
    }
}

/// BuiltinOperator, in enum order (an op code is its index).
const BUILTINS: [&str; 210] = [
    "ADD", "AVERAGE_POOL_2D", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D", "DEPTH_TO_SPACE", "DEQUANTIZE",
    "EMBEDDING_LOOKUP", "FLOOR", "FULLY_CONNECTED", "HASHTABLE_LOOKUP", "L2_NORMALIZATION", "L2_POOL_2D",
    "LOCAL_RESPONSE_NORMALIZATION", "LOGISTIC", "LSH_PROJECTION", "LSTM", "MAX_POOL_2D", "MUL", "RELU", "RELU_N1_TO_1",
    "RELU6", "RESHAPE", "RESIZE_BILINEAR", "RNN", "SOFTMAX", "SPACE_TO_DEPTH", "SVDF", "TANH", "CONCAT_EMBEDDINGS",
    "SKIP_GRAM", "CALL", "CUSTOM", "EMBEDDING_LOOKUP_SPARSE", "PAD", "UNIDIRECTIONAL_SEQUENCE_RNN", "GATHER",
    "BATCH_TO_SPACE_ND", "SPACE_TO_BATCH_ND", "TRANSPOSE", "MEAN", "SUB", "DIV", "SQUEEZE",
    "UNIDIRECTIONAL_SEQUENCE_LSTM", "STRIDED_SLICE", "BIDIRECTIONAL_SEQUENCE_RNN", "EXP", "TOPK_V2", "SPLIT",
    "LOG_SOFTMAX", "DELEGATE", "BIDIRECTIONAL_SEQUENCE_LSTM", "CAST", "PRELU", "MAXIMUM", "ARG_MAX", "MINIMUM", "LESS",
    "NEG", "PADV2", "GREATER", "GREATER_EQUAL", "LESS_EQUAL", "SELECT", "SLICE", "SIN", "TRANSPOSE_CONV",
    "SPARSE_TO_DENSE", "TILE", "EXPAND_DIMS", "EQUAL", "NOT_EQUAL", "LOG", "SUM", "SQRT", "RSQRT", "SHAPE", "POW",
    "ARG_MIN", "FAKE_QUANT", "REDUCE_PROD", "REDUCE_MAX", "PACK", "LOGICAL_OR", "ONE_HOT", "LOGICAL_AND", "LOGICAL_NOT",
    "UNPACK", "REDUCE_MIN", "FLOOR_DIV", "REDUCE_ANY", "SQUARE", "ZEROS_LIKE", "FILL", "FLOOR_MOD", "RANGE",
    "RESIZE_NEAREST_NEIGHBOR", "LEAKY_RELU", "SQUARED_DIFFERENCE", "MIRROR_PAD", "ABS", "SPLIT_V", "UNIQUE", "CEIL",
    "REVERSE_V2", "ADD_N", "GATHER_ND", "COS", "WHERE", "RANK", "ELU", "REVERSE_SEQUENCE", "MATRIX_DIAG", "QUANTIZE",
    "MATRIX_SET_DIAG", "ROUND", "HARD_SWISH", "IF", "WHILE", "NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5",
    "SCATTER_ND", "SELECT_V2", "DENSIFY", "SEGMENT_SUM", "BATCH_MATMUL", "PLACEHOLDER_FOR_GREATER_OP_CODES", "CUMSUM",
    "CALL_ONCE", "BROADCAST_TO", "RFFT2D", "CONV_3D", "IMAG", "REAL", "COMPLEX_ABS", "HASHTABLE", "HASHTABLE_FIND",
    "HASHTABLE_IMPORT", "HASHTABLE_SIZE", "REDUCE_ALL", "CONV_3D_TRANSPOSE", "VAR_HANDLE", "READ_VARIABLE",
    "ASSIGN_VARIABLE", "BROADCAST_ARGS", "RANDOM_STANDARD_NORMAL", "BUCKETIZE", "RANDOM_UNIFORM", "MULTINOMIAL", "GELU",
    "DYNAMIC_UPDATE_SLICE", "RELU_0_TO_1", "UNSORTED_SEGMENT_PROD", "UNSORTED_SEGMENT_MAX", "UNSORTED_SEGMENT_SUM",
    "ATAN2", "UNSORTED_SEGMENT_MIN", "SIGN", "BITCAST", "BITWISE_XOR", "RIGHT_SHIFT", "STABLEHLO_LOGISTIC",
    "STABLEHLO_ADD", "STABLEHLO_DIVIDE", "STABLEHLO_MULTIPLY", "STABLEHLO_MAXIMUM", "STABLEHLO_RESHAPE",
    "STABLEHLO_CLAMP", "STABLEHLO_CONCATENATE", "STABLEHLO_BROADCAST_IN_DIM", "STABLEHLO_CONVOLUTION",
    "STABLEHLO_SLICE", "STABLEHLO_CUSTOM_CALL", "STABLEHLO_REDUCE", "STABLEHLO_ABS", "STABLEHLO_AND",
    "STABLEHLO_COSINE", "STABLEHLO_EXPONENTIAL", "STABLEHLO_FLOOR", "STABLEHLO_LOG", "STABLEHLO_MINIMUM",
    "STABLEHLO_NEGATE", "STABLEHLO_OR", "STABLEHLO_POWER", "STABLEHLO_REMAINDER", "STABLEHLO_RSQRT", "STABLEHLO_SELECT",
    "STABLEHLO_SUBTRACT", "STABLEHLO_TANH", "STABLEHLO_SCATTER", "STABLEHLO_COMPARE", "STABLEHLO_CONVERT",
    "STABLEHLO_DYNAMIC_SLICE", "STABLEHLO_DYNAMIC_UPDATE_SLICE", "STABLEHLO_PAD", "STABLEHLO_IOTA",
    "STABLEHLO_DOT_GENERAL", "STABLEHLO_REDUCE_WINDOW", "STABLEHLO_SORT", "STABLEHLO_WHILE", "STABLEHLO_GATHER",
    "STABLEHLO_TRANSPOSE", "DILATE", "STABLEHLO_RNG_BIT_GENERATOR", "REDUCE_WINDOW", "STABLEHLO_COMPOSITE",
    "STABLEHLO_SHIFT_LEFT", "STABLEHLO_CBRT", "STABLEHLO_CASE",
];
const CUSTOM: i64 = 32;

/// `DEPTHWISE_CONV_2D` gives `DepthwiseConv2D`, like the op names of the other formats.
fn camel(name: &str) -> String {
    name.split('_').map(|p| match p {
        "LSH" | "SVDF" | "RNN" | "LSTM" | "ND" => p.to_string(),
        _ if p.starts_with(|c: char| c.is_ascii_digit()) => p.to_string(),
        _ => p[..1].to_string() + &p[1..].to_ascii_lowercase(),
    }).collect()
}

/// TensorType, in enum order, named as `analysis::dtype_bits` expects.
fn dtype_name(t: u64) -> &'static str {
    const NAMES: [&str; 23] = [
        "float", "float16", "int32", "uint8", "int64", "string", "bool", "int16", "complex64", "int8", "double",
        "complex128", "uint64", "resource", "variant", "uint32", "uint16", "int4", "bfloat16", "int2", "uint4",
        "float8e4m3fn", "float8e5m2",
    ];
    NAMES.get(t as usize).copied().unwrap_or("unknown")
}

/// Builtin options shown as attributes, by BuiltinOptions union type: field names in table order, each with a kind
/// (`_` skips a field). Kinds: i int, f float, b bool, [i] int vector, and the byte enums pad (Padding), act
/// (ActivationFunctionType), mode (MirrorPadMode) and type (TensorType). `=1` is a default other than zero.
const OPTIONS: [(u64, &str); 36] = [
    (1, "padding:pad stride_w:i stride_h:i fused_activation_function:act dilation_w_factor:i=1 dilation_h_factor:i=1"),
    (2, "padding:pad stride_w:i stride_h:i depth_multiplier:i fused_activation_function:act dilation_w_factor:i=1 dilation_h_factor:i=1"),
    (5, "padding:pad stride_w:i stride_h:i filter_width:i filter_height:i fused_activation_function:act"),
    (8, "fused_activation_function:act _ keep_num_dims:b"),
    (9, "beta:f"),
    (10, "axis:i fused_activation_function:act"),
    (11, "fused_activation_function:act"),
    (12, "fused_activation_function:act"),
    (13, "radius:i bias:f alpha:f beta:f"),
    (15, "_ _ align_corners:b half_pixel_centers:b"),
    (17, "new_shape:[i]"),
    (19, "block_size:i"),
    (21, "fused_activation_function:act"),
    (23, "axis:i batch_dims:i"),
    (27, "keep_dims:b"),
    (28, "fused_activation_function:act"),
    (29, "fused_activation_function:act"),
    (30, "squeeze_dims:[i]"),
    (32, "begin_mask:i end_mask:i ellipsis_mask:i new_axis_mask:i shrink_axis_mask:i offset:b"),
    (35, "num_splits:i"),
    (37, "in_data_type:type out_data_type:type"),
    (40, "output_type:type"),
    (49, "padding:pad stride_w:i stride_h:i fused_activation_function:act"),
    (57, "output_type:type"),
    (59, "values_count:i axis:i"),
    (64, "num:i axis:i"),
    (74, "align_corners:b half_pixel_centers:b"),
    (75, "alpha:f"),
    (77, "mode:mode"),
    (79, "num_splits:i"),
    (92, "then_subgraph_index:i else_subgraph_index:i"),
    (93, "cond_subgraph_index:i body_subgraph_index:i"),
    (94, "block_size:i"),
    (101, "adj_x:b adj_y:b"),
    (106, "padding:pad stride_d:i stride_w:i stride_h:i fused_activation_function:act dilation_d_factor:i=1 dilation_w_factor:i=1 dilation_h_factor:i=1"),
    (116, "approximate:b"),
];

fn shape_text(dims: &[i32]) -> String {
    let d: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("[{}]", d.join(","))
}

fn options<S: ByteSource>(r: &mut Reader<S>, kind: u64, t: Table) -> Result<Vec<(String, String)>, String> {
    let Some(&(_, spec)) = OPTIONS.iter().find(|o| o.0 == kind) else { return Ok(Vec::new()) };
    let mut attrs = Vec::new();
    for (id, f) in spec.split(' ').enumerate() {
        let Some((name, ty)) = f.split_once(':') else { continue };
        let (ty, default) = ty.split_once('=').map_or((ty, 0), |(k, d)| (k, d.parse().unwrap_or(0)));
        let id = id as u64;
        let byte = |r: &mut Reader<S>| r.scalar(t, id, 1, default);
        let value = match ty {
            "i" => (r.scalar(t, id, 4, default)? as u32 as i32).to_string(),
            "f" => f32::from_bits(r.scalar(t, id, 4, default)? as u32).to_string(),
            "b" => (byte(r)? != 0).to_string(),
            "pad" => ["SAME", "VALID"].get(byte(r)? as usize).unwrap_or(&"?").to_string(),
            "act" => ["NONE", "RELU", "RELU_N1_TO_1", "RELU6", "TANH", "SIGN_BIT"].get(byte(r)? as usize).unwrap_or(&"?").to_string(),
            "mode" => ["REFLECT", "SYMMETRIC"].get(byte(r)? as usize).unwrap_or(&"?").to_string(),
            "type" => dtype_name(byte(r)?).to_string(),
            _ => match r.ints(t, id)? {
                Some(v) => shape_text(&v),
                None => continue,
            },
        };
        attrs.push((name.to_string(), value));
    }
    Ok(attrs)
}

/// Whether buffer `b` holds data: a non-empty `data` vector (0), or, in files over 2 GB, data stored after the
/// flatbuffer at `offset` (1), valid when above 1, with `size` (2). Buffer 0 is the empty sentinel.
fn holds_data<S: ByteSource>(r: &mut Reader<S>, buffers: (u64, u64), b: u64) -> Result<bool, String> {
    if b == 0 {
        return Ok(false);
    }
    let t = r.table_at(buffers, b).map_err(|e| format!("tensor buffer: {e}"))?;
    Ok(r.vector(t, 0, 1)?.is_some_and(|(_, n)| n > 0) || (r.scalar(t, 1, 8, 0)? > 1 && r.scalar(t, 2, 8, 0)? > 0))
}

/// Model metadata entries: short text values as they are (`min_runtime_version`), and the name, version, author
/// and license of a TFLite Support `TFLITE_METADATA` flatbuffer (metadata_schema.fbs, identifier "M001").
fn metadata<S: ByteSource>(r: &mut Reader<S>, model: Table, buffers: (u64, u64)) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    // Model.metadata (6): Metadata { name 0, buffer 1 }
    for m in r.tables(model, 6)? {
        let name = r.string(m, 0)?;
        let t = r.scalar(m, 1, 4, 0).and_then(|b| r.table_at(buffers, b))?;
        let Some((p, n)) = r.vector(t, 0, 1)? else { continue };
        if name == "TFLITE_METADATA" && (8..=1 << 20).contains(&n) && r.bytes(p + 4, 4)? == b"M001" {
            // ModelMetadata: name 0, description 1, version 2, author 4, license 5
            let root = r.offset(p)?;
            let t = r.table(root)?;
            for (id, key) in [(0, "name"), (2, "version"), (4, "author"), (5, "license")] {
                let v = r.string(t, id)?;
                if !v.is_empty() {
                    out.push((format!("{name}.{key}"), v));
                }
            }
        } else if n <= 64 {
            let text = r.bytes(p, n)?;
            let text = String::from_utf8_lossy(&text);
            let text = text.trim_end_matches('\0');
            if !text.is_empty() && text.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
                out.push((name, text.to_string()));
            }
        }
    }
    Ok(out)
}

/// Parse a TFLite model's main subgraph. Weight buffers are never read; only the pages holding the tables are.
pub fn parse<S: ByteSource>(src: &mut S) -> Result<Graph, String> {
    parse_model(src).map_err(|e| format!("not a readable TFLite model ({e})"))
}

fn parse_model<S: ByteSource>(src: &mut S) -> Result<Graph, String> {
    let len = src.len();
    let mut r = Reader { src, len, pages: HashMap::new() };
    if len < 8 || r.bytes(4, 4)? != b"TFL3" {
        return Err("no TFL3 identifier".into());
    }
    let root = r.offset(0)?;
    let model = r.table(root)?;
    // Model: version 0, operator_codes 1, subgraphs 2, description 3, buffers 4, metadata 6
    let version = r.scalar(model, 0, 4, 0)?;
    // OperatorCode: deprecated_builtin_code 0 (a byte, for codes under 127), custom_code 1, builtin_code 3
    let mut codes = Vec::new();
    for c in r.tables(model, 1)? {
        let code = (r.scalar(c, 0, 1, 0)? as u8 as i8 as i64).max(r.scalar(c, 3, 4, 0)? as u32 as i32 as i64);
        codes.push(match code {
            CUSTOM => Some(r.string(c, 1)?).filter(|s| !s.is_empty()).unwrap_or_else(|| "Custom".into()),
            _ => BUILTINS.get(code as usize).filter(|_| code >= 0).map_or_else(|| format!("Builtin{code}"), |n| camel(n)),
        });
    }
    let subgraphs = r.tables(model, 2)?;
    let main = *subgraphs.first().ok_or("no subgraph")?;
    // buffer tables sit next to their data, so they are only visited for the tensors that use them
    let buffers = r.vector(model, 4, 4)?.unwrap_or((0, 0));

    // SubGraph: tensors 0, inputs 1, outputs 2, operators 3, name 4
    let mut g = Graph::default();
    let mut has_data: HashMap<u64, bool> = HashMap::new();
    for t in r.tables(main, 0)? {
        // Tensor: shape 0, type 1, buffer 2, name 3, shape_signature 7 (-1 for a dynamic dim), external_buffer 10
        let mut shape = r.ints(t, 0)?.unwrap_or_default();
        if let Some(sig) = r.ints(t, 7)?.filter(|s| s.len() == shape.len()) {
            shape = sig;
        }
        let b = r.scalar(t, 2, 4, 0)?;
        let constant = match has_data.get(&b) {
            Some(&c) => c,
            None => {
                let c = holds_data(&mut r, buffers, b)?;
                has_data.insert(b, c);
                c
            }
        } || r.scalar(t, 10, 4, 0)? != 0;
        g.tensors.push(Tensor {
            name: r.string(t, 3)?,
            dtype: dtype_name(r.scalar(t, 1, 1, 0)?).into(),
            shape: Some(shape.into_iter().map(i64::from).collect()),
            producer: NONE,
            constant,
            ..Default::default()
        });
    }
    let n_tensors = g.tensors.len();
    let index = |i: i32| match i {
        -1 => Ok(NONE),
        _ if (0..n_tensors as i32).contains(&i) => Ok(i as u32),
        _ => Err(format!("tensor index {i} out of range ({n_tensors} tensors)")),
    };
    for (i, op) in r.tables(main, 3)?.into_iter().enumerate() {
        // Operator: opcode_index 0, inputs 1, outputs 2, builtin_options_type 3, builtin_options 4
        let code = r.scalar(op, 0, 4, 0)?;
        let name = codes.get(code as usize).ok_or_else(|| format!("op code {code} out of range ({} codes)", codes.len()))?;
        let mut n = Node { op: name.clone(), ..Default::default() };
        n.inputs = r.ints(op, 1)?.unwrap_or_default().into_iter().map(index).collect::<Result<_, _>>()?;
        n.outputs = r.ints(op, 2)?.unwrap_or_default().into_iter().map(index).collect::<Result<_, _>>()?;
        if n.outputs.contains(&NONE) {
            return Err(format!("operator {i} has an omitted output"));
        }
        if let Some(p) = r.field(op, 4)? {
            let kind = r.scalar(op, 3, 1, 0)?;
            let t = r.offset(p)?;
            let t = r.table(t)?;
            n.attrs = options(&mut r, kind, t)?;
        }
        // operators have no names: like other tools, name one after its first output
        n.name = n.outputs.first().map(|&t| g.tensors[t as usize].name.clone()).filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("{}_{i}", n.op));
        g.nodes.push(n);
    }
    if g.nodes.is_empty() {
        return Err("no operators in the main subgraph".into());
    }
    g.finish();

    g.meta.push(("format".into(), format!("TFLite, schema v{version}")));
    let description = r.string(model, 3)?;
    if !description.is_empty() {
        g.meta.push(("description".into(), description));
    }
    if subgraphs.len() > 1 {
        let mut others = Vec::new();
        for (i, &s) in subgraphs.iter().enumerate().skip(1) {
            let name = r.string(s, 4)?;
            let ops = r.vector(s, 3, 4)?.map_or(0, |v| v.1);
            let name = if name.is_empty() { format!("#{i}") } else { name };
            others.push(format!("{name} ({ops} {})", if ops == 1 { "op" } else { "ops" }));
        }
        g.meta.push(("subgraphs".into(), format!("main graph shown; not shown: {}", others.join(", "))));
    }
    let entries = metadata(&mut r, model, buffers)?;
    g.meta.extend(entries);
    Ok(g)
}
