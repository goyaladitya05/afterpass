//! Streaming ONNX reader. Walks the protobuf through ranged reads and seeks past initializer payloads,
//! so weights are never read: only node, value_info and initializer headers are decoded.

use crate::ir::{Graph, Node, NONE};

/// Random-access bytes: a file in the browser (`File.slice` via FileReaderSync) or on disk in tests.
pub trait ByteSource {
    fn len(&self) -> u64;
    fn read(&mut self, offset: u64, buf: &mut [u8]);
}

impl ByteSource for &[u8] {
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }
    fn read(&mut self, offset: u64, buf: &mut [u8]) {
        let o = offset as usize;
        buf.copy_from_slice(&self[o..o + buf.len()]);
    }
}

const WINDOW: usize = 1 << 20;
/// Graph-level messages larger than this (other than initializers) are skipped instead of decoded.
const MAX_MESSAGE: u64 = 64 << 20;

struct Stream<'a, S: ByteSource> {
    src: &'a mut S,
    buf: Vec<u8>,
    start: u64,
    pos: u64,
    len: u64,
}

impl<'a, S: ByteSource> Stream<'a, S> {
    fn byte(&mut self) -> Result<u8, String> {
        if self.pos >= self.len {
            return Err("unexpected end of ONNX file".into());
        }
        if self.pos < self.start || self.pos >= self.start + self.buf.len() as u64 {
            let n = WINDOW.min((self.len - self.pos) as usize);
            self.buf.resize(n, 0);
            self.src.read(self.pos, &mut self.buf);
            self.start = self.pos;
        }
        let b = self.buf[(self.pos - self.start) as usize];
        self.pos += 1;
        Ok(b)
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= ((b & 0x7f) as u64) << shift;
            if b < 0x80 {
                return Ok(v);
            }
        }
        Err("bad varint".into())
    }

    fn bytes(&mut self, n: u64) -> Result<Vec<u8>, String> {
        if self.pos + n > self.len {
            return Err("message runs past end of file".into());
        }
        let mut out = vec![0u8; n as usize];
        let buffered = self.pos >= self.start && self.pos + n <= self.start + self.buf.len() as u64;
        if buffered {
            let o = (self.pos - self.start) as usize;
            out.copy_from_slice(&self.buf[o..o + n as usize]);
        } else {
            self.src.read(self.pos, &mut out);
        }
        self.pos += n;
        Ok(out)
    }

    /// Skip a field value of the given wire type without reading its payload.
    fn skip(&mut self, wire: u64) -> Result<(), String> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 => {
                let n = self.varint()?;
                self.pos += n;
            }
            5 => self.pos += 4,
            _ => return Err(format!("unsupported wire type {wire}")),
        }
        Ok(())
    }
}

/// In-memory field iterator over one decoded message.
struct Fields<'a> {
    data: &'a [u8],
    pos: usize,
}

enum Value<'a> {
    Int(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
    Fixed64,
}

fn varint(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *data.get(*pos)?;
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            return Some(v);
        }
    }
    None
}

impl<'a> Iterator for Fields<'a> {
    type Item = (u64, Value<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let tag = varint(self.data, &mut self.pos)?;
        let value = match tag & 7 {
            0 => Value::Int(varint(self.data, &mut self.pos)?),
            1 => {
                self.data.get(self.pos..self.pos + 8)?;
                self.pos += 8;
                Value::Fixed64
            }
            2 => {
                let n = varint(self.data, &mut self.pos)? as usize;
                let b = self.data.get(self.pos..self.pos + n)?;
                self.pos += n;
                Value::Bytes(b)
            }
            5 => {
                let b = self.data.get(self.pos..self.pos + 4)?;
                self.pos += 4;
                Value::Fixed32(u32::from_le_bytes(b.try_into().ok()?))
            }
            _ => return None,
        };
        Some((tag >> 3, value))
    }
}

fn fields(data: &[u8]) -> Fields<'_> {
    Fields { data, pos: 0 }
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// Repeated int64 that may be packed (wire type 2) or not.
fn push_ints(v: Value, out: &mut Vec<i64>) {
    match v {
        Value::Int(i) => out.push(i as i64),
        Value::Bytes(b) => {
            let mut p = 0;
            while let Some(i) = varint(b, &mut p) {
                out.push(i as i64);
            }
        }
        _ => {}
    }
}

pub fn dtype_name(t: u64) -> &'static str {
    const NAMES: [&str; 24] = [
        "undefined", "float", "uint8", "int8", "uint16", "int16", "int32", "int64", "string", "bool", "float16",
        "double", "uint32", "uint64", "complex64", "complex128", "bfloat16", "float8e4m3fn", "float8e4m3fnuz",
        "float8e5m2", "float8e5m2fnuz", "uint4", "int4", "float4e2m1",
    ];
    NAMES.get(t as usize).copied().unwrap_or("unknown")
}

/// TensorProto fields we keep: dims (1), data_type (2), name (8). Payload fields are never decoded.
struct TensorHeader {
    name: String,
    dtype: u64,
    dims: Vec<i64>,
}

fn tensor_header(data: &[u8]) -> TensorHeader {
    let mut t = TensorHeader { name: String::new(), dtype: 0, dims: Vec::new() };
    for (f, v) in fields(data) {
        match (f, v) {
            (1, v) => push_ints(v, &mut t.dims),
            (2, Value::Int(d)) => t.dtype = d,
            (8, Value::Bytes(b)) => t.name = text(b),
            _ => {}
        }
    }
    t
}

fn shape_text(dims: &[i64]) -> String {
    let d: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("[{}]", d.join(","))
}

fn attribute(data: &[u8]) -> (String, String) {
    let (mut name, mut ints, mut floats, mut strings) = (String::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut scalar, mut kind) = (None, 0u64);
    for (f, v) in fields(data) {
        match (f, v) {
            (1, Value::Bytes(b)) => name = text(b),
            (2, Value::Fixed32(x)) => scalar = Some(format!("{}", f32::from_bits(x))),
            (3, Value::Int(i)) => scalar = Some((i as i64).to_string()),
            (4, Value::Bytes(b)) => scalar = Some(text(b)),
            (5, Value::Bytes(b)) => {
                let t = tensor_header(b);
                scalar = Some(format!("tensor {} {}", dtype_name(t.dtype), shape_text(&t.dims)));
            }
            (6, Value::Bytes(b)) => {
                let nodes = fields(b).filter(|(f, _)| *f == 1).count();
                scalar = Some(format!("graph ({nodes} nodes)"));
            }
            (7, Value::Fixed32(x)) => floats.push(f32::from_bits(x)),
            (7, Value::Bytes(b)) => floats.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
            (8, v) => push_ints(v, &mut ints),
            (9, Value::Bytes(b)) => strings.push(text(b)),
            (20, Value::Int(t)) => kind = t,
            _ => {}
        }
    }
    let value = scalar.unwrap_or_else(|| match kind {
        6 => format!("{floats:?}"),
        8 => shape_text(&ints),
        9 => format!("{strings:?}"),
        _ if !ints.is_empty() => shape_text(&ints),
        _ if !floats.is_empty() => format!("{floats:?}"),
        _ => String::new(),
    });
    (name, value)
}

fn node(g: &mut Graph, data: &[u8]) -> Node {
    let mut n = Node::default();
    let mut domain = String::new();
    for (f, v) in fields(data) {
        if let Value::Bytes(b) = v {
            match f {
                1 => n.inputs.push(if b.is_empty() { NONE } else { g.tensor(&text(b)) }),
                2 => n.outputs.push(g.tensor(&text(b))),
                3 => n.name = text(b),
                4 => n.op = text(b),
                5 => n.attrs.push(attribute(b)),
                7 => domain = text(b),
                _ => {}
            }
        }
    }
    if !domain.is_empty() && domain != "ai.onnx" {
        n.op = format!("{domain}::{}", n.op);
    }
    n
}

fn value_info(g: &mut Graph, data: &[u8]) {
    let (mut name, mut dtype, mut shape) = (String::new(), 0, None);
    for (f, v) in fields(data) {
        match (f, v) {
            (1, Value::Bytes(b)) => name = text(b),
            (2, Value::Bytes(ty)) => {
                for (f, v) in fields(ty) {
                    if let (1, Value::Bytes(tt)) = (f, v) {
                        for (f, v) in fields(tt) {
                            match (f, v) {
                                (1, Value::Int(d)) => dtype = d,
                                (2, Value::Bytes(sh)) => {
                                    let mut dims = Vec::new();
                                    for (f, v) in fields(sh) {
                                        if let (1, Value::Bytes(dim)) = (f, v) {
                                            let mut d = -1;
                                            for (f, v) in fields(dim) {
                                                if let (1, Value::Int(x)) = (f, v) {
                                                    d = x as i64;
                                                }
                                            }
                                            dims.push(d);
                                        }
                                    }
                                    shape = Some(dims);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if name.is_empty() {
        return;
    }
    let t = g.tensor(&name);
    let t = &mut g.tensors[t as usize];
    if dtype != 0 {
        t.dtype = dtype_name(dtype).to_string();
    }
    if shape.is_some() {
        t.shape = shape;
    }
}

/// Decode an initializer's header in place, seeking past its payload.
fn initializer<S: ByteSource>(s: &mut Stream<S>, g: &mut Graph, end: u64) -> Result<(), String> {
    let mut t = TensorHeader { name: String::new(), dtype: 0, dims: Vec::new() };
    while s.pos < end {
        let tag = s.varint()?;
        match (tag >> 3, tag & 7) {
            (1, 0) => t.dims.push(s.varint()? as i64),
            (1, 2) => {
                let n = s.varint()?;
                push_ints(Value::Bytes(&s.bytes(n)?), &mut t.dims);
            }
            (2, 0) => t.dtype = s.varint()?,
            (8, 2) => {
                let n = s.varint()?;
                t.name = text(&s.bytes(n)?);
            }
            (_, wire) => s.skip(wire)?,
        }
    }
    let id = g.tensor(&t.name);
    let tensor = &mut g.tensors[id as usize];
    tensor.dtype = dtype_name(t.dtype).to_string();
    tensor.shape = Some(t.dims);
    tensor.constant = true;
    Ok(())
}

fn graph<S: ByteSource>(s: &mut Stream<S>, end: u64) -> Result<Graph, String> {
    let mut g = Graph::default();
    let mut infos = Vec::new();
    while s.pos < end {
        let tag = s.varint()?;
        let (field, wire) = (tag >> 3, tag & 7);
        if wire != 2 {
            s.skip(wire)?;
            continue;
        }
        let n = s.varint()?;
        match field {
            5 => initializer(s, &mut g, s.pos + n)?,
            1 | 11 | 12 | 13 if n <= MAX_MESSAGE => {
                let data = s.bytes(n)?;
                if field == 1 {
                    let node = node(&mut g, &data);
                    g.nodes.push(node);
                } else {
                    infos.push(data);
                }
            }
            _ => s.pos += n,
        }
    }
    for data in infos {
        value_info(&mut g, &data);
    }
    g.finish();
    Ok(g)
}

/// Parse an ONNX model's topology. Only the graph protobuf is read; external data files are never touched.
pub fn parse<S: ByteSource>(src: &mut S) -> Result<Graph, String> {
    parse_model(src).map_err(|e| format!("not a readable ONNX model ({e})"))
}

fn parse_model<S: ByteSource>(src: &mut S) -> Result<Graph, String> {
    let len = src.len();
    let mut s = Stream { src, buf: Vec::new(), start: 0, pos: 0, len };
    while s.pos < len {
        let tag = s.varint()?;
        if tag >> 3 == 7 && tag & 7 == 2 {
            let n = s.varint()?;
            let end = s.pos + n;
            return graph(&mut s, end);
        }
        s.skip(tag & 7)?;
    }
    Err("no graph in ONNX file".into())
}
