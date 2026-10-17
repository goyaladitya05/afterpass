// WebGL2 graph renderer: instanced rounded boxes, instanced edge segments and SDF text.
// World units are layout pixels; the camera maps them to the canvas with a pan and a zoom.

import type { Atlas } from "./glyphs";

export interface Scene {
  boxes: Float32Array;
  kinds: Uint8Array;
  flags: Uint8Array;
  times: Float32Array;
  segs: Float32Array;
  labels: string;
  width: number;
  height: number;
}

export const enum Kind { Node = 0, Group = 1, Stack = 2, Frame = 3, Port = 4 }

export type RGBA = [number, number, number, number];
export type Theme = Record<string, RGBA>;

export interface Style {
  selected: number;
  /** linked box -> best provenance grade (0 explicit, 1 inferred) */
  linked: Map<number, number>;
  timeOverlay: boolean;
}

const BOX_VS = `#version 300 es
layout(location=0) in vec2 corner;
layout(location=1) in vec4 rect;
layout(location=2) in vec4 fill;
layout(location=3) in vec4 stroke;
layout(location=4) in vec2 params;
uniform vec3 view;
uniform vec2 res;
out vec2 local; out vec2 half_; out vec4 vFill; out vec4 vStroke; out float vRadius; out float vBorder; out float vPx;
void main() {
  vec2 pad = vec2(2.0) / view.z;
  vec2 p = rect.xy - pad + corner * (rect.zw + 2.0 * pad);
  half_ = rect.zw * 0.5;
  local = p - rect.xy - half_;
  vec2 s = (p - view.xy) * view.z;
  gl_Position = vec4(s.x / res.x * 2.0 - 1.0, 1.0 - s.y / res.y * 2.0, 0.0, 1.0);
  vFill = fill; vStroke = stroke; vRadius = params.x; vBorder = params.y; vPx = 1.0 / view.z;
}`;

const BOX_FS = `#version 300 es
precision highp float;
in vec2 local; in vec2 half_; in vec4 vFill; in vec4 vStroke; in float vRadius; in float vBorder; in float vPx;
out vec4 color;
float sdRound(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
void main() {
  float r = min(vRadius, min(half_.x, half_.y));
  float d = sdRound(local, half_, r);
  float outer = 1.0 - smoothstep(-0.5 * vPx, 0.5 * vPx, d);
  float inner = 1.0 - smoothstep(-0.5 * vPx, 0.5 * vPx, d + vBorder * vPx);
  vec4 c = mix(vStroke, vFill, inner);
  c.a *= outer;
  color = vec4(c.rgb * c.a, c.a);
}`;

const SEG_VS = `#version 300 es
layout(location=0) in vec2 corner;
layout(location=1) in vec4 seg;
uniform vec3 view;
uniform vec2 res;
uniform float width;
void main() {
  vec2 a = (seg.xy - view.xy) * view.z, b = (seg.zw - view.xy) * view.z;
  vec2 dir = b - a;
  float len = length(dir);
  vec2 n = len > 0.0 ? vec2(-dir.y, dir.x) / len : vec2(0.0, 1.0);
  vec2 s = mix(a, b, corner.x) + n * (corner.y - 0.5) * width;
  gl_Position = vec4(s.x / res.x * 2.0 - 1.0, 1.0 - s.y / res.y * 2.0, 0.0, 1.0);
}`;

const SEG_FS = `#version 300 es
precision mediump float;
uniform vec4 tint;
out vec4 color;
void main() { color = vec4(tint.rgb * tint.a, tint.a); }`;

const TEXT_VS = `#version 300 es
layout(location=0) in vec2 corner;
layout(location=1) in vec3 glyph;
layout(location=2) in float gi;
layout(location=3) in vec4 tint;
uniform vec3 view;
uniform vec2 res;
uniform vec4 atlas; // cols, cell u, cell v, font size / cell
uniform float minPx;
out vec2 uv; out vec4 vTint; out float vCellPx;
void main() {
  float cellPx = glyph.z * view.z;
  if (cellPx * atlas.w < minPx) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 s = (glyph.xy + corner * glyph.z - view.xy) * view.z;
  gl_Position = vec4(s.x / res.x * 2.0 - 1.0, 1.0 - s.y / res.y * 2.0, 0.0, 1.0);
  uv = (vec2(mod(gi, atlas.x), floor(gi / atlas.x)) + corner) * atlas.yz;
  vTint = tint; vCellPx = cellPx;
}`;

const TEXT_FS = `#version 300 es
precision mediump float;
uniform sampler2D tex;
uniform vec2 sdf; // radius, cell (atlas px)
in vec2 uv; in vec4 vTint; in float vCellPx;
out vec4 color;
void main() {
  float d = texture(tex, uv).r;
  float s = 0.7 * sdf.y / (sdf.x * vCellPx);
  float a = smoothstep(0.75 - s, 0.75 + s, d) * vTint.a;
  color = vec4(vTint.rgb * a, a);
}`;

const FONT = 12, SUB_FONT = 10, HEAD_FONT = 11;
const BOX_STRIDE = 8; // floats per box instance: rect(4) fill(1 packed) stroke(1 packed) params(2)

function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

function rgba8(c: RGBA): number {
  const b = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
  return (b(c[0]) | (b(c[1]) << 8) | (b(c[2]) << 16) | (b(c[3]) << 24)) >>> 0;
}

/** Growable instance buffer with float32 and uint32 views over the same bytes. */
class Instances {
  f = new Float32Array(1024);
  u = new Uint32Array(this.f.buffer);
  n = 0;
  constructor(readonly stride: number) {}
  slot(): number {
    if (this.n + this.stride > this.f.length) {
      const f = new Float32Array(this.f.length * 2);
      f.set(this.f);
      this.f = f;
      this.u = new Uint32Array(f.buffer);
    }
    const at = this.n;
    this.n += this.stride;
    return at;
  }
  get count() { return this.n / this.stride; }
  view() { return this.f.subarray(0, this.n); }
}

export class Renderer {
  readonly gl: WebGL2RenderingContext;
  private box: WebGLProgram; private seg: WebGLProgram; private glyph: WebGLProgram;
  private vaoBox: WebGLVertexArrayObject; private vaoSeg: WebGLVertexArrayObject; private vaoText: WebGLVertexArrayObject;
  private bufBox: WebGLBuffer; private bufSeg: WebGLBuffer; private bufText: WebGLBuffer;
  private frameCount = 0; private boxCount = 0; private segCount = 0; private glyphCount = 0;
  private atlas: Atlas;
  private theme: Theme = {};
  scene: Scene | null = null;
  view = { x: 0, y: 0, zoom: 1 };
  private pending = false;
  onDraw: (() => void) | null = null;

  constructor(readonly canvas: HTMLCanvasElement, atlas: Atlas) {
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, alpha: false });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    this.atlas = atlas;
    this.box = program(gl, BOX_VS, BOX_FS);
    this.seg = program(gl, SEG_VS, SEG_FS);
    this.glyph = program(gl, TEXT_VS, TEXT_FS);
    const quad = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    const vao = (setup: (buf: WebGLBuffer) => void): [WebGLVertexArrayObject, WebGLBuffer] => {
      const v = gl.createVertexArray()!;
      gl.bindVertexArray(v);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      setup(buf);
      gl.bindVertexArray(null);
      return [v, buf];
    };
    const attr = (loc: number, size: number, type: number, norm: boolean, stride: number, off: number) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, type, norm, stride, off);
      gl.vertexAttribDivisor(loc, 1);
    };
    [this.vaoBox, this.bufBox] = vao(() => {
      const st = BOX_STRIDE * 4;
      attr(1, 4, gl.FLOAT, false, st, 0);
      attr(2, 4, gl.UNSIGNED_BYTE, true, st, 16);
      attr(3, 4, gl.UNSIGNED_BYTE, true, st, 20);
      attr(4, 2, gl.FLOAT, false, st, 24);
    });
    [this.vaoSeg, this.bufSeg] = vao(() => attr(1, 4, gl.FLOAT, false, 16, 0));
    [this.vaoText, this.bufText] = vao(() => {
      attr(1, 3, gl.FLOAT, false, 20, 0);
      attr(2, 1, gl.FLOAT, false, 20, 12);
      attr(3, 4, gl.UNSIGNED_BYTE, true, 20, 16);
    });
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, atlas.width, atlas.height, 0, gl.RED, gl.UNSIGNED_BYTE, atlas.data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  setTheme(theme: Theme) {
    this.theme = theme;
  }

  /** Upload a scene with the given highlight state. */
  setScene(scene: Scene | null, style: Style) {
    this.scene = scene;
    const gl = this.gl;
    if (!scene) {
      this.frameCount = this.boxCount = this.segCount = this.glyphCount = 0;
      this.requestDraw();
      return;
    }
    const t = this.theme, n = scene.kinds.length;
    const maxTime = scene.times.reduce((m, v, i) => (scene.kinds[i] !== Kind.Frame && v > m ? v : m), 0);
    const ramp = [t["time-0"], t["time-1"], t["time-2"], t["time-3"], t["time-4"], t["time-5"]];
    const heat = (v: number): RGBA | null => {
      if (!style.timeOverlay || !(v > 0) || maxTime <= 0) return null;
      const k = Math.min(1, Math.log1p(v) / Math.log1p(maxTime)) * (ramp.length - 1);
      const i = Math.floor(k), f = k - i, a = ramp[i], b = ramp[Math.min(i + 1, ramp.length - 1)];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, 1];
    };
    // frames first (drawn under edges), then the rest; stacks get two offset shadow copies
    const frames = new Instances(BOX_STRIDE), boxes = new Instances(BOX_STRIDE);
    const push = (out: Instances, i: number, dx: number, fill: RGBA, stroke: RGBA, radius: number, border: number) => {
      const b = scene.boxes, k = out.slot();
      out.f[k] = b[i * 4] + dx; out.f[k + 1] = b[i * 4 + 1] + dx; out.f[k + 2] = b[i * 4 + 2]; out.f[k + 3] = b[i * 4 + 3];
      out.u[k + 4] = rgba8(fill); out.u[k + 5] = rgba8(stroke); out.f[k + 6] = radius; out.f[k + 7] = border;
    };
    const labels = scene.labels.split("\n");
    const glyphs = new Instances(5);
    for (let i = 0; i < n; i++) {
      const kind = scene.kinds[i];
      let fill = t["node-fill"], stroke = t["node-stroke"], border = 1, radius = 3;
      if (kind === Kind.Group || kind === Kind.Stack) { fill = t["bg-panel"]; stroke = t["rule"]; radius = 2; }
      if (kind === Kind.Frame) { fill = t["bg"]; stroke = t["rule"]; radius = 4; }
      if (kind === Kind.Port) { fill = t["edge"]; stroke = t["edge"]; radius = 100; border = 0; }
      const h = kind !== Kind.Frame ? heat(scene.times[i]) : null;
      if (h) fill = h;
      if (scene.flags[i] & 1) { stroke = t["warn"]; border = 1.5; }
      const link = style.linked.get(i);
      if (link !== undefined) { stroke = link === 0 ? t["node-linked"] : t["prov-inferred"]; border = 2.5; }
      if (i === style.selected) { stroke = t["node-selected"]; border = 3; }
      if (kind === Kind.Stack) {
        push(boxes, i, 6, t["bg-panel"], t["rule"], radius, 1);
        push(boxes, i, 3, t["bg-panel"], t["rule"], radius, 1);
      }
      push(kind === Kind.Frame ? frames : boxes, i, 0, fill, stroke, radius, border);
      if (kind !== Kind.Port) this.layoutLabel(glyphs, scene, i, labels[i] ?? "", h);
    }
    this.frameCount = frames.count;
    this.boxCount = boxes.count;
    const all = new Float32Array(frames.n + boxes.n);
    all.set(frames.view());
    all.set(boxes.view(), frames.n);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufBox);
    gl.bufferData(gl.ARRAY_BUFFER, all, gl.STATIC_DRAW);
    this.segCount = scene.segs.length / 4;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufSeg);
    gl.bufferData(gl.ARRAY_BUFFER, scene.segs, gl.STATIC_DRAW);
    this.glyphCount = glyphs.count;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufText);
    gl.bufferData(gl.ARRAY_BUFFER, glyphs.view(), gl.STATIC_DRAW);
    this.requestDraw();
  }

  private layoutLabel(out: Instances, scene: Scene, i: number, label: string, heat: RGBA | null) {
    const [title, sub] = label.split("\t");
    const b = scene.boxes, x = b[i * 4], y = b[i * 4 + 1], w = b[i * 4 + 2], h = b[i * 4 + 3];
    const kind = scene.kinds[i], t = this.theme;
    const ink = heat && heat[0] + heat[1] + heat[2] > 1.8 ? t["bg"] : t["node-text"];
    if (kind === Kind.Node) {
      this.text(out, title, x + 8, y + h / 2 + FONT * 0.35, FONT, w - 14, ink);
    } else if (kind === Kind.Frame) {
      this.text(out, title + (sub ? "  " + sub : ""), x + 10, y + 16, HEAD_FONT, w - 18, t["text-muted"]);
    } else {
      const badge = title.indexOf("  ×");
      const main = badge >= 0 ? title.slice(0, badge) : title;
      const end = this.text(out, main, x + 10, y + 18, FONT, w - 18, ink);
      if (badge >= 0) this.text(out, title.slice(badge), end, y + 18, FONT, x + w - 8 - end, t["accent-text"]);
      this.text(out, sub, x + 10, y + 34, SUB_FONT, w - 18, heat ? ink : t["text-muted"]);
    }
  }

  /** Lay out one line of monospace text; returns the pen x after the last glyph. */
  private text(out: Instances, s: string, x: number, baseline: number, font: number, maxW: number, color: RGBA): number {
    const a = this.atlas, scale = font / a.size, adv = a.advance * font;
    let chars = [...s];
    const fit = Math.floor(maxW / adv);
    if (fit <= 0) return x;
    if (chars.length > fit) chars = chars.slice(0, Math.max(0, fit - 1)).concat("…");
    const c = rgba8(color), cellW = a.cell * scale;
    for (const ch of chars) {
      if (ch !== " ") {
        const k = out.slot();
        out.f[k] = x - a.buffer * scale; out.f[k + 1] = baseline - a.baseline * scale; out.f[k + 2] = cellW;
        out.f[k + 3] = a.index.get(ch) ?? a.index.get("?")!; out.u[k + 4] = c;
      }
      x += adv;
    }
    return x;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.requestDraw();
  }

  fit() {
    if (!this.scene) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = this.canvas.width / dpr, ch = this.canvas.height / dpr, m = 40;
    const z = Math.min((cw - m) / Math.max(this.scene.width, 1), (ch - m - 30) / Math.max(this.scene.height, 1), 1.25);
    this.view.zoom = z;
    this.view.x = this.scene.width / 2 - cw / 2 / z;
    this.view.y = -(m / 2 + 30) / z;
    this.requestDraw();
  }

  /** Pan so that a world point appears at the given canvas CSS position. */
  anchor(wx: number, wy: number, sx: number, sy: number) {
    this.view.x = wx - sx / this.view.zoom;
    this.view.y = wy - sy / this.view.zoom;
    this.requestDraw();
  }

  zoomAt(sx: number, sy: number, factor: number) {
    const wx = this.view.x + sx / this.view.zoom, wy = this.view.y + sy / this.view.zoom;
    this.view.zoom = Math.min(8, Math.max(0.002, this.view.zoom * factor));
    this.anchor(wx, wy, sx, sy);
  }

  toWorld(sx: number, sy: number): [number, number] {
    return [this.view.x + sx / this.view.zoom, this.view.y + sy / this.view.zoom];
  }

  /** Topmost box under a canvas CSS position, or -1. */
  hit(sx: number, sy: number): number {
    const s = this.scene;
    if (!s) return -1;
    const [x, y] = this.toWorld(sx, sy), b = s.boxes;
    for (let i = s.kinds.length - 1; i >= 0; i--) {
      if (s.kinds[i] === Kind.Port) continue;
      if (x >= b[i * 4] && y >= b[i * 4 + 1] && x <= b[i * 4] + b[i * 4 + 2] && y <= b[i * 4 + 1] + b[i * 4 + 3]) return i;
    }
    return -1;
  }

  requestDraw() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.draw();
    });
  }

  draw() {
    const gl = this.gl, dpr = window.devicePixelRatio || 1, t = this.theme;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const bg = t["bg-canvas"] ?? [0, 0, 0, 1];
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const res = [this.canvas.width / dpr, this.canvas.height / dpr];
    const view = [this.view.x, this.view.y, this.view.zoom];
    const uniforms = (p: WebGLProgram) => {
      gl.useProgram(p);
      gl.uniform3fv(gl.getUniformLocation(p, "view"), view);
      gl.uniform2fv(gl.getUniformLocation(p, "res"), res);
    };
    if (this.frameCount) {
      uniforms(this.box);
      gl.bindVertexArray(this.vaoBox);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.frameCount);
    }
    if (this.segCount) {
      uniforms(this.seg);
      gl.uniform1f(gl.getUniformLocation(this.seg, "width"), 1.25);
      gl.uniform4fv(gl.getUniformLocation(this.seg, "tint"), t["edge"] ?? [0.5, 0.5, 0.5, 1]);
      gl.bindVertexArray(this.vaoSeg);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.segCount);
    }
    if (this.boxCount) {
      uniforms(this.box);
      gl.bindVertexArray(this.vaoBox);
      this.drawFrom(this.frameCount, this.boxCount);
    }
    if (this.glyphCount) {
      const a = this.atlas;
      uniforms(this.glyph);
      gl.uniform4f(gl.getUniformLocation(this.glyph, "atlas"), a.cols, a.cell / a.width, a.cell / a.height, a.size / a.cell);
      gl.uniform2f(gl.getUniformLocation(this.glyph, "sdf"), 8, a.cell);
      gl.uniform1f(gl.getUniformLocation(this.glyph, "minPx"), 5);
      gl.bindVertexArray(this.vaoText);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.glyphCount);
    }
    gl.bindVertexArray(null);
    this.onDraw?.();
  }

  /** Draw box instances [first, first + count) by re-pointing the instance attributes. */
  private drawFrom(first: number, count: number) {
    const gl = this.gl, st = BOX_STRIDE * 4, off = first * st;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufBox);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, st, off);
    gl.vertexAttribPointer(2, 4, gl.UNSIGNED_BYTE, true, st, off + 16);
    gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, st, off + 20);
    gl.vertexAttribPointer(4, 2, gl.FLOAT, false, st, off + 24);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, st, 0);
    gl.vertexAttribPointer(2, 4, gl.UNSIGNED_BYTE, true, st, 16);
    gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, st, 20);
    gl.vertexAttribPointer(4, 2, gl.FLOAT, false, st, 24);
  }
}
