import { buildAtlas } from "./glyphs";
import { Kind, Renderer, type RGBA, type Scene, type Style, type Theme } from "./gl";
import type { Request } from "./worker";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const panesEl = $<HTMLElement>("#panes");
const info = $<HTMLElement>("#info");
const perf = $<HTMLElement>("#perf");
const params = new URLSearchParams(location.search);

const THEME_KEYS = [
  "bg", "bg-panel", "bg-canvas", "rule", "rule-soft", "text", "text-muted", "text-faint", "accent", "accent-text", "warn",
  "node-fill", "node-stroke", "node-text", "edge", "node-selected", "node-linked", "prov-inferred",
  "time-0", "time-1", "time-2", "time-3", "time-4", "time-5",
];

function readTheme(): Theme {
  const cs = getComputedStyle(document.documentElement), theme: Theme = {};
  for (const k of THEME_KEYS) {
    const hex = cs.getPropertyValue(`--${k}`).trim().replace("#", "");
    theme[k] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).concat(1) as RGBA;
  }
  return theme;
}

// worker RPC
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let nextId = 0;
const waiting = new Map<number, [(v: any) => void, (e: Error) => void]>();
worker.onmessage = (e) => {
  const [resolve, reject] = waiting.get(e.data.id)!;
  waiting.delete(e.data.id);
  if (e.data.error) reject(new Error(e.data.error));
  else resolve(e.data.result);
};
type NoId<T> = T extends unknown ? Omit<T, "id"> : never;
function rpc(req: NoId<Request>): Promise<any> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, [resolve, reject]);
    worker.postMessage({ ...req, id });
  });
}

// panes
const atlas = buildAtlas(getComputedStyle(document.documentElement).getPropertyValue("--font-mono") || "monospace");
let theme = readTheme();
const panes = [...document.querySelectorAll<HTMLElement>(".pane")].map((el) => {
  const r = new Renderer(el.querySelector("canvas")!, atlas);
  r.setTheme(theme);
  return { el, r, style: { selected: -1, linked: new Map(), timeOverlay: false } as Style };
});
let paneCount = 0;
const timeToggle = $<HTMLInputElement>("#overlay-time");

function restyle(p: number) {
  const { r, style } = panes[p];
  style.timeOverlay = timeToggle.checked;
  r.setScene(r.scene, style);
}

function boxRect(s: Scene, b: number) {
  return [s.boxes[b * 4], s.boxes[b * 4 + 1], s.boxes[b * 4 + 2], s.boxes[b * 4 + 3]];
}

async function select(p: number, box: number) {
  for (const q of panes) {
    q.style.selected = -1;
    q.style.linked = new Map();
  }
  panes[p].style.selected = box;
  if (box >= 0) {
    const res = await rpc({ op: "select", pane: p, box });
    info.textContent = res.text;
    const o = 1 - p;
    if (paneCount > 1) {
      res.boxes.forEach((b: number, i: number) => panes[o].style.linked.set(b, res.grades[i]));
      revealLinked(o, res.boxes);
    }
  } else info.textContent = "";
  for (let q = 0; q < paneCount; q++) restyle(q);
}

/** Pan the other pane to the linked boxes when none of them is on screen. */
function revealLinked(p: number, boxes: Uint32Array) {
  const { r } = panes[p], s = r.scene;
  if (!s || !boxes.length) return;
  const cw = r.canvas.clientWidth, ch = r.canvas.clientHeight;
  const onScreen = [...boxes].some((b) => {
    const [x, y, w, h] = boxRect(s, b);
    const [sx, sy] = [(x - r.view.x) * r.view.zoom, (y - r.view.y) * r.view.zoom];
    return sx + w * r.view.zoom > 0 && sy + h * r.view.zoom > 0 && sx < cw && sy < ch;
  });
  if (onScreen) return;
  const [x, y, w, h] = boxRect(s, boxes[0]);
  r.anchor(x + w / 2, y + h / 2, cw / 2, ch / 2);
}

async function toggle(p: number, box: number, op: "toggle" | "step", delta = 1) {
  const { r } = panes[p];
  const s = r.scene!;
  const [x, y] = boxRect(s, box);
  const [sx, sy] = [(x - r.view.x) * r.view.zoom, (y - r.view.y) * r.view.zoom];
  const res = await rpc({ op, pane: p, box, delta });
  if (res.anchor < 0) return;
  r.scene = res.scene;
  const [nx, ny] = boxRect(res.scene, res.anchor);
  r.anchor(nx, ny, sx, sy);
  await select(p, res.anchor);
  report(res.wasmBytes);
}

// interaction
panes.forEach(({ r }, p) => {
  const c = r.canvas;
  let drag: { x: number; y: number; moved: boolean } | null = null;
  c.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, moved: false };
    c.setPointerCapture(e.pointerId);
  });
  c.addEventListener("pointermove", (e) => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
      if (drag.moved) {
        r.view.x -= dx / r.view.zoom;
        r.view.y -= dy / r.view.zoom;
        drag.x = e.clientX;
        drag.y = e.clientY;
        r.requestDraw();
      }
    } else {
      const b = r.hit(e.offsetX, e.offsetY), k = b >= 0 ? r.scene!.kinds[b] : -1;
      c.style.cursor = k === Kind.Group || k === Kind.Stack || k === Kind.Frame ? "pointer" : "default";
    }
  });
  c.addEventListener("pointerup", (e) => {
    if (drag && !drag.moved && e.detail < 2) select(p, r.hit(e.offsetX, e.offsetY));
    drag = null;
  });
  c.addEventListener("dblclick", (e) => {
    const b = r.hit(e.offsetX, e.offsetY);
    if (b >= 0) toggle(p, b, "toggle");
  });
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    r.zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)));
  }, { passive: false });
});

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.key === "f") panes.forEach(({ r }) => r.fit());
  const p = panes.findIndex((q) => q.style.selected >= 0);
  if (p < 0) return;
  const box = panes[p].style.selected;
  if (e.key === "Enter") toggle(p, box, "toggle");
  if (e.key === "]" || e.key === "[") toggle(p, box, "step", e.key === "]" ? 1 : -1);
  if (e.key === "Escape") select(p, -1);
});

new ResizeObserver(() => panes.forEach(({ r }) => r.resize())).observe(panesEl);
timeToggle.addEventListener("change", () => { for (let p = 0; p < paneCount; p++) restyle(p); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  theme = readTheme();
  panes.forEach(({ r }, p) => { r.setTheme(theme); if (p < paneCount) restyle(p); });
});

// loading
let loadStart = 0;

function report(wasmBytes: number) {
  const heap = (performance as any).memory?.usedJSHeapSize as number | undefined;
  const mb = (b: number) => `${(b / 2 ** 20).toFixed(0)} MiB`;
  perf.dataset.wasm = String(wasmBytes);
  perf.textContent = `${perf.dataset.load ?? ""}  wasm ${mb(wasmBytes)}${heap ? `  js ${mb(heap)}` : ""}`;
}

async function load(files: File[]) {
  if (!files.length) return;
  loadStart = performance.now();
  $<HTMLElement>("#loading").hidden = false;
  try {
    const res = await rpc({ op: "load", files, flat: params.has("flat") });
    const tParsed = performance.now();
    paneCount = res.scenes.length;
    panesEl.classList.toggle("single", paneCount < 2);
    $<HTMLElement>("#empty").classList.add("hidden");
    const names = files.map((f) => f.name).join(" + ");
    $<HTMLElement>("#crumbs").textContent = `/ ${names}${res.label ? " / " + res.label : ""}`;
    panes.forEach(({ el, r, style }, p) => {
      style.selected = -1;
      style.linked = new Map();
      r.resize();
      r.setScene(res.scenes[p] ?? null, style);
      r.fit();
      const lat = p === 1 && res.latency === res.latency ? `  latency ${(res.latency / 1000).toFixed(2)} ms` : "";
      el.querySelector(".meta")!.textContent = res.nodes[p] !== undefined ? `${res.nodes[p]} nodes${lat}` : "";
      el.querySelector(".title")!.textContent = p === 0 ? "source" : res.label ? "compiled" : "graph b";
    });
    timeToggle.checked = paneCount > 1 && res.latency === res.latency;
    for (let p = 0; p < paneCount; p++) restyle(p);
    // time to the first drawn frame of every pane
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const total = performance.now() - loadStart;
    const boxes = res.scenes.reduce((n: number, s: Scene) => n + s.kinds.length, 0);
    perf.dataset.load = `load ${total.toFixed(0)} ms (core ${res.ms.toFixed(0)} ms)  ${res.nodes.join(" + ")} nodes  ${boxes} boxes`;
    report(res.wasmBytes);
    console.log(`[afterpass] load ${names}: interactive ${total.toFixed(0)} ms, worker ${res.ms.toFixed(0)} ms, transfer+upload ${(total - (tParsed - loadStart)).toFixed(0)} ms, nodes ${res.nodes.join("+")}, boxes ${boxes}, wasm ${res.wasmBytes} B, js heap ${(performance as any).memory?.usedJSHeapSize ?? "n/a"} B`);
    if (params.has("bench")) bench();
  } catch (err) {
    info.textContent = `error: ${(err as Error).message}`;
    console.error(err);
  } finally {
    $<HTMLElement>("#loading").hidden = true;
  }
}

/** Animate pan and zoom on every pane for 300 frames (from fit to 1:1, sweeping across the graph) and
 * report the frame rate and the per-frame draw cost. */
function bench() {
  const frames: number[] = [], draws: number[] = [];
  const active = panes.slice(0, paneCount);
  const start = active.map(({ r }) => ({ ...r.view }));
  let last = performance.now(), i = 0;
  const step = (now: number) => {
    frames.push(now - last);
    last = now;
    const t0 = performance.now();
    active.forEach(({ r }, p) => {
      const s = r.scene!, t = i / 300;
      const phase = (1 - Math.cos(t * Math.PI * 4)) / 2; // 0 = fit, 1 = 1:1 zoom
      const z = start[p].zoom * Math.pow(Math.max(1, 1 / start[p].zoom), phase);
      const cx = s.width * (0.5 + 0.4 * Math.sin(t * Math.PI * 2)), cy = s.height * t;
      r.view.zoom = z;
      r.anchor(cx, cy, r.canvas.clientWidth / 2, r.canvas.clientHeight / 2);
      r.draw();
      r.gl.finish(); // bench only: include GPU completion in the draw cost
    });
    draws.push(performance.now() - t0);
    if (++i < 300) requestAnimationFrame(step);
    else {
      active.forEach(({ r }, p) => { Object.assign(r.view, start[p]); r.requestDraw(); });
      const f = frames.slice(10).sort((a, b) => a - b), d = draws.slice(10).sort((a, b) => a - b);
      const avg = f.reduce((a, b) => a + b, 0) / f.length;
      const p95 = (v: number[]) => v[Math.floor(v.length * 0.95)].toFixed(2);
      const msg = `bench: ${(1000 / avg).toFixed(1)} fps avg, p95 frame ${p95(f)} ms, worst ${f[f.length - 1].toFixed(1)} ms, draw+finish p95 ${p95(d)} ms max ${d[d.length - 1].toFixed(2)} ms`;
      console.log(`[afterpass] ${msg}`);
      perf.textContent += `  ${msg}`;
    }
  };
  requestAnimationFrame(step);
}

const fileInput = $<HTMLInputElement>("#file");
$<HTMLButtonElement>("#open").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => load([...(fileInput.files ?? [])]));
const empty = $<HTMLElement>("#empty");
window.addEventListener("dragover", (e) => { e.preventDefault(); empty.classList.add("drag"); });
window.addEventListener("dragleave", () => empty.classList.remove("drag"));
window.addEventListener("drop", (e) => {
  e.preventDefault();
  empty.classList.remove("drag");
  load([...(e.dataTransfer?.files ?? [])]);
});

// ?src=<url>[&cmp=<url>]: load by URL (demo bundles, benchmarks)
async function fetchFile(url: string): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return new File([await res.blob()], url.split("/").pop()!);
}
const urls = [params.get("src"), params.get("cmp")].filter((u): u is string => !!u);
if (urls.length) {
  loadStart = performance.now();
  Promise.all(urls.map(fetchFile)).then(load, (e) => (info.textContent = `error: ${e.message}`));
}

// empty-state texture: our own token soup, chrome only
const TOKENS = "fuse lower tile emit pass kernel reorder fp16 int8 bf16 nchw nhwc %0 %1 -> => :: {} [] () 0x1f gemm conv mvn rope sdpa x4 x12 x64 ; | / * + -".split(" ");
const ascii = $<HTMLElement>(".ascii");
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
ascii.textContent = Array.from({ length: 80 }, () => {
  let line = "";
  while (line.length < 260) line += (rand() < 0.12 ? "    " : TOKENS[Math.floor(rand() * TOKENS.length)]) + " ";
  return line;
}).join("\n");

if (import.meta.env.DEV) (window as any).__ap = { panes, rpc };
