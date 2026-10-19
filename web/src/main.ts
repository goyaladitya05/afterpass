import { asciiTexture, loader, scramble, ticker } from "./chrome";
import { buildAtlas } from "./glyphs";
import { Kind, RAMP, Renderer, type RGBA, type Scene, sceneMaxTime, type Style, type Theme } from "./gl";
import { type Finding, h, pressable, renderFindings, renderInspector, us } from "./panels";
import { STALE } from "./stale";
import type { Request } from "./worker";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const params = new URLSearchParams(location.search);
const panesEl = $<HTMLElement>("#panes");
const info = $<HTMLElement>("#info");
const perf = $<HTMLElement>("#perf");
const overlaySel = $<HTMLSelectElement>("#overlay");
const landing = $<HTMLElement>("#landing");

// theme
const CATS = ["cat-1", "cat-2", "cat-3", "cat-4", "cat-5", "cat-6", "cat-7", "cat-8"];
const THEME_KEYS = [
  "bg", "bg-panel", "bg-canvas", "rule", "text-muted", "accent-text", "warn",
  "node-fill", "node-stroke", "node-text", "edge", "node-selected", "node-linked", "prov-inferred",
  ...RAMP, ...CATS,
];
/** Fixed precision colours so fp32 means the same colour in every graph (order of the core's legend). */
const PRECISION_CATS = ["cat-1", "cat-2", "cat-6", "cat-3", "cat-4", "cat-7"];

function readTheme(): Theme {
  const cs = getComputedStyle(document.documentElement), theme: Theme = {};
  for (const k of THEME_KEYS) {
    const hex = cs.getPropertyValue(`--${k}`).trim().replace("#", "");
    theme[k] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).concat(1) as RGBA;
  }
  return theme;
}
const css = (c: RGBA) => `rgb(${c.slice(0, 3).map((x) => Math.round(x * 255)).join(" ")})`;

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
// a worker script that fails to load (deleted by a newer deploy) never answers: fail every request instead
worker.onerror = () => {
  for (const [, reject] of waiting.values()) reject(new Error(STALE));
  waiting.clear();
};
type NoId<T> = T extends unknown ? Omit<T, "id"> : never;
function rpc(req: NoId<Request>): Promise<any> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, [resolve, reject]);
    worker.postMessage({ ...req, id });
  });
}

// state
let theme: Theme = {};
let panes: { el: HTMLElement; r: Renderer; style: Style; legend: HTMLElement; legendNames: string[] }[] = [];
let paneCount = 0;
let paneNames: string[] = [];
let latencies: number[] = [];
let selection: { pane: number; box: number } | null = null;

function boxRect(s: Scene, b: number) {
  return [s.boxes[b * 4], s.boxes[b * 4 + 1], s.boxes[b * 4 + 2], s.boxes[b * 4 + 3]];
}

function restyle(p: number) {
  const { r, style } = panes[p];
  r.setScene(r.scene, style);
}

// overlays
async function applyOverlay() {
  const mode = overlaySel.value;
  const cat = mode !== "none" && mode !== "time";
  const res = cat ? await rpc({ op: "overlay", mode }) : [];
  panes.slice(0, paneCount).forEach((p, i) => {
    p.style.overlay = mode;
    p.style.cats = cat ? res[i]?.cats : undefined;
    const names: string[] = cat && res[i]?.legend ? res[i].legend.split("\n") : [];
    p.legendNames = names;
    p.style.palette = names.map((_, k) => theme[mode === "precision" ? PRECISION_CATS[k] : CATS[Math.min(k, CATS.length - 1)]]);
  });
  syncTimeScale();
}

/** Compared compiled panes share one time ramp, so the same colour means the same time on both sides. */
function syncTimeScale() {
  const compiled = paneCount === 3 ? [1, 2] : [];
  const max = Math.max(0, ...compiled.map((p) => (panes[p].r.scene ? sceneMaxTime(panes[p].r.scene!) : 0)));
  panes.slice(0, paneCount).forEach((p, i) => {
    p.style.timeMax = compiled.includes(i) ? max : undefined;
    renderLegend(i);
    restyle(i);
  });
}

function renderLegend(i: number) {
  const { legend, style, legendNames, r } = panes[i];
  legend.replaceChildren();
  if (style.overlay === "time") {
    const max = style.timeMax ?? (r.scene ? sceneMaxTime(r.scene) : 0);
    if (!max) return;
    const ramp = RAMP.map((k) => css(theme[k])).join(",");
    legend.append(h("li", {}, h("i", { class: "ramp", style: `background: linear-gradient(90deg, ${ramp})` }), `0 to ${us(max)}`));
    return;
  }
  if (!style.cats) return;
  const present = new Set(style.cats);
  legendNames.forEach((name, k) => {
    if (!present.has(k) || k >= CATS.length) return;
    legend.append(h("li", {}, h("i", { style: `background: ${css(style.palette![k])}` }), k === CATS.length - 1 && legendNames.length > CATS.length ? "other" : name));
  });
}

// selection
async function select(p: number, box: number, opts: { center?: boolean } = {}) {
  for (const q of panes) {
    q.style.selected = -1;
    q.style.linked = new Map();
  }
  selection = box >= 0 ? { pane: p, box } : null;
  panes[p].style.selected = box;
  if (box >= 0 && !loading) {
    const [linked, insp] = await Promise.all([rpc({ op: "select", pane: p, box }), rpc({ op: "inspect", pane: p, box })]);
    say(linked.text);
    linked.linked.forEach((l: { boxes: Uint32Array; grades: Uint8Array }, q: number) => {
      if (q === p || !l.boxes.length) return;
      l.boxes.forEach((b, i) => panes[q].style.linked.set(b, l.grades[i]));
      revealLinked(q, l.boxes);
    });
    const title = panes[p].r.scene!.labels.split("\n")[box]?.split("\t")[0] ?? "";
    renderInspector($("#inspect-main"), $("#inspect-links"), $("#inspector-meta"), insp, paneNames, title, reveal);
    if (opts.center) centerOn(p, box);
  } else {
    say("");
  }
  for (let q = 0; q < paneCount; q++) restyle(q);
}

function centerOn(p: number, box: number) {
  const { r } = panes[p];
  const [x, y, w, hh] = boxRect(r.scene!, box);
  if (w * r.view.zoom < 24) r.view.zoom = Math.min(1.2, 120 / Math.max(w, 1));
  r.anchor(x + w / 2, y + hh / 2, r.canvas.clientWidth / 2, r.canvas.clientHeight / 2);
}

/** Pan the other pane to the linked boxes when none of them is on screen. */
function revealLinked(p: number, boxes: Uint32Array) {
  const { r } = panes[p], s = r.scene;
  if (!s || !boxes.length) return;
  const cw = r.canvas.clientWidth, ch = r.canvas.clientHeight;
  const visible = [...boxes].some((b) => {
    const [x, y, w, hh] = boxRect(s, b);
    const [sx, sy] = [(x - r.view.x) * r.view.zoom, (y - r.view.y) * r.view.zoom];
    return sx + w * r.view.zoom > 0 && sy + hh * r.view.zoom > 0 && sx < cw && sy < ch;
  });
  if (!visible) {
    const [x, y, w, hh] = boxRect(s, boxes[0]);
    r.anchor(x + w / 2, y + hh / 2, cw / 2, ch / 2);
  }
}

async function afterSceneChange(p: number, scene: Scene, wasmBytes: number) {
  panes[p].r.scene = scene;
  await applyOverlay();
  report(wasmBytes);
}

async function toggle(p: number, box: number, op: "toggle" | "step", delta = 1) {
  if (loading) return;
  const { r } = panes[p];
  const [x, y] = boxRect(r.scene!, box);
  const [sx, sy] = [(x - r.view.x) * r.view.zoom, (y - r.view.y) * r.view.zoom];
  const res = await rpc({ op, pane: p, box, delta });
  if (res.anchor < 0) return;
  await afterSceneChange(p, res.scene, res.wasmBytes);
  const [nx, ny] = boxRect(res.scene, res.anchor);
  r.anchor(nx, ny, sx, sy);
  await select(p, res.anchor);
}

/** Open whatever hides a node (or group) and select it. */
async function reveal(p: number, target: number, group = false) {
  if (p >= paneCount || loading) return;
  const res = await rpc({ op: "reveal", pane: p, target, group });
  if (res.scene) await afterSceneChange(p, res.scene, res.wasmBytes);
  if (res.box >= 0) await select(p, res.box, { center: true });
  else say("that node is a folded constant and is not drawn");
}

// findings
async function loadFindings() {
  const items: Finding[] = paneCount ? await rpc({ op: "findings" }) : [];
  renderFindings($("#findings-body"), items, (f, k) => {
    const [p, id, group] = f.targets[k];
    reveal(p, id, group === 1);
  });
}

// search
const searchInput = $<HTMLInputElement>("#search");
const results = $<HTMLUListElement>("#results");
let searchTimer = 0;
searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = window.setTimeout(async () => {
    const q = searchInput.value.trim();
    results.replaceChildren();
    if (!q || !paneCount) {
      results.hidden = true;
      return;
    }
    const hits: { pane: number; node: number; name: string; op: string }[] = await rpc({ op: "search", query: q });
    for (const hit of hits) {
      const li = h("li", {}, h("span", { class: "tag" }, hit.pane === 0 ? "src" : paneCount === 3 ? paneNames[hit.pane] : "cmp"), h("span", {}, hit.op), h("span", {}, hit.name));
      pressable(li, () => {
        results.hidden = true;
        reveal(hit.pane, hit.node);
      });
      results.append(li);
    }
    if (!hits.length) results.append(h("li", { class: "empty" }, "no match"));
    results.hidden = false;
  }, 150);
});
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") (results.querySelector("li:not(.empty)") as HTMLElement | null)?.click();
  if (e.key === "Escape") {
    results.hidden = true;
    searchInput.blur();
  }
});
document.addEventListener("click", (e) => { if (!(e.target as HTMLElement).closest(".search")) results.hidden = true; });

// interaction
function bindPane(p: number) {
  const { r } = panes[p];
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
}

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === "/") {
    e.preventDefault();
    searchInput.focus();
  }
  if (e.key === "f") panes.forEach(({ r }) => r.fit());
  // arrows pan and +/- zoom the selected pane, or every pane when nothing is selected
  const moves: Record<string, [number, number, number]> = { ArrowLeft: [-1, 0, 1], ArrowRight: [1, 0, 1], ArrowUp: [0, -1, 1], ArrowDown: [0, 1, 1], "+": [0, 0, 1.25], "=": [0, 0, 1.25], "-": [0, 0, 0.8] };
  const m = moves[e.key];
  if (m && paneCount) {
    e.preventDefault();
    for (const { r } of selection ? [panes[selection.pane]] : panes.slice(0, paneCount)) {
      r.view.x += (m[0] * r.canvas.clientWidth) / 5 / r.view.zoom;
      r.view.y += (m[1] * r.canvas.clientHeight) / 5 / r.view.zoom;
      if (m[2] !== 1) r.zoomAt(r.canvas.clientWidth / 2, r.canvas.clientHeight / 2, m[2]);
      r.requestDraw();
    }
  }
  if (!selection) return;
  if (e.key === "Enter") toggle(selection.pane, selection.box, "toggle");
  if (e.key === "]" || e.key === "[") toggle(selection.pane, selection.box, "step", e.key === "]" ? 1 : -1);
  if (e.key === "Escape") select(selection.pane, -1);
});

overlaySel.addEventListener("change", applyOverlay);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  theme = readTheme();
  panes.forEach(({ r }) => r.setTheme(theme));
  applyOverlay();
});

// loading; while a load runs, clicks on the old graph or Findings would address the wrong session
let loading = false;
const inspectHint = $("#inspect-main").firstElementChild!.cloneNode(true);

/** The status line; errors are coloured so a failed load is not missed. */
function say(text: string, error = false) {
  info.textContent = error ? `error: ${text}` : text;
  info.classList.toggle("error", error);
}

function report(wasmBytes: number) {
  const heap = (performance as any).memory?.usedJSHeapSize as number | undefined;
  const mb = (b: number) => `${(b / 2 ** 20).toFixed(0)} MiB`;
  perf.textContent = `${perf.dataset.load ?? ""}  wasm ${mb(wasmBytes)}${heap ? `  js ${mb(heap)}` : ""}`;
}

/** "openvino 2026.4.1 GPU Intel(R) Arc(TM) 140V" gives "openvino 2026.4.1 GPU" (the pane heads carry the rest). */
const short = (label: string) => label.split(" ").slice(0, 3).join(" ");

/** What a bundle says about its capture (`Session.about`), or what imported files said; empty for a source pane. */
type About = { runtime?: string; version?: string; device?: string; device_name?: string; captured?: string; model?: string; phase?: string; imported?: string };
const RUNTIMES: Record<string, string> = { openvino: "OpenVINO", tensorrt: "TensorRT", onnxruntime: "ONNX Runtime" };
/** "Intel(R) Arc(TM) 140V GPU (16GB)" gives "Intel Arc 140V GPU (16GB)". */
const tidy = (name: string) => name.replace(/\((R|TM)\)/g, "").replace(/\s+/g, " ").trim();

/** A compiled pane's title and first words of its header: who compiled it, for which device, and when. Imported
 * files may not say the version or the device. */
function compiledBy(a: About): [string, string] {
  const device = a.device_name ? `on ${tidy(a.device_name)} (${a.device})` : a.device ? `on ${a.device}` : "device not recorded";
  const when = `${a.captured ? `, captured ${a.captured}` : ""}${a.phase ? `, ${a.phase}` : ""}${a.imported ? `, imported from ${a.imported}` : ""}`;
  return [`compiled by ${RUNTIMES[a.runtime ?? ""] ?? a.runtime}${a.version ? ` ${a.version}` : ""}`, `${device}${when}`];
}

const INSTALL = `uv tool install "git+https://github.com/goyaladitya05/afterpass#subdirectory=capture"`;

/** The compiled pane of a lone model file: nothing compiled it, and how to get a compiled graph. */
function showUncompiled(file: string | null) {
  const el = $<HTMLDetailsElement>("#uncompiled");
  el.hidden = !file;
  if (!file) return;
  const command = (text: string) => {
    const copy = h("button", { class: "copy" }, "copy");
    copy.addEventListener("click", () => navigator.clipboard.writeText(text).then(() => (copy.textContent = "copied"), () => undefined));
    return h("div", { class: "command" }, h("code", {}, text), copy);
  };
  const capture = (model: string, runtime: string, device: string) => command(`afterpass capture ${model} --runtime ${runtime} --device ${device}`);
  const name = file.toLowerCase();
  // capture reads ONNX and IR only; OpenVINO's converter turns a TFLite model into an IR it can read
  const tflite = name.endsWith(".tflite") ? file.slice(0, -".tflite".length) + ".xml" : null;
  el.open = false;
  el.replaceChildren(
    h("summary", {}, "not compiled: see what a compiler does to it"),
    h("p", {}, "Nothing has compiled this model. This page only reads files; a compiled graph comes from running the model through a runtime on real hardware, with afterpass capture."),
    ...(tflite ? [
      h("p", {}, "afterpass capture reads ONNX models and OpenVINO IR, not TFLite. OpenVINO reads TFLite: to see what it does to this model, on a machine with an Intel CPU, GPU or NPU, convert it to an IR with ovc (installed with the openvino package) and capture the .xml it writes:"),
      command(`ovc ${file}`),
      capture(tflite, "openvino", "CPU"),
      h("p", {}, "The bundle's source graph is then that IR, not this file. TensorRT reads ONNX models only."),
    ] : [
      h("p", {}, "To see what OpenVINO does to it, on a machine with an Intel CPU, GPU or NPU:"),
      capture(file, "openvino", "CPU"),
      ...(name.endsWith(".xml") ? [] : [h("p", {}, "To see what TensorRT does to it, on a machine with an NVIDIA GPU:"), capture(file, "tensorrt", "GPU"),
        h("p", {}, "To see what ONNX Runtime does to it on a CPU (install the tool below with --with onnxruntime):"), capture(file, "onnxruntime", "CPU")]),
    ]),
    h("p", {}, "Then open the .afterpass.json.gz it writes. Install the capture tool once with:"),
    command(INSTALL),
  );
}

function showLanding(show: boolean, error = "") {
  landing.hidden = !show;
  const box = $<HTMLElement>("#load-error");
  box.hidden = !error;
  box.textContent = error;
  if (show) $<HTMLElement>("#crumbs").textContent = "/ start";
}

/** True the first time in this tab; false after that, or when storage is blocked and a reload loop could not
 * be prevented. */
function firstReload(): boolean {
  try {
    if (sessionStorage.getItem("reloaded")) return false;
    sessionStorage.setItem("reloaded", "1");
    return true;
  } catch {
    return false;
  }
}

async function load(files: File[]) {
  // an IR's .bin chosen along with its .xml is not needed: weights are never read
  files = files.filter((f) => !(f.name.endsWith(".bin") && files.some((x) => x.name === f.name.replace(/\.bin$/, ".xml"))));
  if (!files.length) return;
  say("");
  loading = true;
  $("#findings-body").replaceChildren();
  const loadStart = performance.now();
  showLanding(false);
  $<HTMLElement>("#loading").hidden = false;
  try {
    const res = await rpc({ op: "load", files, flat: params.has("flat") });
    const tParsed = performance.now();
    paneCount = res.scenes.length;
    latencies = res.latencies;
    const labels: string[] = res.labels;
    const abouts: About[] = res.abouts;
    paneNames = paneCount === 3 ? ["source", "A", "B"] : paneCount === 2 ? (labels[1] ? ["source", "compiled"] : ["graph a", "graph b"]) : ["source"];
    panesEl.classList.toggle("single", paneCount === 1);
    panesEl.classList.toggle("triple", paneCount === 3);
    showUncompiled(paneCount === 1 ? files[0].name : null);
    const names = files.map((f) => f.name).join(" + ");
    $<HTMLElement>("#crumbs").textContent = paneCount === 3 ? `/ compare / A ${short(labels[1])} / B ${short(labels[2])}` : `/ ${names}${labels[1] ? " / " + labels[1] : ""}`;
    selection = null;
    // every header says what the pane is: the model as exported, or who compiled it for which device
    const model = abouts[1]?.model || (paneCount === 1 ? files[0].name : "");
    panes.forEach(({ el, r, style }, p) => {
      style.selected = -1;
      style.linked = new Map();
      r.resize();
      r.scene = res.scenes[p] ?? null;
      r.fit();
      const lat = latencies[p] === latencies[p] && latencies[p] !== undefined ? `, latency ${us(latencies[p])}` : "";
      let [title, what] = [paneNames[p] ?? "", ""];
      if (abouts[p]?.runtime) {
        const [by, on] = compiledBy(abouts[p]);
        [title, what] = [paneCount === 3 ? `${paneNames[p]}: ${by}` : by, `${on}, `];
      } else if (p === 0 && model) what = `${model} as exported, not compiled, `;
      el.querySelector(".title")!.textContent = title;
      el.querySelector(".meta")!.textContent = res.nodes[p] !== undefined ? `${what}${res.nodes[p]} nodes${lat}` : "";
    });
    overlaySel.value = latencies.some((l) => l === l) ? "time" : "none";
    await applyOverlay();
    $("#inspect-main").replaceChildren(inspectHint.cloneNode(true));
    $("#inspect-links").replaceChildren();
    $("#inspector-meta").textContent = "";
    loadFindings();
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const total = performance.now() - loadStart;
    const boxes = res.scenes.reduce((n: number, s: Scene) => n + s.kinds.length, 0);
    perf.dataset.load = `load ${total.toFixed(0)} ms (core ${res.ms.toFixed(0)} ms)  ${res.nodes.join(" + ")} nodes  ${boxes} boxes`;
    report(res.wasmBytes);
    console.log(`[afterpass] load ${names}: interactive ${total.toFixed(0)} ms, worker ${res.ms.toFixed(0)} ms, transfer+upload ${(total - (tParsed - loadStart)).toFixed(0)} ms, nodes ${res.nodes.join("+")}, boxes ${boxes}, wasm ${res.wasmBytes} B, js heap ${(performance as any).memory?.usedJSHeapSize ?? "n/a"} B`);
    if (params.has("bench")) bench();
  } catch (err) {
    // an outdated page reloads itself once; the session flag stops a loop if the reload does not help
    if ((err as Error).message === STALE && firstReload()) {
      location.reload();
      return;
    }
    // a failed load never leaves the previous model on screen, where it would pass for the new file
    console.error(err);
    paneCount = 0;
    selection = null;
    $("#findings-body").replaceChildren();
    showLanding(true, (err as Error).message);
    landing.scrollTop = 0;
    perf.textContent = "";
    delete perf.dataset.load;
    say("");
  } finally {
    loading = false;
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
      r.view.zoom = start[p].zoom * Math.pow(Math.max(1, 1 / start[p].zoom), phase);
      r.anchor(s.width * (0.5 + 0.4 * Math.sin(t * Math.PI * 2)), s.height * t, r.canvas.clientWidth / 2, r.canvas.clientHeight / 2);
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

async function fetchFile(url: string): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return new File([await res.blob()], url.split("/").pop()!);
}

/** Demo bundles appear when demo/index.json exists: [{ "title": "...", "files": ["demo/x.afterpass.json.gz"] }]. */
async function loadDemos() {
  try {
    const res = await fetch("demo/index.json");
    if (!res.ok) return;
    const demos: { title: string; files: string[] }[] = await res.json();
    const ul = $("#demos ul");
    for (const d of demos) {
      const b = h("button", {}, d.title);
      b.addEventListener("click", async () => load(await Promise.all(d.files.map(fetchFile))));
      ul.append(h("li", {}, b));
    }
    $("#demos").hidden = !demos.length;
  } catch {
    // no demos
  }
}

// start
async function start() {
  await document.fonts.load('40px "Source Code Pro"').catch(() => undefined);
  theme = readTheme();
  const atlas = buildAtlas(getComputedStyle(document.documentElement).getPropertyValue("--font-mono") || "monospace");
  panes = [...document.querySelectorAll<HTMLElement>(".pane")].map((el) => {
    const r = new Renderer(el.querySelector("canvas")!, atlas);
    r.setTheme(theme);
    return { el, r, style: { selected: -1, linked: new Map(), overlay: "none" }, legend: el.querySelector(".legend")!, legendNames: [] };
  });
  panes.forEach((_, p) => bindPane(p));
  new ResizeObserver(() => panes.forEach(({ r }) => r.resize())).observe(panesEl);

  scramble(document);
  asciiTexture($(".landing .ascii"));
  ticker($(".ticker-track"), ["source graph", "compiled graph", "fused kernels", "provenance", "openvino", "tensorrt", "time", "precision", "layout", "reorders", "x48 blocks", "30B graphs under a second"]);
  const fileInput = $<HTMLInputElement>("#file");
  for (const id of ["#open", "#open-landing"]) $(id).addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => load([...(fileInput.files ?? [])]));
  $("#home").addEventListener("click", () => showLanding(!paneCount || landing.hidden !== false));
  window.addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("drag"); });
  window.addEventListener("dragleave", () => document.body.classList.remove("drag"));
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    document.body.classList.remove("drag");
    load([...(e.dataTransfer?.files ?? [])]);
  });
  loadDemos();

  // ?src=<url>[&cmp=<url>]: load by URL (demo bundles, benchmarks); skips the loading screen
  const urls = [params.get("src"), params.get("cmp")].filter((u): u is string => !!u);
  const loaderEl = $("#loader");
  if (urls.length || params.has("bench")) loaderEl.remove();
  else loader(loaderEl);
  if (urls.length) {
    Promise.all(urls.map(fetchFile)).then(load, (e) => say(e.message, true));
  }
}

start();

// any failed request (not only a load) reaches the status line, not just the console
window.addEventListener("unhandledrejection", (e) => say(e.reason instanceof Error ? e.reason.message : String(e.reason), true));

if (import.meta.env.DEV) (window as any).__ap = { get panes() { return panes; }, rpc };
