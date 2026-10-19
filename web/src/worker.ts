// Hosts the wasm core. Files are read here: ONNX and TFLite through ranged FileReaderSync reads (weights are
// skipped, never copied), OpenVINO IR .xml whole, bundles through the browser's gzip DecompressionStream.

import init, { Session } from "../pkg/afterpass_core.js";
import { STALE } from "./stale";

export type Request =
  | { id: number; op: "load"; files: File[]; flat?: boolean }
  | { id: number; op: "toggle" | "step" | "select" | "inspect"; pane: number; box: number; delta?: number }
  | { id: number; op: "overlay"; mode: string }
  | { id: number; op: "findings" }
  | { id: number; op: "search"; query: string }
  | { id: number; op: "reveal"; pane: number; target: number; group: boolean };

let session: Session | null = null;
// a deploy replaces every hashed file, so a page opened before it asks for a core that no longer exists: a 404,
// or a host's HTML page served in its place (no wasm magic word)
const wasm = init().catch((e) => {
  throw new Error(/404|magic word/.test(String(e)) ? STALE : `could not start the core (${e instanceof Error ? e.message : e})`);
});

/** The file's kind from its first bytes, since names are often changed: gzip or JSON is a bundle, XML an IR,
 * the identifier TFL3 at bytes 4..8 a TFLite flatbuffer, a protobuf starting with field 1 (ir_version) an ONNX
 * model. TFLite goes first: its first byte is an offset, which can look like `<` or `{`. */
async function open(file: File): Promise<Session> {
  const head = new Uint8Array(await file.slice(0, 64).arrayBuffer());
  const text = new TextDecoder().decode(head).trimStart();
  const name = file.name.toLowerCase();
  const ranged = () => {
    const reader = new FileReaderSync();
    return (offset: number, length: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(offset, offset + length)));
  };
  try {
    if (String.fromCharCode(...head.subarray(4, 8)) === "TFL3" || name.endsWith(".tflite")) return Session.load_tflite(ranged(), file.size);
    if (head[0] === 0x1f && head[1] === 0x8b) {
      const stream = file.stream().pipeThrough(new DecompressionStream("gzip"));
      return Session.load_bundle(new Uint8Array(await new Response(stream).arrayBuffer()));
    }
    if (text.startsWith("{")) return Session.load_bundle(new Uint8Array(await file.arrayBuffer()));
    if (text.startsWith("<")) return Session.load_ir(new Uint8Array(await file.arrayBuffer()));
    if (head[0] === 0x08 || name.endsWith(".onnx")) return Session.load_onnx(ranged(), file.size);
  } catch (e) {
    throw new Error(`${file.name}: ${e instanceof Error ? e.message : e}`);
  }
  if (name.endsWith(".bin")) throw new Error(`${file.name} holds OpenVINO weights only: open the .xml next to it`);
  throw new Error(`${file.name}: not an ONNX or TFLite model, OpenVINO IR .xml or afterpass bundle`);
}

/** What benchmark_app wrote, by content: the execution graph (-exec_graph_path: an IR whose layers carry the
 * runtime's rt_info, primitiveType among it), the statistics report (CSV, or JSON with -json_stats) and the
 * per-layer counters report, which repeats the execution graph's times. */
type Kind = "exec" | "report" | "counters" | "model";

async function kind(file: File): Promise<Kind> {
  const text = new TextDecoder().decode(await file.slice(0, 4096).arrayBuffer()).trimStart();
  if (text.startsWith("<") && text.includes('primitiveType="')) return "exec";
  if (text.startsWith("Command line parameters") || /^\{\s*"cmd_options"/.test(text)) return "report";
  if (text.startsWith("layerName;execStatus") || /^\{\s*"report_type"/.test(text)) return "counters";
  return "model";
}

const EXEC = "the execution graph benchmark_app writes with -exec_graph_path exec.xml";

/** A model with benchmark_app's execution graph, and its statistics report if given: the compiled graph comes from
 * the execution graph, linked to the model as a capture links it. */
async function imported(files: File[], kinds: Kind[]): Promise<Session> {
  const of = (k: Kind) => files.filter((_, i) => kinds[i] === k);
  const [exec, reports, models] = [of("exec"), of("report"), of("model")];
  if (exec.length > 1) throw new Error(`open one execution graph at a time, not ${exec.map((f) => f.name).join(" and ")}`);
  if (reports.length > 1) throw new Error(`open one benchmark_app report at a time, not ${reports.map((f) => f.name).join(" and ")}`);
  if (models.length !== 1) {
    throw new Error(models.length
      ? `${exec[0].name} is an OpenVINO execution graph: open it with the one model it was compiled from, not ${models.map((f) => f.name).join(" and ")}`
      : `${exec[0].name} is an OpenVINO execution graph: open it together with the model it was compiled from (.onnx or OpenVINO IR .xml)`);
  }
  const next = await open(models[0]);
  try {
    if (next.panes() !== 1) throw new Error(`${models[0].name} already holds a compiled graph: open ${exec[0].name} with the .onnx or .xml it was compiled from`);
    const report = reports[0] ? new Uint8Array(await reports[0].arrayBuffer()) : new Uint8Array();
    try {
      next.import_exec(new Uint8Array(await exec[0].arrayBuffer()), report, models[0].name);
    } catch (e) {
      throw new Error(`${exec[0].name}${reports[0] ? ` with ${reports[0].name}` : ""}: ${e instanceof Error ? e.message : e}`);
    }
  } catch (e) {
    next.free();
    throw e;
  }
  return next;
}

function scene(pane: number) {
  const s = session!.scene(pane) as Record<string, unknown>;
  return { scene: s, transfer: ["boxes", "kinds", "flags", "times", "segs"].map((k) => (s[k] as ArrayBufferView).buffer as ArrayBuffer) };
}

async function handle(req: Request): Promise<[unknown, Transferable[]]> {
  const memory = (await wasm).memory;
  if (req.op === "load") {
    const t0 = performance.now();
    // benchmark_app's counters report repeats the execution graph's times: skip it next to one
    let files = req.files;
    let kinds = await Promise.all(files.map(kind));
    if (kinds.includes("exec")) [files, kinds] = [files.filter((_, i) => kinds[i] !== "counters"), kinds.filter((k) => k !== "counters")];
    const side = kinds.findIndex((k) => k === "report" || k === "counters");
    if (side >= 0 && !kinds.includes("exec")) {
      throw new Error(kinds[side] === "report"
        ? `${files[side].name} is benchmark_app's statistics report, latency only: open it with the model and ${EXEC}`
        : `${files[side].name} holds benchmark_app's per-layer counters: open the model with ${EXEC} instead, which holds the same times and the compiled graph`);
    }
    const most = kinds.includes("exec") ? 3 : 2;
    if (files.length > most) {
      throw new Error(`open one or two files, or a model with ${EXEC} and benchmark_report.csv, not ${files.length}: ${files.map((f) => f.name).join(", ")}`);
    }
    const next = kinds.includes("exec") ? await imported(files, kinds) : await open(files[0]);
    if (files[1] && !kinds.includes("exec")) {
      // two bundles of one source are compared; anything else is shown side by side without links
      let other: Session;
      try {
        other = await open(files[1]);
      } catch (e) {
        next.free();
        throw e;
      }
      if (next.panes() === 2 && other.panes() === 2) {
        try {
          next.compare(other);
        } catch (e) {
          next.free();
          throw e;
        }
      } else next.attach(other);
    }
    if (req.flat) next.flatten();
    session?.free();
    session = next;
    const panes = session.panes();
    const scenes = [], transfer: Transferable[] = [];
    for (let p = 0; p < panes; p++) {
      const s = scene(p);
      scenes.push(s.scene);
      transfer.push(...s.transfer);
    }
    const all = Array.from({ length: panes }, (_, p) => p);
    const nodes = all.map((p) => session!.node_count(p));
    const labels = all.map((p) => session!.label(p)), latencies = all.map((p) => session!.latency_us(p));
    const abouts = all.map((p) => JSON.parse(session!.about(p)));
    return [{ scenes, nodes, labels, latencies, abouts, ms: performance.now() - t0, wasmBytes: memory.buffer.byteLength }, transfer];
  }
  if (!session) throw new Error("nothing loaded");
  const ses = session;
  const panes = Array.from({ length: ses.panes() }, (_, p) => p);
  if (req.op === "overlay") return [panes.map((p) => ses.overlay(p, req.mode)), []];
  if (req.op === "findings") return [JSON.parse(ses.findings()), []];
  if (req.op === "search") return [panes.flatMap((p) => JSON.parse(ses.search(p, req.query, 40)).map((r: object) => ({ ...r, pane: p }))), []];
  if (req.op === "inspect") return [JSON.parse(ses.inspect(req.pane, req.box)), []];
  if (req.op === "reveal") {
    if (req.pane >= ses.panes()) return [{ box: -1 }, []];
    const box = ses.reveal(req.pane, req.target, req.group); // may expand groups even when it returns -1
    const sc = scene(req.pane);
    return [{ box, scene: sc.scene, wasmBytes: memory.buffer.byteLength }, sc.transfer];
  }
  if (req.op === "select") {
    const linked = panes.map((q) => {
      const l = (q === req.pane ? {} : ses.linked(req.pane, req.box, q)) as { boxes?: Uint32Array; grades?: Uint8Array };
      return { boxes: l.boxes ?? new Uint32Array(), grades: l.grades ?? new Uint8Array() };
    });
    return [{ linked, text: session.describe(req.pane, req.box) }, []];
  }
  const anchor = req.op === "toggle" ? session.toggle(req.pane, req.box) : session.step(req.pane, req.box, req.delta ?? 1);
  if (anchor < 0) return [{ anchor }, []];
  const s = scene(req.pane);
  return [{ anchor, scene: s.scene, wasmBytes: memory.buffer.byteLength }, s.transfer];
}

self.onmessage = async (e: MessageEvent<Request>) => {
  try {
    const [result, transfer] = await handle(e.data);
    self.postMessage({ id: e.data.id, result }, { transfer });
  } catch (err) {
    // a panic in the core aborts with a RuntimeError and leaves the wasm instance unusable
    const fatal = err instanceof WebAssembly.RuntimeError;
    if (fatal) session = null;
    const message = fatal ? "internal error in the core; reload the page" : String(err instanceof Error ? err.message : err);
    self.postMessage({ id: e.data.id, error: message });
  }
};
