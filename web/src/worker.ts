// Hosts the wasm core. Files are read here: ONNX through ranged FileReaderSync reads (weights are skipped,
// never copied), OpenVINO IR .xml whole, bundles through the browser's gzip DecompressionStream.

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
 * a protobuf starting with field 1 (ir_version) an ONNX model. */
async function open(file: File): Promise<Session> {
  const head = new Uint8Array(await file.slice(0, 64).arrayBuffer());
  const text = new TextDecoder().decode(head).trimStart();
  try {
    if (head[0] === 0x1f && head[1] === 0x8b) {
      const stream = file.stream().pipeThrough(new DecompressionStream("gzip"));
      return Session.load_bundle(new Uint8Array(await new Response(stream).arrayBuffer()));
    }
    if (text.startsWith("{")) return Session.load_bundle(new Uint8Array(await file.arrayBuffer()));
    if (text.startsWith("<")) return Session.load_ir(new Uint8Array(await file.arrayBuffer()));
    if (head[0] === 0x08 || file.name.toLowerCase().endsWith(".onnx")) {
      const reader = new FileReaderSync();
      const read = (offset: number, length: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(offset, offset + length)));
      return Session.load_onnx(read, file.size);
    }
  } catch (e) {
    throw new Error(`${file.name}: ${e instanceof Error ? e.message : e}`);
  }
  if (file.name.toLowerCase().endsWith(".bin")) throw new Error(`${file.name} holds OpenVINO weights only: open the .xml next to it`);
  throw new Error(`${file.name}: not an ONNX model, OpenVINO IR .xml or afterpass bundle`);
}

function scene(pane: number) {
  const s = session!.scene(pane) as Record<string, unknown>;
  return { scene: s, transfer: ["boxes", "kinds", "flags", "times", "segs"].map((k) => (s[k] as ArrayBufferView).buffer as ArrayBuffer) };
}

async function handle(req: Request): Promise<[unknown, Transferable[]]> {
  const memory = (await wasm).memory;
  if (req.op === "load") {
    const t0 = performance.now();
    const files = req.files;
    if (files.length > 2) throw new Error(`open one or two files, not ${files.length}`);
    const next = await open(files[0]);
    if (files[1]) {
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
