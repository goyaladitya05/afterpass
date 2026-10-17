// Hosts the wasm core. Files are read here: ONNX through ranged FileReaderSync reads (weights are skipped,
// never copied), OpenVINO IR .xml whole, bundles through the browser's gzip DecompressionStream.

import init, { Session } from "../pkg/afterpass_core.js";

export type Request =
  | { id: number; op: "load"; files: File[]; flat?: boolean }
  | { id: number; op: "toggle" | "step" | "select" | "inspect"; pane: number; box: number; delta?: number }
  | { id: number; op: "overlay"; mode: string }
  | { id: number; op: "findings" }
  | { id: number; op: "search"; query: string }
  | { id: number; op: "reveal"; pane: number; target: number; group: boolean };

let session: Session | null = null;
const wasm = init();

async function open(file: File): Promise<Session> {
  const name = file.name.toLowerCase();
  if (name.endsWith(".onnx")) {
    const reader = new FileReaderSync();
    const read = (offset: number, length: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(offset, offset + length)));
    return Session.load_onnx(read, file.size);
  }
  if (name.endsWith(".xml")) return Session.load_ir(new Uint8Array(await file.arrayBuffer()));
  let bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return Session.load_bundle(bytes);
}

function scene(pane: number) {
  const s = session!.scene(pane) as Record<string, unknown>;
  return { scene: s, transfer: ["boxes", "kinds", "flags", "times", "segs"].map((k) => (s[k] as ArrayBufferView).buffer as ArrayBuffer) };
}

async function handle(req: Request): Promise<[unknown, Transferable[]]> {
  const memory = (await wasm).memory;
  if (req.op === "load") {
    const t0 = performance.now();
    const next = await open(req.files[0]);
    if (req.files[1]) {
      // two bundles of one source are compared; anything else is shown side by side without links
      const other = await open(req.files[1]);
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
    return [{ scenes, nodes, labels, latencies, ms: performance.now() - t0, wasmBytes: memory.buffer.byteLength }, transfer];
  }
  if (!session) throw new Error("nothing loaded");
  const ses = session;
  const panes = Array.from({ length: ses.panes() }, (_, p) => p);
  if (req.op === "overlay") return [panes.map((p) => ses.overlay(p, req.mode)), []];
  if (req.op === "findings") return [JSON.parse(ses.findings()), []];
  if (req.op === "search") return [panes.flatMap((p) => JSON.parse(ses.search(p, req.query, 40)).map((r: object) => ({ ...r, pane: p }))), []];
  if (req.op === "inspect") return [JSON.parse(ses.inspect(req.pane, req.box)), []];
  if (req.op === "reveal") {
    const box = ses.reveal(req.pane, req.target, req.group);
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
    self.postMessage({ id: e.data.id, error: String(err instanceof Error ? err.message : err) });
  }
};
